"""LTX-Video runner (text-to-video and image-to-video). Runs inside /opt/venv-video.

Mirrors ltx_video.inference.infer(), with two differences:
  * infer() builds the whole pipeline on every call; here it is built once at startup.
  * ltx_video's create_ltx_video_pipeline() moves the T5-XXL text encoder to the GPU in
    float32 (~19 GB) and only then casts it to bfloat16 (~9.5 GB). Next to Chatterbox
    that overflows a 24 GB card, so build_pipeline() below loads T5 in bfloat16 from
    the start: same weights and same final dtypes, half the peak memory.
"""

from __future__ import annotations

import gc
import json
import os
import random
import threading
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from ltx_video.inference import (
    create_latent_upsampler,
    create_transformer,
    prepare_conditioning,
)
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline, LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from PIL import Image, ImageOps
from safetensors import safe_open
from transformers import T5EncoderModel, T5Tokenizer

from app.runners.common import (
    ModelRunner,
    UserError,
    clamp,
    env_int,
    frames_for_duration,
    main,
    video_dimensions,
)

CONFIG_DIR = Path(__file__).with_name("ltx_configs")
LTX_REPO = "Lightricks/LTX-Video"
DEFAULT_NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text"

STG_MODES = {
    "attention_values": SkipLayerStrategy.AttentionValues,
    "stg_av": SkipLayerStrategy.AttentionValues,
    "attention_skip": SkipLayerStrategy.AttentionSkip,
    "stg_as": SkipLayerStrategy.AttentionSkip,
    "residual": SkipLayerStrategy.Residual,
    "stg_r": SkipLayerStrategy.Residual,
    "transformer_block": SkipLayerStrategy.TransformerBlock,
    "stg_t": SkipLayerStrategy.TransformerBlock,
}


def fit_image(path: str, width: int, height: int) -> str:
    """Centre-crop ("cover") and resize an image to exactly width x height; returns the new path.

    Honours EXIF rotation (phone photos) and drops alpha so LTX gets plain RGB.
    """
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            fitted = ImageOps.fit(img, (width, height), method=Image.Resampling.LANCZOS)
    except Exception as err:  # noqa: BLE001
        raise UserError(f"Could not read the scene image: {err}") from err
    out = str(Path(path).with_name("fitted.png"))
    fitted.save(out)
    return out


class GpuPinnedLTXVideoPipeline(LTXVideoPipeline):
    """LTXVideoPipeline that always runs on CUDA.

    diffusers derives `_execution_device` from the first registered component, which is
    the text encoder. With offload_to_cpu the text encoder is parked on the CPU after
    encoding, so on the next pass LTX concludes it should run on the CPU, moves the
    transformer there and crashes with "Expected all tensors to be on the same device".
    """

    @property
    def _execution_device(self):
        return torch.device("cuda")


def build_pipeline(
    ckpt_path: str,
    precision: str,
    text_encoder_name: str,
    sampler: str | None,
    device: str,
) -> LTXVideoPipeline:
    """create_ltx_video_pipeline() without prompt enhancement, with T5 loaded in bfloat16."""
    with safe_open(ckpt_path, framework="pt") as f:
        allowed_inference_steps = json.loads(f.metadata()["config"]).get("allowed_inference_steps")

    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path).to(torch.bfloat16).to(device)
    transformer = create_transformer(ckpt_path, precision).to(device)
    if sampler == "from_checkpoint" or not sampler:
        scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)
    else:
        scheduler = RectifiedFlowScheduler(
            sampler="Uniform" if sampler.lower() == "uniform" else "LinearQuadratic"
        )
    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_name,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_name, subfolder="tokenizer")

    pipeline = GpuPinnedLTXVideoPipeline(
        transformer=transformer,
        patchifier=SymmetricPatchifier(patch_size=1),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        vae=vae,
        prompt_enhancer_image_caption_model=None,
        prompt_enhancer_image_caption_processor=None,
        prompt_enhancer_llm_model=None,
        prompt_enhancer_llm_tokenizer=None,
        allowed_inference_steps=allowed_inference_steps,
    )
    return pipeline.to(device)


class LtxRunner(ModelRunner):
    name = "video"

    def load(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU visible to the video runner")
        self.device = "cuda"

        config_name = os.environ.get("LTX_PIPELINE_CONFIG", "ltxv-2b-0.9.8-distilled.yaml")
        config_path = Path(config_name) if os.path.isabs(config_name) else CONFIG_DIR / config_name
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)

        checkpoint = hf_hub_download(LTX_REPO, self.config["checkpoint_path"])
        # No prompt enhancement: it would load Florence-2 + Llama-3.2-3B (~10 GB more),
        # and the backend's LLM already writes detailed visual prompts.
        pipeline = build_pipeline(
            ckpt_path=checkpoint,
            precision=self.config["precision"],
            text_encoder_name=self.config["text_encoder_model_name_or_path"],
            sampler=self.config.get("sampler"),
            device=self.device,
        )
        if self.config.get("pipeline_type") == "multi-scale":
            upsampler = hf_hub_download(LTX_REPO, self.config["spatial_upscaler_model_path"])
            pipeline = LTXMultiScalePipeline(
                pipeline, latent_upsampler=create_latent_upsampler(upsampler, pipeline.device)
            )
        self.pipeline = pipeline
        # The single-scale pipeline that owns the text encoder and transformer.
        self.video_pipe = getattr(pipeline, "video_pipeline", pipeline)
        stg_mode = self.config.get("stg_mode", "attention_values").lower()
        self.skip_layer_strategy = STG_MODES[stg_mode]
        # On by default: the T5 text encoder (~9.5 GB) only reads the prompt. It lives
        # in CPU RAM and visits the GPU alone, once per clip, before generation starts.
        self.offload = os.environ.get("LTX_OFFLOAD_TO_CPU", "1") == "1"
        self.fps = env_int("LTX_FPS", 24)
        self.long_edge = env_int("LTX_LONG_EDGE", 960)
        # 5 s (121 frames at 544x960) is what fits next to Chatterbox on 24 GB.
        self.max_seconds = env_int("LTX_MAX_SECONDS", 5)
        # If GPU memory still held after cleanup grows this far past the idle baseline,
        # something is leaking: the runner restarts itself before the next job fails.
        self.leak_limit = env_int("LTX_LEAK_LIMIT_MB", 2048) * 1024 * 1024

        self._park_text_encoder()
        gc.collect()
        torch.cuda.empty_cache()
        self.baseline = torch.cuda.memory_allocated()
        print(f"[video] idle GPU memory: {self._mem()}", flush=True)

    def _mem(self) -> str:
        gib = 1024**3
        return (
            f"allocated {torch.cuda.memory_allocated() / gib:.2f} GiB, "
            f"reserved {torch.cuda.memory_reserved() / gib:.2f} GiB"
        )

    def _park_text_encoder(self) -> None:
        if self.offload and getattr(self.video_pipe, "text_encoder", None) is not None:
            self.video_pipe.text_encoder = self.video_pipe.text_encoder.cpu()

    def cleanup(self) -> None:
        # A failed run's traceback keeps its GPU tensors alive until the GC runs, and a
        # run that died mid-encode may have left T5 on the GPU: undo both.
        gc.collect()
        self._park_text_encoder()
        torch.cuda.empty_cache()
        held = torch.cuda.memory_allocated()
        print(f"[video] after job: {self._mem()}", flush=True)
        if held - self.baseline > self.leak_limit:
            # The job's response has already been sent; a fresh process (the worker
            # restarts it automatically) is the only reliable way to get memory back.
            print(
                f"[video] {(held - self.baseline) / 1024**3:.2f} GiB still held after "
                "cleanup — restarting the video runner to free it",
                flush=True,
            )
            threading.Timer(1.0, os._exit, (3,)).start()

    def _encode(self, prompt: str, negative: str) -> dict:
        """Prompt -> T5 embeddings, with T5 the only thing that moves onto the GPU.

        LTX would otherwise move T5 onto the GPU at the start of EVERY pass (twice per
        multi-scale clip) next to whatever generation still holds, which is where 24 GB
        cards run out of memory.
        """
        video = self.video_pipe
        if self.offload:
            video.text_encoder = video.text_encoder.to(self.device)
        try:
            with torch.no_grad():
                embeds, mask, neg_embeds, neg_mask = video.encode_prompt(
                    prompt,
                    True,
                    negative_prompt=negative,
                    device=self.device,
                    text_encoder_max_tokens=256,
                )
        finally:
            self._park_text_encoder()
            torch.cuda.empty_cache()
        return {
            "prompt_embeds": embeds,
            "prompt_attention_mask": mask,
            "negative_prompt_embeds": neg_embeds,
            "negative_prompt_attention_mask": neg_mask,
        }

    def run(self, payload: dict) -> dict:
        params = payload["params"]
        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            raise UserError("prompt is empty")
        aspect = params.get("aspectRatio", "9:16")
        if aspect not in ("9:16", "16:9", "1:1"):
            raise UserError(f"Unsupported aspectRatio '{aspect}'")
        seconds = clamp(float(params.get("durationSeconds", 5)), 1.0, float(self.max_seconds))
        seed = params.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)

        width, height = video_dimensions(aspect, self.long_edge)
        num_frames = frames_for_duration(seconds, self.fps)

        image = payload.get("inputs", {}).get("imagePath")
        if image:
            # LTX's image loader (just_crop=True) crops to the aspect ratio but never
            # resizes, so a 1080x1920 photo would not match the 544x960 latent. Fit it
            # to the exact frame first.
            image = fit_image(image, width, height)

        # Start from a clean GPU, then retry once after freeing memory if it runs out.
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[video] before job: {self._mem()}", flush=True)
        for attempt in (1, 2):
            try:
                frames = self._generate(
                    prompt,
                    params.get("negativePrompt") or DEFAULT_NEGATIVE,
                    image,
                    width,
                    height,
                    num_frames,
                    seed,
                )
                break
            except torch.OutOfMemoryError:
                if attempt == 2:
                    raise
                print("[video] GPU out of memory — freeing memory and retrying once", flush=True)
            # Outside the except block, so the failed attempt's frames are released.
            self._park_text_encoder()
            gc.collect()
            torch.cuda.empty_cache()

        with imageio.get_writer(
            payload["outputPath"],
            fps=self.fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=16,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)

        return {
            "durationSec": round(len(frames) / self.fps, 3),
            "width": width,
            "height": height,
            "frames": len(frames),
            "fps": self.fps,
            "seed": seed,
        }

    def _generate(
        self,
        prompt: str,
        negative: str,
        image: str | None,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
    ) -> np.ndarray:
        embeds = self._encode(prompt, negative)
        conditioning = None
        if image:
            conditioning = prepare_conditioning(
                conditioning_media_paths=[image],
                conditioning_strengths=[1.0],
                conditioning_start_frames=[0],
                height=height,
                width=width,
                num_frames=num_frames,
                padding=(0, 0, 0, 0),  # dimensions are already multiples of 32
                pipeline=self.pipeline,
            )

        call_config = {k: v for k, v in self.config.items() if k != "stg_mode"}
        generator = torch.Generator(device=self.device).manual_seed(seed)
        video = self.video_pipe
        text_encoder = video.text_encoder
        # With the embeddings supplied and no text encoder, LTX skips T5 entirely, so
        # generation never shares the GPU with it and the transformer can stay put.
        video.text_encoder = None
        try:
            images = self.pipeline(
                **call_config,
                **embeds,
                skip_layer_strategy=self.skip_layer_strategy,
                generator=generator,
                output_type="pt",
                callback_on_step_end=None,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=self.fps,
                # LTX refuses text together with embeddings; the embeddings carry it.
                prompt=None,
                negative_prompt=None,
                media_items=None,
                conditioning_items=conditioning,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=0.15,
                mixed_precision=self.config["precision"] == "mixed_precision",
                offload_to_cpu=False,
                device=self.device,
                enhance_prompt=False,
            ).images
            frames = images[0][:, :num_frames].permute(1, 2, 3, 0).cpu().float().numpy()
            del images
            return (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
        finally:
            video.text_encoder = text_encoder
            del embeds, conditioning
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(LtxRunner())
