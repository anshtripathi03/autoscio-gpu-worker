"""LTX-Video runner (text-to-video and image-to-video). Runs inside /opt/venv-video.

Mirrors ltx_video.inference.infer(), with one important difference: infer() builds
the whole pipeline (several GB of weights) on every call. Here it is built once at
startup and reused, so a job costs only the generation itself.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from ltx_video.inference import (
    create_latent_upsampler,
    create_ltx_video_pipeline,
    prepare_conditioning,
)
from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

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
        pipeline = create_ltx_video_pipeline(
            ckpt_path=checkpoint,
            precision=self.config["precision"],
            text_encoder_model_name_or_path=self.config["text_encoder_model_name_or_path"],
            sampler=self.config.get("sampler"),
            device=self.device,
            # Prompt enhancement would load Florence-2 + Llama-3.2-3B (~10 GB more).
            # The backend's LLM already writes detailed visual prompts.
            enhance_prompt=False,
        )
        if self.config.get("pipeline_type") == "multi-scale":
            upsampler = hf_hub_download(LTX_REPO, self.config["spatial_upscaler_model_path"])
            pipeline = LTXMultiScalePipeline(
                pipeline, latent_upsampler=create_latent_upsampler(upsampler, pipeline.device)
            )
        self.pipeline = pipeline
        stg_mode = self.config.get("stg_mode", "attention_values").lower()
        self.skip_layer_strategy = STG_MODES[stg_mode]
        self.offload = os.environ.get("LTX_OFFLOAD_TO_CPU", "0") == "1"
        self.fps = env_int("LTX_FPS", 24)
        self.long_edge = env_int("LTX_LONG_EDGE", 960)
        self.max_seconds = env_int("LTX_MAX_SECONDS", 8)

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

        conditioning = None
        image = payload.get("inputs", {}).get("imagePath")
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
        try:
            images = self.pipeline(
                **call_config,
                skip_layer_strategy=self.skip_layer_strategy,
                generator=generator,
                output_type="pt",
                callback_on_step_end=None,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=self.fps,
                prompt=prompt,
                prompt_attention_mask=None,
                negative_prompt=params.get("negativePrompt") or DEFAULT_NEGATIVE,
                negative_prompt_attention_mask=None,
                media_items=None,
                conditioning_items=conditioning,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=0.15,
                mixed_precision=self.config["precision"] == "mixed_precision",
                offload_to_cpu=self.offload,
                device=self.device,
                enhance_prompt=False,
            ).images

            frames = images[0][:, :num_frames].permute(1, 2, 3, 0).cpu().float().numpy()
            frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
        finally:
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


if __name__ == "__main__":
    main(LtxRunner())
