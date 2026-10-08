"""Chatterbox Multilingual voice-cloning runner. Runs inside /opt/venv-tts.

Zero-shot cloning: no training. Each job brings its own 10-30s voice sample, the
model conditions on it, speaks the text, and the conditioning is reset afterwards
so one tenant's voice can never carry into the next job.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS

from app.runners.common import ModelRunner, UserError, clamp, env_int, ffmpeg, main, split_text

PAUSE_BETWEEN_CHUNKS_S = 0.2


class ChatterboxRunner(ModelRunner):
    name = "tts"

    def load(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU visible to the TTS runner")
        self.model = ChatterboxMultilingualTTS.from_pretrained(
            device="cuda", t3_model=os.environ.get("CHATTERBOX_T3_MODEL", "v3")
        )
        # The repo ships a built-in default voice; used when a job has no sample.
        self.default_conds = self.model.conds
        self.max_chars = env_int("CHATTERBOX_CHUNK_CHARS", 250)

    def run(self, payload: dict) -> dict:
        params = payload["params"]
        text = str(params.get("text", "")).strip()
        language = str(params.get("language", "en")).lower()
        if language not in SUPPORTED_LANGUAGES:
            raise UserError(
                f"Unsupported language '{language}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        chunks = split_text(text, self.max_chars)
        if not chunks:
            raise UserError("Text is empty")
        exaggeration = clamp(float(params.get("exaggeration", 0.5)), 0.0, 2.0)
        cfg_weight = clamp(float(params.get("cfgWeight", 0.5)), 0.0, 1.0)

        sample = payload.get("inputs", {}).get("voiceSamplePath")
        try:
            if sample:
                # Browser recordings arrive as webm/m4a/mp3; normalise to mono WAV first.
                reference = str(Path(sample).with_name("voice_reference.wav"))
                ffmpeg(sample, reference, ["-ac", "1", "-ar", "24000"])
                self.model.prepare_conditionals(reference, exaggeration=exaggeration)
            elif self.default_conds is None:
                raise UserError("A voice sample is required (inputs.voiceSampleUrl)")
            else:
                self.model.conds = self.default_conds

            pause = np.zeros(int(self.model.sr * PAUSE_BETWEEN_CHUNKS_S), dtype=np.float32)
            parts: list[np.ndarray] = []
            for i, chunk in enumerate(chunks):
                wav = self.model.generate(
                    chunk, language_id=language, exaggeration=exaggeration, cfg_weight=cfg_weight
                )
                if i:
                    parts.append(pause)
                parts.append(wav.squeeze(0).detach().cpu().numpy().astype(np.float32))
            audio = np.concatenate(parts)
        finally:
            self.model.conds = self.default_conds
            torch.cuda.empty_cache()

        out = Path(payload["outputPath"])
        wav_path = out if out.suffix == ".wav" else out.with_suffix(".tmp.wav")
        sf.write(str(wav_path), audio, self.model.sr)
        if out.suffix == ".mp3":
            ffmpeg(str(wav_path), str(out), ["-codec:a", "libmp3lame", "-b:a", "128k"])
            wav_path.unlink(missing_ok=True)

        return {
            "durationSec": round(len(audio) / self.model.sr, 3),
            "chunks": len(chunks),
            "language": language,
            "clonedVoice": bool(sample),
        }


if __name__ == "__main__":
    main(ChatterboxRunner())
