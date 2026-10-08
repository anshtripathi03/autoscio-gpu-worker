"""Worker settings, read once from the environment at startup."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Shared secret with the Autoscio backend. Authenticates every request to this
    # worker AND signs every webhook this worker sends back.
    api_key: str
    # Local debugging only. Never set on a Pod with a public URL.
    allow_no_auth: bool
    # "real" spawns the GPU model runners; "fake" writes placeholder files (CI/tests).
    runner_mode: str
    enabled_runners: tuple[str, ...]
    work_dir: str
    tts_python: str
    video_python: str
    tts_port: int
    video_port: int
    runner_timeout_s: int
    max_input_bytes: int
    max_queue: int
    job_ttl_s: int
    version: str


def load_settings() -> Settings:
    enabled = tuple(
        name.strip()
        for name in os.environ.get("ENABLED_RUNNERS", "tts,video").split(",")
        if name.strip()
    )
    return Settings(
        api_key=os.environ.get("WORKER_API_KEY", "").strip(),
        allow_no_auth=_bool("ALLOW_INSECURE_NO_AUTH"),
        runner_mode=os.environ.get("RUNNER_MODE", "real").strip().lower(),
        enabled_runners=enabled,
        work_dir=os.environ.get("WORK_DIR", "/tmp/jobs"),
        tts_python=os.environ.get("TTS_PYTHON", "/opt/venv-tts/bin/python"),
        video_python=os.environ.get("VIDEO_PYTHON", "/opt/venv-video/bin/python"),
        tts_port=_int("TTS_RUNNER_PORT", 8101),
        video_port=_int("VIDEO_RUNNER_PORT", 8102),
        runner_timeout_s=_int("RUNNER_TIMEOUT_SECONDS", 1800),
        max_input_bytes=_int("MAX_INPUT_BYTES", 25 * 1024 * 1024),
        max_queue=_int("MAX_QUEUE", 100),
        job_ttl_s=_int("JOB_TTL_SECONDS", 24 * 3600),
        version=os.environ.get("GIT_SHA", "dev"),
    )
