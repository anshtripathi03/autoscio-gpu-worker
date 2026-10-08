"""Request/response contract between the Autoscio backend and this worker.

Every file moves through S3 presigned URLs — the backend never sends media bytes
in a request body and this worker never returns them in a response.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

JobKind = Literal["tts", "t2v", "i2v"]

# What each kind may write, and the file extension the model runner writes it as.
OUTPUT_TYPES: dict[str, dict[str, str]] = {
    "tts": {"audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav"},
    "t2v": {"video/mp4": ".mp4"},
    "i2v": {"video/mp4": ".mp4"},
}


class JobInputs(BaseModel):
    # Presigned S3 GET URLs. Short-lived; downloaded once when the job starts.
    voiceSampleUrl: Optional[str] = None
    imageUrl: Optional[str] = None


class JobOutput(BaseModel):
    # Presigned S3 PUT URL the result is uploaded to. Must still be valid when the
    # job FINISHES (queue wait + render), so mint it with a long expiry (~2h).
    putUrl: str = Field(min_length=1)
    s3Key: str = Field(min_length=1, max_length=1024)
    contentType: str
    # Any extra headers the presigned PUT was signed with (e.g. SSE).
    headers: dict[str, str] = Field(default_factory=dict)


class WebhookSpec(BaseModel):
    url: str = Field(min_length=1)


class JobRequest(BaseModel):
    # The backend's own id for this GPU task. Submitting the same id twice returns
    # the existing job instead of rendering again, so retries are safe.
    taskId: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    kind: JobKind
    params: dict
    inputs: JobInputs = Field(default_factory=JobInputs)
    output: JobOutput
    webhook: Optional[WebhookSpec] = None


class TtsParams(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    # ISO 639-1 code supported by Chatterbox Multilingual (en, hi, es, fr, ...).
    language: str = Field(default="en", min_length=2, max_length=5)
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0)
    cfgWeight: float = Field(default=0.5, ge=0.0, le=1.0)


class VideoParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negativePrompt: Optional[str] = Field(default=None, max_length=1000)
    aspectRatio: Literal["9:16", "16:9", "1:1"] = "9:16"
    # Clamped by the runner to LTX_MAX_SECONDS.
    durationSeconds: float = Field(default=5.0, gt=0, le=60)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)


PARAMS_MODEL = {"tts": TtsParams, "t2v": VideoParams, "i2v": VideoParams}
