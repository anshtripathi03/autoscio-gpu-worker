"""In-memory job queue. One GPU, so jobs run strictly one at a time, in order.

State lives in memory on purpose: if the Pod restarts, GET /jobs/{id} returns 404
and the backend resubmits. The backend's database is the source of truth.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx

from . import transfer
from .config import Settings
from .errors import JobFailure
from .runner_manager import RunnerManager
from .schemas import OUTPUT_TYPES, JobRequest
from .transfer import _extension

log = logging.getLogger("worker.jobs")


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


FINISHED = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


def _iso(ts: Optional[float]) -> Optional[str]:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts else None


@dataclass
class Job:
    request: JobRequest
    status: JobStatus = JobStatus.QUEUED
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    retryable: bool = False
    webhook_delivered: Optional[bool] = None

    @property
    def id(self) -> str:
        return self.request.taskId

    def view(self) -> dict:
        queued_ms = run_ms = None
        if self.started_at:
            queued_ms = int((self.started_at - self.created_at) * 1000)
            if self.finished_at:
                run_ms = int((self.finished_at - self.started_at) * 1000)
        return {
            "taskId": self.id,
            "kind": self.request.kind,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retryable": self.retryable,
            "createdAt": _iso(self.created_at),
            "startedAt": _iso(self.started_at),
            "finishedAt": _iso(self.finished_at),
            "timings": {"queuedMs": queued_ms, "runMs": run_ms},
        }


class QueueFull(Exception):
    pass


class JobService:
    def __init__(self, settings: Settings, runners: RunnerManager, http: httpx.AsyncClient):
        self.settings = settings
        self.runners = runners
        self.http = http
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.current: Optional[str] = None
        root = Path(settings.work_dir)
        self.jobs_dir = root / "jobs"
        self.uploads_dir = root / "uploads"
        self.results_dir = root / "results"
        for d in (self.jobs_dir, self.uploads_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    def queue_depth(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == JobStatus.QUEUED)

    def submit(self, request: JobRequest) -> tuple[Job, bool]:
        existing = self.jobs.get(request.taskId)
        if existing:
            return existing, False
        if self.queue_depth() >= self.settings.max_queue:
            raise QueueFull()
        job = Job(request=request, created_at=time.time())
        self.jobs[job.id] = job
        self.queue.put_nowait(job.id)
        return job, True

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    # Test mode: inputs uploaded to, and outputs kept on, the worker.

    def save_upload(self, filename: str, content_type: str, data: bytes) -> str:
        file_id = uuid4().hex
        ext = _extension(content_type, filename, ".bin")
        (self.uploads_dir / f"{file_id}{ext}").write_bytes(data)
        return file_id

    def uploaded_file(self, file_id: str) -> Optional[Path]:
        return next(self.uploads_dir.glob(f"{file_id}.*"), None)

    def result_file(self, job_id: str) -> Optional[Path]:
        return next(self.results_dir.glob(f"{job_id}.*"), None)

    def cancel(self, job_id: str) -> Optional[Job]:
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.finished_at = time.time()
        return job

    async def run_forever(self) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != JobStatus.QUEUED:
                continue
            self.current = job_id
            try:
                await self._process(job)
            finally:
                self.current = None

    async def _process(self, job: Job) -> None:
        req = job.request
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        workdir = self.jobs_dir / job.id
        workdir.mkdir(parents=True, exist_ok=True)
        log.info("Job %s (%s) started", job.id, req.kind)
        try:
            inputs: dict[str, str] = {}
            voice = await self._input(
                req.inputs.voiceSampleUrl, req.inputs.voiceSampleFileId, workdir / "voice", ".wav"
            )
            if voice:
                inputs["voiceSamplePath"] = voice
            image = await self._input(
                req.inputs.imageUrl, req.inputs.imageFileId, workdir / "image", ".jpg"
            )
            if image:
                inputs["imagePath"] = image

            ext = OUTPUT_TYPES[req.kind][req.output.contentType]
            output_path = workdir / f"output{ext}"
            meta = await self.runners.run(
                req.kind,
                {
                    "kind": req.kind,
                    "params": req.params,
                    "inputs": inputs,
                    "outputPath": str(output_path),
                },
            )
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise JobFailure("The model finished but wrote no output file", retryable=True)

            result = {
                "s3Key": req.output.s3Key,
                "contentType": req.output.contentType,
                "bytes": output_path.stat().st_size,
                "durationSec": meta.get("durationSec"),
                "meta": {k: v for k, v in meta.items() if k not in {"ok", "durationSec"}},
            }
            if req.output.putUrl:
                await transfer.upload(
                    self.http,
                    req.output.putUrl,
                    req.output.contentType,
                    req.output.headers,
                    output_path,
                )
            else:
                shutil.move(str(output_path), self.results_dir / f"{job.id}{ext}")
                result["downloadUrl"] = f"/jobs/{job.id}/output"
            job.result = result
            job.status = JobStatus.COMPLETED
        except JobFailure as err:
            job.status, job.error, job.retryable = JobStatus.FAILED, str(err), err.retryable
        except Exception as err:  # noqa: BLE001 — a job must never take the worker down
            log.exception("Job %s crashed", job.id)
            job.status, job.error, job.retryable = (
                JobStatus.FAILED,
                f"Unexpected worker error: {err}",
                True,
            )
        finally:
            job.finished_at = time.time()
            shutil.rmtree(workdir, ignore_errors=True)
        log.info("Job %s finished: %s %s", job.id, job.status.value, job.error or "")

        if req.webhook:
            job.webhook_delivered = await transfer.send_webhook(
                self.http, req.webhook.url, self.settings.api_key, job.view()
            )

    async def _input(
        self, url: Optional[str], file_id: Optional[str], stem: Path, default_ext: str
    ) -> Optional[str]:
        if file_id:
            src = self.uploaded_file(file_id)
            if src is None:
                raise JobFailure(
                    f"Unknown fileId {file_id}: upload it with POST /files first",
                    retryable=False,
                )
            # Copy so the runner's scratch files land in the job dir, not uploads/.
            dest = stem.with_suffix(src.suffix)
            shutil.copyfile(src, dest)
            return str(dest)
        if url:
            path = await transfer.download(
                self.http, url, stem, self.settings.max_input_bytes, default_ext
            )
            return str(path)
        return None

    async def cleanup_forever(self, interval_s: int = 600) -> None:
        while True:
            await asyncio.sleep(interval_s)
            cutoff = time.time() - self.settings.job_ttl_s
            for job_id in [
                j.id
                for j in self.jobs.values()
                if j.status in FINISHED and (j.finished_at or 0) < cutoff
            ]:
                self.jobs.pop(job_id, None)
            for folder in (self.uploads_dir, self.results_dir):
                for f in folder.iterdir():
                    if f.stat().st_mtime < cutoff:
                        f.unlink(missing_ok=True)
