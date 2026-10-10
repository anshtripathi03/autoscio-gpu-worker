"""Autoscio GPU worker — job API in front of the Chatterbox and LTX-Video runners.

POST   /jobs              submit a job (returns immediately; result arrives by webhook)
GET    /jobs/{id}         job status (the backend's fallback when a webhook is missed)
DELETE /jobs/{id}         cancel a job that has not started
GET    /health            liveness + whether the models are loaded (no auth)
Test mode (no S3):
POST   /files             upload a voice sample / image, returns a fileId
GET    /jobs/{id}/output  download a result that was not sent to S3
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from . import transfer
from .config import load_settings
from .jobs import JobService, QueueFull
from .runner_manager import RunnerManager
from .schemas import OUTPUT_TYPES, PARAMS_MODEL, JobRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs every request at INFO — the 5-second runner health polls would bury the
# lines that matter. Runner state changes are logged by RunnerManager instead.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("worker")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    if not settings.api_key and not settings.allow_no_auth:
        raise RuntimeError("WORKER_API_KEY is not set. Refusing to start an open GPU worker.")
    http = transfer.build_client()
    runners = RunnerManager(settings, http)
    await runners.start()
    jobs = JobService(settings, runners, http)
    tasks = [asyncio.create_task(jobs.run_forever()), asyncio.create_task(jobs.cleanup_forever())]
    app.state.settings, app.state.runners, app.state.jobs = settings, runners, jobs
    log.info("Worker %s up; runners: %s", settings.version, ", ".join(runners.runners) or "none")
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await runners.stop()
        await http.aclose()


app = FastAPI(title="Autoscio GPU worker", lifespan=lifespan, docs_url=None, redoc_url=None)


def require_auth(request: Request, authorization: Optional[str] = Header(default=None)) -> None:
    settings = request.app.state.settings
    if not settings.api_key and settings.allow_no_auth:
        return
    expected = f"Bearer {settings.api_key}".encode()
    if not authorization or not hmac.compare_digest(authorization.encode(), expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
async def health(request: Request) -> dict:
    runners: RunnerManager = request.app.state.runners
    jobs: JobService = request.app.state.jobs
    return {
        "ok": True,
        "ready": runners.ready(),
        "runners": runners.status(),
        "queueDepth": jobs.queue_depth(),
        "running": jobs.current,
        "version": request.app.state.settings.version,
    }


@app.post("/jobs", status_code=202, dependencies=[Depends(require_auth)])
async def submit_job(body: JobRequest, request: Request, response: Response) -> dict:
    allowed = OUTPUT_TYPES[body.kind]
    if body.output.contentType not in allowed:
        raise HTTPException(
            422, f"output.contentType for {body.kind} must be one of {sorted(allowed)}"
        )
    try:
        PARAMS_MODEL[body.kind].model_validate(body.params)
    except ValidationError as err:
        raise HTTPException(422, f"Invalid params for {body.kind}: {err.errors()}") from err
    if body.kind == "i2v" and not (body.inputs.imageUrl or body.inputs.imageFileId):
        raise HTTPException(422, "i2v jobs need inputs.imageUrl or inputs.imageFileId")

    jobs: JobService = request.app.state.jobs
    try:
        job, created = jobs.submit(body)
    except QueueFull as err:
        raise HTTPException(429, "Worker queue is full; retry later") from err
    if not created:
        response.status_code = 200  # same taskId already submitted — idempotent
    view = job.view()
    view["queuePosition"] = jobs.queue_depth() if created else None
    return view


@app.post("/files", status_code=201, dependencies=[Depends(require_auth)])
async def upload_file(request: Request, file: UploadFile = File(...)) -> dict:
    """Test mode: store an input on the worker and get a fileId for inputs.*FileId."""
    limit = request.app.state.settings.max_input_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {limit // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(422, "File is empty")
    jobs: JobService = request.app.state.jobs
    file_id = jobs.save_upload(file.filename or "", file.content_type or "", data)
    return {"fileId": file_id, "bytes": len(data)}


@app.get("/jobs/{job_id}/output", dependencies=[Depends(require_auth)])
async def job_output(job_id: str, request: Request) -> FileResponse:
    """Test mode: the result of a job submitted without output.putUrl."""
    jobs: JobService = request.app.state.jobs
    job = jobs.get(job_id)
    path = jobs.result_file(job_id)
    if job is None or path is None or not job.result:
        raise HTTPException(404, "No stored output for this job (not finished, or sent to S3)")
    return FileResponse(path, media_type=job.result["contentType"], filename=path.name)


@app.get("/jobs/{job_id}", dependencies=[Depends(require_auth)])
async def get_job(job_id: str, request: Request) -> dict:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job (it may have been lost in a worker restart)")
    return job.view()


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_auth)])
async def cancel_job(job_id: str, request: Request) -> dict:
    jobs: JobService = request.app.state.jobs
    job = jobs.cancel(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    if job.status.value == "RUNNING":
        raise HTTPException(409, "Job is already running and cannot be cancelled")
    return job.view()
