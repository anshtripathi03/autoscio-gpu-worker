"""S3 presigned-URL transfers and signed webhooks."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from .errors import JobFailure

log = logging.getLogger("worker.transfer")

# Tests swap this for an httpx.MockTransport before the app starts.
transport_override: Optional[httpx.AsyncBaseTransport] = None

_EXT_BY_TYPE = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "video/webm": ".webm",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

WEBHOOK_BACKOFF_S = (2, 5, 15)


def build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=transport_override,
        timeout=httpx.Timeout(120.0, connect=15.0),
        follow_redirects=False,
    )


def _safe_url(url: str) -> str:
    """Host + path only — presigned query strings carry credentials and never get logged."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _extension(content_type: str, url: str, default: str) -> str:
    ext = _EXT_BY_TYPE.get(content_type.split(";")[0].strip().lower())
    if ext:
        return ext
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if 1 < len(suffix) <= 5 else default


async def download(
    http: httpx.AsyncClient, url: str, dest_stem: Path, max_bytes: int, default_ext: str
) -> Path:
    """Stream a presigned GET to disk, enforcing a size cap. Returns the written path."""
    try:
        async with http.stream("GET", url) as res:
            if res.status_code != 200:
                raise JobFailure(
                    f"Could not download input {_safe_url(url)} (HTTP {res.status_code}) — "
                    "the presigned URL may have expired",
                    retryable=res.status_code >= 500 or res.status_code == 403,
                )
            dest = dest_stem.with_suffix(
                _extension(res.headers.get("content-type", ""), url, default_ext)
            )
            written = 0
            with dest.open("wb") as fh:
                async for chunk in res.aiter_bytes():
                    written += len(chunk)
                    if written > max_bytes:
                        raise JobFailure(
                            f"Input file is larger than {max_bytes // (1024 * 1024)} MB",
                            retryable=False,
                        )
                    fh.write(chunk)
            if written == 0:
                raise JobFailure("Input file is empty", retryable=False)
            return dest
    except httpx.HTTPError as err:
        raise JobFailure(f"Could not download input: {err}", retryable=True) from err


async def upload(
    http: httpx.AsyncClient, put_url: str, content_type: str, headers: dict, path: Path
) -> None:
    body = path.read_bytes()
    try:
        res = await http.put(
            put_url,
            content=body,
            headers={**headers, "Content-Type": content_type},
            timeout=httpx.Timeout(600.0, connect=15.0),
        )
    except httpx.HTTPError as err:
        raise JobFailure(f"Upload to S3 failed: {err}", retryable=True) from err
    if res.status_code not in (200, 201, 204):
        hint = (
            " — the presigned PUT URL probably expired, or was signed with different headers"
            if res.status_code == 403
            else ""
        )
        raise JobFailure(
            f"S3 rejected the upload (HTTP {res.status_code}){hint}",
            retryable=res.status_code in (403, 408, 429) or res.status_code >= 500,
        )


def sign(secret: str, timestamp: str, body: bytes) -> str:
    digest = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256)
    return "sha256=" + digest.hexdigest()


async def send_webhook(http: httpx.AsyncClient, url: str, secret: str, payload: dict) -> bool:
    """POST the job result, signed with the shared key. Retries; never raises.

    A missed webhook is not fatal: the backend also polls GET /jobs/{id}.
    """
    body = json.dumps(payload, separators=(",", ":")).encode()
    for attempt in range(len(WEBHOOK_BACKOFF_S) + 1):
        timestamp = str(int(time.time()))
        headers = {"Content-Type": "application/json", "X-Worker-Timestamp": timestamp}
        if secret:
            headers["X-Worker-Signature"] = sign(secret, timestamp, body)
        try:
            res = await http.post(url, content=body, headers=headers, timeout=30.0)
            if res.status_code < 300:
                return True
            # 4xx means the backend rejected it deliberately; retrying won't help.
            if 400 <= res.status_code < 500 and res.status_code not in (408, 429):
                log.warning("Webhook %s rejected with %s", _safe_url(url), res.status_code)
                return False
            log.warning("Webhook %s returned %s", _safe_url(url), res.status_code)
        except httpx.HTTPError as err:
            log.warning("Webhook %s failed: %s", _safe_url(url), err)
        if attempt < len(WEBHOOK_BACKOFF_S):
            await asyncio.sleep(WEBHOOK_BACKOFF_S[attempt])
    return False
