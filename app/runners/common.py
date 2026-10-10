"""Shared runner plumbing. Standard library ONLY: this module is imported from both
model venvs, which share nothing but the Python stdlib.

A runner loads its model once in a background thread, then serves
    GET  /health -> {"state": "loading"|"ready"|"error", "error": str|None}
    POST /run    -> {"ok": true, ...meta} | {"ok": false, "error": str, "retryable": bool}
on 127.0.0.1 only. Requests are handled one at a time, which is what we want on one GPU.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

# How long a runner whose model failed to load keeps reporting the error before it
# exits, so the supervisor has time to read the reason before restarting it.
LOAD_ERROR_LINGER_S = 30


class UserError(Exception):
    """Bad input (unsupported language, unreadable voice sample...). Not retryable."""


class ModelRunner:
    name = "runner"

    def load(self) -> None:
        raise NotImplementedError

    def run(self, payload: dict) -> dict:
        raise NotImplementedError


def is_oom(err: BaseException) -> bool:
    return "out of memory" in str(err).lower()


def serve(runner: ModelRunner) -> None:
    port = int(os.environ["RUNNER_PORT"])
    status = {"state": "loading", "error": None}

    def _load() -> None:
        started = time.time()
        try:
            runner.load()
            status["state"] = "ready"
            print(f"[{runner.name}] model ready in {time.time() - started:.0f}s", flush=True)
        except BaseException as err:  # noqa: BLE001
            traceback.print_exc()
            status["state"], status["error"] = "error", f"model failed to load: {err}"
            time.sleep(LOAD_ERROR_LINGER_S)
            os._exit(1)

    threading.Thread(target=_load, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, status)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/run":
                self._send(404, {"ok": False, "error": "not found"})
                return
            if status["state"] != "ready":
                self._send(
                    503, {"ok": False, "error": f"model {status['state']}", "retryable": True}
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = runner.run(payload)
                self._send(200, {"ok": True, **result})
            except UserError as err:
                self._send(400, {"ok": False, "error": str(err), "retryable": False})
            except BaseException as err:  # noqa: BLE001
                traceback.print_exc()
                message = (
                    "GPU ran out of memory — try a shorter clip or a lower LTX_LONG_EDGE"
                    if is_oom(err)
                    else f"{type(err).__name__}: {err}"
                )
                self._send(500, {"ok": False, "error": message, "retryable": True})

        def log_message(self, *args) -> None:
            pass  # the API process already logs every job

    print(f"[{runner.name}] listening on 127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def ffmpeg(src: str, dst: str, args: list[str]) -> None:
    """Convert media with ffmpeg; a failure here means the input file is bad."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, *args, dst],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise UserError(f"Could not read media file: {proc.stderr.strip()[:300]}")


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")


def split_text(text: str, max_chars: int) -> list[str]:
    """Split narration into chunks Chatterbox can say in one pass.

    Chatterbox stops after ~1000 speech tokens (~40s of audio) per call, so a 60s
    script must be generated in pieces. Splits on sentence ends (incl. the Hindi
    danda), then commas, then words, and packs pieces up to `max_chars`.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        for clause in re.split(r"(?<=[,;:])\s+", sentence):
            while len(clause) > max_chars:
                cut = clause.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                pieces.append(clause[:cut].strip())
                clause = clause[cut:].strip()
            if clause:
                pieces.append(clause)

    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return chunks


def round_to(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def video_dimensions(aspect_ratio: str, long_edge: int) -> tuple[int, int]:
    """(width, height), both multiples of 32 as LTX-Video requires."""
    long_px = round_to(long_edge, 32)
    if aspect_ratio == "9:16":
        return round_to(long_px * 9 / 16, 32), long_px
    if aspect_ratio == "16:9":
        return long_px, round_to(long_px * 9 / 16, 32)
    side = round_to(long_px * 0.75, 32)
    return side, side


def frames_for_duration(seconds: float, fps: int) -> int:
    """LTX-Video needs 8k+1 frames."""
    k = max(1, round((seconds * fps - 1) / 8))
    return 8 * k + 1


def main(runner: ModelRunner) -> None:
    try:
        serve(runner)
    except KeyboardInterrupt:
        sys.exit(0)
