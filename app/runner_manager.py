"""Starts, supervises and calls the per-model runner processes.

Chatterbox pins transformers 5.x and LTX-Video requires transformers < 4.52, so
the two models cannot share a Python environment. Each runs in its own venv as a
long-lived subprocess that keeps its model loaded on the GPU and serves
127.0.0.1-only HTTP. This process talks to them; nothing outside the Pod can.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from .config import Settings
from .errors import JobFailure

log = logging.getLogger("worker.runners")

APP_ROOT = Path(__file__).resolve().parent.parent
RUNNER_FOR_KIND = {"tts": "tts", "t2v": "video", "i2v": "video"}
HEALTH_INTERVAL_S = 5
MAX_BACKOFF_S = 120


@dataclass
class Runner:
    name: str
    module: str
    python: str
    port: int
    state: str = "starting"  # starting | loading | ready | error | crashed
    error: Optional[str] = None
    restarts: int = 0
    proc: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)


class RunnerManager:
    def __init__(self, settings: Settings, http: httpx.AsyncClient):
        self.settings = settings
        self.http = http
        self.fake = settings.runner_mode == "fake"
        self._tasks: list[asyncio.Task] = []
        catalog = {
            "tts": Runner("tts", "app.runners.tts_runner", settings.tts_python, settings.tts_port),
            "video": Runner(
                "video", "app.runners.video_runner", settings.video_python, settings.video_port
            ),
        }
        self.runners = {name: catalog[name] for name in settings.enabled_runners if name in catalog}

    async def start(self) -> None:
        for runner in self.runners.values():
            if self.fake:
                runner.state = "ready"
            else:
                self._tasks.append(asyncio.create_task(self._supervise(runner)))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for runner in self.runners.values():
            if runner.proc and runner.proc.returncode is None:
                runner.proc.terminate()
                try:
                    await asyncio.wait_for(runner.proc.wait(), timeout=10)
                except TimeoutError:
                    runner.proc.kill()

    def status(self) -> dict:
        return {
            name: {"state": r.state, "error": r.error, "restarts": r.restarts}
            for name, r in self.runners.items()
        }

    def ready(self) -> bool:
        return bool(self.runners) and all(r.state == "ready" for r in self.runners.values())

    async def _supervise(self, runner: Runner) -> None:
        """Keep the runner process alive, restarting with backoff if it dies.

        A runner exits on purpose when its model fails to load (e.g. a Hugging Face
        download hiccup), so a restart is also the retry.
        """
        backoff = 5
        while True:
            env = {**os.environ, "RUNNER_PORT": str(runner.port), "PYTHONPATH": str(APP_ROOT)}
            python = runner.python if Path(runner.python).exists() else sys.executable
            log.info("Starting %s runner (%s -m %s)", runner.name, python, runner.module)
            runner.state, runner.error = "loading", None
            runner.proc = await asyncio.create_subprocess_exec(
                python, "-m", runner.module, cwd=str(APP_ROOT), env=env
            )
            waiter = asyncio.create_task(runner.proc.wait())
            while not waiter.done():
                await self._poll_health(runner)
                if runner.state == "ready":
                    backoff = 5
                await asyncio.wait({waiter}, timeout=HEALTH_INTERVAL_S)
            code = waiter.result()
            runner.state = "crashed"
            runner.error = runner.error or f"runner process exited with code {code}"
            runner.restarts += 1
            log.error("%s runner exited (%s); restarting in %ss", runner.name, code, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)

    async def _poll_health(self, runner: Runner) -> None:
        try:
            res = await self.http.get(f"http://127.0.0.1:{runner.port}/health", timeout=3.0)
            data = res.json()
            state = data.get("state", runner.state)
            if state != runner.state:
                log.info("%s runner: %s -> %s", runner.name, runner.state, state)
            runner.state = state
            runner.error = data.get("error")
        except (httpx.HTTPError, ValueError):
            pass  # still booting, or busy rendering (single-threaded); keep last state

    async def run(self, kind: str, payload: dict) -> dict:
        name = RUNNER_FOR_KIND[kind]
        runner = self.runners.get(name)
        if runner is None:
            raise JobFailure(f"The {name} model is not enabled on this worker", retryable=False)
        if self.fake:
            return _fake_run(kind, payload)
        if runner.state != "ready":
            raise JobFailure(
                f"The {name} model is not ready yet (state: {runner.state})", retryable=True
            )
        try:
            res = await self.http.post(
                f"http://127.0.0.1:{runner.port}/run",
                json=payload,
                timeout=httpx.Timeout(float(self.settings.runner_timeout_s), connect=5.0),
            )
        except httpx.TimeoutException as err:
            raise JobFailure(
                f"The {name} model took longer than {self.settings.runner_timeout_s}s",
                retryable=True,
            ) from err
        except httpx.HTTPError as err:
            raise JobFailure(f"The {name} model process is unreachable: {err}", True) from err
        try:
            data = res.json()
        except ValueError:
            data = {}
        if res.status_code != 200 or not data.get("ok"):
            raise JobFailure(
                data.get("error") or f"The {name} model failed (HTTP {res.status_code})",
                retryable=bool(data.get("retryable", res.status_code >= 500)),
            )
        return data


def _fake_run(kind: str, payload: dict) -> dict:
    """Placeholder output for CI and local API testing — no GPU, no model."""
    Path(payload["outputPath"]).write_bytes(b"fake-" + kind.encode())
    return {"ok": True, "durationSec": 1.0, "fake": True}
