import asyncio
import dataclasses

import pytest

from app import runner_manager
from app.config import load_settings
from app.errors import JobFailure
from app.runner_manager import RunnerManager


def manager(wait_s: int) -> RunnerManager:
    settings = dataclasses.replace(load_settings(), runner_mode="real", runner_ready_wait_s=wait_s)
    return RunnerManager(settings, http=None)


def test_job_waits_for_a_restarting_runner(monkeypatch):
    monkeypatch.setattr(runner_manager, "READY_POLL_S", 0.01)
    mgr = manager(wait_s=5)
    runner = mgr.runners["video"]
    runner.state = "crashed"

    async def scenario():
        async def reload():
            await asyncio.sleep(0.05)
            runner.state = "loading"
            await asyncio.sleep(0.05)
            runner.state = "ready"

        reloading = asyncio.create_task(reload())
        await mgr._wait_until_ready(runner)
        await reloading

    asyncio.run(scenario())
    assert runner.state == "ready"


def test_wait_gives_up_as_retryable_after_the_limit(monkeypatch):
    monkeypatch.setattr(runner_manager, "READY_POLL_S", 0.01)
    mgr = manager(wait_s=0)
    runner = mgr.runners["video"]
    runner.state, runner.error = "crashed", "runner process exited with code 3"

    with pytest.raises(JobFailure) as err:
        asyncio.run(mgr._wait_until_ready(runner))
    assert err.value.retryable is True
    assert "not ready yet" in str(err.value)
    assert "code 3" in str(err.value)
