"""The event-loop stall watchdog.

A wedged env-server reports healthy CPU, RAM and thread counts; the only thing
wrong is that the loop is not running, and while it is not running the server
cannot answer ``/host_status`` either. These tests pin the two properties that
make the WARNING in :func:`~lite.gym.remote.server._watch_loop_stalls` the
surviving evidence: it fires for a real stall, and it stays quiet otherwise.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from lite.gym.remote import server as srv


class _FakeState:
    """Only the two fields the watcher writes."""

    def __init__(self) -> None:
        self.loop_stall_max_s = 0.0
        self.loop_stall_max_at: float | None = None


async def _run_watcher_for(state, seconds: float, block: float = 0.0):
    task = asyncio.create_task(srv._watch_loop_stalls(state))
    if block:
        # Block the loop the way a GIL-holding metric did: a synchronous sleep
        # inside the loop thread, which the watcher cannot preempt.
        await asyncio.sleep(srv._STALL_POLL_S)
        time.sleep(block)
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_blocking_the_loop_is_recorded_and_logged(caplog):
    state = _FakeState()
    block = srv._STALL_WARN_S + 1.0
    with caplog.at_level(logging.WARNING, logger=srv.logger.name):
        await _run_watcher_for(state, seconds=1.0, block=block)

    assert state.loop_stall_max_s >= block * 0.9, state.loop_stall_max_s
    assert state.loop_stall_max_at is not None
    assert any("event loop stalled" in r.message for r in caplog.records), caplog.records


@pytest.mark.asyncio
async def test_a_healthy_loop_logs_nothing(caplog):
    state = _FakeState()
    with caplog.at_level(logging.WARNING, logger=srv.logger.name):
        await _run_watcher_for(state, seconds=2.0)

    assert state.loop_stall_max_s < 1.0, state.loop_stall_max_s
    assert not [r for r in caplog.records if "event loop stalled" in r.message]


@pytest.mark.asyncio
async def test_the_watcher_survives_being_cancelled_mid_sleep():
    """Lifespan cancels it alongside the reapers; it must not swallow that."""
    state = _FakeState()
    task = asyncio.create_task(srv._watch_loop_stalls(state))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_host_status_reports_the_worst_stall_seen():
    """The recorded stall must reach the operator surface, not just the State.

    Greping the source for the dict key would pass against any build that merely
    mentions it, so this reads the value back through the endpoint itself.
    """
    from fastapi.testclient import TestClient

    from lite.gym.remote.server import State, make_app
    from tests.gym.remote.conftest import make_test_admission

    state = State(admission=make_test_admission(max_live_envs=1), idle_ttl_sec=3600.0)
    state.loop_stall_max_s = 12.5
    state.loop_stall_max_at = 1_700_000_000.0

    with TestClient(make_app(state, token="t")) as client:
        body = client.get("/host_status", headers={"Authorization": "Bearer t"}).json()

    assert body["loop"]["stall_max_seconds"] == 12.5
    assert body["loop"]["stall_max_at"] == 1_700_000_000.0
