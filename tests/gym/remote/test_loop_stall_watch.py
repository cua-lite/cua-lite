from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress

import pytest
from gym.remote.conftest import make_test_admission

from lite.gym.remote.server import State, _watch_loop_stalls


def _state() -> State:
    return State(admission=make_test_admission(max_live_envs=10), idle_ttl_sec=60.0)


def test_loop_stall_snapshot_carries_lifetime_and_recent_window() -> None:
    state = _state()
    state.record_loop_stall(4.0, observed_at=90.0, window_s=60.0)
    state.record_loop_stall(1.5, observed_at=155.0, window_s=60.0)

    snapshot = state.loop_stall_snapshot(now=160.0, window_s=60.0)

    assert snapshot["stall_max_seconds"] == 4.0
    assert snapshot["stall_max_at"] == 90.0
    assert snapshot["stall_max_60s"] == 1.5


@pytest.mark.asyncio
async def test_loop_stall_watchdog_records_warning(caplog) -> None:
    state = _state()
    caplog.set_level(logging.WARNING, logger="lite.gym.remote.server")

    task = asyncio.create_task(
        _watch_loop_stalls(
            state,
            interval_s=0.01,
            warn_after_s=0.01,
            window_s=1.0,
        )
    )
    await asyncio.sleep(0.02)
    time.sleep(0.05)
    await asyncio.sleep(0.03)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    snapshot = state.loop_stall_snapshot(window_s=1.0)
    assert snapshot["stall_max_seconds"] >= 0.02
    assert snapshot["stall_max_60s"] >= 0.02
    assert "event loop stalled" in caplog.text
