"""Metric isolation: a runaway third-party metric must not starve the env-server.

The regression this guards is measured, not hypothetical. One ``compare_audios`` over a
366-second track (8_070_912 samples) held the GIL for 15+ minutes inside a pure-Python
``fastdtw``. Because the metric ran on the loop's default THREAD executor, the env-server's
asyncio loop could not get the GIL to run any bytecode and stopped answering EVERY request —
twice measured, 15 minutes silent with 100+ connections queued in the accept backlog, every
in-flight client past its deadline, a whole eval batch lost. A thread cannot be cancelled either,
so no timeout could be enforced against it.

``runner`` now runs metrics in a forkserver child under a deadline. These pin both halves.
"""
import asyncio
import time

import pytest

from lite.gym.envs.lite.osworld.src.eval import runner

_FIXTURES = "tests.gym.envs.lite.osworld.metric_fixtures"


@pytest.fixture(autouse=True)
def _resolve_metrics_from_fixtures(monkeypatch):
    monkeypatch.setattr(runner, "_METRIC_MODULES", (_FIXTURES,))
    monkeypatch.setattr(runner, "_metric_ctx_cached", None)


@pytest.mark.asyncio
async def test_loop_stays_live_while_a_metric_burns_cpu(monkeypatch):
    monkeypatch.setattr(runner, "_METRIC_TIMEOUT_S", 60.0)
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    t0 = time.monotonic()
    score = await runner._run_metric_isolated("spin", 2.0, None, {})
    elapsed = time.monotonic() - t0
    hb.cancel()

    assert score == 1.0
    assert elapsed >= 2.0
    # ~40 ticks fit in 2s. Pre-fix this was 0 — the loop never got the GIL.
    assert ticks > 10, f"event loop was starved: {ticks} ticks in {elapsed:.1f}s"


@pytest.mark.asyncio
async def test_runaway_metric_is_killed_on_deadline(monkeypatch):
    monkeypatch.setattr(runner, "_METRIC_TIMEOUT_S", 2.0)
    t0 = time.monotonic()
    with pytest.raises(runner.MetricTimeout):
        await runner._run_metric_isolated("spin", 600.0, None, {})
    assert time.monotonic() - t0 < 15.0, "deadline not enforced"


@pytest.mark.asyncio
async def test_child_exception_surfaces_as_runtime_error(monkeypatch):
    monkeypatch.setattr(runner, "_METRIC_TIMEOUT_S", 30.0)
    with pytest.raises(RuntimeError, match="ValueError"):
        await runner._run_metric_isolated("explode", None, None, {})


@pytest.mark.asyncio
async def test_cancelling_the_step_kills_the_child(monkeypatch):
    """The final eval runs inside ``step()`` under ``asyncio.wait_for(step_timeout)``.

    Cancellation has to reach the metric: a wait that ignores it leaves the child burning a core
    behind a step that already failed, and (when the wait parks a pool thread) delays graceful
    shutdown by the whole deadline. Both were measured before this was an async poll.
    """
    import multiprocessing

    monkeypatch.setattr(runner, "_METRIC_TIMEOUT_S", 600.0)
    task = asyncio.create_task(runner._run_metric_isolated("spin", 600.0, None, {}))
    await asyncio.sleep(3)  # let the child actually start
    assert multiprocessing.active_children(), "child never started; test proves nothing"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(100):  # the finally-block kill is synchronous, but reaping is not instant
        if not multiprocessing.active_children():
            break
        await asyncio.sleep(0.1)
    assert not multiprocessing.active_children(), "child outlived the cancelled step"


@pytest.mark.asyncio
async def test_metric_deadline_stays_under_the_step_timeout():
    """A deadline above ``step_timeout`` is unreachable — the step fails first."""
    import yaml

    for cfg in (
        "lite/gym/envs/lite/osworld/configs/default.yaml",
        "lite/gym/envs/lite/scalecua/configs/default.yaml",
    ):
        with open(cfg) as fh:
            step_timeout = yaml.safe_load(fh)["make_kwargs"]["step_timeout"]
        assert runner._METRIC_TIMEOUT_S < step_timeout, (
            f"metric deadline {runner._METRIC_TIMEOUT_S}s >= {cfg} step_timeout {step_timeout}s: "
            "MetricTimeout would never fire"
        )


@pytest.mark.asyncio
async def test_killing_a_metric_never_signals_our_own_process_group(monkeypatch):
    """The deadline kill must not be able to take the env-server down with it.

    ``_kill_metric_child`` signals the child's process GROUP so a metric that shelled out to
    ``soffice`` cannot leave a grandchild behind. But a forkserver child only joins its own group
    once it reaches ``os.setsid()`` — which happens *after* ``spawn.prepare`` re-executes the
    caller's ``__main__``, ~0.33s with an env-server-shaped one. Kill inside that window and
    ``getpgid(child)`` is still OURS: signalling it killed the parent outright (reproduced,
    exit 143). This pins the guard: whatever the timing, our own group is never the target.
    """
    import os

    our_pgid = os.getpgid(0)
    killed_groups = []
    real_killpg = os.killpg

    def spy_killpg(pgid, sig):
        killed_groups.append(pgid)
        assert pgid != our_pgid, f"killpg targeted OUR OWN group {pgid} — this kills the server"
        return real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", spy_killpg)
    monkeypatch.setattr(runner, "_METRIC_TIMEOUT_S", 600.0)

    # Cancel at several offsets so at least one lands in the pre-setsid window.
    for delay in (0.0, 0.05, 0.15, 0.35):
        task = asyncio.create_task(runner._run_metric_isolated("spin", 600.0, None, {}))
        await asyncio.sleep(delay)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert our_pgid not in killed_groups


# ---------------------------------------------------------------------------
# Scratch-dir ownership
# ---------------------------------------------------------------------------
# The eval scratch dir had a mkdtemp with no matching cleanup: measured on a long-lived host,
# 92,275 /tmp/osworld_eval_* dirs holding 88 GB, oldest two months old, on the filesystem
# dockerd and the checkpoints share. Cleanup has to stay ownership-scoped, because
# lite.scalecua passes its OWN cache_dir and keeps reading it after this returns.


@pytest.mark.asyncio
async def test_own_scratch_dir_is_removed(monkeypatch, tmp_path):
    import glob
    import os

    from lite.gym.envs.lite.osworld.src.eval import runner as r

    monkeypatch.setattr(__import__("tempfile"), "mkdtemp",
                        lambda prefix="": str(tmp_path / "owned"))
    os.makedirs(tmp_path / "owned", exist_ok=True)
    (tmp_path / "owned" / "artifact.bin").write_bytes(b"x")

    await r.evaluate_osworld_task(None, {"func": "infeasible"})
    assert not glob.glob(str(tmp_path / "owned")), "a scratch dir we created must not survive"


@pytest.mark.asyncio
async def test_caller_supplied_scratch_dir_is_left_alone(tmp_path):
    from lite.gym.envs.lite.osworld.src.eval import runner as r

    theirs = tmp_path / "theirs"
    theirs.mkdir()
    (theirs / "artifact.bin").write_bytes(b"x")

    await r.evaluate_osworld_task(None, {"func": "infeasible"}, cache_dir=str(theirs))
    assert (theirs / "artifact.bin").exists(), "lite.scalecua still reads the dir it passed in"


@pytest.mark.asyncio
async def test_debug_run_keeps_its_artifacts(monkeypatch, tmp_path):
    import os

    from lite.gym.envs.lite.osworld.src.eval import runner as r

    monkeypatch.setattr(__import__("tempfile"), "mkdtemp",
                        lambda prefix="": str(tmp_path / "dbg"))
    os.makedirs(tmp_path / "dbg", exist_ok=True)
    (tmp_path / "dbg" / "artifact.bin").write_bytes(b"x")

    await r.evaluate_osworld_task(None, {"func": "infeasible"}, debug=True)
    assert (tmp_path / "dbg" / "artifact.bin").exists(), "debug exists to keep the artifacts"
