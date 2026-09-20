"""``docker run`` pacing for the sandbox family: env-server only, never direct.

``SandboxBaseEnv._attempt_boot_computer`` holds the create slot only
``if self._env_id`` — i.e. only under an env-server — and
``docker_create_slot_async`` itself yields unpaced whenever no
:class:`AdmissionGate` has configured the semaphore. Both carve-outs are
DELIBERATE, and this module is the regression guard for them.

Direct mode deliberately has no host-derived default bound: it has no 503-retry
client, so a slot timeout there becomes a hard task failure rather than
back-pressure, and a direct-mode caller already bounds its own fan-out through
its own concurrency settings.

So the two things pinned here are:
  1. a direct-mode boot takes NO slot and installs NO semaphore;
  2. the env-server path still runs at exactly its configured capacity —
     neither below it (a throughput regression) nor above it.

Run: uv run pytest tests/gym/sandbox/test_create_pacing.py
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager

import pytest

from lite.gym.errors import CapacityExhausted
from lite.gym.remote import admission as admission_mod
from lite.gym.sandbox import SandboxBaseEnv, SandboxTaskConfig
from lite.gym.sandbox import base as sandbox_base
from lite.gym.utils.backend import freshness as freshness_mod

_TASK = SandboxTaskConfig(
    task_id="cb_create_pacing",
    instruction="smoke",
    platform="desktop",
    computer={
        "image": "lite.placeholder:latest",
        "display": "1920x1080",
        "memory": "4GB",
        "cpu": "1",
        "timeout": 120,
    },
    max_steps=1,
)

_BOOTS = 9


class _Peak:
    def __init__(self) -> None:
        self.now = 0
        self.peak = 0
        self._lock = threading.Lock()

    @asynccontextmanager
    async def counted(self):
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            yield
        finally:
            with self._lock:
                self.now -= 1


@pytest.fixture()
def boot_harness(monkeypatch):
    """A sandbox boot harness with no docker: metered provisioner, stub attach.

    Also clears the process-wide docker semaphore. ``monkeypatch`` restores it,
    which matters under xdist: a semaphore installed here must not leak into a
    sibling test module that builds an :class:`AdmissionGate` of its own.
    """
    peak = _Peak()

    class MeteredProvisioner:
        def __init__(self, name, cfg, **kwargs):
            self.name = name

        async def run(self):
            async with peak.counted():
                await asyncio.sleep(0.05)

    class FakeComputer:
        async def stop(self):
            return None

    async def fake_attach(_name, **_kwargs):
        return FakeComputer()

    async def fake_rm(name, *, timeout, label):
        return None

    # The env-server path (``env_id`` set) additionally runs an image-freshness
    # check, which this harness's placeholder image has no provider for. Patched
    # on the freshness module, not on ``sandbox_base``: ``_attempt_boot_computer``
    # imports ``image_for`` locally, inside the function.
    class _AlwaysFresh:
        def ensure_runnable(self):
            return None

    monkeypatch.setattr(freshness_mod, "image_for", lambda *_a, **_kw: _AlwaysFresh())
    monkeypatch.setattr(admission_mod, "_docker_sema", None)
    monkeypatch.setattr(admission_mod, "_docker_sema_capacity", 0)
    monkeypatch.setattr(sandbox_base, "DockerProvisioner", MeteredProvisioner)
    monkeypatch.setattr(sandbox_base, "attach", fake_attach)
    monkeypatch.setattr(sandbox_base, "docker_rm_f_async", fake_rm)
    return peak


async def _boot_many(n: int, *, env_id: str | None = None) -> list[SandboxBaseEnv]:
    envs = [SandboxBaseEnv(task=_TASK) for _ in range(n)]
    for e in envs:
        e._env_id = env_id
    assert all((e._env_id is None) == (env_id is None) for e in envs)
    await asyncio.wait_for(asyncio.gather(*(e.boot() for e in envs)), timeout=60)
    return envs


@pytest.mark.asyncio
async def test_direct_mode_boots_take_no_create_slot(boot_harness, monkeypatch):
    """9 concurrent direct-mode boots all run at once, and nothing is installed.

    This is the oracle validators' exact configuration: no
    :class:`AdmissionGate`, no ``env_id``. The assertion is the DIRECTIVE, not a
    performance preference — `direct不用节流`. ``derive_docker_create_concurrency``
    is stubbed to 1 so that a reintroduced bound would be unmissable (peak 1
    instead of 9) rather than hidden behind a large derived number.
    """
    peak = boot_harness
    monkeypatch.setattr(admission_mod, "derive_docker_create_concurrency",
                        lambda **_kw: 1)
    envs = await _boot_many(_BOOTS)
    try:
        assert peak.peak == _BOOTS, (
            f"{_BOOTS} concurrent direct-mode boots ran only {peak.peak} docker "
            "creates at once — something is throttling direct mode again"
        )
        assert peak.now == 0
        assert all(e._computer is not None for e in envs)
        # Nothing lazily installed: a direct-mode process must not acquire a
        # process-wide semaphore that a later AdmissionGate would inherit.
        assert admission_mod._docker_sema is None
        assert admission_mod._docker_sema_capacity == 0
    finally:
        await asyncio.gather(*(e.close() for e in envs))


@pytest.mark.asyncio
async def test_env_server_path_is_paced_at_its_configured_capacity(
    boot_harness, monkeypatch,
):
    """The env-server path IS paced, at exactly the gate-configured capacity.

    Both halves matter. Below it would reduce env-server throughput; above it
    would mean the pacing is not happening at all. Set up the way
    :func:`configure_docker_sema` leaves the module rather than by
    building an :class:`AdmissionGate`, whose install-once guard is process-wide
    global state this test has no business fighting.
    """
    peak = boot_harness
    capacity = 3
    monkeypatch.setattr(admission_mod, "_docker_sema", threading.Semaphore(capacity))
    monkeypatch.setattr(admission_mod, "_docker_sema_capacity", capacity)

    envs = await _boot_many(_BOOTS, env_id="lite.osworld")
    try:
        assert peak.peak == capacity, (
            f"configured capacity {capacity} but {peak.peak} creates ran "
            "concurrently under an env_id"
        )
        assert peak.now == 0
        assert all(e._computer is not None for e in envs)
    finally:
        await asyncio.gather(*(e.close() for e in envs))


@pytest.mark.asyncio
async def test_env_server_path_is_not_capped_below_its_configured_capacity(
    boot_harness, monkeypatch,
):
    """The env-server keeps its throughput: with a configured capacity of 6, six
    concurrent boots all run at once, and the derived default is NOT consulted."""
    peak = boot_harness
    capacity = 6
    monkeypatch.setattr(admission_mod, "_docker_sema", threading.Semaphore(capacity))
    monkeypatch.setattr(admission_mod, "_docker_sema_capacity", capacity)
    monkeypatch.setattr(admission_mod, "derive_docker_create_concurrency",
                        lambda **_kw: 1)   # would cap at 1 if it were consulted

    envs = await _boot_many(capacity, env_id="lite.osworld")
    try:
        assert peak.peak == capacity, (
            f"configured capacity {capacity} but only {peak.peak} creates ran "
            "concurrently — something is capping the env-server path"
        )
        assert admission_mod._docker_sema_capacity == capacity
    finally:
        await asyncio.gather(*(e.close() for e in envs))


@pytest.mark.asyncio
async def test_env_server_image_freshness_check_runs_off_loop(
    boot_harness, monkeypatch,
):
    calls = []

    async def fake_to_thread(fn, /, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    monkeypatch.setattr(sandbox_base.asyncio, "to_thread", fake_to_thread)
    envs = await _boot_many(1, env_id="lite.osworld")
    try:
        assert len(calls) == 1
        assert calls[0][0].__name__ == "_ensure_image_runnable"
    finally:
        await asyncio.gather(*(e.close() for e in envs))


@pytest.mark.asyncio
async def test_readiness_timeout_keeps_its_own_diagnosis(boot_harness, monkeypatch):
    """A ``TimeoutError`` from the boot must not be relabelled as slot exhaustion.

    Slot exhaustion is also a :exc:`CapacityExhausted`, so the readiness-marker
    timeout must keep its own text or an operator is sent after the wrong
    bottleneck. This module pins the boundary because it owns the pacing
    decision.
    """
    class TimingOutProvisioner:
        def __init__(self, name, cfg, **kwargs):
            pass

        async def run(self):
            raise TimeoutError("/tmp/gnome-ready never appeared")

    monkeypatch.setattr(sandbox_base, "DockerProvisioner", TimingOutProvisioner)
    env = SandboxBaseEnv(task=_TASK)
    with pytest.raises(CapacityExhausted) as excinfo:
        await env.boot()
    assert "container boot transient (TimeoutError)" in excinfo.value.what
    assert "slot" not in excinfo.value.what
