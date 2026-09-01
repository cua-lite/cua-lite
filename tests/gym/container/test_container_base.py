"""LiteContainerBase lifecycle contract tests.

The properties PoC-verified before the migration, kept as durable tests:
ABC-mixin + @dataclass composes; template destroy() is idempotent, never
raises, and dispatches an overridden `_pre_destroy`; a subclass missing
`start()` fails AT INSTANTIATION; a double-destroy never re-releases ports;
the five production containers all conform.

Run: uv run pytest tests/gym/container/test_container_base.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pytest

import lite.gym.container as cb
from lite.gym.container import _TRACKED, _TRACKED_LOCK, LiteContainerBase


@dataclass
class _Box(LiteContainerBase):
    name: str
    api_port: int
    _ports_owned: tuple[int, ...] = field(default=(), repr=False)

    rm_label: ClassVar[str] = "testbox"
    pre_destroy_calls: ClassVar[list[str]] = []

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.api_port}"

    def start(self) -> None:
        self._register()

    def _pre_destroy(self) -> None:
        _Box.pre_destroy_calls.append(self.name)


@pytest.fixture()
def patched(monkeypatch):
    rm_calls: list[tuple] = []
    released: list[tuple] = []
    monkeypatch.setattr(
        cb, "docker_rm_f",
        lambda name, *, timeout, label: rm_calls.append((name, timeout, label)) or 1,
    )
    monkeypatch.setattr(cb, "release_ports", lambda *ports: released.append(ports))
    monkeypatch.setattr(cb, "docker_container_gone", lambda _name: True)
    _Box.pre_destroy_calls = []
    yield rm_calls, released
    with _TRACKED_LOCK:
        _TRACKED[:] = [c for c in _TRACKED if not isinstance(c, _Box)]


def test_dataclass_mixin_composes_and_registers(patched):
    b = _Box(name="b1", api_port=1234, _ports_owned=(1234,))
    assert b.base_url == "http://localhost:1234"
    b.start()
    b.start()   # double-register is a no-op
    with _TRACKED_LOCK:
        assert _TRACKED.count(b) == 1


def test_destroy_template_order_and_idempotence(patched):
    rm_calls, released = patched
    b = _Box(name="b2", api_port=1235, _ports_owned=(1235, 1236))
    b.start()
    b.destroy()
    assert _Box.pre_destroy_calls == ["b2"], "hook runs before rm"
    assert rm_calls == [("b2", 60.0, "testbox")], \
        "rm routed through docker_rm_f with ClassVar knobs"
    assert released == [(1235, 1236)], "owned ports released"
    assert b._ports_owned == (), "ownership cleared after release"
    with _TRACKED_LOCK:
        assert b not in _TRACKED
    # Second destroy: rm again (idempotent op) but NEVER a second port release.
    b.destroy()
    assert released == [(1235, 1236)], "double-destroy must not re-release ports"


def test_pre_destroy_exception_never_blocks_teardown(patched, monkeypatch):
    rm_calls, _ = patched

    def boom(self):
        raise RuntimeError("diagnostic probe died")

    monkeypatch.setattr(_Box, "_pre_destroy", boom)
    b = _Box(name="b4", api_port=1238)
    b.destroy()   # must not raise
    assert rm_calls and rm_calls[-1][0] == "b4", "rm still ran after hook failure"


def test_destroy_retains_tracking_and_port_when_removal_is_unconfirmed(
    patched, monkeypatch,
):
    rm_calls, released = patched
    monkeypatch.setattr(
        cb, "docker_rm_f",
        lambda name, *, timeout, label: rm_calls.append((name, timeout, label)) or 0,
    )
    monkeypatch.setattr(cb, "docker_container_gone", lambda _name: False)
    b = _Box(name="still-running", api_port=1239, _ports_owned=(1239,))
    b.start()

    assert b.destroy() is False
    assert rm_calls[-1][0] == "still-running"
    assert released == []
    assert b._ports_owned == (1239,)
    with _TRACKED_LOCK:
        assert b in _TRACKED


def test_missing_start_fails_at_instantiation():
    @dataclass
    class _Broken(LiteContainerBase):
        name: str
        api_port: int
        _ports_owned: tuple[int, ...] = ()

        @property
        def base_url(self) -> str:
            return "http://x"
        # no start()

    with pytest.raises(TypeError):
        _Broken(name="nope", api_port=1)


def test_all_five_production_containers_conform():
    from lite.gym.envs.androidlab.container import AndroidLabContainer
    from lite.gym.envs.androidworld.container import AndroidWorldContainer
    from lite.gym.envs.mobileworld.container import MobileWorldContainer
    from lite.gym.envs.osworld.container import OSWorldContainer
    from lite.gym.envs.osworld_2.container import OSWorldV2Container

    for cls in (OSWorldContainer, OSWorldV2Container, AndroidWorldContainer,
                AndroidLabContainer, MobileWorldContainer):
        assert issubclass(cls, LiteContainerBase)
        c = cls(name=f"conform-{cls.__name__}", api_port=19999)
        assert c.base_url.startswith("http://"), cls.__name__
        assert c._ports_owned == (), cls.__name__
        assert cls.destroy is LiteContainerBase.destroy, (
            f"{cls.__name__} must not override the destroy() template (override _pre_destroy)"
        )


def test_every_lite_container_subclass_keeps_the_destroy_template():
    """6th-class gate: the conformance test above enumerates today's five
    containers by hand — a SIXTH env's container that overrides destroy()
    would sail past it. Walk ALL LiteContainerBase subclasses transitively
    (after importing the five production container modules) and auto-conscript
    every ``lite.*`` class into the template pin."""
    import lite.gym.envs.androidlab.container  # noqa: F401
    import lite.gym.envs.androidworld.container  # noqa: F401
    import lite.gym.envs.mobileworld.container  # noqa: F401
    import lite.gym.envs.osworld.container  # noqa: F401
    import lite.gym.envs.osworld_2.container  # noqa: F401

    def _walk(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from _walk(sub)

    lite_subs = sorted(
        {c for c in _walk(LiteContainerBase) if c.__module__.startswith("lite.")},
        key=lambda c: c.__qualname__,
    )
    assert len(lite_subs) >= 5, "expected at least the five production containers"
    for cls in lite_subs:
        assert cls.destroy is LiteContainerBase.destroy, (
            f"{cls.__module__}.{cls.__qualname__} overrides the destroy() template "
            f"(override _pre_destroy instead)"
        )


# ── boot_with_retry retry contract ─────────────────────────────────────────────────

def test_boot_with_retry_transient_then_success(patched):
    from lite.gym.container import boot_with_retry
    from lite.gym.errors import CapacityExhausted

    built, destroyed = [], []

    class _Flaky(_Box):
        def start(self):
            self._register()
            if len(built) == 1:
                raise CapacityExhausted.warming(what="boot deadline")

        def destroy(self2):  # record via base? _Box inherits template; count via hook
            destroyed.append(self2.name)
            return super().destroy()

    # NOTE: overriding destroy here is a TEST spy on the retry shell, not an
    # env pattern (envs must never override the template).
    def build():
        b = _Flaky(name=f"flaky-{len(built)}", api_port=1300 + len(built))
        built.append(b)
        return b

    c = boot_with_retry(build, start=lambda c: c.start(), max_attempts=2, label="t")
    assert c.name == "flaky-1", "second (fresh) build won"
    assert destroyed == ["flaky-0"], "failed attempt destroyed"


def test_boot_with_retry_does_not_overlap_unremoved_failed_attempt(patched):
    from lite.gym.container import boot_with_retry
    from lite.gym.errors import CapacityExhausted

    built = []

    class _Stuck(_Box):
        def start(self):
            raise RuntimeError("boot timed out")

        def destroy(self):
            return False

    def build():
        box = _Stuck(name=f"stuck-{len(built)}", api_port=1500 + len(built))
        built.append(box)
        return box

    with pytest.raises(CapacityExhausted, match="refusing to overlap"):
        boot_with_retry(build, start=lambda c: c.start(), max_attempts=3)
    assert len(built) == 1, "must not start a replacement before teardown is proven"


def test_boot_with_retry_exhausted_reraises_last(patched):
    from lite.gym.container import boot_with_retry

    class _AlwaysFails(_Box):
        def start(self):
            raise RuntimeError("docker run failed (exit 125)")

    with pytest.raises(RuntimeError, match="exit 125"):
        boot_with_retry(lambda: _AlwaysFails(name="af", api_port=1400),
                        start=lambda c: c.start(),
                        max_attempts=2, label="t")


def test_boot_with_retry_nontransient_no_retry(patched):
    from lite.gym.container import boot_with_retry

    calls = []

    class _HardFail(_Box):
        def start(self):
            calls.append(1)
            raise FileNotFoundError("docker binary missing")

    with pytest.raises(FileNotFoundError):
        boot_with_retry(lambda: _HardFail(name="hf", api_port=1401),
                        start=lambda c: c.start(),
                        max_attempts=3, label="t")
    assert calls == [1], "non-transient must not retry"


def test_boot_with_retry_gate_and_start_wrapper(patched):
    from lite.gym.container import boot_with_retry

    events = []

    class _Gate:
        def __enter__(self):
            events.append("gate+")
        def __exit__(self, *a):
            events.append("gate-")

    c = boot_with_retry(
        lambda: _Box(name="g", api_port=1402),
        start=lambda c: events.append("start"),
        attempt_gate=_Gate,
        max_attempts=1, label="t",
    )
    assert c.name == "g"
    assert events == ["gate+", "start", "gate-"], "gate held across the attempt"


# ── spawn_background_destroy callback contract ─────────────────────────────────────

def test_spawn_background_destroy_result_then_done():
    import concurrent.futures
    import threading

    from lite.gym.container import spawn_background_destroy

    cf = concurrent.futures.Future()
    cf.set_result("boxval")
    events: list = []
    done = threading.Event()
    spawn_background_destroy(
        cf,
        on_result=lambda res: events.append(("result", res)),
        on_done=lambda: (events.append("done"), done.set()),
        thread_name="t-bg-destroy-result",
    )
    assert done.wait(2.0), "background thread never reached on_done"
    assert events == [("result", "boxval"), "done"], (
        "on_result(result) must run first, on_done ALWAYS afterwards"
    )


def test_spawn_background_destroy_timeout_dispatches_failure_then_done():
    import concurrent.futures
    import threading

    from lite.gym.container import spawn_background_destroy

    cf = concurrent.futures.Future()   # never completes
    events: list = []
    done = threading.Event()
    spawn_background_destroy(
        cf,
        on_result=lambda res: events.append(("result", res)),
        on_failure=lambda exc: events.append(("failure", exc)),
        on_done=lambda: (events.append("done"), done.set()),
        timeout=0.05,
        thread_name="t-bg-destroy-timeout",
    )
    assert done.wait(2.0), "background thread never reached on_done"
    assert [e[0] if isinstance(e, tuple) else e for e in events] == ["failure", "done"]
    assert isinstance(events[0][1], (TimeoutError, concurrent.futures.TimeoutError))


def test_spawn_background_destroy_on_result_error_never_blocks_on_done(caplog):
    import concurrent.futures
    import logging
    import threading

    from lite.gym.container import spawn_background_destroy

    cf = concurrent.futures.Future()
    cf.set_result("boxval")
    done = threading.Event()

    def boom(res):
        raise RuntimeError("callback died")

    with caplog.at_level(logging.WARNING, logger="lite.gym.container"):
        spawn_background_destroy(
            cf,
            on_result=boom,
            on_done=done.set,
            thread_name="t-bg-destroy-boom",
        )
        # Nothing propagates (daemon thread), and on_done STILL fires.
        assert done.wait(2.0), "on_done must fire even when on_result raises"
    assert any("on_result callback failed" in r.message for r in caplog.records), (
        "the swallowed callback exception must stay observable via the warning log"
    )
