"""Unit tests for ContainerReaper boot-vs-steady reap.

Mocks the docker subprocess helpers so no real docker is needed. Run:
``uv run pytest tests/gym/remote/test_reaper.py``.
"""

from __future__ import annotations

import subprocess
import time

import pytest

from lite.gym.remote import reaper as cr
from lite.gym.remote.reaper import MAX_ORPHANS_PER_CYCLE, ContainerReaper, _DockerPsScan
from lite.gym.remote.scope import ServerScope

SCOPE = ServerScope(server_port=30100, token_hash=None)


def _patch(monkeypatch, scan: _DockerPsScan):
    rmd: list[str] = []
    monkeypatch.setattr(cr, "_docker_ps_with_time", lambda *a, **k: scan)
    monkeypatch.setattr(cr, "_prune_quarantine", lambda **k: None)
    monkeypatch.setattr(cr, "_try_orphan_rm", lambda name, t: (rmd.append(name) or True))
    monkeypatch.setattr(cr.time, "sleep", lambda *_a: None)
    return rmd


def test_boot_reap_is_widenet_no_age_guard_no_cap(monkeypatch):
    """A boot reap (framework-supplied boot=True) reaps EVERY in-scope container
    regardless of age and with no per-cycle cap — matching the old reap_zombies."""
    now = time.time()
    # 5 YOUNG containers (created "now"): a steady reap would age-guard them out.
    scan = _DockerPsScan(all=tuple((f"c{i}", now) for i in range(5)))
    rmd = _patch(monkeypatch, scan)
    r = ContainerReaper()
    n = r.reap("e", SCOPE, set(), boot=True)
    assert n == 5, "boot must reap all 5 young containers (wide-net)"
    assert len(rmd) == 5


def test_steady_reap_age_guards_and_caps(monkeypatch):
    """A steady reap (boot=False) applies the orphan age guard +
    MAX_ORPHANS_PER_CYCLE cap."""
    now = time.time()
    young = tuple((f"y{i}", now) for i in range(4))            # < 600s → spared
    old = tuple((f"o{i}", now - 10_000) for i in range(8))     # old → reapable, capped
    scan = _DockerPsScan(all=young + old)
    rmd = _patch(monkeypatch, scan)
    r = ContainerReaper()
    n = r.reap("e", SCOPE, set(), boot=False)
    assert n == MAX_ORPHANS_PER_CYCLE, f"steady cap not applied: {n}"
    assert all(name.startswith("o") for name in rmd), "young (age-guarded) reaped"


def test_steady_reap_spares_in_use(monkeypatch):
    """Containers backing live instances (in_use) are never reaped."""
    now = time.time()
    scan = _DockerPsScan(
        all=tuple((f"o{i}", now - 10_000) for i in range(3)),
    )
    rmd = _patch(monkeypatch, scan)
    r = ContainerReaper()
    n = r.reap("e", SCOPE, in_use={"o0", "o1"}, boot=False)
    assert n == 1 and rmd == ["o2"], f"only the untracked orphan reaped: {rmd}"


def test_boot_and_steady_are_stateless_across_calls(monkeypatch):
    """The reaper holds NO boot latch — boot vs steady is decided per call by the
    framework flag, NOT by call order. The same reaper instance does a wide-net
    boot reap and a guarded steady reap with no shared/mutated state between."""
    now = time.time()
    r = ContainerReaper()
    # A young container under boot=True is wide-net reaped...
    rmd = _patch(monkeypatch, _DockerPsScan(all=(("young", now),)))
    assert r.reap("e", SCOPE, set(), boot=True) == 1 and rmd == ["young"]
    # ...and the SAME instance spares an equally-young container under boot=False
    # (no latch was consumed; the flag alone governs).
    rmd = _patch(monkeypatch, _DockerPsScan(all=(("young2", now),)))
    assert r.reap("e", SCOPE, set(), boot=False) == 0 and rmd == []


def test_recover_drives_boot_widenet_even_when_steady_oracle_would_fail(monkeypatch):
    """REGRESSION: boot recovery via reaper.recover() must run wide-net (boot=True)
    independent of the live_ids ghost oracle — recover() calls reap directly with
    boot=True, bypassing the live_ids gate that a docker hiccup fails closed."""
    from lite.gym.remote.reconcile import recover
    now = time.time()
    # A YOUNG container exists at boot; live_ids (the ghost oracle) fails closed
    # to None, but recover bypasses it and still wide-net reaps the young leftover.
    rmd = _patch(monkeypatch, _DockerPsScan(all=(("young", now),)))
    monkeypatch.setattr(cr, "_docker_ps_running_or_none", lambda *a, **k: None)
    r = ContainerReaper()
    assert recover(r, "e", SCOPE) == 1, "boot recovery must wide-net reap the young leftover"
    assert rmd == ["young"]


def test_singleton_container_services_reap_boot_only_and_shutdown(monkeypatch):
    """SingletonContainerServices gives a SINGLETON env boot-only reap +
    shutdown (both docker rm -f the one container_name, with the subclass's rm_timeout_s/
    rm_label) and inherits live_ids -> None. A subclass declares only the name."""
    from lite.gym.remote.reaper import SingletonContainerServices
    calls: list[tuple] = []
    monkeypatch.setattr(
        cr, "docker_rm_f",
        lambda name, *, timeout, label: (calls.append((name, timeout, label)) or 1),
    )

    class _Svc(SingletonContainerServices):
        rm_timeout_s = 12.0
        rm_label = "myenv"
        def container_name(self, scope):
            return f"myenv-{scope.server_port}"

    svc = _Svc()
    name = f"myenv-{SCOPE.server_port}"
    # Steady tick (boot=False) is a no-op — never reap the shared backend mid-run.
    assert svc.reap("e", SCOPE, set(), boot=False) == 0
    assert calls == []
    # Boot reap docker-rm's the one container with the subclass's timeout + label.
    assert svc.reap("e", SCOPE, set(), boot=True) == 1
    assert calls == [(name, 12.0, "myenv")]
    # shutdown docker-rm's it too.
    svc.shutdown("e", SCOPE)
    assert calls[-1] == (name, 12.0, "myenv")
    # live_ids inherited -> None ("no per-instance world"; a SINGLETON is never reconciled).
    assert svc.live_ids("e", SCOPE) is None


def test_singleton_container_services_requires_container_name():
    """container_name is a REAL abstractmethod (section A3): forgetting it fails at
    instantiation (TypeError), not at the first reap."""
    from lite.gym.remote.reaper import SingletonContainerServices
    with pytest.raises(TypeError):
        SingletonContainerServices()

    class _NoName(SingletonContainerServices):
        pass

    with pytest.raises(TypeError):
        _NoName()


def test_live_ids_fail_closed_raises(monkeypatch):
    """live_ids RAISES ReconcileProbeError when the docker ps probe fails (fail closed).
    The reconcile loop catches it → skips the cycle, byte-identical
    to the old None-return. None is no longer used for probe failure."""
    from lite.gym.errors import ReconcileProbeError
    monkeypatch.setattr(cr, "_docker_ps_running_or_none", lambda *a, **k: None)
    r = ContainerReaper()
    with pytest.raises(ReconcileProbeError):
        r.live_ids("e", SCOPE)
    # And a successful empty scan is a real empty set, never a raise.
    monkeypatch.setattr(cr, "_docker_ps_running_or_none", lambda *a, **k: set())
    assert r.live_ids("e", SCOPE) == set()


def test_live_ids_refuses_unscoped_scope(monkeypatch):
    from lite.gym.errors import ReconcileProbeError

    def _boom(*a, **k):
        raise AssertionError("docker must not be consulted on an unscoped probe")

    monkeypatch.setattr(cr, "_docker_ps_running_or_none", _boom)
    r = ContainerReaper()
    with pytest.raises(ReconcileProbeError, match="UNSCOPED"):
        r.live_ids("e", ServerScope(server_port=None))


def test_reap_refuses_unscoped_scope(monkeypatch, caplog):
    """server_port=None → the name prefix degrades to bare ``lite-env-`` and a
    sweep would rm EVERY server's containers host-wide (observed in the wild
    via a TestClient lifespan in another checkout).
    reap() must refuse loudly and never touch docker."""
    import logging

    from lite.gym.remote import reaper as cr
    from lite.gym.remote.scope import ServerScope

    def _boom(*a, **k):
        raise AssertionError("docker must not be consulted on an unscoped sweep")

    monkeypatch.setattr(cr, "_docker_ps_with_time", _boom)
    monkeypatch.setattr(cr, "_try_orphan_rm", _boom)
    reaper = cr.ContainerReaper()
    with caplog.at_level(logging.ERROR, logger="lite.gym.remote.reaper"):
        n = reaper.reap("lite.osworld", ServerScope(server_port=None), set(), boot=True)
    assert n == 0
    assert any("UNSCOPED" in r.message for r in caplog.records)
    # steady-state path refuses too
    assert reaper.reap("lite.osworld", ServerScope(server_port=None), set(), boot=False) == 0


def test_env_facing_surface_is_the_two_services_classes() -> None:
    """The module docstring's "why this stays under ``remote/``" argument rests
    on a measurement — that what envs import here is the ``EnvServices`` pair,
    never the docker primitives. Keep the measurement executable: if a primitive
    (or a whole-module alias, which hides the reach from a name grep) ever
    escapes into ``lite/``, the KEEP rationale is dead and this says so instead
    of the docstring quietly aging into fiction.
    """
    import ast
    from pathlib import Path

    from lite.utils.path import project_root

    root = project_root()
    module = "lite.gym.remote.reaper"
    public = {"ContainerServices", "SingletonContainerServices"}

    # Same exclusions as tests/static/test_tier_dag.py's walk: per-env task caches and
    # the vendored osworld generator are not this tree's source (and some do not
    # even parse).
    skip = (".cache", "__pycache__", ".venv", "lite/gym/envs/lite/osworld/src")

    sites: dict[str, set[str]] = {}
    aliases: list[str] = []
    for path in sorted((root / "lite").glob("**/*.py")):
        rel = Path(path).relative_to(root).as_posix()
        if any(part in rel for part in skip) or rel == "lite/gym/remote/reaper.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == module:
                sites.setdefault(rel, set()).update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module == "lite.gym.remote":
                if any(a.name == "reaper" for a in node.names):
                    aliases.append(rel)
            elif isinstance(node, ast.Import):
                if any(a.name == module for a in node.names):
                    aliases.append(rel)

    assert sites, "found no importer at all — the scan is broken, not the tree"
    leaked = {rel: names - public for rel, names in sites.items() if names - public}
    assert not leaked, (
        f"only {sorted(public)} are this module's env-facing surface; "
        f"the docker primitives are module-private: {leaked}"
    )
    assert not aliases, (
        "whole-module import of the reaper hides which names are reached "
        f"(and from a name grep): {sorted(set(aliases))}"
    )


# ---------------------------------------------------------------------------
# _try_orphan_rm: what the docker exit code means
#
# The tests above stub _try_orphan_rm out, so nothing above reaches the rm
# itself. `rc != 0` is the daemon REFUSING the removal -- the container is still
# running -- and it used to return True, which counted it as reaped.
# ---------------------------------------------------------------------------

@pytest.fixture
def _clean_quarantine():
    cr._RM_QUARANTINE.clear()
    yield cr._RM_QUARANTINE
    cr._RM_QUARANTINE.clear()


def _fake_rm(monkeypatch, returncode: int, stderr: str = ""):
    def run(argv, **kw):
        return subprocess.CompletedProcess(argv, returncode, stdout="", stderr=stderr)
    monkeypatch.setattr(cr.subprocess, "run", run)


def test_a_refused_rm_is_not_reaped_and_gets_quarantined(monkeypatch, _clean_quarantine):
    _fake_rm(monkeypatch, 1, "device or resource busy")
    assert cr._try_orphan_rm("c0", 10.0) is False, "rc != 0 means the container is still there"
    next_retry, tier = _clean_quarantine["c0"]
    assert tier == 0 and next_retry > time.time(), "first failure enters the 60s tier"


def test_a_refused_rm_escalates_the_same_tiers_as_a_timeout(monkeypatch, _clean_quarantine):
    """Both failures share one backoff ladder: the name is unremovable either way."""
    _fake_rm(monkeypatch, 1)
    for expected_tier in range(len(cr._RM_QUARANTINE_BACKOFF_S)):
        cr._RM_QUARANTINE["c0"] = (0.0, expected_tier - 1)  # due now, at the prior tier
        assert cr._try_orphan_rm("c0", 10.0) is False
        assert _clean_quarantine["c0"][1] == expected_tier
    cr._RM_QUARANTINE["c0"] = (0.0, len(cr._RM_QUARANTINE_BACKOFF_S) - 1)
    cr._try_orphan_rm("c0", 10.0)
    assert _clean_quarantine["c0"][1] == len(cr._RM_QUARANTINE_BACKOFF_S) - 1, "top tier saturates"


def test_rc_zero_is_gone_and_clears_a_stale_quarantine(monkeypatch, _clean_quarantine):
    """rc 0 is BOTH "removed" and "no such container" -- either way the name is gone."""
    _fake_rm(monkeypatch, 0)
    _clean_quarantine["c0"] = (0.0, 1)   # due now, previously failed
    assert cr._try_orphan_rm("c0", 10.0) is True
    assert "c0" not in _clean_quarantine


def test_reap_does_not_count_containers_the_daemon_refused(monkeypatch, _clean_quarantine):
    """End-to-end through the real _try_orphan_rm: orphans=N must not overstate."""
    now = time.time()
    scan = _DockerPsScan(all=tuple((f"c{i}", now) for i in range(3)))
    monkeypatch.setattr(cr, "_docker_ps_with_time", lambda *a, **k: scan)
    monkeypatch.setattr(cr.time, "sleep", lambda *_a: None)
    _fake_rm(monkeypatch, 1, "device or resource busy")
    assert ContainerReaper().reap("e", SCOPE, set(), boot=True) == 0
    assert set(_clean_quarantine) == {"c0", "c1", "c2"}
