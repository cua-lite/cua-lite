"""Unit tests for lite.gym.utils.backend.docker helpers.

Run: uv run pytest tests/gym/utils/backend/test_docker.py
"""
from __future__ import annotations

import asyncio
import subprocess
import threading

import pytest

from lite.gym.errors import EnvDepsMissingError
from lite.gym.utils.backend import docker as docker_mod
from lite.gym.utils.backend.docker import (
    _rm_argv,
    docker_rm_f,
    docker_rm_f_async,
    docker_run,
    redact_env_args,
)

# ── the one removal argv (section A0) ───────────────────────────────────────────────

def test_rm_argv_is_the_one_removal_shape():
    # -v is part of the DEFINITION of removal (anon-volume clean); every rm
    # path shares this argv so the flag can never diverge again.
    assert _rm_argv("c1") == ["docker", "rm", "-f", "-v", "c1"]


def test_docker_rm_f_uses_rm_argv_and_reports_removed(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(argv, 0, stdout="c1\n", stderr="")

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    assert docker_rm_f("c1", timeout=5.0) == 1
    assert seen["argv"] == ["docker", "rm", "-f", "-v", "c1"]
    # ``timeout`` bounds the WHOLE removal, so the CLI gets what is left of it.
    assert 0 < seen["timeout"] <= 5.0


# ── the receipt, measured: what `docker rm -f -v` actually answers ────────────
#
# On this host (2026-08-08): removed -> rc 0 + the id on stdout; absent -> rc 0,
# stdout EMPTY, "No such container" on stderr; refused -> rc 1, stdout EMPTY,
# "could not kill container: ... did not receive an exit event" on stderr. So the
# exit code cannot tell a removal from a no-op and the echoed id is the only
# receipt. Both cases below are rc 0; only one removed a container.

def test_docker_rm_f_absent_name_is_rc_zero_and_removed_nothing_quietly(
    monkeypatch, caplog,
):
    calls = []

    def fake_run(argv, **_k):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, stdout="", stderr="Error response from daemon: No such container: gone")

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    with caplog.at_level("WARNING"):
        assert docker_rm_f("gone", timeout=5.0) == 0
    # The idempotent pre-reap is the commonest removal call in the tree
    # (every env clears a same-name container before booting): exactly one CLI,
    # and not a word of warning.
    assert len(calls) == 1
    assert caplog.text == ""


def test_docker_rm_f_refused_is_re_issued_and_the_second_receipt_wins(monkeypatch):
    """A refusal is a HALF-DONE removal, so it is finished rather than reported.

    Measured: 7/7 concurrent removals of running containers were refused at
    12.0-12.4 s with rc 1, the SIGKILL landed ~15 s later, the remove step never
    ran, and all 7 were still present at +60 s. An immediate re-issue removed 4/4.
    """
    calls = []

    def fake_run(argv, **_k):
        calls.append(argv)
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                argv, 1, stdout="",
                stderr='cannot remove container "c1": could not kill container: '
                       "tried to kill container, but did not receive an exit event")
        return subprocess.CompletedProcess(argv, 0, stdout="c1\n", stderr="")

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    assert docker_rm_f("c1", timeout=5.0) == 1
    assert len(calls) == 2


def test_docker_rm_f_refused_to_the_end_says_refused_and_quotes_the_daemon(
    monkeypatch, caplog,
):
    monkeypatch.setattr(
        docker_mod.subprocess, "run",
        lambda argv, **_k: subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="could not kill container: no exit event"),
    )
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_PORT", raising=False)
    with caplog.at_level("WARNING"):
        assert docker_rm_f("stuck", timeout=0.2) == 0
    assert "REFUSED by the daemon" in caplog.text
    assert "could not kill container: no exit event" in caplog.text
    # Direct mode: name the command, don't promise a reaper that never runs.
    assert "direct mode has no drift reaper" in caplog.text

    monkeypatch.setenv("CUA_LITE_ENV_SERVER_PORT", "30999")
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert docker_rm_f("stuck", timeout=0.2) == 0
    assert "the drift reaper will backstop it" in caplog.text


def test_docker_rm_f_swallows_timeout_returns_zero(monkeypatch):
    def raise_timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

    monkeypatch.setattr(docker_mod.subprocess, "run", raise_timeout)
    assert docker_rm_f("wedged", timeout=0.1) == 0   # never raises


def test_docker_rm_f_async_uses_rm_argv_and_never_raises(monkeypatch):
    seen = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"c2\n", b""

    async def fake_exec(*argv, **kwargs):
        seen["argv"] = list(argv)
        return FakeProc()

    monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(docker_rm_f_async("c2", timeout=5.0))
    assert seen["argv"] == ["docker", "rm", "-f", "-v", "c2"]

    async def boom(*argv, **kwargs):
        raise FileNotFoundError("docker missing")

    monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", boom)
    asyncio.run(docker_rm_f_async("c3", timeout=5.0))   # swallows — must not raise


def test_docker_rm_f_async_reports_its_three_failures_distinguishably(
    monkeypatch, caplog,
):
    """NEVER ISSUED / ABANDONED / REFUSED call for different operator actions.

    The message they replaced was one string for all three ("did not complete
    cleanly (timeout/spawn error)"), and its docstring claimed the timeout branch
    was benign because "the daemon accepted it and finishes on its own schedule".
    Measured false: containers whose removal was refused sat as ``Exited (137)``
    at +60 s, and removing them by hand then took 0.04-0.05 s.
    """
    async def never_issued(*_argv, **_k):
        raise FileNotFoundError("docker missing")

    class Hang:
        returncode = None

        async def communicate(self):
            await asyncio.sleep(3600)

        def kill(self) -> None:
            raise AssertionError("the abandoned CLI must not be killed (Q2 revert)")

    class Refuse:
        returncode = 1

        async def communicate(self):
            return b"", b"could not kill container: no exit event"

    async def hang(*_argv, **_k):
        return Hang()

    async def refuse(*_argv, **_k):
        return Refuse()

    messages = {}
    for tag, stub, budget in (("never", never_issued, 5.0),
                              ("abandoned", hang, 0.05),
                              ("refused", refuse, 0.2)):
        monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", stub)
        caplog.clear()
        with caplog.at_level("WARNING"):
            asyncio.run(docker_rm_f_async(f"c-{tag}", timeout=budget))
        messages[tag] = caplog.text

    assert "NEVER ISSUED" in messages["never"]
    assert "ABANDONED" in messages["abandoned"]
    assert "REFUSED by the daemon" in messages["refused"]
    assert len({m.split(" — ")[0] for m in messages.values()}) == 3


def test_docker_rm_f_async_refused_then_removed_says_nothing(monkeypatch, caplog):
    spawned = []

    class Proc:
        def __init__(self, n: int) -> None:
            self.returncode = 1 if n == 1 else 0
            self._n = n

        async def communicate(self):
            return (b"", b"could not kill container") if self._n == 1 else (b"c1\n", b"")

    async def fake_exec(*argv, **_k):
        spawned.append(list(argv))
        return Proc(len(spawned))

    monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", fake_exec)
    with caplog.at_level("WARNING"):
        asyncio.run(docker_rm_f_async("c1", timeout=5.0))
    assert len(spawned) == 2          # the refusal was finished, not reported
    assert caplog.text == ""


# ── removal is DELIBERATELY UNBOUNDED (2026-08-08 revert) ───────────────────
#
# Q2 added ``RM_CONCURRENCY = 3`` and a shared semaphore across both removal
# helpers; it was reverted the same day on the owner's directive that neither
# direct mode nor the env-server's throttling mode may differ from
# ``origin/main``, which bounds no removal path. The test below is the guard on
# that decision: it meters concurrency under a stub — no docker, no wedged host
# required — and asserts the fan-out is N-for-N, so a reintroduced process-wide
# bound fails here instead of silently landing on ``env.close()``.
#
# The exposure this accepts is explicit: 57 simultaneous ``docker rm`` processes
# were once measured (oldest stuck 3 m 26 s). A caller that needs a bound must
# apply it at the call site, not here.

class _Peak:
    """Concurrency meter: ``peak`` is the max simultaneous holders seen."""

    def __init__(self) -> None:
        self.now = 0
        self.peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def leave(self) -> None:
        with self._lock:
            self.now -= 1


def _metered_rm_stub(peak: _Peak, *, hang: bool = False):
    """Stand-in for ``asyncio.create_subprocess_exec`` that counts overlap."""
    spawned: list[list[str]] = []

    class FakeProc:
        def __init__(self) -> None:
            self.killed = False
            self.returncode = 0

        async def communicate(self):
            if hang:
                await asyncio.sleep(3600)
            await asyncio.sleep(0.05)
            peak.leave()
            return b"removed\n", b""

        def kill(self) -> None:
            self.killed = True
            peak.leave()

    procs: list[FakeProc] = []

    async def fake_exec(*argv, **_kwargs):
        spawned.append(list(argv))
        peak.enter()
        p = FakeProc()
        procs.append(p)
        return p

    return fake_exec, spawned, procs


def test_docker_rm_f_async_fanout_is_unbounded(monkeypatch):
    """20 gathered removals spawn 20 simultaneous ``docker rm`` CLIs.

    The regression guard on the Q2 revert: a process-wide removal semaphore
    would show up here as a peak below 20. ``env.close()`` goes through this
    function, so a bound added here lands on every task.
    """
    peak = _Peak()
    fake_exec, spawned, _ = _metered_rm_stub(peak)
    monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", fake_exec)

    async def go():
        await asyncio.wait_for(asyncio.gather(*(
            docker_rm_f_async(f"c{i}", timeout=5.0) for i in range(20)
        )), timeout=60)

    asyncio.run(go())
    assert peak.peak == 20, (
        f"only {peak.peak} of 20 removals ran at once — removal is paced again"
    )
    assert len(spawned) == 20
    assert spawned[0] == _rm_argv("c0")


def test_docker_rm_f_sync_fanout_is_unbounded(monkeypatch):
    """The sync sibling is unpaced too: 10 threads all reach the subprocess.

    A barrier, not a sleep: on a loaded host a 50 ms hold is not reliably long
    enough for the threads to overlap, and the test would read "peak 1" as
    success. The barrier makes a bound a HANG-then-timeout rather than a
    false pass.
    """
    peak = _Peak()
    rendezvous = threading.Barrier(10, timeout=30)

    def fake_run(argv, **_kwargs):
        peak.enter()
        rendezvous.wait()
        peak.leave()
        return subprocess.CompletedProcess(argv, 0, stdout="x\n", stderr="")

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    ts = [threading.Thread(target=docker_rm_f, args=(f"c{i}",),
                           kwargs={"timeout": 5.0}) for i in range(10)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
        assert not t.is_alive()
    assert peak.peak == 10


def test_docker_rm_f_async_swallows_a_timeout_and_names_the_container(
    monkeypatch, caplog,
):
    """A removal that blows its budget warns, names the container, never raises.

    The abandoned CLI child is NOT killed — that was Q2's second half, reverted
    with the first so this path matches ``origin/main`` — and nothing is
    re-issued, because the budget is spent and a second CLI would race the one
    still holding the daemon's lock. The message says ABANDONED rather than
    claiming either outcome: this is the one branch whose verdict is genuinely
    unknown.
    """
    peak = _Peak()
    fake_exec, _, procs = _metered_rm_stub(peak, hang=True)
    monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", fake_exec)

    with caplog.at_level("WARNING"):
        asyncio.run(docker_rm_f_async("wedged", timeout=0.05))   # never raises

    assert "ABANDONED" in caplog.text
    assert "wedged" in caplog.text
    assert not procs[0].killed


def test_reaper_orphan_rm_shares_the_argv(monkeypatch):
    """The drift reaper's quarantined rm builds its argv from _rm_argv too —
    the -v flag structurally cannot diverge between the two rm paths."""
    from lite.gym.remote import reaper as cr

    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    assert cr._try_orphan_rm("orphan1", 5.0) is True
    assert seen["argv"] == _rm_argv("orphan1")


# ── quarantine backoff escalation (moved from tests/gym/remote/test_reaper_quarantine.py;
# fully stubbed, no docker, so it belongs in the default
# suite; the real-docker clean-exit variant stays live-marked there) ──────────

@pytest.fixture()
def _clear_rm_quarantine():
    from lite.gym.remote.reaper import _RM_QUARANTINE
    _RM_QUARANTINE.clear()
    yield
    _RM_QUARANTINE.clear()


def test_quarantine_tier_escalates_on_repeated_timeout(_clear_rm_quarantine, monkeypatch):
    """Simulate consecutive `docker rm -f` timeouts on the same name;
    verify the backoff tier escalates 0→1→2→2 (capped)."""
    from lite.gym.remote import reaper as dd

    test_name = "lite-env-30200-abc123-stuck-androidworld-x-99"

    # Stub subprocess.run to raise TimeoutExpired on docker rm -f.
    original_run = dd.subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd[:3] == ["docker", "rm", "-f"]:
            raise dd.subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 60.0))
        return original_run(cmd, *args, **kwargs)
    monkeypatch.setattr(dd.subprocess, "run", fake_run)

    # Tier escalation: 0 → 1 → 2 → 2 (capped).
    expected_tiers = [0, 1, 2, 2]
    for i, expected in enumerate(expected_tiers):
        # Fast-forward past prior backoff so the next call isn't a no-op.
        if test_name in dd._RM_QUARANTINE:
            _, t = dd._RM_QUARANTINE[test_name]
            dd._RM_QUARANTINE[test_name] = (0.0, t)
        assert not dd._try_orphan_rm(test_name, rm_timeout=60.0), \
            f"iteration {i}: expected timeout"
        _, tier = dd._RM_QUARANTINE[test_name]
        assert tier == expected, f"iteration {i}: tier {tier} != {expected}"


def test_redact_env_args_masks_secrets_only():
    args = ["-e", "OPENAI_API_KEY=sk-secret123", "-e", "WEBSITE_HOST_SUFFIX=web.hku.icu",
            "-e", "GITLAB_TOKEN=glpat-xyz", "-v", "/a:/b", "--name", "foo"]
    out = redact_env_args(args, ("OPENAI_API_KEY", "GITLAB_TOKEN"))
    assert "OPENAI_API_KEY=***" in out and "GITLAB_TOKEN=***" in out
    assert "WEBSITE_HOST_SUFFIX=web.hku.icu" in out          # non-secret passes through
    assert "sk-secret123" not in " ".join(out)               # real values never appear
    assert "glpat-xyz" not in " ".join(out)
    assert out[6:] == ["-v", "/a:/b", "--name", "foo"]       # non -e args untouched


def test_redact_env_args_no_secrets_is_identity():
    args = ["-e", "SCREEN_W=1920", "-v", "/x:/y", "--name", "c"]
    assert redact_env_args(args, ("OPENAI_API_KEY",)) == args


def test_redact_env_args_value_with_equals_fully_masked():
    # a value containing '=' (e.g. a base64 token) still masks to KEY=*** (split on FIRST =)
    out = redact_env_args(["-e", "OPENAI_API_KEY=ab=cd=ef"], ("OPENAI_API_KEY",))
    assert out == ["-e", "OPENAI_API_KEY=***"]


def test_docker_run_checks_freshness_before_launch(monkeypatch):
    events = []

    class FakeImage:
        tag = "cua-lite/test:latest"
        install = "install.sh"
        see = "README.md"

        def ensure_runnable(self):
            events.append("ensure")

    def fake_run(argv, **kwargs):
        events.append("run")
        return subprocess.CompletedProcess(argv, 0, stdout="cid", stderr="")

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    docker_run(
        "c1",
        FakeImage(),
        mem="4g",
        port=(12345, 8000),
        env={"A": "1"},
    )
    assert events == ["ensure", "run"]


def test_docker_run_failure_does_not_expose_stderr(monkeypatch, caplog):
    class FakeImage:
        tag = "cua-lite/test:latest"
        install = "uv run --no-sync bash install.sh"
        see = "README.md"

        def ensure_runnable(self):
            return None

    def fake_run(argv, **kwargs):
        raise subprocess.CalledProcessError(
            125,
            argv,
            stderr="bind mount /mnt/private/path failed",
        )

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    with pytest.raises(EnvDepsMissingError) as excinfo:
        docker_run("c1", FakeImage(), mem="4g", port=(12345, 8000))

    assert "docker run failed for image cua-lite/test:latest" in str(excinfo.value)
    assert "/mnt/private/path" not in str(excinfo.value)
    assert excinfo.value.install == "uv run --no-sync bash install.sh"
    assert "bind mount /mnt/private/path failed" in caplog.text


# ── docker_run_detached (section G-DX): the ONE DEDICATED docker run ────────────────

def test_docker_run_detached_argv_and_lease_restamp(monkeypatch):
    from lite.gym.utils.backend.docker import docker_run_detached

    seen = {}
    monkeypatch.setattr(docker_mod, "touch_ports",
                        lambda *ports: seen.setdefault("touched", ports))

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout=b"cid", stderr=b"")

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    docker_run_detached(
        name="c1", image="img:latest",
        auto_remove=True, privileged=True,
        cap_add=("NET_ADMIN",), devices=("/dev/kvm",),
        sysctls=("net.ipv4.ip_forward=1",), memory="8g", group_add=(993,),
        env={"A": "1", "SECRET": "s"}, volumes=("/x:/y:ro",),
        ports=((22001, 6000),), command=("sleep", "1"),
        redact=("SECRET",), timeout=9.0, label="t",
    )
    assert seen["argv"] == [
        "docker", "run", "--rm", "-d", "--privileged",
        "--cap-add", "NET_ADMIN", "--device", "/dev/kvm",
        "--sysctl", "net.ipv4.ip_forward=1",
        "--memory", "8g", "--memory-swap", "8g", "--memory-swappiness", "0",
        "--group-add", "993",
        "-e", "A=1", "-e", "SECRET=s",
        "-v", "/x:/y:ro", "-p", "22001:6000",
        "--name", "c1", "img:latest", "sleep", "1",
    ]
    assert seen["touched"] == (22001,), "section D3: lease re-stamped right before the run"


def test_docker_run_detached_error_mapping(monkeypatch):
    from lite.gym.errors import CapacityExhausted
    from lite.gym.utils.backend.docker import docker_run_detached

    monkeypatch.setattr(docker_mod, "touch_ports", lambda *p: None)

    def raise_timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

    monkeypatch.setattr(docker_mod.subprocess, "run", raise_timeout)
    with pytest.raises(CapacityExhausted):   # warming → retriable 503
        docker_run_detached(name="c2", image="i", timeout=0.1)

    monkeypatch.setattr(
        docker_mod.subprocess, "run",
        lambda argv, **k: subprocess.CompletedProcess(argv, 125, stdout=b"", stderr=b"name in use"),
    )
    with pytest.raises(RuntimeError, match="name in use"):
        docker_run_detached(name="c3", image="i", timeout=5.0)
