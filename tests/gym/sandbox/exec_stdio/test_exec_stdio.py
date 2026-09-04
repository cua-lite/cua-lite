#!/usr/bin/env python3
"""Protocol-level tests for the exec-stdio cb transport — no docker, no X.

``ExecStdioSession(_local=True)`` runs the in-container ``server.py`` as a plain
host subprocess over the SAME JSON-lines pipe the real ``docker exec -i`` path
uses, so these exercise the host↔server protocol (framing, error mapping,
auto-respawn) without a container. GUI ops (screenshot/input) need X and are
NOT covered here; only the transport-pure ops (run_command / read_bytes /
write_bytes / ping) are.

Run:
    PYTHONPATH=. uv run pytest tests/gym/sandbox/exec_stdio/test_exec_stdio.py -q
"""
from __future__ import annotations

import ast
import asyncio
import subprocess
from pathlib import Path

import pytest

from lite.gym.sandbox.exec_stdio.client import ExecStdioError, ExecStdioSession

_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
async def session():
    s = ExecStdioSession("local-test", _local=True)
    await s.start()
    try:
        yield s
    finally:
        await s.close()


async def test_run_command_stdout_stderr_rc(session):
    """run_command surfaces stdout, stderr and the real exit code separately."""
    res = await session.call(
        "run_command",
        cmd="printf out; printf err 1>&2; exit 7",
    )
    assert res["stdout"] == "out"
    assert res["stderr"] == "err"
    assert res["returncode"] == 7


async def test_run_command_success_rc_zero(session):
    res = await session.call("run_command", cmd="echo hello")
    assert res["stdout"] == "hello\n"
    assert res["stderr"] == ""
    assert res["returncode"] == 0


async def test_write_read_bytes_binary_roundtrip(session, tmp_path):
    """write_bytes/read_bytes round-trip arbitrary bytes (base64 framed) —
    incl. NULs and non-UTF8, which a text channel would corrupt."""
    path = str(tmp_path / "blob.bin")
    payload = bytes(range(256)) + b"\x00\xff\xfe nul-and-high-bytes"
    import base64

    await session.call(
        "write_bytes", path=path, b64=base64.b64encode(payload).decode("ascii")
    )
    res = await session.call("read_bytes", path=path)
    assert base64.b64decode(res["b64"]) == payload


async def test_auto_respawn_after_proc_kill(session):
    """A dead server pipe is transparently respawned and the call retried once
    (the server is stateless) — the channel survives a crash."""
    old_proc = session._proc
    assert old_proc is not None
    old_proc.kill()
    await old_proc.wait()
    assert session._proc.returncode is not None  # confirmed dead

    res = await session.call("run_command", cmd="echo back")
    assert res["stdout"] == "back\n"
    assert session._proc is not None and session._proc.returncode is None
    assert session._proc is not old_proc  # a fresh subprocess


async def test_bad_op_raises_exec_stdio_error(session):
    """An unknown op is reported by the server as ``ok: false`` and mapped to
    ExecStdioError on the host (the pipe stays alive afterwards)."""
    with pytest.raises(ExecStdioError):
        await session.call("no_such_op")
    # session is still usable after a reported failure (pipe outlives bad reqs)
    res = await session.call("run_command", cmd="echo still-alive")
    assert res["stdout"] == "still-alive\n"


async def test_ping(session):
    """ping returns the server protocol version (liveness + version handshake)."""
    assert (await session.call("ping")).get("version") == 4


async def test_docker_spawn_uses_fixed_env_python_contract(monkeypatch):
    """Maintained Docker images use the baked server on the env Python.

    This intentionally does not shell through a ``python fallback`` selector:
    missing /opt/env/venv/bin/python or /opt/lite/stdio_server.py is an image
    contract failure, not something the host transport should paper over.
    """
    from lite.gym.sandbox.exec_stdio import client

    captured: dict = {}

    async def fake_create_subprocess_exec(*argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs

        class _Proc:
            returncode = None
            stdin = None
            stdout = None

        return _Proc()

    monkeypatch.setattr(client.asyncio, "create_subprocess_exec",
                        fake_create_subprocess_exec)
    s = client.ExecStdioSession("ctr")
    await s._spawn()

    assert captured["argv"] == (
        "docker", "exec", "-i", "-u", "user", "ctr",
        client._ENV_PY, client._SERVER_BAKED,
    )


async def test_image_contract_probe_reports_stale_install(monkeypatch):
    """A missing maintained-image dependency should fail fast with rebuild
    guidance instead of trying docker-cp or system-Python fallbacks."""
    from lite.gym.sandbox.exec_stdio import client

    captured: dict = {}

    class _Proc:
        returncode = 90

        async def communicate(self):
            return b"", b"missing executable: /opt/env/venv/bin/python\n"

    async def fake_create_subprocess_exec(*argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(client.asyncio, "create_subprocess_exec",
                        fake_create_subprocess_exec)
    s = client.ExecStdioSession("ctr")

    with pytest.raises(ExecStdioError, match="image contract failed") as excinfo:
        await s._check_image_contract()

    assert captured["argv"][:6] == (
        "docker", "exec", "-u", "user", "ctr", "sh",
    )
    assert "install.sh" in str(excinfo.value)


async def test_image_contract_probe_runtime_failure_is_not_stale_install(monkeypatch):
    """Docker/probe runtime failures should not be diagnosed as stale images."""
    from lite.gym.sandbox.exec_stdio import client

    class _Proc:
        returncode = 125

        async def communicate(self):
            return b"", b"container is not running\n"

    async def fake_create_subprocess_exec(*argv, **kwargs):
        return _Proc()

    monkeypatch.setattr(client.asyncio, "create_subprocess_exec",
                        fake_create_subprocess_exec)
    s = client.ExecStdioSession("ctr")

    with pytest.raises(RuntimeError, match="exec-stdio image probe failed") as excinfo:
        await s._check_image_contract()

    assert "image contract failed" not in str(excinfo.value)
    assert "install.sh" not in str(excinfo.value)


async def test_ping_timeout_propagates_as_retryable_timeout(monkeypatch):
    """A slow first ping stays a raw TimeoutError for the sandbox retry path."""
    from lite.gym.sandbox.exec_stdio import client

    async def fake_check(self):
        return None

    async def fake_spawn(self):
        self._proc = object()

    async def fake_call(self, op, **kwargs):
        raise TimeoutError

    async def fake_close(self):
        self._proc = None

    monkeypatch.setattr(client.ExecStdioSession, "_check_image_contract", fake_check)
    monkeypatch.setattr(client.ExecStdioSession, "_spawn", fake_spawn)
    monkeypatch.setattr(client.ExecStdioSession, "call", fake_call)
    monkeypatch.setattr(client.ExecStdioSession, "close", fake_close)

    s = client.ExecStdioSession("ctr")
    with pytest.raises(TimeoutError):
        await s.start()


async def test_ping_version_mismatch_raises_contract_error(monkeypatch):
    """A v1 (pre-plus-token) baked server must fail the contract loudly —
    this is the branch every pre-existing image hits after the v3 bump."""
    from lite.gym.sandbox.exec_stdio import client

    async def fake_check(self):
        return None

    async def fake_spawn(self):
        self._proc = object()

    async def fake_call(self, op, **kwargs):
        return {"version": 1}

    async def fake_close(self):
        self._proc = None

    monkeypatch.setattr(client.ExecStdioSession, "_check_image_contract", fake_check)
    monkeypatch.setattr(client.ExecStdioSession, "_spawn", fake_spawn)
    monkeypatch.setattr(client.ExecStdioSession, "call", fake_call)
    monkeypatch.setattr(client.ExecStdioSession, "close", fake_close)

    s = client.ExecStdioSession("ctr")
    with pytest.raises(client.ExecStdioError, match="protocol version"):
        await s.start()


async def test_transient_ping_failure_matches_boot_retry_classifier(monkeypatch):
    """Pin the cross-file magic-string contract: the RuntimeError start()
    raises for a transient ping failure must be classified transient by the
    sandbox boot-retry loop — rewording either side silently converts
    retryable boot hiccups into hard failures."""
    from lite.gym.sandbox import base
    from lite.gym.sandbox.exec_stdio import client

    async def fake_check(self):
        return None

    async def fake_spawn(self):
        self._proc = object()

    async def fake_call(self, op, **kwargs):
        raise ConnectionResetError("peer reset")

    async def fake_close(self):
        self._proc = None

    monkeypatch.setattr(client.ExecStdioSession, "_check_image_contract", fake_check)
    monkeypatch.setattr(client.ExecStdioSession, "_spawn", fake_spawn)
    monkeypatch.setattr(client.ExecStdioSession, "call", fake_call)
    monkeypatch.setattr(client.ExecStdioSession, "close", fake_close)

    s = client.ExecStdioSession("ctr")
    with pytest.raises(RuntimeError) as exc_info:
        await s.start()
    assert base._is_transient_boot_error(exc_info.value)


def test_exec_stdio_client_has_no_silent_legacy_fallback_path():
    """Default transport should not support stale images by copying host code
    into /tmp or falling back to the agent-facing system Python."""
    source = (
        _ROOT / "lite" / "gym" / "sandbox" / "exec_stdio" / "client.py"
    ).read_text()

    assert "_SERVER_CP_DST" not in source
    assert "_IN_CTR_PY" not in source
    assert "docker\", \"cp\"" not in source
    assert "/tmp/lite_stdio_server.py" not in source
    assert "[ -x /opt/env/venv/bin/python ]" not in source


def test_shared_docker_exec_session_has_neutral_owner():
    """The shared subprocess lifecycle is env-internal transport, not agent shell."""
    exec_stdio_root = _ROOT / "lite" / "gym" / "sandbox" / "exec_stdio"
    core_root = _ROOT / "lite" / "core"
    session_path = exec_stdio_root / "session.py"
    client_source = (exec_stdio_root / "client.py").read_text()
    agent_source = (exec_stdio_root / "agent_shell.py").read_text()
    session_source = session_path.read_text()

    assert session_path.is_file()
    assert session_path.is_relative_to(exec_stdio_root)
    assert not session_path.is_relative_to(core_root)
    assert "lite.core" not in session_source
    assert "class DockerExecSession" in session_source
    assert "class _DockerExecSession" not in agent_source
    assert "_DockerExecSession" not in client_source
    assert (
        "from lite.gym.sandbox.exec_stdio.session import DockerExecSession"
        in client_source
    )
    assert (
        "from lite.gym.sandbox.exec_stdio.session import DockerExecSession"
        in agent_source
    )

    client_tree = ast.parse(client_source)
    agent_tree = ast.parse(agent_source)
    client_exec_session = next(
        node for node in client_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ExecStdioSession"
    )
    agent_shell_session = next(
        node for node in agent_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentShellSession"
    )
    assert [
        base.id for base in client_exec_session.bases if isinstance(base, ast.Name)
    ] == ["DockerExecSession"]
    assert [
        base.id for base in agent_shell_session.bases if isinstance(base, ast.Name)
    ] == ["DockerExecSession"]


def test_sandbox_family_images_inherit_baked_exec_stdio_contract():
    """The envs that use SandboxBaseEnv must inherit the maintained image ABI.

    - sandbox.linux creates /opt/env/venv and bakes /opt/lite/stdio_server.py.
    - lite.osworld builds from sandbox.linux.
    - lite.cuagym builds from lite.osworld.
    - lite.cuaworld.base is sandbox.linux rebuilt with USER=ga, then software
      images build from that base.
    """
    sandbox = (
        _ROOT / "lite" / "gym" / "sandbox" / "docker" / "Dockerfile.linux"
    ).read_text()
    osworld = (
        _ROOT / "lite" / "gym" / "envs" / "lite" / "osworld"
        / "docker" / "Dockerfile"
    ).read_text()
    cuagym = (
        _ROOT / "lite" / "gym" / "envs" / "lite" / "cuagym"
        / "docker" / "Dockerfile"
    ).read_text()
    cuaworld_install = (
        _ROOT / "lite" / "gym" / "envs" / "lite" / "cuaworld"
        / "scripts" / "install.sh"
    ).read_text()
    cuaworld_docker = (
        _ROOT / "lite" / "gym" / "envs" / "lite" / "cuaworld"
        / "docker" / "Dockerfile"
    ).read_text()

    assert "/opt/env/bin/uv venv --python 3.12 /opt/env/venv" in sandbox
    assert "COPY exec_stdio/server.py /opt/lite/stdio_server.py" in sandbox
    assert "FROM cua-lite/sandbox.linux:${BASE_TAG}" in osworld
    assert "ARG BASE_IMAGE=cua-lite/lite.osworld:latest" in cuagym
    assert "FROM ${BASE_IMAGE} AS xcftools-builder" in cuagym
    assert cuagym.splitlines().count("FROM ${BASE_IMAGE}") == 1
    assert "BASE_IMAGE=\"cua-lite/lite.cuaworld.base:latest\"" in cuaworld_install
    assert "--build-arg USER=ga" in cuaworld_install
    assert "ARG BASE_IMAGE=cua-lite/lite.cuaworld.base:latest" in cuaworld_docker
    assert "FROM ${BASE_IMAGE}" in cuaworld_docker


def _inject_first_line(monkeypatch, session, line: bytes):
    """Make the current proc's stdout yield ``line`` on the first readline, then
    fall through to the real reader (which, after a respawn, is the fresh proc)."""
    real = session._proc.stdout.readline
    n = {"i": 0}

    async def flaky(*a, **k):
        n["i"] += 1
        return line if n["i"] == 1 else await real(*a, **k)

    monkeypatch.setattr(session._proc.stdout, "readline", flaky)


async def test_blank_line_response_resyncs_and_retries(session, monkeypatch):
    """A stray blank line (b'\\n') is a desync, not a response: it is truthy so a
    plain ``not raw`` check would pass it to json.loads → 'Expecting value: char
    0'. The session must instead respawn and retry the idempotent op."""
    _inject_first_line(monkeypatch, session, b"\n")
    assert (await session.call("ping")).get("version") == 4  # resynced


async def test_malformed_line_resyncs_and_retries(session, monkeypatch):
    """A non-blank but non-JSON line (interleaved stdout) raises JSONDecodeError;
    call() treats it as a pipe desync and retries the idempotent op."""
    _inject_first_line(monkeypatch, session, b"not json at all\n")
    assert (await session.call("ping")).get("version") == 4  # resynced


async def test_desync_nonidempotent_raises_no_double_execute(session, monkeypatch):
    """A desync on a NON-idempotent op raises after respawn — never silently
    retried (run_command is correctly excluded from _RETRYABLE_OPS)."""
    async def always_blank(*a, **k):
        return b"\n"

    monkeypatch.setattr(session._proc.stdout, "readline", always_blank)
    with pytest.raises(asyncio.IncompleteReadError):
        await session.call("run_command", cmd="true")


# =============================================================================
# Binary-frame completion: the screenshot op is now a
# ``header\n`` + raw-bytes frame. Two failure modes the header-length contract
# must survive — an *error* frame (op failed → ``shot_n:0``, NO body) and a
# mid-frame timeout — driven off a fake StreamReader so they need no X/ffmpeg
# and stay deterministic with a fake proc/StreamReader.
# =============================================================================

class _FakeStdin:
    """Swallows the request line — these tests script the *responses*, so what
    the client writes is irrelevant."""
    def write(self, _data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass


class _FakeProc:
    """A live-looking (returncode is None) proc backed by a scripted stdout —
    enough for ``_roundtrip``/``call`` without a real subprocess or X."""
    returncode = None

    def __init__(self, stdout):
        self.stdin = _FakeStdin()
        self.stdout = stdout


def _reader_with(*lines: bytes) -> asyncio.StreamReader:
    r = asyncio.StreamReader()
    for ln in lines:
        r.feed_data(ln)
    r.feed_eof()
    return r


async def test_error_frame_shot_n_zero_leaves_no_desync():
    """A framed op (screenshot) that FAILS emits ``ok:false, shot_n:0`` with NO
    body. The client must raise ExecStdioError for it and — crucially — read no
    body (it honors the header length, never assumes a body from the op name), so
    the NEXT op reads its own frame cleanly with no desync."""
    s = ExecStdioSession("fake", _local=True)
    s._proc = _FakeProc(_reader_with(
        b'{"id": 1, "ok": false, "shot_n": 0, "error": "RuntimeError: no X"}\n',
        b'{"id": 2, "ok": true, "result": {"version": 3}}\n',
    ))

    with pytest.raises(ExecStdioError):
        await s.call("screenshot")
    # No stray body bytes were left in the pipe: the follow-up op reads correctly.
    res = await s.call("ping")
    assert res == {"version": 3}


async def test_respawn_on_timeout_recovers_retryable_op(monkeypatch):
    """A mid-frame ``asyncio.TimeoutError`` (e.g. partway through the screenshot
    body) must trigger ``_respawn()`` — discarding the half-frame on a fresh
    pipe — after which the retryable op (screenshot ∈ _RETRYABLE_OPS) recovers.
    TimeoutError being in call()'s except set is the whole point."""
    s = ExecStdioSession("fake", _local=True)

    async def timeout_readline(*a, **k):
        raise TimeoutError

    class _TimeoutStdout:
        readline = staticmethod(timeout_readline)

    s._proc = _FakeProc(_TimeoutStdout())

    # A healthy proc the respawn swaps in; its frame id must match the retry's
    # id (_roundtrip increments _id: 1 on the timed-out attempt, 2 on the retry).
    good = _FakeProc(_reader_with(
        b'{"id": 2, "ok": true, "result": "recovered"}\n',
    ))

    async def fake_respawn():
        s._proc = good

    monkeypatch.setattr(s, "_respawn", fake_respawn)

    res = await s.call("screenshot")
    assert res == "recovered"
    assert s._proc is good  # respawn actually swapped the dead pipe out


async def test_readline_over_limit_raises_plain_valueerror_not_limitoverrun():
    """PIN the stdlib fact the resync arm is built on: a response line longer
    than the StreamReader limit makes ``readline()`` raise a PLAIN
    ``ValueError`` — asyncio catches its own ``LimitOverrunError`` internally
    and never lets it out. An ``except asyncio.LimitOverrunError`` arm can
    therefore NEVER fire for this case."""
    r = asyncio.StreamReader(limit=64)
    r.feed_data(b"x" * 4096)  # no separator, buffer already past the limit
    with pytest.raises(ValueError) as ei:
        await r.readline()
    assert not isinstance(ei.value, asyncio.LimitOverrunError)


async def test_over_limit_response_line_resyncs_and_retries(monkeypatch):
    """A response line past ``_STREAM_LIMIT`` (e.g. read_bytes of a huge file)
    is a pipe desync: unread bytes of the giant line are still queued, so the
    NEXT op would read them as its own frame. ``call()`` must respawn and — the
    op being idempotent — retry on the fresh pipe."""
    s = ExecStdioSession("fake", _local=True)
    over = asyncio.StreamReader(limit=64)
    over.feed_data(b"x" * 4096)
    s._proc = _FakeProc(over)

    # _roundtrip increments _id: 1 on the over-limit attempt, 2 on the retry.
    good = _FakeProc(_reader_with(
        b'{"id": 2, "ok": true, "result": {"b64": "aGk="}}\n',
    ))
    respawns = {"n": 0}

    async def fake_respawn():
        respawns["n"] += 1
        s._proc = good

    monkeypatch.setattr(s, "_respawn", fake_respawn)

    res = await s.call("read_bytes", path="/huge")
    assert res == {"b64": "aGk="}
    assert respawns["n"] == 1  # the over-limit line actually drove a respawn


async def test_bad_timeout_kwarg_raises_loudly_and_never_respawns():
    """The coercion trap the widened ``ValueError`` arm must NOT swallow: a
    malformed ``timeout`` is a CALLER bug, not a wire desync. ``float()`` runs
    outside the retry ``try``, so it raises straight through with no respawn and
    no retry."""
    s = ExecStdioSession("fake", _local=True)
    s._proc = _FakeProc(_reader_with(b'{"id": 1, "ok": true, "result": 1}\n'))

    async def boom():
        raise AssertionError("a malformed timeout must not trigger a respawn")

    s._respawn = boom  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        await s.call("ping", timeout="not-a-number")


# =============================================================================
# composite_cursor — the sandbox capture-time cursor composite. Tap only proves
# transport byte-identity (emitted==decoded), so a
# mis-placed / oversized / brightened sprite is transport-identical and passes
# Tap — this golden is the only guard on composite CORRECTNESS. Uses the REAL
# baked 16x24 sprite, so it also pins that the asset stays tip-at-top-left
# straight-alpha.
# =============================================================================

_CURSOR_PNG = (
    _ROOT / "lite" / "gym" / "sandbox" / "docker" / "assets" / "cursor.png"
)


def _blank(w: int, h: int, color=(0, 0, 0)):
    from PIL import Image
    return Image.new("RGB", (w, h), color)


def test_composite_cursor_lands_tip_at_xy_sized_16x24():
    """The sprite composites tip-at-(x,y) confined to a 16x24 box — NO hotspot
    offset, NO oversize. Asserted by pixel confinement: over a black frame, every
    non-background pixel must fall inside ``[x, x+16) x [y, y+24)`` (top-left =
    the tip), and the box must actually contain drawn pixels."""
    from PIL import Image

    from lite.gym.sandbox.exec_stdio.server import composite_cursor

    sprite = Image.open(_CURSOR_PNG).convert("RGBA")
    assert sprite.size == (16, 24), "golden pins the baked sprite at 16x24"
    x, y = 30, 40
    out = composite_cursor(_blank(100, 100), sprite, x, y)

    assert out.mode == "RGB" and out.size == (100, 100)
    px = out.load()
    drawn = 0
    for j in range(100):
        for i in range(100):
            inside = (x <= i < x + 16) and (y <= j < y + 24)
            if inside:
                if px[i, j] != (0, 0, 0):
                    drawn += 1
            else:
                assert px[i, j] == (0, 0, 0), (
                    f"cursor pixel leaked outside the 16x24 box at ({i},{j}) — "
                    "oversized or mis-offset sprite"
                )
    assert drawn > 0, "sprite drew nothing inside its box"
    # The arrow tip sits at the sprite's top-left, so (x, y) itself is drawn
    # (proves tip-at-(x,y), not center-anchored).
    assert px[x, y] != (0, 0, 0), "tip must land exactly at (x, y)"


def test_composite_cursor_is_straight_alpha_not_premultiplied():
    """Straight-alpha, no brightening: a fully-opaque sprite pixel reproduces its
    source RGB exactly, and a semi-transparent pixel is the straight ``src*a +
    bg*(1-a)`` blend — NOT the XFixes un-premultiply (``r*255//a``), which would
    blow a low-alpha white tip up to full white."""
    from PIL import Image

    from lite.gym.sandbox.exec_stdio.server import composite_cursor

    sprite = Image.open(_CURSOR_PNG).convert("RGBA")
    sp = sprite.load()
    x, y = 30, 40
    out = composite_cursor(_blank(100, 100), sprite, x, y).load()

    # Opaque sprite pixel over any bg → exactly the sprite RGB (straight alpha).
    op = next(
        (sx, sy) for sy in range(24) for sx in range(16) if sp[sx, sy][3] == 255
    )
    assert out[x + op[0], y + op[1]] == sp[op[0], op[1]][:3]

    # Semi-transparent pixel over black → src*a/255 (rounded), well below the
    # un-premultiply blow-up. The baked tip (0,0) is white at alpha 63 → ~(63,).
    sr, sg, sb, sa = sp[0, 0]
    assert 0 < sa < 255, "golden assumes an antialiased (semi-alpha) tip pixel"
    got = out[x, y]
    exp = tuple(round(c * sa / 255) for c in (sr, sg, sb))
    assert all(abs(g - e) <= 1 for g, e in zip(got, exp)), (
        f"tip blend {got} != straight-alpha {exp}; a premultiply/brighten "
        "regression would push it toward full white"
    )


# =============================================================================
# _norm_key — raw exec-stdio/container token → xdotool keysym. This is not the
# Lite canonical env-input contract; Lite envs validate/project before RPC.
# Cases are real raw tokens harvested from the 2026-04-26 multi-model eval run.
# =============================================================================

@pytest.mark.parametrize(
    "token,expected",
    [
        # letters lowercase (a capital keysym would be the SHIFTED key, so
        # ctrl+O would become ctrl+shift+o — the wrong shortcut)
        ("O", "o"), ("V", "v"), ("a", "a"),
        # single punctuation → keysym NAME (xdotool rejects the bare glyph), so
        # ctrl+, (Preferences) and ctrl+. (VLC frame-step) land
        (",", "comma"), (".", "period"), ("/", "slash"), (";", "semicolon"),
        ("+", "plus"),
        # digits and F-keys are valid keysym names as-is
        ("8", "8"), ("F5", "F5"),
        # vocabulary + case + macOS-Cmd aliases all resolve
        ("enter", "Return"), ("return", "Return"), ("esc", "Escape"),
        ("CTRL", "ctrl"), ("pagedown", "Next"),
        ("command", "super"), ("cmd", "super"), ("meta", "super"),
        ("print_screen", "Print"),
    ],
)
def test_norm_key(token, expected):
    from lite.gym.sandbox.exec_stdio.server import _norm_key

    assert _norm_key(token) == expected


def test_type_splits_tab_and_newline_into_discrete_keys(monkeypatch):
    """`type` must emit '\\n'->Return and '\\t'->Tab as DISCRETE key presses
    around per-segment type runs — never left inside the xdotool type stream
    (GTK drops a streamed Linefeed; a streamed Tab races Calc's cell commit and
    chops the value across cells). Verifies the call sequence without X."""
    from lite.gym.sandbox.exec_stdio import server

    calls: list[tuple] = []
    monkeypatch.setattr(server, "_xdo", lambda *a, **k: calls.append(a))
    server.op_input({"action": "type", "text": "a\tb\nc"})

    # type "a"; Tab; type "b"; Return; type "c"
    assert [c[0] for c in calls] == ["type", "key", "type", "key", "type"]
    assert calls[1] == ("key", "--clearmodifiers", "Tab")
    assert calls[3] == ("key", "--clearmodifiers", "Return")
    assert calls[0][-1] == "a" and calls[2][-1] == "b" and calls[4][-1] == "c"


def test_run_command_does_not_share_protocol_stdin(monkeypatch):
    from lite.gym.sandbox.exec_stdio import server

    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    server.op_run_command({"cmd": "true"})

    assert captured["stdin"] is subprocess.DEVNULL


async def test_run_command_cannot_read_protocol_stdin(session):
    """A command that reads stdin sees EOF instead of blocking the JSONL server."""
    res = await session.call(
        "run_command",
        cmd="IFS= read -r protocol_line",
        timeout=0.5,
    )

    assert res["returncode"] == 1
    assert res["stdout"] == ""
    assert res["stderr"] == ""


# =============================================================================
# Host-side timeout budget for ``type_text`` — must outlast the server's
# xdotool budget for the same text, otherwise the client times out, the
# server's reply lands on a dead pipe, and the caller sees a transport
# failure for a type that actually succeeded server-side. ``input`` is
# correctly excluded from auto-retry (no double-type), so a premature
# client bail is unrecoverable.
# =============================================================================

@pytest.mark.parametrize(
    "text_len,min_budget_s",
    [
        # Short types stay at the default — no pessimization for the
        # common case (cb step typing a search query, command-line snippet).
        (10, 120),
        (100, 120),
        # Long types must exceed the pre-fix 130s wait. Server's per-row
        # budget is ``30 + len * 0.024`` at default delay_ms=12; the host
        # must wait at least as long, plus a cushion.
        (5000, 150),    # server budget ≈ 30 + 120 = 150s
        (10000, 270),   # server budget ≈ 30 + 240 = 270s
    ],
)
async def test_type_text_host_budget_outlasts_server(text_len, min_budget_s):
    """ExecStdioInterface.type_text must pass a ``timeout`` kwarg to
    ``session.call`` that is at least as long as the server-side xdotool
    budget — otherwise the asyncio.wait_for in ``_roundtrip`` fires before
    the server's response and the type op fails despite having succeeded."""
    from lite.gym.sandbox.exec_stdio.client import ExecStdioInterface

    captured: dict = {}

    class _StubSession:
        async def call(self, op, **kwargs):
            captured["op"] = op
            captured["kwargs"] = kwargs
            return {}

    iface = ExecStdioInterface(_StubSession())  # type: ignore[arg-type]
    await iface.type_text("x" * text_len)

    assert captured["op"] == "input"
    assert captured["kwargs"]["action"] == "type"
    assert captured["kwargs"]["text"] == "x" * text_len
    # The timeout kwarg is the whole point of the fix.
    assert "timeout" in captured["kwargs"], "type_text must set explicit timeout"
    assert captured["kwargs"]["timeout"] >= min_budget_s, (
        f"type_text timeout {captured['kwargs']['timeout']}s for {text_len}-char "
        f"text would race the server's ≥{min_budget_s}s xdotool budget"
    )


def test_non_ascii_segment_types_at_a_slower_delay(monkeypatch):
    """A codepoint outside the keymap needs a slower ``--delay`` or it is DROPPED.

    xdotool borrows a spare keycode for such a character and announces the remap
    with MappingNotify; at the 12 ms default the client has usually not processed
    it yet and loses the key -- silently, with ok=true. Measured in the cuagym
    image typing an em dash / en dash / ellipsis between ASCII letters: delay=12
    delivered only the four ASCII characters, delay=30 delivered all seven.
    Pure-ASCII text keeps the fast default.
    """
    from lite.gym.sandbox.exec_stdio import server

    calls: list[tuple] = []
    monkeypatch.setattr(server, "_xdo", lambda *a, **k: calls.append(a))

    server.op_input({"action": "type", "text": "plain ascii"})
    assert calls[0][calls[0].index("--delay") + 1] == "12"

    calls.clear()
    server.op_input({"action": "type", "text": "A—B"})
    assert calls[0][calls[0].index("--delay") + 1] == str(server._NON_ASCII_DELAY_MS)
    assert server._NON_ASCII_DELAY_MS >= 30


def test_explicit_delay_above_the_floor_is_respected(monkeypatch):
    from lite.gym.sandbox.exec_stdio import server

    calls: list[tuple] = []
    monkeypatch.setattr(server, "_xdo", lambda *a, **k: calls.append(a))
    server.op_input({"action": "type", "text": "A—B", "delay_ms": 90})

    assert calls[0][calls[0].index("--delay") + 1] == "90"
