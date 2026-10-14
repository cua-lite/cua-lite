#!/usr/bin/env python3
"""Host side of the exec-stdio transport — THE sandbox-family host↔container path.

Replaces the retired cua computer-server stack (websocket :8000 + the
``Computer`` object + api-port publishing + ``wait_for_ready``). Root cause
and measurements: an
in-container userspace server is CPU-starved under concurrent load and its
deadline-bound client turns slow into failure ("API Server not ready"), while
dockerd degrades by queueing; and the lean in-container backend (ffmpeg/
xdotool) measured ~10-13x faster per op than the old server stack.

Three pieces:

* :class:`ExecStdioSession` — ONE persistent ``docker exec -i -u <exec_user>
  <name> /opt/env/venv/bin/python /opt/lite/stdio_server.py`` per container
  (``exec_user`` defaults to ``user``; cuaworld runs it as ``root`` — see
  :func:`attach`). Maintained images must bake both files. Missing files or an
  older server protocol are image freshness / install bugs, not a cue to
  silently copy code into the container or fall back to the agent-facing system
  Python. JSON-lines request/response, serialized behind an asyncio.Lock (a
  sandbox env's steps are serial), per-call timeout, auto-respawn on pipe death.

* :class:`ExecStdioInterface` — duck-types the exact ``computer.interface``
  contract the sandbox family uses (the ~20 methods),
  so ALL existing call sites (step dispatch, screenshots, osworld
  setup/dispatch/eval) work unchanged.

* :class:`DockerProvisioner` — provisions the container for this family:
  ``docker run -d --name <n> --memory --cpus -e VNC_PW/-e VNCOPTIONS/-e
  VNC_RESOLUTION [-p novnc:6901] <image>``; readiness = a marker file probed
  over ``docker exec`` (osworld: /tmp/gnome-ready) — not a server handshake.
  No api_port; noVNC is opt-in (humans only), so headless rollouts publish
  ZERO host ports.

Run a protocol smoke locally (no docker, shell ops only):
    uv run python -c "import asyncio
    from lite.gym.sandbox.exec_stdio.client import ExecStdioSession
    async def m():
        s = ExecStdioSession('x', _local=True); await s.start()
        print(await s.call('run_command', cmd='echo hi')); await s.close()
    asyncio.run(m())"
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lite.gym.sandbox.exec_stdio.agent_shell import AgentShellSession
from lite.gym.sandbox.exec_stdio.session import DockerExecSession

logger = logging.getLogger(__name__)

_SERVER_SRC = Path(__file__).resolve().parent / "server.py"
#: Where maintained images bake server.py. This is part of the image contract:
#: callers attach to a container capability; they do not manage server code.
_SERVER_BAKED = "/opt/lite/stdio_server.py"
#: The exec_stdio server is ENV infrastructure, so it runs on the ENV python
#: (``/opt/env/venv``, base-provided) — which carries Pillow + python-xlib for
#: the fast in-process screenshot and keeps env deps off the AGENT python.
_ENV_PY = "/opt/env/venv/bin/python"

#: Must equal server.PROTOCOL_VERSION. A baked server older than the running
#: client is an image freshness / install bug. Rebuild the image instead of
#: silently downgrading the contract.
EXPECTED_SERVER_VERSION = 4

DEFAULT_CALL_TIMEOUT = 120.0
DEFAULT_RUN_COMMAND_TIMEOUT = 300.0
#: Budget for the post-spawn identity ping. Independent of the per-call timeout: a
#: server that cannot answer a ping on a freshly opened pipe is broken, not slow.
_IDENTITY_PING_TIMEOUT = 20.0

# Single-line base64 payloads — read_bytes / write_bytes (file content) and
# any future big-blob op — would trip asyncio StreamReader's default 64 KiB
# line limit (readline() raises a plain ValueError there — NOT
# asyncio.LimitOverrunError, which it catches internally). (screenshot is exempt: its raw
# PNG body is a ``header\n`` + bytes frame read with ``readexactly``, which is
# NOT bounded by this readline limit.) 64 MiB wire ≈ 48 MiB raw bytes (4/3
# base64 expansion + JSON envelope). If a future eval starts pulling files larger than ~48 MiB
# (large PDFs, embedded-media pptx, full VS Code project tars) this knob
# is what to raise — alongside chunking the op at the protocol layer.
_STREAM_LIMIT = 64 * 1024 * 1024


class ExecStdioError(RuntimeError):
    """An op the in-container server reported as failed (``ok: false``)."""


class ExecStdioSession(DockerExecSession):
    """A persistent ``docker exec`` JSON-lines channel to one container.

    ``_local=True`` runs server.py as a plain host subprocess instead of a
    docker exec — protocol-level tests need no docker/X.

    Shares its spawn/close core — and its in-container reap — with the
    agent-facing shell via
    :class:`~lite.gym.sandbox.exec_stdio.session.DockerExecSession`; the
    JSON-lines protocol (``start``/``call``/``_roundtrip``/``_respawn``/version
    probe) is this class's own. The only piece of the reap this class owns is
    LEARNING the identity to aim it at, which rides on ``ping``
    (:meth:`_note_identity`).
    """

    def __init__(self, name: str, *, timeout: float = DEFAULT_CALL_TIMEOUT,
                 exec_user: str = "user", _local: bool = False):
        super().__init__(name, timeout=timeout, exec_user=exec_user, _local=_local)
        self._id = 0
        # The in-container server path this session execs. Docker sessions use
        # the baked image path; local protocol tests use the repo source path.
        self._server_path: str = _SERVER_BAKED

    def _contract_error(self, detail: str) -> ExecStdioError:
        return ExecStdioError(
            "exec-stdio image contract failed for "
            f"{self._name!r}: {detail}. Maintained images must contain "
            f"{_ENV_PY} and {_SERVER_BAKED}; rebuild the image via the "
            "environment install.sh instead of relying on a host-side fallback."
        )

    async def _check_image_contract(self) -> None:
        """Fail fast if a Docker image predates the maintained exec-stdio ABI."""
        if self._local:
            return
        probe = (
            f"test -x {shlex.quote(_ENV_PY)} || "
            f"{{ echo missing executable: {shlex.quote(_ENV_PY)} >&2; exit 90; }}; "
            f"test -r {shlex.quote(_SERVER_BAKED)} || "
            f"{{ echo missing readable file: {shlex.quote(_SERVER_BAKED)} >&2; exit 91; }}"
        )
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-u", self._exec_user, self._name,
            "sh", "-c", probe,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            _out, err = await asyncio.wait_for(proc.communicate(), timeout=30)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            raise
        if proc.returncode != 0:
            msg = err.decode("utf-8", "replace").strip()
            detail = f"contract probe exited rc={proc.returncode}"
            if msg:
                detail += f": {msg}"
            if proc.returncode in (90, 91):
                raise self._contract_error(detail)
            raise RuntimeError(f"exec-stdio image probe failed: {detail}")

    async def _spawn(self) -> None:
        """Spawn the persistent exec running ``self._server_path``."""
        if self._local:
            argv = [sys.executable, str(_SERVER_SRC)]
        else:
            argv = ["docker", "exec", "-i", "-u", self._exec_user, self._name,
                    _ENV_PY, self._server_path]
        await self._spawn_core(
            argv,
            limit=_STREAM_LIMIT,
            # stderr → DEVNULL, NOT PIPE: nothing ever reads this persistent
            # proc's stderr, so a PIPE would fill its ~64KiB OS buffer (ffmpeg/
            # xdotool/X11 warnings from shelled-out children) and block the
            # child on write(2) → the server stops reading stdin → transport-
            # wide hang. The server reports all real errors in-band (ok:false).
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def start(self) -> None:
        """Bring the channel up and prove it with a ping handshake.

        Docker sessions require the maintained image contract:
        ``/opt/env/venv/bin/python /opt/lite/stdio_server.py``. A missing file,
        unreadable file, or older protocol is reported as image staleness
        instead of being papered over with a host-side fallback.
        """
        if self._local:
            self._server_path = str(_SERVER_SRC)
            await self._spawn()
            self._note_identity(await self.call("ping"))
            return
        self._server_path = _SERVER_BAKED
        # The image-contract probe is LAZY — the happy path (spawn +
        # ping succeed) skips it entirely, saving one `docker exec` per boot.
        # It only runs in the transient-error branch below, where a
        # server-missing / EOF / connection-style ping failure is the exact
        # signal that the baked ABI might be missing/old and the actionable
        # "rebuild the image" message is warranted.
        try:
            await self._spawn()
            res = await self.call("ping")
            v = res.get("version")
            # EXACT-match: the binary screenshot frame is version-coupled, so a host
            # and a baked image must speak the SAME protocol. Unlike the old
            # `>= EXPECTED` accept-ahead, an image ahead OR behind is rejected — the
            # image and host checkout upgrade together.
            if isinstance(v, int) and v == EXPECTED_SERVER_VERSION:
                self._note_identity(res)
                return
            await self.close()
            raise self._contract_error(
                f"server protocol version {v!r} != {EXPECTED_SERVER_VERSION} "
                "(rebuild the desktop image and the host checkout in lockstep)"
            )
        except asyncio.CancelledError:
            raise  # cancellation must propagate, not look like "baked missing"
        except ExecStdioError:
            await self.close()
            raise
        except TimeoutError:
            await self.close()
            raise
        except (OSError, asyncio.IncompleteReadError, ConnectionError) as e:
            await self.close()
            # Lazy contract probe: ping failed with a server-missing / EOF /
            # connection-style error. If the ABI is actually missing/old it
            # raises the actionable "rebuild the image" ExecStdioError; if the
            # contract holds, this was a genuine transient and we fall through
            # to raise the transient RuntimeError.
            await self._check_image_contract()
            raise RuntimeError(
                f"exec-stdio ping transient ({type(e).__name__}): {e}"
            ) from e

    #: Ops safe to AUTO-RETRY after a dead-pipe respawn: read-only / idempotent.
    #: NOT here: "input" (a re-sent click/type would double-execute — silent
    #: corrupt gestures, an honesty hazard), "write_bytes", "run_command"
    #: (often non-idempotent, e.g. ``>> file``). Those raise on mid-call pipe
    #: death so the caller sees the failure rather than a duplicated side effect.
    _RETRYABLE_OPS = frozenset({"ping", "screenshot", "read_bytes"})

    def _note_identity(self, res: Any) -> None:
        """Record the in-container pid/pgid a ``ping`` reported.

        The base ``close()`` -> ``_reap()`` kills that process GROUP inside the
        container, which is the only thing that reaches a ``run_command`` child
        still running after the host client dies. An older baked server omits the
        fields; the identity then stays ``None`` and the reap is skipped.
        """
        pid = res.get("pid") if isinstance(res, dict) else None
        pgid = res.get("pgid") if isinstance(res, dict) else None
        self._pid = pid if isinstance(pid, int) else None
        self._pgid = pgid if isinstance(pgid, int) else None

    async def _respawn(self) -> None:
        """Re-establish the exec proc using the same fixed image contract.

        ``_retire()``, NOT ``close()``: the old host client goes, its in-container
        process group STAYS. ``call()`` respawns on ``asyncio.TimeoutError`` for
        every op, and a host timeout means "no answer within the margin", not
        "the work is wrong" — so reaping here SIGKILLs a command that is merely
        slow (a ``setup_fn`` install, an evaluator getter, a chrome/sqlite
        config-flush) and converts a transport fault into a wrong SCORE. The
        retired identity rides on the session so teardown still reaps it. Then the
        fresh proc is re-identified, or the replacement server would be unreapable.
        ``_roundtrip``, not ``call``: ``call()`` already owns ``self._lock`` while
        it drives this.
        """
        await self._retire()
        await self._spawn()
        self._note_identity(
            await self._roundtrip("ping", {}, _IDENTITY_PING_TIMEOUT)
        )

    async def call(self, op: str, **kwargs: Any) -> dict | bytes:
        """One request/response. Serialized + timeout-bounded. A dead pipe is
        respawned; the op is auto-retried ONCE only if idempotent (see
        _RETRYABLE_OPS) — non-idempotent ops raise so we never double-execute.

        ``kwargs["timeout"]`` (when present) must be a positive number or a
        numeric string; it is coerced HERE, deliberately OUTSIDE the retry
        ``try``, so a caller-side bad value raises loudly instead of being
        mistaken for a wire desync and retried (the widened ``ValueError`` arm
        below would otherwise swallow it)."""
        async with self._lock:
            timeout = float(kwargs.get("timeout") or self._timeout)
            if not self._alive():
                await self._respawn()
            try:
                return await self._roundtrip(op, kwargs, timeout)
            except (BrokenPipeError, ConnectionResetError,
                    asyncio.IncompleteReadError,
                    asyncio.TimeoutError, ValueError):
                # Every way the wire can hand back a line we cannot turn into a
                # response is a pipe desync — respawn and retry idempotent ops:
                #   * ValueError covers BOTH json.JSONDecodeError (malformed /
                #     blank line) and UnicodeDecodeError (garbled bytes), and —
                #     the reason the arm is spelled ValueError and not
                #     JSONDecodeError — the OVER-LIMIT line. StreamReader.
                #     readline() does NOT raise asyncio.LimitOverrunError; it
                #     catches that internally and re-raises a PLAIN ValueError
                #     ("Separator is not found, and chunk exceed the limit").
                #     A LimitOverrunError arm therefore never fired, and an
                #     over-limit read_bytes escaped without resyncing the pipe.
                #   * asyncio.TimeoutError so a timeout mid-frame (e.g. partway
                #     through a screenshot body read) discards the half-frame via
                #     _respawn rather than leaving unread bytes for the next op.
                await self._respawn()
                if op in self._RETRYABLE_OPS:
                    return await self._roundtrip(op, kwargs, timeout)
                raise

    async def _roundtrip(self, op: str, kwargs: dict, timeout: float) -> dict | bytes:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._id += 1
        req = {"id": self._id, "op": op, **kwargs}
        self._proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()
        # readline raises a plain ValueError if a single response line exceeds
        # _STREAM_LIMIT (e.g. read_bytes of a huge file) — caught by call()'s
        # ValueError arm so a giant payload fails cleanly + resyncs instead of
        # desyncing the pipe.
        raw = await asyncio.wait_for(
            self._proc.stdout.readline(), timeout=timeout + 10)
        if not raw.strip():
            # EOF (b"") OR a stray blank line (b"\n" is truthy, so a plain
            # `not raw` check lets it through to json.loads → "Expecting value:
            # char 0"). Both are a desync, not a response — raise so call()
            # respawns + retries idempotent ops instead of crashing the op.
            raise asyncio.IncompleteReadError(b"", None)
        resp = json.loads(raw.decode("utf-8"))
        # Guard against a one-off desync returning a stale/foreign response.
        if resp.get("id") != self._id:
            raise asyncio.IncompleteReadError(b"", None)
        if not resp.get("ok"):
            raise ExecStdioError(f"{op} failed: {resp.get('error')}")
        if "shot_n" in resp:
            # Binary-body op (screenshot): the header line is followed by EXACTLY
            # shot_n raw PNG bytes. readexactly is NOT bounded by the readline
            # _STREAM_LIMIT, so multi-MB PNGs are fine. A timeout/EOF here is caught
            # by call() -> _respawn (the half-frame is discarded, no desync).
            n = resp["shot_n"]
            if not n:
                return b""
            return await asyncio.wait_for(
                self._proc.stdout.readexactly(n), timeout=timeout + 10)
        return resp["result"]


_BTN = {"left": 1, "middle": 2, "right": 3}


class ExecStdioInterface:
    """The sandbox family's ``computer.interface`` contract over an
    :class:`ExecStdioSession`."""

    def __init__(self, session: ExecStdioSession):
        self._s = session

    # -- screen / shell / fs --------------------------------------------------

    async def screenshot(self) -> bytes:
        # call("screenshot") returns the raw PNG bytes directly — the exec_stdio
        # binary frame (header line + body), decoded by _roundtrip. No base64.
        return await self._s.call("screenshot")

    async def run_command(self, command: str,
                          timeout: float | None = None) -> Any:
        """``timeout`` bounds BOTH the in-container subprocess (server-side,
        default 300s) and the host-side wait — long replays pass theirs. The command
        runs as the server's uid (the env's desktop user; root for cuaworld)."""
        if timeout is not None and timeout <= 0:
            raise ValueError("run_command timeout must be positive")
        kw: dict[str, Any] = {
            "timeout": timeout if timeout is not None else DEFAULT_RUN_COMMAND_TIMEOUT,
        }
        res = await self._s.call("run_command", cmd=command, **kw)
        return SimpleNamespace(stdout=res["stdout"], stderr=res["stderr"],
                               returncode=res["returncode"])

    async def read_bytes(self, path: str) -> bytes:
        res = await self._s.call("read_bytes", path=path)
        return base64.b64decode(res["b64"])

    async def write_bytes(self, path: str, data: bytes) -> None:
        """The file is written by the server's uid (the env's desktop user), so it is
        born user-owned — no ownership channel."""
        await self._s.call("write_bytes", path=path,
                           b64=base64.b64encode(data).decode("ascii"))

    async def get_screen_size(self) -> dict:
        res = await self.run_command("xdotool getdisplaygeometry")
        parts = res.stdout.split()
        if len(parts) >= 2:
            return {"width": int(parts[0]), "height": int(parts[1])}
        # bare context has no DISPLAY — retry with the canonical one
        res = await self.run_command("DISPLAY=:1 xdotool getdisplaygeometry")
        parts = res.stdout.split()
        return {"width": int(parts[0]), "height": int(parts[1])} \
            if len(parts) >= 2 else {"width": 1920, "height": 1080}

    async def get_cursor_position(self) -> dict:
        res = await self.run_command(
            "DISPLAY=:1 xdotool getmouselocation --shell")
        vals: dict[str, int] = {}
        for ln in res.stdout.splitlines():
            k, _, v = ln.partition("=")
            if k in ("X", "Y") and v.strip().lstrip("-").isdigit():
                vals[k.lower()] = int(v)
        return {"x": vals.get("x", 0), "y": vals.get("y", 0)}

    # -- keyboard -------------------------------------------------------------

    async def type_text(self, text: str) -> None:
        # Mirror server.py's op_input per-row xdotool budget on the host so
        # the client wait outlasts the server work. Server-side each row's
        # ``_xdo("type", "--delay", delay_ms, ..., cell)`` gets a budget of
        # ``30 + len(cell) * delay_ms/1000 * 2`` seconds, and rows run
        # sequentially → server total ≤ ``30 + len(text) * delay_ms/1000 * 2``.
        # Pre-fix the host wait was a flat 130s (``DEFAULT_CALL_TIMEOUT + 10``),
        # so any single ``type`` of more than ~4 KiB at delay=12ms made the
        # client bail BEFORE the server finished — the response then arrived
        # on a dead pipe, ``call()`` saw IncompleteRead, and because ``input``
        # is correctly NOT in ``_RETRYABLE_OPS`` (no double-typing), the caller
        # got an exception even though the server actually typed it. Compute
        # a budget that strictly dominates the server's total; never go below
        # the default to avoid pessimizing the common short-type case.
        # Mirror the server's own row budget, including the floor it raises the
        # delay to for a segment with a non-keymap codepoint (12 ms drops those
        # characters outright). Use that floor whenever the text is not pure
        # ASCII so the host wait keeps strictly dominating the server's.
        _server_default_delay_ms = 12 if text.isascii() else 30
        _server_fixed_per_row_s = 30.0
        budget = max(
            DEFAULT_CALL_TIMEOUT,
            _server_fixed_per_row_s
            + len(text) * (_server_default_delay_ms / 1000.0) * 2
            + 30.0,  # host-side cushion beyond the server's row budget
        )
        # ``call()`` reads ``timeout`` from kwargs to bound the host-side
        # readline; the server's op_input ignores the field, so this is
        # purely a client-side wait extension.
        await self._s.call("input", action="type", text=text, timeout=budget)

    async def hotkey(self, *keys: str) -> None:
        await self._s.call("input", action="key", keys=list(keys))

    async def press_key(self, key: str) -> None:
        await self._s.call("input", action="key", keys=[key])

    async def key_down(self, key: str) -> None:
        await self._s.call("input", action="keydown", key=key)

    async def key_up(self, key: str) -> None:
        await self._s.call("input", action="keyup", key=key)

    # -- mouse ----------------------------------------------------------------

    async def left_click(self, x: int | None = None, y: int | None = None) -> None:
        await self._s.call("input", action="click", x=x, y=y, button=1)

    async def right_click(self, x: int | None = None, y: int | None = None) -> None:
        await self._s.call("input", action="click", x=x, y=y, button=3)

    async def multi_click(self, x: int | None = None, y: int | None = None,
                          button: str = "left", clicks: int = 2) -> None:
        """``clicks`` presses of ``button`` at (x, y), as ONE xdotool repeat.

        ``--repeat`` keeps the presses inside the X server's double-click interval,
        which a Python-side loop cannot guarantee -- that is why this is one call
        and not N ``left_click``s.
        """
        await self._s.call("input", action="click", x=x, y=y,
                           button=_BTN[button], repeat=clicks)

    async def double_click(self, x: int | None = None, y: int | None = None,
                           button: str = "left") -> None:
        await self.multi_click(x, y, button=button, clicks=2)

    async def move_cursor(self, x: int, y: int) -> None:
        await self._s.call("input", action="move", x=x, y=y)

    async def mouse_down(self, x: int | None = None, y: int | None = None,
                         button: str = "left") -> None:
        await self._s.call("input", action="mouse_down", x=x, y=y,
                           button=_BTN[button])

    async def mouse_up(self, x: int | None = None, y: int | None = None,
                       button: str = "left") -> None:
        await self._s.call("input", action="mouse_up", x=x, y=y,
                           button=_BTN[button])

    async def drag(self, path: list, button: str = "left",
                   duration: float = 0.5) -> None:
        # press at the start, move through the path, release at the end.
        #
        # TIMING IS LOAD-BEARING for toolkit drag-and-drop (e.g. moving a field
        # in the Pivot Table Layout dialog, or any GtkTreeView DnD). GTK starts a
        # DnD operation only when it sees the pointer move PAST the drag threshold
        # (~8px) WHILE the button is held, delivered as a sequence its main loop
        # can process. A bare mousedown→teleport-to-target→mouseup arrives faster
        # than the widget iterates, so GTK reads it as a plain click/select and
        # the item never moves — the exact drag no-op seen in rollouts (agent
        # drags a pivot field, nothing happens, it falls back to double-click).
        # So: dwell after the press (arm the source), move in SMALL interpolated
        # steps paced over ``duration`` (cross the threshold + feed continuous
        # motion), then dwell on the target before releasing (let the drop land).
        # Verified on the Pivot Table dialog: the un-timed form no-ops, this
        # form moves the field. ``duration`` paces the interpolated motion, matching the
        # official OSWorld ``pyautogui.dragTo(..., duration=0.5)`` shape. The
        # release runs in a ``finally`` so a failed intermediate move can never
        # leave the mouse button stuck down.
        if not path:
            return
        pts = [(int(x), int(y)) for (x, y) in path]
        sx, sy = pts[0]
        # Interpolate every segment to ~15px steps so GTK sees the drag-threshold
        # crossing and continuous motion (a 2-point endpoints-only path included).
        steps: list[tuple[int, int]] = []
        prev = (sx, sy)
        for (tx, ty) in pts[1:]:
            n = max(1, int(max(abs(tx - prev[0]), abs(ty - prev[1])) / 15))
            for i in range(1, n + 1):
                steps.append((round(prev[0] + (tx - prev[0]) * i / n),
                              round(prev[1] + (ty - prev[1]) * i / n)))
            prev = (tx, ty)
        # Pace moves over ``duration`` but keep a per-step floor so GTK's main
        # loop actually processes each motion event (very long drags stay smooth
        # without becoming instant); this floor matches the validated pacing.
        per_step = max(0.015, duration / len(steps)) if steps else 0.0
        ex, ey = pts[-1]
        await self.mouse_down(sx, sy, button=button)
        try:
            await asyncio.sleep(0.3)               # arm the source widget's DnD
            for (x, y) in steps:
                await self.move_cursor(x, y)
                await asyncio.sleep(per_step)
            await asyncio.sleep(0.3)               # dwell on target before drop
        finally:
            await self.mouse_up(ex, ey, button=button)

    async def scroll(self, x: int, y: int) -> None:
        # cua/pynput semantics: scroll(x, y) drives BOTH axes — y>0 up / y<0
        # down, x>0 right / x<0 left. base.py's left/right branch calls
        # scroll(±amount, 0); a y-only handler would mis-scroll those down
        # (old pynput backend honored x — parity matrix EXEC_STDIO_PARITY).
        if x and not y:
            await self._s.call("input", action="scroll", axis="x",
                               direction="right" if x > 0 else "left",
                               amount=abs(int(x)) or 1)
        else:
            await self._s.call("input", action="scroll",
                               direction="up" if y > 0 else "down",
                               amount=abs(int(y)) or 1)

    async def scroll_down(self, clicks: int = 1) -> None:
        await self._s.call("input", action="scroll", direction="down",
                           amount=clicks)

    async def scroll_up(self, clicks: int = 1) -> None:
        await self._s.call("input", action="scroll", direction="up",
                           amount=clicks)


class _ContainerHandle:
    """What ``env._computer`` IS on the exec-stdio path: the two members the
    cb code actually reads (verified surface: base.py reads only ``.run()`` —
    never on this path — ``.stop()`` and ``.interface``). NOT a cua Computer."""

    def __init__(self, name: str, session: ExecStdioSession, *, agent_user: str = "user"):
        self.name = name
        self._session = session
        self._agent_user = agent_user
        self._agent_shell: AgentShellSession | None = None
        self.interface = ExecStdioInterface(session)

    @property
    def agent_shell(self) -> AgentShellSession:
        if self._agent_shell is None:
            self._agent_shell = AgentShellSession(self.name, exec_user=self._agent_user)
        return self._agent_shell

    async def close_agent_shell(self) -> None:
        if self._agent_shell is not None:
            await self._agent_shell.close()
            self._agent_shell = None

    async def stop(self) -> None:
        """Close the stdio session. Container removal is the owner's job
        (env.close() routes it through ``docker_rm_f_async``)."""
        await self.close_agent_shell()
        await self._session.close()


def _mem_to_docker(mem: Any) -> str:
    """'4GB'/'2048MB'/4096 → docker --memory string (parity with the retired
    provider's _parse_memory)."""
    if isinstance(mem, int):
        return f"{mem}m"
    s = str(mem).strip().upper()
    for suf, unit in (("GB", "g"), ("G", "g"), ("MB", "m"), ("M", "m")):
        if s.endswith(suf) and s[: -len(suf)].isdigit():
            return s[: -len(suf)] + unit
    return "4g"


class DockerProvisioner:
    """Boot/teardown one sandbox-family container — no cua provider, no api_port.

    ``run()`` arg-parity with the retired cua docker provider (memory/cpus/
    VNC_PW/VNCOPTIONS/VNC_RESOLUTION), minus the computer-server port. noVNC
    is published only when ``vnc_port`` is given (humans watching); headless
    rollouts publish zero host ports. Readiness = ``ready_marker`` file probed
    over ``docker exec`` (osworld images touch /tmp/gnome-ready when the
    desktop is up); ``None`` falls back to "exec answers" (generic images).
    """

    def __init__(self, name: str, computer_cfg: dict, *,
                 image: str | None = None,
                 vnc_port: int | None = None,
                 ready_marker: str | None = "/tmp/gnome-ready",
                 ready_timeout: float = 120.0):
        self.name = name
        self._cfg = computer_cfg
        self._image = image or computer_cfg.get("image")
        self._vnc_port = vnc_port
        self._ready_marker = ready_marker
        self._ready_timeout = ready_timeout

    async def run(self) -> None:
        cmd = ["docker", "run", "-d", "--name", self.name]
        # A container can't change its own hostname (no CAP_SYS_ADMIN), so an env
        # that needs a fixed guest hostname requests it here as a docker-run flag —
        # e.g. osworld pins "user-virtual-machine" so terminal prompts/titles and
        # `hostname`/`uname -n` match its reference VM instead of the container id.
        if self._cfg.get("hostname"):
            cmd += ["--hostname", str(self._cfg["hostname"])]
        if self._cfg.get("memory"):
            cmd += ["--memory", _mem_to_docker(self._cfg["memory"])]
        if self._cfg.get("cpu"):
            cmd += ["--cpus", str(self._cfg["cpu"])]
        if self._cfg.get("gpus"):
            # int N → reserve N GPUs; str → pass through (e.g. "all", "device=2")
            g = self._cfg["gpus"]
            cmd += ["--gpus", str(g) if not isinstance(g, str) else g]
        cmd += ["-e", "VNC_PW=password", "-e", "VNCOPTIONS=-disableBasicAuth"]
        if self._cfg.get("display"):
            cmd += ["-e", f"VNC_RESOLUTION={self._cfg['display']}"]
        if self._vnc_port:
            cmd += ["-p", f"127.0.0.1:{self._vnc_port}:6901"]
        cmd += [self._image]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        _out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker run {self.name} failed (rc={proc.returncode}): "
                f"{err.decode('utf-8', 'replace')[:300]}")
        await self._wait_ready()

    async def _wait_ready(self) -> None:
        # Poll readiness with ONE guest-side loop instead of spawning a
        # fresh `docker exec` every second. The old host-side poll was N docker
        # execs/s of daemon churn under concurrency; here a SINGLE `docker exec`
        # runs the until-marker loop inside the container (sleep 0.5 guest-side),
        # and the outer asyncio.wait_for is the hard host deadline so a hung
        # exec can't outlive the budget — it's killed on timeout. Semantics are
        # identical to before: same marker path, same overall timeout budget
        # (self._ready_timeout), same failure (RuntimeError, same message). The
        # None-marker case keeps its old meaning ("exec answers" → `true`).
        budget = int(self._ready_timeout)
        if self._ready_marker:
            marker = shlex.quote(self._ready_marker)
            probe = (f"end=$((SECONDS+{budget})); until test -f {marker}; do "
                     f"[ $SECONDS -ge $end ] && exit 1; sleep 0.5; done")
        else:
            probe = "true"
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", self.name, "bash", "-c", probe,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL)
        try:
            # Slack beyond the guest budget covers `docker exec` spin-up and the
            # final `sleep 0.5` granularity, so the guest loop's own `exit 1`
            # wins the race and we report the same failure as the old poll.
            last_rc = await asyncio.wait_for(
                proc.wait(), timeout=self._ready_timeout + 15)
        except (TimeoutError, asyncio.TimeoutError):
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            last_rc = None
        if last_rc == 0:
            return
        raise RuntimeError(
            f"{self.name}: ready marker {self._ready_marker!r} not up after "
            f"{self._ready_timeout:.0f}s (last rc={last_rc})")

async def attach(name: str, *, call_timeout: float = DEFAULT_CALL_TIMEOUT,
                 exec_user: str = "user", agent_user: str = "user") -> _ContainerHandle:
    """Open an exec-stdio session to a RUNNING container (provisioned by us or
    injected by name) and return the env-facing handle. ``exec_user`` is the
    in-container user the persistent ``docker exec`` runs as. The default is
    ``user`` — and the sandbox path uses exactly that: sandbox ``base.py`` passes
    ``exec_user=EXEC_USER`` which is ``"user"`` for every env EXCEPT cuaworld. The
    server runs as the desktop user and reads its own baked
    ``/opt/lite/{stdio_server.py,cursor.png}`` (``a+rX``, user-readable). cuaworld
    overrides ``EXEC_USER="root"`` because its upstream hooks drop to the desktop
    user via ``su - <desktop-user> -c`` (passwordless only from root)."""
    session = ExecStdioSession(name, timeout=call_timeout, exec_user=exec_user)
    await session.start()
    return _ContainerHandle(name, session, agent_user=agent_user)


__all__ = [
    "AgentShellSession", "DEFAULT_CALL_TIMEOUT", "DEFAULT_RUN_COMMAND_TIMEOUT",
    "DockerProvisioner", "ExecStdioError", "ExecStdioInterface", "ExecStdioSession",
    "_ContainerHandle", "attach",
]
