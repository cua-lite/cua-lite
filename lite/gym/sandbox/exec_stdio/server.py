#!/usr/bin/env python3
"""In-container command server for the exec-stdio transport (sandbox family).

Runs INSIDE the env container on ``/opt/env/venv/bin/python`` and is reached
over ONE persistent ``docker exec -i`` pipe (not a network port).
Protocol: one JSON object per stdin line in, one JSON object per
stdout line out — except ``screenshot``, whose result is framed as ``header\n``
+ raw PNG bytes (read with ``readexactly``, not a JSON line). Multi-MB base64
responses (``read_bytes``/``write_bytes``) still ride one line, so the host
reader must raise its stream limit — see client.py.

Maintained images bake this file at ``/opt/lite/stdio_server.py``; client.py
execs that baked copy with the env Python. Production host↔container traffic
does not copy repo source into the container at runtime. Local protocol tests
may execute the repo copy directly under ``_local=True``. Missing baked files
or stale protocol versions are image freshness / install bugs.

Ops (request ``{"id": n, "op": <name>, ...}`` → response
``{"id": n, "ok": true, "result": {...}}`` or ``{"ok": false, "error": str}``):

- ``ping``        → {version, pid, pgid}         (liveness + version + the
      in-container identity the host's reap targets; pid/pgid are additive and
      absent from images baked before them — see :func:`op_ping`)
- ``run_command`` {cmd}                          → {stdout, stderr, returncode}
      Session context: every run_command child inherits the GNOME session bus by
      exporting DBUS_SESSION_BUS_ADDRESS (read from /tmp/dbus-session-bus-address)
      + GTK_MODULES=gail:atk-bridge — gsettings/dconf, notifications, AT-SPI all
      depend on it. We re-read the marker per call (cheap; robust to session
      restarts). DISPLAY/HOME ride the image env.
- ``screenshot``  {}                             → raw PNG bytes (``header\n`` + body frame, not a JSON line)
      ffmpeg x11grab on $DISPLAY (in the base image); ImageMagick ``import``
      fallback (no xwd/scrot in the image).
- ``input``       {action, ...}                  → {}
      xdotool (XTest) on $DISPLAY: type / key / keydown / keyup / click /
      move / mouse_down / mouse_up / scroll. xdotool resolves keysyms
      directly over XTest, with no "pagedown"-alias key-mapping bug.
- ``read_bytes``  {path}                         → {b64}
- ``write_bytes`` {path, b64}                    → {}
      (used by osworld dispatch.py:638 / eval runner.py:1544)

Run (only ever by client.py): /opt/env/venv/bin/python /opt/lite/stdio_server.py
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import pwd
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

# Bump when the request/response protocol changes. client.py checks this against
# EXPECTED_SERVER_VERSION; an older baked server means the image must be rebuilt.
PROTOCOL_VERSION = 4
DEFAULT_RUN_COMMAND_TIMEOUT = 300.0

#: Floor for ``xdotool type --delay`` on a segment holding a non-keymap codepoint.
#: 12 ms drops them outright; 30 ms was the lowest measured value that kept every
#: character. See ``op_input``.
_NON_ASCII_DELAY_MS = 30

# The env venv (/opt/env/venv, the eval getters' python + bs4/websocket/openpyxl) and
# vendored CLIs (/opt/env/bin: xdotool/jq/pandoc/pdftk/xclip/convert, off /usr/bin) are
# HARNESS tools. The server runs as the env's desktop user (EXEC_USER — `user` for
# osworld/scalecua/cuagym, `root` for cuaworld), so prepend /opt/env to THIS server's PATH
# once, at module level: it feeds both op_run_command (env = dict(os.environ)) and _X_ENV
# below (op_input/op_screenshot). If a setup opens an agent-visible shell surface,
# image-level launcher wrappers (terminal aliases, VS Code) scrub PATH before that
# shell appears to the agent; do not remove those wrappers while keeping this
# env-facing server PATH.
os.environ["PATH"] = "/opt/env/venv/bin:/opt/env/bin:" + os.environ.get("PATH", "")

DISPLAY = os.environ.get("DISPLAY", ":1")

# The desktop user this container runs as. The image bakes `ENV HOME=/home/<user>`, so
# derive the account from it rather than hardcoding a name (default build `user`; the same
# base built with `--build-arg USER=ga` is cuaworld's `ga`).
_HOME = os.environ.get("HOME", "/home/user")
_DESKTOP_USER = os.path.basename(_HOME) or "user"
#: uid of the desktop user, for its /run/user/<uid> session runtime dir. Looked up
#: rather than assumed: the server runs as that user for osworld/scalecua/cuagym but
#: as ROOT for cuaworld, so `os.getuid()` would give 0 there and misdirect every
#: session tool at a /run/user/0 no image creates.
#:
#: The fallback is NOT defensive programming — it is what makes this module
#: importable OUTSIDE a container. The host test suite imports it directly, and a
#: dev host's HOME can have a basename that names no local account, so an
#: unguarded `getpwnam` raises KeyError at import and takes 23 tests with it. In a
#: container the lookup always succeeds, so the fallback is dead weight there and
#: only the looked-up value is ever used.
try:
    _DESKTOP_UID = pwd.getpwnam(_DESKTOP_USER).pw_uid
except KeyError:
    _DESKTOP_UID = 1000  # host import only; `useradd` gives the image's first user 1000

# The GUI session context for INPUT + SCREENSHOT ops (they must land on the env's desktop).
# Inherits the /opt/env PATH set above. LC_CTYPE/LANG pinned to UTF-8: `xdotool type` aborts
# on the first non-ASCII byte under a POSIX/C locale — pin it so accented/CJK typing can't
# silently regress if the baked image's user locale ever changes.
_X_ENV = {**os.environ, "DISPLAY": DISPLAY, "HOME": _HOME,
          "LC_CTYPE": "C.UTF-8", "LANG": os.environ.get("LANG", "C.UTF-8")}


def _run(argv: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=_X_ENV, capture_output=True, timeout=timeout)


@contextlib.contextmanager
def _tmpfile(suffix: str):
    """A unique temp path, removed on exit. UNIQUE (mkstemp), not a fixed name:
    two ExecStdioSessions can attach to the SAME container (owner + env), so a
    fixed path would let concurrent ops clobber each other's files and return
    the wrong bytes to the wrong call."""
    fd, path = tempfile.mkstemp(prefix=".lite_", suffix=suffix)
    os.close(fd)
    try:
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def op_ping(req: dict) -> dict:
    """Liveness + version handshake, and the ONE place this server's in-container
    identity crosses the pipe.

    ``pid``/``pgid`` are what the host's reap targets: killing the host
    ``docker exec`` client does not signal anything inside the container, so
    without them a command still running here (or a wedged ``_grab_xlib``)
    outlives the session that started it. They are ADDITIVE and optional — an
    older baked server omits them and the host simply skips the reap, so this
    deliberately does NOT bump ``PROTOCOL_VERSION`` (which is exact-matched by
    the client and would force every image to be rebuilt in lockstep for a
    field that is backward-compatible in both directions).
    """
    return {"version": PROTOCOL_VERSION, "pid": os.getpid(), "pgid": os.getpgrp()}


# `pkill/pgrep -f <pat>` matches the transport's OWN `bash -c '… <pat> …'` command
# line and kills/flags it — masking the real returncode (empty response, rc coerced
# to 0) or self-terminating the backend. Bracket the pattern's first char
# (`[c]hrome`) so it still matches the target process but never the literal `<pat>`
# text in our own argv. Emit it single-quoted (an unquoted `[c]hrome` would glob).
# VERIFIED: still kills, rc reaches the caller, both uids.
_PKILL_F_RE = re.compile(
    # match `-f` whether standalone or bundled in a short-flag group (`-if`, `-af`,
    # `pgrep -fi`): a flag token containing `f`. (Signal `-9` is separate, consumed
    # by the preceding `[^\n;|&]*?`.)
    r"(\bp(?:kill|grep)\b[^\n;|&]*?\s-[A-Za-z]*f[A-Za-z]*\s+)(?:'([^']*)'|\"([^\"]*)\"|([^\s;&|<>()]+))")


def _bracket_pkill(cmd: str) -> str:
    def _sub(m: re.Match) -> str:
        pat = next((g for g in m.groups()[1:] if g is not None), "")
        # A pattern already containing a char-class `[...]` is self-avoiding (its
        # own argv can't match) — and bracketing a LEADING regex metachar would
        # corrupt the ERE: `\[STUB\]` -> `[\][STUB\]` (unterminated class),
        # `^qemu` -> `[^]qemu` (bad negated class), `.*http` -> `[.]*http` (loses
        # `.*`). So skip if `[` is present, else bracket the first ALPHANUMERIC
        # char — regex-inert inside a class, and still breaks the self-match
        # (guards against latent _bracket_pkill metachar corruption).
        if not pat or "[" in pat:  # empty or already class-bearing → leave it
            return m.group(0)
        idx = next((i for i, c in enumerate(pat) if c.isalnum()), None)
        if idx is None:  # no alnum char to bracket (all-metachar) → leave it
            return m.group(0)
        return f"{m.group(1)}'{pat[:idx]}[{pat[idx]}]{pat[idx + 1:]}'"
    return _PKILL_F_RE.sub(_sub, cmd)


def op_run_command(req: dict) -> dict:
    # stdout/stderr go to FILES, not pipes: a command that daemonizes a
    # long-lived child (soffice, chrome, launch.sh ...) leaves the pipe held
    # open after bash exits, and pipe-capture would block on EOF until the
    # daemon dies — the classic run_command hang. File redirection waits only
    # for the bash process itself.
    env = dict(os.environ)
    # USER/LOGNAME: run_command children must see $USER/$LOGNAME, but
    # `docker exec -u <user>` does NOT set them (it sets uid only) and `bash -c`
    # is non-login, so they'd be empty — GUI apps / scripts that read $USER would
    # diverge. Use the container's actual desktop account (derived from HOME) so
    # this is correct for both the `user` and `ga` builds.
    env.setdefault("USER", _DESKTOP_USER)
    env.setdefault("LOGNAME", _DESKTOP_USER)
    # GNOME session bus: run_command children must attach to it, else gsettings
    # setups silently no-op and chrome launches miss the session → trivial_pass
    # false-greens (oracle validate caught this drift).
    try:
        with open("/tmp/dbus-session-bus-address") as fh:
            env["DBUS_SESSION_BUS_ADDRESS"] = fh.read().strip()
    except OSError:
        pass
    env.setdefault("GTK_MODULES", "gail:atk-bridge")
    # XDG_RUNTIME_DIR: session getters reach the per-user pulseaudio socket + session
    # bus through it (e.g. the audio `pactl get-sink-volume` getter). `docker exec -u`
    # sets uid only and `bash -c` is non-login, so it would be unset — pactl then fails
    # "Connection refused". Restore it to the session runtime dir (same value the
    # supervisord programs export). setdefault: a hook that exports its own value wins.
    #
    # /run/user/<uid>, the STANDARD freedesktop location systemd-logind would create —
    # not the /tmp/runtime-<user> this used to invent. The OSWorld corpus hardcodes the
    # standard paths (it was authored against the QEMU VM), so presenting them makes
    # those commands correct instead of silently misdirected; /tmp/runtime-<user>
    # remains as a compat symlink. See the session-bus block in docker/Dockerfile.linux.
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{_DESKTOP_UID}")
    # The server runs as the env's desktop user (EXEC_USER; user for
    # osworld/scalecua/cuagym, root for cuaworld), so every command runs
    # as that uid — plain `bash -c`, no per-command uid routing, no privilege drop. /opt/env is
    # already on env["PATH"] (inherited from the module-level os.environ prepend), so a
    # getter's bare `python3`/`jq` resolves to the env venv/CLIs.
    _cmd = _bracket_pkill(req["cmd"])
    argv = ["bash", "-c", _cmd]
    with _tmpfile(".out") as out_f, _tmpfile(".err") as err_f:
        with open(out_f, "wb") as fo, open(err_f, "wb") as fe:
            # Child processes must never consume the server's JSONL protocol stdin.
            p = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=fo,
                stderr=fe,
                env=env,
                timeout=float(
                    req["timeout"] if req.get("timeout") is not None
                    else DEFAULT_RUN_COMMAND_TIMEOUT
                ),
            )
        with open(out_f, "rb") as fo, open(err_f, "rb") as fe:
            return {"stdout": fo.read().decode("utf-8", "replace"),
                    "stderr": fe.read().decode("utf-8", "replace"),
                    "returncode": p.returncode}


_CURSOR_SPRITE = None
_CURSOR_SPRITE_LOADED = False
_CURSOR_SPRITE_PATH = "/opt/lite/cursor.png"


def _cursor_sprite():
    """The baked ~16x24 straight-alpha cursor sprite, loaded once (or ``None`` if
    the image predates the bake — then capture works, just cursor-less)."""
    global _CURSOR_SPRITE, _CURSOR_SPRITE_LOADED
    if not _CURSOR_SPRITE_LOADED:
        _CURSOR_SPRITE_LOADED = True
        try:
            from PIL import Image
            _CURSOR_SPRITE = Image.open(_CURSOR_SPRITE_PATH).convert("RGBA")
        except Exception:
            _CURSOR_SPRITE = None
    return _CURSOR_SPRITE


def composite_cursor(img: "Image.Image", sprite: "Image.Image",
                     x: int, y: int) -> "Image.Image":
    """Alpha-composite the straight-alpha cursor ``sprite`` onto ``img`` at
    ``(x, y)`` and return an RGB image. Pure (no X, no I/O), so it is unit-golden
    testable without a live capture.

    The sprite's arrow tip is at its top-left (≈0,0) and it is straight
    (non-premultiplied) alpha, so paste at the RAW ``(x, y)`` — NO XFixes hotspot
    (``- xhot/- yhot``) offset and NO ``r*255//a`` un-premultiply (both are
    XFixes-only and would mis-place / brighten the sprite)."""
    img = img.convert("RGBA")
    img.alpha_composite(sprite, (x, y))
    return img.convert("RGB")


def _grab_xlib() -> bytes:
    """Fast in-process capture: Xlib ``XGetImage`` (screen) + capture-time cursor
    composite + PIL PNG. ~24ms vs ffmpeg x11grab's ~400ms, and pixel-identical.
    Runs on the ENV python (/opt/env/venv), which carries Pillow + python-xlib;
    op_screenshot falls back to ffmpeg if this raises. The exec_stdio server is env
    infrastructure, so it runs on the env python (not the agent python) — these
    imports are env-facing and never touch the agent's interpreter."""
    import io

    from PIL import Image
    from Xlib import X, display

    d = display.Display(DISPLAY)
    try:
        root = d.screen().root
        geo = root.get_geometry()
        W, H = geo.width, geo.height
        raw = root.get_image(0, 0, W, H, X.ZPixmap, 0xffffffff)
        img = Image.frombytes("RGB", (W, H), raw.data, "raw", "BGRX")

        # Draw the cursor at capture. The lite.* desktop is headless, so the X
        # server exposes no cursor sprite (XFixes returns nothing) — read the
        # pointer via XQueryPointer and alpha_composite the baked ~16x24 straight-
        # alpha sprite at the raw (x, y): the sprite's tip is at its top-left, so
        # NO hotspot offset, and it is straight (non-premultiplied) alpha, so NO
        # un-premultiply.
        #
        # This composite is UNCONDITIONAL by design — the sandbox always draws its
        # own cursor in the capture path, so no host/client repaint should be
        # layered on top.
        sprite = _cursor_sprite()
        if sprite is not None:
            p = root.query_pointer()
            img = composite_cursor(img, sprite, p.root_x, p.root_y)

        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
    finally:
        d.close()


def op_screenshot(req: dict) -> bytes:
    # Raw PNG **bytes** (not base64 in a JSON line): the main loop frames a
    # ``bytes`` result as ``header\n`` + the raw body, so the host reads it with
    # ``readexactly`` and never json-walks/b64-decodes the multi-MB blob on its
    # event loop. Fast in-process capture (Xlib + capture-cursor + PIL PNG, ~24ms);
    # falls back to ffmpeg x11grab (~400ms), then ImageMagick import.
    try:
        return _grab_xlib()
    except Exception as e:
        _xlib_err = f"{type(e).__name__}: {e}"
    with _tmpfile(".png") as out:
        p = _run(["ffmpeg", "-loglevel", "error", "-f", "x11grab", "-i", DISPLAY,
                  "-frames:v", "1", "-y", out], timeout=20)
        if p.returncode != 0 or os.path.getsize(out) == 0:
            # ImageMagick fallback (xwd/scrot absent in the base image).
            p = _run(["import", "-window", "root", out], timeout=20)
            if p.returncode != 0:
                raise RuntimeError(
                    f"screenshot failed: xlib({_xlib_err}) + ffmpeg/import "
                    f"rc={p.returncode} {p.stderr.decode('utf-8', 'replace')[:200]}")
        with open(out, "rb") as fh:
            return fh.read()


def _xdo(*args: str, timeout: float = 30.0) -> None:
    p = _run(["xdotool", *args], timeout=timeout)
    err = p.stderr.decode("utf-8", "replace")
    # xdotool returns rc=0 even when a keysym does not resolve, only warning
    # "No such key name '<x>'. Ignoring it." on stderr → the keypress silently
    # does NOTHING but the op looks successful (an honesty hazard, same class as
    # the uppercase-key bug). Treat that warning as a hard failure so an
    # unmapped key raises loudly instead of no-op'ing.
    if p.returncode != 0 or "No such key name" in err:
        raise RuntimeError(
            f"xdotool {' '.join(args[:3])}... rc={p.returncode} {err[:200]}")


def _move_if_xy(req: dict) -> None:
    """Move the cursor first when the action carries coordinates (click /
    mouse_down / mouse_up may target a point or act at the current position)."""
    if req.get("x") is not None and req.get("y") is not None:
        _xdo("mousemove", str(int(req["x"])), str(int(req["y"])))


# Low-level exec-stdio/raw-container key name -> X keysym. Lite env runtime input
# is validated and projected before it reaches this server; this table exists
# only for the raw xdotool adapter surface. It is orthogonal to case: _norm_key
# lowercases the token BEFORE the lookup, so 'CTRL'/'Ctrl'/'ctrl' all hit the
# 'ctrl' entry. (_KEYSYM = name→keysym map; _norm_key = case-fold +
# single-char handling + this lookup.) xdotool needs the exact keysym and does
# not resolve these names itself.
_KEYSYM = {
    "enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape",
    "del": "Delete", "delete": "Delete", "backspace": "BackSpace", "bksp": "BackSpace",
    "tab": "Tab", "space": "space", "spacebar": "space",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    # Raw direct callers may send "arrowdown" etc. instead of bare "down".
    "arrowup": "Up", "arrowdown": "Down", "arrowleft": "Left", "arrowright": "Right",
    # common aliases that xdotool does NOT resolve on its own (it needs the
    # exact X keysym — it is NOT the case-insensitive/alias-rich resolver one
    # might assume; the _KEYSYM table is the authority, not a convenience layer).
    "menu": "Menu", "contextmenu": "Menu", "printscreen": "Print",
    "print_screen": "Print", "prtsc": "Print", "clear": "Clear",
    "kp_enter": "KP_Enter", "numpad_enter": "KP_Enter",
    "pageup": "Prior", "page_up": "Prior", "pgup": "Prior",
    "pagedown": "Next", "page_down": "Next", "pgdn": "Next",
    "home": "Home", "end": "End", "insert": "Insert",
    "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "option": "alt",
    "shift": "shift", "super": "super", "win": "super", "cmd": "super",
    # 'command'/'meta' = the macOS Cmd key. At this raw-container boundary both
    # map to GNOME's Super keysym, matching the existing cmd/win aliases.
    "command": "super", "meta": "super", "capslock": "Caps_Lock",
    "minus": "minus", "plus": "plus", "equal": "equal",
    **{f"f{i}": f"F{i}" for i in range(1, 13)},
}

# Single printable punctuation → X keysym NAME. xdotool's ``key`` wants keysym
# names, not the literal glyph: ``xdotool key ,`` warns "No such key name ','"
# and no-ops, while ``xdotool key comma`` works. Agents emit chords like
# ``ctrl+,`` (Preferences) and ``ctrl+.`` (VLC frame-step) with the bare glyph,
# so a chord member that is one punctuation char is rewritten to its keysym.
# (Digits 0-9 and letters ARE valid keysym names as-is, so they stay literal.)
_PUNCT = {
    ",": "comma", ".": "period", "/": "slash", "\\": "backslash",
    ";": "semicolon", "'": "apostrophe", "`": "grave",
    "[": "bracketleft", "]": "bracketright",
    "-": "minus", "+": "plus", "=": "equal",
}


def _norm_key(k: str) -> str:
    """Map one key token to an xdotool keysym (case-insensitive + aliases).

    CRITICAL: a single ASCII LETTER is lowercased. In xdotool a capital letter
    keysym ('O') means the SHIFTED key, so a chord like ``ctrl+O`` is Ctrl+Shift+o
    — NOT Ctrl+o. Agents emit keys uppercase ('CTRL','O','V'), so without this a
    'ctrl o' (Open) became Ctrl+Shift+O and 'ctrl v' (Paste) became Ctrl+Shift+V
    (Paste-Special) — the action landed as the WRONG shortcut. Shift is expressed
    by an explicit 'shift' modifier token, never by letter case. A single
    punctuation char is mapped to its X keysym name (',' -> 'comma'); digits pass
    through (they are valid keysym names already)."""
    low = k.strip().lower()
    if low in _KEYSYM:
        return _KEYSYM[low]
    if len(k) == 1:
        if k.isalpha():
            return k.lower()
        return _PUNCT.get(k, k)
    return k  # assume it's already a valid keysym (e.g. 'F5', 'Menu')


def op_input(req: dict) -> dict:
    a = req["action"]
    if a == "type":
        # `xdotool type` synthesizes keysyms directly over XTEST — it handles
        # UTF-8 (CJK/em-dash) too on this single Unicode-capable path. The
        # previous xclip-paste fallback is REMOVED:
        # `xclip` forks a daemon that holds the X selection until a paste reads
        # it, so subprocess.run(capture_output=True) blocked on its stdout EOF
        # for the full 60s timeout — every long/non-ASCII type hung (the v7
        # rollout's writer task wasted ~all turns on it). timeout scales with
        # length (long paragraphs at --delay ms/char take real time).
        text = req["text"]
        delay = int(req.get("delay_ms", 12))
        # Budget the row at the delay it will actually run at: a segment holding a
        # non-keymap codepoint types at ``_NON_ASCII_DELAY_MS`` (see below).
        _row_delay = max(delay, _NON_ASCII_DELAY_MS) if not text.isascii() else delay
        budget = 30.0 + len(text) * (_row_delay / 1000.0) * 2
        # Uniform rule: a REAL control char means "press that key", a LITERAL
        # backslash-escape means "type those characters".
        #  * real '\n' (chr 10) -> press Return; real '\t' (chr 9) -> press Tab,
        #    emitted as DISCRETE keypresses, not left in the xdotool type stream.
        #    Required for '\n' (xdotool type maps it to keysym Linefeed/Ctrl-J,
        #    which GTK widgets silently DROP — a multi-line type would lose every
        #    break while still reporting ok=true); applied to '\t' for the same
        #    rule, so a Tab navigates/commits as a discrete keypress.
        #  * a LITERAL '\\n' / '\\t' (backslash + letter, two chars) is NOT a
        #    control char, so split() never matches it: it stays in the segment
        #    and is typed verbatim — preserving source intent like a shell
        #    `echo -e "a\nb"`. (Models that double-escape a newline they MEANT as
        #    Return are emitting the wrong bytes; that is the model adapter's job
        #    to un-escape, not the generic transport's.)
        # Accents/CJK/symbols stay on the working type path.
        for ri, row in enumerate(text.split("\n")):
            if ri:
                _xdo("key", "--clearmodifiers", "Return")
            for ci, cell in enumerate(row.split("\t")):
                if ci:
                    _xdo("key", "--clearmodifiers", "Tab")
                if cell:
                    # A codepoint outside the keymap forces xdotool to borrow a
                    # spare keycode and announce it with MappingNotify. At the
                    # 12 ms default the client has typically not processed that
                    # remap yet and DROPS the key -- silently, with ok=true.
                    # Measured in this image typing "A<em>B<en>C<ellipsis>D":
                    # delay=12 delivered "ABCD" (every non-ASCII gone, every
                    # ASCII kept); delay=30 delivered all 7 characters, as did
                    # 60 and 120. Pay the slower rate only for the cells that
                    # need it -- pure-ASCII typing keeps the 12 ms default.
                    cell_delay = max(delay, _NON_ASCII_DELAY_MS) if not cell.isascii() else delay
                    _xdo("type", "--clearmodifiers", "--delay", str(cell_delay),
                         "--", cell, timeout=budget)
    elif a == "key":
        # One key = a press; several = a chord (xdotool "ctrl+c" syntax).
        keys = [_norm_key(k) for k in req["keys"]]
        _xdo("key", "+".join(keys) if len(keys) > 1 else keys[0])
    elif a == "keydown":
        _xdo("keydown", _norm_key(req["key"]))
    elif a == "keyup":
        _xdo("keyup", _norm_key(req["key"]))
    elif a == "click":
        _move_if_xy(req)
        args = ["click"]
        if int(req.get("repeat", 1)) > 1:
            args += ["--repeat", str(int(req["repeat"]))]
        _xdo(*args, str(int(req.get("button", 1))))
    elif a == "move":
        _xdo("mousemove", str(int(req["x"])), str(int(req["y"])))
    elif a == "mouse_down":
        _move_if_xy(req)
        _xdo("mousedown", str(int(req.get("button", 1))))
    elif a == "mouse_up":
        _move_if_xy(req)
        _xdo("mouseup", str(int(req.get("button", 1))))
    elif a == "scroll":
        # xdotool wheel buttons: 4=up 5=down 6=left 7=right. BOTH axes are
        # supported — base.py drives horizontal scroll via interface.scroll(dx, 0)
        # (cb step "scroll" direction left/right), so a vertical-only handler
        # would silently mis-scroll those down. axis="x" => horizontal.
        axis = req.get("axis", "y")
        if axis == "x":
            button = "6" if req.get("direction", "right") == "left" else "7"
        else:
            button = "4" if req.get("direction", "down") == "up" else "5"
        for _ in range(max(1, int(req.get("amount", 1)))):
            _xdo("click", button)
    else:
        raise ValueError(f"unknown input action {a!r}")
    return {}


def op_read_bytes(req: dict) -> dict:
    with open(req["path"], "rb") as fh:
        return {"b64": base64.b64encode(fh.read()).decode("ascii")}


def op_write_bytes(req: dict) -> dict:
    # The server runs as the env's desktop user (EXEC_USER; user for
    # osworld/scalecua/cuagym, root for cuaworld), so the file is born owned by that
    # uid — no per-command uid routing, no post-write chown.
    data = base64.b64decode(req["b64"])
    path = req["path"]
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return {}


_OPS = {
    "ping": op_ping,
    "run_command": op_run_command,
    "screenshot": op_screenshot,
    "input": op_input,
    "read_bytes": op_read_bytes,
    "write_bytes": op_write_bytes,
}


def main() -> None:
    # OWN PROCESS GROUP, first thing: every child spawned below (op_run_command's
    # `bash -c`, op_input's xdotool) inherits this pgid, which is what makes the
    # host's ONE `kill -TERM -<pgid>` reach the whole tree instead of just this
    # process. `docker exec` does not put its process in a group of its own, so
    # without this the pgid names a SHARED group the host must refuse to signal
    # (see DockerExecSession._reap's leader check) and every command that
    # outlives its session leaks for the container's lifetime.
    # EPERM only if we are already a session leader — in which case we already
    # lead our own group and there is nothing to do.
    with contextlib.suppress(OSError):
        os.setpgrp()
    # PROTOCOL ISOLATION. This server speaks JSON-lines over stdout, but code it
    # calls also writes to fd 1: python-xlib's Display() prints
    # "Xlib.xauth: warning, no xauthority details available" to stdout, and shelled
    # children can too. That noise interleaves with responses, so the client
    # readline()s a warning line and json.loads() it -> "Expecting value: line 1
    # column 1 (char 0)". Dup the real stdout for our responses, then point fd 1 at
    # stderr: every stray stdout write (Python OR C level) lands on stderr (the
    # client DEVNULLs it), never in the protocol stream.
    resp_out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    # Line-buffered loop; any single-op failure is reported, never fatal to the
    # session (the pipe outlives bad requests).
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            result = _OPS[req["op"]](req)
            if isinstance(result, (bytes, bytearray)):
                # Binary-body op (screenshot): a tiny JSON header line stating the
                # body length, then the raw bytes. Flush the TEXT header before the
                # BINARY body (resp_out is text-mode, line-buffered) so the header
                # can't be buffered after the body.
                hdr = {"id": rid, "ok": True, "shot_n": len(result)}
                resp_out.write(json.dumps(hdr) + "\n")
                resp_out.flush()
                resp_out.buffer.write(result)
                resp_out.buffer.flush()
                continue
            resp = {"id": rid, "ok": True, "result": result}
        except Exception as e:  # noqa: BLE001 — protocol boundary
            # shot_n=0 so a *screenshot* client honoring the header reads no body on
            # an error frame (no desync); harmless for other ops (they raise on !ok).
            resp = {"id": rid, "ok": False, "shot_n": 0,
                    "error": f"{type(e).__name__}: {e}"}
        resp_out.write(json.dumps(resp) + "\n")
        resp_out.flush()


if __name__ == "__main__":
    main()
