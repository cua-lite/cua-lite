"""Bridge DISPLAY :0 → the container's real Xvnc :1.

CUA-Gym's official scripts hardcode ``DISPLAY=:0`` (their VMs ran the desktop on
:0), but the lite.osworld image's Xvnc — the display the exec-stdio server
screenshots and sends input to — is ``:1``. A plain ``X0 -> X1`` symlink does NOT
work: X clients probe the *abstract* socket ``@/tmp/.X11-unix/X0`` first, which a
symlink can't satisfy (and the exec-stdio server runs commands as the non-root
``user``, for whom the symlink fallback also fails). So we socat-forward BOTH the
abstract and filesystem :0 sockets to :1. Idempotent + fast (gnome-ready ~3s).

Run: not directly — imported by ``src/browser/scripts.py`` and
``src/desktop/scripts.py`` setup_fn.
"""

from __future__ import annotations

# One shell snippet: if :0 isn't already reachable, stand up the socat forwarders
# and wait (briefly) for :0 to come up. Runs as the server's user.
_BRIDGE = (
    "DISPLAY=:0 xdpyinfo >/dev/null 2>&1 || { "
    "rm -f /tmp/.X11-unix/X0; "
    "setsid socat ABSTRACT-LISTEN:/tmp/.X11-unix/X0,fork,reuseaddr "
    "ABSTRACT-CONNECT:/tmp/.X11-unix/X1 >/dev/null 2>&1 </dev/null & "
    "setsid socat UNIX-LISTEN:/tmp/.X11-unix/X0,fork,mode=0777 "
    "ABSTRACT-CONNECT:/tmp/.X11-unix/X1 >/dev/null 2>&1 </dev/null & "
    "for _ in $(seq 1 20); do DISPLAY=:0 xdpyinfo >/dev/null 2>&1 && break; sleep 0.5; done; }; "
    "DISPLAY=:0 xdpyinfo >/dev/null 2>&1"
)
_WAIT_DESKTOP = (
    "for _ in $(seq 1 150); do "
    "[ -f /tmp/gnome-failed ] && exit 1; "
    "[ -f /tmp/gnome-ready ] && exit 0; "
    "sleep 1; "
    "done; exit 1"
)
# Setup's first act is a baseline window snapshot, so a reachable display is not
# enough: mutter must already have published _NET_CLIENT_LIST. `/tmp/gnome-ready`
# fires when gnome-shell starts, which on a loaded host can precede that, and
# `wmctrl -l` exits non-zero in the gap -- which window_ids cannot tell apart
# from a broken X connection, so it failed the task instead of waiting a beat.
# Before this wait existed, `wmctrl -l` failed on the order of a hundred times
# per desktop collection run (188, 246 and 404 occurrences in three of them).
_WAIT_WINDOW_LIST = (
    "for _ in $(seq 1 40); do "
    "DISPLAY=:0 wmctrl -l >/dev/null 2>&1 && exit 0; "
    "sleep 0.5; "
    "done; DISPLAY=:0 wmctrl -l >/dev/null"
)


async def bridge_display(computer) -> None:
    """Make DISPLAY=:0 reach the real Xvnc :1, WM ready to list windows."""
    desktop = await computer.interface.run_command(_WAIT_DESKTOP, timeout=180)
    if getattr(desktop, "returncode", 0) != 0:
        detail = (getattr(desktop, "stderr", "") or "").strip()
        raise RuntimeError(
            f"lite.cuagym GNOME desktop did not become ready: {detail[-300:]}"
        )
    result = await computer.interface.run_command(_BRIDGE, timeout=30)
    if getattr(result, "returncode", 0) != 0:
        detail = (getattr(result, "stderr", "") or "").strip()
        raise RuntimeError(f"failed to bridge DISPLAY=:0 to :1: {detail[-300:]}")
    windows = await computer.interface.run_command(_WAIT_WINDOW_LIST, timeout=40)
    if getattr(windows, "returncode", 0) != 0:
        detail = (getattr(windows, "stderr", "") or "").strip()
        raise RuntimeError(
            f"lite.cuagym window manager published no client list: {detail[-300:]}"
        )
