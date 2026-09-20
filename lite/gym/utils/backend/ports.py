"""Automatic port allocation for parallel containerised environments.

Scans a configurable port range to find free ports for in-container RPC
servers (e.g. ``androidworld`` Computer-API on ``api_port`` or ``androidlab``'s
in-container HTTP API) so that multiple Docker-backed environments can run in
parallel without manual port management. The sandbox family (lite.osworld,
cua-lite/sandbox.linux) does NOT use this — it rides exec-stdio over the docker
daemon socket and publishes no host ports for headless rollouts.

Cross-process safety is achieved via a shared reservation file protected by
``fcntl.flock``.  The file maps each port to ``{"pid": int, "ts": float}``
(allocation wall-clock seconds). Reservations survive until explicit release
or owner exit, including while Docker has not yet bound the port. Dead owners'
entries are pruned automatically on each allocation.

Band ownership — the mechanism, deliberately NOT a table. This module owns two
things: the scan-and-reserve mechanism, and the DEFAULT band declared below. It
owns no env's band. Every other caller passes its own ``range_start`` /
``range_end``, declared beside its own allocation site (an ``_API_PORT_RANGE`` /
``_PORT_RANGE_START`` module constant, or a ``configs/default.yaml`` key), so a
band and the code that allocates from it cannot drift apart. To enumerate the
live bands, read the call sites:

    git grep -n 'allocate_ports(' lite scripts

Do not re-add a per-env map here: an enumeration in a shared util is a copy no
env owner updates, and not every band on this host is even reserved through the
file below — sglang's and Ray's are not.

Bands must not overlap, since one host runs many of them concurrently — but
that is a property of the call sites and is NOT enforced here: ``allocate_ports``
hands out any free port in the range it is given, including one inside another
env's band. Nor is the upper bound one convention: ``range_end`` is EXCLUSIVE
here (``range(range_start, range_end)``), while some callers pass an inclusive
constant plus one (``…_END + 1``) and browsergym's miniwob picker scans its own
band inclusively with a separate loop. Read the call site, not a summary.

Usage (called automatically by ``SandboxBaseEnv.reset``):
    from lite.gym.utils.backend.ports import allocate_ports
    ports = allocate_ports()  # [20000, 20001]

Exhaustion: when no free port (or contiguous-port block) is available
in the configured range, the allocators raise
:class:`~lite.gym.errors.CapacityExhausted`. The env-server's exception
handler maps that to HTTP 503 + Retry-After so the client retries (by
which time other envs' ports may have been released). Direct mode (no env-server) sees
the raw exception and should treat it the same way.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import random
import socket
import threading
import time

from lite.gym.errors import CapacityExhausted
from lite.utils.path import project_root

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────
# 20000-20999 sits cleanly above Ray workers (11000-19999) and well clear
# of android emulator's grpc projection (=console+3000, up to 10554 for a
# 1000-slot console pool). ``range_end`` is exclusive, so the last allocatable
# port is 20998: 999 ports / 2 ports per env = 499 concurrent osworld VMs —
# generously more than any practical single-host load.
DEFAULT_PORT_RANGE_START = 20000
DEFAULT_PORT_RANGE_END = 20999

# ── Module-level state ───────────────────────────────────────────────────────
_lock = threading.Lock()

# Reservation file lives under <repo>/.tmp/ (bind-mounted into every slime
# container at /workspaces/cua-lite/.tmp/) so all callers — host-direct rollout
# and any number of slime containers running env recipes concurrently — share
# one host inode. Per-container /tmp would let two containers each pick the
# same port and collide at host-level docker port bind.
# Repo-root ``.tmp`` (NOT parents[4], which was the parent-OF-repo — an off-by-one:
# in a slime container the repo mounts at /workspaces/cua-lite, so parents[4] gave
# the unmounted /workspaces/.tmp and silently broke the host-direct↔container share
# this file's docstring promises). project_root() is the marker-found repo root, so
# the host's .tmp and every container's /workspaces/cua-lite/.tmp are the same inode.
_SHARED_TMP = project_root() / ".tmp"
_RESERVATION_FILE = _SHARED_TMP / "sandbox-port-reservations.json"
_LOCK_FILE = _SHARED_TMP / "sandbox-port-alloc.lock"

def _is_port_free(port: int) -> bool:
    """Return *True* if *port* is not bound on any local address.

    Checks IPv4 (0.0.0.0, 127.0.0.1) and IPv6 (::1) WITHOUT SO_REUSEADDR,
    so that ports held by Docker containers (which bind to both 127.0.0.1
    and [::1]) are correctly detected as in-use.
    """
    for family, host in [
        (socket.AF_INET, "0.0.0.0"),
        (socket.AF_INET, "127.0.0.1"),
        (socket.AF_INET6, "::1"),
    ]:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.bind((host, port))
        except OSError:
            return False
    return True

def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Process exists but we lack permission to signal it

def _read_reservations() -> dict[int, dict]:
    """Read the reservation file. Returns ``{port: {"pid": int, "ts": float}}``.

    Backward-compat: a legacy ``{port: pid_int}`` file is normalized to the new
    shape with ``ts=0.0`` (unknown allocation time).
    """
    try:
        data = _RESERVATION_FILE.read_text()
        raw = json.loads(data)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}
    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            port = int(k)
        except ValueError:
            continue
        if isinstance(v, int):
            out[port] = {"pid": v, "ts": 0.0}
        elif isinstance(v, dict) and "pid" in v:
            out[port] = {"pid": int(v["pid"]), "ts": float(v.get("ts", 0.0))}
    return out

def _write_reservations(reservations: dict[int, dict]) -> None:
    """Write the reservation file atomically. Uses ``{port: {"pid", "ts"}}`` shape."""
    _RESERVATION_FILE.write_text(json.dumps(reservations))

def _prune_dead(reservations: dict[int, dict]) -> dict[int, dict]:
    """Remove reservations only after their owning process exits.

    An unbound port can belong to a queued or booting container. Its live
    owner must release it explicitly when that container is destroyed.
    """
    return {port: entry for port, entry in reservations.items() if _pid_alive(entry["pid"])}

def allocate_ports(
    *,
    n: int = 2,
    range_start: int = DEFAULT_PORT_RANGE_START,
    range_end: int = DEFAULT_PORT_RANGE_END,
) -> list[int]:
    """Find *n* free, non-overlapping TCP ports in ``[range_start, range_end)``.

    Uses a file lock for cross-process safety + a threading lock for
    in-process safety.  Ports are reserved in a shared file so other
    processes know not to use them, even before Docker binds them.
    """
    found: list[int] = []
    my_pid = os.getpid()

    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.touch(exist_ok=True)
    lock_fd = open(_LOCK_FILE)

    try:
        # Cross-process lock
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        with _lock:
            reservations = _read_reservations()
            reservations = _prune_dead(reservations)
            reserved_ports = set(reservations.keys())

            # Randomize scan order to reduce collisions when many workers
            # allocate simultaneously (sequential scan causes thundering-herd
            # on the lowest free ports).
            candidates = list(range(range_start, range_end))
            random.shuffle(candidates)
            now_wall = time.time()
            for port in candidates:
                if port in reserved_ports:
                    continue
                if not _is_port_free(port):
                    continue
                found.append(port)
                reservations[port] = {"pid": my_pid, "ts": now_wall}
                if len(found) == n:
                    break

            if len(found) < n:
                # Roll back partial allocation
                for p in found:
                    reservations.pop(p, None)
                _write_reservations(reservations)
                # Translate to CapacityExhausted so the env-server's
                # exception handler maps to 503 + Retry-After uniformly
                # with other bounded-pool envs. The client retries by
                # which time another environment may have released its ports.
                raise CapacityExhausted(
                    what=(
                        f"port range [{range_start}, {range_end}) exhausted: "
                        f"found only {len(found)}/{n} free"
                    ),
                    retry_after_s=30.0,
                )

            _write_reservations(reservations)

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    logger.info("Allocated ports: %s (pid=%d)", found, my_pid)
    return found

def resolve_env_server_port() -> int | None:
    """Read the host env-server's listen port from the ambient ``CUA_LITE_ENV_SERVER_PORT``
    env var (set by ``lite.gym.remote.server.create_app``).

    Returns ``None`` in direct mode (var unset/empty → no env-server in the
    loop) or when the value isn't a valid int. Container-name scoping (per
    env-server) is built on top of this — see :func:`env_server_scope`."""
    p = os.environ.get("CUA_LITE_ENV_SERVER_PORT")
    try:
        return int(p) if p else None
    except ValueError:
        return None


def env_server_scope(default: str) -> str:
    """Stringified env-server port for container-name scoping, falling back to
    *default* in direct mode (port is ``None``).

    Callers pick *default* to match their cleanup convention:
      * webgym/mobilegym use ``f"d{os.getpid()}"`` (unique per direct process).
      * browsergym uses ``"default"`` (matches its start.sh / cleanup.sh
        ``${CUA_LITE_ENV_SERVER_PORT:-default}`` shell convention)."""
    p = resolve_env_server_port()
    return str(p) if p is not None else default


def touch_ports(*ports: int) -> None:
    """Update timestamps of this process's reservations before Docker starts.

    Timestamps do not expire reservations; ownership lasts until release or
    process exit. Another process's entries are never changed.
    """
    if not ports:
        return
    now = time.time()

    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.touch(exist_ok=True)
    lock_fd = open(_LOCK_FILE)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        with _lock:
            reservations = _read_reservations()
            my_pid = os.getpid()
            changed = False
            for p in ports:
                entry = reservations.get(p)
                if entry is not None and entry.get("pid") == my_pid:
                    entry["ts"] = now
                    changed = True
            if changed:
                _write_reservations(reservations)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def release_ports(*ports: int) -> None:
    """Mark previously allocated ports as available again.

    Removes entries from the shared reservation file so other processes
    can claim them.
    """
    if not ports:
        return
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.touch(exist_ok=True)
    lock_fd = open(_LOCK_FILE)

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        with _lock:
            reservations = _read_reservations()
            for p in ports:
                reservations.pop(p, None)
            _write_reservations(reservations)

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    logger.debug("Released ports: %s", list(ports))
