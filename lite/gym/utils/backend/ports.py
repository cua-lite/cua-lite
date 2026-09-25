"""Automatic port allocation for parallel containerised environments.

Scans a configurable port range to find free ports for in-container RPC
servers (e.g. ``androidworld`` Computer-API on ``api_port`` or ``androidlab``'s
in-container HTTP API) so that multiple Docker-backed environments can run in
parallel without manual port management. The sandbox family (lite.osworld,
cua-lite/sandbox.linux) does NOT use this — it rides exec-stdio over the docker
daemon socket and publishes no host ports for headless rollouts.

Cross-process safety is achieved via a shared reservation file protected by
``fcntl.flock``.  The file maps each port to ``{"pid": int, "ts": float}``
(allocation wall-clock seconds, used as a TOCTOU grace before docker actually
binds). Stale entries (dead pid OR port no longer bound past the grace window)
are pruned automatically on each allocation.

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
which time other envs' ports may have been released and the recent-
allocation grace window has lapsed). Direct mode (no env-server) sees
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

# Grace window after allocate_ports during which _prune_dead will NOT drop a
# reservation even if its OS-level port still tests as free. Docker takes
# 1-3 s to actually bind the host port after ``docker run -p`` returns;
# without this guard, a parallel allocator can call _is_port_free, see the
# port still un-bound, drop the reservation, and pick the same port — causing
# the second docker run to fail with "Bind for 127.0.0.1:N failed: port is
# already allocated" (TOCTOU race observed under androidlab concurrency=16).
_RECENT_GRACE_S = 30.0

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

# In-process cache of recently-allocated ports → monotonic timestamp. Protects
# only this Python process's threads (cross-process race still possible, but
# cua-lite rollouts are single-process per run-id, so concurrency-16 within
# one scripts/rollout.py process is the dominant collision case).
_RECENT_ALLOCATIONS: dict[int, float] = {}

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
    shape with ``ts=0.0`` (no grace info → falls through to existing port-free
    check during prune).
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
    """Remove entries whose owning process no longer exists OR whose port is
    no longer bound.

    The PID-only check is insufficient when multiple Docker containers share
    a single RolloutManager PID: a container may crash (freeing its port)
    while the PID stays alive. Checking ``_is_port_free`` catches these
    stale reservations and prevents the port space from slowly exhausting
    under high concurrency.

    Two grace mechanisms protect against TOCTOU between allocate→docker-bind:
      1. cross-process: ``ts`` field in reservation file (wall-clock seconds).
         Any process pruning sees the freshly-allocated entry as recent and
         keeps it, even when its own ``_RECENT_ALLOCATIONS`` is empty.
      2. in-process: ``_RECENT_ALLOCATIONS`` (monotonic). Survives wall-clock
         jumps within the allocator's own lifetime.
    """
    live: dict[int, dict] = {}
    now_mono = time.monotonic()
    now_wall = time.time()
    for port, entry in reservations.items():
        pid = entry["pid"]
        ts_wall = entry.get("ts", 0.0)
        if not _pid_alive(pid):
            continue
        # Cross-process grace via reservation-file timestamp.
        if ts_wall > 0 and (now_wall - ts_wall) < _RECENT_GRACE_S:
            live[port] = entry
            continue
        # In-process grace (legacy path; redundant once ts is set, but kept
        # for safety against wall-clock jumps).
        if (alloc_t := _RECENT_ALLOCATIONS.get(port)) is not None:
            if now_mono - alloc_t < _RECENT_GRACE_S:
                live[port] = entry
                continue
            _RECENT_ALLOCATIONS.pop(port, None)
        if _is_port_free(port):
            # Port not bound anymore — container died, release reservation.
            continue
        live[port] = entry
    return live

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
                # which time the recent-allocation grace window
                # (``_RECENT_GRACE_S``) has lapsed and a freshly-released
                # emulator's port is reusable.
                raise CapacityExhausted(
                    what=(
                        f"port range [{range_start}, {range_end}) exhausted: "
                        f"found only {len(found)}/{n} free"
                    ),
                    retry_after_s=30.0,
                )

            _write_reservations(reservations)

            # Mark recently allocated so concurrent _prune_dead in this
            # process doesn't drop these before docker actually binds.
            _now = time.monotonic()
            for p in found:
                _RECENT_ALLOCATIONS[p] = _now

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
    """Re-stamp the reservation lease for *ports* to NOW.

    Closes the stale-reservation window: callers hold a reservation across a potentially
    slow pre-run step (a zombie ``docker rm -f`` on a wedged daemon can take
    30-60 s) BEFORE the ``docker run`` that actually binds the port; once
    ``_RECENT_GRACE_S`` lapses a parallel allocator may prune-and-reuse the
    port. Calling this immediately before ``docker run`` (the shared
    ``docker_run_detached`` does it for every DEDICATED env) re-bases the
    grace on the real bind gap. Only entries owned by THIS pid are
    re-stamped — never another process's reservation.
    """
    if not ports:
        return
    now = time.time()
    now_mono = time.monotonic()
    for p in ports:
        if p in _RECENT_ALLOCATIONS:
            _RECENT_ALLOCATIONS[p] = now_mono

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
    # Drop from in-process recent-allocation cache so future _prune_dead
    # calls treat these as available.
    for p in ports:
        _RECENT_ALLOCATIONS.pop(p, None)

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
