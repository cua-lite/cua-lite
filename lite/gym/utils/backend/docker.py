"""Shared docker helpers for env adapters.

Run: not directly. Imported by ``ensure_services`` hooks in env modules.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from lite.gym.errors import CapacityExhausted, EnvDepsMissingError
from lite.gym.utils.backend.ports import resolve_env_server_port, touch_ports

if TYPE_CHECKING:
    from lite.gym.utils.backend.freshness import ContainerImage

logger = logging.getLogger(__name__)


# ── Removal concurrency: DELIBERATELY UNBOUNDED ──────────────────────────────
#
# There is no removal semaphore here, and its absence is a decision rather than
# an oversight. A ``RM_CONCURRENCY = 3`` bound was added on 2026-08-08 and
# REVERTED the same day on the owner's directive that neither direct mode
# nor the env-server's throttling mode may differ from ``origin/main``, which has
# no bound on any removal path.
#
# The exposure is explicit: a caller that ``asyncio.gather``s N teardowns spawns
# N simultaneous ``docker rm`` CLIs (57 were once measured, oldest stuck 3 m
# 26 s). Bound the fan-out at the CALLER if a caller ever needs it; do not
# reintroduce a process-wide bound here, where it
# lands on ``env.close()`` — the one path every task traverses.


def redact_env_args(args: list[str], secret_keys: tuple[str, ...]) -> list[str]:
    """Mask secret ``-e KEY=VALUE`` values for logging."""
    return [
        f"{a.split('=', 1)[0]}=***"
        if "=" in a and a.split("=", 1)[0] in secret_keys
        else a
        for a in args
    ]


def require_image_present(image: ContainerImage) -> None:
    """Assert an env's image exists AND was built from its current sources;
    raise ``EnvDepsMissingError`` when it is missing/stale, or
    ``CapacityExhausted`` when Docker inspection itself is transiently failing.

    Called by an env's module-level ``ensure_services`` hook (see
    :mod:`lite.gym.registry` for the contract) so the env-server fails fast at
    startup probe time — both when the image was never built and when a
    Dockerfile/patch edit left it stale — not on the first ``gym.make`` minutes
    into rollout iter 0. Freshness lives on the :class:`ContainerImage` so the
    check can't be forgotten. See :mod:`lite.gym.utils.backend.freshness`.
    """
    image.ensure_runnable()


def _mem_argv(mem: str) -> list[str]:
    """The ONE argv for "cap this container's memory":
    ``--memory M --memory-swap M --memory-swappiness 0``.

    The three flags are a single decision, not three options: ``--memory-swap``
    equal to ``--memory`` means NO swap allowance, and ``--memory-swappiness 0``
    keeps the container off host swap entirely — together they make an
    over-cap container OOM-kill the offender instead of thrashing the host's
    disk. Setting ``--memory`` alone silently grants an equal amount of swap
    (docker's default), i.e. double the intended cap paid for in host I/O.
    Every capped run path builds its argv here so the trio can never partially
    diverge (cf. :func:`_rm_argv` for removal)."""
    return ["--memory", mem, "--memory-swap", mem, "--memory-swappiness", "0"]


def docker_run(
    name: str,
    image: ContainerImage,
    *,
    mem: str,
    port: tuple[int, int],
    env: dict[str, str] | None = None,
) -> None:
    """``docker run -d`` a memory-capped, ``--init``-reaped, single-port container.

    THE CONDITION for using it (not a caller list — every list of envs written
    into a shared util's docstring has gone stale): your env's backend is ONE
    long-lived container per server port that publishes ONE port, i.e. the
    shared-backend (``BackendFamily.SINGLETON``) shape. It gives that container
    a hard memory cap (:func:`_mem_argv`) and ``--init`` so PID 1 reaps zombie
    browser children. A DEDICATED env — one container per trajectory, often
    several ports — does not fit and builds its own argv.

    On ``docker run`` failure (image not built, port
    bind race, daemon down) raises :class:`EnvDepsMissingError` so the server
    fails fast and tells the operator how to build the image. Docker stderr is
    logged server-side, not sent back in the remote error payload.

    Asserts the image is present and fresh before launching (so a forgotten
    rebuild fails here, not via a confusing in-container error). The operator
    hints printed on a ``docker run`` failure come from the ``ContainerImage``.

    Args:
        name: ``--name`` (the per-server-port container name).
        image: The env's :class:`ContainerImage` (tag + build sources + hints).
        mem: Memory cap string (e.g. ``"8g"``) → :func:`_mem_argv`.
        port: ``(host_port, container_port)`` → ``-p host:container``.
        env: ``-e KEY=VALUE`` pairs forwarded into the container (the only
            config channel — the container has no host filesystem access).
            Insertion order is preserved (matches the hand-rolled ``-e`` order).
    """
    image.ensure_runnable()
    cmd = ["docker", "run", "-d", "--name", name, "--init", *_mem_argv(mem)]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    host_port, container_port = port
    cmd += ["-p", f"{host_port}:{container_port}", image.tag]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.warning(
            "docker run %s failed: %s",
            image.tag,
            (e.stderr or "").strip()[:500],
        )
        raise EnvDepsMissingError(
            what=f"docker run failed for image {image.tag}",
            install=image.install,
            see=image.see,
        ) from e


def docker_run_detached(
    *,
    name: str,
    image: str,
    ports: Sequence[tuple[int, int]] = (),
    env: Mapping[str, str] | None = None,
    volumes: Sequence[str] = (),
    devices: Sequence[str] = (),
    sysctls: Sequence[str] = (),
    cap_add: Sequence[str] = (),
    group_add: Sequence[int | str] = (),
    memory: str | None = None,
    privileged: bool = False,
    auto_remove: bool = False,
    command: Sequence[str] = (),
    redact: Sequence[str] = (),
    timeout: float = 180.0,
    label: str = "docker",
) -> None:
    """THE ``docker run -d`` for DEDICATED per-episode containers.

    Owns exactly the copied surface: argv assembly (canonical flag order),
    secret-redacted logging, the port-lease re-stamp
    (:func:`~lite.gym.utils.backend.ports.touch_ports` immediately before the run, so
    a slow pre-reap can't let a parallel allocator prune-and-reuse the
    reserved port), the bounded run, and its error mapping. **Readiness
    probes stay env-specific** — the env calls this, then runs its own
    boot-wait pipeline.

    Args:
        ports: ``(host_port, container_port)`` pairs → ``-p host:container``.
            Host ports are also the lease re-stamp targets.
        env: ``-e KEY=VALUE`` pairs, insertion order preserved.
        volumes: raw ``src:dst[:ro]`` bind specs → ``-v``.
        memory: cap string (e.g. ``"8g"``) → :func:`_mem_argv`; ``None`` = uncapped.
        redact: secret ``-e`` KEY names masked in the log line.
        command: argv appended after the image (e.g. ``("sleep", "86400")``).

    Raises:
        CapacityExhausted: (warming, retriable) — the run timed out; a
            wedged/overloaded daemon is a capacity condition, and every
            DEDICATED acquire loop retries ``(RuntimeError, CapacityExhausted)``.
        RuntimeError: the run returned non-zero (name conflict, bad flags…).
    """
    argv = ["docker", "run"]
    if auto_remove:
        argv.append("--rm")
    argv.append("-d")
    if privileged:
        argv.append("--privileged")
    for c in cap_add:
        argv += ["--cap-add", c]
    for d in devices:
        argv += ["--device", d]
    for s in sysctls:
        argv += ["--sysctl", s]
    if memory is not None:
        argv += _mem_argv(memory)
    for g in group_add:
        argv += ["--group-add", str(g)]
    for k, v in (env or {}).items():
        argv += ["-e", f"{k}={v}"]
    for vol in volumes:
        argv += ["-v", vol]
    for host_port, container_port in ports:
        argv += ["-p", f"{host_port}:{container_port}"]
    argv += ["--name", name, image, *command]

    logger.info("docker run %s: %s", label,
                " ".join(redact_env_args(argv[2:], tuple(redact))))
    touch_ports(*(host for host, _ in ports))
    try:
        r = subprocess.run(argv, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise CapacityExhausted.warming(
            what=f"docker run {name} timed out after {timeout:.0f}s "
                 "(daemon overloaded — too many concurrent spawns)",
        ) from e
    if r.returncode != 0:
        stderr = (r.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"docker run {name} failed (exit {r.returncode}): {stderr}")


def _rm_argv(name: str) -> list[str]:
    """The ONE argv for "remove a container": ``docker rm -f -v <name>``.

    ``-v`` is part of the framework's definition of removal, not an option:
    anonymous volumes (osworld's ``/storage``, mobileworld's nested
    ``/var/lib/docker``, …) are auto-cleaned only on a graceful ``--rm`` exit —
    a force-rm without ``-v`` orphans them forever, invisibly eating disk.
    ``-v`` never touches named volumes (e.g. VWA's ``classifieds_db``) or bind
    mounts, so it is safe to hardcode for every env.

    Every removal path — :func:`docker_rm_f`, :func:`docker_rm_f_async`, the
    drift reaper's quarantined rm (remote/reaper.py), and the env-side
    pre-reap/destroy sites — builds its argv here so the flag can never
    silently diverge again (androidlab once lost ``-v`` exactly this way).
    """
    return ["docker", "rm", "-f", "-v", name]


# ── What counts as a REMOVAL, and why the exit code does not ─────────────────
#
# Measured on this host (2026-08-08, load ~107, ``/srv`` 89% full, image
# ``cua-lite/lite.osworld:latest``). ``docker rm -f -v`` answers three ways:
#
#   removed  rc 0, stdout ``<name>``  — the echoed id IS the receipt
#   absent   rc 0, stdout EMPTY, stderr ``No such container`` (idempotent no-op)
#   refused  rc 1, stdout EMPTY, stderr ``cannot remove container …: could not
#            kill container: … did not receive an exit event``
#
# So the exit code cannot separate a removal from a no-op — both are 0 — and the
# echoed id is the only evidence the container went away.
#
# The third answer is the one that leaks, and it is the NORMAL outcome of a
# gathered teardown, not an edge case: 7 of 7 concurrent removals of running
# containers were refused at 12.0–12.4 s, the SIGKILL landed ~15 s later
# (``Exited (137)``), the remove step never ran, and all 7 were still present at
# +60 s. Re-issuing IMMEDIATELY removed 4/4 in 6.8–8.1 s, and 0.04–0.05 s once a
# container is already ``Exited``. A refusal is therefore a HALF-DONE removal
# whose expensive half (the force-stop) is already paid for — so every removal
# path here finishes the job instead of reporting a leak it could have prevented.
#
# The daemon's own kill-wait expiring at ~12 s is also why a sub-12 s client
# budget is the wrong shape: it guarantees the client gives up before the daemon
# can tell it what happened.


def _rm_removed(returncode: int, stdout: str) -> bool:
    """Did this ``docker rm -f -v`` actually remove a container? The echoed id is
    the only receipt — ``returncode == 0`` also covers "no such container" (see
    the note above)."""
    return returncode == 0 and bool(stdout.strip())


def _rm_aftermath(name: str) -> str:
    """What an operator should do about a container that was NOT removed.

    The drift reaper backstops leftovers ONLY under an env-server: it is a
    lifespan task of ``lite/gym/remote/server.py`` and both of its halves refuse
    an unscoped ``server_port`` (remote/reaper.py). Direct mode has no such
    sweep, so the message names the command instead of pointing at a reaper that
    will never run."""
    if resolve_env_server_port() is not None:
        return "the drift reaper will backstop it"
    return f"direct mode has no drift reaper — remove by hand: `docker rm -f -v {name}`"


def docker_rm_f(name: str, *, timeout: float, label: str = "docker") -> int:
    """``docker rm -f -v <name>`` (idempotent). Returns 1 if a container was
    removed, 0 if there was nothing to remove. ``-v`` cleans the container's
    anonymous volumes (see :func:`_rm_argv` for why it is hardcoded).

    ``timeout`` bounds the WHOLE removal, not one CLI invocation, so a wedged
    docker daemon still can't block the server thread on ensure/shutdown/reap: a
    *refused* removal is re-issued inside whatever budget is left, because the
    refusal already paid for the force-stop (see the note above). ``label``
    prefixes the warnings (e.g. ``"webgym"`` / ``"mobilegym"``).

    Unpaced: see the "Removal concurrency" note at the top of this module for why
    there is no semaphore here."""
    deadline = time.monotonic() + timeout
    stderr = ""
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            r = subprocess.run(_rm_argv(name), capture_output=True,
                               text=True, timeout=remaining)
        except subprocess.TimeoutExpired:
            logger.warning(
                "%s: `docker rm -f -v %s` ABANDONED after %.0fs — the CLI was killed "
                "at the budget, so whether the container was removed is unknown; %s",
                label, name, timeout, _rm_aftermath(name))
            return 0
        if _rm_removed(r.returncode, r.stdout):
            return 1
        if r.returncode == 0:
            return 0                       # nothing to remove: the idempotent case
        stderr = (r.stderr or "").strip()   # refused — re-issue while budget remains
    logger.warning(
        "%s: `docker rm -f -v %s` REFUSED by the daemon within %.0fs — the container "
        "is still there (a force-stop whose exit event was lost leaves it running, "
        "then `Exited`, but never removed); last error: %s; %s",
        label, name, timeout, stderr or "<none>", _rm_aftermath(name))
    return 0


async def docker_rm_f_async(name: str, *, timeout: float, label: str = "docker") -> None:
    """Async sibling of :func:`docker_rm_f` for event-loop callers (sandbox
    teardown paths). Same re-issue of a refused removal, and ``timeout`` is
    likewise the budget for the whole removal. It returns nothing, so it needs
    only the *refusal* half of the receipt (a non-zero rc); the id echo that
    separates "removed" from "nothing to remove" is what :func:`_rm_removed`
    exists for, and only the counting sibling needs it. Best-effort by contract:
    swallows *every* failure — because its callers (close / failed-boot release)
    must never let a removal hiccup escape.

    Unpaced, one CLI per container: see the "Removal concurrency" note at the top
    of this module. A caller that gathers N teardowns gets N simultaneous
    ``docker rm`` processes — which is exactly the case the daemon refuses, hence
    the re-issue.

    Each way of NOT removing the container says which one happened, because they
    call for different operator actions: ``NEVER ISSUED`` (no CLI ran, the
    container is untouched), ``ABANDONED`` (the CLI outlived the budget and is
    left running, so the verdict is unknown) and ``REFUSED`` (the daemon declined
    and the container is still there)."""
    deadline = time.monotonic() + timeout
    stderr = ""
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *_rm_argv(name),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as e:
                logger.warning(
                    "%s: async `docker rm -f -v %s` was NEVER ISSUED (%s: %s) — "
                    "nothing was killed and the container is untouched; %s",
                    label, name, type(e).__name__, e, _rm_aftermath(name))
                return
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=remaining)
            except TimeoutError:
                # The CLI child is left running, NOT killed: killing it was Q2's
                # second half and was reverted with the first, so this path matches
                # ``origin/main`` (test_docker_rm_f_async_swallows_a_timeout...).
                # Nothing is re-issued either — the budget is spent, and a second
                # CLI here would race the one still holding the daemon's lock.
                logger.warning(
                    "%s: async `docker rm -f -v %s` ABANDONED after %.0fs — the CLI "
                    "outlived its budget and was left running, so whether the "
                    "container was removed is unknown; %s",
                    label, name, timeout, _rm_aftermath(name))
                return
            # ``communicate()`` reaped the child, so ``returncode`` is an int.
            rc: int = proc.returncode            # type: ignore[assignment]
            if rc == 0:
                # rc 0 is BOTH "removed" (id echoed) and "nothing to remove"
                # (no such container) — either way there is nothing left to do.
                return
            stderr = err.decode("utf-8", "replace").strip()
        logger.warning(
            "%s: async `docker rm -f -v %s` REFUSED by the daemon within %.0fs — the "
            "container is still there (a force-stop whose exit event was lost leaves "
            "it running, then `Exited`, but never removed); last error: %s; %s",
            label, name, timeout, stderr or "<none>", _rm_aftermath(name))
    except Exception as e:
        logger.warning(
            "%s: async `docker rm -f -v %s` FAILED unexpectedly (%s: %s); %s",
            label, name, type(e).__name__, e, _rm_aftermath(name))
