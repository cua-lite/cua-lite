"""Composed container reconcile-view + its docker-ps/rm primitives.

The ``live_ids`` / ``reap`` a container-backed env-server backend exposes, as a
standalone object container ``EnvServices`` *compose* (hold-and-delegate), NOT a
mixin they inherit — composition has no MRO and avoids the
``class X(EnvServices, …)`` shadow footgun. ``lite.osworld`` / ``lite.demo`` /
``androidworld`` / ``androidlab`` all share this one body.

  * :meth:`ContainerReaper.live_ids` — the RUNNING container names this run owns;
    the framework computes ghosts (``tracked − live``) from it. **Fails closed**:
    raises :class:`~lite.gym.errors.ReconcileProbeError` (the reconcile loop catches
    it and skips this cycle) on any ``docker ps`` failure — never returns ``None`` —
    so a docker hiccup never mass-false-ghosts live instances (a latent bug in the old
    code, which treated a failed scan as "nothing running" → ghosted everything).
  * :meth:`ContainerReaper.reap` — ``docker rm -f`` of orphan containers (live,
    NOT in ``in_use``, older than the orphan age guard), capped + quarantine-backed.

**Scoping is by ``server_port`` alone.** The server's listen port already
isolates co-resident servers; within one server, every container is that
server's to reap regardless of which client (token_hash) created it — the
framework's exact-match on ``in_use`` keeps any live instance's container safe.
A port-scoped ``docker ps`` catches strict and passthrough containers uniformly
without relying on token-hash sentinels.

The low-level docker-ps / docker-rm primitives (``_docker_ps_*``, ``_try_orphan_rm``,
``_quarantine``, ``_prune_quarantine``) + their tuning constants live at module scope below — they
were a separate ``docker_drift`` module, folded in here as its only consumer.

WHY THIS LIVES UNDER ``remote/`` even though env modules import it directly.
The public imports are :class:`ContainerServices` and
:class:`SingletonContainerServices` — ``EnvServices`` subclasses, i.e. the
env-server integration surface. The docker primitives are module-private, and
``tests/gym/remote/test_reaper.py::test_env_facing_surface_is_the_two_services_classes``
fails if a primitive or whole-module alias escapes. Every public method is typed
on :class:`lite.gym.remote.scope.ServerScope`, and the pair ``live_ids`` /
``reap`` is :class:`lite.gym.remote.reconcile.ReconcileView`. Moving this under
``utils/backend`` would point that package at ``lite.gym.remote`` and separate
a protocol from its implementation.

Not to be confused with :mod:`lite.gym.utils.backend.reaper` — same word, two
concepts. That one is a parametrized ``atexit`` sweep of a process-local
registry on interpreter exit; this one is the env-server's steady-state,
``server_port``-scoped reconcile of the docker host.

Run: not directly. Composed by container-backed env ``EnvServices`` objects;
the docker subprocess paths are exercised by the live/stress suites.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from lite.gym.services import EnvServices
from lite.gym.utils.backend.docker import _rm_argv, docker_rm_f
from lite.gym.utils.config.naming import container_name_prefix, match_pattern

if TYPE_CHECKING:
    from lite.gym.remote.scope import ServerScope

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tuning constants (docker / orphan-reaping side; the ghost-side knobs —
# safety_margin + MAX_GHOSTS_PER_CYCLE — live in the framework loop
# ``lite.gym.remote.reconcile``, NOT here: ghost detection is the server's job,
# not the container layer's).
#
# All knobs read env vars at import time. They stay env-var-only: useful for
# deployment tuning, but too low-level for ``scripts/serve_env.py --help``.
# --------------------------------------------------------------------------- #

# 600 s is the empirically validated grace. Earlier 120 s let the drift-reaper
# race transient ``_current_container = None`` windows during recycle /
# snapshot-fail / close (now fixed at source — the Layer-B ``destroy_backend``
# contract nulls the tracked container only AFTER destroy returns) and reap LIVE
# containers whose task was still running → /step Connection-refused → task FAIL.
# The destroy-then-null fix closes the race, but 600 s is kept as
# belt-and-suspenders for any teardown path that ever slips through.
DEFAULT_MIN_ORPHAN_AGE_S = float(
    os.environ.get("CUA_LITE_DRIFT_MIN_ORPHAN_AGE_S", "600.0")
)
DEFAULT_RM_TIMEOUT_S = float(os.environ.get("CUA_LITE_DRIFT_RM_TIMEOUT_S", "60.0"))
DEFAULT_PS_TIMEOUT_S = float(os.environ.get("CUA_LITE_DRIFT_PS_TIMEOUT_S", "15.0"))

# Bound per-cycle WORK — not throughput. The dispatcher awaits each cycle before
# the next, so this cap IS a cycle's worst case: that many rms, each bounded by
# ``DEFAULT_RM_TIMEOUT_S``, ``RM_INTER_DELAY_S`` apart. There is no orphan arrival
# rate to size it against: a healthy run produces none (``destroy_backend`` rms
# its own container in-band, via ``docker_rm_f``), an orphan is what a FAILED
# teardown leaves behind, and any excess waits for the next cycle. The cap is per
# env_id per cycle (``recovery.reconcile_all`` loops).
#
# 10 → 3 (2026-05-26): /step ``ConnectionRefused`` spikes on UNRELATED active
# containers coincided with bulk-rm cycles — 10 simultaneous ``docker rm -f``
# briefly lock the daemon (kernel-side iptables / netns teardown serialises
# through it) and stall ``docker exec`` / port-forwarded RPC past env-server's
# urllib timeout. SIMULTANEITY is not this number's job and never was: the loop in
# ``reap`` runs one rm at a time regardless of the cap. Nor is it capped anywhere
# else; env-side ``close()`` remains outside this drift-cycle cap by design, so
# this value only limits orphan cleanup done by the reaper.
MAX_ORPHANS_PER_CYCLE = int(os.environ.get("CUA_LITE_DRIFT_MAX_ORPHANS_PER_CYCLE", "3"))

#: Sleep between successive ``docker rm -f`` calls inside one drift cycle. Gives
#: the docker daemon time to drain its event queue and re-enter a responsive
#: state before the next rm. See :data:`MAX_ORPHANS_PER_CYCLE` rationale.
RM_INTER_DELAY_S = float(os.environ.get("CUA_LITE_DRIFT_RM_INTER_DELAY_S", "1.0"))

# Per-name exponential backoff for `docker rm -f` timeouts; prevents retrying the
# same stuck container every cycle forever. Self-prunes when the name disappears
# from `docker ps` (clean external exit) or after the longest backoff tier
# elapses without retry. Process-local (reset on server restart — that's when
# `recover_all` boot recovery re-scans). Stored as ``(next_retry_epoch, tier)``
# so the backoff actually escalates across timeouts (1m → 5m → 30m).
_RM_QUARANTINE: dict[str, tuple[float, int]] = {}
_RM_QUARANTINE_BACKOFF_S = (60.0, 300.0, 1800.0)   # 1m → 5m → 30m
_RM_QUARANTINE_MAX_AGE_S = 2 * _RM_QUARANTINE_BACKOFF_S[-1]   # 1h


def _name_regex(prefix: str, env_id: str) -> str:
    """docker-ps ``--filter name=`` regex for in-scope containers of one env_id.
    Single source for both ``_docker_ps_*`` scans; the grammar itself lives
    with the name producer (``naming.match_pattern``) so the two sides
    can't diverge."""
    return match_pattern(prefix, env_id)


@dataclasses.dataclass(frozen=True)
class _DockerPsScan:
    all: tuple[tuple[str, float], ...]    # (name, created_at) — running + exited


def _docker_ps_with_time(
    prefix: str, *, env_id: str, timeout: float,
) -> _DockerPsScan:
    """Returns the docker ps scan for containers in scope: ``all`` =
    ``(name, created_at)`` for every in-scope container including Exited
    (orphan classification reaps those too). Running-vs-not is NOT needed
    here — ghost classification uses the separate fail-closed
    :func:`_docker_ps_running_or_none`.

    Scopes by ``name=^{prefix}.*-{env_id}-``. Returns empty on any subprocess
    failure so the calling cycle classifies nothing and next cycle retries.

    Parses docker's human-style ``.CreatedAt`` (``"YYYY-MM-DD HH:MM:SS +ZZZZ
    UTC"`` — the trailing literal ``UTC`` is redundant text that breaks strptime,
    so we strip it first).
    """
    name_regex = _name_regex(prefix, env_id)
    try:
        r = subprocess.run(
            [
                "docker", "ps", "-a",
                "--filter", f"name={name_regex}",
                "--format", "{{.Names}}\t{{.CreatedAt}}",
            ],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
    except Exception as e:
        # Broad catch: covers TimeoutExpired, CalledProcessError, AND
        # FileNotFoundError (docker CLI not installed in this env). All
        # paths return the empty scan so this cycle does nothing.
        logger.warning(
            "container reap(%s): docker ps for prefix=%r failed: %s",
            env_id, prefix, e,
        )
        return _DockerPsScan(all=())
    all_entries: list[tuple[str, float]] = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, created_str = parts[0], parts[1]
        created_str = created_str.rsplit(" ", 1)[0]   # drop trailing " UTC"
        try:
            dt = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S %z")
        except ValueError:
            # Unparseable timestamp → skip (don't misclassify). Logged
            # once per occurrence; next cycle retries.
            logger.warning(
                "container reap(%s): unparseable docker CreatedAt for %s: %r",
                env_id, name, created_str,
            )
            continue
        all_entries.append((name, dt.timestamp()))
    return _DockerPsScan(all=tuple(all_entries))


def _docker_ps_running_or_none(
    prefix: str, *, env_id: str, timeout: float,
) -> set[str] | None:
    """Fail-closed running-set probe for ``ContainerReaper.live_ids``.

    Unlike :func:`_docker_ps_with_time` (which swallows a subprocess failure as
    an empty scan — indistinguishable from a genuinely-idle daemon), this runs
    its own ``docker ps`` and returns:

      * ``set[str]`` of RUNNING container names (``Status`` starts ``"Up "``) on
        success — possibly empty (daemon ok, nothing running);
      * ``None`` on ANY subprocess failure (timeout / CLI missing / non-zero) —
        the fail-closed signal so the framework skips ghost detection this cycle
        instead of treating "docker is down" as "everything is gone".
    """
    name_regex = _name_regex(prefix, env_id)
    try:
        r = subprocess.run(
            [
                "docker", "ps",
                "--filter", f"name={name_regex}",
                "--filter", "status=running",
                "--format", "{{.Names}}",
            ],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
    except Exception as e:
        logger.warning(
            "live_ids(%s): docker ps for prefix=%r failed: %s — failing closed",
            env_id, prefix, e,
        )
        return None
    return {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}


def _quarantine(name: str, now: float, prev_tier: int, why: str) -> bool:
    """Back this name off to the next tier and report it as NOT reaped.

    Both docker failures land here — the rm timed out, or the daemon refused it —
    because they carry the same fact: the container is still there and retrying it
    at full cadence only spends a reap slot the next orphan could have used.
    """
    tier = min(prev_tier + 1, len(_RM_QUARANTINE_BACKOFF_S) - 1)
    delay = _RM_QUARANTINE_BACKOFF_S[tier]
    _RM_QUARANTINE[name] = (now + delay, tier)
    logger.warning(
        "container reap: docker rm -f %s: %s; quarantined for %.0fs (tier %d)",
        name, why, delay, tier,
    )
    return False


def _try_orphan_rm(name: str, rm_timeout: float) -> bool:
    """``docker rm -f`` with per-name exponential backoff on failure.

    Returns ``True`` only when docker says the name is gone: ``rc == 0`` covers
    both "removed" and "no such container", which is what makes the call
    race-safe. Returns ``False`` while quarantined, and quarantines the name when
    the rm times out or the daemon refuses it — a refused rm leaves the container
    running, so counting it as reaped would overstate every ``orphans=N`` line.
    """
    now = time.time()
    next_retry, prev_tier = _RM_QUARANTINE.get(name, (0.0, -1))
    if next_retry > now:
        return False                       # under quarantine
    try:
        # NOT docker_rm_f: that helper swallows TimeoutExpired, which would
        # collapse the 3-state handling here (failed → escalate quarantine
        # vs already-gone → clear quarantine). Only the argv is shared.
        r = subprocess.run(
            _rm_argv(name),
            check=False, capture_output=True, text=True, timeout=rm_timeout,
        )
    except subprocess.TimeoutExpired:
        return _quarantine(name, now, prev_tier, f"timed out after {rm_timeout:.0f}s")
    if r.returncode != 0:
        stderr = (r.stderr or "").strip() or "<none>"
        return _quarantine(name, now, prev_tier, f"REFUSED by the daemon: {stderr}")
    # Gone (removed, or never there) → clear any stale quarantine.
    _RM_QUARANTINE.pop(name, None)
    return True


def _prune_quarantine(*, live_names: set[str], now: float) -> None:
    """Drop quarantine entries for names that vanished from docker (clean exit)
    OR exceeded the longest backoff tier without retry.

    Worst-case dict size = ``MAX_ORPHANS_PER_CYCLE`` × the cycles that fit in
    ``_RM_QUARANTINE_MAX_AGE_S`` (1 h) — tens of entries at any cadence the
    dispatcher runs, i.e. negligible.
    """
    for name, (next_retry, _tier) in list(_RM_QUARANTINE.items()):
        if name not in live_names or now - next_retry > _RM_QUARANTINE_MAX_AGE_S:
            _RM_QUARANTINE.pop(name, None)


class ContainerReaper:
    """Reconcile view for a docker-backed env. Composed by the env's
    ``EnvServices``; see module docstring."""

    def __init__(
        self,
        *,
        min_orphan_age_s: float = DEFAULT_MIN_ORPHAN_AGE_S,
        rm_timeout: float = DEFAULT_RM_TIMEOUT_S,
        ps_timeout: float = DEFAULT_PS_TIMEOUT_S,
    ) -> None:
        self._min_orphan_age_s = min_orphan_age_s
        self._rm_timeout = rm_timeout
        self._ps_timeout = ps_timeout

    # -- the reconcile view ------------------------------------------------- #

    def live_ids(self, env_id: str, scope: ServerScope) -> set[str] | None:
        """RUNNING container names this run owns (``Status`` starts ``Up ``).
        **Raises** :class:`~lite.gym.errors.ReconcileProbeError` on ``docker ps``
        failure (fail closed — never a partial set; the reconcile loop catches it and
        skips the cycle). Never returns ``None``: that value means only "no
        per-instance world", which a container env never is."""
        if scope.server_port is None:
            from lite.gym.errors import ReconcileProbeError
            raise ReconcileProbeError(
                f"live_ids({env_id}): UNSCOPED scope (server_port=None) would "
                "probe containers host-wide — skipping cycle"
            )
        prefix = container_name_prefix(server_port=scope.server_port, token_hash=None)
        ids = _docker_ps_running_or_none(
            prefix, env_id=env_id, timeout=self._ps_timeout,
        )
        if ids is None:
            from lite.gym.errors import ReconcileProbeError
            raise ReconcileProbeError(
                f"live_ids({env_id}): docker ps failed (prefix={prefix!r}) — skipping cycle"
            )
        return ids

    def reap(
        self, env_id: str, scope: ServerScope, in_use: set[str], *, boot: bool = False
    ) -> int:
        """``docker rm -f`` orphan containers: live (incl. Exited), NOT in
        ``in_use``.

        **Boot recovery** (``boot=True`` — the framework's one-shot
        ``recover_all`` call before serving) is WIDE-NET — no orphan age guard,
        no per-cycle cap, no inter-rm delay: nothing races before serving and
        every in-scope container is a prior-lifetime leftover. **Steady-state**
        (``boot=False``) reaps apply the orphan age guard (``min_orphan_age_s``)
        + ``MAX_ORPHANS_PER_CYCLE`` cap + inter-rm breathing window + per-name
        quarantine backoff. The boot/steady split is owned by the framework
        (``recover`` vs ``reconcile``), not latched here."""
        if scope.server_port is None:
            # UNSCOPED scope → the prefix degrades to bare ``lite-env-``,
            # matching EVERY server's and co-tenant's containers host-wide.
            # Happens whenever an embedder enters make_app(port=None)'s
            # lifespan (a TestClient context manager is enough) — observed
            # in the wild: another checkout's pytest swept a sibling server's
            # live containers mid-run. A destructive sweep with no ownership evidence is
            # never safe; refuse loudly.
            logger.error(
                "reap(%s, boot=%s): UNSCOPED scope (server_port=None) would "
                "sweep containers host-wide (make_app(port=None) lifespan?). "
                "REFUSING; pass a real port to reap prior-lifetime leftovers.",
                env_id, boot,
            )
            return 0
        prefix = container_name_prefix(server_port=scope.server_port, token_hash=None)
        scan = _docker_ps_with_time(prefix, env_id=env_id, timeout=self._ps_timeout)
        live_all = scan.all
        # Prune BEFORE the empty-scan early-return: an empty scan (idle daemon OR
        # a `docker ps` failure) must still expire stale quarantine entries, else
        # the dict grows unbounded while the daemon flaps. live_names=∅ → every
        # quarantined name counts as gone → pruned (correct "container gone").
        now = time.time()
        _prune_quarantine(live_names={n for n, _ in live_all}, now=now)
        if not live_all:
            return 0

        min_age = 0.0 if boot else self._min_orphan_age_s
        candidates = [
            name
            for name, created_at in live_all
            if name not in in_use and now - created_at > min_age
        ]
        if not boot:
            candidates = candidates[:MAX_ORPHANS_PER_CYCLE]

        reaped = 0
        for i, name in enumerate(candidates):
            if i > 0 and not boot:
                time.sleep(RM_INTER_DELAY_S)  # daemon breathing window (steady-state)
            if _try_orphan_rm(name, self._rm_timeout):
                reaped += 1
        return reaped


class ContainerServices(EnvServices):
    """``EnvServices`` base for container-backed envs: composes a
    :class:`ContainerReaper` and delegates ``live_ids``/``reap`` to it. A
    container env subclasses this and overrides only ``ensure`` (its image/deps
    check) — removing the identical delegator triplet from every docker env.

    Constructs a default ``ContainerReaper`` (all envs use the default tuning;
    the boot/steady split is framework-supplied, not reaper state, so the reaper
    holds only config). If an env ever needs custom tuning, override ``__init__``
    to pass it to ``ContainerReaper``.
    """

    def __init__(self) -> None:
        self._reaper = ContainerReaper()

    def live_ids(self, env_id: str, scope: ServerScope) -> set[str] | None:
        return self._reaper.live_ids(env_id, scope)

    def reap(
        self, env_id: str, scope: ServerScope, in_use: set[str], *, boot: bool = False
    ) -> int:
        return self._reaper.reap(env_id, scope, in_use, boot=boot)


class SingletonContainerServices(EnvServices, ABC):
    """``EnvServices`` base for SINGLETON envs whose shared backend is ONE container named
    ``container_name(scope)`` (``<svc>-<server_port>``). The sibling of
    :class:`ContainerServices` (DEDICATED, per-instance reaping): a SINGLETON's container is
    reclaimed ONLY boot-side (``reap(boot=True)``) or on explicit ``shutdown`` — both are
    ``docker rm -f`` of that one container, NEVER mid-run (family=SINGLETON gates the env out
    of steady per-instance reconcile, so the live container IS the shared backend).

    A new singleton container env subclasses this, declares the container NAME + family, and
    overrides only ``ensure`` (image/deps) and optionally ``health`` — it never re-copies the
    boot-reap / shutdown boilerplate owned here, the SINGLETON analogue of
    ``ContainerServices`` for DEDICATED.

    Subclass contract:
      * ``container_name(scope) -> str`` — the per-owner name (``<svc>-<server_port>``).
      * class attrs ``rm_timeout_s`` / ``rm_label`` — the ``docker rm -f`` bound + log label.

    (``live_ids`` is intentionally NOT defined — it inherits the base ``-> None`` ("no
    per-instance world"); a SINGLETON is never per-instance reconciled.)
    """

    rm_timeout_s: ClassVar[float] = 60.0
    rm_label: ClassVar[str] = "singleton"

    @abstractmethod
    def container_name(self, scope: ServerScope) -> str:
        """The per-owner shared-container name (``<svc>-<server_port>``).

        Abstract (not a ``NotImplementedError`` pseudo-abstract) so a subclass
        that forgets it fails at INSTANTIATION, not at its first reap."""

    def reap(
        self, env_id: str, scope: ServerScope, in_use: set[str], *, boot: bool = False
    ) -> int:
        # Boot-only: reclaim a prior crashed lifetime's container. Steady ticks never reach
        # here (family gates SINGLETON out of per-instance reconcile) — the live container IS
        # the shared backend, never reap it mid-run.
        if not boot:
            return 0
        return docker_rm_f(
            self.container_name(scope), timeout=self.rm_timeout_s, label=self.rm_label
        )

    def shutdown(self, env_id: str, scope: ServerScope) -> None:
        docker_rm_f(
            self.container_name(scope), timeout=self.rm_timeout_s, label=self.rm_label
        )
