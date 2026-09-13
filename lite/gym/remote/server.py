"""FastAPI app that exposes :class:`LiteBaseEnv` over HTTP.

Library half of the env-server split:
  - This module: schemas + state + endpoints + idle TTL reaper + auth.
  - Thin launcher: [`scripts/serve_env.py`](/scripts/serve_env.py) — argparse +
    ``uvicorn.run(make_app(State(...)))``.

Endpoints (all require ``Authorization: Bearer <token>`` when
``make_app(state, token=...)`` is given a non-None token; open when
``token=None`` for solo-dev). Three disjoint resource trees: ``/envs/*``
for **env types** (registry probes — read-heavy), ``/instances/*``
for **live sessions** (RPC — write-heavy), and ``/admin/*`` for
**cluster introspection** (cross-token, operator-side).

Admission — three independent layers + a structural daemon lock,
each emitting a typed HTTP 503 + ``Retry-After`` that the client
retries (:class:`lite.gym.remote.client.LiteEnvClient._post_with_retry`).
The server's exception handler attributes every 503 to one layer via
the ``layer`` field on :class:`~lite.gym.errors.CapacityExhausted`,
counted at ``cua_lite_admission_503_total{layer="..."}``.

  L1  emergency   — host RAM / load via psutil; pluggable per server
                    via :func:`~lite.gym.remote.admission.register_emergency_check`.
                    Dynamic — re-checked on every POST /instances.
  L2  capacity    — in-flight env cap. Auto-derives from vCPU and RAM at boot
                    (MIN(vCPU × 4, RAM_GB / 4) clamped to [4, 512]);
                    operator may override with ``--max-live-envs``.
                    Static at runtime.
  L3  env_internal — each env raises ``CapacityExhausted`` from its
                     own ``reset()`` / pool acquire when its bounded
                     internal resource is full (mobilegym browser pool,
                     port range, OmniBoxes slot, ...). Env-specific cap.
  docker_sema     — structural, not admission: process-wide semaphore
                    serializing ``docker run`` calls so docker-backed
                    envs don't trample the daemon. Raises L3-shape
                    503 on timeout. See
                    :func:`~lite.gym.remote.admission.docker_create_slot`.

L1+L2 live in this module; L3 is per-env (see each env's main.py);
docker_sema is shared by androidworld / androidlab / lite.osworld
via :mod:`lite.gym.remote.admission`. The framework is env-agnostic:
this file never branches on env_id, never reads env-specific config.

  # ── Env catalog (types) ───────────────────────────────────────────────
  GET    /envs                       -> list env_ids on this server
                                        ?expand=metadata → map of {env_id: {available,
                                        n_tasks, splits, error?}}.
  GET    /envs/{env_id}              -> single-type metadata (404 if not registered;
                                        200 with available=false + error if registered
                                        but import-broken or blocked).
  GET    /envs/{env_id}/tasks        -> task list for one env type.

  # ── Instances (live sessions, self-scoped to caller's token) ──────────
  POST   /instances                  -> create        body: {env_key, env_kwargs, session_id}
                                                       Runs L1+L2 admission; L3 fires
                                                       later inside env.reset() if the
                                                       env's own backend pool is full.
  GET    /instances                  -> list          ?session_id=&env_key=&env_id=
                                                       → {instances: [...]} (filtered to
                                                       caller's own token_hash)
  GET    /instances/{id}             -> single-instance state lookup (own token only;
                                                       404 unknown, 403 owned by another)
  POST   /instances/{id}/reset       -> reset
  POST   /instances/{id}/step        -> step          body: {actions}
  DELETE /instances/{id}             -> close         Always destroys.
  DELETE /instances                  -> bulk close    ?session_id=&env_id=&force=&dry_run=
                                                       Wide-net guard: 400 unless
                                                       session_id or env_id is pinned, or
                                                       force=true is passed.

  # ── Admin / cluster introspection (cross-token, auth via ``--admin-token``)
  # Auth: when --admin-token is set, /admin/* requires it. Else open in
  # passthrough server-mode / 404 (disabled) in strict server-mode. See
  # :func:`_make_admin_bearer`. Filter vocabulary is shared across the
  # two list endpoints (token / session_id / env_id) so admin tooling
  # uses one query string against either.
  GET    /admin/budget               -> {in_flight, max_live_envs, pct_used,
                                        host_ram_percent, host_ram_free_bytes,
                                        host_load_per_cpu,
                                        503_total: {emergency, capacity,
                                                    docker_sema, env_internal}}
                                        — cluster admission pressure.
  GET    /admin/tokens               -> {tokens: [{token, token_hash, first_seen_at,
                                        last_seen_at, instances_created_total,
                                        instances_active}, ...]}
                                        every token the server has ever seen.
  GET    /admin/usage                -> {usage: [{token, token_hash, session_id, env_id,
                                        n_active_instances}, ...]} per
                                        (token, session, env_id) grain.
                                        ?token=&session_id=&env_id= filters.
  GET    /admin/instances            -> {instances: [...session_summary + token]}
                                        cross-token live-instance rows w/ id.
                                        ?token=&session_id=&env_id=&env_key= filters.

  # ── Server ────────────────────────────────────────────────────────────
  GET    /metrics                    -> Prometheus text-format: in_flight,
                                        admission_503_total by layer,
                                        host_ram_percent / load_per_cpu,
                                        per-env-id live counts.
  GET    /host_status                -> CPU / memory / disk / process metrics
                                        (bearer-gated like /metrics: operator
                                        surface, same audience).

Sibling :mod:`lite.gym.remote.client` is wire-compatible with this
module.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
import random
import shutil
import subprocess
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psutil
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import lite
import lite.gym as gym
from lite.core.tools.calls import (
    RUNTIME_RESULT_CALL_ID_KEY,
    RuntimeEnvAction,
    tool_call_id,
    validate_lite_tool_call,
)
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import (
    CapacityExhausted,
    EnvAccessDenied,
    EnvDepsMissingError,
    EnvUnavailable,
    InstanceGone,
    LiteGymError,
    MakeFailed,
    ProtocolMisuse,
)
from lite.gym.remote.admission import AdmissionGate
from lite.gym.remote.conflict import (
    OTHERS_CONFLICT_KEYS,
    OTHERS_MUTATING,
    ConflictKeyTable,
    restore_backend_dispatch,
)
from lite.gym.remote.errors import ModelOutputError, action_payload_metadata
from lite.gym.remote.frame import (
    FRAME_MAGIC,
    FRAME_VERSION,
    encode_reset_observation,
    encode_step_result,
)
from lite.gym.remote.scope import ServerScope
from lite.gym.services import EnvServerResource
from lite.gym.types import (
    InstanceView,
    LiteEnvObservation,
    LiteEnvStepResult,
    StateView,
)
from lite.gym.utils.backend.rpc import may_reissue
from lite.gym.utils.feedback.ingress import (
    invalid_env_tool_call_envelope_message,
    is_internal_finish_tool_call,
)
from lite.gym.utils.server.routing import serve_locally
from lite.utils.registry import split_key

# ``EnvServerPoolable.bind()`` remains the construction-time handoff for
# task-specific soft state in both direct and server cold-spawn paths.

logger = logging.getLogger(__name__)

_LOOP_STALL_INTERVAL_S = 1.0
_LOOP_STALL_WARN_AFTER_S = 5.0
_LOOP_STALL_WINDOW_S = 60.0


# =============================================================================
# State
# =============================================================================

@dataclasses.dataclass
class _EnvSession:
    """In-memory record for one live env instance on the server."""
    env: LiteBaseEnv
    env_key: str          # e.g. "lite.osworld@osworld_chrome_030eeff7"
    env_id: str           # e.g. "lite.osworld"  (parsed once for cleanup filtering)
    session_id: str       # cleanup/session label, e.g. "alice-1"
    token_hash: str | None  # sha256(bearer_token)[:6] — caller identity; used
                          # to scope reset/step/delete ownership + cleanup
                          # filter. In passthrough mode (make_app(token=None))
                          # this is sha256 of whatever bearer the client
                          # sent, or the literal ``"anonymous"`` for requests
                          # with no Authorization header. In strict mode
                          # (token=<T>) it's sha256(T)[:6] — the same for
                          # every accepted caller. ``None`` should only appear
                          # on legacy/unowned rows created by older in-process
                          # tests or tools; ownership comparisons naturally
                          # reject those rows by inequality.
    created_at: float
    last_active_at: float
    n_steps: int = 0
    #: Count of step/reset handler coroutines currently awaiting inside this
    #: env. The idle-TTL reaper and bulk-DELETE skip sessions with
    #: ``in_flight > 0`` — ``last_active_at`` is stamped at handler ENTRY and
    #: never refreshed during the awaited call, so a long op (KVM boot-class
    #: reset under a lowered --idle-ttl-sec) would otherwise be closed
    #: mid-call, tearing the backend down under a live episode.
    in_flight: int = 0
    #: ``True`` once ``/reset`` has succeeded for this instance. ``/step``
    #: before reset is a client state-machine error → 409 (not a 500 from
    #: the env touching un-booted state). Per-instance: a fresh POST
    #: /instances starts ``False`` until its reset.
    reset_done: bool = False
    #: The ``env_kwargs`` the env was originally created with. Used only
    #: as a debug/admin readback.
    env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: Raw token captured at instance creation (denormalised from
    #: :attr:`State.token_registry` for convenience in admin views and
    #: log lines). ``None`` for the anonymous identity (no/empty
    #: Authorization header) and for legacy/unowned rows.
    token: str | None = None


@dataclasses.dataclass
class _TokenInfo:
    """Registry entry tracking a token (HTTP Bearer credential) seen by
    the server.

    Tokens are deliberately not stored on :class:`_EnvSession` instances
    alone — they belong in a server-wide registry so admin views can
    answer "who has touched this server" even for callers whose
    instances all closed. Each entry persists for the server's lifetime
    (no GC); the registry is in-memory and resets on restart.

    The raw ``token`` field is None for the canonical "anonymous"
    identity (no or empty Authorization header) — there's only one such
    identity, keyed by the literal ``"anonymous"`` token_hash.
    """
    token: str | None
    token_hash: str
    first_seen_at: float
    last_seen_at: float
    instances_created_total: int = 0


class State:
    """Holds the live env table + admission policy.

    Admission is delegated to :class:`~lite.gym.remote.admission.AdmissionGate`
    (passed in via the ``admission`` constructor arg). The gate runs L1
    (host sensors) + L2 (in-flight cap) on every POST /instances; L3
    (env-internal) fires when an env's own ``reset()`` raises
    :class:`~lite.gym.errors.CapacityExhausted`.

    The State object owns the env table, token registry, idle TTL,
    and the histogram counters; it does NOT make admission
    decisions itself.
    """

    # Histogram buckets in seconds. Covers fast lite.demo steps (<100ms),
    # typical lite.osworld (1-3s), and the long androidlab tail (up to
    # ~60s on contention). +Inf is implicit.
    _STEP_DURATION_BUCKETS = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
    # Reset is heavier than step: a lite.demo reset is sub-second, a
    # mobilegym browser ctx ~1–3 s, an android KVM cold-boot 30–300 s.
    # Buckets span the full range so the long tail isn't all dumped in
    # the +Inf overflow.
    _RESET_DURATION_BUCKETS = (0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)

    def __init__(
        self,
        admission: "AdmissionGate",
        idle_ttl_sec: float,
        allowed_env_ids: set[str] | None = None,
        blocked_env_ids: frozenset[str] | None = None,
        reset_jitter_sec: float = 0.0,
        boot_jitter_sec: float = 0.0,
        reset_concurrency: int | None = None,
        warm_singleton_enabled: bool = False,
    ) -> None:
        # L1+L2 admission gate. Server holds a reference; never branches
        # on env_id; never reads env-specific config. All admission
        # decisions go through ``admission.require_capacity()`` / ``.admit()`` /
        # ``.release()``.
        self.admission = admission
        self.idle_ttl_sec = idle_ttl_sec
        # Allow-list of env_ids accepted by POST /instances. ``None`` =
        # accept whatever ``gym.make`` accepts (no restriction). Empty
        # set = deny everything (explicit lockdown). See
        # ``scripts/serve_env.py:--env-ids``.
        self.allowed_env_ids = allowed_env_ids
        # Optional hard-block list (always denied, even when
        # ``allowed_env_ids`` is ``None`` / contains them). Most launchers
        # leave this empty; tests and custom embedders can still exercise it.
        self.blocked_env_ids: frozenset[str] = blocked_env_ids or frozenset()
        # Pre-reset jitter (seconds). Each ``POST /instances/<id>/reset`` sleeps
        # ``random.uniform(0, reset_jitter_sec)`` before handing off to
        # ``env.reset()``. Smooths the snapshot-restore NVMe burst when N
        # callers hit reset() simultaneously (observed at n=128 androids
        # K=2 second-pass: 128 × ~1 GB snapshot read in one wall-clock
        # millisecond → NVMe IOPS spike → reset() timeouts). 0.0 disables.
        self.reset_jitter_sec = reset_jitter_sec
        # Pre-create (boot) jitter (seconds). Each cost > 0 ``POST /instances``
        # sleeps ``random.uniform(0, boot_jitter_sec)`` before
        # ``gym.make()``. Smooths NVMe-IO contention when two drivers
        # bursting envs at the same wall-clock second. 0.0 disables.
        self.boot_jitter_sec = boot_jitter_sec
        # Listen port of this env-server instance. Set by ``make_app(port=...)``
        # and threaded into ``gym.make`` as ``server_port``; env-side
        # code uses it (alongside ``token_hash``) to scope its spawned
        # external resources so two co-resident env-server instances on
        # the same host stay mutually isolated. Server.py treats it as
        # an opaque identifier — what each env does with it is owned by
        # that env's main.py (see e.g. ``lite.gym.utils.config.naming`` for
        # docker-spawning envs).
        self.server_port: int | None = None
        #: This server run's identity, for resource-ownership scoping. Built once
        #: in :func:`make_app` from ``(port, strict-token)`` and passed to the
        #: reconcile loop / recovery / shutdown. ``None`` until
        #: ``make_app`` runs.
        self.scope: ServerScope | None = None
        self.envs: dict[str, _EnvSession] = {}
        #: hash → :class:`_TokenInfo` for every token this server has
        #: ever seen (since process start); why it is server-wide rather than
        #: per-session is argued on :class:`_TokenInfo`. Populated by the
        #: ``_bearer`` dependency on each authenticated request.
        #: Lock-free reads + writes: dict[key]=val is GIL-atomic in
        #: CPython, and the only mutated counter
        #: (``instances_created_total``) is incremented under
        #: ``state.lock`` in the POST /instances handler.
        self.token_registry: dict[str, _TokenInfo] = {}
        # Detached close tasks from ``DELETE /envs/<id>`` — the handler
        # pops the env from ``self.envs`` (releasing its L2 slot) then
        # spawns ``_close_quietly`` (calls ``env.close()``, polymorphic)
        # as a background task so the HTTP response can return in <1 ms
        # instead of waiting on the env's resource teardown (which can
        # take 1-60+ s depending on env type).
        # Lifespan shutdown drains the set with a deadline so SIGTERM
        # doesn't orphan in-flight teardowns.
        self.pending_closes: set[asyncio.Task[None]] = set()
        # Single lock guards env table mutations (create / close / reap).
        # Per-env step/reset run unlocked — multiple envs progress concurrently.
        self.lock = asyncio.Lock()
        #: --warm-singleton: background pre-warm of served SINGLETON backends at
        #: startup (non-blocking). It is what makes ``available`` flip true on its own
        #: (lazy never would — health() only probes), so a launcher can wait-for-hot;
        #: and it overlaps the slow boot (WA gitlab ~5-15 min) with server startup.
        self.warm_singleton_enabled: bool = warm_singleton_enabled
        #: Shared-backend isolation. Generic per-key
        #: readers-writer gate — env-agnostic; see
        #: :mod:`lite.gym.remote.conflict`. Mutated only under ``self.lock``.
        #: Empty for every env that doesn't set ``others["conflict_keys"]`` →
        #: the gate is a no-op (zero behavior change).
        self.conflict: ConflictKeyTable = ConflictKeyTable()
        from lite.gym.remote.admission import derive_reset_concurrency
        #: Cap on concurrent ``env.reset()`` calls, **per env_id**. ``None``
        #: defers to the HostCapacity-derived default, imported and called
        #: lazily right above rather than kept in a module-level constant —
        #: that would freeze at first import, before a test could monkeypatch
        #: HostCapacity. Reset is rollout's critical path; ``/step`` is
        #: deliberately NOT gated: step is sub-second for every env we ship, so
        #: a sema there is over-engineering until there is evidence of
        #: step-induced resource pressure.
        self.reset_concurrency: int = (
            reset_concurrency if reset_concurrency is not None
            else derive_reset_concurrency()
        )
        #: Per-``env_id`` asyncio.Semaphores. Lazy-created via
        #: :meth:`reset_sema_for` so a
        #: server hosting N env_ids ends up with N independent queues.
        #: Per-env isolation: a 100-deep lite.osworld spawn queue can't
        #: starve mobilegym's spawns; a saturated lite.osworld reset
        #: queue can't block androidworld's resets. The semas
        #: themselves are bound to the running loop on first acquire —
        #: safe to construct in __init__ (Python 3.10+ asyncio sync
        #: primitives are loop-free at construction).
        self._reset_semas: dict[str, asyncio.Semaphore] = {}
        #: Per-``env_id`` counter of /reset calls past the reset_sema gate.
        self.reset_active: dict[str, int] = {}
        # Lightweight in-process counters + histograms for /metrics.
        # Keyed by ``env_id``. Updated lock-free — increments are
        # race-tolerant for monitoring purposes (a missed increment
        # doesn't break behavior). For correctness-critical accounting
        # use ``state.envs``.
        self.step_total: dict[str, int] = {}
        self.step_5xx_total: dict[str, int] = {}
        self.reset_total: dict[str, int] = {}
        self.reset_5xx_total: dict[str, int] = {}
        self.asgi_errors_total: int = 0
        #: Outer-retry counters for the /reset + /step retry layer. Both
        #: endpoints retry, but on DIFFERENT licences (``/reset`` on
        #: replay-safety, ``/step`` only on a proven never-reached call), so the
        #: endpoint label is load-bearing: a step-retry storm and a reset-retry
        #: storm mean different things. One per endpoint × outcome so /metrics
        #: can compare attempted vs recovered — a high recovered ratio shows the
        #: retry layer earning its keep; a high exhausted ratio means the
        #: underlying problem isn't actually transient and needs a structural
        #: fix (e.g. mid-rollout container death).
        self.outer_retry_fired: dict[str, int] = {"step": 0, "reset": 0}
        self.outer_retry_recovered: dict[str, int] = {"step": 0, "reset": 0}
        self.outer_retry_exhausted: dict[str, int] = {"step": 0, "reset": 0}
        # Per-env_id histograms: bucket_idx → count of latencies that
        # fell in that bucket. Last bucket is implicit +Inf. Reset and
        # step are separate because their distributions barely overlap
        # (step typically sub-second, reset can be minutes for cold
        # KVM boots).
        self.step_duration_buckets: dict[str, list[int]] = {}
        self.step_duration_sum: dict[str, float] = {}
        self.reset_duration_buckets: dict[str, list[int]] = {}
        self.reset_duration_sum: dict[str, float] = {}
        # ----------------------------------------------------------------
        # Per-env-id sema lookups follow at the method level (below).
        # ----------------------------------------------------------------
        # Cumulative count of state.envs entries dropped by the
        # background drift reaper because the env's external resource
        # vanished out-of-band (e.g. host docker daemon restart, manual
        # ``docker kill``, in-container OOM-killer of qemu, an
        # OmniBoxes slot recycled by its own TTL — the framework reconcile
        # loop detects these ghosts via each env's ``live_ids``). Surfaced via
        # /metrics as ``cua_lite_envs_dead_total``. A rising counter
        # signals chronic out-of-band resource loss — worth investigating.
        self.dead_envs_total: int = 0
        # Writers whose shared-backend restore FAILED on close.
        # We keep the conflict key held (never re-lease a dirty stack), so a
        # rising counter means a stack is wedged — exposed on /metrics as
        # ``cua_lite_conflict_restore_fail_total``. Needs operator attention
        # (manual reset + server restart to recover the key).
        self.restore_fail_total: int = 0
        # Shared-backend conflict-gate 503s — claims rejected because a
        # conflicting writer/reader holds the key. Distinct from the admission
        # layers (which self-count); exposed on /metrics as
        # ``cua_lite_conflict_503_total``. A high value = heavy same-stack
        # serialization pressure (raise --max-attempts / use N stacks).
        self.conflict_503_total: int = 0
        # Event-loop stall telemetry for /host_status. Lifetime max helps
        # postmortems; the recent window decays after one bad boot so operators
        # can tell whether the server is still unhealthy.
        self.loop_stall_max_seconds: float = 0.0
        self.loop_stall_max_at: float | None = None
        self.loop_stalls: deque[tuple[float, float]] = deque()

    @classmethod
    def for_env_server(
        cls,
        *,
        admission: "AdmissionGate",
        idle_ttl_sec: float,
        allowed_env_ids: set[str] | None = None,
        reset_concurrency: int | None = None,
        warm_singleton_enabled: bool = False,
    ) -> "State":
        """Build launcher-facing state with env-var-only server tuning."""
        return cls(
            admission=admission,
            idle_ttl_sec=idle_ttl_sec,
            allowed_env_ids=allowed_env_ids,
            reset_jitter_sec=float(os.environ.get("CUA_LITE_RESET_JITTER_S", "3.0")),
            boot_jitter_sec=float(os.environ.get("CUA_LITE_BOOT_JITTER_S", "5.0")),
            reset_concurrency=reset_concurrency,
            warm_singleton_enabled=warm_singleton_enabled,
        )

    def reset_sema_for(self, env_id: str) -> asyncio.Semaphore:
        """Per-``env_id`` reset semaphore.

        Acquired by the /reset handler after jitter, around ``env.reset()`` +
        its retry loop.
        """
        sema = self._reset_semas.get(env_id)
        if sema is None:
            sema = self._reset_semas.setdefault(
                env_id, asyncio.Semaphore(self.reset_concurrency),
            )
        return sema

    def record_loop_stall(
        self,
        stall_s: float,
        *,
        observed_at: float | None = None,
        window_s: float = _LOOP_STALL_WINDOW_S,
    ) -> None:
        observed_at = time.time() if observed_at is None else observed_at
        if stall_s > self.loop_stall_max_seconds:
            self.loop_stall_max_seconds = stall_s
            self.loop_stall_max_at = observed_at
        self.loop_stalls.append((observed_at, stall_s))
        self._prune_loop_stalls(now=observed_at, window_s=window_s)

    def _prune_loop_stalls(self, *, now: float, window_s: float) -> None:
        cutoff = now - window_s
        while self.loop_stalls and self.loop_stalls[0][0] < cutoff:
            self.loop_stalls.popleft()

    def loop_stall_snapshot(
        self,
        *,
        now: float | None = None,
        window_s: float = _LOOP_STALL_WINDOW_S,
    ) -> dict[str, float | None]:
        now = time.time() if now is None else now
        self._prune_loop_stalls(now=now, window_s=window_s)
        recent_max = max((stall for _, stall in self.loop_stalls), default=0.0)
        return {
            "stall_max_seconds": self.loop_stall_max_seconds,
            "stall_max_at": self.loop_stall_max_at,
            "stall_max_60s": recent_max,
        }

    @staticmethod
    def _observe_duration(
        buckets_by_env: dict[str, list[int]],
        sum_by_env: dict[str, float],
        env_id: str,
        duration_s: float,
        bucket_defs: tuple[float, ...],
    ) -> None:
        """Insert one observation into a Prometheus-style histogram.

        Shared between step and reset (and any future op-duration metric)
        so the bucket-indexing logic lives in exactly one place. Last
        slot is the implicit ``+Inf`` overflow bucket.
        """
        buckets = buckets_by_env.setdefault(env_id, [0] * (len(bucket_defs) + 1))
        idx = len(bucket_defs)  # default: overflow bucket
        for i, b in enumerate(bucket_defs):
            if duration_s <= b:
                idx = i
                break
        buckets[idx] += 1
        sum_by_env[env_id] = sum_by_env.get(env_id, 0.0) + duration_s

    def record_step(self, env_id: str, duration_s: float, ok: bool) -> None:
        """Record one step call's latency + success/fail. Race-tolerant."""
        self.step_total[env_id] = self.step_total.get(env_id, 0) + 1
        if not ok:
            self.step_5xx_total[env_id] = self.step_5xx_total.get(env_id, 0) + 1
        self._observe_duration(
            self.step_duration_buckets, self.step_duration_sum,
            env_id, duration_s, self._STEP_DURATION_BUCKETS,
        )

    def record_reset(self, env_id: str, duration_s: float, ok: bool) -> None:
        """Record one reset call's latency + success/fail. Race-tolerant.

        Reset latency observability lets operators spot host-pressure
        signals (recent p95 climbing) before L1 sensors fire.
        """
        self.reset_total[env_id] = self.reset_total.get(env_id, 0) + 1
        if not ok:
            self.reset_5xx_total[env_id] = self.reset_5xx_total.get(env_id, 0) + 1
        self._observe_duration(
            self.reset_duration_buckets, self.reset_duration_sum,
            env_id, duration_s, self._RESET_DURATION_BUCKETS,
        )


# =============================================================================
# Schemas
# =============================================================================

class _CreateBody(BaseModel):
    env_key: str
    env_kwargs: dict[str, Any] = {}
    session_id: str = "default"


class _StepBody(BaseModel):
    actions: list[Any]


def _is_instance_step_path(path: str) -> bool:
    parts = path.strip("/").split("/")
    return len(parts) == 3 and parts[0] == "instances" and parts[2] == "step"


def _validation_error_summary(exc: RequestValidationError) -> str:
    items: list[str] = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = str(err.get("msg", "invalid request body"))
        typ = str(err.get("type", ""))
        suffix = f" ({typ})" if typ else ""
        items.append(f"{loc}: {msg}{suffix}" if loc else f"{msg}{suffix}")
    if not items:
        return "malformed /step request body"
    if len(exc.errors()) > len(items):
        items.append(f"... {len(exc.errors()) - len(items)} more validation errors")
    return "malformed /step request body: " + "; ".join(items)


def _remote_step_action_error(
    index: int,
    action: Any,
    message: str,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> ModelOutputError:
    metadata = action_payload_metadata(index, action)
    if extra_metadata:
        metadata.update(extra_metadata)
    return ModelOutputError(
        f"malformed /step request body: body.actions.{index}: {message}",
        payload_metadata=metadata,
    )


def _validate_remote_step_actions(
    actions: list[Any],
) -> list[RuntimeEnvAction]:
    """Validate public Lite tool-call envelopes at the remote /step boundary.

    The canonical-shape FACT comes from
    :func:`lite.core.tools.calls.validate_lite_tool_call`, the one owner of
    "what shape is a Lite tool call". This boundary only decides the
    CONSEQUENCE: malformed calls with a usable canonical ``id`` can still become
    model-visible env feedback, while unpairable or noncanonical envelopes stop
    here as typed 422s.

    One rule stays boundary-local because no owner upstream can hold it:
    ``id`` uniqueness *within one batch*. The ``RUNTIME_RESULT_CALL_ID_KEY``
    reservation (runtime-only result routing, which a client must never spell) is
    NOT local — :func:`lite.gym.utils.feedback.results
    .invalid_env_tool_call_envelope_message` enforces the same rule on the direct
    path. What is local is only the CONSEQUENCE: a 422 carrying
    ``payload_metadata`` here, a message string there.

    Env implementations only see canonical Lite calls. Tool/schema/action
    validation remains env-owned after this shape gate passes. Missing ``id``
    remains valid for direct/server parity; envs may return aggregate results
    without per-call ids for such actions.
    """
    seen_result_ids: dict[str, int] = {}

    def note_result_id(index: int, action: dict[str, Any], result_id: str) -> None:
        previous = seen_result_ids.get(result_id)
        if previous is not None:
            raise _remote_step_action_error(
                index,
                action,
                f"duplicate call id {result_id!r}",
                extra_metadata={"duplicate_of_action_index": previous},
            )
        seen_result_ids[result_id] = index

    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise _remote_step_action_error(
                index,
                action,
                f"expected an object, got {type(action).__name__}",
            )

        try:
            is_internal_finish = is_internal_finish_tool_call(action)
        except (KeyError, TypeError):
            is_internal_finish = False

        if is_internal_finish:  # runtime-only, not model-emitted
            result_call_id = action.get(RUNTIME_RESULT_CALL_ID_KEY)
            if result_call_id is not None:
                if not isinstance(result_call_id, str) or not result_call_id:
                    raise _remote_step_action_error(
                        index,
                        action,
                        "_result_call_id must be a non-empty string when present",
                    )
                note_result_id(index, action, result_call_id)
            continue

        if RUNTIME_RESULT_CALL_ID_KEY in action:
            raise _remote_step_action_error(
                index,
                action,
                "_result_call_id is reserved for runtime-only result routing",
            )

        if "tool_call_id" in action:
            raise _remote_step_action_error(
                index,
                action,
                "Lite tool calls must use canonical 'id', not 'tool_call_id'",
            )

        call_id = tool_call_id(action)
        shape_error = validate_lite_tool_call(action, "action", require_id=False)
        if shape_error is not None:
            invalid_feedback = invalid_env_tool_call_envelope_message(action)
            pairable = invalid_feedback is not None and call_id is not None
            noncanonical_shape = (
                "noncanonical" in shape_error
                or "tool_call_id" in shape_error
                or ".id must be" in shape_error
            )
            if not pairable or noncanonical_shape:
                raise _remote_step_action_error(index, action, shape_error)

        if call_id is None:
            continue
        if not isinstance(call_id, str) or not call_id:
            raise _remote_step_action_error(
                index,
                action,
                "canonical 'id' must be a non-empty string",
            )
        note_result_id(index, action, call_id)

    return actions  # type: ignore[return-value]


# =============================================================================
# Build info — surfaced via /host_status so operators can confirm which
# cua-lite commit a remote env-server is running without having to ssh.
# Resolved once at module import; missing-git / pip-installed cases
# fall back to ``{"commit": None, "branch": None, "dirty": None}``.
# =============================================================================

def _resolve_cua_lite_build_info() -> dict[str, str | bool | None]:
    unknown: dict[str, str | bool | None] = {"commit": None, "branch": None, "dirty": None}

    pkg_dir = Path(lite.__file__).resolve().parent
    for candidate in (pkg_dir, *pkg_dir.parents):
        if (candidate / ".git").exists():
            git_dir = candidate
            break
    else:
        # pip-installed / Docker image without source — no .git anywhere.
        return unknown

    if shutil.which("git") is None:
        # Source tree present but git binary not on PATH (slim runtime image,
        # etc.). Short-circuit instead of doing 3 doomed subprocess attempts.
        return unknown

    def _git(*args: str) -> str | None:
        # Returns the trimmed stdout on success — possibly empty (e.g.
        # ``git status --porcelain`` on a clean tree). Only returns None
        # when the subprocess itself failed; callers must distinguish
        # "no output but valid result" from "couldn't ask git" themselves.
        try:
            return subprocess.run(
                ["git", "-C", str(git_dir), *args],
                check=True, capture_output=True, text=True, timeout=2,
            ).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None

    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    porcelain = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": None if branch == "HEAD" else branch,  # HEAD = detached
        "dirty": None if porcelain is None else bool(porcelain),
    }


_CUA_LITE_BUILD_INFO = _resolve_cua_lite_build_info()


_SECRET_CONFIG_KEY_PARTS = (
    "api_key",
    "apikey",
    "token",
    "password",
    "secret",
    "credential",
)


def _env_config_dir_for_env_id(env_id: str) -> Path | None:
    """Best-effort env_id -> env config directory.

    Programmatic sub-envs (``browsergym.webarena``, ``cua.bench.kicad``,
    ``lite.cuaworld.<software>``) share a parent package config. Walk from the
    full dotted name toward its parent until a ``configs/default.yaml`` is found.
    """
    import sys

    envs_dir = sys.modules["lite.gym.registry"]._ENVS_DIR
    parts = env_id.split(".")
    for end in range(len(parts), 0, -1):
        candidate = envs_dir / Path(*parts[:end])
        if (candidate / "configs" / "default.yaml").is_file():
            return candidate
    return None


def _is_secret_config_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _SECRET_CONFIG_KEY_PARTS)


def _redact_config_value(key: str, value: Any) -> Any:
    """Return a JSON-safe copy with secret-looking config values redacted."""
    if _is_secret_config_key(key):
        return "<redacted>" if value not in (None, "") else value
    if isinstance(value, dict):
        return {str(k): _redact_config_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_config_value("", v) for v in value]
    if isinstance(value, tuple):
        return [_redact_config_value("", v) for v in value]
    return value


def _server_process_env_status(env_var_prefix: str) -> dict[str, dict[str, bool]]:
    """Secret-safe process-env availability readback for server-mode debugging."""
    names = {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    }
    if env_var_prefix:
        names.add(f"{env_var_prefix}_CONFIG")
        names.add(f"{env_var_prefix}_API_KEY")
        names.add(f"{env_var_prefix}_BASE_URL")
        names.add(f"{env_var_prefix}_JUDGE_API_KEY")
        names.add(f"{env_var_prefix}_JUDGE_BASE_URL")
        # Some envs' judges resolve their API key through a layer of
        # indirection: `{prefix}_EVAL_API_KEY_VAR` names the actual key var
        # (webgym, online_mind2web, webharbor/webvoyager all follow this
        # convention) rather than holding the key itself.
        eval_key_var = os.environ.get(f"{env_var_prefix}_EVAL_API_KEY_VAR")
        if eval_key_var:
            names.add(eval_key_var)
    # Env-specific judge shims with intentionally shared names.
    names.update({"VLM_API_KEY", "VLM_BASE_URL"})
    return {
        name: {
            "set": name in os.environ,
            "available": bool(os.environ.get(name)),
        }
        for name in sorted(names)
    }


def _env_server_config_snapshot(
    env_id: str,
    state: State,
    *,
    token: str | None,
    admin_token: str | None,
) -> dict[str, Any]:
    """Redacted server-side config/provenance for an env catalog response.

    This is deliberately separate from:
      * per-instance ``env_kwargs`` (stored on live session rows), and
      * carried ``make_kwargs`` (the ``kwargs`` / ``env_make_kwargs`` task payload).
    """
    env_server = {
        "port": state.server_port,
        "auth_mode": "strict" if token is not None else "passthrough",
        "admin_auth_mode": (
            "strict" if admin_token is not None
            else "open" if token is None
            else "disabled"
        ),
        "allowed_env_ids": (
            sorted(state.allowed_env_ids)
            if state.allowed_env_ids is not None else None
        ),
        "blocked_env_ids": sorted(state.blocked_env_ids),
        "idle_ttl_sec": state.idle_ttl_sec,
        "boot_jitter_sec": state.boot_jitter_sec,
        "reset_jitter_sec": state.reset_jitter_sec,
        "reset_concurrency": state.reset_concurrency,
        "warm_singleton_enabled": state.warm_singleton_enabled,
    }

    config_dir = _env_config_dir_for_env_id(env_id)
    if config_dir is None:
        return {
            "provenance": {
                "available": False,
                "error": f"no configs/default.yaml resolved for env_id={env_id!r}",
            },
            "server_kwargs": {},
            "server_process_env": _server_process_env_status(""),
            "env_server": env_server,
            "build": dict(_CUA_LITE_BUILD_INFO),
        }

    from lite.gym.utils import config as env_config

    cfg = env_config.load(str(config_dir))
    return {
        "provenance": {"available": True, **cfg.provenance()},
        "server_kwargs": _redact_config_value("", cfg.server_kwargs),
        "server_process_env": _server_process_env_status(cfg.env_var_prefix),
        "env_server": env_server,
        "build": dict(_CUA_LITE_BUILD_INFO),
    }


# Retry-After hint (seconds) for a shared-backend conflict 503. The client
# retries until its own deadline, so this just paces the polling; modest so
# a freed backend is re-claimed promptly.
_CONFLICT_RETRY_AFTER_S = float(os.environ.get("CUA_LITE_CONFLICT_RETRY_AFTER_S", "5.0"))

# =============================================================================
# Auth
# =============================================================================

def _parse_bearer_token(authorization: str | None) -> tuple[str, str | None]:
    """Parse ``Authorization: Bearer <token>`` → ``(token_hash, raw_token)``.

    Single canonical place for the parse logic, shared by all three
    auth dependencies (passthrough / strict / admin). Returns the
    sentinel ``"anonymous"`` hash with ``raw_token=None`` for the
    missing-header and empty-token cases — callers don't have to repeat
    the conditionals.

    The function name keeps "bearer" because it parses the HTTP **Bearer**
    scheme; the data it produces is the application's "token" everywhere
    downstream (registry, log lines, JSON responses, ``--token`` CLI).
    """
    if not authorization or not authorization.startswith("Bearer "):
        return "anonymous", None
    raw = authorization[len("Bearer "):].strip()
    if not raw:
        return "anonymous", None
    return hashlib.sha256(raw.encode()).hexdigest()[:6], raw


def _record_token(state: State, token_hash: str, raw: str | None) -> None:
    """Learn or refresh a token in ``state.token_registry``."""
    now = time.time()
    info = state.token_registry.get(token_hash)
    if info is None:
        state.token_registry[token_hash] = _TokenInfo(
            token=raw,
            token_hash=token_hash,
            first_seen_at=now,
            last_seen_at=now,
        )
    else:
        info.last_seen_at = now


def _make_bearer(expected_token: str | None, state: State):
    """Build the FastAPI ``Depends`` callable that maps an incoming
    ``Authorization: Bearer <token>`` to the caller's identity
    (``token_hash`` = ``sha256(token)[:6]``).

    The token_hash flows into each env's external-resource scoping
    (resource naming + per-env cleanup filters) so two callers with
    different bearer tokens cannot accidentally clobber each other's
    envs even if they happen to share a ``session_id``. What each env
    DOES with token_hash is owned by that env's main.py.

    Two modes:

    * ``expected_token=None`` — **passthrough**. Any bearer token is
      accepted; identity is ``sha256(client_token)[:6]``. Requests with
      no ``Authorization`` header land under the literal identity
      ``"anonymous"``. Lets one env-server host multiple tenants
      identified by their client-side bearer token.
    * ``expected_token=<T>`` — **strict**. Only ``Authorization: Bearer
      T`` is accepted (anything else → 401); identity is ``sha256(T)[:6]``.
      Single-tenant production: rotate the token and everyone using the
      old one is locked out instantly.

    Every accepted request is recorded into :attr:`State.token_registry`
    so ``/admin/tokens`` can answer "who has touched this server".
    """
    if expected_token is None:
        async def _passthrough(authorization: str | None = Header(default=None)) -> str:
            token_hash, raw = _parse_bearer_token(authorization)
            _record_token(state, token_hash, raw)
            return token_hash
        return _passthrough

    expected_hash = hashlib.sha256(expected_token.encode()).hexdigest()[:6]

    async def _bearer(authorization: str | None = Header(default=None)) -> str:
        if authorization != f"Bearer {expected_token}":
            raise HTTPException(status_code=401, detail="invalid bearer token")
        _record_token(state, expected_hash, expected_token)
        return expected_hash

    return _bearer


def _make_admin_bearer(
    admin_token: str | None,
    server_strict: bool,
    state: State,
):
    """Build the ``Depends`` callable for ``/admin/*`` endpoints.

    Three modes, picked at server-build time:

    * ``admin_token=<T>``: require ``Authorization: Bearer T``. Anything
      else → 403. Works regardless of whether the main bearer is
      passthrough or strict — admin auth is independent.
    * ``admin_token=None, server_strict=False`` (passthrough server):
      **open admin**. Any caller can hit ``/admin/*``. Intended for
      single-operator deployments where the server itself is already
      behind a trusted network boundary. The dependency still records
      the caller's token for log context.
    * ``admin_token=None, server_strict=True`` (strict server): **admin
      disabled**. ``/admin/*`` returns 404 — the operator clearly
      configured gating; making admin secretly open would violate that
      expectation. Set ``--admin-token`` explicitly to enable.
    """
    if admin_token is not None:
        async def _admin_strict(authorization: str | None = Header(default=None)) -> str:
            if authorization != f"Bearer {admin_token}":
                raise HTTPException(status_code=403, detail="admin token required")
            return "admin"
        return _admin_strict

    if server_strict:
        async def _admin_disabled() -> str:
            raise HTTPException(
                status_code=404,
                detail=(
                    "admin endpoints disabled: server is in strict mode "
                    "(--token set) but no --admin-token was configured. "
                    "Pass --admin-token to enable."
                ),
            )
        return _admin_disabled

    async def _admin_open(authorization: str | None = Header(default=None)) -> str:
        token_hash, raw = _parse_bearer_token(authorization)
        _record_token(state, token_hash, raw)
        return token_hash
    return _admin_open


def _require_own_env(state: State, id_: str, token_hash: str) -> _EnvSession:
    """Look up env + assert the caller's token_hash matches its owner.
    Returns 404 if unknown id, 403 if owned by another token."""
    s = state.envs.get(id_)
    if s is None:
        # Typed body: the client reconstructs InstanceGone (retryable —
        # re-make recovers) instead of parsing a naked 404.
        raise InstanceGone(what=f"unknown instance id={id_!r}")
    if s.token_hash != token_hash:
        # Typed body: EnvAccessDenied is TERMINAL. A bare HTTPException's
        # {"detail": …} carries no ``error_type``, so the client fell through to
        # raise_for_status and ``is_retryable``'s unknown→True default re-ran the
        # whole task once per attempt against a token that will never be accepted.
        raise EnvAccessDenied(what=f"instance id={id_!r} owned by another token")
    return s


# =============================================================================
# Helpers
# =============================================================================

def _emit_histogram(
    lines: list[str],
    metric: str,
    help_text: str,
    buckets_by_env: dict[str, list[int]],
    sum_by_env: dict[str, float],
    bucket_defs: tuple[float, ...],
) -> None:
    """Append Prometheus histogram lines for one metric. Shared between
    step and reset (and any future op-duration export) so the line shape
    stays in lock-step with whatever observability tooling consumes
    these.

    Skips the metric entirely when no observations have been recorded —
    no HELP/TYPE banner appears, keeps the /metrics output noise-free
    on fresh servers.
    """
    if not buckets_by_env:
        return
    lines += [
        f"# HELP {metric} {help_text}",
        f"# TYPE {metric} histogram",
    ]
    for env_id_ in sorted(buckets_by_env):
        buckets = buckets_by_env[env_id_]
        cum = 0
        for i, b in enumerate(bucket_defs):
            cum += buckets[i]
            lines.append(f'{metric}_bucket{{env_id="{env_id_}",le="{b}"}} {cum}')
        cum += buckets[-1]
        lines.append(f'{metric}_bucket{{env_id="{env_id_}",le="+Inf"}} {cum}')
        lines.append(
            f'{metric}_sum{{env_id="{env_id_}"}} '
            f'{sum_by_env.get(env_id_, 0.0):.6f}'
        )
        lines.append(f'{metric}_count{{env_id="{env_id_}"}} {cum}')


def _frame_response(content: bytes) -> Response:
    return Response(content=content, media_type="application/octet-stream")


def _reset_observation_to_response(observation: LiteEnvObservation) -> Response:
    """Serialize a reset observation to the binary frame.

    The multi-MB screenshot rides raw in the frame body, so FastAPI's default
    ``JSONResponse`` never ``json.dumps`` the blob on the single event-loop
    thread. Only a tiny scalar header is JSON. Error responses are unaffected —
    they still go out as JSON (handlers only call this on the success path), so a
    200 ``application/octet-stream`` is unambiguously a success frame."""
    return _frame_response(encode_reset_observation(observation))


def _step_result_to_response(result: LiteEnvStepResult) -> Response:
    """Serialize a step result to the binary frame."""
    return _frame_response(encode_step_result(result))


def _session_summary(id_: str, s: _EnvSession) -> dict[str, Any]:
    """Self-scoped instance row.

    Used by ``GET /instances`` and ``GET /instances/{id}`` where the
    caller already knows their own token (they sent it). Deliberately
    omits the raw token to keep the self-view free of denormalised
    fields that other callers' rows shouldn't carry — use
    :func:`_admin_session_summary` for cross-token admin views.
    """
    return {
        "id": id_,
        "env_key": s.env_key,
        "env_id": s.env_id,
        "session_id": s.session_id,
        "token_hash": s.token_hash,
        "created_at": s.created_at,
        "last_active_at": s.last_active_at,
        "n_steps": s.n_steps,
        "env_kwargs": dict(s.env_kwargs),
    }


def _admin_session_summary(id_: str, s: _EnvSession) -> dict[str, Any]:
    """Cross-token instance row including the raw token.

    Same fields as :func:`_session_summary` plus ``token`` (captured
    at session creation time). Used by every ``/admin/*`` endpoint that
    enumerates instances — keeps the shape consistent so admin tooling
    can rely on a single schema.
    """
    return {**_session_summary(id_, s), "token": s.token}


# =============================================================================
# Admin aggregations
# =============================================================================
#
# Pure functions over State, kept separate from the route handlers so
# they're trivially unit-testable and reusable (e.g. a CLI tool could
# import these to render a local dashboard against an in-process
# State). Each aggregator takes its own snapshot under ``state.lock``
# and does the GROUP BY work outside the critical section — endpoints
# below become 1-2-line wrappers that just call + return.


@dataclasses.dataclass
class _AdminSnapshot:
    """One coherent point-in-time view of cluster state for admin reads.

    Taken under :attr:`State.lock` so the four ``/admin/*`` handlers
    plus ``/admin/budget`` see a consistent picture (no torn reads
    across concurrent create / close). All fields are independent
    snapshots — mutating them does not affect server state.

    Each ``/admin/*`` handler takes one snapshot then runs pure aggregation
    over it. Extending the admin surface = add a field here + add an
    aggregator over those fields, without touching the lock discipline.
    """
    envs: list[tuple[str, _EnvSession]]
    registry: dict[str, _TokenInfo]
    admission: dict[str, int | float | None]


async def _snapshot_state_for_admin(state: State) -> _AdminSnapshot:
    """Atomic snapshot of live state for admin reads.

    One lock acquisition per request — all admin views share the same
    coherent picture (no torn reads across concurrent create / close).
    Aggregation is pure-Python over the snapshot, so the lock is held
    for milliseconds at most.
    """
    async with state.lock:
        return _AdminSnapshot(
            envs=list(state.envs.items()),
            registry=dict(state.token_registry),
            admission=state.admission.snapshot(),
        )


def _aggregate_tokens(snap: _AdminSnapshot) -> list[dict[str, Any]]:
    """One row per token the server has ever seen.

    Per-row fields:
      - ``instances_active`` — live instance count owned by this token
      - ``instances_created_total`` — lifetime counter for successful creates,
        survives close.

    Sorted by ``instances_active`` desc then recency.
    """
    # Legacy/unowned entries carry token_hash IS NULL instead of caller
    # ownership; skip them so they don't show up as a phantom "None"-token row.
    active_per_hash: dict[str, int] = {}
    for _, s in snap.envs:
        if s.token_hash is None:
            continue
        active_per_hash[s.token_hash] = active_per_hash.get(s.token_hash, 0) + 1
    rows = [
        {
            "token": info.token,
            "token_hash": info.token_hash,
            "first_seen_at": info.first_seen_at,
            "last_seen_at": info.last_seen_at,
            "instances_created_total": info.instances_created_total,
            "instances_active": active_per_hash.get(info.token_hash, 0),
        }
        for info in snap.registry.values()
    ]
    # no per-token cost; sort by active instances then recency.
    rows.sort(key=lambda r: (-r["instances_active"], -r["last_seen_at"]))
    return rows


def _aggregate_usage(snap: _AdminSnapshot) -> list[dict[str, Any]]:
    """One row per active ``(token_hash, session_id, env_id)`` triple.

    Carries ``n_active_instances`` per row; sorted by that desc.
    """
    grouped: dict[tuple[str, str, str], int] = {}
    for _, s in snap.envs:
        if s.token_hash is None:  # unowned — see _aggregate_tokens
            continue
        key = (s.token_hash, s.session_id, s.env_id)
        grouped[key] = grouped.get(key, 0) + 1
    rows = []
    for (token_hash, session_id, env_id), n in grouped.items():
        info = snap.registry.get(token_hash)
        rows.append({
            "token_hash": token_hash,
            "token": info.token if info else None,
            "session_id": session_id,
            "env_id": env_id,
            "n_active_instances": n,
        })
    rows.sort(key=lambda r: (-r["n_active_instances"], r["token"] or "", r["env_id"]))
    return rows


def _budget_snapshot(snap: _AdminSnapshot) -> dict[str, Any]:
    """Cluster-level admission pressure ."""
    adm = snap.admission
    in_flight = int(adm["in_flight"] or 0)
    max_live = int(adm["max_live_envs"] or 1)
    pct = (in_flight / max_live * 100.0) if max_live > 0 else 0.0
    return {
        "in_flight": in_flight,
        "max_live_envs": max_live,
        "pct_used": round(pct, 2),
        "host_ram_percent": adm["host_ram_percent"],
        "host_ram_free_bytes": adm["host_ram_free_bytes"],
        "host_load_per_cpu": adm["host_load_per_cpu"],
        "503_total": {
            "emergency": adm["emergency_503_total"],
            "capacity": adm["capacity_503_total"],
            "docker_sema": adm["docker_sema_503_total"],
            "env_internal": adm["env_internal_503_total"],
        },
    }


async def _close_quietly(state: "State", s: _EnvSession, id_: str) -> None:
    """Close the env, then (for a writer) restore its shared backend and
    release its conflict keys (reset-in-close).

    Generic: the conflict bits are a no-op when the instance held no keys
    (every keyless env — ``holder`` returns ``None``). Runs
    detached / off the request path, so the slow ``restore`` (gitlab
    ~30-60 s) never blocks the API, and the key stays held *through*
    restore so the next claimant sees a clean backend."""
    try:
        await s.env.close()
    except Exception as e:
        logger.warning("env.close() failed for id=%s: %s", id_, e)
    held = state.conflict.holder(id_)
    if held is not None:
        keys, mutating = held
        if mutating:
            try:
                for key in keys:
                    await restore_backend_dispatch(s.env_id, key)
            except Exception as e:
                # Restore failed → the stack may be dirty. KEEP the key held
                # (never release onto a dirty backend — correctness) and
                # surface it; recovery needs a manual reset + server restart.
                async with state.lock:
                    state.restore_fail_total += 1
                logger.error(
                    "restore_backend failed for id=%s env=%s keys=%s: %s — "
                    "KEEPING conflict key(s) held; stack may be dirty. Manual "
                    "reset + server restart required to recover.",
                    id_, s.env_id, keys, e,
                )
                return
        async with state.lock:
            state.conflict.release(id_)


# =============================================================================
# Polymorphic drift reaper (background task)
# =============================================================================
#
# Dispatches periodic reconciliation between ``state.envs`` and each
# env's external resources via the framework reconcile loop (each env's
# ``live_ids``/``reap``). Server-side code stays agnostic about what those
# resources are. Correctness rests on dual age guards,
# prefix-stops-at-session-id, and cross-env validation.

#: Drift-reaper cycle cadence: how often the dispatcher fires (idle
#: gap between cycles). Pure scheduling knob — doesn't affect which
#: containers are reapable, nor how many per cycle. Both of those are
#: chosen (with their measurements) at
#: :data:`~lite.gym.remote.reaper.DEFAULT_MIN_ORPHAN_AGE_S` and
#: :data:`~lite.gym.remote.reaper.MAX_ORPHANS_PER_CYCLE`, which is
#: also where the throughput this cadence multiplies out to, and the
#: "never reap a LIVE container" argument, are stated — restating either
#: number here is how the two copies drift.
#:
#: Set via ``CUA_LITE_DRIFT_CYCLE_INTERVAL_S``. Read lazily so an env
#: var exported just before ``uvicorn.run`` still takes effect.
def _drift_cycle_interval_s() -> float:
    return float(os.environ.get("CUA_LITE_DRIFT_CYCLE_INTERVAL_S", "120.0"))


def build_state_view(state: State) -> StateView:
    """Snapshot ``state.envs`` into an immutable
    :class:`~lite.gym.types.StateView`. Caller MUST hold ``state.lock``.
    """
    by_env_id: dict[str, list[InstanceView]] = {}
    for id_, s in state.envs.items():
        # external_resource_id is the EnvServerResource capability's per-instance
        # handle (container name, slot id, vm id) — opaque to the server. Read it
        # off the UNWRAPPED env so wrapper layers don't hide the capability;
        # ``None`` when the env doesn't own a resource, or hasn't created one yet
        # (between gym.make and first reset).
        env = s.env.unwrapped
        by_env_id.setdefault(s.env_id, []).append(InstanceView(
            id=id_, env_id=s.env_id, env_key=s.env_key,
            session_id=s.session_id, token_hash=s.token_hash,
            created_at=s.created_at,
            external_resource_id=(
                env.external_resource_id
                if isinstance(env, EnvServerResource) else None
            ),
        ))
    return StateView(
        by_env_id={k: tuple(v) for k, v in by_env_id.items()},
        snapshot_at=time.time(),
    )


async def _reap_drift_one_cycle(state: State, scope: ServerScope) -> None:
    """One drift-reaper cycle: snapshot → per-env reconcile-loop dispatch →
    cross-env-validated ghost-id pop under lock. Extracted from the
    sleep loop so tests can fire a cycle directly.
    """
    from lite.gym.remote.recovery import reconcile_all

    t0 = time.monotonic()
    async with state.lock:
        state_view = build_state_view(state)
    # Reconcile does blocking docker ps/rm → run off the event loop. NOT wrapped
    # in wait_for: the dispatcher awaits each cycle to completion before the next,
    # so cycles never overlap, which keeps the module-global _RM_QUARANTINE
    # single-threaded. Each blocking reap/live_ids call carries its own subprocess
    # ``timeout=``, so a cycle is self-bounded (a pure-Python-wedged hook is an env
    # bug, surfaced by the dispatcher going quiet — not something to recover here).
    reports = await asyncio.to_thread(reconcile_all, state_view, scope)

    # Cross-env validation (framework-universal, §5.3): a reconcile for env A
    # cannot drop env B's entries. Single source of truth in the reaper module.
    from lite.gym.remote.reconcile import collect_valid_ghosts
    known = {i.id for v in state_view.by_env_id.values() for i in v}
    ghost_ids = collect_valid_ghosts(
        [g for r in reports for g in r.ghost_ids], known,
    )

    popped = 0
    skipped_in_flight: list[str] = []
    if ghost_ids:
        ghosts: list[tuple[str, _EnvSession]] = []
        async with state.lock:
            for gid in ghost_ids:
                # GET-check-POP inside the lock: never close an env mid-call (same
                # rule as ``_collect_stale``), and the check must precede the pop —
                # popping first and putting it back opens a window where a
                # concurrent DELETE sees no entry and skips its release. A deferred
                # ghost is re-offered on the next cycle once ``in_flight`` drains.
                s = state.envs.get(gid)
                if s is None:
                    continue
                if s.in_flight > 0:
                    skipped_in_flight.append(gid)
                    continue
                state.envs.pop(gid, None)
                popped += 1
                state.dead_envs_total += 1
                state.admission.release()
                ghosts.append((gid, s))
        # Close each ghost like the idle-reaper does — load-bearing for
        # shared-backend isolation: ``_close_quietly`` releases the
        # instance's conflict key (and restores the stack if it was a
        # writer), so a drifted writer doesn't wedge its key forever.
        # env.close() on an already-dead external resource is a
        # best-effort no-op (exceptions swallowed).
        if ghosts:
            await asyncio.gather(
                *(_close_quietly(state, s, gid) for gid, s in ghosts),
                return_exceptions=True,
            )

    total_orphans = sum(r.orphans_reaped for r in reports)
    if skipped_in_flight:
        logger.info(
            "drift-reaper: deferred %d ghost(s) with a call in flight: %s",
            len(skipped_in_flight), ", ".join(skipped_in_flight),
        )
    logger.info(
        "drift-reaper: envs=%d orphans=%d ghosts=%d skipped_in_flight=%d elapsed=%.2fs",
        len(state_view.by_env_id), total_orphans, popped,
        len(skipped_in_flight), time.monotonic() - t0,
    )


async def _reap_drift_dispatcher(state: State, scope: ServerScope) -> None:
    """Background task: run :func:`_reap_drift_one_cycle` every
    :func:`_drift_cycle_interval_s` (default 120 s; env var
    ``CUA_LITE_DRIFT_CYCLE_INTERVAL_S``).
    Per-cycle exceptions are caught and logged; the loop continues.
    """
    while True:
        try:
            await asyncio.sleep(_drift_cycle_interval_s())
            await _reap_drift_one_cycle(state, scope)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("drift-reaper loop error (continuing)")


# =============================================================================
# Idle reaper (background task)
# =============================================================================

def _collect_stale(
    state: State, now: float,
) -> list[tuple[str, _EnvSession, str]]:
    """Return in-use envs whose ``idle_ttl_sec`` has expired.

    Unowned (``token_hash is None``) envs are NOT exempt from TTL cleanup.

    Caller must hold ``state.lock``.
    """
    idle_cutoff = now - state.idle_ttl_sec
    stale: list[tuple[str, _EnvSession, str]] = []
    for id_, s in state.envs.items():
        if s.in_flight > 0:
            # A step/reset is awaited inside this env RIGHT NOW —
            # last_active_at is stamped at handler entry only, so a long op
            # (KVM boot-class reset) can outlive the TTL while fully live.
            # Never close an env mid-call. The drift reaper applies the SAME
            # unconditional rule to its ghosts (:func:`_reap_drift_one_cycle`).
            continue
        if s.last_active_at < idle_cutoff:
            stale.append((id_, s, "idle"))
    return stale


async def _reap_idle(state: State) -> None:
    """Background task: periodically close in-use envs whose TTL has
    expired. See :func:`_collect_stale` for the policy.
    """
    interval = max(2.0, min(60.0, state.idle_ttl_sec / 2.0))
    while True:
        try:
            await asyncio.sleep(interval)
            now = time.time()
            async with state.lock:
                stale = _collect_stale(state, now)
                for id_, _, _ in stale:
                    if state.envs.pop(id_, None) is not None:
                        state.admission.release()
            if stale:
                # Log first so the reaping is visible even if a close
                # hangs; then close all in parallel — a 128-env sweep
                # serialized would block new acquires for minutes.
                for id_, s, kind in stale:
                    age_min = (now - s.last_active_at) / 60.0
                    logger.warning(
                        "%s TTL reaped instance id=%s env_key=%s session_id=%s "
                        "age=%.1fmin", kind, id_, s.env_key, s.session_id,
                        age_min,
                    )
                await asyncio.gather(
                    *(_close_quietly(state, s, id_) for id_, s, _ in stale),
                    return_exceptions=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("reaper loop error (continuing)")


# =============================================================================
# Event-loop stall watchdog (background task)
# =============================================================================

async def _watch_loop_stalls(
    state: State,
    *,
    interval_s: float = _LOOP_STALL_INTERVAL_S,
    warn_after_s: float = _LOOP_STALL_WARN_AFTER_S,
    window_s: float = _LOOP_STALL_WINDOW_S,
) -> None:
    """Record event-loop stalls so /host_status can show loop health."""
    loop = asyncio.get_running_loop()
    expected = loop.time() + interval_s
    while True:
        try:
            await asyncio.sleep(interval_s)
            now = loop.time()
            stall_s = max(0.0, now - expected)
            observed_at = time.time()
            if stall_s >= warn_after_s:
                state.record_loop_stall(
                    stall_s,
                    observed_at=observed_at,
                    window_s=window_s,
                )
                logger.warning("event loop stalled for %.3fs", stall_s)
            else:
                state._prune_loop_stalls(now=observed_at, window_s=window_s)
            expected = now + interval_s
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("loop-stall watchdog error (continuing)")


# =============================================================================
# Background warm-singleton (--warm-singleton)
# =============================================================================
# Non-blocking pre-warm: the server listens immediately, and this background task
# boots each served SINGLETON backend so ``available`` flips true on its own (lazy
# never would — ``health()`` only probes). That is what lets a launcher *wait for hot*
# (poll GET /envs/<id> until available:true) before starting a rollout — the only way
# to make a slow backend (WA gitlab ~5-15 min) rollout-smooth, since the wait happens
# at the launcher layer, not inside a per-task reset deadline it could never survive.
# Generous deadline (covers gitlab); degrade-don't-die on missing deps.

_WARM_SINGLETON_DEADLINE_S = float(
    os.environ.get("CUA_LITE_WARM_SINGLETON_DEADLINE_S", "1800")  # 30 min — covers gitlab
)
_WARM_SINGLETON_RETRY_S = float(
    os.environ.get("CUA_LITE_WARM_SINGLETON_RETRY_S", "10")
)


async def _warm_singletons(state: "State") -> None:
    """Background-warm every served SINGLETON backend (non-blocking; runs as a
    lifespan task). Serial across envs; per-backend retry-until-ready; **degrade,
    don't die** — a backend that never warms (deps missing / deadline) is just left
    ``available: false`` (``health()`` reports it) and the others keep warming.

    Expands a bare umbrella (``browsergym``) to its registered leaves — warming the
    umbrella id boots nothing (``_benchmark_of``→None) yet would report hot."""
    from lite.gym.registry import ensure_registered, ensure_services
    from lite.gym.services import WarmStrategy, health_check, warm_strategy

    allowed = state.allowed_env_ids
    # `is not None`, not truthiness: an empty allow-list means "serve nothing" (the
    # POST path denies all), so warm nothing — don't fall through to ALL.
    served = sorted(allowed) if allowed is not None else list(gym.registry.registered_env_ids())
    leaves: list[str] = []
    seen: set[str] = set()
    for env_id in served:
        if warm_strategy(env_id) is not WarmStrategy.EAGER:
            continue
        ensure_registered(env_id)  # populate any lazy leaf catalog (cheap, no boot)
        subs = gym.registry.sub_env_ids(env_id)
        for leaf in (subs or [env_id]):
            # Don't warm a blocked env (POST /instances refuses it anyway).
            if leaf not in seen and leaf not in state.blocked_env_ids:
                seen.add(leaf)
                leaves.append(leaf)

    if not leaves:
        return
    logger.info("warm-singleton: background-warming %s", ", ".join(leaves))
    for leaf in leaves:
        await _warm_one_singleton(leaf, ensure_services, health_check)


async def _warm_one_singleton(leaf: str, ensure_services, health_check) -> None:
    """Warm one SINGLETON leaf to readiness. ``CapacityExhausted`` (the env's
    by-design "still warming" signal) → retry to the deadline; a hard failure (deps
    missing / deadline / anything else) → log + return (degrade), never raise.
    ``ensure``/``health`` are blocking, so run them off the event loop."""
    import time as _time
    deadline = _time.monotonic() + _WARM_SINGLETON_DEADLINE_S
    while True:
        try:
            await asyncio.to_thread(ensure_services, leaf)
            await asyncio.to_thread(health_check, leaf)
            logger.info("warm-singleton: %s ready", leaf)
            return
        except CapacityExhausted:
            if _time.monotonic() >= deadline:
                logger.warning(
                    "warm-singleton: %s still warming at %ss deadline → available:false"
                    " (others keep warming)", leaf, _WARM_SINGLETON_DEADLINE_S,
                )
                return
            await asyncio.sleep(_WARM_SINGLETON_RETRY_S)
        except EnvDepsMissingError as e:
            logger.warning(
                "warm-singleton: %s deps missing → available:false: %s", leaf, e,
            )
            return
        except Exception as e:  # noqa: BLE001 — degrade on ANY warm failure, never crash
            logger.warning(
                "warm-singleton: %s warm failed → available:false: %s", leaf, e,
            )
            return


# =============================================================================
# Outer retry layer — one predicate, two threaded facts
# =============================================================================
#
# ONE question is asked here: ``may_reissue(exc, replay_safe=...)``. ``/reset``
# passes ``replay_safe=True`` (a second reset re-establishes the same initial
# state, so a replay costs at most time); ``/step`` passes False (without
# server-side request-id dedupe a replay applies the same action twice while the
# trajectory records one) and therefore retries ONLY when the raiser proved the
# call never reached the worker — a connect phase that never completed.
#
# Neither fact is guessed from an exception message: ``replay_safe`` is the
# caller's, ``reached_worker`` reads the type the raiser stamped.
#
# Layered on top of the inner ``_RemoteRPC.post()`` retry (one retry for the
# env-server → container hop). The inner retry covers single transient blips;
# this outer layer covers slower-recovering storms — e.g. a drift-reaper bulk-rm
# burst that hammers daemon iptables / network-ns teardown and stalls unrelated
# containers' port forwards long enough that both inner attempts fail.
#
# 4 attempts with jittered exponential backoff (base 2 s × 2^attempt, ±50 %
# jitter to break load-storm lockstep): worst case ~14 s before the client sees
# 500, which is either
#   * a recoverable storm (bulk-rm / brief daemon stall) — recovers within
#     2-6 s, so 3-4 attempts catches it; or
#   * true container death (in-container uvicorn gone) — never recovers, and the
#     wasted ~14 s saves the rollout client's downstream ``--max-attempts``
#     budget (~3 min per fresh container spawn) for the recoverable case.
# 14 s ≪ the ``httpx`` client timeout (600 s), so this is transparent to the
# client beyond a slightly slower successful response. Not tunable: the two env
# vars that used to set these were read under the same names in ``client.py``
# for a DIFFERENT hop's loop, so "tuning" one silently moved both, and nothing
# in the repo ever set either.
_OUTER_RETRY_ATTEMPTS = 4
_OUTER_RETRY_BASE_DELAY_S = 2.0


def _retry_delay_s(attempt_0_indexed: int) -> float:
    """Exponential backoff (2/4/8 s) × a uniform [0.5, 1.5) jitter factor.

    Without the jitter, 8 instances retrying a shared docker-daemon stall wake
    up at the same wall-clock t on every attempt and the daemon never gets a
    quiet window. Mean is unchanged, so the cumulative recovery budget matches
    the un-jittered design (~14 s worst case across 4 attempts).
    """
    return _OUTER_RETRY_BASE_DELAY_S * (2 ** attempt_0_indexed) * (0.5 + random.random())


# =============================================================================
# App factory
# =============================================================================

def make_app(
    state: State,
    *,
    token: str | None = None,
    admin_token: str | None = None,
    port: int | None = None,
) -> FastAPI:
    """Build the FastAPI app bound to a :class:`State` instance.

    ``token`` controls main-API auth (see :func:`_make_bearer`):
      - ``None`` (default): **passthrough** — accept any client token,
        identity = ``sha256(client_token)[:6]``. One env-server hosts
        multiple tenants distinguished by their bearer.
      - ``<str>``: **strict** — only ``Authorization: Bearer <token>``
        is accepted; everyone else gets 401. Single-tenant production.

    ``admin_token`` controls ``/admin/*`` auth (see
    :func:`_make_admin_bearer`):
      - ``<str>``: strict — only ``Authorization: Bearer <admin_token>``
        is accepted on admin endpoints. Independent of ``token``.
      - ``None``: depends on the main mode. Passthrough → admin open
        (any caller); strict → admin disabled (404). Set ``admin_token``
        explicitly to unlock admin in strict mode.

    ``port`` is the server's listen port. It's threaded into ``gym.make``
    as ``server_port`` so each env's main.py can use it (alongside
    ``token_hash``) to scope its own external resources — naming,
    cleanup filters, drift-reaper scope — and keep two co-resident
    env-server instances on the same host mutually isolated. Server.py
    treats ``port`` as opaque; what each env does with it is owned by
    that env's main.py. Pass ``args.port`` from
    ``scripts/serve_env.py``; ``None`` for in-process / non-server
    gym.make.

    Lifespan tasks (idle-TTL reaper + polymorphic drift reaper) are
    started/cancelled via the app lifespan, and any envs still alive
    at shutdown are best-effort closed via their polymorphic
    ``env.close()``.
    """
    # This process IS the authoritative local registry. Declare it so the
    # registry's routing helper ignores the ambient ``CUA_LITE_ENV_SERVER_URL``
    # (commonly inherited from the shell/``.zshrc`` in the single-host setup) —
    # otherwise every ``gym.registry.task_ids`` / ``gym.make`` inside a request
    # handler would route the server back to ITSELF over HTTP and self-deadlock
    # the single uvicorn worker. Explicit "I am the server" boundary; the client
    # var stays in the environment, just inert here. (Module function — the
    # ``gym.registry`` name is the Registry facade instance, not this module.)
    serve_locally()

    _bearer = _make_bearer(token, state)
    _admin = _make_admin_bearer(
        admin_token, server_strict=(token is not None), state=state,
    )
    state.server_port = port
    # This server run's ownership-scoping identity. Built once
    # here — the single sha256(strict-token) site — and threaded to the
    # reconcile loop / recovery / shutdown.
    state.scope = ServerScope.from_server(server_port=port, strict_token=token)
    # Export ``CUA_LITE_ENV_SERVER_PORT`` so env modules' lifecycle
    # hooks (``ensure`` / ``reap`` / ``shutdown``) running in this same Python
    # process can identify which env-server they belong to. Used by
    # ``webgym/main.py`` to key its per-env-server state file
    # (``~/.cua-lite/webgym/server-<port>.json``).
    if port is not None:
        os.environ["CUA_LITE_ENV_SERVER_PORT"] = str(port)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Size the default thread pool to the host. POST /instances construction
        # (``await asyncio.to_thread(gym.make, ...)``), the drift reconciler, and
        # shutdown all run on the loop's default executor. The stdlib default
        # (``min(32, cpu+4)``) throttles the claim-construction burst under
        # GRPO-scale concurrency on a many-core host — 256 concurrent claims
        # queue behind 32 threads. Scale with the host, capped at 256 so we don't
        # spawn hundreds of idle threads on a 384-core box. (Per-instance docker
        # boot is async and unaffected; this only speeds the sync construction
        # burst.)
        import concurrent.futures as _cf
        _n_pool = max(32, min(os.cpu_count() or 32, 256))
        asyncio.get_running_loop().set_default_executor(
            _cf.ThreadPoolExecutor(max_workers=_n_pool, thread_name_prefix="envserver-pool")
        )
        logger.info("default thread pool sized to %d workers (host cpu=%s)", _n_pool, os.cpu_count())
        # Zombie reaper: env-server SIGKILL leaves state.envs empty next
        # time, but the external resources each env spawned (any kind —
        # docker containers, process trees, remote VMs, ...) stay
        # alive. ``recover_all`` runs the reconcile loop with an empty tracked
        # set per env-id (= boot recovery), so each env's ``reap`` reclaims the
        # orphans a prior lifetime leaked, under its own policy. Logged so an
        # operator can SEE prior-session cleanup happen (a non-zero count after
        # a crash/restart is the no-leak guarantee working, not an anomaly).
        from lite.gym.remote.recovery import recover_all
        n_recovered = recover_all(state.scope)
        logger.info(
            "boot recovery: reclaimed %d orphan resource(s) leaked by a prior "
            "env-server lifetime (server_port=%s)",
            n_recovered, state.scope.server_port,
        )
        idle_task = asyncio.create_task(_reap_idle(state), name="idle-reaper")
        # Polymorphic drift reaper — runs periodic reconciliation (orphan +
        # ghost) via the framework reconcile loop (each env's ``live_ids``/
        # ``reap``). The dispatcher body is :func:`_reap_drift_dispatcher` below.
        drift_task = asyncio.create_task(
            _reap_drift_dispatcher(state, state.scope), name="drift-reaper",
        )
        loop_watch_task = asyncio.create_task(
            _watch_loop_stalls(state), name="loop-stall-watchdog",
        )
        # --warm-singleton: background pre-warm (non-blocking). Boots served SINGLETON
        # backends so ``available`` flips true on its own → a launcher can wait-for-hot
        # before rolling out (the only smooth path for WA gitlab's ~5-15 min boot,
        # which no per-task deadline could survive). Runs after recover_all (can't be
        # reaped) + after CUA_LITE_ENV_SERVER_PORT is set (correct container scope).
        warm_task = (
            asyncio.create_task(_warm_singletons(state), name="warm-singleton")
            if state.warm_singleton_enabled else None
        )
        try:
            yield
        finally:
            for t in (idle_task, drift_task, loop_watch_task, warm_task):
                if t is None:
                    continue
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            # Shutdown contract:
            # SIGTERM does NOT drain in-flight step/reset handler coroutines —
            # the sweep below closes every env immediately, so an in-flight op
            # errors out mid-call. The RECOVERY contract is client-side: the
            # aborted call surfaces as a transient/connection error (or a
            # typed InstanceGone on the next call after restart), and the
            # client's retry → re-make path recovers the trajectory. Chosen
            # over a quiesce (stop-admitting → await in-flight with deadline)
            # because operators SIGTERM to get the port back NOW; a KVM-class
            # reset could hold a quiesce for minutes, and boot recovery +
            # the drift reaper already guarantee no resource leaks either way.
            # Best-effort close of any remaining envs on shutdown.
            # Parallel so a 128-env shutdown doesn't take 128× a single close.
            items = list(state.envs.items())
            if items:
                await asyncio.gather(
                    *(_close_quietly(state, s, id_) for id_, s in items),
                    return_exceptions=True,
                )
            state.envs.clear()
            # Drain detached close tasks (spawned by ``DELETE /envs/<id>``)
            # with a deadline so SIGTERM doesn't orphan in-flight
            # ``env.close()`` calls. Close tasks finish naturally. Bounded
            # wait — any leftover external resource is caught by the
            # next-startup ``recover_all`` boot recovery anyway, so don't
            # block shutdown forever on a hung backend.
            pending = list(state.pending_closes)
            if pending:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "lifespan shutdown: %d detached task(s) still "
                        "running after 30s — startup zombie reaper "
                        "will catch leftovers", len(pending),
                    )
            # Shared-service shutdown (mirror of per-env close above).
            # ``env.close()`` handles per-env-instance resources (each
            # env's own backend — see :mod:`lite.gym.envs`); ``shutdown_all``
            # calls each env's ``shutdown(env_id, scope)`` for SHARED resources
            # whose ownership this env-server claimed at ``ensure`` time
            # (e.g. webgym's OmniBoxes process tree). Runs after env.close drain
            # so we don't yank the underlying service out from under
            # still-closing slots. Off-thread because per-env hooks may invoke
            # blocking I/O (subprocess, RPC).
            try:
                from lite.gym.remote.recovery import shutdown_all
                await asyncio.to_thread(shutdown_all, state.scope)
            except Exception as e:
                logger.warning(
                    "lifespan shutdown: shutdown_all failed: %s — "
                    "next-startup recovery will catch leftovers", e,
                )

    app = FastAPI(title="cua-lite env server", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _request_validation_error_handler(
        request: Any, exc: RequestValidationError,
    ) -> Response:
        if not _is_instance_step_path(request.url.path):
            return await request_validation_exception_handler(request, exc)

        state.asgi_errors_total += 1
        err = ModelOutputError(_validation_error_summary(exc))
        return JSONResponse(status_code=err.http_status, content=err.to_payload())

    @app.exception_handler(CapacityExhausted)
    async def _capacity_exhausted_handler(
        request: Any, exc: CapacityExhausted,
    ) -> Response:
        # L1/L2/docker_sema self-count at their raise site; only L3
        # (env-internal) lands here uncounted.
        retry_after = max(1, int(round(exc.retry_after_s)))
        state.asgi_errors_total += 1
        if exc.layer == "env_internal":
            state.admission.record_env_internal_503()
        elif exc.layer == "conflict":
            # Shared-backend gate rejection. Hot path under same-stack
            # contention → plain increment (GIL-atomic), like the admission
            # counters. Exported at cua_lite_conflict_503_total.
            state.conflict_503_total += 1
        # DEBUG, not INFO: 503-storms during a 64-burst boot produce
        # tens to hundreds of these per second. The per-layer counters
        # exported at ``cua_lite_admission_503_total{layer="..."}`` are
        # the production observability path — this log is for
        # interactive debugging only.
        logger.debug(
            "503 CapacityExhausted on %s: %s layer=%s (retry_after=%ds)",
            getattr(request, "url", "?"), exc.what, exc.layer, retry_after,
        )
        # Body is the class's own ``to_payload`` — identical to the generic handler
        # below; this handler adds only the ``Retry-After`` header and the per-layer
        # 503 attribution above.
        return JSONResponse(
            status_code=exc.http_status,
            headers={"Retry-After": str(retry_after)},
            content={**exc.to_payload(), "detail": str(exc)},
        )

    @app.exception_handler(LiteGymError)
    async def _lite_gym_error_handler(request: Any, exc: LiteGymError) -> Response:
        # Generic handler for ANY server-raised typed LiteGymError: serialize to
        # its declared http_status + a body tagged with the class name so the
        # client reconstructs it typed (lite_error_from_payload). Without this a
        # typed error with no per-type handler would map to a terminal HTTP 500.
        # CapacityExhausted keeps its more-specific handler above (Retry-After +
        # metrics), which FastAPI prefers — this only catches the rest.
        #
        # ``detail`` is mirrored alongside the typed fields: typed clients read
        # ``error_type``/``what``, while curl, logs and the fleet router's verbatim
        # pass-through read ``detail``.
        #
        # ``error_type`` is authoritative — ``lite_error_from_payload`` reconstructs
        # the class (and with it ``retryable``) from that field alone and never reads
        # the status. ``http_status`` is presentation. So a new failure earns a
        # registered ``error_type``, not a new status plus a client branch on it.
        state.asgi_errors_total += 1
        return JSONResponse(
            status_code=exc.http_status,
            content={**exc.to_payload(), "detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_envelope(request: Any, exc: Exception) -> Response:
        # Generic envelope: ANY exception that is not a LiteGymError (the
        # envs' 42 bare RuntimeErrors, library errors, …) serializes as
        # {error_type, what} instead of FastAPI's opaque "Internal Server
        # Error" — the client wraps unknown types as RemoteEnvError with the
        # original class name. Error paths only; success payloads untouched.
        state.asgi_errors_total += 1
        logger.exception("unhandled server error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error_type": type(exc).__name__, "what": str(exc)[:2000]},
        )

    # -- session lifecycle ---------------------------------------------------

    def _blocked_env_detail(env_id: str) -> str:
        return (
            f"env_id {env_id!r} is hard-blocked on this server "
            f"(configured by this server deployment). Use direct mode "
            f"(``gym.make`` in your own process) for this env."
        )

    @app.post("/instances", status_code=201)
    async def create(
        body: _CreateBody,
        token_hash: str = Depends(_bearer),
    ) -> dict[str, Any]:
        # task_id (right of "@") may be empty for envs without per-task
        # registration. Bind-capable envs receive this task_id during cold
        # ``gym.make(env_key, ...)``.
        env_id, task_id = split_key(body.env_key)
        # The three create-time denials below (hard-blocked, not on the allow-list,
        # deps missing) are ONE condition from the caller's side — "this deployment
        # refuses this env_id, by configuration" — so they share one typed identity
        # and one status: EnvUnavailable / 501, terminal rather than retried.
        if env_id in state.blocked_env_ids:
            raise EnvUnavailable(what=_blocked_env_detail(env_id), status_code=501)
        if state.allowed_env_ids is not None and env_id not in state.allowed_env_ids:
            allowed = ", ".join(sorted(state.allowed_env_ids)) or "<empty>"
            raise EnvUnavailable(
                what=(
                    f"env_id {env_id!r} is not allowed on this server. "
                    f"Allowed env_ids: {allowed}"
                ),
                status_code=501,
            )
        # Shared-backend isolation. The conflict gate is a
        # SERVER-MODE-ONLY contract, so a task declares it via opt-in keys in its
        # registered ``metadata.others`` (not typed core fields — direct-mode
        # gym.make never reads them):
        #   others["conflict_keys"]: tuple[str, ...] — opaque labels for shared
        #       resources this task touches. Two live instances sharing a key
        #       can't run concurrently if either is a writer; same key, both
        #       readers → no conflict. Absent/empty → no constraint.
        #   others["mutating"]: bool — True iff the task may WRITE a shared
        #       resource → the gate gives it the key exclusively (readers share).
        # Read from REGISTERED metadata — cheap, no env construction. One
        # instance id is allocated before construction, so the key acquired
        # below ties to the stored env and is released by that id on every
        # exit path.
        try:
            meta = gym.registry.task_metadata(env_id, task_id)
        except EnvDepsMissingError as e:
            # task_metadata now fires lazy registration (import + register_tasks),
            # which raises here when this env's deps aren't installed on the server.
            # Surface 501 EnvUnavailable (symmetric with the gym.make path below +
            # GET /envs/{id}/tasks), not a bare 500. No-op for deps-present envs.
            raise e.as_env_unavailable(env_id) from e
        others = meta.others if meta is not None else {}
        conflict_keys = tuple(others.get(OTHERS_CONFLICT_KEYS) or ())
        mutating = bool(others.get(OTHERS_MUTATING, False))
        inst_id = uuid.uuid4().hex
        async with state.lock:
            # Conflict gate FIRST, atomically with admission's lock (no
            # check-then-act race): a writer takes its keys exclusively, a
            # reader shares; conflict → 503 + Retry-After, which the client
            # already retries (zero client change). The key is held until
            # this instance's close+restore (reset-in-close) and
            # released on every failure path below. No-op when conflict_keys
            # is empty (every env that hasn't opted in → zero behavior change).
            if not state.conflict.try_acquire(inst_id, conflict_keys, mutating):
                raise CapacityExhausted(
                    what=(f"shared backend busy: conflict_keys="
                          f"{list(conflict_keys)} mutating={mutating}"),
                    retry_after_s=_CONFLICT_RETRY_AFTER_S,
                    layer="conflict",
                )
            # Every create cold-constructs the caller's real env_key via
            # gym.make; no parked Env object is rebound under state.lock.
            try:
                state.admission.require_capacity()
            except Exception:
                state.conflict.release(inst_id)
                raise
            state.admission.admit()
        # Lock released; gym.make runs unlocked. release() pairs with
        # admit() on insertion failure (finally below), DELETE, or idle TTL.
        env_inserted = False
        try:
            if state.boot_jitter_sec > 0.0:
                await asyncio.sleep(random.uniform(0.0, state.boot_jitter_sec))
            try:
                # session_id + token_hash + server_port are passed
                # explicitly so each env scopes its own external resources
                # rather than racing through process-global state.
                env = await asyncio.to_thread(
                    gym.make,
                    body.env_key,
                    session_id=body.session_id,
                    token_hash=token_hash,
                    server_port=state.server_port,
                    **body.env_kwargs,
                )
            except EnvDepsMissingError as e:
                logger.warning(
                    "env_id=%s deps missing on this server: %s",
                    env_id, e.what,
                )
                raise e.as_env_unavailable(env_id) from e
            except CapacityExhausted:
                # Transient — e.g. an env's ``ensure_services`` reports its
                # backend is still warming up (browsergym Magento cold boot),
                # or any L3 ``CapacityExhausted`` raised during construction.
                # Re-raise so the global CapacityExhausted handler maps it to
                # 503 + Retry-After and the client retries; the generic 400
                # below is TERMINAL and would fail the task on cold boot.
                raise
            except Exception as e:
                logger.exception("gym.make failed for %s", body.env_key)
                # Typed envelope: 400 with {error_type, what} instead of
                # a bare detail string.
                raise MakeFailed(what=f"gym.make failed: {type(e).__name__}: {e}") from e

            async with state.lock:
                id_ = inst_id  # unified with the gate-acquired conflict key
                now = time.time()
                token_info = state.token_registry.get(token_hash)
                token = token_info.token if token_info is not None else None
                state.envs[id_] = _EnvSession(
                    env=env,
                    env_key=body.env_key,
                    env_id=env_id,
                    session_id=body.session_id,
                    token_hash=token_hash,
                    token=token,
                    created_at=now,
                    last_active_at=now,
                    env_kwargs=dict(body.env_kwargs),
                )
                if token_info is not None:
                    token_info.instances_created_total += 1
                env_inserted = True

            logger.info(
                "created instance id=%s env_key=%s session_id=%s token=%s token_hash=%s",
                id_, body.env_key, body.session_id, token, token_hash,
            )
            return {
                "id": id_,
                "metadata": env.metadata.to_dict() if env.metadata else None,
            }
        finally:
            if not env_inserted:
                state.admission.release()
                # Release the conflict key acquired at the gate — the env was
                # never inserted (gym.make/insert failed), so the episode
                # never ran (no restore needed). Brief lock for the dict-flip.
                async with state.lock:
                    state.conflict.release(inst_id)

    @app.post("/instances/{id_}/reset")
    async def reset(id_: str, token_hash: str = Depends(_bearer)) -> Response:
        s = _require_own_env(state, id_, token_hash)
        s.last_active_at = time.time()  # for the idle-reaper
        if state.reset_jitter_sec > 0.0:
            await asyncio.sleep(random.uniform(0.0, state.reset_jitter_sec))
        # Strict concurrency cap on env.reset(), **per env_id**. Jitter
        # above smooths sub-second start-time burst; this sema bounds
        # steady-state parallelism so e.g. 128 simultaneous resets
        # don't all do a snapshot restore at once (NVMe IOPS spike →
        # reset timeouts). Per-env_id gating: a saturated lite.osworld
        # reset queue can't block a concurrent androidworld rollout.
        # Retries are kept INSIDE the held slot — under a saturated
        # system, the retry's sleep + re-attempt natively back-pressure
        # against the same cap rather than re-queuing.
        async with state.reset_sema_for(s.env_id):
            state.reset_active[s.env_id] = state.reset_active.get(s.env_id, 0) + 1
            s.in_flight += 1   # reap-ineligible while the op is awaited
            try:
                return await _reset_with_retries(state, id_, s)
            except EnvDepsMissingError as e:
                # Convert HERE, where env_id is still known: the terminal branch
                # inside _reset_with_retries pops the session before re-raising,
                # so by the time the app-level handler sees this there is no
                # id→env_id mapping left to name the env in the 501 message.
                # (That handler stays as the never-a-500 net for other paths.)
                raise e.as_env_unavailable(s.env_id) from e
            finally:
                s.in_flight -= 1
                n = state.reset_active.get(s.env_id, 1) - 1
                if n <= 0:
                    state.reset_active.pop(s.env_id, None)
                else:
                    state.reset_active[s.env_id] = n

    async def _reset_with_retries(
        state: "State", id_: str, s: _EnvSession,
    ) -> dict[str, Any]:
        """Run ``env.reset()`` with the outer-retry layer. Split out so
        the /reset handler reads as sema-gate + counter + body. Holds
        the reset_sema slot for the duration including retry sleeps.

        ``replay_safe=True`` is not a parameter but a property of this function:
        it wraps ``env.reset()`` and nothing else, and reset is the one env call
        a replay cannot corrupt. ``/step`` has no counterpart."""
        retried_any = False
        for attempt in range(_OUTER_RETRY_ATTEMPTS):
            t0 = time.monotonic()
            try:
                observation = await s.env.reset()
                state.record_reset(s.env_id, time.monotonic() - t0, ok=True)
                s.reset_done = True  # /step is now allowed (see step handler guard)
                if retried_any:
                    state.outer_retry_recovered["reset"] += 1
                return _reset_observation_to_response(observation)
            except CapacityExhausted:
                # Retriable (e.g. browsergym backend still cold-booting — gitlab
                # / reddit / classifieds — surfaced by the env's reset). Re-raise
                # IMMEDIATELY, BEFORE the terminal reclaim below: the global
                # CapacityExhausted handler returns 503 + Retry-After and the
                # client retries /reset on the SAME instance id, so the instance
                # MUST stay alive (do not pop/close it). On the retry, the env's
                # lazy _create_and_reset runs again — once the backend is
                # HTTP-ready the reset succeeds.
                state.record_reset(s.env_id, time.monotonic() - t0, ok=False)
                raise
            except Exception as e:
                if attempt + 1 < _OUTER_RETRY_ATTEMPTS and may_reissue(
                    e, replay_safe=True,
                ):
                    delay = _retry_delay_s(attempt)
                    state.outer_retry_fired["reset"] += 1
                    retried_any = True
                    logger.warning(
                        "/reset transient %s on env %s; retry %d/%d after %.1fs",
                        type(e).__name__, id_, attempt + 1, _OUTER_RETRY_ATTEMPTS, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                state.record_reset(s.env_id, time.monotonic() - t0, ok=False)
                state.asgi_errors_total += 1
                if retried_any:
                    state.outer_retry_exhausted["reset"] += 1
                # Terminal reset failure: reclaim the admission slot + the
                # state.envs entry NOW instead of leaving it pinned until the
                # idle-TTL reaper fires. The external resource (container /
                # emulator) was already force-removed by the env's own
                # failed-attempt cleanup; this only drops the server-side
                # bookkeeping so a client that abandons the instance without a
                # DELETE (crash / request timeout) can't hold capacity for
                # idle_ttl_sec. Mirrors the DELETE handler: the pop()-guarded
                # release is race-safe against a concurrent client DELETE
                # (whichever pops first releases exactly once), and the close
                # is detached so the error response still returns promptly.
                async with state.lock:
                    if state.envs.pop(id_, None) is not None:
                        state.admission.release()
                bg_task = asyncio.create_task(_close_quietly(state, s, id_))
                state.pending_closes.add(bg_task)
                bg_task.add_done_callback(state.pending_closes.discard)
                raise
        # Type-checker fence: the for-loop always either returns or
        # re-raises, but Python doesn't propagate that. Keep the raise
        # so the function has a known terminal value on every code path.
        raise RuntimeError("reset retry loop exhausted")

    @app.post("/instances/{id_}/step")
    async def step(
        id_: str,
        body: _StepBody,
        token_hash: str = Depends(_bearer),
    ) -> Response:
        s = _require_own_env(state, id_, token_hash)
        actions = _validate_remote_step_actions(body.actions)
        # Client state-machine guard: /step before /reset is invalid (the env's
        # backend isn't booted yet). Return 409 instead of letting env.step()
        # touch un-booted state and surface as a 500 traceback.
        if not s.reset_done:
            # Typed body: ProtocolMisuse is TERMINAL — the old naked 409
            # fell into the client's unknown→retry default and was retried
            # pointlessly.
            raise ProtocolMisuse(
                what="step called before reset; call POST /instances/{id}/reset first",
            )
        s.last_active_at = time.time()
        s.n_steps += 1
        retried_any = False
        s.in_flight += 1   # reap-ineligible while the op is awaited
        try:
            for attempt in range(_OUTER_RETRY_ATTEMPTS):
                t0 = time.monotonic()
                try:
                    # no active-op throttle: step is sub-second on every
                    # env we ship (no snapshot-restore-class work), so
                    # gating it adds latency without preventing any
                    # observed failure mode. /reset is gated because
                    # snapshot restore is heavy IO.
                    result = await s.env.step(actions)
                    state.record_step(s.env_id, time.monotonic() - t0, ok=True)
                    if retried_any:
                        state.outer_retry_recovered["step"] += 1
                    return _step_result_to_response(result)
                except Exception as e:
                    state.record_step(s.env_id, time.monotonic() - t0, ok=False)
                    # ``replay_safe=False``: a step applies an action the
                    # trajectory records once, so the ONLY thing that can license
                    # a re-issue is the raiser proving nothing ran — a connect
                    # phase that never completed. A cut response, or any
                    # un-answered transport failure, is sent exactly once.
                    if attempt + 1 < _OUTER_RETRY_ATTEMPTS and may_reissue(
                        e, replay_safe=False,
                    ):
                        delay = _retry_delay_s(attempt)
                        state.outer_retry_fired["step"] += 1
                        retried_any = True
                        logger.warning(
                            "/step never-reached %s on env %s; retry %d/%d after %.1fs",
                            type(e).__name__, id_, attempt + 1, _OUTER_RETRY_ATTEMPTS, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    state.asgi_errors_total += 1
                    if retried_any:
                        state.outer_retry_exhausted["step"] += 1
                    raise
        # No EnvDepsMissingError arm here, unlike /reset: the deps probe reachable
        # from a step frame already ran identically in ``reset()``.
        finally:
            s.in_flight -= 1
        # Type-checker fence — see /reset's mirror docstring.
        raise RuntimeError("step retry loop exhausted")

    @app.delete("/instances/{id_}")
    async def close(id_: str, token_hash: str = Depends(_bearer)) -> dict[str, Any]:
        # Always destroy on DELETE.
        # Two-stage close keeps iter-boundary close-storms off the
        # client's critical path: pop under the lock (L2 slot
        # released atomically), spawn ``_close_quietly`` as a detached
        # task so the response returns in <1 ms while the env's
        # polymorphic ``env.close()`` (~1-3 s typical, longer for
        # KVM-class backends) runs in the background.
        #
        # Response shape: ``{ok: True}``. ``ok=True`` is set in every
        # success path including the idempotent "already gone" case.
        async with state.lock:
            s = state.envs.get(id_)
            if s is None:
                return {"ok": True}  # idempotent
            if s.token_hash != token_hash:
                # Same typed 403 as _require_own_env (this handler can't reuse it:
                # it must not raise InstanceGone for an unknown id — DELETE is
                # idempotent and answers {ok: True} above).
                raise EnvAccessDenied(
                    what=f"instance id={id_!r} owned by another token",
                )
            if state.envs.pop(id_, None) is not None:
                state.admission.release()
        bg_task = asyncio.create_task(_close_quietly(state, s, id_))
        state.pending_closes.add(bg_task)
        bg_task.add_done_callback(state.pending_closes.discard)
        logger.info("closing instance id=%s (detached)", id_)
        return {"ok": True}

    # -- introspection -------------------------------------------------------

    @app.get("/instances")
    async def list_instances(
        session_id: str | None = None,
        env_key: str | None = None,
        env_id: str | None = None,
        token_hash: str = Depends(_bearer),
    ) -> dict[str, Any]:
        # Scope to caller's own token. Listing across tokens would be an
        # admin-only feature; defer to future work.
        # Snapshot under the lock so iteration is safe against concurrent
        # POST /instances / DELETE / bulk-cleanup mutations. Filtering +
        # formatting run outside — keeps the critical section sub-ms.
        # Response wrapped in {"instances": [...]} so a future ``next_cursor``
        # field can be added without a breaking change.
        async with state.lock:
            snapshot = list(state.envs.items())
        out = []
        for id_, s in snapshot:
            if s.token_hash != token_hash:
                continue
            if env_key is not None and s.env_key != env_key:
                continue
            if env_id is not None and s.env_id != env_id:
                continue
            if session_id is not None and s.session_id != session_id:
                continue
            out.append(_session_summary(id_, s))
        return {"instances": out}

    @app.get("/instances/{id_}")
    async def get_instance(
        id_: str,
        token_hash: str = Depends(_bearer),
    ) -> dict[str, Any]:
        """Single-instance state lookup. 404 if unknown / never existed
        under this token; 403 if owned by another token.

        Mirrors one entry of ``GET /instances`` — operators reach for
        this when they have an id from a log line and want to know the
        env_key / session_id / age without paginating the full list.
        """
        s = _require_own_env(state, id_, token_hash)
        return _session_summary(id_, s)

    # -- admin views ---------------------------------------------------------
    # Cross-token introspection for cluster operators. All endpoints
    # share auth (:func:`_make_admin_bearer`) and take one atomic
    # :class:`_AdminSnapshot` so concurrent admin scrapes give a
    # consistent picture. Aggregation lives in module-level helpers —
    # these handlers stay thin (snapshot → aggregate → filter → return).
    #
    # Four endpoints, one purpose each, no overlap:
    #   /admin/budget     global admission pressure (in_flight / cap / 503 by layer)
    #   /admin/tokens     per-token history + current instances_active
    #   /admin/usage      per (token, session, env_id) active-instance grain
    #   /admin/instances  raw instance rows w/ ID + token
    #
    # ``/admin/usage`` and ``/admin/instances`` share the same filter
    # vocabulary (``token`` / ``session_id`` / ``env_id``) so admin
    # tooling can use one query string against either.

    @app.get("/admin/budget", dependencies=[Depends(_admin)])
    async def admin_budget() -> dict[str, Any]:
        """Cluster-level admission pressure snapshot.

        Single call answers "how loaded is the server" — in_flight vs
        max_live_envs, host RAM/load readings, and 503 counts per
        admission layer. See :func:`_budget_snapshot`.
        """
        snap = await _snapshot_state_for_admin(state)
        return _budget_snapshot(snap)

    @app.get("/admin/tokens", dependencies=[Depends(_admin)])
    async def admin_tokens() -> dict[str, Any]:
        """Every token the server has ever seen, with current
        ``instances_active`` count.

        Answers both "who has touched this server" and "who holds how
        many live envs right now". See :func:`_aggregate_tokens`.
        """
        snap = await _snapshot_state_for_admin(state)
        return {"tokens": _aggregate_tokens(snap)}

    @app.get("/admin/usage", dependencies=[Depends(_admin)])
    async def admin_usage(
        token: str | None = None,
        session_id: str | None = None,
        env_id: str | None = None,
    ) -> dict[str, Any]:
        """Per ``(token, session, env_id)`` resource breakdown.

        Finest-grain "who is using how much, right now" table. The three
        filters drill into the same view:
          * ``?token=alice`` — alice's load across sessions / envs
          * ``?env_id=lite.osworld`` — everyone running lite.osworld
          * ``?session_id=alice-A`` — single session breakdown
        Filters compose (logical AND). Sorted by ``n_active_instances`` desc.
        """
        snap = await _snapshot_state_for_admin(state)
        rows = _aggregate_usage(snap)
        if token is not None:
            rows = [r for r in rows if r["token"] == token]
        if session_id is not None:
            rows = [r for r in rows if r["session_id"] == session_id]
        if env_id is not None:
            rows = [r for r in rows if r["env_id"] == env_id]
        return {"usage": rows}

    @app.get("/admin/instances", dependencies=[Depends(_admin)])
    async def admin_instances(
        token: str | None = None,
        session_id: str | None = None,
        env_id: str | None = None,
        env_key: str | None = None,
    ) -> dict[str, Any]:
        """Cross-token live-instance dump (one row per instance, w/ ID).

        Same row shape as ``GET /instances`` (self-scoped) plus the raw
        ``token``. Filters share vocabulary with ``/admin/usage`` —
        admins typically use ``/admin/usage`` first for aggregates,
        then drill into ``/admin/instances`` to get actual IDs.
        """
        snap = await _snapshot_state_for_admin(state)
        rows = []
        for id_, s in snap.envs:
            if token is not None and s.token != token:
                continue
            if session_id is not None and s.session_id != session_id:
                continue
            if env_id is not None and s.env_id != env_id:
                continue
            if env_key is not None and s.env_key != env_key:
                continue
            rows.append(_admin_session_summary(id_, s))
        return {"instances": rows}

    @app.get("/metrics", dependencies=[Depends(_bearer)])
    async def metrics() -> Response:
        """Prometheus text-format metrics dump.

        Snapshot of admission + per-env_id counts. Mostly gauges
        sampled from in-memory state. Operator-tunable values
        (``max_live_envs``, ``idle_ttl_sec``) are emitted alongside so
        a single scrape captures both observed state and config.
        """
        # Snapshot under the lock; format outside.
        async with state.lock:
            sessions = list(state.envs.values())
            total = len(state.envs)
            reset_active_by_env: dict[str, int] = dict(state.reset_active)
        by_env: dict[str, int] = {}
        reset_cap = state.reset_concurrency
        for s in sessions:
            by_env[s.env_id] = by_env.get(s.env_id, 0) + 1
        lines: list[str] = [
            "# HELP cua_lite_envs_total Total live envs admitted",
            "# TYPE cua_lite_envs_total gauge",
            f"cua_lite_envs_total {total}",
            "# HELP cua_lite_envs_by_id Live envs by env_id",
            "# TYPE cua_lite_envs_by_id gauge",
        ]
        for env_id_, n in sorted(by_env.items()):
            lines.append(f'cua_lite_envs_by_id{{env_id="{env_id_}"}} {n}')
        lines += [
            "# HELP cua_lite_reset_active env.reset() calls past the per-env_id reset_sema gate (currently running, incl. retries)",
            "# TYPE cua_lite_reset_active gauge",
            *(
                f'cua_lite_reset_active{{env_id="{eid}"}} {n}'
                for eid, n in sorted(reset_active_by_env.items())
            ),
            "# HELP cua_lite_reset_concurrency Per-env_id cap on concurrent env.reset() (reset_sema capacity)",
            "# TYPE cua_lite_reset_concurrency gauge",
            f"cua_lite_reset_concurrency {reset_cap}",
        ]
        adm_snap = state.admission.snapshot()
        lines += [
            "# HELP cua_lite_admission_max_live_envs L2 cap on concurrent live envs",
            "# TYPE cua_lite_admission_max_live_envs gauge",
            f'cua_lite_admission_max_live_envs {adm_snap["max_live_envs"]}',
            "# HELP cua_lite_admission_in_flight Current live env count",
            "# TYPE cua_lite_admission_in_flight gauge",
            f'cua_lite_admission_in_flight {adm_snap["in_flight"]}',
            "# HELP cua_lite_admission_503_total 503 responses by layer",
            "# TYPE cua_lite_admission_503_total counter",
            *(
                f'cua_lite_admission_503_total{{layer="{layer}"}} '
                f'{adm_snap[f"{layer}_503_total"]}'
                for layer in ("emergency", "capacity", "docker_sema", "env_internal")
            ),
            "# HELP cua_lite_admission_docker_in_flight Docker daemon serialization slots in use",
            "# TYPE cua_lite_admission_docker_in_flight gauge",
            f'cua_lite_admission_docker_in_flight {adm_snap["docker_in_flight"]}',
            "# HELP cua_lite_admission_docker_sema_pending Callers currently blocked in the docker-create acquire (queue depth)",
            "# TYPE cua_lite_admission_docker_sema_pending gauge",
            f'cua_lite_admission_docker_sema_pending {adm_snap["docker_sema_pending"]}',
            "# HELP cua_lite_admission_docker_create_p50_s EMA of observed docker-create duration; drives adaptive Retry-After",
            "# TYPE cua_lite_admission_docker_create_p50_s gauge",
            f'cua_lite_admission_docker_create_p50_s {adm_snap["docker_create_p50_s"]:.3f}',
            "# HELP cua_lite_host_ram_percent Host RAM usage (psutil)",
            "# TYPE cua_lite_host_ram_percent gauge",
            f'cua_lite_host_ram_percent {adm_snap["host_ram_percent"]:.1f}',
            "# HELP cua_lite_host_ram_free_bytes Host RAM available (psutil)",
            "# TYPE cua_lite_host_ram_free_bytes gauge",
            f'cua_lite_host_ram_free_bytes {adm_snap["host_ram_free_bytes"]}',
            "# HELP cua_lite_host_swap_percent Host swap usage (psutil)",
            "# TYPE cua_lite_host_swap_percent gauge",
            f'cua_lite_host_swap_percent {adm_snap["host_swap_percent"]:.1f}',
            "# HELP cua_lite_host_disk_free_bytes Host root-fs free bytes (psutil)",
            "# TYPE cua_lite_host_disk_free_bytes gauge",
            f'cua_lite_host_disk_free_bytes {adm_snap["host_disk_free_bytes"]}',
            "# HELP cua_lite_host_load_per_cpu 1-min loadavg divided by cpu_count",
            "# TYPE cua_lite_host_load_per_cpu gauge",
            f'cua_lite_host_load_per_cpu {adm_snap["host_load_per_cpu"]:.3f}',
            "# HELP cua_lite_idle_ttl_sec In-use TTL reaper threshold",
            "# TYPE cua_lite_idle_ttl_sec gauge",
            f"cua_lite_idle_ttl_sec {state.idle_ttl_sec}",
        ]
        # Counters — monotonically increase over server lifetime.
        # Prometheus convention: counter names end in _total.
        lines += [
            "# HELP cua_lite_asgi_errors_total Sum of step+reset 5xx errors (request-handler exceptions re-raised to ASGI)",
            "# TYPE cua_lite_asgi_errors_total counter",
            f"cua_lite_asgi_errors_total {state.asgi_errors_total}",
            "# HELP cua_lite_outer_retry_fired_total Outer transport-error retry attempts triggered (labeled by endpoint)",
            "# TYPE cua_lite_outer_retry_fired_total counter",
            f'cua_lite_outer_retry_fired_total{{endpoint="step"}} {state.outer_retry_fired["step"]}',
            f'cua_lite_outer_retry_fired_total{{endpoint="reset"}} {state.outer_retry_fired["reset"]}',
            "# HELP cua_lite_outer_retry_recovered_total Requests that recovered via outer-retry (labeled by endpoint)",
            "# TYPE cua_lite_outer_retry_recovered_total counter",
            f'cua_lite_outer_retry_recovered_total{{endpoint="step"}} {state.outer_retry_recovered["step"]}',
            f'cua_lite_outer_retry_recovered_total{{endpoint="reset"}} {state.outer_retry_recovered["reset"]}',
            "# HELP cua_lite_outer_retry_exhausted_total Requests where outer-retry exhausted all attempts and returned 500 (labeled by endpoint)",
            "# TYPE cua_lite_outer_retry_exhausted_total counter",
            f'cua_lite_outer_retry_exhausted_total{{endpoint="step"}} {state.outer_retry_exhausted["step"]}',
            f'cua_lite_outer_retry_exhausted_total{{endpoint="reset"}} {state.outer_retry_exhausted["reset"]}',
            "# HELP cua_lite_envs_dead_total state.envs entries dropped by the drift reaper (external resource vanished out-of-band)",
            "# TYPE cua_lite_envs_dead_total counter",
            f"cua_lite_envs_dead_total {state.dead_envs_total}",
            "# HELP cua_lite_conflict_restore_fail_total writer closes whose shared-backend restore failed (conflict key kept held; stack may be wedged)",
            "# TYPE cua_lite_conflict_restore_fail_total counter",
            f"cua_lite_conflict_restore_fail_total {state.restore_fail_total}",
            "# HELP cua_lite_conflict_503_total shared-backend conflict-gate 503s (claims rejected; client retries via Retry-After)",
            "# TYPE cua_lite_conflict_503_total counter",
            f"cua_lite_conflict_503_total {state.conflict_503_total}",
            "# HELP cua_lite_step_total Total step() calls per env_id (success + fail)",
            "# TYPE cua_lite_step_total counter",
        ]
        for env_id_ in sorted(state.step_total):
            lines.append(
                f'cua_lite_step_total{{env_id="{env_id_}"}} '
                f'{state.step_total[env_id_]}'
            )
        lines += [
            "# HELP cua_lite_step_5xx_total Failed step() calls per env_id",
            "# TYPE cua_lite_step_5xx_total counter",
        ]
        for env_id_ in sorted(state.step_5xx_total):
            lines.append(
                f'cua_lite_step_5xx_total{{env_id="{env_id_}"}} '
                f'{state.step_5xx_total[env_id_]}'
            )
        lines += [
            "# HELP cua_lite_reset_total Total reset() calls per env_id (success + fail)",
            "# TYPE cua_lite_reset_total counter",
        ]
        for env_id_ in sorted(state.reset_total):
            lines.append(
                f'cua_lite_reset_total{{env_id="{env_id_}"}} '
                f'{state.reset_total[env_id_]}'
            )
        # Step + reset latency histograms per env_id (cumulative buckets
        # per Prometheus convention). Each bucket count includes
        # everything in lower buckets.
        _emit_histogram(
            lines, "cua_lite_step_duration_seconds", "Per-step latency",
            state.step_duration_buckets, state.step_duration_sum,
            State._STEP_DURATION_BUCKETS,
        )
        _emit_histogram(
            lines, "cua_lite_reset_duration_seconds", "Per-reset latency",
            state.reset_duration_buckets, state.reset_duration_sum,
            State._RESET_DURATION_BUCKETS,
        )
        return Response("\n".join(lines) + "\n",
                        media_type="text/plain; version=0.0.4")

    @app.get("/host_status", dependencies=[Depends(_bearer)])
    async def host_status() -> dict[str, Any]:
        """Host-level resource snapshot (CPU / memory / load / disk-IO /
        per-process for the env-server itself).

        Counterpart to :meth:`metrics`: ``/metrics`` reports app-level
        env-server state (envs, throttles, retry counters); this
        reports the OS-level resource the server is running on. Useful
        for one-shot diagnostic curls and for monitoring scripts that
        want a single endpoint instead of shelling out to ``uptime`` /
        ``free`` / ``iostat`` / ``ps`` separately.

        Returns a JSON object with stable keys so downstream tooling can
        parse it. All sizes in bytes unless suffixed (``_gb`` =
        gibibytes for human readability).

        Bearer-gated like ``/metrics``; host
        resource info is not sensitive in our deployment model.
        """
        vm = psutil.virtual_memory()
        load1, load5, load15 = os.getloadavg()
        # cpu_percent(interval=None) returns since-last-call %, so the
        # first call after import returns 0.0; subsequent calls give
        # the percent since the previous call. Acceptable for polled
        # monitoring; operators expect a small skew on the very first
        # hit after server boot.
        cpu_pct = psutil.cpu_percent(interval=None)
        try:
            disk_io = psutil.disk_io_counters()
        except Exception:
            disk_io = None
        try:
            net_io = psutil.net_io_counters()
        except Exception:
            net_io = None
        proc = psutil.Process(os.getpid())
        with proc.oneshot():
            proc_rss = proc.memory_info().rss
            proc_cpu = proc.cpu_percent(interval=None)
            try:
                num_fds = proc.num_fds()
            except Exception:
                num_fds = None
            num_threads = proc.num_threads()
            proc_create_time = proc.create_time()
        return {
            "uptime_seconds": time.time() - proc_create_time,
            "cua_lite": _CUA_LITE_BUILD_INFO,
            "wire": {
                "frame_magic": FRAME_MAGIC,
                "frame_version": FRAME_VERSION,
            },
            "loop": state.loop_stall_snapshot(),
            "cpu": {
                "count_logical": psutil.cpu_count(logical=True),
                "count_physical": psutil.cpu_count(logical=False),
                "percent": cpu_pct,
                "load1": load1,
                "load5": load5,
                "load15": load15,
            },
            "memory": {
                "total_gb": vm.total / (1 << 30),
                "used_gb": vm.used / (1 << 30),
                "available_gb": vm.available / (1 << 30),
                "free_gb": vm.free / (1 << 30),
                "percent": vm.percent,
            },
            "disk_io": None if disk_io is None else {
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count,
            },
            "net_io": None if net_io is None else {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
            },
            "process": {
                "pid": os.getpid(),
                "rss_gb": proc_rss / (1 << 30),
                "cpu_percent": proc_cpu,
                "num_fds": num_fds,
                "num_threads": num_threads,
            },
        }

    # -- env catalog (types) -------------------------------------------------

    def _env_visible(env_id: str) -> bool:
        """Whether this env_id is in scope to list / probe on this server.

        ``allowed_env_ids`` restricts what callers can ``POST /instances``
        for; restrict listing too so admin-controlled deny-listing is
        consistent across endpoints. ``blocked_env_ids`` is NOT a
        visibility gate here — blocked envs are returned by /envs and
        /envs/{env_id} with ``available: false`` so clients see them as
        known-but-unusable rather than missing.
        """
        allowed = state.allowed_env_ids
        return allowed is None or env_id in allowed

    def _env_metadata(env_id: str) -> dict[str, Any]:
        """Build the single-env metadata payload used by both
        ``GET /envs/{env_id}`` and ``GET /envs?expand=metadata``.

        Returns ``{available, n_tasks, splits}`` for healthy envs,
        ``{available: False, error}`` for blocked / import-broken ones.
        Never raises — the error path lives in the response body so a
        single bad env doesn't poison the whole map endpoint.
        """
        if env_id in state.blocked_env_ids:
            return {
                "available": False,
                "error": (
                    "blocked on this server by deployment config; "
                    "use direct mode for this env"
                ),
            }
        try:
            splits = gym.registry.task_ids(env_id)
            assert isinstance(splits, dict)
            n_tasks = sum(len(ids) for ids in splits.values())
            # Uniform runtime probe: envs whose services object implements
            # :meth:`lite.gym.services.EnvServices.health` get a chance
            # to surface runtime-only dep gaps (asset missing, backend
            # service down) as ``available: false`` symmetrically with
            # import-time failures (which already block registration, so
            # the env never reaches here). No-op for the default impl.
            from lite.gym.services import health_check as _health_check
            _health_check(env_id)
            return {
                "available": True,
                "n_tasks": n_tasks,
                "splits": sorted(splits.keys()),
            }
        except Exception as e:
            # Truncate to keep response small; full traceback is in
            # the server log if needed.
            msg = str(e).strip()
            if len(msg) > 500:
                msg = msg[:500] + "... (truncated)"
            return {"available": False, "error": msg or type(e).__name__}

    @app.get("/envs", dependencies=[Depends(_bearer)])
    async def list_env_types(
        expand: str | None = None,
    ) -> Any:
        """List env_ids registered on this server.

        Two response shapes via ``?expand=``:

          * Default → bare list ``["androidworld", "androidlab", ...]``.
            Cheap discovery: no per-env ``task_ids()`` cost, just the
            registry walk.

          * ``?expand=metadata`` → map of
            ``{env_id: {available, n_tasks, splits, error?}}``.
            One-shot pre-flight: replaces the deprecated
            ``/envs/available`` of the old API.

        Walks ``registry.registered_env_ids()`` — the union of directory-
        scanned envs AND programmatically-registered sub-envs (e.g.
        ``browsergym.miniwob``, which has no own dir but is registered
        when ``browsergym/main.py`` imports).

        A scoped server skips that walk: ``--env-ids`` already names the
        answer, so there is nothing to discover and no reason to fire every
        other env's lazy registration (webgym alone costs ~24 s and a
        HuggingFace download). Same precedence as ``_warm_singletons`` —
        ``is not None``, not truthiness, so an empty allow-list serves nothing
        instead of falling through to ALL.
        """
        allowed = state.allowed_env_ids
        if allowed is not None:
            visible = sorted(
                eid for eid in allowed if gym.registry.is_known_env_id(eid)
            )
        else:
            visible = [
                eid for eid in gym.registry.registered_env_ids()
                if _env_visible(eid)
            ]
        if expand == "metadata":
            # _env_metadata is blocking (task_ids → first-call _load_tasks; health()
            # → urllib site probes up to ~15s while warming). Offload each off the
            # event loop so a readiness poll can't stall every concurrent rollout's
            # step/reset. Sequential await is enough — the loop stays free.
            return {eid: await asyncio.to_thread(_env_metadata, eid) for eid in visible}
        if expand is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unsupported expand={expand!r}; "
                    "supported: 'metadata' or unset"
                ),
            )
        return visible

    @app.get("/envs/{env_id}", dependencies=[Depends(_bearer)])
    async def get_env_type(env_id: str) -> dict[str, Any]:
        """Single-type metadata. 404 if env_id isn't registered on this
        server; 200 with ``available: false`` + ``error`` if it's
        registered but import-broken or blocked.

        Cheap pre-flight before starting a long rollout — equivalent to
        ``GET /envs?expand=metadata`` filtered to one env_id, without the
        full registry walk.

        ``is_known_env_id`` is what keeps that last clause true. The obvious
        spelling (``env_id in registered_env_ids()``) fires ``ensure_registered``
        for EVERY env, so probing one id paid webgym's HuggingFace load (~24 s)
        and browsergym's (~21 s) — a rollout launcher's pre-flight blocked for
        minutes on catalogs it never asked about.
        """
        if not _env_visible(env_id) or not gym.registry.is_known_env_id(env_id):
            raise HTTPException(
                status_code=404,
                detail=f"env_id={env_id!r} not registered on this server",
            )
        # Offload: _env_metadata blocks on task_ids (first-call _load_tasks) + health()
        # urllib probes (up to ~15s while warming) — must not run on the event loop, or a
        # wait-for-hot poll stalls every concurrent rollout. (See list_env_types above.)
        return await asyncio.to_thread(_env_metadata, env_id)

    @app.get("/envs/{env_id}/tasks", dependencies=[Depends(_bearer)])
    async def list_env_tasks(env_id: str) -> dict[str, Any]:
        """Task IDs (grouped by split) + their Lite task metadata for one env type."""
        if not _env_visible(env_id):
            raise HTTPException(
                status_code=404,
                detail=f"env_id={env_id!r} not registered on this server",
            )
        if env_id in state.blocked_env_ids:
            raise EnvUnavailable(what=_blocked_env_detail(env_id), status_code=501)
        try:
            # Offload: first task_ids call triggers _load_tasks (~21s for browsergym) —
            # keep it off the event loop.
            splits = await asyncio.to_thread(gym.registry.task_ids, env_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except EnvDepsMissingError as e:
            # deps-missing (the env is directory-scanned but its install.sh never
            # ran) is NO-SERVICE, not a server error: emit 501 so the client types
            # it EnvUnavailable — symmetric with POST /instances and _env_metadata.
            raise e.as_env_unavailable(env_id) from e
        assert isinstance(splits, dict), "task_ids without split should return dict"

        metadata: dict[str, Any] = {}
        kwargs_map: dict[str, Any] = {}
        for task_ids in splits.values():
            for tid in task_ids:
                md = gym.registry.task_metadata(env_id, tid)
                if md is not None:
                    metadata[tid] = md.to_dict()
                # Carry the registered make kwargs (wrapper-owned config such as
                # timeouts, plus env-owned make config such as cursor) so
                # register_from_server rebuilds specs that resolve the SAME
                # config as a direct make() — see CARRIED_SPEC_KWARGS.
                kw = gym.registry.task_kwargs(env_id, tid)
                if kw:
                    kwargs_map[tid] = kw
        # env-wide make-kwarg defaults (layer 2) so the client resolves the same
        # config as direct mode without importing the env module.
        return {
            "splits": splits,
            "metadata": metadata,
            "kwargs": kwargs_map,
            "env_make_kwargs": gym.registry.env_make_kwargs(env_id),
            "env_supported_kwargs": gym.registry.env_supported_kwargs(env_id),
            "server_config": _env_server_config_snapshot(
                env_id,
                state,
                token=token,
                admin_token=admin_token,
            ),
        }

    # -- bulk cleanup --------------------------------------------------------

    @app.delete("/instances")
    async def bulk_close_instances(
        session_id: str | None = None,
        env_id: str | None = None,
        force: bool = False,
        dry_run: bool = False,
        token_hash: str = Depends(_bearer),
    ) -> dict[str, Any]:
        # Filter params live in the query string (canonical ordering:
        # session_id → env_id → force → dry_run). Scoping cases:
        #   session_id + env_id   → precise: matched session_id × env_id
        #   session_id only       → all env_ids of that session
        #   env_id only           → all sessions of that env_id (this token)
        #   neither               → ALL envs under this token (kill switch)
        # The any-unscoped case demands explicit ``force=true``; otherwise
        # 400 — guards against a typo / missing param quietly nuking a
        # co-tenant's session or the user's whole workload.
        if not (env_id and session_id) and not force:
            raise HTTPException(
                status_code=400,
                detail=(
                    "bulk-close with unpinned session_id/env_id is wide-net; "
                    "pass both query params to scope it precisely, or pass "
                    "``force=true`` to confirm. Current scope: "
                    f"session_id={session_id!r}, env_id={env_id!r}"
                    " — would match every instance not excluded by the "
                    "set params under your token."
                ),
            )
        # Scope to caller's own token. token_hash is constant per
        # env-server in strict mode (forward-looking for future
        # multi-token-per-server); in passthrough mode it's the
        # per-client sha256 so this filter is actively load-bearing.
        async with state.lock:
            victims = [
                (id_, s) for id_, s in state.envs.items()
                if s.token_hash == token_hash
                and (env_id is None or s.env_id == env_id)
                and (session_id is None or s.session_id == session_id)
                # Never bulk-close an env with a step/reset awaited
                # inside it — the caller re-issues once the op drains
                # (skipped ids are reported so the omission is visible).
                and s.in_flight == 0
            ]
            skipped_in_flight = [
                id_ for id_, s in state.envs.items()
                if s.token_hash == token_hash
                and (env_id is None or s.env_id == env_id)
                and (session_id is None or s.session_id == session_id)
                and s.in_flight > 0
            ]
            # Dry-run: compute victims under the same lock to get a
            # consistent snapshot, but DON'T mutate state.envs.
            if dry_run:
                ids = [id_ for id_, _ in victims]
                logger.info(
                    "bulk-close DRY_RUN session_id=%s env_id=%s force=%s "
                    "token_hash=%s would_close=%d",
                    session_id, env_id, force, token_hash, len(ids),
                )
                return {"would_close": ids, "dry_run": True}
            for id_, _ in victims:
                if state.envs.pop(id_, None) is not None:
                    state.admission.release()
        # Parallel close — matches the convention of _reap_idle and
        # lifespan shutdown. Each backend's teardown is itself
        # internally serialised (e.g. docker daemon, OmniBoxes master),
        # so the speedup is bounded by backend throughput; what we
        # eliminate is the host-side wait pattern, which matters for
        # 32+ stale-env wipes.
        if victims:
            await asyncio.gather(
                *(_close_quietly(state, s, id_) for id_, s in victims),
                return_exceptions=True,
            )
        ids = [id_ for id_, _ in victims]
        logger.info(
            "bulk-close session_id=%s env_id=%s force=%s token_hash=%s closed=%d",
            session_id, env_id, force, token_hash, len(ids),
        )
        return {"closed": ids, "skipped_in_flight": skipped_in_flight}

    return app
