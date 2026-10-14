"""MobileGym — CUA-Lite gym wrapper for MobileGym simulated mobile environment.

Wraps MobileGym's browser-based Android-like simulator as a CUA-Lite gym
environment using LiteMobileActionSpace ([0, 1000] coordinates).

MobileGym provides 24 simulated apps and 416 parameterized task templates
with deterministic state-diff judges.

**Container-only (SINGLETON).** The whole task lifecycle
(browser pool + Vite UI + setup → step → evaluate → teardown) runs inside ONE
shared ``cua-lite/mobilegym`` container per env-server (``docker/server.py``). The
host is a thin JSON-RPC client (:class:`RemoteMobileGymEnv`) that imports ZERO
``bench_env`` — task metadata is read from a STATIC manifest
(``data/tasks.json``, dumped by ``scripts/utils/tasks.sh`` from the
pinned image; mirrors androidworld). The container resolves the live
``task_cls`` from the ``task_id`` itself.

Prerequisites:
  - Docker + the ``cua-lite/mobilegym:latest`` image
    (``uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh``).

Usage:
    uv run python -c "
    import asyncio, lite.gym as gym
    async def main():
        env = gym.make('mobilegym@bilibili.OpenRankingTask', max_steps=5)
        obs = await env.reset()
        print(obs.text)
        await env.close()
    asyncio.run(main())
    "
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import urllib.error
import urllib.request
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Literal

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.extra_tools import (
    LiteFinishToolSet,
    make_open_app_tool,
)
from lite.core.tools.calls import EnvAction, RuntimeEnvAction
from lite.core.tools.schemas import BaseTools
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import CapacityExhausted
from lite.gym.registry import invalidate_services, register
from lite.gym.services import register_services
from lite.gym.remote.reaper import SingletonContainerServices
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.docker import docker_rm_f, docker_run
from lite.gym.utils.backend.freshness import image_for
from lite.gym.utils.backend.wait import wait_until_ready
from lite.gym.utils.config.naming import container_name
from lite.gym.utils.feedback.errors import (
    BackendExecutionErrorDetail,
    ToolErrorFeedback,
    error_only_feedback,
    parse_container_action_error_record,
    record_model_action_error,
    record_tool_execution_error,
    unsupported_action_message,
)
from lite.gym.utils.feedback.ingress import (
    invalid_action_message,
    prepare_env_tool_calls,
    standalone_tool_call_feedback,
    unsupported_env_action_message,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.utils.feedback.surface import (
    android_supported_actions,
    resolve_extra_tools,
    resolve_schema_valid_actions,
    resolve_valid_actions,
)
from lite.utils.image import png_from_b64_async

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)

logger = logging.getLogger(__name__)


def _derive_max_browsers() -> int:
    # Upper bound on concurrent Chromium processes. Default derived from
    # host RAM (Chromium per-process resident set ~400 MB; reserve 30% of
    # RAM for the pool). Floor 4 protects tiny hosts; ceiling 32 keeps us
    # within the rendering-pipeline serialization sweet spot warned about
    # in upstream KNOWN_ISSUES.md §2 (upstream-validated layout caps at 16
    # × 8 = 128; we've extended to 32 × 8 = 256 only because our hosts
    # have 1+ TB RAM headroom — operators on smaller hosts will land at
    # the RAM-derived value, not the ceiling).
    #
    # If ``capacity`` (psutil) is unavailable in some weird test setup,
    # fall back to the historical hardcoded default so this module
    # remains importable. Module-level evaluation means any ImportError
    # here would break the whole gym registry.
    try:
        from lite.gym.utils.server.capacity import cached_host_capacity
        host = cached_host_capacity()
    except Exception as e:
        logger.warning(
            "mobilegym: capacity probe failed (%s); "
            "_MAX_BROWSERS falls back to historical default 16",
            type(e).__name__,
        )
        return 16
    _BROWSER_RAM_GB = 0.4
    ram_budget_gb = host.ram_total_gb * 0.3
    return max(4, min(int(ram_budget_gb / _BROWSER_RAM_GB), 32))


# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# MOBILEGYM_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
# null: mobilegym registers MANY tasks whose budget varies by
# difficulty, so the concrete value is resolved per-task at registration from the
# static manifest (data/tasks.json). When this is None, RemoteMobileGymEnv falls
# back to the manifest-supplied ``base_max_steps``. A concrete int here would
# override every task.
_MAX_STEPS = CFG.env_kwargs["max_steps"]
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
_EVAL_MODE = CFG.env_kwargs["eval_mode"]
_REWARD_SHAPING = CFG.env_kwargs["reward_shaping"]
_DISPLAY_RESOLUTION = tuple(CFG.env_kwargs["display_resolution"])
_DPR = CFG.env_kwargs["dpr"]
_SEED = CFG.env_kwargs["seed"]
_VALID_ACTIONS_CONFIG = CFG.env_kwargs.get("valid_actions")
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]
# --- server_kwargs (per-deployment) ---
# The pool runs INSIDE the container; these host consts are read from the yaml and
# forwarded to the container via `docker run -e` (MobileGymContainerServices.ensure
# forwards max_browsers / contexts_per_browser / idle_browser_ttl_s / launch_timeout_s),
# so the in-container pool matches the yaml.
_CONTEXTS_PER_BROWSER = CFG.server_kwargs["contexts_per_browser"]
# 0 (the typed sentinel) → auto-derive from host RAM; a non-zero yaml value pins it.
_MAX_BROWSERS = CFG.server_kwargs["max_browsers"] or _derive_max_browsers()
_IDLE_BROWSER_TTL_S = CFG.server_kwargs["idle_browser_ttl_s"]
#: Timeout for the in-container ``chromium.launch()`` + ``browser.new_context()``.
#: Chromium DevTools handshake can hang silently under
#: ``fs.inotify.max_user_instances`` exhaustion (KNOWN_ISSUES.md §3).
_LAUNCH_TIMEOUT_S = CFG.server_kwargs["launch_timeout_s"]
# ============================================================================

# Per-task step budgets are RESOLVED in the static manifest (data/tasks.json,
# field ``max_steps`` = text-mode base, mirroring bench_env/config.py
# get_max_steps). The grounded-mode +15 AnswerSheet bonus is added per-instance
# in RemoteMobileGymEnv.max_steps when eval_mode == "grounded". The host imports
# ZERO bench_env (the container resolves the live ``task_cls`` from ``task_id``).

# ---------------------------------------------------------------------------
# App catalog
# ---------------------------------------------------------------------------
#
# The mobilegym sandbox ships with this exact set of mock apps (see the
# reference's ``bench_env/agent/mai_ui.py:62`` "Available Apps" line).
# Surfaced via ``metadata.others["apps"]`` so action adapters that
# accept an ``open`` action (currently mai_ui mobile) inject the right
# names into their system prompt instead of the SFT-default androidworld
# list. Also embedded as the ``app_name`` enum in the ``open_app`` extra
# tool below, so tool-schema agents (e.g. Qwen3-VL) see the same names.
#
# ⚠️ Every name MUST be resolvable by bench_env's ``MobileGymEnv._open_app``
# (``bench_env/env/mobile_gym.py``): an exact (case-sensitive) key of its
# ``APP_NAME_MAP``, or a string that lowercases to one of the map's appId
# values. Unresolvable names are SILENTLY skipped in-container (the screen
# just doesn't change), so a pretty-but-unmapped name here turns every
# open_app call for that app into a no-op the agent can't see.

_MOBILEGYM_APPS: list[str] = [
    "WeChat", "Alipay", "Map", "Calendar", "Clock", "Notes",
    "Settings", "Phone", "SMS", "Gallery", "File Manager", "Browser",
    "Calculator", "Weather", "Compass", "Bilibili", "RedNote",
    "WeChat Reading", "Tencent Meeting", "12306", "Spotify",
    "Reddit", "X", "eBay",
]


# ---------------------------------------------------------------------------
# Extra tools
# ---------------------------------------------------------------------------
# Routed to MobileGym's ``AWAKE`` action by the in-container server's
# ``_translate_action`` (docker/server.py). Surfaced as an OpenAI function tool through
# ``metadata.extra_tool_schemas``. Qwen-family native
# ``mobile_use(action="open", text=...)`` and MAI-UI app-open surfaces
# canonicalize to this env-owned ``open_app`` boundary when configs opt in.

class MobilegymTools(BaseTools):
    """What mobilegym declares beyond the GUI surface: ``open_app``."""

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "open_app": make_open_app_tool(_MOBILEGYM_APPS),
    }


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = MobilegymTools.get_tool_names() | LiteFinishToolSet.get_tool_names()

_SUPPORTED_ACTIONS = android_supported_actions()
_SCHEMA_VALID_ACTIONS = resolve_schema_valid_actions(
    _VALID_ACTIONS_CONFIG,
    env_name="mobilegym",
    platform="mobile",
    supported_actions=_SUPPORTED_ACTIONS,
)


# ---------------------------------------------------------------------------
# env-server scope helpers (used by the container Services below)
# ---------------------------------------------------------------------------

#: Per-env-id "already ensured" guard for MobileGymContainerServices.ensure.
_services_started: set[str] = set()

# Published RPC-port band for the cua-lite/mobilegym container (the in-container
# vite/nginx default 4173 is irrelevant to the host — only the RPC port is
# published). The range MUST stay inside :mod:`lite.gym.utils.backend.ports`'s
# allocation map (see its docstring for the cross-env range layout).
_MOBILEGYM_FALLBACK_START = 7660
_MOBILEGYM_FALLBACK_END = 7699


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

def _load_tasks_json() -> dict[str, dict[str, Any]]:
    """Load the STATIC task manifest dumped from the pinned docker image.

    Lets the host enumerate task ids + register them with gym.registry
    WITHOUT importing the ``bench_env`` package (which would force the bench
    tree + playwright onto every env-server host). The container resolves the
    live ``task_cls`` from the ``task_id`` itself. Mirrors androidworld.

    See ``scripts/utils/tasks.sh`` — re-run that script when the
    mobilegym pin in ``docker/Dockerfile`` changes.
    """
    tasks_path = Path(__file__).parent / "data" / "tasks.json"
    if not tasks_path.is_file():
        raise FileNotFoundError(
            f"{tasks_path} not found. Re-generate with:\n"
            f"  bash lite/gym/envs/mobilegym/scripts/utils/tasks.sh\n"
            "(requires cua-lite/mobilegym:latest docker image — run scripts/install.sh first)."
        )
    with open(tasks_path) as f:
        return json.load(f)


def _task_metadata(
    *,
    task_id: str,
    difficulty: str,
    scope: str,
    objective: str,
    composition: str,
    capabilities: list[str],
    task_apps: list[str],
    answer_fields: bool,
    base_max_steps: int,
) -> LiteCUAMetadata:
    """Same-source metadata builder, called with
    the manifest-derived ctor kwargs. ``task_name``/``suite`` derive from the
    task_id; ``task_apps`` = apps THIS task involves vs ``apps`` = the full
    launchable catalog (open_app enum source); ``max_steps`` carries the
    text-mode base (the +15 grounded AnswerSheet bonus is the eval_mode
    amendment); extra_tool_schemas mirrors bind()'s default resolution."""
    others: dict[str, Any] = {
        "task_name": task_id.split(".", 1)[1] if "." in task_id else task_id,
        "suite": task_id.split(".", 1)[0] if "." in task_id else "",
        "difficulty": difficulty,
        "scope": scope,
        "objective": objective,
        "composition": composition,
        "capabilities": list(capabilities),
        "task_apps": list(task_apps),
        "max_steps": base_max_steps,
        "apps": list(_MOBILEGYM_APPS),
    }
    # Mark query tasks that need AnswerSheet interaction so callers can
    # filter them with --filter "lambda m: not m.others.get('needs_answer_sheet')"
    if answer_fields:
        others["needs_answer_sheet"] = True
    return LiteCUAMetadata(
        dims=("mobile", "use"),
        extra_tool_schemas=resolve_extra_tools(
            _EXTRA_TOOLS, tools=MobilegymTools, env_name="mobilegym",
        ),
        valid_actions=list(_SCHEMA_VALID_ACTIONS),
        others=others,
    )


def _register_tasks() -> None:
    """Register every mobilegym task from the static manifest.

    Each ``task_id → meta`` entry carries the RESOLVED text-mode ``max_steps``
    base (explicit ``cls.max_steps`` else difficulty default), the split + seed
    (the old ``[("test","eval",42),("train","train",None)]`` loop, pre-encoded),
    and the metadata the host needs. The grounded-mode +15 AnswerSheet bonus is
    per-instance and added in :attr:`RemoteMobileGymEnv.max_steps`.
    """
    for task_id, meta in _load_tasks_json().items():
        base_max_steps = meta["max_steps"]

        # Set the remote factory DIRECTLY on the entry point; nothing rebinds
        # the registered spec after this loop.
        entry_kwargs: dict[str, Any] = {
            "task_id": task_id,
            "difficulty": meta["difficulty"],
            "scope": meta["scope"],
            "objective": meta["objective"],
            "composition": meta["composition"],
            "capabilities": meta["capabilities"],
            "task_apps": meta["apps"],
            "answer_fields": meta["answer_fields"],
            "base_max_steps": base_max_steps,
        }
        reg_kwargs: dict[str, Any] = {}
        if meta["seed"] is not None:
            reg_kwargs["seed"] = meta["seed"]

        register(
            key=f"mobilegym@{task_id}",
            entry_point=partial(_make_remote_env, **entry_kwargs),
            split=meta["split"],
            # Same-source contract: registered copy == the env's
            # builder output, built from EXACTLY the ctor kwargs the
            # entry_point forwards.
            metadata=_task_metadata(**entry_kwargs),
            max_steps=base_max_steps,
            **reg_kwargs,
        )

    logger.debug("Registered mobilegym tasks from static manifest")


# ===========================================================================
# Container backend (SINGLETON)
# ===========================================================================
# One shared cua-lite/mobilegym container per env-server holds the Chromium pool +
# vite dist + an in-container FastAPI RPC server that runs the FULL task
# lifecycle (docker/server.py). The host is a thin JSON-RPC client
# (RemoteMobileGymEnv) with ZERO bench_env dependency, plus a SINGLETON
# SingletonContainerServices (ensure once, boot-only docker rm -f, live_ids=None).

_MOBILEGYM_IMAGE = image_for("mobilegym")


def _mobilegym_container_name(server_port: int | None = None) -> str:
    """``mobilegym-<server_port>`` (or ``mobilegym-d<pid>`` in direct mode) — the
    shared-singleton scheme; see :func:`lite.gym.utils.config.naming.container_name`."""
    return container_name("mobilegym", server_port)


_MOBILEGYM_RM_TIMEOUT_S = float(os.environ.get("CUA_LITE_MOBILEGYM_RM_TIMEOUT_S", str(CFG.server_kwargs["rm_timeout_s"])))


def _mobilegym_docker_rm_f(name: str) -> int:
    """``docker rm -f <name>`` bounded by ``_MOBILEGYM_RM_TIMEOUT_S``; see
    :func:`lite.gym.utils.backend.docker.docker_rm_f`."""
    return docker_rm_f(name, timeout=_MOBILEGYM_RM_TIMEOUT_S, label="mobilegym")


def _mobilegym_healthz(url: str) -> bool:
    """True iff the in-container RPC server at ``url`` answers /healthz ok."""
    if not url:
        return False
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/healthz", timeout=3) as r:
            return bool(json.loads(r.read()).get("ok"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


class RemoteMobileGymEnv(LiteBaseEnv):
    """Thin JSON-RPC client to the cua-lite/mobilegym container's in-container
    server (docker/server.py runs the full task lifecycle). ``metadata`` is a
    pure function of construction kwargs (manifest-derived task fields) — no
    browser / container is touched at construct time, satisfying the
    eager-create contract. The container resolves the live ``task_cls`` from
    ``task_id`` itself, so the host never holds one. reset/step/close POST plain
    JSON and wrap the response into LiteEnvStepResult/LiteEnvObservation."""

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = MobilegymTools

    def __init__(
        self,
        task_id: str,
        *,
        difficulty: str,
        scope: str,
        objective: str,
        composition: str,
        capabilities: list[str],
        task_apps: list[str],
        answer_fields: bool,
        base_max_steps: int,
        max_steps: int | None = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        eval_mode: Literal["text", "grounded"] = _EVAL_MODE,
        reward_shaping: bool = _REWARD_SHAPING,
        seed: int | None = _SEED,
        valid_actions: list[str] | None = _VALID_ACTIONS_CONFIG,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        display_resolution: tuple[int, int] = _DISPLAY_RESOLUTION,
        dpr: float = _DPR,
    ) -> None:
        self._task_id = task_id
        # Manifest-derived task fields (replace the live task_cls — the host
        # imports ZERO bench_env; the container resolves task_cls from task_id).
        self._difficulty = difficulty
        self._scope = scope
        self._objective = objective
        self._composition = composition
        # list(...): the registration entry_kwargs share these list objects
        # across every instance of the task — copy, don't alias.
        self._capabilities = list(capabilities)
        self._task_apps = list(task_apps)
        self._answer_fields = answer_fields
        self._base_max_steps = base_max_steps
        self._seed = seed
        self._max_steps = max_steps
        self._post_action_delay = post_action_delay
        self._eval_mode = eval_mode
        self._reward_shaping = reward_shaping
        self._display_resolution = tuple(display_resolution)
        self._dpr = dpr
        # Soft env_kwarg, resolved through the shared helper so every env
        # answers ``valid_actions`` identically (None = full GUI surface,
        # [] = deliberately none, unknown name = raise at the config
        # boundary). Runtime invalid/unsupported feedback is owned by ``step``.
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="mobilegym", platform="mobile",
        )
        self._schema_valid_actions = resolve_schema_valid_actions(
            valid_actions,
            env_name="mobilegym",
            platform="mobile",
            supported_actions=_SUPPORTED_ACTIONS,
        )
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._instance_id: str | None = None
        # NOTE: deliberately NO host-side step counter. ``inst.step_count`` in
        # docker/server.py is the single source of truth for the budget (the
        # host hands it the authoritative ``max_steps`` at ``/reset`` and reads
        # ``truncated`` back off every ``/step``). A second host-side counter
        # only ever drifts from it, which is exactly what it did.
        self._last_observation_image: bytes | None = None
        self._last_observation_text: str | None = None

    @property
    def max_steps(self) -> int:
        # Grounded-mode tasks that declare answer_fields need +15 steps for the
        # "open AnswerSheet → fill → submit" flow (mirrors bench_env/config.py
        # get_max_steps); text mode never gets the bonus. The raw base comes
        # from the static manifest (base_max_steps) when the caller passed None.
        bonus = 15 if (self._eval_mode == "grounded" and self._answer_fields) else 0
        raw = self._max_steps if self._max_steps is not None else self._base_max_steps
        return raw + bonus

    # Shared with registration (module-level builder); exposed on the class
    # per the family-wide convention.
    _task_metadata = staticmethod(_task_metadata)

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendments: extra_tools resolved at construct;
        # max_steps = live budget (max_steps override + the eval_mode-
        # grounded AnswerSheet bonus — both env_kwargs; defaults leave it
        # at the builder's text-mode base).
        md = self._task_metadata(
            task_id=self._task_id,
            difficulty=self._difficulty,
            scope=self._scope,
            objective=self._objective,
            composition=self._composition,
            capabilities=self._capabilities,
            task_apps=self._task_apps,
            answer_fields=self._answer_fields,
            base_max_steps=self._base_max_steps,
        )
        return dataclasses.replace(
            md,
            valid_actions=list(self._schema_valid_actions),
            extra_tool_schemas=self._extra_tool_schemas,
            others={**md.others, "max_steps": self.max_steps},
        )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to the container RPC port, return the parsed JSON body.
        Runs blocking urllib in a thread so the event loop isn't stalled."""
        url = os.environ.get("MOBILEGYM_RPC_URL", "").rstrip("/") + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )

        def _do() -> dict[str, Any]:
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                # 503 = in-container browser pool saturated (a RECOVERABLE capacity
                # event). Map it to CapacityExhausted (like webgym's RemoteWebGymEnv)
                # so the env-server emits 503+Retry-After and the rollout's group-redraw
                # path fires — instead of a generic RuntimeError that fails the task hard.
                if e.code == 503:
                    raise CapacityExhausted(
                        what=f"mobilegym container pool full ({path}): {detail}",
                        retry_after_s=60.0,
                    ) from e
                raise RuntimeError(f"mobilegym container {path} returned {e.code}: {detail}") from e

        return _do()

    async def reset(self) -> LiteEnvObservation:
        body = {
            "task_id": self._task_id,
            "seed": self._seed,
            "eval_mode": self._eval_mode,
            "reward_shaping": self._reward_shaping,
            "physical_size": list(self._display_resolution),
            "dpr": self._dpr,
            "delay": self._post_action_delay,
            # Authoritative budget computed host-side (honours an env_kwargs
            # max_steps override + the grounded AnswerSheet bonus). The container
            # uses this verbatim for truncation instead of recomputing — so a
            # config/CLI max_steps override actually caps the episode (it was
            # silently dropped when the container owned the number).
            "max_steps": self.max_steps,
        }
        resp = await asyncio.to_thread(self._post, "/reset", body)
        self._instance_id = resp["instance_id"]
        png = await png_from_b64_async(resp["screenshot_b64"])
        self._last_observation_image = png
        self._last_observation_text = resp["instruction"]
        return LiteEnvObservation(image=png,
            text=resp["instruction"],
        )

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        runtime_metadata = dataclasses.replace(
            metadata, valid_actions=self._valid_actions,
        )
        actions_with_result_ids, action_errors = prepare_env_tool_calls(
            actions, runtime_metadata,
        )
        action_errors: dict[str, ToolErrorFeedback] = dict(action_errors)
        actions_to_send: list[EnvAction] = []
        sent_actions_with_result_ids: list[tuple[EnvAction, str | None]] = []
        # R3: a GUI slot the host rejected owes a frame too, but it never
        # reaches the container so no frame comes back for it. Record WHERE it
        # sat so the frames can be re-interleaved in slot order below; the host
        # cannot capture mid-batch (the container runs the whole batch in one
        # RPC), so a rejected slot reuses the frame of the action before it.
        rejected_slot_positions: list[int] = []
        for slot_index, (action, result_call_id) in enumerate(actions_with_result_ids):
            name = action["name"]
            tool_feedback = standalone_tool_call_feedback(
                action,
                _KNOWN_STANDALONE_TOOL_NAMES,
                metadata.extra_tool_schemas,
            )
            if tool_feedback is not None:
                if result_call_id:
                    action_errors[result_call_id] = tool_feedback
                rejected_slot_positions.append(slot_index)
                continue
            invalid_action = invalid_action_message(
                action, runtime_metadata.valid_actions,
            )
            if invalid_action:
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(invalid_action)
                rejected_slot_positions.append(slot_index)
                continue
            unsupported_action = unsupported_env_action_message(
                name,
                _SUPPORTED_ACTIONS,
            )
            if unsupported_action:
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(unsupported_action)
                rejected_slot_positions.append(slot_index)
                continue
            actions_to_send.append(action)
            sent_actions_with_result_ids.append((action, result_call_id))

        # A step whose calls were ALL filtered out still goes to the container
        # with an empty action list. It must not short-circuit here:
        #
        #   * ``inst.step_count`` (docker/server.py) is the ONLY step counter,
        #     and the ONLY truncation authority (this env returns the
        #     container's ``truncated`` verbatim below). A host-side early
        #     return that skipped the RPC advanced nothing, so N filtered steps
        #     left the container N turns behind and it truncated N turns late.
        #   * ``_evaluate`` lives in the container and runs ONLY inside
        #     ``/step`` on ``terminated or truncated`` — there is no host-side
        #     evaluator to fall back on. So a fully-filtered step that exhausts
        #     the budget can only be scored by reaching the container; short
        #     circuiting returned ``reward=None`` and the episode was never
        #     scored at all.
        #   * the container also returns a FRESH frame, where the early return
        #     re-served the previous turn's stale screenshot.
        #
        # The empty batch is cheap and the container handles it (its action loop
        # simply does not iterate), so the saved round-trip was never worth the
        # three bugs. Contrast ``webharbor/webvoyager``, which legitimately keeps
        # its fast path because ITS evaluator is host-side.
        resp = await asyncio.to_thread(
            self._post, "/step",
            {"instance_id": self._instance_id, "actions": actions_to_send},
        )
        # Per-action model feedback from the container (docker/server.py /step).
        # It reports the INDEX of the offending entry in the list we sent, which
        # is the index of ``actions_to_send`` / ``sent_actions_with_result_ids``.
        # That maps the error back to the ORIGINATING top-level call_id (the
        # mobile() batch the action came from) even when earlier actions were
        # filtered out host-side before the RPC.
        for entry in resp.get("action_errors") or []:
            record = parse_container_action_error_record(entry)
            if record is None:
                continue
            index = record.index
            if not 0 <= index < len(sent_actions_with_result_ids):
                continue
            result_call_id = sent_actions_with_result_ids[index][1]
            if not result_call_id:
                continue
            name = record.name
            detail = record.error or record.message
            if record.kind == "unsupported_action":
                action_errors[result_call_id] = error_only_feedback(
                    record.message or unsupported_action_message(str(name))
                )
            elif record.kind == "tool_execution":
                record_tool_execution_error(
                    action_errors,
                    result_call_id,
                    BackendExecutionErrorDetail(detail),
                    action_name=name,
                )
            elif record.kind == "model_action":
                record_model_action_error(
                    action_errors,
                    result_call_id,
                    ValueError(detail),
                    action_name=name,
                )
        # An agent finish action (`terminate`, or `response`/answer) ends the
        # episode. The in-container backend already terminates on `terminate`
        # (→ COMPLETE), but in text mode `ANSWER` "submits, does not terminate" —
        # so an agent that answers (the natural completion of a Q&A task) would
        # otherwise run to max_steps and be scored as an Overdue-Termination
        # failure. OR-in the agent's finish signal so `answer`/`response` also
        # terminates, matching every other lite env (sandbox/base.py:503, webgym,
        # browsergym, captcha, androidworld). The action is still forwarded above,
        # so the backend records the answer + its match reward in this same
        # step's resp["reward"]; in grounded mode `answer` is a misfire (answers go
        # through the AnswerSheet UI) and this is a graceful termination fallback.
        finish_calls = [
            (action, call_id)
            for action, call_id in sent_actions_with_result_ids
            if action["name"] in LiteFinishToolSet.get_tool_names()
        ]
        agent_finished = bool(finish_calls)
        # Model-emitted calls that ENDED the episode. They get no continuation
        # observation: devs/migration/verify.py forbids a tool result for a
        # terminal call. Keyed on the env-local ``action["call_id"]`` and NOT
        # on ``result_call_id``, because an INTERNAL finish call has no local
        # ``call_id`` and the loop-detect wrapper's injected ``terminate``
        # carries the intercepted NON-finish model call's id as
        # ``_result_call_id`` -- that call must still be answered.
        terminal_call_ids = {
            action["call_id"] for action, _ in finish_calls if action.get("call_id")
        }
        # One result frame per EXECUTED action, in action order: the container
        # runs the whole batch, so it is the only side that can see the screen
        # between two actions. ``screenshots_b64`` is the single owner of
        # this step's frames; ``/step`` returns no singular spelling.
        frames_b64 = resp["screenshots_b64"]
        pngs = [
            frame for frame in
            [await png_from_b64_async(entry) for entry in frames_b64]
            if frame is not None
        ]
        if rejected_slot_positions and pngs:
            # Re-interleave: walk the ORIGINAL slot order, taking the container's
            # next frame for a slot that ran and repeating the last emitted frame
            # for one the host rejected. The repeat is honest -- that slot
            # changed nothing -- and it is the only shape a batched RPC allows.
            rebuilt: list[bytes] = []
            executed_frames = iter(pngs)
            rejected = set(rejected_slot_positions)
            for slot_index in range(len(actions_with_result_ids)):
                if slot_index in rejected:
                    rebuilt.append(rebuilt[-1] if rebuilt else pngs[0])
                else:
                    rebuilt.append(next(executed_frames, pngs[-1]))
            pngs = rebuilt
        png = pngs[-1] if pngs else None
        self._last_observation_image = png
        self._last_observation_text = None
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=resp["reward"],
                terminated=resp["terminated"] or agent_finished,
                truncated=resp["truncated"],
                info={EXECUTED_ACTIONS_INFO_KEY: resp["executed"]},
            ),
            ordered_call_ids=result_call_ids,
            continue_call_ids=[
                call_id for call_id in result_call_ids
                if call_id not in terminal_call_ids
            ],
            images=pngs,
            feedback=action_errors,
        )

    async def close(self) -> None:
        if self._instance_id is not None:
            try:
                await asyncio.to_thread(
                    self._post, "/close", {"instance_id": self._instance_id},
                )
            except Exception as e:
                logger.warning("mobilegym container close failed: %s", e)
            self._instance_id = None
        self._last_observation_image = None
        self._last_observation_text = None


class MobileGymContainerServices(SingletonContainerServices):
    """Container variant of mobilegym's env-server capability.

    SINGLETON (one fixed shared container per env-server): ``ensure`` brings up ONE
    ``cua-lite/mobilegym`` container (the Chromium pool + vite dist + the in-container
    task-lifecycle RPC server) and points ``MOBILEGYM_RPC_URL`` at its published
    port. ``shutdown``/``reap(boot=True)`` (inherited from
    :class:`SingletonContainerServices`) ``docker rm -f`` this server's container, named by
    :meth:`container_name`, scoped by ``server_port`` (→ ``d<pid>`` in direct mode).
    Steady-tick ``reap`` is a no-op (the in-container idle reaper handles pool compaction)."""

    rm_timeout_s = _MOBILEGYM_RM_TIMEOUT_S
    rm_label = "mobilegym"

    def container_name(self, scope) -> str:
        return _mobilegym_container_name(scope.server_port)

    def ensure(self, env_id: str) -> None:
        if env_id in _services_started:
            return
        url = os.environ.get("MOBILEGYM_RPC_URL", "")
        if _mobilegym_healthz(url):          # already serving → reuse
            _services_started.add(env_id)
            return
        name = _mobilegym_container_name()
        _mobilegym_docker_rm_f(name)         # clear a dead-but-present same-name container
        from lite.gym.errors import EnvDepsMissingError
        from lite.gym.utils.backend.ports import allocate_ports
        # Published RPC port from mobilegym's documented fallback band 7660-7699.
        host_port = allocate_ports(n=1, range_start=_MOBILEGYM_FALLBACK_START,
                                   range_end=_MOBILEGYM_FALLBACK_END + 1)[0]
        # Memory headroom derived from the pool's CONTEXT capacity, not just
        # the browser count: each browser hosts _CONTEXTS_PER_BROWSER pages that
        # each render the OS + run JS (~0.15 GB/context observed), plus ~0.4 GB
        # per-browser base. The old `max_browsers*0.6` was context-blind and
        # undershot ~2x (32 browsers => 19 GiB, but its 256 contexts need ~38),
            # which throttled concurrency >64. 8 GB floor bounds OOM blast radius.
        _mem_gb = _MAX_BROWSERS * (0.4 + _CONTEXTS_PER_BROWSER * 0.15)
        mem = f"{max(8, round(_mem_gb))}g"
        logger.info("mobilegym container: docker run %s (max_browsers=%d, host_port=%d, mem=%s)",
                    name, _MAX_BROWSERS, host_port, mem)
        # Pool config is sourced from the yaml (CFG) on the host and passed in
        # via -e, so the in-container pool MATCHES main.py's. The container has
        # no access to lite.gym/env_config, so os.environ is its only channel;
        # the defaults in docker/server.py are pure fallbacks. Single source of
        # truth stays the yaml: yaml → CFG → -e → in-container os.environ.
        docker_run(
            name, _MOBILEGYM_IMAGE, mem=mem, port=(host_port, 8000),
            env={
                "MOBILEGYM_MAX_BROWSERS": str(_MAX_BROWSERS),
                "MOBILEGYM_CONTEXTS_PER_BROWSER": str(_CONTEXTS_PER_BROWSER),
                "MOBILEGYM_IDLE_BROWSER_TTL_S": str(_IDLE_BROWSER_TTL_S),
                "MOBILEGYM_LAUNCH_TIMEOUT_S": str(_LAUNCH_TIMEOUT_S),
            },
        )
        new_url = f"http://localhost:{host_port}"
        os.environ["MOBILEGYM_RPC_URL"] = new_url
        if wait_until_ready(lambda: _mobilegym_healthz(new_url), timeout=180, interval=2):
            _services_started.add(env_id)
            logger.info("mobilegym container ready: %s", new_url)
            return
        raise EnvDepsMissingError(
            what=f"mobilegym container {name} started but /healthz not ok within 180s",
            install=f"check `docker logs {name}` and lite/gym/envs/mobilegym/scripts/install.sh",
            see="lite/gym/envs/mobilegym/README.md",
        )

    def health(self, env_id: str) -> None:
        from lite.gym.errors import EnvDepsMissingError
        if not _mobilegym_healthz(os.environ.get("MOBILEGYM_RPC_URL", "")):
            raise EnvDepsMissingError(
                what="mobilegym container not serving (/healthz unreachable)",
                install="uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh",
                see="lite/gym/envs/mobilegym/README.md",
            )

    def _evict_cache(self, env_id: str) -> None:
        _services_started.discard(env_id)
        os.environ.pop("MOBILEGYM_RPC_URL", None)
        invalidate_services(env_id)

    def reap(self, env_id: str, scope, in_use: set[str], *, boot: bool = False) -> int:
        try:
            return super().reap(env_id, scope, in_use, boot=boot)
        finally:
            if boot:
                self._evict_cache(env_id)

    def shutdown(self, env_id: str, scope) -> None:
        try:
            super().shutdown(env_id, scope)
        finally:
            self._evict_cache(env_id)

    # live_ids inherited from SingletonContainerServices. Steady-tick reap is a
    # no-op; the in-container idle reaper compacts the pool.


def _make_remote_env(
    *,
    task_id: str,
    difficulty: str,
    scope: str,
    objective: str,
    composition: str,
    capabilities: list[str],
    task_apps: list[str],
    answer_fields: bool,
    base_max_steps: int,
    **kwargs: Any,
) -> RemoteMobileGymEnv:
    return RemoteMobileGymEnv(
        task_id=task_id,
        difficulty=difficulty,
        scope=scope,
        objective=objective,
        composition=composition,
        capabilities=capabilities,
        task_apps=task_apps,
        answer_fields=answer_fields,
        base_max_steps=base_max_steps,
        **kwargs,
    )


# Register all tasks from the static manifest — each entry point already points
# at _make_remote_env, so nothing rebinds a spec afterwards. Then register the
# container Services unconditionally: there is no env-var gate, and per-task
# shape metadata is resolved at instance time rather than baked in here.
_register_tasks()
register_services("mobilegym", MobileGymContainerServices())
from lite.gym.services import BackendFamily, register_family  # noqa: E402

register_family("mobilegym", BackendFamily.SINGLETON)
