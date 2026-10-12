"""mobileworld — CUA-Lite gym wrapper for Tongyi-MAI's MobileWorld benchmark.

Wraps MobileWorld (201 upstream tasks, 161 registered; 20 mobile apps; Docker-in-Docker boxes
holding a rooted Android emulator + self-hosted app backends) as a CUA-Lite
gym environment with the mobile action space. Tasks are evaluated by the
upstream in-container server against real device / backend state (SQLite,
Mattermost/Mastodon databases, files) — never UI matching.

Each ``gym.make`` drives one ``cua-lite/mobileworld:latest`` container (see
:mod:`lite.gym.envs.mobileworld.container`) through the upstream FastAPI
server on its published port: ``/task/init`` (reloads the ``init_state`` AVD
snapshot — a full device reset), ``/step`` (JSONAction dispatch),
``/screenshot``, ``/task/eval`` (0-1 score). Containers are expensive to boot
(nested dockerd + backends + emulator), so the env keeps the
``EnvServerPoolable`` explicit construction/bind contract for config validation
and task/default binding. Env-server mode still centralizes admission, ownership
cleanup, and retry/recovery.

Task registry loads from the checked-in ``data/tasks.json`` (dumped from the
image by ``scripts/tasks.sh``) — no mobile_world Python import on the host.
``agent-mcp``-tagged tasks are NOT registered (MCP tool integration is not
supported at this stage); ``agent-user-interaction`` tasks remain registered
with ``exclude_reason="ask_user"``. The eval runner filters those 44 tasks by
default, leaving 117 GUI-only tasks (see README).

Prerequisites:
  - uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh
  - /dev/kvm rw-accessible (containers run the Android emulator under KVM)

Usage:
    uv run python -c "import lite.gym as gym; print(gym.registry.task_ids('mobileworld'))"
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import requests

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.extra_tools import (
    LiteFinishToolSet,
    make_open_app_tool,
)
from lite.core.tools.action_space.duration import (
    ACTION_SCHEMA_DURATION_CAPS_SECONDS,
)
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.schemas import BaseTools, make_tool_schema
from lite.gym.container import spawn_background_destroy
from lite.gym.envs.mobileworld.container import (
    ADB_DEVICE,
    MobileWorldContainer,
    MobileWorldContainerFactory,
)
from lite.gym.registry import register, registry
from lite.gym.services import BackendFamily, register_family, register_services
from lite.gym.remote.reaper import ContainerServices
from lite.gym.services import EnvServerPoolable, EnvServerResource
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
    LiteExecutedAction,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.coordinate import norm_to_pixel
from lite.gym.utils.backend.freshness import image_for
from lite.gym.utils.backend.model_inputs import (
    DEFAULT_MODEL_DURATION_CAP_SECONDS,
    coerce_model_numeric,
)
from lite.gym.utils.feedback.errors import (
    append_feedback,
    MODEL_ACTION_ERROR_TYPES,
    ToolErrorFeedback,
    error_only_feedback,
    record_model_action_error,
    record_tool_execution_error,
)
from lite.gym.utils.feedback.ingress import (
    invalid_action_message,
    prepare_env_tool_calls,
    standalone_tool_call_feedback_with_reason,
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
from lite.gym.utils.server.health import CachedEnvDepsHealthCheck
from lite.utils.image import png_from_b64

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# MOBILE_WORLD_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
#: Step budget per episode (upstream ``mw eval --max_round 50``).
#: ``bind()`` signature default.
_MAX_STEPS = CFG.env_kwargs["max_steps"]
#: Seconds to sleep before the post-step screenshot (upstream
#: ``step_wait_time``). ``bind()`` signature default.
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
#: Opt-in extra-tool selection (list of names; ``[]`` = none).
#: ``bind()`` signature default.
_VALID_ACTIONS_CONFIG = CFG.env_kwargs.get("valid_actions")
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]
#: Spawn-image override (default base image). Read by ``__init__``'s signature
#: default and the services ``ensure`` hook.
_IMAGE = CFG.env_kwargs["image"]
# --- server_kwargs (per-deployment) ---
#: ThreadPoolExecutor size. Each live container holds at most 2 threads
#: (boot/reset + step/teardown).
_MAX_WORKERS = CFG.server_kwargs["max_workers"]
#: HTTP timeouts against the in-container server. ``step`` also covers
#: ``ask_user`` (a server-side LLM call); ``task_init`` covers the AVD
#: snapshot reload + app-backend restart; ``eval`` covers backend DB /
#: adb state checks.
_STEP_TIMEOUT = CFG.server_kwargs["step_timeout"]
_TASK_INIT_TIMEOUT = CFG.server_kwargs["task_init_timeout"]
_EVAL_TIMEOUT = CFG.server_kwargs["eval_timeout"]
#: Recycle (destroy + cold-spawn) after N in-place ``/task/init`` reuses —
#: consumed by the framework's ``reset_with_recycle`` (cap counts in-place
#: reuses; cap=N → N+1 episodes per container). MobileWorld reuses ONE
#: container across episodes by design (cold DinD boot is 2-5 min; snapshot
#: reload is seconds) — so this stays LARGE, bounding out-of-snapshot drift
#: (qemu, adb daemon, in-container python globals, /tmp) without turning
#: reuse into cold-boot-per-episode.
_MAX_RESETS_PER_CONTAINER = CFG.server_kwargs["max_resets_per_container"]
# ============================================================================

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(
    max_workers=_MAX_WORKERS,
    thread_name_prefix="mobileworld-exec",
)

# Force-cancel pending futures at interpreter exit so a worker stuck inside a
# long HTTP call can't block ThreadPoolExecutor's own atexit shutdown(wait=True)
# forever (same rationale + LIFO-ordering trick as androidworld's hook).
atexit.register(lambda: _EXECUTOR.shutdown(wait=False, cancel_futures=True))


# ---------------------------------------------------------------------------
# App catalog
# ---------------------------------------------------------------------------
# The launchable apps baked into the MobileWorld image (upstream
# ``runtime/utils/models.py`` APP_DICT; ``launch_app`` matches names
# case-insensitively). Surfaced via ``metadata.others["apps"]`` and
# embedded as the ``app_name`` enum of the ``open_app`` extra tool.

_MOBILE_WORLD_APPS: list[str] = [
    "Calendar", "Camera", "Chrome", "Clock", "Contacts", "Files",
    "Gallery", "Mail", "Maps", "Mastodon", "Mattermost", "SMS",
    "Settings", "Taodian",
]


# ---------------------------------------------------------------------------
# Extra tools
# ---------------------------------------------------------------------------
# Opt-in via ``env_kwargs.extra_tools`` (yaml or per-run env_kwargs).
# ``open_app`` routes to the upstream ``open_app`` JSONAction. ``ask_user``
# routes to the upstream ``ask_user`` JSONAction, whose server-side handler
# queries the simulated-user LLM (host OPENAI_API_KEY / OPENAI_BASE_URL +
# the server_kwargs.user_agent_model knob) and returns its
# reply — REQUIRED for the agent-user-interaction task family; GUI-only
# tasks don't need it.

_ASK_USER_TOOL: dict[str, Any] = make_tool_schema(
    "ask_user",
    description=(
        "Ask the phone's user a question when the task needs information "
        "only they know (preferences, missing details). Returns their reply."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The question to ask the user.",
            },
        },
        "required": ["text"],
    },
)

class MobileworldTools(BaseTools):
    """What mobileworld declares beyond the GUI surface: ``open_app`` /
    ``ask_user``."""

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "open_app": make_open_app_tool(_MOBILE_WORLD_APPS),
        "ask_user": _ASK_USER_TOOL,
    }


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = MobileworldTools.get_tool_names() | LiteFinishToolSet.get_tool_names()

_SUPPORTED_ACTIONS = android_supported_actions()
_SCHEMA_VALID_ACTIONS = resolve_schema_valid_actions(
    _VALID_ACTIONS_CONFIG,
    env_name="mobileworld",
    platform="mobile",
    supported_actions=_SUPPORTED_ACTIONS,
)


def _duration_seconds(
    args: dict[str, Any],
    action_name: str,
    *,
    default: float,
    allow_zero: bool = False,
) -> float:
    raw = args.get("duration", default)
    if raw is None:
        raw = default
    duration = float(coerce_model_numeric(
        raw,
        field="duration",
        action_name=action_name,
        max_value=ACTION_SCHEMA_DURATION_CAPS_SECONDS.get(
            action_name,
            DEFAULT_MODEL_DURATION_CAP_SECONDS,
        ),
    ))
    if duration < 0 or (duration == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "greater than 0"
        raise ValueError(f"{action_name}.duration must be {comparator}")
    return duration


# ---------------------------------------------------------------------------
# In-container server client
# ---------------------------------------------------------------------------

class _Client:
    """Thin sync HTTP client for the upstream in-container FastAPI server.

    All methods are blocking (driven via ``_EXECUTOR``). Mirrors the slice of
    upstream's ``AndroidEnvClient`` the env needs; task/eval endpoints take
    the task's class name.
    """

    def __init__(self, base_url: str, device: str = ADB_DEVICE):
        self._base = base_url
        self._device = device
        self._session = requests.Session()

    def init(self) -> tuple[int, int]:
        """(Re-)build the server-side controller; returns (width, height)."""
        r = self._session.post(
            f"{self._base}/init", json={"device": self._device}, timeout=60,
        )
        r.raise_for_status()
        w, h = r.json()["viewport_size"]
        return int(w), int(h)

    def screenshot_png(self) -> bytes | None:
        """Fetch the container's screenshot and decode its bare-b64 wire form
        to raw PNG bytes at the host boundary. Called inside a ``run_in_executor``
        block, so the sync ``png_from_b64`` decode does not block the loop."""
        r = self._session.get(
            f"{self._base}/screenshot",
            params={"device": self._device, "return_b64": True},
            timeout=60,
        )
        r.raise_for_status()
        return png_from_b64(r.json()["b64_png"])

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """POST /step with a JSONAction dict; returns the response body
        (``result`` carries the ask_user reply)."""
        r = self._session.post(
            f"{self._base}/step",
            json={"device": self._device, "action": action},
            timeout=_STEP_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def task_init(self, task_name: str) -> None:
        """Initialize a task: reloads the ``init_state`` AVD snapshot (full
        device reset), restarts the task's app backends, runs its init hook."""
        r = self._session.post(
            f"{self._base}/task/init",
            json={"task_name": task_name, "req_device": self._device},
            timeout=_TASK_INIT_TIMEOUT,
        )
        r.raise_for_status()

    def task_goal(self, task_name: str) -> str:
        r = self._session.get(
            f"{self._base}/task/goal", params={"task_name": task_name},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def task_eval(self, task_name: str) -> tuple[float, str]:
        """Score the current episode. GET with a JSON body — upstream's
        endpoint reads a request model, mirroring its own client."""
        r = self._session.get(
            f"{self._base}/task/eval",
            json={"task_name": task_name, "req_device": self._device},
            timeout=_EVAL_TIMEOUT,
        )
        r.raise_for_status()
        result = r.json()
        return float(result["score"]), result.get("reason", "")

    def task_tear_down(self, task_name: str) -> None:
        r = self._session.post(
            f"{self._base}/task/tear_down",
            json={"task_name": task_name, "req_device": self._device},
            timeout=60,
        )
        r.raise_for_status()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _check_image(tag: str = _IMAGE) -> None:
    from lite.gym.utils.backend.docker import require_image_present

    require_image_present(image_for("mobileworld", tag=tag))


def _ensure_services(env_id: str) -> None:
    """env-server startup hook: verify the docker image is built + fresh
    before accepting traffic (→ ``available: false`` / HTTP 501 instead of a
    cryptic gym.make failure mid-rollout)."""
    _check_image()


_HEALTH_CHECK = CachedEnvDepsHealthCheck(_ensure_services)


def _health_check(env_id: str) -> None:
    _HEALTH_CHECK(env_id)


class MobileWorldEnv(EnvServerPoolable, EnvServerResource):
    """CUA-Lite wrapper around one MobileWorld container.

    Each episode:
      1. reset() — boots the container if needed (minutes, cold), tears down
         the previous task, POSTs ``/task/init`` (snapshot reload = pristine
         device), takes the first screenshot.
      2. step() — translates CUA-Lite mobile actions to upstream JSONActions
         via ``/step``; ``ask_user`` replies surface in the observation text.
      3. On termination OR truncation — ``/task/eval`` scores the episode
         (upstream parity: state-based checks can pass even on truncation).
      4. close() — tears down the task and destroys the container.
    """

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = MobileworldTools

    def __init__(
        self,
        *,
        image: str = _IMAGE,
        task_id: str = "",
        max_steps: int = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        valid_actions: list[str] | None = _VALID_ACTIONS_CONFIG,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        seed: int | None = None,
    ) -> None:
        self._image = image
        # Container state (lazy-populated by boot()). ``_current_container``
        # is the Layer-B seam attr (drift reaper / reset_with_recycle).
        self._current_container: MobileWorldContainer | None = None
        self._client: _Client | None = None
        self._screen_w = 0
        self._screen_h = 0
        #: Task name initialized on the SERVER side. A recycled container can
        #: still hold the previous task until the next reset tears it down.
        self._active_task: str | None = None
        # In-flight boot tracking, so close() can reap a container acquired
        # by a reset() that was cancelled mid-boot (StepTimeoutWrapper), and
        # so the drift reaper sees the container during the boot window.
        self._pending_container: MobileWorldContainer | None = None
        self._pending_cf_future: Any = None  # concurrent.futures.Future
        #: Reuse cap consumed by ``reset_with_recycle``.
        self._max_resets_per_container = _MAX_RESETS_PER_CONTAINER
        self.bind(
            task_id=task_id,
            max_steps=max_steps,
            post_action_delay=post_action_delay,
            valid_actions=valid_actions,
            extra_tools=extra_tools,
            seed=seed,
        )

    def bind(
        self,
        task_id: str = "",
        *,
        max_steps: int = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        valid_actions: list[str] | None = _VALID_ACTIONS_CONFIG,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        seed: int | None = None,  # noqa: ARG002 — caller symmetry (rollout's group_shared_seed injects seed= uniformly)
    ) -> None:
        """Bind (or re-bind) a task and apply soft kwargs.

        Single entry point for task identity + soft state (direct and server
        construction both land here — see :class:`EnvServerPoolable`). All
        assignments are unconditional; signature defaults are the single source
        of truth. ``task_id=""`` is the legacy no-task seed.
        """
        if task_id:
            meta = _TASKS.get(task_id)
            if meta is None:
                known = ", ".join(sorted(_TASKS)[:5])
                raise ValueError(
                    f"unknown mobileworld task_id={task_id!r}; "
                    f"known examples: {known}..."
                )
            # Copy, don't alias the shared _TASKS entry; bind re-reads it
            # fresh on every rebind, so the copy can't go stale.
            self._task_meta = dict(meta)
            self._instruction = meta["goal"]
        else:
            self._task_meta = {}
            self._instruction = "Interact with the Android device."
        self._task_id = task_id
        self._max_steps = max_steps
        self._post_action_delay = post_action_delay
        # Soft env_kwarg, resolved through the shared helper so every env
        # answers ``valid_actions`` identically (None = full GUI surface,
        # [] = deliberately none, unknown name = raise at the config
        # boundary). Runtime invalid/unsupported feedback is owned by ``step``.
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="mobileworld", platform="mobile",
        )
        self._schema_valid_actions = resolve_schema_valid_actions(
            valid_actions,
            env_name="mobileworld",
            platform="mobile",
            supported_actions=_SUPPORTED_ACTIONS,
        )
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._step_count = 0
        self._terminated = False

    @staticmethod
    def _task_metadata(task_name: str, task_meta: dict[str, Any]) -> LiteCUAMetadata:
        """Same-source metadata builder.
        ``task_name=""`` / ``task_meta={}`` is the legacy no-task seed;
        extra_tool_schemas mirrors bind()'s default resolution."""
        others = {
            "task_name": task_name,
            # Copy task lists so registered/live metadata cannot mutate the catalog.
            "tags": list(task_meta.get("tags", [])),
            "task_apps": list(task_meta.get("apps", [])),
            "apps": list(_MOBILE_WORLD_APPS),  # full launchable catalog / open_app enum
        }
        if "agent-user-interaction" in others["tags"]:
            others["exclude_reason"] = "ask_user"
        return LiteCUAMetadata(
            dims=("mobile", "use"),
            extra_tool_schemas=resolve_extra_tools(
                _EXTRA_TOOLS, tools=MobileworldTools, env_name="mobileworld",
            ),
            valid_actions=list(_SCHEMA_VALID_ACTIONS),
            others=others,
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendment: extra_tool_schemas resolved at bind().
        return dataclasses.replace(
            self._task_metadata(self._task_id, self._task_meta),
            valid_actions=list(self._schema_valid_actions),
            extra_tool_schemas=self._extra_tool_schemas,
        )

    @property
    def external_resource_id(self) -> str | None:
        # Falls back to the in-flight container so the drift reaper sees it
        # during the (minutes-long) boot window — same rationale as
        # androidworld's ``_pending_cf_future``-tracked acquire window.
        c = self._current_container if self._current_container is not None \
            else self._pending_container
        return c.name if c is not None else None

    # -- lifecycle -----------------------------------------------------------

    async def boot(self) -> None:
        """Acquire the container + server client. Idempotent; task-independent
        (reset calls this lazily before task setup)."""
        if self._client is not None:
            return
        loop = asyncio.get_event_loop()
        cf = _EXECUTOR.submit(self._acquire)
        self._pending_cf_future = cf
        container, client, (w, h) = await asyncio.wrap_future(cf, loop=loop)
        self._pending_cf_future = None
        self._pending_container = None
        self._current_container = container
        self._client = client
        self._screen_w, self._screen_h = w, h

    def _acquire(self) -> tuple[MobileWorldContainer, _Client, tuple[int, int]]:
        """Blocking spawn + readiness (runs in ``_EXECUTOR``)."""
        from lite.gym.utils.config.identity import EnvIdentity
        identity = getattr(self, "identity", None) or EnvIdentity()
        _check_image(self._image)
        factory = MobileWorldContainerFactory(
            image=self._image,
            task_id=self._task_id or None,
            session_id=identity.session_id,
            token_hash=identity.token_hash,
            server_port=identity.server_port,
        )
        container = factory.acquire()
        # Expose immediately: from here on the drift reaper must treat this
        # container as owned even though boot() hasn't returned yet.
        self._pending_container = container
        client = _Client(container.base_url)
        viewport = client.init()
        logger.info(
            "mobileworld container %s ready (viewport=%s)",
            container.name, viewport,
        )
        return container, client, viewport

    def _destroy_container(self) -> None:
        """Blocking teardown of the current container (runs in ``_EXECUTOR``).

        Nulls ``_current_container`` only AFTER ``destroy()`` returns (and
        only if still ours) — an eager null would blank
        ``external_resource_id`` while the container is still alive in
        docker, inviting the drift reaper to race our own teardown."""
        container = self._current_container
        self._client = None
        self._active_task = None
        if container is not None:
            try:
                container.destroy()
            except Exception as e:
                # Swallow-and-warn like the sibling envs' destroy_backend —
                # the drift reaper is the documented backstop for a failed
                # rm; propagating would abort the framework's cold reboot.
                logger.warning("mobileworld: container destroy failed: %s", e)
            finally:
                if self._current_container is container:
                    self._current_container = None

    def backend_alive(self) -> bool:
        # Usable-ness is the server CLIENT binding, not the raw container —
        # the seam attr can point at a container whose ``_acquire`` never
        # finished ``client.init()``.
        return self._client is not None

    async def tear_down_task(self) -> None:
        """Clear the previous episode's server-side controller state
        (the ``/task/init`` inside :meth:`init_task` does the rest).
        Tolerates "no previous task" — including a recycled container still
        holding a foreign task before the next reset."""
        if self._active_task is None:
            return
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                _EXECUTOR,
                partial(self._client.task_tear_down, self._active_task),
            )
        except Exception as e:
            logger.warning("mobileworld: previous-task tear_down failed: %s", e)
        self._active_task = None

    async def reset_to_pristine(self) -> bool:
        # The pristine rollback (AVD snapshot reload + app-backend restart)
        # is fused into the server's ``/task/init`` — there is no separate
        # rollback RPC. Report the backend reusable; init_task owns the
        # reload and its own wedged-device recovery.
        return True

    async def init_task(self) -> None:
        """POST ``/task/init`` (snapshot reload = pristine device, then task
        setup) and fetch the authoritative goal.

        ``/task/init`` failing usually means the device/backends wedged —
        retry once on a fresh container before giving up (destroy + boot,
        mirroring the framework's failed-rollback recovery)."""
        loop = asyncio.get_event_loop()
        for attempt in (1, 2):
            if self._client is None:
                await self.boot()
                # Fresh backend on the retry path — restart the reuse budget
                # (reset_with_recycle only zeroes it for ITS OWN boots).
                self._recycle_count = 0
            try:
                await loop.run_in_executor(
                    _EXECUTOR, partial(self._client.task_init, self._task_id),
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(
                    "/task/init failed on %s (%s); re-spawning container",
                    self._current_container.name if self._current_container else "?", e,
                )
                await self.destroy_backend()
        self._active_task = self._task_id
        self._step_count = 0
        self._terminated = False

        # Authoritative goal from the server — some tasks interpolate
        # episode state (dates) into it; tasks.json is the static fallback.
        try:
            self._instruction = await loop.run_in_executor(
                _EXECUTOR, partial(self._client.task_goal, self._task_id),
            )
        except Exception as e:
            logger.warning("task_goal fetch failed (%s); using tasks.json goal", e)

    async def destroy_backend(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_EXECUTOR, self._destroy_container)

    async def reset(self) -> LiteEnvObservation:
        loop = asyncio.get_event_loop()

        # Defensive: a previous reset cancelled mid-boot without close().
        if self._pending_cf_future is not None:
            self._spawn_background_cleanup(self._pending_cf_future)
            self._pending_cf_future = None

        await self.reset_with_recycle()

        await asyncio.sleep(1.0)  # let the post-init UI settle
        screenshot = await loop.run_in_executor(
            _EXECUTOR, self._client.screenshot_png,
        )
        return LiteEnvObservation(
            image=screenshot, text=self._instruction,
        )

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        runtime_metadata = dataclasses.replace(
            metadata, valid_actions=self._valid_actions,
        )
        actions, ingress_errors = prepare_env_tool_calls(actions, runtime_metadata)
        loop = asyncio.get_event_loop()
        terminated = False
        # Model-emitted calls that ENDED the episode. They get no continuation
        # observation: devs/migration/verify.py forbids a tool result for a
        # terminal call. Keyed on the env-local ``action["call_id"]`` and NOT
        # on ``result_call_id``, because an INTERNAL finish call has no local
        # ``call_id`` and the loop-detect wrapper's injected ``terminate``
        # carries the intercepted NON-finish model call's id as
        # ``_result_call_id`` -- that call must still be answered.
        terminal_call_ids: set[str] = set()
        executed_actions: list[LiteExecutedAction] = []
        obs_notes: list[str] = []
        step_screenshots: list[bytes] = []
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)

        for action, result_call_id in actions:
            name = action["name"]
            args = action["arguments"]

            tool_feedback, tool_reason = standalone_tool_call_feedback_with_reason(
                action,
                _KNOWN_STANDALONE_TOOL_NAMES,
                metadata.extra_tool_schemas,
            )
            if tool_feedback is not None:
                if result_call_id:
                    append_feedback(action_errors, result_call_id, tool_feedback)
                executed_actions.append({
                    "call": "noop",
                    "args": {
                        "name": name,
                        "reason": tool_reason,
                    },
                })
                # The shared layer already decided the SURFACE: ``current`` is a
                # GUI slot the model got wrong, which owes a frame even though
                # nothing ran (R2a + R3); ``error_only`` is a text tool, which
                # owes none. Read that decision rather than re-deriving it.
                if tool_feedback.carrier == "current":
                    png = await loop.run_in_executor(
                        _EXECUTOR, self._client.screenshot_png,
                    )
                    if png is not None:
                        step_screenshots.append(png)
                continue

            if name in LiteFinishToolSet.get_tool_names():
                # ``response`` carries the answer text for information-
                # retrieval tasks; upstream evaluators read it from the
                # controller's interaction_cache, populated by the ANSWER
                # action. Without this write, ALL Q&A tasks score 0.
                if name == "response":
                    answer_text = args.get("text", "")
                    if answer_text:
                        try:
                            await loop.run_in_executor(
                                _EXECUTOR,
                                partial(self._client.step, {
                                    "action_type": "answer",
                                    "text": answer_text,
                                }),
                            )
                            executed_actions.append({
                                "call": "json_action.answer",
                                "args": {"text": answer_text},
                            })
                        except Exception as e:
                            record_tool_execution_error(
                                action_errors,
                                result_call_id,
                                f"response side effect failed: {e}",
                                action_name=name,
                            )
                            logger.warning("answer action failed: %s", e)
                terminated = True
                if action.get("call_id"):
                    terminal_call_ids.add(action["call_id"])
                logger.info("Agent terminated (%s): %s", name, args)
                break

            invalid_action = invalid_action_message(
                action, runtime_metadata.valid_actions,
            )
            if invalid_action:
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(invalid_action)
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": invalid_action},
                })
                continue

            unsupported_action = unsupported_env_action_message(
                name,
                _SUPPORTED_ACTIONS,
            )
            if unsupported_action:
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(unsupported_action)
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": unsupported_action},
                })
                continue

            try:
                calls, note = await self._dispatch_action(name, args)
            except Exception as e:
                record_tool_execution_error(action_errors, result_call_id, e, action_name=name)
                logger.warning("Action %s execution failed: %s", name, e)
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": str(e)},
                })
                continue

            executed_actions.extend(calls)
            # An action-batch call can expand into several actions sharing one
            # result_call_id (see prepare_env_tool_calls); the first of them to
            # report an error owns the feedback for that call_id.
            if result_call_id and result_call_id not in action_errors:
                for call in calls:
                    call_args = call.get("args", {})
                    if call.get("call") == "noop" and call_args.get("reason"):
                        if call_args.get("is_error"):
                            record_model_action_error(
                                action_errors,
                                result_call_id,
                                ValueError(str(call_args["reason"])),
                                action_name=name,
                            )
                        else:
                            action_errors[result_call_id] = error_only_feedback(
                                str(call_args["reason"])
                            )
                        break
            if note:
                obs_notes.append(note)
            if any(call.get("call") == "noop" for call in calls):
                # Downgraded at the device boundary: nothing ran, so there is
                # no post-action state to record. The batch is NOT aborted
                # here, so the following actions still get their own frames.
                continue

            # Let the device UI settle, then take ONE frame for THIS action.
            # The capture must follow the settle sleep or the frame records
            # the pre-action screen.
            if self._post_action_delay > 0:
                await asyncio.sleep(self._post_action_delay)
            png = await loop.run_in_executor(
                _EXECUTOR, self._client.screenshot_png,
            )
            if png is not None:
                step_screenshots.append(png)

        if not step_screenshots:
            # Nothing executed (empty batch, terminal-only call, or every call
            # rejected). The turn still owes the model a current observation.
            png = await loop.run_in_executor(
                _EXECUTOR, self._client.screenshot_png,
            )
            if png is not None:
                step_screenshots.append(png)

        self._step_count += 1
        truncated = not terminated and self._step_count >= self._max_steps

        # Evaluate at episode end. Upstream parity: MobileWorld scores the
        # final device/backend state regardless of WHY the episode ended, so
        # truncated episodes get their real score too (unlike androidworld's
        # terminate-only reward).
        reward = None
        info: dict[str, Any] = {EXECUTED_ACTIONS_INFO_KEY: executed_actions}
        if terminated or truncated:
            try:
                score, reason = await loop.run_in_executor(
                    _EXECUTOR, partial(self._client.task_eval, self._task_id),
                )
                reward = score
                info["eval_reason"] = reason
            except Exception as e:
                logger.error("Evaluation failed: %s", e)
                reward = 0.0

        self._terminated = terminated
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            ),
            ordered_call_ids=result_call_ids,
            continue_call_ids=[
                call_id for call_id in result_call_ids
                if call_id not in terminal_call_ids
            ],
            images=step_screenshots,
            text="\n".join(obs_notes) if obs_notes else None,
            feedback=action_errors,
        )

    async def close(self) -> None:
        # Reap a container from a reset() cancelled mid-boot.
        if self._pending_cf_future is not None:
            cf = self._pending_cf_future
            self._pending_cf_future = None
            self._spawn_background_cleanup(cf)

        loop = asyncio.get_event_loop()
        if self._client is not None and self._active_task is not None:
            try:
                await loop.run_in_executor(
                    _EXECUTOR,
                    partial(self._client.task_tear_down, self._active_task),
                )
            except Exception as e:
                logger.warning("tear_down on close failed: %s", e)
            self._active_task = None
        await loop.run_in_executor(_EXECUTOR, self._destroy_container)

    # 60s timeout (vs siblings' 30): a DinD acquire includes a nested dockerd boot.
    def _spawn_background_cleanup(self, cf_future: Any, timeout: float = 60.0) -> None:
        """Reap the container produced by a cancelled boot via the shared
        background cleanup shell: wait for ``_acquire`` (it holds the docker-create slot) and
        destroy its container; if it never finishes, destroy the
        ``_pending_container`` the factory already exposed (the documented
        pending-attr exception to the capture-as-local rule)."""
        def _ok(res: Any) -> None:
            container, _client, _viewport = res
            container.destroy()
            logger.info("mobileworld: background cleanup destroyed leaked container")

        def _fail(e: BaseException) -> None:
            pending = self._pending_container
            self._pending_container = None
            if pending is not None:
                pending.destroy()
            logger.warning("Background cleanup finished with: %s", e)

        spawn_background_destroy(
            cf_future, on_result=_ok, on_failure=_fail, timeout=timeout,
            thread_name="mobileworld-leak-cleanup",
        )

    # -- action dispatch ------------------------------------------------------

    async def _dispatch_action(
        self, name: str, args: dict[str, Any]
    ) -> tuple[list[LiteExecutedAction], str | None]:
        """Translate one CUA-Lite mobile action into an upstream JSONAction
        and execute it. Returns (executed-calls list, observation note)."""
        loop = asyncio.get_event_loop()
        if name == "ask_user":
            question = args.get("text", "")
            resp = await loop.run_in_executor(
                _EXECUTOR,
                partial(self._client.step, {
                    "action_type": "ask_user", "text": question,
                }),
            )
            reply = str(resp.get("result", ""))
            call: LiteExecutedAction = {
                "call": "json_action.ask_user", "args": {"text": question},
            }
            return [call], f'The user replied: "{reply}"'

        if name == "system_button" and args.get("button") in ("Menu", "Recent"):
            button = args["button"]
            executed = await loop.run_in_executor(
                _EXECUTOR, self._exec_system_button_adb, button,
            )
            return [executed], None

        try:
            action = self._translate_action(name, args)
        except MODEL_ACTION_ERROR_TYPES as e:
            logger.error("Action %s failed: %s", name, e)
            return [{
                "call": "noop",
                "args": {"name": name, "reason": str(e), "is_error": True},
            }], None
        if action is None:
            if name == "screenshot":
                # Nothing to dispatch; the step loop takes this action's frame.
                return [], None
            return [{
                "call": "noop",
                "args": {"name": name, "reason": "unknown action"},
            }], None

        await loop.run_in_executor(
            _EXECUTOR, partial(self._client.step, action),
        )
        return [{
            "call": f"json_action.{action['action_type']}",
            "args": {k: v for k, v in action.items() if v is not None},
        }], None

    def _translate_action(
        self, name: str, args: dict[str, Any]
    ) -> dict[str, Any] | None:
        """CUA-Lite mobile action → upstream JSONAction dict.

        ``swipe``/``drag`` both map to the precise ``drag`` JSONAction
        (upstream's ``swipe`` only takes an enum direction, ``drag`` honors
        exact start/end coordinates).
        """
        if name == "tap":
            x, y = self._to_screen_pixels(args.get("coordinate"))
            if args.get("clicks", 1) >= 2:
                return {"action_type": "double_tap", "x": x, "y": y}
            return {"action_type": "click", "x": x, "y": y}

        elif name == "long_press":
            x, y = self._to_screen_pixels(args.get("coordinate"))
            if "duration" in args:
                # MobileWorld's upstream JSONAction has no duration field for
                # long_press, so duration is bounded/validated only.
                _duration_seconds(args, "long_press", default=1.0)
            return {"action_type": "long_press", "x": x, "y": y}

        # No ``double_tap`` branch: the canonical mobile spelling is
        # ``tap(clicks=2)``, handled above, so a bare ``double_tap`` falls through
        # to the unknown-action noop. (``action_type: "double_tap"`` on the wire is
        # the upstream JSONAction name ``tap(clicks>=2)`` emits, not a cua-lite
        # action name.)
        elif name == "type":
            return {"action_type": "input_text", "text": args.get("text", "")}

        elif name in ("swipe", "drag"):
            sx, sy = self._to_screen_pixels(
                args.get("start_coordinate", [500, 500])
            )
            ex, ey = self._to_screen_pixels(args.get("coordinate", [500, 500]))
            if "duration" in args:
                # The precise drag JSONAction has no duration field; this is a
                # validation-only contract until upstream exposes one.
                _duration_seconds(args, name, default=0.4)
            return {
                "action_type": "drag",
                "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
            }

        elif name == "system_button":
            btn = args.get("button", "")
            action_type = {
                "Home": "navigate_home",
                "Back": "navigate_back",
                "Enter": "keyboard_enter",
            }.get(btn)
            if action_type:
                return {"action_type": action_type}
            raise ValueError(
                f"system_button: unknown button {btn!r}; expected one of "
                "['Back', 'Enter', 'Home', 'Menu', 'Recent']"
            )

        elif name == "open_app":
            return {"action_type": "open_app", "app_name": args.get("app_name", "")}

        elif name == "wait":
            if "duration" in args:
                # Upstream wait carries no duration value.
                _duration_seconds(args, "wait", default=1.0, allow_zero=True)
            return {"action_type": "wait"}

        elif name == "screenshot":
            return None

        else:
            logger.warning("Unknown action: %s(%s), skipping", name, args)
            return None

    def _exec_system_button_adb(self, button: str) -> LiteExecutedAction:
        if self._current_container is None:
            raise RuntimeError("mobileworld container is not available")
        keycodes = {"Menu": 82, "Recent": 187}
        keycode = keycodes[button]
        self._current_container.adb_shell(
            "input", "keyevent", str(keycode), timeout=15.0,
        )
        return {
            "call": "adb.input.keyevent",
            "args": {"keycode": keycode, "button": button},
        }

    def _to_screen_pixels(
        self, coordinate: list[int] | None
    ) -> tuple[int, int]:
        """Convert [0, 1000] normalized coords to device pixel coords.

        clamp=False preserves this env's legacy no-clamp behavior until its
        coordinate contract is validated for canonical clamping."""
        return norm_to_pixel(coordinate, self._screen_w, self._screen_h,
                             clamp=False, on_malformed="raise")


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

def _load_tasks_json() -> dict[str, dict[str, Any]]:
    """Load the static task list dumped from the docker image.

    See ``scripts/tasks.sh`` — re-run when the MOBILEWORLD_SHA pin in
    ``docker/Dockerfile`` changes, and check the result in.
    """
    tasks_path = Path(__file__).parent / "data" / "tasks.json"
    if not tasks_path.is_file():
        raise FileNotFoundError(
            f"{tasks_path} not found. Re-generate with:\n"
            f"  bash lite/gym/envs/mobileworld/scripts/tasks.sh\n"
            "(requires cua-lite/mobileworld:latest — run scripts/install.sh first)."
        )
    with open(tasks_path) as f:
        return json.load(f)


_TASKS = _load_tasks_json()

# Env-wide make defaults from default.yaml make_kwargs.
registry.set_env_make_kwargs("mobileworld", CFG.make_kwargs)

# MobileWorld is an eval benchmark with deterministic tasks (no param
# randomization), so every task registers in the EVAL split only.
# ``agent-mcp`` tasks are skipped: MCP tool integration is not supported yet.
for _task_name, _task_meta in _TASKS.items():
    if "agent-mcp" in _task_meta["tags"]:
        continue
    register(
        key=f"mobileworld@{_task_name}",
        entry_point=partial(MobileWorldEnv, task_id=_task_name),
        split="eval",
        metadata=MobileWorldEnv._task_metadata(_task_name, _task_meta),
    )


class MobileWorldServices(ContainerServices):
    """Env-server capabilities for mobileworld: docker-image freshness check
    (``ensure``) + per-container reconciliation (``live_ids``/``reap``
    inherited from :class:`ContainerServices`; the per-instance container name
    is emitted by the env's ``EnvServerResource`` mixin)."""

    def ensure(self, env_id: str) -> None:
        _ensure_services(env_id)

    def health(self, env_id: str) -> None:
        _health_check(env_id)


register_services("mobileworld", MobileWorldServices())
register_family("mobileworld", BackendFamily.DEDICATED)
