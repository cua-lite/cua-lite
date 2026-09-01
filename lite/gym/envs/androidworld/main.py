"""androidworld — CUA-Lite gym wrapper for Google's AndroidWorld benchmark.

Wraps AndroidWorld (116 multi-step mobile tasks across 20 apps) as a CUA-Lite
gym environment with LiteMobileActionSpace. Tasks are evaluated by checking
real device state (SQLite databases, file system, app state) rather than UI
matching.

Task registry auto-loads from the installed androidworld package at import
time. Raises ImportError if the package is not installed.

Prerequisites:
  - uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh
  - cua-lite/androidworld docker image built via this env's
    ``scripts/install.sh``. Each ``gym.make`` spawns one container that
    hosts the emulator + an in-container HTTP env-server; the host
    process drives it via ``_RemoteEnv`` / ``_RemoteTask`` proxies (see
    :mod:`lite.gym.envs.androidworld.container`).

Usage:
    uv run python -c "import lite.gym as gym; print(gym.registry.task_ids('androidworld'))"
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import hashlib
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import pixel_to_norm
from lite.core.tools.action_space.duration import (
    ACTION_SCHEMA_DURATION_CAPS_SECONDS,
)
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.extra_tools import (
    LiteFinishToolSet,
    make_open_app_tool,
)
from lite.core.tools.schemas import BaseTools
from lite.gym.container import spawn_background_destroy
from lite.gym.envs.androidworld.container import (
    AndroidWorldContainer,
    AndroidWorldContainerFactory,
    _block_overlapping_retry,
)
from lite.gym.registry import register, registry
from lite.gym.remote.reaper import ContainerServices
from lite.gym.services import (
    EnvServerPoolable,
    EnvServerResource,
    register_services,
)
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
from lite.gym.utils.backend.rpc import RemoteRPC
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    ToolErrorFeedback,
    append_feedback,
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
from lite.utils.image import encode_png

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# ANDROID_WORLD_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
#: Default step budget. ``None`` → reset() computes ``int(10*complexity)``
#: per task. ``bind()`` signature default.
_MAX_STEPS = CFG.env_kwargs["max_steps"]
#: Seconds to sleep before the post-step screenshot. ``bind()`` default.
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
#: Observation-text mode: ``none`` | ``a11y:pixel`` | ``a11y:norm``.
#: ``bind()`` signature default.
_OBSERVATION_TEXT = CFG.env_kwargs["observation_text"]
#: Default per-instance task seed. ``None`` → train-random params; the
#: EVAL split registers seed=42 explicitly. ``bind()`` signature default.
_SEED = CFG.env_kwargs["seed"]
#: Soft GUI-action surface default (``None`` = full GUI, ``[]`` = no GUI).
_VALID_ACTIONS_CONFIG = CFG.env_kwargs.get("valid_actions")
#: Opt-in extra-tool selection (list of tool names; ``[]`` = none). The
#: ROLLOUT/eval yaml enables ``open_app``; the default is OFF.
#: ``bind()`` signature default.
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]
#: Family-neutral spawn-image override (default base image). Read by the
#: ``register`` call below and by ``__init__``'s signature default.
_IMAGE = CFG.env_kwargs["image"]
# --- server_kwargs (per-deployment) ---
#: Which baked emulator instance to attach (the AVD name inside the docker
#: image). Per-deploy, so server_kwargs; operator override: ANDROID_AVD_NAME.
_AVD_NAME_DEFAULT = CFG.server_kwargs["avd_name"]
#: ThreadPoolExecutor size. Each live android emulator may hold up to 2
#: threads in the pool (acquire() + step/teardown); default 256 covers
#: ~128 emulators × 2 threads. Consumed by ``_EXECUTOR`` below.
_MAX_WORKERS = CFG.server_kwargs["max_workers"]
#: Max resets per container before forcing a destroy + cold-spawn.
#: Mirror of androidlab's same-named constant — bounds qemu / adb-daemon /
#: in-container python server (``docker/server.py``) drift across long
#: runs. ``reset_snapshot`` restores VM-layer state perfectly, but the
#: layers OUTSIDE the VM (qemu process state, adb daemon socket pool,
#: in-container python globals, /tmp logs) accumulate forever otherwise.
#: Default K=0 → every reset (after first cold-spawn) triggers destroy +
#: cold-spawn, eliminating cross-reset residue (e.g. launcher dock-icon
#: leakage from prior task's app usage stats — verified k_bench3 dock
#: comparison 2026-05-25). Wall-clock cost ~14% on contended host; CPU/
#: disk indistinguishable from K=20 (page cache absorbs image extract).
#: Set ``server_kwargs.max_resets_per_container>=1`` to re-enable snapshot
#: reuse if the cleanliness/throughput tradeoff is acceptable.
_MAX_RESETS_PER_CONTAINER = CFG.server_kwargs["max_resets_per_container"]
#: Number of destroy + cold-spawn attempts when /init returns 500.
#: Belt-and-suspenders for residual Android-system-services races even
#: after :meth:`AndroidWorldContainer._wait_until_android_ready` (which
#: probes PackageManager + AccessibilityManagerService before /init).
#: Under extreme host contention adb timing can still slip; 3 attempts
#: amortises the recovery cost (~30-60 s per container respawn) without
#: looping forever. Mirrors ``_acquire_emulator``'s ``max_attempts=3``.
_INIT_MAX_ATTEMPTS = CFG.server_kwargs["init_max_attempts"]
# ============================================================================


def _pixels_to_png(pixels: np.ndarray) -> bytes:
    """HxWx3 uint8 numpy array → raw PNG bytes."""
    from PIL import Image

    return encode_png(Image.fromarray(pixels))

logger = logging.getLogger(__name__)


def _format_ui_elements(
    ui_elements,
    screen_w: int, screen_h: int,
    coord_unit: str = "pixel",
) -> str:
    """Render visible interactable UI elements as a numbered list.

    Centres are emitted in the space the ACTION side reads, and there is one
    conversion for it — ``AndroidWorldEnv._to_screen_pixels``'s
    ``norm_to_pixel(coord, screen_w, screen_h)``:

      - ``"norm"``:  ``pixel_to_norm`` against the same dims, i.e. that
        conversion's exact inverse. A centre the model echoes back as
        ``coordinate`` therefore lands on the element it names.
      - ``"pixel"``: the device-native pixels ``_to_screen_pixels`` *outputs*
        (identity — a11y bboxes are already native). The env's own
        ``coordinate`` is always ``[0, 1000]``, so this spelling round-trips
        only for an agent whose adapter converts the model's pixels back to
        ``[0, 1000]`` against these same device dims (the API-agent configs
        that select ``a11y:pixel``); a model that emits ``[0, 1000]`` itself
        must be given ``a11y:norm``.
    """
    lines: list[str] = []
    for i, el in enumerate(ui_elements):
        if not el.bbox_pixels:
            continue
        if el.is_visible is False:
            continue
        x_min, y_min = el.bbox_pixels.x_min, el.bbox_pixels.y_min
        x_max, y_max = el.bbox_pixels.x_max, el.bbox_pixels.y_max
        if x_max <= 0 or y_max <= 0 or x_min >= screen_w or y_min >= screen_h:
            continue
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        if coord_unit == "norm":
            cx, cy = pixel_to_norm(cx, cy, screen_w, screen_h)
        label = (el.text or el.content_description or "").strip()
        cls = (el.class_name or "").rsplit(".", 1)[-1]
        flags: list[str] = []
        if el.is_clickable: flags.append("clickable")
        if el.is_editable: flags.append("editable")
        if el.is_checkable: flags.append("checkable")
        if el.is_checked: flags.append("checked")
        if el.is_selected: flags.append("selected")
        if el.is_scrollable: flags.append("scrollable")
        # Only keep elements that carry real information.
        if not label and not flags:
            continue
        flags_s = f" [{','.join(flags)}]" if flags else ""
        label_s = f'"{label}"' if label else "(unlabeled)"
        lines.append(f"[{i}] {label_s} center=({cx},{cy}) <{cls}>{flags_s}")
    return "\n".join(lines)


_EXECUTOR = ThreadPoolExecutor(
    max_workers=_MAX_WORKERS,
    thread_name_prefix="android-world-exec",
)

# At interpreter exit, force-cancel any pending futures and abandon running
# ones. Without this, ThreadPoolExecutor's own atexit hook calls
# shutdown(wait=True), which blocks forever if a worker is stuck inside
# task.tear_down() / env_launcher.* / a long-running grpc call -- which is
# exactly what we observed at concurrency=16 with 3/32 errored tasks. Our
# atexit runs in LIFO order BEFORE the executor's own atexit, so we win.
atexit.register(lambda: _EXECUTOR.shutdown(wait=False, cancel_futures=True))


def _to_json_safe(obj: Any) -> Any:
    """Recursively coerce ``obj`` into a JSON-serializable structure.

    androidworld tasks like Expense / Recipe return @dataclass instances
    inside their params dict; ``json.dumps`` chokes on them. Map dataclasses
    to dicts, fall back to ``repr()`` for anything else exotic.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_json_safe(dataclasses.asdict(obj))
    return repr(obj)

# ---------------------------------------------------------------------------
# App catalog
# ---------------------------------------------------------------------------
#
# The androidworld docker image ships with this exact set of apps
# (see ``lite/gym/envs/androidworld/docker/apps.sh`` and the upstream
# google-research/android_world README's "Apps" section). Surfaced via
# ``metadata.others["apps"]`` so action adapters that expose an
# ``open`` shortcut (currently mai_ui mobile) inject env-correct names
# into the model's "Available Apps" prompt line, and embedded as the
# ``app_name`` enum in the ``open_app`` extra tool below, so tool-schema
# agents (e.g. Qwen3-VL) see the same names. Byte-equal to the list
# baked into MAI-UI's SFT distribution at
# ``${CUA_LITE_REFERENCES_ROOT}/MAI-UI/src/prompt.py:47``.

_ANDROID_WORLD_APPS: list[str] = [
    "Camera", "Chrome", "Clock", "Contacts", "Dialer", "Files",
    "Settings", "Markor", "Tasks", "Simple Draw Pro",
    "Simple Gallery Pro", "Simple SMS Messenger", "Audio Recorder",
    "Pro Expense", "Broccoli APP", "OSMand", "VLC", "Joplin",
    "Retro Music", "OpenTracks", "Simple Calendar Pro",
]


# ---------------------------------------------------------------------------
# Extra tools
# ---------------------------------------------------------------------------
# Routed to AndroidWorld's native ``action_type=open_app`` via the
# ``elif name == "open_app"`` branch in the agent→bench action mapper.
# Exposed through ``metadata.extra_tool_schemas``. Qwen-family native
# ``mobile_use(action="open", text=...)`` and MAI-UI app-open surfaces
# canonicalize to this env-owned ``open_app`` boundary when configs opt in.

class AndroidworldTools(BaseTools):
    """What androidworld declares beyond the GUI surface: ``open_app``."""

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "open_app": make_open_app_tool(_ANDROID_WORLD_APPS),
    }


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = AndroidworldTools.get_tool_names() | LiteFinishToolSet.get_tool_names()

_SUPPORTED_ACTIONS = android_supported_actions()
_SCHEMA_VALID_ACTIONS = resolve_schema_valid_actions(
    _VALID_ACTIONS_CONFIG,
    env_name="androidworld",
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
# Config
# ---------------------------------------------------------------------------

@dataclass
class AndroidWorldTaskConfig:
    """Configuration for an AndroidWorld environment instance.

    Emulator ports + adb path are NOT exposed here — they're fixed
    constants baked into the docker image (see
    :mod:`lite.gym.envs.androidworld.container`).
    """

    emulator_setup: bool = False  # True on first run to install 20+ apps
    freeze_datetime: bool = True
    use_fake: bool = False  # Use a mock env for testing (no emulator needed)
    # NO display_resolution: the AVD render size is emulator-fixed (we can't change
    # it), and the screenshot is captured native (``pixels.shape``). Agent / a11y
    # coords are derived from that native size at runtime — see docs/envs.md.
    avd_name: str = os.environ.get("ANDROID_AVD_NAME", _AVD_NAME_DEFAULT)

    # Task-level (set per-task at registration)
    task_class_name: str = ""
    instruction_template: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Emulator acquisition
# ---------------------------------------------------------------------------

def _acquire_emulator(
    avd_name: str = "lite_avd_androidworld",
    task_id: str | None = None,
    session_id: str | None = None,
    token_hash: str | None = None,
    server_port: int | None = None,
    image: str | None = None,
    env_id: str = "androidworld",
) -> AndroidWorldContainer:
    """Spawn one sibling docker container (cua-lite/androidworld:latest) for
    this env. Apps are pre-installed in the AVD at image build time, so
    there's no per-host setup step. ``session_id`` + ``token_hash`` +
    ``server_port`` flow into the container name so the env server's
    ``DELETE /instances`` can scope by session/token, AND so two same-token
    env-server instances on the same host (different ports) stay
    mutually isolated.
    """
    factory = AndroidWorldContainerFactory(
        avd_name=avd_name,
        task_id=task_id,
        session_id=session_id,
        token_hash=token_hash,
        server_port=server_port,
        env_id=env_id,
        # image override (env-kwarg); omit when None so the factory default applies
        **({"image": image} if image else {}),
    )
    return factory.acquire()

# ---------------------------------------------------------------------------
# Fake environment for testing (no emulator required)
# ---------------------------------------------------------------------------

class _FakeAsyncEnv:
    """Minimal mock of androidworld's AsyncEnv for testing."""

    def __init__(self):
        # Mock device-native screenshot size (no real emulator in tests).
        self._w, self._h = (1080, 2400)
        self._closed = False

    def reset(self, go_home: bool = False):
        from types import SimpleNamespace

        pixels = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        return SimpleNamespace(pixels=pixels, ui_elements=[], forest=None)

    def get_state(self, wait_to_stabilize: bool = False):
        return self.reset()

    def execute_action(self, action) -> None:
        pass

    @property
    def device_screen_size(self) -> tuple[int, int]:
        return (self._w, self._h)

    @property
    def logical_screen_size(self) -> tuple[int, int]:
        return (self._w, self._h)

    def close(self) -> None:
        self._closed = True

    def hide_automation_ui(self) -> None:
        pass

    @property
    def interaction_cache(self) -> str:
        return getattr(self, "_interaction_cache", "")

    @interaction_cache.setter
    def interaction_cache(self, value: str) -> None:
        self._interaction_cache = value

    @property
    def controller(self):
        return None

class _FakeTaskEval:
    """Minimal mock of a TaskEval for testing."""

    def __init__(self, params: dict[str, Any] | None = None):
        self._params = params or {}
        self._initialized = False

    @property
    def goal(self) -> str:
        return "Fake task for testing."

    @property
    def app_names(self) -> tuple[str, ...]:
        return ("fake_app",)

    @property
    def complexity(self) -> float:
        return 1.0

    def initialize_task(self, env) -> None:
        self._initialized = True

    def is_successful(self, env) -> float:
        return 1.0

    def tear_down(self, env) -> None:
        self._initialized = False

    @classmethod
    def generate_random_params(cls, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            rng_state = random.getstate()
            try:
                random.seed(seed)
                return {"fake_value": random.randint(0, 1_000_000), "seed": seed}
            finally:
                random.setstate(rng_state)
        return {"fake_value": random.randint(0, 1_000_000)}


# ---------------------------------------------------------------------------
# In-container env-server proxies
# ---------------------------------------------------------------------------
# ``_create_env`` hands the AndroidWorldEnv code a ``_RemoteEnv`` instead
# of an AsyncAndroidEnv, and a ``_RemoteTaskFactory`` wrapping the
# TaskEval class. The reset/step/close orchestration logic doesn't care:
# every env/task method call is redirected over HTTP-RPC to the
# in-container server (docker/server.py).
#
# Why proxies instead of running AndroidWorldEnv-in-container: the host
# tracks step budget, history, action dispatch, and the LiteBaseEnv
# contract — all RL-loop bookkeeping that doesn't need to be inside the
# container. Only the emulator-adjacent methods (get_state,
# execute_action, task.initialize_task etc.) must execute in-container
# so a11y_grpc_wrapper's reverse gRPC stays in the container's network
# namespace (rootless docker's --disable-host-loopback blocks the
# host-side server otherwise).


class _RemoteRPC(RemoteRPC):
    """Tiny HTTP-RPC client used by the container-mode proxies.

    Bodies and responses are pickled Python objects (so we don't lose
    protobuf / numpy fidelity). The hot-path response handling is one
    request → one unpickle; no JSON-schema bookkeeping. The shared
    :class:`lite.gym.utils.backend.rpc.RemoteRPC` body is reused; androidworld
    opts into the one-retry-on-transient behaviour.

    ``ConnectionRefusedError`` retry (one attempt, ~2 s wait): the
    in-container ``server.py`` process is sometimes briefly unreachable
    mid-rollout — uvicorn worker recycling, host TCP backlog overflow
    under fan-in, or an adb subprocess that briefly blocked the event
    loop. A single retry after a short sleep recovers these without
    giving up the whole episode. If the container is truly dead the
    second attempt fails the same way and the exception propagates
    normally (the caller — env-server — then returns 500 to the rollout
    client). Doesn't retry on ``HTTPError`` (those are application-level
    rejects that need different handling at the call site — see
    ``_RemoteEnv.reset`` 409 path and ``_RemoteTask.__init__`` 409 path).
    """

    def __init__(self, base_url: str, timeout: float = 180.0):
        super().__init__(base_url, timeout, retries=1)


class _RemoteControllerEnv:
    """Stub of ``env.controller.env`` — the inner AndroidEnv that
    ``adb_utils`` functions on the host call ``execute_adb_call`` on.
    Forwards each call to the in-container server."""

    def __init__(self, rpc: _RemoteRPC):
        self._rpc = rpc

    def execute_adb_call(self, adb_request: Any) -> Any:
        return self._rpc.post("/env/execute_adb_call", body=adb_request)

    def attempt_enable_networking(self) -> None:
        self._rpc.post("/env/attempt_enable_networking")


class _RemoteController:
    """Stub of ``env.controller`` exposing only the ``env`` attribute
    that the host code actually reaches (via adb_utils)."""

    def __init__(self, rpc: _RemoteRPC):
        self.env = _RemoteControllerEnv(rpc)


class _RemoteEnv:
    """Proxy to env_launcher's ``AsyncAndroidEnv`` running inside the
    androidworld container.

    Implements the slice of the AsyncAndroidEnv API that the host's
    ``AndroidWorldEnv.reset/step/close`` orchestration touches:
    ``reset(go_home)`` / ``get_state()`` / ``execute_action(action)`` /
    ``interaction_cache`` (setter) / ``hide_automation_ui()`` /
    ``close()`` / ``controller.env`` for adb_utils.
    """

    def __init__(self, rpc: _RemoteRPC, init_body: dict[str, Any] | None = None):
        self._rpc = rpc
        self.controller = _RemoteController(rpc)
        # interaction_cache is a *write-only* attribute on the host side
        # (only used to set the Q&A answer that androidworld's evaluator
        # reads). Reads aren't part of the host's surface, so we store
        # the last-set value here purely for debug introspection.
        self._interaction_cache_local: Any = None
        # Cached /init params so :meth:`reset` can re-initialize the
        # in-container env if it returns 409 "env not initialized". The
        # in-container global ``_env`` should survive between /init and
        # /env/reset, but under high concurrency (n ≥ 96) we observed
        # ~20 % reset() failures with that error — likely a uvicorn /
        # supervisord interaction with `_env=None` after some recovery
        # path. The defensive retry is also useful for transient
        # container-server restarts that future hardening may add.
        self._init_body = init_body

    @property
    def interaction_cache(self) -> Any:
        return self._interaction_cache_local

    @interaction_cache.setter
    def interaction_cache(self, value: Any) -> None:
        self._interaction_cache_local = value
        self._rpc.post("/env/set_interaction_cache", body={"text": value})

    def reset(self, go_home: bool = True) -> None:
        try:
            self._rpc.post("/env/reset", body={"go_home": go_home})
        except RuntimeError as e:
            if "409" not in str(e) or "not initialized" not in str(e):
                raise
            if self._init_body is None:
                # No cached init params (legacy construction); re-raise
                # without retry. Callers with init_body wired through get
                # the recovery path; refresh-path callers do not.
                raise
            # In-container ``_env`` is None — re-issue /init (idempotent
            # there: a second /init when _env is already set returns
            # already_initialized) and retry once. If init itself fails,
            # the original RuntimeError is what surfaces.
            logger.warning(
                "env/reset returned 409 'not initialized'; re-running /init "
                "and retrying once",
            )
            self._rpc.post("/init", body=self._init_body)
            self._rpc.post("/env/reset", body={"go_home": go_home})

    def get_state(self) -> Any:
        return self._rpc.post("/env/get_state")

    def execute_action(self, action: Any) -> None:
        self._rpc.post("/env/execute_action", body=action)

    def hide_automation_ui(self) -> None:
        # The in-container server already calls hide_automation_ui()
        # during /init, so this is a no-op when the host code asks for
        # it again during _create_env (kept callable for interface
        # parity with the AsyncAndroidEnv API).
        pass

    def close(self) -> None:
        self._rpc.post("/close")


class _RemoteTask:
    """Proxy for an ``android_world.task_evals`` TaskEval instance.

    Constructed in-container via ``/task/load``; subsequent
    ``initialize_task`` / ``is_successful`` / ``tear_down`` calls
    forward to the in-container instance over HTTP. Static metadata
    (``goal`` / ``complexity`` / ``app_names`` / ``name``) is captured
    from the load response so host-side reads stay zero-RTT.
    """

    def __init__(
        self,
        rpc: _RemoteRPC,
        task_class_name: str,
        params: dict[str, Any],
    ):
        self._rpc = rpc
        load_body = {
            "task_class_name": task_class_name,
            "params": params,
        }
        try:
            info = rpc.post("/task/load", body=load_body)
        except RuntimeError as e:
            # 409 'no stashed params' means the in-container
            # ``_pending_params`` was cleared between the host's
            # ``/task/generate_params`` and this ``/task/load`` —
            # observed under contention (concurrent in-container
            # ``/task/tear_down`` races, /init retry-driven respawn
            # where a fresh container's stash is empty by definition,
            # …). Recover by re-stashing via ``/task/generate_params``
            # and retrying ``/task/load`` once. Re-stash uses the
            # ``seed`` already baked into ``params`` (saved by the
            # original ``/task/generate_params`` call), so the reload
            # produces an identical task instance.
            if "returned 409" not in str(e) or "no stashed params" not in str(e):
                raise
            seed = params.get("seed") if isinstance(params, dict) else None
            logger.warning(
                "/task/load 409 (no stashed params) for %s; re-running "
                "/task/generate_params with seed=%s and retrying once",
                task_class_name, seed,
            )
            rpc.post("/task/generate_params", body={
                "task_class_name": task_class_name,
                "seed": seed,
            })
            info = rpc.post("/task/load", body=load_body)
        self.goal: str = info["goal"]
        self.complexity: float = info["complexity"]
        self.app_names: tuple[str, ...] = tuple(info["app_names"])
        self.name: str = info["name"]

    def initialize_task(self, env: Any) -> None:
        self._rpc.post("/task/initialize")

    def is_successful(self, env: Any) -> float:
        return float(self._rpc.post("/task/is_successful")["reward"])

    def tear_down(self, env: Any) -> None:
        self._rpc.post("/task/tear_down")


class _RemoteTaskFactory:
    """Stand-in for a TaskEval class. The real class lives only inside
    the docker image; on host we identify it by name string + send
    ``__call__`` / ``generate_random_params`` to the container.

    ``AndroidWorldEnv.reset()`` keeps calling ``self._task_class(params)``
    and ``self._task_class.generate_random_params(seed=…)`` as if it
    held the real class. Both go through HTTP RPC instead.
    """

    def __init__(self, task_class_name: str, rpc: _RemoteRPC):
        self._task_class_name = task_class_name
        self._rpc = rpc
        self.__name__ = task_class_name

    def __call__(self, params: dict[str, Any]) -> _RemoteTask:
        return _RemoteTask(self._rpc, self._task_class_name, params)

    def generate_random_params(self, seed: int | None = None) -> dict[str, Any]:
        """Request params from the in-container ``task_class.generate_random_params``.
        When ``seed`` is not None, the container runs the call under a
        scoped ``random.seed(seed)`` (save/restore around it). Same
        semantics as the old host-side path, executed remotely so the
        host doesn't need the real class.
        """
        return self._rpc.post("/task/generate_params", body={
            "task_class_name": self._task_class_name,
            "seed": seed,
        })


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _check_image(tag: str = _IMAGE, *, env_id: str = "androidworld") -> None:
    from lite.gym.utils.backend.docker import require_image_present

    require_image_present(image_for(env_id, tag=tag))


def _ensure_services(env_id: str) -> None:
    """env-server startup hook: verify the docker image is built + fresh
    (rebuilt since its sources changed) before
    accepting traffic. Symmetric with androidlab's same-named hook.
    Image absence surfaces as ``EnvDepsMissingError`` → ``available:
    false`` on ``GET /envs?expand=metadata`` and HTTP 501 on the first
    ``POST /instances`` (instead of a cryptic gym.make failure mid-rollout).
    """
    _check_image()


_HEALTH_CHECK = CachedEnvDepsHealthCheck(_ensure_services)


def _health_check(env_id: str) -> None:
    _HEALTH_CHECK(env_id)


class AndroidWorldEnv(EnvServerPoolable, EnvServerResource):
    """CUA-Lite wrapper around Google's AndroidWorld benchmark.

    Each episode:
      1. reset() — connects to emulator, instantiates task with random params,
         initializes device state, takes screenshot
      2. step() — translates CUA-Lite mobile actions to AndroidWorld JSONActions,
         executes on emulator, returns screenshot
      3. On termination — calls task.is_successful() for reward (truncation → 0.0)
      4. close() — tears down task, optionally closes connection
    """

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = AndroidworldTools
    ENV_ID: ClassVar[str] = "androidworld"

    def __init__(
        self, *,
        config: AndroidWorldTaskConfig,
        image: str = _IMAGE,
        task_id: str = "",
        max_steps: int | None = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        observation_text: str = _OBSERVATION_TEXT,
        seed: int | None = _SEED,
        valid_actions: list[str] | None = _VALID_ACTIONS_CONFIG,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
    ) -> None:
        """Construct emulator-shape fields, then bind task/soft state.

        ``config`` is required because the emulator container's avd_name / etc.
        live inside it; those fields are identical across tasks that share one
        backend shape.
        """
        # Family-neutral spawn-image override (env-kwarg, same contract
        # as the osworld env): boot a derived image instead of the
        # default. None = default.
        self._image = image
        self._config = config
        # Emulator state holders (lazy-populated by boot()).
        self._env = None  # androidworld AsyncEnv
        self._task = None  # Current _RemoteTask instance
        self._task_class: Any = None  # factory wrapper set by _create_env
        self._screen_w = 0
        self._screen_h = 0
        #: THE container-handle attr: what ``external_resource_id`` reads. Assigned
        #: AT ACQUIRE inside ``_create_env`` (right after the factory returns,
        #: before /init) so the drift reaper sees every live container for its
        #: whole life; nulled only after ``destroy()`` completes
        #: (``destroy_backend``). Recycle counters/gate live on the framework
        #: (``_recycle_count`` / ``_recycle_first_done``).
        self._current_container: AndroidWorldContainer | None = None
        # Track in-progress executor future so close() can clean up
        # emulators that were created while reset() was being cancelled.
        self._pending_cf_future: Any = None  # concurrent.futures.Future
        #: Recycle cap for reset_with_recycle — counts IN-PLACE REUSES
        #: (successful snapshot reloads); the operator knob stays
        #: server_kwargs.max_resets_per_container.
        self._max_resets_per_container = _MAX_RESETS_PER_CONTAINER
        # Per-task fields (_task_id, _task_class_name, _task_static,
        # _instruction, _seed, _max_steps, _observation_text,
        # _post_action_delay, _extra_tools, _step_count, _terminated)
        # are ALL set by bind() below, so normal direct/server construction
        # never observes them unset.
        self.bind(
            task_id=task_id,
            max_steps=max_steps,
            post_action_delay=post_action_delay,
            observation_text=observation_text,
            seed=seed,
            valid_actions=valid_actions,
            extra_tools=extra_tools,
        )

    @staticmethod
    def _task_metadata(config_metadata: dict[str, Any]) -> LiteCUAMetadata:
        """Same-source metadata builder. ``apps``
        is the full launchable catalog (module constant) — the registered
        copy carries it too; extra_tool_schemas mirrors bind()'s default
        resolution."""
        return LiteCUAMetadata(
            dims=("mobile", "use"),
            extra_tool_schemas=resolve_extra_tools(
                _EXTRA_TOOLS, tools=AndroidworldTools, env_name="androidworld",
            ),
            valid_actions=list(_SCHEMA_VALID_ACTIONS),
            others={
                **config_metadata,
                # list(...): the spread aliases config_metadata's list values —
                # sever the registered/live others from the task registry's
                # own tags list.
                "tags": list(config_metadata.get("tags", [])),
                "apps": list(_ANDROID_WORLD_APPS),
            },
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendment: extra_tool_schemas resolved at bind. Reads the
        # CURRENT config (bind() REPLACEs it), so re-bind stays fresh.
        return dataclasses.replace(
            self._task_metadata(self._config.metadata),
            valid_actions=list(self._schema_valid_actions),
            extra_tool_schemas=list(self._extra_tool_schemas),
        )

    @property
    def external_resource_id(self) -> str | None:
        # Single container-handle attr: non-None from ACQUIRE (stamped inside
        # _create_env right after the factory returns — cold spawn, init
        # retry, and recycle-respawn windows are all covered while boot is
        # still in flight) until destroy_backend() completes. The drift
        # reaper therefore sees every live container for its whole life.
        lock = self._current_container
        return lock.name if lock is not None else None

    async def boot(self) -> None:
        """Acquire emulator container + AsyncEnv binding. Idempotent.

        Reads no task state: emulator container config is determined by
        env-class state + env_kwargs.

        Uses ``_EXECUTOR.submit`` + ``wrap_future`` (not
        ``run_in_executor``) so a StepTimeoutWrapper cancellation of
        ``reset()`` (which delegates here) leaves us with a
        ``_pending_cf_future`` handle for :meth:`close` to reap the
        orphaned emulator + file lock.
        """
        if self._env is not None:
            return  # idempotent
        loop = asyncio.get_event_loop()
        create_cf = _EXECUTOR.submit(self._create_env)
        self._pending_cf_future = create_cf
        env, emulator_lock = await asyncio.wrap_future(create_cf, loop=loop)
        self._pending_cf_future = None  # only reached on success
        self._env = env
        # Already stamped at acquire by _create_env; re-assign for the fake
        # path (returns None) and to keep boot() self-contained.
        self._current_container = emulator_lock

    async def init_task(self) -> None:
        """Instantiate + initialize the bound task on the (pristine) device —
        identical on the cold-spawn and snapshot-reuse paths (reset_with_recycle
        calls this last)."""
        loop = asyncio.get_event_loop()
        self._step_count = 0
        self._terminated = False

        if self._task_class is not None:
            # Reference (minimal_task_runner.py:93) calls env.reset(go_home=True)
            # before task.initialize_task. Important when reusing the same
            # emulator across resets — without it, leftover UI state can leak
            # into the next task setup.
            await loop.run_in_executor(
                _EXECUTOR, partial(self._env.reset, go_home=True)
            )
            # ``_env.reset`` (via ``coordinator.rl_reset`` →
            # ``_launch_simulator``) re-applies android_env's
            # device_settings which **re-enables**
            # ``pointer_location=1`` / ``show_touches=1`` from config
            # defaults (config_classes.py:43-45). Without re-hiding,
            # the GPU pointer-trace overlay (rendered when
            # ``pointer_location=1``) reappears in obs.image AND
            # the overlay window can intercept / shift input delivery
            # so that the agent's first tap occasionally fails to open
            # the launcher's app drawer (M1 v2: 3/4 success on
            # AudioRecorderRecordAudio snapshot reuse — the 1/4 failure
            # had no state divergence at turn_0, only step-time
            # divergence at turn_1, traced to this overlay-driven
            # race). Issue settings put via the container's adb_shell
            # helper since ``_RemoteEnv.hide_automation_ui`` is a no-op
            # — the in-container server.py only calls hide_automation_ui
            # at /init time and exposes no endpoint to re-invoke it.
            if self._current_container is not None:
                try:
                    self._current_container.adb_shell(
                        "settings", "put", "system",
                        "pointer_location", "0", timeout=10.0,
                    )
                    self._current_container.adb_shell(
                        "settings", "put", "system",
                        "show_touches", "0", timeout=10.0,
                    )
                except Exception as e:
                    logger.warning("post-reset hide_automation_ui failed: %s", e)
            # Pre-flight: verify a11y service is alive. If a previous task left
            # the emulator in a bad state (airplane mode, killed a11y service,
            # stale gRPC), _ensure_a11y_healthy() will refresh_env() to rebuild
            # the controller + a11y wrapper without killing the emulator.
            await loop.run_in_executor(_EXECUTOR, self._ensure_a11y_healthy)
            # Generate params (over RPC, in-container so seeded RNG state
            # is scoped to the container's Python). Same instance_seed
            # derivation as before, just executed remotely.
            if self._seed is not None:
                task_name = self._task_class_name
                instance_seed = int(hashlib.md5(
                    f"{task_name}:{self._seed}".encode()
                ).hexdigest(), 16) % (2**31)
            else:
                instance_seed = None
            # Wrap two sync RPC chains (``generate_random_params`` →
            # ``/task/generate_params``; ``_RemoteTask.__init__`` →
            # ``/task/load``) in ``run_in_executor`` so a slow in-container
            # POST doesn't stall the env-server event loop while other
            # concurrent ``reset()`` coroutines wait. At N=32 concurrent
            # rollouts, two back-to-back ~200 ms blocking POSTs per env
            # would stall the loop ~13 s before yielding — long enough to
            # starve /metrics and the drift reaper.
            params = await loop.run_in_executor(
                _EXECUTOR,
                partial(self._task_class.generate_random_params,
                        seed=instance_seed),
            )
            self._task = await loop.run_in_executor(
                _EXECUTOR, self._task_class, params,
            )
            # initialize_task() does `random.seed(params["seed"])` without
            # restoring state — that would leak the env's seed into the whole
            # process RNG (line 157 of android_world/task_evals/task_eval.py).
            # Save/restore around it to keep env seeds scoped to the env.
            _init_rng_state = random.getstate()
            try:
                await loop.run_in_executor(
                    _EXECUTOR, self._task.initialize_task, self._env
                )
            except Exception as e:
                # Match reference's broad-except pattern (suite_utils.py:249):
                # log and skip the task rather than crashing the entire rollout.
                # The agent will get a generic instruction and the task will
                # evaluate as reward=0 (no initialize → is_successful fails).
                random.setstate(_init_rng_state)
                logger.warning("initialize_task failed for %s: %s — skipping", self._task.name, e)
                self._task = None
            else:
                random.setstate(_init_rng_state)
            if self._task is not None:
                self._instruction = self._task.goal
                # Lazy step-budget allocation: if bind() didn't pin a
                # value (``_max_steps is None``), compute the task-
                # complexity default ``int(10 * complexity)`` matching
                # androidworld's ``_allocate_step_budget``. Caller-
                # pinned value (non-None from bind) wins.
                if self._max_steps is None:
                    self._max_steps = int(10 * self._task.complexity)
                # Deliberate post-reset enrichment of the INSTANCE's own
                # metadata copy (bind() copied config.metadata) with
                # per-reset facts for the trajectory row — outside the
                # construction-time registered==live parity invariant.
                self._config.metadata.update({
                    "task_params": _to_json_safe(params),
                    "task_apps": list(self._task.app_names),   # apps THIS task involves (per-task); catalog is others["apps"]
                    "complexity": self._task.complexity,
                })

    def backend_alive(self) -> bool:
        # Usable-ness is the RPC binding, not the container handle: after a
        # cancelled boot the seam attr may point at a container whose /init
        # never completed.
        return self._env is not None

    async def destroy_backend(self) -> None:
        """Tear down env binding + container, reaper-safe: the seam attr keeps
        pointing at the dying container UNTIL destroy() completes — otherwise
        ``external_resource_id`` returns None while the container is still
        alive, the drift reaper classifies it as an orphan, and races
        ``docker rm`` against our own teardown (see ContainerReaper /
        reconcile)."""
        old_env = self._env
        old_lock = self._current_container
        self._env = None
        loop = asyncio.get_event_loop()
        if old_env is not None:
            try:
                await loop.run_in_executor(_EXECUTOR, old_env.close)
            except Exception as e:
                logger.warning("recycle: old env close failed: %s", e)
        if old_lock is not None:
            try:
                await loop.run_in_executor(_EXECUTOR, old_lock.destroy)
            except Exception as e:
                logger.warning("recycle: old container destroy failed: %s", e)
            finally:
                # Null AFTER destroy completes, and only if nothing restamped
                # the seam attr in between (a concurrent acquire).
                if self._current_container is old_lock:
                    self._current_container = None

    async def tear_down_task(self) -> None:
        # MUST run BEFORE reset_to_pristine: ``task.tear_down(env)`` issues
        # device-state mutations (``pm clear``, file deletions) that assume
        # the post-episode dirty state; after the rollback they would smear
        # residue onto the pristine baseline (D4: A.cold→A.hit reward=0 when
        # reversed). Ordering is enforced by reset_with_recycle.
        if self._task is None:
            return
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                _EXECUTOR, self._task.tear_down, self._env
            )
        except Exception as e:
            logger.warning("androidworld: previous-task tear_down failed: %s", e)
        self._task = None

    async def reset_to_pristine(self) -> bool:
        """Snapshot-reload the VM to baseline (~5 s vs a 35 s+ respawn).

        Pre-snapshot images have no ``reset_snapshot`` — treat as pristine
        and proceed on the live device (the legacy fall-through; env.reset(
        go_home=True) in init_task handles MVP cleanup). ``False`` (a bad
        snapshot state) makes the framework destroy + cold-spawn — mirrors
        androidlab's fallback path."""
        lock = self._current_container
        if lock is None or not hasattr(lock, "reset_snapshot"):
            return True
        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(_EXECUTOR, lock.reset_snapshot)
        if not ok:
            logger.warning(
                "androidworld snapshot reload failed; re-spawning container"
            )
        return bool(ok)

    async def reset(self) -> LiteEnvObservation:
        # If a previous reset was cancelled mid-flight and close() was never
        # called (shouldn't happen in normal flow, but be defensive), hand the
        # orphaned future to a background cleanup thread before proceeding.
        if self._pending_cf_future is not None:
            self._spawn_background_cleanup(self._pending_cf_future)
            self._pending_cf_future = None

        # Framework-owned re-task sequence: recycle cap → cold boot →
        # tear-down-BEFORE-pristine → pristine (failed reload ⇒ respawn) →
        # init_task. The four invariants live ONCE in
        # EnvServerPoolable.reset_with_recycle; this env supplies the hooks
        # (backend_alive / destroy_backend / tear_down_task /
        # reset_to_pristine / init_task) below.
        await self.reset_with_recycle()

        # Get initial screenshot. Match reference's transition_pause=1.0 —
        # initialize_task may have just launched an app, so wait one second
        # for the UI to render before capturing the first frame. Fake mode has
        # no emulator/UI to settle, so skip the wait (keeps the fake tests fast).
        if not self._config.use_fake:
            await asyncio.sleep(1.0)
        state, screenshot = await self._get_state_with_png()
        if state.pixels is not None:
            self._screen_h, self._screen_w = state.pixels.shape[:2]

        obs_text = self._instruction
        if self._observation_text.startswith("a11y"):
            coord_unit = "norm" if self._observation_text.endswith(":norm") else "pixel"
            # a11y bboxes and ``_to_screen_pixels`` share the device-native
            # screen, so ``_format_ui_elements`` needs no second surface.
            ui_list = _format_ui_elements(
                state.ui_elements or [], self._screen_w, self._screen_h,
                coord_unit=coord_unit,
            )
            if ui_list:
                obs_text = f"{self._instruction}\n\nVisible UI elements:\n{ui_list}"
        # (The recycle gate — _recycle_first_done — is armed by
        # reset_with_recycle itself.)
        return LiteEnvObservation(image=screenshot, text=obs_text)

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
        step_screenshots: list[bytes] = []
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        # Last device state seen; ``get_state`` carries the frame AND the a11y
        # elements, so the last executed action's frame and ``obs_text`` come
        # from the same RPC. ``None`` means the loop never reached a capture.
        state: Any = None

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
                    _state, png = await self._get_state_with_png()
                    if png is not None:
                        step_screenshots.append(png)
                continue

            if name in LiteFinishToolSet.get_tool_names():
                # ``response`` carries the answer text for information_retrieval
                # tasks (Q&A). AndroidWorld's evaluator reads the answer from
                # ``env.interaction_cache`` (see information_retrieval.py:111).
                # Without this write, ALL Q&A tasks score 0 regardless of
                # whether the agent's text is correct.
                if name == "response":
                    answer_text = args.get("text", "")
                    if answer_text and self._env is not None:
                        # The setter does a sync ``/env/set_interaction_cache``
                        # POST; wrap it so the event loop stays responsive
                        # while other concurrent step() coroutines run.
                        def _set_ic() -> None:
                            self._env.interaction_cache = answer_text
                        try:
                            await loop.run_in_executor(_EXECUTOR, _set_ic)
                        except Exception as e:
                            record_tool_execution_error(
                                action_errors,
                                result_call_id,
                                f"response side effect failed: {e}",
                                action_name=name,
                            )
                            logger.warning("Failed to set interaction_cache: %s", e)
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
                calls = await self._dispatch_action(name, args)
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
            if any(call.get("call") == "noop" for call in calls):
                # Downgraded before touching the device: nothing ran, so there
                # is no post-action state to record. The batch is NOT aborted
                # here, so the following actions still get their own frames.
                continue

            # Wait for the real device UI to settle, then take ONE frame for
            # THIS action — the capture must follow the settle sleep or the
            # frame records the pre-action screen. Fake mode has no emulator,
            # so there's nothing to settle (the fake lifecycle test would
            # otherwise pay this delay on every one of 50 steps).
            if self._post_action_delay > 0 and not self._config.use_fake:
                await asyncio.sleep(self._post_action_delay)
            state, png = await self._get_state_with_png()
            if png is not None:
                step_screenshots.append(png)

        if state is None:
            # ``state`` is TWO facts: "did a GUI action run" (it is set only on
            # the executed path) and the a11y source ``obs_text`` needs below.
            # Fetch it either way, but only ADD a frame when the step produced
            # none -- a rejected slot already captured its own, and appending
            # here too would ship N+1 frames for N slots.
            state, png = await self._get_state_with_png()
            if png is not None and not step_screenshots:
                step_screenshots.append(png)

        obs_text: str | None = None
        if self._observation_text.startswith("a11y"):
            coord_unit = "norm" if self._observation_text.endswith(":norm") else "pixel"
            # a11y bboxes and ``_to_screen_pixels`` share the device-native
            # screen, so ``_format_ui_elements`` needs no second surface.
            ui_list = _format_ui_elements(
                state.ui_elements or [], self._screen_w, self._screen_h,
                coord_unit=coord_unit,
            )
            if ui_list:
                obs_text = f"Visible UI elements:\n{ui_list}"

        self._step_count += 1
        # _max_steps is resolved by reset() (either from bind's
        # caller-pinned value or task.complexity); the ``is not None``
        # guard is for type narrowing — by step() it's always int.
        truncated = not terminated and (
            self._max_steps is not None and self._step_count >= self._max_steps
        )

        reward = await self._evaluate_episode(terminated, truncated, loop)

        self._terminated = terminated
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info={EXECUTED_ACTIONS_INFO_KEY: executed_actions},
            ),
            ordered_call_ids=result_call_ids,
            continue_call_ids=[
                call_id for call_id in result_call_ids
                if call_id not in terminal_call_ids
            ],
            images=step_screenshots,
            text=obs_text,
            feedback=action_errors,
        )

    async def _evaluate_episode(
        self, terminated: bool, truncated: bool, loop: asyncio.AbstractEventLoop,
    ) -> float | None:
        """Evaluate a finished episode; AndroidWorld scores only explicit finish."""
        if not (terminated or truncated) or self._task is None:
            return None
        try:
            task_reward = await loop.run_in_executor(
                _EXECUTOR, self._task.is_successful, self._env
            )
            return float(task_reward) if terminated else 0.0
        except Exception as e:
            logger.error("Evaluation failed: %s", e)
            return 0.0

    async def _get_state_with_png(self) -> tuple[Any, bytes | None]:
        """One ``get_state`` RPC plus its frame.

        The PNG is ``None`` when the device returned no pixels. Callers that
        need the a11y elements and the frame to describe the same instant must
        use this single call rather than two RPCs.
        """
        loop = asyncio.get_event_loop()
        state = await loop.run_in_executor(_EXECUTOR, self._env.get_state)
        pixels = state.pixels
        png = (
            await asyncio.to_thread(_pixels_to_png, pixels)
            if pixels is not None else None
        )
        return state, png

    def bind(
        self,
        task_id: str = "",
        *,
        max_steps: int | None = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        observation_text: str = _OBSERVATION_TEXT,
        seed: int | None = _SEED,
        valid_actions: list[str] | None = _VALID_ACTIONS_CONFIG,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
    ) -> None:
        """Bind (or re-bind) a task and apply soft kwargs.

        **Single entry point for task identity + soft state.** Both direct and
        server construction (direct mode via
        :meth:`LiteBaseEnv.__init__` partition, or the registry
        factory's gym.make call) flow through this method.
        That's the byte-equal guarantee: identical kwargs → identical
        body run → identical self-state.

        All soft kwargs are assigned UNCONDITIONALLY — defaults in this
        signature are the SINGLE source of truth for what each kwarg
        defaults to when absent. ``post_action_delay=3.0`` matches
        m3a's ``wait_after_action_seconds`` (2.0, m3a.py:342) +
        base_agent's ``transition_pause`` (1.0, base_agent.py:52),
        i.e. the total gap between executing an action and grabbing
        the next decision's screenshot.

        ``observation_text``: ``"none"`` (screenshot only),
        ``"a11y:pixel"`` (flat numbered element list, pixel coords in
        the native screenshot's space), or ``"a11y:norm"``
        (flat numbered element list, [0,1000] normalized). No
        ``:tree`` variant — androidworld's ui_elements come from the
        Google library pre-flattened (only androidlab has both).

        ``task_id=""`` is the legacy no-task compatibility path: bind() runs as a
        soft-state reset without resolving a task spec. Normal direct/server
        callers pass a non-empty task_id.

        Unknown kwargs raise ``TypeError`` automatically (Python parameter
        binding), so unsupported env_kwargs fail fast at this boundary.

        Safe to call between episodes only — never during an active
        ``step()`` / ``reset()``.
        """
        # Resolve task spec from task_id when a real task is given.
        # Replace _config wholesale (the registered _task_config is a
        # module-level singleton; we make a shallow copy of metadata
        # so concurrent reuse doesn't share dict). Mirrors _make_env's
        # behavior for fresh gym.make calls. Empty task_id leaves the config
        # from ``__init__`` in place for the no-task compatibility path.
        # Validate ``observation_text`` up-front: silent fall-through to
        # ``coord_unit="pixel"`` for unsupported suffixes (e.g.
        # ``a11y:tree``) had been a contract gap — the reset/step paths
        # check ``startswith("a11y")`` then ``endswith(":norm")`` only,
        # so ``a11y:foo`` looked like a11y but used pixel coords.
        if observation_text not in {"none", "a11y:pixel", "a11y:norm"}:
            raise ValueError(
                f"unknown observation_text={observation_text!r}; "
                f"supported: 'none', 'a11y:pixel', 'a11y:norm'"
            )
        task_class_name: str | None = None
        if task_id:
            spec = _TASK_CONFIGS.get(task_id)
            if spec is None:
                known = ", ".join(sorted(_TASK_CONFIGS)[:5])
                raise ValueError(
                    f"unknown androidworld task_id={task_id!r}; "
                    f"known examples: {known}..."
                )
            config, task_class_name, task_static = spec
            from dataclasses import replace
            # Copy even though the cold path's _make_env already owns its
            # copy: compatibility bind paths can receive a config from the
            # shared registry, and this is the only site all paths cross.
            self._config = replace(config, metadata={**config.metadata})
            self._task_static = dict(task_static)
            self._instruction = (
                config.instruction_template
                or "Interact with the Android device."
            )
        else:
            # Legacy no-task compatibility seed: preserve baked _config, derive
            # instruction from it, leave task-derived static empty.
            self._task_static = {}
            self._instruction = (
                self._config.instruction_template
                or "Interact with the Android device."
            )
        # Unconditional per-kwarg application — see method docstring.
        # ``max_steps`` stores the *caller-pinned* budget directly;
        # ``None`` means "auto-compute from task.complexity at reset()".
        # One field, one source of truth.
        self._task_id = task_id
        self._task_class_name = task_class_name
        self._seed = seed
        self._max_steps = max_steps
        self._post_action_delay = post_action_delay
        self._observation_text = observation_text
        # Soft env_kwarg, resolved through the shared helper so every env
        # answers ``valid_actions`` identically (None = full GUI surface,
        # [] = deliberately none, unknown name = raise at the config
        # boundary). Runtime invalid/unsupported feedback is owned by ``step``.
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="androidworld", platform="mobile",
        )
        self._schema_valid_actions = resolve_schema_valid_actions(
            valid_actions,
            env_name="androidworld",
            platform="mobile",
            supported_actions=_SUPPORTED_ACTIONS,
        )
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        # Rebuild the remote-task factory bound to the NEW
        # task_class_name. The factory is what ``reset()`` later calls
        # to ``generate_random_params(seed=...)`` → ``task_class(params)``,
        # and it carries its OWN ``_task_class_name``. Without re-
        # binding here, task-id re-binds emit the new ``self._task_id``
        # into log paths but issue ``/task/generate_params`` /
        # ``/task/load`` with the FIRST task's class name, so the
        # agent receives the OLD task's goal text. Bind the new
        # factory iff this env has a remote one (server-mode path;
        # direct mode uses the real task class and is unaffected).
        if task_class_name and isinstance(self._task_class, _RemoteTaskFactory):
            self._task_class = _RemoteTaskFactory(
                task_class_name, self._task_class._rpc
            )
        self._step_count = 0
        self._terminated = False

    async def close(self) -> None:
        """Shut down the emulator and release the emulator lock.

        If reset() was cancelled mid-flight (e.g. by StepTimeoutWrapper timeout),
        the executor thread is still running _create_env(). _pending_cf_future
        lets us wait for it and clean up the leaked env + emulator lock.
        """
        # Handle leaked env from cancelled reset().
        leaked_pending = False
        if self._pending_cf_future is not None:
            leaked_pending = True
            cf_future = self._pending_cf_future
            self._pending_cf_future = None

            if cf_future.done():
                self._close_leaked_env(cf_future)
            else:
                logger.info("reset() executor still running, spawning background cleanup thread")
                self._spawn_background_cleanup(cf_future)

        if self._task is not None and self._env is not None:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    _EXECUTOR, self._task.tear_down, self._env
                )
            except Exception as e:
                logger.warning("Error tearing down task: %s", e)
            self._task = None

        if self._env is not None:
            env = self._env
            self._env = None
            try:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(_EXECUTOR, env.close),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.warning("androidworld env close timed out after 10s")
            except Exception as e:
                logger.warning("Error closing androidworld env: %s", e)

        # Only release emulator lock here if no background cleanup is pending.
        # When a reset was cancelled, the background/inline cleanup handles
        # lock release after closing the leaked env.
        if not leaked_pending and self._current_container is not None:
            self._current_container.destroy()
            self._current_container = None

    def _close_leaked_env(self, cf_future: Any) -> None:
        """Close a leaked env from a completed c.f.Future."""
        try:
            leaked_env, leaked_lock = cf_future.result()
        except Exception:
            return  # _create_env raised — it already released the lock
        try:
            leaked_env.close()
            logger.info("Cleaned up leaked androidworld env")
        except Exception as e:
            logger.warning("leaked androidworld env close() failed: %s", e)
        if leaked_lock is not None:
            leaked_lock.destroy()
        # Only null ``self._current_container`` if it's still pointing at THIS
        # leaked lock — i.e. nothing else assigned a new lock in between.
        # Unconditional null'ing here would wipe a freshly-acquired NEW
        # container's name from ``external_resource_id``, exposing it to
        # the drift-reaper-orphan-race the rest of this file guards
        # against. (Triggered in v20: cleanup of an old leaked future
        # nulled the in-use lock → drift-reaper killed the live
        # container → /step 500.)
        if self._current_container is leaked_lock:
            self._current_container = None

    def _spawn_background_cleanup(self, cf_future: Any, timeout: float = 30.0) -> None:
        """Daemon thread that waits for the c.f.Future and cleans up.

        Survives event loop shutdown — even if the asyncio loop is torn down
        after the rollout, this thread keeps waiting for _create_env() to finish
        and then releases the emulator.

        If the thread doesn't finish within *timeout* seconds (e.g. stuck in
        app setup), we kill the emulator process to unblock it.

        Under the single container-handle attr model there is deliberately NO
        force-kill on timeout: reading the shared attr here could destroy a
        container a concurrent retry just acquired (the shared-attr hazard). If
        _create_env eventually returns, on_result destroys its LOCAL result;
        if it never returns, the container is reclaimed by the drift reaper's
        orphan guard (server mode) / atexit (direct mode).
        """
        cleaned_lock: list[Any] = [None]

        def _ok(res: Any) -> None:
            leaked_env, leaked_lock = res
            try:
                leaked_env.close()
                logger.info("androidworld: background cleanup closed leaked env")
            except Exception as e:
                logger.warning("Background cleanup: close() failed: %s", e)
            if leaked_lock is not None:
                leaked_lock.destroy()
                cleaned_lock[0] = leaked_lock

        def _fail(e: BaseException) -> None:
            if isinstance(e, TimeoutError):
                logger.warning(
                    "Background cleanup: _create_env() stuck for >%.0fs; "
                    "leaving its container to the on_result path / drift "
                    "reaper (no shared-attr kill — shared-attr hazard)", timeout
                )
            else:
                logger.warning("Background cleanup thread failed: %s", e)

        def _done() -> None:
            # Only null the seam attr if it's still pointing at the lock we
            # just released. Otherwise a new container acquired in between
            # loses its ``external_resource_id`` → drift-reaper race. Same
            # shape as ``_close_leaked_env``.
            if cleaned_lock[0] is not None and self._current_container is cleaned_lock[0]:
                self._current_container = None

        spawn_background_destroy(
            cf_future, on_result=_ok, on_failure=_fail, on_done=_done,
            timeout=timeout, thread_name="android-world-leak-cleanup",
        )

    # -----------------------------------------------------------------------
    # Internal: env creation
    # -----------------------------------------------------------------------

    def _create_env(self) -> tuple[Any, AndroidWorldContainer | None]:
        """Create the androidworld env (blocking, called via executor).

        Returns (env, emulator_lock) tuple. The lock is returned instead of
        being assigned to self to avoid cross-thread side effects — the caller
        assigns both after the await succeeds. If _create_env raises after
        acquiring the lock, it releases it before propagating.

        ``/init`` 500 retry: if the in-container ``env_launcher.load_and_setup_env``
        raises (manifests as the host ``rpc.post("/init", ...)`` raising
        ``RuntimeError: server /init returned 500``), the container is
        destroyed and a fresh one is acquired up to ``_INIT_MAX_ATTEMPTS``
        times. This is a belt-and-suspenders for residual Android-system-
        services races even after :meth:`AndroidWorldContainer._wait_until_android_ready`
        — under extreme host contention adb timing can still slip, and an
        early /init failure should not waste the rollout retry budget.
        """
        if self._config.use_fake:
            return _FakeAsyncEnv(), None

        from lite.gym.utils.config.identity import EnvIdentity
        identity = getattr(self, "identity", None) or EnvIdentity()
        _check_image(
            getattr(self, "_image", None) or _IMAGE,
            env_id=self.ENV_ID,
        )

        last_init_err: Exception | None = None
        for attempt in range(1, _INIT_MAX_ATTEMPTS + 1):
            emulator_lock = _acquire_emulator(
                self._config.avd_name,
                task_id=getattr(self, "_task_id", None) or getattr(self._config, "task_id", None),
                session_id=identity.session_id,
                token_hash=identity.token_hash,
                server_port=identity.server_port,
                image=getattr(self, "_image", None),
                env_id=self.ENV_ID,
            )
            # Container-handle attr stamped AT ACQUIRE: the drift reaper must see
            # this container from its first breath, before /init completes.
            self._current_container = emulator_lock
            logger.info(
                "Using container %s (api=%s)",
                emulator_lock.name, emulator_lock.base_url,
            )

            try:
                # env_launcher runs INSIDE the container (via docker/server.py),
                # so a11y_grpc_wrapper's reverse gRPC stays in the container's
                # network namespace — sidesteps rootless docker's
                # --disable-host-loopback block. The host receives a
                # _RemoteEnv proxy whose methods HTTP-RPC into the container.
                rpc = _RemoteRPC(emulator_lock.base_url)
                init_body = {
                    "console_port": emulator_lock.console_port,
                    "grpc_port": emulator_lock.grpc_port,
                    # The in-container adb binary — env_launcher is in
                    # the container too, so this is a same-fs path.
                    "adb_path": "/root/Android/Sdk/platform-tools/adb",
                    "emulator_setup": self._config.emulator_setup,
                    "freeze_datetime": self._config.freeze_datetime,
                }
                rpc.post("/init", body=init_body)
                env = _RemoteEnv(rpc, init_body)
                # Build the remote task factory from the class-name string
                # stored on env (set by the _make_env factory below). Both
                # `self._task_class(params)` and `generate_random_params`
                # round-trip through the in-container server.
                if self._task_class_name is not None:
                    self._task_class = _RemoteTaskFactory(self._task_class_name, rpc)
                break  # /init succeeded
            except RuntimeError as e:
                # Only retry on the specific /init 500 signature. Other
                # RuntimeErrors (network, timeout, container died) bubble
                # up unchanged — those are different failure modes whose
                # right response is also "destroy + retry", but the
                # diagnostic message we'd want to surface differs.
                if "/init returned 500" not in str(e) or attempt == _INIT_MAX_ATTEMPTS:
                    emulator_lock.destroy()
                    if self._current_container is emulator_lock:
                        self._current_container = None
                    raise
                last_init_err = e
                logger.warning(
                    "init attempt %d/%d 500ed on container %s; destroying + "
                    "re-acquiring fresh container",
                    attempt, _INIT_MAX_ATTEMPTS, emulator_lock.name,
                )
                emulator_lock.destroy()
                try:
                    _block_overlapping_retry(emulator_lock)
                finally:
                    if self._current_container is emulator_lock:
                        self._current_container = None
            except Exception:
                # Same cleanup as the retry-path above: null the ref
                # AFTER release so the drift reaper can't snapshot a
                # stale ``external_resource_id`` pointing at a lock
                # that was just torn down. Without this, the reaper
                # sees the destroyed container's name on its next
                # sweep, flags it as a ghost, and pops the session out
                # from under a concurrent retry attempt.
                emulator_lock.destroy()
                if self._current_container is emulator_lock:
                    self._current_container = None
                raise
        else:
            # Loop exited without break — shouldn't reach here because
            # the last attempt re-raises, but defensive.
            raise RuntimeError(
                f"/init failed after {_INIT_MAX_ATTEMPTS} attempts"
            ) from last_init_err

        # hide_automation_ui is already called inside /init; this is a
        # no-op for _RemoteEnv but kept callable for interface parity.
        try:
            env.hide_automation_ui()
        except Exception as e:
            logger.warning("hide_automation_ui() failed: %s", e)

        return env, emulator_lock

    # -----------------------------------------------------------------------
    # Internal: a11y self-healing (matches official androidworld reference)
    # -----------------------------------------------------------------------

    def _refresh_env(self) -> None:
        """Rebuild the AndroidWorldController to recover from a11y failures.

        Mirrors the official androidworld reference's
        ``AndroidWorldController.refresh_env()`` — reconnects to the SAME
        running emulator (no qemu restart) and re-creates the a11y_grpc_wrapper
        (reinstalls forwarder APK, re-enables a11y service, new gRPC channel).

        Typically takes ~5-10s. Only called when a11y tree fetch fails after
        standard retries, so it does not affect happy-path throughput.
        """
        if self._env is None or self._current_container is None:
            return
        console_port = self._current_container.console_port
        grpc_port = self._current_container.grpc_port

        logger.warning(
            "Refreshing androidworld env (console=%d, grpc=%d) — "
            "reconnecting to emulator + rebuilding a11y wrapper",
            console_port, grpc_port,
        )
        try:
            # Close old env FIRST to release gRPC port before new wrapper binds.
            # a11y_grpc_wrapper starts a gRPC server on a picked port; if the old
            # server is still listening, the new one may fail to bind or silently
            # receive stale messages.
            old_env = self._env
            self._env = None  # prevent step() from using half-torn-down env
            try:
                old_env.close()
            except Exception:
                pass
            time.sleep(1.0)  # grace period for gRPC server shutdown

            # Rebuild the in-container env_launcher via the existing
            # server. Reuses the same RPC base URL since the container
            # itself is still alive (only the in-container Python env
            # object is being recreated).
            rpc = _RemoteRPC(self._current_container.base_url)
            init_body = {
                "console_port": console_port,
                "grpc_port": grpc_port,
                "adb_path": "/root/Android/Sdk/platform-tools/adb",
                "emulator_setup": False,
                "freeze_datetime": self._config.freeze_datetime,
            }
            rpc.post("/init", body=init_body)
            self._env = _RemoteEnv(rpc, init_body)
            try:
                self._env.hide_automation_ui()
            except Exception:
                pass
            logger.info("refresh_env succeeded")
        except Exception:
            logger.exception("refresh_env failed — env may be degraded")

    def _ensure_a11y_healthy(self) -> None:
        """Pre-flight check: verify a11y tree is fetchable, refresh if not.

        Called at the start of each reset() before task initialization.
        Matches the official androidworld reference's recovery pattern:
          1. Check airplane mode (blocks gRPC) → fix if on
          2. Try get_a11y_tree with reference defaults (5 retries, 1s sleep)
          3. On failure → refresh_env() to rebuild controller + a11y wrapper

        On the happy path (a11y works), costs ~1s. On failure, ~5-10s.
        """
        if self._env is None or not hasattr(self._env, 'controller') or self._env.controller is None:
            return
        # Step 1: Fix airplane mode if on (matches reference controller.py:81-88).
        # Airplane mode blocks the gRPC channel to the a11y forwarder.
        try:
            resp = self._env._rpc.post("/env/check_airplane_mode")
            if resp.get("airplane_on"):
                logger.warning("Airplane mode on — disabling before a11y check")
                self._env.controller.env.attempt_enable_networking()
                time.sleep(1.0)
        except Exception:
            pass
        # Step 2: Verify a11y tree is fetchable (reference defaults: 5 retries, 1s).
        # Route via /env/check_a11y on the in-container server:
        # `get_a11y_tree` calls `env.accumulate_new_extras()`, a method
        # on the a11y-grpc wrapper that lives only inside the container.
        # The host's _RemoteControllerEnv proxy intentionally does not
        # expose it (we don't want to round-trip a heavy a11y tree just
        # to check its presence — the container can short-circuit).
        try:
            resp = self._env._rpc.post("/env/check_a11y")
            if not resp.get("ok", False):
                raise RuntimeError(
                    f"in-container a11y check failed: {resp.get('error')}"
                )
        except (RuntimeError, Exception):
            logger.warning("a11y pre-flight failed — calling _refresh_env()")
            self._refresh_env()

    # -----------------------------------------------------------------------
    # Internal: action dispatch
    # -----------------------------------------------------------------------

    async def _dispatch_action(
        self, name: str, args: dict[str, Any]
    ) -> list[LiteExecutedAction]:
        """Translate CUA-Lite mobile action to AndroidWorld JSONAction and execute.

        Most actions go through ``JSONAction`` + ``env.execute_action``. The
        only exception is ``swipe``: AW's ``JSONAction(scroll | swipe)`` only
        accepts an enum direction (no distance, no precise endpoints), so we
        bypass it and issue a raw ``adb input swipe`` via ``adb_utils`` so the
        agent's start/end coordinates are honored exactly.
        """
        calls: list[LiteExecutedAction] = []

        try:
            if self._config.use_fake:
                # For fake env, just record the action without executing
                calls.append({"call": name, "args": args})
                return calls

            loop = asyncio.get_event_loop()
            if name in ("swipe", "drag"):
                try:
                    executed = await loop.run_in_executor(_EXECUTOR, self._exec_swipe_adb, args)
                except MODEL_ACTION_ERROR_TYPES as e:
                    logger.error("Action %s failed: %s", name, e)
                    calls.append({
                        "call": "noop",
                        "args": {"name": name, "reason": str(e), "is_error": True},
                    })
                    return calls
                calls.append(executed)
            elif name == "system_button" and args.get("button") in ("Menu", "Recent"):
                executed = await loop.run_in_executor(
                    _EXECUTOR,
                    self._exec_system_button_adb,
                    args["button"],
                )
                calls.append(executed)
            else:
                try:
                    action_dict = self._translate_action(name, args)
                except MODEL_ACTION_ERROR_TYPES as e:
                    logger.error("Action %s failed: %s", name, e)
                    calls.append({
                        "call": "noop",
                        "args": {"name": name, "reason": str(e), "is_error": True},
                    })
                    return calls
                if action_dict is not None:
                    await loop.run_in_executor(
                        _EXECUTOR, self._env.execute_action, action_dict
                    )
                    # Record the AndroidWorld JSONAction-equivalent dict we
                    # actually issued (post coord-translation), so traces show
                    # the env-internal form rather than echoing the cua-lite
                    # request.
                    calls.append({
                        "call": f"json_action.{action_dict['action_type']}",
                        "args": {k: v for k, v in action_dict.items() if v is not None},
                    })
                elif name != "screenshot":
                    calls.append({"call": "noop", "args": {"name": name, "reason": "unknown action"}})

        except Exception as e:
            logger.error("Action %s failed: %s", name, e)
            # DEBUG postmortem: fire immediately at the failure site so the
            # in-container ``/tmp/server.log`` is captured BEFORE the env
            # is closed by the client and the container destroyed. Only
            # for Connection-refused / URL errors (indicating in-container
            # python died mid-action); other failures (parse errors,
            # action-translation bugs) don't need an in-container dump.
            # Gated by ``CUA_LITE_DEBUG_POSTMORTEM_DIR`` inside
            # ``_postmortem_snapshot``; production no-op.
            if self._current_container is not None:
                err = str(e)
                if "Connection refused" in err or "URLError" in err \
                        or "ConnectionRefusedError" in err:
                    try:
                        self._current_container._postmortem_snapshot(
                            f"action_fail_{type(e).__name__}"
                        )
                    except Exception:
                        pass
            raise

        return calls

    def _exec_swipe_adb(self, args: dict[str, Any]) -> LiteExecutedAction:
        """Execute a precise swipe via raw ``adb input swipe`` (in-container).

        AW's JSONAction(scroll/swipe) only takes an enum direction, so it
        cannot honor the agent's exact start/end coordinates or distance.
        We send the pixel coords + duration to the in-container server's
        /env/exec_swipe endpoint, which calls
        ``adb_utils.generate_swipe_command`` + ``issue_generic_request``.
        Coordinates from the agent are normalized [0, 1000]; we map them
        to physical pixels host-side using the device size reported by the
        controller, then ship the pixel values to the container.

        Returns a dict describing the env-internal command actually issued
        (for the trajectory ``executed`` log).
        """
        start = args.get("start_coordinate", [500, 500])
        end = args.get("coordinate", [500, 500])
        sx, sy = self._to_screen_pixels(start)
        ex, ey = self._to_screen_pixels(end)
        duration_ms = int(_duration_seconds(args, "swipe", default=0.5) * 1000)
        self._env._rpc.post("/env/exec_swipe", body={
            "sx": sx, "sy": sy, "ex": ex, "ey": ey,
            "duration_ms": duration_ms,
        })
        return {
            "call": "adb.input.swipe",
            "args": {
                "start_x": sx, "start_y": sy,
                "end_x": ex, "end_y": ey,
                "duration_ms": duration_ms,
            },
        }

    def _exec_system_button_adb(self, button: str) -> LiteExecutedAction:
        """Press Android system keys that JSONAction does not expose."""
        keycodes = {"Menu": 82, "Recent": 187}
        keycode = keycodes[button]
        self._env._rpc.post("/env/exec_keyevent", body={"keycode": keycode})
        return {
            "call": "adb.input.keyevent",
            "args": {"keycode": keycode, "button": button},
        }

    def _translate_action(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Convert CUA-Lite mobile action to an AndroidWorld JSONAction
        dict (which the in-container ``/env/execute_action`` rehydrates
        into the actual ``JSONAction`` dataclass before dispatch).
        """
        if name == "tap":
            x, y = self._to_screen_pixels(args.get("coordinate"))
            clicks = args.get("clicks", 1)
            if clicks >= 2:
                return {"action_type": "double_tap", "x": x, "y": y}
            return {"action_type": "click", "x": x, "y": y}

        elif name == "long_press":
            x, y = self._to_screen_pixels(args.get("coordinate"))
            if "duration" in args:
                # AndroidWorld JSONAction has no duration slot here; keep the
                # model-controlled value as validation-only.
                _duration_seconds(args, "long_press", default=1.0)
            return {"action_type": "long_press", "x": x, "y": y}

        # No ``double_tap`` branch: the canonical mobile spelling is
        # ``tap(clicks=2)``, handled above, so a bare ``double_tap`` falls through
        # to the unknown-action noop. (``action_type: "double_tap"`` on the wire is
        # the upstream JSONAction name ``tap(clicks>=2)`` emits, not a cua-lite
        # action name.)
        elif name == "type":
            text = args.get("text", "")
            return {"action_type": "input_text", "text": text}

        elif name == "system_button":
            btn = args.get("button", "")
            action_type_map = {
                "Home": "navigate_home",
                "Back": "navigate_back",
                "Enter": "keyboard_enter",
            }
            action_type = action_type_map.get(btn)
            if action_type:
                return {"action_type": action_type}
            raise ValueError(
                f"system_button: unknown button {btn!r}; expected one of "
                "['Back', 'Enter', 'Home', 'Menu', 'Recent']"
            )

        elif name == "open_app":
            app_name = args.get("app_name", "")
            return {"action_type": "open_app", "app_name": app_name}

        elif name == "wait":
            if "duration" in args:
                # Upstream wait JSONAction carries no duration. Validate bounds
                # before dispatch but do not pretend the backend honors it.
                _duration_seconds(args, "wait", default=1.0, allow_zero=True)
            return {"action_type": "wait"}

        elif name == "screenshot":
            # Nothing to dispatch; the step loop takes this action's frame.
            return None

        else:
            logger.warning("Unknown action: %s(%s), skipping", name, args)
            return None

    def _to_screen_pixels(
        self, coordinate: list[int] | None
    ) -> tuple[int, int]:
        """Convert [0, 1000] normalized coords to pixel coords.

        clamp=False preserves this env's legacy no-clamp behavior until its
        coordinate contract is validated for canonical clamping."""
        return norm_to_pixel(coordinate, self._screen_w, self._screen_h,
                             clamp=False, on_malformed="raise")

# ---------------------------------------------------------------------------
# Task metadata loading
# ---------------------------------------------------------------------------

def _load_tasks_json() -> dict[str, dict[str, Any]]:
    """Load the static task list dumped from the docker image.

    See ``scripts/utils/tasks.sh`` — re-run that script when
    the androidworld version pinned in ``docker/Dockerfile`` changes.
    """
    tasks_path = Path(__file__).parent / "data" / "tasks.json"
    if not tasks_path.is_file():
        raise FileNotFoundError(
            f"{tasks_path} not found. Re-generate with:\n"
            f"  bash lite/gym/envs/androidworld/scripts/utils/tasks.sh\n"
            "(requires cua-lite/androidworld:latest docker image — run scripts/install.sh first)."
        )
    with open(tasks_path) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

def _make_env(
    *,
    task_class: type | None = None,
    config: AndroidWorldTaskConfig | None = None,
    **kwargs: Any,
) -> AndroidWorldEnv:
    """Factory function for gym.make().

    Real tasks (container path): pass ``task_class_name=<str>`` +
    ``task_static=<tasks.json entry>``. ``_create_env`` then wraps the
    name in a ``_RemoteTaskFactory`` so reset() can call
    ``task_class(params)`` and ``task_class.generate_random_params(seed=…)``.

    Fake/test tasks (``config.use_fake=True``): pass ``task_class=<type>``
    pointing at a real Python class (e.g. ``_FakeTaskEval``). reset()
    uses it directly without any RPC.
    """
    from dataclasses import replace
    if config is None:
        config = AndroidWorldTaskConfig(
            **{k: v for k, v in kwargs.items() if k in AndroidWorldTaskConfig.__dataclass_fields__}
        )
    else:
        # The registered `_task_config` is a module-level singleton shared across
        # every gym.make() of the same task_id. `replace` is shallow — metadata
        # dict would still be shared — so copy it explicitly. Without this,
        # concurrent reset()s overwrite each other's task_params in the shared
        # dict, and the last writer wins in all parquets.
        config_overrides = {
            k: v for k, v in kwargs.items() if k in AndroidWorldTaskConfig.__dataclass_fields__
        }
        config_overrides.setdefault("metadata", {**config.metadata})
        config = replace(config, **config_overrides)
    env_kwargs = {k: v for k, v in kwargs.items() if k not in AndroidWorldTaskConfig.__dataclass_fields__}
    env = AndroidWorldEnv(config=config, **env_kwargs)
    # ``_task_class_name`` + ``_task_static`` are set by ``bind`` via
    # ``_TASK_CONFIGS`` lookup keyed on the ``task_id`` we already
    # forwarded through ``env_kwargs`` (see ``register(...)`` partial
    # call sites — ``task_id=_task_name``). Direct ``setattr`` here
    # would be redundant and would drift from bind's source-of-truth.
    #
    # Fake/test path: ``task_class`` is a real Python class (e.g.
    # ``_FakeTaskEval``) passed via the partial; assign it directly
    # so reset() bypasses the remote factory. ``bind``'s
    # ``_RemoteTaskFactory`` rebuild branch only fires when the
    # existing class IS a remote factory — assigning a real class
    # here is preserved.
    if task_class is not None:
        env._task_class = task_class
    return env

#: Module-level task_id → (config, task_class_name, task_static) table,
#: populated by the registration loop below. Looked up by
#: :meth:`AndroidWorldEnv.bind` when callers pass a task_id string instead of
#: the full config object.
_TASK_CONFIGS: dict[str, tuple["AndroidWorldTaskConfig", str, dict]] = {}


# Auto-register all AndroidWorld tasks from the static tasks.json (no
# androidworld Python package import on host).
_TASKS = _load_tasks_json()

# Env-wide make defaults declared once here, sourced from default.yaml
# make_kwargs. Per-task register() kwargs / gym.make() still override.
registry.set_env_make_kwargs("androidworld", CFG.make_kwargs)

# Register each task twice: once as eval (original name) and once as train
# (perturb_<name>). Both splits share the same factory — the eval/train
# distinction (seeded vs randomized task params) is encoded in the
# registered spec's defaults (e.g. eval pins seed=42, train leaves it open
# for the group-shared seed injector).
for _task_name, _task_meta in _TASKS.items():
    _instruction = _task_meta.get(
        "instruction_template",
        f"Complete the '{_task_name}' task on Android.",
    )
    _task_others = {
        "task_name": _task_name,
        "difficulty": _task_meta.get("difficulty", "unknown"),
        "complexity": _task_meta.get("complexity"),
        "tags": _task_meta.get("tags", []),
        "optimal_steps": _task_meta.get("optimal_steps", ""),
    }
    _task_config = AndroidWorldTaskConfig(
        task_class_name=_task_name,
        instruction_template=_instruction,
        metadata=_task_others,
    )
    # Populate the task_id → spec table for :meth:`AndroidWorldEnv.bind`.
    # Both eval and perturb_ variants share the same task_class + config;
    # ``bind("perturb_X")`` and ``bind("X")`` differ only in the
    # task_id string the env tags with, not in the underlying behavior
    # (the perturb_ split is set at gym.make time via env_kwargs.seed=None
    # → randomized params each reset). Since task_id-string bind passes a single
    # task_id, we register both keys.
    _TASK_CONFIGS[_task_name] = (_task_config, _task_name, _task_meta)
    _TASK_CONFIGS[f"perturb_{_task_name}"] = (_task_config, _task_name, _task_meta)
    # eval: seed=42 baked into registration → deterministic task params on every reset.
    # train: no seed → randomized params each reset.
    # Separate entry points so each env instance receives its own task_id string.
    # reset_timeout is an env-wide default (CFG.make_kwargs, applied via
    # set_env_make_kwargs above) — not repeated per task here. A per-task or
    # per-subset override would still go here as a register() kwarg.
    register(
        key=f"androidworld@{_task_name}",
        entry_point=partial(
            _make_env, config=_task_config, task_id=_task_name,
        ),
        split="eval",
        # Same-source contract: registered copy == the env's builder
        # output (incl. the module-constant ``apps`` catalog and the
        # no-override extra_tool_schemas default).
        metadata=AndroidWorldEnv._task_metadata(_task_others),
        seed=42,
    )
    register(
        key=f"androidworld@perturb_{_task_name}",
        entry_point=partial(
            _make_env, config=_task_config, task_id=f"perturb_{_task_name}",
        ),
        split="train",
        metadata=AndroidWorldEnv._task_metadata(_task_others),
    )
    logger.debug("Registered androidworld task: %s (eval + train:perturb_)", _task_name)


class AndroidWorldServices(ContainerServices):
    """Env-server capabilities for androidworld: docker-image presence check
    (``ensure``) + per-container reconciliation (``live_ids``/``reap`` inherited
    from :class:`ContainerServices`; the per-instance container name is emitted by
    the env's ``EnvServerResource`` mixin)."""

    def ensure(self, env_id: str) -> None:
        _ensure_services(env_id)

    def health(self, env_id: str) -> None:
        _health_check(env_id)


register_services("androidworld", AndroidWorldServices())
from lite.gym.services import BackendFamily, register_family  # noqa: E402

register_family("androidworld", BackendFamily.DEDICATED)
