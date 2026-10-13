"""androidlab — CUA-Lite gym wrapper for THUDM's AndroidLab benchmark.

Wraps AndroidLab (138 multi-step mobile tasks across 9 offline apps) as a
CUA-Lite gym environment with ``LiteMobileActionSpace``. Evaluation is
per-step XML-state inspection via each task's hand-written ``judge()``;
we cache the most recent ``judge_page=True`` outcome across the episode
and reward ``complete=True`` at terminate/truncate — same semantics as
the reference's replay-eval but live.

Per-worker lifecycle: a dedicated docker container runs one emulator,
loads the reference's Quick Boot snapshot (with ABC/AAA contacts,
Pink-Floyd MP3s, SQLite DBs intact), and is driven via HTTP RPC into
the container's ``docker/server.py`` (which runs adb locally to its
own emulator — no host-side ``docker exec`` on the hot path). Between
episodes on the same env, we hot-reload ``default_boot`` via
``adb emu avd snapshot load`` to restore the seeded state in ~3-5s.
See ``container.py`` for the why (snapshot path hardcoding +
emulator version pinning).

Prerequisites:
  - ``uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh`` to build the
    ``cua-lite/androidlab:latest`` docker image (builds THUDM's image
    then swaps in emulator 34.2.15 to match the snapshot).

Usage:
    uv run python -c "import lite.gym as gym; print(gym.registry.task_ids('androidlab'))"
"""
from __future__ import annotations

import asyncio
import atexit
import dataclasses
import json
import logging
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.extra_tools import (
    LiteFinishToolSet,
    make_open_app_tool,
)
from lite.core.tools.action_space import pixel_to_norm
from lite.core.tools.action_space.duration import (
    ACTION_SCHEMA_DURATION_CAPS_SECONDS,
)
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.schemas import BaseTools
from lite.gym.container import spawn_background_destroy
from lite.gym.envs.androidlab.container import AndroidLabContainer, AndroidLabContainerFactory
from lite.gym.registry import register, registry
from lite.gym.services import register_services
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
from lite.gym.utils.backend.rpc import RemoteRPC
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
from lite.utils.image import encode_png

logger = logging.getLogger(__name__)

# Registration + infra defaults — see configs/default.yaml (override the whole
# config via ANDROID_LAB_CONFIG=<path>). The yaml is the single source for these
# values; env_kwargs feed AndroidLabTaskConfig / bind, server_kwargs feed the
# module-level infra constants (executor size, recycle cap, container timeouts).
ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# ANDROID_LAB_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
#: Per-task step budget (bind default; every eval config uses 20).
_MAX_STEPS = CFG.env_kwargs["max_steps"]
#: Wait-after-action before capturing next observation. Reference uses 3s;
#: dropping lower risks stale UI Automator dumps (events not yet indexed).
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
#: Text observation mode (see ``AndroidLabTaskConfig.observation_text``).
_OBSERVATION_TEXT = CFG.env_kwargs["observation_text"]
#: Downscale screenshot's longer side to this many pixels before sending
#: to the agent. See ``_downscale_png`` for rationale.
_SCREENSHOT_MAX_DIM = CFG.env_kwargs["screenshot_max_dim"]
#: Default per-instance task seed. ``None`` → unseeded; the EVAL split
#: registers a fixed seed. androidlab judges are deterministic so this
#: is a no-op (kept for caller-symmetry). ``bind()`` signature default.
_SEED = CFG.env_kwargs["seed"]
#: Soft GUI-action surface default (``None`` = full GUI, ``[]`` = no GUI).
_VALID_ACTIONS_CONFIG = CFG.env_kwargs.get("valid_actions")
#: Opt-in extra-tool selection (list of tool names; ``[]`` = none). The
#: ROLLOUT/eval yaml enables ``open_app``; the default is OFF.
#: ``bind()`` signature default.
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]
#: Docker image built by ``scripts/install.sh`` — has the pinned 34.2.15
#: emulator + reference's AVD + custom x86_64 system image + adb keys.
_IMAGE = CFG.env_kwargs["image"]
# --- server_kwargs (per-deployment) ---
#: ThreadPool size for blocking RPC / sync calls (see ``_EXECUTOR`` below).
_MAX_WORKERS = CFG.server_kwargs["max_workers"]
#: Max resets per container before forcing a destroy + respawn. The QEMU
#: snapshot reload restores VM-layer state perfectly (validated byte-level
#: in T1/T2/T5), but the layers OUTSIDE
#: the VM accumulate state across episodes:
#:   * qemu process — fd pool, memory fragmentation, internal allocator
#:   * adb daemon (in-container) — connection-pool table grows
#:   * in-container ``docker/server.py`` Python process — module-level
#:     globals (``_env``, ``_task``, ``_judge``) reset via /task/release
#:     + /task/load between episodes, but the process itself ages
#:   * /tmp/emu.log, /tmp/server.log — unbounded
#: Forcing a fresh container every K resets bounds all of these without
#: requiring the host to monitor each one.
#:
#: The counter is per-CONTAINER, not per-env-instance: a single gym instance may
#: run multiple resets against the same ``_current_container`` when snapshot reuse
#: is enabled, so K bounds container physical age across that instance.
#:
#: Default K=0 → every reset (after the first cold spawn) destroys the
#: container and acquires a fresh one. Matches androidworld's default
#: stance: prefer byte-clean baselines over the ~13 % wall-clock saving
#: from snapshot reuse. Historical values were K=50 then K=20 (tightened
#: as fd-leak / adb-socket-pool drift was observed in long stress runs);
#: K=0 generalizes the trend — eliminates VM-external residue entirely.
#: Set ``server_kwargs.max_resets_per_container >= 1`` (configs/default.yaml) to
#: re-enable snapshot reuse if the throughput tradeoff is acceptable.
_MAX_RESETS_PER_CONTAINER = CFG.server_kwargs["max_resets_per_container"]
# ============================================================================

# Blocking RPC / sync calls (urllib.request, container docker
# subprocess) run on this pool. 256 covers ~128 emulators × 2 threads
# (acquire + step/teardown). See server_kwargs.max_workers in default.yaml.
_EXECUTOR = ThreadPoolExecutor(
    max_workers=_MAX_WORKERS,
    thread_name_prefix="android-lab-exec",
)
atexit.register(lambda: _EXECUTOR.shutdown(wait=False, cancel_futures=True))


# ---------------------------------------------------------------------------
# In-container env-server proxies (docker/server.py)
# ---------------------------------------------------------------------------

class _RemoteRPC(RemoteRPC):
    """Tiny HTTP-RPC client to the in-container env-server.

    Mirrors androidworld's ``_RemoteRPC`` — reuses the shared
    :class:`lite.gym.utils.backend.rpc.RemoteRPC` body (pickle-POST, so we don't
    lose androidlab judge / XML / package structures across the
    boundary; healthz / OK responses are JSON). Unlike androidworld it
    does NOT retry on a transient connection error (``retries=0``); a
    transient surfaces straight to the caller.
    """

    def __init__(self, base_url: str, timeout: float = 180.0):
        super().__init__(base_url, timeout, retries=0)


class _RemoteController:
    """Host-side proxy mirroring the in-container ``_CmdController``.

    Mirrors androidworld's ``_RemoteEnv`` pattern: every method
    delegates to a single HTTP POST on the container's env-server.
    Drops the legacy ``DockerExecAndroidController`` (host-side
    ``docker exec adb …``) so per-step adb calls no longer pay the
    docker-daemon serialization tax. See
    :class:`lite.gym.envs.androidlab.docker.server._CmdController`
    for the receiver-side implementation + parity notes.
    """

    def __init__(self, rpc: _RemoteRPC, device: str):
        self._rpc = rpc
        self.device = device
        r = rpc.post("/init", body={"device": device})
        self.width = int(r["width"])
        self.height = int(r["height"])

    # ---- input ------------------------------------------------------------

    def tap(self, x: int, y: int) -> None:
        self._rpc.post("/env/tap", body={"x": int(x), "y": int(y)})

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        self._rpc.post("/env/long_press", body={
            "x": int(x), "y": int(y), "duration_ms": int(duration_ms),
        })

    def swipe_precise(
        self, start: tuple[int, int], end: tuple[int, int], duration_ms: int = 400,
    ) -> None:
        sx, sy = start
        ex, ey = end
        self._rpc.post("/env/swipe_precise", body={
            "sx": int(sx), "sy": int(sy), "ex": int(ex), "ey": int(ey),
            "duration_ms": int(duration_ms),
        })

    def text(self, s: str) -> None:
        self._rpc.post("/env/text", body={"text": s})

    def back(self) -> None:
        self._rpc.post("/env/key", body={"key": "back"})

    def home(self) -> None:
        self._rpc.post("/env/key", body={"key": "home"})

    def enter(self) -> None:
        self._rpc.post("/env/key", body={"key": "enter"})

    def menu(self) -> None:
        self._rpc.post("/env/key", body={"key": "menu"})

    def recent(self) -> None:
        self._rpc.post("/env/key", body={"key": "recent"})

    def launch_app(self, package: str) -> None:
        self._rpc.post("/env/launch_app", body={"package": package})

    # ---- observation ------------------------------------------------------

    def get_current_activity(self) -> str:
        r = self._rpc.post("/env/get_current_activity")
        return r.get("activity", "") if isinstance(r, dict) else ""

    def execute_adb(self, cmd: str) -> str:
        r = self._rpc.post("/env/execute_adb", body={"cmd": cmd})
        return r.get("result", "") if isinstance(r, dict) else ""

    def get_screenshot_bytes(self) -> bytes | None:
        """PNG bytes via ``adb exec-out screencap -p`` (streamed)."""
        return self._rpc.post("/env/get_screenshot")

    def get_xml(self) -> str | None:
        """uiautomator-dump XML (covers all apps except map.me/pimusic)."""
        return self._rpc.post("/env/get_xml", body={"ac": False})

    def get_ac_xml(self) -> str | None:
        """XMLParser a11y dump (map.me / pimusic only)."""
        return self._rpc.post("/env/get_xml", body={"ac": True})

    def observe(self, *, ac: bool = False) -> dict[str, Any]:
        """Batched observation: one RPC for screenshot + xml +
        compressed_xml + current_activity. Falls back to individual
        endpoints if the container is older (pre-``/env/observe``).

        The whole package, always: the sole caller wants every field, and
        the batched endpoint costs one RPC either way. ``ac`` stays a
        parameter because it selects WHICH xml dump, not whether to take one.
        """
        # No local 404 fallback: the sole caller already wraps this in
        # ``except Exception`` and falls back per-field, so an older container
        # without /env/observe is handled one frame up.
        return self._rpc.post("/env/observe", body={
            "ac": ac,
            "want_screenshot": True,
            "want_xml": True,
            "want_compressed_xml": True,
            "want_activity": True,
        })


def _downscale_png(png_bytes: bytes, max_dim: int) -> bytes:
    """Half-scale-ish downsample of a PNG so vision tokens stay bounded.

    Qwen3-VL's mobile cookbook was authored against ~720×1520 emulator
    screenshots; feeding raw 1440×3120 (9M pixels, ~4.4K vision tokens
    per image × 4 history) crowds the KV cache and truncates the tool
    call JSON mid-output. Capping the longer side at ``max_dim`` drops
    vision tokens to ~1.1K each without changing our [0, 1000] normalized
    coordinate space.
    """
    try:
        from io import BytesIO

        from PIL import Image
        img = Image.open(BytesIO(png_bytes))
        w, h = img.size
        longest = max(w, h)
        if longest > max_dim:
            scale = max_dim / longest
            img = img.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.LANCZOS,
            )
        # Fast encode (compress_level=1, no optimize): this runs in an executor
        # thread so it doesn't block the loop, but the default PIL level-6 is
        # ~2-3x slower CPU for no size benefit that matters here.
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=1)
        return buf.getvalue()
    except Exception as e:
        logger.warning("png downscale failed (%s); using raw", e)
        return encode_png(png_bytes)


_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


def _rescale_tree_bounds(
    tree: Any, convert: Callable[[int, int], tuple[int, int]]
) -> Any:
    """Walk reference's compressed XML tree and map every ``bounds`` corner
    through ``convert`` (``AndroidLabEnv._a11y_point`` — the env's ONE a11y
    coordinate conversion, so the tree and the flat list cannot diverge).

    The tree is the **parsed** result of
    ``json.loads(get_compressed_xml_from_str(...))`` — a nested dict whose
    keys are UI node descriptors and whose values are either (a) a dict
    with a ``bounds`` string + nested children, or (b) a leaf string. Note
    the upstream helper returns a JSON string despite its docstring saying
    "Returns a nested-dict"; ``_get_obs_text`` is responsible for the
    ``json.loads`` step. We return rebuilt copies so the caller can
    re-serialize without mutating the original (judge still needs the
    native-pixel tree).
    """
    if isinstance(tree, dict):
        out: dict[str, Any] = {}
        for k, v in tree.items():
            if k == "bounds" and isinstance(v, str):
                m = _BOUNDS_RE.match(v)
                if m:
                    x1, y1, x2, y2 = map(int, m.groups())
                    ax, ay = convert(x1, y1)
                    bx, by = convert(x2, y2)
                    out[k] = f"[{ax},{ay}][{bx},{by}]"
                    continue
            out[k] = _rescale_tree_bounds(v, convert)
        return out
    if isinstance(tree, list):
        return [_rescale_tree_bounds(x, convert) for x in tree]
    return tree


# ---------------------------------------------------------------------------
# App catalog
# ---------------------------------------------------------------------------
#
# Apps available for the model's ``open`` shortcut on androidlab. These
# are the canonical English keys from the reference's package lookup
# (``Android-Lab/android_lab/templates/packages.py``); the bench's own
# task suites (``android_lab/evaluation/tasks/``) only directly cover 9
# of them, but the dispatcher uses Levenshtein fuzzy match against the
# full key list so listing the rest still helps the model pick a name
# the lookup will resolve. Surfaced via ``metadata.others["apps"]``
# and embedded as the ``app_name`` enum in the ``open_app`` extra tool
# below, so tool-schema agents (e.g. Qwen3-VL) see the same names.

_ANDROID_LAB_APPS: list[str] = [
    # Directly covered by the bench's task suites
    "bluecoins", "Calendar", "Cantook", "Clock", "Contacts",
    "Map.me", "PiMusicPlayer", "Settings", "Zoom",
    # Common extras present in packages.py (the model may need any of
    # these depending on the task)
    "Chrome", "Firefox", "Gmail", "Google Drive", "Google Maps",
    "YouTube", "Spotify", "Twitter", "X", "Reddit", "LinkedIn",
    "Facebook", "Instagram", "WhatsApp", "Snapchat", "TikTok",
    "Netflix", "Twitch", "Amazon Shopping", "Booking", "Uber",
    "Slack", "Quora", "weather", "tasks", "simple_notepad", "vlc",
]


# ---------------------------------------------------------------------------
# Extra tools
# ---------------------------------------------------------------------------
# Routed to ``self._launch_app(app_name)`` via the ``if name == "open_app"``
# branch in the agent→bench action mapper (uses Levenshtein fuzzy match
# against ``packages.py`` keys). Exposed as an OpenAI function tool through
# ``metadata.extra_tool_schemas``. Qwen-family native ``open`` and MAI-UI
# app-open surfaces canonicalize to this env-owned ``open_app`` boundary when
# configs opt in.

class AndroidlabTools(BaseTools):
    """What androidlab declares beyond the GUI surface: ``open_app``."""

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "open_app": make_open_app_tool(_ANDROID_LAB_APPS),
    }


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = AndroidlabTools.get_tool_names() | LiteFinishToolSet.get_tool_names()

_SUPPORTED_ACTIONS = android_supported_actions()
_SCHEMA_VALID_ACTIONS = resolve_schema_valid_actions(
    _VALID_ACTIONS_CONFIG,
    env_name="androidlab",
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

# Native render size of the pinned Pixel_7_Pro AVD (``wm size``). Pre-reset
# fallback only — ``_screen_w/h`` are pulled fresh from the device at reset. The
# AVD render size is emulator-fixed; there is no settable display_resolution.
_AVD_NATIVE_SIZE: tuple[int, int] = (1440, 3120)


@dataclass
class AndroidLabTaskConfig:
    """Per-task config. Populated at registration, overridable via gym.make()."""

    #: Docker image built by ``scripts/install.sh`` — has the pinned 34.2.15
    #: emulator + reference's AVD + custom x86_64 system image + adb keys.
    #: Name is fixed; keep in sync with ``scripts/install.sh`` if renaming.
    #: Default lives in configs/default.yaml (env_kwargs.image).
    image: str = _IMAGE
    avd_name: str = "Pixel_7_Pro_API_33"

    #: Wait-after-action before capturing next observation. Reference uses 3s;
    #: dropping lower risks stale UI Automator dumps (events not yet indexed).
    post_action_delay: float = _POST_ACTION_DELAY
    #: Text observation mode. All modes derive from the same UIAutomator /
    #: xml_parser a11y dump; they differ only in rendering:
    #:   "none"              — screenshot only (default, matches androidworld)
    #:   "a11y_tree:pixel"   — reference's compressed JSON tree, pixel coords
    #:                         in the native screenshot's pixel space
    #:   "a11y_tree:norm"    — compressed JSON tree, [0,1000] normalized coords
    #:   "a11y_list:pixel"   — flat numbered element list, pixel coords
    #:   "a11y_list:norm"    — flat numbered element list, [0,1000] normalized
    observation_text: str = _OBSERVATION_TEXT
    #: Downscale screenshot's longer side to this many pixels before sending
    #: to the agent. See ``_downscale_png`` for rationale.
    screenshot_max_dim: int = _SCREENSHOT_MAX_DIM
    #: Optional adb command to run after every step; result goes into
    #: ``line["command"][cmd]``. ~30 tasks (mostly Settings) use this.
    adb_query: str | None = None
    #: App name used by ``find_package`` for the ``open_app`` launch.
    app: str = ""
    task_id: str = ""
    #: Reference's `task_template` — the instruction the agent sees.
    instruction: str = ""
    #: Identifies which judge to instantiate inside the container via
    #: ``/task/load``. Set from tasks.json (== upstream's
    #: ``function_map`` key for the task, which equals ``task_id``).
    judge_class_name: str = ""
    #: Submodule under ``android_lab.evaluation.tasks.<this>`` that
    #: holds the judge class. Derived from the YAML's ``metric_func``.
    app_module_name: str = ""
    #: TEST-ONLY backdoor. When set, _run_judge instantiates this class
    #: locally and calls .judge() directly (bypassing RPC). Production
    #: code never sets this — tests use it to inject a stateful
    #: _FakeJudge without spawning a real container.
    judge_class: type | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _check_image(tag: str = _IMAGE) -> None:
    from lite.gym.utils.backend.docker import require_image_present

    require_image_present(image_for("androidlab", tag=tag))


def _ensure_services(env_id: str) -> None:
    """env-server startup hook: verify the docker image is built + fresh
    (rebuilt since its sources changed) before
    accepting traffic. Symmetric with androidworld's same-named hook.
    Image absence surfaces as ``EnvDepsMissingError`` → ``available:
    false`` on ``GET /envs?expand=metadata`` and HTTP 501 on the first
    ``POST /instances`` (instead of a cryptic gym.make failure mid-rollout).
    """
    _check_image()


_HEALTH_CHECK = CachedEnvDepsHealthCheck(_ensure_services)


def _health_check(env_id: str) -> None:
    _HEALTH_CHECK(env_id)


class AndroidLabEnv(EnvServerPoolable, EnvServerResource):
    """CUA-Lite wrapper around AndroidLab's 138 live-eval tasks."""

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = AndroidlabTools

    def __init__(
        self, *,
        config: AndroidLabTaskConfig,
        image: str | None = _IMAGE,
        task_id: str = "",
        max_steps: int = _MAX_STEPS,
        seed: int | None = _SEED,
        valid_actions: list[str] | None = _VALID_ACTIONS_CONFIG,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        post_action_delay: float | None = None,
        observation_text: str | None = None,
        screenshot_max_dim: int | None = None,
    ) -> None:
        """Construct qemu-shape fields, then bind task/soft state.

        ``config`` is required because the baked qemu fields (image, avd_name)
        live inside it; those fields are identical across tasks that share one
        backend shape.
        """
        self._config = config
        # Family-neutral spawn-image override (env-kwarg, same contract
        # as the osworld/androidworld envs): None = use the config's
        # default image.
        self._image = image
        # Per-task fields (_max_steps, _extra_tools, _step_count, _terminated,
        # _config-task-side state) are set by bind() below, so direct-mode never
        # observes them unset.
        self._screen_w = 0
        self._screen_h = 0
        #: Live container with a booted emulator (the Layer-B seam attr —
        #: ``reset_with_recycle`` / the default ``backend_alive`` and the
        #: drift-reaper's ``external_resource_id`` all read it). Lazy-created
        #: on first reset().
        self._current_container: AndroidLabContainer | None = None
        #: Reuse cap consumed by ``reset_with_recycle`` — see the
        #: ``_MAX_RESETS_PER_CONTAINER`` docstring for the drift rationale.
        self._max_resets_per_container = _MAX_RESETS_PER_CONTAINER
        #: Host-side proxy for the in-container ``_CmdController``. Owns
        #: adb invocation, screenshot, XML dump, tap/swipe/text — all via
        #: HTTP RPC to ``docker/server.py`` (no docker exec on the hot
        #: path). Re-bound whenever the container is (re)spawned.
        self._controller: _RemoteController | None = None
        #: Judge cache — most recent entry where ``judge_page=True``.
        self._best_judge: dict[str, Any] | None = None
        #: Judge state lives inside the docker container now. The
        #: container's ``server.py`` holds the (stateful) judge instance
        #: between ``/task/load`` and the next ``/task/release``;
        #: ``_run_judge`` round-trips ``compressed_xml + line`` over RPC.
        #: ~16 task judges (bluecoins_11..15, calendar_6/7/8/9/11/12/14,
        #: cantook_11, pimusic_7/8/12) carry cross-step state on
        #: ``self.<flag>`` to detect "before vs after" edits. Reference's
        #: ``evaluation/task.py`` instantiates the metric ONCE per task
        #: and reuses it across the whole trace; the in-container
        #: ``server.py`` does the same.
        self._rpc: _RemoteRPC | None = None
        self._judge_loaded: bool = False
        #: ``concurrent.futures.Future`` of an in-flight executor call
        #: to ``factory.acquire`` inside :meth:`boot`. Stashed so
        #: :meth:`close` can reap the container even if the boot await
        #: was cancelled before the executor finished. ``None`` when no
        #: boot is in flight.
        self._pending_cf_future: Any = None
        self.bind(
            task_id=task_id,
            max_steps=max_steps,
            seed=seed,
            valid_actions=valid_actions,
            extra_tools=extra_tools,
            post_action_delay=post_action_delay,
            observation_text=observation_text,
            screenshot_max_dim=screenshot_max_dim,
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
                _EXTRA_TOOLS, tools=AndroidlabTools, env_name="androidlab",
            ),
            valid_actions=list(_SCHEMA_VALID_ACTIONS),
            others={**config_metadata, "apps": list(_ANDROID_LAB_APPS)},
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
        # Full container name (unique per spawn, includes api_port).
        # Stable across snapshot reuse. ``None`` until first reset() spawns
        # the container.
        return self._current_container.name if self._current_container is not None else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        """Acquire a qemu emulator container. Idempotent.

        The container image + avd name come from
        ``self._config`` (constructor state); :attr:`_task` is not
        read here.

        Uses ``_EXECUTOR.submit + asyncio.wrap_future`` (not
        ``run_in_executor``) and stashes the future on
        ``self._pending_cf_future`` so a cancellation of the await
        (e.g. ``StepTimeoutWrapper`` aborting reset) leaves us with a
        handle for :meth:`close` to reap the orphaned container —
        otherwise the executor thread keeps running ``factory.acquire``
        and the container it eventually produces is GC'd unreferenced.
        Mirrors :meth:`AndroidWorldEnv.boot`.
        """
        if self._current_container is not None:
            return  # idempotent
        loop = asyncio.get_event_loop()
        from lite.gym.utils.config.identity import EnvIdentity
        identity = getattr(self, "identity", None) or EnvIdentity()
        image = self._image or self._config.image
        _check_image(image)
        factory = AndroidLabContainerFactory(
            # ``image`` env-kwarg override > config default. boot() is now
            # the ONLY spawn path (the framework's snapshot-fail respawn
            # routes back through it), so the override can't diverge
            # between cold-spawn and respawn anymore.
            image=image,
            avd_name=self._config.avd_name,
            task_id=self._config.task_id,
            session_id=identity.session_id,
            token_hash=identity.token_hash,
            server_port=identity.server_port,
        )
        create_cf = _EXECUTOR.submit(factory.acquire)
        self._pending_cf_future = create_cf
        container = await asyncio.wrap_future(create_cf, loop=loop)
        self._pending_cf_future = None  # only reached on success
        self._current_container = container

    async def init_task(self) -> None:
        """Per-episode task init on a pristine backend (Layer-B hook).

        Verbatim body of the old ``reset()`` task block: RPC re-bind,
        fresh in-container judge (atomic /task/release + /task/load — the
        judge lives in the container's ``server.py`` process, OUTSIDE the
        VM snapshot, so there is no VM-side teardown and
        ``tear_down_task`` stays the no-op default), emulator setup,
        controller bind, target-app launch."""
        loop = asyncio.get_event_loop()
        # Bind RPC to the freshly (re-)spawned container's in-container
        # env-server. URL is stable across reset_snapshot calls (same
        # container = same api port); we set it unconditionally so a
        # post-respawn case picks up the new port.
        self._rpc = _RemoteRPC(self._current_container.base_url)

        self._step_count = 0
        self._best_judge = None
        # Fresh judge instance per episode — stateful judges (origin_bill,
        # edit_started_correctly, etc.) must start from defaults each run.
        # Release any prior judge then load a new one in-container. The
        # in-container server's /task/load instantiates a fresh judge
        # instance for this episode (==reference's evaluation/task.py
        # one-metric-per-trace pattern).
        if self._config.judge_class_name:
            # Wrap both sync RPC calls in ``run_in_executor`` so the event
            # loop stays responsive while one env's /task/{release,load}
            # round-trip is in flight — at N=32 concurrent resets, 32 × two
            # back-to-back blocking POSTs (~200 ms each) would stall every
            # other coroutine on the server for several seconds.
            judge_class_name = self._config.judge_class_name
            app_module_name = self._config.app_module_name
            def _release_then_load() -> None:
                try:
                    self._rpc.post("/task/release")
                except Exception as e:
                    # Best-effort: a stale judge (or none) is the normal case
                    # on a fresh container.
                    logger.debug("androidlab: stale-judge release skipped: %s", e)
                self._rpc.post("/task/load", body={
                    "judge_class_name": judge_class_name,
                    "app_module_name": app_module_name,
                })
            await loop.run_in_executor(_EXECUTOR, _release_then_load)
            self._judge_loaded = True
        else:
            self._judge_loaded = False

        # Per-reset emulator setup: adb root, pin clock, pin GPS. The snapshot
        # was saved in a specific state; these small tweaks keep runs
        # deterministic across time-of-day.
        await loop.run_in_executor(_EXECUTOR, self._emulator_setup)

        # Bind the in-container controller proxy. /init runs inside the
        # container, reads ``wm size`` via the local adb daemon, returns
        # width/height. We do this AFTER ``_emulator_setup`` ensures
        # ``adb root`` has run so ``wm size`` (probed in /init) won't
        # race the boot.
        self._controller = await loop.run_in_executor(
            _EXECUTOR, _RemoteController, self._rpc, self._current_container.adb_serial,
        )
        self._screen_w, self._screen_h = self._controller.width, self._controller.height

        # Launch the target app. Snapshot already has the Android UI in a
        # stable post-boot state, so the launch is just a foreground switch.
        if self._config.app:
            await loop.run_in_executor(_EXECUTOR, self._launch_app, self._config.app)
            # Beat for app splash / first render to settle.
            await asyncio.sleep(3.0)

    async def reset_to_pristine(self) -> bool:
        """Hot-reload the ``default_boot`` QEMU snapshot (~3-5s vs ~60s
        respawn). ``False`` → framework destroys + reboots (the old
        snapshot-fail respawn path, now framework-owned — ``boot()``
        already routes its acquire through ``_pending_cf_future``)."""
        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(
            _EXECUTOR, self._current_container.reset_snapshot,
        )
        if not ok:
            logger.warning("androidlab: snapshot reload failed; re-spawning container")
        return ok

    async def destroy_backend(self) -> None:
        old_container = self._current_container
        if old_container is None:
            return
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(_EXECUTOR, old_container.destroy)
        except Exception as e:
            logger.warning("androidlab: container destroy failed: %s", e)
        finally:
            # Null AFTER destroy completes, and only if still ours —
            # keeps ``external_resource_id`` valid for the drift-reaper
            # until the docker container is actually gone.
            if self._current_container is old_container:
                self._current_container = None

    async def reset(self) -> LiteEnvObservation:
        loop = asyncio.get_event_loop()

        # If a previous reset's boot() was cancelled mid-`factory.acquire`,
        # the executor thread is still running and will eventually
        # produce a container we no longer hold a reference to. Hand it
        # to the background-thread destroyer so it doesn't leak. Mirror
        # of ``AndroidWorldEnv.reset``'s pending-future check.
        if self._pending_cf_future is not None:
            self._spawn_background_cleanup(self._pending_cf_future)
            self._pending_cf_future = None

        await self.reset_with_recycle()

        screenshot = await loop.run_in_executor(_EXECUTOR, self._get_screenshot_png)
        obs_text, compressed_xml = await loop.run_in_executor(_EXECUTOR, self._get_obs_text)

        head = self._config.instruction or f"Complete task {self._config.task_id} on Android."
        if obs_text:
            head = f"{head}\n\n{obs_text}"
        return LiteEnvObservation(image=screenshot, text=head)

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
        executed: list[LiteExecutedAction] = []
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        finish_payload: dict[str, Any] | None = None
        # One frame per executed action, in action order. The frame for an
        # action is taken at the top of the NEXT iteration (its settle sleep
        # has already run by then), and the LAST executed action's frame is the
        # trailing batched observation below — that one RPC returns the frame
        # and the XML together, so the model's image and ``obs_text`` describe
        # the same instant.
        step_screenshots: list[bytes] = []
        frame_pending = False

        for action, result_call_id in actions:
            name = action["name"]
            args = action["arguments"]

            if frame_pending:
                png = await loop.run_in_executor(_EXECUTOR, self._get_screenshot_png)
                if png is not None:
                    step_screenshots.append(png)
                frame_pending = False

            tool_feedback, tool_reason = standalone_tool_call_feedback_with_reason(
                action,
                _KNOWN_STANDALONE_TOOL_NAMES,
                metadata.extra_tool_schemas,
            )
            if tool_feedback is not None:
                if result_call_id:
                    append_feedback(action_errors, result_call_id, tool_feedback)
                executed.append({
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
                        _EXECUTOR, self._get_screenshot_png,
                    )
                    if png is not None:
                        step_screenshots.append(png)
                continue

            if name in LiteFinishToolSet.get_tool_names():
                terminated = True
                if action.get("call_id"):
                    terminal_call_ids.add(action["call_id"])
                finish_payload = {
                    "operation": "finish",
                    "action": "finish",
                    "kwargs": {
                        "message": (args.get("text") if name == "response" else args.get("reason")) or "",
                    },
                }
                executed.append({"call": name, "args": args})
                break

            invalid_action = invalid_action_message(
                action, runtime_metadata.valid_actions,
            )
            if invalid_action:
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(invalid_action)
                executed.append({
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
                executed.append({
                    "call": "noop",
                    "args": {"name": name, "reason": unsupported_action},
                })
                continue

            try:
                call = await loop.run_in_executor(_EXECUTOR, self._dispatch_action, name, args)
            except MODEL_ACTION_ERROR_TYPES as e:
                logger.warning("Action %s failed: %s", name, e)
                record_model_action_error(action_errors, result_call_id, e, action_name=name)
                executed.append({"call": "noop", "args": {"name": name, "reason": str(e)}})
                continue
            except Exception as e:
                logger.warning("Action %s execution failed: %s", name, e)
                record_tool_execution_error(action_errors, result_call_id, e, action_name=name)
                executed.append({"call": "noop", "args": {"name": name, "reason": str(e)}})
                continue

            executed.append(call)
            call_args = call.get("args", {})
            # An action-batch call can expand into several actions sharing one
            # result_call_id (see prepare_env_tool_calls); the first of them to
            # report an error owns the feedback for that call_id.
            if (
                result_call_id
                and result_call_id not in action_errors
                and call.get("call") == "noop"
                and call_args.get("reason")
            ):
                action_errors[result_call_id] = error_only_feedback(
                    str(call_args["reason"])
                )
            if call.get("call") == "noop":
                # Downgraded before touching the device: nothing ran, so there
                # is no post-action state to record. The batch is NOT aborted
                # here, so the following actions still get their own frames.
                continue

            # Let the device UI settle. The frame for this action is taken
            # after this sleep — at the top of the next iteration, or from the
            # trailing observation if this was the last executed action.
            if self._config.post_action_delay > 0:
                await asyncio.sleep(self._config.post_action_delay)
            frame_pending = True

        # Batched observation: ONE RPC for screenshot + xml +
        # compressed_xml + current_activity (was 4 separate calls).
        # See ``_RemoteController.observe``.
        obs_pkg = await loop.run_in_executor(
            _EXECUTOR, self._observe_via_rpc,
        )
        screenshot = obs_pkg.get("screenshot")
        # This frame closes out the last executed action; when nothing executed
        # at all it is the one current observation the turn still owes the
        # model. It is NOT appended when the batch's tail was rejected after an
        # earlier action already had its frame taken.
        if screenshot is not None and (frame_pending or not step_screenshots):
            step_screenshots.append(screenshot)
        obs_text = obs_pkg.get("obs_text")
        compressed_xml = obs_pkg.get("compressed_xml")
        current_activity = obs_pkg.get("current_activity", "")

        self._step_count += 1
        truncated = not terminated and self._step_count >= self._max_steps

        last = executed[-1] if executed else {}
        parsed_action = finish_payload or {
            # Dual "operation"+"action" keys match both reference shapes the
            # judges use; kwargs carries the action's arguments.
            "operation": last.get("call", "none"),
            "action":    last.get("call", "none"),
            "kwargs":    last.get("args", {}),
        }
        line: dict[str, Any] = {
            "parsed_action": parsed_action,
            "target": self._config.instruction,
            "window": (self._screen_w, self._screen_h),
            "current_activity": current_activity,
        }
        if self._config.adb_query:
            try:
                cmd_out = await loop.run_in_executor(_EXECUTOR, self._run_adb_query, self._config.adb_query)
                line["command"] = {self._config.adb_query: cmd_out}
            except Exception as e:
                logger.warning("adb_query failed: %s", e)
                line["command"] = {self._config.adb_query: ""}

        await loop.run_in_executor(_EXECUTOR, self._run_judge, compressed_xml, line)

        reward: float | None = None
        if terminated or truncated:
            reward = self._compute_reward()
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info={
                    EXECUTED_ACTIONS_INFO_KEY: executed,
                    "best_judge": self._best_judge,
                    "step": self._step_count,
                },
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

    async def close(self) -> None:
        # If boot() was cancelled mid-`factory.acquire`, the executor
        # thread is still running and will eventually return a live
        # container we no longer hold a reference to. Hand the future
        # to a background-thread destroyer so the container is reaped
        # rather than leaked. Mirror of
        # ``AndroidWorldEnv._spawn_background_cleanup``.
        if self._pending_cf_future is not None:
            self._spawn_background_cleanup(self._pending_cf_future)
            self._pending_cf_future = None
        # Best-effort graceful teardown of in-container task + controller
        # before nuking the container. Matches androidworld's pattern
        # (env_world/main.py:415 → POST /close). Failure is non-fatal —
        # docker rm -f below kills the process anyway.
        if self._rpc is not None:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    _EXECUTOR, partial(self._rpc.post, "/close"),
                )
            except Exception as e:
                logger.debug("in-container /close failed: %s", e)
            self._rpc = None
            self._controller = None
        if self._current_container is not None:
            container = self._current_container
            # Keep ``self._current_container`` set until ``destroy()`` finishes.
            # The env's ``external_resource_id`` property (read by the
            # env-server's state snapshot, consumed by the drift-reaper)
            # falls back to ``None`` once ``_current_container`` is null, so an
            # eager-null + slow-destroy window invites the drift-reaper to
            # treat the still-alive docker container as an orphan and
            # race ``docker rm -f`` against our own destroy.
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(_EXECUTOR, container.destroy)
            except Exception as e:
                logger.warning("androidlab: container destroy failed: %s", e)
            finally:
                # Null AFTER destroy returns (``finally`` so a hung
                # destroy doesn't leave the env latched to a dead lock).
                self._current_container = None

    def _spawn_background_cleanup(
        self, cf_future: Any, timeout: float = 30.0,
    ) -> None:
        """Reap the container produced by a cancelled boot via the shared
        background cleanup shell (androidlab's ``factory.acquire`` returns just a
        container — no unpacking, no pending attr to fall back on; a
        stuck acquire is left to the drift reaper's sweep)."""
        def _ok(leaked_container: Any) -> None:
            leaked_container.destroy()
            logger.info(
                "androidlab background cleanup: destroyed leaked container",
            )

        def _fail(e: BaseException) -> None:
            logger.warning("androidlab background cleanup finished with: %s", e)

        spawn_background_destroy(
            cf_future, on_result=_ok, on_failure=_fail, timeout=timeout,
            thread_name="androidlab-leak-cleanup",
        )

    def bind(
        self,
        task_id: str = "",
        *,
        max_steps: int = _MAX_STEPS,
        seed: int | None = _SEED,  # noqa: ARG002 — accepted for caller symmetry
        valid_actions: list[str] | None = _VALID_ACTIONS_CONFIG,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        post_action_delay: float | None = None,
        observation_text: str | None = None,
        screenshot_max_dim: int | None = None,
    ) -> None:
        """Bind (or re-bind) a task and apply soft kwargs.

        **Single entry point for task identity + soft state.** Direct mode and
        server construction (via :meth:`LiteBaseEnv.__init__` partition or
        registry factory) both route here, so identical kwargs produce
        identical state.

        Task-identity soft kwargs are assigned unconditionally; the three
        config-resident overrides below keep their None-guard (reason follows) — defaults in
        this signature are the SINGLE source of truth for what each
        kwarg defaults to when absent. ``max_steps`` default comes from
        configs/default.yaml (env_kwargs.max_steps); callers override
        per-task via env_kwargs.

        Args:
            task_id: Empty string for the legacy no-task compatibility seed;
                normal callers pass a non-empty value. Non-empty values must be one of the 138 keys
                in ``data/tasks.json``.
            max_steps: Per-task step budget.
            seed: Accepted for caller-symmetry with other envs;
                AndroidLabEnv ignores it (judges are deterministic).
            extra_tools: Per-task extra-tool selection (list of names; the
                resolver expands it to the flat schemas).
            post_action_delay: Host-side wait-after-action override.
                ``None`` keeps the baked/cold config value.
            observation_text: Host-side text-observation mode override.
                ``None`` keeps the baked/cold config value.
            screenshot_max_dim: Host-side screenshot downscale override.
                ``None`` keeps the baked/cold config value.

        These three are HOST-SIDE knobs (host sleep / render / downscale;
        never baked into the container image), so they are soft kwargs
        here rather than baked ones. ``None`` defaults make
        server construction byte-equal with the direct path, which only applies them
        when present (see :func:`_make_env`); a non-``None`` value is
        written onto ``self._config`` so the env honors it.

        Unknown kwargs raise ``TypeError`` automatically, so unsupported
        env_kwargs fail fast at this boundary.

        Safe to call between episodes only.
        """
        # Resolve task spec when a real task_id is given; empty leaves
        # the baked config in place (legacy no-task compatibility seed path).
        if task_id:
            cfg = _TASK_CONFIGS.get(task_id)
            if cfg is None:
                known = ", ".join(sorted(_TASK_CONFIGS)[:5])
                raise ValueError(
                    f"unknown androidlab task_id={task_id!r}; "
                    f"known examples: {known}..."
                )
            # Copy metadata, don't alias: _TASK_CONFIGS is the shared
            # registration-side object — an instance-side metadata write must
            # never reach it (mirror of androidworld's ctor copy).
            self._config = replace(cfg, metadata={**cfg.metadata})
        # Unconditional soft-state application — see docstring.
        self._max_steps = max_steps
        # Soft env_kwarg, resolved through the shared helper so every env
        # answers ``valid_actions`` identically (None = full GUI surface,
        # [] = deliberately none, unknown name = raise at the config
        # boundary). Runtime invalid/unsupported feedback is owned by ``step``.
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="androidlab", platform="mobile",
        )
        self._schema_valid_actions = resolve_schema_valid_actions(
            valid_actions,
            env_name="androidlab",
            platform="mobile",
            supported_actions=_SUPPORTED_ACTIONS,
        )
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        # Host-side config-resident overrides: write only when supplied so
        # an absent kwarg keeps the baked/cold config value (byte-equal with
        # the cold path's _make_env, which applies only present overrides).
        config_overrides = {
            k: v
            for k, v in (
                ("post_action_delay", post_action_delay),
                ("observation_text", observation_text),
                ("screenshot_max_dim", screenshot_max_dim),
            )
            if v is not None
        }
        if config_overrides:
            self._config = replace(self._config, **config_overrides)

    # ------------------------------------------------------------------
    # Judge + reward
    # ------------------------------------------------------------------

    def _run_judge(self, compressed_xml: Any, line: dict[str, Any]) -> None:
        # Test-only path: inject a local fake judge via config.judge_class.
        # We always instantiate (even on the no-xml early-return below) so
        # tests can inspect the judge state.
        if self._config.judge_class is not None:
            if not hasattr(self, "_judge"):
                self._judge = self._config.judge_class()
            if compressed_xml is None:
                return
            result = self._judge.judge(compressed_xml, line)
        else:
            if compressed_xml is None:
                return
            # Production path: round-trip via the in-container judge.
            if not self._judge_loaded or self._rpc is None:
                return
            # A failed evaluator must fail the rollout, not become a zero score.
            result = self._rpc.post("/task/judge", body={
                "compressed_xml": compressed_xml,
                "line": line,
            })
        if not isinstance(result, dict):
            return
        if not result.get("judge_page", True):
            return
        self._best_judge = result

    def _compute_reward(self) -> float:
        if self._best_judge is None:
            return 0.0
        return 1.0 if self._best_judge.get("complete") else 0.0

    # ------------------------------------------------------------------
    # Container adb passthrough
    # ------------------------------------------------------------------

    def _adb_shell(self, *args: str, timeout: float = 15.0) -> str:
        if self._current_container is None:
            raise RuntimeError("adb_shell called before container acquired")
        return self._current_container.adb_shell(*args, timeout=timeout)

    def _emulator_setup(self) -> None:
        """Per-reset tweaks borrowed from reference's auto_test.start_emulator.

        Most of this is no-op because the snapshot already captured the
        state, but we re-apply in case a previous episode's actions
        mutated anything the snapshot doesn't own.
        """
        assert self._current_container is not None
        # All three adb calls in this setup path use a 30 s timeout. adb
        # itself returns in <100 ms when the docker layer is unblocked, but
        # at c=32+ concurrent emulator boots the host serialises per-
        # container ``docker exec`` setup and individual call latencies
        # slip to 5-15 s. The ceiling only kicks in under genuine contention.
        self._current_container.adb("root", timeout=30)
        self._current_container.adb("emu", "geo", "fix", "-122.156", "37.438", timeout=30)
        # Don't pin date for map tasks — map.me tiles depend on real time.
        if "map" not in (self._config.app or "").lower():
            self._adb_shell("date", '"2024-05-10 12:00:00"', timeout=30)

    def _launch_app(self, app_name: str) -> None:
        # All callers (reset post-controller-bind, _dispatch_action:open_app)
        # run after the controller is bound, so self._controller is always set.
        assert self._controller is not None
        self._controller.launch_app(self._resolve_package(app_name))

    def _resolve_package(self, app_name: str) -> str:
        """Fuzzy app_name → package via the in-container server's
        ``/env/resolve_package`` (wraps androidlab's ``find_package``).
        Falls back to ``app_name`` verbatim if RPC fails — the existing
        host-side path did the same on import error.
        """
        if self._rpc is None:
            return app_name
        try:
            return self._rpc.post("/env/resolve_package", body={"app_name": app_name})["package"]
        except Exception:
            return app_name

    def _get_screenshot_png(self) -> bytes | None:
        """Raw PNG bytes. Container streams the raw PNG via
        ``adb exec-out screencap -p``; we downsample it host-side."""
        if self._controller is None:
            return None
        raw = self._controller.get_screenshot_bytes()
        if raw is None:
            return None
        return _downscale_png(raw, max_dim=self._config.screenshot_max_dim)

    def _observe_via_rpc(self) -> dict[str, Any]:
        """One-shot batched observation for ``step``: screenshot +
        xml + compressed_xml + current_activity in a single RPC.
        Returns a dict with keys: ``screenshot``, ``obs_text``,
        ``compressed_xml``, ``current_activity``.

        Falls back to per-field RPCs if anything in the batched
        endpoint hiccups (older container without ``/env/observe``
        or transient backend error on one of the sub-fields).
        """
        if self._controller is None or self._rpc is None:
            return {"screenshot": None, "obs_text": None,
                    "compressed_xml": None, "current_activity": ""}
        prefer_ac = self._prefer_ac_xml()
        try:
            pkg = self._controller.observe(ac=prefer_ac)
        except Exception as e:
            logger.warning("batched observe failed (%s); falling back per-field", e)
            obs_text, compressed_xml = self._get_obs_text()
            return {
                "screenshot": self._get_screenshot_png(),
                "obs_text": obs_text,
                "compressed_xml": compressed_xml,
                "current_activity": self._get_current_activity(),
            }
        raw = pkg.get("screenshot")
        xml_str = pkg.get("xml")
        compressed = pkg.get("compressed_xml")
        # If primary XML source was empty / too short, fall back to the OTHER
        # source — mirrors the two-source try sequence in legacy ``_dump_xml``.
        # Symmetric in ``prefer_ac``: covers ac→non-ac as well as non-ac→ac.
        if not xml_str or len(xml_str) <= 200:
            other_fn = (
                self._controller.get_xml if prefer_ac
                else self._controller.get_ac_xml
            )
            try:
                other_xml = other_fn()
            except Exception as e:
                logger.debug("xml fallback %s failed: %s", other_fn.__name__, e)
                other_xml = None
            if other_xml and len(other_xml) > 200:
                xml_str = other_xml
                try:
                    compressed = self._rpc.post(
                        "/env/compressed_xml", body={"xml_str": xml_str},
                    )
                except Exception:
                    compressed = None
        obs_text = self._render_obs_text(xml_str, compressed) if compressed is not None else None
        return {
            "screenshot": (
                _downscale_png(raw, max_dim=self._config.screenshot_max_dim)
                if raw is not None else None
            ),
            "obs_text": obs_text,
            "compressed_xml": compressed,
            "current_activity": pkg.get("current_activity") or "",
        }

    def _render_obs_text(self, xml_str: str | None, compressed: Any) -> str | None:
        """Render the requested observation_text format from raw xml +
        compressed dict. Mirrors ``_get_obs_text``'s branch logic but
        takes the inputs as args instead of re-fetching them."""
        mode = self._config.observation_text or "none"
        if mode == "none":
            return None
        if mode.startswith("a11y_tree"):
            coord_unit = "norm" if mode.endswith(":norm") else "pixel"
            return self._render_a11y_tree(compressed, coord_unit)
        if mode.startswith("a11y_list") and xml_str:
            coord_unit = "norm" if mode.endswith(":norm") else "pixel"
            return self._render_a11y_list(xml_str, coord_unit)
        return None

    def _get_current_activity(self) -> str:
        if self._controller is None:
            return ""
        try:
            r = self._controller.get_current_activity()
            return r if isinstance(r, str) else ""
        except Exception:
            return ""

    def _run_adb_query(self, cmd: str) -> str:
        """Run an ``adb_query`` string (e.g. ``adb shell settings get ...``) inside the container."""
        if self._controller is None:
            return ""
        # Reference's controller accepts the full ``adb -s <device> ...`` command.
        return self._controller.execute_adb(cmd)

    # ------------------------------------------------------------------
    # XML / text observation
    # ------------------------------------------------------------------

    #: Task-id prefixes whose UI is only visible to the XMLParser a11y
    #: service, not to ``uiautomator dump``. Mirrors reference's rule at
    #: ``evaluation/evaluation.py:20`` (``if "map.me" in instruction or
    #: "pimusic" in instruction``) — those two apps are the only ones
    #: where reference sets ``accessibility=True``. We key on ``task_id``
    #: because our YAML ``APP`` field ("map.me" / "Pi Music Player") and
    #: instruction text vary, but task_ids are stable (``map_*``,
    #: ``pimusic_*``).
    _AC_XML_TASK_PREFIXES = ("map_", "pimusic_")

    def _prefer_ac_xml(self) -> bool:
        """Mirror reference's per-app XML-source selection."""
        tid = self._config.task_id or ""
        return any(tid.startswith(p) for p in self._AC_XML_TASK_PREFIXES)

    def _dump_xml(self) -> str | None:
        """Fetch current UI XML via the in-container controller.

        Each get_xml/get_ac_xml call already retries 5× with sleeps
        inside the container (matches reference's robustness). Falls
        back to the other source if the preferred one returns None,
        so no task is worse off than dump-only.
        """
        if self._controller is None:
            return None
        primary_fn, fallback_fn = (
            (self._controller.get_ac_xml, self._controller.get_xml)
            if self._prefer_ac_xml()
            else (self._controller.get_xml, self._controller.get_ac_xml)
        )
        for fn in (primary_fn, fallback_fn):
            try:
                xml = fn()
            except Exception as e:
                logger.debug("%s failed: %s", fn.__name__, e)
                xml = None
            if xml and len(xml) > 200:
                return xml
        logger.warning(
            "xml dump failed on both sources for task %s; judge will skip this step",
            self._config.task_id,
        )
        return None

    def _get_obs_text(self) -> tuple[str | None, Any]:
        """Return (rendered_text, compressed_xml_tree).

        compressed_xml is always computed (for judge) when a dump succeeds,
        regardless of observation_text.
        """
        xml_str = self._dump_xml()
        if xml_str is None or self._rpc is None:
            return None, None

        # Compression runs inside the container via /env/compressed_xml
        # (which wraps ``get_compressed_xml_from_str`` + ``json.loads``).
        # Container returns a parsed dict (or None on empty / parse fail)
        # — see docker/server.py for the parity contract with reference's
        # ``evaluation/task.py:24`` ``json.loads`` step.
        try:
            compressed = self._rpc.post("/env/compressed_xml", body={"xml_str": xml_str})
        except Exception as e:
            logger.warning("xml compression RPC failed for task %s: %s",
                           self._config.task_id, e)
            return None, None
        if compressed is None:
            logger.warning("xml compression returned empty for task %s", self._config.task_id)
            return None, None

        mode = self._config.observation_text or "none"
        if mode == "none":
            return None, compressed

        if mode.startswith("a11y_tree"):
            coord_unit = "norm" if mode.endswith(":norm") else "pixel"
            return self._render_a11y_tree(compressed, coord_unit), compressed

        if mode.startswith("a11y_list"):
            coord_unit = "norm" if mode.endswith(":norm") else "pixel"
            return self._render_a11y_list(xml_str, coord_unit), compressed

        logger.warning("unknown observation_text=%r; returning no text", mode)
        return None, compressed

    # ------------------------------------------------------------------
    # a11y coordinates — ONE conversion, owned by the action side
    #   _screen_w/h = native emulator pixels (from ``wm size``), the surface
    #                 ``_to_pixels`` projects onto.
    # ------------------------------------------------------------------

    def _a11y_point(self, x: int, y: int, coord_unit: str) -> tuple[int, int]:
        """Map one native-pixel a11y point into the emitted coord space.

        ``:norm`` is ``pixel_to_norm`` against the device dims — the exact
        inverse of the ``norm_to_pixel`` inside ``_to_pixels`` — so a centre
        the model echoes back as ``coordinate`` lands on the element it names.

        ``:pixel`` emits the native pixels ``_to_pixels`` *outputs*, while
        ``_to_pixels`` *consumes* ``[0, 1000]`` -- so it round-trips only for an
        agent whose adapter converts the model's pixels back to ``[0, 1000]``
        against these same dims. A model that emits ``[0, 1000]`` itself must be
        given ``a11y_*:norm``.
        """
        sw = self._screen_w or _AVD_NATIVE_SIZE[0]
        sh = self._screen_h or _AVD_NATIVE_SIZE[1]
        if coord_unit == "norm":
            nx, ny = pixel_to_norm(x, y, sw, sh)
            return nx, ny
        return x, y

    def _render_a11y_tree(self, compressed: Any, coord_unit: str) -> str:
        """Tree-shaped a11y view (reference's compressed JSON) with bounds
        rescaled to the requested coord space."""
        rescaled = _rescale_tree_bounds(
            compressed, lambda x, y: self._a11y_point(x, y, coord_unit)
        )
        try:
            return json.dumps(rescaled, ensure_ascii=False)
        except Exception:
            return str(rescaled)

    def _render_a11y_list(self, xml_str: str, coord_unit: str) -> str:
        """Parse UIAutomator XML into a flat numbered element list (androidworld-compatible)."""
        try:
            from lxml import etree
        except Exception:
            return ""
        try:
            root = etree.fromstring(xml_str.encode("utf-8"), parser=etree.XMLParser(recover=True))
        except Exception:
            return ""
        lines: list[str] = []
        for i, node in enumerate(root.iter("node")):
            attrs = node.attrib
            m = _BOUNDS_RE.match(attrs.get("bounds", ""))
            if not m:
                continue
            x1, y1, x2, y2 = map(int, m.groups())
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy = self._a11y_point((x1 + x2) // 2, (y1 + y2) // 2, coord_unit)
            label = (attrs.get("text") or attrs.get("content-desc") or "").strip()
            cls = (attrs.get("class") or "").rsplit(".", 1)[-1]
            flags: list[str] = []
            for flag in ("clickable", "scrollable", "checkable", "checked", "focused", "selected", "long-clickable"):
                if attrs.get(flag) == "true":
                    flags.append(flag.replace("-", ""))
            if not label and not flags:
                continue
            flags_s = f" [{','.join(flags)}]" if flags else ""
            label_s = f'"{label}"' if label else "(unlabeled)"
            lines.append(f"[{i}] {label_s} center=({cx},{cy}) <{cls}>{flags_s}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Action translation
    # ------------------------------------------------------------------

    def _dispatch_action(self, name: str, args: dict[str, Any]) -> LiteExecutedAction:
        """Translate cua-lite tool_call → in-container adb call via RPC.

        Converts the [0,1000]-normalized ``coordinate`` to native pixels,
        then calls the matching :class:`_RemoteController` method which
        round-trips a single HTTP POST to the container's
        ``/env/<action>`` endpoint. The adb shell command produced
        in-container mirrors android-lab reference's ``AndroidController``
        bit-for-bit (tap → ``input tap X Y``, text → 100×KEYCODE_DEL then
        ADBKeyboard b64 broadcast, etc.), so replay equality against the
        legacy host-side ``docker exec adb`` path is preserved.
        """
        ctrl = self._controller
        assert ctrl is not None

        if name == "tap":
            x, y = self._to_pixels(args.get("coordinate"))
            clicks = args.get("clicks", 1)
            if clicks >= 2:
                ctrl.tap(x, y)
                time.sleep(0.05)
                ctrl.tap(x, y)
                return {"call": "double_tap", "args": {"x": x, "y": y}}
            ctrl.tap(x, y)
            return {"call": "tap", "args": {"x": x, "y": y}}

        if name == "drag":
            sx, sy = self._to_pixels(args.get("start_coordinate"))
            ex, ey = self._to_pixels(args.get("coordinate"))
            duration_ms = int(
                _duration_seconds(args, "drag", default=0.8) * 1000
            )
            ctrl.swipe_precise((sx, sy), (ex, ey), duration_ms=duration_ms)
            return {
                "call": "drag",
                "args": {"from": (sx, sy), "to": (ex, ey), "duration_ms": duration_ms},
            }

        if name == "long_press":
            x, y = self._to_pixels(args.get("coordinate"))
            duration_ms = int(
                _duration_seconds(args, "long_press", default=1.0) * 1000
            )
            ctrl.long_press(x, y, duration_ms=duration_ms)
            return {"call": "long_press", "args": {"x": x, "y": y, "duration_ms": duration_ms}}

        if name in ("swipe", "drag"):
            sx, sy = self._to_pixels(args.get("start_coordinate"))
            ex, ey = self._to_pixels(args.get("coordinate"))
            duration_ms = int(
                _duration_seconds(args, name, default=0.4) * 1000
            )
            ctrl.swipe_precise((sx, sy), (ex, ey), duration_ms=duration_ms)
            return {"call": name, "args": {"from": (sx, sy), "to": (ex, ey)}}

        if name == "type":
            text = args.get("text", "") or ""
            # Reference's ``text`` clears the field via 100x KEYCODE_DEL
            # then broadcasts the b64-encoded payload; we delegate to it
            # to match its exact behavior.
            ctrl.text(text)
            # Reference's TextOnlyExecutor follows ``text`` with Enter
            # (paper numbers assume submit-on-type for search boxes etc.).
            ctrl.enter()
            return {"call": "type", "args": {"text": text}}

        if name == "system_button":
            btn = args.get("button", "")
            handler = {
                "Home": ctrl.home, "Back": ctrl.back,
                "Enter": ctrl.enter, "Menu": ctrl.menu,
                "Recent": ctrl.recent,
            }.get(btn)
            if handler:
                handler()
                return {"call": "system_button", "args": {"button": btn}}
            raise ValueError(
                f"system_button: unknown button {btn!r}; expected one of "
                "['Back', 'Enter', 'Home', 'Menu', 'Recent']"
            )

        if name == "open_app":
            app_name = args.get("app_name", "")
            self._launch_app(app_name)
            return {"call": "open_app", "args": {"app_name": app_name}}

        if name == "wait":
            duration = _duration_seconds(args, "wait", default=1.0, allow_zero=True)
            time.sleep(duration)
            return {"call": "wait", "args": {"duration_s": duration}}

        if name == "screenshot":
            return {"call": "screenshot_noop", "args": {}}

        # No ``pinch`` branch: ``android_supported_actions()`` subtracts it for the
        # whole Android family, so the shared gate in ``step`` noops it before
        # dispatch.
        logger.warning("Unknown action: %s(%s)", name, args)
        return {"call": "noop", "args": {"name": name, "reason": "unknown action"}}

    def _to_pixels(self, coord: list[int] | None) -> tuple[int, int]:
        w = self._screen_w or _AVD_NATIVE_SIZE[0]
        h = self._screen_h or _AVD_NATIVE_SIZE[1]
        # clamp=False preserves this env's legacy no-clamp behavior until its
        # coordinate contract is validated for canonical clamping.
        return norm_to_pixel(coord, w, h, clamp=False, on_malformed="raise")


# ---------------------------------------------------------------------------
# Task registration — walk YAML configs, resolve judge class, register
# ---------------------------------------------------------------------------

#: Module-level task_id → config table, populated by ``_load_and_register_all``.
#: Looked up by :meth:`AndroidLabEnv.bind` when callers pass a task_id string
#: instead of a full config object.
_TASK_CONFIGS: dict[str, AndroidLabTaskConfig] = {}



def _make_env(
    *,
    config: AndroidLabTaskConfig | None = None,
    **kwargs: Any,
) -> AndroidLabEnv:
    """Factory for gym.make()."""
    if config is None:
        config = AndroidLabTaskConfig(
            **{k: v for k, v in kwargs.items() if k in AndroidLabTaskConfig.__dataclass_fields__}
        )
    else:
        overrides = {k: v for k, v in kwargs.items() if k in AndroidLabTaskConfig.__dataclass_fields__}
        if overrides:
            # Don't let concurrent resets mutate a shared metadata dict.
            overrides.setdefault("metadata", {**config.metadata})
            config = replace(config, **overrides)
    env_kwargs = {k: v for k, v in kwargs.items() if k not in AndroidLabTaskConfig.__dataclass_fields__}
    return AndroidLabEnv(config=config, **env_kwargs)


def _load_and_register_all() -> int:
    """Register all 138 androidlab tasks from the checked-in tasks.json.

    See ``scripts/utils/tasks.sh`` — re-run that script when
    the android-lab pin in ``docker/Dockerfile`` changes.
    """
    tasks_path = Path(__file__).parent / "data" / "tasks.json"
    if not tasks_path.is_file():
        raise FileNotFoundError(
            f"{tasks_path} not found. Re-generate with:\n"
            f"  bash lite/gym/envs/androidlab/scripts/utils/tasks.sh\n"
            "(requires cua-lite/androidlab:latest docker image — run scripts/install.sh first)."
        )
    with open(tasks_path) as f:
        tasks = json.load(f)

    # Declare env-wide make defaults (reset_timeout) once from
    # default.yaml make_kwargs. Per-task register() kwargs / gym.make() override.
    registry.set_env_make_kwargs("androidlab", CFG.make_kwargs)

    count = 0
    for task_id, meta in tasks.items():
        app_name = meta.get("app", "")
        # No "task_id" here: identity is framework-injected (registry.register
        # for the spec; the LiteBaseEnv.metadata property for live envs).
        others = {
            "app": app_name,
            "package": meta.get("package", ""),
            "category": meta.get("category", ""),
            "metric_type": meta.get("metric_type", ""),
            "adb_query": meta.get("adb_query"),
        }
        cfg = AndroidLabTaskConfig(
            task_id=task_id,
            app=app_name,
            instruction=(
                f"You should use {app_name} to complete the following task: "
                + meta.get("instruction", "")
            ),
            adb_query=meta.get("adb_query"),
            judge_class_name=meta.get("judge_class_name", task_id),
            app_module_name=meta.get("app_module_name", ""),
            metadata=others,
        )
        _TASK_CONFIGS[task_id] = cfg
        register(
            key=f"androidlab@{task_id}",
            entry_point=partial(_make_env, config=cfg),
            # reset_timeout (600s, for AVD Quick Boot snapshot restore headroom)
            # is an env-wide default (CFG.make_kwargs, applied via
            # set_env_make_kwargs above) — not repeated per task here.
            split="eval",
            # Same-source contract: registered copy == the env's
            # builder output (incl. the module-constant ``apps`` catalog and
            # the no-override extra_tool_schemas default).
            metadata=AndroidLabEnv._task_metadata(others),
        )
        count += 1
    return count


_TASK_COUNT = _load_and_register_all()
logger.info("androidlab: registered %d tasks", _TASK_COUNT)


class AndroidLabServices(ContainerServices):
    """Env-server capabilities for androidlab: docker-image presence check
    (``ensure``) + per-container reconciliation (``live_ids``/``reap`` inherited
    from :class:`ContainerServices`)."""

    def ensure(self, env_id: str) -> None:
        _ensure_services(env_id)

    def health(self, env_id: str) -> None:
        _health_check(env_id)


register_services("androidlab", AndroidLabServices())
from lite.gym.services import BackendFamily, register_family  # noqa: E402

register_family("androidlab", BackendFamily.DEDICATED)
