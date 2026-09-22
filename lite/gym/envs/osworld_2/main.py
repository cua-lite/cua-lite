"""osworld_2 — CUA-Lite gym wrapper for the official OSWorld-v2.1 benchmark.

108 capability-graded desktop tasks (release osworld-v2.1, ids 001–108) run on a
locally-managed VM-in-Docker container (`cua-lite/osworld_2`; QEMU/KVM booting the gated v2
disk osworld-v2-ubuntu-x86.qcow2). DEDICATED, one container per trajectory.

EVAL-IN-CONTAINER (androidworld pattern): OSWorld-V2's `desktop_env` is baked into the
image and driven by an in-container HTTP server (`docker/server.py`); the cua-lite HOST
imports ZERO `desktop_env` and just talks to the container over one JSON-RPC port. THIS is
what lets osworld_2 (V2) coexist with osworld v1 + lite.osworld (v1) in ONE uv — their host
`desktop_env` dist is untouched; osworld_2 carries V2 only inside its image. The host still
owns the container (naming + reaping) and translates LiteDesktopActionSpace → pyautogui
host-side (a pure function, no `desktop_env`).

Prerequisites (see README):
  - the derived image `cua-lite/osworld_2` built (install.sh; needs the local V2 checkout)
  - hf auth login + accepted gates on xlangai/v2-image and xlangai/osworld_v2_tasks
  - uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh
  - KVM (/dev/kvm), /dev/net/tun, the built image, the v2 qcow2, task classes (ensure-checked).

Task ENUMERATION works with none of the above — it reads the vendored data/ (no container,
no `desktop_env`); the image + gated assets are needed only to reset().

Usage:
    uv run python -c "import lite.gym as gym; print(len(gym.registry.task_ids('osworld_2')))"
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import requests

from lite.core.messages.final import (
    MAX_STEPS_STOP_REASON,
    STOP_REASON_INFO_KEY,
)
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.schemas import BaseTools
from lite.gym.base import LiteBaseEnv

# Reuse v1's action translation + report_infeasible tool set VERBATIM — pure functions (no
# `desktop_env`): the LiteDesktop→pyautogui verbs don't change v1↔v2, single source of
# truth. These three are v1's PUBLIC v1/v2-shared surface (no private reach across envs).
# (This import also registers v1 `osworld`, which is harmless.)
from lite.gym.envs.osworld.main import OsworldTools, reap_pending_build, to_pyautogui
from lite.gym.envs.osworld_2.container import OSWorldV2Container, OSWorldV2ContainerFactory
from lite.gym.errors import TrueInfraFailure, EnvDepsMissingError
from lite.gym.registry import register, registry
from lite.gym.services import EnvServerResource
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
    LiteExecutedAction,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.rpc import json_rpc
from lite.gym.utils.config.identity import EnvIdentity
from lite.gym.utils.feedback.errors import (
    append_feedback,
    MODEL_ACTION_ERROR_TYPES,
    ToolErrorFeedback,
    error_only_feedback,
    record_batch_abort,
    record_model_action_error,
    record_tool_execution_error,
    unsupported_action_message,
)
from lite.gym.utils.feedback.ingress import (
    invalid_action_message,
    prepare_env_tool_calls,
    standalone_tool_call_feedback_with_reason,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.utils.feedback.surface import (
    resolve_extra_tools,
    resolve_valid_actions,
)
from lite.utils.image import png_from_b64_async

logger = logging.getLogger(__name__)

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)
_RELEASE = json.loads((Path(ENV_DIR) / "data" / "release.json").read_text(encoding="utf-8"))

# ============================================================================
# Config defaults — read once from configs/default.yaml. Named constants.
# ============================================================================
# env_kwargs (per-instance)
_MAX_STEPS = CFG.env_kwargs["max_steps"]
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]
# server_kwargs (per-deploy infra)
_IMAGE = CFG.server_kwargs["image"]
_RAM = CFG.server_kwargs["ram_size"]
_CPU = CFG.server_kwargs["cpu_cores"]
_DISK = CFG.server_kwargs["disk_size"]
_BOOT_TO = float(CFG.server_kwargs["boot_timeout"])
_SCREEN_W = CFG.server_kwargs["screen_width"]
_SCREEN_H = CFG.server_kwargs["screen_height"]
_MAX_WORKERS = CFG.server_kwargs["max_workers"]
# V2 service knobs — drive exclude_reason.
_WEBSITE_SUFFIX = CFG.server_kwargs["website_host_suffix"]
_GITLAB_URL = CFG.server_kwargs["gitlab_url"]
_GITLAB_TOKEN = CFG.server_kwargs["gitlab_private_token"]
_USER_SIM_MODEL = CFG.server_kwargs["user_sim_model"]

_EVAL_MODEL = CFG.server_kwargs["eval_model"]   # LLM-judge evaluator model override (None → V2 default gpt-4o)
_HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))   # gates the ~18 LLM-judge tasks (else they'd mis-score)
_ASSET_BASE = os.environ.get("OSWORLD_FILE_BASE_URL")
#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = OsworldTools.get_tool_names() | LiteFinishToolSet.get_tool_names()

# Env vars threaded into EACH container (only the set ones) so the in-container task setup + evaluators
# can reach the configured services. WEBSITE_HOST_SUFFIX enables the stateful-website tasks
# (prepare_stateful_website_urls; per-session cookie isolation keeps concurrent trajectories safe).
# OPENAI_API_KEY/OPENAI_BASE_URL feed the ~18 LLM-judge evaluators (desktop_env model_client reads them
# at evaluate(); default provider=openai, model=gpt-4o) — same host→container passthrough as mobileworld
# (USER_AGENT_API_KEY←OPENAI_API_KEY) and the browsergym/online_mind2web host-OPENAI_API_KEY eval convention.
_SERVICE_ENV = {k: v for k, v in {
    "WEBSITE_HOST_SUFFIX": _WEBSITE_SUFFIX,
    "GITLAB_URL": _GITLAB_URL,
    "GITLAB_PRIVATE_TOKEN": _GITLAB_TOKEN,     # V2 controllers/gitlab.py reads GITLAB_PRIVATE_TOKEN (not GITLAB_TOKEN)
    "OSWORLD_USER_SIM_MODEL": _USER_SIM_MODEL,  # V2 user_simulator.py reads OSWORLD_USER_SIM_MODEL
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
    "OSWORLD_EVAL_MODEL_NAME": _EVAL_MODEL,
    "OSWORLD_BENCHMARK_RELEASE": _RELEASE["hf_revision"],
    "OSWORLD_FILE_BASE_URL": _ASSET_BASE,
}.items() if v}

# V2 image sudo password — passed to the in-container server (not used host-side).
_CLIENT_PASSWORD = "osworld-public-evaluation"

_QCOW2_CFG = CFG.server_kwargs["qcow2_path"]
_QCOW2 = _QCOW2_CFG if os.path.isabs(_QCOW2_CFG) else str((Path(ENV_DIR) / _QCOW2_CFG).resolve())
_QCOW2_SIZE = int(_RELEASE["qcow2"]["size"])
_CACHE_DIR = str(Path(ENV_DIR) / ".cache")
_TASK_CLASS_DIR = str(Path(_CACHE_DIR) / "task_class")   # gated task_<id>.py, mounted into the container
_DATA_DIR = Path(ENV_DIR) / "data"                       # vendored id/capability manifests
_TASKS_REPO = str(_RELEASE["tasks"]["repo"])
_HF_REVISION = str(_RELEASE["hf_revision"])
_TASK_IDS = tuple(json.loads((_DATA_DIR / "test_v2.json").read_text(encoding="utf-8"))["tasks"])
_TASK_COUNT = len(_TASK_IDS)
_TASK_CLASS_IDENTITY = f"{_TASKS_REPO}@{_HF_REVISION}:{_TASK_COUNT}"

_README = "lite/gym/envs/osworld_2/README.md"
_INSTALL = "uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh"

# RPC timeouts (blocking HTTP to the in-container eval server).
_RESET_RPC_TIMEOUT = 600.0   # /reset runs the task's setup (downloads, launch commands)
_STEP_RPC_TIMEOUT = 180.0
_EVAL_RPC_TIMEOUT = 300.0   # upstream CAD evaluators allow up to 240s

_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="osworld_2-exec")
atexit.register(lambda: _EXECUTOR.shutdown(wait=False, cancel_futures=True))


# ---------------------------------------------------------------------------
# In-container eval-server RPC (JSON; the host never imports desktop_env)
# ---------------------------------------------------------------------------
def _rpc(base_url: str, path: str, body: dict | None = None, timeout: float = _STEP_RPC_TIMEOUT) -> dict:
    """Thin label/timeout binding over the shared JSON RPC — the two
    osworld envs carried byte-identical copies of this client)."""
    return json_rpc(base_url, path, body, timeout=timeout, label="osworld_2")


# ---------------------------------------------------------------------------
# Prerequisite checks (ensure — terminal, 501). No desktop_env check (it's in the image).
# ---------------------------------------------------------------------------
def _dep_error(what: str) -> EnvDepsMissingError:
    return EnvDepsMissingError(what=what, install=_INSTALL, see=_README)


def _check_kvm() -> None:
    if not os.path.exists("/dev/kvm"):
        raise _dep_error("/dev/kvm not present — osworld_2 needs hardware virtualization (KVM); no software fallback")


def _check_tun() -> None:
    if not os.path.exists("/dev/net/tun"):
        raise _dep_error("/dev/net/tun not present — the VM-in-Docker needs it for guest "
                         "networking; without it QEMU falls back to usermode and the guest server hangs")


def _check_image() -> None:
    # Freshness gate (image_for → ensure_runnable): the image must exist AND its lite.src_hash label
    # must match the current docker/Dockerfile — so editing it without a rebuild fails HERE instead
    # of silently running the stale image (docs/envs.md#image-build-and-freshness).
    from lite.gym.utils.backend.docker import require_image_present
    from lite.gym.utils.backend.freshness import image_for
    require_image_present(image_for("osworld_2", tag=_IMAGE))


def _check_qcow2() -> None:
    if not os.path.exists(_QCOW2):
        raise _dep_error(
            "v2 VM disk cache is missing (osworld-v2-ubuntu-x86.qcow2); "
            "provision it via install.sh — HF-gated"
        )
    size = os.path.getsize(_QCOW2)
    if size != _QCOW2_SIZE:
        raise _dep_error(
            f"v2 VM disk cache has size {size}, expected {_QCOW2_SIZE}; "
            "re-run install.sh provision"
        )


def _check_task_classes() -> None:
    p = Path(_TASK_CLASS_DIR)
    stamp = p / ".task_class_revision"
    try:
        revision = stamp.read_text().strip() if stamp.is_file() else ""
    except OSError as exc:
        raise _dep_error(
            "gated task class revision stamp is unreadable; re-run install.sh "
            "provision"
        ) from exc
    missing = [tid for tid in _TASK_IDS if not (p / f"task_{tid}.py").is_file()]
    if missing or revision != _TASK_CLASS_IDENTITY:
        detail = (
            f"{len(missing)} required task files missing"
            if missing
            else "revision stamp is stale or missing"
        )
        raise _dep_error(
            f"gated task classes are not ready ({detail}); download via "
            "install.sh (accept the xlangai/osworld_v2_tasks gate + "
            "`hf auth login`)"
        )


def _check_website() -> None:
    """EAGER readiness probe for the stateful-website SINGLETON (mimics browsergym's ensure_services
    gate). When `website_host_suffix` is configured, probe one app's /api/state at ensure so a down /
    misconfigured deployment surfaces up front — vs the LAZY fallback where it only shows when a
    website task's in-container setup fails mid-run. NON-fatal (warn, not raise): only ~31 of the
    scored tasks need it, so a transient website outage must not block the other 51.

    The website is an EXTERNAL always-on SINGLETON — cua-lite never
    creates/reaps it, so this is a reachability check, not a start. To SELF-HOST, bring the
    OSWorld-web stack up once per env-server here (lazy) or at startup (eager), browsergym-style."""
    if not _WEBSITE_SUFFIX:
        return
    # Best-effort probe of one known app (mailhub, the largest task group). Short 3s timeout so a
    # slow/unreachable deployment can't stall ensure (it runs under the process-wide services lock).
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # verify=False on the internal probe
    for scheme in ("https", "http"):
        url = f"{scheme}://mailhub.{_WEBSITE_SUFFIX}/api/state"
        try:
            if requests.get(url, timeout=(3, 3), verify=False).status_code == 200:
                return
        except Exception:
            pass
    logger.warning("osworld_2 website %s unreachable — website tasks may error at reset", url)


def _check_runtime_deps() -> None:
    if _ASSET_BASE and "://" not in _ASSET_BASE and (
        not os.path.isabs(_ASSET_BASE) or not os.path.isdir(_ASSET_BASE)
    ):
        raise _dep_error(f"OSWORLD_FILE_BASE_URL must be an existing absolute directory: {_ASSET_BASE}")
    _check_kvm()
    _check_tun()
    _check_image()
    _check_qcow2()
    _check_task_classes()


def _ensure_services(env_id: str) -> None:
    """env-server startup hook: terminal pre-launch checks → EnvDepsMissingError (501)."""
    _check_runtime_deps()
    _check_website()   # eager reachability warning (non-fatal); lazy per-task setup is the fallback


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class OSWorldV2Config:
    """Per-task config. The gated task-class file is mounted in the container's
    evaluation_examples/task_class; the host passes only the task_id — the server loads the live
    BaseTask instance + runs setup/eval."""

    task_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment (thin RPC client — no desktop_env on the host)
# ---------------------------------------------------------------------------
class OSWorldV2Env(LiteBaseEnv, EnvServerResource):
    """CUA-Lite wrapper: owns one VM-in-Docker container per trajectory + an in-container
    eval server it drives over JSON-RPC. Non-poolable DEDICATED; cancellation-safe local-
    container leak handling (copied from v1 osworld)."""

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = OsworldTools

    def __init__(
        self,
        *,
        config: OSWorldV2Config,
        max_steps: int = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        **kwargs: Any,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"OSWorldV2Env got unexpected env kwargs: {unknown}")
        # Copy metadata, don't alias: `config` is the shared registration-side
        # object (one per task, all instances) — an instance-side metadata
        # write must never reach it.
        self._config = dataclasses.replace(config, metadata=dict(config.metadata))
        self._max_steps = max_steps
        self._post_action_delay = post_action_delay
        # ``valid_actions`` used to land in ``**kwargs`` and be silently
        # ignored. Resolved here so a typo fails at the config boundary and a
        # real subset reaches ``metadata.valid_actions``. ``None`` = full GUI
        # surface, which is this env's registered default. Runtime
        # invalid/unsupported feedback is owned by ``step``.
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="osworld_2", platform="desktop",
        )
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._step_count = 0
        self._container: OSWorldV2Container | None = None
        self._pending: OSWorldV2Container | None = None
        self._pending_cf_future: Any = None

    @staticmethod
    def _task_metadata(config) -> LiteCUAMetadata:
        """Same-source metadata builder.
        extra_tool_schemas mirrors bind()'s default resolution, so the yaml
        default flows to BOTH sides."""
        return LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=resolve_extra_tools(
                _EXTRA_TOOLS, tools=OsworldTools, env_name="osworld_2",
            ),
            others={
                **config.metadata,
                # list(...): the spread aliases config.metadata's list values —
                # sever the registered/live others from the load-time
                # capabilities list.
                "capabilities": list(config.metadata.get("capabilities", [])),
            },
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendments: valid_actions / extra_tool_schemas resolved
        # at construction.
        return dataclasses.replace(
            self._task_metadata(self._config),
            valid_actions=self._valid_actions,
            extra_tool_schemas=self._extra_tool_schemas,
        )

    @property
    def external_resource_id(self) -> str | None:
        c = self._container or self._pending
        return c.name if c is not None else None

    async def reset(self) -> LiteEnvObservation:
        loop = asyncio.get_event_loop()
        _check_runtime_deps()

        if self._pending_cf_future is not None:
            self._spawn_background_destroy(self._pending_cf_future, self._pending)
            self._pending_cf_future = None
            self._pending = None
        await self._teardown_existing()

        self._step_count = 0

        identity = getattr(self, "identity", None) or EnvIdentity()
        volume_size = self._config.metadata.get("volume_size")
        factory = OSWorldV2ContainerFactory(
            qcow2_path=_QCOW2, task_class_dir=_TASK_CLASS_DIR, image=_IMAGE,
            asset_dir=_ASSET_BASE if _ASSET_BASE and "://" not in _ASSET_BASE else "",
            ram_size=_RAM, cpu_cores=_CPU, disk_size=_DISK, volume_size=volume_size,
            website=bool(self._config.metadata.get("website")),
            boot_timeout=_BOOT_TO,
            screen_width=_SCREEN_W, screen_height=_SCREEN_H, client_password=_CLIENT_PASSWORD,
            service_env=_SERVICE_ENV,
            session_id=identity.session_id, token_hash=identity.token_hash,
            server_port=identity.server_port, task_id=self._config.task_id,
        )
        # build() is fast (port + name) — tracked future so a cancel can't orphan the port.
        build_cf = _EXECUTOR.submit(factory.build)
        self._pending_cf_future = build_cf
        try:
            container = await asyncio.wrap_future(build_cf, loop=loop)
        except BaseException:
            self._pending_cf_future = None
            self._spawn_background_destroy(build_cf, None)
            raise
        self._pending = container
        self._pending_cf_future = None

        # start() = docker run + launch server + wait /healthz vm_ready (the slow boot).
        start_cf = _EXECUTOR.submit(container.start)
        self._pending_cf_future = start_cf
        try:
            await asyncio.wrap_future(start_cf, loop=loop)
        except BaseException:
            self._pending_cf_future = None
            self._pending = None
            if start_cf.done():
                await loop.run_in_executor(_EXECUTOR, container.destroy)
            else:
                self._spawn_background_destroy(start_cf, container)
            raise
        self._pending_cf_future = None
        self._container = container
        self._pending = None

        # /reset: the in-container server loads the BaseTask instance + runs its setup.
        res = await loop.run_in_executor(
            _EXECUTOR, _rpc, container.base_url, "/reset",
            {"task_id": self._config.task_id},
            _RESET_RPC_TIMEOUT + (660.0 if volume_size else 0.0))
        return LiteEnvObservation(image=await png_from_b64_async(res.get("screenshot_b64")),
            text=res.get("instruction", ""),
        )

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        actions, ingress_errors = prepare_env_tool_calls(actions, metadata)
        loop = asyncio.get_event_loop()
        terminated = False
        # Model-emitted calls that ENDED the episode. They get no continuation
        # observation: devs/migration/verify.py forbids a tool result for a
        # terminal call. Keyed on the env-internal projected ``action["call_id"]``
        # and NOT on ``result_call_id``, because an INTERNAL finish call has no
        # model-call id and the loop-detect wrapper's injected ``terminate``
        # carries the intercepted NON-finish model call's id as
        # ``_result_call_id`` -- that call must still be answered.
        terminal_call_ids: set[str] = set()
        stop_reason: str | None = None
        executed: list[LiteExecutedAction] = []
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        base = self._container.base_url if self._container is not None else None
        # One frame PER EXECUTED ACTION, in action order (the contract the agent
        # layer consumes). Filled inside the loop below.
        step_screenshots: list[bytes] = []

        async def capture_frame() -> bytes | None:
            """The guest's current frame; ``None`` when the server returns none."""
            sr = await loop.run_in_executor(
                _EXECUTOR, _rpc, base, "/screenshot", {}, _STEP_RPC_TIMEOUT
            )
            return await png_from_b64_async(sr.get("screenshot_b64"))

        for index, (action, result_call_id) in enumerate(actions):
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
                if tool_feedback.carrier == "current" and base is not None:
                    if (frame := await capture_frame()) is not None:
                        step_screenshots.append(frame)
                continue

            # Infeasible-report paths → OSWorld "FAIL" (feeds v2 evaluate()'s last-action check).
            if name == "report_infeasible" or (
                name == "terminate" and args.get("status") == "failure"
            ) or (name == "response" and "[infeasible]" in str(args.get("text", "")).lower()):
                terminated = True
                stop_reason = name
                # ``report_infeasible`` shares this branch but is NOT a canonical
                # finish tool: ``lite/data/utils/rows.py``'s row contract keys
                # finish calls on ``LiteFinishToolSet`` alone, and sibling envs pair a
                # result to it on purpose. Only the canonical finish calls are
                # terminal for this rule.
                if name in LiteFinishToolSet.get_tool_names() and action.get("call_id"):
                    terminal_call_ids.add(action["call_id"])
                if base is not None:
                    try:
                        await loop.run_in_executor(
                            _EXECUTOR,
                            _rpc,
                            base,
                            "/step",
                            {"cmd": "FAIL", "pause": 0},
                            _STEP_RPC_TIMEOUT,
                        )
                    except Exception as e:
                        record_tool_execution_error(
                            action_errors, result_call_id, e, action_name=name
                        )
                        executed.append({
                            "call": "noop",
                            "args": {"name": name, "reason": str(e)},
                        })
                break
            if name in LiteFinishToolSet.get_tool_names():
                terminated = True
                stop_reason = name
                if action.get("call_id"):
                    terminal_call_ids.add(action["call_id"])
                break

            invalid_action = invalid_action_message(action, metadata.valid_actions)
            if invalid_action:
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(invalid_action)
                executed.append({
                    "call": "noop",
                    "args": {"name": name, "reason": invalid_action},
                })
                continue

            try:
                pyautogui_str = to_pyautogui(name, args, _SCREEN_W, _SCREEN_H)
            except MODEL_ACTION_ERROR_TYPES as e:
                record_model_action_error(
                    action_errors, result_call_id, e, action_name=name
                )
                executed.append({"call": "noop", "args": {"name": name, "reason": str(e)}})
                # POLICY: a MODEL fault costs this action and nothing more. It
                # never reached the screen, so the state the tail was chosen
                # against is exactly the state still on it -- there is nothing to
                # protect the tail from. The model gets the reason as text plus a
                # frame for this slot, then the batch continues. Contrast the
                # interface-failure arm: THERE the action may have half executed
                # and the screen is unknown, which is a different fault.
                if base is not None and (frame := await capture_frame()) is not None:
                    step_screenshots.append(frame)
                continue
            if pyautogui_str and base is not None:
                try:
                    await loop.run_in_executor(
                        _EXECUTOR,
                        _rpc,
                        base,
                        "/step",
                        {"cmd": pyautogui_str, "pause": 0},
                        _STEP_RPC_TIMEOUT,
                    )
                except (ConnectionResetError, ConnectionRefusedError) as e:
                    # R1: the BACKEND is gone (connect refused, reset, retries
                    # exhausted). The model cannot cause this and must never be
                    # shown it -- degrading it to feedback trains the model
                    # against a screen that stopped updating. Raise so the
                    # trajectory is discarded and retried against a fresh env.
                    raise TrueInfraFailure(
                        f"osworld_2 backend failed during {name}: {e}"
                    ) from e
                except Exception as e:
                    # A non-200 is NOT separable here: the guest crashing on a
                    # pyautogui string the MODEL supplied looks the same as a
                    # server fault. Keep it model-visible rather than discard a
                    # trajectory that may be the model's own mistake.
                    record_tool_execution_error(
                        action_errors, result_call_id, e, action_name=name
                    )
                    executed.append({
                        "call": "noop",
                        "args": {"name": name, "reason": str(e)},
                    })
                    record_batch_abort(action_errors, result_call_id, actions[index + 1:])
                    break
                # Settle AFTER EACH action — mirrors upstream's per-action `time.sleep(pause)` (step()
                # sleeps at its end, default 2s). In a batched turn this lets action N's UI update before
                # action N+1 fires (else e.g. a `type` runs before a preceding `click` focuses the field).
                # Done host-side + async so the RPC executor thread isn't held for the delay (concurrency).
                if self._post_action_delay > 0:
                    await asyncio.sleep(self._post_action_delay)
                executed.append({"call": pyautogui_str})
                # Capture AFTER the settle delay, or the frame records the
                # pre-settle screen.
                if (frame := await capture_frame()) is not None:
                    step_screenshots.append(frame)
            elif name in ("screenshot", "cursor_position"):
                # Read-only: no pyautogui command to send, but the action DID run
                # and owes its frame like any other. No settle delay — it cannot
                # have changed the screen.
                if base is not None and (frame := await capture_frame()) is not None:
                    step_screenshots.append(frame)
            else:
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(
                        unsupported_action_message(name)
                    )
                executed.append({"call": "noop", "args": {"name": name}})

        self._step_count += 1
        truncated = not terminated and self._step_count >= self._max_steps
        if truncated:
            stop_reason = MAX_STEPS_STOP_REASON

        if base is not None and not step_screenshots:
            # Nothing ran (empty batch, a terminal-only turn, or every call
            # rejected). The turn still owes the model one current observation.
            if (frame := await capture_frame()) is not None:
                step_screenshots.append(frame)

        reward = None
        info: dict[str, Any] = {EXECUTED_ACTIONS_INFO_KEY: executed}
        if stop_reason is not None:
            info[STOP_REASON_INFO_KEY] = stop_reason
        if (terminated or truncated) and base is not None:
            # Upstream task 064 allows 3,180s for three motion-planning replays.
            eval_timeout = 3600.0 if self._config.task_id == "064" else _EVAL_RPC_TIMEOUT
            er = await loop.run_in_executor(_EXECUTOR, _rpc, base, "/evaluate", {}, eval_timeout)
            reward = er.get("reward")
            if er.get("payload") is not None:
                info["evaluate_payload"] = er["payload"]
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward, terminated=terminated, truncated=truncated, info=info,
            ),
            ordered_call_ids=result_call_ids,
            continue_call_ids=[
                call_id for call_id in result_call_ids
                if call_id not in terminal_call_ids
            ],
            images=step_screenshots,
            text=None,
            feedback=action_errors,
        )

    async def close(self) -> None:
        loop = asyncio.get_event_loop()
        if self._container is not None:
            try:
                await loop.run_in_executor(_EXECUTOR, _rpc, self._container.base_url, "/close", {}, 15.0)
            except Exception:
                pass
        if self._pending_cf_future is not None:
            self._spawn_background_destroy(self._pending_cf_future, self._pending)
            self._pending_cf_future = None
            self._pending = None
        await self._teardown_existing()

    # ── internal ──────────────────────────────────────────────────────────────
    async def _teardown_existing(self) -> None:
        loop = asyncio.get_event_loop()
        for attr in ("_container", "_pending"):
            c = getattr(self, attr)
            if c is not None:
                try:
                    await loop.run_in_executor(_EXECUTOR, c.destroy)
                except Exception as e:
                    logger.warning("destroy %s failed: %s", c.name, e)
                finally:
                    # Null AFTER destroy returns, and only if still ours —
                    # an eager null blanks external_resource_id while the
                    # container is still alive in docker, inviting the drift
                    # reaper to race our own teardown; same fix as
                    # mobileworld's _destroy_container.
                    if getattr(self, attr) is c:
                        setattr(self, attr, None)

    def _spawn_background_destroy(self, cf: Any, container: OSWorldV2Container | None) -> None:
        reap_pending_build(cf, container, thread_name="osworld_2-bg-destroy")


# ---------------------------------------------------------------------------
# Task registration — two-tier; enumeration needs NO gated code / no container.
# ---------------------------------------------------------------------------
def _exclude_reason(task_id: str, hitl_ids: set[str], dep: dict[str, Any], *,
                    user_sim_model: str | None, website_suffix: str | None,
                    gitlab_ok: bool, has_openai_key: bool) -> str | None:
    """Registration-time exclude reason (first-match-wins). The gate config is PASSED IN (not read
    from module globals) so this is a pure function — testable by injecting configs, no patching."""
    if (task_id in hitl_ids or dep.get("user_sim")) and not user_sim_model:
        return "human_in_the_loop"
    if dep.get("website") and not website_suffix:
        return "website"
    if dep.get("gitlab") and not gitlab_ok:
        return "gitlab"
    if dep.get("llm_judge") and not has_openai_key:
        return "llm_judge"   # evaluator calls an LLM (desktop_env model_client) at evaluate(); no host OPENAI_API_KEY → would 500/mis-score
    # Fidelity limits — cua-lite can't reproduce the OFFICIAL score for these, so exclude them from
    # the scored set rather than report a wrong reward (see osworld_2 eval-fidelity audit D1/D2):
    if dep.get("multi_phase"):
        return "multi_phase"   # MultiPhaseTask: official runs N phase sub-trajectories; cua-lite runs one → phase-1-only score
    return None


def _load_tasks() -> None:
    index_file = _DATA_DIR / "test_v2.json"
    if not index_file.exists():
        logger.warning("osworld_2 data/test_v2.json missing at %s", index_file)
        return
    task_ids = json.loads(index_file.read_text())["tasks"]

    id_caps: dict[str, list[str]] = {}
    hitl_ids: set[str] = set()
    for cap_file in sorted((_DATA_DIR / "capabilities").glob("*.json")):
        (cap_name, ids), = json.loads(cap_file.read_text()).items()
        for tid in ids:
            id_caps.setdefault(tid, []).append(cap_name)
        if cap_name == "human_in_the_loop":
            hitl_ids = set(ids)

    service_deps: dict[str, dict] = {}
    # Prefer the freshly-scanned copy (install.sh service_scan → .cache); fall back to the vendored
    # snapshot (data/) so the service-dep flags + service exclude_reasons also resolve on
    # enumeration-only hosts without the gated download.
    for sdp in (Path(_TASK_CLASS_DIR) / "_service_deps.json", _DATA_DIR / "_service_deps.json"):
        if sdp.exists():
            service_deps = json.loads(sdp.read_text())
            break

    for tid in task_ids:
        dep = service_deps.get(tid, {})
        # No "task_id" here: identity is framework-injected (registry.register
        # for the spec; the LiteBaseEnv.metadata property for live envs).
        others: dict[str, Any] = {"capabilities": id_caps.get(tid, [])}
        # Expose the static service dependencies (browsergym-style) so they're filterable +
        # visible, e.g. `--filter "lambda m: m.others.get('website')"` to run just website tasks.
        for _d in ("website", "gitlab", "multi_phase", "user_sim", "llm_judge"):
            if dep.get(_d) or (_d == "user_sim" and tid in hitl_ids):
                others[_d] = True
        if "volume_size" in dep:
            others["volume_size"] = dep["volume_size"]
        reason = _exclude_reason(tid, hitl_ids, dep,
                                 user_sim_model=_USER_SIM_MODEL, website_suffix=_WEBSITE_SUFFIX,
                                 gitlab_ok=bool(_GITLAB_URL and _GITLAB_TOKEN), has_openai_key=_HAS_OPENAI_KEY)
        if reason:
            others["exclude_reason"] = reason
        config = OSWorldV2Config(task_id=tid, metadata=dict(others))
        register(
            key=f"osworld_2@{tid}",
            entry_point=lambda *, cfg=config, **kw: OSWorldV2Env(config=cfg, **kw),
            split="eval",
            # Same-source contract: registered copy == the env's
            # builder output; the two sides cannot drift.
            metadata=OSWorldV2Env._task_metadata(config),
            **({"reset_timeout": 1700.0} if dep.get("volume_size") else {}),
            **({"step_timeout": 3900.0} if tid == "064" else {}),
        )


_load_tasks()
registry.set_env_make_kwargs("osworld_2", CFG.make_kwargs)

# ---------------------------------------------------------------------------
# Services + backend family
# ---------------------------------------------------------------------------
from lite.gym.services import BackendFamily, register_family, register_services  # noqa: E402
from lite.gym.remote.reaper import ContainerServices  # noqa: E402


class OSWorldV2Services(ContainerServices):
    """DEDICATED container services — image/KVM/tun/qcow2/task-class presence check on
    `ensure` (NO host desktop_env — it's in the image); `live_ids`/`reap` inherited
    (name-regex `docker ps`, scoped by server_port; `-osworld_2-` disjoint from `-osworld-`
    and `.osworld-`)."""

    def ensure(self, env_id: str) -> None:
        _ensure_services(env_id)


register_services("osworld_2", OSWorldV2Services())
register_family("osworld_2", BackendFamily.DEDICATED)
