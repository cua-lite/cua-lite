"""Online-Mind2Web CUA-Lite wrapper.

SINGLETON browser env: one shared ``cua-lite/online_mind2web`` container per env-server owns a
Playwright Chromium browser. Each trajectory gets a fresh browser context/page
through the in-container FastAPI RPC server.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import dataclasses
import importlib.util
import json
import logging
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import LITE_DESKTOP_KEY_ACTION_NAMES
from lite.core.tools.calls import (
    EnvAction,
    RuntimeEnvAction,
)
from lite.core.tools.extra_tools import (
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
)
from lite.core.tools.schemas import BaseTools, tool_schema_name
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import CapacityExhausted, EnvDepsMissingError
from lite.gym.registry import invalidate_services, register, registry
from lite.gym.remote.reaper import SingletonContainerServices
from lite.gym.services import register_services
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.coordinate import norm_to_pixel
from lite.gym.utils.backend.docker import docker_rm_f, docker_run
from lite.gym.utils.backend.freshness import image_for
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
    project_model_keys,
)
from lite.gym.utils.config import naming
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    BackendExecutionErrorDetail,
    ToolErrorFeedback,
    current_feedback,
    parse_container_action_error_record,
    record_model_action_error,
    record_tool_execution_error,
    unsupported_action_message,
)
from lite.gym.utils.feedback.ingress import (
    invalid_action_message,
    prepare_env_tool_calls,
    standalone_tool_call_feedback,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.utils.feedback.surface import (
    copy_valid_actions,
    resolve_extra_tools,
    resolve_valid_actions,
)
from lite.utils.image import png_from_b64_async

logger = logging.getLogger(__name__)

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)




# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
# Strict-index every key — default.yaml is the single source of truth (no literal
# fallbacks that duplicate/contradict it), matching webharbor.webvoyager + CLAUDE.md (trust
# the config contract; don't defensively .get internal keys). Only the documented
# runtime sentinels keep an `or` (max_steps: null, rpc_url/hf_revision: "").
_DEFAULT_MAX_STEPS = 30
_MAX_STEPS = CFG.env_kwargs["max_steps"]
if _MAX_STEPS is not None:
    try:
        _MAX_STEPS = int(_MAX_STEPS)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"online_mind2web env_kwargs.max_steps must be an int or null, got {_MAX_STEPS!r}"
        ) from exc
_STEP_TIMEOUT = CFG.env_kwargs["step_timeout"]
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
_VIEWPORT = tuple(CFG.env_kwargs["viewport"])
_VALID_ACTIONS = resolve_valid_actions(
    CFG.env_kwargs["valid_actions"], env_name="online_mind2web", platform="browser",
)
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]
_TRAJECTORY_DIR = CFG.env_kwargs["trajectory_dir"]
_OVERWRITE_TRAJECTORY = bool(CFG.env_kwargs["overwrite_trajectory"])
_SKIP_EVAL = bool(CFG.env_kwargs["skip_eval"])
_EVAL_CONFIG: dict[str, Any] = CFG.env_kwargs.get("eval_config") or {}

_HF_REPO_ID = CFG.server_kwargs["hf_repo_id"]
_HF_REVISION = CFG.server_kwargs["hf_revision"] or None
_RPC_PORT = CFG.server_kwargs["rpc_port"]
_RPC_URL = CFG.server_kwargs["rpc_url"] or f"http://localhost:{_RPC_PORT}"
_INSTANCES = int(CFG.server_kwargs["instances"] or 0)
_PAGE_LOAD_TIMEOUT_S = float(CFG.server_kwargs["page_load_timeout_s"])
_RESET_SETTLE_S = float(CFG.server_kwargs["reset_settle_s"])
_INSTANCE_TTL_S = float(CFG.server_kwargs["instance_ttl_s"])
_REAPER_INTERVAL_S = float(CFG.server_kwargs["reaper_interval_s"])
_RM_TIMEOUT_S = float(
    os.environ.get(
        "CUA_LITE_ONLINE_MIND2WEB_RM_TIMEOUT_S",
        str(CFG.server_kwargs["rm_timeout_s"]),
    )
)

_SOURCE = "online_mind2web"
_ONLINE_MIND2WEB_IMAGE = "cua-lite/online_mind2web:latest"
_ONLINE_MIND2WEB_PORT_START = 7900
_ONLINE_MIND2WEB_PORT_END = 7999
#: The nav subset this env is actually wired for. OmniBoxes-style single-window
#: browsing: no tab tools.
_ONLINE_MIND2WEB_NAV_TOOLS: tuple[str, ...] = ("goto", "back", "forward")


class OnlineMind2WebTools(BaseTools):
    """What online_mind2web declares beyond the GUI surface: three nav tools.

    Bodies come from the shared ``LiteBrowserNavToolSet`` source, so a nav-tool wording
    change lands here without an edit. ``response``/``terminate`` are resolved
    centrally and must NOT be listed.
    """

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        tool_schema_name(schema): schema
        for schema in LiteBrowserNavToolSet.get_tool_schemas(
            include=list(_ONLINE_MIND2WEB_NAV_TOOLS)
        )
    }


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = (
    OnlineMind2WebTools.get_tool_names() | LiteFinishToolSet.get_tool_names()
)

#: Builder-side no-override default, resolved ONCE at import.
_DEFAULT_EXTRA_TOOL_SCHEMAS = resolve_extra_tools(
    _EXTRA_TOOLS, tools=OnlineMind2WebTools, env_name="online_mind2web",
)
_services_started: set[str] = set()

_LEVEL_DIFFICULTY = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
}


def _derive_instances() -> int:
    if _INSTANCES:
        return _INSTANCES
    try:
        from lite.gym.utils.server.capacity import cached_host_capacity

        host = cached_host_capacity()
        return max(2, min(int((host.ram_total_gb * 0.20) / 0.6), 32))
    except Exception as e:
        logger.warning(
            "online_mind2web: capacity probe failed (%s); instances fallback to 8",
            type(e).__name__,
        )
        return 8


_RESOLVED_INSTANCES = _derive_instances()


def _container_name(server_port: int | None = None) -> str:
    """``online_mind2web-<server_port>`` (or ``online_mind2web-d<pid>`` in direct mode).

    Thin wrapper over the canonical shared-singleton helper so reaper scope
    stays identical to webgym/mobilegym."""
    return naming.container_name("online_mind2web", server_port)


def _healthz(url: str) -> bool:
    if not url:
        return False
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/healthz", timeout=3) as r:
            data = json.loads(r.read())
        return bool(data.get("ok"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_url(value: Any) -> str:
    url = "" if value is None else str(value).strip()
    if url and "://" not in url:
        return f"https://{url}"
    return url


def _rows_from_json(raw: Any) -> list[dict[str, Any]]:
    """Accept the current list manifest plus simple future container shapes."""
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = None
        for key in ("tasks", "data", "examples"):
            value = raw.get(key)
            if isinstance(value, list):
                rows = value
                break
        if rows is None and all(isinstance(v, dict) for v in raw.values()):
            rows = []
            for key, value in raw.items():
                row = dict(value)
                row.setdefault("task_id", key)
                rows.append(row)
        if rows is None:
            raise ValueError(
                "Mind2Web manifest must be a list, a dict with tasks/data/examples, "
                "or a task_id -> row mapping"
            )
    else:
        raise ValueError(f"Mind2Web manifest must be JSON object/list, got {type(raw).__name__}")

    normalized: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Mind2Web row {i} must be an object, got {type(row).__name__}")
        normalized.append(dict(row))
    return normalized


def _convert_task(row: dict[str, Any], index: int) -> dict[str, Any]:
    task_id = str(row.get("task_id") or row.get("id") or index).strip()
    if not task_id:
        raise ValueError(f"Mind2Web row {index} has an empty task_id")

    instruction = str(
        row.get("confirmed_task")
        or row.get("instruction")
        or row.get("task")
        or ""
    ).strip()
    if not instruction:
        raise ValueError(f"Mind2Web task {task_id!r} has an empty confirmed_task")

    website = _normalize_url(row.get("website") or row.get("start_url") or row.get("url"))
    reference_length = _optional_int(row.get("reference_length"))
    level = str(row.get("level") or "").strip().lower()
    difficulty = _LEVEL_DIFFICULTY.get(level)

    task = dict(row)
    task.update(
        {
            "task_id": task_id,
            "instruction": instruction,
            "task": instruction,
            "confirmed_task": instruction,
            "website": website,
            "start_url": website,
            "reference_length": reference_length,
            "level": level,
            "difficulty": difficulty,
            "source": _SOURCE,
            "hf_repo_id": _HF_REPO_ID,
            "hf_revision": _HF_REVISION,
        }
    )
    return task


def _load_tasks() -> list[dict[str, Any]]:
    """Load tasks from the checked-in static manifest (offline, no network).

    Mirrors the webgym/webharbor.webvoyager pattern: the 300-task Online-Mind2Web list is
    committed to ``data/tasks.json`` (gated upstream — regenerate with
    ``scripts/utils/tasks.sh``), so registration is import-time and offline-clean. No
    ``huggingface_hub`` at import.
    """
    tasks_path = Path(__file__).parent / "data" / "tasks.json"
    if not tasks_path.is_file():
        raise FileNotFoundError(
            f"{tasks_path} not found. Re-generate with:\n"
            f"  uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/utils/tasks.sh"
        )
    with tasks_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [_convert_task(row, i) for i, row in enumerate(_rows_from_json(raw))]


def _task_metadata_others(task: dict[str, Any]) -> dict[str, Any]:
    others: dict[str, Any] = {}
    for key, value in {
        "source": task["source"],
        "hf_repo_id": task["hf_repo_id"],
        "hf_revision": task.get("hf_revision"),
        "website": task.get("website", ""),
        "start_url": task.get("start_url", ""),
        "confirmed_task": task.get("confirmed_task", ""),
    }.items():
        if value not in (None, "", []):
            others[key] = value
    for key in ("reference_length", "level", "difficulty", "max_steps"):
        value = task.get(key)
        if value is not None and value != "":
            others[key] = value
    return others


def _task_metadata(task: dict[str, Any]) -> LiteCUAMetadata:
    """Same-source metadata builder.
    ``schema_version`` is a constant task fact; viewport's DEFAULT is a yaml
    fact; ``max_steps`` carries the task default pre-stamped at registration
    (the live env amends on env_kwargs override)."""
    others = _task_metadata_others(task)
    others["schema_version"] = "online-mind2web-v2"
    # viewport's DEFAULT is a yaml fact → registered copy carries it too
    # (invariant: registered == no-override live).
    others["viewport"] = tuple(_VIEWPORT)
    return LiteCUAMetadata(
        dims=("browser", "use"),
        valid_actions=copy_valid_actions(_VALID_ACTIONS),
        # yaml default via the ctor's own resolver — registered ==
        # no-override live by construction.
        extra_tool_schemas=list(_DEFAULT_EXTRA_TOOL_SCHEMAS),
        others=others,
    )


# ---------------------------------------------------------------------------
# Online-Mind2Web v2 trajectory + WebJudge helpers
# ---------------------------------------------------------------------------
def _jsonable_text(value: Any, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _quote_text(value: Any, limit: int = 180) -> str:
    text = _jsonable_text(value, limit=limit)
    return repr(text)


def _action_description(action: EnvAction) -> str | None:
    for key in ("action_description", "description"):
        value = (action or {}).get(key)
        if value:
            return _jsonable_text(value, limit=1000)
    return None


def _action_thought(action: EnvAction) -> str | None:
    for key in ("thought", "reasoning", "reason"):
        value = (action or {}).get(key)
        if value:
            return _jsonable_text(value, limit=1000)
    return _action_description(action)


def _pair_online_mind2web_action_errors(
    actions: list[EnvAction],
    result_call_ids: list[str | None],
    *,
    action_errors: list[Any] | None = None,
) -> tuple[dict[str, ToolErrorFeedback], set[int]]:
    """Attribute typed container ``action_errors[]`` to sent actions.

    The host/container contract is indexed and typed. Aggregate ``errors[]``
    remains debug-only response metadata and must not drive model-visible
    feedback by text-prefix matching.
    """
    paired: dict[str, ToolErrorFeedback] = {}
    failed: set[int] = set()

    def add(record) -> None:
        index = record.index
        if index < 0 or index >= len(actions):
            return
        failed.add(index)
        call_id = result_call_ids[index]
        if not call_id:
            return
        name = record.name
        detail = record.error or record.message
        if record.kind == "unsupported_action":
            paired[call_id] = current_feedback(
                record.message or unsupported_action_message(name)
            )
            return
        if record.kind == "tool_execution":
            record_tool_execution_error(
                paired,
                call_id,
                BackendExecutionErrorDetail(detail),
                action_name=name,
            )
            return
        record_model_action_error(
            paired,
            call_id,
            ValueError(detail),
            action_name=name,
        )

    for raw_record in action_errors or []:
        record = parse_container_action_error_record(raw_record)
        if record is None:
            continue
        add(record)
    return paired, failed


def _coord_px(coord: Any, viewport: tuple[int, int]) -> tuple[int, int] | None:
    # on_malformed="none" → the caller targets "page" (this env's contract).
    return norm_to_pixel(coord, int(viewport[0]), int(viewport[1]),
                         clamp=True, on_malformed="none")


def _coord_target(coord: Any, viewport: tuple[int, int]) -> str:
    xy = _coord_px(coord, viewport)
    if xy is None:
        return "page"
    return f"coords({xy[0]}, {xy[1]})"


def _status_suffix(status: str | None) -> str:
    return f" | {status}" if status in {"SUCCESS", "FAILED"} else ""


def _keys_text(args: dict[str, Any]) -> str:
    keys = args.get("keys")
    if keys is None and args.get("key") is not None:
        keys = [args["key"]]
    if isinstance(keys, str):
        keys = [keys]
    return "+".join(str(k) for k in (keys or [])) or "key"


def _terminal_action(
    name: str,
    args: dict[str, Any],
) -> str:
    """Render a terminal action into its v2 trajectory string.

    Renders the *action the agent issued*, not the final answer of record, which
    is the container's. ``terminate`` carries no answer at all: it is a completion
    status, and reading its status as an answer would turn a normal
    ``terminate(status=success)`` into the bogus final answer "success".
    """
    if name == "response":
        return f"TASK_COMPLETE -> ANSWER: {str(args.get('text', ''))}"

    status = str(args.get("status") or "success")
    reason = _jsonable_text(args.get("reason"), limit=400)
    action = f"TASK_COMPLETE -> STATUS: {status}"
    if reason:
        action = f"{action} | REASON: {reason}"
    return action


def _to_container_action(action: EnvAction) -> EnvAction:
    """Project one canonical action onto the container's action wire.

    Only ``type`` differs: canonical spells "submit" as a trailing newline, but
    the container prefers ``locator.fill()``, which stores a newline as text
    instead of pressing Return. So the newline becomes the ``press_enter`` flag
    the container already reads. ``actions_to_send`` itself stays canonical --
    it is also what the judged v2 trajectory is rendered from.
    """
    if action["name"] != "type":
        return action
    text = str(action["arguments"].get("text", ""))
    press_enter = text.endswith("\n")
    return {
        **action,
        "arguments": {
            **action["arguments"],
            "text": text[:-1] if press_enter else text,
            "press_enter": press_enter,
        },
    }


def _format_schema_action(
    name: str,
    args: dict[str, Any],
    *,
    status: str | None,
    viewport: tuple[int, int],
    description: str | None = None,
) -> tuple[str, str | None]:
    """Return (action string, action_status)."""
    if name in LiteFinishToolSet.get_tool_names():
        return _terminal_action(name, args), None

    suffix = _status_suffix(status)
    action_status = status if status in {"SUCCESS", "FAILED"} else None

    if name == "click":
        target = _coord_target(args.get("coordinate"), viewport)
        desc = description or "click at the target coordinate"
        return f"CLICK {target} -> {desc}{suffix}", action_status

    if name == "type":
        raw = str(args.get("text", ""))
        press_enter = raw.endswith("\n")
        text = _quote_text(raw[:-1] if press_enter else raw)
        desc = description or f"type {text} into the focused input"
        if press_enter:
            desc += " and press Enter"
        return f"TYPE page -> {desc}{suffix}", action_status

    if name == "scroll":
        direction = str(args.get("direction", "down")).lower()
        amount = int(args.get("amount", 3) or 3)
        target = _coord_target(args.get("coordinate"), viewport) if args.get("coordinate") else "page"
        desc = description or f"scroll {direction} by {amount} unit(s)"
        return f"SCROLL {target} -> {desc}{suffix}", action_status

    if name == "key":
        desc = description or f"press {_keys_text(args)}"
        return f"PRESS_KEY page -> {desc}{suffix}", action_status

    if name == "key_down":
        return f"PRESS_KEY page -> hold {_keys_text(args)}{suffix}", action_status

    if name == "key_up":
        return f"PRESS_KEY page -> release {_keys_text(args)}{suffix}", action_status

    if name == "wait":
        duration = coerce_model_duration(args.get("duration", 1.0), action_name="wait")
        desc = description or f"wait {duration:g} second(s)"
        return f"WAIT page -> {desc}{suffix}", action_status

    if name in {"noop", "screenshot"}:
        desc = description or "observe current page"
        return f"WAIT page -> {desc}{suffix}", action_status

    if name in {"back", "goback"}:
        desc = description or "return to the previous browser history entry"
        return f"page -> GO_BACK -> {desc}{suffix}", action_status

    if name in {"forward", "goforward"}:
        desc = description or "advance to the next browser history entry"
        return f"page -> GO_FORWARD -> {desc}{suffix}", action_status

    if name == "goto":
        url = _jsonable_text(args.get("url"), limit=300)
        desc = description or f"Direct navigation to {url}"
        return f"page -> NAVIGATE -> {desc}{suffix}", action_status

    if name in {"mouse_move", "hover"}:
        target = _coord_target(args.get("coordinate"), viewport)
        desc = description or "move pointer to the target coordinate"
        return f"HOVER {target} -> {desc}{suffix}", action_status

    if name == "select":
        selector = _jsonable_text(args.get("selector") or "page", limit=180)
        choice = args.get("value")
        if choice is None:
            choice = args.get("label")
        if choice is None:
            choice = args.get("index")
        desc = description or f"select {_quote_text(choice)}"
        return f"SELECT {selector} -> {desc}{suffix}", action_status

    if name == "drag":
        start = _coord_target(args.get("start_coordinate") or args.get("from") or args.get("source"), viewport)
        end = _coord_target(args.get("coordinate") or args.get("end_coordinate") or args.get("to"), viewport)
        desc = description or f"drag from {start} to {end}"
        return f"CLICK {end} -> {desc}{suffix}", action_status

    if name in {"click_elem", "fill"}:
        # Mind2Web's Playwright server does not expose element indices today,
        # but keep the formatter stable for callers that pass these through.
        verb = "CLICK" if name == "click_elem" else "TYPE"
        desc = description or f"{name} with arguments {_jsonable_text(args, limit=240)}"
        return f"{verb} page -> {desc}{suffix}", action_status

    desc = f"attempt unhandled action {name} with arguments {_jsonable_text(args, limit=240)}"
    return f"WAIT page -> {desc}{suffix}", action_status


def _extract_prediction(response: str, mode: str) -> int:
    if mode == "WebVoyager_eval":
        return 0 if "FAILURE" in response else 1
    if mode in {
        "Autonomous_eval",
        "AgentTrek_eval",
        "WebJudge_Online_Mind2Web_eval",
        "WebJudge_general_eval",
    }:
        try:
            return 1 if "success" in response.lower().split("status:", 1)[1] else 0
        except Exception:
            return 0
    raise ValueError(f"Unknown mode: {mode}")


def _require_eval_deps(eval_config: dict[str, Any]) -> None:
    if importlib.util.find_spec("openai") is None:
        raise EnvDepsMissingError(
            what="openai package not importable (needed for Mind2Web WebJudge)",
            install="pip install openai",
            see="lite/gym/envs/online_mind2web/README.md",
        )
    if importlib.util.find_spec("PIL") is None:
        raise EnvDepsMissingError(
            what="Pillow package not importable (needed to encode WebJudge images)",
            install="pip install pillow",
            see="lite/gym/envs/online_mind2web/README.md",
        )
    api_key_var = eval_config.get("api_key_env_var", "OPENAI_API_KEY")
    if not os.environ.get(api_key_var):
        raise EnvDepsMissingError(
            what=f"Mind2Web WebJudge needs {api_key_var}",
            install=f"export {api_key_var}=...   # or pass skip_eval=True",
            see="lite/gym/envs/online_mind2web/README.md",
        )


def _is_judge_rate_limit_error(e: BaseException) -> bool:
    """A transient judge rate-limit (retryable), vs. a hard infra error (raise).

    Mirrors webharbor.webvoyager's check so the two browser envs agree on the boundary: a 429
    is retried; anything else propagates so the caller ERRs instead of fabricating
    a verdict.
    """
    return "429" in str(e) or "rate" in str(e).lower() or "RateLimitError" in type(e).__name__


class _OpenAIJudgeEngine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        import openai

        api_key_var = cfg.get("api_key_env_var", "OPENAI_API_KEY")
        base_url = (
            cfg.get("base_url")
            or os.environ.get("ONLINE_MIND2WEB_EVAL_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        self.model = os.environ.get("ONLINE_MIND2WEB_EVAL_MODEL") or cfg.get("model", "o4-mini")
        self.max_tokens = int(cfg.get("max_tokens", 2048))
        timeout = cfg.get("request_timeout")
        self.client = openai.OpenAI(
            api_key=os.environ.get(api_key_var),
            base_url=base_url or None,
            timeout=float(timeout) if timeout is not None else None,
        )

    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[str]:
        request = {
            "model": kwargs.get("model") or self.model,
            "messages": messages,
        }
        if "temperature" in kwargs:
            request["temperature"] = kwargs["temperature"]
        max_tokens = int(kwargs.get("max_new_tokens") or self.max_tokens)
        # Retry transient rate-limits with backoff; a non-rate-limit or
        # retries-exhausted error PROPAGATES (never coalesced into a verdict) so
        # the caller ERRs + retries instead of committing a corrupt reward.
        for attempt in range(5):
            try:
                response = self._create(request, max_tokens)
                break
            except Exception as e:
                if attempt < 4 and _is_judge_rate_limit_error(e):
                    wait = 10 * (2**attempt)
                    logger.warning(
                        "online_mind2web judge rate-limited (attempt %d/5), retrying in %ds: %s",
                        attempt + 1,
                        wait,
                        e,
                    )
                    time.sleep(wait)
                else:
                    raise
        return [choice.message.content or "" for choice in response.choices]

    def _create(self, request: dict[str, Any], max_tokens: int) -> Any:
        """One chat-completion call with the max_tokens/max_completion_tokens fallback."""
        try:
            return self.client.chat.completions.create(**request, max_tokens=max_tokens)
        except Exception as e:
            # Some reasoning models reject max_tokens and require
            # max_completion_tokens. Keep this fallback local to eval.
            if "max_tokens" not in str(e) and "max_completion_tokens" not in str(e):
                raise
            return self.client.chat.completions.create(**request, max_completion_tokens=max_tokens)


def _encode_image_file(path: str | Path) -> str:
    import io

    from PIL import Image

    image = Image.open(path)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _identify_key_points(task: str, model: _OpenAIJudgeEngine) -> str:
    system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": [{"type": "text", "text": f"Task: {task}"}]},
    ]
    return model.generate(messages)[0]


def _normalize_key_points(text: str) -> str:
    key_points = text.replace("\n\n", "\n")
    try:
        key_points = key_points.split("**Key Points**:", 1)[1]
    except Exception:
        key_points = key_points.split("Key Points:", 1)[-1]
    return "\n".join(line.lstrip() for line in key_points.splitlines()).strip()


def _judge_image(task: str, image_path: str, key_points: str, model: _OpenAIJudgeEngine) -> str:
    system_msg = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.
- Does the image show actions, progress indicators, or critical information directly related to completing the task?
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.
- **Score**: Assign a score based on the reasoning, using the following scale:
    - **1**: The image does not contain any necessary steps or relevant information.
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:
1. **Reasoning**: [Your explanation]
2. **Score**: [1-5]"""
    text = (
        f"**Task**: {task}\n\n"
        f"**Key Points for Task Completion**: {key_points}\n\n"
        "The snapshot of the web page is shown in the image."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_encode_image_file(image_path)}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    return model.generate(messages)[0]


def _webjudge_online_mind2web_eval(
    *,
    task: str,
    last_actions: list[str],
    image_paths: list[str],
    cfg: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    mode = str(cfg.get("mode") or "WebJudge_Online_Mind2Web_eval")
    if mode != "WebJudge_Online_Mind2Web_eval":
        raise ValueError(f"online_mind2web only wires WebJudge_Online_Mind2Web_eval, got {mode!r}")

    model = _OpenAIJudgeEngine(cfg)
    score_threshold = int(cfg.get("score_threshold", 1))  # 1 = lenient (N5); see _build_eval_config
    max_images = int(cfg.get("max_images", 50))
    image_judge_workers = max(
        1,
        int(cfg.get("image_judge_workers", min(16, max(1, len(image_paths)))) or 1),
    )

    key_points = _normalize_key_points(_identify_key_points(task, model))
    record: list[dict[str, Any]] = []
    important_images: list[dict[str, Any]] = []
    important_thoughts: list[str] = []
    score_pattern = r"[1-5]"

    def _judge_one(image_path: str) -> str:
        return _judge_image(task, image_path, key_points, model)

    if image_paths:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(image_judge_workers, len(image_paths))
        ) as executor:
            # A judge CALL error (429/timeout/network, after generate()'s retries)
            # propagates out of list() -> _evaluate re-raises -> rollout ERR +
            # retry. We never silently score an un-judged frame as 0 — that would
            # drop it from the verdict input and corrupt the reward.
            image_responses = list(executor.map(_judge_one, image_paths))
    else:
        image_responses = []

    for image_path, response in zip(image_paths, image_responses):
        thought = ""
        try:
            score_text = response.split("Score", 1)[1]
            thought = response.split("**Reasoning**:")[-1].strip().lstrip("\n").split("\n\n")[0]
            thought = thought.replace("\n", " ")
            score = int(re.findall(score_pattern, score_text)[0])
        except Exception:
            # Successful call, unparseable text -> legitimately "not important".
            score = 0
        entry: dict[str, Any] = {"Response": response, "Score": score}
        record.append(entry)
        if score >= score_threshold:
            important_images.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_encode_image_file(image_path)}",
                        "detail": "high",
                    },
                }
            )
            if thought:
                important_thoughts.append(thought)

    important_images = important_images[:max_images]
    important_thoughts = important_thoughts[:max_images]

    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
Examples of Failure Cases:
- If the requirement is less than $50, but the applied filter is less than $25, it is a failure.
- If the requirement is $1500-$2500, but the applied filter is $2000-$2500, it is a failure.
- If the requirement is $25-$200, but the applied filter is $0-$200, it is a failure.
- If the required years are 2004-2012, but the filter applied is 2001-2012, it is a failure.
- If the required years are before 2015, but the applied filter is 2000-2014, it is a failure.
- If the task requires exactly 2 beds, but the filter applied is 2+ beds, it is a failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements (e.g., the cheapest or lowest price), the task is still considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""
    if important_images:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{thoughts}"""
        text = prompt.format(
            task=task,
            key_points=key_points,
            last_actions="\n".join(f"{i + 1}. {action}" for i, action in enumerate(last_actions)),
            thoughts="\n".join(f"{i + 1}. {thought}" for i, thought in enumerate(important_thoughts)),
        )
    else:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}"""
        text = prompt.format(
            task=task,
            key_points=key_points,
            last_actions="\n".join(f"{i + 1}. {action}" for i, action in enumerate(last_actions)),
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": [{"type": "text", "text": text}] + important_images},
    ]
    response = model.generate(messages)[0]
    predicted_label = _extract_prediction(response, mode)
    reward = float(predicted_label)
    return reward, {
        "eval_model": model.model,
        "eval_mode": mode,
        "eval_verdict": response,
        "predicted_label": predicted_label,
        "reward": reward,
        "input_text": text,
        "system_msg": system_msg,
        "image_judge_record": record,
        "key_points": key_points,
    }


# ---------------------------------------------------------------------------
# Thin JSON RPC client env
# ---------------------------------------------------------------------------
class RemoteOnlineMind2WebEnv(LiteBaseEnv):
    EXTRA_TOOLS: ClassVar[type[BaseTools]] = OnlineMind2WebTools

    def __init__(
        self,
        *,
        task_id: str,
        task: dict[str, Any],
        base_max_steps: int,
        max_steps: int | None = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        viewport: tuple[int, int] = _VIEWPORT,
        valid_actions: list[str] | None = _VALID_ACTIONS,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        trajectory_dir: str | None = _TRAJECTORY_DIR,
        overwrite_trajectory: bool = _OVERWRITE_TRAJECTORY,
        eval_config: dict[str, Any] | None = None,
        skip_eval: bool = _SKIP_EVAL,
        cursor: bool = True,
        **_: Any,
    ) -> None:
        self._task_id = task_id
        # dict(...): `task` is the shared registration-side row — copy,
        # don't alias.
        self._task = dict(task)
        self._base_max_steps = base_max_steps
        self._max_steps = max_steps
        self._post_action_delay = post_action_delay
        self._cursor = cursor
        self._viewport = tuple(viewport)
        self._rpc_url = os.environ.get("ONLINE_MIND2WEB_RPC_URL") or _RPC_URL
        self._instance_id: str | None = None
        # Unconditional assignment: the signature default (yaml-sourced) is
        # the single source of truth for "omitted", so an EXPLICIT
        # ``valid_actions: null`` keeps its shared meaning (no filtering).
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="online_mind2web", platform="browser",
        )
        self._trajectory_root = Path(trajectory_dir).expanduser() if trajectory_dir else None
        self._overwrite_trajectory = bool(overwrite_trajectory)
        self._eval_config: dict[str, Any] = {**_EVAL_CONFIG, **(eval_config or {})}
        self._skip_eval = bool(skip_eval)
        self._tmp_trajectory: tempfile.TemporaryDirectory[str] | None = None
        self._run_dir: Path | None = None
        self._trajectory_steps: list[dict[str, Any]] = []
        self._image_paths: list[str] = []
        self._agent_final_answer: str | None = None
        self._last_obs: dict[str, Any] = {}
        self._last_observation_image: bytes | None = None
        self._last_observation_text: str | None = None
        self._step_count = 0

        # Resolve + validate via the same helper the builder default uses
        # (registered == no-override live by construction).
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)

    @property
    def max_steps(self) -> int:
        return int(self._max_steps if self._max_steps is not None else self._base_max_steps)

    # Shared with registration (module-level builder).
    _task_metadata = staticmethod(_task_metadata)

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendments: valid_actions / extra_tools overrides,
        # max_steps override, viewport, and the optional trajectory_dir
        # (env_kwarg-derived, absent by default).
        md = self._task_metadata(self._task)
        others = {**md.others, "viewport": self._viewport, "max_steps": self.max_steps}
        if self._trajectory_root is not None:
            others["trajectory_dir"] = str(self._trajectory_root)
        return dataclasses.replace(
            md,
            valid_actions=self._valid_actions,
            extra_tool_schemas=self._extra_tool_schemas,
            others=others,
        )

    # -----------------------------------------------------------------------
    # Trajectory + evaluation helpers
    # -----------------------------------------------------------------------

    def _build_eval_config(self) -> dict[str, Any]:
        cfg = dict(self._eval_config)
        cfg.setdefault("mode", "WebJudge_Online_Mind2Web_eval")
        cfg.setdefault("model", "o4-mini")
        # 1 = lenient: an image scoring >=1 of 1-5 counts as relevant context for
        # the verdict (the cua-lite default.yaml sets this; upstream WebJudge used
        # 3). Keep the code fallback aligned so a partial eval_config override
        # doesn't silently jump back to the stricter 3 (N5).
        cfg.setdefault("score_threshold", 1)
        cfg.setdefault("api_key_env_var", "OPENAI_API_KEY")
        cfg.setdefault("max_images", 50)
        if os.environ.get("ONLINE_MIND2WEB_EVAL_MODEL"):
            cfg["model"] = os.environ["ONLINE_MIND2WEB_EVAL_MODEL"]
        if os.environ.get("ONLINE_MIND2WEB_EVAL_API_KEY_VAR"):
            cfg["api_key_env_var"] = os.environ["ONLINE_MIND2WEB_EVAL_API_KEY_VAR"]
        if os.environ.get("ONLINE_MIND2WEB_EVAL_BASE_URL"):
            cfg["base_url"] = os.environ["ONLINE_MIND2WEB_EVAL_BASE_URL"]
        return cfg

    def _reset_trajectory_state(self) -> None:
        self._trajectory_steps = []
        self._image_paths = []
        self._agent_final_answer = None
        self._last_obs = {}
        self._last_observation_image = None
        self._last_observation_text = None
        self._step_count = 0
        if self._tmp_trajectory is not None:
            self._tmp_trajectory.cleanup()
            self._tmp_trajectory = None
        if self._trajectory_root is None:
            self._tmp_trajectory = tempfile.TemporaryDirectory(
                prefix=f"online_mind2web-{self._task_id}-"
            )
            self._run_dir = Path(self._tmp_trajectory.name) / self._task_id
        else:
            self._run_dir = self._trajectory_root / self._task_id
            if self._run_dir.exists():
                if self._overwrite_trajectory:
                    shutil.rmtree(self._run_dir)
                else:
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    self._run_dir = self._trajectory_root / f"{self._task_id}__{stamp}"
        (self._run_dir / "trajectory").mkdir(parents=True, exist_ok=True)

    @property
    def trajectory_dir(self) -> str | None:
        return str(self._run_dir) if self._run_dir is not None else None

    @property
    def result_path(self) -> str | None:
        return str(self._run_dir / "result.json") if self._run_dir is not None else None

    def _reference_length(self) -> int:
        return (
            _optional_int(self._task.get("reference_length"))
            or _optional_int(self._task.get("max_steps"))
            or max(1, int(self._base_max_steps))
        )

    def _write_screenshot(self, screenshot_b64: str, step_idx: int) -> str:
        if self._run_dir is None:
            self._reset_trajectory_state()
        assert self._run_dir is not None
        name = f"{step_idx:04d}.png"
        path = self._run_dir / "trajectory" / name
        path.write_bytes(base64.b64decode(screenshot_b64))
        return name

    def _write_result_json(self) -> None:
        if self._run_dir is None:
            return
        result = {
            "schema_version": "online-mind2web-v2",
            "task": self._task.get("instruction") or self._task.get("confirmed_task") or "",
            "task_id": self._task_id,
            "agent_final_answer": self._agent_final_answer,
            "reference_length": self._reference_length(),
            "action_history": self._trajectory_steps,
        }
        with (self._run_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def _write_eval_json(self, eval_info: dict[str, Any]) -> None:
        if self._run_dir is None or not eval_info or eval_info.get("eval_skipped"):
            return
        payload = {
            "task_id": self._task_id,
            "mode": eval_info.get("eval_mode"),
            "model": eval_info.get("eval_model"),
            "reward": eval_info.get("reward"),
            "predicted_label": eval_info.get("predicted_label"),
            "eval_verdict": eval_info.get("eval_verdict"),
            "key_points": eval_info.get("key_points"),
            "image_judge_record": eval_info.get("image_judge_record"),
        }
        with (self._run_dir / "evaluation.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _append_v2_step(
        self,
        *,
        action: str,
        screenshot_b64: str,
        url: str | None,
        thought: str | None,
        action_status: str | None,
    ) -> None:
        step_idx = len(self._trajectory_steps)
        screenshot_name = self._write_screenshot(screenshot_b64, step_idx)
        step = {
            "step": step_idx,
            "screenshot": screenshot_name,
            "url": url or None,
            "action": action,
            "action_status": action_status,
            "thought": thought,
        }
        self._trajectory_steps.append(step)
        assert self._run_dir is not None
        self._image_paths.append(str(self._run_dir / "trajectory" / screenshot_name))
        self._write_result_json()

    def _append_initial_step(self, resp: dict[str, Any]) -> None:
        navigation_error = resp.get("navigation_error")
        status = "FAILED" if navigation_error else None
        suffix = _status_suffix(status)
        self._append_v2_step(
            action=f"page -> NAVIGATE -> Initial navigation to the provided URL{suffix}",
            screenshot_b64=resp["screenshot_b64"],
            url=resp.get("url") or self._task.get("start_url"),
            thought="Navigating to initial URL that was provided.",
            action_status=status,
        )

    def _append_action_steps(
        self,
        actions: list[EnvAction],
        resp: dict[str, Any],
        failed_indices: set[int],
    ) -> None:
        # The v2 trajectory is the Online-Mind2Web eval artifact, not the Lite
        # result-image contract: every action step in one batch shares the
        # post-batch frame, as it always has. Per-action frames now exist in
        # ``screenshots_b64`` and giving each judged step its own would be more
        # faithful, but that changes what the judge scores -- a separate change.
        screenshot_b64 = resp["screenshots_b64"][-1]
        url = resp.get("url")

        for index, action in enumerate(actions):
            name = action["name"]
            args = dict(action["arguments"])
            if not name:
                continue
            status = "FAILED" if index in failed_indices else "SUCCESS"
            if name in LiteFinishToolSet.get_tool_names():
                status = None
            schema_action, action_status = _format_schema_action(
                name,
                args,
                status=status,
                viewport=self._viewport,
                description=_action_description(action),
            )
            self._append_v2_step(
                action=schema_action,
                screenshot_b64=screenshot_b64,
                url=url,
                thought=_action_thought(action),
                action_status=action_status,
            )
            if name in LiteFinishToolSet.get_tool_names():
                break

    async def _evaluate(self) -> tuple[float | None, dict[str, Any]]:
        if self._skip_eval:
            return None, {"eval_skipped": True, "eval_skip_reason": "skip_eval"}
        if not self._trajectory_steps or not self._image_paths:
            return None, {"eval_skipped": True, "eval_skip_reason": "empty_trajectory"}
        cfg = self._build_eval_config()
        _require_eval_deps(cfg)
        # A WebJudge infra error (rate-limit after retries, timeout, network,
        # auth) PROPAGATES — the terminal step ERRs and the rollout retries it.
        # We must never coalesce a non-verdict into reward=0.0: that silently
        # poisons RL data. webgym/webharbor.webvoyager both re-raise here.
        return await asyncio.to_thread(
            _webjudge_online_mind2web_eval,
            task=str(self._task.get("instruction") or ""),
            last_actions=[step["action"] for step in self._trajectory_steps],
            image_paths=list(self._image_paths),
            cfg=cfg,
        )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self._rpc_url.rstrip("/") + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code == 503:
                raise CapacityExhausted(
                    what=f"online_mind2web container pool full ({path}): {detail}",
                    retry_after_s=60.0,
                ) from e
            raise RuntimeError(f"online_mind2web container {path} returned {e.code}: {detail}") from e

    @staticmethod
    def _obs_metadata(resp: dict[str, Any]) -> dict[str, Any]:
        url = resp.get("url", "")
        title = resp.get("title", "")
        body_text = resp.get("body_text", "")
        navigation_error = resp.get("navigation_error", "")
        page_lines = []
        if url:
            page_lines.append(f"Current page URL: {url}")
        if title:
            page_lines.append(f"Current page title: {title}")
        if navigation_error:
            page_lines.append(f"Navigation error: {navigation_error}")
        if body_text:
            if page_lines:
                page_lines.append("")
            page_lines.append(str(body_text))
        web_text = "\n".join(page_lines)
        return {
            "url": url,
            "title": title,
            "a11y_tree": resp.get("a11y_tree", []),
            "body_text": body_text,
            "navigation_error": navigation_error,
            # web_text is the page-context string history protocols may surface
            # through explicit image+text history policy; body_text is kept for
            # generic consumers.
            # (online_mind2web's shipped configs use the base qwen3_vl/qwen3_5 history,
            # not the webharbor.webvoyager SoM web_text splice.)
            "web_text": web_text,
        }

    @staticmethod
    def _obs_text(resp: dict[str, Any]) -> str | None:
        text = RemoteOnlineMind2WebEnv._obs_metadata(resp).get("web_text", "")
        return text or None

    async def reset(self) -> LiteEnvObservation:
        if not self._skip_eval:
            _require_eval_deps(self._build_eval_config())
        if self._instance_id is not None:
            await self.close()
        self._reset_trajectory_state()

        resp = await asyncio.to_thread(
            self._post,
            "/reset",
            {
                "task_id": self._task_id,
                "instruction": self._task.get("instruction", ""),
                "confirmed_task": self._task.get("confirmed_task", ""),
                "start_url": self._task.get("start_url", "https://www.google.com/"),
                "website": self._task.get("website", ""),
                "max_steps": self.max_steps,
                "viewport": list(self._viewport),
                "cursor": self._cursor,
            },
        )
        self._instance_id = resp["instance_id"]
        self._last_obs = self._obs_metadata(resp)
        png = await png_from_b64_async(resp["screenshot_b64"])
        self._last_observation_image = png
        self._last_observation_text = self._obs_text(resp)
        self._append_initial_step(resp)
        return LiteEnvObservation(image=png,
            text=resp.get("instruction") or self._task.get("instruction", ""),
            metadata={
                **self._last_obs,
                "trajectory_dir": self.trajectory_dir,
                "result_path": self.result_path,
            },
        )

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        if self._instance_id is None:
            raise RuntimeError("online_mind2web step() called before reset()")
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        actions_with_result_ids, action_errors = prepare_env_tool_calls(actions, metadata)
        action_errors: dict[str, ToolErrorFeedback] = dict(action_errors)
        actions_to_send: list[EnvAction] = []
        sent_result_call_ids: list[str | None] = []
        for action, result_call_id in actions_with_result_ids:
            tool_feedback = standalone_tool_call_feedback(
                action,
                _KNOWN_STANDALONE_TOOL_NAMES,
                metadata.extra_tool_schemas,
            )
            if tool_feedback is not None:
                if result_call_id:
                    action_errors[result_call_id] = tool_feedback
                continue
            name = action["name"]
            invalid_action = invalid_action_message(action, metadata.valid_actions)
            if invalid_action:
                if result_call_id:
                    action_errors[result_call_id] = current_feedback(invalid_action)
                continue
            if name == "wait":
                try:
                    duration = coerce_model_duration(
                        action["arguments"].get("duration", 1.0),
                        action_name="wait",
                    )
                except MODEL_ACTION_ERROR_TYPES as e:
                    record_model_action_error(
                        action_errors,
                        result_call_id,
                        e,
                        action_name=name,
                    )
                    continue
                action = {
                    **action,
                    "arguments": {**action["arguments"], "duration": duration},
                }
            if name in LITE_DESKTOP_KEY_ACTION_NAMES:
                try:
                    # ``keys`` is REQUIRED and must be non-empty: the container's
                    # `if keys:` presses nothing for an empty list, so accepting
                    # one returned a normal post-action screenshot for a keypress
                    # that never happened -- the model could not tell. A missing
                    # or empty list is a malformed argument, reported through the
                    # same ``record_model_action_error`` channel as a bad
                    # ``wait.duration``.
                    keys = project_model_keys(
                        action["arguments"].get("keys", []),
                        action_name=name,
                        backend="playwright",
                    )
                except MODEL_ACTION_ERROR_TYPES as e:
                    record_model_action_error(
                        action_errors,
                        result_call_id,
                        e,
                        action_name=name,
                    )
                    continue
                action = {
                    **action,
                    "arguments": {**action["arguments"], "keys": keys},
                }
            actions_to_send.append(action)
            sent_result_call_ids.append(result_call_id)

        self._step_count += 1
        if not actions_to_send:
            truncated = self._step_count >= self.max_steps
            reward: float | None = None
            eval_info: dict[str, Any] = {}
            if truncated:
                reward, eval_info = await self._evaluate()
                if eval_info:
                    self._write_result_json()
                    self._write_eval_json(eval_info)
            return build_tool_results_from_decisions(
                LiteEnvStepResult(
                    reward=reward,
                    terminated=False,
                    truncated=truncated,
                    info={
                        EXECUTED_ACTIONS_INFO_KEY: [],
                        "errors": [],
                        "answer": self._agent_final_answer,
                        "trajectory_dir": self.trajectory_dir,
                        "result_path": self.result_path,
                        **eval_info,
                    },
                ),
                ordered_call_ids=result_call_ids,
                continue_call_ids=result_call_ids,
                images=[self._last_observation_image],
                text=self._last_observation_text,
                metadata={
                    **self._last_obs,
                    "trajectory_dir": self.trajectory_dir,
                    "result_path": self.result_path,
                },
                feedback=action_errors,
            )

        resp = await asyncio.to_thread(
            self._post,
            "/step",
            {
                "instance_id": self._instance_id,
                "actions": [_to_container_action(a) for a in actions_to_send],
                "post_action_delay": self._post_action_delay,
                "cursor": self._cursor,
                # The host owns the budget: it is the side that sees the turns
                # whose calls were all rejected, which the fast path above scores
                # without reaching the container.
                "step_count": self._step_count,
            },
        )
        self._last_obs = self._obs_metadata(resp)
        # One result frame per EXECUTED action, in action order: the container
        # runs the whole batch, so it is the only side that can see the page
        # between two actions. ``screenshots_b64`` is the single owner of
        # this step's frames; ``/step`` returns no singular spelling.
        frames_b64 = resp["screenshots_b64"]
        pngs = [
            frame for frame in
            [await png_from_b64_async(entry) for entry in frames_b64]
            if frame is not None
        ]
        # R3: a GUI slot the host rejected owes a frame too, but it never
        # reaches the container so none comes back for it. Derive WHERE those
        # slots sat from the sent list -- no filter arm has to remember -- and
        # re-interleave. The host cannot capture mid-batch (the container runs
        # the whole batch in one RPC), so a rejected slot repeats the frame of
        # the action before it; that slot changed nothing, so the repeat is the
        # honest record.
        if pngs and len(actions_to_send) < len(actions_with_result_ids):
            sent_ids = [id(a) for a in actions_to_send]
            rebuilt: list[bytes] = []
            executed_frames = iter(pngs)
            for action, _rid in actions_with_result_ids:
                if id(action) in sent_ids:
                    rebuilt.append(next(executed_frames, pngs[-1]))
                else:
                    rebuilt.append(rebuilt[-1] if rebuilt else pngs[0])
            pngs = rebuilt
        png = pngs[-1] if pngs else None
        self._last_observation_image = png
        self._last_observation_text = self._obs_text(resp)
        # The container executes the whole batch and reports failures as one
        # aggregate ``errors[]``; pair each entry back to the originating
        # top-level call_id so the agent sees WHICH of its calls failed. Same
        # pairing drives the FAILED status in the v2 trajectory.
        container_action_errors, failed_indices = _pair_online_mind2web_action_errors(
            actions_to_send,
            sent_result_call_ids,
            action_errors=list(resp.get("action_errors") or []),
        )
        action_errors.update(container_action_errors)
        # The container owns the final answer: it executes ``response`` and reports
        # its own ``answer`` on every step. Stored BEFORE appending the trajectory
        # (``_append_v2_step`` rewrites result.json) so the artifact and the fast
        # path below both ship the container's value.
        self._agent_final_answer = resp["answer"]
        self._append_action_steps(actions_to_send, resp, failed_indices)
        terminated = bool(resp.get("terminated"))
        # Model-emitted calls that ENDED the episode. They get no continuation
        # observation: devs/migration/verify.py forbids a tool result for a
        # terminal call. Keyed on ``action["call_id"]`` and NOT on
        # ``result_call_id``, because an INTERNAL finish call has no ``call_id``
        # and the loop-detect wrapper's injected ``terminate`` carries the
        # intercepted NON-finish model call's id as ``_result_call_id`` -- that
        # call must still be answered. Gated on ``terminated`` because the
        # CONTAINER owns the verdict here: a finish call it did not honour leaves
        # a non-terminal step, whose calls all still need their result.
        terminal_call_ids = {
            action["call_id"]
            for action in actions_to_send
            if action["name"] in LiteFinishToolSet.get_tool_names() and action.get("call_id")
        } if terminated else set()
        truncated = bool(resp.get("truncated"))
        reward: float | None = resp.get("reward")
        eval_info: dict[str, Any] = {}
        if terminated or truncated:
            reward, eval_info = await self._evaluate()
            if eval_info:
                self._write_result_json()
                self._write_eval_json(eval_info)
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info={
                    EXECUTED_ACTIONS_INFO_KEY: resp.get("executed", []),
                    "errors": resp.get("errors", []),
                    "answer": self._agent_final_answer,
                    "trajectory_dir": self.trajectory_dir,
                    "result_path": self.result_path,
                    **eval_info,
                },
            ),
            ordered_call_ids=result_call_ids,
            continue_call_ids=[
                call_id for call_id in result_call_ids
                if call_id not in terminal_call_ids
            ],
            images=pngs,
            text=self._last_observation_text,
            metadata={
                **self._last_obs,
                "trajectory_dir": self.trajectory_dir,
                "result_path": self.result_path,
            },
            feedback=action_errors,
        )

    async def close(self) -> None:
        if self._instance_id is None:
            return
        iid = self._instance_id
        self._instance_id = None
        try:
            await asyncio.to_thread(self._post, "/close", {"instance_id": iid})
        except Exception as e:
            logger.warning("online_mind2web container close failed: %s", e)
        if self._tmp_trajectory is not None:
            self._tmp_trajectory.cleanup()
            self._tmp_trajectory = None
        self._last_observation_image = None
        self._last_observation_text = None
        self._step_count = 0


class OnlineMind2WebContainerServices(SingletonContainerServices):
    rm_timeout_s = _RM_TIMEOUT_S
    rm_label = "online_mind2web"

    def container_name(self, scope) -> str:
        return _container_name(scope.server_port)

    def ensure(self, env_id: str) -> None:
        if env_id in _services_started:
            return
        url = os.environ.get("ONLINE_MIND2WEB_RPC_URL", "")
        if _healthz(url):
            _services_started.add(env_id)
            return

        name = _container_name()
        docker_rm_f(name, timeout=_RM_TIMEOUT_S, label="online_mind2web")

        from lite.gym.utils.backend.ports import allocate_ports

        host_port = allocate_ports(
            n=1,
            range_start=_ONLINE_MIND2WEB_PORT_START,
            range_end=_ONLINE_MIND2WEB_PORT_END + 1,
        )[0]
        vw, vh = _VIEWPORT
        mem = f"{max(4, int(_RESOLVED_INSTANCES * 0.7))}g"
        logger.info(
            "online_mind2web container: docker run %s "
            "(instances=%d, host_port=%d, mem=%s, viewport=%dx%d)",
            name,
            _RESOLVED_INSTANCES,
            host_port,
            mem,
            vw,
            vh,
        )
        docker_run(
            name,
            image_for("online_mind2web", tag=_ONLINE_MIND2WEB_IMAGE),
            mem=mem,
            port=(host_port, 8000),
            env={
                "ONLINE_MIND2WEB_INSTANCES": str(_RESOLVED_INSTANCES),
                "ONLINE_MIND2WEB_VIEWPORT_W": str(vw),
                "ONLINE_MIND2WEB_VIEWPORT_H": str(vh),
                "ONLINE_MIND2WEB_PAGE_LOAD_TIMEOUT_S": str(_PAGE_LOAD_TIMEOUT_S),
                "ONLINE_MIND2WEB_RESET_SETTLE_S": str(_RESET_SETTLE_S),
                "ONLINE_MIND2WEB_INSTANCE_TTL_S": str(_INSTANCE_TTL_S),
                "ONLINE_MIND2WEB_REAPER_INTERVAL_S": str(_REAPER_INTERVAL_S),
            },
        )

        new_url = f"http://localhost:{host_port}"
        os.environ["ONLINE_MIND2WEB_RPC_URL"] = new_url
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if _healthz(new_url):
                _services_started.add(env_id)
                logger.info("online_mind2web container ready: %s", new_url)
                return
            time.sleep(2)
        raise EnvDepsMissingError(
            what=f"online_mind2web container {name} started but /healthz not ok within 120s",
            install=f"check `docker logs {name}` and lite/gym/envs/online_mind2web/scripts/install.sh",
            see="lite/gym/envs/online_mind2web/README.md",
        )

    def health(self, _env_id: str) -> None:
        if not _healthz(os.environ.get("ONLINE_MIND2WEB_RPC_URL", "")):
            raise EnvDepsMissingError(
                what="online_mind2web container not serving (/healthz unreachable)",
                install="uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh",
                see="lite/gym/envs/online_mind2web/README.md",
            )

    def _evict_cache(self, env_id: str) -> None:
        _services_started.discard(env_id)
        os.environ.pop("ONLINE_MIND2WEB_RPC_URL", None)
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

    # live_ids inherited from SingletonContainerServices: SINGLETON backends have
    # no per-instance reconcile view.


def _make_remote_env(
    *,
    task_id: str,
    task: dict[str, Any],
    base_max_steps: int,
    **kwargs: Any,
) -> RemoteOnlineMind2WebEnv:
    return RemoteOnlineMind2WebEnv(task_id=task_id, task=task, base_max_steps=base_max_steps, **kwargs)


def _register_tasks() -> None:
    # Env-wide make() defaults from default.yaml make_kwargs.
    registry.set_env_make_kwargs("online_mind2web", CFG.make_kwargs)
    tasks = _load_tasks()
    if not tasks:
        raise ValueError("Mind2Web manifest loaded but contained no tasks.")

    seen: set[str] = set()
    for task in tasks:
        task_id = task["task_id"]
        if task_id in seen:
            raise ValueError(f"Duplicate Mind2Web task_id {task_id!r}")
        seen.add(task_id)

        base_max_steps = _MAX_STEPS if _MAX_STEPS is not None else _DEFAULT_MAX_STEPS
        task["max_steps"] = base_max_steps
        register(
            key=f"online_mind2web@{task_id}",
            entry_point=partial(
                _make_remote_env,
                task_id=task_id,
                task=task,
                base_max_steps=base_max_steps,
            ),
            split=str(task.get("split") or "eval"),
            metadata=_task_metadata(task),
            max_steps=base_max_steps,
            step_timeout=_STEP_TIMEOUT,
        )

    logger.info("Registered %d Mind2Web tasks", len(tasks))


_register_tasks()
register_services("online_mind2web", OnlineMind2WebContainerServices())
from lite.gym.services import BackendFamily, register_family  # noqa: E402

register_family("online_mind2web", BackendFamily.SINGLETON)
