"""OSWorld-G grounding benchmark for CUA-Lite.

Single-step click grounding: the agent sees a screenshot + instruction and
either (a) outputs a click coordinate, evaluated against the ground-truth
region (bbox or polygon), or (b) calls ``report_infeasible`` for the 54
"refusal" tasks where the requested element does not exist.

Three task types live in the same benchmark — distinguishable by
``box_type``:
  - ``bbox``     — 470 tasks. ``box_coordinates`` is ``[x, y, w, h]``.
  - ``polygon``  — 40 tasks. ``box_coordinates`` is a flat list of
                   ``[x0, y0, x1, y1, ...]`` polygon vertices.
  - ``refusal``  — 54 tasks. Element does NOT exist in the screenshot;
                   model is expected to call ``report_infeasible``. Tagged
                   with ``metadata.others["exclude_reason"] = "refusal"``
                   so users can filter via
                   ``--filter "lambda m: not m.others.get('exclude_reason')"``.

Two instruction variants per task (env_kwarg ``instruction_style``):
  - ``"original"`` (default) — short label / intent (e.g. "Open the
    filter function for search settings.")
  - ``"refined"`` — longer, more visually-grounded description
    (e.g. "Click the button that includes an icon of funnel on the
    right of the 'search settings' bar")

Data is auto-cloned from ``https://github.com/xlang-ai/OSWorld-G`` into
``.cache/OSWorld-G/`` (gitignored). To pre-clone manually::

    uv run python lite/gym/envs/osworld_g/scripts/utils/download_tasks.py

Reference: https://arxiv.org/abs/2505.13227 (Xie et al. 2025).

Usage:
    uv run python -c "import lite.gym as gym; print(len(gym.registry.task_ids('osworld_g')))"
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, ClassVar

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import EnvAction, RuntimeEnvAction
from lite.core.tools.extra_tools import LiteFinishToolSet, make_report_infeasible_tool
from lite.core.tools.schemas import BaseTools
from lite.gym.base import LiteBaseEnv
from lite.gym.registry import register
from lite.gym.services import register_services
from lite.gym.services import EnvServices
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
    LiteExecutedAction,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.coordinate import norm_to_pixel, parse_norm_pair_values
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    ToolErrorFeedback,
    current_feedback,
    error_only_feedback,
    record_model_action_error,
    unavailable_action_message,
    unknown_tool_message,
)
from lite.gym.utils.feedback.ingress import (
    classify_standalone_tool_call,
    invalid_action_message,
    prepare_env_tool_calls,
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
from lite.utils.image import encode_png

logger = logging.getLogger(__name__)

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# OSWORLD_G_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
_INSTRUCTION_STYLE = CFG.env_kwargs["instruction_style"]
_REWARD_SHAPING    = CFG.env_kwargs["reward_shaping"]
_VALID_ACTIONS     = resolve_valid_actions(  # ⚠ advanced — defines the action enum
    CFG.env_kwargs["valid_actions"],
    env_name="osworld_g", platform="desktop", task_type="grounding.point",
)
_EXTRA_TOOLS       = CFG.env_kwargs["extra_tools"]
# ============================================================================

# ---------------------------------------------------------------------------
# Data paths — local clone of xlang-ai/OSWorld-G under .cache/
# ---------------------------------------------------------------------------

_ENV_DIR = Path(__file__).parent
_DATA_DIR = _ENV_DIR / ".cache" / "OSWorld-G"
_BENCHMARK_DIR = _DATA_DIR / "benchmark"
_IMAGES_DIR = _BENCHMARK_DIR / "images"
_TASKS_ORIGINAL = _BENCHMARK_DIR / "OSWorld-G.json"
_TASKS_REFINED = _BENCHMARK_DIR / "OSWorld-G_refined.json"
# Paper-canonical 5-category classification (text_matching /
# element_recognition / layout_understanding / fine_grained_manipulation /
# refusal). Multi-tag — a task can land in 1+ buckets. Exact mapping is
# the upstream-provided ``classification_result.json`` (NOT inferable from
# raw ``GUI_types``; some buckets share GUI_types like
# ``Accordion/Collapsible Panel`` appearing in both layout and fine-grained).
_PAPER_CLASSIFICATION_FILE = _BENCHMARK_DIR / "classification_result.json"


def _data_present() -> bool:
    """Lightweight check: do the benchmark JSONs and images dir exist?"""
    return (
        _TASKS_ORIGINAL.is_file()
        and _TASKS_REFINED.is_file()
        and _IMAGES_DIR.is_dir()
    )


def _load_paper_categories() -> dict[str, list[str]]:
    """Build ``task_id → [paper categories]`` from the upstream
    ``classification_result.json``. Returns ``{}`` if the file is absent
    so the env still loads (each task's ``paper_category`` becomes an
    empty list).
    """
    if not _PAPER_CLASSIFICATION_FILE.is_file():
        # The env still loads (paper_category is optional metadata), but warn loudly
        # rather than silently degrading every task to ``[]`` — but only when the env
        # is actually present (a missing-everything checkout shouldn't spam). Re-fetch
        # via scripts/utils/download_tasks.py (the file ships in the upstream clone).
        if _data_present():
            logger.warning(
                "osworld_g: %s absent — every task's `paper_category` will be empty; "
                "re-run scripts/utils/download_tasks.py to fetch it.",
                _PAPER_CLASSIFICATION_FILE.name,
            )
        return {}
    raw = json.loads(_PAPER_CLASSIFICATION_FILE.read_text())
    out: dict[str, list[str]] = {}
    for category, entries in (raw.get("classified") or {}).items():
        for entry in entries:
            tid = entry["id"] if isinstance(entry, dict) else str(entry)
            out.setdefault(tid, []).append(category)
    return out


_PAPER_CATEGORIES: dict[str, list[str]] = _load_paper_categories()


# ---------------------------------------------------------------------------
# Refusal extra tool (mirrors OSWorld's report_infeasible)
# ---------------------------------------------------------------------------

class OsworldGTools(BaseTools):
    """What osworld_g declares beyond the grounding surface: ``report_infeasible``."""

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "report_infeasible": make_report_infeasible_tool(
            description=(
                "Report that the requested UI element does NOT exist in "
                "the screenshot, or that the task cannot be completed as "
                "described. The score is 1.0 if and only if the task is a "
                "true refusal task (no element to click); for tasks with "
                "an actual target this scores 0.0."
            ),
            reason_description="Why the requested element is not present.",
        ),
    }


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = OsworldGTools.get_tool_names() | LiteFinishToolSet.get_tool_names()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class OSWorldGEnv(LiteBaseEnv):
    """Single-step grounding env for OSWorld-G.

    Each episode:
      1. ``reset()`` — loads the screenshot from disk, returns it as
         base64 alongside the chosen instruction (original or refined).
      2. ``step()``  — accepts a single ``click`` or ``report_infeasible``
         action, evaluates against the ground-truth region (bbox /
         polygon / refusal), terminates immediately.
      3. ``close()`` — no-op.
    """

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = OsworldGTools

    def __init__(
        self,
        *,
        annotation_original: dict[str, Any],
        annotation_refined: dict[str, Any],
        images_dir: Path,
        task_id: str = "",
        instruction_style: str = _INSTRUCTION_STYLE,
        reward_shaping: bool = _REWARD_SHAPING,
        valid_actions: list[str] | None = _VALID_ACTIONS,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        **kwargs: Any,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"OSWorldGEnv got unexpected env kwargs: {unknown}")
        if instruction_style not in ("original", "refined"):
            raise ValueError(
                f"instruction_style must be 'original' or 'refined', got {instruction_style!r}"
            )
        # dict(...): both annotations are shared registration-side records
        # (entry_point default-arg bound) — copy, don't alias.
        self._ann_original = dict(annotation_original)
        self._ann_refined = dict(annotation_refined)
        self._images_dir = images_dir
        self._task_id = task_id
        self._instruction_style = instruction_style
        self._reward_shaping = reward_shaping
        # Unconditional assignment through the shared resolver: the signature
        # default (yaml-sourced) is the single source of truth for "omitted",
        # ``None`` means no filtering, ``[]`` deliberately strips the grounding
        # tool, and an unknown name (typo / finish tool) raises here rather
        # than silently reshaping the action enum.
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="osworld_g",
            platform="desktop", task_type="grounding.point",
        )
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._screenshot: bytes | None = None

    @property
    def _annotation(self) -> dict[str, Any]:
        return (
            self._ann_refined
            if self._instruction_style == "refined"
            else self._ann_original
        )

    @staticmethod
    def _task_metadata(annotation: dict[str, Any], task_id: str) -> LiteCUAMetadata:
        """Same-source metadata builder.
        ``annotation`` is the ORIGINAL variant (ground-truth shape identical
        across variants); extra_tool_schemas / valid_actions /
        instruction_style carry the yaml no-override defaults, amended live
        only on env_kwargs override."""
        # Two distinct fields by intent:
        #   - ``box_type``: descriptive label of the eval shape — one of
        #     ``"bbox"`` / ``"polygon"`` / ``"refusal"``. Useful for
        #     ``--filter`` by shape (e.g. only ``box_type == "bbox"``).
        #   - ``exclude_reason``: aligns with the OSWorld convention — set
        #     ONLY for tasks the user typically wants to skip (refusal
        #     here, matching OSWorld's ``"infeasible"`` / ``"google_auth"``
        #     pattern). Filter via
        #     ``--filter "lambda m: not m.others.get('exclude_reason')"``.
        others: dict[str, Any] = {
            "box_type": annotation["box_type"],
            # list(...): sever the registered/live others from the annotation
            # record's / module registry's own lists.
            "image_size": list(annotation["image_size"]),
            "image_path": annotation["image_path"],
            "GUI_types": list(annotation.get("GUI_types") or []),
            # Paper-canonical 5-class label set (text_matching /
            # element_recognition / layout_understanding /
            # fine_grained_manipulation / refusal); multi-tag, sourced from
            # upstream ``classification_result.json``. Empty if the
            # classification file is absent (older clones).
            "paper_category": list(_PAPER_CATEGORIES.get(task_id, [])),
            "instruction_style": _INSTRUCTION_STYLE,
        }
        if annotation["box_type"] == "refusal":
            others["exclude_reason"] = "refusal"
        return LiteCUAMetadata(
            # Single-step click prediction → routes to the adapter family's
            # GroundingPoint adapter (trimmed click-only schema; FullHistoryProtocol).
            dims=("desktop", "grounding.point"),
            valid_actions=copy_valid_actions(_VALID_ACTIONS),
            extra_tool_schemas=resolve_extra_tools(
                _EXTRA_TOOLS, tools=OsworldGTools, env_name="osworld_g",
            ),
            others=others,
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendments: valid_actions / extra_tools resolved at
        # construction + the instruction_style override.
        md = self._task_metadata(self._ann_original, self._task_id)
        return dataclasses.replace(
            md,
            valid_actions=self._valid_actions,
            extra_tool_schemas=self._extra_tool_schemas,
            others={**md.others, "instruction_style": self._instruction_style},
        )

    async def reset(self) -> LiteEnvObservation:
        ann = self._annotation
        img_path = self._images_dir / ann["image_path"]

        def _load_image():
            with open(img_path, "rb") as f:
                return encode_png(f.read())

        self._screenshot = await asyncio.to_thread(_load_image)
        return LiteEnvObservation(image=self._screenshot,
            text=ann["instruction"],
        )

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        unsupported_reasons: dict[str, str] = {}
        actions_with_result_ids, ingress_errors = prepare_env_tool_calls(
            actions,
            metadata,
            validate_top_level_action=True,
        )
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        unsupported_current_ids: set[str] = set()
        model_error_actions: list[LiteExecutedAction] = []
        executable_actions: list[EnvAction] = []
        inactive_action_ids: set[int] = set()
        unknown_action_ids: set[int] = set()
        unsupported_reason_by_action_id: dict[int, str] = {}
        malformed_action_ids: set[int] = set()
        for action, result_call_id in actions_with_result_ids:
            name = action["name"]
            tool_availability = classify_standalone_tool_call(
                action,
                _KNOWN_STANDALONE_TOOL_NAMES,
                metadata.extra_tool_schemas,
            )
            if tool_availability == "inactive":
                reason = unavailable_action_message(name)
                unsupported_reason_by_action_id[id(action)] = reason
                if result_call_id:
                    unsupported_reasons[result_call_id] = reason
                inactive_action_ids.add(id(action))
                continue
            if tool_availability == "unknown":
                reason = unknown_tool_message(name)
                unsupported_reason_by_action_id[id(action)] = reason
                unknown_action_ids.add(id(action))
                if result_call_id:
                    unsupported_reasons[result_call_id] = reason
                continue
            if name == "point":
                invalid_action = invalid_action_message(action, self._valid_actions)
                if invalid_action:
                    if result_call_id:
                        unsupported_reasons[result_call_id] = invalid_action
                        unsupported_current_ids.add(result_call_id)
                    continue
                try:
                    self._extract_click([action])
                except MODEL_ACTION_ERROR_TYPES as e:
                    malformed_action_ids.add(id(action))
                    record_model_action_error(
                        action_errors, result_call_id, e, action_name=name
                    )
                    model_error_actions.append({
                        "call": "noop",
                        "args": {"name": name, "reason": str(e)},
                    })
                    continue
            executable_actions.append(action)

        # Feedback/logging still records malformed or rejected calls, but the
        # benchmark score follows the first surviving well-formed answer. If no
        # usable answer remains, ``_evaluate`` returns 0.0.
        reward = self._evaluate(executable_actions)
        executed_actions: list[LiteExecutedAction] = []
        img_w, img_h = self._annotation["image_size"]
        for action, result_call_id in actions_with_result_ids:
            if id(action) in malformed_action_ids or result_call_id in action_errors:
                continue
            name = action["name"]
            args = action["arguments"]
            if id(action) in unsupported_reason_by_action_id:
                executed_actions.append({
                    "call": "noop",
                    "args": {
                        "name": name,
                        "reason": (
                            "inactive extra tool"
                            if id(action) in inactive_action_ids
                            else "unknown tool"
                            if id(action) in unknown_action_ids
                            else unsupported_reason_by_action_id[id(action)]
                        ),
                    },
                })
                continue
            if name == "point":
                coord = args.get("coordinate")
                if coord and len(coord) >= 2:
                    # De-normalize cua-lite [0, 1000] → native pixels so the
                    # log line is directly comparable with ``box_coordinates``
                    # (mirrors screenspot_pro / OSWorld convention).
                    # clamp=False preserves this grounding task's pixel contract.
                    # as_float=True because ``_evaluate`` scores the UNROUNDED
                    # coordinate: rounding here makes the log disagree with the
                    # recorded reward at a box boundary. Measured: task
                    # ``1YgyNsIUQY-0``, point [44, 355] on 1920x1080 → y = 383.4,
                    # exactly the top edge of box [75, 383.4, 176.2, 17.8]. Logged
                    # as int 383 it reads as OUTSIDE, so a log-only audit sees 0.0
                    # against the recorded (correct) 1.0. ``box_coordinates`` are
                    # floats already, so this also makes the two directly comparable.
                    px, py = norm_to_pixel(coord, img_w, img_h, clamp=False, as_float=True)
                    executed_actions.append({"call": "point", "args": {"x": px, "y": py}})
                else:
                    executed_actions.append({"call": "point", "args": {"reason": "missing coordinate"}})
            else:
                # ``report_infeasible`` and any other (screenshot / noop) end up here.
                executed_actions.append({"call": name, "args": args})
        executed_actions.extend(model_error_actions)
        feedback = dict(action_errors)
        feedback.update({
            call_id: current_feedback(message)
            if call_id in unsupported_current_ids
            else error_only_feedback(message)
            for call_id, message in unsupported_reasons.items()
        })
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward,
                terminated=True,  # always single-step
                truncated=False,
                info={EXECUTED_ACTIONS_INFO_KEY: executed_actions, "annotation": self._annotation},
            ),
            ordered_call_ids=result_call_ids,
            continue_call_ids=(),
            # Single frame, deliberately: the observation is the task's dataset
            # PNG, and this env is single-step (``terminated=True`` above) with no
            # live screen to re-capture. A per-action frame list here could only
            # repeat these same bytes -- N copies carrying no more information
            # than one.
            images=[self._screenshot],
            text=None,
            feedback=feedback,
        )

    async def close(self) -> None:
        pass

    # -------------------------------------------------------------------
    # Evaluation — three modes by ``box_type``
    # -------------------------------------------------------------------

    def _evaluate(self, actions: list[EnvAction]) -> float:
        """Score the first usable answer call against the annotation.

        The first well-formed ``point`` or ``report_infeasible`` answer scores.
        Zero answer calls is no answer. Later answers are ignored so migration
        results stay comparable with the original first-valid benchmark scoring
        rule.
        """
        ann = self._annotation
        box_type = ann["box_type"]
        answer_action = next(
            (
                action for action in actions
                if action["name"] in {"point", "report_infeasible"}
            ),
            None,
        )
        if answer_action is None:
            return 0.0

        if box_type == "refusal":
            # A refusal task scores 1.0 iff the model called report_infeasible.
            return 1.0 if answer_action["name"] == "report_infeasible" else 0.0

        if answer_action["name"] == "report_infeasible":
            return 0.0

        click = self._extract_click([answer_action])
        if click is None:
            return 0.0

        # Normalize click from cua-lite [0, 1000] to [0, 1].
        click_x = click[0] / 1000.0
        click_y = click[1] / 1000.0
        img_w, img_h = ann["image_size"]
        px = click_x * img_w
        py = click_y * img_h

        if box_type == "bbox":
            # OSWorld-G stores ``[x, y, w, h]`` (top-left + size) — normalize
            # to ``[x1, y1, x2, y2]`` before the in-rect check.
            x, y, w, h = ann["box_coordinates"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            inside = (x1 <= px <= x2) and (y1 <= py <= y2)
            if inside:
                return 1.0
            if not self._reward_shaping:
                return 0.0
            # Distance-based reward (same shape as screenspot_pro): max=0
            # at corners, 1.0 inside.
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = math.hypot((px - cx) / img_w, (py - cy) / img_h)
            max_dist = math.hypot(0.5, 0.5)
            return max(0.0, 1.0 - dist / max_dist)

        if box_type == "polygon":
            polygon = ann["box_coordinates"]
            if _is_point_in_polygon(px, py, polygon):
                return 1.0
            return 0.0

        # Unknown box_type — fail closed.
        logger.warning("Unknown box_type %r for task %s", box_type, self._task_id)
        return 0.0

    @staticmethod
    def _extract_click(actions: list[EnvAction]) -> list[float] | None:
        for action in actions:
            if action["name"] != "point":
                continue
            coord = action["arguments"].get("coordinate")
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                raise ValueError(f"malformed normalized coordinate: {coord!r}")
            x, y = parse_norm_pair_values(coord)
            return [x, y]
        return None


def _is_point_in_polygon(x: float, y: float, polygon: list[float]) -> bool:
    """Ray-casting: is (x, y) inside the polygon defined as a flat
    ``[x0, y0, x1, y1, ...]`` vertex list? Mirrors the upstream reference
    ``OSWorld-G/evaluation/eval.py:_is_point_in_polygon``.
    """
    n = len(polygon) // 2
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i * 2], polygon[i * 2 + 1]
        xj, yj = polygon[j * 2], polygon[j * 2 + 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_tasks() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Load (task_id, original_ann, refined_ann) triples in stable order.

    Matches original ↔ refined by ``id``; raises if the two files don't
    cover identical task ids (the reference always ships matched pairs).
    """
    if not _data_present():
        from lite.gym.errors import EnvDepsMissingError
        raise EnvDepsMissingError(
            what="OSWorld-G benchmark data not downloaded",
            install="uv run python lite/gym/envs/osworld_g/scripts/utils/download_tasks.py",
            see="lite/gym/envs/osworld_g/README.md",
        )

    with open(_TASKS_ORIGINAL) as f:
        originals = json.load(f)
    with open(_TASKS_REFINED) as f:
        refined = json.load(f)

    refined_by_id = {t["id"]: t for t in refined}
    if set(refined_by_id) != {t["id"] for t in originals}:
        warnings.warn(
            "OSWorld-G original and refined task id sets differ — using "
            "intersection only.",
            stacklevel=2,
        )

    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for orig in originals:
        rid = orig["id"]
        ref = refined_by_id.get(rid)
        if ref is None:
            continue
        out.append((rid, orig, ref))
    return out


# ---------------------------------------------------------------------------
# Task registration (lazy — see ensure_services below)
# ---------------------------------------------------------------------------
#
# Module import does NOT touch the benchmark data. Loading is deferred to
# :func:`ensure_services` so the env-server can come up before benchmark
# data is present — ``GET /envs/{env_id}`` then surfaces a clean
# ``EnvDepsMissingError`` on probe, and a later request (after running
# ``download_tasks.py``) auto-recovers without restarting the server.
#
# A previous version called ``_load_tasks()`` at module level, which
# (a) raced background data downloads (first ``GET /envs/{env_id}`` saw
# ``available=false`` even when data appeared 1 s later) and
# (b) wasted RAM on every process whether the env was used or not.

_tasks_registered = False

# Populated in-place by :func:`_register_tasks` so test modules can do
# ``from lite.gym.envs.osworld_g.main import _all_tasks`` and see the
# loaded tasks after the lazy hook fires (reassigning would break the
# imported alias).
_all_tasks: list[tuple[str, dict[str, Any], dict[str, Any]]] = []


def _register_tasks() -> None:
    """Load benchmark JSONs and register every osworld_g task. Idempotent.

    Raises :class:`lite.gym.errors.EnvDepsMissingError` if the benchmark
    data is still missing — callers (``ensure_services`` from the registry,
    or ``make()``) propagate this to the user as a clear install hint.
    """
    global _tasks_registered
    if _tasks_registered:
        return
    _all_tasks.extend(_load_tasks())
    for tid, orig, refined in _all_tasks:
        register(
            key=f"osworld_g@{tid}",
            entry_point=lambda *, _orig=orig, _refined=refined, _tid=tid, **kw: OSWorldGEnv(
                annotation_original=_orig,
                annotation_refined=_refined,
                images_dir=_IMAGES_DIR,
                task_id=_tid,
                **kw,
            ),
            split="eval",
            metadata=OSWorldGEnv._task_metadata(orig, tid),
        )
    _tasks_registered = True
    logger.debug("Registered %d osworld_g tasks from %s", len(_all_tasks), _BENCHMARK_DIR)


def _ensure_services(env_id: str) -> None:
    """Registry hook: realise the task list on first probe.

    Re-checks data presence on every (uncached) call so a download that
    completes after server start is picked up automatically. Subsequent
    calls fast-path once registration succeeds.
    """
    if env_id != "osworld_g":
        return
    _register_tasks()


class OSWorldGServices(EnvServices):
    """Env-server capability for osworld_g: lazy task discovery on first probe.
    No shared backend, no drift — a static grounding dataset, so it overrides the
    register-only ``register_tasks`` hook (fired by ``task_ids``) and leaves
    ``ensure`` a no-op (nothing to boot)."""

    def register_tasks(self, env_id: str) -> None:
        _ensure_services(env_id)


register_services("osworld_g", OSWorldGServices())
from lite.gym.services import BackendFamily, register_family  # noqa: E402

register_family("osworld_g", BackendFamily.PURE)
