"""ScreenSpot-Pro benchmark for CUA-Lite.

Single-step click grounding: the agent sees a screenshot + instruction,
produces a click coordinate, evaluated against the ground-truth bounding box.

Data auto-downloads from HuggingFace on first use. To pre-download manually:
    uv run python lite/gym/envs/screenspot_pro/scripts/utils/download_tasks.py

Dataset: https://huggingface.co/datasets/likaixin/ScreenSpot-Pro

Usage:
    uv run python -c "import lite.gym as gym; print(gym.registry.task_ids('screenspot_pro'))"
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.calls import EnvAction, RuntimeEnvAction
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
    record_model_action_error,
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
# SCREENSPOT_PRO_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
_REWARD_SHAPING = CFG.env_kwargs["reward_shaping"]
_VALID_ACTIONS = resolve_valid_actions(  # ⚠ advanced — defines the action enum
    CFG.env_kwargs["valid_actions"],
    env_name="screenspot_pro", platform="desktop", task_type="grounding.point",
)
_EXTRA_TOOLS   = CFG.env_kwargs["extra_tools"]
# ============================================================================

# ---------------------------------------------------------------------------
# Extra tools
# ---------------------------------------------------------------------------
# EMPTY on purpose, and that is the whole point of it existing: ScreenSpot-Pro
# has no refusal split (every task has a real target), so this env owns no
# extra tool of its own — unlike its sibling ``osworld_g``, which offers
# ``report_infeasible``. Declaring the (empty) tool set anyway routes
# ``extra_tools`` through :func:`resolve_extra_tools`, so a yaml naming
# something this env cannot execute RAISES instead of being silently
# swallowed. The canonical finish tools (``response`` / ``terminate``) are
# resolved centrally by that helper and must NOT be listed here.
class ScreenspotProTools(BaseTools):
    """What screenspot_pro declares beyond the grounding surface: nothing."""

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {}


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = ScreenspotProTools.get_tool_names() | LiteFinishToolSet.get_tool_names()

# ---------------------------------------------------------------------------
# Data paths — resolved from HuggingFace cache via snapshot_download
# ---------------------------------------------------------------------------

_REPO_ID = "likaixin/ScreenSpot-Pro"

_ALLOW_PATTERNS = ["annotations/*.json", "images/**/*.png"]

def _cached_data_dir() -> Path | None:
    """The ScreenSpot-Pro snapshot dir IF already cached — local only, no network.

    Split out of _ensure_data_dir so callers that must not download (e.g. a test's
    skipif) can probe availability without triggering the networked fallback."""
    from huggingface_hub import snapshot_download

    try:
        snap = Path(snapshot_download(
            _REPO_ID, repo_type="dataset",
            allow_patterns=_ALLOW_PATTERNS, local_files_only=True,
        ))
        return snap if (snap / "annotations").is_dir() else None
    except Exception:
        return None


def _ensure_data_dir() -> Path | None:
    """Ensure ScreenSpot-Pro data is available, auto-downloading if needed."""
    from huggingface_hub import snapshot_download

    # Fast path: already cached (local only)
    cached = _cached_data_dir()
    if cached is not None:
        return cached

    # Auto-download
    try:
        logger.info(
            "ScreenSpot-Pro data not cached. Downloading from HuggingFace (%s) ...",
            _REPO_ID,
        )
        path = snapshot_download(
            _REPO_ID,
            repo_type="dataset",
            allow_patterns=_ALLOW_PATTERNS,
        )
        snap = Path(path)
        if (snap / "annotations").is_dir():
            logger.info("ScreenSpot-Pro data cached at: %s", snap)
            return snap
    except Exception as e:
        logger.warning(
            "Failed to download ScreenSpot-Pro data: %s\n"
            "  Download manually: uv run python lite/gym/envs/screenspot_pro/scripts/utils/download_tasks.py\n"
            "  See: lite/gym/envs/screenspot_pro/README.md", e,
        )

    return None

# Data paths are resolved LAZILY in _register_tasks(), never at module import:
# calling _ensure_data_dir() here would run a blocking snapshot_download on
# whatever thread imports this module — including the env-server asyncio loop
# via the reaper's registry._import_all() — freezing the whole server. The HF
# snapshot is resolved on the first task_ids()/make() probe instead (the lazy
# pattern webgym uses).

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ScreenSpotProEnv(LiteBaseEnv):
    """Single-step click grounding environment for ScreenSpot-Pro.

    Each episode:
    1. reset() — loads a screenshot from disk, returns it as base64.
    2. step()  — accepts a single click action, evaluates against ground-truth bbox.
    3. close() — no-op.
    """

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = ScreenspotProTools

    def __init__(
        self,
        *,
        annotation: dict[str, Any],
        images_dir: Path,
        reward_shaping: bool = _REWARD_SHAPING,
        valid_actions: list[str] | None = _VALID_ACTIONS,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        **kwargs: Any,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"ScreenSpotProEnv got unexpected env kwargs: {unknown}")
        # dict(...): `annotation` is the shared registration-side record —
        # copy, don't alias (step() also hands it to the observation info).
        self._annotation = dict(annotation)
        self._images_dir = images_dir
        self._reward_shaping = reward_shaping
        # Unconditional assignment through the shared resolver: ``None`` means
        # no filtering, ``[]`` deliberately strips the grounding tool, and an
        # unknown name fails at the config boundary instead of becoming a
        # silently different action enum.
        self._valid_actions = resolve_valid_actions(
            valid_actions,
            env_name="screenspot_pro", platform="desktop", task_type="grounding.point",
        )
        # Raises ValueError on a name this env cannot execute — same contract
        # as the sibling ``osworld_g``. Without this, an ``extra_tools`` yaml
        # entry was absorbed by ``**kwargs`` and dropped without a word.
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._screenshot: bytes | None = None

    @staticmethod
    def _task_metadata(annotation: dict[str, Any]) -> LiteCUAMetadata:
        """Same-source metadata builder.

        Carries the yaml no-override defaults for
        ``valid_actions`` / ``extra_tool_schemas``; ``_runtime_metadata``
        amends them from the live env_kwargs.
        """
        return LiteCUAMetadata(
            # Single-step click prediction → routes to the adapter family's
            # GroundingPoint adapter (trimmed click-only schema; FullHistoryProtocol).
            dims=("desktop", "grounding.point"),
            valid_actions=copy_valid_actions(_VALID_ACTIONS),
            extra_tool_schemas=resolve_extra_tools(
                _EXTRA_TOOLS, tools=ScreenspotProTools, env_name="screenspot_pro",
            ),
            others={
                # list(...): sever the registered/live others from the
                # annotation record's own lists.
                "bbox": list(annotation["bbox"]),
                "img_size": list(annotation["img_size"]),
                "application": annotation.get("application", ""),
                "group": annotation.get("group", ""),
                "ui_type": annotation.get("ui_type", ""),
                "platform_os": annotation.get("platform", ""),
                "img_filename": annotation.get("img_filename", ""),
            },
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendments: valid_actions / extra_tools resolved at
        # construction. (reward_shaping only shapes the reward.)
        md = self._task_metadata(self._annotation)
        return dataclasses.replace(
            md,
            valid_actions=self._valid_actions,
            extra_tool_schemas=self._extra_tool_schemas,
        )

    async def reset(self) -> LiteEnvObservation:
        img_path = self._images_dir / self._annotation["img_filename"]
        def _load_image():
            with open(img_path, "rb") as f:
                return encode_png(f.read())
        self._screenshot = await asyncio.to_thread(_load_image)
        instruction = self._annotation["instruction"]
        return LiteEnvObservation(image=self._screenshot, text=instruction)

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        unsupported_reasons: dict[int, str] = {}
        actions_with_result_ids, ingress_errors = prepare_env_tool_calls(
            actions,
            self.metadata,
            validate_top_level_action=True,
        )
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        model_error_actions: list[LiteExecutedAction] = []
        had_model_action_error = False
        # Classifier-gate inactive finish tools exactly as sibling grounding
        # envs do: a model calling ``terminate`` when it was never advertised
        # must not be scored as if it had answered.
        executable_actions: list[tuple[EnvAction, str | None]] = []
        malformed_action_ids: set[int] = set()
        for action, result_call_id in actions_with_result_ids:
            name = action["name"]
            tool_feedback, noop_reason = standalone_tool_call_feedback_with_reason(
                action,
                _KNOWN_STANDALONE_TOOL_NAMES,
                self.metadata.extra_tool_schemas,
            )
            if tool_feedback is not None:
                unsupported_reasons[id(action)] = noop_reason or tool_feedback.message
                if result_call_id:
                    action_errors[result_call_id] = tool_feedback
                continue
            if name == "point":
                invalid_action = invalid_action_message(action, self._valid_actions)
                if invalid_action:
                    unsupported_reasons[id(action)] = invalid_action
                    if result_call_id:
                        action_errors[result_call_id] = current_feedback(invalid_action)
                    continue
                try:
                    self._extract_click([action])
                except MODEL_ACTION_ERROR_TYPES as e:
                    had_model_action_error = True
                    malformed_action_ids.add(id(action))
                    record_model_action_error(action_errors, result_call_id, e, action_name=name)
                    model_error_actions.append({
                        "call": "noop",
                        "args": {"name": name, "reason": str(e)},
                    })
                    continue
            executable_actions.append((action, result_call_id))
        actions = [action for action, _ in executable_actions]
        # The reward scores the ANSWER. It is 0.0 when the answer is missing,
        # ambiguous, or malformed -- and ONLY then. Two inputs decide that, and
        # the split between them is what the gate is FOR:
        #
        #   * a botched ATTEMPT TO ANSWER poisons the score. That is
        #     ``had_model_action_error`` (a ``point`` whose coordinate would not
        #     parse) and ``ingress_errors`` (a call rejected as an invalid
        #     action for this task -- notably answering through the wrong
        #     wrapper, ``computer(actions=[{"action": "point", ...}])``). Both
        #     are the model reaching for the answer and missing, which makes the
        #     surviving ``point`` one of SEVERAL attempts, not a lone answer.
        #   * a rejected NON-ANSWER call does not. An unknown tool, or a
        #     ``terminate`` this task never advertised, is reported to the model
        #     as feedback and otherwise ignored. These pass ingress and are
        #     caught in the loop above as inactive/unknown STANDALONE tools, so
        #     they land in ``unsupported_reasons`` -- never in ``ingress_errors``.
        #
        # A model emitting ONLY ``terminate`` needs no gate: ``_evaluate`` sees
        # no ``point``, so there is no answer and it returns 0.0 by itself.
        answer_is_malformed = had_model_action_error or bool(ingress_errors)
        reward = 0.0 if answer_is_malformed else self._evaluate(actions)
        # Log executed actions in de-normalized native pixels so the trajectory
        # log line is directly comparable with ``annotation.bbox`` (mirrors
        # osworld_g convention).
        executed_actions: list[LiteExecutedAction] = []
        img_w, img_h = self._annotation["img_size"]
        for action, result_call_id in actions_with_result_ids:
            action_id = id(action)
            if action_id in malformed_action_ids:
                continue
            name = action["name"]
            args = action["arguments"]
            if action_id in unsupported_reasons:
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": unsupported_reasons[action_id]},
                })
                continue
            if result_call_id in action_errors:
                continue
            if name == "point":
                coord = args.get("coordinate")
                if coord and len(coord) >= 2:
                    # clamp=False preserves this grounding task's pixel contract.
                    # as_float=True because ``_evaluate`` scores the UNROUNDED
                    # coordinate: rounding here makes the log disagree with the
                    # recorded reward at a box boundary. Measured: task
                    # ``linux_common_linux_0``, point [163, 192] on 3456x2160 →
                    # x = 563.328, just OUTSIDE bbox [545, 402, 563, 425]. Logged as
                    # int 563 it lands exactly on the right edge, so a log-only audit
                    # sees 1.0 against the recorded (correct) 0.0.
                    px, py = norm_to_pixel(coord, img_w, img_h, clamp=False, as_float=True)
                    executed_actions.append({"call": "point", "args": {"x": px, "y": py}})
                else:
                    executed_actions.append({"call": "point", "args": {"reason": "missing coordinate"}})
            else:
                executed_actions.append({"call": name, "args": args})
        executed_actions.extend(model_error_actions)
        feedback = dict(action_errors)
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
    # Evaluation
    # -------------------------------------------------------------------

    def _evaluate(self, actions: list[EnvAction]) -> float:
        """Evaluate click against ground-truth bounding box.

        CUA-Lite coordinates are in [0, 1000] range.
        Ground-truth bbox is in pixel coordinates: [x1, y1, x2, y2].

        If ``reward_shaping=False`` (default): 1.0 if inside bbox, else 0.0 --
        the benchmark's own metric, which is what eval must report.
        If ``reward_shaping=True``: 1.0 if inside bbox, otherwise a distance-based
        reward that decays from 1.0 to 0.0 as the click moves away from the bbox
        center, using ``reward = max(0, 1 - dist / max_dist)`` where max_dist is
        half the image diagonal (so clicks at the image edge get ~0).

        EXACTLY ONE ``point`` scores. Zero points is no answer. Two or more is
        an ambiguous answer, and taking the first (which is what this used to
        do) pays a model for hedging -- emitting several candidate clicks and
        letting the grader find the hit turns a single-shot grounding benchmark
        into a multiple-choice one. Both are 0.0.
        """
        answer_actions = [action for action in actions if action["name"] == "point"]
        if len(answer_actions) != 1:
            return 0.0

        click = self._extract_click(answer_actions)
        if click is None:
            return 0.0

        # Normalize click from [0, 1000] to [0, 1]
        click_x = click[0] / 1000.0
        click_y = click[1] / 1000.0

        # Normalize bbox from pixels to [0, 1]
        bbox = self._annotation["bbox"]
        img_w, img_h = self._annotation["img_size"]
        norm_bbox = [
            bbox[0] / img_w,
            bbox[1] / img_h,
            bbox[2] / img_w,
            bbox[3] / img_h,
        ]

        # Inside bbox → always 1.0
        if norm_bbox[0] <= click_x <= norm_bbox[2] and norm_bbox[1] <= click_y <= norm_bbox[3]:
            return 1.0

        if not self._reward_shaping:
            return 0.0

        # Distance-based reward: compute distance from click to bbox center,
        # normalize by half the image diagonal so reward ∈ [0, 1).
        import math
        cx = (norm_bbox[0] + norm_bbox[2]) / 2
        cy = (norm_bbox[1] + norm_bbox[3]) / 2
        dist = math.sqrt((click_x - cx) ** 2 + (click_y - cy) ** 2)
        max_dist = math.sqrt(0.5**2 + 0.5**2)  # half diagonal of [0,1]×[0,1]
        return max(0.0, 1.0 - dist / max_dist)

    @staticmethod
    def _extract_click(actions: list[EnvAction]) -> list[float] | None:
        """Extract click coordinate from actions list."""
        for action in actions:
            name = action["name"]
            args = action["arguments"]
            if name == "point":
                coord = args.get("coordinate")
                if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                    raise ValueError(f"malformed normalized coordinate: {coord!r}")
                x, y = parse_norm_pair_values(coord)
                return [x, y]
        return None

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_annotations(annotations_dir: Path | None) -> list[tuple[str, int, dict[str, Any]]]:
    """Load all annotations from JSON files.

    Returns list of (filename_stem, index, annotation_dict).
    """
    if annotations_dir is None or not annotations_dir.is_dir():
        from lite.gym.errors import EnvDepsMissingError
        raise EnvDepsMissingError(
            what="ScreenSpot-Pro annotation data not downloaded",
            install="uv run python lite/gym/envs/screenspot_pro/scripts/utils/download_tasks.py",
            see="lite/gym/envs/screenspot_pro/README.md",
        )

    results = []

    for json_file in sorted(annotations_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        stem = json_file.stem
        for idx, ann in enumerate(data):
            results.append((stem, idx, ann))

    return results

# ---------------------------------------------------------------------------
# Task registration (lazy — see ensure_services below)
# ---------------------------------------------------------------------------
#
# Module import does NOT touch the HF cache. Loading is deferred to
# :func:`ensure_services` so the env-server can boot before annotations
# are fetched. ``GET /envs/{env_id}`` then surfaces a clean
# ``EnvDepsMissingError`` on probe, and a later request (after the HF
# snapshot completes) auto-recovers without restarting the server. See
# ``osworld_g/main.py`` for the same pattern + rationale.

_tasks_registered = False


def _register_tasks() -> None:
    """Load annotations from the HF snapshot and register every task. Idempotent.

    Raises :class:`lite.gym.errors.EnvDepsMissingError` if the annotation
    data is still missing.
    """
    global _tasks_registered
    if _tasks_registered:
        return
    # Resolve the HF snapshot HERE (lazy), not at module import — this runs on
    # the task_ids()/make() probe via ScreenSpotProServices, so importing the
    # module never triggers a blocking snapshot_download on the event loop.
    data_dir = _ensure_data_dir()
    annotations_dir = data_dir / "annotations" if data_dir else None
    images_dir = data_dir / "images" if data_dir else None
    all_annotations = _load_annotations(annotations_dir)
    for stem, idx, ann in all_annotations:
        task_id = f"{stem}_{idx}"
        register(
            key=f"screenspot_pro@{task_id}",
            entry_point=lambda *, _ann=ann, _imgs=images_dir, **kw: ScreenSpotProEnv(
                annotation=_ann, images_dir=_imgs, **kw,
            ),
            split="eval",
            metadata=ScreenSpotProEnv._task_metadata(ann),
        )
    _tasks_registered = True
    logger.debug(
        "Registered %d screenspot_pro tasks from %s",
        len(all_annotations),
        annotations_dir,
    )


def _ensure_services(env_id: str) -> None:
    """Registry hook: realise the task list on first probe.

    Re-checks data presence on every (uncached) call so an HF snapshot
    that completes after server start is picked up automatically.
    """
    if env_id != "screenspot_pro":
        return
    _register_tasks()


class ScreenSpotProServices(EnvServices):
    """Env-server capability for screenspot_pro: lazy task discovery (HF
    snapshot download) on first probe. No shared backend, no drift — a static
    grounding dataset, so it overrides the register-only ``register_tasks`` hook
    (fired by ``task_ids``) and leaves ``ensure`` a no-op (nothing to boot)."""

    def register_tasks(self, env_id: str) -> None:
        _ensure_services(env_id)


register_services("screenspot_pro", ScreenSpotProServices())
from lite.gym.services import BackendFamily, register_family  # noqa: E402

register_family("screenspot_pro", BackendFamily.PURE)
