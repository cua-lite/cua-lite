"""
MAI-UI Action Space

Wraps actions in a single ``mobile_use`` tool with an ``action`` parameter,
matching the format Tongyi-MAI's MAI-UI mobile model was SFT-trained on.

Reference: ${CUA_LITE_REFERENCES_ROOT}/MAI-UI/src/prompt.py (action
space JSON examples in MAI_MOBILE_SYS_PROMPT) and src/mai_naivigation_agent.py
(parse + render).

Key differences from Qwen3VLMobileActionSpace:

- **Coordinate system: absolute [0, 999]** (not 1000). The model emits ints in
  [0, 999]; we rescale on the way in/out.
- **swipe** uses ``direction`` (up/down/left/right enum) plus an OPTIONAL
  ``coordinate`` anchor — not start/end coordinates. Distance is implicit.
- **drag** is a separate action with ``start_coordinate``/``end_coordinate``
  for precise gestures (used by MAI-UI's drag-and-drop scenarios).
- **system_button** enum values are LOWERCASE (back/home/menu/enter), unlike
  Qwen3VL's Capitalized form. The Lite native space does not include "Menu",
  so we pass it through via the raw tool-call dict on the reverse path.
  MAI-UI has NO spelling for the canonical "Recent" button, so the forward
  path raises rather than emitting an undecodable value.
- **terminate** ``status`` enum is ``success``/``fail`` (not ``failure``).
- No ``key`` action. Reference doesn't expose adb keyevents to the model.

For the swipe semantics specifically, see the design note in the adapter
file — we use "Option A" (direction-only forward, anchor-from-start; reverse
synthesizes start/end with a fixed offset from the anchor).

Usage:
    from lite.agents.models.mai_ui.action_space import MAIUIMobileActionSpace

    space = MAIUIMobileActionSpace()
    tools = space.get_tool_schemas()
    tc = MAIUIMobileActionSpace.mobile_use(action="click", coordinate=[500, 500])
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Literal

from lite.agents.core.action_space import (
    ActionSpaceRegistry,
    BaseActionSpace,
)
from lite.agents.core.action_space.base import (
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.action_space.utils.geometry import (
    compact_number,
    optional_coord,
    required_coord,
    required_model_text,
)
from lite.agents.core.action_space.utils.grounding_point import (
    convert_non_point_call_for_grounding_space,
)
from lite.agents.core.action_space.utils.unknown_wrapper_action import unknown_wrapper_action_batch
from lite.agents.core.action_space.utils.wrapper_enum import filter_wrapper_action_enum
from lite.core.tools.action_space import (
    LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
    clamp_norm,
    merge_adjacent_lite_action_batches,
)
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.extra_tools import (
    LiteFinishToolSet,
    extra_tool_name_and_arguments_are_admitted,
)
from lite.core.tools.schemas import tool

logger = logging.getLogger(__name__)

# The single provider-native wrapper tool MAI-UI was SFT-trained on: every GUI
# and semantic action travels as one ``action`` enum value inside it.
_MAI_NATIVE_TOOL_NAME = "mobile_use"


# ---------------------------------------------------------------------------
# Coordinate rescale: cua-lite [0, 1000] <-> MAI-UI [0, 999]
# ---------------------------------------------------------------------------

# MAI-UI's SCALE_FACTOR (mai_naivigation_agent.py:38)
_SCALE_FACTOR = 999


def _scale_to_mai(pts: list[int]) -> list[int]:
    return [int(round(p * _SCALE_FACTOR / 1000)) for p in pts]


def _scale_from_mai(pts: list[int]) -> list[int]:
    return [int(round(p * 1000 / _SCALE_FACTOR)) for p in pts]


def _required_from_mai(c: Any, *, name: str = "coordinate") -> list[int]:
    """Required MAI-UI [0, 999] -> cua-lite [0, 1000]."""
    pts = required_coord(c, name=name)
    if len(pts) == 4:
        # MAI-UI bbox outputs name a target by rectangle; upstream clicks the
        # center point before execution (mai_naivigation_agent.py:135-137).
        x1, y1, x2, y2 = pts
        pts = [(x1 + x2) // 2, (y1 + y2) // 2]
    elif len(pts) != 2:
        raise ModelToolCallParseError(
            f"{name} must contain exactly 2 or 4 numeric values; got {len(pts)}"
        )
    return _scale_from_mai(pts)


def _optional_from_mai(c: Any) -> list[int] | None:
    """Optional MAI-UI [0, 999] -> cua-lite [0, 1000]."""
    pts = optional_coord(c)
    if pts is None:
        return None
    if len(pts) == 4:
        x1, y1, x2, y2 = pts
        pts = [(x1 + x2) // 2, (y1 + y2) // 2]
    elif len(pts) != 2:
        return None
    return _scale_from_mai(pts)


# ---------------------------------------------------------------------------
# Swipe direction <-> start/end synthesis (Option A — see plan)
# ---------------------------------------------------------------------------

# Default swipe length when synthesizing start/end from a direction-only swipe.
# Expressed in cua-lite normalized [0, 1000] units. ~30% of screen.
_SWIPE_OFFSET = 300


def _direction_from_endpoints(
    start: list[int] | None, end: list[int] | None,
) -> Literal["up", "down", "left", "right"]:
    """Compute the dominant swipe direction from a (start, end) finger trajectory.

    Returns the direction the FINGER moves, matching MAI-UI's prompt
    convention ("Swipe from x to y").
    """
    sx, sy = (start or [500, 500])[:2]
    ex, ey = (end or [500, 500])[:2]
    dx, dy = ex - sx, ey - sy
    if abs(dy) >= abs(dx):
        return "down" if dy > 0 else "up"
    return "right" if dx > 0 else "left"


def _endpoints_from_direction(
    direction: str | None,
    anchor: list[int] | None,
) -> tuple[list[int], list[int]]:
    """Synthesize a (start, end) pair from a direction + optional anchor.

    Used on the reverse path (MAI-UI -> cua-lite). The anchor is the user's
    "swipe this element" hint; we treat it as the finger START and project
    a fixed-distance endpoint in the chosen direction. If no anchor is given
    we use the screen center.
    """
    sx, sy = (anchor or [500, 500])[:2]
    dx = dy = 0
    if direction == "up":
        dy = -_SWIPE_OFFSET
    elif direction == "down":
        dy = +_SWIPE_OFFSET
    elif direction == "left":
        dx = -_SWIPE_OFFSET
    elif direction == "right":
        dx = +_SWIPE_OFFSET
    ex = clamp_norm(sx + dx)
    ey = clamp_norm(sy + dy)
    return [sx, sy], [ex, ey]


# ---------------------------------------------------------------------------
# system_button case mapping (cua-lite Capitalized <-> MAI-UI lowercase)
# ---------------------------------------------------------------------------

# MAI-UI's prompt enumerates exactly four buttons (reference prompt.py:39 /
# adapter.py: `# Options: back, home, menu, enter`). The canonical mobile
# surface has a fifth, "Recent", with no MAI-UI spelling — there is nothing to
# map it to, so the forward path RAISES instead of inventing one. Silently
# lowercasing it produced `recent`, which the reverse map cannot decode and
# which is not a legal canonical button either.
_BUTTON_LITE_TO_MAI = {
    "Home": "home",
    "Back": "back",
    "Enter": "enter",
    "Menu": "menu",
}

_BUTTON_MAI_TO_LITE = {v: k for k, v in _BUTTON_LITE_TO_MAI.items()}


# ---------------------------------------------------------------------------
# terminate status mapping (cua-lite success/failure <-> MAI-UI success/fail)
# ---------------------------------------------------------------------------

_STATUS_LITE_TO_MAI = {"success": "success", "failure": "fail"}
_STATUS_MAI_TO_LITE = {"success": "success", "fail": "failure"}


def _mai_action_values(cls) -> frozenset[str]:
    """Every MAI-UI ``action`` value ``cls``'s from-agent dispatch understands.

    The wrapped GUI values plus the native spellings of standalone canonical
    tools — i.e. exactly the ``mobile_use`` action enum the model was trained
    on. Read when provider output drops the wrapper and uses an action value as
    the tool name, so that flat output converts through the same branches as
    nested output.
    """
    return frozenset(
        action_value
        for action_values in cls.LITE_ACTION_NAME_TO_MAI_ACTION_VALUES.values()
        for action_value in action_values
    ) | frozenset(cls.MAI_ACTION_VALUE_TO_EXTRA_TOOL_NAMES)


# =============================================================================
# Action Space
# =============================================================================

@dataclasses.dataclass
class MAIUIMobileActionSpace(BaseActionSpace, key="mai_ui@mobile"):
    """
    MAI-UI-style action space for mobile interactions.

    Uses a single ``mobile_use`` tool with an ``action`` parameter, matching
    the format Tongyi-MAI MAI-UI was SFT-trained on (see prompt.py:33-42 in
    the reference).

    Coordinate system: absolute [0, 999]. Conversion to/from cua-lite's
    normalized [0, 1000] is done by ``_to_mai`` / ``_from_mai`` helpers
    inside ``convert_tool_calls_to/from_agent``.

    Actions:
        click, long_press, type, swipe (direction + optional anchor),
        drag (start_coordinate + end_coordinate), open, system_button,
        wait, terminate, answer.

    Example:
        space = MAIUIMobileActionSpace()
        tc = MAIUIMobileActionSpace.mobile_use(action="click", coordinate=[500, 500])
    """

    platform: str = "mobile"

    _ACTION_DESCRIPTION = """
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `type`: Input the specified text into the activated input box.
* `swipe`: Swipe in a direction. `coordinate` is optional — provide it to anchor the gesture on a specific UI element.
* `open`: Open an app on the device.
* `drag`: Drag from the start point with coordinate (start_coordinate) to the end point with coordinates (end_coordinate).
* `system_button`: Press the system button.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Output the answer.
"""

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        coordinate="(x, y): coordinate for click/long_press, or anchor for swipe. Range [0, 999].",
        start_coordinate="(x, y): drag start. Range [0, 999]. Required only by `action=drag`.",
        end_coordinate="(x, y): drag end. Range [0, 999]. Required only by `action=drag`.",
        direction="Swipe direction. Required only by `action=swipe`.",
        text="Required only by `action=type`, `action=open`, and `action=answer`.",
        time="Seconds. Required only by `action=long_press` and `action=wait`.",
        button="back, home, menu, or enter. Required only by `action=system_button`.",
        status="The status of the task. Required only by `action=terminate`.",
    )
    def mobile_use(
        action: Literal[
            "click", "long_press", "type", "swipe", "drag",
            "open", "system_button", "wait", "terminate", "answer",
        ],
        coordinate: list[int] | None = None,
        start_coordinate: list[int] | None = None,
        end_coordinate: list[int] | None = None,
        direction: Literal["up", "down", "left", "right"] | None = None,
        text: str | None = None,
        time: float | None = None,
        button: Literal["back", "home", "menu", "enter"] | None = None,
        status: Literal["success", "fail"] | None = None,
    ) -> dict[str, Any]:
        """
        Use a touchscreen to interact with a mobile device, and take screenshots.

        * This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
        * The screen's resolution is 999x999.
        * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element.
        """
        return make_tool_call(
            _MAI_NATIVE_TOOL_NAME,
            {
                "action": action,
                "coordinate": coordinate,
                "start_coordinate": start_coordinate,
                "end_coordinate": end_coordinate,
                "direction": direction,
                "text": text,
                "time": time,
                "button": button,
                "status": status,
            },
        )

    # -------------------------------------------------------------------------
    # Action filtering (cua-lite GUI name -> MAI-UI action values it maps to)
    # -------------------------------------------------------------------------

    LITE_ACTION_NAME_TO_MAI_ACTION_VALUES = {
        "tap": ["click"],
        "long_press": ["long_press"],
        "swipe": ["swipe"],
        "drag": ["drag"],
        "type": ["type"],
        "system_button": ["system_button"],
        "wait": ["wait"],
    }
    # MAI action values that are the wire spelling of canonical STANDALONE extra
    # tools. ``valid_actions`` never touches them, and neither does the system
    # prompt: MAI-UI's Action Space block is SFT text, so the adapter advertises
    # all three whatever ``metadata.extra_tool_schemas`` holds.
    MAI_ACTION_VALUE_TO_EXTRA_TOOL_NAMES = {
        "open": frozenset({"open_app"}),
        "answer": frozenset({"response"}),
        "terminate": frozenset({"terminate"}),
    }

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        allowed = {
            action_value
            for name in valid_actions
            for action_value in cls.LITE_ACTION_NAME_TO_MAI_ACTION_VALUES.get(name) or ()
        }
        allowed.update(cls.MAI_ACTION_VALUE_TO_EXTRA_TOOL_NAMES)
        return filter_wrapper_action_enum(
            schemas,
            wrapper_tool_name=_MAI_NATIVE_TOOL_NAME,
            keep_action_value=lambda action_value: action_value in allowed,
        )

    # There is no active-extras counterpart to the gate above. MAI-UI's
    # model-visible action vocabulary is the ``## Action Space`` block of the
    # system prompt — SFT text, not a rendered tool schema — so the sample's
    # active extra tools have nothing to trim.

    # -------------------------------------------------------------------------
    # Tool call conversion: cua-lite -> MAI-UI
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == "mobile":
            result = []
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                result.extend(self._convert_single_to_agent(make_tool_call(action, action_args)))
            return result

        if name == "tap":
            # The enum has ``click`` and no double-tap counterpart, so ``clicks``
            # has no carrier. Rendering ``tap(clicks=2)`` as a plain ``click``
            # re-parses as ``tap(clicks=1)``: a SINGLE tap where a double tap was
            # asked for, which is a different gesture rather than a coarser one,
            # and one the caller cannot recover -- every retry degrades
            # identically. Raise rather than silently substitute, as
            # ``fara@desktop`` does for the same ``clicks`` loss on ``click``.
            clicks = args.get("clicks", 1)
            if clicks != 1:
                raise ValueError(
                    f"MAI-UI cannot render tap(clicks={clicks}): its action enum "
                    "has only 'click' and the wire carries no repeat count"
                )
            return [MAIUIMobileActionSpace.mobile_use(
                action="click",
                coordinate=_scale_to_mai(required_coord(args.get("coordinate"), dimensions=2)),
            )["function"]]

        if name == "long_press":
            return [MAIUIMobileActionSpace.mobile_use(
                action="long_press",
                coordinate=_scale_to_mai(required_coord(args.get("coordinate"), dimensions=2)),
                time=args.get("duration"),
            )["function"]]

        if name == "swipe":
            start = args.get("start_coordinate")
            end = args.get("coordinate")
            direction = _direction_from_endpoints(start, end)
            anchor = optional_coord(start, dimensions=2)
            return [MAIUIMobileActionSpace.mobile_use(
                action="swipe",
                direction=direction,
                coordinate=_scale_to_mai(anchor) if anchor is not None else None,
            )["function"]]

        if name == "drag":
            # MAI-UI's wire declares ``drag`` with ``start_coordinate`` /
            # ``end_coordinate`` and parses that pair back, so the endpoints survive
            # exactly -- unlike ``swipe``, which the grammar carries as a coarse
            # direction.
            return [MAIUIMobileActionSpace.mobile_use(
                action="drag",
                start_coordinate=_scale_to_mai(
                    required_coord(
                        args.get("start_coordinate"),
                        dimensions=2,
                        name="start_coordinate",
                    )
                ),
                end_coordinate=_scale_to_mai(required_coord(args.get("coordinate"), dimensions=2)),
            )["function"]]

        if name == "type":
            return [MAIUIMobileActionSpace.mobile_use(
                action="type",
                text=args.get("text", ""),
            )["function"]]

        if name == "open_app":
            # MAI-UI uses `text` for app name (prompt.py:37: {"action": "open", "text": "app_name"})
            return [MAIUIMobileActionSpace.mobile_use(
                action="open",
                text=args.get("app_name", ""),
            )["function"]]

        if name == "system_button":
            btn = args.get("button", "")
            if btn not in _BUTTON_LITE_TO_MAI:
                raise ValueError(
                    f"MAI-UI has no native system button for {btn!r}; "
                    f"supported: {sorted(_BUTTON_LITE_TO_MAI)}"
                )
            return [MAIUIMobileActionSpace.mobile_use(
                action="system_button",
                button=_BUTTON_LITE_TO_MAI[btn],
            )["function"]]

        if name == "response":
            return [MAIUIMobileActionSpace.mobile_use(
                action="answer",
                text=args.get("text", ""),
            )["function"]]

        if name == "wait":
            return [MAIUIMobileActionSpace.mobile_use(
                action="wait",
                time=compact_number(args.get("duration")),
            )["function"]]

        if name == "terminate":
            cua_status = args.get("status", "success")
            return [MAIUIMobileActionSpace.mobile_use(
                action="terminate",
                status=_STATUS_LITE_TO_MAI.get(cua_status, cua_status),
            )["function"]]

        if name in LiteMobileActionSpace.get_action_names():
            # A canonical MOBILE action with no branch above has no MAI-UI enum
            # entry (``pinch``/``screenshot``): the enum is
            # {click, long_press, type, swipe, drag, open, system_button, wait,
            # terminate, answer}. Warning and returning ``[]`` deleted the action
            # from the wire, so the round trip came back EMPTY and the caller
            # could not tell. Raise, as the unsupported system button above does.
            raise ValueError(
                f"MAI-UI cannot render canonical tool {name!r}: its action enum "
                "has no counterpart, and dropping it silently loses the action"
            )

        logger.warning("Unsupported cua-lite mobile action for MAI-UI: %s(%s)", name, args)
        return []

    # -------------------------------------------------------------------------
    # Tool call conversion: MAI-UI -> cua-lite
    # -------------------------------------------------------------------------

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert MAI-UI ``mobile_use`` tool_calls back to cua-lite format.

        Unknown-action policy: a ``mobile_use`` action this family cannot
        express becomes an INVALID ``mobile`` action-batch naming the model's
        own action value, so env ingress rejects it and the model sees the
        rejection rather than the action vanishing.

        A NON-wrapper tool name is claimed in three steps: an active extra-tool
        schema first, then a native ``action`` value used AS the tool name
        (wrapper dropped), then an ``action`` argument carried under a wrong
        wrapper name. What survives all three is a standalone Lite call under
        its own wire name — it satisfies no advertised schema and names no
        action this space owns, and env ingress owns that rejection.
        """
        result = []
        for tc in agent_tool_calls:
            result.extend(self._convert_single_from_agent(
                tc,
                active_extra_tool_names=active_extra_tool_names,
                active_extra_tool_schemas=active_extra_tool_schemas,
            ))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(
        self,
        agent_tool_call: dict[str, Any],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]
        action = args.get("action", "")

        if name != _MAI_NATIVE_TOOL_NAME:
            admitted = extra_tool_name_and_arguments_are_admitted(
                name,
                args,
                active_extra_tool_names=active_extra_tool_names,
                active_extra_tool_schemas=active_extra_tool_schemas,
            )
            # The model dropped the ``mobile_use`` wrapper and used the native
            # action value AS the tool name, e.g. ``click(coordinate=...)``. Read
            # it as that action so flat output converts through the same branches
            # as nested output — ``open``, ``answer`` and ``terminate`` included.
            # An ACTIVE env extra outranks a colliding native value: browsergym's
            # ``type(bid=...)`` is the env's tool, not the native ``type`` the
            # dispatch would read as a keyboard action.
            if not action and not admitted and name in _mai_action_values(type(self)):
                action = name
            elif not action or admitted:
                # A standalone tool: an active env extra, or a name this family
                # does not own at all. Its Lite name IS its wire name, so env
                # ingress answers it with model-visible feedback keyed to the
                # call id. Returning ``[]`` here instead deleted the call before
                # any id reached the model, which read downstream as a
                # no-tool-call parse-failure final and ended the episode.
                return [make_tool_call(name, args)]
            # Otherwise ``action`` is set under a non-wrapper name: the model kept
            # the native nesting but got the tool name wrong; the dispatch claims it.

        if action == "click":
            return [LiteMobileActionSpace.tap(
                coordinate=_required_from_mai(args.get("coordinate")),
            )]

        if action == "long_press":
            t = args.get("time")
            return [LiteMobileActionSpace.long_press(
                coordinate=_required_from_mai(args.get("coordinate")),
                duration=float(t) if t is not None else None,
            )]

        if action == "swipe":
            anchor = _optional_from_mai(args.get("coordinate"))
            start, end = _endpoints_from_direction(args.get("direction"), anchor)
            return [LiteMobileActionSpace.swipe(
                start_coordinate=start, coordinate=end,
            )]

        if action == "drag":
            return [LiteMobileActionSpace.drag(
                start_coordinate=_required_from_mai(
                    args.get("start_coordinate"),
                    name="start_coordinate",
                ),
                coordinate=_required_from_mai(
                    args.get("end_coordinate"),
                    name="end_coordinate",
                ),
            )]

        if action == "type":
            return [LiteMobileActionSpace.type(text=required_model_text(args, action=action))]

        if action == "open":
            # MAI-UI uses `text` for app name (NOT `app_name`)
            return [make_tool_call("open_app", {"app_name": args.get("text", "")})]

        if action == "answer":
            return [LiteFinishToolSet.response(text=args.get("text", ""))]

        if action == "system_button":
            btn = args.get("button", "")
            lite_btn = _BUTTON_MAI_TO_LITE.get(btn, btn)
            return [LiteMobileActionSpace.system_button(button=lite_btn)]

        if action == "wait":
            t = args.get("time")
            return [LiteMobileActionSpace.wait(
                duration=float(t) if t is not None else 1.0,
            )]

        if action == "terminate":
            mai_status = args.get("status", "success")
            return [LiteFinishToolSet.terminate(
                status=_STATUS_MAI_TO_LITE.get(mai_status, mai_status),
            )]

        logger.warning("Unknown MAI-UI action: %s(%s)", action, args)
        return [unknown_wrapper_action_batch(LITE_MOBILE_ACTION_BATCH_TOOL_NAME, args)]


# Ensure registry import side-effects
ActionSpaceRegistry.register("mai_ui@mobile", MAIUIMobileActionSpace)


# =============================================================================
# MAI-UI Grounding (single-step click) action space
# =============================================================================
#
# Distinct wire format from ``use``: NO ``mobile_use`` tool_call, NO action
# enum. The model emits free-form text wrapped in
# ``<grounding_think>...</grounding_think>`` (note: NOT the ``use``
# ``<thinking>`` BPE tag) followed by ``<answer>{"coordinate":[x,y]}</answer>``.
#
# This action space's role is purely to convert between cua-lite's
# :class:`LitePointActionSpace.point(coord)` and a synthetic ``answer``
# tool_call that the adapter parses out of the ``<answer>`` block. There are
# no schemas to advertise — the model gets the format from the system prompt
# only (see :data:`MAI_GROUNDING_SYS_PROMPT`).
#
# Naming: per the MAI-UI cookbook the prompt is named
# ``MAI_MOBILE_SYS_PROMPT_GROUNDING`` but the harness is platform-agnostic.
# We register under all 3 platforms so env routing works uniformly.

@dataclasses.dataclass
class MAIUIGroundingPointActionSpace(BaseActionSpace, key=r"mai_ui@(desktop|browser|mobile)@point"):
    """MAI-UI grounding action space (single-step click, all platforms).

    Coordinate frame: identity. MAI-UI's ``use`` harness uses [0, 999]; the
    grounding cookbook uses the same [0, 999] frame in the ``<answer>``
    payload, which lines up with cua-lite [0, 1000] within sub-pixel
    rounding (consistent with the ``use`` harness's pass-through
    rationale — see :class:`MAIUIMobileActionSpace`).
    """

    platform: str | None = None

    # Native action declaration. The wire has NO tool schemas at all — the
    # ``<answer>{"coordinate":[x,y]}</answer>`` block is prompt-only — so the
    # synthetic ``answer`` name exists nowhere except this table and
    # ``convert_tool_calls_{to,from}_agent``. This class subclasses
    # ``BaseActionSpace`` directly, so there is no inherited table to reset.
    LITE_ACTION_NAME_TO_MAI_ACTION_VALUES = {"point": ["answer"]}

    # No tool schemas — the format lives entirely in the system prompt's
    # ``<grounding_think>`` / ``<answer>`` instructions.
    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        return []

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """cua-lite ``point(coord)`` → synthetic ``answer(coordinate=coord)``.

        The adapter renders the resulting tool_call as the literal
        ``<answer>{"coordinate":[x,y]}</answer>`` text block.
        """
        result = []
        for tc in tool_calls:
            name = tool_call_name(tc)
            args = tool_call_arguments(tc)
            if name == "point":
                result.append({
                    "name": "answer",
                    "arguments": {"coordinate": args.get("coordinate")},
                })
            else:
                result.extend(convert_non_point_call_for_grounding_space(
                    tc, surface="MAI-UI grounding action_space",
                ))
        return result

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Synthetic ``answer(coordinate=coord)`` → cua-lite ``point(coord)``.

        Anything else (extra tools like ``report_infeasible``, off-schema
        names) is returned as canonical Lite output — same pattern as
        :class:`Qwen3VLDesktopActionSpace._convert_single_from_agent` for
        non-``computer_use`` names.
        """
        result = []
        for tc in agent_tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            if name == "answer" and "coordinate" in args:
                result.append(LitePointActionSpace.point(
                    coordinate=required_coord(args.get("coordinate"), dimensions=2),
                ))
            else:
                result.append(make_tool_call(name, args))
        return result


# MAI-UI's upstream grounding cookbook runs ScreenSpot-Pro / OSWorld-G /
# UI-Vision / MMBench across desktop + browser + mobile uniformly. This regex
# key registers the same grounding surface under all three platforms.
