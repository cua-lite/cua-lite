"""
STEP-GUI Action Space

Wraps actions in a single ``mobile_use`` tool with an ``action`` parameter,
matching the STEP-GUI mobile model's tab-separated action format.

Reference: gelab-zero/copilot_tools/parser_0920_summary.py (Parser0920Summary)

Key characteristics:

- **Coordinate system: absolute [0, 1000]** — same as cua-lite, no conversion
  needed.
- **Actions**: CLICK, TYPE, COMPLETE, WAIT, AWAKE, INFO, ABORT, SLIDE, LONGPRESS.
- **COMPLETE** signals task success and reports a result (maps to cua-lite
  ``terminate(status="success")``).
- **ABORT** signals task failure (maps to cua-lite ``terminate(status="failure")``).
- **INFO** submits a final answer (maps to cua-lite ``response``).
- **AWAKE** opens/wakes an app (maps to cua-lite ``open_app``).

Usage:
    from lite.agents.models.step_gui.action_space import STEPGUIMobileActionSpace

    space = STEPGUIMobileActionSpace()
    tools = space.get_tool_schemas()
    tc = STEPGUIMobileActionSpace.mobile_use(action="CLICK", point=[500, 500])
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Collection
from typing import Any, Literal

from lite.agents.core.action_space.base import (
    ActionSpaceRegistry,
    BaseActionSpace,
    LiteMobileActionSpace,
)
from lite.agents.core.action_space.utils.geometry import (
    compact_number,
    required_coord,
)
from lite.agents.core.action_space.utils.unknown_wrapper_action import unknown_wrapper_action_batch
from lite.agents.core.action_space.utils.wrapper_enum import filter_wrapper_action_enum
from lite.core.tools.action_space import (
    LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
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

#: The one provider-native tool STEP-GUI exposes; its ``action`` parameter
#: carries every STEP-GUI action value.
_STEP_GUI_NATIVE_TOOL_NAME = "mobile_use"


def _step_gui_action_values(cls) -> frozenset[str]:
    """Every STEP-GUI ``action`` value ``cls``'s from-agent dispatch understands.

    The wrapped GUI values plus the native spellings of standalone canonical
    tools — i.e. the single ``Literal[...]`` enum the model was trained on.
    Read when provider output drops the wrapper and uses an action value as the
    tool name, so that flat output converts through the same branches as nested
    output.
    """
    return frozenset(
        action_value
        for action_values in cls.LITE_ACTION_NAME_TO_STEP_GUI_ACTION_VALUES.values()
        for action_value in action_values
    ) | frozenset(cls.STEP_GUI_ACTION_VALUE_TO_EXTRA_TOOL_NAMES)


# =============================================================================
# Action Space
# =============================================================================

@dataclasses.dataclass
class STEPGUIMobileActionSpace(BaseActionSpace, key="step_gui@mobile"):
    """
    STEP-GUI-style action space for mobile interactions.

    Uses a single ``mobile_use`` tool with an ``action`` parameter, matching
    the STEP-GUI tab-separated format (parser_0920_summary.py).

    Coordinate system: absolute [0, 1000] — same as cua-lite, no conversion.

    Actions:
        CLICK, LONGPRESS, TYPE, SLIDE, AWAKE, WAIT, INFO, COMPLETE, ABORT

    Example:
        space = STEPGUIMobileActionSpace()
        tc = STEPGUIMobileActionSpace.mobile_use(action="CLICK", point=[500, 500])
    """

    platform: str = "mobile"

    _ACTION_DESCRIPTION = """
1. CLICK：点击手机屏幕坐标，需包含点击的坐标位置 point。
例如：action:CLICK\tpoint:x,y
2. TYPE：在手机输入框中输入文字，需包含输入内容 value、输入框的位置 point。
例如：action:TYPE\tvalue:输入内容\tpoint:x,y
3. COMPLETE：任务完成后向用户报告结果，需包含报告的内容 return_value。
例如：action:COMPLETE\treturn:完成任务后向用户报告的内容
4. WAIT：等待指定时长，需包含等待时间 value（秒）。
例如：action:WAIT\tvalue:等待时间
5. AWAKE：唤醒指定应用，需包含唤醒的应用名称 value。
例如：action:AWAKE\tvalue:应用名称
6. INFO：向用户提交最终回答，需包含回答内容 value。
例如：action:INFO\tvalue:最终回答内容
7. ABORT：终止当前任务，仅在当前任务无法继续执行时使用，需包含 value 说明原因。
例如：action:ABORT\tvalue:终止任务的原因
8. SLIDE：在手机屏幕上滑动，滑动的方向不限，需包含起点 point1 和终点 point2。
例如：action:SLIDE\tpoint1:x1,y1\tpoint2:x2,y2
9. LONGPRESS：长按手机屏幕坐标，需包含长按的坐标位置 point。
例如：action:LONGPRESS\tpoint:x,y
"""

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        point="(x, y): coordinate for CLICK, LONGPRESS, TYPE. Range [0, 1000].",
        point1="(x, y): slide start coordinate. Required only by `action=SLIDE`.",
        point2="(x, y): slide end coordinate. Required only by `action=SLIDE`.",
        value=(
            "Text for TYPE, wait seconds for WAIT, app name for AWAKE, final answer "
            "for INFO, reason for ABORT."
        ),
        return_value="Result text to report. Required only by `action=COMPLETE`.",
    )
    def mobile_use(
        action: Literal[
            "CLICK", "LONGPRESS", "TYPE", "SLIDE",
            "AWAKE", "WAIT", "INFO", "COMPLETE", "ABORT",
        ],
        value: str | None = None,
        point: list[int] | None = None,
        point1: list[int] | None = None,
        point2: list[int] | None = None,
        return_value: str | None = None,
    ) -> dict[str, Any]:
        """
        Use a touchscreen to interact with a mobile device, and take screenshots.

        * This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
        * The screen's coordinate range is [0, 1000] with origin at the top-left corner, x-axis to the right, y-axis downward.
        * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element.
        """
        # Field order matches the STEP_GUI_MOBILE_SYS_PROMPT example lines,
        # e.g. TYPE is `action:TYPE\tvalue:...\tpoint:x,y` (value BEFORE point).
        return make_tool_call(
            _STEP_GUI_NATIVE_TOOL_NAME,
            {
                "action": action,
                "value": value,
                "point": point,
                "point1": point1,
                "point2": point2,
                "return_value": return_value,
            },
        )

    # -------------------------------------------------------------------------
    # Action-value tables
    # -------------------------------------------------------------------------
    # STEP-GUI packs BOTH layers into one ``Literal[...]`` on one ``@tool``:
    # ``WAIT`` (an action) and ``ABORT`` (a finish tool) are two strings in the
    # same enum, so nothing class-visible separates them. Declared here.
    #
    # The extra-tool names are the ones ``_convert_single_from_agent`` PRODUCES.

    LITE_ACTION_NAME_TO_STEP_GUI_ACTION_VALUES = {
        "tap": ["CLICK"],
        "long_press": ["LONGPRESS"],
        "type": ["TYPE"],
        "swipe": ["SLIDE"],
        "wait": ["WAIT"],
    }
    #: ``COMPLETE`` and ``ABORT`` both parse to ``terminate`` (success vs
    #: failure status); ``AWAKE`` is the app-launch channel and ``INFO`` the
    #: answer channel.
    STEP_GUI_ACTION_VALUE_TO_EXTRA_TOOL_NAMES = {
        "AWAKE": frozenset({"open_app"}),
        "INFO": frozenset({"response"}),
        "COMPLETE": frozenset({"terminate"}),
        "ABORT": frozenset({"terminate"}),
    }

    # -------------------------------------------------------------------------
    # Action-value gate
    # -------------------------------------------------------------------------
    # ``valid_actions`` is the one fact that trims the ``mobile_use`` schema's
    # action values. The sample's active extra tools are NOT a second gate here:
    # Step-GUI's model-visible action vocabulary is the numbered action rows of
    # its system prompt (``STEPGUIMobileBaseAdapter``), SFT-trained text that is
    # always advertised whole, so there is nothing for an active-extras gate to
    # trim. Env ingress answers a call the current surface does not admit.

    @classmethod
    def step_gui_action_value_allowed_by_valid_actions(
        cls,
        action_value: str,
        valid_actions: Collection[str],
    ) -> bool:
        """True when ``valid_actions`` still reaches ``action_value``.

        ``valid_actions`` lists canonical GUI action names and never names an
        extra tool, so action values that spell one (``AWAKE``, ``INFO``,
        ``COMPLETE``, ``ABORT``) are out of this gate's scope and always pass.
        """
        if action_value in cls.STEP_GUI_ACTION_VALUE_TO_EXTRA_TOOL_NAMES:
            return True
        return any(
            action_value in (cls.LITE_ACTION_NAME_TO_STEP_GUI_ACTION_VALUES.get(name) or ())
            for name in valid_actions
        )

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        return filter_wrapper_action_enum(
            schemas,
            wrapper_tool_name=_STEP_GUI_NATIVE_TOOL_NAME,
            keep_action_value=lambda value: (
                cls.step_gui_action_value_allowed_by_valid_actions(value, valid_actions)
            ),
        )

    # -------------------------------------------------------------------------
    # Tool call conversion: cua-lite -> STEP-GUI
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
            # The enum has ``CLICK`` and no double-tap counterpart, so ``clicks``
            # has no carrier. Rendering ``tap(clicks=2)`` as a plain ``CLICK``
            # re-parses as ``tap(clicks=1)``: a SINGLE tap where a double tap was
            # asked for, which is a different gesture rather than a coarser one,
            # and one the caller cannot recover -- every retry degrades
            # identically. Raise rather than silently substitute, as
            # ``fara@desktop`` does for the same ``clicks`` loss on ``click``.
            clicks = args.get("clicks", 1)
            if clicks != 1:
                raise ValueError(
                    f"STEP-GUI cannot render tap(clicks={clicks}): its action enum "
                    "has only 'CLICK' and the wire carries no repeat count"
                )
            return [STEPGUIMobileActionSpace.mobile_use(
                action="CLICK",
                point=args.get("coordinate"),
            )["function"]]

        if name == "long_press":
            return [STEPGUIMobileActionSpace.mobile_use(
                action="LONGPRESS",
                point=args.get("coordinate"),
            )["function"]]

        if name == "type":
            # Reference `action2action` omits `point` for TYPE, so the SFT
            # distribution never saw it. We omit it here too to match.
            return [STEPGUIMobileActionSpace.mobile_use(
                action="TYPE",
                value=args.get("text", ""),
            )["function"]]

        if name == "swipe":
            return [STEPGUIMobileActionSpace.mobile_use(
                action="SLIDE",
                point1=args.get("start_coordinate"),
                point2=args.get("coordinate"),
            )["function"]]

        if name == "open_app":
            return [STEPGUIMobileActionSpace.mobile_use(
                action="AWAKE",
                value=args.get("app_name", ""),
            )["function"]]

        if name == "response":
            return [STEPGUIMobileActionSpace.mobile_use(
                action="INFO",
                value=args.get("text", ""),
            )["function"]]

        if name == "wait":
            # Reference emits `value:5` (integer-looking string), not `value:5.0`.
            # `compact_number` preserves integer form when duration has no
            # fractional part; str() turns everything into the wire form.
            duration = compact_number(args.get("duration"))
            value = str(duration) if duration is not None else None
            return [STEPGUIMobileActionSpace.mobile_use(
                action="WAIT",
                value=value,
            )["function"]]

        if name == "terminate":
            status = args.get("status", "success")
            reason = args.get("reason", "")
            if status == "success":
                return [STEPGUIMobileActionSpace.mobile_use(
                    action="COMPLETE",
                    return_value=reason or "",
                )["function"]]
            else:
                # Reference `action2action` keeps nothing beyond base for ABORT
                # (parser_0920_summary.py:188 `pass`). `value` from the input is
                # discarded and never re-emitted. We match by omitting it.
                return [STEPGUIMobileActionSpace.mobile_use(action="ABORT")["function"]]

        if name in LiteMobileActionSpace.get_action_names():
            # A canonical MOBILE action with no branch above has no STEP-GUI enum
            # entry: the enum is {CLICK, LONGPRESS, TYPE, SLIDE, AWAKE, WAIT,
            # INFO, COMPLETE, ABORT}, which spells no ``drag`` (``SLIDE`` parses
            # back as ``swipe``), no ``pinch``, no ``screenshot`` and NO system
            # button at all — not even ``Back``. Warning and returning ``[]``
            # deleted the action from the wire, so a canonical ``Back`` press
            # vanished with only a log line to show for it.
            raise ValueError(
                f"STEP-GUI cannot render canonical tool {name!r}: its action enum "
                "has no counterpart, and dropping it silently loses the action"
            )

        logger.warning("Unknown cua-lite mobile action for STEP-GUI: %s(%s)", name, args)
        return []

    # -------------------------------------------------------------------------
    # Tool call conversion: STEP-GUI -> cua-lite
    # -------------------------------------------------------------------------

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert STEP-GUI mobile_use tool_calls back to cua-lite format.

        ``summary`` is **not** stashed onto cua-lite ``arguments`` — it lives
        as :class:`HistorySummaryContent` on the assistant message (the
        canonical home for trajectory rolling-summary metadata, set by
        ``STEPGUIMobileBaseAdapter.parse_raw_assistant_response``).

        Following reference ``action2action`` (parser_0920_summary.py:107-213),
        TYPE drops ``point`` and ABORT drops ``value`` — both are lost on this
        edge (never reach cua-lite canonical), matching the SFT distribution.
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

        if name != _STEP_GUI_NATIVE_TOOL_NAME:
            admitted = extra_tool_name_and_arguments_are_admitted(
                name,
                args,
                active_extra_tool_names=active_extra_tool_names,
                active_extra_tool_schemas=active_extra_tool_schemas,
            )
            # The model dropped the ``mobile_use`` wrapper and used the native
            # action value AS the tool name, e.g. ``CLICK(point=[x, y])``. Read it
            # as that action so flat output converts through the same branches as
            # nested output — ``AWAKE``/``INFO``/``COMPLETE``/``ABORT`` included.
            # An ACTIVE env extra outranks a colliding native value: an env tool
            # spelled like an action value is the env's, not this space's.
            if not action and not admitted and name in _step_gui_action_values(type(self)):
                action = name
            elif not action or admitted:
                # A standalone tool (env extra, custom tool): its Lite name IS its
                # wire name.
                return [make_tool_call(name, args)]
            # Otherwise ``action`` is set under a non-wrapper name: the model kept
            # the native nesting but got the tool name wrong; the dispatch claims it.

        if action == "CLICK":
            return [LiteMobileActionSpace.tap(
                coordinate=required_coord(args.get("point"), dimensions=2, name="point"),
            )]

        if action == "LONGPRESS":
            return [LiteMobileActionSpace.long_press(
                coordinate=required_coord(args.get("point"), dimensions=2, name="point"),
            )]

        if action == "TYPE":
            return [LiteMobileActionSpace.type(
                text=args.get("value", ""),
            )]

        if action == "SLIDE":
            return [LiteMobileActionSpace.swipe(
                start_coordinate=required_coord(
                    args.get("point1"),
                    dimensions=2,
                    name="point1",
                ),
                coordinate=required_coord(args.get("point2"), dimensions=2, name="point2"),
            )]

        if action == "AWAKE":
            return [make_tool_call("open_app", {"app_name": args.get("value", "")})]

        if action == "INFO":
            return [LiteFinishToolSet.response(text=args.get("value", ""))]

        if action == "WAIT":
            # The wire format is text (``action:WAIT\tvalue:5``) and an omitted
            # ``value`` never reaches here, so the only malformed duration the
            # MODEL can emit is a non-numeric string -> ValueError, absorbed
            # into the reference parser's 1s default. A non-str ``value`` can
            # only come from a hand-built arguments dict, i.e. caller breakage,
            # and must stay loud instead of silently becoming a 1s wait.
            raw = args.get("value", "1")
            try:
                duration = float(raw)
            except ValueError:
                duration = 1.0
            return [LiteMobileActionSpace.wait(duration=duration)]

        if action == "COMPLETE":
            return_text = args.get("return_value", "")
            return [LiteFinishToolSet.terminate(
                status="success",
                reason=return_text if return_text else None,
            )]

        if action == "ABORT":
            return [LiteFinishToolSet.terminate(
                status="failure",
                reason=args.get("value", ""),
            )]

        logger.warning("Unknown STEP-GUI action: %s(%s)", action, args)
        return [unknown_wrapper_action_batch(LITE_MOBILE_ACTION_BATCH_TOOL_NAME, args)]


# Ensure registry import side-effects
ActionSpaceRegistry.register("step_gui@mobile", STEPGUIMobileActionSpace)
