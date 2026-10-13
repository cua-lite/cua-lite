"""
Qwen2.5-VL Action Spaces

Wire format: single ``computer_use`` / ``mobile_use`` tool with an ``action``
parameter — mirrors the official Qwen2.5-VL ``cookbooks/utils/agent_function_call.py``
(11-action desktop enum, 9-action mobile enum).

The schema description contains the placeholder ``{display_width_px}x{display_height_px}``;
the adapter substitutes the actual resized image dims at render time so each
turn declares the right screen resolution to the model. Coordinates ARE the
resized-image pixels — no normalization here. The adapter rescales between
cua-lite [0, 1000] and pixel-in-resized.

Usage::

    from lite.agents.models.qwen2_5_vl.action_space import (
        Qwen2_5VLDesktopActionSpace,
        Qwen2_5VLDesktopGroundingPointActionSpace,
        Qwen2_5VLMobileActionSpace,
        Qwen2_5VLMobileGroundingPointActionSpace,
    )
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Collection
from typing import Any, ClassVar, Literal

from lite.agents.core.action_space.base import (
    BaseActionSpace,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.action_space.utils.geometry import (
    PIXELS_PER_CLICK,
    compact_number,
    optional_coord,
    required_coord,
    required_model_text,
    required_scroll_pixels,
    scroll_clicks,
    scroll_wire_magnitude,
)
from lite.agents.core.action_space.utils.grounding_point import (
    convert_non_point_call_for_grounding_space,
)
from lite.agents.core.action_space.utils.unknown_wrapper_action import unknown_wrapper_action_batch
from lite.agents.core.action_space.utils.wrapper_enum import filter_wrapper_action_enum
from lite.core.tools.action_space import (
    LITE_COMPUTER_ACTION_BATCH_TOOL_NAME,
    LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
    merge_adjacent_lite_action_batches,
)
from lite.core.tools.action_space.duration import duration_description
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

# Qwen2.5-VL outputs scroll in screen pixels; Lite scroll uses wheel clicks. The
# conversion and the raw-notch cutoff are the shared cross-family convention
# owned by ``lite.agents.core.action_space.utils``.


# --- Qwen wrapper ``action`` enum gates --------------------------------------
#
# The mechanical surgery (enum pruning, bullet-line pruning, "Required only by
# `action=...`" prose rewriting) is owned by ``filter_wrapper_action_enum``.
# What stays here is Qwen policy: which action values a ``valid_actions`` gate
# or an active-extra-tool gate leaves reachable. Each function is bound as a
# classmethod on the action spaces below, so ``cls`` reads that class's own
# tables.


def _filter_qwen_action_values_for_valid_actions(
    cls,
    schemas: list[dict[str, Any]],
    valid_actions: list[str],
) -> list[dict[str, Any]]:
    """Keep the Qwen action values ``valid_actions`` can reach.

    Action values in :attr:`QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES` spell
    standalone canonical tools rather than GUI actions, so this GUI-only gate
    never drops them; whether they render is decided by
    :func:`_filter_qwen_action_values_for_active_extra_tools` instead.
    """
    allowed = {
        action_value
        for name in valid_actions
        for action_value in cls.LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES.get(name) or ()
    }
    extra_tool_action_values = cls.QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES
    return filter_wrapper_action_enum(
        schemas,
        wrapper_tool_name=cls._QWEN_NATIVE_TOOL_NAME,
        keep_action_value=(
            lambda action_value: action_value in allowed
            or action_value in extra_tool_action_values
        ),
    )


def _qwen_action_values(cls) -> frozenset[str]:
    """Every Qwen action value ``cls``'s from-agent dispatch understands.

    The wrapped GUI values, the native spellings of standalone canonical tools,
    and the parse-only values in :attr:`QWEN_PARSE_ONLY_ACTION_VALUES` — i.e.
    exactly what the model can put in the wrapper's ``action`` field. Read when
    provider output drops the wrapper and uses an action value as the tool name,
    so that flat output converts through the same branches as nested output.
    """
    return frozenset(
        action_value
        for action_values in cls.LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES.values()
        for action_value in action_values
    ) | frozenset(cls.QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES) | cls.QWEN_PARSE_ONLY_ACTION_VALUES


def _filter_qwen_grounding_action_values_for_valid_actions(
    cls,
    schemas: list[dict[str, Any]],
    valid_actions: list[str],
) -> list[dict[str, Any]]:
    """``valid_actions`` gate for a trimmed grounding wrapper enum.

    A grounding enum carries one click action value and declares no action
    value that spells a standalone canonical tool, so reachability from
    ``valid_actions`` is the whole policy - refusal is the env's standalone
    ``report_infeasible`` schema.
    """
    allowed = {
        action_value
        for name in valid_actions
        for action_value in cls.LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES.get(name) or ()
    }
    return filter_wrapper_action_enum(
        schemas,
        wrapper_tool_name=cls._QWEN_NATIVE_TOOL_NAME,
        keep_action_value=allowed.__contains__,
    )


def _filter_qwen_action_values_for_active_extra_tools(
    cls,
    schemas: list[dict[str, Any]],
    active_extra_tool_names: Collection[str],
) -> list[dict[str, Any]]:
    """Keep a Qwen action value that spells a standalone canonical tool only
    while at least one of the extra tools it stands for is active.

    ``valid_actions=[]`` + ``extra_tools=["terminate"]`` still leaves the agent
    ``computer_use(action="terminate")`` to end the episode with.
    """
    active = set(active_extra_tool_names)
    inactive = {
        action_value
        for action_value, extra_tool_names
        in cls.QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES.items()
        if extra_tool_names.isdisjoint(active)
    }
    if not inactive:
        return list(schemas)
    return filter_wrapper_action_enum(
        schemas,
        wrapper_tool_name=cls._QWEN_NATIVE_TOOL_NAME,
        keep_action_value=lambda action_value: action_value not in inactive,
    )


# =============================================================================
# Qwen2.5-VL Desktop Action Space (full)
# =============================================================================

@dataclasses.dataclass
class Qwen2_5VLDesktopActionSpace(
    BaseActionSpace,
    key=r"qwen2_5_vl@(desktop|browser)",
):
    """Qwen2.5-VL desktop action space — single ``computer_use`` tool.

    Matches the upstream
    ``Qwen2.5-VL/cookbooks/utils/agent_function_call.py`` :class:`ComputerUse`
    byte-for-byte: 11 actions (no ``triple_click`` / ``hscroll`` / ``answer``),
    and the screen-resolution sentence is templated against
    ``{display_width_px}x{display_height_px}`` — the adapter substitutes
    the resized image dims per render.

    Coordinate space is **pixels in the smart-resized image**. cua-lite's
    [0, 1000] normalized coords are rescaled by the adapter (this layer is
    identity on coords).
    """


    #: No Qwen2.5-VL rollouts on this host to measure; keeps the screen-unit
    #: spelling of the Qwen3-VL prompt family it shares.
    SCROLL_WIRE_UNIT: ClassVar[int] = PIXELS_PER_CLICK

    platform: str = "desktop"
    _QWEN_NATIVE_TOOL_NAME = "computer_use"

    _ACTION_DESCRIPTION = (
        "The action to perform. The available actions are:\n"
        "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n"
        "* `type`: Type a string of text on the keyboard.\n"
        "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
        "* `left_click`: Click the left mouse button.\n"
        "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n"
        "* `right_click`: Click the right mouse button.\n"
        "* `middle_click`: Click the middle mouse button.\n"
        "* `double_click`: Double-click the left mouse button.\n"
        "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
        "* `wait`: Wait specified seconds for the change to happen.\n"
        "* `terminate`: Terminate the current task and report its completion status."
    )

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        keys="Required only by `action=key`.",
        text="Required only by `action=type`.",
        coordinate="(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
        pixels="The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
        time=duration_description(
            "The seconds to wait. Required only by `action=wait`."
        ),
        status="The status of the task. Required only by `action=terminate`.",
    )
    def computer_use(
        action: Literal[
            "key", "type", "mouse_move", "left_click", "left_click_drag",
            "right_click", "middle_click", "double_click",
            "scroll", "wait", "terminate",
        ],
        keys: list[str] | None = None,
        text: str | None = None,
        coordinate: list[int] | None = None,
        pixels: int | None = None,
        time: float | None = None,
        status: Literal["success", "failure"] | None = None,
    ) -> dict[str, Any]:
        """Use a mouse and keyboard to interact with a computer, and take screenshots.

        * This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
        * The screen's resolution is {display_width_px}x{display_height_px}.
        * Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
        * If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
        * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
        """
        return make_tool_call(
            "computer_use",
            {
                "action": action,
                "keys": keys,
                "text": text,
                "coordinate": coordinate,
                "pixels": pixels,
                "time": time,
                "status": status,
            },
        )

    # -------------------------------------------------------------------------
    # Action filtering
    # -------------------------------------------------------------------------

    # CUA-Lite GUI name → Qwen2.5-VL action values it maps to.
    # No ``triple_click`` / ``hscroll`` / ``answer`` — these are out of the
    # trained enum. ``terminate`` is native qwen wire grammar; it is gated by
    # ``extra_tool_schemas``, never by ``valid_actions`` (see
    # ``QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES`` below).
    LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {
        "click": ["left_click", "right_click", "middle_click", "double_click"],
        "drag": ["left_click_drag"],
        "type": ["type"], "key": ["key"], "mouse_move": ["mouse_move"],
        "scroll": ["scroll"], "wait": ["wait"],
    }
    QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES = {"terminate": frozenset({"terminate"})}
    # Action values the from-agent dispatch understands but the schema never
    # advertises, so they belong in neither table above. The optional
    # thought-format system prompt names ``answer`` as the question-answering
    # finish action, so provider output carries it even though the upstream
    # ``computer_use`` enum omits it.
    QWEN_PARSE_ONLY_ACTION_VALUES = frozenset({"answer"})

    filter_tool_schemas_for_valid_actions = classmethod(
        _filter_qwen_action_values_for_valid_actions
    )
    # Qwen owns this gate: which provider action values a sample's active extra
    # tools leave reachable is family policy, so the Qwen2.5-VL adapter's tools
    # section calls it.
    filter_qwen_action_values_for_active_extra_tools = classmethod(
        _filter_qwen_action_values_for_active_extra_tools
    )

    # -------------------------------------------------------------------------
    # Tool Call Conversion (cua-lite ↔ agent format; coords pass through;
    # the adapter handles [0,1000] ↔ pixel rescaling).
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        results = []

        if name == "computer":
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                results.extend(self._convert_single_to_agent(make_tool_call(action, action_args)))

        elif name == "click":
            c = args.get("coordinate")
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)
            if button == "right":
                results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                    action="right_click",
                    coordinate=c,
                )["function"])
            elif button == "middle":
                results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                    action="middle_click",
                    coordinate=c,
                )["function"])
            elif clicks == 2:
                results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                    action="double_click",
                    coordinate=c,
                )["function"])
            elif clicks == 3:
                # No triple_click in Qwen2.5-VL enum → emit double + single at the same coord.
                results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                    action="double_click",
                    coordinate=c,
                )["function"])
                results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                    action="left_click",
                    coordinate=c,
                )["function"])
            else:
                results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                    action="left_click",
                    coordinate=c,
                )["function"])

        elif name == "type":
            results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                action="type",
                text=args.get("text", ""),
            )["function"])

        elif name == "key":
            results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                action="key",
                keys=args.get("keys", []),
            )["function"])

        elif name == "scroll":
            direction = args.get("direction", "down")
            amount = args.get("amount", 3)
            if direction not in ("up", "down"):
                # The trained enum has no ``hscroll``, so the signed scalar
                # ``pixels`` is the only scroll carrier and its single axis is
                # vertical -- the parse side reads the sign back as ``up``/``down``
                # and nothing else. There is no faithful spelling of a horizontal
                # scroll, so raise rather than silently substitute the vertical axis.
                raise ValueError(
                    f"Qwen2.5-VL desktop cannot render scroll(direction={direction!r}): "
                    "its action enum has no 'hscroll' and 'pixels' carries the "
                    "vertical axis only"
                )
            # ``int(round(...))``: the schema type is integer, so a float
            # ``amount`` from source data must not leak a float onto the wire.
            scroll_pixels = scroll_wire_magnitude(int(round(amount)),
                                                  wire_unit=type(self).SCROLL_WIRE_UNIT)
            if direction == "down":
                scroll_pixels = -scroll_pixels
            c = args.get("coordinate")
            results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                action="scroll",
                pixels=scroll_pixels,
                coordinate=c,
            )["function"])

        elif name == "drag":
            if args.get("start_coordinate"):
                results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                    action="mouse_move",
                    coordinate=args["start_coordinate"],
                )["function"])
            c = args.get("coordinate")
            if c:
                results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                    action="left_click_drag",
                    coordinate=c,
                )["function"])

        elif name == "mouse_move":
            c = args.get("coordinate")
            if c:
                results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                    action="mouse_move",
                    coordinate=c,
                )["function"])

        elif name == "wait":
            results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                action="wait",
                time=compact_number(args.get("duration", 3)),
            )["function"])

        elif name == "terminate":
            results.append(Qwen2_5VLDesktopActionSpace.computer_use(
                action="terminate",
                status=args.get("status", "success"),
            )["function"])

        elif name == "response":
            raise ValueError("Qwen2.5-VL desktop cannot render canonical tool 'response'")

        elif name in LiteDesktopActionSpace.get_action_names():
            # A canonical GUI action with NO branch above has no entry in the
            # trained ``computer_use`` enum (``key_down``/``key_up``/``hold_key``/
            # ``mouse_down``/``mouse_up``/``screenshot``/``cursor_position``).
            # Warning and returning ``[]`` deleted the action from the wire, so
            # the round trip came back EMPTY and the caller could not tell.
            raise ValueError(
                f"Qwen2.5-VL desktop cannot render canonical tool {name!r}: its "
                "action enum has no counterpart, and dropping it silently loses "
                "the action"
            )

        else:
            logger.warning("Unsupported CUA-lite action for Qwen2.5-VL: %s(%s)", name, args)

        return results

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Parse ``computer_use`` calls back to canonical Lite calls.

        Unknown-action policy: a ``computer_use`` action this family cannot
        express becomes an INVALID ``computer`` action-batch naming the model's
        own action value, so env ingress rejects it and the model sees the
        rejection. Non-wrapper tool names stay by-name calls for env ingress to
        judge. Nothing is dropped.
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

        if name != "computer_use":
            admitted = extra_tool_name_and_arguments_are_admitted(
                name,
                args,
                active_extra_tool_names=active_extra_tool_names,
                active_extra_tool_schemas=active_extra_tool_schemas,
            )
            # The model dropped the ``computer_use`` wrapper and used the native
            # action value AS the tool name, e.g. ``left_click(coordinate=...)``.
            # Read it as that action so flat output converts through the same
            # branches as nested output — ``answer`` and ``terminate`` included.
            # An ACTIVE env extra outranks a colliding native value: browsergym's
            # ``scroll(delta_y=...)`` is the env's tool, not the native ``scroll``
            # the dispatch would read as a pixel-carrying wheel action.
            if not action and not admitted and name in _qwen_action_values(type(self)):
                action = name
            elif not action or admitted:
                # A standalone tool: env extra (browser nav ``goto``/``back``) or a
                # custom ``summarize(text=...)``. Its Lite name IS its wire name.
                # Do NOT infer ``answer`` from a "text" arg — that silently
                # collapsed text-bearing extra tools into ``response``.
                return [make_tool_call(name, args)]
            # Otherwise ``action`` is set under a non-wrapper name: the model kept
            # the native nesting but got the tool name wrong; the dispatch claims it.

        if action == "left_click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.click(coordinate=coordinate)]
        if action == "right_click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.click(coordinate=coordinate, button="right")]
        if action == "middle_click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.click(coordinate=coordinate, button="middle")]
        if action == "double_click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.click(coordinate=coordinate, clicks=2)]
        if action == "type":
            return [LiteDesktopActionSpace.type(text=required_model_text(args, action=action))]
        if action == "key":
            raw_keys = args.get("keys", [])
            return [LiteDesktopActionSpace.key(keys=raw_keys)]
        if action == "scroll":
            # ``pixels`` is REQUIRED — it is the only carrier of the direction.
            # See ``required_scroll_pixels`` for why a default is never right.
            scroll_pixels = required_scroll_pixels(args, action)
            direction = "down" if scroll_pixels < 0 else "up"
            amount = scroll_clicks(scroll_pixels)
            return [LiteDesktopActionSpace.scroll(
                direction=direction, amount=amount,
                coordinate=optional_coord(args.get("coordinate"), dimensions=2),
            )]
        if action == "left_click_drag":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.drag(coordinate=coordinate)]
        if action == "mouse_move":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.mouse_move(coordinate=coordinate)]
        if action == "wait":
            return [LiteDesktopActionSpace.wait(duration=float(args.get("time", 3)))]
        if action == "answer":
            # Parse-side only. The upstream enum does not advertise ``answer``,
            # but the optional thought-format system prompt names it as the
            # question-answering finish action. Preserve it as canonical
            # ``response`` instead of dropping the model's terminal intent.
            return [LiteFinishToolSet.response(text=args.get("text", ""))]
        if action == "terminate":
            return [LiteFinishToolSet.terminate(status=args.get("status", "success"))]

        logger.warning("Unknown Qwen2.5-VL desktop wrapper action: %s(%s)", action, args)
        return [unknown_wrapper_action_batch(LITE_COMPUTER_ACTION_BATCH_TOOL_NAME, args)]

# =============================================================================
# Qwen2.5-VL Desktop Grounding (point) — trimmed single-step click
# =============================================================================

@dataclasses.dataclass
class Qwen2_5VLDesktopGroundingPointActionSpace(
    BaseActionSpace,
    key=r"qwen2_5_vl@(desktop|browser)@point",
):
    """Trimmed ``computer_use`` schema: ``left_click`` only (grounding eval).

    Mirrors :class:`Qwen3VLDesktopGroundingPointActionSpace` but pulls in
    the Qwen2.5-VL screen-resolution wording and resized-pixel coord
    convention. The adapter rescales between cua-lite [0, 1000] and pixel.
    """

    platform: str = "desktop"
    _QWEN_NATIVE_TOOL_NAME = "computer_use"

    _ACTION_DESCRIPTION = (
        "The action to perform. The available actions are:\n"
        "* `left_click`: Click the left mouse button at the (x, y) pixel coordinate of the target element on the screen."
    )

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        coordinate="(x, y): The x (pixels from the left edge) and y (pixels from the top edge) pixel coordinate of the target element. Required by `action=left_click`.",
    )
    def computer_use(
        action: Literal["left_click"],
        coordinate: list[int] | None = None,
    ) -> dict[str, Any]:
        """Use a mouse to click on the target element in the screenshot.

        * The screen's resolution is {display_width_px}x{display_height_px}.
        * Click with the cursor tip centered on the target; avoid edges unless the instruction asks for an edge.
        * Do not use other tools. Only ``left_click`` is available.
        """
        return make_tool_call(
            "computer_use",
            {"action": action, "coordinate": coordinate},
        )

    # This trimmed surface declares no Qwen action value that spells a
    # standalone canonical tool, so it has no active-extra gate.
    LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {"point": ["left_click"]}

    filter_tool_schemas_for_valid_actions = classmethod(
        _filter_qwen_grounding_action_values_for_valid_actions
    )

    def convert_tool_calls_to_agent(
        self, tool_calls: list[dict[str, Any]], **kwargs,
    ) -> list[dict[str, Any]]:
        result = []
        for tc in tool_calls:
            name = tool_call_name(tc)
            args = tool_call_arguments(tc)
            if name == "point":
                result.append(
                    Qwen2_5VLDesktopGroundingPointActionSpace.computer_use(
                        action="left_click",
                        coordinate=args.get("coordinate"),
                    )["function"]
                )
            else:
                result.extend(convert_non_point_call_for_grounding_space(
                    tc, surface="Qwen2.5-VL grounding (point) action_space",
                ))
        return result

    def convert_tool_calls_from_agent(
        self, agent_tool_calls: list[dict[str, Any]], **kwargs,
    ) -> list[dict[str, Any]]:
        result = []
        for tc in agent_tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            action = args.get("action", "")
            if name != "computer_use":
                # Pass-through extras (e.g. ``report_infeasible``).
                result.append(make_tool_call(name, args))
                continue
            if action == "left_click":
                coordinate = required_coord(args.get("coordinate"), dimensions=2)
                result.append(LitePointActionSpace.point(
                    coordinate=coordinate,
                ))
            else:
                logger.warning(
                    "Qwen2.5-VL grounding (point) dropping off-schema action %s(%s)",
                    action, args,
                )
        return result


# =============================================================================
# Qwen2.5-VL Mobile Action Space (full)
# =============================================================================

# adb ``keyevent`` names that a cua-lite ``system_button`` can express. The
# native ``mobile_use(action="key", text=...)`` channel is kept upstream-faithful
# (see the class docstring), but cua-lite mobile has no ``key`` action, so
# only these round-trip; the rest are dropped (see ``_convert_single_from_agent``).
_ADB_KEYEVENT_TO_SYSTEM_BUTTON = {
    "back": "Back",
    "home": "Home",
    "menu": "Menu",
    "enter": "Enter",
    "dpad_center": "Enter",
    "app_switch": "Recent",
    "recent": "Recent",
    "recents": "Recent",
}


@dataclasses.dataclass
class Qwen2_5VLMobileActionSpace(
    BaseActionSpace,
    key="qwen2_5_vl@mobile",
):
    """Qwen2.5-VL mobile action space — single ``mobile_use`` tool.

    Matches upstream :class:`MobileUse`: 9 actions
    (``key``, ``click``, ``long_press``, ``swipe``, ``type``,
    ``system_button``, ``open``, ``wait``, ``terminate``). Coordinate
    space is **pixels in the smart-resized image**; rescaling lives in
    the adapter.
    """

    platform: str = "mobile"
    _QWEN_NATIVE_TOOL_NAME = "mobile_use"

    _ACTION_DESCRIPTION = (
        "The action to perform. The available actions are:\n"
        "* `key`: Perform a key event on the mobile device.\n"
        "    - This supports adb's `keyevent` syntax.\n"
        "    - Examples: \"volume_up\", \"volume_down\", \"power\", \"camera\", \"clear\".\n"
        "* `click`: Click the point on the screen with coordinate (x, y).\n"
        "* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n"
        "* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n"
        "* `type`: Input the specified text into the activated input box.\n"
        "* `system_button`: Press the system button.\n"
        "* `open`: Open an app on the device.\n"
        "* `wait`: Wait specified seconds for the change to happen.\n"
        "* `terminate`: Terminate the current task and report its completion status."
    )

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        coordinate="(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
        coordinate2="(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
        text="Required only by `action=key`, `action=type`, and `action=open`.",
        time=duration_description(
            "The seconds to wait. Required only by `action=long_press` and `action=wait`."
        ),
        button="Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, Enter means pressing the enter, and Recent means opening the recent-apps / task switcher. Required only by `action=system_button`",
        status="The status of the task. Required only by `action=terminate`.",
    )
    def mobile_use(
        action: Literal[
            "key", "click", "long_press", "swipe", "type",
            "system_button", "open", "wait", "terminate",
        ],
        coordinate: list[int] | None = None,
        coordinate2: list[int] | None = None,
        text: str | None = None,
        time: float | None = None,
        button: Literal["Back", "Home", "Menu", "Enter", "Recent"] | None = None,
        status: Literal["success", "failure"] | None = None,
    ) -> dict[str, Any]:
        """Use a touchscreen to interact with a mobile device, and take screenshots.

        * This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
        * The screen's resolution is {display_width_px}x{display_height_px}.
        * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
        """
        return make_tool_call(
            "mobile_use",
            {
                "action": action,
                "coordinate": coordinate,
                "coordinate2": coordinate2,
                "text": text,
                "time": time,
                "button": button,
                "status": status,
            },
        )

    LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {
        "tap": ["click"],
        "long_press": ["long_press"],
        "swipe": ["swipe"],
        "type": ["type"],
        # Canonical MOBILE has NO ``key`` action; the native ``key`` entry is
        # an adb keyevent that ``_convert_single_from_agent`` parses into a
        # canonical ``system_button`` (see below). Declaring ``key`` as its own
        # canonical key would name an action the core mobile set does not
        # carry, and would advertise ``key`` to envs that never allow it.
        "system_button": ["system_button", "key"],
        "wait": ["wait"],
    }
    QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES = {
        "open": frozenset({"open_app"}),
        "terminate": frozenset({"terminate"}),
    }
    # See :class:`Qwen2_5VLDesktopActionSpace` — parse-only, never advertised.
    QWEN_PARSE_ONLY_ACTION_VALUES = frozenset({"answer"})

    filter_tool_schemas_for_valid_actions = classmethod(
        _filter_qwen_action_values_for_valid_actions
    )
    # Qwen owns this gate: which provider action values a sample's active extra
    # tools leave reachable is family policy, so the Qwen2.5-VL adapter's tools
    # section calls it.
    filter_qwen_action_values_for_active_extra_tools = classmethod(
        _filter_qwen_action_values_for_active_extra_tools
    )

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        results = []

        if name == "mobile":
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                results.extend(self._convert_single_to_agent(make_tool_call(action, action_args)))

        elif name == "tap":
            # The trained mobile enum has ``click`` and no double-tap
            # counterpart, so ``clicks`` has no carrier. Rendering
            # ``tap(clicks=2)`` as a plain ``click`` re-parses as
            # ``tap(clicks=1)``: a SINGLE tap where a double tap was asked for,
            # which is a different gesture rather than a coarser one, and one
            # the caller cannot recover -- every retry degrades identically.
            # Raise rather than silently substitute, as ``fara@desktop`` does
            # for the same ``clicks`` loss on ``click``.
            clicks = args.get("clicks", 1)
            if clicks != 1:
                raise ValueError(
                    f"Qwen2.5-VL mobile cannot render tap(clicks={clicks}): its "
                    "action enum has only 'click' and the wire carries no repeat count"
                )
            results.append(Qwen2_5VLMobileActionSpace.mobile_use(
                action="click",
                coordinate=args.get("coordinate"),
            )["function"])
        elif name == "long_press":
            results.append(Qwen2_5VLMobileActionSpace.mobile_use(
                action="long_press",
                coordinate=args.get("coordinate"),
                time=args.get("duration"),
            )["function"])
        elif name == "swipe":
            results.append(Qwen2_5VLMobileActionSpace.mobile_use(
                action="swipe",
                coordinate=args.get("start_coordinate"),
                coordinate2=args.get("coordinate"),
            )["function"])
        elif name == "type":
            results.append(Qwen2_5VLMobileActionSpace.mobile_use(
                action="type",
                text=args.get("text", ""),
            )["function"])
        elif name == "key":
            results.append(Qwen2_5VLMobileActionSpace.mobile_use(
                action="key",
                text=args.get("text", ""),
            )["function"])
        elif name == "system_button":
            results.append(Qwen2_5VLMobileActionSpace.mobile_use(
                action="system_button",
                button=args.get("button"),
            )["function"])
        elif name == "open_app":
            results.append(Qwen2_5VLMobileActionSpace.mobile_use(
                action="open",
                text=args.get("app_name", ""),
            )["function"])
        elif name == "wait":
            results.append(Qwen2_5VLMobileActionSpace.mobile_use(
                action="wait",
                time=compact_number(args.get("duration", 3)),
            )["function"])
        elif name == "terminate":
            results.append(Qwen2_5VLMobileActionSpace.mobile_use(
                action="terminate",
                status=args.get("status", "success"),
            )["function"])
        elif name == "response":
            raise ValueError("Qwen2.5-VL mobile cannot render canonical tool 'response'")
        elif name in LiteMobileActionSpace.get_action_names():
            # A canonical MOBILE action with no branch above has no entry in the
            # trained ``mobile_use`` enum (``drag``/``pinch``/``screenshot``):
            # ``swipe`` is the only two-point gesture and it parses back as
            # ``swipe``, not ``drag``. Warning and returning ``[]`` deleted the
            # action from the wire, so the round trip came back EMPTY.
            raise ValueError(
                f"Qwen2.5-VL mobile cannot render canonical tool {name!r}: its "
                "action enum has no counterpart, and dropping it silently loses "
                "the action"
            )
        else:
            logger.warning("Unsupported CUA-lite mobile action for Qwen2.5-VL: %s(%s)", name, args)

        return results

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Parse ``mobile_use`` calls back to canonical Lite calls.

        Unknown-action policy: a ``mobile_use`` action this family cannot
        express — including a native ``key`` keyevent with no cua-lite mobile
        counterpart — becomes an INVALID ``mobile`` action-batch naming the
        model's own action value, so env ingress rejects it and the model sees
        the rejection. Non-wrapper tool names stay by-name calls for env ingress
        to judge. Nothing is dropped.
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

        if name != "mobile_use":
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
            # ``click(bid=...)`` is the env's tool, not the native ``click`` the
            # dispatch would read as a coordinate tap.
            if not action and not admitted and name in _qwen_action_values(type(self)):
                action = name
            elif not action or admitted:
                # A standalone tool: env extra or a custom ``summarize(text=...)``.
                # Its Lite name IS its wire name. Do NOT infer ``answer`` from a
                # "text" arg — that silently collapsed text-bearing extra tools
                # into ``response``.
                return [make_tool_call(name, args)]
            # Otherwise ``action`` is set under a non-wrapper name: the model kept
            # the native nesting but got the tool name wrong; the dispatch claims it.

        if action == "click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteMobileActionSpace.tap(coordinate=coordinate)]
        if action == "long_press":
            t = args.get("time")
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteMobileActionSpace.long_press(
                coordinate=coordinate,
                duration=float(t) if t is not None else None,
            )]
        if action == "swipe":
            start_coordinate = required_coord(args.get("coordinate"), dimensions=2)
            coordinate = required_coord(
                args.get("coordinate2"),
                dimensions=2,
                name="coordinate2",
            )
            return [LiteMobileActionSpace.swipe(
                start_coordinate=start_coordinate,
                coordinate=coordinate,
            )]
        if action == "type":
            return [LiteMobileActionSpace.type(text=required_model_text(args, action=action))]
        if action == "key":
            # Native ``key`` is an adb ``keyevent`` (upstream examples:
            # "volume_up", "volume_down", "power", "camera", "clear"). cua-lite
            # mobile has NO ``key`` action — the catalog is deliberate — so
            # only the keyevents that name a lite ``system_button`` have an
            # equivalent. Everything else (volume / power / camera / ...) takes
            # the unknown-action path below: an invalid ``key`` action-batch the
            # env rejects by name. It must never raise, because an
            # AttributeError here escapes the env's model-action-error handling
            # and burns the turn as a generic parse error.
            raw = str(args.get("text") or "").strip().lower()
            button = _ADB_KEYEVENT_TO_SYSTEM_BUTTON.get(
                raw[len("keycode_"):] if raw.startswith("keycode_") else raw
            )
            if button is None:
                logger.warning(
                    "Qwen2.5-VL mobile keyevent %r has no cua-lite mobile action",
                    args.get("text"),
                )
                return [unknown_wrapper_action_batch(LITE_MOBILE_ACTION_BATCH_TOOL_NAME, args)]
            return [LiteMobileActionSpace.system_button(button=button)]
        if action == "open":
            return [make_tool_call("open_app", {"app_name": args.get("text", "")})]
        if action == "system_button":
            return [LiteMobileActionSpace.system_button(button=args.get("button", ""))]
        if action == "wait":
            return [LiteMobileActionSpace.wait(duration=float(args.get("time", 3)))]
        if action == "answer":
            # Parse-side only. The upstream enum does not advertise ``answer``,
            # but the optional thought-format system prompt names it as the
            # question-answering finish action. Preserve it as canonical
            # ``response`` instead of dropping the model's terminal intent.
            return [LiteFinishToolSet.response(text=args.get("text", ""))]
        if action == "terminate":
            return [LiteFinishToolSet.terminate(status=args.get("status", "success"))]

        logger.warning("Unknown Qwen2.5-VL mobile wrapper action: %s(%s)", action, args)
        return [unknown_wrapper_action_batch(LITE_MOBILE_ACTION_BATCH_TOOL_NAME, args)]

# =============================================================================
# Qwen2.5-VL Mobile Grounding (point) — trimmed single-step tap
# =============================================================================

@dataclasses.dataclass
class Qwen2_5VLMobileGroundingPointActionSpace(
    BaseActionSpace,
    key="qwen2_5_vl@mobile@point",
):
    """Trimmed ``mobile_use`` schema: ``click`` only (mobile grounding eval)."""

    platform: str = "mobile"
    _QWEN_NATIVE_TOOL_NAME = "mobile_use"

    _ACTION_DESCRIPTION = (
        "The action to perform. The available actions are:\n"
        "* `click`: Click (tap) the point on the screen with coordinate (x, y) of the target element."
    )

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        coordinate="(x, y): The x (pixels from the left edge) and y (pixels from the top edge) pixel coordinate of the target. Required by `action=click`.",
    )
    def mobile_use(
        action: Literal["click"],
        coordinate: list[int] | None = None,
    ) -> dict[str, Any]:
        """Use a touchscreen to tap on the target element in the screenshot.

        * The screen's resolution is {display_width_px}x{display_height_px}.
        * Tap with the touch point centered on the target; avoid edges unless the instruction asks for an edge.
        * Do not use other tools. Only ``click`` is available.
        """
        return make_tool_call(
            "mobile_use",
            {"action": action, "coordinate": coordinate},
        )

    # This trimmed surface declares no Qwen action value that spells a
    # standalone canonical tool, so it has no active-extra gate.
    LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {"point": ["click"]}

    filter_tool_schemas_for_valid_actions = classmethod(
        _filter_qwen_grounding_action_values_for_valid_actions
    )

    def convert_tool_calls_to_agent(
        self, tool_calls: list[dict[str, Any]], **kwargs,
    ) -> list[dict[str, Any]]:
        result = []
        for tc in tool_calls:
            name = tool_call_name(tc)
            args = tool_call_arguments(tc)
            if name == "point":
                result.append(
                    Qwen2_5VLMobileGroundingPointActionSpace.mobile_use(
                        action="click",
                        coordinate=args.get("coordinate"),
                    )["function"]
                )
            else:
                result.extend(convert_non_point_call_for_grounding_space(
                    tc, surface="Qwen2.5-VL mobile grounding (point) action_space",
                ))
        return result

    def convert_tool_calls_from_agent(
        self, agent_tool_calls: list[dict[str, Any]], **kwargs,
    ) -> list[dict[str, Any]]:
        result = []
        for tc in agent_tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            action = args.get("action", "")
            if name != "mobile_use":
                result.append(make_tool_call(name, args))
                continue
            if action == "click":
                coordinate = required_coord(args.get("coordinate"), dimensions=2)
                result.append(LitePointActionSpace.point(
                    coordinate=coordinate,
                ))
            else:
                logger.warning(
                    "Qwen2.5-VL mobile grounding (point) dropping off-schema action %s(%s)",
                    action, args,
                )
        return result
