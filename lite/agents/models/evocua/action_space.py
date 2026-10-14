"""
EvoCUA Action Spaces

Subclasses the Qwen3VL action space with EvoCUA-specific action enum:
adds key_down/key_up, removes hscroll/answer.

- EvoCUADesktopActionSpace: desktop+browser interactions via "computer_use".
  Registered under the regex key ``r"evocua@(desktop|browser)"`` so
  ``ActionSpaceRegistry.get("evocua@browser")`` resolves to the same class.

Usage:
    from lite.agents.models.evocua.action_space import EvoCUADesktopActionSpace

    action_space = EvoCUADesktopActionSpace()
    tools = action_space.get_tool_schemas()
    action = EvoCUADesktopActionSpace.computer_use(action="key_down", keys=["shift"])
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, ClassVar, Literal

from lite.agents.core.action_space.base import LiteDesktopActionSpace
from lite.agents.core.action_space.utils.geometry import (
    compact_number,
    optional_coord,
    required_coord,
    required_model_text,
    required_scroll_pixels,
    scroll_clicks,
    scroll_wire_magnitude,
)
from lite.agents.core.action_space.utils.unknown_wrapper_action import unknown_wrapper_action_batch
from lite.agents.models.qwen3_vl.action_space import (
    Qwen3VLDesktopActionSpace,
    Qwen3VLDesktopGroundingPointActionSpace,
    _qwen_action_values,
)
from lite.core.tools.action_space import LITE_COMPUTER_ACTION_BATCH_TOOL_NAME
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


@dataclasses.dataclass
class EvoCUADesktopActionSpace(Qwen3VLDesktopActionSpace, key=r"evocua@(desktop|browser)"):
    """
    EvoCUA-style action space for desktop interactions.

    Same as Qwen3VL but with key_down/key_up instead of hscroll/answer.

    Actions:
        - left_click, right_click, middle_click, double_click, triple_click
        - type, key, key_down, key_up
        - mouse_move, left_click_drag
        - scroll
        - wait, terminate

    Example:
        action_space = EvoCUADesktopActionSpace()
        tools = action_space.get_tool_schemas()
        action = EvoCUADesktopActionSpace.computer_use(action="key_down", keys=["shift"])
    """

    #: EvoCUA writes raw notch counts: 261 (value, trajectory) pairs measured,
    #: every one in 1..20, never a value that could be a screen-unit reading.
    SCROLL_WIRE_UNIT: ClassVar[int] = 1


    # -------------------------------------------------------------------------
    # Single computer_use tool with all parameters
    # -------------------------------------------------------------------------

    _ACTION_DESCRIPTION = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and HOLD the specified key(s) down in order (no release). Use this for stateful holds like holding Shift while clicking.
* `key_up`: Release the specified key(s) in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
"""

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        keys="Required only by `action=key`, `action=key_down`, and `action=key_up`.",
        text="Required only by `action=type`.",
        coordinate="The x,y coordinates for mouse actions.",
        pixels="The amount of scrolling. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
        time=duration_description("The seconds to wait."),
        status="The status of the task.",
    )
    def computer_use(
        action: Literal[
            "key", "key_down", "key_up", "type",
            "mouse_move", "left_click", "left_click_drag",
            "right_click", "middle_click", "double_click", "triple_click",
            "scroll", "wait", "terminate",
        ],
        keys: list[str] | None = None,
        text: str | None = None,
        coordinate: list[int] | None = None,
        pixels: int | None = None,
        time: float | None = None,
        status: Literal["success", "failure"] | None = None,
    ) -> dict[str, Any]:
        """
        Use a mouse and keyboard to interact with a computer, and take screenshots.

        * This is an interface to a desktop GUI. You must click on desktop icons to start applications.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
        * The screen's resolution is 1000x1000.
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

    LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {
        "click": ["left_click", "right_click", "middle_click", "double_click", "triple_click"],
        "drag": ["left_click_drag"],
        "type": ["type"], "key": ["key"], "key_down": ["key_down"], "key_up": ["key_up"],
        "mouse_move": ["mouse_move"],
        "scroll": ["scroll"], "wait": ["wait"],
    }
    #: EvoCUA dropped Qwen3-VL's ``answer`` enum member, so it has NO native
    #: answer channel — override the inherited map rather than claiming an
    #: ``answer`` entry this family cannot emit.
    QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES = {"terminate": frozenset({"terminate"})}

    # -------------------------------------------------------------------------
    # Tool Call Conversion: CUA-lite → EvoCUA
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single CUA-lite tool call to EvoCUA format (may return multiple)."""
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
                results.append(EvoCUADesktopActionSpace.computer_use(
                    action="right_click",
                    coordinate=c,
                )["function"])
            elif button == "middle":
                results.append(EvoCUADesktopActionSpace.computer_use(
                    action="middle_click",
                    coordinate=c,
                )["function"])
            elif clicks == 2:
                results.append(EvoCUADesktopActionSpace.computer_use(
                    action="double_click",
                    coordinate=c,
                )["function"])
            elif clicks == 3:
                results.append(EvoCUADesktopActionSpace.computer_use(
                    action="triple_click",
                    coordinate=c,
                )["function"])
            else:
                results.append(EvoCUADesktopActionSpace.computer_use(
                    action="left_click",
                    coordinate=c,
                )["function"])

        elif name == "type":
            results.append(EvoCUADesktopActionSpace.computer_use(
                action="type",
                text=args.get("text", ""),
            )["function"])

        elif name == "key":
            results.append(EvoCUADesktopActionSpace.computer_use(
                action="key",
                keys=args.get("keys", []),
            )["function"])

        elif name == "key_down":
            results.append(EvoCUADesktopActionSpace.computer_use(
                action="key_down",
                keys=args.get("keys", []),
            )["function"])

        elif name == "key_up":
            results.append(EvoCUADesktopActionSpace.computer_use(
                action="key_up",
                keys=args.get("keys", []),
            )["function"])

        elif name == "scroll":
            direction = args.get("direction", "down")
            amount = args.get("amount", 3)
            if direction not in ("up", "down"):
                # This family's only scroll carrier is the signed scalar
                # ``pixels``, whose single axis is vertical -- the parse side reads
                # the sign back as ``up``/``down`` and nothing else. There is no
                # faithful spelling of a horizontal scroll, so raise rather than
                # silently substitute the vertical axis.
                raise ValueError(
                    f"EvoCUA cannot render scroll(direction={direction!r}): its "
                    "action enum has no 'hscroll' and 'pixels' carries the "
                    "vertical axis only"
                )
            # ``int(round(...))``: the schema type is integer, so a float
            # ``amount`` from source data must not leak a float onto the wire.
            scroll_pixels = scroll_wire_magnitude(int(round(amount)),
                                                  wire_unit=type(self).SCROLL_WIRE_UNIT)
            if direction == "down":
                scroll_pixels = -scroll_pixels
            c = args.get("coordinate")
            results.append(EvoCUADesktopActionSpace.computer_use(
                action="scroll",
                pixels=scroll_pixels,
                coordinate=c,
            )["function"])

        elif name == "drag":
            if args.get("start_coordinate"):
                results.append(EvoCUADesktopActionSpace.computer_use(
                    action="mouse_move",
                    coordinate=args["start_coordinate"],
                )["function"])
            c = args.get("coordinate")
            if c:
                results.append(EvoCUADesktopActionSpace.computer_use(
                    action="left_click_drag",
                    coordinate=c,
                )["function"])

        elif name == "mouse_move":
            c = args.get("coordinate")
            if c:
                results.append(EvoCUADesktopActionSpace.computer_use(
                    action="mouse_move",
                    coordinate=c,
                )["function"])

        elif name == "wait":
            results.append(EvoCUADesktopActionSpace.computer_use(
                action="wait",
                time=compact_number(args.get("duration", 3)),
            )["function"])

        elif name == "terminate":
            results.append(EvoCUADesktopActionSpace.computer_use(
                action="terminate",
                status=args.get("status", "success"),
            )["function"])

        elif name == "response":
            # EvoCUA dropped Qwen3-VL's ``answer`` enum member, so the family has
            # NO native answer channel (see ``QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES``).
            # Fail loudly instead of leaking the canonical name onto the model
            # surface.
            raise ValueError("EvoCUA cannot render canonical tool 'response'")

        else:
            # Standalone schema-backed extras keep their canonical name.
            # EvoCUA's native wrapper is only for the trained GUI enum above.
            results.append({"name": name, "arguments": args})

        return results

    # -------------------------------------------------------------------------
    # Tool Call Conversion: EvoCUA → CUA-lite
    # -------------------------------------------------------------------------

    def _convert_single_from_agent(
        self,
        agent_tool_call: dict[str, Any],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert a single EvoCUA tool call to CUA-lite format (may return multiple)."""
        args = agent_tool_call["arguments"]
        action = args.get("action", "")

        name = agent_tool_call["name"]
        if name != "computer_use":
            admitted = extra_tool_name_and_arguments_are_admitted(
                name,
                args,
                active_extra_tool_names=active_extra_tool_names,
                active_extra_tool_schemas=active_extra_tool_schemas,
            )
            # The model dropped the ``computer_use`` wrapper and used the native
            # action value AS the tool name, e.g. ``left_click(coordinate=...)``.
            # An ACTIVE env extra outranks a colliding name in either fallback:
            # browsergym's ``click(bid=...)`` is the env's tool, and feeding it to
            # ``LiteDesktopActionSpace.click`` raises TypeError on the unknown
            # keyword instead of ever reaching env ingress.
            if not action and not admitted and name in _qwen_action_values(type(self)):
                action = name
            elif not action and not admitted and name in LiteDesktopActionSpace.get_action_names():
                # Desktop actions this enum cannot spell (``hold_key``,
                # ``mouse_down``, ``mouse_up``, ``screenshot``,
                # ``cursor_position``) leave ``to_agent`` as bare model-function
                # projections; parsing through the action owner keeps them in the
                # ``computer`` action-batch shape.
                return [getattr(LiteDesktopActionSpace, name)(**args)]
            elif not action or admitted:
                return [make_tool_call(name, args)]
            # Otherwise ``action`` is set under a non-wrapper name: the model kept
            # the native nesting but got the tool name wrong; the dispatch claims it.

        if action == "left_click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.click(coordinate=coordinate)]

        elif action == "right_click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.click(coordinate=coordinate, button="right")]

        elif action == "middle_click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.click(coordinate=coordinate, button="middle")]

        elif action == "double_click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.click(coordinate=coordinate, clicks=2)]

        elif action == "triple_click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.click(coordinate=coordinate, clicks=3)]

        elif action == "type":
            return [LiteDesktopActionSpace.type(text=required_model_text(args, action=action))]

        elif action == "key":
            raw_keys = args.get("keys", [])
            return [LiteDesktopActionSpace.key(keys=raw_keys)]

        elif action == "key_down":
            raw_keys = args.get("keys", [])
            return [LiteDesktopActionSpace.key_down(keys=raw_keys)]

        elif action == "key_up":
            raw_keys = args.get("keys", [])
            return [LiteDesktopActionSpace.key_up(keys=raw_keys)]

        elif action == "scroll":
            # ``pixels`` is REQUIRED — it is the only carrier of the direction.
            # Model may output it as a string float (e.g. "5.0"); the helper
            # float()s first. See ``required_scroll_pixels``.
            scroll_pixels = required_scroll_pixels(args, action)
            direction = "down" if scroll_pixels < 0 else "up"
            amount = scroll_clicks(scroll_pixels)
            return [LiteDesktopActionSpace.scroll(
                direction=direction,
                amount=amount,
                coordinate=optional_coord(args.get("coordinate"), dimensions=2),
            )]

        elif action == "left_click_drag":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.drag(coordinate=coordinate)]

        elif action == "mouse_move":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteDesktopActionSpace.mouse_move(coordinate=coordinate)]

        elif action == "wait":
            return [LiteDesktopActionSpace.wait(duration=float(args.get("time", 3)))]

        elif action == "terminate":
            terminate_args: dict[str, Any] = {}
            if "status" in args:
                terminate_args["status"] = args["status"]
            elif active_extra_tool_names is None and active_extra_tool_schemas is None:
                terminate_args["status"] = "success"
            if "status" in terminate_args:
                return [LiteFinishToolSet.terminate(status=terminate_args["status"])]
            return [make_tool_call("terminate", terminate_args)]

        else:
            # EvoCUA narrows its parent's enum (no ``answer``, no ``hscroll``),
            # so an out-of-enum value here is ordinary model output, not a bug.
            # Emit the invalid batch naming the model's own value, exactly as
            # ``Qwen3VLDesktopActionSpace`` and the other wrapper families do:
            # env ingress then answers it with feedback keyed to the call id.
            # Returning ``[]`` instead left the turn with zero tool calls, which
            # ``AgentBase.sample`` treats as terminal -- one off-enum value
            # ENDED the episode rather than costing a turn.
            logger.warning("Unknown EvoCUA action: %s(%s)", action, args)
            return [unknown_wrapper_action_batch(LITE_COMPUTER_ACTION_BATCH_TOOL_NAME, args)]


# =============================================================================
# Grounding (single-step click) action spaces
# =============================================================================
#
# EvoCUA inherits the trimmed Qwen3-VL grounding harness verbatim — same
# ``computer_use`` tool with a single ``left_click`` action enum. Desktop
# and browser share one class via the ``(desktop|browser)`` regex key.

@dataclasses.dataclass
class EvoCUADesktopGroundingPointActionSpace(
    Qwen3VLDesktopGroundingPointActionSpace, key=r"evocua@(desktop|browser)@point"
):
    """Desktop+browser grounding (single-step click) for EvoCUA. See parent."""
