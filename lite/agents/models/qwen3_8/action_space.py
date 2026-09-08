"""
Qwen3.8 action spaces (expanded desktop ``computer_use`` enum).

Qwen3.8 keeps Qwen3.5's XML tool_call wire format (handled by
:mod:`lite.agents.models.qwen3_8.adapter`) but is served through the
expanded OSWorld harness — ``${CUA_LITE_REFERENCES_ROOT}/OSWorld/mm_agents/qwen/``,
whose exported ``QwenAgent`` builds its schema from
``prompts.build_internal_tools_def``. That schema differs from the Qwen3.5
one (``prompts.build_base_tools_def``, mirrored by
:class:`~lite.agents.models.qwen3_5.action_space.Qwen3_5DesktopActionSpace`)
in exactly two ways:

* five actions are added — ``key_down``, ``key_up``, ``left_mouse_down``,
  ``left_mouse_up``, ``screenshot``;
* ``answer`` is removed and ``call_user`` takes over the text-to-user
  terminal channel.

Every added action already has a canonical counterpart in
:class:`~lite.core.tools.action_space.base.LiteDesktopActionSet`, so this
module only adds the projection; the Lite contract is unchanged.

Mobile inherits :class:`~lite.agents.models.qwen3_vl.action_space.Qwen3VLMobileActionSpace`
directly: the upstream harness declares no ``mobile_use`` tool, and the
Qwen3.5-family ``left_click`` tolerance is an empirical fix for measured 3.5
leakage, not a family-invariant.

Usage:
    from lite.agents.models.qwen3_8.action_space import (
        Qwen3_8DesktopActionSpace,
        Qwen3_8MobileActionSpace,
    )
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Literal

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.action_space.base import LiteDesktopActionSpace
from lite.agents.core.action_space.utils.geometry import (
    compact_number,
    model_keys,
    required_coord,
)
from lite.agents.models.qwen3_5.action_space import Qwen3_5DesktopActionSpace
from lite.agents.models.qwen3_vl.action_space import (
    Qwen3VLDesktopGroundingPointActionSpace,
    Qwen3VLMobileActionSpace,
    Qwen3VLMobileGroundingPointActionSpace,
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


# =============================================================================
# Qwen3.8 Desktop Action Space
# =============================================================================


@dataclasses.dataclass
class Qwen3_8DesktopActionSpace(Qwen3_5DesktopActionSpace, key=r"qwen3_8@(desktop|browser)"):
    """Desktop+browser action space for Qwen3.8 (expanded ``computer_use`` enum).

    Inherits Qwen3.5's coordinate space (1000x1000 normalized), scroll-pixel
    convention, and click/type/key projections; overrides the ``computer_use``
    schema and adds the five new action projections plus the
    ``call_user`` ↔ ``response`` terminal mapping.

    One class for desktop and browser: browser nav arrives as env
    ``extra_tools``, not as a per-platform action space.
    """

    platform: str = "desktop"

    # -------------------------------------------------------------------------
    # Single computer_use tool with all parameters
    # -------------------------------------------------------------------------

    _ACTION_DESCRIPTION = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and hold a single key without releasing it.
* `key_up`: Release a previously held single key.
* `left_mouse_down`: Press and hold the left mouse button.
* `left_mouse_up`: Release the left mouse button.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `triple_click`: Triple-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `screenshot`: Capture a new screenshot of the current screen.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `call_user`: Ask user for information or confirmation.
"""

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        keys="Required only by `action=key`, `action=key_down`, or `action=key_up`.",
        text="Required only by `action=type` and `action=call_user`.",
        coordinate="The x,y coordinates for mouse actions.",
        pixels="The amount of scrolling. Positive values scroll up, negative values scroll down. Required only by `action=scroll` and `action=hscroll`.",
        time=duration_description("The seconds to wait."),
        status="The status of the task.",
    )
    def computer_use(
        action: Literal[
            "key", "key_down", "key_up",
            "left_mouse_down", "left_mouse_up",
            "type", "mouse_move", "left_click", "left_click_drag",
            "right_click", "middle_click", "double_click", "triple_click",
            "scroll", "hscroll", "screenshot", "wait", "terminate", "call_user",
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

        * This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
        * The screen's resolution is 1000x1000.
        * Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
        * If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
        * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges.
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

    # Parent's table plus the five actions the expanded enum adds. Both gates
    # (``valid_actions`` and active extra tools) read these off ``cls``, so
    # redeclaring here is what makes the new values reachable.
    LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {
        **Qwen3_5DesktopActionSpace.LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES,
        "key_down": ["key_down"],
        "key_up": ["key_up"],
        "mouse_down": ["left_mouse_down"],
        "mouse_up": ["left_mouse_up"],
        "screenshot": ["screenshot"],
    }

    # ``answer`` is gone from the enum; ``call_user`` is the only text-bearing
    # terminal action left, so it spells canonical ``response``. Final answers
    # that need no tool call ride the no-tool-call text path instead — that is
    # what the expanded system prompt tells the model to do.
    QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES = {
        "call_user": frozenset({"response"}),
        "terminate": frozenset({"terminate"}),
    }

    # -------------------------------------------------------------------------
    # Tool Call Conversion
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert one CUA-lite tool call to Qwen3.8 format (may return multiple)."""
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == "key_down":
            return [Qwen3_8DesktopActionSpace.computer_use(
                action="key_down", keys=args.get("keys", []),
            )["function"]]

        if name == "key_up":
            return [Qwen3_8DesktopActionSpace.computer_use(
                action="key_up", keys=args.get("keys", []),
            )["function"]]

        if name == "mouse_down":
            return [Qwen3_8DesktopActionSpace.computer_use(action="left_mouse_down")["function"]]

        if name == "mouse_up":
            return [Qwen3_8DesktopActionSpace.computer_use(action="left_mouse_up")["function"]]

        if name == "screenshot":
            return [Qwen3_8DesktopActionSpace.computer_use(action="screenshot")["function"]]

        if name == "wait":
            return [Qwen3_8DesktopActionSpace.computer_use(
                action="wait", time=compact_number(args.get("duration", 3)),
            )["function"]]

        if name == "terminate":
            return [Qwen3_8DesktopActionSpace.computer_use(
                action="terminate", status=args.get("status", "success"),
            )["function"]]

        if name == "response":
            return [Qwen3_8DesktopActionSpace.computer_use(
                action="call_user", text=args.get("text", ""),
            )["function"]]

        # Everything else (click/drag/scroll/mouse_move/key, standalone extra
        # tools) is unchanged from the Qwen3.5 enum.
        return super()._convert_single_to_agent(tool_call)

    def _convert_single_from_agent(
        self,
        agent_tool_call: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Convert one Qwen3.8 tool call back to CUA-lite format."""
        args = agent_tool_call.get("arguments") or {}
        action = args.get("action", "")
        if not action:
            # The model dropped the ``computer_use`` wrapper and used the action
            # value AS the tool name. The parent promotes name → action off
            # ``LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES``, which reads this class's
            # expanded enum — so without the same promotion here the parent would
            # promote an expanded value it has no branch for and fall through to
            # ``unknown``. An ACTIVE env extra outranks a colliding native value,
            # same rule as the parent.
            name = agent_tool_call.get("name", "")
            claims_name = name in _EXPANDED_ONLY_ACTION_VALUES
            if claims_name and not extra_tool_name_and_arguments_are_admitted(
                name,
                args,
                active_extra_tool_names=kwargs.get("active_extra_tool_names"),
                active_extra_tool_schemas=kwargs.get("active_extra_tool_schemas"),
            ):
                action = name
        if action not in _EXPANDED_ONLY_ACTION_VALUES:
            return super()._convert_single_from_agent(agent_tool_call, **kwargs)

        if action in ("key_down", "key_up"):
            # Through ``model_keys`` for the same reason ``key`` is: the shared
            # key vocabulary raises a BARE ``ValueError`` on an unknown token,
            # which escapes the agent's parse boundary and loses the whole
            # trajectory instead of reaching the model as feedback.
            keys = model_keys(args.get("keys", []), action=action)
            ctor = (
                LiteDesktopActionSpace.key_down
                if action == "key_down"
                else LiteDesktopActionSpace.key_up
            )
            return [ctor(keys=keys)]

        if action in ("left_mouse_down", "left_mouse_up"):
            # ``coordinate`` is optional here: with one, the press/release
            # happens after a move, so it lowers to two canonical actions.
            calls: list[dict[str, Any]] = []
            coordinate = args.get("coordinate")
            if coordinate:
                calls.append(LiteDesktopActionSpace.mouse_move(
                    coordinate=required_coord(coordinate, dimensions=2),
                ))
            if action == "left_mouse_down":
                calls.append(LiteDesktopActionSpace.mouse_down(button="left"))
            else:
                calls.append(LiteDesktopActionSpace.mouse_up(button="left"))
            return calls

        if action == "screenshot":
            return [LiteDesktopActionSpace.screenshot()]

        # ``call_user`` — the expanded enum's text-to-user terminal channel.
        # Upstream decides DONE vs FAIL by sniffing the prose for infeasibility
        # phrases; cua-lite does not, because refusal has its own env-gated
        # ``report_infeasible`` tool and a text heuristic would silently
        # relabel ordinary answers.
        return [LiteFinishToolSet.response(text=args.get("text", ""))]


# Action values that exist only in the expanded enum. Membership here decides
# whether ``_convert_single_from_agent`` handles a call locally or delegates to
# the inherited Qwen3.5 branches.
#
_EXPANDED_ONLY_ACTION_VALUES = frozenset({
    "key_down", "key_up", "left_mouse_down", "left_mouse_up",
    "screenshot", "call_user",
})


# =============================================================================
# Qwen3.8 Mobile Action Space
# =============================================================================


@dataclasses.dataclass
class Qwen3_8MobileActionSpace(Qwen3VLMobileActionSpace, key="qwen3_8@mobile"):
    """Mobile action space for Qwen3.8.

    Inherits :class:`Qwen3VLMobileActionSpace` verbatim rather than the
    Qwen3.5 subclass: the expanded harness declares only the desktop
    ``computer_use`` tool, and Qwen3.5's ``left_click`` → ``click`` tolerance
    was justified by measured 3.5-family leakage rates, which say nothing
    about 3.8. Qwen3.8's mobile delta, if any, is the XML wire format handled
    by :mod:`lite.agents.models.qwen3_8.adapter`.
    """


# =============================================================================
# Grounding (single-step click) action spaces
# =============================================================================
#
# Trimmed schemas inherited verbatim from Qwen3-VL — a grounding surface
# exposes one click action, which the expanded enum does not change. Only the
# registry keys are new so the qwen3_8 adapters resolve.


@dataclasses.dataclass
class Qwen3_8DesktopGroundingPointActionSpace(
    Qwen3VLDesktopGroundingPointActionSpace, key=r"qwen3_8@(desktop|browser)@point"
):
    """Desktop+browser grounding (single-step click) for Qwen3.8."""


@dataclasses.dataclass
class Qwen3_8MobileGroundingPointActionSpace(
    Qwen3VLMobileGroundingPointActionSpace, key="qwen3_8@mobile@point"
):
    """Mobile grounding (single-step click) for Qwen3.8."""


__all__ = [
    "Qwen3_8DesktopActionSpace",
    "Qwen3_8MobileActionSpace",
    "Qwen3_8DesktopGroundingPointActionSpace",
    "Qwen3_8MobileGroundingPointActionSpace",
    "BaseActionSpace",
]
