"""
Qwen3VL Action Spaces

Wraps actions in a single tool with an "action" parameter,
matching the Qwen3VL expected format.

- Qwen3VLDesktopActionSpace: desktop+browser interactions via "computer_use"
  (click, type, scroll, …). Registered under the regex key
  ``r"qwen3_vl@(desktop|browser)"`` so ``ActionSpaceRegistry.get(
  "qwen3_vl@browser")`` resolves to the same class.
- Qwen3VLMobileActionSpace: mobile interactions via "mobile_use" (click, swipe, …)

Usage:
    from lite.agents.models.qwen3_vl.action_space import (
        Qwen3VLDesktopActionSpace,
        Qwen3VLMobileActionSpace,
    )

    action_space = Qwen3VLDesktopActionSpace()
    tools = action_space.get_tool_schemas()
    action = Qwen3VLDesktopActionSpace.computer_use(action="left_click", coordinate=[500, 500])
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
    model_duration,
    model_keys,
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
from lite.core.tools.schemas import (
    tool,
    tool_schema_name,
    tool_schema_parameters,
)

logger = logging.getLogger(__name__)

# Qwen3VL outputs scroll in screen pixels; Lite scroll uses wheel clicks. The
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
    """Every Qwen ``action`` value ``cls``'s from-agent dispatch understands.

    The wrapped GUI values plus the native spellings of standalone canonical
    tools — i.e. exactly what the model was trained to put in the wrapper's
    ``action`` field. Read when provider output drops the wrapper and uses an
    action value as the tool name, so that flat output converts through the
    same branches as nested output.
    """
    return frozenset(
        action_value
        for action_values in cls.LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES.values()
        for action_value in action_values
    ) | frozenset(cls.QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES)


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


def _normalize_open_app_name(
    app_name: Any,
    active_extra_tool_schemas: list[dict[str, Any]] | None,
) -> Any:
    """Normalize only exact case-insensitive app-catalog matches.

    Qwen often copies visible app names with different capitalization. When the
    env provided a unique ``open_app.app_name`` enum match, use the env spelling
    so app launch can proceed. Non-matches pass through unchanged and are
    reported by env-owned schema/action feedback with the current observation.
    """
    if not isinstance(app_name, str):
        return app_name
    query = app_name.strip()
    if not query:
        return app_name
    for schema in active_extra_tool_schemas or []:
        if tool_schema_name(schema) != "open_app":
            continue
        app_prop = (
            tool_schema_parameters(schema)
            .get("properties", {})
            .get("app_name", {})
        )
        enum = app_prop.get("enum") if isinstance(app_prop, dict) else None
        if not isinstance(enum, list):
            continue
        names = [str(name) for name in enum]
        if query in names:
            return query
        folded = [name for name in names if name.casefold() == query.casefold()]
        if len(folded) == 1:
            return folded[0]
    return app_name


def _qwen_mobile_open_app_call(
    args: dict[str, Any],
    active_extra_tool_schemas: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Canonicalize Qwen mobile app-open intent to Lite ``open_app``.

    Full argument validity remains env-owned so bad app names produce a
    role:tool result with fresh observation, not an adapter parse failure.
    """
    app_name = args.get("app_name") if "app_name" in args else args.get("text", "")
    app_name = _normalize_open_app_name(
        app_name,
        active_extra_tool_schemas,
    )
    return [make_tool_call("open_app", {"app_name": app_name})]


@dataclasses.dataclass
class Qwen3VLDesktopActionSpace(
    BaseActionSpace,
    key=r"qwen3_vl@(desktop|browser)",
):
    """
    Qwen3VL-style action space for desktop interactions.

    Uses absolute 1000x1000 coordinate system (same as CUA-lite normalized).
    All actions are wrapped in a single "computer_use" tool with an "action" parameter.

    Actions:
        - left_click, right_click, middle_click, double_click, triple_click
        - type, key
        - mouse_move, left_click_drag
        - scroll, hscroll
        - wait, answer, terminate

    Example:
        action_space = Qwen3VLDesktopActionSpace()
        tools = action_space.get_tool_schemas()
        action = Qwen3VLDesktopActionSpace.computer_use(action="left_click", coordinate=[500, 500])
    """


    #: The spelling this family writes on the wire: one click as this many
    #: ``pixels``. Measured per family, deduplicated per trajectory: Qwen3-VL
    #: is 84.5% screen units; Qwen3.5/3.8/EvoCUA write raw notch counts.
    #: Rendering a notch writer in screen units teaches a student to write
    #: 300 where its teacher wrote 3.
    SCROLL_WIRE_UNIT: ClassVar[int] = PIXELS_PER_CLICK

    platform: str = "desktop"
    _QWEN_NATIVE_TOOL_NAME = "computer_use"

    # -------------------------------------------------------------------------
    # Single computer_use tool with all parameters
    # -------------------------------------------------------------------------

    _ACTION_DESCRIPTION = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
"""

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        keys="Required only by `action=key`.",
        text="Required only by `action=type` and `action=answer`.",
        coordinate="The x,y coordinates for mouse actions.",
        pixels="The amount of scrolling. Positive values scroll up, negative values scroll down. Required only by `action=scroll` and `action=hscroll`.",
        time=duration_description("The seconds to wait."),
        status="The status of the task.",
    )
    def computer_use(
        action: Literal[
            "key", "type", "mouse_move", "left_click", "left_click_drag",
            "right_click", "middle_click", "double_click", "triple_click",
            "scroll", "hscroll", "wait", "terminate", "answer",
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

    # CUA-Lite action name → the Qwen action values it maps to.
    LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {
        "click": ["left_click", "right_click", "middle_click", "double_click", "triple_click"],
        "drag": ["left_click_drag"],
        "type": ["type"], "key": ["key"], "mouse_move": ["mouse_move"],
        "scroll": ["scroll", "hscroll"], "wait": ["wait"],
    }
    # Qwen action value → the canonical STANDALONE extra tool it spells, e.g.
    # ``computer_use(action="answer")`` is the native spelling of ``response``.
    # Gated from ``metadata.extra_tool_schemas``, NOT by ``valid_actions`` — so
    # ``valid_actions=[]`` + ``extra_tools=["terminate"]`` still leaves the
    # agent a way to end the episode.
    QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES = {
        "answer": frozenset({"response"}),
        "terminate": frozenset({"terminate"}),
    }

    filter_tool_schemas_for_valid_actions = classmethod(
        _filter_qwen_action_values_for_valid_actions
    )
    # Qwen owns this gate: which provider action values a sample's active extra
    # tools leave reachable is family policy, so the Qwen adapter calls it.
    filter_qwen_action_values_for_active_extra_tools = classmethod(
        _filter_qwen_action_values_for_active_extra_tools
    )

    # -------------------------------------------------------------------------
    # Tool Call Conversion
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single CUA-lite tool call to Qwen3VL format (may return multiple)."""
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        results = []

        if name == "computer":
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                results.extend(self._convert_single_to_agent(make_tool_call(action, action_args)))

        elif name == "click":
            coord = args.get("coordinate")
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)

            if button == "right":
                results.append(Qwen3VLDesktopActionSpace.computer_use(
                    action="right_click",
                    coordinate=coord,
                )["function"])
            elif button == "middle":
                results.append(Qwen3VLDesktopActionSpace.computer_use(
                    action="middle_click",
                    coordinate=coord,
                )["function"])
            elif clicks == 2:
                results.append(Qwen3VLDesktopActionSpace.computer_use(
                    action="double_click",
                    coordinate=coord,
                )["function"])
            elif clicks == 3:
                results.append(Qwen3VLDesktopActionSpace.computer_use(
                    action="triple_click",
                    coordinate=coord,
                )["function"])
            else:
                results.append(Qwen3VLDesktopActionSpace.computer_use(
                    action="left_click",
                    coordinate=coord,
                )["function"])

        elif name == "type":
            results.append(Qwen3VLDesktopActionSpace.computer_use(
                action="type",
                text=args.get("text", ""),
            )["function"])

        elif name == "key":
            results.append(Qwen3VLDesktopActionSpace.computer_use(
                action="key",
                keys=args.get("keys", []),
            )["function"])

        elif name == "scroll":
            direction = args.get("direction", "down")
            amount = args.get("amount", 3)
            # ``int(round(...))``: the schema type is integer, so a float
            # ``amount`` from source data must not leak onto the wire.
            scroll_pixels = scroll_wire_magnitude(int(round(amount)),
                                                  wire_unit=type(self).SCROLL_WIRE_UNIT)
            # Native Qwen3-VL splits vertical (``scroll``) vs horizontal
            # (``hscroll``); both are pixels-only with NO coordinate (the
            # ``coordinate`` arg is for click/move). Signs mirror the from_agent
            # parse: scroll → neg=down/pos=up; hscroll → neg=left/pos=right. A
            # horizontal lite scroll must become ``hscroll`` (was wrongly emitted
            # as a vertical ``scroll``), and no coordinate is attached (was carried
            # from GPT-origin element-anchored scrolls → off-distribution).
            if direction in ("left", "right"):
                if direction == "left":
                    scroll_pixels = -scroll_pixels
                results.append(Qwen3VLDesktopActionSpace.computer_use(
                    action="hscroll",
                    pixels=scroll_pixels,
                )["function"])
            else:
                if direction == "down":
                    scroll_pixels = -scroll_pixels
                results.append(Qwen3VLDesktopActionSpace.computer_use(
                    action="scroll",
                    pixels=scroll_pixels,
                )["function"])

        elif name == "drag":
            # If start_coordinate is provided, need mouse_move first
            if args.get("start_coordinate"):
                results.append(Qwen3VLDesktopActionSpace.computer_use(
                    action="mouse_move",
                    coordinate=args["start_coordinate"],
                )["function"])
            coord = args.get("coordinate")
            if coord:
                results.append(Qwen3VLDesktopActionSpace.computer_use(
                    action="left_click_drag",
                    coordinate=coord,
                )["function"])

        elif name == "mouse_move":
            coord = args.get("coordinate")
            if coord:
                results.append(Qwen3VLDesktopActionSpace.computer_use(
                    action="mouse_move",
                    coordinate=coord,
                )["function"])

        elif name == "wait":
            results.append(Qwen3VLDesktopActionSpace.computer_use(
                action="wait",
                time=compact_number(args.get("duration", 3)),
            )["function"])

        elif name == "terminate":
            results.append(Qwen3VLDesktopActionSpace.computer_use(
                action="terminate",
                status=args.get("status", "success"),
            )["function"])

        elif name == "response":
            results.append(Qwen3VLDesktopActionSpace.computer_use(
                action="answer",
                text=args.get("text", ""),
            )["function"])

        elif name in LiteDesktopActionSpace.get_action_names():
            raise ValueError(
                f"Qwen3-VL desktop cannot render canonical tool {name!r}: its "
                "action enum has no counterpart, and dropping it silently loses "
                "the action"
            )

        else:
            # Not a ``computer_use``-wrapped action (the wrapped set is exactly
            # ``LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES``'s keys, handled above)
            # → a STANDALONE function tool: browser nav (goto/back/...), env extra
            # tools, etc. Its CUA-lite
            # name IS its wire name, so pass it through unchanged rather than
            # forcing it into a bogus ``computer_use(action=<name>)``.
            results.append({"name": name, "arguments": args})

        return results

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Convert Qwen3VL tool_calls back to CUA-lite format.

        Qwen3VL format:
            {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [500, 500]}}

        CUA-lite format keeps GUI actions under an action-batch call,
        even when the run has length 1:
            {"name": "computer", "arguments": {"actions": [
                {"action": "click", "coordinate": [500, 500]}
            ]}}

        Adjacent provider-native wrapper calls in the same assistant turn are grouped:
            {"name": "computer", "arguments": {"actions": [
                {"action": "click", "coordinate": [500, 500]},
                {"action": "type", "text": "hello"}
            ]}}

        Unknown provider-native wrapper actions become invalid action-batch
        calls. The env owns availability and feedback for standalone extra
        tools; this shared model action space does not repair env-specific
        extra-tool names emitted inside ``computer_use``.
        """
        result = []
        for tool_call in agent_tool_calls:
            result.extend(self._convert_single_from_agent(
                tool_call,
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
        """Convert a single Qwen3VL tool call to CUA-lite format (may return multiple)."""
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
            # ``click(bid=...)`` is the env's tool, not the native ``click`` the
            # dispatch would read as a coordinate action.
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
            # Raw key payloads pass through the shared key vocabulary owner so
            # every model family reaches the same canonical tokens; ``model_keys``
            # adds the list-element ``+`` split and makes a bad token model-visible
            # feedback instead of an uncaught error that loses the trajectory.
            return [LiteDesktopActionSpace.key(keys=model_keys(args.get("keys", []), action=action))]

        elif action == "scroll":
            # ``pixels`` is REQUIRED here — it is the only carrier of the
            # direction, so ``required_scroll_pixels`` raises rather than
            # defaulting. Model may output it as a string float (e.g. "5.0");
            # int("5.0") raises ValueError, so the helper float()s first.
            scroll_pixels = required_scroll_pixels(args, action)
            direction = "down" if scroll_pixels < 0 else "up"
            amount = scroll_clicks(scroll_pixels)
            return [LiteDesktopActionSpace.scroll(
                direction=direction,
                amount=amount,
                coordinate=optional_coord(args.get("coordinate"), dimensions=2),
            )]

        elif action == "hscroll":
            # Same required-argument + string-float guard as scroll above.
            scroll_pixels = required_scroll_pixels(args, action)
            direction = "left" if scroll_pixels < 0 else "right"
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
            return [LiteDesktopActionSpace.wait(duration=model_duration(args, action=action, default=3))]

        elif action == "answer":
            return [LiteFinishToolSet.response(text=args.get("text", ""))]

        elif action == "terminate":
            return [LiteFinishToolSet.terminate(status=args.get("status", "success"))]

        else:
            logger.warning("Unknown Qwen3VL action: %s(%s)", action, args)
            return [unknown_wrapper_action_batch(LITE_COMPUTER_ACTION_BATCH_TOOL_NAME, args)]


# =============================================================================
# Qwen3VL Mobile Action Space
# =============================================================================

@dataclasses.dataclass
class Qwen3VLMobileActionSpace(
    BaseActionSpace,
    key="qwen3_vl@mobile",
):
    """
    Qwen3-VL mobile action space.

    Reference: ${CUA_LITE_REFERENCES_ROOT}/Qwen3-VL/cookbooks/mobile_agent.ipynb
    and ${CUA_LITE_REFERENCES_ROOT}/Qwen3-VL/cookbooks/utils/agent_function_call.py.

    Uses a single "mobile_use" tool with an "action" parameter. Trained enum
    includes app launch as ``action="open"``:

        click, long_press, swipe, type, open, answer, system_button, wait, terminate

    Coordinate system: declared as ``999x999`` in the system prompt (model SFT
    distribution). cua-lite normalized space is ``[0, 1000]``. The 0.1%
    difference is sub-pixel on real Android resolutions, so we **pass
    coordinates through unchanged**: model emits ints in ``[0, 999]``, we
    treat them as cua-lite ``[0, 1000]`` directly. Adding an explicit rescale
    would only inject ±1-unit rounding noise.

    Example:
        action_space = Qwen3VLMobileActionSpace()
        tools = action_space.get_tool_schemas()
        action = Qwen3VLMobileActionSpace.mobile_use(action="click", coordinate=[500, 500])
    """

    platform: str = "mobile"
    _QWEN_NATIVE_TOOL_NAME = "mobile_use"

    # Derived from the official mobile_use tool. ``answer`` is Qwen's action
    # value for the canonical ``response`` tool, and ``open`` is its action
    # value for canonical ``open_app``.
    _ACTION_DESCRIPTION = """The action to perform. The available actions are:
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `open`: Open an app on the device.
* `answer`: Output the answer.
* `system_button`: Press the system button.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status."""

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        coordinate="(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
        coordinate2="(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
        text="Required only by `action=type`, `action=open`, and `action=answer`.",
        time=duration_description(
            "The seconds to wait. Required only by `action=long_press` and `action=wait`."
        ),
        button="Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
        status="The status of the task. Required only by `action=terminate`.",
    )
    def mobile_use(
        action: Literal[
            "click", "long_press", "swipe", "type",
            "open", "answer", "system_button", "wait", "terminate",
        ],
        coordinate: list[int] | None = None,
        coordinate2: list[int] | None = None,
        text: str | None = None,
        time: float | None = None,
        button: Literal["Back", "Home", "Menu", "Enter", "Recent"] | None = None,
        status: Literal["success", "failure"] | None = None,
    ) -> dict[str, Any]:
        """
        Use a touchscreen to interact with a mobile device, and take screenshots.

        * This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
        * The screen's resolution is 999x999.
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

    # -------------------------------------------------------------------------
    # Action filtering
    # -------------------------------------------------------------------------

    # cua-lite name → Qwen3-VL mobile enum values it maps to.
    # ``open_app`` is canonical Lite/env vocabulary; Qwen's native wire spells it
    # ``mobile_use(action="open", text=...)``. The prompt advertises and parses
    # that enum entry only when the env injects the canonical ``open_app`` extra
    # schema; ``valid_actions`` does not control app launch.
    # ``drag`` is not trained for mobile: no native entry parses back to it, and
    # the to-agent path RAISES rather than borrowing ``swipe`` (which would parse
    # back as ``swipe``, a different canonical action).
    # ``tap`` with ``clicks=2`` (double-tap) also RAISES: the enum has no
    # double-tap entry, and mapping it to a single ``click`` re-parsed as
    # ``tap(clicks=1)`` -- a different gesture, silently.
    LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {
        "tap": ["click"],
        "long_press": ["long_press"],
        "swipe": ["swipe"],
        "type": ["type"],
        "drag": [],  # not in trained enum; the to-agent path raises
        "system_button": ["system_button"],
        "wait": ["wait"],
    }
    # See :class:`Qwen3VLDesktopActionSpace` — scoped by ``extra_tool_schemas``.
    QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES = {
        "open": frozenset({"open_app"}),
        "answer": frozenset({"response"}),
        "terminate": frozenset({"terminate"}),
    }

    filter_tool_schemas_for_valid_actions = classmethod(
        _filter_qwen_action_values_for_valid_actions
    )
    # Qwen owns this gate: which provider action values a sample's active extra
    # tools leave reachable is family policy, so the Qwen adapter calls it.
    filter_qwen_action_values_for_active_extra_tools = classmethod(
        _filter_qwen_action_values_for_active_extra_tools
    )

    # -------------------------------------------------------------------------
    # Tool Call Conversion
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single CUA-lite mobile tool call to Qwen3VL format."""
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
            # for the same ``clicks`` loss on ``click``. Inherited by
            # qwen3_5@mobile, which shares this enum.
            clicks = args.get("clicks", 1)
            if clicks != 1:
                raise ValueError(
                    f"Qwen3-VL mobile cannot render tap(clicks={clicks}): its "
                    "action enum has only 'click' and the wire carries no repeat count"
                )
            results.append(Qwen3VLMobileActionSpace.mobile_use(
                action="click",
                coordinate=args.get("coordinate"),
            )["function"])

        elif name == "long_press":
            results.append(Qwen3VLMobileActionSpace.mobile_use(
                action="long_press",
                coordinate=args.get("coordinate"),
                time=args.get("duration"),
            )["function"])

        elif name == "swipe":
            results.append(Qwen3VLMobileActionSpace.mobile_use(
                action="swipe",
                coordinate=args.get("start_coordinate"),
                coordinate2=args.get("coordinate"),
            )["function"])

        elif name == "type":
            results.append(Qwen3VLMobileActionSpace.mobile_use(
                action="type",
                text=args.get("text", ""),
            )["function"])

        elif name == "system_button":
            results.append(Qwen3VLMobileActionSpace.mobile_use(
                action="system_button",
                button=args.get("button"),
            )["function"])

        elif name == "drag":
            # The trained mobile enum has no ``drag``. ``swipe`` is the only
            # two-point gesture and it parses back as canonical ``swipe``, so
            # borrowing it would substitute a different action; dropping the call
            # (the previous behaviour) deleted it from the wire and the round trip
            # came back EMPTY. Raise instead -- the ``ui_tars@mobile`` convention
            # for exactly this cell. Inherited by qwen3_5@mobile and
            # the mobile families that share this enum.
            raise ValueError(
                "Qwen3-VL mobile cannot render canonical tool 'drag': its action "
                "enum has only 'swipe', which parses back as 'swipe'"
            )

        elif name == "open_app":
            results.append(Qwen3VLMobileActionSpace.mobile_use(
                action="open",
                text=args.get("app_name", ""),
            )["function"])

        elif name == "response":
            results.append(Qwen3VLMobileActionSpace.mobile_use(
                action="answer",
                text=args.get("text", ""),
            )["function"])

        elif name == "wait":
            results.append(Qwen3VLMobileActionSpace.mobile_use(
                action="wait",
                time=compact_number(args.get("duration", 3)),
            )["function"])

        elif name == "terminate":
            results.append(Qwen3VLMobileActionSpace.mobile_use(
                action="terminate",
                status=args.get("status", "success"),
            )["function"])

        elif name in LiteMobileActionSpace.get_action_names():
            raise ValueError(
                f"Qwen3-VL mobile cannot render canonical tool {name!r}: its "
                "action enum has no counterpart, and dropping it silently loses "
                "the action"
            )

        else:
            results.append({"name": name, "arguments": args})

        return results

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Convert Qwen3VL mobile_use tool_calls back to CUA-lite format.

        Qwen3VL format:
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 500]}}

        CUA-lite format keeps GUI actions under an action-batch call,
        even when the run has length 1:
            {"name": "mobile", "arguments": {"actions": [
                {"action": "tap", "coordinate": [500, 500]}
            ]}}

        Adjacent provider-native wrapper calls in the same assistant turn are grouped:
            {"name": "mobile", "arguments": {"actions": [
                {"action": "tap", "coordinate": [500, 500]},
                {"action": "type", "text": "hello"}
            ]}}

        ``active_extra_tool_schemas`` lets Qwen mobile's native
        ``action="open"`` spelling normalize against the active ``open_app``
        app catalog. Other unknown provider-native wrapper actions become
        invalid action-batch calls so env availability and feedback remain
        env-owned.
        """
        result = []
        for tool_call in agent_tool_calls:
            result.extend(self._convert_single_from_agent(
                tool_call,
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
        """Convert a single Qwen3VL mobile_use tool call to CUA-lite format."""
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
            # An ACTIVE env extra outranks a colliding native value.
            if not action and not admitted and name in _qwen_action_values(type(self)):
                action = name
            elif not action or admitted:
                # A standalone tool (env extra tool, custom tool). Its Lite name
                # IS its wire name. Do NOT infer ``answer`` from a "text" arg —
                # that silently collapsed text-bearing extra tools into
                # ``response``.
                return [make_tool_call(name, args)]
            # Otherwise ``action`` is set under a non-wrapper name: the model kept
            # the native nesting but got the tool name wrong; the dispatch claims it.

        if action == "click":
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteMobileActionSpace.tap(coordinate=coordinate)]

        elif action == "long_press":
            t = args.get("time")
            coordinate = required_coord(args.get("coordinate"), dimensions=2)
            return [LiteMobileActionSpace.long_press(
                coordinate=coordinate,
                duration=float(t) if t is not None else None,
            )]

        elif action == "swipe":
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

        elif action == "type":
            return [LiteMobileActionSpace.type(text=required_model_text(args, action=action))]

        elif action == "open":
            return _qwen_mobile_open_app_call(
                args,
                active_extra_tool_schemas,
            )

        elif action == "open_app":
            return _qwen_mobile_open_app_call(
                args,
                active_extra_tool_schemas,
            )

        elif action == "system_button":
            btn = args.get("button", "")
            return [LiteMobileActionSpace.system_button(button=btn)]

        elif action == "wait":
            return [LiteMobileActionSpace.wait(duration=model_duration(args, action=action, default=3))]

        elif action == "answer":
            return [LiteFinishToolSet.response(text=args.get("text", ""))]

        elif action == "terminate":
            return [LiteFinishToolSet.terminate(status=args.get("status", "success"))]

        else:
            logger.warning("Unknown Qwen3VL mobile action: %s(%s)", action, args)
            return [unknown_wrapper_action_batch(LITE_MOBILE_ACTION_BATCH_TOOL_NAME, args)]


# =============================================================================
# Qwen3-VL Grounding Action Space (point) — single-step click harness
# =============================================================================
#
# Bridge between cua-lite's :class:`LitePointActionSpace` (canonical
# ``point(coordinate=[x,y])``) and Qwen3-VL's ``computer_use`` wire format,
# but with a TRIMMED 2-action enum (``left_click`` + ``terminate``) — no
# scroll / type / drag / mouse_move / etc. The simpler schema concentrates
# the model's attention on coordinate output and matches the grounding
# cookbook's "Only left_click is allowed" guidance.
#
# Refusal path: agent ``terminate(status="failure")`` round-trips through
# this action space as the env-side extra tool ``report_infeasible(reason)``
# (osworld_g / screenspot_pro convention). Agent ``terminate(status="success")``
# is dropped — single-step grounding has no "I succeeded without clicking".

@dataclasses.dataclass
class Qwen3VLDesktopGroundingPointActionSpace(
    BaseActionSpace,
    key=r"qwen3_vl@(desktop|browser)@point",
):
    """Qwen3-VL trimmed grounding action space (single-step click).

    Wire format: same single ``computer_use`` tool, with the ``action`` enum
    trimmed to a single value (``left_click``). Refusal is handled by the
    env's ``report_infeasible`` extra tool, NOT by an in-schema ``terminate``
    — keeps the action space focused purely on coordinate output. Coordinate
    normalization is identity ([0, 1000] cua-lite ↔ [0, 1000] Qwen3-VL).
    """

    platform: str = "desktop"
    _QWEN_NATIVE_TOOL_NAME = "computer_use"

    _ACTION_DESCRIPTION = """The action to perform. The available actions are:
* `left_click`: Click the left mouse button at the (x, y) pixel coordinate of the target element on the screen."""

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        coordinate="The (x, y) pixel coordinate to click. Required by `action=left_click`.",
    )
    def computer_use(
        action: Literal["left_click"],
        coordinate: list[int] | None = None,
    ) -> dict[str, Any]:
        """Use a mouse to click on the target element in the screenshot.

        * The screen's resolution is 1000x1000.
        * Click with the cursor tip centered on the target; avoid edges
          unless the instruction asks for an edge.
        * Do not use other tools. Only ``left_click`` is available.
        """
        return make_tool_call(
            "computer_use",
            {"action": action, "coordinate": coordinate},
        )

    # This trimmed surface declares no Qwen action value that spells a
    # standalone canonical tool, so it has no active-extra gate.
    LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {
        "point": ["left_click"],
    }

    filter_tool_schemas_for_valid_actions = classmethod(
        _filter_qwen_grounding_action_values_for_valid_actions
    )

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """cua-lite ``point(coord)`` → Qwen3-VL ``computer_use(action=left_click, coord)``.

        Refusal path (``report_infeasible``) is treated as an extra tool by
        the adapter — it does not flow through here.
        """
        result = []
        for tool_call in tool_calls:
            name = tool_call_name(tool_call)
            args = tool_call_arguments(tool_call)
            if name == "point":
                result.append(
                    Qwen3VLDesktopGroundingPointActionSpace.computer_use(
                        action="left_click",
                        coordinate=args.get("coordinate"),
                    )["function"]
                )
            else:
                result.extend(convert_non_point_call_for_grounding_space(
                    tool_call, surface="Qwen3-VL grounding (point) action_space",
                ))
        return result

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Qwen3-VL agent → cua-lite single-call mapping.

        ``computer_use(left_click, coord)`` → ``LitePointActionSpace.point(coord)``.
        Non-``computer_use`` names (e.g. extra tools like ``report_infeasible``)
        pass through unchanged. Anything else is dropped with a warning.
        """
        result = []
        for agent_tool_call in agent_tool_calls:
            name = agent_tool_call["name"]
            args = agent_tool_call["arguments"]
            action = args.get("action", "")
            if name != "computer_use":
                # Non-computer_use names — including extra tools like
                # ``report_infeasible`` — become canonical Lite standalone calls.
                result.append(make_tool_call(name, args))
                continue
            if action == "left_click":
                coordinate = required_coord(args.get("coordinate"), dimensions=2)
                result.append(LitePointActionSpace.point(
                    coordinate=coordinate,
                ))
            else:
                logger.warning(
                    "Qwen3-VL grounding (point) dropping off-schema action %s(%s)",
                    action, args,
                )
        return result


@dataclasses.dataclass
class Qwen3VLMobileGroundingPointActionSpace(
    BaseActionSpace,
    key="qwen3_vl@mobile@point",
):
    """Qwen3-VL mobile grounding action space (single-step tap).

    Mirror of the desktop grounding harness, but using the **mobile** wire
    format: ``mobile_use`` tool name (not ``computer_use``), ``click`` enum
    value (not ``left_click``), and touchscreen-flavored description.
    Schema is trimmed to a single enum value (``click``). Refusal is handled
    by the env's ``report_infeasible`` extra tool. Coordinate space is
    identity ([0, 999] mobile cookbook ↔ [0, 1000] cua-lite — sub-pixel
    difference, see :class:`Qwen3VLMobileActionSpace` for rationale).
    """

    platform: str = "mobile"
    _QWEN_NATIVE_TOOL_NAME = "mobile_use"

    _ACTION_DESCRIPTION = """The action to perform. The available actions are:
* `click`: Click (tap) the point on the screen with coordinate (x, y) of the target element."""

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        coordinate="(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates of the target. Required by `action=click`.",
    )
    def mobile_use(
        action: Literal["click"],
        coordinate: list[int] | None = None,
    ) -> dict[str, Any]:
        """Use a touchscreen to tap on the target element in the screenshot.

        * This is an interface to a mobile device with a touchscreen.
        * The screen's resolution is 999x999.
        * Tap with the touch point centered on the target; avoid edges
          unless the instruction asks for an edge.
        * Do not use other tools. Only ``click`` is available.
        """
        return make_tool_call(
            "mobile_use",
            {"action": action, "coordinate": coordinate},
        )

    # This trimmed surface declares no Qwen action value that spells a
    # standalone canonical tool, so it has no active-extra gate.
    LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {
        "point": ["click"],
    }

    filter_tool_schemas_for_valid_actions = classmethod(
        _filter_qwen_grounding_action_values_for_valid_actions
    )

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """cua-lite ``point(coord)`` → Qwen3-VL mobile ``mobile_use(click, coord)``."""
        result = []
        for tool_call in tool_calls:
            name = tool_call_name(tool_call)
            args = tool_call_arguments(tool_call)
            if name == "point":
                result.append(
                    Qwen3VLMobileGroundingPointActionSpace.mobile_use(
                        action="click",
                        coordinate=args.get("coordinate"),
                    )["function"]
                )
            else:
                result.extend(convert_non_point_call_for_grounding_space(
                    tool_call, surface="Qwen3-VL mobile grounding (point) action_space",
                ))
        return result

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Qwen3-VL mobile agent → cua-lite single-call mapping.

        ``mobile_use(click, coord)`` → ``LitePointActionSpace.point(coord)``.
        Non-``mobile_use`` names (e.g. extra tools like ``report_infeasible``)
        pass through unchanged. Anything else is dropped with a warning.
        """
        result = []
        for agent_tool_call in agent_tool_calls:
            name = agent_tool_call["name"]
            args = agent_tool_call["arguments"]
            action = args.get("action", "")
            if name != "mobile_use":
                # Non-mobile_use names — including extra tools like
                # ``report_infeasible`` — become canonical Lite standalone calls.
                result.append(make_tool_call(name, args))
                continue
            if action == "click":
                coordinate = required_coord(args.get("coordinate"), dimensions=2)
                result.append(LitePointActionSpace.point(
                    coordinate=coordinate,
                ))
            else:
                logger.warning(
                    "Qwen3-VL mobile grounding (point) dropping off-schema action %s(%s)",
                    action, args,
                )
        return result
