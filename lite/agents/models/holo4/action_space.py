"""Holo4 flat tool dialects and canonical Lite action conversion."""

from __future__ import annotations

import ast
import dataclasses
from typing import Any, ClassVar, Literal

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.core.tools.action_space import LiteDesktopActionSet, LiteMobileActionSet
from lite.core.tools.action_space.batches import (
    LITE_COMPUTER_ACTION_BATCH_TOOL_NAME,
    LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
    merge_adjacent_lite_action_batches,
    unpack_action_batch_call,
)
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import BASH_TOOL_NAME, LiteFinishToolSet
from lite.core.tools.schemas import tool, tool_schema_name

_DESKTOP_SCHEMA_ACTIONS: dict[str, frozenset[str]] = {
    "click_desktop": frozenset({"click"}),
    "double_click_desktop": frozenset({"click"}),
    "triple_click_desktop": frozenset({"click"}),
    "move_to_desktop": frozenset({"mouse_move"}),
    "mouse_down_desktop": frozenset({"mouse_down"}),
    "mouse_up_desktop": frozenset({"mouse_up"}),
    "scroll_desktop": frozenset({"scroll"}),
    "drag_and_drop": frozenset({"drag"}),
    "drag_to_desktop": frozenset({"drag"}),
    "hotkey_desktop": frozenset({"key"}),
    "key_down_desktop": frozenset({"key_down"}),
    "key_up_desktop": frozenset({"key_up"}),
    "hold_and_tap_key_desktop": frozenset({"key"}),
    "write_desktop": frozenset({"key", "type"}),
    "write_at_desktop": frozenset({"click", "key", "type"}),
    "wait_desktop": frozenset({"wait"}),
}

_MOBILE_SCHEMA_ACTIONS: dict[str, frozenset[str]] = {
    "mobile_click": frozenset({"tap"}),
    "mobile_write": frozenset({"tap", "type", "system_button"}),
    "mobile_scroll": frozenset({"swipe"}),
    "mobile_swipe": frozenset({"swipe"}),
    "mobile_long_press": frozenset({"long_press"}),
    "mobile_drag": frozenset({"drag"}),
    "mobile_go_home": frozenset({"system_button"}),
    "mobile_go_back": frozenset({"system_button"}),
    "mobile_hide_keyboard": frozenset({"system_button"}),
    "mobile_go_to_all_apps": frozenset({"system_button", "swipe"}),
}

_COORDINATE_KEYS: tuple[str, ...] = (
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "x_touch",
    "y_touch",
    "x_lift",
    "y_lift",
)
_COORDINATE_PAIR_KEYS: tuple[tuple[str, str], ...] = (
    ("x", "y"),
    ("x1", "y1"),
    ("x2", "y2"),
    ("x_touch", "y_touch"),
    ("x_lift", "y_lift"),
)

# Last line of a Holo4 final answer that declares the task infeasible.
_INFEASIBLE_ANSWER_MARKER = "[TASK=INFEASIBLE]"


def _normalize_coordinate_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Coerce model-emitted coordinate strings to the declared ints.

    The model occasionally emits a pointer coordinate as a numeric string
    (``"509"``) or packs both values of a pair into its first key
    (``"272, 306"``, ``"[601, 925]"``); official trajectories show the same
    string forms. Every pointer tool on both surfaces shares this one
    normalizer. Mutates and returns ``arguments``; values that do not
    literal-eval stay as-is and fail loudly at converter binding.
    """
    for key in _COORDINATE_KEYS:
        value = arguments.get(key)
        if isinstance(value, str):
            try:
                arguments[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass
    for first, second in _COORDINATE_PAIR_KEYS:
        value = arguments.get(first)
        if isinstance(value, (list, tuple)) and len(value) == 2 and second not in arguments:
            arguments[first], arguments[second] = value
    return arguments


class _Holo4ActionSpace(BaseActionSpace):
    """Shared flat-schema filtering and finish/shell projection."""

    _SCHEMA_ACTIONS: ClassVar[dict[str, frozenset[str]]]

    @classmethod
    def get_action_names(cls) -> frozenset[str]:
        return frozenset().union(*cls._SCHEMA_ACTIONS.values())

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        allowed = set(valid_actions)
        return [
            schema
            for schema in schemas
            if cls._SCHEMA_ACTIONS.get(tool_schema_name(schema), frozenset()) <= allowed
        ]

    @staticmethod
    def _project_finish_from_agent(
        content: str,
        active_extra_tool_names: set[str],
    ) -> dict[str, Any]:
        # The Holo4 prompt declares a task infeasible with a final answer
        # whose last line is exactly ``[TASK=INFEASIBLE]``; canonical Lite
        # declares it with ``terminate(status="failure")``, which the envs
        # score as the infeasible verdict.
        if (
            "terminate" in active_extra_tool_names
            and content.strip().rpartition("\n")[2].strip() == _INFEASIBLE_ANSWER_MARKER
        ):
            return LiteFinishToolSet.terminate(status="failure", reason=content)
        if "response" in active_extra_tool_names:
            return LiteFinishToolSet.response(text=content)
        if "terminate" in active_extra_tool_names:
            return LiteFinishToolSet.terminate(status="success")
        raise ValueError("answer requires active response or terminate extra tool")

    @staticmethod
    def _project_extra_to_agent(tool_call: dict[str, Any]) -> dict[str, Any] | None:
        name = tool_call_name(tool_call)
        arguments = tool_call_arguments(tool_call)
        if name == BASH_TOOL_NAME:
            return {"name": "shell", "arguments": {"command": arguments["command"]}}
        if name == "response":
            return {"name": "answer", "arguments": {"content": arguments["text"]}}
        if name == "terminate":
            content = arguments.get("reason") or "Task completed."
            return {"name": "answer", "arguments": {"content": content}}
        return None


@dataclasses.dataclass
class Holo4DesktopActionSpace(
    _Holo4ActionSpace,
    key=r"holo4@(desktop|browser)",
):
    """Holo4 desktop/browser flat named tools."""

    platform: str = "desktop"
    _SCHEMA_ACTIONS: ClassVar[dict[str, frozenset[str]]] = _DESKTOP_SCHEMA_ACTIONS

    @staticmethod
    @tool(
        element="Visible element to click.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
        button="Mouse button to click.",
    )
    def click_desktop(
        element: str,
        x: int,
        y: int,
        button: Literal["left", "right", "middle"] = "left",
    ) -> dict[str, Any]:
        """Click a visible desktop element."""
        return LiteDesktopActionSet.click(coordinate=[x, y], button=button)

    @staticmethod
    @tool(
        element="Visible element to double-click.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
    )
    def double_click_desktop(element: str, x: int, y: int) -> dict[str, Any]:
        """Double-click a visible desktop element."""
        return LiteDesktopActionSet.click(coordinate=[x, y], clicks=2)

    @staticmethod
    @tool(
        element="Visible element to triple-click.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
    )
    def triple_click_desktop(element: str, x: int, y: int) -> dict[str, Any]:
        """Triple-click a visible desktop element."""
        return LiteDesktopActionSet.click(coordinate=[x, y], clicks=3)

    @staticmethod
    @tool(
        element="Visible element to move the pointer to.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
    )
    def move_to_desktop(element: str, x: int, y: int) -> dict[str, Any]:
        """Move the pointer to a visible desktop element."""
        return LiteDesktopActionSet.mouse_move(coordinate=[x, y])

    @staticmethod
    @tool(button="Mouse button to hold down.")
    def mouse_down_desktop(
        button: Literal["left", "right", "middle"] = "left",
    ) -> dict[str, Any]:
        """Press and hold a mouse button at the current pointer position."""
        return LiteDesktopActionSet.mouse_down(button=button)

    @staticmethod
    @tool(button="Mouse button to release.")
    def mouse_up_desktop(
        button: Literal["left", "right", "middle"] = "left",
    ) -> dict[str, Any]:
        """Release a held mouse button."""
        return LiteDesktopActionSet.mouse_up(button=button)

    @staticmethod
    @tool(
        element="Visible scrollable element.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
        direction="Scroll direction.",
        scroll_size="Number of scroll units.",
    )
    def scroll_desktop(
        element: str,
        x: int,
        y: int,
        direction: Literal["up", "down", "left", "right"],
        scroll_size: int,
    ) -> dict[str, Any]:
        """Scroll a visible desktop region."""
        return LiteDesktopActionSet.scroll(
            coordinate=[x, y], direction=direction, amount=scroll_size
        )

    @staticmethod
    @tool(
        description="What is being dragged and where.",
        x1="Horizontal start coordinate normalized to [0, 1000].",
        y1="Vertical start coordinate normalized to [0, 1000].",
        x2="Horizontal end coordinate normalized to [0, 1000].",
        y2="Vertical end coordinate normalized to [0, 1000].",
        hold_keys="Keys to hold during the drag.",
    )
    def drag_and_drop(
        description: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        hold_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Drag from one point to another."""
        # The official wire name carries no ``_desktop`` suffix. The official
        # ``hold_keys`` argument is declared but not carried: canonical ``drag``
        # holds no modifiers, and 192 of 193 official calls use none.
        return LiteDesktopActionSet.drag(start_coordinate=[x1, y1], coordinate=[x2, y2])

    @staticmethod
    @tool(
        element="Visible element to drag to.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
    )
    def drag_to_desktop(element: str, x: int, y: int) -> dict[str, Any]:
        """Drag from the current pointer position to a visible element."""
        return LiteDesktopActionSet.drag(coordinate=[x, y])

    @staticmethod
    @tool(
        keys="Keys to press together.",
        repeat_count="Number of times to press the chord.",
    )
    def hotkey_desktop(keys: list[str], repeat_count: int = 1) -> dict[str, Any]:
        """Press and release a desktop key chord."""
        if repeat_count < 1:
            raise ValueError("repeat_count must be at least 1")
        return merge_adjacent_lite_action_batches(
            [LiteDesktopActionSet.key(keys=keys) for _ in range(repeat_count)]
        )[0]

    @staticmethod
    @tool(key="Key to press and hold.")
    def key_down_desktop(key: str) -> dict[str, Any]:
        """Press and hold a key."""
        return LiteDesktopActionSet.key_down(keys=[key])

    @staticmethod
    @tool(key="Held key to release.")
    def key_up_desktop(key: str) -> dict[str, Any]:
        """Release a held key."""
        return LiteDesktopActionSet.key_up(keys=[key])

    @staticmethod
    @tool(
        hold_keys="Keys to press and hold.",
        tap_keys="Keys to tap while the held keys are down.",
    )
    def hold_and_tap_key_desktop(
        hold_keys: list[str],
        tap_keys: list[str],
    ) -> dict[str, Any]:
        """Hold modifiers down, tap keys, then release: one chord."""
        return LiteDesktopActionSet.key(keys=[*hold_keys, *tap_keys])

    @staticmethod
    @tool(
        content="Text to write.",
        overwrite="Whether to replace the current field contents.",
        press_enter="Whether to press Enter after writing.",
    )
    def write_desktop(
        content: str,
        overwrite: bool = False,
        press_enter: bool = False,
    ) -> dict[str, Any]:
        """Write text into the focused desktop field."""
        calls = []
        if overwrite:
            calls.append(LiteDesktopActionSet.key(keys=["ctrl", "a"]))
        # Canonical ``type`` carries submit as a trailing newline in the text.
        calls.append(LiteDesktopActionSet.type(text=f"{content}\n" if press_enter else content))
        return merge_adjacent_lite_action_batches(calls)[0]

    @staticmethod
    @tool(
        content="Text to write.",
        element="Visible text field.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
        overwrite="Whether to replace the current field contents.",
        press_enter="Whether to press Enter after writing.",
    )
    def write_at_desktop(
        content: str,
        element: str,
        x: int,
        y: int,
        overwrite: bool = False,
        press_enter: bool = False,
    ) -> dict[str, Any]:
        """Focus a visible desktop field and write text into it."""
        calls = [LiteDesktopActionSet.click(coordinate=[x, y])]
        if overwrite:
            calls.append(LiteDesktopActionSet.key(keys=["ctrl", "a"]))
        # Canonical ``type`` carries submit as a trailing newline in the text.
        calls.append(LiteDesktopActionSet.type(text=f"{content}\n" if press_enter else content))
        return merge_adjacent_lite_action_batches(calls)[0]

    @staticmethod
    @tool(seconds="Time to wait in seconds.")
    def wait_desktop(seconds: float) -> dict[str, Any]:
        """Wait for the desktop to settle."""
        return LiteDesktopActionSet.wait(duration=seconds)

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        *,
        active_extra_tool_names: set[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        converters = {
            "click_desktop": self.click_desktop,
            "double_click_desktop": self.double_click_desktop,
            "triple_click_desktop": self.triple_click_desktop,
            "move_to_desktop": self.move_to_desktop,
            "mouse_down_desktop": self.mouse_down_desktop,
            "mouse_up_desktop": self.mouse_up_desktop,
            "scroll_desktop": self.scroll_desktop,
            "drag_and_drop": self.drag_and_drop,
            "drag_to_desktop": self.drag_to_desktop,
            "hotkey_desktop": self.hotkey_desktop,
            "key_down_desktop": self.key_down_desktop,
            "key_up_desktop": self.key_up_desktop,
            "hold_and_tap_key_desktop": self.hold_and_tap_key_desktop,
            "write_desktop": self.write_desktop,
            "write_at_desktop": self.write_at_desktop,
            "wait_desktop": self.wait_desktop,
        }
        validated = super().convert_tool_calls_from_agent(agent_tool_calls, **kwargs)
        active_extras = active_extra_tool_names or set()
        result: list[dict[str, Any]] = []
        for call in validated:
            name = tool_call_name(call)
            arguments = {**tool_call_arguments(call)}
            try:
                if name in converters:
                    result.append(converters[name](**_normalize_coordinate_arguments(arguments)))
                elif name == "shell" and BASH_TOOL_NAME in active_extras:
                    result.append(make_tool_call(BASH_TOOL_NAME, {"command": arguments["command"]}))
                elif name == "answer":
                    result.append(
                        self._project_finish_from_agent(arguments["content"], active_extras)
                    )
                elif name in active_extras:
                    result.append(make_tool_call(name, arguments))
                else:
                    raise ModelToolCallParseError(f"unsupported Holo4 desktop tool {name!r}")
            except ModelToolCallParseError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelToolCallParseError(
                    f"invalid Holo4 desktop tool {name!r}: {exc}"
                ) from exc
        return merge_adjacent_lite_action_batches(result)

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Render canonical calls in the Holo4 desktop vocabulary.

        A canonical-form renderer, not an inverse of the parse direction:
        expanded lowerings render back as their primitive spellings
        (``write_at_desktop`` as click/hotkey/write, for example).
        """
        result: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            projected_extra = self._project_extra_to_agent(tool_call)
            if projected_extra is not None:
                result.append(projected_extra)
                continue
            if tool_call_name(tool_call) != LITE_COMPUTER_ACTION_BATCH_TOOL_NAME:
                # lite -> agent re-emits an open set: a name outside the
                # canonical batch is the env extra-tool channel, so it passes
                # through as bare agent wire instead of raising.
                result.append(
                    {
                        "name": tool_call_name(tool_call),
                        "arguments": tool_call_arguments(tool_call),
                    }
                )
                continue
            for child in unpack_action_batch_call(tool_call):
                name = child["name"]
                arguments = child["arguments"]
                coordinate = arguments.get("coordinate")
                if name == "click":
                    # double/triple_click_desktop carry no ``button`` argument;
                    # official calls used the default left button throughout.
                    click_arguments: dict[str, Any] = {
                        "element": "screen coordinate",
                        "x": coordinate[0],
                        "y": coordinate[1],
                    }
                    clicks = arguments.get("clicks", 1)
                    if clicks == 2:
                        result.append(
                            {"name": "double_click_desktop", "arguments": click_arguments}
                        )
                    elif clicks == 3:
                        result.append(
                            {"name": "triple_click_desktop", "arguments": click_arguments}
                        )
                    else:
                        click_arguments["button"] = arguments.get("button", "left")
                        result.append({"name": "click_desktop", "arguments": click_arguments})
                elif name == "mouse_move":
                    result.append(
                        {
                            "name": "move_to_desktop",
                            "arguments": {
                                "element": "screen coordinate",
                                "x": coordinate[0],
                                "y": coordinate[1],
                            },
                        }
                    )
                elif name == "mouse_down":
                    result.append(
                        {
                            "name": "mouse_down_desktop",
                            "arguments": {"button": arguments.get("button", "left")},
                        }
                    )
                elif name == "mouse_up":
                    result.append(
                        {
                            "name": "mouse_up_desktop",
                            "arguments": {"button": arguments.get("button", "left")},
                        }
                    )
                elif name == "drag":
                    start = arguments.get("start_coordinate")
                    if start:
                        result.append(
                            {
                                "name": "drag_and_drop",
                                "arguments": {
                                    "description": "screen element",
                                    "x1": start[0],
                                    "y1": start[1],
                                    "x2": coordinate[0],
                                    "y2": coordinate[1],
                                },
                            }
                        )
                    else:
                        result.append(
                            {
                                "name": "drag_to_desktop",
                                "arguments": {
                                    "element": "screen element",
                                    "x": coordinate[0],
                                    "y": coordinate[1],
                                },
                            }
                        )
                elif name == "key_down":
                    result.append(
                        {"name": "key_down_desktop", "arguments": {"key": arguments["keys"][0]}}
                    )
                elif name == "key_up":
                    result.append(
                        {"name": "key_up_desktop", "arguments": {"key": arguments["keys"][0]}}
                    )
                elif name == "scroll":
                    result.append(
                        {
                            "name": "scroll_desktop",
                            "arguments": {
                                "element": "scrollable region",
                                "x": coordinate[0] if coordinate else 500,
                                "y": coordinate[1] if coordinate else 500,
                                "direction": arguments["direction"],
                                "scroll_size": arguments["amount"],
                            },
                        }
                    )
                elif name == "key":
                    result.append(
                        {"name": "hotkey_desktop", "arguments": {"keys": arguments["keys"]}}
                    )
                elif name == "type":
                    # Canonical ``type``'s trailing newline is Holo4's
                    # ``press_enter``; project it back so replayed history
                    # reads like the model's own calls.
                    content = arguments["text"]
                    press_enter = content.endswith("\n")
                    result.append(
                        {
                            "name": "write_desktop",
                            "arguments": {
                                "content": content[:-1] if press_enter else content,
                                "overwrite": False,
                                "press_enter": press_enter,
                            },
                        }
                    )
                elif name == "wait":
                    result.append(
                        {"name": "wait_desktop", "arguments": {"seconds": arguments["duration"]}}
                    )
                else:
                    result.append({"name": name, "arguments": arguments})
        return result


@dataclasses.dataclass
class Holo4MobileActionSpace(_Holo4ActionSpace, key="holo4@mobile"):
    """Holo4 mobile flat named tools."""

    platform: str = "mobile"
    _SCHEMA_ACTIONS: ClassVar[dict[str, frozenset[str]]] = _MOBILE_SCHEMA_ACTIONS

    @staticmethod
    @tool(
        element="Visible element to tap.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
    )
    def mobile_click(element: str, x: int, y: int) -> dict[str, Any]:
        """Tap a visible mobile element."""
        return LiteMobileActionSet.tap(coordinate=[x, y])

    @staticmethod
    @tool(
        element="Visible text field.",
        text="Text to write.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
        overwrite="Has no effect: the written text is appended to the field's current contents.",
        enter="Whether to press Enter after writing.",
    )
    def mobile_write(
        element: str,
        text: str,
        x: int,
        y: int,
        overwrite: bool = False,
        enter: bool = False,
    ) -> dict[str, Any]:
        """Focus a visible mobile field and write text into it."""
        # ``overwrite`` is accepted so official-tool history replays, but it is
        # not carried: canonical mobile ``type`` appends to whatever the
        # focused field already holds.
        calls = [
            LiteMobileActionSet.tap(coordinate=[x, y]),
            LiteMobileActionSet.type(text=text),
        ]
        if enter:
            calls.append(LiteMobileActionSet.system_button(button="Enter"))
        return merge_adjacent_lite_action_batches(calls)[0]

    @staticmethod
    @tool(direction="Scroll direction.", factor="Fraction of the screen to scroll.")
    def mobile_scroll(
        direction: Literal["up", "down", "left", "right"],
        factor: float,
    ) -> dict[str, Any]:
        """Scroll the current mobile view."""
        # The official runtime's factor-to-swipe mapping is closed-source; this
        # approximation clamps the [0, 1000]-axis distance to [100, 800] and
        # swipes through the screen center.
        distance = max(100, min(800, round(factor * 1000)))
        half = distance // 2
        start = [500, 500]
        end = [500, 500]
        if direction == "down":
            start, end = [500, 500 + half], [500, 500 - half]
        elif direction == "up":
            start, end = [500, 500 - half], [500, 500 + half]
        elif direction == "right":
            start, end = [500 + half, 500], [500 - half, 500]
        elif direction == "left":
            start, end = [500 - half, 500], [500 + half, 500]
        return LiteMobileActionSet.swipe(start_coordinate=start, coordinate=end)

    @staticmethod
    @tool(
        x_touch="Starting horizontal coordinate normalized to [0, 1000].",
        y_touch="Starting vertical coordinate normalized to [0, 1000].",
        x_lift="Ending horizontal coordinate normalized to [0, 1000].",
        y_lift="Ending vertical coordinate normalized to [0, 1000].",
    )
    def mobile_swipe(
        x_touch: int,
        y_touch: int,
        x_lift: int,
        y_lift: int,
    ) -> dict[str, Any]:
        """Swipe between two mobile screen coordinates."""
        return LiteMobileActionSet.swipe(
            start_coordinate=[x_touch, y_touch], coordinate=[x_lift, y_lift]
        )

    @staticmethod
    @tool(
        element="Visible element to long press.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
        duration="Press duration in milliseconds.",
    )
    def mobile_long_press(
        element: str,
        x: int,
        y: int,
        duration: int = 1000,
    ) -> dict[str, Any]:
        """Long press a visible mobile element."""
        return LiteMobileActionSet.long_press(coordinate=[x, y], duration=duration / 1000)

    @staticmethod
    @tool(
        element="Visible element to drag.",
        x_touch="Starting horizontal coordinate normalized to [0, 1000].",
        y_touch="Starting vertical coordinate normalized to [0, 1000].",
        x_lift="Ending horizontal coordinate normalized to [0, 1000].",
        y_lift="Ending vertical coordinate normalized to [0, 1000].",
        duration="Drag duration in milliseconds.",
    )
    def mobile_drag(
        element: str,
        x_touch: int,
        y_touch: int,
        x_lift: int,
        y_lift: int,
        duration: int = 400,
    ) -> dict[str, Any]:
        """Drag a visible mobile element."""
        # The official ``duration`` argument is not carried: canonical mobile
        # ``drag`` defines no duration.
        return LiteMobileActionSet.drag(
            start_coordinate=[x_touch, y_touch],
            coordinate=[x_lift, y_lift],
        )

    @staticmethod
    @tool()
    def mobile_go_home() -> dict[str, Any]:
        """Open the mobile home screen."""
        return LiteMobileActionSet.system_button(button="Home")

    @staticmethod
    @tool()
    def mobile_go_back() -> dict[str, Any]:
        """Navigate back on the mobile device."""
        return LiteMobileActionSet.system_button(button="Back")

    @staticmethod
    @tool()
    def mobile_hide_keyboard() -> dict[str, Any]:
        """Hide the mobile soft keyboard."""
        return LiteMobileActionSet.system_button(button="Back")

    @staticmethod
    @tool()
    def mobile_go_to_all_apps() -> dict[str, Any]:
        """Open the mobile app drawer."""
        return merge_adjacent_lite_action_batches(
            [
                LiteMobileActionSet.system_button(button="Home"),
                LiteMobileActionSet.swipe(start_coordinate=[500, 850], coordinate=[500, 200]),
            ]
        )[0]

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        *,
        active_extra_tool_names: set[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        validated = super().convert_tool_calls_from_agent(agent_tool_calls, **kwargs)
        active_extras = active_extra_tool_names or set()
        result: list[dict[str, Any]] = []
        converters = {
            "mobile_click": self.mobile_click,
            "mobile_write": self.mobile_write,
            "mobile_scroll": self.mobile_scroll,
            "mobile_swipe": self.mobile_swipe,
            "mobile_long_press": self.mobile_long_press,
            "mobile_drag": self.mobile_drag,
            "mobile_go_home": self.mobile_go_home,
            "mobile_go_back": self.mobile_go_back,
            "mobile_hide_keyboard": self.mobile_hide_keyboard,
            "mobile_go_to_all_apps": self.mobile_go_to_all_apps,
        }
        for call in validated:
            name = tool_call_name(call)
            arguments = {**tool_call_arguments(call)}
            try:
                if name in converters:
                    arguments = _normalize_coordinate_arguments(arguments)
                    if name == "mobile_swipe" and "x_touch" not in arguments:
                        arguments = {
                            "x_touch": arguments["x"],
                            "y_touch": arguments["y"],
                            "x_lift": arguments["x_lift"],
                            "y_lift": arguments["y_lift"],
                        }
                    result.append(converters[name](**arguments))
                elif name == "shell" and BASH_TOOL_NAME in active_extras:
                    result.append(make_tool_call(BASH_TOOL_NAME, {"command": arguments["command"]}))
                elif name == "answer":
                    result.append(
                        self._project_finish_from_agent(arguments["content"], active_extras)
                    )
                elif name in active_extras:
                    result.append(make_tool_call(name, arguments))
                else:
                    raise ModelToolCallParseError(f"unsupported Holo4 mobile tool {name!r}")
            except ModelToolCallParseError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelToolCallParseError(f"invalid Holo4 mobile tool {name!r}: {exc}") from exc
        return merge_adjacent_lite_action_batches(result)

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Render canonical calls in the Holo4 mobile vocabulary.

        A canonical-form renderer, not an inverse of the parse direction:
        ``mobile_hide_keyboard`` renders as ``mobile_go_back`` and
        ``mobile_go_to_all_apps`` as home + swipe, for example.
        """
        result: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            projected_extra = self._project_extra_to_agent(tool_call)
            if projected_extra is not None:
                result.append(projected_extra)
                continue
            if tool_call_name(tool_call) != LITE_MOBILE_ACTION_BATCH_TOOL_NAME:
                # lite -> agent re-emits an open set: a name outside the
                # canonical batch is the env extra-tool channel, so it passes
                # through as bare agent wire instead of raising.
                result.append(
                    {
                        "name": tool_call_name(tool_call),
                        "arguments": tool_call_arguments(tool_call),
                    }
                )
                continue
            children = unpack_action_batch_call(tool_call)
            child_index = 0
            while child_index < len(children):
                child = children[child_index]
                name = child["name"]
                arguments = child["arguments"]
                coordinate = arguments.get("coordinate")
                next_child = children[child_index + 1] if child_index + 1 < len(children) else None
                if name == "tap" and next_child and next_child["name"] == "type":
                    enter_child = (
                        children[child_index + 2] if child_index + 2 < len(children) else None
                    )
                    press_enter = bool(
                        enter_child
                        and enter_child["name"] == "system_button"
                        and enter_child["arguments"].get("button") == "Enter"
                    )
                    write_arguments = {
                        "element": "screen field",
                        "text": next_child["arguments"]["text"],
                        "x": coordinate[0],
                        "y": coordinate[1],
                        "enter": press_enter,
                    }
                    result.append({"name": "mobile_write", "arguments": write_arguments})
                    child_index += 3 if press_enter else 2
                    continue
                if name == "tap":
                    result.append(
                        {
                            "name": "mobile_click",
                            "arguments": {
                                "element": "screen coordinate",
                                "x": coordinate[0],
                                "y": coordinate[1],
                            },
                        }
                    )
                elif name == "type":
                    # Out-of-family canonical input only: in-family
                    # ``mobile_write`` lowers to tap + type, so a bare type
                    # renders as mobile_write with screen-center placeholder
                    # coordinates for the focused field.
                    write_arguments = {
                        "element": "focused field",
                        "text": arguments["text"],
                        "x": 500,
                        "y": 500,
                        "enter": False,
                    }
                    result.append({"name": "mobile_write", "arguments": write_arguments})
                elif name == "swipe":
                    start = arguments["start_coordinate"]
                    result.append(
                        {
                            "name": "mobile_swipe",
                            "arguments": {
                                "x_touch": start[0],
                                "y_touch": start[1],
                                "x_lift": coordinate[0],
                                "y_lift": coordinate[1],
                            },
                        }
                    )
                elif name == "long_press":
                    long_press_arguments = {
                        "element": "screen coordinate",
                        "x": coordinate[0],
                        "y": coordinate[1],
                    }
                    if "duration" in arguments:
                        long_press_arguments["duration"] = round(arguments["duration"] * 1000)
                    result.append(
                        {
                            "name": "mobile_long_press",
                            "arguments": long_press_arguments,
                        }
                    )
                elif name == "drag":
                    start = arguments["start_coordinate"]
                    drag_arguments = {
                        "element": "screen element",
                        "x_touch": start[0],
                        "y_touch": start[1],
                        "x_lift": coordinate[0],
                        "y_lift": coordinate[1],
                    }
                    result.append(
                        {
                            "name": "mobile_drag",
                            "arguments": drag_arguments,
                        }
                    )
                elif name == "system_button" and arguments["button"] == "Home":
                    result.append({"name": "mobile_go_home", "arguments": {}})
                elif name == "system_button" and arguments["button"] == "Back":
                    result.append({"name": "mobile_go_back", "arguments": {}})
                else:
                    result.append({"name": name, "arguments": arguments})
                child_index += 1
        return result


__all__ = ["Holo4DesktopActionSpace", "Holo4MobileActionSpace"]
