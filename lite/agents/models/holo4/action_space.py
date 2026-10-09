"""Holo4 flat tool dialects and canonical Lite action conversion."""

from __future__ import annotations

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
    "move_to_desktop": frozenset({"mouse_move"}),
    "scroll_desktop": frozenset({"scroll"}),
    "hotkey_desktop": frozenset({"key"}),
    "write_desktop": frozenset({"type"}),
    "write_at_desktop": frozenset({"click", "type"}),
    "wait_desktop": frozenset({"wait"}),
}

_MOBILE_SCHEMA_ACTIONS: dict[str, frozenset[str]] = {
    "mobile_click": frozenset({"tap"}),
    "mobile_write": frozenset({"tap", "type"}),
    "mobile_scroll": frozenset({"swipe"}),
    "mobile_swipe": frozenset({"swipe"}),
    "mobile_long_press": frozenset({"long_press"}),
    "mobile_drag": frozenset({"drag"}),
    "mobile_go_home": frozenset({"system_button"}),
    "mobile_go_back": frozenset({"system_button"}),
    "mobile_hide_keyboard": frozenset({"system_button"}),
    "mobile_go_to_all_apps": frozenset({"system_button", "swipe"}),
}


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
        element="Visible element to move the pointer to.",
        x="Horizontal coordinate normalized to [0, 1000].",
        y="Vertical coordinate normalized to [0, 1000].",
    )
    def move_to_desktop(element: str, x: int, y: int) -> dict[str, Any]:
        """Move the pointer to a visible desktop element."""
        return LiteDesktopActionSet.mouse_move(coordinate=[x, y])

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
    @tool(keys="Keys to press together.")
    def hotkey_desktop(keys: list[str]) -> dict[str, Any]:
        """Press and release a desktop key chord."""
        return LiteDesktopActionSet.key(keys=keys)

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
        calls.append(LiteDesktopActionSet.type(text=content, press_enter=press_enter))
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
        calls.append(LiteDesktopActionSet.type(text=content, press_enter=press_enter))
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
        validated = super().convert_tool_calls_from_agent(agent_tool_calls, **kwargs)
        active_extras = active_extra_tool_names or set()
        result: list[dict[str, Any]] = []
        for call in validated:
            name = tool_call_name(call)
            arguments = tool_call_arguments(call)
            try:
                if name == "click_desktop":
                    result.append(self.click_desktop(**arguments))
                elif name == "move_to_desktop":
                    result.append(self.move_to_desktop(**arguments))
                elif name == "scroll_desktop":
                    result.append(self.scroll_desktop(**arguments))
                elif name == "hotkey_desktop":
                    result.append(self.hotkey_desktop(**arguments))
                elif name == "write_desktop":
                    result.append(self.write_desktop(**arguments))
                elif name == "write_at_desktop":
                    result.append(self.write_at_desktop(**arguments))
                elif name == "wait_desktop":
                    result.append(self.wait_desktop(**arguments))
                elif name == "shell" and BASH_TOOL_NAME in active_extras:
                    result.append(make_tool_call(BASH_TOOL_NAME, {"command": arguments["command"]}))
                elif name == "answer":
                    result.append(
                        self._project_finish_from_agent(arguments["content"], active_extras)
                    )
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
        result: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            projected_extra = self._project_extra_to_agent(tool_call)
            if projected_extra is not None:
                result.append(projected_extra)
                continue
            if tool_call_name(tool_call) != LITE_COMPUTER_ACTION_BATCH_TOOL_NAME:
                raise ValueError(
                    f"unsupported canonical desktop tool {tool_call_name(tool_call)!r}"
                )
            for child in unpack_action_batch_call(tool_call):
                name = child["name"]
                arguments = child["arguments"]
                coordinate = arguments.get("coordinate")
                if name == "click":
                    result.append(
                        {
                            "name": "click_desktop",
                            "arguments": {
                                "element": "screen coordinate",
                                "x": coordinate[0],
                                "y": coordinate[1],
                                "button": arguments.get("button", "left"),
                            },
                        }
                    )
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
                    result.append(
                        {
                            "name": "write_desktop",
                            "arguments": {
                                "content": arguments["text"],
                                "overwrite": False,
                                "press_enter": bool(arguments.get("press_enter")),
                            },
                        }
                    )
                elif name == "wait":
                    result.append(
                        {"name": "wait_desktop", "arguments": {"seconds": arguments["duration"]}}
                    )
                else:
                    raise ValueError(f"unsupported canonical desktop action {name!r}")
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
        enter="Whether to press Enter after writing.",
    )
    def mobile_write(
        element: str,
        text: str,
        x: int,
        y: int,
        enter: bool = False,
    ) -> dict[str, Any]:
        """Focus a visible mobile field and write text into it."""
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
    )
    def mobile_drag(
        element: str,
        x_touch: int,
        y_touch: int,
        x_lift: int,
        y_lift: int,
    ) -> dict[str, Any]:
        """Drag a visible mobile element."""
        return LiteMobileActionSet.drag(
            start_coordinate=[x_touch, y_touch], coordinate=[x_lift, y_lift]
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
            arguments = tool_call_arguments(call)
            try:
                if name in converters:
                    result.append(converters[name](**arguments))
                elif name == "answer":
                    result.append(
                        self._project_finish_from_agent(arguments["content"], active_extras)
                    )
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
        result: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            projected_extra = self._project_extra_to_agent(tool_call)
            if projected_extra is not None:
                result.append(projected_extra)
                continue
            if tool_call_name(tool_call) != LITE_MOBILE_ACTION_BATCH_TOOL_NAME:
                raise ValueError(
                    f"unsupported canonical mobile tool {tool_call_name(tool_call)!r}"
                )
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
                    result.append(
                        {
                            "name": "mobile_write",
                            "arguments": {
                                "element": "screen field",
                                "text": next_child["arguments"]["text"],
                                "x": coordinate[0],
                                "y": coordinate[1],
                                "enter": press_enter,
                            },
                        }
                    )
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
                    result.append(
                        {
                            "name": "mobile_write",
                            "arguments": {
                                "element": "focused field",
                                "text": arguments["text"],
                                "x": 500,
                                "y": 500,
                                "enter": False,
                            },
                        }
                    )
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
                    result.append(
                        {
                            "name": "mobile_long_press",
                            "arguments": {
                                "element": "screen coordinate",
                                "x": coordinate[0],
                                "y": coordinate[1],
                            },
                        }
                    )
                elif name == "drag":
                    start = arguments["start_coordinate"]
                    result.append(
                        {
                            "name": "mobile_drag",
                            "arguments": {
                                "element": "screen element",
                                "x_touch": start[0],
                                "y_touch": start[1],
                                "x_lift": coordinate[0],
                                "y_lift": coordinate[1],
                            },
                        }
                    )
                elif name == "system_button" and arguments["button"] == "Home":
                    result.append({"name": "mobile_go_home", "arguments": {}})
                elif name == "system_button" and arguments["button"] == "Back":
                    result.append({"name": "mobile_go_back", "arguments": {}})
                else:
                    raise ValueError(f"unsupported canonical mobile action {name!r}")
                child_index += 1
        return result


__all__ = ["Holo4DesktopActionSpace", "Holo4MobileActionSpace"]
