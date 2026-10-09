"""CPU-only contract tests for Holo4 action spaces."""

from __future__ import annotations

import pytest

from lite.agents.models.holo4.action_space import (
    Holo4DesktopActionSpace,
    Holo4MobileActionSpace,
)
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters


def _actions(call: dict) -> list[dict]:
    return tool_call_arguments(call)["actions"]


def test_desktop_schema_is_the_holo_flat_surface() -> None:
    space = Holo4DesktopActionSpace()
    assert space.get_tool_names() == {
        "click_desktop",
        "move_to_desktop",
        "scroll_desktop",
        "hotkey_desktop",
        "write_desktop",
        "write_at_desktop",
        "wait_desktop",
    }
    params = tool_schema_parameters(space.get_tool_schema("write_at_desktop"))
    assert params["required"] == ["content", "element", "x", "y"]
    assert set(params["properties"]) == {
        "content",
        "element",
        "x",
        "y",
        "overwrite",
        "press_enter",
    }


def test_mobile_schema_is_the_holo_flat_surface() -> None:
    assert Holo4MobileActionSpace.get_tool_names() == {
        "mobile_click",
        "mobile_write",
        "mobile_scroll",
        "mobile_swipe",
        "mobile_long_press",
        "mobile_drag",
        "mobile_go_home",
        "mobile_go_back",
        "mobile_hide_keyboard",
        "mobile_go_to_all_apps",
    }
    write_properties = tool_schema_parameters(
        Holo4MobileActionSpace.get_tool_schema("mobile_write")
    )["properties"]
    drag_properties = tool_schema_parameters(
        Holo4MobileActionSpace.get_tool_schema("mobile_drag")
    )["properties"]
    assert "overwrite" not in write_properties
    assert "duration" not in drag_properties


def test_valid_actions_filter_uses_canonical_action_names() -> None:
    space = Holo4DesktopActionSpace()
    filtered = space.filter_tool_schemas_for_valid_actions(
        space.get_tool_schemas(), ["click", "wait"]
    )
    assert [tool_schema_name(schema) for schema in filtered] == [
        "click_desktop",
        "wait_desktop",
    ]


def test_write_at_desktop_expands_to_one_canonical_batch() -> None:
    calls = Holo4DesktopActionSpace().convert_tool_calls_from_agent(
        [
            {
                "name": "write_at_desktop",
                "arguments": {
                    "content": "hello",
                    "element": "Name",
                    "x": 420,
                    "y": 310,
                    "overwrite": True,
                    "press_enter": True,
                },
            }
        ]
    )
    assert len(calls) == 1
    assert _actions(calls[0]) == [
        {"action": "click", "coordinate": [420, 310]},
        {"action": "key", "keys": ["ctrl", "a"]},
        {"action": "type", "text": "hello", "press_enter": True},
    ]


def test_shell_requires_active_bash_and_projects_to_canonical_name() -> None:
    call = {"name": "shell", "arguments": {"command": "pwd"}}
    space = Holo4DesktopActionSpace()
    with pytest.raises(ValueError, match="unsupported Holo4 desktop tool"):
        space.convert_tool_calls_from_agent([call], active_extra_tool_names=set())
    converted = space.convert_tool_calls_from_agent([call], active_extra_tool_names={"bash"})
    assert tool_call_name(converted[0]) == "bash"
    assert tool_call_arguments(converted[0]) == {"command": "pwd"}


def test_answer_prefers_response_then_falls_back_to_terminate() -> None:
    call = {"name": "answer", "arguments": {"content": "done"}}
    space = Holo4DesktopActionSpace()
    response = space.convert_tool_calls_from_agent(
        [call], active_extra_tool_names={"response", "terminate"}
    )
    assert response == [LiteFinishToolSet.response(text="done")]
    terminate = space.convert_tool_calls_from_agent([call], active_extra_tool_names={"terminate"})
    assert terminate == [LiteFinishToolSet.terminate(status="success")]


def test_desktop_canonical_calls_render_as_holo_calls() -> None:
    canonical = Holo4DesktopActionSpace.write_at_desktop("hello", "Name", 420, 310, True, True)
    rendered = Holo4DesktopActionSpace().convert_tool_calls_to_agent([canonical])
    assert [call["name"] for call in rendered] == [
        "click_desktop",
        "hotkey_desktop",
        "write_desktop",
    ]


def test_mobile_calls_lower_to_canonical_actions() -> None:
    calls = Holo4MobileActionSpace().convert_tool_calls_from_agent(
        [
            {
                "name": "mobile_write",
                "arguments": {
                    "element": "Name",
                    "text": "Ada",
                    "x": 500,
                    "y": 300,
                    "enter": True,
                },
            },
            {
                "name": "mobile_long_press",
                "arguments": {
                    "element": "file",
                    "x": 200,
                    "y": 400,
                    "duration": 2000,
                },
            },
        ]
    )
    assert len(calls) == 1
    assert _actions(calls[0]) == [
        {"action": "tap", "coordinate": [500, 300], "clicks": 1},
        {"action": "type", "text": "Ada"},
        {"action": "system_button", "button": "Enter"},
        {"action": "long_press", "coordinate": [200, 400], "duration": 2.0},
    ]
    assert Holo4MobileActionSpace().convert_tool_calls_to_agent(calls) == [
        {
            "name": "mobile_write",
            "arguments": {
                "element": "screen field",
                "text": "Ada",
                "x": 500,
                "y": 300,
                "enter": True,
            },
        },
        {
            "name": "mobile_long_press",
            "arguments": {
                "element": "screen coordinate",
                "x": 200,
                "y": 400,
            },
        },
    ]


def test_mobile_swipe_round_trip() -> None:
    native = {
        "name": "mobile_swipe",
        "arguments": {"x_touch": 800, "y_touch": 500, "x_lift": 200, "y_lift": 500},
    }
    canonical = Holo4MobileActionSpace().convert_tool_calls_from_agent([native])
    rendered = Holo4MobileActionSpace().convert_tool_calls_to_agent(canonical)
    assert rendered == [native]


def test_canonical_finish_and_shell_render_with_holo_names() -> None:
    calls = [
        make_tool_call("bash", {"command": "ls"}),
        LiteFinishToolSet.response(text="found it"),
    ]
    assert Holo4DesktopActionSpace().convert_tool_calls_to_agent(calls) == [
        {"name": "shell", "arguments": {"command": "ls"}},
        {"name": "answer", "arguments": {"content": "found it"}},
    ]


def test_unknown_canonical_desktop_tool_fails_on_render() -> None:
    with pytest.raises(ValueError, match="unsupported canonical desktop tool"):
        Holo4DesktopActionSpace().convert_tool_calls_to_agent(
            [make_tool_call("totally_bogus_action", {"foo": 1})]
        )


def test_unknown_canonical_mobile_tool_fails_on_render() -> None:
    with pytest.raises(ValueError, match="unsupported canonical mobile tool"):
        Holo4MobileActionSpace().convert_tool_calls_to_agent(
            [make_tool_call("totally_bogus_action", {"foo": 1})]
        )
