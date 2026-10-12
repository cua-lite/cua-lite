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
        "double_click_desktop",
        "triple_click_desktop",
        "move_to_desktop",
        "mouse_down_desktop",
        "mouse_up_desktop",
        "scroll_desktop",
        "drag_and_drop",
        "drag_to_desktop",
        "hotkey_desktop",
        "key_down_desktop",
        "key_up_desktop",
        "hold_and_tap_key_desktop",
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


def test_official_desktop_low_frequency_tools_lower_to_canonical_actions() -> None:
    calls = Holo4DesktopActionSpace().convert_tool_calls_from_agent(
        [
            {
                "name": "double_click_desktop",
                "arguments": {"element": "icon", "x": 120, "y": 80},
            },
            {
                "name": "drag_and_drop",
                "arguments": {
                    "description": "file onto folder",
                    "x1": 100,
                    "y1": 200,
                    "x2": 700,
                    "y2": 400,
                },
            },
            {"name": "mouse_down_desktop", "arguments": {"button": "right"}},
            {"name": "key_down_desktop", "arguments": {"key": "shift"}},
            {
                "name": "hold_and_tap_key_desktop",
                "arguments": {"hold_keys": ["ctrl"], "tap_keys": ["a"]},
            },
        ]
    )
    assert len(calls) == 1
    assert _actions(calls[0]) == [
        {"action": "click", "coordinate": [120, 80], "clicks": 2},
        {"action": "drag", "start_coordinate": [100, 200], "coordinate": [700, 400]},
        {"action": "mouse_down", "button": "right"},
        {"action": "key_down", "keys": ["shift"]},
        {"action": "key", "keys": ["ctrl", "a"]},
    ]


def test_official_desktop_actions_render_with_holo_names() -> None:
    canonical = [
        Holo4DesktopActionSpace.double_click_desktop("icon", 120, 80),
        Holo4DesktopActionSpace.drag_and_drop("file onto folder", 100, 200, 700, 400),
        Holo4DesktopActionSpace.drag_to_desktop("target", 700, 400),
        Holo4DesktopActionSpace.mouse_down_desktop("right"),
        Holo4DesktopActionSpace.mouse_up_desktop("right"),
        Holo4DesktopActionSpace.key_down_desktop("shift"),
        Holo4DesktopActionSpace.key_up_desktop("shift"),
        Holo4DesktopActionSpace.hold_and_tap_key_desktop(["ctrl"], ["a"]),
    ]
    rendered = Holo4DesktopActionSpace().convert_tool_calls_to_agent(canonical)
    assert rendered == [
        {
            "name": "double_click_desktop",
            "arguments": {"element": "screen coordinate", "x": 120, "y": 80},
        },
        {
            "name": "drag_and_drop",
            "arguments": {
                "description": "screen element",
                "x1": 100,
                "y1": 200,
                "x2": 700,
                "y2": 400,
            },
        },
        {
            "name": "drag_to_desktop",
            "arguments": {"element": "screen element", "x": 700, "y": 400},
        },
        {"name": "mouse_down_desktop", "arguments": {"button": "right"}},
        {"name": "mouse_up_desktop", "arguments": {"button": "right"}},
        {"name": "key_down_desktop", "arguments": {"key": "shift"}},
        {"name": "key_up_desktop", "arguments": {"key": "shift"}},
        {"name": "hotkey_desktop", "arguments": {"keys": ["ctrl", "a"]}},
    ]


def test_hotkey_repeat_count_expands_to_repeated_key_actions() -> None:
    calls = Holo4DesktopActionSpace().convert_tool_calls_from_agent(
        [{"name": "hotkey_desktop", "arguments": {"keys": ["tab"], "repeat_count": 3}}]
    )
    assert _actions(calls[0]) == [{"action": "key", "keys": ["tab"]}] * 3
    rendered = Holo4DesktopActionSpace().convert_tool_calls_to_agent(calls)
    assert rendered == [{"name": "hotkey_desktop", "arguments": {"keys": ["tab"]}}] * 3


def test_official_desktop_string_coordinates_are_normalized() -> None:
    converted = Holo4DesktopActionSpace().convert_tool_calls_from_agent(
        [
            {
                "name": "triple_click_desktop",
                "arguments": {"element": "field", "x": "509", "y": "515"},
            }
        ]
    )
    assert _actions(converted[0]) == [{"action": "click", "coordinate": [509, 515], "clicks": 3}]


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
    drag_properties = tool_schema_parameters(Holo4MobileActionSpace.get_tool_schema("mobile_drag"))[
        "properties"
    ]
    assert "overwrite" in write_properties
    assert "duration" in drag_properties


def test_valid_actions_filter_uses_canonical_action_names() -> None:
    space = Holo4DesktopActionSpace()
    filtered = space.filter_tool_schemas_for_valid_actions(
        space.get_tool_schemas(), ["click", "wait"]
    )
    assert [tool_schema_name(schema) for schema in filtered] == [
        "click_desktop",
        "double_click_desktop",
        "triple_click_desktop",
        "wait_desktop",
    ]


def test_valid_actions_filter_tracks_conditionally_emitted_children() -> None:
    # write_desktop/write_at_desktop emit ``key`` only with overwrite=True and
    # mobile_write emits ``system_button`` only with enter=True; the filter
    # must account for the conditional children, not just the happy path.
    desktop = Holo4DesktopActionSpace()
    filtered = desktop.filter_tool_schemas_for_valid_actions(
        desktop.get_tool_schemas(), ["click", "type"]
    )
    assert [tool_schema_name(schema) for schema in filtered] == [
        "click_desktop",
        "double_click_desktop",
        "triple_click_desktop",
    ]
    mobile = Holo4MobileActionSpace()
    filtered = mobile.filter_tool_schemas_for_valid_actions(
        mobile.get_tool_schemas(), ["tap", "type"]
    )
    assert [tool_schema_name(schema) for schema in filtered] == ["mobile_click"]


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


def test_mobile_answer_lowers_to_the_active_finish_tool() -> None:
    call = {"name": "answer", "arguments": {"content": "done"}}
    assert Holo4MobileActionSpace().convert_tool_calls_from_agent(
        [call], active_extra_tool_names={"response", "terminate"}
    ) == [LiteFinishToolSet.response(text="done")]


def test_mobile_shell_requires_active_bash_and_projects_to_canonical_name() -> None:
    call = {"name": "shell", "arguments": {"command": "pwd"}}
    space = Holo4MobileActionSpace()
    with pytest.raises(ValueError, match="unsupported Holo4 mobile tool"):
        space.convert_tool_calls_from_agent([call], active_extra_tool_names=set())
    converted = space.convert_tool_calls_from_agent([call], active_extra_tool_names={"bash"})
    assert tool_call_name(converted[0]) == "bash"
    assert tool_call_arguments(converted[0]) == {"command": "pwd"}


def test_every_desktop_tool_name_dispatches_in_the_parse_direction() -> None:
    minimal_args = {
        "click_desktop": {"element": "e", "x": 1, "y": 2},
        "double_click_desktop": {"element": "e", "x": 1, "y": 2},
        "triple_click_desktop": {"element": "e", "x": 1, "y": 2},
        "move_to_desktop": {"element": "e", "x": 1, "y": 2},
        "mouse_down_desktop": {},
        "mouse_up_desktop": {},
        "scroll_desktop": {
            "element": "e",
            "x": 1,
            "y": 2,
            "direction": "down",
            "scroll_size": 3,
        },
        "drag_and_drop": {"description": "d", "x1": 1, "y1": 2, "x2": 3, "y2": 4},
        "drag_to_desktop": {"element": "e", "x": 1, "y": 2},
        "hotkey_desktop": {"keys": ["enter"]},
        "key_down_desktop": {"key": "shift"},
        "key_up_desktop": {"key": "shift"},
        "hold_and_tap_key_desktop": {"hold_keys": ["ctrl"], "tap_keys": ["a"]},
        "write_desktop": {"content": "hi"},
        "write_at_desktop": {"content": "hi", "element": "e", "x": 1, "y": 2},
        "wait_desktop": {"seconds": 1.0},
    }
    assert set(minimal_args) == set(Holo4DesktopActionSpace.get_tool_names())
    calls = Holo4DesktopActionSpace().convert_tool_calls_from_agent(
        [{"name": name, "arguments": args} for name, args in minimal_args.items()]
    )
    assert len(calls) == 1
    assert len(_actions(calls[0])) == 17  # write_at_desktop expands to click + type


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
                "duration": 2000,
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


def test_official_mobile_write_overwrite_is_dropped_in_lowering() -> None:
    converted = Holo4MobileActionSpace().convert_tool_calls_from_agent(
        [
            {
                "name": "mobile_write",
                "arguments": {
                    "element": "Amount field",
                    "text": "266.49",
                    "overwrite": True,
                    "x": 499,
                    "y": 267,
                },
            }
        ]
    )

    assert _actions(converted[0]) == [
        {"action": "tap", "coordinate": [499, 267], "clicks": 1},
        {"action": "type", "text": "266.49"},
    ]
    assert Holo4MobileActionSpace().convert_tool_calls_to_agent(converted) == [
        {
            "name": "mobile_write",
            "arguments": {
                "element": "screen field",
                "text": "266.49",
                "x": 499,
                "y": 267,
                "enter": False,
            },
        }
    ]


def test_mobile_swipe_accepts_coordinate_start_names_seen_in_model_output() -> None:
    converted = Holo4MobileActionSpace().convert_tool_calls_from_agent(
        [
            {
                "name": "mobile_swipe",
                "arguments": {"x": 500, "y": 446, "x_lift": 500, "y_lift": 278},
            }
        ]
    )

    assert _actions(converted[0]) == [
        {
            "action": "swipe",
            "start_coordinate": [500, 446],
            "coordinate": [500, 278],
        }
    ]


def test_active_mobile_extra_tool_round_trips() -> None:
    space = Holo4MobileActionSpace()
    call = {"name": "note", "arguments": {"note": "Remember 266.49"}}
    converted = space.convert_tool_calls_from_agent([call], active_extra_tool_names={"note"})
    assert tool_call_name(converted[0]) == "note"
    assert tool_call_arguments(converted[0]) == call["arguments"]


def test_open_app_survives_mobile_history_projection() -> None:
    call = make_tool_call("open_app", {"app_name": "OSMAnd"})

    assert Holo4MobileActionSpace().convert_tool_calls_to_agent([call]) == [
        {"name": "open_app", "arguments": {"app_name": "OSMAnd"}}
    ]


def test_canonical_finish_and_shell_render_with_holo_names() -> None:
    calls = [
        make_tool_call("bash", {"command": "ls"}),
        LiteFinishToolSet.response(text="found it"),
    ]
    assert Holo4DesktopActionSpace().convert_tool_calls_to_agent(calls) == [
        {"name": "shell", "arguments": {"command": "ls"}},
        {"name": "answer", "arguments": {"content": "found it"}},
    ]


def test_unknown_desktop_name_passes_through_on_render() -> None:
    # lite -> agent re-emits an open set: unknown names are the env extra-tool
    # channel and pass through as bare agent wire instead of raising.
    assert Holo4DesktopActionSpace().convert_tool_calls_to_agent(
        [make_tool_call("totally_bogus_action", {"foo": 1})]
    ) == [{"name": "totally_bogus_action", "arguments": {"foo": 1}}]


def test_unknown_mobile_name_passes_through_on_render() -> None:
    assert Holo4MobileActionSpace().convert_tool_calls_to_agent(
        [make_tool_call("totally_bogus_action", {"foo": 1})]
    ) == [{"name": "totally_bogus_action", "arguments": {"foo": 1}}]


def test_official_mobile_click_coordinate_list_is_normalized() -> None:
    converted = Holo4MobileActionSpace().convert_tool_calls_from_agent(
        [{"name": "mobile_click", "arguments": {"element": "icon", "x": [601, 925]}}]
    )
    assert _actions(converted[0]) == [{"action": "tap", "coordinate": [601, 925], "clicks": 1}]


def test_official_mobile_click_stringified_coordinate_list_is_normalized() -> None:
    converted = Holo4MobileActionSpace().convert_tool_calls_from_agent(
        [{"name": "mobile_click", "arguments": {"element": "icon", "x": "[601, 925]"}}]
    )
    assert _actions(converted[0]) == [{"action": "tap", "coordinate": [601, 925], "clicks": 1}]


def test_official_mobile_drag_duration_is_dropped_in_lowering() -> None:
    converted = Holo4MobileActionSpace().convert_tool_calls_from_agent(
        [
            {
                "name": "mobile_drag",
                "arguments": {
                    "element": "slider",
                    "x_touch": 100,
                    "y_touch": 200,
                    "x_lift": 800,
                    "y_lift": 200,
                    "duration": 800,
                },
            }
        ]
    )
    assert _actions(converted[0]) == [
        {
            "action": "drag",
            "start_coordinate": [100, 200],
            "coordinate": [800, 200],
        }
    ]
    assert Holo4MobileActionSpace().convert_tool_calls_to_agent(converted) == [
        {
            "name": "mobile_drag",
            "arguments": {
                "element": "screen element",
                "x_touch": 100,
                "y_touch": 200,
                "x_lift": 800,
                "y_lift": 200,
            },
        }
    ]
