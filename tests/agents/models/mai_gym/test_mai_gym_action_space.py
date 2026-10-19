"""MAI-Gym action space: native ``mobile_use(action_type=...)`` <-> canonical Lite."""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.mai_gym.action_space import MAIGymMobileActionSpace
from lite.core.tools.action_space import LiteMobileActionSet
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet


def _from_agent(arguments: dict, name: str = "mobile_use") -> list[dict]:
    return MAIGymMobileActionSpace().convert_tool_calls_from_agent(
        [{"name": name, "arguments": arguments}]
    )


def _children(calls: list[dict]) -> list[dict]:
    assert [tool_call_name(c) for c in calls] == ["mobile"]
    return tool_call_arguments(calls[0])["actions"]


def test_schema_declares_the_trained_action_type_enum() -> None:
    (schema,) = MAIGymMobileActionSpace.get_tool_schemas()
    params = schema["function"]["parameters"]["properties"]
    assert schema["function"]["name"] == "mobile_use"
    assert params["action_type"]["enum"] == [
        "click",
        "double_click",
        "long_press",
        "type",
        "open",
        "drag",
        "system_button",
        "wait",
        "ask_user",
        "terminate",
    ]


def test_click_and_double_click_rescale_999_to_1000() -> None:
    assert _children(_from_agent({"action_type": "click", "coordinate": [999, 0]})) == [
        {"action": "tap", "coordinate": [1000, 0], "clicks": 1}
    ]
    assert _children(_from_agent({"action_type": "double_click", "coordinate": [500, 500]})) == [
        {"action": "tap", "coordinate": [501, 501], "clicks": 2}
    ]


def test_drag_is_the_scroll_gesture_and_keeps_exact_endpoints() -> None:
    children = _children(
        _from_agent(
            {
                "action_type": "drag",
                "start_coordinate": [500, 800],
                "end_coordinate": [500, 200],
            }
        )
    )
    assert children == [
        {"action": "drag", "start_coordinate": [501, 801], "coordinate": [501, 200]}
    ]


def test_menu_lowers_to_home_like_upstream() -> None:
    for button, expected in (
        ("menu", "Home"),
        ("back", "Back"),
        ("home", "Home"),
        ("enter", "Enter"),
    ):
        assert _children(_from_agent({"action_type": "system_button", "button": button})) == [
            {"action": "system_button", "button": expected}
        ]


def test_terminate_with_answer_is_a_response_and_ignores_status() -> None:
    assert _from_agent({"action_type": "terminate", "answer": "42", "status": "failed"}) == [
        LiteFinishToolSet.response(text="42")
    ]


def test_terminate_without_answer_maps_status() -> None:
    assert _from_agent({"action_type": "terminate", "status": "failed"}) == [
        LiteFinishToolSet.terminate(status="failure")
    ]
    assert _from_agent({"action_type": "terminate", "status": "success"}) == [
        LiteFinishToolSet.terminate(status="success")
    ]


def test_wait_open_type_and_ask_user() -> None:
    assert _children(_from_agent({"action_type": "wait", "waiting_time": "2"})) == [
        {"action": "wait", "duration": 2.0}
    ]
    assert _children(_from_agent({"action_type": "wait", "waiting_time": "xxx"})) == [
        {"action": "wait", "duration": 1.0}
    ]
    assert _from_agent({"action_type": "open", "text": "Files"}) == [
        make_tool_call("open_app", {"app_name": "Files"})
    ]
    assert _children(_from_agent({"action_type": "type", "text": "hi"})) == [
        {"action": "type", "text": "hi"}
    ]
    assert _from_agent({"action_type": "ask_user", "text": "Which one?"}) == [
        make_tool_call("ask_user", {"text": "Which one?"})
    ]


def test_dispatch_reads_action_type_and_ignores_the_tool_name() -> None:
    # Upstream ``parse_action_response`` never checks the name.
    assert _children(
        _from_agent({"action_type": "click", "coordinate": [0, 0]}, name="anything")
    ) == [{"action": "tap", "coordinate": [0, 0], "clicks": 1}]
    # No action key at all: a standalone tool the env answers by its own name.
    assert _from_agent({"query": "x"}, name="search") == [make_tool_call("search", {"query": "x"})]


@pytest.mark.parametrize(
    "arguments",
    [
        {"action_type": "swipe", "direction": "up"},
        {"action_type": "click"},
        {"action_type": "click", "coordinate": [1, 2, 3, 4]},
        {"action_type": "system_button", "button": "recent"},
    ],
)
def test_malformed_native_actions_are_parse_failures(arguments: dict) -> None:
    with pytest.raises(ModelToolCallParseError):
        _from_agent(arguments)


def test_canonical_calls_render_back_to_native() -> None:
    space = MAIGymMobileActionSpace()
    rendered = space.convert_tool_calls_to_agent(
        [
            LiteMobileActionSet.tap(coordinate=[1000, 0], clicks=2),
            LiteMobileActionSet.swipe(start_coordinate=[500, 800], coordinate=[500, 200]),
            LiteMobileActionSet.system_button(button="Menu"),
            LiteMobileActionSet.wait(duration=2.0),
            make_tool_call("open_app", {"app_name": "Files"}),
            LiteFinishToolSet.response(text="42"),
            LiteFinishToolSet.terminate(status="failure"),
        ]
    )
    assert [call["arguments"] for call in rendered] == [
        {"action_type": "double_click", "coordinate": [999, 0]},
        {"action_type": "drag", "start_coordinate": [500, 799], "end_coordinate": [500, 200]},
        {"action_type": "system_button", "button": "menu"},
        {"action_type": "wait", "waiting_time": "2"},
        {"action_type": "open", "text": "Files"},
        {"action_type": "terminate", "answer": "42", "status": "success"},
        {"action_type": "terminate", "status": "failed"},
    ]
    assert {call["name"] for call in rendered} == {"mobile_use"}


def test_canonical_actions_without_a_native_spelling_raise() -> None:
    space = MAIGymMobileActionSpace()
    with pytest.raises(ValueError, match="Recent"):
        space.convert_tool_calls_to_agent([LiteMobileActionSet.system_button(button="Recent")])
    with pytest.raises(ValueError, match="pinch"):
        space.convert_tool_calls_to_agent(
            [LiteMobileActionSet.pinch(coordinate=[500, 500], direction="in")]
        )


def test_env_extra_tool_passes_through_as_bare_function_call() -> None:
    call = make_tool_call("goto", {"url": "http://example.com"})
    assert MAIGymMobileActionSpace().convert_tool_calls_to_agent([call]) == [
        {"name": "goto", "arguments": {"url": "http://example.com"}}
    ]
