"""OpenMobile-2 action-space tests."""

from __future__ import annotations

from lite.agents.models.openmobile_2.action_space import OpenMobile2MobileActionSpace
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet

_ACTION_SPACE = OpenMobile2MobileActionSpace()


def _from_agent(call: dict) -> dict:
    """One mobile_use call → the single canonical call it converts to."""
    converted = _ACTION_SPACE._convert_single_from_agent(call)
    assert len(converted) == 1
    return converted[0]


def test_schema_advertises_the_official_enum() -> None:
    # Official qwen35_thought_session schema: 8 actions (no ``open``), 4
    # buttons (no ``Recent``), success/failure terminate status.
    schemas = _ACTION_SPACE.get_tool_schemas()
    assert [tool_schema["function"]["name"] for tool_schema in schemas] == ["mobile_use"]
    properties = schemas[0]["function"]["parameters"]["properties"]
    assert properties["action"]["enum"] == [
        "click", "long_press", "swipe", "type",
        "answer", "system_button", "wait", "terminate",
    ]
    assert properties["button"]["enum"] == ["Back", "Home", "Menu", "Enter"]
    assert properties["status"]["enum"] == ["success", "failure"]
    # No extra tool folds into the enum: the official surface has no open_app.
    assert _ACTION_SPACE.QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES == {
        "answer": frozenset({"response"}),
        "terminate": frozenset({"terminate"}),
    }


def test_menu_lowers_to_wait_like_the_official_transform() -> None:
    # Official qwen3vl_action_transform turns system_button(Menu) into a
    # bare wait (no keycode); the nested and the wrapper-dropped flat
    # spellings both lower through the parent's wait branch.
    nested = _from_agent(
        {"name": "mobile_use", "arguments": {"action": "system_button", "button": "Menu"}}
    )
    flat = _from_agent({"name": "system_button", "arguments": {"button": "Menu"}})
    for call in (nested, flat):
        assert tool_call_name(call) == "mobile"
        assert tool_call_arguments(call) == {
            "actions": [{"action": "wait", "duration": 3.0}]
        }


def test_other_system_buttons_pass_through() -> None:
    for button in ("Back", "Home", "Enter"):
        call = _from_agent(
            {"name": "mobile_use", "arguments": {"action": "system_button", "button": button}}
        )
        assert tool_call_arguments(call) == {
            "actions": [{"action": "system_button", "button": button}]
        }


def test_answer_and_terminate_map_to_the_finish_tools() -> None:
    answer = _from_agent({"name": "mobile_use", "arguments": {"action": "answer", "text": "42"}})
    assert tool_call_name(answer) == "response"
    assert tool_call_arguments(answer) == {"text": "42"}

    for status, expected in (("success", "success"), ("failure", "failure")):
        call = _from_agent(
            {"name": "mobile_use", "arguments": {"action": "terminate", "status": status}}
        )
        assert tool_call_name(call) == "terminate"
        assert tool_call_arguments(call) == {"status": expected}


def test_click_coordinates_pass_through_unchanged() -> None:
    # Official coordinates are [0, 999]; cua-lite's [0, 1000] normalized
    # space differs by 0.1% (sub-pixel), so they pass through untouched.
    call = _from_agent(
        {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [163, 718]}}
    )
    assert tool_call_arguments(call) == {
        "actions": [{"action": "tap", "coordinate": [163, 718], "clicks": 1}]
    }


def test_left_click_leakage_alias_is_inherited() -> None:
    # The Qwen3.5 base leaks the desktop ``left_click`` spelling on mobile;
    # the inherited alias keeps those turns from becoming env-side noops.
    call = _from_agent(
        {"name": "mobile_use", "arguments": {"action": "left_click", "coordinate": [100, 200]}}
    )
    assert tool_call_arguments(call) == {
        "actions": [{"action": "tap", "coordinate": [100, 200], "clicks": 1}]
    }


def test_open_maps_to_open_app_and_stays_env_rejected() -> None:
    # The official enum has no ``open``; if the model emits it anyway, the
    # inherited mapping lands on canonical open_app, which the family's
    # configs never enable — the env rejects it with model-visible feedback
    # instead of a silent no-op.
    call = _from_agent({"name": "mobile_use", "arguments": {"action": "open", "text": "Settings"}})
    assert tool_call_name(call) == "open_app"
    assert tool_call_arguments(call) == {"app_name": "Settings"}


def test_to_agent_rendering_round_trips_the_native_actions() -> None:
    space = OpenMobile2MobileActionSpace()
    canonical = [
        make_tool_call(
            "mobile",
            {
                "actions": [
                    {"action": "tap", "coordinate": [500, 500]},
                    {"action": "type", "text": "hi"},
                ]
            },
        ),
        LiteFinishToolSet.response(text="done"),
    ]

    rendered = space.convert_tool_calls_to_agent(canonical)
    assert rendered == [
        {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 500]}},
        {"name": "mobile_use", "arguments": {"action": "type", "text": "hi"}},
        {"name": "mobile_use", "arguments": {"action": "answer", "text": "done"}},
    ]

    back = space.convert_tool_calls_from_agent(
        rendered, active_extra_tool_names={"response", "terminate"}
    )
    assert back == [
        make_tool_call(
            "mobile",
            {
                "actions": [
                    {"action": "tap", "coordinate": [500, 500], "clicks": 1},
                    {"action": "type", "text": "hi"},
                ]
            },
        ),
        LiteFinishToolSet.response(text="done"),
    ]
