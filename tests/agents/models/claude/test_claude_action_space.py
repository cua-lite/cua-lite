"""Characterization tests for ClaudeMobileActionSpace.

Covers:
  1. Tool schema inventory matches "claude@mobile" variant
  2. Round-trip (agent → lite → agent) for each action, pixel-based coords
  3. Coordinate conversion from Claude pixel tools to canonical Lite calls

Run:
    uv run pytest tests/agents/models/claude/test_claude_action_space.py -v
"""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import (
    OPEN_APP_SCHEMA,
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    tool_names,
)

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.claude.action_space import ClaudeDesktopActionSpace, ClaudeMobileActionSpace
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import (
    tool_schema_name,
    tool_schema_parameters,
)

PIXEL_6 = (1080, 2400)


def _provider_call(name, **arguments):
    return {"name": name, "arguments": arguments}


def _single_mobile_action(tool_calls):
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert tool_call_name(call) == "mobile"
    actions = tool_call_arguments(call)["actions"]
    assert len(actions) == 1
    return actions[0]


def _single_desktop_action(tool_calls):
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert tool_call_name(call) == "computer"
    actions = tool_call_arguments(call)["actions"]
    assert len(actions) == 1
    return actions[0]


def _single_provider_action(tool_calls):
    assert len(tool_calls) == 1
    return tool_calls[0]


class TestToolInventory:
    """Tool schema surface — claude:mobile variant."""

    def test_tool_names_match_mobile_set(self):
        a = ClaudeMobileActionSpace()
        names = {tool_schema_name(t) for t in a.get_tool_schemas()}
        # open_app is NOT native — it's an env extra_tool (make_open_app_tool),
        # surfaced via extra_tools like qwen. response/terminate are hidden finish
        # tools surfaced only through metadata.extra_tool_schemas.
        expected_actions = {
            "tap",
            "long_press",
            "swipe",
            "drag",
            "pinch",
            "type",
            "system_button",
            "wait",
            "screenshot",
        }
        assert names == expected_actions, f"Missing or extra: {expected_actions ^ names}"
        assert "open_app" not in names
        assert a.get_tool_schema("mobile") is None
        assert a.get_tool_schema("tap") is not None

    def test_no_desktop_actions_present(self):
        a = ClaudeMobileActionSpace()
        names = {tool_schema_name(t) for t in a.get_tool_schemas()}
        desktop_only = {
            "left_click",
            "right_click",
            "double_click",
            "scroll",
            "mouse_move",
            "left_click_drag",
            "key",
            "hold_key",
        }
        leaked = names & desktop_only
        assert not leaked, f"Desktop actions leaked into mobile space: {leaked}"

    def test_tap_description_anchors_platform_and_units(self):
        """Mobile agents rely on description text — guard the anchors."""
        a = ClaudeMobileActionSpace()
        tap_schema = a.get_tool_schema("tap")
        assert tap_schema is not None
        desc = tap_schema["function"]["description"].lower()
        # These anchors help the model understand the provider-flat mobile tool.
        assert "android" in desc or "phone" in desc or "mobile" in desc, desc
        params_desc = tool_schema_parameters(tap_schema)["properties"]["coordinate"][
            "description"
        ].lower()
        assert "pixel" in params_desc, params_desc


def test_claude_empty_valid_actions_drops_native_computer_tool() -> None:
    """``valid_actions=[]`` drops Claude's versioned native computer tool."""
    native = [{"type": "computer_20250124"}]
    assert ClaudeDesktopActionSpace.filter_tool_schemas_for_valid_actions(native, []) == []
    kept = ClaudeDesktopActionSpace.filter_tool_schemas_for_valid_actions(native, ["click"])
    assert [s.get("type") for s in kept] == ["computer_20250124"]


def test_claude_mobile_filter_keeps_non_native_extras() -> None:
    native = ClaudeMobileActionSpace.get_tool_schemas()
    assert {tool_schema_name(s) for s in native} & {"tap"}, "fixture expects a native tap tool"
    schemas = native + [RESPONSE_SCHEMA, TERMINATE_SCHEMA, OPEN_APP_SCHEMA]

    out = ClaudeMobileActionSpace.filter_tool_schemas_for_valid_actions(schemas, ["tap"])
    names = tool_names(out)

    assert {"response", "terminate", "open_app"} <= names
    assert "swipe" not in names
    assert "tap" in names


def test_claude_mobile_filter_keeps_extras_even_with_empty_valid_actions() -> None:
    out = ClaudeMobileActionSpace.filter_tool_schemas_for_valid_actions(
        ClaudeMobileActionSpace.get_tool_schemas() + [RESPONSE_SCHEMA],
        [],
    )

    assert tool_names(out) == {"response"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("+", ["+"]),
        ("-", ["-"]),
        ("=", ["="]),
        (",", [","]),
        ("ctrl++", ["ctrl", "+"]),
        ("ctrl+-", ["ctrl", "-"]),
        ("ctrl+=", ["ctrl", "="]),
        ("ctrl+,", ["ctrl", ","]),
        ("Next", ["pagedown"]),
        ("Prior", ["pageup"]),
    ],
)
def test_claude_desktop_key_text_uses_core_key_grammar(text, expected):
    space = ClaudeDesktopActionSpace()

    out = space.convert_tool_calls_from_agent([{"action": "key", "text": text}])

    assert _single_desktop_action(out) == {"action": "key", "keys": expected}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("+", ["+"]),
        ("-", ["-"]),
        ("=", ["="]),
        (",", [","]),
        ("ctrl++", ["ctrl", "+"]),
        ("ctrl+-", ["ctrl", "-"]),
        ("ctrl+=", ["ctrl", "="]),
        ("ctrl+,", ["ctrl", ","]),
    ],
)
def test_claude_desktop_hold_key_text_uses_core_key_grammar(text, expected):
    space = ClaudeDesktopActionSpace()

    out = space.convert_tool_calls_from_agent(
        [{"action": "hold_key", "key": text, "duration": 2.0}]
    )

    assert _single_desktop_action(out) == {
        "action": "hold_key",
        "keys": expected,
        "duration": 2.0,
    }


@pytest.mark.parametrize("action,key_arg", [("key", "text"), ("hold_key", "key")])
@pytest.mark.parametrize("raw_keys", ["ctrl left", "ctrl -"])
def test_claude_desktop_key_actions_reject_phrase_like_strings(action, key_arg, raw_keys):
    space = ClaudeDesktopActionSpace()

    with pytest.raises(ModelToolCallParseError, match="unknown key token"):
        space.convert_tool_calls_from_agent(
            [{"action": action, key_arg: raw_keys, "duration": 2.0}]
        )


@pytest.mark.parametrize("action,key_arg", [("key", "text"), ("hold_key", "key")])
def test_claude_desktop_key_actions_reject_non_string_payload(action, key_arg):
    space = ClaudeDesktopActionSpace()

    with pytest.raises(ModelToolCallParseError, match=f"{action} requires string {key_arg}"):
        space.convert_tool_calls_from_agent(
            [{"action": action, key_arg: ["ctrl", "c"], "duration": 2.0}]
        )


@pytest.mark.parametrize("action", ["key", "hold_key"])
def test_claude_desktop_key_constructor_errors_stay_loud(monkeypatch, action):
    def broken_constructor(**kwargs):
        raise ValueError("internal converter bug")

    monkeypatch.setattr(
        f"lite.agents.models.claude.action_space.LiteDesktopActionSpace.{action}",
        broken_constructor,
    )
    with pytest.raises(ValueError, match="internal converter bug") as caught:
        ClaudeDesktopActionSpace().convert_tool_calls_from_agent(
            [{"action": action, "text": "Return", "duration": 2.0}]
        )
    assert not isinstance(caught.value, ModelToolCallParseError)


def test_claude_desktop_key_repeat_and_hold_text():
    space = ClaudeDesktopActionSpace()
    repeated = space.convert_tool_calls_from_agent([{"action": "key", "text": "Tab", "repeat": 3}])
    assert tool_call_arguments(repeated[0])["actions"] == [{"action": "key", "keys": ["tab"]}] * 3
    held = space.convert_tool_calls_from_agent(
        [{"action": "hold_key", "text": "Shift", "duration": 2}]
    )
    assert _single_desktop_action(held) == {
        "action": "hold_key", "keys": ["shift"], "duration": 2.0,
    }


@pytest.mark.parametrize("repeat", [0, 101, True, 1.5])
def test_claude_desktop_rejects_invalid_repeat(repeat):
    with pytest.raises(ModelToolCallParseError, match="repeat"):
        ClaudeDesktopActionSpace().convert_tool_calls_from_agent(
            [{"action": "key", "text": "Tab", "repeat": repeat}]
        )


@pytest.mark.parametrize(
    "payload,expected_action",
    [
        ({"action": "left_click", "coordinate": [500, 300]}, "click"),
        ({"action": "right_click", "coordinate": [500, 300]}, "click"),
        ({"action": "middle_click", "coordinate": [500, 300]}, "click"),
        ({"action": "double_click", "coordinate": [500, 300]}, "click"),
        ({"action": "triple_click", "coordinate": [500, 300]}, "click"),
        ({"action": "scroll", "scroll_direction": "down", "scroll_amount": 2}, "scroll"),
        (
            {"action": "left_click_drag", "start_coordinate": [0, 0], "coordinate": [500, 500]},
            "drag",
        ),
    ],
)
def test_claude_mouse_modifiers_surround_the_action(payload, expected_action):
    out = ClaudeDesktopActionSpace().convert_tool_calls_from_agent(
        [{**payload, "text": "ctrl+shift"}], resolution=(1000, 1000)
    )
    actions = tool_call_arguments(out[0])["actions"]
    assert actions[0] == {"action": "key_down", "keys": ["ctrl", "shift"]}
    assert actions[1]["action"] == expected_action
    assert actions[2] == {"action": "key_up", "keys": ["ctrl", "shift"]}
    if expected_action == "click":
        assert actions[1]["coordinate"] == [500, 300]


class TestRoundTripPixel:
    """Pixel coordinates round-trip through CUA-lite normalized [0, 1000]."""

    @pytest.mark.parametrize(
        "px",
        [
            [0, 0],
            [540, 1200],  # screen center for 1080x2400
            [1079, 2399],  # corner
            [200, 1500],
        ],
    )
    def test_tap_round_trip(self, px):
        a = ClaudeMobileActionSpace()
        raw = [_provider_call("tap", coordinate=px)]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite)["action"] == "tap"

        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        call = _single_provider_action(back)
        assert call["name"] == "tap"
        # Allow ±1 pixel rounding
        for i in range(2):
            assert abs(call["arguments"]["coordinate"][i] - px[i]) <= 1, (back, px)

    def test_tap_clicks_round_trip(self):
        a = ClaudeMobileActionSpace()
        raw = [_provider_call("tap", coordinate=[540, 1200], clicks=2)]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite)["clicks"] == 2
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert _single_provider_action(back)["arguments"]["clicks"] == 2

    def test_swipe_round_trip(self):
        a = ClaudeMobileActionSpace()
        raw_action = {"start_coordinate": [100, 2000], "coordinate": [100, 400]}
        raw = [_provider_call("swipe", **raw_action)]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite)["action"] == "swipe"
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        call = _single_provider_action(back)
        assert call["name"] == "swipe"
        for key in ("start_coordinate", "coordinate"):
            for i in range(2):
                assert abs(call["arguments"][key][i] - raw_action[key][i]) <= 1

    def test_long_press_preserves_duration(self):
        a = ClaudeMobileActionSpace()
        raw = [_provider_call("long_press", coordinate=[540, 1200], duration=2.5)]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite)["duration"] == 2.5
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert _single_provider_action(back)["arguments"]["duration"] == 2.5

    def test_pinch_round_trip(self):
        a = ClaudeMobileActionSpace()
        raw = [
            _provider_call(
                "pinch",
                coordinate=[540, 1200],
                direction="out",
                amount=30,
            )
        ]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        action = _single_mobile_action(lite)
        assert action["action"] == "pinch"
        assert action["direction"] == "out"
        assert action["amount"] == 30
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        provider = _single_provider_action(back)
        assert provider["name"] == "pinch"
        assert provider["arguments"]["direction"] == "out"
        assert provider["arguments"]["amount"] == 30
        for i in range(2):
            assert abs(provider["arguments"]["coordinate"][i] - [540, 1200][i]) <= 1

    def test_type_text_preserved(self):
        a = ClaudeMobileActionSpace()
        raw = [_provider_call("type", text="hello android 🤖")]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite)["text"] == "hello android 🤖"
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert _single_provider_action(back) == _provider_call("type", text="hello android 🤖")

    @pytest.mark.parametrize("btn", ["Home", "Back", "Enter", "Menu", "Recent"])
    def test_system_button_round_trip(self, btn):
        a = ClaudeMobileActionSpace()
        raw = [_provider_call("system_button", button=btn)]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite) == {"action": "system_button", "button": btn}
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert _single_provider_action(back) == _provider_call("system_button", button=btn)

    def test_open_app_round_trip(self):
        a = ClaudeMobileActionSpace()
        raw = [{"name": "open_app", "arguments": {"app_name": "Settings"}}]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert back[0] == {"name": "open_app", "arguments": {"app_name": "Settings"}}

    def test_response_round_trip(self):
        a = ClaudeMobileActionSpace()
        raw = [{"name": "response", "arguments": {"text": "42%"}}]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert tool_call_name(lite[0]) == "response"
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert back[0] == {"name": "response", "arguments": {"text": "42%"}}

    def test_terminate_round_trip_with_reason(self):
        a = ClaudeMobileActionSpace()
        raw = [{"name": "terminate", "arguments": {"status": "success", "reason": "done"}}]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert back[0]["name"] == "terminate"
        assert back[0]["arguments"]["status"] == "success"
        assert back[0]["arguments"]["reason"] == "done"

    def test_screenshot_round_trip(self):
        a = ClaudeMobileActionSpace()
        raw = [_provider_call("screenshot")]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite) == {"action": "screenshot"}
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert _single_provider_action(back) == _provider_call("screenshot")


class TestNormalizationTarget:
    """CUA-lite target is normalized [0, 1000]; verify conversion math."""

    def test_tap_center_normalizes_to_500(self):
        a = ClaudeMobileActionSpace()
        raw = [_provider_call("tap", coordinate=[540, 1200])]  # exact center of 1080x2400
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        coord = _single_mobile_action(lite)["coordinate"]
        assert 499 <= coord[0] <= 501
        assert 499 <= coord[1] <= 501

    def test_tap_origin_is_zero(self):
        a = ClaudeMobileActionSpace()
        raw = [_provider_call("tap", coordinate=[0, 0])]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        coord = _single_mobile_action(lite)["coordinate"]
        assert coord == [0, 0]


class TestBatchConversion:
    """Adjacent mobile actions normalize to one canonical mobile wrapper."""

    def test_multi_action_batch(self):
        a = ClaudeMobileActionSpace()
        raw = [
            _provider_call("tap", coordinate=[100, 200]),
            _provider_call("type", text="search"),
            _provider_call("system_button", button="Enter"),
        ]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert len(lite) == 1
        assert tool_call_name(lite[0]) == "mobile"
        assert [a["action"] for a in tool_call_arguments(lite[0])["actions"]] == [
            "tap",
            "type",
            "system_button",
        ]
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert [call["name"] for call in back] == ["tap", "type", "system_button"]
        assert all(
            abs(back[0]["arguments"]["coordinate"][i] - [100, 200][i]) <= 1 for i in range(2)
        )

    def test_adjacent_mobile_calls_merge_without_crossing_extra_tools(self):
        a = ClaudeMobileActionSpace()
        raw = [
            _provider_call("tap", coordinate=[100, 200]),
            _provider_call("type", text="search"),
            {"name": "open_app", "arguments": {"app_name": "Settings"}},
            _provider_call("wait", duration=1.0),
        ]

        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)

        assert [tool_call_name(call) for call in lite] == ["mobile", "open_app", "mobile"]
        assert [action["action"] for action in tool_call_arguments(lite[0])["actions"]] == [
            "tap",
            "type",
        ]
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert [call["name"] for call in back] == ["tap", "type", "open_app", "wait"]

    def test_adjacent_canonical_mobile_calls_unwrap_to_provider_flat_calls(self):
        a = ClaudeMobileActionSpace()
        lite = [
            make_tool_call(
                "mobile",
                {"actions": [{"action": "tap", "coordinate": [93, 83]}]},
            ),
            make_tool_call(
                "mobile",
                {"actions": [{"action": "type", "text": "search"}]},
            ),
        ]

        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)

        assert [call["name"] for call in back] == ["tap", "type"]

    def test_undeclared_provider_mobile_wrapper_does_not_pass_through_as_extra(self):
        a = ClaudeMobileActionSpace()
        raw = [
            {
                "name": "mobile",
                "arguments": {
                    "actions": [{"action": "open_app", "app_name": "Settings"}],
                },
            }
        ]

        with pytest.raises(ValueError, match="unknown Claude mobile action: mobile"):
            a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)

    @pytest.mark.parametrize(
        "raw",
        [
            [_provider_call("tap")],
            [_provider_call("tap", coordinate=[None, 7])],
            [_provider_call("tap", coordinate=[7])],
            [_provider_call("swipe", start_coordinate=[10, 20])],
        ],
    )
    def test_malformed_native_provider_calls_raise_value_error(self, raw):
        a = ClaudeMobileActionSpace()

        with pytest.raises(ValueError, match="malformed Claude mobile arguments"):
            a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)

    def test_undeclared_provider_mobile_wrapper_raises_by_default(self):
        a = ClaudeMobileActionSpace()

        with pytest.raises(ValueError, match="unknown Claude mobile action: mobile"):
            a.convert_tool_calls_from_agent(
                [{"name": "mobile", "arguments": {"actions": [{"action": "tap"}]}}],
                resolution=PIXEL_6,
            )


class TestReplayCoordinateBoundary:
    @pytest.mark.parametrize(
        ("action", "match"),
        [
            ({"action": "tap"}, "missing coordinate"),
            ({"action": "long_press"}, "missing coordinate"),
            ({"action": "pinch", "direction": "in"}, "missing coordinate"),
            (
                {"action": "swipe", "coordinate": [300, 400]},
                "missing start_coordinate",
            ),
            (
                {"action": "swipe", "start_coordinate": [100, 200]},
                "missing coordinate",
            ),
            (
                {"action": "drag", "coordinate": [300, 400]},
                "missing start_coordinate",
            ),
            (
                {"action": "drag", "start_coordinate": [100, 200]},
                "missing coordinate",
            ),
        ],
    )
    def test_missing_nested_canonical_coordinates_raise(self, action, match):
        a = ClaudeMobileActionSpace()
        lite = [make_tool_call("mobile", {"actions": [action]})]

        with pytest.raises(ValueError, match=match):
            a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)

    def test_missing_top_level_canonical_coordinate_raises(self):
        a = ClaudeMobileActionSpace()

        with pytest.raises(ValueError, match="missing coordinate"):
            a.convert_tool_calls_to_agent(
                [make_tool_call("tap", {})],
                resolution=PIXEL_6,
            )
