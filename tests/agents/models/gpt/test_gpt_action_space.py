"""Characterization tests for GPTMobileActionSpace.

Parallel to test_claude_mobile.py. GPT's wire format uses separate x/y fields
rather than Claude's coordinate list, so per-action assertions differ.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_action_space.py -v
"""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import (
    OPEN_APP_SCHEMA,
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    tool_names,
)

from lite.agents.models.gpt.action_space import GPTDesktopActionSpace, GPTMobileActionSpace
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import (
    tool_schema_name,
    tool_schema_parameters,
)

PIXEL_6 = (1080, 2400)


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


def _mobile_provider_call(*actions):
    return [
        {
            "name": action["action"],
            "arguments": {k: v for k, v in action.items() if k != "action"},
        }
        for action in actions
    ]


def _single_provider_action(tool_calls):
    assert len(tool_calls) == 1
    call = tool_calls[0]
    return {"action": call["name"], **call["arguments"]}


class TestToolInventory:
    def test_tool_names_match_mobile_set(self):
        a = GPTMobileActionSpace()
        names = {tool_schema_name(t) for t in a.get_tool_schemas()}
        # open_app is NOT native — it's an env extra_tool (make_open_app_tool),
        # surfaced via extra_tools like qwen. response/terminate are hidden finish
        # tools surfaced only through metadata.extra_tool_schemas.
        expected = {
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
        assert names == expected
        assert "mobile" not in names
        assert "open_app" not in names
        assert a.get_tool_schema("mobile") is None

    def test_provider_flat_tool_required_params_are_action_specific(self):
        a = GPTMobileActionSpace()
        tap = a.get_tool_schema("tap")
        swipe = a.get_tool_schema("swipe")
        pinch = a.get_tool_schema("pinch")
        long_press = a.get_tool_schema("long_press")
        type_tool = a.get_tool_schema("type")
        screenshot = a.get_tool_schema("screenshot")

        assert tool_schema_parameters(tap)["required"] == ["x", "y"]
        assert tool_schema_parameters(swipe)["required"] == [
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]
        assert tool_schema_parameters(pinch)["required"] == ["x", "y", "direction"]
        assert tool_schema_parameters(long_press)["required"] == ["x", "y"]
        assert tool_schema_parameters(type_tool)["required"] == ["text"]
        assert tool_schema_parameters(screenshot)["required"] == []

    def test_no_desktop_actions_present(self):
        a = GPTMobileActionSpace()
        names = {tool_schema_name(t) for t in a.get_tool_schemas()}
        desktop_only = {"click", "double_click", "scroll", "keypress", "hotkey"}
        leaked = names & desktop_only
        assert not leaked, f"Desktop actions leaked: {leaked}"

    def test_tap_description_anchors_platform(self):
        a = GPTMobileActionSpace()
        tap = a.get_tool_schema("tap")
        assert tap is not None
        desc = tap["function"]["description"].lower()
        assert "screen" in desc or "phone" in desc or "mobile" in desc
        # Params should mention pixel + coordinate
        params = tool_schema_parameters(tap)["properties"]
        x_description = params["x"]["description"].lower()
        assert "pixel" in x_description or "coordinate" in x_description


def test_gpt_empty_valid_actions_drops_native_computer_tool() -> None:
    """``valid_actions=[]`` drops GPT's opaque native computer tool."""
    native = [{"type": "computer"}]
    assert GPTDesktopActionSpace.filter_tool_schemas_for_valid_actions(native, []) == []
    kept = GPTDesktopActionSpace.filter_tool_schemas_for_valid_actions(native, ["click"])
    assert [s.get("type") for s in kept] == ["computer"]


def test_gpt_mobile_filter_keeps_non_native_extras() -> None:
    native = GPTMobileActionSpace.get_tool_schemas()
    assert {tool_schema_name(s) for s in native} & {"tap"}, "fixture expects a native tap tool"
    schemas = native + [RESPONSE_SCHEMA, TERMINATE_SCHEMA, OPEN_APP_SCHEMA]

    out = GPTMobileActionSpace.filter_tool_schemas_for_valid_actions(schemas, ["tap"])
    names = tool_names(out)

    assert {"response", "terminate", "open_app"} <= names
    assert "swipe" not in names
    assert "tap" in names


def test_gpt_mobile_filter_keeps_extras_even_with_empty_valid_actions() -> None:
    out = GPTMobileActionSpace.filter_tool_schemas_for_valid_actions(
        GPTMobileActionSpace.get_tool_schemas() + [RESPONSE_SCHEMA],
        [],
    )

    assert tool_names(out) == {"response"}


@pytest.mark.parametrize(
    "raw_keys,expected",
    [
        ("+", ["+"]),
        ("ctrl++", ["ctrl", "+"]),
        ("ctrl+plus", ["ctrl", "+"]),
        ("ctrl+-", ["ctrl", "-"]),
        ("ctrl+=", ["ctrl", "="]),
        (["CTRL+P"], ["ctrl", "p"]),
        (["CTRL", "P"], ["ctrl", "p"]),
        (["CTRL+SHIFT", "P"], ["ctrl", "shift", "p"]),
        (["ctrl+;"], ["ctrl", ";"]),
        (["ctrl+semicolon"], ["ctrl", ";"]),
        (["ctrl++"], ["ctrl", "+"]),
        (["ctrl", "+"], ["ctrl", "+"]),
    ],
)
def test_gpt_desktop_keypress_uses_core_key_grammar(raw_keys, expected):
    space = GPTDesktopActionSpace()

    out = space.convert_tool_calls_from_agent([{"type": "keypress", "keys": raw_keys}])

    assert _single_desktop_action(out) == {"action": "key", "keys": expected}


@pytest.mark.parametrize("raw_keys", ["ctrl +", ["ctrl +"], ["ctrl+"]])
def test_gpt_desktop_keypress_rejects_incomplete_chord(raw_keys):
    space = GPTDesktopActionSpace()

    with pytest.raises(ValueError, match="invalid trailing '\\+' chord syntax"):
        space.convert_tool_calls_from_agent([{"type": "keypress", "keys": raw_keys}])


@pytest.mark.parametrize("raw_keys", ["ctrl left", "ctrl -", ["ctrl+；"]])
def test_gpt_desktop_keypress_rejects_phrase_like_key_strings(raw_keys):
    space = GPTDesktopActionSpace()

    with pytest.raises(ValueError, match="unknown key token"):
        space.convert_tool_calls_from_agent([{"type": "keypress", "keys": raw_keys}])


class TestRoundTripPixel:
    @pytest.mark.parametrize(
        "px",
        [
            (0, 0),
            (540, 1200),
            (1079, 2399),
            (200, 1500),
        ],
    )
    def test_tap_round_trip(self, px):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call({"action": "tap", "x": px[0], "y": px[1]})
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite)["action"] == "tap"
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        action = _single_provider_action(back)
        assert action["action"] == "tap"
        assert abs(action["x"] - px[0]) <= 1
        assert abs(action["y"] - px[1]) <= 1

    def test_tap_clicks_round_trip(self):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call({"action": "tap", "x": 540, "y": 1200, "clicks": 2})
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite)["clicks"] == 2
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert _single_provider_action(back)["clicks"] == 2

    def test_swipe_round_trip(self):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call(
            {
                "action": "swipe",
                "start_x": 100,
                "start_y": 2000,
                "end_x": 100,
                "end_y": 400,
            }
        )
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite)["action"] == "swipe"
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        action = _single_provider_action(back)
        for k in ("start_x", "start_y", "end_x", "end_y"):
            assert abs(action[k] - raw[0]["arguments"][k]) <= 1

    def test_long_press_preserves_duration(self):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call({"action": "long_press", "x": 540, "y": 1200, "duration": 2.5})
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite)["duration"] == 2.5
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert _single_provider_action(back)["duration"] == 2.5

    def test_pinch_round_trip(self):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call(
            {
                "action": "pinch",
                "x": 540,
                "y": 1200,
                "direction": "out",
                "amount": 30,
            }
        )
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        action = _single_mobile_action(lite)
        assert action["action"] == "pinch"
        assert action["direction"] == "out"
        assert action["amount"] == 30
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        provider = _single_provider_action(back)
        assert provider["action"] == "pinch"
        assert provider["direction"] == "out"
        assert provider["amount"] == 30
        assert abs(provider["x"] - 540) <= 1
        assert abs(provider["y"] - 1200) <= 1

    def test_type_text_preserved(self):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call({"action": "type", "text": "hello android 🤖"})
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert _single_mobile_action(lite)["text"] == "hello android 🤖"
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert _single_provider_action(back)["text"] == "hello android 🤖"

    @pytest.mark.parametrize("btn", ["Home", "Back", "Enter", "Menu", "Recent"])
    def test_system_button_round_trip(self, btn):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call({"action": "system_button", "button": btn})
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert _single_provider_action(back) == {"action": "system_button", "button": btn}

    def test_open_app_round_trip(self):
        a = GPTMobileActionSpace()
        raw = [{"name": "open_app", "arguments": {"app_name": "Settings"}}]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert back[0] == {"name": "open_app", "arguments": {"app_name": "Settings"}}

    def test_response_round_trip(self):
        a = GPTMobileActionSpace()
        raw = [{"name": "response", "arguments": {"text": "42%"}}]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert back[0] == {"name": "response", "arguments": {"text": "42%"}}

    def test_standalone_extra_to_agent_does_not_require_resolution(self):
        a = GPTMobileActionSpace()
        lite = [make_tool_call("response", {"text": "42%"})]
        assert a.convert_tool_calls_to_agent(lite) == [
            {"name": "response", "arguments": {"text": "42%"}}
        ]

    def test_terminate_round_trip_with_reason(self):
        a = GPTMobileActionSpace()
        raw = [{"name": "terminate", "arguments": {"status": "success", "reason": "done"}}]
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert back[0] == {
            "name": "terminate",
            "arguments": {"status": "success", "reason": "done"},
        }

    def test_screenshot_round_trip(self):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call({"action": "screenshot"})
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert _single_provider_action(back) == {"action": "screenshot"}


class TestNormalizationTarget:
    def test_tap_center_normalizes_to_500(self):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call({"action": "tap", "x": 540, "y": 1200})
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        coord = _single_mobile_action(lite)["coordinate"]
        assert 499 <= coord[0] <= 501
        assert 499 <= coord[1] <= 501

    def test_tap_origin_is_zero(self):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call({"action": "tap", "x": 0, "y": 0})
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        coord = _single_mobile_action(lite)["coordinate"]
        assert coord == [0, 0]


class TestBatchConversion:
    def test_multi_action_batch(self):
        a = GPTMobileActionSpace()
        raw = _mobile_provider_call(
            {"action": "tap", "x": 100, "y": 200},
            {"action": "type", "text": "search"},
            {"action": "system_button", "button": "Enter"},
        )
        lite = a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)
        assert len(lite) == 1
        assert tool_call_name(lite[0]) == "mobile"
        assert [a["action"] for a in tool_call_arguments(lite[0])["actions"]] == [
            "tap",
            "type",
            "system_button",
        ]
        back = a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)
        assert len(back) == 3
        actions = [{"action": call["name"], **call["arguments"]} for call in back]
        assert [action["action"] for action in actions] == ["tap", "type", "system_button"]
        assert abs(actions[0]["x"] - 100) <= 1
        assert abs(actions[0]["y"] - 200) <= 1

    def test_adjacent_mobile_calls_merge_without_crossing_extra_tools(self):
        a = GPTMobileActionSpace()
        raw = [
            {"name": "tap", "arguments": {"x": 100, "y": 200}},
            {"name": "type", "arguments": {"text": "search"}},
            {"name": "open_app", "arguments": {"app_name": "Settings"}},
            {"name": "wait", "arguments": {"duration": 1.0}},
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
        a = GPTMobileActionSpace()
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

    def test_undeclared_provider_mobile_wrapper_is_rejected(self):
        a = GPTMobileActionSpace()
        raw = [
            {
                "name": "mobile",
                "arguments": {
                    "actions": [
                        {"action": "tap", "x": 100, "y": 200},
                        {"action": "type", "text": "search"},
                    ]
                },
            }
        ]

        with pytest.raises(ValueError, match="unknown GPT mobile action: mobile"):
            a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)

    def test_unknown_nested_action_does_not_pass_through_as_extra(self):
        a = GPTMobileActionSpace()
        raw = [
            {
                "name": "mobile",
                "arguments": {
                    "actions": [
                        {"action": "open_app", "app_name": "Settings"},
                    ]
                },
            }
        ]

        with pytest.raises(ValueError, match="unknown GPT mobile action: mobile"):
            a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)

    @pytest.mark.parametrize(
        "raw",
        [
            [{"name": "tap", "arguments": {}}],
            [{"name": "tap", "arguments": None}],
            [{"name": "swipe", "arguments": {"start_x": 10, "start_y": 20, "end_x": 30}}],
        ],
    )
    def test_malformed_native_provider_calls_raise_value_error(self, raw):
        a = GPTMobileActionSpace()

        with pytest.raises(ValueError, match="malformed GPT mobile arguments"):
            a.convert_tool_calls_from_agent(raw, resolution=PIXEL_6)

    @pytest.mark.parametrize(
        "arguments",
        [
            {},
            {"actions": "tap"},
            {"actions": [None]},
            {"actions": [{"action": "tap"}]},
        ],
    )
    def test_malformed_undeclared_mobile_wrapper_is_rejected_not_parsed(self, arguments):
        a = GPTMobileActionSpace()

        with pytest.raises(ValueError, match="unknown GPT mobile action: mobile"):
            a.convert_tool_calls_from_agent(
                [{"name": "mobile", "arguments": arguments}],
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
        a = GPTMobileActionSpace()
        lite = [make_tool_call("mobile", {"actions": [action]})]

        with pytest.raises(ValueError, match=match):
            a.convert_tool_calls_to_agent(lite, resolution=PIXEL_6)

    def test_missing_top_level_canonical_coordinate_raises(self):
        a = GPTMobileActionSpace()

        with pytest.raises(ValueError, match="missing coordinate"):
            a.convert_tool_calls_to_agent(
                [make_tool_call("tap", {})],
                resolution=PIXEL_6,
            )
