"""GUI-360 preproc tool/result contract tests."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
from data.preproc._tool_io_helpers import (
    _actions,
    _all_calls,
    _assert_final_action_row,
    _assert_first_action_result_is_tool,
    _assert_structural_done_row,
)

from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.data.preproc.gui360 import use as gui360_use


def test_gui360_post_action_screenshot_is_tool_and_type_batches(monkeypatch):
    class _FakeImage:
        size = (1000, 1000)

    monkeypatch.setattr(gui360_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    monkeypatch.setattr(gui360_use.Image, "open", lambda _path: nullcontext(_FakeImage()))
    monkeypatch.setattr(
        gui360_use,
        "load_steps",
        lambda _path: [
            {
                "execution_id": "exec",
                "request": "finish",
                "complete": "yes",
                "status": "RUNNING",
                "thought": "focus",
                "subtask": "type",
                "screenshot": "0.png",
                "action": {
                    "function": "type",
                    "coordinate_x": 100,
                    "coordinate_y": 200,
                    "args": {"text": "hello", "clear_current_text": True},
                },
            },
            {
                "execution_id": "exec",
                "request": "finish",
                "complete": "yes",
                "status": "OVERALL_FINISH",
                "thought": "finish",
                "subtask": "done",
                "screenshot": "1.png",
                "action": {"function": "", "args": {}},
            },
        ],
    )

    row = gui360_use.build_trajectory("fake.jsonl", "word", "forms")
    assert row is not None
    _assert_first_action_result_is_tool(row)
    assert tool_call_name(row["messages"][1]["tool_calls"][0]) == "computer"
    actions = tool_call_arguments(row["messages"][1]["tool_calls"][0])["actions"]
    assert [a["action"] for a in actions] == [
        "click",
        "key",
        "type",
    ]
    assert tool_call_arguments(row["messages"][1]["tool_calls"][0])["actions"][1] == {
        "action": "key",
        "keys": ["ctrl", "a"],
    }
    assert len(_all_calls(row)) == 1
    _assert_structural_done_row(row)


def test_gui360_loader_fails_loud_on_malformed_jsonl(tmp_path):
    source = tmp_path / "broken.jsonl"
    source.write_text("{\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.jsonl line 1"):
        gui360_use.load_steps(str(source))


def test_gui360_type_focuses_its_target_coordinate() -> None:
    kind, calls = gui360_use.map_action(
        {
            "function": "type",
            "coordinate_x": 100,
            "coordinate_y": 200,
            "args": {"text": "hello"},
        },
        1000,
        1000,
        "exec",
    )

    assert kind == "emit"
    assert [tool_call_arguments(call)["actions"][0]["action"] for call in calls] == [
        "click",
        "type",
    ]


def test_gui360_keeps_terminal_executable_action_at_eof(monkeypatch):
    """A terminal OVERALL_FINISH GUI action is still a supervised label.

    GUI-360 stamps ``OVERALL_FINISH`` either on a dedicated terminal record (whose
    screenshot is the previous action's result) or on the step that finished the
    task. The second shape has no later screenshot; the action now stays as the
    row's EOF target.
    """
    class _FakeImage:
        size = (1000, 1000)

    monkeypatch.setattr(gui360_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    monkeypatch.setattr(gui360_use.Image, "open", lambda _path: nullcontext(_FakeImage()))
    monkeypatch.setattr(
        gui360_use,
        "load_steps",
        lambda _path: [
            {
                "execution_id": "exec",
                "request": "finish",
                "complete": "yes",
                "status": "RUNNING",
                "thought": "focus",
                "subtask": "type",
                "screenshot": "0.png",
                "action": {
                    "function": "type",
                    "coordinate_x": 100,
                    "coordinate_y": 200,
                    "args": {"text": "hello"},
                },
            },
            {
                "execution_id": "exec",
                "request": "finish",
                "complete": "yes",
                "status": "OVERALL_FINISH",
                "thought": "finish",
                "subtask": "click",
                "screenshot": "1.png",
                "action": {
                    "function": "click",
                    "coordinate_x": 300,
                    "coordinate_y": 400,
                    "args": {},
                },
            },
        ],
    )

    row = gui360_use.build_trajectory("fake.jsonl", "word", "forms")

    assert len(row["images"]) == 2
    assert [m["role"] for m in row["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert len(_all_calls(row)) == 2
    first_actions = tool_call_arguments(_all_calls(row)[0])["actions"]
    assert [a["action"] for a in first_actions] == ["click", "type"]
    assert [a["action"] for a in tool_call_arguments(_all_calls(row)[1])["actions"]] == ["click"]
    assert row["messages"][2]["content"] == [{"type": "image", "index": 1}]
    _assert_final_action_row(row)


def test_gui360_sendkeys_preserves_order_repeats_and_hold_state() -> None:
    calls = gui360_use.parse_keys_string(
        "{HOME}{RIGHT 2}Annual{ENTER}{VK_CONTROL down}l{VK_CONTROL up}", "ep"
    )
    assert [action["action"] for action in _actions(calls)] == [
        "key", "key", "key", "type", "key", "key_down", "type", "key_up"
    ]
    assert _actions(gui360_use.parse_keys_string("^a", "ep")) == [
        {"action": "key", "keys": ["^"]},
        {"action": "type", "text": "a"},
    ]
    assert _actions(gui360_use.parse_keys_string("^+", "ep")) == [
        {"action": "key", "keys": ["^"]},
        {"action": "key", "keys": ["+"]},
    ]
    assert _actions(gui360_use.parse_keys_string("^+%", "ep")) == [
        {"action": "key", "keys": ["^"]},
        {"action": "key", "keys": ["+"]},
        {"action": "key", "keys": ["%"]},
    ]
    with pytest.raises(gui360_use.SkipTrajectory, match="dangling SendKeys modifier"):
        gui360_use.parse_keys_string("{VK_CONTROL}{VK_SHIFT}", "ep")
    assert _actions(gui360_use.parse_keys_string("{VK_CONTROL}({F1})", "ep")) == [
        {"action": "key", "keys": ["ctrl", "f1"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_CONTROL}{+}", "ep")) == [
        {"action": "key", "keys": ["ctrl", "+"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{+}", "ep")) == [
        {"action": "key", "keys": ["+"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_SHIFT}{+}", "ep")) == [
        {"action": "key", "keys": ["shift", "+"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_ADD}", "ep")) == [
        {"action": "key", "keys": ["+"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_OEM_PLUS}", "ep")) == [
        {"action": "key", "keys": ["+"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_OEM_MINUS}", "ep")) == [
        {"action": "key", "keys": ["-"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_RETURN}", "ep")) == [
        {"action": "key", "keys": ["enter"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_TAB}", "ep")) == [
        {"action": "key", "keys": ["tab"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_ESCAPE}", "ep")) == [
        {"action": "key", "keys": ["esc"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_BACK}", "ep")) == [
        {"action": "key", "keys": ["backspace"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_NEXT}", "ep")) == [
        {"action": "key", "keys": ["pagedown"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_CAPITAL}", "ep")) == [
        {"action": "key", "keys": ["capslock"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{F1}", "ep")) == [
        {"action": "key", "keys": ["f1"]}
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_MENU}", "ep")) == [
        {"action": "key", "keys": ["alt"]}
    ]


def test_gui360_sendkeys_distinguishes_text_punctuation_from_shortcuts() -> None:
    assert _actions(gui360_use.parse_keys_string("2 + 3 = 5", "ep")) == [
        {"action": "type", "text": "2 "},
        {"action": "key", "keys": ["+"]},
        {"action": "type", "text": " 3 = 5"},
    ]
    assert _actions(gui360_use.parse_keys_string("K^alpha is 50%", "ep")) == [
        {"action": "type", "text": "K"},
        {"action": "key", "keys": ["^"]},
        {"action": "type", "text": "alpha is 50"},
        {"action": "key", "keys": ["%"]},
    ]
    assert _actions(gui360_use.parse_keys_string("^α", "ep")) == [
        {"action": "key", "keys": ["^"]},
        {"action": "type", "text": "α"},
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_CONTROL} {SPACE}", "ep")) == [
        {"action": "key", "keys": ["ctrl", "space"]},
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_SHIFT} ", "ep")) == [
        {"action": "key", "keys": ["shift", "space"]},
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_ALT}n{VK_P}", "ep")) == [
        {"action": "key", "keys": ["alt", "n"]},
        {"action": "key", "keys": ["p"]},
    ]
    assert _actions(gui360_use.parse_keys_string("{VK_CONTROL}a", "ep")) == [
        {"action": "key", "keys": ["ctrl", "a"]},
    ]


@pytest.mark.parametrize(
    "keys",
    ["{ }", "{}", "{VK_NOT_A_KEY}", "{VK_PLUS}", "{VK_MINUS}", "{VK_EQUAL}"],
)
def test_gui360_sendkeys_rejects_empty_or_unknown_tokens(keys: str) -> None:
    with pytest.raises(gui360_use.SkipTrajectory, match="Unknown VK token"):
        gui360_use.parse_keys_string(keys, "ep")


def test_gui360_rejects_select_text_and_invalid_terminal_api(monkeypatch) -> None:
    with pytest.raises(gui360_use.SkipTrajectory, match="Office-API"):
        gui360_use.map_action(
            {
                "function": "select_text",
                "args": {"text": "agent"},
                "coordinate_x": 1,
                "coordinate_y": 2,
            },
            100, 100, "ep",
        )
    with pytest.raises(gui360_use.SkipTrajectory, match="unsupported click button"):
        gui360_use.map_action(
            {"function": "click", "args": {"button": "down"}, "coordinate_x": 1, "coordinate_y": 2},
            100, 100, "ep",
        )

    class _FakeImage:
        size = (100, 100)

    monkeypatch.setattr(gui360_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    monkeypatch.setattr(gui360_use.Image, "open", lambda _path: nullcontext(_FakeImage()))
    monkeypatch.setattr(gui360_use, "load_steps", lambda _path: [
        {
            "execution_id": "ep", "request": "make blue", "complete": "yes",
            "status": "RUNNING", "thought": "click", "subtask": "click", "screenshot": "0.png",
            "action": {"function": "click", "coordinate_x": 1, "coordinate_y": 2, "args": {}},
        },
        {
            "execution_id": "ep", "request": "make blue", "complete": "yes",
            "status": "OVERALL_FINISH", "thought": "blue", "subtask": "blue", "screenshot": "1.png",
            "action": {"function": "set_background_color", "args": {"color": "0000FF"}},
        },
    ])
    with pytest.raises(gui360_use.SkipTrajectory, match="Office-API"):
        gui360_use.build_trajectory("fake.jsonl", "ppt", "search")


def test_gui360_drag_preserves_modifier_hold() -> None:
    _, calls = gui360_use.map_action(
        {
            "function": "drag",
            "args": {"start_x": 1, "start_y": 2, "end_x": 3, "end_y": 4, "key_hold": "shift"},
        },
        100, 100, "ep",
    )
    assert [action["action"] for action in _actions(calls)] == ["key_down", "drag", "key_up"]
    assert _actions(calls)[0]["keys"] == ["shift"]
    assert _actions(calls)[2]["keys"] == ["shift"]

    _, calls = gui360_use.map_action(
        {
            "function": "drag",
            "args": {"start_x": 1, "start_y": 2, "end_x": 3, "end_y": 4, "key_hold": "+"},
        },
        100, 100, "ep",
    )
    assert [action["keys"] for action in _actions(calls) if action["action"] != "drag"] == [
        ["+"],
        ["+"],
    ]

    for key_hold in ("Insert a blank line", " "):
        with pytest.raises(gui360_use.SkipTrajectory, match="unsupported drag key_hold"):
            gui360_use.map_action(
                {
                    "function": "drag",
                    "args": {
                        "start_x": 1,
                        "start_y": 2,
                        "end_x": 3,
                        "end_y": 4,
                        "key_hold": key_hold,
                    },
                },
                100,
                100,
                "ep",
            )
