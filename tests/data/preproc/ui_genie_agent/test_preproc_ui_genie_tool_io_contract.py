"""UI-Genie preproc tool/result contract tests."""

from __future__ import annotations

import json

import pytest
from data.preproc._tool_io_helpers import (
    _all_calls,
    _assert_final_action_row,
    _assert_first_action_result_is_tool,
    _assert_no_terminate_outcome,
    _assert_structural_done_row,
    _assert_terminate_outcome,
)

from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.data.preproc.ui_genie_agent import use as ui_genie_use
from lite.data.utils.rows import validate_canonical_rows


def test_ui_genie_post_action_screenshot_is_tool(monkeypatch):
    monkeypatch.setattr(ui_genie_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")

    def rec(step: int, x: int) -> tuple[int, dict]:
        return _ui_genie_rec(
            step, {"action": "click", "coordinate": [x, 200], "action_desc": "tap"}
        )

    terminate = (
        2,
        {
            "images": ["data/screenshots/u/screenshot-2.png"],
            "messages": [
                {"content": "resolution is 1000x1000"},
                {"content": "The user query: finish\nTask progress: Step1: prior\nStep2: prior"},
                {
                    "content": (
                        "<tool_call>"
                        + json.dumps({
                            "arguments": {
                                "action": "terminate",
                                "status": "success",
                                "action_info": "done",
                            }
                        })
                        + "</tool_call>"
                    )
                },
            ],
        },
    )
    row = ui_genie_use.build_trajectory(
        "u",
        [rec(0, 100), rec(1, 300), terminate],
        ui_genie_use.SUBSETS["ui_genie"],
    )
    _assert_first_action_result_is_tool(row)
    assert row["messages"][-1]["content"] == [{"type": "text", "text": "done"}]


def test_ui_genie_rejects_steps_after_terminal(monkeypatch):
    monkeypatch.setattr(ui_genie_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    terminal = _ui_genie_rec(1, {"action": "terminate", "status": "success"})
    suffix = _ui_genie_rec(2, {"action": "click", "coordinate": [1, 1]})
    with pytest.raises(ui_genie_use.SkipTrajectoryError, match="post_terminal_steps"):
        ui_genie_use.build_trajectory(
            "u",
            [_ui_genie_rec(0, {"action": "click", "coordinate": [1, 1]}), terminal, suffix],
            ui_genie_use.SUBSETS["ui_genie"],
        )


def test_ui_genie_missing_terminal_terminate_keeps_final_action(monkeypatch):
    monkeypatch.setattr(ui_genie_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")

    row = ui_genie_use.build_trajectory(
        "u",
        [
            _ui_genie_rec(
                0,
                {"action": "click", "coordinate": [100, 200], "action_desc": "tap"},
            ),
            _ui_genie_rec(
                1,
                {"action": "click", "coordinate": [300, 400], "action_desc": "tap"},
            ),
        ],
        ui_genie_use.SUBSETS["ui_genie"],
    )

    assert [m["role"] for m in row["messages"]] == ["user", "assistant", "tool", "assistant"]
    _assert_final_action_row(row)


def _ui_genie_rec(step: int, args: dict, query: str = "finish") -> tuple[int, dict]:
    progress = "\n".join(f"Step{i}: prior" for i in range(1, step + 1)) or "none"
    return step, {
        "images": [f"data/screenshots/u/screenshot-{step}.png"],
        "messages": [
            {"content": "resolution is 1000x1000"},
            {"content": f"The user query: {query}\nTask progress: {progress}"},
            {"content": f"<tool_call>{json.dumps({'arguments': args})}</tool_call>"},
        ],
    }


def _ui_genie_terminal_row(monkeypatch, *, query: str, action_info: str | None) -> dict:
    """A 2-step ui_genie trajectory ending on ``terminate(success, action_info)``."""
    monkeypatch.setattr(ui_genie_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    terminate: dict = {"action": "terminate", "status": "success", "action_desc": "done"}
    if action_info is not None:
        terminate["action_info"] = action_info
    return ui_genie_use.build_trajectory(
        "u",
        [
            _ui_genie_rec(
                0,
                {"action": "click", "coordinate": [100, 200], "action_desc": "tap"},
                query=query,
            ),
            _ui_genie_rec(1, terminate, query=query),
        ],
        ui_genie_use.SUBSETS["ui_genie"],
    )


_UI_GENIE_QUESTION = (
    "You should use bluecoins to complete the following task: "
    "What is the current daily household budget?"
)


def test_ui_genie_terminal_answer_becomes_the_final_turn_text(monkeypatch):
    """A ``terminate`` that answers the question is re-homed onto the final turn.

    ``ui_genie`` hangs QA answers off ``terminate.action_info`` -- a field the canonical
    vocabulary reserves for ``response`` ("the only tool that carries answer text") --
    and ``terminate_outcome_others`` records nothing for ``status='success'``, so
    without this the source-authored result is deleted. No finish call is persisted:
    the text IS the content-only final, exactly as ``guiact`` / ``scalecua`` do it.
    """
    row = _ui_genie_terminal_row(
        monkeypatch,
        query=_UI_GENIE_QUESTION,
        action_info="The current daily household budget is set to $0.00.",
    )

    assert row["messages"][-1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "The current daily household budget is set to $0.00."}
        ],
    }
    assert all(tool_call_name(call) not in {"terminate", "response"} for call in _all_calls(row))
    assert row["metadata"]["extra_tool_schemas"] == []
    _assert_no_terminate_outcome(row)
    validate_canonical_rows([row], "ui_genie_answer")


def test_ui_genie_preserves_source_completion_text(monkeypatch):
    row = _ui_genie_terminal_row(
        monkeypatch,
        query="How to disable automatic time zone?",
        action_info="Successfully disabled automatic time zone.",
    )

    assert row["messages"][-1]["content"] == [
        {"type": "text", "text": "Successfully disabled automatic time zone."}
    ]
    _assert_no_terminate_outcome(row)
    validate_canonical_rows([row], "ui_genie_completion_text")


@pytest.mark.parametrize("action_info", ["The book ", '"The book "'])
def test_ui_genie_rejects_truncated_terminal_text(monkeypatch, action_info):
    with pytest.raises(ui_genie_use.SkipTrajectoryError, match="truncated_terminal_text"):
        _ui_genie_terminal_row(
            monkeypatch,
            query="How many pages does the first book have?",
            action_info=action_info,
        )


def test_ui_genie_preserves_source_text_for_action_goal(monkeypatch):
    row = _ui_genie_terminal_row(
        monkeypatch,
        query="Open the calendar app and create a new event titled 'Finance Meeting'.",
        action_info="The 'Finance Meeting' event was created.",
    )

    assert row["messages"][-1]["content"] == [
        {"type": "text", "text": "The 'Finance Meeting' event was created."}
    ]
    validate_canonical_rows([row], "ui_genie_action_goal_text")


def test_ui_genie_amex_shaped_terminate_has_no_answer(monkeypatch):
    """``amex`` authors no ``action_info`` at all (2,816/2,816 carry ``{status}``).

    Its rows must therefore be byte-identical to the pre-fix output, question-shaped
    goal or not.
    """
    row = _ui_genie_terminal_row(monkeypatch, query=_UI_GENIE_QUESTION, action_info=None)

    _assert_structural_done_row(row)


def test_ui_genie_source_terminal_text_replaces_done(monkeypatch):
    """Source-authored successful terminal text is preserved as the final turn.

    Every UI-Genie-Agent terminate in the raw corpus carries ``status='success'``
    (4 992/4 992 across both subsets), so the terminate turn only marks the end of
    the episode and carries no source-asserted outcome label.
    """
    monkeypatch.setattr(ui_genie_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = ui_genie_use.build_trajectory(
        "u",
        [
            _ui_genie_rec(0, {"action": "click", "coordinate": [100, 200], "action_desc": "tap"}),
            _ui_genie_rec(
                1,
                {"action": "terminate", "status": "success", "action_info": "all set"},
            ),
        ],
        ui_genie_use.SUBSETS["ui_genie"],
    )

    assert row["messages"][-1]["content"] == [{"type": "text", "text": "all set"}]
    assert [tool_call_name(call) for call in _all_calls(row)] == ["mobile"]
    validate_canonical_rows([row], "ui_genie_source_terminal_text")


def test_ui_genie_failure_terminate_moves_to_metadata_others(monkeypatch):
    """``status='failure'`` keeps its status and reason -- in ``others``."""
    monkeypatch.setattr(ui_genie_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = ui_genie_use.build_trajectory(
        "u",
        [
            _ui_genie_rec(0, {"action": "click", "coordinate": [100, 200], "action_desc": "tap"}),
            _ui_genie_rec(
                1,
                {"action": "terminate", "status": "failure", "action_info": "blocked"},
            ),
        ],
        ui_genie_use.SUBSETS["ui_genie"],
    )

    _assert_structural_done_row(row)
    assert [tool_call_name(call) for call in _all_calls(row)] == ["mobile"]
    _assert_terminate_outcome(row, status="failure", reason="blocked")
    validate_canonical_rows([row], "ui_genie")


def test_ui_genie_rejects_pixel_oob_before_normalization_clamp() -> None:
    with pytest.raises(ui_genie_use.SkipTrajectoryError, match="outside resolution"):
        ui_genie_use.args_to_tool_call(
            {"action": "click", "coordinate": [1200, 500]},
            width=1000,
            height=1000,
        )

    call = ui_genie_use.args_to_tool_call(
        {"action": "click", "coordinate": [-1e-7, 1000 + 1e-7]},
        width=1000,
        height=1000,
    )
    assert tool_call_arguments(call)["actions"][0]["coordinate"] == [0, 1000]


@pytest.mark.parametrize("action", ["wait", "long_press"])
def test_ui_genie_rejects_malformed_source_duration(action: str) -> None:
    args = {"action": action, "time": "not-a-duration"}
    if action == "long_press":
        args["coordinate"] = [1, 1]
    with pytest.raises(ui_genie_use.SkipTrajectoryError, match="malformed_duration"):
        ui_genie_use.args_to_tool_call(args, width=1000, height=1000)


def test_ui_genie_orders_by_logical_history_and_rejects_real_gap(monkeypatch) -> None:
    monkeypatch.setattr(ui_genie_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    first = _ui_genie_rec(0, {"action": "click", "coordinate": [1, 2]})
    second = _ui_genie_rec(1, {"action": "terminate", "status": "success"})
    second[1]["images"] = first[1]["images"]
    row = ui_genie_use.build_trajectory(
        "u", [second, first], ui_genie_use.SUBSETS["ui_genie"]
    )
    assert len(row["images"]) == 2
    assert row["messages"][2]["role"] == "tool"

    with pytest.raises(ui_genie_use.SkipTrajectoryError, match="non_contiguous"):
        ui_genie_use.build_trajectory(
            "u",
            [first, _ui_genie_rec(2, {"action": "terminate", "status": "success"})],
            ui_genie_use.SUBSETS["ui_genie"],
        )
