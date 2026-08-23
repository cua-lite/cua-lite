"""OpenCUA preproc tool/result contract tests."""

from __future__ import annotations

import json

import pytest
from data.preproc._tool_io_helpers import (
    _actions,
    _all_calls,
    _assert_final_action_row,
    _assert_first_action_result_is_tool,
    _assert_no_terminate_outcome,
    _assert_structural_done_row,
    _assert_terminate_outcome,
)

from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.data.preproc.opencua import use as opencua_use
from lite.data.staging import iter_parquet_rows, write_partition
from lite.data.utils.rows import validate_canonical_rows


def test_opencua_terminal_only_terminate_becomes_done_and_tool_result(monkeypatch):
    monkeypatch.setattr(opencua_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    monkeypatch.setattr(opencua_use, "get_image_resolution", lambda _path: [1000, 1000])
    row = opencua_use.record_to_example(
        {
            "task_id": "task_1",
            "instruction": "finish",
            "traj": [
                {
                    "image": "s0.png",
                    "value": {
                        "thought": "click",
                        "action": "click",
                        "code": "pyautogui.click(x=0.1, y=0.2)",
                    },
                },
                {
                    "image": "s1.png",
                    "value": {
                        "thought": "done",
                        "action": "done",
                        "code": "computer.terminate(status='success')",
                    },
                },
            ],
        },
        images_dir=None,
        relative_root="AgentNet/images",
        dataset_type="ubuntu",
        record_idx=0,
        os_by_task_id={"task_1": "ubuntu"},
    )

    _assert_first_action_result_is_tool(row)
    _assert_structural_done_row(row)
    calls = _all_calls(row)
    assert [tool_call_name(call) for call in calls] == ["computer"]
    assert tool_call_arguments(calls[0])["actions"] == [
        {"action": "click", "coordinate": [100, 200]},
    ]
    assert row["metadata"]["extra_tool_schemas"] == []


def _opencua_row(monkeypatch, final_code: str, *, record_extra: dict | None = None) -> dict:
    monkeypatch.setattr(opencua_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    monkeypatch.setattr(opencua_use, "get_image_resolution", lambda _path: [1000, 1000])
    return opencua_use.record_to_example(
        {
            **(record_extra or {}),
            "task_id": "task_fail",
            "instruction": "finish",
            "traj": [
                {
                    "image": "s0.png",
                    "value": {
                        "thought": "click",
                        "action": "click",
                        "code": "pyautogui.click(x=0.1, y=0.2)",
                    },
                },
                {
                    "image": "s1.png",
                    "value": {"thought": "give up", "action": "stop", "code": final_code},
                },
            ],
        },
        images_dir=None,
        relative_root="AgentNet/images",
        dataset_type="ubuntu",
        record_idx=0,
        os_by_task_id={"task_fail": "ubuntu"},
    )


def test_opencua_failure_terminate_moves_to_metadata_others(monkeypatch):
    """AgentNet parses ``status`` as a first-class outcome.

    This terminal-marker row ends on ``Done.``, and the self-reported failure
    moves to ``metadata.others`` -- dropping it outright would relabel a
    demonstration the source explicitly marks as failed.
    """
    row = _opencua_row(monkeypatch, "computer.terminate(status='failure')")

    _assert_structural_done_row(row)
    assert [tool_call_name(call) for call in _all_calls(row)] == ["computer"]
    _assert_terminate_outcome(row, status="failure")
    validate_canonical_rows([row], "opencua")


def test_opencua_self_report_never_overwrites_external_task_completed(monkeypatch):
    """``task_completed`` is an external judgement; ``status`` a self-report.

    On published rows the two disagree (``task_completed=False`` alongside
    ``status='success'``), so they keep separate keys and neither collapses into
    the other.
    """
    row = _opencua_row(
        monkeypatch,
        "computer.terminate(status='success')",
        record_extra={"task_completed": False},
    )

    _assert_structural_done_row(row)
    _assert_no_terminate_outcome(row)
    assert row["metadata"]["others"]["task_completed"] is False


def test_opencua_normalizes_fail_alias_to_failure(monkeypatch):
    """``status='fail'`` is normalized so the recorded label stays in the enum."""
    row = _opencua_row(monkeypatch, "computer.terminate(status='fail')")

    _assert_terminate_outcome(row, status="failure")
    validate_canonical_rows([row], "opencua_fail_alias")


def test_opencua_non_terminating_final_keeps_final_action(monkeypatch):
    """A demonstration that ends on an executable action keeps that label."""
    monkeypatch.setattr(opencua_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    monkeypatch.setattr(opencua_use, "get_image_resolution", lambda _path: [1000, 1000])
    row = opencua_use.record_to_example(
        {
            "task_id": "task_2",
            "instruction": "type",
            "traj": [
                {
                    "image": "s0.png",
                    "value": {
                        "thought": "click",
                        "action": "click",
                        "code": "pyautogui.click(x=0.1, y=0.2)",
                    },
                },
                {
                    "image": "s1.png",
                    "value": {
                        "thought": "type",
                        "action": "type",
                        "code": "pyautogui.write('hi')",
                    },
                },
            ],
        },
        images_dir=None,
        relative_root="AgentNet/images",
        dataset_type="ubuntu",
        record_idx=0,
        os_by_task_id={"task_2": "ubuntu"},
    )

    assert [m["role"] for m in row["messages"]] == ["user", "assistant", "tool", "assistant"]
    _assert_final_action_row(row)


def test_opencua_iter_examples_head_stops_at_trajectory_boundary(tmp_path, monkeypatch) -> None:
    """``--head`` counts whole AgentNet trajectories, not partial steps."""
    monkeypatch.setattr(opencua_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    monkeypatch.setattr(opencua_use, "get_image_resolution", lambda _path: [1000, 1000])

    def record(i: int) -> dict:
        return {
            "task_id": f"task_{i}",
            "instruction": "finish",
            "traj": [
                {
                    "image": f"{i}_s0.png",
                    "value": {
                        "thought": "click",
                        "action": "click",
                        "code": "pyautogui.click(x=0.1, y=0.2)",
                    },
                },
                {
                    "image": f"{i}_s1.png",
                    "value": {
                        "thought": "done",
                        "action": "done",
                        "code": "computer.terminate(status='success')",
                    },
                },
            ],
        }

    jsonl = tmp_path / "agentnet.jsonl"
    jsonl.write_text("".join(json.dumps(record(i)) + "\n" for i in range(5)))

    kwargs = dict(
        jsonl_path=jsonl,
        images_dir=None,
        relative_root="AgentNet/images",
        dataset_type="ubuntu",
        os_by_task_id={f"task_{i}": "ubuntu" for i in range(5)},
    )
    bounded = list(opencua_use.iter_examples(head=2, **kwargs))
    full = list(opencua_use.iter_examples(**kwargs))

    assert len(full) == 5
    assert len(bounded) == 2
    assert [r["messages"] for r in bounded] == [r["messages"] for r in full[:2]]

    parquet_path = tmp_path / "persisted" / "opencua_use.parquet"
    write_partition(bounded, parquet_path)
    persisted = list(iter_parquet_rows(parquet_path))
    assert len(persisted) == len(bounded)
    validate_canonical_rows(persisted, "opencua/use")


def test_opencua_preserves_press_repeats_and_media_key_aliases() -> None:
    actions = _actions(opencua_use.agentnet_code_to_tool_calls(
        "pyautogui.press('left', presses=3)\n"
        "pyautogui.press('media_volume_down', presses=2)"
    ))
    assert [action["keys"] for action in actions] == [
        ["left"], ["left"], ["left"], ["volumedown"], ["volumedown"]
    ]


def test_opencua_accepts_agentnet_numpad_multiply_alias() -> None:
    assert _actions(opencua_use.agentnet_code_to_tool_calls("pyautogui.press('num*')")) == [
        {"action": "key", "keys": ["*"]},
    ]


def test_opencua_validates_keys_and_accepts_the_keyword_press_signature() -> None:
    actions = _actions(opencua_use.agentnet_code_to_tool_calls(
        "pyautogui.press(keys=[')'], presses=2)"
    ))
    assert [action["keys"] for action in actions] == [[")"], [")"]]
    assert _actions(opencua_use.agentnet_code_to_tool_calls(
        "pyautogui.hotkey(['ctrl', '+'])"
    )) == [{"action": "key", "keys": ["ctrl", "+"]}]
    assert _actions(opencua_use.agentnet_code_to_tool_calls(
        "pyautogui.hotkey(['ctrl', 'plus'])"
    )) == [{"action": "key", "keys": ["ctrl", "+"]}]
    assert _actions(opencua_use.agentnet_code_to_tool_calls(
        "pyautogui.hotkey(['controlright', 'cmdleft', 'a'])"
    )) == [{"action": "key", "keys": ["ctrl", "meta", "a"]}]
    assert [action["keys"] for action in _actions(opencua_use.agentnet_code_to_tool_calls(
        "pyautogui.press('+')\npyautogui.press('-')\npyautogui.press('=')\n"
        "pyautogui.press('plus')"
    ))] == [["+"], ["-"], ["="], ["+"]]

    for code in (
        "pyautogui.press('8000')",
        "pyautogui.hotkey(['ctrl', 'ac'])",
        "pyautogui.press(' ')",
        "pyautogui.press('\\n')",
    ):
        with pytest.raises(opencua_use.AgentNetCodeParseError, match="Unsupported key token"):
            opencua_use.agentnet_code_to_tool_calls(code)
