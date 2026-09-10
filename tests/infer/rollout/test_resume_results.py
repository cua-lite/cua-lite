"""Tests for rollout summary CI gates and debug log checks.

Validates the opt-in CI assertions wired into the rollout entrypoints
(``scripts/rollout.py``) via
``--min-valid-frac``, ``--min-mean-return``, and ``--debug``.

Run::

    uv run pytest tests/infer/rollout/test_resume_results.py -v
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lite.core.tools.calls import (
    make_tool_call,
)
from lite.infer.rollout import (
    TaskSpec,
    get_pending,
    print_results,
    rebuild_results,
)
from lite.utils.parquet import write_records_to_parquet


def _write_summary(tmp_path: Path, stats: dict) -> Path:
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"config": {}, "stats": stats, "tasks": []}))
    return p


def _metadata(
    *,
    task_type: str = "use",
    platform: str = "desktop",
    extra_tool_schemas: list[dict] | None = None,
    terminated: bool | None = None,
    truncated: bool | None = None,
) -> dict:
    metadata = {
        "metadata_kind": "cua",
        "dims": [platform, task_type],
        "extra_tool_schemas": extra_tool_schemas or [],
        "valid_actions": None,
        "others": {
            "env_id": "test.env",
            "task_id": "task_0",
        },
    }
    if terminated is not None:
        metadata["others"]["terminated"] = terminated
    if truncated is not None:
        metadata["others"]["truncated"] = truncated
    return metadata


def _write_trajectory(
    sample_dir: Path,
    messages: list[dict],
    *,
    metadata: dict | None = None,
    images: list[str] | None = None,
    json_fields: tuple[str, ...] | None = None,
) -> Path:
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / "trajectory.parquet"
    write_records_to_parquet([{
        "images": images or [],
        "messages": messages,
        "metadata": metadata or _metadata(),
    }], path, json_fields=json_fields)
    return path


def _text(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


def _computer_call(call_id: str = "call_0") -> dict:
    return make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": [10, 20]}]},
        call_id=call_id,
    )


def _messages_with_tool_result(tool_text: str, *, call_id: str = "call_0") -> list[dict]:
    return [
        {"role": "user", "content": _text("task")},
        {"role": "assistant", "content": [], "tool_calls": [_computer_call(call_id)]},
        {"role": "tool", "tool_call_id": call_id, "content": _text(tool_text)},
        {"role": "assistant", "content": _text("Done.")},
    ]


def _write_turn_debug_payload(
    sample_dir: Path,
    *,
    call_id: str = "call_0",
    tool_name: str = "computer",
    arguments: dict | None = None,
    text: str | None = "obs",
    error: str | None = "bad",
) -> Path:
    turn = sample_dir / "turn_0000"
    turn.mkdir(parents=True, exist_ok=True)
    (turn / "01_prompt.txt").write_text("prompt")
    (turn / "02_response.txt").write_text("response")
    if arguments is None:
        arguments = {"actions": [{"action": "click", "coordinate": [10, 20]}]}
    (turn / "03_actions.json").write_text(json.dumps({
        "lite_message": {
            "role": "assistant",
            "content": [],
            "tool_calls": [make_tool_call(tool_name, arguments, call_id=call_id)],
        },
    }))
    result = {
        "tool_call_id": call_id,
        "images": [],
        "text": text,
        "metadata": None,
        "error": error,
    }
    (turn / "04_results.json").write_text(json.dumps({
        "reward": 0.0,
        "terminated": True,
        "truncated": False,
        "results": [result],
        "info": {},
    }))
    return turn


def _write_empty_turn_debug_payload(sample_dir: Path) -> Path:
    turn = sample_dir / "turn_0000"
    turn.mkdir(parents=True, exist_ok=True)
    (turn / "01_prompt.txt").write_text("prompt")
    (turn / "02_response.txt").write_text("unparseable output")
    (turn / "03_actions.json").write_text(json.dumps({"lite_message": {}}))
    (turn / "04_results.json").write_text(json.dumps({
        "reward": 0.0,
        "terminated": True,
        "truncated": False,
        "results": [],
        "info": {},
    }))
    return turn


def _write_result_image(
    turn: Path,
    name: str = "0000_0000_from_call_0.png",
    data: bytes = b"png",
) -> dict:
    path = turn / "env_result_images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": str(Path("env_result_images") / name),
        "source": "env_result_images",
        "bytes": len(data),
        "sha1": hashlib.sha1(data).hexdigest(),
    }


def _params() -> dict:
    return {
        "type": "object",
        "properties": {
            "app_name": {"type": "string"},
            "command": {"type": "string"},
            "coordinate": {"type": "array"},
            "reason": {"type": "string"},
            "status": {"type": "string"},
            "text": {"type": "string"},
            "url": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": [],
    }


_PROJECTED_NATIVE_MISMATCH = "does not match projected native text/error"


def test_parse_failure_summary_is_a_scored_measurement_not_an_error(tmp_path: Path):
    """A parse failure is a measurement: not re-run, and scored at the env's return.

    The env evaluates the parse-failure turn (the agent sends it as a ``response``
    finish call), so its ``episode_return`` is real. Only ``error`` marks a summary
    as unmeasured; a durable ``stop_reason`` must not.
    """
    spec = TaskSpec(task_id="task_0", env_id="test.env")
    sample_dir = spec.sample_dir(tmp_path, 0)
    sample_dir.mkdir(parents=True)
    (sample_dir / "summary.json").write_text(json.dumps({
        "n_turns": 1,
        "episode_return": 0.0,
        "terminated": True,
        "truncated": False,
        "stop_reason": "parse_failure",
    }))

    assert get_pending(tmp_path, [spec], group_size=1) == []

    results = rebuild_results(tmp_path, [spec], group_size=1)
    assert results[0]["error"] is None
    assert results[0]["stop_reason"] == "parse_failure"
    assert results[0]["episode_return"] == 0.0

    stats = print_results(results, [spec], group_size=1)
    assert stats["num_valid"] == 1
    assert stats["mean_episode_return"] == 0.0
    assert stats["stop_reasons"] == {"parse_failure": 1}


def test_parse_failure_that_solved_the_task_keeps_its_score(tmp_path: Path):
    """The env's verdict decides, not the status: a 1.0 parse failure scores 1.0."""
    spec = TaskSpec(task_id="task_0", env_id="test.env")
    sample_dir = spec.sample_dir(tmp_path, 0)
    sample_dir.mkdir(parents=True)
    (sample_dir / "summary.json").write_text(json.dumps({
        "n_turns": 3,
        "episode_return": 1.0,
        "terminated": True,
        "truncated": False,
        "stop_reason": "parse_failure",
    }))

    results = rebuild_results(tmp_path, [spec], group_size=1)
    assert results[0]["error"] is None

    stats = print_results(results, [spec], group_size=1)
    assert stats["num_valid"] == 1
    assert stats["mean_episode_return"] == 1.0


def test_terminal_error_summary_stays_out_of_the_score(tmp_path: Path):
    """A summary carrying ``error`` is a tombstone: resolved, but never scored.

    The counterpart to the two parse-failure tests above — the whole change turns
    on ``error`` vs ``stop_reason``, so both sides belong here.
    ``test_task_resolution.py`` pins the resolved/error half; this pins the
    ``num_valid`` half, which is the number the change moves.
    """
    spec = TaskSpec(task_id="task_0", env_id="test.env")
    sample_dir = spec.sample_dir(tmp_path, 0)
    sample_dir.mkdir(parents=True)
    (sample_dir / "summary.json").write_text(json.dumps({
        "n_turns": 0,
        "episode_return": 0.0,
        "terminated": False,
        "truncated": False,
        "error": "EnvBlocked: task is unscoreable",
    }))

    assert get_pending(tmp_path, [spec], group_size=1) == []

    results = rebuild_results(tmp_path, [spec], group_size=1)
    assert results[0]["error"] == "EnvBlocked: task is unscoreable"

    stats = print_results(results, [spec], group_size=1)
    assert stats["num_valid"] == 0


def test_summary_without_parquet_is_resolved_for_resume(tmp_path: Path):
    spec = TaskSpec(task_id="task_0", env_id="test.env")
    sample_dir = spec.sample_dir(tmp_path, 0)
    sample_dir.mkdir(parents=True)
    (sample_dir / "summary.json").write_text(json.dumps({
        "n_turns": 0,
        "episode_return": 1.0,
        "terminated": True,
        "truncated": False,
    }))

    assert not (sample_dir / "trajectory.parquet").exists()
    assert get_pending(tmp_path, [spec], group_size=1) == []

    results = rebuild_results(tmp_path, [spec], group_size=1)
    assert results == [{
        "task": "task_0",
        "env_id": "test.env",
        "group_idx": 0,
        "sample_idx": 0,
        "turns": 0,
        "episode_return": 1.0,
        "terminated": True,
        "truncated": False,
        "stop_reason": None,
        "error": None,
    }]

