"""Tests for rollout summary CI gates and debug log checks.

Validates the opt-in CI assertions wired into the rollout entrypoints
(``scripts/rollout.py``) via
``--min-valid-frac``, ``--min-mean-return``, and ``--debug``.

Run::

    uv run pytest tests/infer/debug/test_log_contract.py -v
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lite.core import LiteGenericMetadata
from lite.core.tools.calls import (
    make_tool_call,
)
from lite.core.tools.results import project_tool_result_text
from lite.core.tools.schemas import make_tool_schema
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


def test_log_contract_helper_stays_in_debug_namespace():
    from lite.infer import rollout

    assert not hasattr(rollout, "check_log_contract")
    assert importlib.util.find_spec("lite.infer.debug.log_contract") is not None


def test_log_contract_stays_read_only_raw_rollout_checker():
    source = (
        Path(__file__).resolve().parents[3] / "lite" / "infer" / "debug"
        / "log_contract.py"
    ).read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    used_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }

    assert "validate_raw_rollout_rows" in imported
    assert "validate_canonical_rows" not in imported
    assert source.count("validate_raw_rollout_rows(") == 1
    assert "_GROUNDING_TASK_TYPES" not in source
    assert "legacy_grounding_tool_results" not in source
    assert "_without_explicit_legacy_grounding_tool_result_messages" not in source
    assert "orphan role:tool result" not in source
    assert "no previous assistant tool_call" not in source
    assert not (
        used_attrs
        & {
            "mkdir",
            "rename",
            "replace",
            "rmdir",
            "touch",
            "unlink",
            "write_bytes",
            "write_text",
        }
    )


def test_log_contract_gate_rejects_summary_only_run(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    summary = _write_summary(
        tmp_path,
        {"num_tasks": 1, "num_valid": 1, "mean_episode_return": 1.0},
    )
    sample_dir = tmp_path / "sample_00"
    sample_dir.mkdir()
    (sample_dir / "summary.json").write_text(summary.read_text())

    reasons = check_log_contract(tmp_path)
    assert any("trajectory.parquet" in r for r in reasons)


def test_log_contract_gate_accepts_explicit_envblocked_void_sample(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    sample_dir.mkdir()
    error_text = "lite.gym.errors.EnvBlocked: site blocked"
    (sample_dir / "summary.json").write_text(json.dumps({
        "n_turns": 0,
        "episode_return": 0.0,
        "terminated": False,
        "truncated": False,
        "error": error_text,
    }))
    (sample_dir / "error.txt").write_text(error_text)

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_rejects_non_envblocked_void_sample(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    sample_dir.mkdir()
    error_text = "RuntimeError: worker crashed"
    (sample_dir / "summary.json").write_text(json.dumps({
        "n_turns": 0,
        "episode_return": 0.0,
        "terminated": False,
        "truncated": False,
        "error": error_text,
    }))
    (sample_dir / "error.txt").write_text(error_text)

    reasons = check_log_contract(tmp_path)
    assert any("trajectory.parquet" in r for r in reasons)


def test_log_contract_gate_rejects_missing_sample_trajectories(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_summary(
        tmp_path,
        {"num_tasks": 1, "num_valid": 1, "mean_episode_return": 1.0},
    )

    reasons = check_log_contract(tmp_path)
    assert any("no sample_* trajectory directories found" in r for r in reasons)


def test_log_contract_gate_accepts_json_string_messages_and_metadata(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    sample_dir.mkdir(parents=True)
    write_records_to_parquet(
        [{
            "images": [],
            "messages": _messages_with_tool_result("ok"),
            "metadata": _metadata(),
        }],
        sample_dir / "trajectory.parquet",
        json_fields=("messages", "metadata"),
    )

    assert check_log_contract(sample_dir) == []


def test_log_contract_gate_accepts_tagged_generic_metadata(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        [
            {"role": "user", "content": _text("answer")},
            {
                "role": "assistant",
                "tool_calls": [
                    make_tool_call("response", {"text": "42"}, call_id="call_response")
                ],
            },
        ],
        metadata=LiteGenericMetadata(
            dims=("geo3k", "sft"),
            extra_tool_schemas=[make_tool_schema("response", parameters=_params())],
            others={"source": "unit"},
        ).to_dict(),
        json_fields=("messages", "metadata"),
    )

    assert check_log_contract(sample_dir) == []


def test_log_contract_gate_checks_image_paths_and_indices(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    image_path = tmp_path / "obs.png"
    image_path.write_bytes(b"png")
    good_messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}] + _text("task")},
        {"role": "assistant", "content": _text("Done.")},
    ]
    _write_trajectory(
        tmp_path / "sample_good",
        good_messages,
        images=[str(image_path)],
        json_fields=("messages", "metadata"),
    )
    assert check_log_contract(tmp_path / "sample_good") == []

    _write_trajectory(
        tmp_path / "sample_bad_path",
        good_messages,
        images=[str(tmp_path / "missing.png")],
        json_fields=("messages", "metadata"),
    )
    missing_reasons = check_log_contract(tmp_path / "sample_bad_path")
    assert any("image path does not exist" in r for r in missing_reasons)

    _write_trajectory(
        tmp_path / "sample_bad_index",
        [
            {"role": "user", "content": [{"type": "image", "index": 1}] + _text("task")},
            {"role": "assistant", "content": _text("Done.")},
        ],
        images=[str(image_path)],
        json_fields=("messages", "metadata"),
    )
    index_reasons = check_log_contract(tmp_path / "sample_bad_index")
    assert any("out of range for images length 1" in r for r in index_reasons)


def test_log_contract_accepts_project_root_relative_logger_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from lite.infer.debug import log_contract
    from lite.infer.debug.log_contract import check_log_contract

    project = tmp_path / "project"
    rel_image = Path(".logs/rollout/task/sample_00/images/000000.png")
    image_path = project / rel_image
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")

    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(log_contract, "project_root", lambda: project)

    sample_dir = tmp_path / "run" / "sample_00"
    _write_trajectory(
        sample_dir,
        [
            {"role": "user", "content": [{"type": "image", "index": 0}] + _text("task")},
            {"role": "assistant", "content": _text("Done.")},
        ],
        images=[str(rel_image)],
        json_fields=("messages", "metadata"),
    )

    assert check_log_contract(sample_dir) == []


def test_log_contract_module_cli_reports_failures(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "lite.infer.debug.log_contract", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "no sample_* trajectory directories found" in result.stderr


def test_log_contract_gate_checks_turn_debug_artifact_shape(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        _messages_with_tool_result(
            "obs\n\n## Error from previous action:\nbad",
        ),
    )
    turn = _write_turn_debug_payload(sample_dir)
    (turn / "prompt_images").mkdir(parents=True)
    (turn / "prompt_images_annotated").mkdir()
    (turn / "prompt_images" / "0000_reset.png").write_bytes(b"png")
    (turn / "prompt_images_annotated" / "0000_reset.png").write_bytes(b"png")

    assert check_log_contract(tmp_path) == []

    (turn / "04_results.json").write_text(json.dumps({
        "reward": 0.0,
        "terminated": True,
        "truncated": False,
        "results": [{
            "tool_call_id": "call_0",
            "images": [],
            "text": "obs",
            "metadata": None,
        }],
        "info": {},
    }))
    (turn / "prompt_images_annotated" / "9999_missing.png").write_bytes(b"png")
    (turn / "00_screenshot.png").write_bytes(b"legacy")

    reasons = check_log_contract(tmp_path)
    assert any("missing keys ['error']" in r for r in reasons)
    assert any(
        "annotated image has no matching prompt_images/9999_missing.png" in r
        for r in reasons
    )
    assert any("stale legacy artifact 00_screenshot.png" in r for r in reasons)


def test_log_contract_gate_rejects_legacy_turn_prompt_image_dirs(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        _messages_with_tool_result(
            "obs\n\n## Error from previous action:\nbad",
        ),
    )
    turn = _write_turn_debug_payload(sample_dir)
    (turn / "images").mkdir(parents=True)
    (turn / "annotated").mkdir()
    (turn / "images" / "0000_reset.png").write_bytes(b"png")
    (turn / "annotated" / "0000_reset.png").write_bytes(b"png")

    reasons = check_log_contract(tmp_path)
    assert any("stale legacy artifact images/" in r for r in reasons)
    assert any("stale legacy artifact annotated/" in r for r in reasons)


def test_log_contract_gate_allows_terminal_turn_results_to_omit_call_id_feedback(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        _messages_with_tool_result("current screenshot after max_steps"),
        metadata=_metadata(truncated=True),
    )
    turn = _write_turn_debug_payload(
        sample_dir,
        text="current screenshot after max_steps",
        error=None,
    )
    payload = json.loads((turn / "04_results.json").read_text())
    payload["terminated"] = False
    payload["truncated"] = True
    payload["results"] = []
    (turn / "04_results.json").write_text(json.dumps(payload))

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_rejects_legacy_result_images_payload_in_current_results(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        _messages_with_tool_result(
            "obs\n\n## Error from previous action:\nbad",
        ),
    )
    turn = _write_turn_debug_payload(sample_dir)
    legacy_dir = turn / "result_images"
    legacy_dir.mkdir()
    (legacy_dir / "0000_from_call_0.png").write_bytes(b"result-png")
    payload = json.loads((turn / "04_results.json").read_text())
    payload["results"][0]["images"] = [{
        "path": "result_images/0000_from_call_0.png",
        "source": "result_images",
        "bytes": len(b"result-png"),
        "sha1": hashlib.sha1(b"result-png").hexdigest(),
    }]
    (turn / "04_results.json").write_text(json.dumps(payload))

    reasons = check_log_contract(tmp_path)
    assert any("image.path must be relative under env_result_images/" in r for r in reasons)
    assert any("stale legacy artifact result_images/" in r for r in reasons)


def test_log_contract_gate_accepts_env_result_image_refs(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        [
            {"role": "user", "content": [{"type": "image", "index": 0}] + _text("task")},
            {"role": "assistant", "content": [], "tool_calls": [_computer_call("call_0")]},
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": [
                    {"type": "image", "index": 1},
                    {"type": "text", "text": "obs\n\n## Error from previous action:\nbad"},
                ],
            },
            {"role": "assistant", "content": _text("Done.")},
        ],
        images=["images/000000.png", "images/000001.png"],
        json_fields=("messages", "metadata"),
    )
    image_dir = sample_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "000000.png").write_bytes(b"initial-png")
    (image_dir / "000001.png").write_bytes(b"canonical-png")
    turn = _write_turn_debug_payload(sample_dir)
    image_info = _write_result_image(turn, data=b"env-png")
    payload = json.loads((turn / "04_results.json").read_text())
    payload["results"][0]["images"] = [image_info]
    (turn / "04_results.json").write_text(json.dumps(payload))

    assert check_log_contract(tmp_path) == []

    payload["results"][0]["images"][0]["path"] = "images/000000.png"
    (turn / "04_results.json").write_text(json.dumps(payload))
    reasons = check_log_contract(tmp_path)
    assert any("image.path must be relative under env_result_images/" in r for r in reasons)


def test_log_contract_gate_rejects_legacy_05_results_json(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        _messages_with_tool_result(
            "obs\n\n## Error from previous action:\nbad",
        ),
    )
    turn = _write_turn_debug_payload(sample_dir)
    (turn / "04_results.json").replace(turn / "05_results.json")

    reasons = check_log_contract(tmp_path)
    assert any("missing 04_results.json" in r for r in reasons)
    assert any("stale legacy artifact 05_results.json" in r for r in reasons)


def test_log_contract_gate_rejects_malformed_results_json_shape(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(sample_dir, _messages_with_tool_result("obs"))
    turn = _write_turn_debug_payload(sample_dir)
    payload = json.loads((turn / "04_results.json").read_text())
    payload.pop("info")
    payload["terminated"] = "true"
    payload["results"][0]["images"] = [{
        "path": "../bad.png",
        "source": "env_result_images",
        "bytes": 0,
        "sha1": hashlib.sha1(b"").hexdigest(),
    }]
    (turn / "04_results.json").write_text(json.dumps(payload))

    reasons = check_log_contract(tmp_path)
    assert any("missing top-level key 'info'" in r for r in reasons)
    assert any("terminated must be a bool" in r for r in reasons)
    assert any("image.path must be relative under env_result_images/" in r for r in reasons)


def test_log_contract_gate_rejects_duplicate_result_image_references(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(sample_dir, _messages_with_tool_result("obs"))
    turn = _write_turn_debug_payload(sample_dir)
    image_info = _write_result_image(turn, data=b"shared")
    payload = json.loads((turn / "04_results.json").read_text())
    payload["results"] = [
        {**payload["results"][0], "images": [image_info]},
        {
            "tool_call_id": "call_1",
            "images": [image_info],
            "text": None,
            "metadata": None,
            "error": None,
        },
    ]
    (turn / "04_results.json").write_text(json.dumps(payload))

    reasons = check_log_contract(tmp_path)
    assert any(
        "duplicate result image path env_result_images/0000_0000_from_call_0.png" in r
        for r in reasons
    )


@pytest.mark.parametrize(
    "projected_text",
    [
        "obs",
        "## Error from previous action:\nbad",
    ],
)
def test_log_contract_gate_checks_projected_error_text_from_native_results(
    tmp_path: Path,
    projected_text: str,
):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(sample_dir, _messages_with_tool_result(projected_text))
    _write_turn_debug_payload(sample_dir)

    reasons = check_log_contract(tmp_path)
    assert any(_PROJECTED_NATIVE_MISMATCH in r for r in reasons)


def test_log_contract_gate_checks_projected_error_text_without_native_text(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(sample_dir, _messages_with_tool_result("unrelated text"))
    _write_turn_debug_payload(sample_dir, text=None, error="unsupported action: search")

    reasons = check_log_contract(tmp_path)
    assert any(_PROJECTED_NATIVE_MISMATCH in r for r in reasons)


def test_log_contract_gate_accepts_truncated_native_observation_text_with_error(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    full_observation = "## AXTree:\n" + ("node\n" * 120)
    truncated_observation = full_observation[:200] + f"... [{len(full_observation)} chars total]"
    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        _messages_with_tool_result(
            f"{full_observation}\n## Error from previous action:\nbad"
        ),
    )
    _write_turn_debug_payload(sample_dir, text=truncated_observation)

    reasons = check_log_contract(tmp_path)
    assert not any(_PROJECTED_NATIVE_MISMATCH in r for r in reasons)


def test_log_contract_gate_accepts_truncated_native_error_text(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    observation = "CURRENT URL: http://example.test\nDOM:\n[Turn 4/15]"
    full_error = "invalid action arguments: " + ("stack frame\n" * 120)
    truncated_error = full_error[:200] + f"... [{len(full_error)} chars total]"
    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        _messages_with_tool_result(
            f"{observation}\n\n## Error from previous action:\n{full_error}"
        ),
    )
    _write_turn_debug_payload(sample_dir, text=observation, error=truncated_error)

    reasons = check_log_contract(tmp_path)
    assert not any(_PROJECTED_NATIVE_MISMATCH in r for r in reasons)


def test_log_contract_gate_rejects_provider_envelope_and_tool_call_id(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {
                        "tool_call_id": "call_0",
                        "type": "function",
                        "function": {"name": "click", "arguments": "{}"},
                    },
                ],
            },
        ],
    )
    _write_trajectory(
        tmp_path / "sample_01",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {
                        "call_id": "call_1",
                        "name": "bash",
                        "arguments": "{}",
                    },
                ],
            },
        ],
    )

    reasons = check_log_contract(tmp_path)
    assert any("must use id on tool calls, not tool_call_id" in r for r in reasons)
    assert any("noncanonical outer keys ['arguments', 'call_id', 'name']" in r for r in reasons)
    assert any("devs/migration" in r for r in reasons)


def test_log_contract_gate_treats_parquet_null_padding_as_absent(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "computer",
                        "arguments": {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    },
                    "tool_call_id": None,
                    "call_id": None,
                    "name": None,
                    "arguments": None,
                }],
            },
            {
                "role": "tool",
                "call_id": None,
                "tool_call_id": "call_0",
                "content": _text("ok"),
            },
            {"role": "assistant", "content": _text("Done.")},
        ],
        metadata=_metadata(extra_tool_schemas=[{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run bash.",
                "parameters": _params(),
            },
            "name": None,
            "parameters": None,
        }]),
    )

    reasons = check_log_contract(tmp_path)
    assert reasons == []


def test_log_contract_gate_rejects_call_id_pairing_errors(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [{
                    "id": "paired",
                    "type": "function",
                    "function": {"name": "click", "arguments": {"coordinate": [1, 2]}},
                }],
            },
            {"role": "tool", "tool_call_id": "paired", "content": _text("ok")},
            {"role": "tool", "tool_call_id": "paired", "content": _text("again")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("click", {"coordinate": [3, 4]}, call_id="dup"),
                    make_tool_call("wait", {"duration": 1}, call_id="dup"),
                    make_tool_call("click", {"coordinate": [5, 6]}),
                    make_tool_call("click", {"coordinate": [7, 8]}, call_id="missing_result"),
                ],
            },
            {"role": "tool", "tool_call_id": "orphan", "content": _text("orphan")},
            {"role": "tool", "content": _text("no id")},
            {"role": "assistant", "content": _text("Done.")},
        ],
    )

    reasons = check_log_contract(tmp_path)
    assert any("missing non-empty id" in r for r in reasons)


def test_log_contract_gate_rejects_flat_and_duplicate_extra_tool_schemas(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [{"role": "user", "content": _text("task")}],
        metadata=_metadata(extra_tool_schemas=[
            {"type": "function", "name": "dup", "parameters": _params()},
        ]),
    )
    _write_trajectory(
        tmp_path / "sample_01",
        [{"role": "user", "content": _text("task")}],
        metadata=_metadata(extra_tool_schemas=[
            make_tool_schema("dup", parameters=_params()),
            make_tool_schema("dup", parameters=_params()),
        ]),
    )

    reasons = check_log_contract(tmp_path)
    assert any("noncanonical outer keys ['name', 'parameters']" in r for r in reasons)
    assert any("duplicate metadata.extra_tool_schemas name 'dup'" in r for r in reasons)


def test_log_contract_gate_rejects_raw_terminal_answer_alias(tmp_path: Path):
    """Ending the row on the dialect spelling does not exempt it (tool_call side).

    The sibling below covers the ``extra_tool_schemas`` side of the same rule.
    """
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("answer", {"text": "Done."}, call_id="answer_0")
                ],
            },
        ],
        metadata=_metadata(terminated=True),
    )

    reasons = check_log_contract(tmp_path)
    assert any("'answer' is dialect-only" in r and "'response'" in r for r in reasons)


def test_log_contract_gate_rejects_metadata_only_answer_schema(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [{"role": "user", "content": _text("task")}],
        metadata=_metadata(extra_tool_schemas=[make_tool_schema("answer", parameters=_params())]),
    )

    reasons = check_log_contract(tmp_path)
    assert any("canonical 'response'" in r for r in reasons)


def test_log_contract_gate_accepts_content_only_final_paired_results_and_terminal_tools(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {
                        "actions": [{"action": "click", "coordinate": [1, 2]}],
                        },
                        call_id="call_0",
                    )
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": _text("ok")},
            {"role": "assistant", "content": _text("Done.")},
        ],
    )
    _write_trajectory(
        tmp_path / "sample_01",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("response", {"text": "Done."}, call_id="terminal_0")
                ],
            },
            {"role": "tool", "tool_call_id": "terminal_0", "content": _text("Done.")},
        ],
        metadata=_metadata(
            extra_tool_schemas=[
                make_tool_schema(
                    "response",
                    description="Submit a final answer.",
                    parameters=_params(),
                )
            ],
            terminated=True,
        ),
    )
    _write_trajectory(
        tmp_path / "sample_02",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "terminate",
                        {"status": "success"},
                        call_id="terminal_1",
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "terminal_1",
                "content": _text("Task terminated: success"),
            },
        ],
        metadata=_metadata(
            extra_tool_schemas=[
                make_tool_schema(
                    "terminate",
                    description="End the task.",
                    parameters=_params(),
                )
            ],
            terminated=True,
        ),
    )

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_allows_paired_terminal_eof_only_when_episode_done(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    def write_sample(
        name: str,
        messages: list[dict],
        *,
        terminated: bool,
    ) -> Path:
        sample_dir = tmp_path / name
        _write_trajectory(
            sample_dir,
            messages,
            metadata=_metadata(terminated=terminated),
        )
        return sample_dir

    final_tool_call = {
        "role": "assistant",
        "tool_calls": [_computer_call("call_0000")],
    }
    final_tool_result = {
        "role": "tool",
        "tool_call_id": "call_0000",
        "content": _text("terminal feedback"),
    }

    assert check_log_contract(
        write_sample(
            "sample_terminal",
            [final_tool_call, final_tool_result],
            terminated=True,
        )
    ) == []

    nonterminal_reasons = check_log_contract(
        write_sample(
            "sample_nonterminal",
            [final_tool_call, final_tool_result],
            terminated=False,
        )
    )
    assert any("trailing role:tool result" in r for r in nonterminal_reasons)

    assert check_log_contract(
        write_sample(
            "sample_unpaired_terminal",
            [final_tool_call],
            terminated=True,
        )
    ) == []

    mid_turn_reasons = check_log_contract(
        write_sample(
            "sample_mid_turn",
            [
                final_tool_call,
                {"role": "assistant", "content": _text("Done.")},
            ],
            terminated=True,
        )
    )
    assert any("role:tool result" in r for r in mid_turn_reasons)


def test_log_contract_gate_accepts_executed_tool_result_at_truncated_eof(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [_computer_call("call_0")],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": _text("current screenshot after max_steps"),
            },
        ],
        metadata=_metadata(truncated=True),
    )

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_accepts_unknown_tool_error_only_result(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [make_tool_call("foo", call_id="foo_0")],
            },
            {
                "role": "tool",
                "tool_call_id": "foo_0",
                "content": _text("## Error from previous action:\nunknown tool: foo"),
            },
            {"role": "assistant", "content": _text("Done.")},
        ],
    )
    _write_turn_debug_payload(
        sample_dir,
        call_id="foo_0",
        tool_name="foo",
        arguments={},
        text=None,
        error="unknown tool: foo",
    )

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_accepts_extra_tool_invalid_arguments_result(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    sample_dir.mkdir(parents=True)
    (sample_dir / "screen.png").write_bytes(b"png")
    error = "invalid arguments for click: click.arguments.index must be an integer"
    click_schema = make_tool_schema(
        "click",
        parameters={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
    )
    projected = project_tool_result_text("obs", error)
    assert projected is not None
    _write_trajectory(
        sample_dir,
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("click", {"index": "[3]"}, call_id="click_0")
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "click_0",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": projected},
                    {"type": "metadata", "data": {"is_error": True}},
                ],
            },
            {"role": "assistant", "content": _text("Done.")},
        ],
        metadata=_metadata(
            platform="browser",
            extra_tool_schemas=[click_schema],
        ),
        images=["screen.png"],
        json_fields=("messages", "metadata"),
    )
    _write_turn_debug_payload(
        sample_dir,
        call_id="click_0",
        tool_name="click",
        arguments={"index": "[3]"},
        text="obs",
        error=error,
    )

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_accepts_unparseable_output_without_tool_result(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    sample_dir = tmp_path / "sample_00"
    _write_trajectory(
        sample_dir,
        [
            {"role": "user", "content": _text("task")},
            {"role": "assistant", "content": _text("Done.")},
        ],
        metadata=_metadata(terminated=True),
    )
    _write_empty_turn_debug_payload(sample_dir)

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_rejects_schema_less_explicit_finish(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("response", {"text": "Done."}, call_id="terminal_0")
                ],
            },
        ],
    )
    _write_trajectory(
        tmp_path / "sample_01",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "terminate",
                        {"status": "success"},
                        call_id="terminal_1",
                    )
                ],
            },
        ],
    )

    reasons = check_log_contract(tmp_path)
    assert any("tool_call 'response' is standalone but missing" in r for r in reasons)
    assert any("tool_call 'terminate' is standalone but missing" in r for r in reasons)


def test_log_contract_gate_accepts_schema_backed_terminal_eof_without_result(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("response", {"text": "Done."}, call_id="terminal_0")
                ],
            },
        ],
        metadata=_metadata(
            extra_tool_schemas=[
                make_tool_schema(
                    "response",
                    description="Submit a final answer.",
                    parameters=_params(),
                )
            ],
            terminated=True,
        ),
    )

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_accepts_grounding_terminal_predictions(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("point task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("point", {"coordinate": [10, 20]}, call_id="point_0")
                ],
            },
        ],
        metadata=_metadata(
            task_type="grounding.point",
            terminated=True,
        ),
    )
    _write_trajectory(
        tmp_path / "sample_01",
        [
            {"role": "user", "content": _text("bbox task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("bbox", {"coordinate": [1, 2, 3, 4]}, call_id="bbox_0")
                ],
            },
        ],
        metadata=_metadata(
            task_type="grounding.bbox",
            terminated=True,
        ),
    )
    _write_trajectory(
        tmp_path / "sample_02",
        [
            {"role": "user", "content": _text("action task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {
                        "actions": [
                            {"action": "click", "coordinate": [1, 2]},
                            {"action": "type", "text": "x"},
                        ],
                        },
                        call_id="act_0",
                    )
                ],
            },
        ],
        metadata=_metadata(
            task_type="grounding.action",
            terminated=True,
        ),
    )
    _write_trajectory(
        tmp_path / "sample_03",
        [
            {"role": "user", "content": _text("mobile action task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "mobile",
                        {
                        "actions": [
                            {"action": "tap", "coordinate": [1, 2]},
                            {"action": "type", "text": "x"},
                        ],
                        },
                        call_id="mobile_0",
                    )
                ],
            },
        ],
        metadata=_metadata(
            task_type="grounding.action",
            platform="mobile",
            terminated=True,
        ),
    )
    _write_trajectory(
        tmp_path / "sample_04",
        [
            {"role": "user", "content": _text("extra action task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("open_app", {"app_name": "Clock"}, call_id="open_0")
                ],
            },
        ],
        metadata=_metadata(
            task_type="grounding.action",
            platform="mobile",
            terminated=True,
            extra_tool_schemas=[
                make_tool_schema(
                    "open_app",
                    description="Open an app.",
                    parameters=_params(),
                )
            ],
        ),
    )

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_accepts_grounding_terminal_tool_result_feedback(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("point task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("point", {"coordinate": [10, 20]}, call_id="point_0")
                ],
            },
            {"role": "tool", "tool_call_id": "point_0", "content": _text("terminal feedback")},
        ],
        metadata=_metadata(
            task_type="grounding.point",
            terminated=True,
        ),
    )
    _write_trajectory(
        tmp_path / "sample_01",
        [
            {"role": "user", "content": _text("action task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {
                        "actions": [{"action": "click", "coordinate": [1, 2]}],
                        },
                        call_id="act_0",
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "act_0",
                "content": _text("## Error from previous action:\ntarget disappeared"),
            },
        ],
        metadata=_metadata(
            task_type="grounding.action",
            terminated=True,
        ),
    )

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_rejects_nonterminal_grounding_tool_results(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("point task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("point", {"coordinate": [10, 20]}, call_id="point_0")
                ],
            },
            {"role": "tool", "tool_call_id": "point_0", "content": _text("optional feedback")},
            {"role": "assistant", "content": _text("Done.")},
        ],
        metadata=_metadata(
            task_type="grounding.point",
        ),
    )
    _write_trajectory(
        tmp_path / "sample_01",
        [
            {"role": "user", "content": _text("bbox task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("bbox", {"coordinate": [1, 2, 3, 4]}, call_id="bbox_0")
                ],
            },
            {"role": "tool", "tool_call_id": "bbox_0", "content": _text("optional feedback")},
            {"role": "assistant", "content": _text("Done.")},
        ],
        metadata=_metadata(
            task_type="grounding.bbox",
        ),
    )

    reasons = check_log_contract(tmp_path)
    assert sum("orphan role:tool result" in r for r in reasons) == 2


def test_log_contract_gate_does_not_strip_grounding_tool_results_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from lite.infer.debug import log_contract
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("point task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("point", {"coordinate": [10, 20]}, call_id="point_0")
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "point_0",
                "content": _text("terminal feedback"),
            },
        ],
        metadata=_metadata(task_type="grounding.point", terminated=True),
    )
    validator_roles: list[list[str]] = []

    def fake_validate_raw_rollout_rows(rows: list[dict], _label: str) -> None:
        roles = [
            msg["role"]
            for msg in rows[0]["messages"]
            if isinstance(msg, dict) and isinstance(msg.get("role"), str)
        ]
        validator_roles.append(roles)
        if "tool" in roles:
            raise ValueError("structured legacy grounding feedback diagnostic")

    monkeypatch.setattr(
        log_contract,
        "validate_raw_rollout_rows",
        fake_validate_raw_rollout_rows,
    )

    reasons = check_log_contract(tmp_path)
    assert any("structured legacy grounding feedback diagnostic" in r for r in reasons)
    assert validator_roles == [["user", "assistant", "tool"]]


def test_log_contract_gate_rejects_paired_schema_less_standalone_extra(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("finish task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "terminate",
                        {"status": "success"},
                        call_id="terminal_0",
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "terminal_0",
                "content": _text("terminate is not available in this task."),
            },
            {"role": "assistant", "content": _text("Done.")},
        ],
        metadata=_metadata(task_type="use"),
    )
    _write_trajectory(
        tmp_path / "sample_01",
        [
            {"role": "user", "content": _text("bash task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("bash", {"command": "echo hi"}, call_id="bash_0")
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "bash_0",
                "content": _text("bash is not available in this task."),
            },
            {"role": "assistant", "content": _text("Done.")},
        ],
        metadata=_metadata(task_type="use"),
    )

    reasons = check_log_contract(tmp_path)
    assert any("tool_call 'terminate' is standalone but missing" in r for r in reasons)
    assert any("tool_call 'bash' is standalone but missing" in r for r in reasons)


def test_log_contract_gate_rejects_paired_tool_result_at_nonterminal_eof(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {
                        "actions": [{"action": "click", "coordinate": [1, 2]}],
                        },
                        call_id="call_0",
                    )
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": _text("unused final screenshot")},
        ],
        metadata=_metadata(task_type="use"),
    )

    reasons = check_log_contract(tmp_path)
    assert any("trailing role:tool result" in r for r in reasons)


def test_log_contract_gate_accepts_final_use_tool_call_without_result(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                        call_id="call_0",
                    )
                ],
            },
        ],
        metadata=_metadata(task_type="use"),
    )

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_accepts_unpaired_terminal_use_wrapper_call_in_raw_rollout(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {
                        "actions": [{"action": "click", "coordinate": [1, 2]}],
                        },
                        call_id="call_0",
                    )
                ],
            },
        ],
        metadata=_metadata(task_type="use", terminated=True),
    )

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_accepts_paired_final_use_wrapper_call_with_terminal_evidence(
    tmp_path: Path,
):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {
                        "actions": [{"action": "click", "coordinate": [1, 2]}],
                        },
                        call_id="call_0",
                    )
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": _text("final screenshot")},
        ],
        metadata=_metadata(task_type="use", terminated=True),
    )

    assert check_log_contract(tmp_path) == []


def test_log_contract_gate_rejects_non_action_inside_use_wrapper(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {
                        "actions": [{"action": "terminate", "status": "success"}],
                        },
                        call_id="call_0",
                    )
                ],
            },
        ],
        metadata=_metadata(task_type="use", terminated=True),
    )

    reasons = check_log_contract(tmp_path)
    assert any("is not valid for computer" in r for r in reasons)


def test_log_contract_gate_requires_use_tool_result_before_next_assistant(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                        call_id="call_0",
                    )
                ],
            },
            {"role": "assistant", "content": _text("Done.")},
        ],
        metadata=_metadata(task_type="use"),
    )

    reasons = check_log_contract(tmp_path)
    assert any("assistant message arrived before role:tool result" in r for r in reasons)


def test_log_contract_gate_does_not_globalize_grounding_names(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("point", {"coordinate": [1, 2]}, call_id="point_0")
                ],
            },
        ],
        metadata=_metadata(task_type="use"),
    )
    _write_trajectory(
        tmp_path / "sample_01",
        [
            {"role": "user", "content": _text("task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("bbox", {"coordinate": [1, 2, 3, 4]}, call_id="bbox_0")
                ],
            },
        ],
        metadata=_metadata(task_type="use"),
    )

    reasons = check_log_contract(tmp_path)
    assert any(
        "tool_call 'point' is not valid for metadata_kind=cua dims=('desktop', 'use')" in r
        for r in reasons
    )
    assert any(
        "tool_call 'bbox' is not valid for metadata_kind=cua dims=('desktop', 'use')" in r
        for r in reasons
    )


def test_log_contract_gate_rejects_invalid_grounding_terminal_prediction(tmp_path: Path):
    from lite.infer.debug.log_contract import check_log_contract

    _write_trajectory(
        tmp_path / "sample_00",
        [
            {"role": "user", "content": _text("typo action task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("clik", {"coordinate": [1, 2]}, call_id="bad_0")
                ],
            },
        ],
        metadata=_metadata(task_type="grounding.action"),
    )
    _write_trajectory(
        tmp_path / "sample_01",
        [
            {"role": "user", "content": _text("desktop action task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("tap", {"coordinate": [1, 2]}, call_id="bad_1")
                ],
            },
        ],
        metadata=_metadata(task_type="grounding.action", platform="desktop"),
    )
    _write_trajectory(
        tmp_path / "sample_02",
        [
            {"role": "user", "content": _text("mobile action task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("click", {"coordinate": [1, 2]}, call_id="bad_2")
                ],
            },
        ],
        metadata=_metadata(task_type="grounding.action", platform="mobile"),
    )
    _write_trajectory(
        tmp_path / "sample_03",
        [
            {"role": "user", "content": _text("undeclared extra task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("open_app", {"app_name": "Clock"}, call_id="bad_3")
                ],
            },
        ],
        metadata=_metadata(task_type="grounding.action", platform="mobile"),
    )
    _write_trajectory(
        tmp_path / "sample_04",
        [
            {"role": "user", "content": _text("bbox task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call("bbox", {"coordinate": [1, 2, 3, 4]}, call_id="bad_4")
                ],
            },
        ],
        metadata=_metadata(task_type="grounding.action"),
    )
    _write_trajectory(
        tmp_path / "sample_05",
        [
            {"role": "user", "content": _text("desktop wrapper task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "mobile",
                        {"actions": [{"action": "tap", "coordinate": [1, 2]}]},
                        call_id="bad_5",
                    )
                ],
            },
        ],
        metadata=_metadata(task_type="grounding.action", platform="desktop"),
    )
    _write_trajectory(
        tmp_path / "sample_06",
        [
            {"role": "user", "content": _text("mobile wrapper task")},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                        call_id="bad_6",
                    )
                ],
            },
        ],
        metadata=_metadata(task_type="grounding.action", platform="mobile"),
    )

    reasons = check_log_contract(tmp_path)
    assert any("tool_call 'clik' is standalone but missing" in r for r in reasons)
    assert any("tool_call 'tap' is standalone but missing" in r for r in reasons)
    assert any("tool_call 'click' is standalone but missing" in r for r in reasons)
    assert any("tool_call 'open_app' is standalone but missing" in r for r in reasons)
    bbox_reason = (
        "tool_call 'bbox' is not valid for metadata_kind=cua "
        "dims=('desktop', 'grounding.action')"
    )
    mobile_reason = (
        "tool_call 'mobile' is not valid for metadata_kind=cua "
        "dims=('desktop', 'grounding.action')"
    )
    computer_reason = (
        "tool_call 'computer' is not valid for metadata_kind=cua "
        "dims=('mobile', 'grounding.action')"
    )
    assert any(bbox_reason in r for r in reasons)
    assert any(mobile_reason in r for r in reasons)
    assert any(computer_reason in r for r in reasons)


# =============================================================================
# I17 — ``arguments: {}`` must not be persisted as ``null`` in the parquet
# =============================================================================
#
# ``lite/utils/parquet.py::_normalize_for_arrow`` used to map EVERY empty dict to
# ``None`` to dodge PyArrow's ``struct<>`` limitation. A zero-argument tool call
# (``back``, ``new_tab``) has ``arguments == {}``, so the golden record stored
# ``null`` where ``03_actions.json`` correctly held ``{}``.
#
# MEASURED: 11 / 11 zero-arg calls affected (100%), of 3 997 calls across 4 of
# 18 rows. Each cascades into a SECOND, spurious violation — ``check_log_contract``
# treats the malformed call as not-a-call, so its paired result is reported as an
# orphan ``role:"tool"`` — and 11 defects surfaced as 22.
#
# RULE 0: not a regression. ``_normalize_for_arrow`` is byte-identical at the
# merge-base; ``check_log_contract`` simply had never looked before.



def _browser_nav_metadata() -> dict:
    """``back`` / ``goto`` declared as standalone extras — the checker requires a
    standalone call to be backed by ``metadata.extra_tool_schemas``, and this is
    exactly how the browsergym browser rows advertise them."""
    return _metadata(
        platform="browser",
        extra_tool_schemas=[
            make_tool_schema(
                "back",
                description="back",
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            make_tool_schema(
                "goto",
                description="goto",
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            ),
        ],
        terminated=True,
    )


def _zero_arg_trajectory_messages() -> list[dict]:
    """A zero-arg tool call and its paired result, plus an arg-bearing sibling.

    The sibling matters: it is what gives the ``arguments`` path a child field,
    i.e. the common case where PyArrow's ``struct<>`` limitation does NOT apply
    and the old blanket rewrite was pure loss.
    """
    return [
        {"role": "user", "content": _text("go back")},
        {"role": "assistant", "content": [], "tool_calls": [
            make_tool_call(
                "goto",
                {"url": "https://example.com"},
                call_id="call_0000",
            ),
        ]},
        {"role": "tool", "tool_call_id": "call_0000", "content": _text("ok")},
        {"role": "assistant", "content": [], "tool_calls": [
            make_tool_call("back", {}, call_id="call_0001"),
        ]},
        {"role": "tool", "tool_call_id": "call_0001", "content": _text("ok")},
        # A trajectory may not END on a tool result — close it with a
        # content-only final so the only thing under test is ``arguments``.
        {"role": "assistant", "content": _text("Done.")},
    ]
