"""Pure launcher command and result-evidence tests; no Codex inference is run."""

from __future__ import annotations

import json
import tomllib
import uuid
from types import SimpleNamespace

import pytest

from examples.not_a_robot.codex_smoke import (
    FIRST_TEN,
    PROMPT,
    _parse_args,
    build_command,
    collect_result,
)


@pytest.fixture
def run_files(tmp_path):
    task_root = tmp_path / "run" / "neal_01"
    workspace = task_root / "solver-workspace"
    workspace.mkdir(parents=True)
    args = SimpleNamespace(
        codex="/bin/codex",
        python="/runtime/python",
        browser_executable="/browser/chrome",
        model="gpt-6-astra",
        reasoning_effort="xhigh",
        max_seconds=600,
        max_steps=150,
    )
    thread_id = str(uuid.uuid4())
    (task_root / "codex.stdout.jsonl").write_text(
        json.dumps({"type": "thread.started", "thread_id": thread_id}) + "\n"
    )
    home = tmp_path / "client-home"
    sessions = home / "sessions" / "2026" / "09" / "09"
    sessions.mkdir(parents=True)
    context = {
        "model": args.model,
        "effort": args.reasoning_effort,
        "cwd": str(workspace),
        "sandbox_policy": {"type": "read-only"},
    }
    session = sessions / f"rollout-2026-09-09-{thread_id}.jsonl"
    session.write_text(
        json.dumps({"timestamp": "2026-09-09", "type": "turn_context", "payload": context}) + "\n"
    )
    attempt = task_root / "trajectory" / "attempt"
    attempt.mkdir(parents=True)
    manifest = attempt / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"outcome": "success", "recording_complete": True, "data": {"cleanup_complete": True}}
        )
    )
    return args, task_root, home, session, context, manifest


def test_command_has_exact_model_and_gui_only_configuration(run_files):
    args, task_root, *_ = run_files
    command = build_command(args, "neal_01", task_root, "/bin/uv")
    assert command[:5] == [
        "/bin/codex",
        "exec",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--json",
    ]
    assert command[command.index("-m") + 1] == "gpt-6-astra"
    assert command[command.index("-s") + 1] == "read-only"
    assert command[command.index("-C") + 1] == str(task_root / "solver-workspace")
    assert command[-1] == PROMPT
    config = {}
    for index, token in enumerate(command):
        if token == "-c":
            config.update(tomllib.loads(command[index + 1]))
    assert config["model_reasoning_effort"] == "xhigh"
    # Dotted overrides are independent command arguments, so parse together too.
    merged = tomllib.loads(
        "\n".join(command[index + 1] for index, token in enumerate(command) if token == "-c")
    )
    assert merged["features"] == {
        "shell_tool": False,
        "apps": False,
        "multi_agent": False,
        "memories": False,
    }
    assert merged["approval_policy"] == "never"
    assert merged["web_search"] == "disabled"
    assert merged["project_doc_max_bytes"] == 0
    assert list(merged["mcp_servers"]) == ["local_game"]
    server = merged["mcp_servers"]["local_game"]
    assert server["enabled_tools"] == ["get_observation", "computer", "finish"]
    assert server["tools"] == {
        name: {"approval_mode": "approve"} for name in ("get_observation", "computer", "finish")
    }
    assert "default_tools_approval_mode" not in server
    assert server["command"] == "/bin/uv"
    assert server["args"][:6] == [
        "run",
        "--no-project",
        "--python",
        "/runtime/python",
        "python",
        "-m",
    ]
    assert server["args"][6] == "examples.not_a_robot.codex_bridge"
    assert server["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert server["args"][server["args"].index("--task") + 1] == "neal_01"
    assert server["args"][server["args"].index("--model") + 1] == "gpt-6-astra"


def test_success_requires_current_thread_context_and_environment_manifest(run_files):
    args, task_root, home, _, _, _ = run_files
    result = collect_result(args, task_root, home, 0, False)
    assert result["evaluated"] and result["success"]
    assert result["environment_attempt_count"] == result["environment_manifest_count"] == 1
    evidence = json.loads((task_root / "client_model_evidence.json").read_text())
    assert evidence["client_configuration_matches"]
    assert not evidence["provider_model_cryptographically_verified"]


@pytest.mark.parametrize(
    "outcome", ["failure", "agent_stopped", "budget_exhausted", "controller_timeout"]
)
def test_model_failure_can_be_a_complete_evaluated_attempt(run_files, outcome):
    args, task_root, home, _, _, manifest = run_files
    manifest.write_text(
        json.dumps(
            {"outcome": outcome, "recording_complete": True, "data": {"cleanup_complete": True}}
        )
    )
    (task_root / "last_message.txt").write_text("I claim success regardless of the game.")
    result = collect_result(args, task_root, home, 0, False)
    assert result["evaluated"] and not result["success"] and result["outcome"] == outcome


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "different-model"),
        ("effort", "high"),
        ("cwd", "/wrong"),
        ("sandbox_policy", {"type": "danger-full-access"}),
    ],
)
def test_context_mismatch_is_not_verified(run_files, field, value):
    args, task_root, home, session, context, _ = run_files
    context[field] = value
    session.write_text(
        json.dumps({"timestamp": "2026-09-09", "type": "turn_context", "payload": context}) + "\n"
    )
    result = collect_result(args, task_root, home, 0, False)
    assert not result["client_configuration_matches"] and not result["evaluated"]


def test_non_context_payloads_are_not_parsed_or_exported(run_files):
    args, task_root, home, session, context, _ = run_files
    # Deliberately invalid non-context payload: reading it as JSON would fail.
    skipped = (
        '{"timestamp":"2026-09-09","type":"response_item",'
        '"payload":DO NOT PARSE PRIVATE MATERIAL}\n'
    )
    context["other_private_field"] = "DO NOT EXPORT"
    session.write_text(
        skipped
        + json.dumps({"timestamp": "2026-09-09", "type": "turn_context", "payload": context})
        + "\n"
    )
    result = collect_result(args, task_root, home, 0, False)
    assert result["evaluated"]
    exported = (task_root / "client_model_evidence.json").read_text()
    assert "DO NOT" not in exported and "other_private_field" not in exported


@pytest.mark.parametrize(
    "failure",
    [
        "missing_session",
        "second_attempt",
        "missing_manifest",
        "corrupt_manifest",
        "incomplete_recording",
        "incomplete_cleanup",
        "infra_error",
        "truncated_stdout",
        "client_error",
        "launcher_timeout",
    ],
)
def test_infrastructure_or_missing_evidence_never_counts_as_pass(run_files, failure):
    args, task_root, home, session, _, manifest = run_files
    returncode, timed_out = 0, False
    if failure == "missing_session":
        session.unlink()
    elif failure == "second_attempt":
        (manifest.parent.parent / "other-attempt").mkdir()
    elif failure == "missing_manifest":
        manifest.unlink()
    elif failure == "corrupt_manifest":
        manifest.write_text("{")
    elif failure in ("incomplete_recording", "incomplete_cleanup", "infra_error"):
        data = json.loads(manifest.read_text())
        if failure == "incomplete_recording":
            data["recording_complete"] = False
        elif failure == "incomplete_cleanup":
            data["data"]["cleanup_complete"] = False
        else:
            data["outcome"] = "infra_error"
        manifest.write_text(json.dumps(data))
    elif failure == "truncated_stdout":
        with (task_root / "codex.stdout.jsonl").open("a") as stream:
            stream.write("{")
    elif failure == "client_error":
        returncode = 1
    elif failure == "launcher_timeout":
        timed_out = True
    result = collect_result(args, task_root, home, returncode, timed_out)
    assert not result["evaluated"] and not result["success"]


def test_cli_defaults_and_existing_artifacts_are_not_overwritten(tmp_path):
    root = tmp_path / "new-run"
    args = _parse_args(["--artifact-root", str(root), "--browser-executable", "/chrome"])
    assert args.tasks == list(FIRST_TEN)
    assert args.model == "gpt-6-astra" and args.reasoning_effort == "xhigh"
    assert args.max_seconds == 600 and args.max_steps == 150
    root.mkdir()
    with pytest.raises(SystemExit):
        _parse_args(["--artifact-root", str(root), "--browser-executable", "/chrome"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-seconds", "nan"],
        ["--max-seconds", "0"],
        ["--max-steps", "0"],
        ["--tasks", "neal_01", "neal_01"],
        ["--tasks", "neal_11"],
    ],
)
def test_cli_rejects_invalid_or_duplicate_work(tmp_path, extra):
    with pytest.raises(SystemExit):
        _parse_args(
            ["--artifact-root", str(tmp_path / "new"), "--browser-executable", "/chrome", *extra]
        )
