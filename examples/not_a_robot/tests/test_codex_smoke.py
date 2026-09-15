"""Pure launcher command and result-evidence tests; no Codex inference is run."""

from __future__ import annotations

import json
import subprocess
import tomllib
import uuid
from types import SimpleNamespace

import pytest

from examples.not_a_robot import codex_smoke
from examples.not_a_robot.codex_smoke import (
    PROMPT,
    REFERENCE_TASKS,
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
        campaign=False,
        seed=0,
        reference_instance="default",
        record_video=False,
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
        "shell_snapshot": False,
        "view_image": False,
        "workspace_dependencies": False,
        "apps": False,
        "multi_agent": False,
        "memories": False,
        "plugins": False,
        "hooks": False,
        "browser_use": False,
        "computer_use": False,
        "image_generation": False,
        "skill_search": False,
        "skip_host_skill_discovery": True,
    }
    assert merged["approval_policy"] == "never"
    assert merged["web_search"] == "disabled"
    assert merged["project_doc_max_bytes"] == 0
    assert list(merged["mcp_servers"]) == ["local_game"]
    server = merged["mcp_servers"]["local_game"]
    assert server["required"] is True
    assert server["tool_timeout_sec"] == args.max_seconds + 5
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
    assert server["args"][server["args"].index("--seed") + 1] == "0"
    assert server["args"][server["args"].index("--reference-instance") + 1] == "default"


def test_prompt_describes_optional_sequence_without_expanding_solver_access():
    assert 'sequence={"duration_seconds": 5, "interval_seconds": 0.1}' in PROMPT
    assert "first get_observation may also request a sequence" in PROMPT
    assert "Each extra screenshot consumes a step" in PROMPT
    assert "not video or audio" in PROMPT
    assert "cannot accompany a region" in PROMPT
    assert "Do not use a shell, files, source code, DOM" in PROMPT


def test_success_requires_current_thread_context_and_environment_manifest(run_files):
    args, task_root, home, _, _, _ = run_files
    result = collect_result(args, task_root, home, 0, False)
    assert result["evaluated"] and result["success"]
    assert result["environment_attempt_count"] == result["environment_manifest_count"] == 1
    evidence = json.loads((task_root / "client_model_evidence.json").read_text())
    assert evidence["client_configuration_matches"]
    assert not evidence["provider_model_cryptographically_verified"]


@pytest.mark.parametrize("status", ["saved", "missing", "failed", "not_requested"])
def test_video_evidence_is_reported_separately_from_game_outcome(run_files, status):
    args, task_root, home, _, _, manifest = run_files
    args.record_video = status != "not_requested"
    data = json.loads(manifest.read_text())
    data["video"] = {"status": status, "playback_validation": "not_performed"}
    manifest.write_text(json.dumps(data))
    result = collect_result(args, task_root, home, 0, False)
    assert result["outcome"] == "success" and result["evaluated"]
    assert result["video_requested"] is args.record_video
    assert result["video_saved"] is (status == "saved")
    assert result["video"]["status"] == status
    assert result["video"]["playback_validation"] == "not_performed"


@pytest.mark.parametrize("max_seconds", [30, 600, 900.5])
def test_tool_timeout_covers_episode_budget_without_an_unbounded_wait(run_files, max_seconds):
    args, task_root, *_ = run_files
    args.max_seconds = max_seconds
    command = build_command(args, "neal_01", task_root, "/bin/uv")
    config = tomllib.loads(
        "\n".join(command[i + 1] for i, token in enumerate(command) if token == "-c")
    )
    assert config["mcp_servers"]["local_game"]["tool_timeout_sec"] == max_seconds + 5
    assert "within 45 seconds" in PROMPT


def test_utf8_client_and_session_text_does_not_use_windows_ansi_encoding(run_files):
    args, task_root, home, session, _, _ = run_files
    with (task_root / "codex.stdout.jsonl").open("ab") as stream:
        stream.write('{"type":"fixture","text":"中文 ’ ”"}\n'.encode())
    with session.open("ab") as stream:
        stream.write('{"type":"response_item","payload":"中文 ’ ”"}\n'.encode())
    assert collect_result(args, task_root, home, 0, False)["evaluated"]


def test_campaign_launcher_targets_one_full_game(run_files, tmp_path):
    args, task_root, *_ = run_files
    parsed = _parse_args(
        [
            "--campaign",
            "--artifact-root",
            str(tmp_path / "new-run"),
            "--browser-executable",
            "/chrome",
        ]
    )
    assert parsed.campaign and parsed.tasks == ["full_game"]
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--campaign",
                "--tasks",
                "neal_01",
                "--artifact-root",
                str(tmp_path / "other"),
                "--browser-executable",
                "/chrome",
            ]
        )
    args.campaign = True
    command = build_command(args, "full_game", task_root, "/bin/uv")
    config = tomllib.loads(
        "\n".join(command[i + 1] for i, token in enumerate(command) if token == "-c")
    )
    bridge_args = config["mcp_servers"]["local_game"]["args"]
    assert "--campaign" in bridge_args and "--task" not in bridge_args


def test_missing_task_is_not_evaluated_as_a_model_failure_or_success(run_files):
    args, task_root, home, _, _, manifest = run_files
    manifest.write_text(
        json.dumps(
            {
                "outcome": "unsupported_task",
                "recording_complete": True,
                "data": {"cleanup_complete": True},
            }
        )
    )
    result = collect_result(args, task_root, home, 0, False)
    assert result["blocked_by_missing_task"]
    assert result["recording_complete"] and result["cleanup_complete"]
    assert not result["evaluated"] and not result["success"]


def test_infra_error_with_clean_exit_and_complete_video_is_not_evaluated(run_files):
    """Collector-only fixture, not an actual model run or a decoded video."""
    args, task_root, home, _, _, manifest = run_files
    args.record_video = True
    manifest.write_text(
        json.dumps(
            {
                "outcome": "infra_error",
                "recording_complete": True,
                "recording_scope": "events_and_png_images",
                "video": {"status": "saved", "playback_validation": "not_performed"},
                "data": {"cleanup_complete": True},
            }
        )
    )
    result = collect_result(args, task_root, home, 0, False)
    assert result["cli_returncode"] == 0 and result["client_configuration_matches"]
    assert result["recording_complete"] and result["cleanup_complete"] and result["video_saved"]
    assert result["outcome"] == "infra_error"
    assert not result["evaluated"] and not result["success"]


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


@pytest.mark.parametrize("outcome", ["controller_disconnected", "controller_shutdown"])
def test_transport_end_without_terminal_game_result_is_not_a_model_failure(run_files, outcome):
    args, task_root, home, _, _, manifest = run_files
    manifest.write_text(
        json.dumps(
            {"outcome": outcome, "recording_complete": True, "data": {"cleanup_complete": True}}
        )
    )
    result = collect_result(args, task_root, home, 0, False)
    assert result["outcome"] == outcome
    assert result["recording_complete"] and result["cleanup_complete"]
    assert not result["evaluated"] and not result["success"]


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
    assert args.tasks == list(REFERENCE_TASKS)
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
        ["--tasks", "neal_49"],
    ],
)
def test_cli_rejects_invalid_or_duplicate_work(tmp_path, extra):
    with pytest.raises(SystemExit):
        _parse_args(
            ["--artifact-root", str(tmp_path / "new"), "--browser-executable", "/chrome", *extra]
        )


@pytest.mark.parametrize("platform", ["nt", "posix"])
def test_timeout_uses_only_owned_client_resources(tmp_path, monkeypatch, platform):
    """Platform-boundary mock, not a real Codex/model or process-tree benchmark."""
    calls = []
    parent_env = {
        "CODEX_HOME": str(tmp_path / "test-home"),
        "CODEX_APP_TOOLS_PIPE_PATH": "fixture-app-pipe",
        "CODEX_SESSION_ID": "fixture-session",
        "CODEX_THREAD_ID": "fixture-thread",
        "CODEX_INTERNAL_ORIGINATOR_OVERRIDE": "fixture-originator",
        "CODEX_SANDBOX_NETWORK_DISABLED": "1",
        "CODEX_SANDBOX": "fixture-restriction",
        "CODEX_MCP_NODE_PATH": "fixture-runtime-path",
        "PATH": "fixture-path",
    }
    expected_env = {
        key: value
        for key, value in parent_env.items()
        if key
        not in {
            "CODEX_APP_TOOLS_PIPE_PATH",
            "CODEX_SESSION_ID",
            "CODEX_THREAD_ID",
            "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
        }
    }
    args = SimpleNamespace(
        artifact_root=tmp_path / "run",
        tasks=["neal_15"],
        codex="test-client",
        max_seconds=1,
    )
    monkeypatch.setattr(codex_smoke.shutil, "which", lambda _: "test-uv")
    monkeypatch.setattr(
        codex_smoke,
        "os",
        SimpleNamespace(
            name=platform,
            environ=parent_env,
            killpg=lambda pid, sig: (
                calls.append(("group", pid, sig))
                if sig
                else (_ for _ in ()).throw(ProcessLookupError())
            ),
        ),
    )
    monkeypatch.setattr(
        codex_smoke,
        "signal",
        SimpleNamespace(
            SIGTERM=15,
            SIGKILL=9,
            CTRL_BREAK_EVENT=1,
        ),
    )
    monkeypatch.setattr(codex_smoke.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)

    def version_probe(*args, **kwargs):
        assert kwargs["env"] == expected_env
        assert kwargs["env"] is not parent_env
        return SimpleNamespace(stdout="fixture-client")

    monkeypatch.setattr(codex_smoke.subprocess, "run", version_probe)
    monkeypatch.setattr(codex_smoke, "build_command", lambda *a: ["fixture-client"])

    class Owned:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive):
            assert recursive
            return [Owned(202)]

        def terminate(self):
            calls.append(("terminate", self.pid))

        def kill(self):
            calls.append(("kill", self.pid))

    class Client:
        pid = 101
        returncode = -1
        waits = 0

        def __init__(self, command, **kwargs):
            assert command == ["fixture-client"]
            assert kwargs["env"] == expected_env
            assert kwargs["env"] is not parent_env
            assert kwargs["start_new_session"] == (platform != "nt")
            assert kwargs["creationflags"] == (512 if platform == "nt" else 0)
            manifest = kwargs["cwd"].parent / "trajectory" / "attempt" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("fixture-client", timeout)

        def send_signal(self, value):
            calls.append(("break", self.pid, value))
            raise OSError("Fixture has no attached console")

    monkeypatch.setattr(codex_smoke.psutil, "Process", Owned)
    monkeypatch.setattr(codex_smoke.psutil, "wait_procs", lambda owned, timeout: (owned, []))
    monkeypatch.setattr(codex_smoke.subprocess, "Popen", Client)
    monkeypatch.setattr(
        codex_smoke,
        "collect_result",
        lambda *a: {
            "outcome": "timeout",
            "evaluated": False,
            "success": False,
        },
    )
    assert codex_smoke.run(args) == 1
    assert parent_env["CODEX_APP_TOOLS_PIPE_PATH"] == "fixture-app-pipe"
    assert parent_env["CODEX_SESSION_ID"] == "fixture-session"
    assert parent_env["CODEX_THREAD_ID"] == "fixture-thread"
    assert parent_env["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"] == "fixture-originator"
    if platform == "nt":
        assert calls == [("break", 101, 1), ("terminate", 101), ("terminate", 202)]
    else:
        assert calls == [("group", 101, 15)]
