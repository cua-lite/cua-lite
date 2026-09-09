"""Run bounded, screenshot-only Codex exec attempts against the first ten tasks.

Use ``python -m examples.not_a_robot.codex_smoke --artifact-root /new/run/path
--browser-executable /path/to/chrome``. The artifact root must not exist.
The logged CLI-selected model is checked against this run's turn contexts;
this is client evidence, not cryptographic proof of the provider's model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

FIRST_TEN = tuple(f"neal_{level:02d}" for level in range(1, 11))
PROMPT = """Evaluate the synthetic visual game using only the local_game MCP tools:
get_observation, computer, and finish. Call get_observation first. Read the visible
instructions and reason from screenshots only. Computer coordinates are normalized
0..1000 across the complete screenshot. Use deliberate, bounded mouse and keyboard
actions, then observe their results. Do not blindly click every cell or brute-force
answers. Do not use a shell, files, source code, DOM, selectors, browser evaluation,
external sites, other tools, or prior answers. Give only short public action
summaries, not private reasoning. When the game completes or you are stuck, call
finish with an honest status and concise summary. A claim alone is not completion.
"""


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=FIRST_TEN, default=list(FIRST_TEN))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--browser-executable", required=True)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="xhigh")
    args = parser.parse_args(argv)
    if args.max_seconds <= 0 or not math.isfinite(args.max_seconds) or args.max_steps <= 0:
        parser.error("Time and step budgets must be finite and positive")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Each task may appear only once per run")
    if args.artifact_root.exists():
        parser.error(
            "--artifact-root must name a new directory; existing runs are never overwritten"
        )
    return args


def build_command(args, task: str, task_root: Path, uv: str) -> list[str]:
    """Build an explicit isolated client plus its single GUI-only MCP server."""
    repo = Path(__file__).resolve().parents[2]
    bridge_args = [
        "run",
        "--no-project",
        "--python",
        args.python,
        "python",
        "-m",
        "examples.not_a_robot.codex_bridge",
        "--task",
        task,
        "--artifact-root",
        str(task_root / "trajectory"),
        "--browser-executable",
        args.browser_executable,
        "--max-seconds",
        str(args.max_seconds),
        "--max-steps",
        str(args.max_steps),
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
    ]
    server = (
        "{command="
        + json.dumps(uv)
        + ",args="
        + json.dumps(bridge_args)
        + ",env={PYTHONPATH="
        + json.dumps(str(repo))
        + ',PYTHONDONTWRITEBYTECODE="1"},startup_timeout_sec=30,tool_timeout_sec=60,'
        + 'enabled_tools=["get_observation","computer","finish"],'
        + 'tools={get_observation={approval_mode="approve"},'
        + 'computer={approval_mode="approve"},finish={approval_mode="approve"}}}'
    )
    config = {
        "model_reasoning_effort": args.reasoning_effort,
        "features.shell_tool": False,
        "web_search": "disabled",
        "features.apps": False,
        "features.multi_agent": False,
        "features.memories": False,
        "project_doc_max_bytes": 0,
        "approval_policy": "never",
    }
    command = [
        args.codex,
        "exec",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--json",
        "--color",
        "never",
        "-m",
        args.model,
        "-s",
        "read-only",
        "-C",
        str(task_root / "solver-workspace"),
        "--output-last-message",
        str(task_root / "last_message.txt"),
    ]
    for key, value in config.items():
        command.extend(["-c", f"{key}={json.dumps(value)}"])
    command.extend(["-c", "mcp_servers={local_game=" + server + "}", PROMPT])
    return command


def collect_result(
    args, task_root: Path, codex_home: Path, returncode: int, timed_out: bool
) -> dict:
    """Grade from the environment manifest; inspect only this client's model context."""
    thread_ids = []
    malformed_lines = 0
    with (task_root / "codex.stdout.jsonl").open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                malformed_lines += 1
                continue
            if not isinstance(event, dict):
                malformed_lines += 1
                continue
            if event.get("type") == "thread.started":
                try:
                    thread_ids.append(str(uuid.UUID(event["thread_id"])))
                except (KeyError, ValueError, TypeError):
                    malformed_lines += 1
    contexts = []
    session_files = []
    context_errors = []
    if len(thread_ids) == 1:
        session_files = list((codex_home / "sessions").glob(f"*/*/*/*{thread_ids[0]}.jsonl"))
    if len(session_files) == 1:
        with session_files[0].open() as stream:
            for line in stream:
                # Inspect the envelope first; never parse/export reasoning or other payloads.
                envelope = line.partition('"payload"')[0]
                if not re.search(r'"type"\s*:\s*"turn_context"', envelope):
                    continue
                try:
                    payload = json.loads(line)["payload"]
                except ValueError as error:
                    context_errors.append(str(error))
                    continue
                contexts.append(
                    {key: payload.get(key) for key in ("model", "effort", "cwd", "sandbox_policy")}
                )
    matched = (
        bool(contexts)
        and not context_errors
        and all(
            context["model"] == args.model
            and context["effort"] == args.reasoning_effort
            and context["cwd"] == str(task_root / "solver-workspace")
            and isinstance(context["sandbox_policy"], dict)
            and context["sandbox_policy"].get("type") == "read-only"
            for context in contexts
        )
    )
    evidence = {
        "requested_model": args.model,
        "requested_reasoning_effort": args.reasoning_effort,
        "thread_ids": thread_ids,
        "session_files": [str(path) for path in session_files],
        "turn_contexts": contexts,
        "context_errors": context_errors,
        "client_configuration_matches": matched,
        "provider_model_cryptographically_verified": False,
    }
    with (task_root / "client_model_evidence.json").open("x") as output:
        json.dump(evidence, output, indent=2)
        output.write("\n")
    manifests = list((task_root / "trajectory").glob("*/manifest.json"))
    attempts = [path for path in (task_root / "trajectory").glob("*") if path.is_dir()]
    manifest = None
    manifest_error = None
    if len(manifests) == 1:
        try:
            manifest = json.loads(manifests[0].read_text())
        except ValueError as error:
            manifest_error = str(error)
    outcome = manifest["outcome"] if manifest else "missing_or_ambiguous_trajectory"
    recording_complete = bool(manifest and manifest.get("recording_complete"))
    cleanup_complete = bool(manifest and manifest.get("data", {}).get("cleanup_complete"))
    evaluated = (
        returncode == 0
        and not timed_out
        and malformed_lines == 0
        and matched
        and len(attempts) == 1
        and recording_complete
        and cleanup_complete
        and outcome
        in (
            "success",
            "failure",
            "agent_stopped",
            "budget_exhausted",
            "timeout",
            "controller_timeout",
            "controller_disconnected",
            "controller_shutdown",
        )
    )
    return {
        "task": task_root.name,
        "cli_returncode": returncode,
        "launcher_timed_out": timed_out,
        "malformed_client_jsonl_lines": malformed_lines,
        "client_configuration_matches": matched,
        "environment_attempt_count": len(attempts),
        "environment_manifest_count": len(manifests),
        "environment_manifest": str(manifests[0]) if len(manifests) == 1 else None,
        "environment_manifest_error": manifest_error,
        "outcome": outcome,
        "recording_complete": recording_complete,
        "cleanup_complete": cleanup_complete,
        "evaluated": evaluated,
        "success": evaluated and outcome == "success",
    }


def run(args) -> int:
    """Run tasks serially in fresh solver workspaces without changing global config."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to launch the configured Python runtime")
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).resolve()
    version = subprocess.run(
        [args.codex, "--version"], capture_output=True, text=True, check=True, timeout=15
    ).stdout.strip()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    run_root = root / uuid.uuid4().hex
    run_root.mkdir(mode=0o700)
    results = []
    for task in args.tasks:
        task_root = run_root / task
        workspace = task_root / "solver-workspace"
        workspace.mkdir(parents=True, mode=0o700)
        command = build_command(args, task, task_root, uv)
        with (task_root / "command.json").open("x") as output:
            json.dump(
                {"argv": command, "cwd": str(workspace), "codex_version": version}, output, indent=2
            )
            output.write("\n")
        with (workspace / "prompt.txt").open("x") as output:
            output.write(PROMPT)
        print(f"Starting {task}: {task_root}", flush=True)
        timed_out = False
        interrupted = False
        with (
            (task_root / "codex.stdout.jsonl").open("x") as stdout,
            (task_root / "codex.stderr.log").open("x") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                process.wait(timeout=args.max_seconds + 45)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                interrupted = isinstance(error, KeyboardInterrupt)
                timed_out = not interrupted
                # The process group was created here and belongs only to this attempt.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    # The leader can exit before its MCP child finishes cleanup.
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        os.killpg(process.pid, 0)
                        time.sleep(0.1)
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        # Normal client exit can precede the bridge's final atomic manifest write.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not list(
            (task_root / "trajectory").glob("*/manifest.json")
        ):
            time.sleep(0.1)
        result = collect_result(args, task_root, codex_home, process.returncode, timed_out)
        result["launcher_interrupted"] = interrupted
        if interrupted:
            result["evaluated"] = result["success"] = False
        results.append(result)
        with (task_root / "result.json").open("x") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        print(
            f"Finished {task}: outcome={result['outcome']} evaluated={result['evaluated']}",
            flush=True,
        )
        if interrupted:
            break
    with (run_root / "run_manifest.json").open("x") as output:
        json.dump(
            {"codex_version": version, "requested_tasks": args.tasks, "results": results},
            output,
            indent=2,
        )
        output.write("\n")
    print(f"Run manifest: {run_root / 'run_manifest.json'}", flush=True)
    return (
        0
        if len(results) == len(args.tasks) and all(result["evaluated"] for result in results)
        else 1
    )


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGINT))
    sys.exit(run(_parse_args()))
