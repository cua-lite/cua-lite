"""Protocol, provenance, GUI-only surface and lifecycle tests without a browser."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from types import SimpleNamespace

import pytest

from examples.not_a_robot.codex_bridge import CodexBridge, _parse_args, serve
from examples.not_a_robot.recorder import EventRecorder
from lite import gym
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


class FakeEnv:
    """Exercise the bridge with the real recorder and canonical result shapes."""

    def __init__(self, root):
        self.unwrapped = self
        self.attempt_dir = root / "attempt"
        self.recorder = None
        self.last_observation = None
        self.display_resolution = (1280, 800)
        self.outcome = "not_started"
        self.calls = []
        self.reset_count = self.close_count = 0

    async def reset(self):
        self.reset_count += 1
        self.outcome = "in_progress"
        self.recorder = EventRecorder(self.attempt_dir, {"fixture": True})
        self.last_observation = self.recorder.image(PNG, variant="model_visible")
        return LiteEnvObservation(image=PNG, text="Complete the visible task.")

    async def step(self, calls):
        self.calls.extend(calls)
        function = calls[0]["function"]
        if function["name"] == "terminate":
            self.outcome = "agent_stopped"
        elif function["arguments"]["actions"] == [{"action": "click", "coordinate": [50, 50]}]:
            self.outcome = "success"
        self.last_observation = self.recorder.image(PNG, variant="model_visible")
        terminal = self.outcome != "in_progress"
        return LiteEnvStepResult(
            results=[LiteToolResult(tool_call_id=calls[0]["id"], images=[PNG])],
            reward=float(self.outcome == "success") if terminal else None,
            terminated=terminal,
            info={"evaluation_only_secret": "must not reach model"},
        )

    async def close(self):
        self.close_count += 1
        if self.recorder:
            self.recorder.finalize(self.outcome, cleanup_complete=True)
            self.recorder = None


@pytest.fixture
def setup_bridge(tmp_path, monkeypatch):
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_PORT", raising=False)
    args = SimpleNamespace(
        task="click",
        seed=12,
        artifact_root=str(tmp_path),
        browser_executable=None,
        max_steps=20,
        max_seconds=5,
        model="gpt-6-astra",
        reasoning_effort="xhigh",
    )
    env = FakeEnv(tmp_path)
    created = []

    def make(key, **kwargs):
        created.append((key, kwargs))
        return env

    monkeypatch.setattr(gym, "make", make)
    return args, env, created


def rpc(method, request_id=1, **params):
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})


async def exchange(args, lines):
    reader, writer = os.pipe()
    output = io.StringIO()
    try:
        os.write(writer, ("\n".join(lines) + "\n").encode())
        os.close(writer)
        writer = None
        code = await serve(args, input_fd=reader, output=output)
        return code, [json.loads(line) for line in output.getvalue().splitlines()]
    finally:
        os.close(reader)
        if writer is not None:
            os.close(writer)


def test_tools_reuse_canonical_actions_without_dom_or_source_tools():
    first = CodexBridge.tool_definitions()
    second = CodexBridge.tool_definitions()
    assert first == second  # Schema projection must not mutate the owning schema.
    assert [tool["name"] for tool in first] == ["get_observation", "computer", "finish"]
    schema = first[1]["inputSchema"]
    assert schema["required"] == ["actions", "decision_summary"]
    actions = schema["properties"]["actions"]["items"]["properties"]["action"]["enum"]
    assert {"click", "type", "key", "drag", "screenshot", "wait"} <= set(actions)
    assert not {"evaluate", "javascript", "selector", "read_file", "shell"} & set(actions)


async def test_handshake_notifications_and_shutdown_are_lazy(setup_bridge):
    args, env, created = setup_bridge
    code, replies = await exchange(
        args,
        [
            rpc("initialize", protocolVersion="2025-06-18"),
            '{"jsonrpc":"2.0","method":"notifications/initialized"}',
            rpc("tools/list", 2),
            rpc("ping", 3),
            rpc("shutdown", 4),
        ],
    )
    assert code == 0
    assert len(replies) == 4
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18"
    assert len(replies[1]["result"]["tools"]) == 3
    assert replies[2]["result"] == replies[3]["result"] == {}
    assert created == [] and env.reset_count == env.close_count == 0


async def test_protocol_errors_do_not_break_following_requests(setup_bridge):
    args, _, _ = setup_bridge
    code, replies = await exchange(
        args,
        [
            "not JSON",
            "[]",
            rpc("tools/list"),
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc("unknown"),
            rpc("tools/call", name="get_observation", arguments={"selector": "body"}),
            rpc("ping"),
        ],
    )
    assert code == 0
    assert [reply["error"]["code"] for reply in replies[:3]] == [-32700, -32600, -32002]
    assert replies[4]["error"]["code"] == -32601
    assert replies[5]["result"]["isError"]
    assert replies[6]["result"] == {}


async def test_actual_image_provenance_and_terminal_action_suppression(setup_bridge):
    args, env, created = setup_bridge
    summary = "Click the visible checkbox."
    click = {"actions": [{"action": "click", "coordinate": [50, 50]}], "decision_summary": summary}
    lines = [
        rpc("initialize", protocolVersion="2025-11-25"),
        rpc("tools/call", 2, name="get_observation", arguments={}),
        rpc("tools/call", 3, name="computer", arguments=click),
        rpc("tools/call", 4, name="computer", arguments=click),
        rpc("tools/call", 5, name="finish", arguments={"status": "success", "summary": "Done."}),
    ]
    code, replies = await exchange(args, lines)
    assert code == 0
    assert created[0][0] == "visual_tasks@click"
    assert created[0][1]["seed"] == 12
    assert env.reset_count == env.close_count == 1
    assert len(env.calls) == 1
    for reply in replies[1:]:
        assert base64.b64decode(reply["result"]["content"][1]["data"]) == PNG
        assert "evaluation_only_secret" not in json.dumps(reply)
        assert str(env.attempt_dir) not in json.dumps(reply)
    feedback = json.loads(replies[2]["result"]["content"][0]["text"])
    assert feedback["outcome"] == "success" and feedback["reward"] == 1
    events = [
        json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    inputs = [row["data"] for row in events if row["type"] == "controller_input"]
    assert [row["raw_text"] for row in inputs] == lines[1:]
    started = next(row["data"] for row in events if row["type"] == "controller_started")
    assert started["requested_model"] == "gpt-6-astra"
    assert started["requested_reasoning_effort"] == "xhigh"
    assert started["actual_model_verified"] is False
    decisions = [row["data"] for row in events if row["type"] == "model_decision"]
    assert decisions[0]["summary"] == summary
    assert decisions[0]["based_on"]["sha256"] == env.last_observation["sha256"]
    assert not decisions[0]["private_reasoning_available"]
    assert decisions[1]["ignored_after_terminal"]
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "success" and manifest["data"]["cleanup_complete"]
    result = json.loads((env.attempt_dir / "bridge_result.json").read_text())
    assert result["finish_claim"] == {"status": "success", "summary": "Done."}


async def test_finish_cannot_manufacture_success(setup_bridge):
    args, env, _ = setup_bridge
    code, replies = await exchange(
        args,
        [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc("tools/call", 2, name="get_observation"),
            rpc("tools/call", 3, name="get_observation"),
            rpc(
                "tools/call", 4, name="finish", arguments={"status": "success", "summary": "Claim."}
            ),
        ],
    )
    assert code == 0
    assert env.calls[0]["function"]["arguments"]["actions"] == [{"action": "screenshot"}]
    feedback = json.loads(replies[-1]["result"]["content"][0]["text"])
    assert feedback["outcome"] == "agent_stopped" and feedback["reward"] == 0


async def test_bad_tool_args_are_archived_and_retry_is_possible(setup_bridge):
    args, env, _ = setup_bridge
    code, replies = await exchange(
        args,
        [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc("tools/call", name="get_observation"),
            rpc("tools/call", name="computer", arguments={"actions": [], "decision_summary": 12}),
            rpc("tools/call", name="get_observation"),
        ],
    )
    assert code == 0
    assert replies[2]["result"]["isError"]
    assert not replies[3]["result"]["isError"]
    events = (env.attempt_dir / "events.jsonl").read_text()
    assert "controller_input_error" in events
    assert env.close_count == 1


@pytest.mark.parametrize("partial", [False, True])
async def test_idle_or_partial_stdin_timeout_closes_owned_env(setup_bridge, partial):
    args, env, _ = setup_bridge
    args.max_seconds = 0.15
    reader, writer = os.pipe()
    output = io.StringIO()
    try:
        messages = [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc("tools/call", name="get_observation"),
        ]
        os.write(writer, ("\n".join(messages) + "\n" + ("{" if partial else "")).encode())
        code = await asyncio.wait_for(serve(args, input_fd=reader, output=output), timeout=2)
    finally:
        os.close(reader)
        os.close(writer)
    assert code == 1 and env.close_count == 1
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "controller_timeout"
    assert manifest["data"]["cleanup_complete"]


async def test_timeout_during_action_closes_owned_env(setup_bridge):
    args, env, _ = setup_bridge
    args.max_seconds = 0.15

    async def slow_step(calls):
        await asyncio.sleep(5)

    env.step = slow_step
    code, _ = await exchange(
        args,
        [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc("tools/call", name="get_observation"),
            rpc("tools/call", name="get_observation"),
        ],
    )
    assert code == 1 and env.close_count == 1
    assert env.outcome == "controller_timeout"


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf"])
def test_cli_rejects_unbounded_time(tmp_path, budget):
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--task",
                "click",
                "--artifact-root",
                str(tmp_path),
                "--max-seconds",
                budget,
                "--model",
                "gpt-6-astra",
                "--reasoning-effort",
                "xhigh",
            ]
        )


def test_cli_task_choices_follow_local_catalog(tmp_path):
    from examples.not_a_robot.local_tasks import LOCAL_TASKS

    for task in LOCAL_TASKS:
        assert (
            _parse_args(
                [
                    "--task",
                    task,
                    "--artifact-root",
                    str(tmp_path),
                    "--model",
                    "gpt-6-astra",
                    "--reasoning-effort",
                    "xhigh",
                ]
            ).task
            == task
        )


def test_env_server_configuration_is_rejected(setup_bridge, monkeypatch):
    args, _, _ = setup_bridge
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://127.0.0.1:9999")
    with pytest.raises(RuntimeError, match="without env-server"):
        CodexBridge(args)


@pytest.mark.parametrize("disconnect", [False, True])
async def test_finish_finalizes_before_protocol_reply_even_if_client_disconnects(
    setup_bridge, disconnect
):
    args, env, _ = setup_bridge

    class CheckedOutput(io.StringIO):
        def write(self, text):
            reply = json.loads(text)
            if reply["id"] == 3:
                assert env.close_count == 1 and env.recorder is None
                manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
                assert manifest["recording_complete"]
                assert manifest["data"]["cleanup_complete"]
                assert (env.attempt_dir / "bridge_result.json").is_file()
                if disconnect:
                    raise BrokenPipeError("Client disconnects immediately on finish reply")
            return super().write(text)

    reader, writer = os.pipe()
    try:
        messages = [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc("tools/call", 2, name="get_observation"),
            rpc(
                "tools/call", 3, name="finish", arguments={"status": "failure", "summary": "Stop."}
            ),
        ]
        os.write(writer, ("\n".join(messages) + "\n").encode())
        os.close(writer)
        writer = None
        code = await serve(args, input_fd=reader, output=CheckedOutput())
    finally:
        os.close(reader)
        if writer is not None:
            os.close(writer)
    assert code == int(disconnect)
    assert env.close_count == 1
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "agent_stopped"


async def test_finish_duplicate_close_and_later_tools_preserve_immutable_archive(setup_bridge):
    args, env, _ = setup_bridge
    bridge = CodexBridge(args)
    await bridge.call_tool("get_observation", {}, "initial")
    final = await bridge.call_tool("finish", {"status": "failure", "summary": "Stop."}, "finish")
    events = (env.attempt_dir / "events.jsonl").read_bytes()
    manifest = (env.attempt_dir / "manifest.json").read_bytes()
    await bridge.close("controller_disconnected")
    await bridge.close("controller_shutdown")
    for name, arguments in [
        ("get_observation", {}),
        ("computer", {"actions": [{"action": "click"}], "decision_summary": "Late click."}),
        ("finish", {"status": "success", "summary": "Changed claim."}),
    ]:
        assert await bridge.call_tool(name, arguments, "late request") == final
    assert env.close_count == 1 and len(env.calls) == 1
    assert bridge.finish_claim == {"status": "failure", "summary": "Stop."}
    assert (env.attempt_dir / "events.jsonl").read_bytes() == events
    assert (env.attempt_dir / "manifest.json").read_bytes() == manifest


async def test_repeated_cancellation_during_finish_cannot_interrupt_cleanup(setup_bridge):
    args, env, _ = setup_bridge
    bridge = CodexBridge(args)
    await bridge.call_tool("get_observation", {}, "initial")
    closing = asyncio.Event()
    release = asyncio.Event()
    original_close = env.close

    async def slow_close():
        closing.set()
        await release.wait()
        await original_close()

    env.close = slow_close
    finish = asyncio.create_task(
        bridge.call_tool("finish", {"status": "failure", "summary": "Stop."}, "finish")
    )
    await closing.wait()
    finish.cancel()
    await asyncio.sleep(0)
    finish.cancel()
    release.set()
    response = await asyncio.wait_for(finish, timeout=1)
    assert not response["isError"] and bridge.closed
    assert env.close_count == 1
    assert (env.attempt_dir / "manifest.json").is_file()
    await bridge.close("later_shutdown")
    assert env.close_count == 1
