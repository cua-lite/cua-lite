"""Protocol, provenance and lifecycle checks, plus an explicitly live campaign smoke."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import time
from types import SimpleNamespace

import pytest

from examples.not_a_robot import codex_bridge as bridge_module
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
        self._steps = 0
        self._started = time.monotonic()
        self.png = PNG

    async def reset(self):
        self.reset_count += 1
        self.outcome = "in_progress"
        self.recorder = EventRecorder(self.attempt_dir, {"fixture": True})
        self.last_observation = self.recorder.image(
            self.png,
            variant="model_visible",
            capture_started_unix_s=time.time(),
            capture_completed_unix_s=time.time(),
        )
        return LiteEnvObservation(image=self.png, text="Complete the visible task.")

    async def step(self, calls):
        self._steps += 1
        self.calls.extend(calls)
        function = calls[0]["function"]
        if function["name"] == "terminate":
            self.outcome = "agent_stopped"
        elif function["arguments"]["actions"] == [{"action": "click", "coordinate": [50, 50]}]:
            self.outcome = "success"
        self.last_observation = self.recorder.image(
            self.png,
            variant="model_visible",
            capture_started_unix_s=time.time(),
            capture_completed_unix_s=time.time(),
        )
        terminal = self.outcome != "in_progress"
        return LiteEnvStepResult(
            results=[LiteToolResult(tool_call_id=calls[0]["id"], images=[self.png])],
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
        campaign=False,
        seed=12,
        reference_instance="default",
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
    assert '["ctrl", "a"]' in first[1]["description"]
    assert "lowercase canonical tokens" in first[1]["description"]


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
    assert created[0][1]["post_action_delay"] == 0
    assert env.reset_count == env.close_count == 1
    assert len(env.calls) == 1
    for reply in replies[1:]:
        assert base64.b64decode(reply["result"]["content"][1]["data"]) == PNG
        assert "evaluation_only_secret" not in json.dumps(reply)
        assert str(env.attempt_dir) not in json.dumps(reply)
    feedback = json.loads(replies[2]["result"]["content"][0]["text"])
    assert feedback["outcome"] == "success" and feedback["reward"] == 1
    assert 0 <= feedback["remaining_seconds"] <= args.max_seconds
    assert feedback["remaining_steps"] == args.max_steps - 1
    assert feedback["clock"] == "real_time"
    timing = feedback["observation"]
    assert timing["capture_started_unix_s"] <= timing["capture_completed_unix_s"]
    assert timing["age_seconds_at_feedback"] >= 0
    assert timing["age_seconds_at_feedback"] == pytest.approx(
        timing["feedback_at_unix_s"] - timing["capture_completed_unix_s"]
    )
    assert timing["sha256"] == env.last_observation["sha256"]
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


async def test_zoom_uses_only_returned_screenshot_and_records_its_source(setup_bridge):
    from PIL import Image

    args, env, _ = setup_bridge
    frame = Image.new("RGB", (4, 2))
    frame.putdata([(x * 50, y * 100, 30) for y in range(2) for x in range(4)])
    buffer = io.BytesIO()
    frame.save(buffer, format="PNG")
    env.png = buffer.getvalue()
    env.display_resolution = (4, 2)
    code, replies = await exchange(
        args,
        [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc("tools/call", 2, name="get_observation", arguments={"region": [0, 0, 500, 1000]}),
            rpc(
                "tools/call",
                3,
                name="computer",
                arguments={
                    "actions": [{"action": "click", "coordinate": [50, 50]}],
                    "decision_summary": "Click a visible target after inspecting the crop.",
                },
            ),
            rpc(
                "tools/call", 4, name="finish", arguments={"status": "success", "summary": "Done."}
            ),
        ],
    )
    assert code == 0
    content = replies[1]["result"]["content"]
    assert [block["type"] for block in content] == ["text", "image", "image"]
    assert base64.b64decode(content[1]["data"]) == env.png
    enlarged_bytes = base64.b64decode(content[2]["data"])
    enlarged = Image.open(io.BytesIO(enlarged_bytes))
    assert enlarged.size == (8, 8)
    assert enlarged.getpixel((0, 0)) == frame.getpixel((0, 0))
    assert enlarged.getpixel((7, 7)) == frame.getpixel((1, 1))
    feedback = json.loads(content[0]["text"])
    assert feedback["viewport"] == [4, 2]
    assert feedback["coordinate_space"] == "normalized_0_1000"
    assert feedback["zoom"]["source_pixel_box"] == [0, 0, 2, 2]
    assert feedback["zoom"]["sha256"] == hashlib.sha256(enlarged_bytes).hexdigest()
    assert str(env.attempt_dir) not in json.dumps(replies)
    events = [
        json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    response = next(row["data"] for row in events if row["type"] == "controller_result")
    zoom = response["model_visible_zoom"]
    assert (env.attempt_dir / zoom["path"]).read_bytes() == enlarged_bytes
    assert zoom["source_sha256"] == response["model_visible_image"]["sha256"]
    assert (
        zoom["capture_completed_unix_s"]
        == response["model_visible_image"]["capture_completed_unix_s"]
    )
    decisions = [row["data"] for row in events if row["type"] == "model_decision"]
    assert decisions[0]["based_on_zoom"] == zoom
    assert decisions[1]["based_on_zoom"] is None
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]


async def test_second_zoom_refreshes_pixels_and_links_each_decision_to_its_observation(
    setup_bridge,
):
    from PIL import Image

    args, env, _ = setup_bridge
    screenshots = []
    for color in [(200, 10, 20), (30, 190, 40)]:
        buffer = io.BytesIO()
        Image.new("RGB", (4, 2), color).save(buffer, format="PNG")
        screenshots.append(buffer.getvalue())
    env.display_resolution = (4, 2)
    bridge = CodexBridge(args)
    try:
        env.png = screenshots[0]
        await bridge.call_tool("get_observation", {"region": [0, 0, 500, 1000]}, "first crop")
        first_image, first_zoom = env.last_observation, bridge.last_zoom
        env.png = screenshots[1]
        result = await bridge.call_tool(
            "get_observation", {"region": [500, 0, 1000, 1000]}, "second crop"
        )
        second_image, second_zoom = env.last_observation, bridge.last_zoom
        await bridge.call_tool(
            "computer",
            {
                "actions": [{"action": "click", "coordinate": [50, 50]}],
                "decision_summary": "Act on the newly observed pixels.",
            },
            "click",
        )
    finally:
        await bridge.close("test_finished")

    assert env.calls[0]["function"]["arguments"] == {"actions": [{"action": "screenshot"}]}
    assert env._steps == 2
    assert first_image["sha256"] != second_image["sha256"]
    assert first_zoom["sha256"] != second_zoom["sha256"]
    assert base64.b64decode(result["content"][1]["data"]) == screenshots[1]
    with Image.open(io.BytesIO(base64.b64decode(result["content"][2]["data"]))) as enlarged:
        assert enlarged.getpixel((0, 0)) == enlarged.getpixel((7, 7)) == (30, 190, 40)
    for image, zoom in [(first_image, first_zoom), (second_image, second_zoom)]:
        assert zoom["source_sha256"] == image["sha256"]
        assert zoom["capture_started_unix_s"] == image["capture_started_unix_s"]
        assert zoom["capture_completed_unix_s"] == image["capture_completed_unix_s"]
    events = [
        json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    decisions = [row["data"] for row in events if row["type"] == "model_decision"]
    assert [decision["based_on"] for decision in decisions] == [first_image, second_image]
    assert [decision["based_on_zoom"] for decision in decisions] == [first_zoom, second_zoom]
    responses = [row["data"] for row in events if row["type"] == "controller_result"]
    assert [response["model_visible_zoom"] for response in responses] == [
        first_zoom,
        second_zoom,
        None,
    ]


@pytest.mark.parametrize(
    ("frame_size", "region", "pixel_box", "output_size"),
    [
        ((8, 4), [125.1, 250.1, 499.9, 749.9], [1, 1, 4, 3], [12, 8]),
        ((8, 4), [999.25, 749.5, 1000, 1000], [7, 2, 8, 4], [4, 8]),
        ((1280, 800), [0, 0, 1000, 1000], [0, 0, 1280, 800], [1600, 1000]),
        ((800, 1280), [0, 0, 1000, 1000], [0, 0, 800, 1280], [1000, 1600]),
    ],
)
async def test_zoom_pixel_rounding_and_longest_edge_limit(
    setup_bridge, frame_size, region, pixel_box, output_size
):
    from PIL import Image

    args, env, _ = setup_bridge
    frame = Image.new("RGB", frame_size, (20, 40, 60))
    frame.putpixel((pixel_box[0], pixel_box[1]), (200, 10, 20))
    frame.putpixel((pixel_box[2] - 1, pixel_box[3] - 1), (30, 190, 40))
    buffer = io.BytesIO()
    frame.save(buffer, format="PNG")
    env.png = buffer.getvalue()
    env.display_resolution = frame_size
    bridge = CodexBridge(args)
    try:
        result = await bridge.call_tool("get_observation", {"region": region}, "inspect region")
    finally:
        await bridge.close("test_finished")

    feedback = json.loads(result["content"][0]["text"])
    assert feedback["viewport"] == list(frame_size)
    assert feedback["zoom"]["region"] == region
    assert feedback["zoom"]["source_pixel_box"] == pixel_box
    assert feedback["zoom"]["output_size"] == output_size
    assert max(output_size) <= 1600
    assert base64.b64decode(result["content"][1]["data"]) == env.png
    with Image.open(io.BytesIO(base64.b64decode(result["content"][2]["data"]))) as enlarged:
        assert list(enlarged.size) == output_size
        assert enlarged.getpixel((0, 0)) == (200, 10, 20)
        assert enlarged.getpixel((enlarged.width - 1, enlarged.height - 1)) == (30, 190, 40)


async def test_invalid_zoom_during_active_attempt_preserves_provenance_and_allows_retry(
    setup_bridge,
):
    args, env, _ = setup_bridge
    code, replies = await exchange(
        args,
        [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc("tools/call", 2, name="get_observation", arguments={"region": [0, 0, 1000, 1000]}),
            rpc("tools/call", 3, name="get_observation", arguments={"region": [750, 0, 250, 1000]}),
            rpc(
                "tools/call",
                4,
                name="computer",
                arguments={
                    "actions": [{"action": "click", "coordinate": [50, 50]}],
                    "decision_summary": "Use the last valid screenshot after the rejected crop.",
                },
            ),
            rpc(
                "tools/call", 5, name="finish", arguments={"status": "success", "summary": "Done."}
            ),
        ],
    )
    assert code == 0 and env.reset_count == env.close_count == 1
    assert replies[2]["result"]["isError"]
    assert "positive area" in replies[2]["result"]["content"][0]["text"]
    assert not replies[3]["result"]["isError"]
    assert json.loads(replies[3]["result"]["content"][0]["text"])["outcome"] == "success"
    assert len(env.calls) == env._steps == 1
    events = [
        json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    responses = [row["data"] for row in events if row["type"] == "controller_result"]
    decisions = [row["data"] for row in events if row["type"] == "model_decision"]
    assert [response["turn"] for response in responses] == [1, 3, 4]
    assert decisions[0]["based_on_zoom"] == responses[0]["model_visible_zoom"]
    assert decisions[0]["based_on"] == responses[0]["model_visible_image"]
    errors = [row["data"] for row in events if row["type"] == "controller_input_error"]
    assert len(errors) == 1 and errors[0]["turn"] == 2
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "success" and manifest["recording_complete"]


@pytest.mark.parametrize(
    "region",
    [
        None,
        [],
        [0, 0, 1],
        [0, 0, 0, 1],
        [-1, 0, 2, 3],
        [0, 0, 1001, 1000],
        [False, 0, 2, 3],
        [0, "0", 2, 3],
        [0, 0, 10**400, 1000],
        [0, 0, float("nan"), 1000],
        [0, 0, float("inf"), 1000],
    ],
)
async def test_invalid_zoom_is_rejected_before_environment_start(setup_bridge, region):
    args, env, created = setup_bridge
    code, replies = await exchange(
        args,
        [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc("tools/call", 2, name="get_observation", arguments={"region": region}),
        ],
    )
    assert code == 0 and replies[-1]["result"]["isError"]
    assert "positive area" in replies[-1]["result"]["content"][0]["text"]
    assert not created and env.reset_count == 0


async def test_remaining_time_uses_controller_deadline_not_later_env_start(
    setup_bridge, monkeypatch
):
    args, env, _ = setup_bridge
    monkeypatch.setattr(
        bridge_module, "time", SimpleNamespace(monotonic=lambda: 200.0, time=time.time)
    )
    bridge = CodexBridge(args)
    bridge.deadline = 203.0
    env._started = 900.0
    try:
        result = await bridge.call_tool("get_observation", {}, "{}")
        feedback = json.loads(result["content"][0]["text"])
        assert feedback["remaining_seconds"] == 3.0
    finally:
        await bridge.close("test_finished")


def test_campaign_cli_is_explicit_and_excludes_independent_task(tmp_path):
    common = ["--artifact-root", str(tmp_path), "--model", "test", "--reasoning-effort", "test"]
    args = _parse_args([*common, "--campaign"])
    assert args.campaign and args.task is None
    with pytest.raises(SystemExit):
        _parse_args([*common, "--campaign", "--task", "neal_01"])
    with pytest.raises(SystemExit):
        _parse_args(common)


@pytest.mark.live
async def test_campaign_bridge_returns_new_page_and_keeps_previous_crop_provenance(tmp_path):
    from examples.not_a_robot.tests.test_local_tasks import center

    args = _parse_args(
        [
            "--campaign",
            "--artifact-root",
            str(tmp_path),
            "--model",
            "scripted-test",
            "--reasoning-effort",
            "none",
            "--max-seconds",
            "60",
            *(
                ["--browser-executable", os.environ["NEAL_BROWSER_EXECUTABLE"]]
                if "NEAL_BROWSER_EXECUTABLE" in os.environ
                else []
            ),
        ]
    )
    bridge = CodexBridge(args)
    try:
        initial = await bridge.call_tool(
            "get_observation", {"region": [250, 250, 750, 750]}, "initial"
        )
        raw = bridge.env.unwrapped
        first_image, first_zoom = raw.last_observation, bridge.last_zoom
        result = await bridge.call_tool(
            "computer",
            {
                "actions": [
                    {
                        "action": "click",
                        "coordinate": await center(bridge.env, raw._page.get_by_role("checkbox")),
                    },
                    {"action": "wait", "duration": 0.8},
                    {"action": "click", "coordinate": [500, 500]},
                ],
                "decision_summary": "Scripted boundary test: click the visible checkbox then wait.",
            },
            "complete first level",
        )
        feedback = json.loads(result["content"][0]["text"])
        assert feedback["task"] == "neal_02"
        assert feedback["campaign"]["completed_tasks"] == ["neal_01"]
        assert feedback["campaign"]["unexecuted_actions"] >= 1
        assert feedback["outcome"] == "in_progress" and not feedback["terminal"]
        assert feedback["reward"] is None
        assert bridge.last_zoom is None
        assert feedback["observation"]["sha256"] != first_image["sha256"]
        assert len(result["content"]) == 2
        assert (
            base64.b64decode(result["content"][1]["data"])
            == (raw.attempt_dir / raw.last_observation["path"]).read_bytes()
        )
        assert "state" not in feedback and "reference" not in feedback
        assert "level 1 through 48" in json.loads(initial["content"][0]["text"])["instruction"]
        cropped = await bridge.call_tool(
            "get_observation", {"region": [250, 250, 750, 750]}, "new crop"
        )
        assert json.loads(cropped["content"][0]["text"])["task"] == "neal_02"
        assert bridge.last_zoom["source_sha256"] == raw.last_observation["sha256"]
        assert bridge.last_zoom["source_sha256"] != first_image["sha256"]
        finished = await bridge.call_tool(
            "finish",
            {
                "status": "failure",
                "summary": "End this bounded infrastructure test after the transition.",
            },
            "stop",
        )
        final = json.loads(finished["content"][0]["text"])
        assert final["terminal"] and final["outcome"] == "agent_stopped"
        assert final["reward"] == 0
        assert final["campaign"]["completed_tasks"] == ["neal_01"]
        assert await bridge.call_tool("get_observation", {}, "after close") == finished
    finally:
        await bridge.close("test_finished")
    events = [
        json.loads(line) for line in (raw.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    decisions = [row["data"] for row in events if row["type"] == "model_decision"]
    assert decisions[0]["based_on"] == first_image
    assert decisions[0]["based_on_zoom"] == first_zoom
    manifest = json.loads((raw.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "agent_stopped"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]


async def test_serve_enforces_the_same_bridge_deadline(setup_bridge, monkeypatch):
    args, env, _ = setup_bridge
    bridge = CodexBridge(args)
    bridge.deadline = time.monotonic() - 1
    monkeypatch.setattr(bridge_module, "CodexBridge", lambda _: bridge)
    reader, writer = os.pipe()
    os.close(writer)
    try:
        code = await serve(args, input_fd=reader, output=io.StringIO())
    finally:
        os.close(reader)
    assert code == 1 and bridge.closed and env.reset_count == 0
