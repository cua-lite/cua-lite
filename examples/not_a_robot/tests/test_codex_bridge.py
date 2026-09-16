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
        self._terminal = False
        self._started = time.monotonic()
        self.timestamp = time.time
        self.png = PNG

    async def reset(self):
        self.reset_count += 1
        self.outcome = "in_progress"
        self.recorder = EventRecorder(self.attempt_dir, {"fixture": True})
        self.last_observation = self.recorder.image(
            self.png,
            variant="model_visible",
            capture_started_unix_s=self.timestamp(),
            capture_completed_unix_s=self.timestamp(),
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
            capture_started_unix_s=self.timestamp(),
            capture_completed_unix_s=self.timestamp(),
        )
        terminal = self.outcome != "in_progress"
        self._terminal = terminal
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
        record_video=False,
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
    assert '"up", "down", "left", "right"' in first[1]["description"]
    assert 'for ArrowDown use ["down"]' in first[1]["description"]


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


@pytest.mark.parametrize("record_video", [False, True])
async def test_actual_image_provenance_and_terminal_action_suppression(setup_bridge, record_video):
    args, env, created = setup_bridge
    args.record_video = record_video
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
    assert created[0][1]["record_video"] is record_video
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
                    {"action": "wait", "duration": 1.7},
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


@pytest.mark.asyncio
async def test_windows_break_cancels_serve_and_restores_signal_handlers(monkeypatch):
    """CTRL_BREAK must enter graceful shutdown, not kill the bridge before finalize."""
    from examples.not_a_robot import codex_bridge

    previous = {2: object(), 15: object(), 21: object()}
    registered = dict(previous)
    closed = []

    def install(signum, handler):
        old, registered[signum] = registered[signum], handler
        return old

    async def transport(*args, **kwargs):
        assert all(callable(registered[number]) for number in previous)
        registered[21](21, None)
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            closed.append(True)
            return 1

    monkeypatch.setattr(codex_bridge, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        codex_bridge,
        "signal",
        SimpleNamespace(
            SIGTERM=15,
            SIGINT=2,
            SIGBREAK=21,
            signal=install,
        ),
    )
    monkeypatch.setattr(
        codex_bridge,
        "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(fileno=lambda: 99),
        ),
    )
    monkeypatch.setattr(codex_bridge, "serve", transport)
    assert await codex_bridge._main(SimpleNamespace(), None) == 1
    assert closed == [True]
    assert registered == previous


@pytest.fixture
def sequence_bridge(setup_bridge, monkeypatch):
    """Synthetic protocol clock and PNGs, never model gameplay or video evidence."""
    from PIL import Image

    args, env, created = setup_bridge
    args.max_seconds, args.max_steps = 60, 200
    clock = SimpleNamespace(now=1000.0)

    async def advance(seconds):
        clock.now += seconds
        await asyncio.sleep(0)

    monkeypatch.setattr(
        bridge_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, time=lambda: 1700000000 + clock.now),
    )
    monkeypatch.setattr(
        bridge_module, "asyncio", SimpleNamespace(**(vars(asyncio) | {"sleep": advance}))
    )
    env.timestamp = bridge_module.time.time
    original_step = env.step

    async def numbered_step(calls):
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), ((env._steps + 1) % 256, 20, 30)).save(buffer, format="PNG")
        env.png = buffer.getvalue()
        return await original_step(calls)

    env.step = numbered_step
    return args, env, clock, created


@pytest.mark.parametrize("tool", ["get_observation", "computer"])
@pytest.mark.parametrize(
    "sequence",
    [
        None,
        {},
        [],
        {"duration_seconds": 1},
        {"duration_seconds": 1, "interval_seconds": 0.25, "selector": "body"},
        {"duration_seconds": 0, "interval_seconds": 0.25},
        {"duration_seconds": 10.01, "interval_seconds": 0.25},
        {"duration_seconds": 1, "interval_seconds": 0.099},
        {"duration_seconds": 1, "interval_seconds": 10.01},
        {"duration_seconds": True, "interval_seconds": 0.25},
        {"duration_seconds": 1, "interval_seconds": False},
        {"duration_seconds": "1", "interval_seconds": 0.25},
        {"duration_seconds": float("nan"), "interval_seconds": 0.25},
        {"duration_seconds": 1, "interval_seconds": float("inf")},
        {"duration_seconds": 10**400, "interval_seconds": 0.25},
    ],
)
async def test_sequence_invalid_fields_rejected_without_capture(setup_bridge, tool, sequence):
    args, env, created = setup_bridge
    bridge = CodexBridge(args)
    arguments = {"sequence": sequence}
    if tool == "computer":
        arguments.update(actions=[{"action": "screenshot"}], decision_summary="Protocol fixture.")
    with pytest.raises(ValueError):
        await bridge.call_tool(tool, arguments, "invalid sequence fixture")
    assert not created and env.reset_count == 0 and not env.calls
    await bridge.close("unit_fixture_done")


async def test_sequence_region_is_exclusive_before_environment_start(setup_bridge):
    args, env, created = setup_bridge
    bridge = CodexBridge(args)
    with pytest.raises(ValueError):
        await bridge.call_tool(
            "get_observation",
            {"sequence": {"duration_seconds": 1, "interval_seconds": 0.25}, "region": [0, 0, 1, 1]},
            "mutually exclusive fixture",
        )
    assert not created and env.reset_count == 0
    await bridge.close("unit_fixture_done")


@pytest.mark.parametrize("after_action", [False, True])
async def test_sequence_ordered_frames_real_steps_and_private_info_not_returned(
    sequence_bridge, after_action
):
    args, env, _, _ = sequence_bridge
    bridge = CodexBridge(args)
    sequence = {"duration_seconds": 1, "interval_seconds": 0.25}
    try:
        if after_action:
            await bridge.call_tool("get_observation", {}, "initial fixture")
            response = await bridge.call_tool(
                "computer",
                {
                    "actions": [{"action": "click", "coordinate": [30, 30]}],
                    "decision_summary": "Protocol-only action, not a game win.",
                    "sequence": sequence,
                },
                "action and sequence fixture",
            )
        else:
            response = await bridge.call_tool(
                "get_observation", {"sequence": sequence}, "first sequence fixture"
            )
        feedback = json.loads(response["content"][0]["text"])
        sampled = feedback["sequence"]
        frames = sampled["frames"]
        assert sampled["status"] == "complete" and sampled["stop_reason"] == "duration_elapsed"
        assert sampled["frame_count"] == len(frames) == 5
        assert [item["target_seconds"] for item in frames] == [0, 0.25, 0.5, 0.75, 1]
        assert frames[0]["request_relative_seconds"] is frames[0]["lag_seconds"] is None
        assert [item["index"] for item in frames] == list(range(5))
        assert env._steps == len(env.calls) == 4 + int(after_action)
        assert all(
            call["function"]["arguments"] == {"actions": [{"action": "screenshot"}]}
            for call in env.calls[int(after_action) :]
        )
        assert [block["type"] for block in response["content"]] == ["text"] + ["image"] * 5
        pngs = [base64.b64decode(block["data"]) for block in response["content"][1:]]
        assert len(set(pngs)) == 5
        assert sampled["png_bytes"] == sum(map(len, pngs))
        for frame, png in zip(frames, pngs, strict=True):
            assert frame["sha256"] == hashlib.sha256(png).hexdigest()
            assert frame["viewport"] == list(env.display_resolution)
            assert frame["capture_started_unix_s"] <= frame["capture_completed_unix_s"]
            assert frame["capture_relative_seconds"] == pytest.approx(
                frame["capture_started_unix_s"] - sampled["window_started_unix_s"]
            )
        assert feedback["observation"]["sha256"] == frames[-1]["sha256"]
        assert "evaluation_only_secret" not in json.dumps(response)
        assert str(env.attempt_dir) not in json.dumps(response)
        assert "state" not in feedback and "reference" not in feedback
        events = [
            json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
        ]
        prepared = [row["data"] for row in events if row["type"] == "controller_result"][-1]
        assert prepared["response_delivery"] == "prepared"
        assert [item["sha256"] for item in prepared["model_visible_frames"]] == [
            item["sha256"] for item in frames
        ]
        assert not any(row["type"] == "controller_response_sent" for row in events)
    finally:
        await bridge.close("unit_fixture_done")


async def test_sequence_actual_maximum_frame_limit(sequence_bridge):
    args, env, _, _ = sequence_bridge
    bridge = CodexBridge(args)
    try:
        response = await bridge.call_tool(
            "get_observation",
            {"sequence": {"duration_seconds": 10, "interval_seconds": 0.1}},
            "frame limit fixture",
        )
        sampled = json.loads(response["content"][0]["text"])["sequence"]
        assert sampled["status"] == "partial" and sampled["stop_reason"] == "frame_limit"
        assert sampled["frame_count"] == 64 and len(response["content"]) == 65
        assert env._steps == 63
    finally:
        await bridge.close("unit_fixture_done")


async def test_sequence_oversized_first_png_is_not_returned(sequence_bridge):
    args, env, _, _ = sequence_bridge
    # Byte accounting fixture: valid PNG prefix with intentionally oversized padding.
    env.png = PNG + b"\0" * (8 * 1024 * 1024)
    bridge = CodexBridge(args)
    try:
        response = await bridge.call_tool(
            "get_observation",
            {"sequence": {"duration_seconds": 1, "interval_seconds": 0.25}},
            "byte limit fixture",
        )
        feedback = json.loads(response["content"][0]["text"])
        assert feedback["sequence"]["stop_reason"] == "byte_limit"
        assert feedback["sequence"]["frame_count"] == feedback["sequence"]["png_bytes"] == 0
        assert feedback["observation"] is None and len(response["content"]) == 1
        assert env._steps == 0
    finally:
        await bridge.close("unit_fixture_done")


async def test_sequence_late_capture_skips_slots_without_catchup(sequence_bridge):
    args, env, clock, _ = sequence_bridge
    original_step = env.step

    async def delayed_step(calls):
        if env._steps == 0:
            clock.now += 0.6
        return await original_step(calls)

    env.step = delayed_step
    bridge = CodexBridge(args)
    try:
        response = await bridge.call_tool(
            "get_observation",
            {"sequence": {"duration_seconds": 1, "interval_seconds": 0.25}},
            "synthetic late capture fixture",
        )
        sampled = json.loads(response["content"][0]["text"])["sequence"]
        assert sampled["skipped_slots"] == [2, 3]
        assert [frame["target_seconds"] for frame in sampled["frames"]] == [0, 0.25, 1]
        assert sampled["frames"][1]["capture_relative_seconds"] == pytest.approx(0.85)
        assert sampled["frames"][2]["request_relative_seconds"] == pytest.approx(1)
        assert env._steps == 2
    finally:
        await bridge.close("unit_fixture_done")


async def test_sequence_archived_overflow_frame_cannot_be_next_decision_basis(
    sequence_bridge, monkeypatch
):
    args, env, _, _ = sequence_bridge
    monkeypatch.setattr(bridge_module, "SEQUENCE_MAX_PNG_BYTES", len(PNG) + 1)
    bridge = CodexBridge(args)
    try:
        response = await bridge.call_tool(
            "get_observation",
            {"sequence": {"duration_seconds": 1, "interval_seconds": 0.25}},
            "synthetic overflow fixture",
        )
        feedback = json.loads(response["content"][0]["text"])
        sampled = feedback["sequence"]
        assert sampled["stop_reason"] == "byte_limit" and sampled["frame_count"] == 1
        returned_hash = hashlib.sha256(base64.b64decode(response["content"][1]["data"])).hexdigest()
        assert feedback["observation"]["sha256"] == returned_hash
        assert env.last_observation["sha256"] != returned_hash and env._steps == 1
        await bridge.call_tool(
            "computer",
            {
                "actions": [{"action": "click", "coordinate": [30, 30]}],
                "decision_summary": "Use only actually returned fixture pixels.",
            },
            "next decision fixture",
        )
        events = [
            json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
        ]
        decision = [row["data"] for row in events if row["type"] == "model_decision"][-1]
        assert decision["based_on"]["sha256"] == returned_hash
        assert decision["based_on_sequence"]["frames"][-1]["sha256"] == returned_hash
    finally:
        await bridge.close("unit_fixture_done")


@pytest.mark.parametrize("limit", ["step", "time"])
async def test_sequence_stops_at_actual_budget(sequence_bridge, limit):
    args, env, _, _ = sequence_bridge
    if limit == "step":
        args.max_steps = 2
    else:
        args.max_seconds = 0.4
    bridge = CodexBridge(args)
    try:
        response = await bridge.call_tool(
            "get_observation",
            {"sequence": {"duration_seconds": 1, "interval_seconds": 0.25}},
            "synthetic budget fixture",
        )
        sampled = json.loads(response["content"][0]["text"])["sequence"]
        assert sampled["status"] == "partial"
        assert sampled["stop_reason"] == ("step_budget" if limit == "step" else "time_budget")
        assert env._steps == (2 if limit == "step" else 1)
        assert sampled["frame_count"] == env._steps + 1
    finally:
        await bridge.close("unit_fixture_done")


@pytest.mark.parametrize("stop", ["terminal", "infra_error", "tool_error", "campaign_boundary"])
async def test_sequence_stops_on_current_result_and_retains_intermediate_errors(
    sequence_bridge, stop
):
    args, env, _, _ = sequence_bridge
    original_step = env.step
    if stop == "campaign_boundary":
        args.campaign = True
        env.state = SimpleNamespace(task_id="neal_01")

    async def stop_step(calls):
        result = await original_step(calls)
        if stop == "terminal":
            env.outcome, env._terminal = "success", True
            result.terminated, result.reward = True, 1
        elif stop == "infra_error":
            env.outcome, env._terminal = "infra_error", True
            result.truncated, result.reward = True, 0
        elif stop == "tool_error":
            result.results[0].error = "Synthetic intermediate GUI error"
        else:
            env.state.task_id = "neal_02"
            result.info["campaign"] = {
                "displayed_task": "neal_02",
                "completed_tasks": ["neal_01"],
                "total_tasks": 48,
                "next_task": "neal_02",
                "unexecuted_actions": 0,
                "notice": "Completed neal_01; now displaying neal_02.",
            }
        return result

    env.step = stop_step
    bridge = CodexBridge(args)
    try:
        response = await bridge.call_tool(
            "get_observation",
            {"sequence": {"duration_seconds": 1, "interval_seconds": 0.25}},
            "synthetic state boundary, not actual game completion",
        )
        feedback = json.loads(response["content"][0]["text"])
        assert feedback["sequence"]["stop_reason"] == stop
        assert env._steps == 1 and feedback["sequence"]["frame_count"] == 2
        if stop == "tool_error":
            assert response["isError"] and feedback["errors"] == [
                "Synthetic intermediate GUI error"
            ]
        if stop == "infra_error":
            assert feedback["outcome"] == "infra_error" and feedback["truncated"]
    finally:
        await bridge.close("unit_fixture_done")


async def test_sequence_computer_action_error_is_not_overwritten_by_sampling(sequence_bridge):
    args, env, _, _ = sequence_bridge
    bridge = CodexBridge(args)
    try:
        await bridge.call_tool("get_observation", {}, "initial fixture")
        original_step = env.step

        async def errored_step(calls):
            result = await original_step(calls)
            result.results[0].error = "Synthetic action error before sampling"
            return result

        env.step = errored_step
        response = await bridge.call_tool(
            "computer",
            {
                "actions": [{"action": "key", "keys": ["bad_fixture_key"]}],
                "decision_summary": "Protocol-only error fixture.",
                "sequence": {"duration_seconds": 1, "interval_seconds": 0.25},
            },
            "action error fixture",
        )
        feedback = json.loads(response["content"][0]["text"])
        assert env._steps == 1 and response["isError"]
        assert feedback["errors"] == ["Synthetic action error before sampling"]
        assert feedback["sequence"]["stop_reason"] == "tool_error"
        assert feedback["sequence"]["frame_count"] == 1
    finally:
        await bridge.close("unit_fixture_done")


@pytest.mark.parametrize("flush_outcome", ["sent", "broken_pipe", "cancelled"])
async def test_sequence_sent_proof_requires_successful_protocol_flush(
    sequence_bridge, flush_outcome
):
    args, env, _, _ = sequence_bridge

    class CheckedOutput(io.StringIO):
        reply_id = None

        def write(self, text):
            self.reply_id = json.loads(text)["id"]
            return super().write(text)

        def flush(self):
            if self.reply_id == 2:
                events = [
                    json.loads(line)
                    for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
                ]
                assert not any(row["type"] == "controller_response_sent" for row in events)
                if flush_outcome == "broken_pipe":
                    raise BrokenPipeError("Synthetic sequence flush failure")
                if flush_outcome == "cancelled":
                    raise asyncio.CancelledError("Synthetic cancellation at flush")
            return super().flush()

    reader, writer = os.pipe()
    output = CheckedOutput()
    try:
        requests = [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc(
                "tools/call",
                2,
                name="get_observation",
                arguments={"sequence": {"duration_seconds": 1, "interval_seconds": 0.25}},
            ),
            rpc(
                "tools/call",
                3,
                name="finish",
                arguments={"status": "failure", "summary": "End protocol fixture."},
            ),
        ]
        os.write(writer, ("\n".join(requests) + "\n").encode())
        os.close(writer)
        writer = None
        code = await serve(args, input_fd=reader, output=output)
    finally:
        os.close(reader)
        if writer is not None:
            os.close(writer)
    events = [
        json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    sent = [row for row in events if row["type"] == "controller_response_sent"]
    assert code == int(flush_outcome != "sent") and env.close_count == 1
    if flush_outcome == "sent":
        assert len(sent) == 1 and sent[0]["data"]["turn"] == 1
        assert sent[0]["data"]["response_id"] == 2
        prepared = next(row for row in events if row["type"] == "controller_result")
        assert prepared["sequence"] < sent[0]["sequence"]
    else:
        assert sent == []
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]


async def test_cancel_during_sequence_capture_keeps_archive_without_sent_claim(sequence_bridge):
    args, env, _, _ = sequence_bridge
    entered, never = asyncio.Event(), asyncio.Event()

    async def blocked_capture(calls):
        entered.set()
        await never.wait()

    env.step = blocked_capture
    reader, writer = os.pipe()
    serving = None
    try:
        requests = [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc(
                "tools/call",
                2,
                name="get_observation",
                arguments={"sequence": {"duration_seconds": 1, "interval_seconds": 0.25}},
            ),
        ]
        os.write(writer, ("\n".join(requests) + "\n").encode())
        os.close(writer)
        writer = None
        output = io.StringIO()
        serving = asyncio.create_task(serve(args, input_fd=reader, output=output))
        await asyncio.wait_for(entered.wait(), timeout=2)
        serving.cancel()
        assert await asyncio.wait_for(serving, timeout=2) == 1
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        assert [reply["id"] for reply in replies] == [1]
    finally:
        if serving is not None and not serving.done():
            serving.cancel()
            await serving
        os.close(reader)
        if writer is not None:
            os.close(writer)
    events = [
        json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    assert any(row["type"] == "observation" for row in events)
    assert not any(row["type"] == "controller_response_sent" for row in events)
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "controller_cancelled" and env.close_count == 1
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]


@pytest.mark.live
async def test_sequence_real_level35_captures_without_answer_or_mouse_input(tmp_path):
    """Bounded real screenshot-channel smoke, not Astra or a game-success proof."""
    args = _parse_args(
        [
            "--task",
            "neal_35",
            "--artifact-root",
            str(tmp_path),
            "--model",
            "scripted-channel-test",
            "--reasoning-effort",
            "none",
            "--max-seconds",
            "30",
            "--max-steps",
            "100",
            *(
                ["--browser-executable", os.environ["NEAL_BROWSER_EXECUTABLE"]]
                if "NEAL_BROWSER_EXECUTABLE" in os.environ
                else []
            ),
        ]
    )
    code, replies = await exchange(
        args,
        [
            rpc("initialize", protocolVersion="2025-11-25"),
            rpc(
                "tools/call",
                2,
                name="get_observation",
                arguments={"sequence": {"duration_seconds": 8, "interval_seconds": 0.25}},
            ),
            rpc(
                "tools/call",
                3,
                name="finish",
                arguments={
                    "status": "failure",
                    "summary": "End the bounded screenshot-channel test without guessing.",
                },
            ),
        ],
    )
    assert code == 0 and not replies[1]["result"]["isError"]
    content = replies[1]["result"]["content"]
    feedback = json.loads(content[0]["text"])
    sampled = feedback["sequence"]
    frames = sampled["frames"]
    assert feedback["task"] == "neal_35" and feedback["outcome"] == "in_progress"
    assert sampled["stop_reason"] == "duration_elapsed"
    assert 8 <= sampled["frame_count"] <= 33 and len(content) == len(frames) + 1
    assert frames[-1]["capture_started_unix_s"] - frames[0]["capture_started_unix_s"] >= 7
    assert len({frame["sha256"] for frame in frames}) >= 3
    manifests = list(tmp_path.glob("*/manifest.json"))
    assert len(manifests) == 1
    archive = manifests[0].parent
    events = [json.loads(line) for line in (archive / "events.jsonl").read_text().splitlines()]
    prepared = next(row["data"] for row in events if row["type"] == "controller_result")
    for index, (frame, block, reference) in enumerate(
        zip(frames, content[1:], prepared["model_visible_frames"], strict=True)
    ):
        png = base64.b64decode(block["data"])
        assert png.startswith(b"\x89PNG") and png == (archive / reference["path"]).read_bytes()
        assert frame["index"] == index and hashlib.sha256(png).hexdigest() == frame["sha256"]
        assert frame["sha256"] == reference["sha256"]
        assert frame["capture_started_unix_s"] == reference["capture_started_unix_s"]
        assert frame["capture_completed_unix_s"] == reference["capture_completed_unix_s"]
    assert not any(
        row["type"] == "input_started" and row["data"]["call"].startswith(("mouse.", "keyboard."))
        for row in events
    )
    assert not any(
        row["type"] == "game_event" and row["data"]["kind"] in {"rejected", "completed"}
        for row in events
    )
    assert len([row for row in events if row["type"] == "controller_response_sent"]) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["outcome"] == "agent_stopped"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
