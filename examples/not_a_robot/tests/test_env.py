"""Registration and browser boundary checks for the experimental environment."""

from __future__ import annotations

import hashlib
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from examples.not_a_robot import env as env_module
from examples.not_a_robot import registration  # noqa: F401
from examples.not_a_robot.catalog import LEVELS
from examples.not_a_robot.env import LIVE_ENVS, NotARobotEnv
from examples.not_a_robot.evaluator import GameState
from examples.not_a_robot.recorder import EventRecorder
from lite import gym
from lite.core.tools.calls import make_tool_call


def test_metadata_before_reset_and_registry_identity():
    env = gym.make("not_a_robot@level_001", mode="fixture")
    assert env.metadata.dims == ("browser", "use")
    assert env.metadata.others["env_id"] == "not_a_robot"
    assert env.metadata.others["task_id"] == "level_001"
    assert env.metadata.others["execution_mode"] == "fixture"
    assert env.unwrapped.recorder is None
    assert len(LEVELS) == 48
    assert sum(row["independent_start_implemented"] for row in LEVELS) == 1


def test_unimplemented_level_is_explicit():
    with pytest.raises(NotImplementedError):
        NotARobotEnv(target_level=2)


@pytest.fixture
def mock_env():
    """Active episode with mocked browser I/O and an in-memory recorder."""
    env = NotARobotEnv(mode="fixture", target_level=None, cursor=False, post_action_delay=0)
    env._page = SimpleNamespace(
        url="https://neal.fun/not-a-robot/",
        title=AsyncMock(return_value="I'm not a robot"),
        locator=Mock(
            return_value=SimpleNamespace(
                inner_text=AsyncMock(return_value="Level 1: Checkbox\nI'm not a robot")
            )
        ),
        screenshot=AsyncMock(return_value=b"mock screenshot"),
        mouse=SimpleNamespace(
            move=AsyncMock(), click=AsyncMock(), down=AsyncMock(), up=AsyncMock(), wheel=AsyncMock()
        ),
        keyboard=SimpleNamespace(
            type=AsyncMock(), press=AsyncMock(), down=AsyncMock(), up=AsyncMock()
        ),
    )
    recorder = Mock(spec=EventRecorder)
    recorder.emit.side_effect = lambda *args, **kwargs: {"sequence": recorder.emit.call_count}
    recorder.image.return_value = {"path": "images/mock.png"}
    env.recorder = recorder
    env.state = GameState(1, "Checkbox", "in_progress", "Verified initial mock observation")
    env.outcome = "in_progress"
    env._started = time.monotonic()
    return env


@pytest.mark.parametrize("gate", ["title", "http_status", "cf_mitigated"])
async def test_late_access_gate_prevents_input(mock_env, gate):
    if gate == "title":
        mock_env._page.title.return_value = "Just a moment..."
    elif gate == "http_status":
        mock_env._main_status = 403
    else:
        mock_env._main_gate = "challenge"

    result = await mock_env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [500, 350]}]},
                call_id="late_gate",
            )
        ]
    )

    assert mock_env.state.status == "access_blocked"
    assert result.truncated and not result.terminated
    assert result.info["outcome"] == "access_blocked"
    mock_env._page.mouse.move.assert_not_awaited()
    mock_env._page.mouse.click.assert_not_awaited()
    assert "input_started" not in [call.args[0] for call in mock_env.recorder.emit.call_args_list]


async def test_access_gate_between_typing_and_enter_stops_compound_action(mock_env):
    mock_env._page.title.side_effect = [
        "I'm not a robot",
        "Just a moment...",
        "Just a moment...",
    ]
    result = await mock_env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "type", "text": "READY", "press_enter": True}]},
                call_id="compound_gate",
            )
        ]
    )

    mock_env._page.keyboard.type.assert_awaited_once_with(text="READY")
    mock_env._page.keyboard.press.assert_not_awaited()
    assert result.truncated and result.info["outcome"] == "access_blocked"
    inputs = [
        call.kwargs["call"]
        for call in mock_env.recorder.emit.call_args_list
        if call.args[0] == "input_started"
    ]
    assert inputs == ["keyboard.type"]


@pytest.mark.parametrize("failed_read", ["screenshot", "title", "body_text"])
async def test_observation_failure_is_retained_as_infrastructure_error(mock_env, failed_read):
    error = RuntimeError(f"{failed_read} failed")
    if failed_read == "body_text":
        mock_env._page.locator.return_value.inner_text.side_effect = error
    else:
        getattr(mock_env._page, failed_read).side_effect = error
    recorder = mock_env.recorder

    with pytest.raises(RuntimeError, match=f"{failed_read} failed"):
        await mock_env.step(
            [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "screenshot"}]},
                    call_id="failed_observation",
                )
            ]
        )

    assert mock_env.outcome == "infra_error"
    assert mock_env._terminal
    recorder.emit.assert_any_call("error", phase="post_action", error=repr(error))
    await mock_env.close()
    recorder.finalize.assert_called_once_with(
        "infra_error", cleanup_errors=[], cleanup_complete=True
    )


@pytest.mark.parametrize("terminal_status", [None, "success", "failure"])
@pytest.mark.parametrize("cursor", [False, True])
async def test_observation_capture_intervals_follow_returned_image(
    tmp_path, monkeypatch, terminal_status, cursor
):
    """A recaptured frame owns its timestamps, including after cursor projection."""
    env = NotARobotEnv(mode="local", target_level=None, local_task="click", cursor=cursor)
    env._cursor_xy = (19.2, 22.8)
    initial_state = {
        "task_id": "click",
        "label": "Click",
        "version": "test",
        "seed": 0,
        "reference_instance": "default",
        "status": "in_progress",
        "mistakes": 0,
        "progress": 0,
        "reason": "",
        "elapsed_ms": 500,
    }
    final_state = dict(initial_state, status=terminal_status or "in_progress", elapsed_ms=600)
    # Screenshot bytes are opaque at this boundary; the renderer is mocked.
    frames = [b"first browser frame"]
    intervals = [(1000.0, 1000.125)]
    if terminal_status:
        frames.append(b"recaptured terminal browser frame")
        intervals.append((1001.0, 1001.25))
    env._page = SimpleNamespace(
        url="http://127.0.0.1/mock-game",
        evaluate=AsyncMock(
            side_effect=[dict(initial_state, events=[]), dict(final_state, events=[])]
        ),
        screenshot=AsyncMock(side_effect=frames),
    )
    capture_clock = Mock(side_effect=[value for interval in intervals for value in interval])
    monkeypatch.setattr(env_module, "time", SimpleNamespace(time=capture_clock))
    model_frame = b"cursor overlay of " + frames[-1] if cursor else frames[-1]
    overlay = Mock(return_value=model_frame)
    monkeypatch.setattr(env_module, "overlay_cursor_px", overlay)
    recorder = EventRecorder(tmp_path / "capture", {"fixture": True})
    env.recorder = recorder
    try:
        image = await env._observe("test_capture")
    finally:
        recorder.close()

    assert image == model_frame
    assert env._page.screenshot.await_count == len(frames)
    assert capture_clock.call_count == 2 * len(frames)
    if cursor:
        overlay.assert_called_once_with(frames[-1], 19, 23)
    else:
        overlay.assert_not_called()
    events = [
        json.loads(line) for line in (recorder.root / "events.jsonl").read_text().splitlines()
    ]
    observations = [event["data"]["image"] for event in events if event["type"] == "observation"]
    expected = [("raw", frames[-1], intervals[-1]), ("model_visible", model_frame, intervals[-1])]
    if terminal_status:
        expected.insert(0, ("before_terminal_transition", frames[0], intervals[0]))
    assert len(observations) == len(expected)
    for observation, (variant, content, interval) in zip(observations, expected, strict=True):
        assert observation["phase"] == "test_capture"
        assert observation["variant"] == variant
        assert observation["capture_started_unix_s"] == interval[0]
        assert observation["capture_completed_unix_s"] == interval[1]
        assert observation["sha256"] == hashlib.sha256(content).hexdigest()
        assert (recorder.root / observation["path"]).read_bytes() == content
    assert env.last_observation == observations[-1]
    assert events[-1]["type"] == "task_state"
    assert events[-1]["data"]["state"] == final_state


@pytest.mark.parametrize("press_enter", [True, False])
async def test_type_honors_canonical_press_enter(mock_env, press_enter):
    result = await mock_env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "type", "text": "READY", "press_enter": press_enter}]},
                call_id="type_ready",
            )
        ]
    )

    assert result.results[0].error is None
    mock_env._page.keyboard.type.assert_awaited_once_with(text="READY")
    if press_enter:
        mock_env._page.keyboard.press.assert_awaited_once_with(key="Enter")
    else:
        mock_env._page.keyboard.press.assert_not_awaited()


async def test_drag_honors_canonical_right_button(mock_env):
    result = await mock_env.step(
        [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {
                            "action": "drag",
                            "start_coordinate": [100, 200],
                            "coordinate": [500, 600],
                            "button": "right",
                        }
                    ]
                },
                call_id="right_drag",
            )
        ]
    )

    assert result.results[0].error is None
    mock_env._page.mouse.down.assert_awaited_once_with(button="right")
    mock_env._page.mouse.up.assert_awaited_once_with(button="right")
    assert mock_env._page.mouse.move.await_args.kwargs == {"x": 640.0, "y": 480.0}


async def test_coordinate_less_click_uses_previous_cursor_position(mock_env):
    result = await mock_env.step(
        [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "mouse_move", "coordinate": [250, 500]},
                        {"action": "click"},
                    ]
                },
                call_id="move_then_click",
            )
        ]
    )

    assert result.results[0].error is None
    mock_env._page.mouse.move.assert_awaited_once_with(x=320, y=400)
    mock_env._page.mouse.click.assert_awaited_once_with(x=320, y=400, button="left", click_count=1)


async def test_gui_argument_error_returns_current_screenshot(mock_env):
    result = await mock_env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "key", "keys": ["UNKNOWN"]}]},
                call_id="invalid_gui",
            )
        ]
    )

    assert not result.terminated and not result.truncated
    assert result.results[0].tool_call_id == "invalid_gui"
    assert result.results[0].error
    assert result.results[0].images == [b"mock screenshot"]
    mock_env._page.keyboard.press.assert_not_awaited()


@pytest.mark.live
async def test_fixture_success_and_cleanup(tmp_path):
    env = gym.make(
        "not_a_robot@level_001",
        mode="fixture",
        artifact_root=str(tmp_path),
        browser_executable=os.environ.get("NEAL_BROWSER_EXECUTABLE"),
    )
    raw = env.unwrapped
    try:
        observation = await env.reset()
        assert observation.image.startswith(b"\x89PNG")
        result = await env.step(
            [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "click", "coordinate": [500, 355]},
                        ]
                    },
                    call_id="click_1",
                )
            ]
        )
        assert result.terminated and not result.truncated and result.reward == 1
        assert raw.state.level == 2
        assert result.results[0].tool_call_id == "click_1"
    finally:
        await env.close()
    manifest = json.loads((raw.attempt_dir / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["outcome"] == "success"
    assert manifest["data"]["cleanup_complete"]
    assert not LIVE_ENVS


@pytest.mark.live
async def test_rejected_action_retry_and_terminate(tmp_path):
    env = gym.make(
        "not_a_robot_campaign@full_game",
        mode="fixture",
        artifact_root=str(tmp_path),
        browser_executable=os.environ.get("NEAL_BROWSER_EXECUTABLE"),
    )
    raw = env.unwrapped
    try:
        await env.reset()
        rejected = await env.step(
            [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "key", "keys": ["UNKNOWN"]},
                        ]
                    },
                    call_id="bad_key",
                )
            ]
        )
        assert rejected.results[0].error
        assert not rejected.terminated and not rejected.truncated
        retried = await env.step(
            [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "key", "keys": ["tab"]},
                        ]
                    },
                    call_id="retry",
                )
            ]
        )
        assert retried.results[0].error is None
        stopped = await env.step(
            [make_tool_call("terminate", {"status": "failure"}, call_id="stop")]
        )
        assert stopped.terminated and stopped.reward == 0
        assert stopped.results == []
    finally:
        await env.close()
    manifest = json.loads((raw.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "agent_stopped"


@pytest.mark.live
async def test_fixture_keyboard_drag_and_cursor_rendering(tmp_path):
    env = gym.make(
        "not_a_robot_campaign@full_game",
        mode="fixture",
        artifact_root=str(tmp_path),
        browser_executable=os.environ.get("NEAL_BROWSER_EXECUTABLE"),
    )
    raw = env.unwrapped
    try:
        await env.reset()
        result = await env.step(
            [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "click", "coordinate": [385, 386]},
                            {"action": "click", "coordinate": [336, 325]},
                            {"action": "type", "text": "READY", "press_enter": True},
                            {
                                "action": "drag",
                                "start_coordinate": [256, 578],
                                "coordinate": [618, 576],
                            },
                        ]
                    },
                    call_id="fixture_actions",
                )
            ]
        )
        assert not result.truncated and not result.terminated
        assert result.results[0].error is None
        assert all(frame.startswith(b"\x89PNG") for frame in result.results[0].images)
        assert await raw._page.locator("#complete").is_visible()
    finally:
        await env.close()
