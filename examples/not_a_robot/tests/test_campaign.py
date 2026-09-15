"""Local campaign boundaries, not proof of a completed 48-level model run.

Browser checks solve the real first checkbox through canonical GUI actions.
DOM geometry and status are scripted test oracles, never policy inputs. The
missing-next and final-level cases explicitly inject controller boundary fixtures;
they do not claim traversal of the unimplemented levels.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.not_a_robot import env as env_module
from examples.not_a_robot import registration  # noqa: F401
from examples.not_a_robot.codex_bridge import CodexBridge
from examples.not_a_robot.env import LIVE_ENVS, LOCAL_CAMPAIGN_TASKS, NotARobotEnv
from examples.not_a_robot.local_tasks import CATALOG, LocalTaskState
from examples.not_a_robot.recorder import EventRecorder
from examples.not_a_robot.tests.test_expansion_grids import GRIDS
from examples.not_a_robot.tests.test_first10 import (
    ACCEPTED_SELECTIONS,
    RECURSIVE_ROWS,
    win_tic_tac_toe,
)
from examples.not_a_robot.tests.test_local_tasks import center, click, gui
from lite import gym
from lite.core.tools.calls import make_tool_call


@pytest.fixture
async def campaign_env(tmp_path):
    created = []

    async def create(**kwargs):
        env = gym.make(
            "visual_tasks_campaign@full_game",
            artifact_root=str(tmp_path),
            browser_executable=os.environ.get("NEAL_BROWSER_EXECUTABLE"),
            post_action_delay=0,
            cursor=False,
            **kwargs,
        )
        created.append(env)
        observation = await env.reset()
        assert observation.image.startswith(b"\x89PNG")
        assert env.unwrapped.state.task_id == "neal_01"
        assert env.unwrapped.state.status == "in_progress"
        return env

    yield create
    for env in created:
        resource_id = env.unwrapped.external_resource_id
        await env.close()
        assert resource_id not in LIVE_ENVS


def test_campaign_registration_is_distinct_and_declares_all_48_in_order():
    env = gym.make("visual_tasks_campaign@full_game", seed=37)
    assert LOCAL_CAMPAIGN_TASKS == tuple(f"neal_{level:02d}" for level in range(1, 49))
    assert env.metadata.others["env_id"] == "visual_tasks_campaign"
    assert env.metadata.others["task_id"] == "full_game"
    assert env.metadata.others["execution_mode"] == "local"
    assert env.metadata.others["local_campaign"] is True
    assert env.metadata.others["ordered_tasks"] == list(LOCAL_CAMPAIGN_TASKS)
    assert env.metadata.others["seed"] == 37
    assert env.unwrapped.current_local_task == "neal_01"
    assert env.unwrapped._page is None
    remote = gym.make("not_a_robot_campaign@full_game")
    assert remote.metadata.others["execution_mode"] == "live"
    assert not remote.unwrapped.local_campaign


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "local", "target_level": None, "local_task": "neal_01"},
        {"mode": "fixture", "target_level": None},
        {"mode": "live", "target_level": None},
        {"mode": "local", "target_level": 1},
    ],
)
def test_campaign_constructor_rejects_conflicting_modes(kwargs):
    with pytest.raises(ValueError):
        NotARobotEnv(local_campaign=True, **kwargs)


@pytest.mark.live
async def test_first_stage_transition_preserves_episode_and_stops_action_tail(campaign_env):
    env = await campaign_env(max_steps=9, seed=23)
    raw, page = env.unwrapped, env.unwrapped._page
    owned = (raw._browser, raw._context, page, raw.recorder, raw._local_server)
    attempt, resource_id, started = raw.attempt_dir, raw.external_resource_id, raw._started
    result = await gui(
        env,
        [
            {"action": "click", "coordinate": await center(env, page.get_by_role("checkbox"))},
            {"action": "wait", "duration": 0.8},
            {"action": "type", "text": "UNEXECUTED_CAMPAIGN_TAIL"},
        ],
    )
    assert not result.terminated and not result.truncated and result.reward is None
    assert raw.outcome == "in_progress" and raw.state.task_id == "neal_02"
    assert raw.state.status == "in_progress" and raw.state.progress == 0
    assert raw._completed_tasks == ["neal_01"]
    assert (raw._browser, raw._context, raw._page, raw.recorder, raw._local_server) == owned
    assert (raw.attempt_dir, raw.external_resource_id, raw._started) == (
        attempt,
        resource_id,
        started,
    )
    assert raw._steps == 1 and raw.max_steps == 9
    assert [action["call"] for action in result.info["executed_actions"]] == ["click", "wait"]
    assert "not executed" in result.info["campaign"]["notice"]
    assert result.info["campaign"]["displayed_task"] == "neal_02"
    assert result.info["campaign"]["total_tasks"] == 48
    assert not raw._scope_violation
    assert any(
        raw.last_observation["sha256"] == hashlib.sha256(image).hexdigest()
        for item in result.results
        for image in item.images
    )

    await env.close()
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert [event["elapsed_seconds"] for event in events] == sorted(
        event["elapsed_seconds"] for event in events
    )
    assert sum(event["type"] == "attempt_start" for event in events) == 1
    assert sum(event["type"] == "browser_started" for event in events) == 1
    starts = [event for event in events if event["type"] == "campaign_stage_started"]
    assert [event["data"]["task_id"] for event in starts] == ["neal_01", "neal_02"]
    completed = next(event for event in events if event["type"] == "campaign_stage_completed")
    assert completed["data"]["state"]["task_id"] == "neal_01"
    assert completed["data"]["state"]["status"] == "success"
    final_image = completed["data"]["observation"]
    assert (
        hashlib.sha256((attempt / final_image["path"]).read_bytes()).hexdigest()
        == final_image["sha256"]
    )
    assert completed["sequence"] < starts[1]["sequence"]
    assert all(
        event["data"]["args"].get("text") != "UNEXECUTED_CAMPAIGN_TAIL"
        for event in events
        if event["type"] == "input_started"
    )
    for task in ("neal_01", "neal_02"):
        assert any(
            event["type"] == "game_event" and event["data"]["task_id"] == task for event in events
        )
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["outcome"] == "aborted"
    assert manifest["recording_complete"]
    final_event = next(event for event in events if event["type"] == "attempt_end")
    assert final_event["data"]["cleanup_complete"] and not final_event["data"]["cleanup_errors"]
    assert not owned[4]._thread.is_alive() and resource_id not in LIVE_ENVS


@pytest.mark.live
async def test_page_timer_completion_advances_on_observation_only(campaign_env):
    env = await campaign_env()
    raw, page = env.unwrapped, env.unwrapped._page
    await click(env, page.get_by_role("checkbox"))
    # The page's genuine checkbox timer completes while no controller tool runs.
    await page.locator('.neal-game[data-status="success"]').wait_for()
    result = await gui(env, [{"action": "screenshot"}])
    assert raw._completed_tasks == ["neal_01"] and raw.state.task_id == "neal_02"
    assert raw._steps == 2 and result.info["executed_actions"] == []
    assert not result.terminated and not result.truncated and result.reward is None


@pytest.mark.live
async def test_transition_releases_held_keys_and_buttons_on_old_page(campaign_env):
    env = await campaign_env()
    raw, page = env.unwrapped, env.unwrapped._page
    releases = []
    await page.expose_function("recordCampaignRelease", lambda event: releases.append(event))
    await page.evaluate("""() => {
      const kinds = ['keyup', 'mouseup', 'mousedown'];
      for (const kind of kinds) document.addEventListener(kind, event => {
        window.recordCampaignRelease({kind, key: event.key ?? null,
          button: event.button ?? null, buttons: event.buttons ?? null,
          trusted: event.isTrusted, url: location.href});
      });
    }""")
    await gui(
        env,
        [
            {"action": "key_down", "keys": ["shift"]},
            {"action": "mouse_down", "button": "right", "coordinate": [950, 950]},
        ],
    )
    assert raw._pressed_keys == {"Shift"} and raw._pressed_buttons == {"right"}
    result = await gui(
        env,
        [
            {"action": "click", "coordinate": await center(env, page.get_by_role("checkbox"))},
            # Disabled controls suppress mouse events after success. Observe the
            # eventual release over ordinary page background instead.
            {"action": "mouse_move", "coordinate": [950, 950]},
            {"action": "wait", "duration": 0.8},
        ],
    )
    assert not result.terminated and raw.state.task_id == "neal_02"
    assert not raw._pressed_keys and not raw._pressed_buttons
    assert any(
        event["kind"] == "mouseup" and event["button"] == 0 and event["buttons"] == 2
        for event in releases
    )
    assert any(
        event["kind"] == "keyup"
        and event["key"] == "Shift"
        and event["trusted"]
        and "task=neal_01" in event["url"]
        for event in releases
    )
    assert any(
        event["kind"] == "mouseup"
        and event["button"] == 2
        and event["trusted"]
        and "task=neal_01" in event["url"]
        for event in releases
    ), releases
    events = [
        json.loads(line) for line in (raw.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    next_stage = next(
        event
        for event in events
        if event["type"] == "campaign_stage_started" and event["data"]["task_id"] == "neal_02"
    )
    for primitive in ("mouse.up", "keyboard.up"):
        release = next(
            event
            for event in events
            if event["type"] == "input_started"
            and event["data"]["call"] == primitive
            and event["data"]["tool_call_id"] is None
        )
        assert release["sequence"] < next_stage["sequence"]


@pytest.mark.live
async def test_real_first_fourteen_stages_stop_at_actual_missing_fifteen(campaign_env):
    """Exercise the implemented prefix, not a model or original-site benchmark."""
    prefix = [f"neal_{level:02d}" for level in range(1, 15)]
    assert all(task in env_module.LOCAL_TASKS for task in prefix)
    assert "neal_15" not in env_module.LOCAL_TASKS, "Extend this oracle when level 15 is built"
    env = await campaign_env(max_steps=300, max_seconds=180, seed=0)
    raw, page = env.unwrapped, env.unwrapped._page
    owned = (raw._browser, raw._context, page, raw.recorder, raw._local_server)
    identity = (raw.attempt_dir, raw.external_resource_id, raw._started)
    previous_steps = 0

    for level, task_id in enumerate(prefix, start=1):
        assert raw.state.task_id == task_id and raw.state.status == "in_progress"
        assert await page.evaluate("window.scrollY") == 0
        verify = page.get_by_role("button", name="Verify", exact=True)
        tiles = page.locator(".neal-grid .neal-tile")
        if level == 1:
            result = await gui(
                env,
                [
                    {
                        "action": "click",
                        "coordinate": await center(env, page.get_by_role("checkbox")),
                    },
                    {"action": "wait", "duration": 0.8},
                ],
            )
        elif task_id in ACCEPTED_SELECTIONS or task_id in GRIDS:
            accepted = (
                ACCEPTED_SELECTIONS[task_id]
                if task_id in ACCEPTED_SELECTIONS
                else GRIDS[task_id][2]
            )
            for index in accepted:
                await click(env, tiles.nth(index))
            result = await click(env, verify)
        elif level in (3, 8):
            await click(env, page.locator("#neal-answer"))
            result = await gui(
                env,
                [
                    {
                        "action": "type",
                        "text": "YHRPCD" if level == 3 else "867V 309",
                        "press_enter": True,
                    }
                ],
            )
        elif level == 5:
            for index in [0, 1, 2, 2, 5, 6, 6, 8, 8, 0, 0, 2, 2, 6, 6, 8, 8, 1, 1, 5, 5]:
                await click(env, tiles.nth(index))
            await gui(env, [{"action": "wait", "duration": 0.2}])
            result = await click(env, verify)
        elif level == 6:
            # A visible-board test oracle, not model or original-site evidence.
            result = await win_tic_tac_toe(env)
        elif level == 9:
            photo = await page.locator(".neal-recursive-board").bounding_box()
            leaves, actions = [(0, 0, 16)], []
            width, height = raw.display_resolution
            for row, columns in RECURSIVE_ROWS.items():
                for column in columns:
                    coordinate = [
                        (photo["x"] + (column + 0.5) / 16 * photo["width"]) * 1000 / width,
                        (photo["y"] + (row + 0.5) / 16 * photo["height"]) * 1000 / height,
                    ]
                    while True:
                        leaf = next(
                            (x, y, size)
                            for x, y, size in leaves
                            if x <= column < x + size and y <= row < y + size
                        )
                        actions.append({"action": "click", "coordinate": coordinate})
                        if leaf[2] == 1:
                            break
                        leaves.remove(leaf)
                        x, y, size = leaf
                        half = size // 2
                        leaves.extend(
                            [
                                (x, y, half),
                                (x + half, y, half),
                                (x, y + half, half),
                                (x + half, y + half, half),
                            ]
                        )
            for offset in range(0, len(actions), 8):
                await gui(env, actions[offset : offset + 8])
            result = await click(env, verify)
        elif level == 10:
            for _ in range(20):
                visible = page.locator('.neal-tile.mole-visible[aria-pressed="false"]').first
                await visible.wait_for(state="visible", timeout=5000)
                await click(env, visible)
                if raw.state.progress == 5:
                    break
            assert raw.state.progress == 5
            result = await click(env, verify)
        else:
            assert level == 14
            await gui(
                env,
                [
                    {
                        "action": "scroll",
                        "direction": "down",
                        "amount": 26,
                        "coordinate": [200, 300],
                    },
                    {"action": "wait", "duration": 0.15},
                ],
            )
            target = page.get_by_role("checkbox", name="I'm not a robot", exact=True)
            # Completion may happen during the click's screenshot. One batch
            # lets the environment discard the wait if it is already terminal.
            result = await gui(
                env,
                [
                    {"action": "click", "coordinate": await center(env, target)},
                    {"action": "wait", "duration": 0.8},
                ],
            )
        assert raw._completed_tasks == prefix[:level], (task_id, raw.state)
        assert (raw._browser, raw._context, raw._page, raw.recorder, raw._local_server) == owned
        assert (raw.attempt_dir, raw.external_resource_id, raw._started) == identity
        assert raw._steps > previous_steps
        previous_steps = raw._steps
        if level < 14:
            assert not result.terminated and not result.truncated and result.reward is None

    assert result.truncated and not result.terminated and result.reward == 0
    assert raw.outcome == "unsupported_task" and raw.state.task_id == "neal_14"
    assert raw.state.status == "success" and "task=neal_14" in page.url
    assert result.info["campaign"]["next_task"] == "neal_15"
    assert result.info["campaign"]["total_tasks"] == 48
    with pytest.raises(RuntimeError, match="not active"):
        await gui(env, [{"action": "screenshot"}])
    await env.close()
    archive = identity[0]
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["outcome"] == "unsupported_task"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    payload = (archive / "events.jsonl").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["events"]["sha256"]
    events = [json.loads(line) for line in payload.splitlines()]
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert [e["elapsed_seconds"] for e in events] == sorted(e["elapsed_seconds"] for e in events)
    assert sum(e["type"] == "browser_started" for e in events) == 1
    starts = [e for e in events if e["type"] == "campaign_stage_started"]
    ends = [e for e in events if e["type"] == "campaign_stage_completed"]
    assert [e["data"]["task_id"] for e in starts] == prefix
    assert [e["data"]["state"]["task_id"] for e in ends] == prefix
    assert all(e["data"]["state"]["status"] == "success" for e in ends)
    for index, completed in enumerate(ends):
        assert starts[index]["sequence"] < completed["sequence"]
        if index + 1 < len(starts):
            assert completed["sequence"] < starts[index + 1]["sequence"]
    for image in manifest["images"]:
        png = (archive / image["path"]).read_bytes()
        assert len(png) == image["bytes"] and hashlib.sha256(png).hexdigest() == image["sha256"]
    assert not any(e["type"] in {"error", "page_error", "request_blocked"} for e in events)
    assert not owned[4]._thread.is_alive() and identity[1] not in LIVE_ENVS


@pytest.mark.live
async def test_injected_missing_next_task_stops_without_skipping(campaign_env, monkeypatch):
    env = await campaign_env()
    raw, page = env.unwrapped, env.unwrapped._page
    # Inject the next-task absence; this is not an actual traversal to level 15.
    available = dict(env_module.LOCAL_TASKS)
    del available["neal_02"]
    monkeypatch.setattr(env_module, "LOCAL_TASKS", available)
    old_url = page.url
    result = await gui(
        env,
        [
            {"action": "click", "coordinate": await center(env, page.get_by_role("checkbox"))},
            {"action": "wait", "duration": 0.8},
        ],
    )
    assert result.truncated and not result.terminated and result.reward == 0
    assert raw.outcome == "unsupported_task" and page.url == old_url
    assert raw.state.task_id == "neal_01" and raw.state.status == "success"
    assert raw._completed_tasks == ["neal_01"]
    assert result.info["campaign"]["next_task"] == "neal_02"
    assert result.info["campaign"]["displayed_task"] == "neal_01"
    with pytest.raises(RuntimeError, match="not active"):
        await gui(env, [{"action": "screenshot"}])
    await env.close()
    manifest = json.loads((raw.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "unsupported_task" and manifest["recording_complete"]


@pytest.mark.live
async def test_finish_after_page_completion_stops_partial_campaign_without_navigation(campaign_env):
    env = await campaign_env()
    raw, page = env.unwrapped, env.unwrapped._page
    old_url = page.url
    await click(env, page.get_by_role("checkbox"))
    await page.locator('.neal-game[data-status="success"]').wait_for()
    result = await env.step(
        [make_tool_call("terminate", {"status": "success"}, call_id="partial_finish_claim")]
    )
    assert result.terminated and not result.truncated and result.reward == 0
    assert raw.outcome == "agent_stopped" and page.url == old_url
    assert raw._completed_tasks == ["neal_01"] and raw.state.task_id == "neal_01"


@pytest.mark.live
async def test_step_budget_at_boundary_does_not_start_next_stage(campaign_env):
    env = await campaign_env(max_steps=1)
    raw, page = env.unwrapped, env.unwrapped._page
    old_url = page.url
    result = await gui(
        env,
        [
            {"action": "click", "coordinate": await center(env, page.get_by_role("checkbox"))},
            {"action": "wait", "duration": 0.8},
        ],
    )
    assert result.truncated and not result.terminated and result.reward == 0
    assert raw.outcome == "budget_exhausted" and page.url == old_url
    assert raw._completed_tasks == ["neal_01"] and raw._steps == 1


@pytest.mark.live
async def test_reset_starts_fresh_campaign_and_finalizes_previous_archive(campaign_env):
    env = await campaign_env(seed=18)
    raw, page = env.unwrapped, env.unwrapped._page
    first_attempt, first_resource = raw.attempt_dir, raw.external_resource_id
    first_browser, first_server = raw._browser, raw._local_server
    await gui(
        env,
        [
            {"action": "click", "coordinate": await center(env, page.get_by_role("checkbox"))},
            {"action": "wait", "duration": 0.8},
        ],
    )
    assert raw._completed_tasks == ["neal_01"] and raw.state.task_id == "neal_02"
    await env.reset()
    assert raw.attempt_dir != first_attempt and raw.external_resource_id != first_resource
    assert raw._browser is not first_browser and not first_browser.is_connected()
    assert not first_server._thread.is_alive() and first_resource not in LIVE_ENVS
    assert raw.state.task_id == "neal_01" and raw.state.status == "in_progress"
    assert raw.state.seed == 18 and raw._steps == 0 and raw._completed_tasks == []
    manifest = json.loads((first_attempt / "manifest.json").read_text())
    assert manifest["outcome"] == "aborted" and manifest["recording_complete"]


@pytest.mark.live
async def test_navigation_failure_retains_final_stage_and_infrastructure_outcome(
    campaign_env, monkeypatch
):
    env = await campaign_env()
    raw, page = env.unwrapped, env.unwrapped._page
    await click(env, page.get_by_role("checkbox"))
    await page.locator('.neal-game[data-status="success"]').wait_for()
    monkeypatch.setattr(
        page, "goto", AsyncMock(side_effect=RuntimeError("injected navigation failure"))
    )
    with pytest.raises(RuntimeError, match="injected navigation failure"):
        await gui(env, [{"action": "screenshot"}])
    assert raw.outcome == "infra_error" and raw._terminal
    assert raw._completed_tasks == ["neal_01"]
    await env.close()
    events = [
        json.loads(line) for line in (raw.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    assert any(event["type"] == "campaign_stage_completed" for event in events)
    assert any(
        event["type"] == "error" and event["data"]["phase"] == "campaign_transition"
        for event in events
    )
    manifest = json.loads((raw.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "infra_error" and manifest["recording_complete"]


async def test_mocked_final_stage_awards_success_only_after_48th_completion(tmp_path):
    """Pure controller boundary fixture, not a real level-48 game or model result."""
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
    )
    env = NotARobotEnv(mode="local", target_level=None, local_campaign=True, cursor=False)
    env._completed_tasks = list(LOCAL_CAMPAIGN_TASKS[:-1])
    snapshot = {
        "task_id": "neal_48",
        "label": "Mocked final controller boundary",
        "version": CATALOG["version"],
        "seed": 0,
        "status": "success",
        "mistakes": 0,
        "progress": 1,
        "reason": "unit_fixture",
        "elapsed_ms": 10,
    }
    env._page = SimpleNamespace(
        url="http://127.0.0.1/unit-fixture",
        evaluate=AsyncMock(side_effect=lambda *_: {**snapshot, "events": []}),
        screenshot=AsyncMock(return_value=png),
    )
    env.outcome = "in_progress"
    env._started = time.monotonic()
    env.attempt_dir = tmp_path / "mock-final-boundary"
    env.recorder = EventRecorder(env.attempt_dir, {"unit_fixture": True, "real_gameplay": False})
    try:
        result = await gui(env, [{"action": "screenshot"}])
        assert result.terminated and not result.truncated and result.reward == 1
        assert env.outcome == "success" and env._completed_tasks == list(LOCAL_CAMPAIGN_TASKS)
        assert result.info["executed_actions"] == []
    finally:
        await env.close()


@pytest.mark.parametrize(
    "phase",
    ["navigation", "evaluate", "screenshot", "completed_screenshot", "stage_end_screenshot"],
)
async def test_mocked_controller_deadline_is_not_infrastructure_failure(tmp_path, phase):
    """Inject pending browser I/O; this is controller lifecycle, not game traversal."""
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
    )
    env = NotARobotEnv(
        mode="local", target_level=None, local_campaign=True, cursor=False, post_action_delay=0
    )
    env._completed_tasks = list(LOCAL_CAMPAIGN_TASKS[:3])
    completed_page = phase in ("navigation", "completed_screenshot", "stage_end_screenshot")
    snapshot = {
        "task_id": "neal_04",
        "label": "Mocked controller cancellation boundary",
        "version": CATALOG["version"],
        "seed": 0,
        "status": "success" if completed_page else "in_progress",
        "mistakes": 0,
        "progress": 1 if completed_page else 0,
        "reason": "unit_fixture",
        "elapsed_ms": 10,
    }
    env.state = LocalTaskState(**snapshot)
    env._page = SimpleNamespace(
        url="http://127.0.0.1/unit-fixture",
        evaluate=AsyncMock(side_effect=lambda *_: {**snapshot, "events": []}),
        screenshot=AsyncMock(return_value=png),
    )
    env.outcome = "in_progress"
    env._started = time.monotonic()
    env.attempt_dir = tmp_path / f"mock-{phase}-deadline"
    env.recorder = EventRecorder(env.attempt_dir, {"unit_fixture": True, "real_gameplay": False})
    timeout = asyncio.timeout(None)

    async def pending_browser_io(*args, **kwargs):
        # Start the real timeout only after reaching the intended await boundary.
        # This avoids cancelling earlier synchronous recorder writes on slow I/O.
        timeout.reschedule(asyncio.get_running_loop().time() + 0.01)
        await asyncio.sleep(10)

    if phase == "navigation":
        env._open_local_task = pending_browser_io
    elif phase == "stage_end_screenshot":
        captures = 0

        async def capture_until_stage_end(**kwargs):
            nonlocal captures
            captures += 1
            if captures == 1:
                return png
            await pending_browser_io()

        env._page.screenshot = AsyncMock(side_effect=capture_until_stage_end)
    else:
        browser_method = "screenshot" if phase == "completed_screenshot" else phase
        setattr(env._page, browser_method, AsyncMock(side_effect=pending_browser_io))
    bridge = CodexBridge(
        SimpleNamespace(max_seconds=180, model="unit_fixture_model", reasoning_effort="xhigh")
    )
    bridge.env = env
    try:
        with pytest.raises(TimeoutError):
            async with timeout:
                await gui(env, [{"action": "screenshot"}])
        assert env.outcome == "in_progress" and env._terminal
        expected_prefix = list(LOCAL_CAMPAIGN_TASKS[: 4 if phase == "navigation" else 3])
        assert env._completed_tasks == expected_prefix
        with pytest.raises(RuntimeError, match="not active"):
            await gui(env, [{"action": "screenshot"}])
        # Use the real bridge classification/finalizer, not a fabricated manifest.
        await bridge.close("controller_timeout")
        assert env.outcome == "controller_timeout" and bridge.closed
        manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
        assert manifest["outcome"] == "controller_timeout" and manifest["recording_complete"]
        assert manifest["data"]["cleanup_complete"]
        events = [
            json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
        ]
        assert not any(event["type"] == "error" for event in events)
        campaign_end = next(event for event in events if event["type"] == "campaign_end")
        assert campaign_end["data"]["completed_tasks"] == expected_prefix
        assert campaign_end["data"]["outcome"] == "controller_timeout"
    finally:
        await bridge.close("test_cleanup")
