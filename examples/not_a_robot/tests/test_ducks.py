"""Canonical-GUI tests of real-time duck movement and distinct-entity capture.

Visible geometry is a scripted test oracle, not a screenshot-model result.
Tests never freeze, teleport, force-click, or directly change game state.
"""

from __future__ import annotations

import json
import math

import pytest

from examples.not_a_robot.tests.test_local_tasks import gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env


async def visible_control_point(env, locator):
    """Find an actually unobscured point; inputs still use browser hit testing."""
    width, height = env.unwrapped.display_resolution
    for _ in range(20):
        point = await locator.evaluate_all(
            """nodes => {
                for (const node of nodes) {
                    const r = node.getBoundingClientRect();
                    for (const [fx, fy] of [[.5,.5],[.35,.35],[.65,.35],[.35,.65],[.65,.65]]) {
                        const x = r.x + r.width * fx, y = r.y + r.height * fy;
                        const top = document.elementFromPoint(x, y);
                        if (top && node.contains(top)) return [x, y];
                    }
                }
                return null;
            }"""
        )
        if point is not None:
            return [point[0] * 1000 / width, point[1] * 1000 / height]
        await gui(env, [{"action": "wait", "duration": 0.05}])
    raise AssertionError("No unobscured visible control point during the bounded check")


async def duck_layout(page):
    return await page.locator(".neal-duck").evaluate_all(
        """nodes => nodes.map(node => ({id: Number(node.dataset.duck), phase: node.dataset.phase,
            x: parseFloat(node.style.left), y: parseFloat(node.style.top),
            frame: node.firstElementChild.style.backgroundPosition,
            facing: node.firstElementChild.style.transform}))"""
    )


@pytest.mark.live
async def test_ducks_keep_moving_in_real_time_and_background_inputs_do_not_capture(local_env):
    env = await local_env("neal_22", max_steps=60)
    raw, page = env.unwrapped, env.unwrapped._page
    field = await page.locator(".neal-duck-field").bounding_box()
    card = await page.locator(".neal-game").bounding_box()
    assert field["width"] == 600 and field["height"] == 590
    assert card["width"] == 460
    assert field["x"] + field["width"] > card["x"] + card["width"]
    assert await page.locator(".neal-duck").count() == 9
    before = await duck_layout(page)
    elapsed = raw.state.elapsed_ms
    # Each click is outside the motion field; nine arbitrary inputs cannot clear nine ducks.
    await gui(env, [{"action": "click", "coordinate": [900, 900]}] * 9)
    await gui(env, [{"action": "key", "keys": ["left"]}, {"action": "wait", "duration": 0.3}])
    after = await duck_layout(page)
    assert raw.state.elapsed_ms > elapsed + 250
    assert raw.state.progress == 0 and all(row["phase"] == "roaming" for row in after)
    assert any(math.hypot(a["x"] - b["x"], a["y"] - b["y"]) > 3 for a, b in zip(after, before))
    assert all(0 <= row["x"] <= 500 and 0 <= row["y"] <= 490 for row in after)
    assert (
        await page.locator(".neal-duck-sprite").first.evaluate(
            "node => getComputedStyle(node).backgroundSize"
        )
        == "300% 100%"
    )
    await gui(env, [{"action": "mouse_move", "coordinate": [100, 100]}])
    net = page.locator(".neal-duck-net")
    assert await net.is_visible()
    assert await net.evaluate("node => getComputedStyle(node).pointerEvents") == "none"
    verify = page.get_by_role("button", name="Verify", exact=True)
    result = await gui(
        env, [{"action": "click", "coordinate": await visible_control_point(env, verify)}]
    )
    assert not result.terminated and raw.state.status == "in_progress"
    assert raw.state.mistakes == 1


@pytest.mark.live
async def test_ducks_catch_each_entity_once_and_require_completed_return_then_verify(local_env):
    env = await local_env("neal_22", max_steps=150)
    raw, page = env.unwrapped, env.unwrapped._page
    roaming = page.locator('.neal-duck[data-phase="roaming"]')
    coordinate = await visible_control_point(env, roaming)
    await gui(env, [{"action": "click", "coordinate": coordinate, "clicks": 3}])
    await gui(env, [{"action": "wait", "duration": 0.4}])
    first = await page.evaluate("window.syntheticTask.snapshot()")
    started = [event for event in first["events"] if event["kind"] == "duck_return_started"]
    assert len(started) == 1
    assert raw.state.progress == 1
    assert len({event["duck"] for event in started}) == 1
    for _ in range(40):
        if await roaming.count() == 0:
            break
        await gui(
            env, [{"action": "click", "coordinate": await visible_control_point(env, roaming)}]
        )
    assert await roaming.count() == 0
    await gui(env, [{"action": "wait", "duration": 0.4}])
    assert raw.state.progress == 9 and raw.state.status == "in_progress"
    final = await page.evaluate("window.syntheticTask.snapshot()")
    caught = [event for event in final["events"] if event["kind"] == "duck_caught"]
    assert len(caught) == 9 and {event["duck"] for event in caught} == set(range(9))
    layout = await duck_layout(page)
    assert all(row["phase"] == "caught" and row["frame"] == "0% 50%" for row in layout)
    assert all(row["facing"] == "none" for row in layout)
    grid = await page.locator(".neal-duck-grid").bounding_box()
    field = await page.locator(".neal-duck-field").bounding_box()
    for row in layout:
        expected_x = grid["x"] - field["x"] + (row["id"] % 3 + 0.5) * grid["width"] / 3 - 50
        expected_y = grid["y"] - field["y"] + (row["id"] // 3 + 0.5) * grid["height"] / 3 - 50
        assert row["x"] == pytest.approx(expected_x, abs=0.01)
        assert row["y"] == pytest.approx(expected_y, abs=0.01)
    # Repeated clicks on an already caught entity never create extra progress.
    coordinate = await visible_control_point(env, page.locator(".neal-duck").first)
    await gui(env, [{"action": "click", "coordinate": coordinate}] * 9)
    assert raw.state.progress == 9 and raw.state.status == "in_progress"
    verify = page.get_by_role("button", name="Verify", exact=True)
    result = await gui(
        env, [{"action": "click", "coordinate": await visible_control_point(env, verify)}]
    )
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "all_distinct_ducks_caught"
    frozen = await page.evaluate("window.syntheticTask.snapshot()")
    await page.wait_for_timeout(250)
    assert await duck_layout(page) == layout
    assert await page.evaluate("window.syntheticTask.snapshot()") == frozen
    attempt = raw.attempt_dir
    await env.close()
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    assert manifest["outcome"] == "success"


@pytest.mark.live
async def test_ducks_refresh_cancels_old_returns_and_reuses_seeded_paths(local_env):
    env = await local_env("neal_22", max_steps=80)
    raw, page = env.unwrapped, env.unwrapped._page
    snapshot = await page.evaluate("window.syntheticTask.snapshot()")
    first_reset = next(event for event in snapshot["events"] if event["kind"] == "duck_field_reset")
    assert first_reset["field_px"] == [600, 590]
    assert first_reset["return_ms"] == 300 and first_reset["sprite_frame_ms"] == 160
    assert all(100 <= math.hypot(path["vx"], path["vy"]) <= 160 for path in first_reset["paths"])
    roaming = page.locator('.neal-duck[data-phase="roaming"]')
    catch_point = await visible_control_point(env, roaming)
    refresh = page.get_by_role("button", name="Refresh challenge", exact=True)
    refresh_point = await visible_control_point(env, refresh)
    await gui(
        env,
        [
            {"action": "click", "coordinate": catch_point},
            {"action": "click", "coordinate": refresh_point},
            {"action": "wait", "duration": 0.45},
        ],
    )
    snapshot = await page.evaluate("window.syntheticTask.snapshot()")
    resets = [event for event in snapshot["events"] if event["kind"] == "duck_field_reset"]
    assert len(resets) == 2
    assert resets[0]["paths"] == resets[1]["paths"]
    assert not any(
        event["kind"] == "duck_caught" and event["sequence"] > resets[-1]["sequence"]
        for event in snapshot["events"]
    )
    assert raw.state.progress == 0 and await roaming.count() == 9
    previous = raw.attempt_dir
    await env.reset()
    page = raw._page
    fresh = await page.evaluate("window.syntheticTask.snapshot()")
    fresh_reset = next(event for event in fresh["events"] if event["kind"] == "duck_field_reset")
    assert fresh_reset["paths"] == first_reset["paths"]
    assert raw.attempt_dir != previous
    assert raw.state.progress == 0 and raw.state.mistakes == 0
    await env.close()
    for attempt in [previous, raw.attempt_dir]:
        manifest = json.loads((attempt / "manifest.json").read_text())
        assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
        events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
        assert not any(event["type"] == "page_error" for event in events)
