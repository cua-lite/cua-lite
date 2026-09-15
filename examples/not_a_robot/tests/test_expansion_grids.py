"""Captured grid instances through canonical GUI actions, not a model benchmark."""

from __future__ import annotations

import json

import pytest

from examples.not_a_robot.tests.test_local_tasks import click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env

GRIDS = {
    "neal_11": (25, 25, [218, 243, 244]),
    "neal_12": (4, 4, [2, 3, 7, 10, 11, 15]),
    "neal_13": (4, 4, [0, 3, 4, 7, 8, 11, 12, 13, 14, 15]),
    "neal_29": (3, 3, [0, 2, 5, 7]),
    "neal_31": (4, 4, [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    "neal_46": (14, 57, [409, 410, 411, 412, 413, 414, 415, 416]),
}


async def scroll_click(env, locator):
    """Bring the visible test oracle into view using recorded wheel actions."""
    _, viewport_height = env.unwrapped.display_resolution
    for _ in range(10):
        box = await locator.bounding_box()
        center_y = box["y"] + box["height"] / 2
        if 0 <= box["y"] and box["y"] + box["height"] <= viewport_height:
            return await click(env, locator)
        await gui(
            env,
            [
                {
                    "action": "scroll",
                    "direction": "down" if center_y > viewport_height / 2 else "up",
                    "amount": min(10, max(1, abs(center_y - viewport_height / 2) / 100)),
                    "coordinate": [200, 500],
                },
                {"action": "wait", "duration": 0.05},
            ],
        )
    raise AssertionError("The grid control did not enter the viewport after real scrolling")


@pytest.mark.live
@pytest.mark.parametrize("task_id", GRIDS)
async def test_expansion_grid_reset_and_same_instance_refresh(local_env, task_id):
    env = await local_env(task_id, max_steps=100)
    raw, page = env.unwrapped, env.unwrapped._page
    columns, rows, accepted = GRIDS[task_id]
    tiles = page.locator(".neal-reference-selection .neal-tile")
    assert await tiles.count() == columns * rows
    assert await page.locator(".neal-game").get_attribute("data-level") == task_id[-2:]
    initial = await page.locator(".neal-game").inner_html()
    await scroll_click(env, tiles.nth(accepted[0]))
    assert raw.state.progress == 1
    await scroll_click(env, page.get_by_role("button", name="Refresh challenge", exact=True))
    assert raw.state.status == "in_progress" and raw.state.progress == 0
    assert (
        await page.locator('.neal-reference-selection .neal-tile[aria-pressed="true"]').count() == 0
    )
    assert await page.locator(".neal-game").inner_html() == initial
    first_attempt = raw.attempt_dir
    await env.reset()
    assert raw.attempt_dir != first_attempt
    assert raw.state.progress == raw.state.mistakes == 0
    assert await raw._page.locator(".neal-game").inner_html() == initial
    events = [
        json.loads(line) for line in (first_attempt / "events.jsonl").read_text().splitlines()
    ]
    assert not any(row["type"] == "page_error" for row in events)


@pytest.mark.live
@pytest.mark.parametrize("task_id", GRIDS)
async def test_expansion_grid_accepts_only_captured_selection(local_env, task_id):
    env = await local_env(task_id, max_steps=200)
    raw, page = env.unwrapped, env.unwrapped._page
    _, _, accepted = GRIDS[task_id]
    tiles = page.locator(".neal-reference-selection .neal-tile")
    verify = page.get_by_role("button", name="Verify", exact=True)
    result = await scroll_click(env, verify)
    assert not result.terminated and raw.state.mistakes == 1
    for index in accepted[:-1]:
        await scroll_click(env, tiles.nth(index))
    result = await scroll_click(env, verify)
    assert not result.terminated and raw.state.mistakes == 2
    await scroll_click(env, tiles.nth(accepted[-1]))
    extra = next(index for index in range(await tiles.count()) if index not in accepted)
    await scroll_click(env, tiles.nth(extra))
    result = await scroll_click(env, verify)
    assert not result.terminated and raw.state.mistakes == 3
    await scroll_click(env, tiles.nth(extra))
    assert raw.state.progress == len(accepted)
    result = await scroll_click(env, verify)
    assert result.terminated and not result.truncated and result.reward == 1
    assert raw.state.status == raw.outcome == "success"
    await page.screenshot(path=str(raw.attempt_dir / "success.png"))
    frozen = await page.evaluate("window.syntheticTask.snapshot()")
    await page.mouse.click(100, 200)
    assert await page.evaluate("window.syntheticTask.snapshot()") == frozen
    attempt = raw.attempt_dir
    await env.close()
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["outcome"] == "success"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    assert not any(row["type"] == "page_error" for row in events)
    if task_id == "neal_46":
        assert any(
            row["type"] == "input_started" and row["data"].get("call") == "mouse.wheel"
            for row in events
        )
