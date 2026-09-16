"""L18 replacement semantics via real canonical GUI input and real timers.

These scripted visual-control oracles are not model accuracy measurements.
"""

from __future__ import annotations

import json

import pytest

from examples.not_a_robot.tests.test_local_tasks import center, click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env


async def visible_frames(page):
    """Read which private source image the visible tile faces currently display."""
    return await page.locator(".neal-replacement-grid .neal-tile-face").evaluate_all(
        "faces => faces.map(face => "
        "Number(face.style.backgroundImage.match(/state(\\d+)\\.jpg/)[1]))"
    )


@pytest.mark.live
async def test_replacements_pending_verify_and_duplicate_click_do_not_advance_twice(local_env):
    env = await local_env("neal_18", max_steps=100)
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-replacement-grid .neal-tile")
    verify = page.get_by_role("button", name="Verify", exact=True)
    assert await visible_frames(page) == [0] * 9
    result = await gui(
        env,
        [
            {"action": "click", "coordinate": await center(env, tiles.nth(1)), "clicks": 2},
            {"action": "click", "coordinate": await center(env, verify)},
        ],
    )
    assert not result.terminated and raw.state.mistakes == 1
    events = (await page.evaluate("window.syntheticTask.snapshot()"))["events"]
    assert any(
        e["kind"] == "ignored_click" and e["reason"] == "replacement_pending" for e in events
    )
    assert any(e["kind"] == "rejected" and e["reason"] == "replacement_pending" for e in events)
    await gui(env, [{"action": "wait", "duration": 0.8}])
    assert await visible_frames(page) == [0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert raw.state.progress == 1
    assert await tiles.nth(1).get_attribute("aria-pressed") == "false"
    # More than thirty background clicks still leave every other queue unchanged.
    coordinate = await center(env, tiles.nth(0))
    for _ in range(11):
        await gui(env, [{"action": "click", "coordinate": coordinate, "clicks": 3}])
    assert await tiles.nth(0).get_attribute("aria-pressed") == "true"
    result = await click(env, verify)
    assert not result.terminated and raw.state.progress == 1
    events = (await page.evaluate("window.syntheticTask.snapshot()"))["events"]
    assert events[-1]["kind"] == "rejected" and events[-1]["reason"] == "distractor_selected"
    await click(env, tiles.nth(0))
    result = await click(env, verify)
    assert not result.terminated
    assert await visible_frames(page) == [0, 1, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.live
async def test_replacements_independent_reordered_queues_require_clean_final_board(local_env):
    env = await local_env("neal_18", max_steps=200)
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-replacement-grid .neal-tile")
    verify = page.get_by_role("button", name="Verify", exact=True)
    await page.screenshot(path=str(raw.attempt_dir / "initial.png"))
    await click(env, tiles.nth(0))  # Keep a selected distractor throughout the replacements.
    frames = [0] * 9
    # Complete whole cell queues in reverse order, unlike the captured row rounds.
    # A global click counter or slideshow cannot satisfy each per-cell assertion.
    for index, final_frame in [(8, 2), (7, 7), (6, 8), (3, 5), (1, 8)]:
        for frame in range(1, final_frame + 1):
            result = await click(env, tiles.nth(index))
            assert not result.terminated
            await gui(env, [{"action": "wait", "duration": 0.7}])
            frames[index] = frame
            assert await visible_frames(page) == frames
            assert raw.state.progress == sum(frames)
        assert await tiles.nth(index).get_attribute("aria-pressed") == "false"
    assert frames == [0, 8, 0, 5, 0, 0, 8, 7, 2]
    result = await click(env, verify)
    assert not result.terminated and raw.state.mistakes == 1
    await click(env, tiles.nth(0))
    result = await click(env, verify)
    assert result.terminated and not result.truncated and result.reward == 1
    assert raw.state.reason == "no_hydrants_or_pending_replacements"
    await page.screenshot(path=str(raw.attempt_dir / "success.png"))
    frozen = await page.evaluate("window.syntheticTask.snapshot()")
    await page.wait_for_timeout(750)
    assert await page.evaluate("window.syntheticTask.snapshot()") == frozen
    attempt = raw.attempt_dir
    await env.close()
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["outcome"] == "success" and manifest["recording_complete"]
    assert manifest["data"]["cleanup_complete"]
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    replaced = [
        e["data"]
        for e in events
        if e["type"] == "game_event" and e["data"]["kind"] == "image_replaced"
    ]
    assert len(replaced) == 30
    assert not any(e["type"] == "page_error" for e in events)


@pytest.mark.live
async def test_replacements_refresh_cancels_pending_images_and_reset_is_independent(local_env):
    env = await local_env("neal_18")
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-replacement-grid .neal-tile")
    refresh = page.get_by_role("button", name="Refresh challenge")
    initial = await page.locator(".neal-game").inner_html()
    await gui(
        env,
        [
            {"action": "click", "coordinate": await center(env, tiles.nth(1))},
            {
                "action": "click",
                "coordinate": await center(env, refresh),
            },
            {"action": "wait", "duration": 0.8},
        ],
    )
    assert await visible_frames(page) == [0] * 9
    assert raw.state.progress == 0
    assert await page.locator(".neal-game").inner_html() == initial
    events = (await page.evaluate("window.syntheticTask.snapshot()"))["events"]
    assert any(e["kind"] == "replacement_reset" and e["canceled_replacements"] == 1 for e in events)
    assert not any(e["kind"] == "image_replaced" for e in events)
    await click(env, tiles.nth(3))
    await gui(env, [{"action": "wait", "duration": 0.8}])
    assert await visible_frames(page) == [0, 0, 0, 1, 0, 0, 0, 0, 0]
    first_attempt = raw.attempt_dir
    await env.reset()
    assert raw.attempt_dir != first_attempt
    assert raw.state.progress == raw.state.mistakes == 0
    assert await visible_frames(raw._page) == [0] * 9
    assert await raw._page.locator(".neal-game").inner_html() == initial


@pytest.mark.live
async def test_replacements_close_releases_pending_page_and_owned_resources(local_env):
    env = await local_env("neal_18")
    raw, page = env.unwrapped, env.unwrapped._page
    await click(env, page.locator(".neal-replacement-grid .neal-tile").nth(1))
    attempt = raw.attempt_dir
    await env.close()
    assert page.is_closed()
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    assert manifest["outcome"] != "success"
