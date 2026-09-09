"""Reference-instance GUI regression tests, separate from model gameplay.

These scripted oracles use the uploaded accepted inputs and visible control
geometry. They do not measure screenshot-only policy accuracy.
"""

from __future__ import annotations

import json

import pytest

from examples.not_a_robot.tests.test_local_tasks import click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env

REFERENCE_TASKS = tuple(f"neal_{level:02d}" for level in range(1, 11))
ACCEPTED_SELECTIONS = {
    "neal_02": [2, 3, 6, 7],
    "neal_04": [1, 2, 5],
    "neal_07": [87, 76, 65, 54, 43, 32, 21, 10, 95, 96, 97, 98],
}
RECURSIVE_ROWS = {
    4: [7, 8, 9, 10],
    5: [7, 8, 9, 10, 11],
    6: [6, 7, 8, 9, 10, 11],
    7: [6, 7, 8, 9, 10, 11],
    8: [6, 7, 8, 9, 10, 11],
    9: [7, 8, 9, 10],
}


@pytest.mark.live
@pytest.mark.parametrize("task_id", REFERENCE_TASKS)
async def test_reference_reset_is_game_only_and_independent(local_env, task_id):
    env = await local_env(task_id)
    raw = env.unwrapped
    page = raw._page
    assert await page.locator(".neal-game").is_visible()
    assert not await page.locator(".masthead").is_visible()
    assert not await page.locator("#reference-note").is_visible()
    assert await page.locator(".neal-game").get_attribute("data-level") == str(int(task_id[-2:]))
    if task_id in {"neal_02", "neal_04", "neal_05", "neal_06", "neal_07", "neal_10"}:
        boxes = await page.locator(".neal-grid .neal-tile").evaluate_all(
            "els => els.map(el => ({width: el.getBoundingClientRect().width, "
            "height: el.getBoundingClientRect().height}))"
        )
        assert all(abs(box["width"] - box["height"]) < 1 for box in boxes)
        assert max(box["height"] for box in boxes) - min(box["height"] for box in boxes) < 1
    if task_id == "neal_08":
        assert not await page.locator(".neal-refresh svg").is_visible()
    await page.screenshot(path=str(raw.attempt_dir / "initial.png"))
    first_attempt = raw.attempt_dir
    initial = await page.locator(".neal-game").inner_html()
    await env.reset()
    assert raw.attempt_dir != first_attempt
    assert raw.state.status == "in_progress"
    assert raw.state.progress == raw.state.mistakes == 0
    if task_id not in {"neal_03", "neal_10"}:
        assert await raw._page.locator(".neal-game").inner_html() == initial
    assert not raw._scope_violation
    events = [
        json.loads(line) for line in (first_attempt / "events.jsonl").read_text().splitlines()
    ]
    assert not any(row["type"] == "page_error" for row in events)


@pytest.mark.live
@pytest.mark.parametrize("task_id", [task for task in REFERENCE_TASKS if task != "neal_06"])
async def test_reference_success_through_canonical_gui(local_env, task_id):
    env = await local_env(task_id, max_steps=250)
    raw, page = env.unwrapped, env.unwrapped._page
    await page.screenshot(path=str(raw.attempt_dir / "initial.png"))
    if task_id == "neal_01":
        result = await click(env, page.get_by_text("I'm not a robot", exact=True))
        assert not result.terminated
        assert await page.locator(".neal-checkbox-mark").evaluate(
            "el => el.classList.contains('loading')"
        )
        result = await gui(env, [{"action": "wait", "duration": 0.8}])
        assert await page.get_by_role("checkbox").get_attribute("aria-checked") == "true"
    elif task_id in ACCEPTED_SELECTIONS:
        result = await click(env, page.get_by_role("button", name="Verify", exact=True))
        assert not result.terminated and raw.state.mistakes == 1
        tiles = page.locator(".neal-grid .neal-tile")
        expected = ACCEPTED_SELECTIONS[task_id]
        extra = next(index for index in range(await tiles.count()) if index not in expected)
        for index in [*expected, extra]:
            await click(env, tiles.nth(index))
        result = await click(env, page.get_by_role("button", name="Verify", exact=True))
        assert not result.terminated and raw.state.mistakes == 2
        await click(env, tiles.nth(extra))
        result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    elif task_id in {"neal_03", "neal_08"}:
        expected = "YHRPCD" if task_id == "neal_03" else "867V 309"
        await click(env, page.locator("#neal-answer"))
        result = await gui(env, [{"action": "type", "text": "WRONG", "press_enter": True}])
        assert not result.terminated and raw.state.mistakes == 1
        result = await gui(
            env,
            [
                {"action": "key", "keys": ["ctrl", "a"]},
                {"action": "type", "text": expected, "press_enter": True},
            ],
        )
    elif task_id == "neal_05":
        verify = page.get_by_role("button", name="Verify", exact=True)
        result = await click(env, verify)
        assert not result.terminated
        sequence = [0, 1, 2, 2, 5, 6, 6, 8, 8, 0, 0, 2, 2, 6, 6, 8, 8, 1, 1, 5, 5]
        for index in sequence:
            await click(env, page.locator(".neal-grid .neal-tile").nth(index))
        assert raw.state.progress == 21
        await gui(env, [{"action": "wait", "duration": 0.2}])
        result = await click(env, verify)
    elif task_id == "neal_09":
        photo = await page.locator(".neal-recursive-board").bounding_box()
        leaves = [(0, 0, 16)]
        actions = []
        width, height = raw.display_resolution
        for row, columns in RECURSIVE_ROWS.items():
            for column in columns:
                coordinate = [
                    (photo["x"] + (column + 0.5) / 16 * photo["width"]) * 1000 / width,
                    (photo["y"] + (row + 0.5) / 16 * photo["height"]) * 1000 / height,
                ]
                # Replay the reference quadtree through genuine mouse clicks.
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
        # Small action batches respect the same canonical ingress as model calls.
        for offset in range(0, len(actions), 8):
            await gui(env, actions[offset : offset + 8])
        assert raw.state.progress == 31
        assert await page.locator('.neal-recursive-leaf[aria-pressed="true"]').count() == 31
        result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    else:
        verify = page.get_by_role("button", name="Verify", exact=True)
        assert await verify.is_disabled()
        # The test oracle detects rendered moles; the Codex smoke has no DOM access.
        for _ in range(20):
            visible = page.locator('.neal-tile.mole-visible[aria-pressed="false"]').first
            await visible.wait_for(state="visible", timeout=5000)
            await click(env, visible)
            if raw.state.progress == 5:
                break
        assert raw.state.progress == 5
        assert await verify.is_enabled()
        sprite = await page.locator('.neal-tile[aria-pressed="true"] .neal-mole').first.evaluate(
            "el => ({size: getComputedStyle(el).backgroundSize, "
            "position: getComputedStyle(el).backgroundPosition})"
        )
        assert sprite == {"size": "200% 100%", "position": "100% 50%"}
        result = await click(env, verify)
    assert result.terminated and not result.truncated and result.reward == 1
    assert raw.state.status == "success" and raw.outcome == "success"
    await page.screenshot(path=str(raw.attempt_dir / "success.png"))
    frozen = await page.evaluate("window.syntheticTask.snapshot()")
    await page.mouse.click(100, 200)
    await page.wait_for_timeout(100)
    assert await page.evaluate("window.syntheticTask.snapshot()") == frozen
    attempt = raw.attempt_dir
    await env.close()
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["outcome"] == "success"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    assert any(row["type"] == "input_completed" for row in events)
    assert not any(row["type"] == "page_error" for row in events)


@pytest.mark.live
@pytest.mark.parametrize(
    "player,opponent", [([0, 6, 5, 1], [2, 3, 8, 7]), ([1, 8, 6, 3], [0, 2, 7, 5])]
)
async def test_tic_tac_toe_replays_both_observed_draws_without_false_success(
    local_env, player, opponent
):
    env = await local_env("neal_06")
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-tile")
    await click(env, tiles.nth(4))
    assert raw.state.progress == 0
    expected = [""] * 9
    expected[4] = "o"
    for x, o in zip(player, opponent, strict=True):
        await click(env, tiles.nth(x))
        await gui(env, [{"action": "wait", "duration": 0.45}])
        expected[x], expected[o] = "x", "o"
        visible = await tiles.locator(".neal-tile-face").evaluate_all(
            "els => els.map(el => el.classList.contains('neal-mark-x') ? 'x' : "
            "el.classList.contains('neal-mark-o') ? 'o' : '')"
        )
        assert visible == expected
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and result.reward is None
    assert raw.outcome == "in_progress" and raw.state.status == "in_progress"
    await page.screenshot(path=str(raw.attempt_dir / "draw.png"))
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert raw.state.progress == 0
    assert await page.locator(".neal-mark-o").count() == 1
    assert await page.locator(".neal-mark-x").count() == 0


@pytest.mark.live
async def test_recursive_terminal_depth_toggles_instead_of_subdividing(local_env):
    env = await local_env("neal_09")
    page = env.unwrapped._page
    for depth in range(4):
        await click(
            env, page.get_by_role("button", name=f"Image region, depth {depth}", exact=True).first
        )
    smallest = page.get_by_role("button", name="Image region, depth 4", exact=True).first
    count = await page.locator(".neal-recursive-leaf").count()
    await click(env, smallest)
    assert env.unwrapped.state.progress == 1
    await click(env, smallest)
    assert env.unwrapped.state.progress == 0
    assert await page.locator(".neal-recursive-leaf").count() == count
