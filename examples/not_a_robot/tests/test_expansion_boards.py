"""Scripted GUI oracles for the captured crafting, sliding, and math boards.

DOM reads measure visible controls and rendered inventory, not model accuracy.
Every game input passes through the canonical computer action tool.
"""

from __future__ import annotations

import json

import pytest

from examples.not_a_robot.tests.test_local_tasks import center, click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env

SLIDING_INITIAL = [7, 4, 8, 2, 0, 6, 5, 1, 3]
SLIDING_MOVES = [4, 1, 2, 5, 8, 9, 6, 3, 2, 5, 8, 9, 6, 5, 8, 7, 4, 1, 2, 3, 6, 5, 8, 9]
MATH_ORDER = [3, 5, 6, 4, 8, 0, 2, 1, 7]


async def right_click(env, locator):
    return await gui(
        env, [{"action": "click", "coordinate": await center(env, locator), "button": "right"}]
    )


async def inventory_slots(locator):
    """Read the stack types/counts represented by visible slot images and labels."""
    return await locator.evaluate_all(
        "nodes => nodes.map(node => [node.dataset.item, Number(node.dataset.count)])"
    )


@pytest.mark.live
async def test_crafting_recipes_conserve_materials_and_require_inventory_pickaxe(local_env):
    env = await local_env("neal_21", max_steps=100)
    raw, page = env.unwrapped, env.unwrapped._page
    grid = page.locator(".neal-crafting-grid .neal-crafting-slot")
    inventory = page.locator(".neal-crafting-inventory .neal-crafting-slot")
    output = page.locator(".neal-crafting-output")
    verify = page.get_by_role("button", name="Verify", exact=True)
    assert await inventory_slots(inventory) == [
        ["log", 2],
        ["diamond", 3],
        ["", 0],
        ["", 0],
        ["", 0],
        ["", 0],
    ]
    result = await click(env, verify)
    assert not result.terminated and raw.state.mistakes == 1

    # A misplaced diamond stack does not become a tool because enough clicks occurred.
    await click(env, inventory.nth(1))
    await click(env, grid.nth(4))
    assert await output.get_attribute("data-item") == ""
    result = await click(env, verify)
    assert not result.terminated and raw.state.mistakes == 2
    await click(env, grid.nth(4))
    await click(env, inventory.nth(1))

    await click(env, inventory.nth(0))
    await click(env, grid.nth(0))
    assert await inventory_slots(output) == [["planks", 8]]
    await click(env, output)
    assert await inventory_slots(grid) == [["", 0]] * 9
    await right_click(env, grid.nth(1))
    assert await inventory_slots(output) == [["button", 1]]
    assert await page.locator(".neal-crafting-held").get_attribute("data-count") == "7"
    await right_click(env, grid.nth(4))
    await click(env, inventory.nth(0))
    assert await inventory_slots(output) == [["stick", 4]]
    await click(env, output)
    await right_click(env, grid.nth(4))
    await right_click(env, grid.nth(7))
    await click(env, inventory.nth(2))
    await click(env, inventory.nth(1))
    for index in [0, 1, 2]:
        await right_click(env, grid.nth(index))
    assert await inventory_slots(output) == [["pickaxe", 1]]

    # An extra ingredient invalidates the recipe; moving it back restores the preview.
    await click(env, inventory.nth(0))
    await right_click(env, grid.nth(3))
    await click(env, inventory.nth(0))
    assert await output.get_attribute("data-item") == ""
    await click(env, grid.nth(3))
    await click(env, inventory.nth(0))
    assert await inventory_slots(output) == [["pickaxe", 1]]
    result = await click(env, verify)
    assert not result.terminated and raw.state.progress == 0
    await click(env, output)
    assert await inventory_slots(grid) == [["", 0]] * 9
    result = await click(env, verify)
    assert not result.terminated  # Held output is not yet the captured inventory state.
    await click(env, inventory.nth(1))
    assert await inventory_slots(inventory) == [
        ["planks", 6],
        ["pickaxe", 1],
        ["stick", 2],
        ["", 0],
        ["", 0],
        ["", 0],
    ]
    result = await click(env, verify)
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "crafted_pickaxe_in_inventory"


@pytest.mark.live
async def test_crafting_refresh_clears_held_stack_and_restores_materials(local_env):
    env = await local_env("neal_21")
    raw, page = env.unwrapped, env.unwrapped._page
    inventory = page.locator(".neal-crafting-inventory .neal-crafting-slot")
    grid = page.locator(".neal-crafting-grid .neal-crafting-slot")
    await click(env, inventory.nth(0))
    await click(env, grid.nth(0))
    await click(env, page.locator(".neal-crafting-output"))
    await right_click(env, grid.nth(1))
    assert await page.locator(".neal-crafting-held").is_visible()
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert await inventory_slots(grid) == [["", 0]] * 9
    assert await inventory_slots(inventory) == [
        ["log", 2],
        ["diamond", 3],
        ["", 0],
        ["", 0],
        ["", 0],
        ["", 0],
    ]
    assert not await page.locator(".neal-crafting-held").is_visible()
    assert raw.state.progress == 0 and raw.state.status == "in_progress"
    await right_click(env, grid.nth(0))
    assert await inventory_slots(grid) == [["", 0]] * 9


@pytest.mark.live
async def test_sliding_ignores_illegal_moves_and_accepts_actual_solved_board(local_env):
    env = await local_env("neal_30", max_steps=80)
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-sliding-grid .neal-tile")
    verify = page.get_by_role("button", name="Verify", exact=True)
    read_fragments = "nodes => nodes.map(node => Number(node.dataset.fragment))"
    assert await tiles.evaluate_all(read_fragments) == SLIDING_INITIAL
    await click(env, tiles.nth(0))  # Diagonal to the central hole, not a legal move.
    await click(env, tiles.nth(4))  # Clicking the hole is not a move either.
    assert await tiles.evaluate_all(read_fragments) == SLIDING_INITIAL
    result = await click(env, verify)
    assert not result.terminated and raw.state.mistakes == 1
    # Add and undo a legal move: completion is not a fixed 24-step counter.
    await click(env, tiles.nth(3))
    await click(env, tiles.nth(4))
    assert await tiles.evaluate_all(read_fragments) == SLIDING_INITIAL
    expected = list(SLIDING_INITIAL)
    for cell in SLIDING_MOVES:
        index, empty = cell - 1, expected.index(0)
        expected[index], expected[empty] = expected[empty], expected[index]
        result = await click(env, tiles.nth(index))
        assert not result.terminated
        assert await tiles.evaluate_all(read_fragments) == expected
    assert expected == [1, 2, 3, 4, 5, 6, 7, 8, 0]
    assert raw.state.progress == 8
    result = await click(env, verify)
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "sliding_image_reassembled"


@pytest.mark.live
async def test_sliding_refresh_reinstates_captured_permutation(local_env):
    env = await local_env("neal_30")
    page = env.unwrapped._page
    tiles = page.locator(".neal-sliding-grid .neal-tile")
    await click(env, tiles.nth(3))
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert (
        await tiles.evaluate_all("nodes => nodes.map(node => Number(node.dataset.fragment))")
        == SLIDING_INITIAL
    )


@pytest.mark.live
async def test_math_order_requires_all_expressions_in_increasing_order(local_env):
    env = await local_env("neal_34", max_steps=60)
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-math-grid .neal-tile")
    verify = page.get_by_role("button", name="Verify", exact=True)
    assert await page.locator(".neal-math-grid math").count() == 9
    pi = tiles.nth(0).locator('mi[mathvariant="normal"]')
    assert await pi.text_content() == "π"
    assert "DejaVu Serif" in await pi.evaluate("node => getComputedStyle(node).fontFamily")
    for index in range(9):
        await click(env, tiles.nth(index))
    result = await click(env, verify)
    assert not result.terminated and raw.state.progress == 9 and raw.state.mistakes == 1
    await click(env, tiles.nth(0))
    assert raw.state.progress == 8
    assert await tiles.nth(1).locator(".neal-expression-rank").inner_text() == "1"
    await click(env, tiles.nth(0))
    assert await tiles.nth(0).locator(".neal-expression-rank").inner_text() == "9"
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert raw.state.progress == 0
    assert await page.locator('.neal-math-grid .neal-tile[aria-pressed="true"]').count() == 0
    for rank, index in enumerate(MATH_ORDER[:-1], 1):
        await click(env, tiles.nth(index))
        assert await tiles.nth(index).locator(".neal-expression-rank").inner_text() == str(rank)
    result = await click(env, verify)
    assert not result.terminated and raw.state.mistakes == 2
    await click(env, tiles.nth(MATH_ORDER[-1]))
    result = await click(env, verify)
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "expressions_in_ascending_order"


@pytest.mark.live
@pytest.mark.parametrize("task_id", ["neal_21", "neal_30", "neal_34"])
async def test_board_episode_reset_is_independent_and_closes_cleanly(local_env, task_id):
    env = await local_env(task_id)
    raw = env.unwrapped
    previous = raw.attempt_dir
    initial = await raw._page.locator(".neal-game").inner_html()
    await click(env, raw._page.get_by_role("button", name="Verify", exact=True))
    assert raw.state.mistakes == 1
    await env.reset()
    assert raw.attempt_dir != previous
    assert raw.state.mistakes == 0
    assert await raw._page.locator(".neal-game").inner_html() == initial
    await env.close()
    for attempt in [previous, raw.attempt_dir]:
        manifest = json.loads((attempt / "manifest.json").read_text())
        assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
        events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
        assert not any(event["type"] == "page_error" for event in events)
