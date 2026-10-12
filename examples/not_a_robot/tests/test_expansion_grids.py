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
@pytest.mark.parametrize("task_id", ["neal_12", "neal_13", "neal_31"])
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


@pytest.mark.live
@pytest.mark.parametrize(
    ("task_id", "selected"),
    [pytest.param("neal_29", [0, 5, 7], id="soul-exact")]
    + [
        pytest.param(
            "neal_29", [i for i in (0, 5, 7) if i != missing], id=f"soul-missing-{missing}"
        )
        for missing in (0, 5, 7)
    ]
    + [
        pytest.param("neal_29", [0, 5, 7, extra], id=f"soul-extra-{extra}")
        for extra in (1, 2, 4, 6, 8)
    ]
    + [
        pytest.param("neal_46", [*range(409, 417), *extras], id=f"floor-extras-{extras}")
        for extras in ((), (0,), (408,), (797,), (0, 797), (408, 417))
    ],
)
async def test_source_tolerant_grids_accept_allowed_selection(local_env, task_id, selected):
    """Source 1117/1077 predicates through real GUI, not autonomous model evidence."""
    env = await local_env(task_id, max_steps=80)
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-reference-selection .neal-tile")
    for index in selected:
        await scroll_click(env, tiles.nth(index))
    assert raw.state.progress == len(selected)
    result = await scroll_click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and not result.truncated and result.reward == 1
    assert raw.state.status == raw.outcome == "success"
    attempt = raw.attempt_dir
    await page.screenshot(path=str(attempt / "success.png"))
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


@pytest.mark.live
@pytest.mark.parametrize(
    ("task_id", "rejected_selections"),
    [
        pytest.param(
            "neal_29",
            ([], [0], [5], [7], [0, 5, 1], [0, 5, 7, 1, 2], [0, 5, 7, 3]),
            id="soul-two-errors-or-forbidden-cell",
        ),
        pytest.param(
            "neal_46",
            [
                [],
                [*range(409, 417), 408, 417, 418],
                [*range(410, 417), 408, 417, 418],
                [*range(410, 417), 408],
            ]
            + [[i for i in range(409, 417) if i != missing] for missing in range(409, 417)],
            id="floor-missing-required-or-three-extras",
        ),
    ],
)
async def test_source_tolerant_grids_reject_outside_boundary_and_refresh(
    local_env, task_id, rejected_selections
):
    """Invalid selections reuse one real page and never inject game state."""
    env = await local_env(task_id, max_steps=400)
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-reference-selection .neal-tile")
    for mistakes, selected in enumerate(rejected_selections, start=1):
        for index in selected:
            await scroll_click(env, tiles.nth(index))
        result = await scroll_click(env, page.get_by_role("button", name="Verify", exact=True))
        assert not result.terminated and not result.truncated, selected
        assert raw.state.status == "in_progress", selected
        assert raw.state.mistakes == mistakes and raw.state.progress == len(selected)
        await scroll_click(env, page.get_by_role("button", name="Refresh challenge", exact=True))
        assert raw.state.progress == 0 and raw.state.mistakes == mistakes
        assert (
            await page.locator('.neal-reference-selection .neal-tile[aria-pressed="true"]').count()
            == 0
        )


@pytest.mark.live
@pytest.mark.parametrize("extra", [None, 0, 244, 624])
async def test_waldo_accepts_required_cells_with_any_one_extra(local_env, extra):
    env = await local_env("neal_11", max_steps=40)
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-reference-selection .neal-tile")
    accepted = [218, 243] if extra is None else [218, 243, extra]
    for index in accepted:
        await scroll_click(env, tiles.nth(index))
    assert raw.state.progress == len(accepted)
    result = await scroll_click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and not result.truncated and result.reward == 1
    assert raw.state.status == raw.outcome == "success"
    attempt = raw.attempt_dir
    await env.close()
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["outcome"] == "success"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    assert not any(row["type"] == "page_error" for row in events)


@pytest.mark.live
async def test_waldo_rejects_missing_required_cells_or_four_cells_and_refreshes(local_env):
    env = await local_env("neal_11", max_steps=140)
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-reference-selection .neal-tile")
    initial = await page.locator(".neal-game").inner_html()
    for mistakes, rejected in enumerate(
        ([], [218], [243], [218, 244], [243, 244], [218, 243, 0, 244]), start=1
    ):
        for index in rejected:
            await scroll_click(env, tiles.nth(index))
        result = await scroll_click(env, page.get_by_role("button", name="Verify", exact=True))
        assert not result.terminated and raw.state.status == "in_progress"
        assert raw.state.mistakes == mistakes and raw.state.progress == len(rejected)
        await scroll_click(env, page.get_by_role("button", name="Refresh challenge", exact=True))
        assert raw.state.progress == 0 and raw.state.mistakes == mistakes
        assert (
            await page.locator('.neal-reference-selection .neal-tile[aria-pressed="true"]').count()
            == 0
        )
        assert await page.locator(".neal-game").inner_html() == initial


@pytest.mark.live
@pytest.mark.parametrize("width", [390, 799, 800, 801, 1280, 1920])
async def test_waldo_uses_source_responsive_width_and_cell_geometry(local_env, width):
    """Source modules 1126/2025/382 define independent width and mount-border breakpoints."""
    env = await local_env("neal_11", display_resolution=(width, 800), max_steps=40)
    page = env.unwrapped._page
    card = page.locator(".neal-waldo-card")
    parent_box = await page.locator("main").bounding_box()
    card_box = await card.bounding_box()
    expected_width = parent_box["width"] if width <= 800 else min(parent_box["width"] - 50, 1800)
    assert card_box["width"] == pytest.approx(expected_width, abs=0.05)
    assert await card.evaluate("e => e.style.getPropertyValue('--neal-waldo-border')") == (
        "0.2px" if width < 800 else "1px"
    )
    grid = page.locator(".neal-reference-selection")
    assert await grid.evaluate("e => getComputedStyle(e).gap") == "0px"
    grid_box = await grid.bounding_box()
    assert grid_box["width"] == pytest.approx(grid_box["height"], abs=0.1)
    tiles = grid.locator(".neal-tile")
    assert await tiles.count() == 625
    first_box = await tiles.nth(0).bounding_box()
    second_box = await tiles.nth(1).bounding_box()
    assert first_box["width"] == pytest.approx(grid_box["width"] / 25, abs=0.05)
    assert second_box["x"] == pytest.approx(first_box["x"] + first_box["width"], abs=0.05)
    await page.screenshot(path=str(env.unwrapped.attempt_dir / "initial_geometry.png"))
    # The unchanged source predicate still requires actual selections and Verify after scrolling.
    for index in (218, 243):
        await scroll_click(env, tiles.nth(index))
    result = await scroll_click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and not result.truncated and env.unwrapped.outcome == "success"


@pytest.mark.live
@pytest.mark.parametrize("width", [390, 1280])
async def test_waldo_scaled_selection_and_mount_border_survive_resize(local_env, width):
    """Real GUI/CSS transitions with DOM geometry as a test oracle, not a model run."""
    env = await local_env("neal_11", display_resolution=(width, 800), max_steps=30)
    raw, page = env.unwrapped, env.unwrapped._page
    card = page.locator(".neal-waldo-card")
    tiles = page.locator(".neal-reference-selection .neal-tile")
    tile, neighbor = tiles.nth(52), tiles.nth(53)
    mounted_border = await card.evaluate("e => e.style.getPropertyValue('--neal-waldo-border')")
    assert mounted_border == ("0.2px" if width < 800 else "1px")
    await scroll_click(env, neighbor)
    before = await tile.bounding_box()
    await scroll_click(env, tile)
    await gui(env, [{"action": "wait", "duration": 0.2}])
    selected = await tile.bounding_box()
    assert selected["width"] == pytest.approx(before["width"] * 0.75, abs=0.05)
    assert selected["height"] == pytest.approx(before["height"] * 0.75, abs=0.05)
    for axis, dimension in (("x", "width"), ("y", "height")):
        assert selected[axis] + selected[dimension] / 2 == pytest.approx(
            before[axis] + before[dimension] / 2, abs=0.05
        )
    assert (
        await tile.locator(".neal-tile-face").evaluate("e => getComputedStyle(e).transform")
        == "none"
    )
    assert raw.state.progress == 2
    await page.screenshot(path=str(raw.attempt_dir / "selected_geometry.png"))
    # scroll_click reads the transformed bounding box and sends its actual center.
    await scroll_click(env, tile)
    await gui(env, [{"action": "wait", "duration": 0.2}])
    assert await tile.get_attribute("aria-pressed") == "false"
    assert await neighbor.get_attribute("aria-pressed") == "true"
    assert (
        await page.locator('.neal-reference-selection .neal-tile[aria-pressed="true"]').count() == 1
    )
    assert raw.state.progress == 1 and raw.state.mistakes == 0
    restored = await tile.bounding_box()
    assert restored["width"] == pytest.approx(before["width"], abs=0.05)
    assert restored["height"] == pytest.approx(before["height"], abs=0.05)
    before_card = await card.bounding_box()
    resized_width = 1280 if width < 800 else 390
    await page.set_viewport_size({"width": resized_width, "height": 800})
    raw.recorder.emit("test_viewport_resized", viewport=[resized_width, 800])
    parent_box = await page.locator("main").bounding_box()
    resized_card = await card.bounding_box()
    expected_width = parent_box["width"] if resized_width <= 800 else parent_box["width"] - 50
    assert resized_card["width"] == pytest.approx(expected_width, abs=0.05)
    assert resized_card["width"] != pytest.approx(before_card["width"], abs=0.05)
    assert (
        await card.evaluate("e => e.style.getPropertyValue('--neal-waldo-border')")
        == mounted_border
    )
    assert await tile.get_attribute("aria-pressed") == "false"
    assert await neighbor.get_attribute("aria-pressed") == "true"
    await page.screenshot(path=str(raw.attempt_dir / "resized_geometry.png"))
