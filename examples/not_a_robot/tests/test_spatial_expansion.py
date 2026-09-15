"""GUI-oracle coverage for spatial adaptations, not model benchmark results.

All game inputs use the canonical computer tool. Reading visible glyphs, SVG
geometry, or emitted physical observations is an explicitly privileged test
oracle. No test injects game state, accepted strings, wins, or completed events.
"""

from __future__ import annotations

import math

import pytest

from examples.not_a_robot.tests.test_local_tasks import center, click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env


async def enter(env, text):
    await click(env, env.unwrapped._page.get_by_role("textbox", name="Answer"))
    return await gui(
        env, [{"action": "key", "keys": ["ctrl", "a"]}, {"action": "type", "text": text}]
    )


async def drag_pixels(env, source, target):
    width, height = env.unwrapped.display_resolution
    start = [source[0] * 1000 / width, source[1] * 1000 / height]
    end = [target[0] * 1000 / width, target[1] * 1000 / height]
    return await gui(env, [{"action": "drag", "start_coordinate": start, "coordinate": end}])


@pytest.mark.live
async def test_3d_glyphs_rotate_reject_and_regenerate(local_env):
    env = await local_env("neal_16", max_steps=50)
    page = env.unwrapped._page
    glyphs = page.locator(".neal-text3d-letter span:last-child")
    answer = "".join(await glyphs.all_text_contents())
    assert len(answer) == 6
    initial_transform = await page.locator(".neal-text3d-letter").first.get_attribute("style")
    await gui(env, [{"action": "wait", "duration": 0.15}])
    assert initial_transform != await page.locator(".neal-text3d-letter").first.get_attribute(
        "style"
    )
    await enter(env, answer + " ")
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and env.unwrapped.state.mistakes == 1
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    refreshed = "".join(await glyphs.all_text_contents())
    assert (
        refreshed != answer and await page.get_by_role("textbox", name="Answer").input_value() == ""
    )
    box = await page.locator(".neal-text3d").bounding_box()
    await drag_pixels(env, [box["x"] + 150, box["y"] + 100], [box["x"] + 200, box["y"] + 130])
    await enter(env, refreshed)
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and result.reward == 1


def seeded_flashlight_text(seed, instance=0):
    """Independent seeded-instance oracle; not read from the game's answer closure."""

    def imul(a, b):
        return (a * b) & 0xFFFFFFFF

    def random_values():
        nonlocal seed
        while True:
            seed = (seed + 0x6D2B79F5) & 0xFFFFFFFF
            value = imul(seed ^ (seed >> 15), seed | 1)
            value ^= (value + imul(value ^ (value >> 7), value | 61)) & 0xFFFFFFFF
            yield ((value ^ (value >> 14)) & 0xFFFFFFFF) / 4294967296

    stream = random_values()
    alphabet = "ACEFGHIJKLMNPRSUVWXYZ12456789"
    for _ in range(instance + 1):
        text = ""
        for _ in range(5):
            text += alphabet[int(next(stream) * len(alphabet))]
            next(stream)
            next(stream)
    return text


@pytest.mark.live
async def test_flashlight_pixels_input_case_whitespace_and_refresh(local_env):
    env = await local_env("neal_19", seed=17)
    page = env.unwrapped._page
    canvas = page.locator("canvas")
    first = await canvas.screenshot()
    await gui(env, [{"action": "mouse_move", "coordinate": await center(env, canvas)}])
    second = await canvas.screenshot()
    assert first != second
    await enter(env, seeded_flashlight_text(17) + " ")
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert await page.get_by_role("textbox", name="Answer").input_value() == ""
    await enter(env, seeded_flashlight_text(17, 1).lower())
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and result.reward == 1


@pytest.mark.live
async def test_panorama_requires_direction_and_zoom_then_refresh_cycles(local_env):
    env = await local_env("neal_23", max_steps=70)
    page = env.unwrapped._page
    verify = page.get_by_role("button", name="Verify", exact=True)
    original = await page.locator(".neal-heading strong").inner_text()
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    title = await page.locator(".neal-heading strong").inner_text()
    assert title != original
    await click(env, verify)
    assert env.unwrapped.state.mistakes == 1
    desired = {
        "Couple Kissing": (-157, -24.5, 40),
        "Guitar Cat": (79, -32.5, 16),
        "Chilli's Sign": (72, -6.5, 12),
    }[title]
    box = await page.locator("canvas").bounding_box()
    # At hfov=100, repeated in-bounds drags accumulate the target camera angle.
    for yaw_delta, pitch_delta in [(desired[0] / 4, desired[1] / 4)] * 4:
        source = [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2]
        target = [
            source[0] - yaw_delta * box["width"] / 100,
            source[1] + pitch_delta * box["width"] / 100,
        ]
        await drag_pixels(env, source, target)
    result = await click(env, verify)
    assert not result.terminated  # Correct direction alone is insufficient.
    fov = 100
    while fov > desired[2]:
        await click(env, page.get_by_role("button", name="Zoom in", exact=True))
        fov = max(10, fov * 0.8)
    result = await click(env, verify)
    assert result.terminated and result.reward == 1


@pytest.mark.live
async def test_assembly_dragged_points_not_click_count_and_refresh(local_env):
    env = await local_env("neal_43", max_steps=80)
    page = env.unwrapped._page
    verify = page.get_by_role("button", name="Verify", exact=True)
    await click(env, verify)
    assert env.unwrapped.state.mistakes == 1
    initial = await page.locator("svg.neal-spatial-assembly").inner_html()
    for index in range(1, 5):
        source = page.locator(f'[data-piece="leg{index}"] [data-connector="top"]')
        target = page.locator(f'[data-piece="seat"] [data-connector="corner{index}"]')
        sb, tb = await source.bounding_box(), await target.bounding_box()
        await drag_pixels(
            env,
            [sb["x"] + sb["width"] / 2, sb["y"] + sb["height"] / 2],
            [tb["x"] + tb["width"] / 2, tb["y"] + tb["height"] / 2],
        )
    result = await click(env, verify)
    assert not result.terminated
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert await page.locator("svg.neal-spatial-assembly").inner_html() == initial
    for source_selector, target_selector in [
        *(
            (
                f'[data-piece="leg{i}"] [data-connector="top"]',
                f'[data-piece="seat"] [data-connector="corner{i}"]',
            )
            for i in range(1, 5)
        ),
        (
            '[data-piece="plank1"] [data-connector="connect"]',
            '[data-piece="rod1"] [data-connector="halfway"]',
        ),
        (
            '[data-piece="plank2"] [data-connector="connect"]',
            '[data-piece="rod1"] [data-connector="top"]',
        ),
    ]:
        source, target = page.locator(source_selector).first, page.locator(target_selector).first
        sb, tb = await source.bounding_box(), await target.bounding_box()
        await drag_pixels(
            env,
            [sb["x"] + sb["width"] / 2, sb["y"] + sb["height"] / 2],
            [tb["x"] + tb["width"] / 2, tb["y"] + tb["height"] / 2],
        )
    result = await click(env, verify)
    assert result.terminated and result.reward == 1


@pytest.mark.live
async def test_parking_actual_collision_and_reset(local_env):
    env = await local_env("neal_38", max_steps=30)
    page = env.unwrapped._page
    verify = page.get_by_role("button", name="Verify", exact=True)
    await click(env, verify)
    assert env.unwrapped.state.mistakes == 1
    await gui(env, [{"action": "hold_key", "keys": ["up"], "duration": 2.3}])
    result = await click(env, verify)
    assert not result.terminated and env.unwrapped.state.mistakes > 1
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    await click(env, verify)
    pose = await page.evaluate(
        "syntheticTask.snapshot().events.filter(e => e.kind === 'parking_submit_pose').at(-1)"
    )
    assert pose["x"] == 155 and pose["y"] == 355 and not any(pose["hit"])


@pytest.mark.live
async def test_parking_gui_driving_route_must_reach_target(local_env):
    env = await local_env("neal_38", max_steps=260, max_seconds=180)
    page = env.unwrapped._page
    verify = page.get_by_role("button", name="Verify", exact=True)
    # Outside bounds are allowed by this original scene. Physical waypoints route
    # around both pedestrians and obstacles, not through them or by teleportation.
    waypoints = [(-140, 320), (-140, -160), (90, -160), (155, -80), (155, 30)]
    waypoint = 0
    for _ in range(125):
        result = await click(env, verify)
        if result.terminated:
            assert result.reward == 1
            return
        pose = await page.evaluate(
            "syntheticTask.snapshot().events.filter(e => e.kind === 'parking_submit_pose').at(-1)"
        )
        assert not pose["lost"] and not any(pose["hit"]), (waypoint, pose)
        tx, ty = waypoints[waypoint]
        if math.hypot(tx - pose["x"], ty - pose["y"]) < 35 and waypoint < len(waypoints) - 1:
            waypoint += 1
            tx, ty = waypoints[waypoint]
        desired = math.atan2(ty - pose["y"], tx - pose["x"])
        error = (desired - pose["angle"] + math.pi) % (2 * math.pi) - math.pi
        keys = ["up"]
        if abs(error) > 0.06:
            keys.append("right" if error > 0 else "left")
        await gui(env, [{"action": "hold_key", "keys": keys, "duration": 0.24}])
    pytest.fail("Physical driving controller did not reach the parking target")


@pytest.mark.live
async def test_chess_grid_has_equal_square_rows_with_and_without_pieces(local_env):
    env = await local_env("neal_44")
    raw, page = env.unwrapped, env.unwrapped._page
    cells = await page.locator(".neal-spatial-square").evaluate_all(
        "nodes => nodes.map(node => { const r = node.getBoundingClientRect(); "
        "return {x:r.x, y:r.y, width:r.width, height:r.height, "
        "font_size:getComputedStyle(node).fontSize}; })"
    )
    board = await page.locator(".neal-spatial-chess").bounding_box()
    assert len(cells) == 64
    size = board["width"] / 8
    assert abs(board["width"] - board["height"]) < 0.5
    for index, cell in enumerate(cells):
        row, column = divmod(index, 8)
        assert abs(cell["width"] - cell["height"]) < 0.5, (index, cell)
        assert abs(cell["height"] - size) < 0.5, (index, cell, size)
        assert abs(cell["x"] - board["x"] - column * size) < 0.5
        assert abs(cell["y"] - board["y"] - row * size) < 0.5
        assert cell["font_size"] == "42px"
    raw.recorder.emit("chess_grid_geometry", board=board, squares=cells)
    await page.screenshot(path=str(raw.attempt_dir / "chess-equal-grid.png"))


@pytest.mark.live
async def test_chess_illegal_move_real_opponent_and_refresh(local_env):
    env = await local_env("neal_44", max_steps=50)
    page = env.unwrapped._page
    await page.get_by_text("White to move.", exact=True).wait_for(timeout=25000)
    assert (
        await page.locator('[data-square="e2"]').evaluate("node => getComputedStyle(node).fontSize")
        == "42px"
    )
    await click(env, page.get_by_role("button", name="Verify", exact=True))
    await click(env, page.locator('[data-square="e2"]'))
    await click(env, page.locator('[data-square="e5"]'))
    assert env.unwrapped.state.mistakes == 2
    assert "White pawn" in await page.locator('[data-square="e2"]').get_attribute("aria-label")
    source = await page.locator('[data-square="e2"]').bounding_box()
    target = await page.locator('[data-square="e4"]').bounding_box()
    await drag_pixels(
        env,
        [source["x"] + source["width"] / 2, source["y"] + source["height"] / 2],
        [target["x"] + target["width"] / 2, target["y"] + target["height"] / 2],
    )
    await page.get_by_text("White to move.", exact=True).wait_for()
    assert "White pawn" in await page.locator('[data-square="e4"]').get_attribute("aria-label")
    events = await page.evaluate(
        "syntheticTask.snapshot().events.filter(e => e.kind === 'chess_move')"
    )
    assert [event["side"] for event in events] == ["w", "b"]
    assert events[-1]["opponent"] == "stockfish_17_lite_wasm"
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert "White pawn" in await page.locator('[data-square="e2"]').get_attribute("aria-label")
    assert "empty" in await page.locator('[data-square="e4"]').get_attribute("aria-label")


@pytest.mark.live
@pytest.mark.parametrize(
    "task_id", ["neal_16", "neal_19", "neal_23", "neal_38", "neal_43", "neal_44"]
)
async def test_spatial_reset_cleans_up_and_keeps_snapshot_answers_private(local_env, task_id):
    env = await local_env(task_id)
    raw = env.unwrapped
    await click(env, raw._page.get_by_role("button", name="Verify", exact=True))
    assert raw.state.mistakes == 1
    await env.reset()
    assert raw.state.mistakes == 0
    snapshot = await raw._page.evaluate("syntheticTask.snapshot()")
    assert not any(
        key in snapshot for key in ["answer", "characters", "fen", "score", "connections"]
    )
