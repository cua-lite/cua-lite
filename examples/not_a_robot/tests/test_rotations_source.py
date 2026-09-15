"""Source rotation rules reached through ordinary GUI clicks and Refresh.

Module 1111 supplies the quarter-turn and modulo rules. The independent Python
PRNG models the documented local seed, not the original website random stream.
DOM transforms are privileged test observations, never model input or state writes.
"""

from __future__ import annotations

import pytest

from examples.not_a_robot.tests.test_local_tasks import click
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env
from examples.not_a_robot.tests.test_word_search_source import local_random

pytestmark = pytest.mark.live
INITIAL = [90, 90, 0, 0, 0, 90, 0, 0, 0]


async def visible_rotations(page):
    """Read the nine rendered image transforms, not the evaluator state."""
    return await page.locator(".neal-rotation-grid .neal-tile-face").evaluate_all(
        "nodes => nodes.map(node => "
        "Number(node.style.transform.match(/rotate\\(([-\\d.]+)deg\\)/)[1]))"
    )


async def test_rotation_transition_matches_source_css(local_env):
    env = await local_env("neal_05")
    transitions = await env.unwrapped._page.locator(
        ".neal-rotation-grid .neal-tile-face"
    ).evaluate_all(
        "nodes => nodes.map(node => {const css = getComputedStyle(node); "
        "return [css.transitionProperty, css.transitionDuration, css.transitionTimingFunction]})"
    )
    assert transitions == [["transform", "0.2s", "ease-out"]] * 9


@pytest.mark.parametrize("seed", [0, 17])
async def test_refresh_samples_nine_source_rotations_then_gui_solves(local_env, seed):
    env = await local_env("neal_05", seed=seed)
    page = env.unwrapped._page
    assert await visible_rotations(page) == INITIAL
    random = local_random(seed)
    expected = [90 * int(next(random) * 4) for _ in range(9)]
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    await page.screenshot(path=str(env.unwrapped.attempt_dir / "refreshed.png"))
    assert await visible_rotations(page) == expected
    assert any(expected)
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and not result.truncated and result.reward is None
    assert env.unwrapped.state.mistakes == 1
    clicks = 0
    for index in range(9):
        for _ in range((-expected[index] // 90) % 4):
            await click(env, page.locator(".neal-tile").nth(index))
            expected[index] += 90
            clicks += 1
            assert await visible_rotations(page) == expected
    assert all(angle % 360 == 0 for angle in expected)
    assert env.unwrapped.state.progress == clicks
    await page.screenshot(path=str(env.unwrapped.attempt_dir / "aligned.png"))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and not result.truncated and result.reward == 1


@pytest.mark.parametrize("seed", [0, 17])
async def test_refresh_advances_rng_and_reset_restores_capture(local_env, seed):
    env = await local_env("neal_05", seed=seed)
    first_attempt = env.unwrapped.attempt_dir
    for replay in range(2):
        if replay:
            await env.reset()
            assert env.unwrapped.attempt_dir != first_attempt
        page = env.unwrapped._page
        assert await visible_rotations(page) == INITIAL
        assert env.unwrapped.state.progress == 0
        random = local_random(seed)
        for refresh in range(3):
            await click(env, page.locator(".neal-tile").nth(refresh))
            assert env.unwrapped.state.progress == 1
            expected = [90 * int(next(random) * 4) for _ in range(9)]
            await click(env, page.get_by_role("button", name="Refresh challenge"))
            assert await visible_rotations(page) == expected
            assert env.unwrapped.state.progress == 0
            assert env.unwrapped.state.status == "in_progress"
            if refresh == 2:
                await page.screenshot(path=str(env.unwrapped.attempt_dir / "third-refresh.png"))
        # Random outcomes can repeat. Only the exact continued RNG stream is
        # required; no assertion demands that successive boards differ.


async def test_clockwise_accumulation_and_modulo_360_verification(local_env):
    env = await local_env("neal_05")
    page = env.unwrapped._page
    expected = INITIAL.copy()
    for index in (0, 1, 5):
        for _ in range(3):
            await click(env, page.locator(".neal-tile").nth(index))
            expected[index] += 90
            assert await visible_rotations(page) == expected
    # The board is aligned without submitting. One extra quarter-turn makes
    # exactly one tile wrong, while a full cycle must still accumulate.
    await click(env, page.locator(".neal-tile").nth(0))
    expected[0] += 90
    assert await visible_rotations(page) == expected
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and not result.truncated and result.reward is None
    for _ in range(3):
        await click(env, page.locator(".neal-tile").nth(0))
        expected[0] += 90
        assert await visible_rotations(page) == expected
    assert expected[0] == 720
    assert env.unwrapped.state.progress == 13
    events = await page.evaluate("window.syntheticTask.snapshot().events")
    rotations = [event for event in events if event["kind"] == "tile_rotated"]
    assert len(rotations) == 13
    assert all(
        event["direction"] == "clockwise" and event["quarter_turns"] == 1 for event in rotations
    )
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and not result.truncated and result.reward == 1
