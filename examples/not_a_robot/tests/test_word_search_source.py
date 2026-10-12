"""GUI checks of source word placement with the explicitly local seeded RNG.

The Python oracle enumerates legal geometry, not original JavaScript execution.
Its privileged answers drive ordinary clicks only; these are not model results.
Captured initial boards remain distinct from source-generated Refresh boards.
"""

from __future__ import annotations

import pytest

from examples.not_a_robot.tests.test_local_tasks import click
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env


def local_random(seed):
    """Independent integer arithmetic for the documented local PRNG."""
    mask = (1 << 32) - 1
    while True:
        seed = (seed + 0x6D2B79F5) & mask
        value = ((seed ^ (seed >> 15)) * (seed | 1)) & mask
        value ^= (value + (((value ^ (value >> 7)) * (value | 61)) & mask)) & mask
        yield ((value ^ (value >> 14)) & mask) / (1 << 32)


def source_puzzle(random):
    """Module 1072: sorted words, ordered directions, maximal shared letters."""
    alphabet = "abcdefghijklmnoprstuvwy"
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    board = [""] * 100
    answer = set()
    placements = []
    for word in sorted(("STOPSIGN", "BIKE")):
        candidates = []
        for dx, dy in directions:
            for row in range(10):
                for column in range(10):
                    cells = [(column + dx * i, row + dy * i) for i in range(len(word))]
                    if not all(0 <= x < 10 and 0 <= y < 10 for x, y in cells):
                        continue
                    indices = tuple(y * 10 + x for x, y in cells)
                    if any(
                        board[index] not in ("", letter) for index, letter in zip(indices, word)
                    ):
                        continue
                    overlap = sum(board[index] == letter for index, letter in zip(indices, word))
                    candidates.append((overlap, indices, (dx, dy)))
        best = max(item[0] for item in candidates)
        candidates = [item for item in candidates if item[0] == best]
        _, indices, direction = candidates[int(next(random) * len(candidates))]
        for index, letter in zip(indices, word):
            board[index] = letter
        answer.update(indices)
        placements.append((word, indices, direction))
    for index, letter in enumerate(board):
        if not letter:
            board[index] = alphabet[int(next(random) * 23)]
    return "".join(board).upper(), answer, placements


async def visible_letters(page):
    return "".join(await page.locator(".neal-tile").all_text_contents()).upper()


@pytest.mark.live
@pytest.mark.parametrize(
    "seed,count,directions",
    [
        pytest.param(9, 11, ((-1, 0), (1, -1)), id="left-up-right-crossing"),
        pytest.param(10, 12, ((0, -1), (-1, 1)), id="up-down-left-disjoint"),
        pytest.param(17, 11, ((1, 1), (0, 1)), id="down-right-down-crossing"),
        pytest.param(45, 11, ((1, 0), (-1, -1)), id="right-up-left-crossing"),
    ],
)
async def test_refresh_generates_source_word_search(local_env, seed, count, directions):
    env = await local_env("neal_07", seed=seed)
    page = env.unwrapped._page
    before = await visible_letters(page)
    expected, answer, placements = source_puzzle(local_random(seed))
    assert len(answer) == count
    assert tuple(direction for _, _, direction in placements) == directions
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    await page.screenshot(path=str(env.unwrapped.attempt_dir / "refreshed.png"))
    assert await visible_letters(page) == expected
    assert expected != before
    for word, indices, _ in placements:
        assert "".join(expected[index] for index in indices) == word
    # One shared letter is clicked only once. Exact set membership, rather than
    # the nominal word lengths, determines the required count on crossing boards.
    missing = max(answer)
    for index in sorted(answer - {missing}):
        await click(env, page.locator(".neal-tile").nth(index))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and not result.truncated and result.reward is None
    assert env.unwrapped.state.mistakes == 1
    await click(env, page.locator(".neal-tile").nth(missing))
    extra = min(set(range(100)) - answer)
    await click(env, page.locator(".neal-tile").nth(extra))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and not result.truncated and result.reward is None
    assert env.unwrapped.state.mistakes == 2
    await click(env, page.locator(".neal-tile").nth(extra))
    assert await page.locator('.neal-tile[aria-pressed="true"]').count() == count
    assert env.unwrapped.state.progress == count
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and not result.truncated and result.reward == 1


@pytest.mark.live
@pytest.mark.parametrize("instance", ["default", "incremental"])
async def test_refresh_clears_selection_and_reset_replays_from_capture(local_env, instance):
    env = await local_env("neal_07", seed=17, reference_instance=instance)
    original = await visible_letters(env.unwrapped._page)
    previous_attempt = env.unwrapped.attempt_dir
    for replay in range(2):
        if replay:
            await env.reset()
            assert env.unwrapped.attempt_dir != previous_attempt
            assert await visible_letters(env.unwrapped._page) == original
        page = env.unwrapped._page
        assert not await page.locator('.neal-tile[aria-pressed="true"]').count()
        random = local_random(17)
        seen = {original}
        await click(env, page.locator(".neal-tile").nth(0))
        for refresh in range(3):
            expected, _, _ = source_puzzle(random)
            await click(env, page.get_by_role("button", name="Refresh challenge"))
            assert await visible_letters(page) == expected
            assert expected not in seen
            seen.add(expected)
            assert not await page.locator('.neal-tile[aria-pressed="true"]').count()
            assert env.unwrapped.state.progress == 0
            assert env.unwrapped.state.status == "in_progress"
            if refresh in (0, 2):
                await page.screenshot(
                    path=str(env.unwrapped.attempt_dir / f"refresh-{refresh + 1}.png")
                )
            await click(env, page.locator(".neal-tile").nth(refresh))
            assert await page.locator('.neal-tile[aria-pressed="true"]').count() == 1
            assert env.unwrapped.state.progress == 1
