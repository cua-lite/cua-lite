"""Normal UI tests for eight source-rule games and explicit original media variants.

DOM reads observe visible grids, symbols, highlights and controls, not model skill.
Browser virtual-time tests exercise timers, not real-time recording/audio fidelity.
No test assigns evaluator state or invokes a completion function.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from examples.not_a_robot.tests.test_local_tasks import center, click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env


async def freeze_clock(page):
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    await page.clock.install(time=instant)
    await page.clock.pause_at(instant)


async def game_events(page):
    """Read-only evaluator audit; never used to modify state or inject a result."""
    return await page.evaluate("window.syntheticTask.snapshot().events")


@pytest.mark.live
@pytest.mark.parametrize("accepted", ["abcdef", "      ", "😀😀😀"])
async def test_inkblot_raw_utf16_length_and_captured_refresh(local_env, accepted):
    env = await local_env("neal_20")
    raw, page = env.unwrapped, env.unwrapped._page
    image = page.locator(".neal-inkblot")
    initial = await image.get_attribute("src")
    await click(env, page.get_by_role("textbox", name="Answer"))
    await gui(env, [{"action": "type", "text": "abcde", "press_enter": True}])
    assert raw.state.mistakes == 1 and raw.state.status == "in_progress"
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert await image.get_attribute("src") == initial
    assert await page.get_by_role("textbox", name="Answer").input_value() == ""
    await click(env, page.get_by_role("textbox", name="Answer"))
    result = await gui(env, [{"action": "type", "text": accepted, "press_enter": True}])
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "description_length_accepted"


NETWORK_ROUTES = [
    [0, 6, 12, 18, 19, 20, 21, 27],
    [5, 4, 10, 9, 15],
    [26, 25, 24, 30, 31, 32, 33, 34, 28],
    [13, 7, 1, 2, 3],
    [11, 17, 16, 22, 23, 29, 35],
    [8, 14],
]


async def network_drag(env, route):
    tiles = env.unwrapped._page.locator(".neal-network-grid .neal-tile")
    positions = [await center(env, tiles.nth(index)) for index in route]
    return await gui(
        env,
        [{"action": "mouse_down", "coordinate": positions[0]}]
        + [{"action": "mouse_move", "coordinate": coordinate} for coordinate in positions[1:]]
        + [{"action": "mouse_up"}],
    )


@pytest.mark.live
async def test_network_requires_real_adjacent_paths_and_complete_coverage(local_env):
    env = await local_env("neal_27", max_steps=80)
    raw, page = env.unwrapped, env.unwrapped._page
    verify = page.get_by_role("button", name="Verify", exact=True)
    await network_drag(env, [0, 7])  # A diagonal cannot extend the red path.
    assert await page.locator(".neal-network-lines polyline").count() == 0
    await click(env, verify)
    assert raw.state.mistakes == 1
    # All six pairs connected through short routes still do not cover the board.
    shortcuts = [
        NETWORK_ROUTES[0],
        NETWORK_ROUTES[1],
        [26, 32, 33, 34, 28],
        NETWORK_ROUTES[3],
        NETWORK_ROUTES[4],
        NETWORK_ROUTES[5],
    ]
    for route in shortcuts:
        await network_drag(env, route)
    assert await page.locator(".neal-network-grid").get_attribute("data-connected") == "6"
    result = await click(env, verify)
    assert not result.terminated and raw.state.mistakes == 2
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert await page.locator(".neal-network-lines polyline").count() == 0
    for route in NETWORK_ROUTES:
        await network_drag(env, route)
    result = await click(env, verify)
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "all_network_pairs_cover_board"


async def watch_memory(page, length):
    await page.clock.run_for(1500)
    sequence = []
    for _ in range(length):
        lit = await page.locator(".neal-memory-grid .neal-tile").evaluate_all(
            "nodes => nodes.flatMap((node, i) => node.classList.contains('lit') ? [i] : [])"
        )
        assert len(lit) == 1
        sequence.append(lit[0])
        await page.clock.run_for(900)
    assert await page.locator(".neal-memory-grid").get_attribute("data-phase") == "input"
    return sequence


@pytest.mark.live
async def test_memory_wrong_order_resets_then_three_four_five_succeed(local_env):
    env = await local_env("neal_32", max_steps=80)
    raw, page = env.unwrapped, env.unwrapped._page
    await freeze_clock(page)
    assert not await page.get_by_role("button", name="Refresh challenge").is_visible()
    verify = page.get_by_role("button", name="Verify", exact=True)
    assert await verify.is_disabled()
    await click(env, page.get_by_role("button", name="Start with sound"))
    tiles = page.locator(".neal-memory-grid .neal-tile")
    assert await tiles.nth(0).is_disabled()
    sequence = await watch_memory(page, 3)
    await click(env, tiles.nth((sequence[0] + 1) % 25))
    assert raw.state.mistakes == 1
    assert await page.locator(".neal-memory-grid").get_attribute("data-phase") == "failed"
    await page.clock.run_for(1000)
    for length in [3, 4, 5]:
        sequence = await watch_memory(page, length)
        # There is deliberately no rhythm deadline in the input phase.
        await page.clock.run_for(9000)
        for index in sequence:
            await click(env, tiles.nth(index))
    assert not await verify.is_disabled()
    events = await game_events(page)
    assert any(event["kind"] == "audio_tone" for event in events)
    result = await click(env, verify)
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "memory_three_rounds_verified"


def match_cells(board):
    found = set()
    for outer in range(8):
        for vertical in [False, True]:
            indices = [step * 8 + outer if vertical else outer * 8 + step for step in range(8)]
            for start in range(6):
                triple = indices[start : start + 3]
                if len({board[index] for index in triple}) == 1:
                    found.update(triple)
    return found


def source_candy_draws(seed):
    """Local seeded stream, not an original-site seed or captured random sequence."""
    mask = 0xFFFFFFFF
    state = seed
    while True:
        state = (state + 0x6D2B79F5) & mask
        value = ((state ^ (state >> 15)) * (state | 1)) & mask
        product = ((value ^ (value >> 7)) * (value | 61)) & mask
        value = (value ^ ((value + product) & mask)) & mask
        yield 1 + ((value ^ (value >> 14)) * 6 // (1 << 32))


def source_candy_matches(board):
    """Source module 1078: all horizontal runs, then vertical runs, unique in order."""
    found: dict[int, None] = {}
    for vertical in (False, True):
        for outer in range(8):
            line = [step * 8 + outer if vertical else outer * 8 + step for step in range(8)]
            for start in range(6):
                triple = line[start : start + 3]
                if len({board[index] for index in triple}) == 1:
                    found.update(dict.fromkeys(triple))
    return list(found)


def source_candy_initial(draws):
    """Reimplement the reviewed source semantics in Python; never execute source JS."""
    board = [next(draws) for _ in range(64)]
    while matches := source_candy_matches(board):
        for index in matches:
            board[index] = next(draws)
    return board


def source_candy_refill(board, matches, draws):
    result = list(board)
    for column in range(8):
        destination = 7
        for row in range(7, -1, -1):
            index = row * 8 + column
            if index not in matches:
                result[destination * 8 + column] = board[index]
                destination -= 1
        # Source consumes fresh draws at the lowest empty row first, per column.
        for row in range(destination, -1, -1):
            result[row * 8 + column] = next(draws)
    return result


@pytest.mark.parametrize(
    "seed,expected",
    [(0, [2, 1, 2, 1, 3, 4]), (17, [5, 2, 4, 1, 1, 5]), (2026, [3, 2, 4, 4, 1, 2])],
)
def test_candy_oracle_local_rng_vectors(seed, expected):
    draws = source_candy_draws(seed)
    assert [next(draws) for _ in expected] == expected


def test_candy_oracle_scan_order_and_bottom_up_refill():
    """Synthetic oracle unit fixtures, not injected browser boards or model evidence."""
    board = [(row + column) % 6 + 1 for row in range(8) for column in range(8)]
    ordered = list(board)
    for index in (48, 49, 50):
        ordered[index] = 6
    for index in (9, 17, 25):
        ordered[index] = 5
    assert source_candy_matches(ordered) == [48, 49, 50, 9, 17, 25]
    matches = {8, 24, 40, 1, 57}
    filled = source_candy_refill(board, matches, iter([6, 5, 4, 3, 2]))
    assert [filled[row * 8] for row in range(8)] == [4, 5, 6] + [
        board[row * 8] for row in (0, 2, 4, 6, 7)
    ]
    assert [filled[row * 8 + 1] for row in range(8)] == [2, 3] + [
        board[row * 8 + 1] for row in range(1, 7)
    ]
    assert all(filled[index] == board[index] for index in range(64) if index % 8 >= 2)


def candy_swaps(board):
    options = []
    for first in range(64):
        for second in [first + 1 if first % 8 < 7 else -1, first + 8 if first < 56 else -1]:
            if second < 0:
                continue
            candidate = list(board)
            candidate[first], candidate[second] = candidate[second], candidate[first]
            options.append((len(match_cells(candidate)), first, second))
    return options


async def candy_board(page):
    return await page.locator(".neal-candy-grid .neal-tile").evaluate_all(
        "nodes => nodes.map(node => "
        "Number(node.getAttribute('aria-label').split(',')[0].split(' ')[1]))"
    )


async def settle_candy(page):
    for _ in range(60):
        if await page.locator(".neal-candy-grid").get_attribute("data-phase") != "animating":
            return
        await page.clock.run_for(600)
    raise AssertionError("Cascade did not settle within bounded virtual time")


@pytest.mark.live
@pytest.mark.parametrize("seed", [0, 17, 2026])
async def test_candy_source_initial_and_refresh_continue_local_seed_stream(local_env, seed):
    env = await local_env("neal_36", seed=seed)
    page = env.unwrapped._page
    draws = source_candy_draws(seed)
    for revision in range(3):
        if revision:
            await click(env, page.get_by_role("button", name="Refresh challenge"))
        expected = source_candy_initial(draws)
        assert await candy_board(page) == expected
        assert not source_candy_matches(expected)
        assert await page.locator(".neal-puzzle-info").get_attribute("data-score") == "0"
        assert await page.locator(".neal-puzzle-info").get_attribute("data-moves") == "30"


@pytest.mark.live
async def test_candy_gui_refill_waves_match_independent_source_oracle(local_env):
    """Privileged Python/DOM oracle under a test clock, not original-site or Astra play."""
    env = await local_env("neal_36", seed=17, max_steps=30)
    page = env.unwrapped._page
    await freeze_clock(page)
    draws = source_candy_draws(17)
    expected = source_candy_initial(draws)
    tiles = page.locator(".neal-candy-grid .neal-tile")
    info = page.locator(".neal-puzzle-info")
    score = 0
    wave_counts = []
    for move in range(3):
        visible = await candy_board(page)
        assert visible == expected
        count, first, second = max(candy_swaps(visible))
        assert count > 0, "Dead board: keep this failure; do not pick a replacement seed"
        expected[first], expected[second] = expected[second], expected[first]
        await click(env, tiles.nth(first))
        await click(env, tiles.nth(second))
        assert await candy_board(page) == expected
        await page.clock.run_for(399)
        assert await info.get_attribute("data-score") == str(score)
        assert await info.get_attribute("data-moves") == str(30 - move)
        await page.clock.run_for(1)
        waves = 0
        while matches := source_candy_matches(expected):
            assert waves < 60, "Reference cascade exceeded the bounded test budget"
            points = len(matches) * 10 + max(0, len(matches) - 3) * 20
            score += points
            assert await info.get_attribute("data-score") == str(score)
            assert await info.get_attribute("data-moves") == str(29 - move)
            actual = [
                event for event in await game_events(page) if event["kind"] == "candies_matched"
            ][-1]
            assert actual["cells"] == matches
            assert (actual["count"], actual["points"], actual["score"], actual["cascade"]) == (
                len(matches),
                points,
                score,
                waves,
            )
            await page.clock.run_for(200)
            expected = source_candy_refill(expected, matches, draws)
            assert await candy_board(page) == expected
            await page.clock.run_for(400)
            waves += 1
        wave_counts.append(waves)
        assert await page.locator(".neal-candy-grid").get_attribute("data-phase") == "ready"
    assert 1 in wave_counts and any(count > 1 for count in wave_counts)
    assert env.unwrapped.state.status == "in_progress"
    # Refill draws must also remain consumed when the user later refreshes.
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert await candy_board(page) == source_candy_initial(draws)
    assert await info.get_attribute("data-score") == "0"
    assert await info.get_attribute("data-moves") == "30"


@pytest.mark.live
async def test_candy_invalid_swap_reverts_and_valid_cascades_reach_score(local_env):
    env = await local_env("neal_36", seed=17, max_steps=150)
    raw, page = env.unwrapped, env.unwrapped._page
    await freeze_clock(page)
    tiles = page.locator(".neal-candy-grid .neal-tile")
    verify = page.get_by_role("button", name="Verify", exact=True)
    initial = await candy_board(page)
    assert not match_cells(initial)
    _, first, second = next(option for option in candy_swaps(initial) if option[0] == 0)
    await click(env, tiles.nth(first))
    await click(env, tiles.nth(second))
    await settle_candy(page)
    assert await candy_board(page) == initial
    assert await page.locator(".neal-puzzle-info").get_attribute("data-moves") == "30"
    await click(env, verify)
    assert raw.state.mistakes == 2
    for _ in range(30):
        if int(await page.locator(".neal-puzzle-info").get_attribute("data-score")) >= 1000:
            break
        count, first, second = max(candy_swaps(await candy_board(page)))
        assert count > 0, "The source game has no reshuffle; a dead board must not auto-pass"
        await click(env, tiles.nth(first))
        await click(env, tiles.nth(second))
        await settle_candy(page)
    waves = [event for event in await game_events(page) if event["kind"] == "candies_matched"]
    assert waves
    for event in waves:
        assert event["count"] == len(set(event["cells"]))
        assert event["points"] == 10 * event["count"] + 20 * max(0, event["count"] - 3)
    score = int(await page.locator(".neal-puzzle-info").get_attribute("data-score"))
    assert score == sum(event["points"] for event in waves) and score >= 1000
    result = await click(env, verify)
    assert result.terminated and result.reward == 1


@pytest.mark.live
async def test_candy_refresh_cancels_inflight_swap(local_env):
    env = await local_env("neal_36")
    page = env.unwrapped._page
    await freeze_clock(page)
    tiles = page.locator(".neal-candy-grid .neal-tile")
    _, first, second = max(candy_swaps(await candy_board(page)))
    await click(env, tiles.nth(first))
    await click(env, tiles.nth(second))
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    refreshed = await candy_board(page)
    await page.clock.run_for(5000)
    assert await candy_board(page) == refreshed
    assert await page.locator(".neal-puzzle-info").get_attribute("data-score") == "0"
    assert await page.locator(".neal-puzzle-info").get_attribute("data-moves") == "30"


@pytest.mark.live
@pytest.mark.parametrize("accepted_ids", [{1, 6, 7}, {1, 6, 7, 9}, {1, 2, 6, 7, 9}])
async def test_face_symmetric_difference_allows_only_one_error(local_env, accepted_ids):
    env = await local_env("neal_37", max_steps=40)
    raw, page = env.unwrapped, env.unwrapped._page
    verify = page.get_by_role("button", name="Verify", exact=True)
    tiles = page.locator(".neal-grid .neal-tile")

    async def select_ids(ids):
        styles = await tiles.locator(".neal-tile-face").evaluate_all(
            "nodes => nodes.map(node => node.style.backgroundImage)"
        )
        for index, style in enumerate(styles):
            image_id = int(style.split("level37_image_")[1][:2])
            if image_id in ids:
                await click(env, tiles.nth(index))

    await select_ids({1, 6, 7, 2})  # Missing 9 and extra 2 are two errors, not one.
    result = await click(env, verify)
    assert not result.terminated and raw.state.mistakes == 1
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert await page.locator('.neal-grid .neal-tile[aria-pressed="true"]').count() == 0
    await select_ids(accepted_ids)
    result = await click(env, verify)
    assert result.terminated and result.reward == 1


@pytest.mark.live
async def test_slots_character_locking_and_deletion_resumes(local_env):
    env = await local_env("neal_40", max_steps=80)
    raw, page = env.unwrapped, env.unwrapped._page
    input_node = page.get_by_role("textbox", name="Answer")
    verify = page.get_by_role("button", name="Verify", exact=True)
    await click(env, input_node)
    await gui(env, [{"action": "type", "text": "AAAAA"}])
    await click(env, verify)
    assert raw.state.status == "in_progress" and raw.state.mistakes == 1
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    reels = page.locator(".neal-slot-reel")
    await input_node.focus()
    read_symbol = (
        "node => {const p=parseFloat(getComputedStyle(node).translate.split(/\\s+/)[1]||'0');"
        "const i=Math.round(Math.abs(p)/100*10)%5;"
        "return node.children[i].getAttribute('aria-label')[0];}"
    )
    for column in range(5):
        reel = reels.nth(column)
        for _ in range(12):
            # Predict from the currently visible animated column, then type via UI.
            symbol = await reel.evaluate(read_symbol)
            await page.keyboard.type(symbol)
            frozen = await reel.evaluate(read_symbol)
            if frozen.lower() == symbol.lower():
                break
            await page.keyboard.press("Backspace")
        else:
            raise AssertionError("Could not type observed reel symbol within bounded retries")
        assert await reel.evaluate("node => node.style.animationPlayState") == "paused"
    result = await click(env, verify)
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "typed_symbols_match_frozen_reels"


@pytest.mark.live
async def test_finale_pause_refresh_confetti_and_natural_end_gate(local_env):
    env = await local_env("neal_48", max_steps=30)
    raw, page = env.unwrapped, env.unwrapped._page
    await freeze_clock(page)
    verify = page.get_by_role("button", name="Verify", exact=True)
    assert await verify.is_disabled()
    await click(env, page.get_by_role("button", name="Play finale"))
    await page.clock.run_for(5000)
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert await page.locator(".neal-finale-canvas").get_attribute("data-elapsed") == "0.000"
    await click(env, page.get_by_role("button", name="Play finale"))
    await page.clock.run_for(30000)
    await click(env, page.get_by_role("button", name="Pause finale"))
    paused = await page.locator(".neal-finale-canvas").get_attribute("data-elapsed")
    await page.clock.run_for(71000)
    assert await page.locator(".neal-finale-canvas").get_attribute("data-elapsed") == paused
    assert await verify.is_disabled()
    await click(env, page.get_by_role("button", name="Resume finale"))
    await page.clock.run_for(35100)
    assert await verify.is_disabled()
    assert (
        len([event for event in await game_events(page) if event["kind"] == "finale_confetti"]) == 1
    )
    await page.clock.run_for(5000)
    assert not await verify.is_disabled()
    result = await click(env, verify)
    assert result.terminated and result.reward == 1
    assert await page.locator(".neal-local-terminal").inner_text() == "Local finale complete"
    assert raw.state.reason == "original_local_finale_played_and_verified"


@pytest.mark.live
@pytest.mark.parametrize(
    "task_id",
    ["neal_20", "neal_27", "neal_32", "neal_36", "neal_37", "neal_40", "neal_47", "neal_48"],
)
async def test_puzzle_episode_reset_and_resource_cleanup(local_env, task_id):
    env = await local_env(task_id)
    raw = env.unwrapped
    original = raw.attempt_dir
    await env.reset()
    assert raw.attempt_dir != original and raw.state.status == "in_progress"
    await env.close()
    for attempt in [original, raw.attempt_dir]:
        manifest = json.loads((attempt / "manifest.json").read_text())
        assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
        events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
        assert not any(event["type"] == "page_error" for event in events)
