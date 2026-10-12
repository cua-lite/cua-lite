"""Reference-instance GUI regression tests, separate from model gameplay.

These scripted oracles use uploaded reference inputs, source-derived rules,
and visible control geometry. They do not measure screenshot-only policy accuracy.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import cache

import pytest

from examples.not_a_robot.env import NotARobotEnv
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

TIC_LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


@cache
def tic_value(board, mark):
    """Test-only minimax from visible marks, not the local opponent's policy."""
    for player, value in (("X", 1), ("O", -1)):
        if any(all(board[index] == player for index in line) for line in TIC_LINES):
            return value
    empty = [index for index, value in enumerate(board) if not value]
    if not empty:
        return 0
    values = [
        tic_value(board[:index] + (mark,) + board[index + 1 :], "O" if mark == "X" else "X")
        for index in empty
    ]
    return (max if mark == "X" else min)(values)


async def tic_board(page):
    return tuple(
        await page.locator(".neal-tile-face").evaluate_all(
            "els => els.map(el => el.classList.contains('neal-mark-x') ? 'X' : "
            "el.classList.contains('neal-mark-o') ? 'O' : '')"
        )
    )


async def wait_for_tic_reply(page, previous_count):
    """Observe the real callback's visible O mark, not an exact-duration sleep."""
    await page.wait_for_function(
        "count => document.querySelectorAll('.neal-mark-o').length > count",
        arg=previous_count,
        timeout=2500,
    )


@pytest.fixture
async def tic_clock_env(local_env, monkeypatch):
    """Pause only the test browser before navigation to expose the opening timer."""
    original = NotARobotEnv._open_local_task

    async def open_with_clock(env):
        epoch = datetime(2026, 1, 1, tzinfo=UTC)
        await env._page.clock.install(time=epoch)
        await env._page.clock.pause_at(epoch)
        await original(env)

    monkeypatch.setattr(NotARobotEnv, "_open_local_task", open_with_clock)
    return local_env


async def win_tic_tac_toe(env):
    """Bounded GUI oracle using a normal refresh and visible-board minimax."""
    page = env.unwrapped._page
    for _ in range(5):
        await click(env, page.get_by_role("button", name="Refresh challenge"))
        for _ in range(5):
            board = await tic_board(page)
            empty = [index for index, mark in enumerate(board) if not mark]
            move = max(
                empty, key=lambda index: tic_value(board[:index] + ("X",) + board[index + 1 :], "O")
            )
            await click(env, page.locator(".neal-tile").nth(move))
            after_x = await tic_board(page)
            if any(all(after_x[index] == "X" for index in line) for line in TIC_LINES):
                return await click(env, page.get_by_role("button", name="Verify", exact=True))
            if not all(after_x):
                await wait_for_tic_reply(page, board.count("O"))
            board = await tic_board(page)
            if any(all(board[index] == "X" for index in line) for line in TIC_LINES):
                return await click(env, page.get_by_role("button", name="Verify", exact=True))
            assert not any(all(board[index] == "O" for index in line) for line in TIC_LINES)
            if all(board):
                result = await click(env, page.get_by_role("button", name="Verify", exact=True))
                assert not result.terminated and result.reward is None
                break
    pytest.fail("No GUI-oracle win within five legal games; opponent was not changed")


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
    if task_id == "neal_06":
        await wait_for_tic_reply(page, 0)
    await page.screenshot(path=str(raw.attempt_dir / "initial.png"))
    first_attempt = raw.attempt_dir
    initial = await page.locator(".neal-game").inner_html()
    await env.reset()
    assert raw.attempt_dir != first_attempt
    assert raw.state.status == "in_progress"
    assert raw.state.progress == raw.state.mistakes == 0
    if task_id == "neal_06":
        await wait_for_tic_reply(raw._page, 0)
    if task_id not in {"neal_03", "neal_10"}:
        assert await raw._page.locator(".neal-game").inner_html() == initial
    assert not raw._scope_violation
    events = [
        json.loads(line) for line in (first_attempt / "events.jsonl").read_text().splitlines()
    ]
    assert not any(row["type"] == "page_error" for row in events)


@pytest.mark.live
async def test_distorted_text_does_not_use_plate_normalization(local_env):
    env = await local_env("neal_03")
    raw, page = env.unwrapped, env.unwrapped._page
    await click(env, page.locator("#neal-answer"))
    for mistakes, answer in enumerate(("YHR PCD", "YHR-PCD", "yhrpcd"), start=1):
        result = await gui(
            env,
            [
                {"action": "key", "keys": ["ctrl", "a"]},
                {"action": "type", "text": answer, "press_enter": True},
            ],
        )
        assert not result.terminated and raw.state.mistakes == mistakes
    result = await gui(
        env,
        [
            {"action": "key", "keys": ["ctrl", "a"]},
            {"action": "type", "text": "YHRPCD", "press_enter": True},
        ],
    )
    assert result.terminated and result.reward == 1


@pytest.mark.live
@pytest.mark.parametrize(
    "selected,accepted",
    [
        pytest.param([1, 2, 5], True, id="one-missing-required"),
        pytest.param([1, 2, 5, 7], True, id="all-required"),
        pytest.param([1, 2, 5, 8], True, id="optional-plus-one-missing"),
        pytest.param([0, 1, 2, 5, 7], True, id="one-wrong-extra"),
        pytest.param([0, 1, 2, 5], False, id="missing-plus-wrong-extra"),
        pytest.param([1, 2], False, id="two-missing-required"),
        pytest.param([], False, id="empty-selection"),
        pytest.param(list(range(9)), False, id="all-selected"),
    ],
)
async def test_vegetable_source_error_tolerance_through_canonical_gui(
    local_env, selected, accepted
):
    """Source module 1125 cases, tested through real GUI, not a predicate substitute."""
    env = await local_env("neal_04")
    raw, page = env.unwrapped, env.unwrapped._page
    tiles = page.locator(".neal-grid .neal-tile")
    assert await tiles.count() == 9
    # Manifest-verified pictures follow tomato/carrot/onion/banana/grape/corn/
    # avocado/potato/eggplant; the potato picture depicts Mr. Potato Head.
    backgrounds = await page.locator(".neal-tile-face").evaluate_all(
        "els => els.map(el => el.style.backgroundImage)"
    )
    for index, background in enumerate(backgrounds, start=1):
        assert f"level04_image_{index:02d}.webp" in background
    for index in selected:
        await click(env, tiles.nth(index))
    assert (
        await tiles.evaluate_all(
            "els => els.flatMap((el, index) => "
            "el.getAttribute('aria-pressed') === 'true' ? [index] : [])"
        )
        == selected
    )
    assert raw.state.mistakes == 0
    await page.screenshot(path=str(raw.attempt_dir / "before_submit.png"))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    await page.screenshot(path=str(raw.attempt_dir / "after_submit.png"))
    assert not result.truncated
    assert result.terminated is accepted
    if accepted:
        assert result.reward == 1
        assert raw.state.status == raw.outcome == "success"
        assert raw.state.mistakes == 0
    else:
        assert result.reward is None
        assert raw.state.status == "in_progress"
        assert raw.state.mistakes == 1


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
        result = await gui(env, [{"action": "wait", "duration": 1.7}])
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
    "initial,player,opponent,game_num,samples",
    [
        ([4], [0, 6, 5, 1], [2, 3, 8, 7], 0, [[0.1]] * 4),
        ([4], [1, 8, 6, 3], [0, 2, 7, 5], 0, [[0.1]] * 4),
        ([], [0, 4, 6], [7, 8, 3], 1, [[0.1, 0.75], [0.9], [0.9]]),
    ],
)
async def test_tic_tac_toe_source_policy_exact_recorded_replies(
    local_env, initial, player, opponent, game_num, samples
):
    """Recorded moves checked against pure policy, not original RNG or GUI replay."""
    env = await local_env("neal_06")
    page = env.unwrapped._page
    board = [""] * 9
    for index in initial:
        board[index] = "O"
    for x, o, draws in zip(player, opponent, samples, strict=True):
        assert not board[x] and not board[o]
        board[x] = "X"
        actual = await page.evaluate(
            """({board, gameNum, samples}) => {
                let calls = 0;
                const move = ticTacToeMove(board, gameNum, () => {
                    if (calls === samples.length) throw new Error('Unexpected random draw');
                    return samples[calls++];
                });
                return {move, calls, board};
            }""",
            {"board": board, "gameNum": game_num, "samples": draws},
        )
        assert actual == {"move": o, "calls": len(draws), "board": board}
        board[o] = "O"
    if initial:
        assert all(board) and tic_value(tuple(board), "X") == 0
    else:
        board[2] = "X"
        assert all(board[index] == "X" for index in (2, 4, 6))


@pytest.mark.live
async def test_tic_tac_toe_source_refresh_policy_allows_a_legal_gui_win(local_env):
    env = await local_env("neal_06", seed=0)
    raw, page = env.unwrapped, env.unwrapped._page
    await wait_for_tic_reply(page, 0)
    assert await page.locator(".neal-mark-o").count() == 1
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and raw.state.mistakes == 1
    result = await win_tic_tac_toe(env)
    assert result.terminated and result.reward == 1
    assert raw.outcome == "success"


@pytest.mark.live
async def test_tic_tac_toe_opening_and_reply_timer_boundaries(tic_clock_env):
    """Synthetic-clock timing regression through genuine GUI actions."""
    env = await tic_clock_env("neal_06")
    raw, page = env.unwrapped, env.unwrapped._page
    assert await tic_board(page) == ("",) * 9
    await click(env, page.locator(".neal-tile").nth(0))
    assert await tic_board(page) == ("",) * 9
    await page.clock.run_for(99)
    assert await tic_board(page) == ("",) * 9
    await page.clock.run_for(1)
    assert await tic_board(page) == ("", "", "", "", "O", "", "", "", "")
    await click(env, page.locator(".neal-tile").nth(0))
    await page.clock.run_for(449)
    assert (await tic_board(page)).count("O") == 1
    await page.clock.run_for(1)
    assert await tic_board(page) == ("X", "", "O", "", "O", "", "", "", "")
    assert raw.state.progress == 1


@pytest.mark.live
async def test_tic_tac_toe_early_refresh_cancels_opening_timer(tic_clock_env):
    env = await tic_clock_env("neal_06")
    page = env.unwrapped._page
    assert await tic_board(page) == ("",) * 9
    await page.clock.run_for(99)
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    await page.clock.run_for(1000)
    assert await tic_board(page) == ("",) * 9
    events = await page.evaluate("window.syntheticTask.snapshot().events")
    assert not any(event["kind"] == "move" for event in events)
    assert events[-1]["kind"] == "board_reset" and events[-1]["first_player"] == "X"


@pytest.mark.live
async def test_tic_tac_toe_refresh_cancels_pending_opponent_move(tic_clock_env):
    env = await tic_clock_env("neal_06")
    page = env.unwrapped._page
    await page.clock.run_for(100)
    await click(env, page.locator(".neal-tile").nth(0))
    await page.clock.run_for(449)
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    await page.clock.run_for(1000)
    assert await tic_board(page) == ("",) * 9
    events = await page.evaluate("window.syntheticTask.snapshot().events")
    assert [(event["mark"], event["index"]) for event in events if event["kind"] == "move"] == [
        ("O", 4),
        ("X", 0),
    ]
    # A fresh turn still works; the cancelled old callback never consumes RNG.
    await click(env, page.locator(".neal-tile").nth(4))
    await page.clock.run_for(449)
    assert (await tic_board(page)).count("O") == 0
    await page.clock.run_for(1)
    assert await tic_board(page) == ("", "", "", "", "X", "O", "", "", "")


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
