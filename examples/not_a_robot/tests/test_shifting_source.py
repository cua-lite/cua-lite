"""Slot-machine input transitions from captured source module 1115.

The source is read, not executed. Its onChange compares input length with the
sampled answer length, ignores lengths above five, and only unlocks indices in
the old sampled-answer suffix. Module 414 calls it when the input value changes.

DOM reads are test-only oracles for visible reels and logged outcomes. Normal
actions use the canonical environment. Bulk insertion uses the trusted, recorded
keyboard.insert_text GUI primitive, not the model's canonical type action, which
dispatches per-character keyboard.type. No DOM values, evaluator state, RNG,
animation position, or answers are injected. Tests do not claim original-site
replay or cover the separately declared local Refresh/empty-input differences.
"""

from __future__ import annotations

import pytest

from examples.not_a_robot.tests.test_local_tasks import click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env
from examples.not_a_robot.tests.test_puzzles_expansion import game_events

pytestmark = pytest.mark.live

REEL_STATES = "nodes => nodes.map(node => node.style.animationPlayState)"
VISIBLE_SYMBOL = (
    "node => {const p=parseFloat(getComputedStyle(node).translate.split(/\\s+/)[1]||'0');"
    "const i=Math.round(Math.abs(p)/100*10)%5;"
    "return node.children[i].getAttribute('aria-label')[0];}"
)


async def type_visible_symbol(env, column):
    """Type a currently displayed symbol, retrying only after a visible mismatch."""
    reel = env.unwrapped._page.locator(".neal-slot-reel").nth(column)
    for _ in range(12):
        symbol = await reel.evaluate(VISIBLE_SYMBOL)
        await gui(env, [{"action": "type", "text": symbol}])
        frozen = await reel.evaluate(VISIBLE_SYMBOL)
        if symbol.lower() == frozen.lower():
            assert await reel.evaluate("node => node.style.animationPlayState") == "paused"
            return symbol
        await gui(env, [{"action": "key", "keys": ["backspace"]}])
    raise AssertionError("Could not enter the observed moving symbol within bounded retries")


async def test_sixth_character_is_ignored_then_backspace_samples_again(local_env):
    env = await local_env("neal_40", max_steps=160)
    raw, page = env.unwrapped, env.unwrapped._page
    input_node = page.get_by_role("textbox", name="Answer")
    reels = page.locator(".neal-slot-reel")
    await click(env, input_node)
    for column in range(5):
        await type_visible_symbol(env, column)
    accepted = await input_node.input_value()
    before = await game_events(page)
    locked_before = sum(event["kind"] == "slot_reel_locked" for event in before)

    await gui(env, [{"action": "type", "text": "X"}])
    assert await input_node.input_value() == accepted + "X"
    assert raw.state.progress == 5
    assert await game_events(page) == before  # Length six is ignored, not sampled.
    assert await reels.evaluate_all(REEL_STATES) == ["paused"] * 5

    await gui(env, [{"action": "key", "keys": ["backspace"]}])
    assert await input_node.input_value() == accepted
    locked = [event for event in await game_events(page) if event["kind"] == "slot_reel_locked"]
    assert len(locked) == locked_before + 1 and locked[-1]["column"] == 4
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and raw.state.status == "in_progress"
    assert raw.state.mistakes == 1  # Sampled answer now has six, not five, characters.

    # Source skips absent reel 5, then unlocks reel 4 and truncates answer to four.
    await click(env, input_node)
    await gui(env, [{"action": "key", "keys": ["backspace"]}])
    assert await input_node.input_value() == accepted[:4]
    assert raw.state.progress == 4
    assert await reels.evaluate_all(REEL_STATES) == ["paused"] * 4 + ["running"]
    await type_visible_symbol(env, 4)
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "typed_symbols_match_frozen_reels"


async def test_bulk_five_then_delete_to_four_locks_another_reel(local_env):
    env = await local_env("neal_40")
    raw, page = env.unwrapped, env.unwrapped._page
    input_node = page.get_by_role("textbox", name="Answer")
    reels = page.locator(".neal-slot-reel")
    await click(env, input_node)
    await raw._primitive("keyboard.insert_text", text="AAAAA")
    await gui(env, [{"action": "screenshot"}])
    assert await input_node.input_value() == "AAAAA" and raw.state.progress == 1
    assert await reels.evaluate_all(REEL_STATES) == ["running"] * 4 + ["paused"]

    await gui(env, [{"action": "key", "keys": ["backspace"]}])
    assert await input_node.input_value() == "AAAA" and raw.state.progress == 2
    assert await reels.evaluate_all(REEL_STATES) == ["running"] * 3 + ["paused"] * 2
    locked = [event for event in await game_events(page) if event["kind"] == "slot_reel_locked"]
    assert [event["column"] for event in locked] == [4, 3]
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and raw.state.mistakes == 1


async def test_bulk_five_then_clear_only_unlocks_the_sampled_answer_suffix(local_env):
    env = await local_env("neal_40")
    raw, page = env.unwrapped, env.unwrapped._page
    input_node = page.get_by_role("textbox", name="Answer")
    reels = page.locator(".neal-slot-reel")
    await click(env, input_node)
    await raw._primitive("keyboard.insert_text", text="AAAAA")
    await gui(env, [{"action": "screenshot"}])
    assert raw.state.progress == 1
    await gui(
        env,
        [
            {"action": "key", "keys": ["ctrl", "a"]},
            {"action": "key", "keys": ["backspace"]},
        ],
    )
    assert await input_node.input_value() == "" and raw.state.progress == 0
    # Old answer length was one: only reel 0 resumes, not the locked reel 4.
    assert await reels.evaluate_all(REEL_STATES) == ["running"] * 4 + ["paused"]
    events = await game_events(page)
    assert [event["column"] for event in events if event["kind"] == "slot_reel_locked"] == [4]
    assert len([event for event in events if event["kind"] == "slot_suffix_unlocked"]) == 1
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and raw.state.mistakes == 1


async def test_regular_deletion_releases_only_the_matching_last_reel(local_env):
    env = await local_env("neal_40")
    raw, page = env.unwrapped, env.unwrapped._page
    input_node = page.get_by_role("textbox", name="Answer")
    reels = page.locator(".neal-slot-reel")
    await click(env, input_node)
    await gui(env, [{"action": "type", "text": "ABC"}])
    assert raw.state.progress == 3
    assert await reels.evaluate_all(REEL_STATES) == ["paused"] * 3 + ["running"] * 2
    await gui(env, [{"action": "key", "keys": ["backspace"]}])
    assert await input_node.input_value() == "AB" and raw.state.progress == 2
    assert await reels.evaluate_all(REEL_STATES) == ["paused"] * 2 + ["running"] * 3
    await gui(env, [{"action": "type", "text": "D"}])
    assert raw.state.progress == 3
    locked = [event for event in await game_events(page) if event["kind"] == "slot_reel_locked"]
    assert [event["column"] for event in locked] == [0, 1, 2, 2]


async def test_same_length_replacement_appends_a_sample_before_shortening(local_env):
    env = await local_env("neal_40")
    raw, page = env.unwrapped, env.unwrapped._page
    input_node = page.get_by_role("textbox", name="Answer")
    reels = page.locator(".neal-slot-reel")
    await click(env, input_node)
    await gui(env, [{"action": "type", "text": "AA"}])
    await gui(env, [{"action": "key", "keys": ["shift", "left"]}])
    await raw._primitive("keyboard.insert_text", text="B")
    await gui(env, [{"action": "screenshot"}])
    assert await input_node.input_value() == "AB" and raw.state.progress == 3
    locked = [event for event in await game_events(page) if event["kind"] == "slot_reel_locked"]
    assert [event["column"] for event in locked] == [0, 1, 1]
    await gui(env, [{"action": "key", "keys": ["backspace"]}])
    assert await input_node.input_value() == "A" and raw.state.progress == 1
    assert await reels.evaluate_all(REEL_STATES) == ["paused"] + ["running"] * 4


async def test_replacing_bulk_input_with_identical_text_does_not_trigger_the_watcher(local_env):
    env = await local_env("neal_40")
    raw, page = env.unwrapped, env.unwrapped._page
    input_node = page.get_by_role("textbox", name="Answer")
    await click(env, input_node)
    await raw._primitive("keyboard.insert_text", text="AAAAA")
    await gui(env, [{"action": "screenshot"}])
    before = await game_events(page)
    assert raw.state.progress == 1
    await gui(env, [{"action": "key", "keys": ["ctrl", "a"]}])
    await raw._primitive("keyboard.insert_text", text="AAAAA")
    await gui(env, [{"action": "screenshot"}])
    assert await input_node.input_value() == "AAAAA"
    assert raw.state.progress == 1
    assert await game_events(page) == before
    assert await page.locator(".neal-slot-reel").evaluate_all(REEL_STATES) == [
        "running",
        "running",
        "running",
        "running",
        "paused",
    ]


async def test_five_visible_characters_can_complete_through_normal_input(local_env):
    env = await local_env("neal_40", max_steps=160)
    raw, page = env.unwrapped, env.unwrapped._page
    await click(env, page.get_by_role("textbox", name="Answer"))
    for column in range(5):
        await type_visible_symbol(env, column)
    assert raw.state.progress == 5 and raw.state.status == "in_progress"
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "typed_symbols_match_frozen_reels"
