"""Source-chart timing contracts using real GUI keys and synthetic browser time.

These white-box regressions observe generated game events and the imported chart.
They do not measure model skill, real-time audio fidelity or original video end.
No evaluator state or completion event is assigned by a test.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import groupby

import pytest

from examples.not_a_robot.local_tasks import REFERENCE_ROOT
from examples.not_a_robot.tests.test_local_tasks import click
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env
from examples.not_a_robot.tests.test_puzzles_expansion import freeze_clock, game_events


async def start_rhythm(env):
    page = env.unwrapped._page
    await click(env, page.get_by_role("button", name="Start music"))
    events = await game_events(page)
    return [event for event in events if event["kind"] == "rhythm_started"][-1]


@pytest.mark.live
async def test_rhythm_first_spawn_geometry_oldest_note_and_held_key(local_env):
    env = await local_env("neal_47")
    page = env.unwrapped._page
    await freeze_clock(page)
    started = await start_rhythm(env)
    align_age = started["align_age_seconds"]
    board = await page.locator(".neal-rhythm-board").bounding_box()
    target = await page.get_by_role("button", name="ArrowLeft", exact=True).bounding_box()
    assert align_age == pytest.approx(
        2 * (target["y"] + target["height"] / 2 - board["y"] + 22.5) / (board["height"] + 45)
    )
    spawned = [event for event in await game_events(page) if event["kind"] == "rhythm_note_spawned"]
    assert [(event["note"], event["spawn_elapsed_seconds"]) for event in spawned] == [(0, 0)]
    assert spawned[0]["chart_time_seconds"] == 1.55 < align_age
    style = await page.locator(".neal-rhythm-note").evaluate(
        "node => ({height: getComputedStyle(node).height, "
        "duration: getComputedStyle(node).animationDuration})"
    )
    assert style == {"height": "45px", "duration": "2s"}
    await page.keyboard.press("ArrowLeft")  # No left note has spawned yet.
    assert not any(event["kind"] == "rhythm_judged" for event in await game_events(page))
    target_ms = math.ceil(align_age * 1000)
    await page.clock.run_for(target_ms)
    # Two down notes exist. The older note is judged, and holding cannot judge twice.
    await page.keyboard.down("ArrowDown")
    await page.keyboard.down("ArrowDown")
    judged = [event for event in await game_events(page) if event["kind"] == "rhythm_judged"]
    assert len(judged) == 1 and judged[0]["note"] == 0 and judged[0]["hit"]
    assert judged[0]["timing_error_seconds"] == pytest.approx(target_ms / 1000 - align_age)
    assert abs(judged[0]["timing_error_seconds"]) < 0.001
    await page.keyboard.up("ArrowDown")
    await page.keyboard.press("ArrowDown")
    judged = [event for event in await game_events(page) if event["kind"] == "rhythm_judged"]
    assert [event["note"] for event in judged] == [0, 1]
    assert not judged[1]["hit"]


@pytest.mark.live
@pytest.mark.parametrize("error_ms,expected_hit", [(-251, False), (-249, True), (249, True)])
async def test_rhythm_strict_window_around_250ms(local_env, error_ms, expected_hit):
    env = await local_env("neal_47")
    page = env.unwrapped._page
    await freeze_clock(page)
    started = await start_rhythm(env)
    target_ms = round(started["align_age_seconds"] * 1000) + error_ms
    await page.clock.run_for(target_ms)
    await page.keyboard.press("ArrowDown")
    judged = [event for event in await game_events(page) if event["kind"] == "rhythm_judged"]
    assert len(judged) == 1 and judged[0]["note"] == 0
    assert judged[0]["hit"] is expected_hit
    assert judged[0]["timing_error_seconds"] == pytest.approx(
        target_ms / 1000 - started["align_age_seconds"]
    )


@pytest.mark.live
@pytest.mark.parametrize("error_ms", [-250, 250])
async def test_rhythm_exact_250ms_boundaries_with_synthetic_geometry(local_env, error_ms):
    env = await local_env("neal_47")
    page = env.unwrapped._page
    await freeze_clock(page)
    # Test-only geometry makes alignAge exactly representable as 1.75 seconds.
    # This changes layout, not the chart, evaluator, clock math or game outcome.
    await page.locator(".neal-rhythm-targets").evaluate("node => node.style.bottom = '5.625px'")
    started = await start_rhythm(env)
    assert started["align_age_seconds"] == 1.75
    await page.clock.run_for(1750 + error_ms)
    if error_ms < 0:
        await page.keyboard.press("ArrowDown")
        judged = [event for event in await game_events(page) if event["kind"] == "rhythm_judged"]
        assert len(judged) == 1 and judged[0]["note"] == 0
        assert judged[0]["timing_error_seconds"] == -0.25 and not judged[0]["hit"]
    else:
        # At +250ms the independent real game timer has already rejected note 0.
        missed = [event for event in await game_events(page) if event["kind"] == "rhythm_missed"]
        assert [(event["note"], event["reason"]) for event in missed] == [(0, "late")]
        await page.keyboard.press("ArrowDown")
        judged = [event for event in await game_events(page) if event["kind"] == "rhythm_judged"]
        assert all(event["note"] != 0 for event in judged)
    assert await page.get_by_role("button", name="Verify", exact=True).is_disabled()


@pytest.mark.live
async def test_rhythm_timeout_and_refresh_cancel_previous_round(local_env):
    env = await local_env("neal_47")
    page = env.unwrapped._page
    await freeze_clock(page)
    started = await start_rhythm(env)
    await page.clock.run_for(math.ceil((started["align_age_seconds"] + 0.25) * 1000))
    missed = [event for event in await game_events(page) if event["kind"] == "rhythm_missed"]
    assert [(event["note"], event["reason"]) for event in missed] == [(0, "late")]
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert await page.locator(".neal-rhythm-note").count() == 0
    boundary = len(await game_events(page))
    await page.clock.run_for(5000)
    assert not any(
        event["kind"].startswith("rhythm_") for event in (await game_events(page))[boundary:]
    )
    await start_rhythm(env)
    boundary = len(await game_events(page))
    await page.clock.run_for(1000)
    assert not any(
        event["kind"] == "rhythm_missed" for event in (await game_events(page))[boundary:]
    )
    assert await page.get_by_role("button", name="Verify", exact=True).is_disabled()


@pytest.mark.live
async def test_rhythm_stalled_frame_uses_actual_spawn_time(local_env):
    env = await local_env("neal_47")
    page = env.unwrapped._page
    await freeze_clock(page)
    started = await start_rhythm(env)
    # fast_forward fires a delayed frame once, instead of fabricating intermediate frames.
    await page.clock.fast_forward(800)
    spawned = [event for event in await game_events(page) if event["kind"] == "rhythm_note_spawned"]
    second = next(event for event in spawned if event["note"] == 1)
    assert second["spawn_elapsed_seconds"] == pytest.approx(0.8)
    assert (
        second["spawn_elapsed_seconds"]
        > second["chart_time_seconds"] - started["align_age_seconds"] + 0.3
    )
    target_ms = math.ceil((second["spawn_elapsed_seconds"] + started["align_age_seconds"]) * 1000)
    await page.clock.run_for(target_ms - 800)
    await page.keyboard.press("ArrowDown")
    judged = [event for event in await game_events(page) if event["kind"] == "rhythm_judged"]
    assert len(judged) == 1 and judged[0]["note"] == 1 and judged[0]["hit"]
    assert abs(judged[0]["timing_error_seconds"]) < 0.001


@pytest.mark.live
async def test_rhythm_early_loss_then_all_336_source_notes_and_chords(local_env):
    env = await local_env("neal_47", max_steps=40, max_seconds=240)
    page = env.unwrapped._page
    await freeze_clock(page)
    verify = page.get_by_role("button", name="Verify", exact=True)
    await start_rhythm(env)
    await page.clock.run_for(11000)
    assert await page.get_by_role("button", name="Start music").is_visible()
    assert await verify.is_disabled()
    assert any(
        event["kind"] == "rhythm_ended" and event["early"] for event in await game_events(page)
    )
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    boundary = len(await game_events(page))
    started = await start_rhythm(env)
    chart = json.loads((REFERENCE_ROOT / "level47_chart.json").read_text())
    assert len(chart) == 336
    assert chart[251]["time"] > chart[253]["time"]  # The canonical source order is not sorted.
    assert sum(count > 1 for count in Counter(note["time"] for note in chart).values()) == 49
    previous_ms = 0
    # Sort only the GUI action plan, never the source chart or expected note identities.
    action_plan = sorted(chart, key=lambda note: note["time"])
    for seconds, chord in groupby(action_plan, key=lambda note: note["time"]):
        target_ms = max(round(seconds * 1000) + 30, math.ceil(started["align_age_seconds"] * 1000))
        await page.clock.run_for(target_ms - previous_ms)
        previous_ms = target_ms
        keys = ["Arrow" + note["key"].capitalize() for note in chord]
        for key in keys:
            await page.keyboard.down(key)
        for key in keys:
            await page.keyboard.up(key)
    assert await verify.is_disabled()
    await page.clock.run_for(102100 - previous_ms)
    assert not await verify.is_disabled()
    events = (await game_events(page))[boundary:]
    judged = [event for event in events if event["kind"] == "rhythm_judged"]
    assert len(judged) == 336 and all(event["hit"] for event in judged)
    assert {event["note"] for event in judged} == set(range(336))
    assert len({event["elapsed_ms"] for event in judged[-4:]}) == 1
    assert {event["lane"] for event in judged[-4:]} == {0, 1, 2, 3}
    spawned = [event for event in events if event["kind"] == "rhythm_note_spawned"]
    assert len(spawned) == 336
    assert [event["note"] for event in spawned].index(253) < [
        event["note"] for event in spawned
    ].index(251)
    final = [event for event in events if event["kind"] == "rhythm_ended"][-1]
    assert not final["early"] and final["hits"] == 336 and final["missed"] == 0
    assert final["chart"] == "source_chart" and final["media"] == "local_synthesized_media"
    assert final["end_signal"] == "local_synthesized_media_end"
    assert not final["original_media_end_observed"]
    result = await click(env, verify)
    assert result.terminated and result.reward == 1
    assert env.unwrapped.state.reason == "source_chart_local_synthesized_media_verified"
