"""Source-derived mole boundaries, using GUI input and an explicit test clock.

These tests are not model gameplay or original-site execution. Seeds and expected
times come from the authored local PRNG, without replacing that RNG or injecting
game state. DOM classes and diagnostic events are test-only observation oracles.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pytest

from examples.not_a_robot.env import NotARobotEnv
from examples.not_a_robot.tests.test_local_tasks import click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env

pytestmark = pytest.mark.live


@pytest.fixture
async def mole_clock_env(local_env, monkeypatch):
    """Pause the owned test page before the normal navigation/render lifecycle."""
    original = NotARobotEnv._open_local_task

    async def open_with_clock(env):
        epoch = datetime(2026, 1, 1, tzinfo=UTC)
        await env._page.clock.install(time=epoch)
        await env._page.clock.pause_at(epoch)
        await original(env)

    monkeypatch.setattr(NotARobotEnv, "_open_local_task", open_with_clock)
    return local_env


async def mole_events(page, kind):
    """Read append-only diagnostics without revealing them to a model client."""
    snapshot = await page.evaluate("window.syntheticTask.snapshot()")
    return [event for event in snapshot["events"] if event["kind"] == kind]


async def advance_to(page, elapsed):
    """Observe at ceil(due): Playwright's clock retains fractional timer delays."""
    now = (await page.evaluate("window.syntheticTask.snapshot()"))["elapsed_ms"]
    target = math.ceil(elapsed)
    assert target >= now
    await page.clock.run_for(target - now)


async def hit_schedule(env, schedule):
    """Accumulate real hits at known seeded times using normal GUI clicks."""
    page = env.unwrapped._page
    for count, (at, index) in enumerate(schedule, start=1):
        await advance_to(page, at)
        tile = page.locator(".neal-tile").nth(index)
        assert "mole-visible" in (await tile.get_attribute("class")).split()
        await click(env, tile)
        assert env.unwrapped.state.progress == count


# Independent cumulative sums of the authored PRNG's unrounded timer delays.
# Event timestamps round callback time; these are not original wall-clock data.
ZERO_HITS = [(0, 5), (500.6594914011657, 4), (1293.063787277788, 9), (2883.1619527190924, 11)]
OVERLAP_HITS = [(0, 10), (973.4644619747996, 4), (1884.0996464714408, 7), (4178.360729943961, 8)]
OVERLAP_LAST_SPAWN = 6557.903371285647


async def reach_threshold_overlap(env):
    """Reach four hits, then observe two naturally overlapping live moles."""
    page = env.unwrapped._page
    await hit_schedule(env, OVERLAP_HITS)
    await advance_to(page, OVERLAP_LAST_SPAWN)
    assert env.unwrapped.state.progress == 4
    for index in (13, 2):
        assert (
            "mole-visible"
            in (await page.locator(".neal-tile").nth(index).get_attribute("class")).split()
        )


@pytest.mark.parametrize("seed,index,visible", [(5, 12, 1), (36, 16, 0)])
async def test_mount_spawns_immediately_with_source_pool_and_grid_mismatch(
    mole_clock_env, seed, index, visible
):
    env = await mole_clock_env("neal_10", seed=seed)
    page = env.unwrapped._page
    assert await page.locator(".neal-tile").count() == 16
    shown = await mole_events(page, "mole_shown")
    assert len(shown) == 1
    assert (shown[0]["index"], shown[0]["elapsed_ms"], shown[0]["duration_ms"]) == (
        index,
        0,
        1500,
    )
    assert await page.locator(".mole-visible").count() == visible
    assert (
        "mole-visible"
        not in (await page.locator(".neal-tile").nth(0).get_attribute("class")).split()
    )
    assert env.unwrapped.state.progress == 0


@pytest.mark.parametrize(
    "seed,hits,at,index,lifetime",
    [
        (5, [], 0, 12, 1500),
        (1, [(0, 11)], 505.47144236043096, 8, 1000),
        (0, ZERO_HITS[:2], 1293.063787277788, 9, 750),
        (0, ZERO_HITS[:3], 2883.1619527190924, 11, 625),
        (0, ZERO_HITS, 4681.13271240145, 8, 600),
    ],
)
async def test_lifetime_is_fixed_by_hit_count_at_spawn(
    mole_clock_env, seed, hits, at, index, lifetime
):
    env = await mole_clock_env("neal_10", seed=seed)
    page = env.unwrapped._page
    await hit_schedule(env, hits)
    await advance_to(page, at)
    shown = (await mole_events(page, "mole_shown"))[-1]
    assert (shown["index"], shown["elapsed_ms"], shown["duration_ms"]) == (
        index,
        math.floor(at + 0.5),
        lifetime,
    )
    tile = page.locator(".neal-tile").nth(index)
    await advance_to(page, math.ceil(at + lifetime) - 1)
    assert "mole-visible" in (await tile.get_attribute("class")).split()
    await advance_to(page, at + lifetime)
    assert "mole-visible" not in (await tile.get_attribute("class")).split()
    hidden = [event for event in await mole_events(page, "mole_hidden") if event["index"] == index]
    assert hidden[-1]["elapsed_ms"] == math.floor(at + lifetime + 0.5)
    await gui(env, [{"action": "screenshot"}])
    assert env.unwrapped.state.progress == len(hits)


async def test_spawn_rng_schedule_allows_overlap_and_excludes_active_cells(mole_clock_env):
    env = await mole_clock_env("neal_10", seed=0)
    page = env.unwrapped._page
    await advance_to(page, 500)
    assert len(await mole_events(page, "mole_shown")) == 1
    await advance_to(page, 500.6594914011657)
    assert await page.locator(".mole-visible").count() == 2
    await advance_to(page, 1293.063787277788)
    assert await page.locator(".mole-visible").count() == 3
    await advance_to(page, 2883.1619527190924)
    shown = await mole_events(page, "mole_shown")
    assert [(event["index"], event["elapsed_ms"]) for event in shown] == [
        (5, 0),
        (4, 501),
        (9, 1293),
        (10, 2883),
    ]
    assert all(event["index"] != 0 for event in shown)
    assert env.unwrapped.state.progress == 0


async def test_existing_mole_keeps_its_spawn_lifetime_when_score_changes(mole_clock_env):
    env = await mole_clock_env("neal_10", seed=0)
    page = env.unwrapped._page
    await advance_to(page, 500.6594914011657)
    await click(env, page.locator(".neal-tile").nth(5))
    assert env.unwrapped.state.progress == 1
    # ID 4 was already born at score zero, so it retains its 1500 ms life.
    target = page.locator(".neal-tile").nth(4)
    await advance_to(page, 2000)
    assert "mole-visible" in (await target.get_attribute("class")).split()
    await advance_to(page, 2000.6594914011657)
    assert "mole-visible" not in (await target.get_attribute("class")).split()


async def test_empty_selection_and_preselected_mole_do_not_count_as_hits(mole_clock_env):
    env = await mole_clock_env("neal_10", seed=0)
    page = env.unwrapped._page
    empty = page.locator(".neal-tile").nth(0)
    await click(env, empty)
    assert await empty.get_attribute("aria-pressed") == "true"
    assert "mole-whacked" not in (await empty.get_attribute("class")).split()
    assert not await empty.locator(".neal-mole").is_visible()
    assert env.unwrapped.state.progress == env.unwrapped.state.mistakes == 0
    target = page.locator(".neal-tile").nth(4)
    await click(env, target)
    await advance_to(page, 500.6594914011657)
    assert "mole-visible" in (await target.get_attribute("class")).split()
    assert await target.get_attribute("aria-pressed") == "true"
    await click(env, target)
    assert await target.get_attribute("aria-pressed") == "false"
    assert env.unwrapped.state.progress == 0
    assert "mole-visible" in (await target.get_attribute("class")).split()
    await click(env, target)
    assert env.unwrapped.state.progress == 1
    assert "mole-whacked" in (await target.get_attribute("class")).split()
    assert "mole-visible" not in (await target.get_attribute("class")).split()
    await click(env, target)
    assert env.unwrapped.state.progress == 0
    assert await target.get_attribute("aria-pressed") == "false"
    assert "mole-whacked" not in (await target.get_attribute("class")).split()


@pytest.mark.parametrize("hit,remaining,persistent", [(13, 2, True), (2, 13, False)])
async def test_threshold_cancels_latest_hide_only_and_verify_accepts_more_than_five(
    mole_clock_env, hit, remaining, persistent
):
    env = await mole_clock_env("neal_10", seed=22)
    page = env.unwrapped._page
    await reach_threshold_overlap(env)
    await click(env, page.locator(".neal-tile").nth(hit))
    assert env.unwrapped.state.progress == 5 and env.unwrapped.state.status == "in_progress"
    verify = page.get_by_role("button", name="Verify", exact=True)
    assert await verify.is_enabled()
    remaining_tile = page.locator(".neal-tile").nth(remaining)
    assert "mole-visible" in (await remaining_tile.get_attribute("class")).split()
    await page.clock.run_for(66)
    assert "mole-visible" in (await remaining_tile.get_attribute("class")).split()
    await page.clock.run_for(1)
    assert ("mole-visible" in (await remaining_tile.get_attribute("class")).split()) is persistent
    await page.clock.run_for(3000)
    assert len(await mole_events(page, "mole_shown")) == 6
    if persistent:
        assert "mole-visible" in (await remaining_tile.get_attribute("class")).split()
        await click(env, remaining_tile)
        assert env.unwrapped.state.progress == 6
    result = await click(env, verify)
    assert result.terminated and result.reward == 1 and env.unwrapped.outcome == "success"


async def test_unhitting_after_threshold_disables_verify_without_restarting_spawns(mole_clock_env):
    env = await mole_clock_env("neal_10", seed=22)
    page = env.unwrapped._page
    await reach_threshold_overlap(env)
    tile = page.locator(".neal-tile").nth(13)
    await click(env, tile)
    await click(env, tile)
    assert env.unwrapped.state.progress == 4
    assert await page.get_by_role("button", name="Verify", exact=True).is_disabled()
    await page.clock.run_for(5000)
    assert len(await mole_events(page, "mole_shown")) == 6
    assert env.unwrapped.state.status == "in_progress"
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert env.unwrapped.state.progress == 0
    assert len(await mole_events(page, "mole_shown")) == 7
    assert await page.locator('.neal-tile[aria-pressed="true"]').count() == 0
    assert await page.locator(".mole-whacked").count() == 0


async def test_refresh_cancels_older_hide_before_same_id_is_spawned_again(mole_clock_env):
    """Explicit local cleanup difference: an old callback cannot hide the new round."""
    env = await mole_clock_env("neal_10", seed=15)
    page = env.unwrapped._page
    await advance_to(page, 1308.2494349218905)
    assert [
        (event["index"], event["elapsed_ms"]) for event in await mole_events(page, "mole_shown")
    ] == [
        (4, 0),
        (8, 1308),
    ]
    await click(env, page.locator(".neal-tile").nth(0))
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    shown = await mole_events(page, "mole_shown")
    assert (shown[-1]["index"], shown[-1]["elapsed_ms"]) == (4, 1309)
    assert await page.locator('.neal-tile[aria-pressed="true"]').count() == 0
    assert env.unwrapped.state.progress == 0
    target = page.locator(".neal-tile").nth(4)
    await advance_to(page, 1500)
    # The first round's untracked hide would fire at t=1500 without cleanup.
    assert "mole-visible" in (await target.get_attribute("class")).split()
    await advance_to(page, 2808)
    assert "mole-visible" in (await target.get_attribute("class")).split()
    await advance_to(page, 2809)
    assert "mole-visible" not in (await target.get_attribute("class")).split()
    hidden = [event for event in await mole_events(page, "mole_hidden") if event["index"] == 4]
    assert [event["elapsed_ms"] for event in hidden] == [2809]


async def test_success_keeps_observed_state_stable_and_close_finalizes(mole_clock_env):
    env = await mole_clock_env("neal_10", seed=22)
    page = env.unwrapped._page
    await reach_threshold_overlap(env)
    # Hitting the newest mole leaves the older hide scheduled 67 ms later.
    await click(env, page.locator(".neal-tile").nth(2))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and result.reward == 1
    # This observes terminal stability, not whether every timer was cancelled.
    frozen = await page.evaluate("window.syntheticTask.snapshot()")
    await page.clock.run_for(10000)
    assert await page.evaluate("window.syntheticTask.snapshot()") == frozen
    attempt = env.unwrapped.attempt_dir
    browser = env.unwrapped._browser
    await env.close()
    assert page.is_closed() and not browser.is_connected()
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["outcome"] == "success"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
