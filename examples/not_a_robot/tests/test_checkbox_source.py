"""Checkbox source timing through recorded GUI input and a virtual test clock.

Source modules 478/1112/1096 supply the deadlines, not executable fixtures.
DOM and rendered pixels are test-only oracles. No game state or RNG is injected.
Reset/close tests prove episode isolation and archival, not individual timer calls.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import BytesIO

import pytest
from PIL import Image

from examples.not_a_robot.env import NotARobotEnv
from examples.not_a_robot.tests.test_expansion_grids import scroll_click
from examples.not_a_robot.tests.test_local_tasks import click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env

pytestmark = pytest.mark.live

CORRECT = "I'm not a robot"
WRONG = "I'm potentially a robot"


@pytest.fixture
async def checkbox_clock_env(local_env, monkeypatch):
    """Freeze only the owned page clock before normal task navigation."""
    original = NotARobotEnv._open_local_task

    async def open_with_clock(env):
        epoch = datetime(2026, 1, 1, tzinfo=UTC)
        await env._page.clock.install(time=epoch)
        await env._page.clock.pause_at(epoch)
        await original(env)

    monkeypatch.setattr(NotARobotEnv, "_open_local_task", open_with_clock)
    return local_env


async def assert_feedback_color(page, mark, color):
    """Check actual mark pixels, not just a class claiming colored feedback."""
    assert await mark.is_visible()
    bounds = await mark.bounding_box()
    image = Image.open(BytesIO(await page.screenshot(clip=bounds))).convert("RGB")
    if color == "red":
        pixels = [
            (x, y)
            for y in range(image.height)
            for x in range(image.width)
            if (rgb := image.getpixel((x, y)))[0] > 150 and rgb[1] < 130 and rgb[2] < 130
        ]
        # Both strokes of an X reach all four quadrants of the mark.
        assert {(x >= image.width / 2, y >= image.height / 2) for x, y in pixels} == {
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        }
    else:
        pixels = [
            rgb
            for rgb in image.getdata()
            if rgb[1] > 100 and rgb[1] > rgb[0] * 1.3 and rgb[1] > rgb[2] * 1.2
        ]
    assert len(pixels) > 8


@pytest.mark.parametrize("task_id", ["neal_01", "neal_14"])
async def test_correct_check_appears_at_700ms_but_completion_waits_until_1600ms(
    checkbox_clock_env, task_id
):
    env = await checkbox_clock_env(task_id)
    raw, page = env.unwrapped, env.unwrapped._page
    checkbox = page.get_by_role("checkbox", name=CORRECT, exact=True)
    mark = checkbox.locator(".neal-checkbox-mark")
    result = await scroll_click(env, checkbox)
    assert not result.terminated
    assert "loading" in (await mark.get_attribute("class")).split()
    await page.clock.run_for(699)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated and raw.state.status == "in_progress"
    assert "loading" in (await mark.get_attribute("class")).split()
    assert await checkbox.get_attribute("aria-checked") == "false"
    await page.clock.run_for(1)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated and raw.state.status == "in_progress"
    assert "loading" not in (await mark.get_attribute("class")).split()
    assert "checked" in (await mark.get_attribute("class")).split()
    assert await checkbox.get_attribute("aria-checked") == "true"
    await assert_feedback_color(page, mark, "green")
    await page.clock.run_for(899)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated and raw.outcome == "in_progress"
    await page.clock.run_for(1)
    result = await gui(env, [{"action": "screenshot"}])
    assert result.terminated and result.reward == 1 and raw.outcome == "success"
    snapshot = await page.evaluate("window.syntheticTask.snapshot()")
    loading = [event for event in snapshot["events"] if event["kind"] == "checkbox_loading"]
    success = [event for event in snapshot["events"] if event["kind"] == "success"]
    assert len(loading) == len(success) == 1
    assert success[0]["elapsed_ms"] - loading[0]["elapsed_ms"] == 1600


@pytest.mark.parametrize("task_id", ["neal_01", "neal_14"])
async def test_logo_click_uses_the_same_card_completion_path(checkbox_clock_env, task_id):
    env = await checkbox_clock_env(task_id)
    page = env.unwrapped._page
    checkbox = page.get_by_role("checkbox", name=CORRECT, exact=True)
    card = (
        page.locator(".neal-checkbox-card")
        if task_id == "neal_01"
        else page.locator(".neal-statement-card").filter(has=checkbox)
    )
    result = await scroll_click(env, card.locator(".neal-recaptcha-mark"))
    assert not result.terminated
    assert (
        "loading" in (await checkbox.locator(".neal-checkbox-mark").get_attribute("class")).split()
    )
    await page.clock.run_for(1600)
    result = await gui(env, [{"action": "screenshot"}])
    assert result.terminated and result.reward == 1


async def test_wrong_checkbox_shows_a_persistent_red_cross_after_800ms(checkbox_clock_env):
    env = await checkbox_clock_env("neal_14")
    raw, page = env.unwrapped, env.unwrapped._page
    checkbox = page.get_by_role("checkbox", name=WRONG, exact=True)
    mark = checkbox.locator(".neal-checkbox-mark")
    await click(env, checkbox)
    await page.clock.run_for(799)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated
    assert "loading" in (await mark.get_attribute("class")).split()
    assert await checkbox.get_attribute("aria-checked") == "false"
    await page.clock.run_for(1)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated and raw.state.progress == raw.state.mistakes == 0
    assert "wrong" in (await mark.get_attribute("class")).split()
    assert "loading" not in (await mark.get_attribute("class")).split()
    assert await checkbox.get_attribute("aria-checked") == "true"
    await assert_feedback_color(page, mark, "red")
    await page.clock.run_for(3000)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated and raw.outcome == "in_progress"
    assert "wrong" in (await mark.get_attribute("class")).split()
    await assert_feedback_color(page, mark, "red")
    snapshot = await page.evaluate("window.syntheticTask.snapshot()")
    assert not any(event["kind"] in {"success", "rejected"} for event in snapshot["events"])


async def test_different_wrong_cards_have_independent_800ms_deadlines(checkbox_clock_env):
    env = await checkbox_clock_env("neal_14")
    page = env.unwrapped._page
    first = page.get_by_role("checkbox", name=WRONG, exact=True)
    second = page.get_by_role("checkbox", name="I'm a chatbot", exact=True)
    first_mark, second_mark = (
        first.locator(".neal-checkbox-mark"),
        second.locator(".neal-checkbox-mark"),
    )
    await click(env, first)
    await page.clock.run_for(300)
    await click(env, second)
    await page.clock.run_for(500)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated
    assert "wrong" in (await first_mark.get_attribute("class")).split()
    assert "loading" in (await second_mark.get_attribute("class")).split()
    assert await second.get_attribute("aria-checked") == "false"
    await page.clock.run_for(299)
    assert "loading" in (await second_mark.get_attribute("class")).split()
    await page.clock.run_for(1)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated and env.unwrapped.state.progress == 0
    for checkbox, mark in ((first, first_mark), (second, second_mark)):
        assert "wrong" in (await mark.get_attribute("class")).split()
        assert await checkbox.get_attribute("aria-checked") == "true"
        await assert_feedback_color(page, mark, "red")


@pytest.mark.parametrize("task_id", ["neal_01", "neal_14"])
async def test_repeated_click_keeps_first_completion_deadline_and_emits_success_once(
    checkbox_clock_env, task_id
):
    env = await checkbox_clock_env(task_id)
    page = env.unwrapped._page
    checkbox = page.get_by_role("checkbox", name=CORRECT, exact=True)
    await scroll_click(env, checkbox)
    await page.clock.run_for(200)
    await click(env, checkbox)
    snapshot = await page.evaluate("window.syntheticTask.snapshot()")
    loading = [event for event in snapshot["events"] if event["kind"] == "checkbox_loading"]
    assert len(loading) == 2
    assert loading[1]["elapsed_ms"] - loading[0]["elapsed_ms"] == 200
    await page.clock.run_for(500)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated
    assert await checkbox.get_attribute("aria-checked") == "true"
    assert (
        "loading"
        not in (await checkbox.locator(".neal-checkbox-mark").get_attribute("class")).split()
    )
    await page.clock.run_for(900)
    result = await gui(env, [{"action": "screenshot"}])
    assert result.terminated and result.reward == 1
    frozen = await page.evaluate("window.syntheticTask.snapshot()")
    assert len([event for event in frozen["events"] if event["kind"] == "success"]) == 1
    await page.clock.run_for(1000)
    assert await page.evaluate("window.syntheticTask.snapshot()") == frozen


@pytest.mark.parametrize(
    "task_id,label,elapsed",
    [
        ("neal_01", CORRECT, 200),
        ("neal_01", CORRECT, 900),
        ("neal_14", CORRECT, 200),
        ("neal_14", CORRECT, 900),
        ("neal_14", WRONG, 200),
    ],
)
async def test_reset_isolates_pending_checkbox_timers_from_the_new_episode(
    checkbox_clock_env, task_id, label, elapsed
):
    env = await checkbox_clock_env(task_id)
    raw, page = env.unwrapped, env.unwrapped._page
    await scroll_click(env, page.get_by_role("checkbox", name=label, exact=True))
    await page.clock.run_for(elapsed)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated
    attempt = raw.attempt_dir
    await env.reset()
    assert page.is_closed() and raw.attempt_dir != attempt
    await raw._page.clock.run_for(2500)
    result = await gui(env, [{"action": "screenshot"}])
    assert not result.terminated and raw.state.progress == raw.state.mistakes == 0
    assert await raw._page.locator('.neal-checkbox-target[aria-checked="true"]').count() == 0
    assert (
        await raw._page.locator(".neal-checkbox-mark.loading, .neal-checkbox-mark.wrong").count()
        == 0
    )
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["outcome"] == "aborted"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]


@pytest.mark.parametrize(
    "task_id,label",
    [
        ("neal_01", CORRECT),
        ("neal_14", CORRECT),
        ("neal_14", WRONG),
    ],
)
async def test_close_with_pending_feedback_closes_resources_and_archives_without_success(
    checkbox_clock_env, task_id, label
):
    env = await checkbox_clock_env(task_id)
    raw, page = env.unwrapped, env.unwrapped._page
    await scroll_click(env, page.get_by_role("checkbox", name=label, exact=True))
    await page.clock.run_for(200)
    attempt, browser = raw.attempt_dir, raw._browser
    await env.close()
    assert page.is_closed() and not browser.is_connected()
    manifest = json.loads((attempt / "manifest.json").read_text())
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    assert manifest["outcome"] == "aborted"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    assert not any(
        row["type"] == "game_event" and row["data"]["kind"] == "success" for row in events
    )
