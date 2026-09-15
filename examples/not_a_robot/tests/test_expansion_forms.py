"""Scripted GUI oracles for captured forms, not model gameplay scores."""

from __future__ import annotations

import json

import pytest

from examples.not_a_robot.tests.test_local_tasks import click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env


@pytest.mark.live
@pytest.mark.parametrize("task_id", ["neal_14", "neal_24", "neal_33"])
async def test_form_reference_reset_is_independent_and_game_only(local_env, task_id):
    env = await local_env(task_id)
    raw, page = env.unwrapped, env.unwrapped._page
    assert await page.locator(".neal-game").is_visible()
    assert not await page.locator(".masthead").is_visible()
    assert not await page.locator("#reference-note").is_visible()
    initial = await page.locator(".neal-game").inner_html()
    attempt = raw.attempt_dir
    if task_id == "neal_14":
        await click(env, page.get_by_role("checkbox", name="I'm potentially a robot", exact=True))
        assert await page.locator(".neal-checkbox-mark.loading").count() == 1
    else:
        await click(env, page.locator(".neal-game input"))
        await gui(env, [{"action": "type", "text": "WRONG"}])
    await env.reset()
    assert raw.attempt_dir != attempt
    assert raw.state.progress == raw.state.mistakes == 0
    assert await raw._page.locator(".neal-game").inner_html() == initial
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    assert not raw._scope_violation


@pytest.mark.live
async def test_statement_task_scrolls_past_distractors_to_exact_checkbox(local_env):
    env = await local_env("neal_14")
    raw, page = env.unwrapped, env.unwrapped._page
    boxes = page.get_by_role("checkbox")
    assert await boxes.count() == 56
    assert await boxes.nth(33).get_attribute("aria-label") == "I'm not a robot"
    assert await boxes.nth(0).get_attribute("aria-label") == "I'm a robot"
    assert await boxes.nth(55).get_attribute("aria-label") == "I'm a hot single in your area"
    target = page.get_by_role("checkbox", name="I'm not a robot", exact=True)
    assert (await target.bounding_box())["y"] > raw.display_resolution[1]
    await click(env, page.get_by_role("checkbox", name="I'm potentially a robot", exact=True))
    result = await gui(env, [{"action": "wait", "duration": 0.8}])
    assert not result.terminated and raw.state.progress == 0
    # No original rejection was observed: preserve unknown feedback in the log.
    state = await page.evaluate("window.syntheticTask.snapshot()")
    assert any(event["kind"] == "local_distractor_reset" for event in state["events"])
    assert not any(event["kind"] == "rejected" for event in state["events"])
    await gui(
        env,
        [
            {"action": "scroll", "direction": "down", "amount": 26, "coordinate": [200, 300]},
            {"action": "wait", "duration": 0.15},
        ],
    )
    bounds = await target.bounding_box()
    assert 0 < bounds["y"] < 550
    result = await click(env, target)
    assert not result.terminated
    result = await gui(env, [{"action": "wait", "duration": 0.8}])
    assert result.terminated and result.reward == 1
    assert await target.get_attribute("aria-checked") == "true"
    assert await page.locator(".neal-checkbox-mark.loading").count() == 0
    frozen = await page.evaluate("window.syntheticTask.snapshot()")
    await page.wait_for_timeout(700)
    assert await page.evaluate("window.syntheticTask.snapshot()") == frozen


@pytest.mark.live
async def test_eye_exam_rejects_wrong_stage_input_and_requires_all_four_stages(local_env):
    env = await local_env("neal_24", max_steps=70)
    raw, page = env.unwrapped, env.unwrapped._page
    await page.screenshot(path=str(raw.attempt_dir / "initial.png"))
    for stage, answer in enumerate(("EDFCZP", "8", "34"), start=1):
        assert await page.locator(".neal-game").get_attribute("data-stage") == str(stage)
        await click(env, page.locator(".neal-game input"))
        await gui(env, [{"action": "type", "text": "WRONG"}])
        result = await click(env, page.get_by_role("button", name="Verify", exact=True))
        assert not result.terminated and raw.state.progress == stage - 1
        assert raw.state.mistakes == stage
        await click(env, page.locator(".neal-game input"))
        await gui(
            env,
            [{"action": "key", "keys": ["ctrl", "a"]}, {"action": "type", "text": answer}],
        )
        result = await click(env, page.get_by_role("button", name="Verify", exact=True))
        assert not result.terminated and raw.state.progress == stage
        await page.screenshot(path=str(raw.attempt_dir / f"stage_{stage + 1}.png"))
    assert await page.locator(".neal-game").get_attribute("data-stage") == "4"
    assert not await page.locator(".neal-game input").count()
    tiles = page.locator(".neal-eye-colors .neal-tile")
    assert await tiles.count() == 16
    await click(env, tiles.nth(0))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and raw.state.mistakes == 4
    await click(env, tiles.nth(12))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and raw.state.mistakes == 5
    await click(env, tiles.nth(0))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and result.reward == 1 and raw.state.progress == 4
    state = await page.evaluate("window.syntheticTask.snapshot()")
    assert [e["stage"] for e in state["events"] if e["kind"] == "stage_completed"] == [1, 2, 3, 4]
    assert sum(e["kind"] == "success" for e in state["events"]) == 1
    attempt = raw.attempt_dir
    await env.close()
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["outcome"] == "success"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    assert any(event["type"] == "input_completed" for event in events)
    assert not any(event["type"] == "page_error" for event in events)


@pytest.mark.live
async def test_eye_exam_refresh_restarts_the_captured_stage_sequence(local_env):
    env = await local_env("neal_24")
    raw, page = env.unwrapped, env.unwrapped._page
    initial = await page.locator(".neal-game").inner_html()
    await click(env, page.locator(".neal-game input"))
    await gui(env, [{"action": "type", "text": "EDFCZP"}])
    await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert raw.state.progress == 1
    await click(env, page.get_by_role("button", name="Refresh challenge", exact=True))
    assert raw.state.progress == 0
    assert await page.locator(".neal-game").inner_html() == initial
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and raw.state.mistakes == 1


@pytest.mark.live
async def test_brand_text_uses_captured_pixels_and_exact_observed_input(local_env):
    env = await local_env("neal_33")
    raw, page = env.unwrapped, env.unwrapped._page
    assert "XWBKN" not in await page.locator(".neal-game").inner_text()
    image = page.locator(".neal-brand-picture img")
    assert await image.get_attribute("src") == "/reference_assets/level33_reference.jpg"
    assert await image.evaluate("image => [image.naturalWidth, image.naturalHeight]") == [1265, 712]
    await click(env, page.locator(".neal-game input"))
    result = await gui(env, [{"action": "type", "text": "WRONG", "press_enter": True}])
    assert not result.terminated and raw.state.mistakes == 1
    result = await gui(
        env,
        [
            {"action": "key", "keys": ["ctrl", "a"]},
            {"action": "type", "text": "XWBKN", "press_enter": True},
        ],
    )
    assert result.terminated and result.reward == 1 and raw.state.progress == 1
    assert raw.outcome == "success"
