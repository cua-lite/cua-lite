"""Rhythm feedback through real keys, pointers, CSS frames and screenshots.

Synthetic time checks JavaScript removal timers only. Native CSS animationend
events are observed, never dispatched, and no evaluator state is assigned.
These local GUI regressions do not demonstrate screenshot-model performance.
"""

from __future__ import annotations

import asyncio
import json
import math
from contextlib import suppress
from io import BytesIO

import pytest
from PIL import Image, ImageChops

from examples.not_a_robot.tests.test_local_tasks import click
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env
from examples.not_a_robot.tests.test_puzzles_expansion import freeze_clock, game_events
from examples.not_a_robot.tests.test_rhythm_source import start_rhythm

pytestmark = pytest.mark.live


async def note_style(note):
    return await note.evaluate(
        "n => { const s=getComputedStyle(n); return {connected:n.isConnected,"
        "classes:[...n.classList], opacity:parseFloat(s.opacity),"
        "scale:s.scale==='none'?1:parseFloat(s.scale),"
        "playState:s.animationPlayState, transition:s.transitionDuration,"
        "top:n.getBoundingClientRect().top}; }"
    )


async def board_image(page, path):
    box = await page.locator(".neal-rhythm-board").bounding_box()
    assert box
    return Image.open(BytesIO(await page.screenshot(clip=box, path=str(path)))).convert("RGB")


async def test_rhythm_hit_pauses_fades_and_removes_at_three_hundred_ms(local_env, tmp_path):
    env = await local_env("neal_47")
    page = env.unwrapped._page
    await freeze_clock(page)
    started = await start_rhythm(env)
    note = await page.locator(".neal-rhythm-note").first.element_handle()
    assert note
    await page.clock.run_for(math.ceil(started["align_age_seconds"] * 1000))
    await asyncio.sleep(0.25)  # Native CSS has moved the glyph into the visible board.
    before = await board_image(page, tmp_path / "before-hit.png")
    await page.keyboard.down("ArrowDown")
    await page.keyboard.down("ArrowDown")
    await board_image(page, tmp_path / "hit-first-frame.png")
    initial = await note_style(note)
    assert initial["connected"] and "played" in initial["classes"]
    assert initial["playState"] == "paused"
    assert all(float(value.removesuffix("s")) == 0.2 for value in initial["transition"].split(", "))
    await asyncio.sleep(0.25)
    faded = await note_style(note)
    assert faded["connected"] and faded["opacity"] == pytest.approx(0, abs=0.01)
    assert faded["scale"] == pytest.approx(1.3, abs=0.01)
    after = await board_image(page, tmp_path / "hit-faded.png")
    assert ImageChops.difference(before, after).getbbox() is not None
    judged = [event for event in await game_events(page) if event["kind"] == "rhythm_judged"]
    assert len(judged) == 1 and judged[0]["note"] == 0 and judged[0]["hit"]
    await page.keyboard.up("ArrowDown")
    await page.clock.run_for(299)
    assert (await note_style(note))["connected"]
    await page.clock.run_for(1)
    assert not (await note_style(note))["connected"]


@pytest.mark.parametrize("reason", ["early", "late"])
async def test_rhythm_miss_keeps_falling_until_real_animation_end(local_env, tmp_path, reason):
    env = await local_env("neal_47")
    page = env.unwrapped._page
    await freeze_clock(page)
    started = await start_rhythm(env)
    note = await page.locator(".neal-rhythm-note").first.element_handle()
    assert note
    await asyncio.sleep(0.25)
    await board_image(page, tmp_path / "before-miss.png")
    if reason == "early":
        await page.keyboard.press("ArrowDown")
        events = [event for event in await game_events(page) if event["kind"] == "rhythm_judged"]
        assert len(events) == 1 and events[0]["note"] == 0 and not events[0]["hit"]
    else:
        await page.clock.run_for(math.ceil((started["align_age_seconds"] + 0.25) * 1000))
        events = [event for event in await game_events(page) if event["kind"] == "rhythm_missed"]
        assert [(event["note"], event["reason"]) for event in events] == [(0, "late")]
    await board_image(page, tmp_path / "miss-first-frame.png")
    initial = await note_style(note)
    assert initial["connected"] and "missed" in initial["classes"]
    assert initial["playState"] == "running"
    ended = asyncio.create_task(
        note.evaluate(
            "n => new Promise(resolve => n.addEventListener('animationend',"
            "e=>resolve({trusted:e.isTrusted,name:e.animationName,elapsed:e.elapsedTime}),"
            "{once:true}))"
        )
    )
    try:
        await asyncio.sleep(0.25)
        later = await note_style(note)
        assert later["connected"] and later["playState"] == "running"
        assert later["opacity"] == pytest.approx(0.4, abs=0.01)
        assert later["scale"] == pytest.approx(0.8, abs=0.01)
        assert later["top"] > initial["top"] + 5
        await board_image(page, tmp_path / "miss-continues-falling.png")
        event = await asyncio.wait_for(ended, timeout=5)
        assert event == {"trusted": True, "name": "neal-source-rhythm-drop", "elapsed": 2}
        assert (await note_style(note))["connected"]
        await page.clock.run_for(299)
        assert (await note_style(note))["connected"]
        await page.clock.run_for(1)
        assert not (await note_style(note))["connected"]
    finally:
        if not ended.done():
            ended.cancel()
        with suppress(asyncio.CancelledError):
            await ended


async def test_rhythm_keyboard_pressed_feedback_while_idle(local_env, tmp_path):
    env = await local_env("neal_47")
    page = env.unwrapped._page
    await freeze_clock(page)
    target = page.get_by_role("button", name="ArrowLeft", exact=True)
    baseline = await target.bounding_box()
    neutral = await board_image(page, tmp_path / "keyboard-neutral.png")
    await page.keyboard.down("ArrowLeft")
    await page.keyboard.down("ArrowLeft")
    await asyncio.sleep(0.15)
    held = await board_image(page, tmp_path / "keyboard-held.png")
    assert "pressed" in (await target.get_attribute("class")).split()
    assert (await target.bounding_box())["width"] == pytest.approx(baseline["width"] * 1.06, abs=1)
    assert ImageChops.difference(neutral, held).getbbox() is not None
    assert not any(event["kind"] == "rhythm_judged" for event in await game_events(page))
    await page.keyboard.up("ArrowLeft")
    await asyncio.sleep(0.15)
    released = await board_image(page, tmp_path / "keyboard-released.png")
    assert "pressed" not in (await target.get_attribute("class")).split()
    assert (await target.bounding_box())["width"] == pytest.approx(baseline["width"], abs=0.1)
    assert ImageChops.difference(held, released).getbbox() is not None


async def test_rhythm_pointer_pressed_and_release_outside_target(local_env, tmp_path):
    env = await local_env("neal_47")
    page = env.unwrapped._page
    await freeze_clock(page)
    started = await start_rhythm(env)
    await page.clock.run_for(math.ceil(started["align_age_seconds"] * 1000))
    target = page.get_by_role("button", name="ArrowDown", exact=True)
    bounds = await target.bounding_box()
    await page.mouse.move(bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2)
    await page.mouse.down()
    await asyncio.sleep(0.15)
    await board_image(page, tmp_path / "pointer-held.png")
    assert "pressed" in (await target.get_attribute("class")).split()
    assert (await target.bounding_box())["width"] == pytest.approx(bounds["width"] * 1.06, abs=1)
    await page.mouse.move(2, 2)
    await page.mouse.up()
    await asyncio.sleep(0.15)
    await board_image(page, tmp_path / "pointer-released-outside.png")
    assert "pressed" not in (await target.get_attribute("class")).split()
    assert (await target.bounding_box())["width"] == pytest.approx(bounds["width"], abs=0.1)
    judged = [event for event in await game_events(page) if event["kind"] == "rhythm_judged"]
    assert len(judged) == 1 and judged[0]["note"] == 0 and judged[0]["hit"]


async def test_rhythm_early_end_does_not_freeze_existing_css_notes(local_env, tmp_path):
    env = await local_env("neal_47")
    page = env.unwrapped._page
    await freeze_clock(page)
    await start_rhythm(env)
    await page.clock.run_for(11000)
    events = await game_events(page)
    ended_rounds = [event for event in events if event["kind"] == "rhythm_ended"]
    assert len(ended_rounds) == 1 and ended_rounds[0]["early"]
    judged = [event for event in events if event["kind"] in ("rhythm_judged", "rhythm_missed")]
    note = await page.locator(".neal-rhythm-note").last.element_handle()
    assert note
    initial = await note_style(note)
    assert initial["connected"] and initial["playState"] == "running"
    await board_image(page, tmp_path / "early-end-existing-notes.png")
    animation_end = asyncio.create_task(
        note.evaluate(
            "n => new Promise(resolve => n.addEventListener('animationend',"
            "e=>resolve({trusted:e.isTrusted,name:e.animationName,elapsed:e.elapsedTime}),"
            "{once:true}))"
        )
    )
    try:
        await asyncio.sleep(0.2)
        assert (await note_style(note))["top"] > initial["top"] + 5
        await board_image(page, tmp_path / "early-end-notes-keep-falling.png")
        event = await asyncio.wait_for(animation_end, timeout=5)
        assert event == {"trusted": True, "name": "neal-source-rhythm-drop", "elapsed": 2}
        await page.clock.run_for(299)
        assert (await note_style(note))["connected"]
        await page.clock.run_for(1)
        assert not (await note_style(note))["connected"]
        after = await game_events(page)
        assert [event for event in after if event["kind"] == "rhythm_ended"] == ended_rounds
        assert [
            event for event in after if event["kind"] in ("rhythm_judged", "rhythm_missed")
        ] == judged
    finally:
        if not animation_end.done():
            animation_end.cancel()
        with suppress(asyncio.CancelledError):
            await animation_end


async def test_rhythm_refresh_and_close_clean_pending_feedback(local_env, tmp_path):
    env = await local_env("neal_47")
    raw, page = env.unwrapped, env.unwrapped._page
    await freeze_clock(page)
    started = await start_rhythm(env)
    old_note = await page.locator(".neal-rhythm-note").first.element_handle()
    await page.clock.run_for(math.ceil(started["align_age_seconds"] * 1000))
    await page.keyboard.down("ArrowDown")
    await board_image(page, tmp_path / "pending-before-refresh.png")
    assert (await note_style(old_note))["connected"]
    assert await page.locator(".neal-rhythm-target.pressed").count() == 1
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    assert await page.locator(".neal-rhythm-note").count() == 0
    assert await page.locator(".neal-rhythm-target.pressed").count() == 0
    assert not (await note_style(old_note))["connected"]
    await board_image(page, tmp_path / "cleared-after-refresh.png")
    boundary = len(await game_events(page))
    await page.clock.run_for(5000)
    assert not any(
        event["kind"].startswith("rhythm_") for event in (await game_events(page))[boundary:]
    )
    await page.keyboard.up("ArrowDown")
    restarted = await start_rhythm(env)
    new_note = await page.locator(".neal-rhythm-note").first.element_handle()
    await page.clock.run_for(300)
    assert (await note_style(new_note))["connected"]
    await page.clock.run_for(math.ceil(restarted["align_age_seconds"] * 1000) - 300)
    await page.keyboard.down("ArrowDown")
    assert (await note_style(new_note))["connected"]
    await env.close()
    assert page.is_closed()
    manifest = json.loads((raw.attempt_dir / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    assert manifest["data"]["cleanup_errors"] == []
    records = [
        json.loads(line) for line in (raw.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    assert not any(record["type"] == "page_error" for record in records)
