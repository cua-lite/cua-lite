"""Owned-browser calibration for future driving and continuous drawing tasks.

Test-only DOM listeners observe genuine browser inputs and animation frames;
they neither change the game nor expose a new tool to model clients.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from examples.not_a_robot.tests.test_local_tasks import gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env


@pytest.mark.live
async def test_held_keys_span_observations_and_model_idle_time(local_env):
    env = await local_env("input", max_steps=30)
    raw, page = env.unwrapped, env.unwrapped._page
    await page.evaluate("""() => {
      const metrics = {events: [], heldFrames: 0, held: false};
      window.inputCalibration = metrics;
      for (const type of ['keydown', 'keyup']) document.addEventListener(type, event => {
        if (event.key !== 'ArrowUp') return;
        event.preventDefault();
        metrics.held = type === 'keydown';
        metrics.events.push({type, key: event.key, trusted: event.isTrusted,
          time: performance.now()});
      });
      const tick = () => {
        if (metrics.held) metrics.heldFrames++;
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }""")
    await gui(env, [{"action": "hold_key", "keys": ["up"], "duration": 0.25}])
    measured = await page.evaluate("window.inputCalibration")
    assert [event["type"] for event in measured["events"]] == ["keydown", "keyup"]
    assert all(event["trusted"] for event in measured["events"])
    assert measured["events"][1]["time"] - measured["events"][0]["time"] >= 230
    assert not measured["held"] and measured["heldFrames"] >= 2

    await gui(env, [{"action": "key_down", "keys": ["up"]}])
    before = await page.evaluate("window.inputCalibration.heldFrames")
    # No env call runs here, like a model taking time between observations.
    await asyncio.sleep(0.2)
    after = await page.evaluate("window.inputCalibration.heldFrames")
    assert after >= before + 2
    await gui(env, [{"action": "screenshot"}])
    assert await page.evaluate("window.inputCalibration.held")
    await gui(env, [{"action": "key_up", "keys": ["up"]}])
    stopped = await page.evaluate("window.inputCalibration.heldFrames")
    await asyncio.sleep(0.1)
    assert await page.evaluate("window.inputCalibration.heldFrames") == stopped
    measured = await page.evaluate("window.inputCalibration")
    assert [event["type"] for event in measured["events"]] == [
        "keydown",
        "keyup",
        "keydown",
        "keyup",
    ]
    raw.recorder.emit("input_calibration", kind="held_key_wall_clock", measurement=measured)
    first = raw.attempt_dir
    await env.reset()
    assert await raw._page.evaluate("window.inputCalibration === undefined")
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]


@pytest.mark.live
async def test_mouse_press_path_release_is_continuous_and_recorded(local_env):
    env = await local_env("input", max_steps=30)
    raw, page = env.unwrapped, env.unwrapped._page
    await page.evaluate("""() => {
      window.pointerCalibration = [];
      for (const type of ['mousedown', 'mousemove', 'mouseup', 'dragstart']) {
        document.addEventListener(type, event => window.pointerCalibration.push({
          type, x: event.clientX, y: event.clientY, buttons: event.buttons,
          trusted: event.isTrusted, time: performance.now()
        }));
      }
    }""")
    # Stay below the footer: crossing selectable text would test browser-native
    # text drag-and-drop, which suppresses mousemove, not a drawing surface.
    await gui(env, [{"action": "mouse_down", "coordinate": [850, 900]}])
    await gui(env, [{"action": "mouse_move", "coordinate": [880, 920]}])
    await gui(env, [{"action": "wait", "duration": 0.1}])
    await gui(env, [{"action": "mouse_move", "coordinate": [920, 950]}])
    await gui(env, [{"action": "mouse_up"}])
    events = await page.evaluate("window.pointerCalibration")
    assert all(event["trusted"] for event in events)
    assert not any(event["type"] == "dragstart" for event in events)
    down = next(index for index, event in enumerate(events) if event["type"] == "mousedown")
    up = next(index for index, event in enumerate(events) if event["type"] == "mouseup")
    moves = [event for event in events[down + 1 : up] if event["type"] == "mousemove"]
    assert len(moves) == 2 and all(event["buttons"] == 1 for event in moves)
    assert events[up]["buttons"] == 0
    assert events[up]["time"] - events[down]["time"] >= 90

    await page.evaluate("window.pointerCalibration.length = 0")
    await gui(env, [{"action": "drag", "start_coordinate": [850, 900], "coordinate": [950, 950]}])
    drag = await page.evaluate("window.pointerCalibration")
    assert not any(event["type"] == "dragstart" for event in drag)
    held_moves = [event for event in drag if event["type"] == "mousemove" and event["buttons"]]
    assert len(held_moves) == 20
    assert all(event["trusted"] and event["buttons"] == 1 for event in held_moves)
    assert [event["x"] for event in held_moves] == sorted(event["x"] for event in held_moves)
    assert [event["y"] for event in held_moves] == sorted(event["y"] for event in held_moves)
    down = next(event for event in drag if event["type"] == "mousedown")
    up = next(event for event in drag if event["type"] == "mouseup")
    assert up["time"] - down["time"] >= 380
    raw.recorder.emit("input_calibration", kind="continuous_mouse_path", measurement=drag)
    attempt = raw.attempt_dir
    await env.close()
    saved = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    primitives = [event["data"]["call"] for event in saved if event["type"] == "input_started"]
    assert primitives.count("mouse.down") == primitives.count("mouse.up") == 2
    assert primitives.count("mouse.move") == 24
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
