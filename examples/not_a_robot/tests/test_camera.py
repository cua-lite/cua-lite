"""Unavailable-device reference path; these tests never request camera access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.not_a_robot.tests.test_local_tasks import click
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env


def test_camera_reference_source_contains_no_media_request_or_recognition():
    source = Path(__file__).parents[1].joinpath("local/neal_camera.js").read_text()
    for unavailable_api in (
        "getUserMedia",
        "getDisplayMedia",
        "enumerateDevices",
        "requestPermission",
        "navigator.permissions",
        "navigator.mediaDevices",
        'createElement("video")',
        'createElement("canvas")',
    ):
        assert unavailable_api not in source


@pytest.mark.live
async def test_camera_unavailable_reference_verify_without_device_or_permission_request(local_env):
    env = await local_env("neal_39")
    raw, page = env.unwrapped, env.unwrapped._page
    # Instrument a fresh document before its scripts execute. Any accidental
    # media request fails the test without ever reaching the real device API.
    await page.add_init_script(
        """window.testMediaRequests = [];
        for (const name of ['getUserMedia', 'getDisplayMedia', 'enumerateDevices']) {
          Object.defineProperty(navigator.mediaDevices, name, {value: async () => {
            window.testMediaRequests.push(name);
            throw new Error('Media access is forbidden in this reference test');
          }});
        }"""
    )
    await page.reload(wait_until="load")
    await page.wait_for_function("window.syntheticTask !== undefined")
    assert await page.locator(".neal-game").get_attribute("data-level") == "39"
    assert await page.locator(".neal-heading").inner_text() == (
        "Please showcase the emotion\nhappiness"
    )
    assert await page.get_by_role("progressbar", name="HAPPY").get_attribute("aria-valuenow") == "0"
    assert await page.locator(".neal-happiness-meter").inner_text() == "0% HAPPY"
    assert await page.get_by_text(
        "Please enable your camera to complete facial exam", exact=True
    ).is_visible()
    assert await page.locator(".neal-camera-unavailable").get_attribute("aria-label") == (
        "Unable to play media."
    )
    assert await page.locator(".neal-game button").count() == 2
    assert await page.locator(".neal-game video, .neal-game canvas, .neal-game input").count() == 0
    before_permission = await page.evaluate(
        "async () => (await navigator.permissions.query({name: 'camera'})).state"
    )
    assert before_permission != "granted"
    await page.screenshot(path=str(raw.attempt_dir / "initial.png"))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and result.reward == 1
    assert raw.state.reason == "reference_camera_unavailable_path_completed"
    assert raw.state.progress == 1
    assert await page.locator(".neal-happiness-meter").inner_text() == "0% HAPPY"
    assert await page.get_by_text(
        "Please enable your camera to complete facial exam", exact=True
    ).is_visible()
    assert await page.evaluate("window.testMediaRequests") == []
    assert (
        await page.evaluate(
            "async () => (await navigator.permissions.query({name: 'camera'})).state"
        )
        == before_permission
    )
    await page.screenshot(path=str(raw.attempt_dir / "success.png"))
    state = await page.evaluate("window.syntheticTask.snapshot()")
    verified = [
        event for event in state["events"] if event["kind"] == "camera_unavailable_path_verified"
    ]
    assert len(verified) == 1
    assert verified[0]["displayed_happiness_percent"] == 0
    assert not verified[0]["camera_flow_tested"]
    assert not verified[0]["facial_expression_detection_verified"]
    attempt = raw.attempt_dir
    await env.close()
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["outcome"] == "success"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    assert not any(event["type"] == "page_error" for event in events)
    success = [
        event["data"]
        for event in events
        if event["type"] == "game_event" and event["data"]["kind"] == "success"
    ]
    assert len(success) == 1
    assert success[0]["reason"] == "reference_camera_unavailable_path_completed"


@pytest.mark.live
async def test_camera_unavailable_refresh_and_independent_reset_do_not_complete(local_env):
    env = await local_env("neal_39")
    raw, page = env.unwrapped, env.unwrapped._page
    initial = await page.locator(".neal-game").inner_html()
    result = await click(env, page.get_by_role("button", name="Refresh challenge", exact=True))
    assert not result.terminated and raw.state.progress == 0
    assert await page.locator(".neal-game").inner_html() == initial
    state = await page.evaluate("window.syntheticTask.snapshot()")
    assert any(event["kind"] == "camera_unavailable_path_reset" for event in state["events"])
    assert not any(event["kind"] == "success" for event in state["events"])
    first_attempt = raw.attempt_dir
    await env.reset()
    assert raw.attempt_dir != first_attempt
    assert raw.state.progress == raw.state.mistakes == 0
    assert await raw._page.locator(".neal-game").inner_html() == initial
    manifest = json.loads((first_attempt / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
