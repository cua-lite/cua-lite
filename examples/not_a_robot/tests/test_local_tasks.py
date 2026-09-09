"""Original local task contracts and browser smokes; no external websites.

Browser tests use visible DOM controls as scripted test oracles. They test the
environment, not vision-model ability. No task implementation is bypassed.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlsplit

import pytest

from examples.not_a_robot import registration  # noqa: F401
from examples.not_a_robot.env import LIVE_ENVS, NotARobotEnv
from examples.not_a_robot.local_tasks import CATALOG, LOCAL_TASKS, LocalTaskServer
from lite import gym
from lite.core.tools.calls import make_tool_call

ORIGINAL_TASKS = tuple(task_id for task_id in LOCAL_TASKS if not task_id.startswith("neal_"))


def test_original_tasks_have_distinct_registration_and_metadata():
    assert len(ORIGINAL_TASKS) == 8
    for task_id in ORIGINAL_TASKS:
        env = gym.make(f"visual_tasks@{task_id}", seed=42)
        assert env.metadata.others["source"] == "original_visual_tasks"
        assert env.metadata.others["task_id"] == task_id
        assert env.metadata.others["seed"] == 42
        assert env.metadata.others["game_version"] == CATALOG["version"]
        assert "target_level" not in env.metadata.others
        assert env.unwrapped._page is None
        if task_id in ("checkbox", "stop_signs", "wiggles"):
            assert env.metadata.others["reference"]["fidelity"] == "mechanic_adaptation"
        else:
            assert "reference" not in env.metadata.others


@pytest.mark.parametrize("seed", [-1, 4294967296, 0.5, True, "1"])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError, match="seed must be"):
        NotARobotEnv(mode="local", local_task="click", target_level=None, seed=seed)


def test_unknown_local_task_rejected():
    with pytest.raises(ValueError, match="known local_task"):
        NotARobotEnv(mode="local", local_task="unknown", target_level=None)


def test_local_server_only_serves_bundled_assets():
    server = LocalTaskServer()
    parsed = urlsplit(server.origin)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=3)
    try:
        connection.request("GET", "/?task=click&seed=1")
        response = connection.getresponse()
        body = response.read()
        assert response.status == 200
        assert b"Visual Tasks" in body
        assert "default-src 'none'" in response.getheader("Content-Security-Policy")
        assert hashlib.sha256(body).hexdigest() == server.asset_hashes["index.html"]
        for path in ("/../env.py", "/%2e%2e/env.py", "/env.py", "/local/", "/index.html"):
            connection.request("GET", path)
            response = connection.getresponse()
            response.read()
            assert response.status == 404
    finally:
        connection.close()
        server.close()
    assert not server._thread.is_alive()


@pytest.fixture
async def local_env(tmp_path):
    created = []

    async def create(task_id, seed=17, **kwargs):
        env = gym.make(
            f"visual_tasks@{task_id}",
            seed=seed,
            artifact_root=str(tmp_path),
            cursor=False,
            post_action_delay=0,
            browser_executable=os.environ.get("NEAL_BROWSER_EXECUTABLE"),
            **kwargs,
        )
        created.append(env)
        observation = await env.reset()
        assert observation.image.startswith(b"\x89PNG")
        assert env.unwrapped.state.status == "in_progress"
        return env

    yield create
    for env in created:
        await env.close()
    assert not LIVE_ENVS


async def gui(env, actions, call_id="local_test"):
    return await env.step([make_tool_call("computer", {"actions": actions}, call_id=call_id)])


async def center(env, locator):
    """Scripted test-only coordinates from a visible control, never agent input."""
    box = await locator.bounding_box()
    width, height = env.unwrapped.display_resolution
    return [
        (box["x"] + box["width"] / 2) * 1000 / width,
        (box["y"] + box["height"] / 2) * 1000 / height,
    ]


async def click(env, locator):
    return await gui(env, [{"action": "click", "coordinate": await center(env, locator)}])


@pytest.mark.live
@pytest.mark.parametrize("task_id", ORIGINAL_TASKS)
async def test_local_tasks_complete_through_canonical_actions(local_env, task_id):
    env = await local_env(task_id)
    raw = env.unwrapped
    page = raw._page
    if task_id == "checkbox":
        result = await click(env, page.locator(".checkbox-caption"))
        assert not result.terminated and raw.state.progress == 0
        result = await click(env, page.get_by_role("checkbox", name="I'm not a robot"))
        assert await page.get_by_role("checkbox").get_attribute("aria-checked") == "true"
    elif task_id == "stop_signs":
        verify = page.get_by_role("button", name="Verify", exact=True)
        result = await click(env, verify)
        assert not result.terminated and raw.state.mistakes == 1
        tiles = await page.locator(".sign-tile").all()
        result = await gui(
            env, [{"action": "click", "coordinate": await center(env, tile)} for tile in tiles]
        )
        assert not result.terminated and raw.state.progress == 9
        result = await click(env, verify)
        assert not result.terminated and raw.state.mistakes == 2
        # The SVG text is visibly rendered; no hidden answer field is queried.
        non_targets = [
            tile for tile in tiles if "STOP" not in await tile.locator("text").all_text_contents()
        ]
        await gui(
            env,
            [{"action": "click", "coordinate": await center(env, tile)} for tile in non_targets],
        )
        assert raw.state.progress == 4
        result = await click(env, verify)
    elif task_id == "click":
        prompt = await page.locator("#instruction").inner_text()
        target = prompt.split("Click the ", 1)[1].split(".", 1)[0]
        wrong = page.get_by_role(
            "button",
            name="orange circle" if target != "orange circle" else "blue circle",
            exact=True,
        )
        result = await click(env, wrong)
        assert not result.terminated and raw.state.mistakes == 1
        result = await click(env, page.get_by_role("button", name=target, exact=True))
    elif task_id in ("input", "wiggles"):
        code = (
            "".join(await page.locator(".code span").all_text_contents())
            if task_id == "wiggles"
            else await page.locator(".code").inner_text()
        )
        await click(env, page.locator("#code-input"))
        result = await gui(env, [{"action": "type", "text": "WRONG", "press_enter": True}])
        assert not result.terminated and raw.state.mistakes == 1
        result = await gui(
            env,
            [
                {"action": "key", "keys": ["ctrl", "a"]},
                {"action": "type", "text": code, "press_enter": True},
            ],
        )
    elif task_id == "drag":
        start = await center(env, page.locator(".block"))
        missed = await gui(
            env,
            [
                {
                    "action": "drag",
                    "start_coordinate": start,
                    "coordinate": [start[0] + 10, start[1]],
                }
            ],
        )
        assert not missed.terminated and raw.state.mistakes == 1
        result = await gui(
            env,
            [
                {
                    "action": "drag",
                    "start_coordinate": await center(env, page.locator(".block")),
                    "coordinate": await center(env, page.locator(".drop-slot")),
                }
            ],
        )
    elif task_id == "sequence":
        await click(env, page.get_by_role("button", name="1", exact=True))
        result = await click(env, page.get_by_role("button", name="3", exact=True))
        assert not result.terminated and raw.state.progress == 0 and raw.state.mistakes == 1
        for number in range(1, 5):
            result = await click(env, page.get_by_role("button", name=str(number), exact=True))
    else:
        result = await click(env, page.get_by_role("button", name="Moving dot"))
        assert not result.terminated and raw.state.mistakes == 1
        await click(env, page.get_by_role("button", name="Start", exact=True))
        await page.get_by_role("button", name="Moving dot").focus()
        result = await gui(env, [{"action": "key", "keys": ["enter"]}])
        assert not result.terminated and raw.state.mistakes == 2
        result = await click(env, page.get_by_role("button", name="Moving dot"))
    assert result.terminated and not result.truncated and result.reward == 1
    assert raw.state.status == "success" and raw.outcome == "success"
    assert result.info["state"].keys() == {
        "task_id",
        "label",
        "version",
        "seed",
        "status",
        "mistakes",
        "progress",
        "reason",
        "elapsed_ms",
    }
    server = raw._local_server
    await env.close()
    manifest = json.loads((raw.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "success" and manifest["recording_complete"]
    assert manifest["data"]["cleanup_complete"]
    assert not server._thread.is_alive()
    events = [
        json.loads(line) for line in (raw.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        item["type"] == "local_game_started"
        and {"index.html", "game.js", "style.css", "tasks.json"}.issubset(
            item["data"]["asset_hashes"]
        )
        for item in events
    )
    assert any(
        item["type"] == "game_event" and item["data"]["kind"] == "success" for item in events
    )
    assert not any(item["type"] == "page_error" for item in events)


@pytest.mark.live
async def test_checkbox_reset_clears_checked_state(local_env):
    env = await local_env("checkbox")
    await click(env, env.unwrapped._page.get_by_role("checkbox"))
    previous_attempt = env.unwrapped.attempt_dir
    await env.reset()
    assert env.unwrapped.attempt_dir != previous_attempt
    assert env.unwrapped.state.status == "in_progress"
    assert (
        await env.unwrapped._page.get_by_role("checkbox").get_attribute("aria-checked") == "false"
    )


@pytest.mark.live
async def test_stop_signs_missing_target_rejected_and_reset_clears_selection(local_env):
    env = await local_env("stop_signs", seed=42)
    raw = env.unwrapped
    tiles = await raw._page.locator(".sign-tile").all()
    initial_art = await raw._page.locator(".sign-tile").evaluate_all(
        "els => els.map(el => el.innerHTML)"
    )
    targets = [tile for tile in tiles if "STOP" in await tile.locator("text").all_text_contents()]
    assert len(targets) == 4
    await gui(
        env, [{"action": "click", "coordinate": await center(env, tile)} for tile in targets[:-1]]
    )
    result = await click(env, raw._page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and raw.state.mistakes == 1 and raw.state.progress == 3
    await env.reset()
    assert raw.state.mistakes == 0 and raw.state.progress == 0
    assert await raw._page.locator('.sign-tile[aria-pressed="true"]').count() == 0
    assert (
        await raw._page.locator(".sign-tile").evaluate_all("els => els.map(el => el.innerHTML)")
        == initial_art
    )


@pytest.mark.live
async def test_wiggles_animate_and_seeded_reset_retains_code_not_input(local_env):
    env = await local_env("wiggles", seed=41)
    raw = env.unwrapped
    code = "".join(await raw._page.locator(".code span").all_text_contents())
    assert len(code) == 6 and code.isalpha() and code.isupper()
    before = await raw._page.locator(".code span").evaluate_all(
        "els => els.map(el => getComputedStyle(el).transform)"
    )
    await raw._page.wait_for_timeout(180)
    after = await raw._page.locator(".code span").evaluate_all(
        "els => els.map(el => getComputedStyle(el).transform)"
    )
    assert before != after
    await click(env, raw._page.locator("#code-input"))
    result = await gui(env, [{"action": "type", "text": code.lower(), "press_enter": True}])
    assert not result.terminated and raw.state.mistakes == 1
    await env.reset()
    assert "".join(await raw._page.locator(".code span").all_text_contents()) == code
    assert await raw._page.locator("#code-input").input_value() == ""
    assert raw.state.mistakes == 0
    other = await local_env("wiggles", seed=42)
    assert "".join(await other.unwrapped._page.locator(".code span").all_text_contents()) != code


@pytest.mark.live
async def test_seeded_reset_and_cross_instance_isolation(local_env):
    first = await local_env("sequence", seed=42)
    second = await local_env("sequence", seed=42)
    first_raw, second_raw = first.unwrapped, second.unwrapped
    original = await first_raw._page.screenshot()
    assert original == await second_raw._page.screenshot()
    initial_order = await first_raw._page.locator(".number").all_text_contents()
    await click(first, first_raw._page.get_by_role("button", name="1", exact=True))
    assert first_raw.state.progress == 1
    assert (await second_raw._page.evaluate("window.syntheticTask.snapshot()")).get("progress") == 0
    previous_attempt = first_raw.attempt_dir
    await first.reset()
    assert first_raw.attempt_dir != previous_attempt
    assert first_raw.state.progress == 0 and first_raw.state.mistakes == 0
    assert await first_raw._page.screenshot() == original
    assert await first_raw._page.locator(".number").all_text_contents() == initial_order
    assert json.loads((previous_attempt / "manifest.json").read_text())["outcome"] == "aborted"
    third = await local_env("sequence", seed=99)
    assert await third.unwrapped._page.locator(".number").all_text_contents() != initial_order


@pytest.mark.live
async def test_input_reset_clears_partial_text(local_env):
    env = await local_env("input")
    await click(env, env.unwrapped._page.locator("#code-input"))
    await gui(env, [{"action": "type", "text": "PARTIAL"}])
    await env.reset()
    assert await env.unwrapped._page.locator("#code-input").input_value() == ""


@pytest.mark.live
async def test_terminal_stops_batch_and_last_step_success_wins(local_env):
    env = await local_env("sequence", max_steps=1, display_resolution=(1000, 760))
    actions = [
        {
            "action": "click",
            "coordinate": await center(
                env, env.unwrapped._page.get_by_role("button", name=str(number), exact=True)
            ),
        }
        for number in range(1, 5)
    ]
    actions.append({"action": "type", "text": "MUST_NOT_EXECUTE"})
    result = await gui(env, actions)
    assert result.terminated and not result.truncated and result.reward == 1
    assert len(result.info["executed_actions"]) == 4
    assert env.unwrapped.state.progress == 4


@pytest.mark.live
async def test_finish_claim_and_budget_do_not_award_success(local_env):
    env = await local_env("click", max_steps=1)
    result = await env.step([make_tool_call("terminate", {"status": "success"}, call_id="claim")])
    assert result.terminated and result.reward == 0 and env.unwrapped.outcome == "agent_stopped"
    await env.reset()
    result = await gui(env, [{"action": "screenshot"}])
    assert result.truncated and not result.terminated and result.reward == 0
    assert env.unwrapped.outcome == "budget_exhausted"


@pytest.mark.live
@pytest.mark.parametrize("finish_call", [False, True])
async def test_moving_deadline_fails_without_waiting_twenty_seconds(local_env, finish_call):
    env = await local_env("moving")
    page = env.unwrapped._page
    # Clock manipulation is only a unit-test mechanism, not a benchmark mode.
    await page.clock.install()
    await page.reload(wait_until="networkidle")
    await page.wait_for_function("window.syntheticTask !== undefined")
    await click(env, page.get_by_role("button", name="Start", exact=True))
    await page.clock.fast_forward(21000)
    if finish_call:
        result = await env.step(
            [make_tool_call("terminate", {"status": "success"}, call_id="expired")]
        )
        assert result.results == []
    else:
        result = await gui(env, [{"action": "screenshot"}])
    assert result.terminated and not result.truncated and result.reward == 0
    assert env.unwrapped.state.reason == "time_limit"


@pytest.mark.live
async def test_local_state_read_failure_is_archived_as_infrastructure_error(local_env, monkeypatch):
    env = await local_env("click")
    raw = env.unwrapped
    monkeypatch.setattr(raw._page, "evaluate", AsyncMock(side_effect=RuntimeError("page crashed")))
    with pytest.raises(RuntimeError, match="page crashed"):
        await gui(env, [{"action": "screenshot"}])
    assert raw.outcome == "infra_error" and raw._terminal
    await env.close()
    manifest = json.loads((raw.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "infra_error"
    events = [
        json.loads(line) for line in (raw.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        item["type"] == "error" and item["data"]["phase"] == "local_state" for item in events
    )


@pytest.mark.live
async def test_deadline_crossing_during_screenshot_is_reobserved(local_env, monkeypatch):
    env = await local_env("moving")
    raw = env.unwrapped
    page = raw._page
    await page.clock.install()
    await page.reload(wait_until="networkidle")
    await page.wait_for_function("window.syntheticTask !== undefined")
    await click(env, page.get_by_role("button", name="Start", exact=True))
    screenshot = page.screenshot
    captured = []

    async def crossing_screenshot(**kwargs):
        png = await screenshot(**kwargs)
        captured.append(png)
        if len(captured) == 1:
            await page.clock.fast_forward(21000)
        return png

    monkeypatch.setattr(page, "screenshot", crossing_screenshot)
    result = await gui(env, [{"action": "screenshot"}])
    assert len(captured) == 2
    assert result.terminated and result.reward == 0
    assert raw.state.status == "failure"
    assert result.results[0].images[-1] == captured[-1]


@pytest.mark.live
async def test_local_boundary_blocks_external_requests_without_sending_them(local_env):
    env = await local_env("click")
    page = env.unwrapped._page
    # Call the route boundary directly; no request is sent to the external host.
    route = type("Route", (), {})()
    route.request = Mock(url="https://example.invalid/asset.js")
    route.request.is_navigation_request.return_value = False
    route.abort, route.continue_ = AsyncMock(), AsyncMock()
    await env.unwrapped._route(route)
    route.abort.assert_awaited_once_with("blockedbyclient")
    route.continue_.assert_not_awaited()
    assert page.url.startswith(env.unwrapped._local_server.origin)
