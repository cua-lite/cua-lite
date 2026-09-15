"""Owned-video lifecycle and archive tests; saved fixtures are not decoded media."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from playwright.async_api import Error as PlaywrightError

from examples.not_a_robot import (
    codex_bridge,
    codex_smoke,
    registration,  # noqa: F401
    smoke,
)
from examples.not_a_robot.env import NotARobotEnv
from examples.not_a_robot.recorder import EventRecorder
from lite import gym


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_video_parameter_is_explicit(tmp_path, monkeypatch, enabled):
    flags = ["--record-video"] if enabled else []
    args = codex_bridge._parse_args(
        [
            "--task",
            "click",
            "--artifact-root",
            str(tmp_path),
            "--model",
            "test-model",
            "--reasoning-effort",
            "high",
            *flags,
        ]
    )
    assert args.record_video is enabled
    monkeypatch.setattr(sys, "argv", ["smoke", "--artifact-root", str(tmp_path), *flags])
    assert smoke._parse_args().record_video is enabled
    launcher = codex_smoke._parse_args(
        [
            "--tasks",
            "neal_01",
            "--artifact-root",
            str(tmp_path / "new-run"),
            "--browser-executable",
            "/browser",
            *flags,
        ]
    )
    command = codex_smoke.build_command(launcher, "neal_01", tmp_path / "task", "/uv")
    config = tomllib.loads(
        "\n".join(command[index + 1] for index, token in enumerate(command) if token == "-c")
    )
    assert ("--record-video" in config["mcp_servers"]["local_game"]["args"]) is enabled
    assert command[-1] == codex_smoke.PROMPT
    assert config["mcp_servers"]["local_game"]["enabled_tools"] == [
        "get_observation",
        "computer",
        "finish",
    ]


@pytest.mark.parametrize(
    "key",
    [
        "not_a_robot@level_001",
        "not_a_robot_campaign@full_game",
        "visual_tasks@click",
        "visual_tasks_campaign@full_game",
    ],
)
def test_registry_accepts_video_without_starting_browser(key):
    assert gym.make(key).unwrapped.record_video is False
    assert gym.make(key, record_video=True).unwrapped.record_video is True


def test_video_not_requested_has_no_artifact_or_playback_claim(tmp_path):
    recorder = EventRecorder(tmp_path / "attempt", {})
    manifest = recorder.finalize("failure")
    assert manifest["recording_complete"]
    assert manifest["recording_scope"] == "events_and_png_images"
    assert manifest["video"]["status"] == "not_requested"
    assert manifest["video"]["playback_validation"] == "not_performed"
    assert not manifest["video"]["audio"] and not manifest["video"]["model_input"]
    assert "path" not in manifest["video"]
    assert not (recorder.root / "videos").exists()


def test_stream_hash_does_not_claim_decoding(tmp_path, monkeypatch):
    recorder = EventRecorder(tmp_path / "attempt", {})
    video_dir = recorder.root / "videos"
    video_dir.mkdir()
    # Deliberately not valid media: this is an integrity test, not a playback test.
    content = b"not decoded video fixture" * 100000
    (video_dir / "page.webm").write_bytes(content)
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("Video must be hashed in chunks"))
    manifest = recorder.finalize("success", video={"status": "saved", "path": "videos/page.webm"})
    assert manifest["video"]["status"] == "saved"
    assert manifest["video"]["bytes"] == len(content)
    assert manifest["video"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert manifest["video"]["playback_validation"] == "not_performed"
    assert manifest["outcome"] == "success"


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("missing", "missing"),
        ("empty", "failed"),
        ("outside", "failed"),
        ("nested", "failed"),
        ("extension", "failed"),
    ],
)
def test_invalid_video_preserves_event_archive_without_saved_claim(tmp_path, kind, expected):
    recorder = EventRecorder(tmp_path / "attempt", {})
    (recorder.root / "videos").mkdir()
    relative = "videos/page.webm"
    if kind == "empty":
        (recorder.root / relative).touch()
    elif kind == "outside":
        relative = "../outside.webm"
        (tmp_path / "outside.webm").write_bytes(b"private outside file not to be read")
    elif kind == "nested":
        relative = "videos/../events.jsonl"
    elif kind == "extension":
        relative = "videos/page.png"
        (recorder.root / relative).write_bytes(b"wrong extension")
    manifest = recorder.finalize("success", video={"status": "saved", "path": relative})
    assert manifest["recording_complete"] and manifest["outcome"] == "success"
    assert manifest["video"]["status"] == expected
    assert manifest["video"]["error"]
    assert "sha256" not in manifest["video"]
    assert "bytes" not in manifest["video"]


@pytest.mark.parametrize("failure", [None, "context", "path", "missing_artifact", "browser"])
async def test_context_closes_before_main_video_hash_and_browser(tmp_path, failure):
    env = NotARobotEnv(mode="fixture", record_video=True)
    env.attempt_dir = tmp_path / "attempt"
    env.recorder = EventRecorder(env.attempt_dir, {})
    env.outcome = "controller_timeout"
    events = []
    video_path = env.attempt_dir / "videos" / "main.webm"
    video_path.parent.mkdir()
    # A distraction file must never be chosen instead of the main page's video.
    (video_path.parent / "other-page.webm").write_bytes(b"other owned page")

    async def close_context():
        events.append("context_close")
        if failure == "context":
            raise RuntimeError("context close failed")
        video_path.write_bytes(b"main-page finalized fixture")

    async def video_file():
        events.append("video_path")
        assert video_path.exists()
        if failure == "path":
            raise PlaywrightError("video path failed")
        return str(video_path)

    async def close_browser():
        events.append("browser_close")
        if failure == "browser":
            raise RuntimeError("browser close failed")

    env._context = SimpleNamespace(close=AsyncMock(side_effect=close_context))
    env._video = (
        None
        if failure == "missing_artifact"
        else SimpleNamespace(path=AsyncMock(side_effect=video_file))
    )
    env._browser = SimpleNamespace(close=AsyncMock(side_effect=close_browser))
    await env.close()
    await env.close()
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert events[0] == "context_close" and events[-1] == "browser_close"
    assert events.count("context_close") == events.count("browser_close") == 1
    expected = {
        None: "saved",
        "context": "failed",
        "path": "failed",
        "missing_artifact": "missing",
        "browser": "saved",
    }[failure]
    assert manifest["video"]["status"] == expected
    assert manifest["data"]["cleanup_complete"] is (failure not in ("context", "browser"))
    assert manifest["outcome"] == "controller_timeout"
    assert manifest["video"]["playback_validation"] == "not_performed"
    if expected == "saved":
        assert manifest["video"]["path"] == "videos/main.webm"
        assert manifest["video"]["sha256"] == hashlib.sha256(video_path.read_bytes()).hexdigest()
    archive = [
        json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    assert sum(row["type"] == "attempt_end" for row in archive) == 1
    assert any(row["type"] == "video_context_close_started" for row in archive)


async def test_requested_video_before_browser_start_is_missing(tmp_path):
    env = NotARobotEnv(mode="fixture", record_video=True)
    env.attempt_dir = tmp_path / "attempt"
    env.recorder = EventRecorder(env.attempt_dir, {})
    env.outcome = "infra_error"
    await env.close()
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["video"]["status"] == "missing"
    assert manifest["recording_complete"]
    assert manifest["outcome"] == "infra_error"


async def test_unrecorded_context_never_requests_video_or_creates_directory(tmp_path):
    env = NotARobotEnv(mode="fixture")
    env.attempt_dir = tmp_path / "attempt"
    env.recorder = EventRecorder(env.attempt_dir, {})
    env._context = SimpleNamespace(close=AsyncMock())
    env._browser = SimpleNamespace(close=AsyncMock())
    env._video = SimpleNamespace(path=AsyncMock(side_effect=AssertionError("Not requested")))
    await env.close()
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["video"]["status"] == "not_requested"
    assert not (env.attempt_dir / "videos").exists()


async def test_repeated_controller_cancellation_waits_for_video_finalization(tmp_path):
    env = NotARobotEnv(mode="fixture", record_video=True)
    env.attempt_dir = tmp_path / "attempt"
    env.recorder = EventRecorder(env.attempt_dir, {})
    env.outcome = "in_progress"
    video_path = env.attempt_dir / "videos" / "page.webm"
    video_path.parent.mkdir()
    started = asyncio.Event()
    release = asyncio.Event()

    async def close_context():
        started.set()
        await release.wait()
        video_path.write_bytes(b"finished after repeated cancellation")

    context = SimpleNamespace(close=AsyncMock(side_effect=close_context))
    browser = SimpleNamespace(close=AsyncMock())
    env._context, env._browser = context, browser
    env._video = SimpleNamespace(path=AsyncMock(return_value=str(video_path)))
    bridge = codex_bridge.CodexBridge(
        SimpleNamespace(max_seconds=5, model="unit_fixture", reasoning_effort="high")
    )
    bridge.env = env
    task = asyncio.create_task(bridge.close("controller_timeout"))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    await asyncio.wait_for(task, timeout=2)
    await bridge.close("later_shutdown")
    context.close.assert_awaited_once()
    browser.close.assert_awaited_once()
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["video"]["status"] == "saved"
    assert manifest["outcome"] == "controller_timeout"
    assert manifest["data"]["cleanup_complete"]
