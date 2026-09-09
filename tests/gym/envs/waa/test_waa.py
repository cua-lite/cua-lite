"""Tests for the WindowsAgentArena CUA-Lite integration."""

from __future__ import annotations

import base64
import inspect
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import yaml

import lite.gym as gym
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name
from lite.core.utils.filters import parse_filter
from lite.gym.envs.waa.main import (
    CFG,
    WindowsAgentArenaBridgeClient,
    WindowsAgentArenaEnv,
    _asset_manifest_digest,
    _check_runtime_dependencies,
    _ensure_services,
    _load_tasks,
    _to_pyautogui,
)
from lite.gym.envs.waa.qemu import QemuConfig, QemuInstance, reap_runtime_slots
from lite.gym.errors import CapacityExhausted, EnvDepsMissingError
from lite.gym.utils.backend.freshness import image_for
from lite.gym.utils.config.identity import EnvIdentity

REPO_ROOT = Path(__file__).resolve().parents[4]
PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@pytest.mark.parametrize(
    "reward,terminated", [(None, False), (0.0, False), (2.0, True), (0.0, True)]
)
async def test_task_verification_requires_completed_evaluation(monkeypatch, reward, terminated):
    from lite.gym.envs.waa.scripts.utils import verify_all_tasks

    env = SimpleNamespace(
        reset=AsyncMock(return_value=SimpleNamespace(image=b"screenshot")),
        step=AsyncMock(return_value=SimpleNamespace(reward=reward, terminated=terminated)),
        close=AsyncMock(),
    )
    constructor = Mock(return_value=env)
    monkeypatch.setattr(verify_all_tasks, "WindowsAgentArenaEnv", constructor)
    row = {
        "config": {}, "domain": "notepad", "variant": "standard",
        "task_id": "test", "covered_task_ids": ["test"],
    }
    result = await verify_all_tasks._run_attempt(
        row,
        base_disk=Path("disk"),
        assets_dir=Path("assets"),
        reset_timeout=1,
        evaluate_timeout=1,
    )
    assert constructor.call_args.kwargs["extra_tools"] == ["terminate"]
    assert result["ok"] is (terminated and reward == 0.0)
    assert result["evaluate_ok"] is result["ok"]
    env.close.assert_awaited_once()


def _registry_metadata(task_id: str):
    metadata = gym.registry.task_metadata("waa", task_id)
    assert metadata is not None
    return metadata


def test_published_task_catalog_registered():
    assert len(gym.registry.task_ids("waa", split="eval")) == 154
    assert len(gym.registry.task_ids("waa", split="eval_noctxt")) == 154


def test_original_catalog_contains_thirteen_infeasible_tasks_per_variant():
    tasks = _load_tasks()
    infeasible = [
        task for task in tasks if task["config"].get("evaluator", {}).get("func") == "infeasible"
    ]

    assert len(infeasible) == 26
    assert sum(task["variant"] == "standard" for task in infeasible) == 13
    assert sum(task["variant"] == "no_context" for task in infeasible) == 13


def test_infeasible_tasks_are_filterable_from_registry_metadata():
    keep = parse_filter("lambda m: not m.others.get('exclude_reason')")

    for split in ("eval", "eval_noctxt"):
        task_ids = gym.registry.task_ids("waa", split=split)
        reasons = [
            _registry_metadata(task_id).others.get("exclude_reason") for task_id in task_ids
        ]
        tagged = [r for r in reasons if r]

        # 13 upstream `infeasible` contracts + 3 hand-curated trivial-reward
        # `block:` exclusions per split (six across both splits, split-specific).
        assert sum(r == "infeasible" for r in tagged) == 13
        assert sum(r.startswith("block:") for r in tagged) == 3
        assert len(tagged) == 16
        assert sum(keep(_registry_metadata(task_id)) for task_id in task_ids) == 138


def test_constructed_env_metadata_matches_registry_exclusion_label():
    task = next(
        task
        for task in _load_tasks()
        if task["config"].get("evaluator", {}).get("func") == "infeasible"
    )
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
    )

    assert env.metadata.others["exclude_reason"] == "infeasible"


def test_report_infeasible_is_opt_in_on_constructed_env():
    task = _load_tasks()[0]
    default_env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
    )
    enabled_env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=["report_infeasible"],
    )

    assert default_env.metadata.extra_tool_schemas == []
    names = [
        tool_schema_name(schema)
        for schema in enabled_env.metadata.extra_tool_schemas
    ]
    assert names == ["report_infeasible"]


def test_environment_accepts_rollout_seed():
    task = next(task for task in _load_tasks() if task["variant"] == "standard")
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        seed=42,
    )

    assert env._seed == 42


def test_config_kwargs_are_wired_to_the_correct_boundary():
    # Per-instance kwargs cross explicit construction + bind; server-wide kwargs
    # are module constants, never per-instance params.
    constructor = set(inspect.signature(WindowsAgentArenaEnv.__init__).parameters)
    soft = set(inspect.signature(WindowsAgentArenaEnv.bind).parameters)
    parameters = (constructor | soft) - {"self"}

    assert set(CFG.env_kwargs) <= parameters
    assert set(CFG.server_kwargs).isdisjoint(parameters)


def test_custom_assets_dir_reaches_qemu_config(tmp_path):
    task = _load_tasks()[0]
    assets_dir = tmp_path / "assets"
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        assets_dir=str(assets_dir),
    )

    assert env._qemu_config.assets_dir == assets_dir.resolve()


def test_action_translation_uses_normalized_coordinates_and_safe_text_repr():
    assert _to_pyautogui("click", {"coordinate": [500, 250]}, 1440, 900) == (
        "pyautogui.click(720, 225, clicks=1, button='left')"
    )
    command = _to_pyautogui("type", {"text": "it's ok"}, 1440, 900)
    assert command == 'pyautogui.write("it\'s ok", interval=0.02)'


def test_action_translation_coerces_drag_duration():
    command = _to_pyautogui(
        "drag",
        {"coordinate": [500, 500], "duration": "1.25"},
        1000,
        1000,
    )

    assert "pyautogui.dragTo(" in command
    assert "duration=1.25" in command


@pytest.mark.parametrize(
    ("name", "arguments", "expected_error"),
    [
        ("wait", {"duration": 31}, "wait.duration must be <= 30"),
        ("hold_key", {"keys": ["a"], "duration": 6}, "hold_key.duration must be <= 5"),
        ("drag", {"coordinate": [500, 500], "duration": 6}, "drag.duration must be <= 5"),
    ],
)
def test_action_translation_rejects_bad_model_durations(
    name,
    arguments,
    expected_error,
):
    with pytest.raises(ValueError, match=expected_error):
        _to_pyautogui(name, arguments, 1000, 1000)


def test_action_translation_rejects_empty_key_lists():
    # ``keys`` is required with no default on every canonical key action, and env
    # ingress checks envelope shape only -- so an empty list is a malformed
    # argument, reported through ``step``'s ``except MODEL_ACTION_ERROR_TYPES``.
    # It used to translate to ``None`` (reported to the model as "unsupported
    # action: key"), and ``hold_key`` used to degrade to a bare ``time.sleep``.
    for name in ("key", "key_down", "key_up"):
        with pytest.raises(ValueError, match=f"{name}.keys must not be empty"):
            _to_pyautogui(name, {"keys": []}, 1440, 900)
    with pytest.raises(ValueError, match="hold_key.keys must not be empty"):
        _to_pyautogui("hold_key", {"keys": [], "duration": 2}, 1440, 900)


def test_action_translation_rejects_scalar_keys():
    with pytest.raises(ValueError, match="key.keys must be a list of strings"):
        _to_pyautogui("key", {"keys": "enter"}, 1440, 900)


def test_action_translation_accepts_canonical_key_glyphs():
    assert _to_pyautogui("key", {"keys": ["ctrl", "+", "-", "="]}, 1440, 900) == (
        "pyautogui.hotkey('ctrl', '+', '-', '=')"
    )


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["plus"], "unknown key token 'plus'"),
        ([" "], "unknown key token ' '"),
    ],
)
def test_action_translation_rejects_noncanonical_key_tokens(keys, expected):
    with pytest.raises(ValueError, match=expected):
        _to_pyautogui("key", {"keys": keys}, 1440, 900)


def test_runtime_dependency_check_rejects_missing_base_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lite.gym.utils.backend.docker.require_image_present",
        lambda image: None,
    )

    with pytest.raises(EnvDepsMissingError, match="base disk is missing or empty"):
        _check_runtime_dependencies(
            base_disk=Path(tmp_path / "missing.qcow2"),
            runner_image="test-runner:latest",
        )


def test_runtime_dependency_check_rejects_stale_asset_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lite.gym.utils.backend.docker.require_image_present",
        lambda image: None,
    )
    disk = tmp_path / "base.qcow2"
    disk.write_bytes(b"qcow2")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / ".complete.json").write_text(
        json.dumps({"manifest_sha256": "stale"}),
        encoding="utf-8",
    )

    with pytest.raises(EnvDepsMissingError, match="asset cache is missing or stale"):
        _check_runtime_dependencies(
            base_disk=disk,
            runner_image="test-runner:latest",
            assets_dir=assets,
        )


# A tiny synthetic manifest keeps the blob-presence tests fast (the real cache is
# ~121 MB across 123 blobs — too heavy to materialise per test).
_SYNTHETIC_MANIFEST = {
    "schema_version": 1,
    "assets": [
        {"source_url": "u1", "download_url": "u1", "sha256": "a" * 64, "size": 3},
        {"source_url": "u2", "download_url": "u2", "sha256": "b" * 64, "size": 5},
    ],
}


def _install_synthetic_asset_cache(monkeypatch, assets_dir: Path) -> None:
    """Point main at a small manifest, then write its current marker + blobs."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = assets_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(_SYNTHETIC_MANIFEST), encoding="utf-8")
    monkeypatch.setattr("lite.gym.envs.waa.main.ASSET_MANIFEST_PATH", manifest_path)
    (assets_dir / ".complete.json").write_text(
        json.dumps({"manifest_sha256": _asset_manifest_digest()}),
        encoding="utf-8",
    )
    for entry in _SYNTHETIC_MANIFEST["assets"]:
        (assets_dir / entry["sha256"]).write_bytes(b"\0" * entry["size"])


def test_runtime_dependency_check_accepts_current_asset_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lite.gym.utils.backend.docker.require_image_present",
        lambda image: None,
    )
    disk = tmp_path / "base.qcow2"
    disk.write_bytes(b"qcow2")
    assets = tmp_path / "assets"
    _install_synthetic_asset_cache(monkeypatch, assets)

    _check_runtime_dependencies(
        base_disk=disk,
        runner_image="test-runner:latest",
        assets_dir=assets,
    )


def test_runtime_dependency_check_rejects_missing_asset_blob(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lite.gym.utils.backend.docker.require_image_present",
        lambda image: None,
    )
    disk = tmp_path / "base.qcow2"
    disk.write_bytes(b"qcow2")
    assets = tmp_path / "assets"
    _install_synthetic_asset_cache(monkeypatch, assets)
    # A current marker must not paper over a truncated/deleted blob.
    (assets / _SYNTHETIC_MANIFEST["assets"][0]["sha256"]).unlink()

    with pytest.raises(EnvDepsMissingError, match="asset cache is missing or stale"):
        _check_runtime_dependencies(
            base_disk=disk,
            runner_image="test-runner:latest",
            assets_dir=assets,
        )


def test_service_check_only_validates_runner_image(monkeypatch):
    checked = []
    monkeypatch.setattr(
        "lite.gym.envs.waa.main._check_runner_image",
        checked.append,
    )

    _ensure_services("waa")

    assert checked == ["cua-lite/waa:latest"]


def test_runner_freshness_excludes_guest_prep_sources():
    sources = image_for("waa").sources

    assert "lite/gym/envs/waa/docker/Dockerfile" in sources
    assert "lite/gym/envs/waa/data/assets.json" in sources
    assert not any("docker/prep" in source for source in sources)


def test_runtime_slot_reaper_is_scoped_and_preserves_live_slots(monkeypatch, tmp_path):
    slots = tmp_path / "slots"
    slots.mkdir()
    orphan = slots / "lite-env-30100-user-waa-task-dead"
    live = slots / "lite-env-30100-user-waa-task-live"
    other_scope = slots / "lite-env-30101-user-waa-task-dead"
    for slot in (orphan, live, other_scope):
        slot.mkdir()
        os.utime(slot, (1, 1))

    monkeypatch.setattr(
        "lite.gym.envs.waa.qemu.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{live.name}\n"),
    )

    assert reap_runtime_slots(tmp_path, server_port=30100, boot=False) == 1
    assert not orphan.exists()
    assert live.exists()
    assert other_scope.exists()


def test_runtime_slot_reaper_refuses_unscoped_prefix(monkeypatch, tmp_path):
    slots = tmp_path / "slots"
    slots.mkdir()
    orphan = slots / "lite-env-30100-user-waa-task-dead"
    orphan.mkdir()

    def _boom(*args, **kwargs):
        raise AssertionError("docker must not be consulted without server scope")

    monkeypatch.setattr("lite.gym.envs.waa.qemu.subprocess.run", _boom)

    assert reap_runtime_slots(tmp_path, server_port=None, boot=True) == 0
    assert orphan.exists()


def test_standard_rollout_configs_do_not_require_infeasible_tool():
    for family in ("gpt", "qwen3_vl", "qwen3_5"):
        path = REPO_ROOT / "scripts" / "configs" / family / "default" / "waa.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        extra_tools = config["env_kwargs"].get("extra_tools", [])
        assert "report_infeasible" not in extra_tools


@pytest.mark.asyncio
async def test_bridge_client_preserves_server_error_detail():
    client = WindowsAgentArenaBridgeClient("http://example.test")
    response = Mock(
        is_error=True,
        status_code=500,
        text='{"error":"setup failed"}',
    )
    response.json.return_value = {"error": "setup failed"}
    client._client.post = AsyncMock(return_value=response)

    with pytest.raises(RuntimeError, match="setup failed"):
        await client.post("/reset", timeout=1)
    await client.close()


def _env_with_start_error(monkeypatch, error):
    """A WAA env whose QemuInstance.start() raises `error` (deps mocked out)."""
    task = _load_tasks()[0]
    instance = SimpleNamespace(
        name="waa-test",
        bridge_url=None,
        novnc_url=None,
        start=AsyncMock(side_effect=error),
        close=AsyncMock(),
    )

    async def close_instance():
        instance.name = None

    instance.close.side_effect = close_instance
    monkeypatch.setattr("lite.gym.envs.waa.main._check_runtime_dependencies", lambda **kwargs: None)
    monkeypatch.setattr("lite.gym.envs.waa.main.QemuInstance", lambda **kwargs: instance)
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=["terminate"],
    )
    return env, instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boot_error",
    [
        # qemu.py raises CapacityExhausted directly for "still warming" boot failures
        # (container exited / bridge not ready); a bare transport hiccup is retryable too.
        CapacityExhausted.warming("runner exited before readiness"),
        TimeoutError("bridge did not become ready"),
        ConnectionError("connection reset"),
        httpx.ConnectError("connection refused"),
    ],
)
async def test_boot_transient_failures_become_capacity_exhausted(monkeypatch, boot_error):
    env, instance = _env_with_start_error(monkeypatch, boot_error)
    with pytest.raises(CapacityExhausted):
        await env.reset()
    instance.close.assert_awaited_once()
    assert env._instance is None


def test_pinned_upstream_refs_do_not_drift_across_files():
    # The prep-image digest and WAA source commit are each pinned in several files.
    # Drift silently breaks disk_id gating (published label vs what built the disk)
    # or the task catalog. Guard that every copy agrees.
    import re
    from pathlib import Path

    waa = Path(__file__).resolve().parents[4] / "lite" / "gym" / "envs" / "waa"

    def _unique(pattern: str, files: list[Path]) -> set[str]:
        found = set()
        for f in files:
            m = re.search(pattern, f.read_text(encoding="utf-8"))
            assert m, f"pinned ref '{pattern}' not found in {f}"
            found.add(m.group(0))
        return found

    digests = _unique(
        r"winarena@sha256:[0-9a-f]{64}",
        [
            waa / "scripts" / "install.sh",
            waa / "scripts" / "utils" / "image_provenance.py",
            waa / "scripts" / "utils" / "prepare_image.sh",
            waa / "docker" / "prep" / "Dockerfile",
        ],
    )
    assert len(digests) == 1, f"prep-image digest drift: {digests}"

    commits = _unique(
        r"6d39ed88c545a0d40a7a02e39b928e278df7332b|WAA_COMMIT[= ]+\"[0-9a-f]{40}\"",
        [waa / "scripts" / "utils" / "sync_tasks.py", waa / "docker" / "Dockerfile"],
    )
    # sync_tasks.py wraps it as WAA_COMMIT="…"; normalize both to the bare 40-hex.
    commits = {re.search(r"[0-9a-f]{40}", c).group(0) for c in commits}
    assert len(commits) == 1, f"WAA source-commit drift: {commits}"


def test_default_construction_metadata_shape_is_valid():
    # A default-constructed instance can exist before a real task is bound;
    # reading its metadata must not dereference a None task_config.
    env = WindowsAgentArenaEnv(
        base_disk="/tmp/waa.qcow2",
        assets_dir="/tmp/waa-assets",
        runner_image="cua-lite/waa:latest",
    )
    md = env.metadata
    assert md.platform == "desktop"
    assert md.others["benchmark"] == "WindowsAgentArena"
    assert "exclude_reason" not in md.others


@pytest.mark.asyncio
async def test_boot_terminal_failure_surfaces_and_is_not_warming(monkeypatch):
    # A non-transport RuntimeError (real config fault) must NOT be laundered into a
    # retryable warming 503 — classification is by exception TYPE, not message text.
    env, _ = _env_with_start_error(
        monkeypatch, RuntimeError("WindowsAgentArena local QEMU requires /dev/kvm")
    )
    with pytest.raises(RuntimeError, match="requires /dev/kvm") as exc_info:
        await env.reset()
    assert not isinstance(exc_info.value, CapacityExhausted)


@pytest.mark.asyncio
async def test_qemu_start_publishes_reserved_ports(monkeypatch, tmp_path):
    from lite.gym.envs.waa import qemu

    base = tmp_path / "base.qcow2"
    base.write_bytes(b"disk")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / ".complete.json").write_text("{}")
    instance = QemuInstance(
        config=QemuConfig(
            base_disk=base, runner_image="test-runner", runtime_root=tmp_path,
            assets_dir=assets, snapshot_dir=tmp_path / "snapshot",
            vcpus=1, memory_gb=1, shm_size="1g", bind_address="127.0.0.1",
            ready_timeout_s=1, readiness_poll_interval_s=0.1,
        ),
        task_id="task", identity=EnvIdentity(),
    )
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: str(p) == "/dev/kvm" or exists(p))
    monkeypatch.setattr(qemu.shutil, "which", lambda _: "/usr/bin/docker")
    ports = Mock(return_value=[25101, 25102])
    monkeypatch.setattr(qemu, "allocate_ports", ports)
    calls = []
    in_create_slot = False
    slot_entries = 0

    @asynccontextmanager
    async def create_slot():
        nonlocal in_create_slot, slot_entries
        slot_entries += 1
        in_create_slot = True
        try:
            yield
        finally:
            in_create_slot = False

    monkeypatch.setattr(qemu, "docker_create_slot_async", create_slot)

    async def run(*args, **kwargs):
        assert in_create_slot
        calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(qemu, "_run", run)
    async def ready():
        assert not in_create_slot

    monkeypatch.setattr(instance, "_wait_ready", ready)
    await instance.start()

    ports.assert_called_once_with(n=2, range_start=25000, range_end=26000)
    assert "127.0.0.1:25101:5050" in calls[0]
    assert "127.0.0.1:25102:8006" in calls[0]
    assert instance.bridge_url == "http://127.0.0.1:25101"
    assert instance.novnc_url == "http://127.0.0.1:25102"
    assert len(calls) == 1
    assert slot_entries == 1


@pytest.mark.asyncio
async def test_qemu_close_timeout_preserves_slot_for_reaping(monkeypatch, tmp_path):
    async def timeout(*args, **kwargs):
        raise TimeoutError("docker rm timed out")

    monkeypatch.setattr("lite.gym.envs.waa.qemu._run", timeout)
    instance = QemuInstance(
        config=QemuConfig(
            base_disk=tmp_path / "base.qcow2",
            runner_image="test-runner",
            runtime_root=tmp_path,
            assets_dir=tmp_path / "assets",
            snapshot_dir=tmp_path / "snapshot",
            vcpus=1,
            memory_gb=1,
            shm_size="1g",
            bind_address="127.0.0.1",
            ready_timeout_s=1,
            readiness_poll_interval_s=0.1,
        ),
        task_id="task",
        identity=EnvIdentity(),
    )
    instance.name = "waa-test"
    instance.slot_root = tmp_path / "slots" / "waa-test"

    await instance.close()

    assert instance.name == "waa-test"
    assert instance.slot_root == tmp_path / "slots" / "waa-test"


@pytest.mark.asyncio
async def test_failure_termination_marks_fail_before_evaluation():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=["terminate"],
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True},
        {"ok": True, "reward": 1.0},
        {"ok": True, "screenshot_b64": base64.b64encode(b"shot").decode()},
    ]
    env._bridge = bridge

    result = await env.step(
        [make_tool_call("terminate", {"status": "failure"}, call_id="call_terminate")]
    )

    assert bridge.post.await_args_list[0].args[:2] == (
        "/step",
        {"action": "FAIL", "pause": 0.0},
    )
    assert bridge.post.await_args_list[1].args == ("/evaluate",)
    assert env._step_count == 1
    assert result.terminated is True
    assert result.reward == 1.0
    assert result.info["stop_reason"] == "terminate"
    assert result.info["executed_actions"] == [
        {"call": "terminate", "args": {"command": None}}
    ]


@pytest.mark.asyncio
async def test_success_termination_marks_done_before_evaluation():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=["terminate"],
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True},
        {"ok": True, "reward": 1.0},
        {"ok": True, "screenshot_b64": base64.b64encode(b"shot").decode()},
    ]
    env._bridge = bridge

    result = await env.step(
        [make_tool_call("terminate", {"status": "success"}, call_id="call_terminate")]
    )

    assert bridge.post.await_args_list[0].args[:2] == (
        "/step",
        {"action": "DONE", "pause": 0.0},
    )
    assert bridge.post.await_args_list[1].args == ("/evaluate",)
    assert env._step_count == 1
    assert result.terminated is True
    assert result.reward == 1.0
    assert result.info["stop_reason"] == "terminate"
    assert result.info["executed_actions"] == [
        {"call": "terminate", "args": {"command": None}}
    ]


@pytest.mark.asyncio
async def test_report_infeasible_marks_fail_before_evaluation():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=["report_infeasible"],
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True},
        {"ok": True, "reward": 1.0},
        {"ok": True, "screenshot_b64": base64.b64encode(b"shot").decode()},
    ]
    env._bridge = bridge

    result = await env.step(
        [
            make_tool_call(
                "report_infeasible",
                {"reason": "blocked"},
                call_id="call_report_infeasible",
            )
        ]
    )

    assert bridge.post.await_args_list[0].args[:2] == (
        "/step",
        {"action": "FAIL", "pause": 0.0},
    )
    assert bridge.post.await_args_list[1].args == ("/evaluate",)
    assert env._step_count == 1
    assert result.terminated is True
    assert result.reward == 1.0
    assert result.info["stop_reason"] == "report_infeasible"
    assert result.info["executed_actions"] == [
        {"call": "report_infeasible", "args": {"command": None}}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "expected_error", "expect_current_image", "expected_reason"),
    [
        (
            "response",
            {"text": "done"},
            "response is not available in this task.",
            False,
            "inactive extra tool",
        ),
        (
            "terminate",
            {"status": "success"},
            "terminate is not available in this task.",
            False,
            "inactive extra tool",
        ),
        (
            "report_infeasible",
            {"reason": "blocked"},
            "report_infeasible is not available in this task.",
            False,
            "inactive extra tool",
        ),
        ("frobnicate", {}, "unknown tool: frobnicate", False, "unknown tool"),
    ],
)
async def test_inactive_or_unknown_tool_returns_paired_feedback(
    name,
    arguments,
    expected_error,
    expect_current_image,
    expected_reason,
):
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=[],
    )
    env._last_screenshot = b"last-shot"
    env._bridge = AsyncMock()

    result = await env.step([
        make_tool_call(name, arguments, call_id=f"call_{name}")
    ])

    env._bridge.post.assert_not_called()
    assert result.terminated is False
    assert result.truncated is False
    assert result.results[0].tool_call_id == f"call_{name}"
    assert (result.results[0].images[-1] if result.results[0].images else None) == (
        b"last-shot" if expect_current_image else None
    )
    assert result.results[0].text is None
    assert result.results[0].error == expected_error
    assert result.results[0].metadata == {"is_error": True}
    assert result.info["executed_actions"] == [{
        "call": "noop",
        "args": {"name": name, "reason": expected_reason},
    }]


@pytest.mark.asyncio
async def test_release_rows_active_known_tool():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        post_action_delay=0.0,
    )
    bridge = AsyncMock()
    bridge.post.return_value = {
        "ok": True,
        "screenshot_b64": base64.b64encode(b"post-action-bound-frame").decode(),
    }
    env._bridge = bridge

    result = await env.step([
        make_tool_call(
            "click",
            {"coordinate": [500, 500]},
            call_id="active_known_tool",
        )
    ])

    assert bridge.post.await_args_list[0].args[:2] == (
        "/step",
        {"action": "pyautogui.click(720, 450, clicks=1, button='left')", "pause": 0.0},
    )
    assert result.terminated is False
    assert result.truncated is False
    assert result.results[0].tool_call_id == "active_known_tool"
    assert result.results[0].images[-1] == b"post-action-bound-frame"
    assert result.results[0].text is None
    assert result.results[0].error is None
    assert result.results[0].metadata is None


@pytest.mark.asyncio
async def test_waa_env_owned_cursor_tracks_successful_actions(monkeypatch):
    import lite.gym.envs.waa.main as waa_main

    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        post_action_delay=0.0,
    )
    calls: list[tuple[int, int]] = []

    def fake_overlay(png: bytes, x: int, y: int) -> bytes:
        calls.append((x, y))
        return f"overlay:{x}:{y}".encode()

    monkeypatch.setattr(waa_main, "overlay_cursor_px", fake_overlay)
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {
            "ok": True,
            "screen_size": {"width": 1440, "height": 900},
            "screenshot_b64": PNG_1X1_B64,
        },
        # reset's pointer-parking /step (see _park_cursor): its response frame is
        # the one grabbed AFTER the pointer actually reached screen centre.
        {
            "ok": True,
            "screenshot_b64": PNG_1X1_B64,
        },
        {
            "ok": True,
            "screenshot_b64": PNG_1X1_B64,
        },
    ]
    env._bridge = bridge
    env._instance = SimpleNamespace(novnc_url="http://novnc.test/")

    obs = await env._setup_task()
    # The centre is ESTABLISHED, not assumed: reset issues a real pyautogui
    # moveTo so the composited coordinate is where the guest pointer really is.
    assert bridge.post.await_args_list[1].args[:2] == (
        "/step",
        {"action": "pyautogui.moveTo(720, 450)", "pause": 0.0},
    )
    assert obs.image == b"overlay:720:450"
    assert calls == [(720, 450)]

    result = await env.step([
        make_tool_call(
            "click",
            {"coordinate": [250, 750]},
            call_id="call_click",
        )
    ])

    assert bridge.post.await_args_list[2].args[:2] == (
        "/step",
        {"action": "pyautogui.click(360, 675, clicks=1, button='left')", "pause": 0.0},
    )
    assert result.results[0].images[-1] == b"overlay:360:675"
    assert calls == [(720, 450), (360, 675)]


@pytest.mark.asyncio
async def test_t2_release_rows_malformed_known_action():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
    )
    env._last_screenshot = b"last-shot"
    env._bridge = AsyncMock()
    env._bridge.post.return_value = {
        "ok": True,
        "screenshot_b64": base64.b64encode(b"fresh-shot").decode(),
    }

    result = await env.step([
        make_tool_call(
            "click",
            {"coordinate": [None, None]},
            call_id="malformed_known_action",
        )
    ])

    # R3: a rejected GUI slot owes a frame, captured fresh rather than reusing
    # the previous turn's cached one.
    assert env._bridge.post.await_args_list[0].args[0] == "/screenshot"
    assert result.terminated is False
    assert result.truncated is False
    assert result.results[0].tool_call_id == "malformed_known_action"
    assert result.results[0].images[-1] == b"fresh-shot"
    assert result.results[0].text is None
    assert result.results[0].error == (
        "invalid arguments for click: coordinate values must be finite numbers"
    )
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("keys", "expected_error"),
    [
        (["plus"], "invalid arguments for key: unknown key token 'plus'"),
        ([" "], "invalid arguments for key: unknown key token ' '"),
    ],
)
async def test_key_noncanonical_token_returns_current_feedback_without_step(
    keys,
    expected_error,
):
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        post_action_delay=0.0,
    )
    bridge = AsyncMock()
    bridge.post.return_value = {
        "ok": True,
        "screenshot_b64": base64.b64encode(b"fresh-shot").decode(),
    }
    env._bridge = bridge

    result = await env.step([
        make_tool_call("key", {"keys": keys}, call_id="bad_key")
    ])

    assert [call.args[0] for call in bridge.post.await_args_list] == ["/screenshot"]
    assert result.results[0].tool_call_id == "bad_key"
    assert result.results[0].images[-1] == b"fresh-shot"
    assert result.results[0].error == expected_error
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_t2_release_rows_literal_unknown_tool():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
    )
    env._last_screenshot = b"last-shot"
    env._bridge = AsyncMock()

    result = await env.step([
        make_tool_call("foo", {}, call_id="literal_unknown_tool")
    ])

    env._bridge.post.assert_not_called()
    assert result.terminated is False
    assert result.truncated is False
    assert result.results[0].tool_call_id == "literal_unknown_tool"
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.results[0].error == "unknown tool: foo"
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_t2_release_rows_content_only_final_text():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True},
        {"ok": True, "reward": 0.75},
        {"ok": True, "screenshot_b64": base64.b64encode(b"final-shot").decode()},
    ]
    env._bridge = bridge

    actions = make_no_tool_call_final_actions("final text")
    result = await env.step(actions)

    assert tool_call_name(actions[0]) == "response"
    assert tool_call_arguments(actions[0]) == {"text": "final text"}
    assert "id" not in actions[0]
    assert "call_id" not in actions[0]
    assert bridge.post.await_args_list[0].args[:2] == (
        "/step",
        {"action": "DONE", "pause": 0.0},
    )
    assert bridge.post.await_args_list[1].args == ("/evaluate",)
    assert bridge.post.await_args_list[2].args == ("/screenshot",)
    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 0.75
    assert result.results == []


@pytest.mark.asyncio
async def test_t2_release_rows_image_data_binding():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        post_action_delay=0.0,
    )
    env._last_screenshot = b"reset-frame"
    bridge = AsyncMock()
    bridge.post.return_value = {
        "ok": True,
        "screenshot_b64": base64.b64encode(b"post-action-bound-frame").decode(),
    }
    env._bridge = bridge

    result = await env.step([
        make_tool_call(
            "click",
            {"coordinate": [500, 500]},
            call_id="image_data_binding",
        )
    ])

    assert bridge.post.await_args_list[0].args[:2] == (
        "/step",
        {"action": "pyautogui.click(720, 450, clicks=1, button='left')", "pause": 0.0},
    )
    assert result.results[0].tool_call_id == "image_data_binding"
    assert result.results[0].images[-1] == b"post-action-bound-frame"
    assert result.results[0].images[-1] != b"reset-frame"
    assert result.results[0].error is None
    assert result.results[0].metadata is None


@pytest.mark.asyncio
async def test_t2_release_rows_response_terminal_tool():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=["response"],
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True},
        {"ok": True, "reward": 1.0},
        {"ok": True, "screenshot_b64": base64.b64encode(b"final-shot").decode()},
    ]
    env._bridge = bridge

    result = await env.step([
        make_tool_call(
            "response",
            {"text": "done"},
            call_id="response_terminal_tool",
        )
    ])

    assert bridge.post.await_args_list[0].args[:2] == (
        "/step",
        {"action": "DONE", "pause": 0.0},
    )
    assert bridge.post.await_args_list[1].args == ("/evaluate",)
    assert bridge.post.await_args_list[2].args == ("/screenshot",)
    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 1.0
    assert result.info["stop_reason"] == "response"
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    # Same shape the unstamped internal ``response`` already had.
    assert result.results == []


@pytest.mark.asyncio
async def test_terminal_turn_preserves_preceding_malformed_feedback():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=["terminate"],
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True},
        {"ok": True, "reward": 1.0},
        {"ok": True, "screenshot_b64": base64.b64encode(b"shot").decode()},
    ]
    env._bridge = bridge

    result = await env.step([
        make_tool_call(
            "computer",
            {"actions": [{"coordinate": [1, 2]}]},
            call_id="bad_batch",
        ),
        make_tool_call("terminate", {"status": "success"}),
    ])

    assert env._step_count == 1
    assert result.terminated is True
    assert len(result.results) == 1
    assert result.results[0].tool_call_id == "bad_batch"
    assert result.results[0].images[-1] == b"shot"
    assert result.results[0].text is None
    assert result.results[0].error == (
        "invalid arguments for computer: "
        "computer.arguments.actions[0].action must be a non-empty string"
    )
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_termination_records_preceding_and_terminal_actions():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=["response"],
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True, "screenshot_b64": base64.b64encode(b"step-shot").decode()},
        {"ok": True},
        {"ok": True, "reward": 0.0},
        {"ok": True, "screenshot_b64": base64.b64encode(b"final-shot").decode()},
    ]
    env._bridge = bridge

    result = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 500]}),
            make_tool_call("response", {"text": "done"}),
        ]
    )

    assert result.info["executed_actions"] == [
        {
            "call": "click",
            "args": {"command": "pyautogui.click(720, 450, clicks=1, button='left')"},
        },
        {"call": "response", "args": {"command": None}},
    ]


@pytest.mark.asyncio
async def test_action_batch_returns_one_distinct_frame_per_action():
    """N executed actions → N frames, in action order, none a repeat of another.

    The bridge already returns the frame it grabbed after each action (post
    ``pause``), so the per-action record costs no extra round-trip. Distinct
    bytes are the point: repeating one cached frame N times would satisfy the
    count while carrying no new information. The third action is ``screenshot``:
    read-only actions get a frame too.
    """
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        post_action_delay=0.0,
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True, "screenshot_b64": base64.b64encode(b"frame-1").decode()},
        {"ok": True, "screenshot_b64": base64.b64encode(b"frame-2").decode()},
        {"ok": True, "screenshot_b64": base64.b64encode(b"frame-3").decode()},
    ]
    env._bridge = bridge

    result = await env.step([
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [100, 100]},
                    {"action": "type", "text": "hi"},
                    {"action": "screenshot"},
                ],
            },
            call_id="call_batch",
        )
    ])

    assert [call.args[0] for call in bridge.post.await_args_list] == [
        "/step", "/step", "/screenshot",
    ]
    assert result.results[0].tool_call_id == "call_batch"
    assert result.results[0].images == [b"frame-1", b"frame-2", b"frame-3"]
    assert result.results[0].error is None


@pytest.mark.asyncio
async def test_a_rejected_child_earns_a_frame_and_costs_no_sibling():
    """A slot the model got wrong owes a frame, and does not abort the batch.

    ``bogus`` is not a name the batch tool carries. Ingress rejects it and
    forwards it anyway, so this env answers it per slot: one frame repeating the
    screen it did not change, plus a model-visible reason. The valid sibling
    still reaches the bridge -- a model fault costs the model its own slot,
    never the batch.
    """
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        post_action_delay=0.0,
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True, "screenshot_b64": base64.b64encode(b"frame-1").decode()},
        {"ok": True, "screenshot_b64": base64.b64encode(b"frame-2").decode()},
    ]
    env._bridge = bridge

    result = await env.step([
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [100, 100]},
                    {"action": "bogus"},
                ],
            },
            call_id="call_batch",
        )
    ])

    # the sibling still ran
    assert "/step" in [call.args[0] for call in bridge.post.await_args_list]
    # two slots, two frames -- the rejected one included
    assert len(result.results[0].images) == 2
    assert "bogus" in result.results[0].error


@pytest.mark.asyncio
async def test_terminal_call_frame_closes_a_batch_of_per_action_frames():
    """A terminal call ran its own guest command, so ``_finish``'s frame is its own.

    One frame per action that ran: the click's frame, then the frame grabbed
    after the terminal DONE.
    """
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=["report_infeasible"],
        post_action_delay=0.0,
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True, "screenshot_b64": base64.b64encode(b"click-shot").decode()},
        {"ok": True},                                   # _finish's FAIL /step
        {"ok": True, "reward": 0.0},                    # /evaluate
        {"ok": True, "screenshot_b64": base64.b64encode(b"final-shot").decode()},
    ]
    env._bridge = bridge

    result = await env.step([
        make_tool_call("click", {"coordinate": [100, 100]}, call_id="call_click"),
        make_tool_call("report_infeasible", {"reason": "no app"}, call_id="call_stop"),
    ])

    assert result.terminated is True
    assert result.results[0].images == [b"click-shot", b"final-shot"]


@pytest.mark.asyncio
async def test_action_at_max_steps_truncates_and_evaluates():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        max_steps=1,
        post_action_delay=0.0,
    )
    bridge = AsyncMock()
    bridge.post.side_effect = [
        {"ok": True, "screenshot_b64": base64.b64encode(b"step-shot").decode()},
        {"ok": True, "reward": 0.25},
        {"ok": True, "screenshot_b64": base64.b64encode(b"final-shot").decode()},
    ]
    env._bridge = bridge

    result = await env.step([
        make_tool_call("click", {"coordinate": [500, 500]}, call_id="call_click")
    ])

    assert bridge.post.await_args_list[0].args[:2] == (
        "/step",
        {"action": "pyautogui.click(720, 450, clicks=1, button='left')", "pause": 0.0},
    )
    assert bridge.post.await_args_list[1].args == ("/evaluate",)
    assert bridge.post.await_args_list[2].args == ("/screenshot",)
    assert result.terminated is False
    assert result.truncated is True
    assert result.reward == 0.25
    assert result.info["stop_reason"] == "max_steps"
    assert env._finished is True
    assert result.results[0].tool_call_id == "call_click"
    assert result.results[0].images[-1] == b"final-shot"
    assert result.results[0].error is None


@pytest.mark.asyncio
async def test_unknown_action_fails_as_unsupported_action():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
    )
    env._bridge = AsyncMock()

    result = await env.step([make_tool_call("frobnicate", call_id="call_frob")])

    env._bridge.post.assert_not_called()
    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_frob"
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.results[0].error == "unknown tool: frobnicate"
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_bridge_step_failure_returns_error_with_last_screenshot():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
    )
    env._last_screenshot = b"last-shot"

    async def fail_step(path, *args, **kwargs):
        if path == "/step":
            raise RuntimeError("bridge step failed")
        return {"ok": True, "screenshot_b64": base64.b64encode(b"new-shot").decode()}

    bridge = AsyncMock()
    bridge.post.side_effect = fail_step
    env._bridge = bridge

    result = await env.step([
        make_tool_call("click", {"coordinate": [500, 500]}, call_id="call_click")
    ])

    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_click"
    assert result.results[0].images[-1] == b"last-shot"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "click failed: execution failed"


@pytest.mark.asyncio
async def test_malformed_click_coordinate_returns_error_with_last_screenshot():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
    )
    env._last_screenshot = b"last-shot"
    env._bridge = AsyncMock()
    env._bridge.post.return_value = {
        "ok": True,
        "screenshot_b64": base64.b64encode(b"fresh-shot").decode(),
    }

    result = await env.step([
        make_tool_call("click", {}, call_id="call_click")
    ])

    # R3: the rejected slot owes a frame, and waa CAN capture one -- its
    # read-only arm already does exactly this. So the model sees the screen as
    # it is now, not the cached frame from the previous turn.
    assert env._bridge.post.await_args_list[0].args[0] == "/screenshot"
    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_click"
    assert result.results[0].images[-1] == b"fresh-shot"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == (
        "invalid arguments for click: coordinate is required"
    )


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("WAA_DOCKER") != "1", reason="Set WAA_DOCKER=1")
@pytest.mark.asyncio
async def test_live_reset_returns_png():
    task_id = gym.registry.task_ids("waa", split="eval")[0]
    env = gym.make(f"waa@{task_id}", extra_tools=["terminate"])
    try:
        obs = await env.reset()
        screenshot = obs.image or b""
        assert screenshot.startswith(b"\x89PNG")
        evaluated = await env.step(
            [make_tool_call("terminate", {"status": "success"})]
        )
        assert evaluated.reward is not None
        assert 0.0 <= evaluated.reward <= 1.0
    finally:
        await env.close()
