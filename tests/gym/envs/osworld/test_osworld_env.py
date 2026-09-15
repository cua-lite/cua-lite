"""Tests for the osworld (v1) CUA-Lite gym environment.

Eval-in-container design: the host is a thin JSON-RPC client (no desktop_env). Unit tests
mock the in-container server RPC (`_rpc`) — no Docker / desktop_env needed. Live Docker/VM
lifecycle tests are `pytest.mark.live` (need install.sh's build + the qcow2).

Run:
    uv run python -m pytest tests/gym/envs/osworld/test_osworld_env.py -v
"""

from __future__ import annotations

import ast
import base64
import ctypes
import errno
import fnmatch
import json
import os
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest

import lite.gym.envs.osworld.main as m
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.gym.errors import TrueInfraFailure

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PNG_B64 = base64.b64encode(b"screenshot").decode("ascii")


class _FakeContainer:
    base_url = "http://localhost:12345"
    name = "lite-env-local-osworld-task-abc"


def _make_env(
    monkeypatch,
    *,
    max_steps=15,
    eval_ret=None,
    extra_tools=None,
    screenshot_b64=_PNG_B64,
):
    """An OSWorldEnv whose container RPC is mocked; returns (env, calls)."""
    env = m.OSWorldEnv.__new__(m.OSWorldEnv)
    env._config = m.OSWorldConfig(domain="libreoffice_calc", task_id="t1")
    env._max_steps = max_steps
    env._post_action_delay = 0.0
    env._step_count = 0
    env._pending_cf_future = None
    env._container = _FakeContainer()
    env._valid_actions = None
    env._extra_tool_schemas = m.OSWorldEnv.extra_tool_schemas(extra_tools)

    calls: list[tuple[str, dict]] = []

    def fake_rpc(base, path, body=None, timeout=None):
        calls.append((path, body or {}))
        if path == "/evaluate":
            return eval_ret if eval_ret is not None else {"reward": 1.0}
        if path == "/screenshot":
            return {"screenshot_b64": screenshot_b64}
        return {"ok": True}

    monkeypatch.setattr(m, "_rpc", fake_rpc)
    return env, calls


def _step_cmds(calls):
    return [b.get("cmd") for p, b in calls if p == "/step"]


def test_qcow2_gate_checks_size(tmp_path, monkeypatch):
    qcow2 = tmp_path / "Ubuntu.qcow2"
    qcow2.write_bytes(b"tiny")
    monkeypatch.setattr(m, "_QCOW2", str(qcow2))
    monkeypatch.setattr(m, "_QCOW2_SIZE", 10)
    with pytest.raises(Exception, match="expected 10"):
        m._check_qcow2()


def test_release_manifest_matches_runtime_constants():
    release = json.loads((Path(m.ENV_DIR) / "data" / "release.json").read_text())
    assert int(release["qcow2"]["size"]) == m._QCOW2_SIZE
    assert release["qcow2"]["filename"] == "Ubuntu.qcow2"
    assert release["qcow2"]["url"]


def test_image_disables_thp_before_original_entrypoint():
    dockerfile = (Path(m.ENV_DIR) / "docker" / "Dockerfile").read_text()
    entrypoint = next(
        line.removeprefix("ENTRYPOINT ")
        for line in dockerfile.splitlines()
        if line.startswith("ENTRYPOINT ")
    )
    assert json.loads(entrypoint) == [
        "/usr/bin/tini",
        "-s",
        "/opt/venv/bin/python",
        "/usr/local/bin/disable_thp.py",
        "/run/entry.sh",
    ]
    assert "COPY disable_thp.py /usr/local/bin/disable_thp.py" in dockerfile


@pytest.mark.skipif(sys.platform != "linux", reason="THP prctl is Linux-specific")
def test_disable_thp_survives_exec_and_preserves_arguments_and_exit_code():
    wrapper = Path(m.ENV_DIR) / "docker" / "disable_thp.py"
    args = ["with spaces", "", "literal $HOME; 'quoted'", "中文"]
    # Start with THP enabled, so this proves the wrapper changes its process only.
    launcher = (
        "import ctypes, os, runpy, sys; "
        "assert ctypes.CDLL(None).prctl(41, 0, 0, 0, 0) == 0; "
        "print(os.getpid(), flush=True); "
        "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    probe = (
        "import ctypes, json, os, sys; "
        "print(json.dumps([ctypes.CDLL(None).prctl(42, 0, 0, 0, 0), "
        "os.getpid(), sys.argv[1:]])); sys.exit(23)"
    )
    result = subprocess.run(
        [sys.executable, "-c", launcher, str(wrapper), sys.executable, "-c", probe, *args],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 23, result.stderr
    original_pid, payload = result.stdout.splitlines()
    assert json.loads(payload) == [1, int(original_pid), args]


def test_disable_thp_failure_does_not_exec(monkeypatch):
    libc = Mock()
    libc.prctl.return_value = -1
    monkeypatch.setattr(ctypes, "CDLL", Mock(return_value=libc))
    monkeypatch.setattr(ctypes, "get_errno", lambda: errno.EPERM)
    execvp = Mock()
    monkeypatch.setattr(os, "execvp", execvp)
    monkeypatch.setattr(sys, "argv", ["disable_thp.py", "/run/entry.sh"])

    with pytest.raises(OSError, match="PR_SET_THP_DISABLE") as error:
        runpy.run_path(str(Path(m.ENV_DIR) / "docker" / "disable_thp.py"), run_name="__main__")

    assert error.value.errno == errno.EPERM
    execvp.assert_not_called()


def test_osworld_import_time_json_is_declared_for_wheel():
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    patterns = package_data["*"]
    assert "lite.gym.envs.osworld" not in package_data
    assert "lite.gym.envs.osworld_2" not in package_data
    required = {
        "data/release.json",
        "data/tasks.json",
        "data/test_v2.json",
        "data/_service_deps.json",
        "data/capabilities/visual_spatial_precision.json",
    }
    assert all(any(fnmatch.fnmatch(path, pattern) for pattern in patterns) for path in required)


def test_pyautogui_translation_coerces_drag_duration():
    command = m.to_pyautogui(
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
def test_pyautogui_translation_rejects_bad_model_durations(
    name,
    arguments,
    expected_error,
):
    with pytest.raises(ValueError, match=expected_error):
        m.to_pyautogui(name, arguments, 1000, 1000)


@pytest.mark.asyncio
async def test_direct_reset_runs_full_dependency_gate(monkeypatch):
    env, _calls = _make_env(monkeypatch)
    checks: list[str] = []

    monkeypatch.setattr(m, "_check_kvm", lambda: checks.append("kvm"))
    monkeypatch.setattr(m, "_check_tun", lambda: checks.append("tun"))
    monkeypatch.setattr(m, "_check_image", lambda: checks.append("image"))

    def fail_qcow2():
        checks.append("qcow2")
        raise RuntimeError("qcow2 gate")

    monkeypatch.setattr(m, "_check_qcow2", fail_qcow2)

    with pytest.raises(RuntimeError, match="qcow2 gate"):
        await env.reset()
    assert checks == ["kvm", "tun", "image", "qcow2"]


@pytest.mark.asyncio
async def test_fail_forwarded_on_terminate_failure(monkeypatch):
    env, calls = _make_env(monkeypatch, extra_tools=["terminate"])
    r = await env.step(
        [make_tool_call("terminate", {"status": "failure"}, call_id="call_terminate")]
    )
    assert ("/step", {"cmd": "FAIL", "pause": 0}) in calls
    assert any(p == "/evaluate" for p, _ in calls)
    assert r.terminated is True and r.reward == 1.0
    assert r.info["stop_reason"] == "terminate"


@pytest.mark.asyncio
async def test_no_fail_on_terminate_success(monkeypatch):
    env, calls = _make_env(monkeypatch, extra_tools=["terminate"])
    await env.step([make_tool_call("terminate", {"status": "success"})])
    assert "FAIL" not in _step_cmds(calls)


@pytest.mark.asyncio
async def test_no_fail_on_response(monkeypatch):
    env, calls = _make_env(monkeypatch, extra_tools=["response"])
    await env.step([make_tool_call("response", {"text": "done"})])
    assert "FAIL" not in _step_cmds(calls)


@pytest.mark.asyncio
async def test_report_infeasible_forwards_fail(monkeypatch):
    env, calls = _make_env(monkeypatch, extra_tools=["report_infeasible"])
    r = await env.step(
        [
            make_tool_call(
                "report_infeasible",
                {"reason": "no app"},
                call_id="call_report_infeasible",
            )
        ]
    )
    assert ("/step", {"cmd": "FAIL", "pause": 0}) in calls
    assert r.terminated is True and r.reward == 1.0
    assert r.info["stop_reason"] == "report_infeasible"


@pytest.mark.asyncio
async def test_response_infeasible_text_forwards_fail(monkeypatch):
    env, calls = _make_env(monkeypatch, extra_tools=["response"])
    r = await env.step([make_tool_call("response", {"text": "This is [INFEASIBLE]."})])
    assert ("/step", {"cmd": "FAIL", "pause": 0}) in calls
    assert r.terminated is True


@pytest.mark.asyncio
async def test_pyautogui_action_translated_and_forwarded(monkeypatch):
    env, calls = _make_env(monkeypatch)
    await env.step([make_tool_call("click", {"coordinate": [500, 500]})])
    cmds = _step_cmds(calls)
    assert len(cmds) == 1 and cmds[0].startswith("pyautogui.click(")


@pytest.mark.asyncio
async def test_gui_action_at_max_steps_truncates_and_evaluates(monkeypatch):
    env, calls = _make_env(
        monkeypatch,
        max_steps=1,
        eval_ret={"reward": 0.25},
    )

    result = await env.step(
        [make_tool_call("click", {"coordinate": [500, 500]}, call_id="call_click")]
    )

    assert [p for p, _ in calls] == ["/step", "/screenshot", "/evaluate"]
    assert result.terminated is False
    assert result.truncated is True
    assert result.reward == 0.25
    assert result.info["stop_reason"] == "max_steps"
    assert result.results[0].tool_call_id == "call_click"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].error is None
    assert result.info["executed_actions"][0]["call"].startswith("pyautogui.click(")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "expected_error", "expect_current_image"),
    [
        ("response", {"text": "done"}, "response is not available in this task.", False),
        (
            "terminate",
            {"status": "success"},
            "terminate is not available in this task.",
            False,
        ),
        (
            "report_infeasible",
            {"reason": "blocked"},
            "report_infeasible is not available in this task.",
            False,
        ),
        ("frobnicate", {}, "unknown tool: frobnicate", False),
    ],
)
async def test_inactive_or_unknown_tool_returns_paired_feedback(
    monkeypatch,
    name,
    arguments,
    expected_error,
    expect_current_image,
):
    env, calls = _make_env(monkeypatch)

    result = await env.step([make_tool_call(name, arguments, call_id=f"call_{name}")])

    assert not any(p == "/step" for p, _ in calls)
    assert not any(p == "/evaluate" for p, _ in calls)
    assert result.terminated is False
    assert result.truncated is False
    assert result.results[0].tool_call_id == f"call_{name}"
    assert (result.results[0].images[-1] if result.results[0].images else None) == (
        b"screenshot" if expect_current_image else None
    )
    assert result.results[0].text is None
    assert result.results[0].error == expected_error
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "expected_error", "expect_current_image"),
    [
        ("type", {"text": ""}, "unsupported action: type", False),
        (
            "type",
            {"text": 123},
            "invalid arguments for type: type.arguments.text must be a string",
            True,
        ),
        ("key", {"keys": []}, "invalid arguments for key: key.keys must not be empty", True),
        (
            "key",
            {"keys": ["plus"]},
            "invalid arguments for key: unknown key token 'plus'",
            True,
        ),
        (
            "key",
            {"keys": [" "]},
            "invalid arguments for key: unknown key token ' '",
            True,
        ),
        (
            "key",
            {"keys": ["Ctrl"]},
            "invalid arguments for key: keys must be lowercase tokens; "
            "split chords into separate keys; got 'Ctrl'",
            True,
        ),
    ],
)
async def test_route_known_gui_errors_return_expected_carrier(
    monkeypatch,
    name,
    arguments,
    expected_error,
    expect_current_image,
):
    env, calls = _make_env(monkeypatch)

    result = await env.step([make_tool_call(name, arguments, call_id=f"call_{name}")])

    assert [p for p, _ in calls] == ["/screenshot"]
    assert env._step_count == 1
    assert result.reward is None
    assert result.terminated is False
    assert result.truncated is False
    assert result.results[0].tool_call_id == f"call_{name}"
    assert (result.results[0].images[-1] if result.results[0].images else None) == (
        b"screenshot" if expect_current_image else None
    )
    assert result.results[0].text is None
    assert expected_error in result.results[0].error
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_bad_wait_duration_returns_tool_result_error_with_screenshot(monkeypatch):
    env, calls = _make_env(monkeypatch)

    result = await env.step([make_tool_call("wait", {"duration": 31}, call_id="call_wait")])

    assert [p for p, _ in calls] == ["/screenshot"]
    assert result.results[0].tool_call_id == "call_wait"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].text is None
    assert result.results[0].error == ("invalid arguments for wait: wait.duration must be <= 30")
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_route_known_gui_error_at_max_steps_truncates_and_evaluates(monkeypatch):
    env, calls = _make_env(
        monkeypatch,
        max_steps=1,
        eval_ret={"reward": 0.5},
    )

    result = await env.step([make_tool_call("key", {"keys": ["Ctrl"]}, call_id="call_key")])

    assert [p for p, _ in calls] == ["/screenshot", "/evaluate"]
    assert env._step_count == 1
    assert result.reward == 0.5
    assert result.terminated is False
    assert result.truncated is True
    assert result.results[0].tool_call_id == "call_key"
    assert result.results[0].images[-1] == b"screenshot"
    assert (
        "invalid arguments for key: keys must be lowercase tokens; "
        "split chords into separate keys; got 'Ctrl'"
    ) in (result.results[0].error)
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_backend_step_rpc_exception_returns_tool_result_error_with_screenshot(monkeypatch):
    env, _ = _make_env(monkeypatch)
    calls: list[tuple[str, dict]] = []

    def fake_rpc(base, path, body=None, timeout=None):
        calls.append((path, body or {}))
        if path == "/step":
            raise RuntimeError("display pipe closed")
        if path == "/screenshot":
            return {"screenshot_b64": _PNG_B64}
        return {"ok": True}

    monkeypatch.setattr(m, "_rpc", fake_rpc)

    result = await env.step(
        [make_tool_call("click", {"coordinate": [500, 500]}, call_id="call_click")]
    )

    assert any(p == "/screenshot" for p, _ in calls)
    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_click"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].error == "click failed: execution failed"
    assert "display pipe closed" not in result.results[0].error
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_malformed_coordinate_returns_tool_result_error_with_screenshot(monkeypatch):
    env, calls = _make_env(monkeypatch)

    result = await env.step([make_tool_call("click", {}, call_id="call_click")])

    assert not any(p == "/step" for p, _ in calls)
    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_click"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].error == (
        "invalid arguments for click: arguments could not be interpreted"
    )
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_fail_rpc_exception_returns_tool_result_error_with_screenshot(monkeypatch):
    env, _ = _make_env(monkeypatch, extra_tools=["terminate"])

    def fake_rpc(base, path, body=None, timeout=None):
        if path == "/step":
            raise RuntimeError("fail command rejected")
        if path == "/screenshot":
            return {"screenshot_b64": _PNG_B64}
        if path == "/evaluate":
            return {"reward": 0.0}
        return {"ok": True}

    monkeypatch.setattr(m, "_rpc", fake_rpc)

    result = await env.step(
        [
            make_tool_call(
                "terminate",
                {"status": "failure"},
                call_id="call_fail",
            )
        ]
    )

    assert result.terminated is True
    assert result.results[0].tool_call_id == "call_fail"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].error == "terminate failed: execution failed"
    assert "fail command rejected" not in result.results[0].error
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_release_rows_active_known_tool(monkeypatch):
    env, calls = _make_env(monkeypatch)

    result = await env.step(
        [
            make_tool_call(
                "click",
                {"coordinate": [500, 500]},
                call_id="active_known_tool",
            )
        ]
    )

    assert [p for p, _ in calls] == ["/step", "/screenshot"]
    assert result.terminated is False
    assert result.truncated is False
    assert result.results[0].tool_call_id == "active_known_tool"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].text is None
    assert result.results[0].error is None
    assert result.results[0].metadata is None


@pytest.mark.asyncio
async def test_t2_release_rows_malformed_known_action(monkeypatch):
    env, calls = _make_env(monkeypatch)

    result = await env.step(
        [
            make_tool_call(
                "click",
                {"coordinate": [None, None]},
                call_id="malformed_known_action",
            )
        ]
    )

    assert [p for p, _ in calls] == ["/screenshot"]
    assert result.terminated is False
    assert result.truncated is False
    assert result.results[0].tool_call_id == "malformed_known_action"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].text is None
    assert result.results[0].error == (
        "invalid arguments for click: arguments could not be interpreted"
    )
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_t2_release_rows_literal_unknown_tool(monkeypatch):
    env, calls = _make_env(monkeypatch)

    result = await env.step([make_tool_call("foo", {}, call_id="literal_unknown_tool")])

    assert [p for p, _ in calls] == ["/screenshot"]
    assert result.terminated is False
    assert result.truncated is False
    assert result.results[0].tool_call_id == "literal_unknown_tool"
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.results[0].error == "unknown tool: foo"
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_t2_release_rows_content_only_final_text(monkeypatch):
    env, calls = _make_env(monkeypatch, eval_ret={"reward": 0.75})

    actions = make_no_tool_call_final_actions("final text")
    result = await env.step(actions)

    assert tool_call_name(actions[0]) == "response"
    assert tool_call_arguments(actions[0]) == {"text": "final text"}
    assert "id" not in actions[0]
    assert "call_id" not in actions[0]
    assert [p for p, _ in calls] == ["/screenshot", "/evaluate"]
    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 0.75
    assert result.results == []


@pytest.mark.asyncio
async def test_t2_release_rows_image_data_binding(monkeypatch):
    bound_b64 = base64.b64encode(b"post-action-bound-frame").decode("ascii")
    env, calls = _make_env(monkeypatch, screenshot_b64=bound_b64)

    result = await env.step(
        [
            make_tool_call(
                "click",
                {"coordinate": [500, 500]},
                call_id="image_data_binding",
            )
        ]
    )

    assert [p for p, _ in calls] == ["/step", "/screenshot"]
    assert result.results[0].tool_call_id == "image_data_binding"
    assert result.results[0].images[-1] == b"post-action-bound-frame"
    assert result.results[0].images[-1] != b"reset-frame"
    assert result.results[0].error is None
    assert result.results[0].metadata is None


@pytest.mark.asyncio
async def test_t2_release_rows_response_terminal_tool(monkeypatch):
    env, calls = _make_env(
        monkeypatch,
        extra_tools=["response"],
        eval_ret={"reward": 1.0},
    )

    result = await env.step(
        [
            make_tool_call(
                "response",
                {"text": "done"},
                call_id="response_terminal_tool",
            )
        ]
    )

    assert [p for p, _ in calls] == ["/screenshot", "/evaluate"]
    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 1.0
    assert result.info["stop_reason"] == "response"
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    # Same shape the unstamped internal ``response`` already had.
    assert result.results == []


def _frame_sequence_rpc(calls, frames, *, failing_step_index=None):
    """An ``_rpc`` stub whose ``/screenshot`` returns a DIFFERENT frame each time.

    Distinct bytes are the point: a per-action frame list that repeats one cached
    frame satisfies the count while carrying no new information.
    """
    shots = iter(frames)
    steps = 0

    def fake_rpc(base, path, body=None, timeout=None):
        nonlocal steps
        calls.append((path, body or {}))
        if path == "/step":
            steps += 1
            if failing_step_index is not None and steps - 1 == failing_step_index:
                raise RuntimeError("display pipe closed")
            return {"ok": True}
        if path == "/screenshot":
            return {"screenshot_b64": base64.b64encode(next(shots)).decode("ascii")}
        return {"ok": True}

    return fake_rpc


@pytest.mark.asyncio
async def test_action_batch_returns_one_distinct_frame_per_action(monkeypatch):
    """N executed actions → N frames, in action order, none a repeat of another.

    The third action is ``screenshot``: read-only actions get a frame too, so the
    frame count never depends on what the actions were.
    """
    env, _ = _make_env(monkeypatch)
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(m, "_rpc", _frame_sequence_rpc(calls, [b"frame-1", b"frame-2", b"frame-3"]))

    result = await env.step(
        [
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
        ]
    )

    # Each frame is grabbed right after its own action, not once at the end.
    assert [p for p, _ in calls] == [
        "/step",
        "/screenshot",
        "/step",
        "/screenshot",
        "/screenshot",
    ]
    assert result.results[0].tool_call_id == "call_batch"
    assert result.results[0].images == [b"frame-1", b"frame-2", b"frame-3"]
    assert result.results[0].error is None


@pytest.mark.asyncio
async def test_a_dead_backend_raises_but_a_guest_error_stays_model_visible(monkeypatch):
    """R1 vs R2, at the one place they are easy to confuse.

    A transport failure means the container is GONE -- the model cannot cause it
    and must never be shown it, so it raises and the trajectory is retried. A
    non-200 is NOT separable: the guest crashing on a pyautogui string the MODEL
    supplied looks identical to a server fault, so it stays model-visible rather
    than discarding a trajectory that may be the model's own mistake.
    """
    env, _ = _make_env(monkeypatch)

    def dead_backend(base, path, body=None, timeout=None):
        if path == "/step":
            raise ConnectionRefusedError("osworld server unreachable")
        return {"screenshot_b64": base64.b64encode(b"shot").decode()}

    monkeypatch.setattr(m, "_rpc", dead_backend)
    with pytest.raises(TrueInfraFailure):
        await env.step(
            [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "click", "coordinate": [1, 1]},
                        ]
                    },
                    call_id="c1",
                )
            ]
        )

    env2, _ = _make_env(monkeypatch)

    def guest_error(base, path, body=None, timeout=None):
        if path == "/step":
            raise RuntimeError("osworld server /step -> HTTP 500: pyautogui blew up")
        return {"screenshot_b64": base64.b64encode(b"shot").decode()}

    monkeypatch.setattr(m, "_rpc", guest_error)
    result = await env2.step(
        [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [1, 1]},
                    ]
                },
                call_id="c1",
            )
        ]
    )
    assert result.results[0].error, "a guest-side failure must stay model-visible"


@pytest.mark.asyncio
async def test_a_rejected_child_earns_a_frame_and_costs_no_sibling(monkeypatch):
    """Contract rows 2 and 9: a GUI action the model got wrong owes a frame.

    ``bogus`` is a name the batch tool does not carry. Ingress rejects it and
    forwards it anyway, so the env answers it per action: one frame (repeating
    the screen it did not change) plus a model-visible reason -- and its valid
    sibling still runs. Without the frame the model is told it was wrong and
    shown nothing to correct against.
    """
    env, _ = _make_env(monkeypatch)
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(m, "_rpc", _frame_sequence_rpc(calls, [b"frame-1", b"frame-2"]))

    result = await env.step(
        [
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
        ]
    )

    # the valid sibling still reached the guest
    assert "/step" in [p for p, _ in calls]
    # two slots, two frames -- the rejected one included
    assert len(result.results[0].images) == 2
    assert "bogus" in result.results[0].error


@pytest.mark.asyncio
async def test_aborted_action_batch_returns_a_frame_per_action_that_ran(monkeypatch):
    """An aborted batch returns frames only for the actions that DID run."""
    env, _ = _make_env(monkeypatch)
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        m,
        "_rpc",
        _frame_sequence_rpc(calls, [b"frame-1", b"frame-2"], failing_step_index=1),
    )

    result = await env.step(
        [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [100, 100]},
                        {"action": "click", "coordinate": [200, 200]},
                        {"action": "click", "coordinate": [300, 300]},
                    ],
                },
                call_id="call_batch",
            )
        ]
    )

    assert [p for p, _ in calls] == ["/step", "/screenshot", "/step"]
    assert result.results[0].images == [b"frame-1"]
    assert result.results[0].error == (
        "click failed: execution failed\n"
        "batch aborted: the 1 later action was not executed; "
        "a batch stops at the first rejected action"
    )


@pytest.mark.asyncio
async def test_turn_with_no_executed_action_still_returns_one_frame(monkeypatch):
    """Zero executed actions still owe the model one current observation."""
    env, calls = _make_env(monkeypatch)

    result = await env.step([make_tool_call("click", {}, call_id="call_click")])

    assert [p for p, _ in calls] == ["/screenshot"]
    assert result.results[0].images == [b"screenshot"]


# ── Registration (from vendored data/tasks.json — no desktop_env / no container) ──
def test_registration_369_tasks_325_scored():
    from lite.gym.registry import registry

    ids = registry.task_ids("osworld", split="eval")
    assert len(ids) == 369
    excl = {}
    for i in ids:
        r = registry.task_metadata("osworld", i).others.get("exclude_reason")
        if r:
            excl[r] = excl.get(r, 0) + 1
    assert excl == {"infeasible": 29, "google_auth": 8, "blocked": 7}
    assert len(ids) - sum(excl.values()) == 325


def test_container_name_disjoint_from_v2_and_lite():
    from lite.gym.envs.osworld.container import OSWorldContainerFactory

    f = OSWorldContainerFactory(qcow2_path="/x.qcow2", session_id="s", task_id="t1")
    name = f._make_name("abc123")
    assert "-osworld-" in name
    assert "-osworld_2-" not in name
    assert ".osworld-" not in name


# ── to_pyautogui action translation (the single shared source of truth for osworld + osworld_2) ──
_W, _H = 1920, 1080


@pytest.mark.parametrize(
    "name,args,expect",
    [
        ("click", {"coordinate": [500, 500]}, "pyautogui.click("),
        ("click", {"coordinate": [500, 500], "clicks": 2}, "pyautogui.doubleClick("),
        ("click", {"coordinate": [500, 500], "clicks": 3}, "pyautogui.tripleClick("),
        ("click", {"coordinate": [500, 500], "button": "right"}, "button='right'"),
        ("mouse_move", {"coordinate": [100, 200]}, "pyautogui.moveTo("),
        ("mouse_down", {"button": "middle"}, "pyautogui.mouseDown(button='middle')"),
        ("drag", {"start_coordinate": [10, 10], "coordinate": [90, 90]}, "pyautogui.dragTo("),
        ("drag", {"coordinate": [90, 90]}, "duration=0.5"),
        ("type", {"text": "hi there"}, "pyautogui.typewrite('hi there')"),
        ("key", {"keys": ["ctrl", "c"]}, "pyautogui.hotkey("),
        (
            "key",
            {"keys": ["ctrl", "+", "-", "=", "/"]},
            "pyautogui.hotkey('ctrl', '+', '-', '=', '/')",
        ),
        ("hold_key", {"keys": ["shift"], "duration": 0.5}, "time.sleep(0.5)"),
        ("wait", {"duration": 2.5}, "time.sleep(2.5)"),
    ],
)
def test_to_pyautogui_verbs(name, args, expect):
    from lite.gym.envs.osworld.main import to_pyautogui

    assert expect in to_pyautogui(name, args, _W, _H)


def test_drag_without_start_omits_moveto():
    """start_coordinate optional → no pre-`moveTo`, so pyautogui.dragTo starts from
    the current cursor (matches waa / SandboxBaseEnv "drag from current cursor")."""
    from lite.gym.envs.osworld.main import to_pyautogui

    with_start = to_pyautogui(
        "drag", {"start_coordinate": [10, 10], "coordinate": [90, 90]}, _W, _H
    )
    without_start = to_pyautogui("drag", {"coordinate": [90, 90]}, _W, _H)
    assert "moveTo" in with_start and "dragTo" in with_start
    assert "moveTo" not in without_start and "dragTo" in without_start


def test_to_pyautogui_type_splits_less_than():
    """`<` is emitted as hotkey('shift', ',') host-side so no typewrite literal contains `<`.

    Upstream OSWorld's _fix_pyautogui_less_than_bug corrupts typewrite literals containing
    `<` plus escapes (unicode_escape + unescaped re-embed → SyntaxError → the whole action
    is silently dropped); with `<`-free literals it leaves the command untouched.
    """
    from lite.gym.envs.osworld.main import to_pyautogui as tp

    heredoc = "python3 - <<'PY'\nx = 1 < 2\nprint('done')\nPY\n"
    cmd = tp("type", {"text": heredoc}, _W, _H)
    assert "<" not in cmd
    assert cmd.count("pyautogui.hotkey('shift', ',')") == 3
    compile(cmd, "<cmd>", "exec")  # the guest runs this via `python -c`
    # segments reassemble to the original text
    rebuilt = ""
    for call in cmd.split("; "):
        if call == "pyautogui.hotkey('shift', ',')":
            rebuilt += "<"
        else:
            rebuilt += ast.literal_eval(call.removeprefix("pyautogui.typewrite(").removesuffix(")"))
    assert rebuilt == heredoc
    assert tp("type", {"text": "<"}, _W, _H) == "pyautogui.hotkey('shift', ',')"


def test_to_pyautogui_scroll_sign_convention():
    from lite.gym.envs.osworld.main import to_pyautogui as tp

    assert tp("scroll", {"direction": "down", "amount": 3}, _W, _H).startswith(
        "pyautogui.scroll(-3"
    )
    assert tp("scroll", {"direction": "up", "amount": 3}, _W, _H).startswith("pyautogui.scroll(3")
    assert tp("scroll", {"direction": "right", "amount": 2}, _W, _H).startswith(
        "pyautogui.hscroll(2"
    )
    assert tp("scroll", {"direction": "left", "amount": 2}, _W, _H).startswith(
        "pyautogui.hscroll(-2"
    )


def test_to_pyautogui_noop_and_coord_scaling():
    from lite.gym.envs.osworld.main import to_pyautogui as tp

    assert tp("screenshot", {}, _W, _H) is None
    assert tp("cursor_position", {}, _W, _H) is None
    assert tp("type", {"text": ""}, _W, _H) is None
    # coordinates are on a 0-1000 grid → [500,500] maps to the viewport centre (960, 540)
    s = tp("click", {"coordinate": [500, 500]}, _W, _H)
    assert "960" in s and "540" in s


def test_to_pyautogui_rejects_scalar_keys():
    from lite.gym.envs.osworld.main import to_pyautogui as tp

    with pytest.raises(ValueError, match="key.keys must be a list of strings"):
        tp("key", {"keys": "enter"}, _W, _H)


def test_to_pyautogui_rejects_empty_keys():
    # NOT a noop: ``keys`` is required with no default and env ingress does not
    # check argument presence, so an empty list must reach the model as an
    # argument error. It used to translate to ``None``, which ``step`` reported as
    # "unsupported action: key" -- a keypress that never happened, misdiagnosed.
    from lite.gym.envs.osworld.main import to_pyautogui as tp

    with pytest.raises(ValueError, match="key.keys must not be empty"):
        tp("key", {"keys": []}, _W, _H)


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["plus"], "unknown key token 'plus'"),
        ([" "], "unknown key token ' '"),
    ],
)
def test_to_pyautogui_rejects_noncanonical_key_tokens(keys, expected):
    from lite.gym.envs.osworld.main import to_pyautogui as tp

    with pytest.raises(ValueError, match=expected):
        tp("key", {"keys": keys}, _W, _H)


def test_to_pyautogui_rejects_invalid_type_text():
    from lite.gym.envs.osworld.main import to_pyautogui as tp

    with pytest.raises(TypeError, match="type.arguments.text must be a string"):
        tp("type", {"text": 123}, _W, _H)


@pytest.mark.parametrize(
    "name,args",
    [
        ("click", {}),
        ("mouse_move", {"coordinate": []}),
        ("drag", {"start_coordinate": [10, 10]}),
        ("drag", {"start_coordinate": [], "coordinate": [90, 90]}),
        ("scroll", {"coordinate": []}),
    ],
)
def test_to_pyautogui_rejects_malformed_coordinates(name, args):
    from lite.gym.envs.osworld.main import to_pyautogui as tp

    with pytest.raises(ValueError, match="arguments could not be interpreted"):
        tp(name, args, _W, _H)
