"""Tests for the osworld_2 (OSWorld-V2) CUA-Lite gym environment.

Eval-in-container design: the host is a thin JSON-RPC client (no desktop_env). Unit tests
mock the in-container server RPC (`_rpc`) — no Docker / V2 dist / gated downloads needed.
Live Docker/VM lifecycle tests are `pytest.mark.live` (need install.sh's build + gated assets).

Run:
    uv run python -m pytest tests/gym/envs/osworld_2/test_osworld_2.py -v
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import lite.gym.envs.osworld_2.main as m
import lite.gym.envs.osworld_2.container as c
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.results import project_tool_result_text

_PNG_B64 = base64.b64encode(b"screenshot").decode("ascii")
NO_BACKEND_LEAK_STRINGS = (
    "backend",
    "xdotool",
    "pyautogui",
    "pynput",
    "Playwright",
    "unsupported operand",
    "TypeError:",
    "ValueError:",
    "RuntimeError:",
    "NoneType",
)


def _assert_no_backend_leaks(text: str | None) -> None:
    assert text is not None
    for leak in NO_BACKEND_LEAK_STRINGS:
        assert leak not in text


class _FakeContainer:
    base_url = "http://localhost:12345"
    name = "lite-env-local-osworld_2-001-abc"


def _make_env(
    monkeypatch,
    *,
    max_steps=15,
    eval_ret=None,
    extra_tools=None,
    screenshot_b64=_PNG_B64,
):
    """An OSWorldV2Env whose container RPC is mocked; returns (env, calls)."""
    env = m.OSWorldV2Env.__new__(m.OSWorldV2Env)
    env._config = m.OSWorldV2Config(task_id="001")
    env._max_steps = max_steps
    env._post_action_delay = 0.0
    env._step_count = 0
    env._pending_cf_future = None
    env._container = _FakeContainer()
    env._valid_actions = None
    env._extra_tool_schemas = m.resolve_extra_tools(extra_tools, tools=m.OsworldTools)

    calls: list[tuple[str, dict]] = []

    def fake_rpc(base, path, body=None, timeout=None):
        calls.append((path, body or {}))
        if path == "/evaluate":
            return eval_ret if eval_ret is not None else {"reward": 1.0, "payload": None}
        if path == "/screenshot":
            return {"screenshot_b64": screenshot_b64}
        return {"ok": True}

    monkeypatch.setattr(m, "_rpc", fake_rpc)
    return env, calls


def _step_cmds(calls):
    return [b.get("cmd") for p, b in calls if p == "/step"]


def _tc(name: str, arguments: dict | None = None, *, call_id: str | None = None) -> dict:
    return make_tool_call(name, arguments, call_id=call_id)


def test_task_class_gate_requires_revision_stamp(tmp_path, monkeypatch):
    task_dir = tmp_path / "task_class"
    task_dir.mkdir()
    for task_id in m._TASK_IDS:
        (task_dir / f"task_{task_id}.py").write_text("# task\n")
    monkeypatch.setattr(m, "_TASK_CLASS_DIR", str(task_dir))
    with pytest.raises(Exception, match="gated task classes"):
        m._check_task_classes()

    (task_dir / ".task_class_revision").write_text(m._TASK_CLASS_IDENTITY + "\n")
    m._check_task_classes()

    (task_dir / f"task_{m._TASK_IDS[-1]}.py").unlink()
    with pytest.raises(Exception, match="required task files missing"):
        m._check_task_classes()


def test_release_manifest_matches_runtime_constants():
    release = json.loads((Path(m.ENV_DIR) / "data" / "release.json").read_text())
    assert int(release["qcow2"]["size"]) == m._QCOW2_SIZE
    assert release["qcow2"]["filename"] == "osworld-v2-ubuntu-x86.qcow2"
    assert release["qcow2"]["repo"] == "xlangai/v2-image"
    assert str(release["hf_revision"]) == m._HF_REVISION
    assert str(release["tasks"]["repo"]) == m._TASKS_REPO
    assert len(m._TASK_IDS) == 108
    assert m._TASK_CLASS_IDENTITY == (f"{m._TASKS_REPO}@{m._HF_REVISION}:{len(m._TASK_IDS)}")
    assert m._SERVICE_ENV["OSWORLD_BENCHMARK_RELEASE"] == release["hf_revision"]


def test_qcow2_gate_checks_size(tmp_path, monkeypatch):
    qcow2 = tmp_path / "disk.qcow2"
    qcow2.write_bytes(b"tiny")
    monkeypatch.setattr(m, "_QCOW2", str(qcow2))
    monkeypatch.setattr(m, "_QCOW2_SIZE", 10)
    with pytest.raises(Exception, match="expected 10"):
        m._check_qcow2()


def test_v21_assets_mounted_for_in_container_task_setup(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    call = {}
    monkeypatch.setattr(c, "docker_run_detached", lambda **kwargs: call.update(kwargs))
    monkeypatch.setattr(c.OSWorldV2Container, "_register", lambda self: None)
    monkeypatch.setattr(c.OSWorldV2Container, "_launch_server", lambda self: None)
    monkeypatch.setattr(c.OSWorldV2Container, "_wait_ready", lambda self: None)
    container = c.OSWorldV2Container(
        name="test-osworld_2-001", api_port=23001, task_id="103",
        qcow2_path="/tmp/disk.qcow2", task_class_dir="/tmp/task_class",
        asset_dir=str(assets),
        service_env={"OSWORLD_FILE_BASE_URL": str(assets), "OSWORLD_BENCHMARK_RELEASE": "osworld-v2.1"},
    )

    container._start_locked()

    assert f"{assets}:{assets}:ro" in call["volumes"]
    assert "/tmp/task_class/task_103.py:/opt/osworld-v2/evaluation_examples/task_class/task_103.py:ro" in call["volumes"]
    assert call["env"]["TASK_CLASS_DIR"] == "/opt/osworld-v2/evaluation_examples/task_class"
    assert call["env"]["OSWORLD_FILE_BASE_URL"] == str(assets)
    assert call["env"]["OSWORLD_BENCHMARK_RELEASE"] == "osworld-v2.1"

    volume_vm = c.OSWorldV2Container(
        name="test-osworld_2-082", api_port=23002, task_id="082", volume_size=100,
    )
    volume_vm._start_locked()
    assert call["env"]["DISK_SIZE"] == "100G"
    assert call["env"]["OSWORLD_VOLUME_SIZE"] == "100"


def test_nonstandard_website_port_reaches_generated_port_80_links(monkeypatch):
    calls = []
    monkeypatch.setattr(c.socket, "gethostbyname", lambda host: "169.229.219.180")
    monkeypatch.setattr(c.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    monkeypatch.setattr(c, "docker_run_detached", lambda **kwargs: None)
    monkeypatch.setattr(c.OSWorldV2Container, "_register", lambda self: None)
    monkeypatch.setattr(c.OSWorldV2Container, "_launch_server", lambda self: None)
    monkeypatch.setattr(c.OSWorldV2Container, "_wait_ready", lambda self: None)
    vm = c.OSWorldV2Container(
        name="test-osworld_2-057", api_port=23001, task_id="057", website=True,
        service_env={"WEBSITE_HOST_SUFFIX": "169.229.219.180.nip.io:30881"},
    )
    vm._start_locked()
    assert calls[0][-2:] == ["--to-destination", "169.229.219.180:30881"]
    vm.service_env["WEBSITE_HOST_SUFFIX"] = "web.hku.icu"
    vm._start_locked()
    assert len(calls) == 1


def test_freecad_reset_rejects_missing_guest_app(monkeypatch, tmp_path):
    import importlib.util
    import sys

    path = Path(c.__file__).parent / "docker" / "server.py"
    spec = importlib.util.spec_from_file_location("osworld2_server_test", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    (tmp_path / "task_103.py").touch()
    monkeypatch.setattr(server, "_TASK_CLASS_DIR", str(tmp_path))
    waits = []
    monkeypatch.setitem(sys.modules, "desktop_env.controllers.setup", SimpleNamespace(
        SetupController=lambda **kwargs: SimpleNamespace(
            execute=lambda command, **options: waits.append((command, options))
        ),
    ))
    checks = []

    def check(command, **kwargs):
        checks.append((command, kwargs))
        if len(checks) == 1:
            raise RuntimeError("FreeCAD missing")

    fake_env = SimpleNamespace(
        reset=lambda **kwargs: None,
        close=lambda: None,
        setup_controller=SimpleNamespace(execute=check),
        controller=SimpleNamespace(get_screenshot=lambda: b"image"),
    )
    monkeypatch.setattr(server, "_load_task_from_file", lambda _: {"related_apps": ["freecad"], "instruction": "cad"})
    monkeypatch.setattr(server, "_attached_desktop_env", lambda: fake_env)
    monkeypatch.setattr(server, "_encode_screenshot", lambda _: "image")
    with pytest.raises(RuntimeError, match="FreeCAD missing"):
        server.reset(server.ResetBody(task_id="103"))
    assert server.reset(server.ResetBody(task_id="103"))["instruction"] == "cad"
    assert len(waits) == 2 and all("fuser" in command and options["check"] for command, options in waits)
    assert all("{CLIENT_PASSWORD}" not in command and server._CLIENT_PASSWORD in command for command, _ in waits)
    assert checks == [(["which", "freecad"], {"timeout": 10, "check": True})] * 2


def test_freecad_scorer_runs_on_four_cpus_with_bounded_allocator(monkeypatch):
    import os
    import runpy
    import sys

    calls = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(16)))
    monkeypatch.setattr(os, "sched_setaffinity", lambda _, cpus: calls.append(("cpus", cpus)))
    monkeypatch.setattr(os, "execv", lambda *args: calls.append(("exec", args)))
    monkeypatch.setattr(sys, "argv", ["freecad-python", "-c", "import FreeCAD"])
    monkeypatch.setenv("MALLOC_ARENA_MAX", "8")
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")

    runpy.run_path(str(Path(c.__file__).parent / "docker" / "freecad_python.py"), run_name="__main__")

    assert len(calls[0][1]) == 4
    assert calls[1] == ("exec", ("/opt/cad-score/bin/python", [
        "/opt/cad-score/bin/python", "-c", "import FreeCAD",
    ]))
    assert os.environ["MALLOC_ARENA_MAX"] == "1"
    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"


def test_v21_local_website_probe_falls_back_to_http(monkeypatch):
    monkeypatch.setattr(m, "_WEBSITE_SUFFIX", "169.229.219.180.nip.io:30881")
    urls = []

    def get(url, **kwargs):
        urls.append(url)
        if url.startswith("https:"):
            raise requests.exceptions.SSLError("HTTP-only site")
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(m.requests, "get", get)
    m._check_website()
    assert [url.split("://", 1)[0] for url in urls] == ["https", "http"]


@pytest.mark.asyncio
async def test_direct_reset_runs_full_dependency_gate(monkeypatch):
    env, _calls = _make_env(monkeypatch)
    checks: list[str] = []

    monkeypatch.setattr(m, "_check_kvm", lambda: checks.append("kvm"))
    monkeypatch.setattr(m, "_check_tun", lambda: checks.append("tun"))
    monkeypatch.setattr(m, "_check_image", lambda: checks.append("image"))
    monkeypatch.setattr(m, "_check_qcow2", lambda: checks.append("qcow2"))

    def fail_task_classes():
        checks.append("task_classes")
        raise RuntimeError("task class gate")

    monkeypatch.setattr(m, "_check_task_classes", fail_task_classes)

    with pytest.raises(RuntimeError, match="task class gate"):
        await env.reset()
    assert checks == ["kvm", "tun", "image", "qcow2", "task_classes"]


@pytest.mark.asyncio
async def test_fail_forwarded_on_terminate_failure(monkeypatch):
    env, calls = _make_env(monkeypatch, extra_tools=["terminate"])
    r = await env.step([_tc("terminate", {"status": "failure"}, call_id="call_terminate")])
    assert ("/step", {"cmd": "FAIL", "pause": 0}) in calls
    assert any(p == "/evaluate" for p, _ in calls)
    assert r.terminated is True and r.reward == 1.0
    assert r.info["stop_reason"] == "terminate"


@pytest.mark.asyncio
async def test_no_fail_on_terminate_success(monkeypatch):
    env, calls = _make_env(monkeypatch, extra_tools=["terminate"])
    r = await env.step([_tc("terminate", {"status": "success"})])
    assert "FAIL" not in _step_cmds(calls)
    assert any(p == "/evaluate" for p, _ in calls)
    assert r.terminated is True and r.reward == 1.0


@pytest.mark.asyncio
async def test_no_fail_on_response(monkeypatch):
    env, calls = _make_env(monkeypatch, extra_tools=["response"])
    r = await env.step([_tc("response", {"text": "done"})])
    assert "FAIL" not in _step_cmds(calls)
    assert any(p == "/evaluate" for p, _ in calls)
    assert r.terminated is True and r.reward == 1.0


@pytest.mark.asyncio
async def test_report_infeasible_forwards_fail(monkeypatch):
    env, calls = _make_env(monkeypatch, extra_tools=["report_infeasible"])
    r = await env.step(
        [
            _tc(
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
    r = await env.step([_tc("response", {"text": "This is [INFEASIBLE]."})])
    assert ("/step", {"cmd": "FAIL", "pause": 0}) in calls
    assert r.terminated is True


@pytest.mark.asyncio
async def test_dict_evaluate_extracts_score_and_payload(monkeypatch):
    """A dict-returning evaluate() (server extracts ['score']) → reward + payload in info."""
    env, calls = _make_env(
        monkeypatch,
        eval_ret={"reward": 0.5, "payload": {"score": 0.5, "cp": [1, 0]}},
        extra_tools=["terminate"],
    )
    r = await env.step([_tc("terminate", {"status": "success"})])
    assert r.reward == 0.5
    assert r.info["evaluate_payload"] == {"score": 0.5, "cp": [1, 0]}


@pytest.mark.asyncio
async def test_pyautogui_action_translated_and_forwarded(monkeypatch):
    """A normal action → translated host-side via to_pyautogui and sent to /step."""
    env, calls = _make_env(monkeypatch)
    await env.step([_tc("click", {"coordinate": [500, 500]})])
    cmds = _step_cmds(calls)
    assert len(cmds) == 1 and cmds[0].startswith("pyautogui.click(")


def test_pyautogui_translation_accepts_canonical_key_glyphs():
    assert m.to_pyautogui("key", {"keys": ["ctrl", "+", "-", "="]}, 1920, 1080) == (
        "pyautogui.hotkey('ctrl', '+', '-', '=')"
    )


@pytest.mark.asyncio
async def test_gui_action_at_max_steps_truncates_and_evaluates(monkeypatch):
    env, calls = _make_env(
        monkeypatch,
        max_steps=1,
        eval_ret={"reward": 0.25, "payload": None},
    )

    result = await env.step([_tc("click", {"coordinate": [500, 500]}, call_id="call_click")])

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

    result = await env.step([_tc(name, arguments, call_id=f"call_{name}")])

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

    result = await env.step([_tc(name, arguments, call_id=f"call_{name}")])

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
async def test_route_known_gui_error_at_max_steps_truncates_and_evaluates(monkeypatch):
    env, calls = _make_env(
        monkeypatch,
        max_steps=1,
        eval_ret={"reward": 0.5, "payload": None},
    )

    result = await env.step([_tc("key", {"keys": ["Ctrl"]}, call_id="call_key")])

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

    def fake_rpc(base, path, body=None, timeout=None):
        if path == "/step":
            raise RuntimeError("display pipe closed")
        if path == "/screenshot":
            return {"screenshot_b64": _PNG_B64}
        return {"ok": True}

    monkeypatch.setattr(m, "_rpc", fake_rpc)

    result = await env.step([_tc("click", {"coordinate": [500, 500]}, call_id="call_click")])

    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_click"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].error == "click failed: execution failed"
    assert "display pipe closed" not in result.results[0].error
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_detail",
    [
        "xdotool key failed",
        "pyautogui.click failed",
        "pynput controller failed",
        "Playwright TimeoutError: target closed",
        "RuntimeError: backend crashed",
    ],
)
async def test_backend_step_error_detail_is_not_model_visible(
    monkeypatch,
    backend_detail: str,
):
    env, _ = _make_env(monkeypatch)

    def fake_rpc(base, path, body=None, timeout=None):
        if path == "/step":
            raise RuntimeError(backend_detail)
        if path == "/screenshot":
            return {"screenshot_b64": _PNG_B64}
        return {"ok": True}

    monkeypatch.setattr(m, "_rpc", fake_rpc)

    result = await env.step([_tc("click", {"coordinate": [500, 500]}, call_id="call_click")])

    visible_error = result.results[0].error
    assert visible_error == "click failed: execution failed"
    _assert_no_backend_leaks(visible_error)
    _assert_no_backend_leaks(project_tool_result_text(result.results[0].text, visible_error))


@pytest.mark.asyncio
async def test_malformed_coordinate_returns_tool_result_error_with_screenshot(monkeypatch):
    env, calls = _make_env(monkeypatch)

    result = await env.step([_tc("click", {}, call_id="call_click")])

    assert not any(p == "/step" for p, _ in calls)
    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_click"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].error == (
        "invalid arguments for click: arguments could not be interpreted"
    )
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_malformed_coordinate_internal_python_detail_is_not_model_visible(
    monkeypatch,
):
    env, calls = _make_env(monkeypatch)

    result = await env.step([_tc("click", {"coordinate": [None, None]}, call_id="call_click")])

    assert not any(p == "/step" for p, _ in calls)
    visible_error = result.results[0].error
    assert visible_error == ("invalid arguments for click: arguments could not be interpreted")
    _assert_no_backend_leaks(visible_error)
    _assert_no_backend_leaks(project_tool_result_text(result.results[0].text, visible_error))


@pytest.mark.asyncio
async def test_fail_rpc_exception_returns_tool_result_error_with_screenshot(monkeypatch):
    env, _ = _make_env(monkeypatch, extra_tools=["terminate"])

    def fake_rpc(base, path, body=None, timeout=None):
        if path == "/step":
            raise RuntimeError("fail command rejected")
        if path == "/screenshot":
            return {"screenshot_b64": _PNG_B64}
        if path == "/evaluate":
            return {"reward": 0.0, "payload": None}
        return {"ok": True}

    monkeypatch.setattr(m, "_rpc", fake_rpc)

    result = await env.step([_tc("terminate", {"status": "failure"}, call_id="call_fail")])

    assert result.terminated is True
    assert result.results[0].tool_call_id == "call_fail"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].error == "terminate failed: execution failed"
    assert "fail command rejected" not in result.results[0].error
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_release_rows_active_known_tool(monkeypatch):
    env, calls = _make_env(monkeypatch)

    result = await env.step([_tc("click", {"coordinate": [500, 500]}, call_id="active_known_tool")])

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
            _tc(
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

    result = await env.step([_tc("foo", {}, call_id="literal_unknown_tool")])

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
    env, calls = _make_env(monkeypatch, eval_ret={"reward": 0.75, "payload": None})

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
        [_tc("click", {"coordinate": [500, 500]}, call_id="image_data_binding")]
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
        eval_ret={"reward": 1.0, "payload": None},
    )

    result = await env.step([_tc("response", {"text": "done"}, call_id="response_terminal_tool")])

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


@pytest.mark.asyncio
async def test_action_batch_returns_one_distinct_frame_per_action(monkeypatch):
    """N executed actions → N frames, in action order, none a repeat of another.

    Distinct bytes are the point: a per-action frame list that repeats one cached
    frame satisfies the count while carrying no new information. The third action
    is ``screenshot``: read-only actions get a frame too, so the frame count never
    depends on what the actions were.
    """
    env, _ = _make_env(monkeypatch)
    calls: list[tuple[str, dict]] = []
    shots = iter([b"frame-1", b"frame-2", b"frame-3"])

    def fake_rpc(base, path, body=None, timeout=None):
        calls.append((path, body or {}))
        if path == "/screenshot":
            return {"screenshot_b64": base64.b64encode(next(shots)).decode("ascii")}
        return {"ok": True}

    monkeypatch.setattr(m, "_rpc", fake_rpc)

    result = await env.step(
        [
            _tc(
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
async def test_a_rejected_child_earns_a_frame_and_costs_no_sibling(monkeypatch):
    """A slot the model got wrong owes a frame, and does not abort the batch.

    ``bogus`` is not a name the batch tool carries. Ingress rejects it and
    forwards it anyway, so this env answers it per slot: one frame repeating the
    screen it did not change, plus a model-visible reason. The valid sibling
    still reaches the guest -- a model fault costs the model its own slot, never
    the batch.
    """
    env, _ = _make_env(monkeypatch)
    calls: list[tuple[str, dict]] = []
    shots = iter([b"frame-1", b"frame-2"])

    def fake_rpc(base, path, body=None, timeout=None):
        calls.append((path, body or {}))
        if path == "/screenshot":
            return {"screenshot_b64": base64.b64encode(next(shots)).decode("ascii")}
        return {"ok": True}

    monkeypatch.setattr(m, "_rpc", fake_rpc)

    result = await env.step(
        [
            _tc(
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

    assert "/step" in [p for p, _ in calls]  # the sibling still ran
    assert len(result.results[0].images) == 2  # both slots, rejected included
    assert "bogus" in result.results[0].error


@pytest.mark.asyncio
async def test_turn_with_no_executed_action_still_returns_one_frame(monkeypatch):
    """Zero executed actions still owe the model one current observation."""
    env, calls = _make_env(monkeypatch)

    result = await env.step([_tc("click", {}, call_id="call_click")])

    assert [p for p, _ in calls] == ["/screenshot"]
    assert result.results[0].images == [b"screenshot"]


# ── Registration (Tier 1, no gated code / no container) ────────────────────
def test_registration_108_tasks_and_hitl_excluded():
    from lite.gym.registry import registry

    ids = registry.task_ids("osworld_2", split="eval")
    assert len(ids) == 108
    excl = {}
    for i in ids:
        r = registry.task_metadata("osworld_2", i).others.get("exclude_reason")
        if r:
            excl[r] = excl.get(r, 0) + 1
    # HITL (>=6) is always excluded (capability manifest). The install-time service scan
    # (_service_deps.json) additionally tags website/gitlab (provisionable), multi_phase
    # (fidelity limit), and llm_judge (gated on host OPENAI_API_KEY — appears only when the key is
    # ABSENT). So the exact set is env-dependent, but HITL is the invariant and only these appear.
    assert excl.get("human_in_the_loop", 0) >= 6
    assert set(excl) <= {
        "human_in_the_loop",
        "website",
        "gitlab",
        "multi_phase",
        "llm_judge",
    }
    assert all(registry.task_metadata("osworld_2", i).others.get("capabilities") for i in ids)


def test_exclude_reason_precedence():
    """First-match-wins order: hitl > website > gitlab > llm_judge > multi_phase.

    The gate config is injected and does not depend on module state or default.yaml/env.
    """
    from lite.gym.envs.osworld_2.main import _exclude_reason

    CLOSED = dict(user_sim_model=None, website_suffix=None, gitlab_ok=False, has_openai_key=False)
    assert _exclude_reason("t", {"t"}, {}, **CLOSED) == "human_in_the_loop"  # hitl via id
    assert (
        _exclude_reason("t", set(), {"user_sim": True}, **CLOSED) == "human_in_the_loop"
    )  # hitl via user_sim dep
    assert (
        _exclude_reason("t", set(), {"website": True, "multi_phase": True}, **CLOSED) == "website"
    )  # website > fidelity
    assert (
        _exclude_reason("t", set(), {"gitlab": True, "llm_judge": True}, **CLOSED) == "gitlab"
    )  # gitlab > llm_judge
    assert (
        _exclude_reason("t", set(), {"llm_judge": True, "volume_size": 60}, **CLOSED) == "llm_judge"
    )
    assert (
        _exclude_reason("t", set(), {"multi_phase": True, "volume_size": 60}, **CLOSED)
        == "multi_phase"
    )
    assert _exclude_reason("t", set(), {"volume_size": 60}, **CLOSED) is None
    assert _exclude_reason("t", set(), {}, **CLOSED) is None
    # gates OPEN: provisioned service / key present → NOT excluded
    OPEN = dict(
        user_sim_model="gpt-4", website_suffix="web.hku.icu", gitlab_ok=True, has_openai_key=True
    )
    for d in ({"website": True}, {"gitlab": True}, {"llm_judge": True}, {"user_sim": True}):
        assert _exclude_reason("t", set(), d, **OPEN) is None


def test_metadata_exposes_service_dep_flags():
    """Static dep flags are surfaced in others for filtering when true."""
    from lite.gym.registry import registry

    metas = [
        registry.task_metadata("osworld_2", i).others
        for i in registry.task_ids("osworld_2", split="eval")
    ]
    for flag in ("website", "llm_judge"):
        assert any(o.get(flag) is True for o in metas), f"no task carries {flag}"
        assert all(o.get(flag) in (True, None) for o in metas)  # never present-and-False
    sizes = {i: registry.task_metadata("osworld_2", i).others.get("volume_size")
             for i in registry.task_ids("osworld_2", split="eval")}
    assert sizes["030"] == 50 and sizes["080"] == 60 and sizes["082"] == 100
    assert sum(size is not None for size in sizes.values()) == 18  # 023 also needs a user simulator
    assert registry.task_kwargs("osworld_2", "082")["reset_timeout"] == 1700.0


def test_service_env_only_carries_set_knobs():
    """_SERVICE_ENV drops unset knobs — no None/empty value can reach the container as an -e var."""
    import lite.gym.envs.osworld_2.main as m

    assert all(v for v in m._SERVICE_ENV.values())  # no None/"" values
    assert (
        "GITLAB_PRIVATE_TOKEN" not in m._SERVICE_ENV
    )  # gitlab null by default → dropped (V2's real var name)


def test_container_name_disjoint_from_v1_and_lite():
    from lite.gym.envs.osworld_2.container import OSWorldV2ContainerFactory

    f = OSWorldV2ContainerFactory(
        qcow2_path="/x.qcow2", task_class_dir="/tc", session_id="s", task_id="001"
    )
    name = f._make_name("abc123")
    assert "-osworld_2-" in name
    assert "-osworld-" not in name
    assert ".osworld-" not in name
