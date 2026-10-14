"""Unit tests for the Online-Mind2Web CUA-Lite wrapper."""

from __future__ import annotations

import base64
import io
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lite.core.tools import make_tool_call
from lite.gym.envs.online_mind2web.main import (
    RemoteOnlineMind2WebEnv,
    _format_schema_action,
    _pair_online_mind2web_action_errors,
    _to_container_action,
)
from lite.gym.errors import EnvDepsMissingError


def _make_png() -> bytes:
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(80, 110, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_PNG_B64 = base64.b64encode(_make_png()).decode("ascii")

_TASK = {
    "task_id": "abc123",
    "instruction": "Find a blue jacket under $100.",
    "confirmed_task": "Find a blue jacket under $100.",
    "start_url": "https://example.com/",
    "website": "https://example.com/",
    "reference_length": 4,
    "source": "online_mind2web",
    "hf_repo_id": "osunlp/Online-Mind2Web",
    "hf_revision": "test-revision",
}


def _reset_resp() -> dict:
    return {
        "instance_id": "iid-1",
        "screenshot_b64": _PNG_B64,
        "instruction": _TASK["instruction"],
        "url": "https://example.com/",
        "title": "Example",
        "a11y_tree": [],
        "body_text": "Example",
        "max_steps": 12,
    }


def _step_resp(
    *,
    terminated: bool = False,
    truncated: bool = False,
    answer: str | None = None,
) -> dict:
    return {
        "screenshots_b64": [_PNG_B64],
        "reward": None,
        "terminated": terminated,
        "truncated": truncated,
        "executed": [{"call": "click", "args": {"coordinate": [500, 250]}}],
        "errors": [],
        "action_errors": [],
        "answer": answer,
        "url": "https://example.com/search",
        "title": "Search",
        "a11y_tree": [],
        "body_text": "Search results",
    }


def _close_resp() -> dict:
    return {"ok": True, "closed": True}


def _make_env(
    tmp_path,
    *,
    skip_eval: bool = True,
    extra_tools: list[str] | None = None,
    valid_actions: list[str] | None = None,
    cursor: bool = True,
    include_page_context_text: bool = True,
) -> RemoteOnlineMind2WebEnv:
    os.environ.setdefault("ONLINE_MIND2WEB_RPC_URL", "http://localhost:7900")
    return RemoteOnlineMind2WebEnv(
        task_id="abc123",
        task=dict(_TASK),
        base_max_steps=12,
        max_steps=12,
        trajectory_dir=str(tmp_path),
        valid_actions=valid_actions,
        extra_tools=extra_tools or [],
        skip_eval=skip_eval,
        cursor=cursor,
        include_page_context_text=include_page_context_text,
    )


def _tool_call_with_raw_arguments(
    name: str,
    arguments,
    *,
    call_id: str,
) -> dict:
    call = make_tool_call(name, {}, call_id=call_id)
    call["function"]["arguments"] = arguments
    return call


def test_container_services_launches_with_freshness_gated_image(monkeypatch):
    import lite.gym.envs.online_mind2web.main as m

    seen = {}
    monkeypatch.setattr(m, "_services_started", set())
    monkeypatch.delenv("ONLINE_MIND2WEB_RPC_URL", raising=False)
    monkeypatch.setattr(m, "_healthz", lambda url: bool(url))
    monkeypatch.setattr(m, "docker_rm_f", lambda *args, **kwargs: None)
    monkeypatch.setattr("lite.gym.utils.backend.ports.allocate_ports", lambda **kwargs: [7901])

    def fake_docker_run(name, image, *, mem, port, env):
        seen["name"] = name
        seen["image"] = image
        seen["mem"] = mem
        seen["port"] = port
        seen["env"] = env

    monkeypatch.setattr(m, "docker_run", fake_docker_run)

    m.OnlineMind2WebContainerServices().ensure("online_mind2web")

    assert seen["image"].tag == m._ONLINE_MIND2WEB_IMAGE
    assert seen["image"].sources == ("lite/gym/envs/online_mind2web/docker",)
    assert seen["port"] == (7901, 8000)
    assert seen["env"]["ONLINE_MIND2WEB_INSTANCES"] == str(m._RESOLVED_INSTANCES)


def test_online_mind2web_catalog_declares_page_context_toggle():
    import lite.gym as gym

    supported = gym.registry.env_supported_kwargs("online_mind2web")

    assert "include_page_context_text" in supported


def test_online_mind2web_default_disables_page_context_text():
    import inspect

    import lite.gym.envs.online_mind2web.main as m

    default = inspect.signature(
        m.RemoteOnlineMind2WebEnv.__init__,
    ).parameters["include_page_context_text"].default

    assert m.CFG.env_kwargs["include_page_context_text"] is False
    assert m._INCLUDE_PAGE_CONTEXT_TEXT is False
    assert default is False


def test_container_shutdown_evicts_singleton_caches(monkeypatch):
    import importlib

    import lite.gym.envs.online_mind2web.main as m

    registry_mod = importlib.import_module("lite.gym.registry")
    env_id = "online_mind2web"
    monkeypatch.setattr(m, "_services_started", {env_id})
    monkeypatch.setenv("ONLINE_MIND2WEB_RPC_URL", "http://localhost:7902")
    registry_mod._services_started.add(env_id)
    monkeypatch.setattr(
        "lite.gym.remote.reaper.docker_rm_f",
        lambda *args, **kwargs: 1,
    )

    m.OnlineMind2WebContainerServices().shutdown(env_id, SimpleNamespace(server_port=1234))

    assert env_id not in m._services_started
    assert env_id not in registry_mod._services_started
    assert "ONLINE_MIND2WEB_RPC_URL" not in os.environ


def test_container_services_health_reports_setup_blocker(monkeypatch):
    import lite.gym.envs.online_mind2web.main as m

    monkeypatch.setenv("ONLINE_MIND2WEB_RPC_URL", "http://localhost:7900")
    monkeypatch.setattr(m, "_healthz", lambda _url: False)

    with pytest.raises(EnvDepsMissingError) as excinfo:
        m.OnlineMind2WebContainerServices().health("online_mind2web")

    message = str(excinfo.value)
    assert "/healthz unreachable" in message
    assert "lite/gym/envs/online_mind2web/scripts/install.sh" in message
    assert "lite/gym/envs/online_mind2web/README.md" in message


def test_format_schema_action_converts_normalized_coords_to_pixels():
    action, status = _format_schema_action(
        "click",
        {"coordinate": [500, 250]},
        status="SUCCESS",
        viewport=(1280, 768),
    )

    assert action == "CLICK coords(640, 192) -> click at the target coordinate | SUCCESS"
    assert status == "SUCCESS"


def test_format_schema_action_targets_page_for_malformed_coordinate():
    action, status = _format_schema_action(
        "click",
        {"coordinate": None},
        status="SUCCESS",
        viewport=(1280, 768),
    )

    assert action == "CLICK page -> click at the target coordinate | SUCCESS"
    assert status == "SUCCESS"


def test_format_schema_action_does_not_treat_terminate_status_as_answer():
    action, status = _format_schema_action(
        "terminate",
        {"status": "success"},
        status=None,
        viewport=(1280, 768),
    )

    assert action == "TASK_COMPLETE -> STATUS: success"
    assert status is None


def test_valid_actions_rejects_standalone_tools(tmp_path):
    """Standalone tools belong to ``extra_tools``, never the GUI enum.

    These names used to be dropped SILENTLY, which turned a wrong yaml into a
    quietly smaller action enum (and ``["terminate"]`` alone into ``[]``, i.e.
    no GUI tool at all). They now fail at the config boundary.
    """
    with pytest.raises(ValueError) as excinfo:
        _make_env(
            tmp_path,
            valid_actions=[
                "click",
                "response",
                "terminate",
                "goto",
                "back",
                "forward",
                "open_app",
                "ask_user",
                "scroll",
            ],
        )
    message = str(excinfo.value)
    for name in ("response", "terminate", "goto", "back", "forward", "open_app", "ask_user"):
        assert name in message
    assert "extra_tools" in message


def test_valid_actions_action_subset_is_verbatim(tmp_path):
    env = _make_env(tmp_path, valid_actions=["click", "scroll"])
    assert env.metadata.valid_actions == ["click", "scroll"]


@pytest.mark.asyncio
async def test_host_client_forwards_cursor_to_reset_and_step(tmp_path):
    env = _make_env(tmp_path, cursor=False)
    captured: list[tuple[str, dict]] = []

    def _route(path: str, body: dict) -> dict:
        captured.append((path, body))
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)

    await env.reset()
    await env.step([make_tool_call("click", {"coordinate": [500, 250]})])
    await env.close()

    by_path = {path: body for path, body in captured if path in {"/reset", "/step"}}
    assert by_path["/reset"]["cursor"] is False
    assert by_path["/step"]["cursor"] is False


@pytest.mark.asyncio
async def test_page_context_text_can_be_disabled_without_losing_screenshot(tmp_path):
    env = _make_env(tmp_path, include_page_context_text=False)

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)

    obs = await env.reset()
    r = await env.step([make_tool_call("click", {"coordinate": [500, 250]}, call_id="call-click")])

    assert obs.text == _TASK["instruction"]
    assert obs.metadata["web_text"] == ""
    result = r.results[0]
    assert result.tool_call_id == "call-click"
    assert result.images[-1] == base64.b64decode(_PNG_B64)
    assert result.text is None
    assert result.error is None
    assert result.metadata["url"] == "https://example.com/search"
    assert result.metadata["body_text"] == "Search results"

    await env.close()


@pytest.mark.asyncio
async def test_page_context_text_disabled_preserves_navigation_error(tmp_path):
    env = _make_env(tmp_path, include_page_context_text=False)

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            resp = _step_resp()
            resp["navigation_error"] = "net::ERR_NAME_NOT_RESOLVED"
            return resp
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)

    await env.reset()
    r = await env.step([make_tool_call("click", {"coordinate": [500, 250]}, call_id="call-click")])

    result = r.results[0]
    assert result.text == "Navigation error: net::ERR_NAME_NOT_RESOLVED"
    assert "Current page URL" not in result.text
    assert "Current page title" not in result.text
    assert "Search results" not in result.text
    assert result.error is None

    await env.close()


@pytest.mark.asyncio
async def test_page_context_text_disabled_keeps_current_action_error(tmp_path):
    env = _make_env(tmp_path, include_page_context_text=False)
    step_calls: list[dict] = []

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            step_calls.append(body)
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            _tool_call_with_raw_arguments("click", ["bad"], call_id="call-bad-click"),
        ]
    )

    assert step_calls == []
    result = r.results[0]
    assert result.images[-1] == base64.b64decode(_PNG_B64)
    assert result.text is None
    assert result.error == (
        "invalid tool call: tool_call.function.arguments must be an object, got list"
    )

    await env.close()


@pytest.mark.asyncio
async def test_trajectory_written_on_reset_and_step(tmp_path):
    env = _make_env(tmp_path)

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)

    await env.reset()
    r = await env.step([make_tool_call("click", {"coordinate": [500, 250]})])

    result_path = tmp_path / "abc123" / "result.json"
    data = json.loads(result_path.read_text())

    assert r.info["result_path"] == str(result_path)
    assert data["schema_version"] == "online-mind2web-v2"
    assert data["task_id"] == "abc123"
    assert data["reference_length"] == 4
    assert [s["step"] for s in data["action_history"]] == [0, 1]
    assert data["action_history"][0]["action"].startswith("page -> NAVIGATE")
    # [500, 250] in [0,1000] → px on the 1280×720 viewport: x=640, y=180.
    assert data["action_history"][1]["action"].startswith("CLICK coords(640, 180)")
    assert (tmp_path / "abc123" / "trajectory" / "0000.png").is_file()
    assert (tmp_path / "abc123" / "trajectory" / "0001.png").is_file()

    await env.close()


@pytest.mark.asyncio
async def test_terminal_step_runs_eval_and_records_final_answer(tmp_path):
    env = _make_env(tmp_path, extra_tools=["response"])

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            return _step_resp(terminated=True, answer="done")
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    async def _fake_eval():
        return 1.0, {"eval_verdict": "Status: success", "predicted_label": 1}

    env._post = MagicMock(side_effect=_route)
    env._evaluate = _fake_eval  # type: ignore[method-assign]

    await env.reset()
    r = await env.step([make_tool_call("response", {"text": "done"})])

    data = json.loads((tmp_path / "abc123" / "result.json").read_text())

    assert r.reward == 1.0
    assert r.info["predicted_label"] == 1
    assert data["agent_final_answer"] == "done"
    assert data["action_history"][-1]["action"] == "TASK_COMPLETE -> ANSWER: done"

    await env.close()


@pytest.mark.asyncio
async def test_final_answer_comes_from_the_container_not_a_host_reparse(tmp_path):
    """The container executes ``response`` and owns the answer; it is the only side
    that knows what actually ran, so its answer wins over anything the host could
    re-derive from the outgoing args."""
    env = _make_env(tmp_path, extra_tools=["response"])

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            return _step_resp(terminated=True, answer="container-answer")
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    async def _fake_eval():
        return 1.0, {"eval_verdict": "Status: success", "predicted_label": 1}

    env._post = MagicMock(side_effect=_route)
    env._evaluate = _fake_eval  # type: ignore[method-assign]

    await env.reset()
    r = await env.step([make_tool_call("response", {"text": "host-parse"})])

    data = json.loads((tmp_path / "abc123" / "result.json").read_text())

    assert r.info["answer"] == "container-answer"
    assert data["agent_final_answer"] == "container-answer"

    await env.close()


@pytest.mark.asyncio
async def test_response_with_call_id_returns_no_tool_result(tmp_path):
    env = _make_env(tmp_path, extra_tools=["response"])

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            return _step_resp(terminated=True, answer="done")
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    async def _fake_eval():
        return 1.0, {"eval_verdict": "Status: success", "predicted_label": 1}

    env._post = MagicMock(side_effect=_route)
    env._evaluate = _fake_eval  # type: ignore[method-assign]

    await env.reset()
    r = await env.step([make_tool_call("response", {"text": "done"}, call_id="call-response")])

    assert r.terminated is True
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    # What this test defends is unchanged: the container's terminal verdict and
    # the result.json it wrote.
    assert r.results == []
    data = json.loads((tmp_path / "abc123" / "result.json").read_text())
    assert data["agent_final_answer"] == "done"

    await env.close()


@pytest.mark.asyncio
async def test_terminate_records_status_without_final_answer(tmp_path):
    env = _make_env(tmp_path, extra_tools=["terminate"])

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            return _step_resp(terminated=True)
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    async def _fake_eval():
        return 0.0, {"eval_verdict": "Status: failure", "predicted_label": 0}

    env._post = MagicMock(side_effect=_route)
    env._evaluate = _fake_eval  # type: ignore[method-assign]

    await env.reset()
    r = await env.step([make_tool_call("terminate", {"status": "success"})])

    data = json.loads((tmp_path / "abc123" / "result.json").read_text())

    assert r.terminated is True
    assert data["agent_final_answer"] is None
    assert data["action_history"][-1]["action"] == "TASK_COMPLETE -> STATUS: success"

    await env.close()


@pytest.mark.asyncio
async def test_terminate_with_call_id_returns_no_tool_result(tmp_path):
    env = _make_env(tmp_path, extra_tools=["terminate"])

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            return _step_resp(terminated=True)
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    async def _fake_eval():
        return 0.0, {"eval_verdict": "Status: success", "predicted_label": 1}

    env._post = MagicMock(side_effect=_route)
    env._evaluate = _fake_eval  # type: ignore[method-assign]

    await env.reset()
    r = await env.step(
        [make_tool_call("terminate", {"status": "success"}, call_id="call-terminate")]
    )

    assert r.terminated is True
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    # What this test defends is unchanged: the container's terminal verdict and
    # the result.json it wrote.
    assert r.results == []
    data = json.loads((tmp_path / "abc123" / "result.json").read_text())
    assert data["agent_final_answer"] is None

    await env.close()


@pytest.mark.asyncio
async def test_container_step_malformed_action_envelopes_return_action_errors(
    monkeypatch,
    tmp_path,
):
    import lite.gym.envs.online_mind2web.docker.server as server

    async def _bounded_observation(_inst, *, cursor: bool = True):
        return {
            "screenshot_b64": _PNG_B64,
            "url": "https://example.com/",
            "title": "Example",
            "a11y_tree": [],
            "body_text": "Example",
        }

    monkeypatch.setattr(server, "_bounded_observation", _bounded_observation)
    iid = "iid-malformed-action-envelope"
    inst = server._Instance(
        context=SimpleNamespace(),
        page=SimpleNamespace(viewport_size={"width": 1280, "height": 720}),
        task_id="abc123",
        instruction=_TASK["instruction"],
        start_url="https://example.com/",
        download_dir=tmp_path,
        max_steps=5,
    )
    monkeypatch.setitem(server._instances, iid, inst)

    class _Request:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    resp = await server.step(
        _Request(
            {
                "instance_id": iid,
                "post_action_delay": 0,
                "step_count": 1,
                "actions": [
                    {"call_id": "missing-name", "arguments": {}},
                    {"name": "click", "call_id": "bad-args", "arguments": ["bad"]},
                    "bad-action",
                    {"name": "select", "call_id": "unsupported", "arguments": {}},
                    {"name": "key", "call_id": "scalar-key", "arguments": {"keys": "enter"}},
                ],
            }
        )
    )

    assert resp["executed"] == []
    assert resp["terminated"] is False
    assert resp["truncated"] is False
    assert [record["index"] for record in resp["action_errors"]] == [0, 1, 2, 3, 4]
    assert [record["kind"] for record in resp["action_errors"]] == [
        "model_action",
        "model_action",
        "model_action",
        "unsupported_action",
        "model_action",
    ]
    assert [record.get("call_id") for record in resp["action_errors"]] == [
        "missing-name",
        "bad-args",
        None,
        "unsupported",
        "scalar-key",
    ]
    assert [record["name"] for record in resp["action_errors"]] == [
        "<invalid>",
        "click",
        "<invalid>",
        "select",
        "key",
    ]
    assert resp["action_errors"][3]["message"] == "unsupported action: select"
    assert "key.keys must be a list of strings, not a string" in (resp["action_errors"][4]["error"])
    assert "Lite tool_call.arguments must be a dict, got list" in resp["errors"][1]
    assert "Lite action must be an object, got str" in resp["errors"][2]
    assert inst.step_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [-1, "NaN", "Infinity", 31])
async def test_container_step_rejects_bad_wait_duration_without_sleep(
    monkeypatch,
    tmp_path,
    duration,
):
    import lite.gym.envs.online_mind2web.docker.server as server

    async def _bounded_observation(_inst, *, cursor: bool = True):
        return {
            "screenshot_b64": _PNG_B64,
            "url": "https://example.com/",
            "title": "Example",
            "a11y_tree": [],
            "body_text": "Example",
        }

    async def _sleep(_seconds):
        raise AssertionError("bad wait duration reached asyncio.sleep")

    monkeypatch.setattr(server, "_bounded_observation", _bounded_observation)
    monkeypatch.setattr(server.asyncio, "sleep", _sleep)
    iid = "iid-bad-wait-duration"
    inst = server._Instance(
        context=SimpleNamespace(),
        page=SimpleNamespace(viewport_size={"width": 1280, "height": 720}),
        task_id="abc123",
        instruction=_TASK["instruction"],
        start_url="https://example.com/",
        download_dir=tmp_path,
        max_steps=5,
    )
    monkeypatch.setitem(server._instances, iid, inst)

    class _Request:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    resp = await server.step(
        _Request(
            {
                "instance_id": iid,
                "post_action_delay": 0,
                "step_count": 1,
                "actions": [
                    {
                        "name": "wait",
                        "call_id": "bad-wait",
                        "arguments": {"duration": duration},
                    },
                ],
            }
        )
    )

    assert resp["executed"] == []
    assert len(resp["action_errors"]) == 1
    error = resp["action_errors"][0]
    assert error["index"] == 0
    assert error["name"] == "wait"
    assert error["kind"] == "model_action"
    assert "wait.duration" in error["error"]
    assert inst.step_count == 1


def test_container_step_non_list_body_actions_returns_model_action_error(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from lite.gym.envs.online_mind2web.docker import server

    async def _bounded_observation(_inst, *, cursor: bool = True):
        return {
            "screenshot_b64": _PNG_B64,
            "url": "https://example.com/",
            "title": "Example",
            "a11y_tree": [],
            "body_text": "Example",
        }

    monkeypatch.setattr(server, "_bounded_observation", _bounded_observation)
    iid = "iid-non-list-actions"
    inst = server._Instance(
        context=SimpleNamespace(),
        page=SimpleNamespace(viewport_size={"width": 1280, "height": 720}),
        task_id="abc123",
        instruction=_TASK["instruction"],
        start_url="https://example.com/",
        download_dir=tmp_path,
        max_steps=5,
    )
    monkeypatch.setitem(server._instances, iid, inst)

    client = TestClient(server.app)
    r = client.post(
        "/step",
        json={
            "instance_id": iid,
            "actions": "click",
            "step_count": 1,
            "post_action_delay": 0,
        },
    )
    assert r.status_code == 200, r.text
    resp = r.json()
    assert resp["errors"] == ["<invalid>: body.actions must be a list, got str"]
    assert resp["action_errors"] == [
        {
            "index": 0,
            "name": "<invalid>",
            "error": "body.actions must be a list, got str",
            "message": "<invalid>: body.actions must be a list, got str",
            "kind": "model_action",
        }
    ]
    assert inst.step_count == 1

    r = client.post("/step", json={"instance_id": "i1", "actions": [], "step_count": 1})
    assert r.status_code == 404, r.text


def test_container_duration_caps_match_host_table_by_value():
    """The in-container copy must agree with the host table on VALUES.

    ``test_container_step_rejects_bad_wait_duration_without_sleep`` above only
    proves ``duration=31`` is rejected -- it stays green if the ``wait`` cap
    drops to 3.0 -- and the family matrix asserts only that the identifier
    ``_MODEL_DURATION_CAPS_SECONDS`` appears in the source text. The container
    script may not import ``lite.gym``, so the copy is forced; the host-side
    test can still read both and compare.
    """
    import lite.gym.envs.online_mind2web.docker.server as server
    from lite.core.tools.action_space.duration import (
        ACTION_SCHEMA_DURATION_CAPS_SECONDS,
    )
    from lite.gym.utils.backend.model_inputs import DEFAULT_MODEL_DURATION_CAP_SECONDS

    assert server._MODEL_DURATION_CAPS_SECONDS == dict(ACTION_SCHEMA_DURATION_CAPS_SECONDS)
    assert server._DEFAULT_MODEL_DURATION_CAP_SECONDS == DEFAULT_MODEL_DURATION_CAP_SECONDS


def test_container_norm_coord_rejects_nonfinite_values():
    import lite.gym.envs.online_mind2web.docker.server as server

    with pytest.raises(ValueError, match="finite numbers"):
        server._norm_coord(["Infinity", 500], 1280, 720)
    with pytest.raises(ValueError, match="finite numbers"):
        server._norm_coord([500, "NaN"], 1280, 720)


@pytest.mark.asyncio
async def test_truncated_executed_action_with_call_id_returns_paired_current_result(tmp_path):
    env = _make_env(tmp_path)

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            return _step_resp(truncated=True)
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    async def _fake_eval():
        return 0.5, {"eval_verdict": "Status: partial", "predicted_label": 0}

    env._post = MagicMock(side_effect=_route)
    env._evaluate = _fake_eval  # type: ignore[method-assign]

    await env.reset()
    r = await env.step([make_tool_call("click", {"coordinate": [500, 250]}, call_id="call-click")])

    assert r.truncated is True
    assert len(r.results) == 1
    result = r.results[0]
    assert result.tool_call_id == "call-click"
    assert result.error is None
    assert result.images[-1] == base64.b64decode(_PNG_B64)
    assert "Search results" in result.text
    assert r.info["predicted_label"] == 0

    await env.close()


# ---------------------------------------------------------------------------
# Bad model action → PAIRED per-call feedback
#
# The container returns typed ``action_errors[]`` records with action indexes;
# aggregate ``errors[]`` is debug-only and must not drive host-side pairing.
# ---------------------------------------------------------------------------


def test_action_error_pairing_uses_record_name_not_action_fallback():
    paired, failed = _pair_online_mind2web_action_errors(
        [{"arguments": {"coordinate": [500, 250]}}],
        ["call-click"],
        action_errors=[
            {
                "index": 0,
                "kind": "model_action",
                "name": "click",
                "error": "bad coordinate",
                "message": "click: bad coordinate",
            }
        ],
    )

    assert failed == {0}
    assert paired["call-click"].carrier == "current"
    assert paired["call-click"].message == ("invalid arguments for click: bad coordinate")


@pytest.mark.asyncio
async def test_action_error_pairs_to_originating_call_id(tmp_path):
    env = _make_env(tmp_path)

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            resp = _step_resp()
            resp["errors"] = ["type: no editable element at coordinate (0, 0)"]
            resp["action_errors"] = [
                {
                    "index": 1,
                    "kind": "model_action",
                    "name": "type",
                    "call_id": "call-b",
                    "error": "no editable element at coordinate (0, 0)",
                    "message": "type: no editable element at coordinate (0, 0)",
                }
            ]
            return resp
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [500, 250]}]},
                call_id="call-a",
            ),
            make_tool_call(
                "computer",
                {"actions": [{"action": "type", "coordinate": [0, 0], "text": "hi"}]},
                call_id="call-b",
            ),
        ]
    )

    by_id = {res.tool_call_id: res for res in r.results}
    assert not (by_id["call-a"].metadata or {}).get("is_error")
    assert by_id["call-b"].metadata["is_error"] is True
    assert "no editable element" not in (by_id["call-b"].text or "")
    assert "no editable element" in by_id["call-b"].error
    # info keeps the raw aggregate list (trajectory/debug consumers).
    assert r.info["errors"] == ["type: no editable element at coordinate (0, 0)"]
    # ... and the same pairing drives the v2 trajectory status.
    data = json.loads((tmp_path / "abc123" / "result.json").read_text())
    statuses = [s["action_status"] for s in data["action_history"][1:]]
    assert statuses == ["SUCCESS", "FAILED"]

    await env.close()


@pytest.mark.asyncio
async def test_scalar_key_returns_current_feedback_without_rpc(tmp_path):
    env = _make_env(tmp_path)

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "key", "keys": "enter"}]},
                call_id="call-key",
            ),
        ]
    )
    await env.close()

    paths = [call.args[0] for call in env._post.call_args_list]
    assert "/step" not in paths
    result = r.results[0]
    assert result.tool_call_id == "call-key"
    assert result.metadata["is_error"] is True
    assert result.error == (
        "invalid arguments for key: key.keys must be a list of strings, not a string"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_text", "expected_flag"),
    # om2w always spells the flag out; webvoyager omits it when false. Both
    # reach the same container default, so each test states its own env's wire.
    [("search me\n", "search me", True), ("search me", "search me", False)],
    ids=["trailing-newline-submits", "no-newline-types-only"],
)
async def test_type_projects_a_trailing_newline_onto_the_container_flag(
    tmp_path, text, expected_text, expected_flag
):
    """The trailing newline must reach the container as its ``press_enter``.

    Driven through ``step()`` on purpose: the projection lives in the ``/step``
    payload literal, and the unit tests for ``_to_container_action`` pass whether
    or not it is still called there. Without this, deleting the call silently
    removes every Enter press.
    """
    env = _make_env(tmp_path)
    captured: list[dict] = []

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            captured.append(body)
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()
    await env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "type", "text": text}]},
                call_id="call-type",
            ),
        ]
    )
    await env.close()

    action = captured[0]["actions"][0]
    assert action["name"] == "type"
    assert action["arguments"]["text"] == expected_text
    assert action["arguments"].get("press_enter") == expected_flag


@pytest.mark.asyncio
async def test_key_glyphs_project_to_playwright_wire(tmp_path):
    env = _make_env(tmp_path)
    captured: list[dict] = []

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            captured.append(body)
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    await env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "key", "keys": ["+", "-", "="]}]},
                call_id="call-key",
            ),
        ]
    )
    await env.close()

    assert captured[0]["actions"][0]["name"] == "key"
    assert captured[0]["actions"][0]["arguments"]["keys"] == ["+", "-", "="]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("keys", "expected_error"),
    [
        (["plus"], "invalid arguments for key: unknown key token 'plus'"),
        ([" "], "invalid arguments for key: unknown key token ' '"),
    ],
)
async def test_noncanonical_key_tokens_return_current_feedback_without_rpc(
    tmp_path,
    keys,
    expected_error,
):
    env = _make_env(tmp_path)

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            raise AssertionError("noncanonical key token must not reach /step")
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "key", "keys": keys}]},
                call_id="call-key",
            ),
        ]
    )
    await env.close()

    result = r.results[0]
    assert result.tool_call_id == "call-key"
    assert result.images[-1] == base64.b64decode(_PNG_B64)
    assert result.metadata["is_error"] is True
    assert result.error == expected_error


@pytest.mark.asyncio
async def test_aggregate_errors_without_typed_records_are_debug_only(tmp_path):
    env = _make_env(tmp_path)

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            resp = _step_resp()
            resp["errors"] = ["click: target detached"]
            resp["action_errors"] = []
            return resp
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 250]}, call_id="call-click"),
        ]
    )

    assert r.results[0].tool_call_id == "call-click"
    assert r.results[0].error is None
    assert r.info["errors"] == ["click: target detached"]

    data = json.loads((tmp_path / "abc123" / "result.json").read_text())
    assert data["action_history"][-1]["action_status"] == "SUCCESS"

    await env.close()


@pytest.mark.asyncio
async def test_same_name_backend_error_pairs_by_action_index(tmp_path):
    env = _make_env(tmp_path)

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            assert [a["name"] for a in body["actions"]] == ["click", "click"]
            assert [a["call_id"] for a in body["actions"]] == [
                "call-click-1",
                "call-click-2",
            ]
            resp = _step_resp()
            resp["executed"] = [
                {"call": "click", "args": {"coordinate": [100, 100]}},
                {
                    "call": "click",
                    "args": {"coordinate": [900, 900]},
                    "error": "target detached",
                },
            ]
            resp["errors"] = ["click: target detached"]
            resp["action_errors"] = [
                {
                    "index": 1,
                    "kind": "tool_execution",
                    "name": "click",
                    "call_id": "call-click-2",
                    "error": "target detached",
                    "message": "click: target detached",
                }
            ]
            return resp
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            make_tool_call("click", {"coordinate": [100, 100]}, call_id="call-click-1"),
            make_tool_call("click", {"coordinate": [900, 900]}, call_id="call-click-2"),
        ]
    )

    by_id = {res.tool_call_id: res for res in r.results}
    assert not (by_id["call-click-1"].metadata or {}).get("is_error")
    assert by_id["call-click-1"].error is None
    assert by_id["call-click-2"].metadata["is_error"] is True
    assert by_id["call-click-2"].error == "click failed: execution failed"

    data = json.loads((tmp_path / "abc123" / "result.json").read_text())
    statuses = [s["action_status"] for s in data["action_history"][1:]]
    assert statuses == ["SUCCESS", "FAILED"]

    await env.close()


@pytest.mark.asyncio
async def test_invalid_valid_action_error_hides_valid_actions_list_from_model_visible_text(
    tmp_path,
):
    env = _make_env(tmp_path, valid_actions=["scroll"])

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            assert body["actions"] == []
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [500, 250]}]},
                call_id="call-click",
            )
        ]
    )

    assert len(r.results) == 1
    result = r.results[0]
    assert result.tool_call_id == "call-click"
    assert result.metadata["is_error"] is True
    assert result.error == ("invalid action: click; choose an available action for this task")
    assert "valid_actions" not in result.error
    assert "['scroll']" not in result.error

    await env.close()


@pytest.mark.asyncio
async def test_backend_action_error_hides_playwright_detail_from_model_visible_text(tmp_path):
    env = _make_env(tmp_path)

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            assert [a["name"] for a in body["actions"]] == ["click"]
            resp = _step_resp()
            resp["errors"] = ["click: Playwright TimeoutError: target closed"]
            resp["action_errors"] = [
                {
                    "index": 0,
                    "kind": "tool_execution",
                    "name": "click",
                    "call_id": "call-click",
                    "error": "Playwright TimeoutError: target closed",
                    "message": "click: Playwright TimeoutError: target closed",
                }
            ]
            return resp
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 250]}, call_id="call-click"),
        ]
    )

    assert len(r.results) == 1
    result = r.results[0]
    assert result.tool_call_id == "call-click"
    assert result.metadata["is_error"] is True
    assert result.error == "click failed: execution failed"
    assert "playwright" not in result.error.lower()
    assert "target closed" not in result.error.lower()
    assert r.info["errors"] == ["click: Playwright TimeoutError: target closed"]

    await env.close()


@pytest.mark.asyncio
async def test_unknown_tool_error_is_error_only(tmp_path):
    env = _make_env(tmp_path)

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            assert body["actions"] == []
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            make_tool_call("foo", {}, call_id="call-foo"),
        ]
    )

    assert len(r.results) == 1
    result = r.results[0]
    assert result.tool_call_id == "call-foo"
    assert result.error == "unknown tool: foo"
    assert result.metadata == {"is_error": True}
    assert result.images == []
    assert result.text is None

    await env.close()


@pytest.mark.asyncio
async def test_malformed_tool_call_with_call_id_returns_current_feedback(tmp_path):
    env = _make_env(tmp_path)

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            assert body["actions"] == []
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            _tool_call_with_raw_arguments("click", ["bad"], call_id="call-bad-click"),
        ]
    )

    assert len(r.results) == 1
    result = r.results[0]
    assert result.tool_call_id == "call-bad-click"
    assert result.error == (
        "invalid tool call: tool_call.function.arguments must be an object, got list"
    )
    assert result.metadata["is_error"] is True
    assert result.images[-1] == base64.b64decode(_PNG_B64)
    assert "Current page URL: https://example.com/" in result.text
    data = json.loads((tmp_path / "abc123" / "result.json").read_text())
    assert [s["step"] for s in data["action_history"]] == [0]

    await env.close()


@pytest.mark.asyncio
async def test_inactive_known_tool_error_is_error_only(tmp_path):
    env = _make_env(tmp_path)

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            assert body["actions"] == []
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            make_tool_call("goto", {"url": "https://example.org"}, call_id="call-goto"),
        ]
    )

    assert len(r.results) == 1
    result = r.results[0]
    assert result.tool_call_id == "call-goto"
    assert result.error == "goto is not available in this task."
    assert result.metadata == {"is_error": True}
    assert result.images == []
    assert result.text is None

    await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [-1, "NaN", "Infinity", 31])
async def test_bad_wait_duration_returns_current_feedback_without_rpc(tmp_path, duration):
    env = _make_env(tmp_path)
    step_calls: list[dict] = []

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            step_calls.append(body)
            return _step_resp()
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            make_tool_call("wait", {"duration": duration}, call_id="call-wait"),
        ]
    )

    assert step_calls == []
    result = r.results[0]
    assert result.tool_call_id == "call-wait"
    assert result.metadata["is_error"] is True
    assert result.images[-1] == base64.b64decode(_PNG_B64)
    assert "Current page URL: https://example.com/" in result.text
    assert result.error.startswith("invalid arguments for wait: wait.duration")

    await env.close()


@pytest.mark.asyncio
async def test_container_infra_failure_still_raises(tmp_path):
    """A dead container / RPC failure must keep hard-raising out of step()."""
    env = _make_env(tmp_path)

    def _route(path: str, _body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            raise RuntimeError("online_mind2web container /step returned 500: boom")
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    with pytest.raises(RuntimeError, match="returned 500"):
        await env.step([make_tool_call("click", {"coordinate": [500, 250]})])

    await env.close()


# ---------------------------------------------------------------------------
# Judge infra-error propagation (B2 / T2)
#
# The terminal-step tests above monkeypatch env._evaluate, so the REAL judge
# branch is never exercised. These hit the live judge code path with a forced
# infra failure and assert it RAISES (so the rollout ERRs + retries) rather than
# coalescing to a corrupt reward=0.0.
# ---------------------------------------------------------------------------
_BAD_JUDGE_CFG = {
    "mode": "WebJudge_Online_Mind2Web_eval",
    "base_url": "http://127.0.0.1:1/v1",  # unroutable -> connection error
    "api_key_env_var": "ONLINE_MIND2WEB_TEST_NO_KEY",
    "model": "gpt-4o-mini",
    "request_timeout": 1,
}


def test_judge_generate_propagates_infra_error():
    from lite.gym.envs.online_mind2web.main import _OpenAIJudgeEngine

    engine = _OpenAIJudgeEngine(_BAD_JUDGE_CFG)
    with pytest.raises(Exception):  # APIConnectionError — NOT swallowed
        engine.generate([{"role": "user", "content": "hi"}])


def test_webjudge_propagates_infra_error_not_zero_reward():
    from lite.gym.envs.online_mind2web.main import _webjudge_online_mind2web_eval

    # A judge infra failure must propagate, never return (0.0, ...): coalescing
    # it would silently poison RL data (B2).
    with pytest.raises(Exception):
        _webjudge_online_mind2web_eval(
            task="find a blue jacket",
            last_actions=["click"],
            image_paths=[],
            cfg=_BAD_JUDGE_CFG,
        )


def test_is_judge_rate_limit_error_classification():
    from lite.gym.envs.online_mind2web.main import _is_judge_rate_limit_error

    class _RateLimitError(Exception):
        pass

    assert _is_judge_rate_limit_error(_RateLimitError("boom"))  # by type name
    assert _is_judge_rate_limit_error(Exception("HTTP 429 Too Many"))  # by 429
    assert _is_judge_rate_limit_error(Exception("rate limit exceeded"))  # by 'rate'
    assert not _is_judge_rate_limit_error(Exception("connection refused"))


@pytest.mark.asyncio
async def test_container_reset_seeds_cursor_at_viewport_centre(monkeypatch, tmp_path):
    """A fresh instance starts with the pointer at the viewport CENTRE.

    Regression: ``last_cursor`` used to stay ``None`` until the first pointer
    action, which made ``_show_capture_cursor`` return ``False`` and shipped a
    turn-0 frame with NO cursor at all — structurally unlike every later frame,
    and unlike the retired ``CursorOverlayWrapper`` which seeded ``(500, 500)``
    normalized (= centre) on ``reset()``.

    Drives the real ``/reset`` handler with the browser stubbed out, so the
    assertion covers the shipped construction site, not a re-derived constant.
    """
    import lite.gym.envs.online_mind2web.docker.server as server

    class _Page:
        def set_default_timeout(self, _ms):
            pass

        def set_default_navigation_timeout(self, _ms):
            pass

    class _Context:
        async def new_page(self):
            return _Page()

        async def close(self):
            pass

    async def _new_context(_w, _h, _dir):
        return _Context()

    async def _safe_goto(_page, _url):
        return None

    async def _auto_accept_cookies(_page):
        return None

    async def _bounded_observation(_inst, *, cursor: bool = True):
        return {
            "screenshot_b64": _PNG_B64,
            "url": "",
            "title": "",
            "a11y_tree": [],
            "body_text": "",
        }

    monkeypatch.setattr(server, "_new_context", _new_context)
    monkeypatch.setattr(server, "_safe_goto", _safe_goto)
    monkeypatch.setattr(server, "_auto_accept_cookies", _auto_accept_cookies)
    monkeypatch.setattr(server, "_bounded_observation", _bounded_observation)
    monkeypatch.setattr(server, "_RESET_SETTLE_S", 0.0)
    monkeypatch.setattr(server, "_BASE_DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(server, "_instances", {})

    class _Request:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    out = await server.reset(_Request({"viewport": [1280, 720]}))
    inst = server._instances[out["instance_id"]]

    assert inst.last_cursor == (640, 360)

    wide = await server.reset(_Request({"viewport": [1920, 1080]}))

    assert server._instances[wide["instance_id"]].last_cursor == (960, 540)


@pytest.mark.asyncio
async def test_host_step_budget_truncates_after_a_fully_filtered_step(tmp_path):
    """A filtered step consumes budget on the host; the container never sees it.

    ``_evaluate`` is HOST-side here (the container always returns
    ``reward: None``), so the fully-filtered fast path legitimately scores
    without a round-trip -- so a container-side counter can only fall behind. The
    host owns the count and SHIPS it in the ``/step`` body; the container assigns
    it and derives ``truncated`` from it. Same shape as
    ``webharbor/webvoyager``; deliberately NOT ``mobilegym``, which deleted its
    host counter because ITS evaluator lives in the container.
    """
    env = RemoteOnlineMind2WebEnv(
        task_id="abc123",
        task=dict(_TASK),
        base_max_steps=2,
        max_steps=2,
        trajectory_dir=str(tmp_path),
        skip_eval=True,
    )

    step_bodies: list[dict] = []

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            step_bodies.append(body)
            # Mirrors docker/server.py: the container adopts the shipped count
            # instead of keeping one of its own, so the turn the host filtered
            # out is already accounted for here.
            return _step_resp(truncated=body["step_count"] >= 2)
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()

    # Turn 1: every call rejected at ingress -> no RPC, budget still consumed.
    filtered = await env.step(
        [
            _tool_call_with_raw_arguments("click", ["bad"], call_id="bad-args"),
        ]
    )
    assert step_bodies == []
    assert filtered.truncated is False

    # Turn 2 exhausts max_steps=2. The container only ever saw this one POST,
    # so it truncates on time only because the host shipped its own count.
    last = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 250]}, call_id="ok"),
        ]
    )

    assert len(step_bodies) == 1
    assert step_bodies[0]["step_count"] == 2
    assert last.truncated is True


# ---------------------------------------------------------------------------
# Per-action result frames
#
# online_mind2web POSTs the WHOLE batch to the container in one ``/step``, so
# the action loop -- and therefore the only place that can see the page BETWEEN
# two actions of one batch -- lives in ``docker/server.py``. The container
# returns ``screenshots_b64`` (one frame per executed action, in action order)
# and the host turns it into the per-action ``images`` of the tool result.
# ---------------------------------------------------------------------------


def _distinct_frames(prefix: str, count: int) -> list[str]:
    return [
        base64.b64encode(f"{prefix}-{index}".encode()).decode("ascii") for index in range(count)
    ]


def _batch_call(call_id: str, count: int) -> dict:
    return make_tool_call(
        "computer",
        {
            "actions": [
                {"action": "click", "coordinate": [100 * (index + 1)] * 2} for index in range(count)
            ]
        },
        call_id=call_id,
    )


@pytest.mark.asyncio
async def test_action_batch_result_carries_one_image_per_executed_action(tmp_path):
    """N executed actions -> N DISTINCT frames on the batch's single result.

    Distinctness is the assertion that matters: re-emitting one frame N times
    would satisfy a length check while carrying no information at all.
    """
    frames = _distinct_frames("om2w-frame", 3)
    env = _make_env(tmp_path)

    def _route(path: str, body: dict) -> dict:
        if path == "/reset":
            return _reset_resp()
        if path == "/step":
            assert [action["name"] for action in body["actions"]] == ["click"] * 3
            return {
                **_step_resp(),
                "screenshots_b64": frames,
                "screenshot_b64": frames[-1],
            }
        if path == "/close":
            return _close_resp()
        raise AssertionError(path)

    env._post = MagicMock(side_effect=_route)
    await env.reset()
    r = await env.step([_batch_call("call-batch", 3)])

    assert len(r.results) == 1
    images = list(r.results[0].images)
    assert images == [base64.b64decode(frame) for frame in frames]
    assert len(set(images)) == 3

    await env.close()


def test_container_step_returns_one_frame_per_executed_action(monkeypatch, tmp_path):
    """The container-side half: the frame count follows the ACTION count.

    The post-loop observation SUPERSEDES the last per-action frame rather than
    extending the list -- it is the same action's state, re-taken after the
    settle delay and carrying the turn's a11y tree / url. A batch that executed
    nothing still owes exactly one observation.
    """
    from fastapi.testclient import TestClient

    from lite.gym.envs.online_mind2web.docker import server

    frames = iter([b"om-1", b"om-2", b"om-3"])

    async def _capture(_inst, *, cursor: bool = True) -> bytes:
        return next(frames)

    async def _observation(_inst, *, cursor: bool = True) -> dict:
        return {
            "screenshot_b64": base64.b64encode(b"om-observation").decode("ascii"),
            "url": "https://example.com/",
            "title": "Example",
            "a11y_tree": [],
            "body_text": "",
        }

    monkeypatch.setattr(server, "_bounded_screenshot_png", _capture)
    monkeypatch.setattr(server, "_bounded_observation", _observation)
    iid = "iid-per-action-frames"
    inst = server._Instance(
        context=SimpleNamespace(),
        page=SimpleNamespace(viewport_size={"width": 1280, "height": 720}),
        task_id="abc123",
        instruction=_TASK["instruction"],
        start_url="https://example.com/",
        download_dir=tmp_path,
        max_steps=5,
    )
    monkeypatch.setitem(server._instances, iid, inst)

    client = TestClient(server.app)
    r = client.post(
        "/step",
        json={
            "instance_id": iid,
            "step_count": 1,
            "post_action_delay": 0,
            "actions": [
                {"name": "screenshot", "call_id": "c1", "arguments": {}},
                {"name": "screenshot", "call_id": "c2", "arguments": {}},
                {"name": "screenshot", "call_id": "c3", "arguments": {}},
            ],
        },
    )
    assert r.status_code == 200, r.text
    resp = r.json()
    assert [base64.b64decode(f) for f in resp["screenshots_b64"]] == [
        b"om-1",
        b"om-2",
        b"om-observation",
    ]

    r = client.post(
        "/step",
        json={
            "instance_id": iid,
            "step_count": 2,
            "post_action_delay": 0,
            "actions": [],
        },
    )
    assert r.status_code == 200, r.text
    assert [base64.b64decode(f) for f in r.json()["screenshots_b64"]] == [
        b"om-observation",
    ]


def test_a_terminating_batch_keeps_the_previous_actions_frame(monkeypatch, tmp_path):
    """``terminate`` breaks the loop owing a frame; the observation EXTENDS.

    ``terminate`` executes but never drives the page, so unlike a normal action
    it appends no frame of its own. If the post-loop observation superseded here
    it would overwrite the PRECEDING action's frame, and a two-action batch would
    report one frame -- losing a frame and miscounting executed actions at once.
    """
    from fastapi.testclient import TestClient

    from lite.gym.envs.online_mind2web.docker import server

    frames = iter([b"om-1", b"om-2"])

    async def _capture(_inst, *, cursor: bool = True) -> bytes:
        return next(frames)

    async def _observation(_inst, *, cursor: bool = True) -> dict:
        return {
            "screenshot_b64": base64.b64encode(b"om-observation").decode("ascii"),
            "url": "https://example.com/",
            "title": "Example",
            "a11y_tree": [],
            "body_text": "",
        }

    monkeypatch.setattr(server, "_bounded_screenshot_png", _capture)
    monkeypatch.setattr(server, "_bounded_observation", _observation)
    iid = "iid-terminating-batch"
    inst = server._Instance(
        context=SimpleNamespace(),
        page=SimpleNamespace(viewport_size={"width": 1280, "height": 720}),
        task_id="abc123",
        instruction=_TASK["instruction"],
        start_url="https://example.com/",
        download_dir=tmp_path,
        max_steps=5,
    )
    monkeypatch.setitem(server._instances, iid, inst)

    client = TestClient(server.app)
    r = client.post(
        "/step",
        json={
            "instance_id": iid,
            "step_count": 1,
            "post_action_delay": 0,
            "actions": [
                {"name": "screenshot", "call_id": "c1", "arguments": {}},
                {"name": "terminate", "call_id": "c2", "arguments": {}},
            ],
        },
    )

    assert r.status_code == 200, r.text
    resp = r.json()
    assert [c["call"] for c in resp["executed"]] == ["screenshot", "terminate"]
    assert [base64.b64decode(f) for f in resp["screenshots_b64"]] == [
        b"om-1",
        b"om-observation",
    ]
    assert resp["terminated"] is True


@pytest.mark.asyncio
async def test_container_type_without_press_enter_types_only():
    """Absent ``press_enter`` means TYPE ONLY, on every env that reads it.

    The container flag is host-derived: canonical spells Enter as a trailing
    newline, and the env strips it onto ``press_enter``. Absent means the model
    did not ask to submit, and the error is asymmetric: a missing Enter costs
    one turn, a spurious Enter
    irreversibly submits a form or navigates away.
    """
    from lite.gym.envs.online_mind2web.docker import server

    keystrokes: list[str] = []

    class _Keyboard:
        async def type(self, text, delay=0):
            keystrokes.append(f"type:{text}")

        async def press(self, key):
            keystrokes.append(f"press:{key}")

    class _Page:
        viewport_size = {"width": 1280, "height": 720}
        keyboard = _Keyboard()

        async def evaluate(self, *_args, **_kwargs):
            return "body"

    inst = SimpleNamespace(page=_Page(), last_cursor=None)

    await server._execute_action(inst, "type", {"text": "query"})
    assert keystrokes == ["type:query"]

    await server._execute_action(inst, "type", {"text": "query", "press_enter": True})
    assert keystrokes == ["type:query", "type:query", "press:Enter"]


@pytest.mark.asyncio
async def test_container_goto_requires_canonical_url_argument():
    from lite.gym.envs.online_mind2web.docker import server

    class _Page:
        viewport_size = {"width": 1280, "height": 720}

    inst = SimpleNamespace(page=_Page(), last_cursor=None)

    for args in ({}, {"web": "https://example.com/"}, {"page": "https://example.com/"}):
        with pytest.raises(ValueError, match="goto requires url"):
            await server._execute_action(inst, "goto", args)


# ---------------------------------------------------------------------------
# Canonical -> container action wire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,value,submits",
    [
        ("query\n", "query", True),
        ("query", "query", False),
        ("line one\nline two", "line one\nline two", False),
        ("", "", False),
    ],
    ids=["trailing", "plain", "interior", "empty"],
)
def test_to_container_action_projects_a_trailing_newline(text, value, submits):
    """Canonical spells Enter as a trailing newline, but the container reaches
    for ``locator.fill()`` first, which STORES a newline instead of pressing
    Return. So the host strips it onto the flag the container already reads.
    An interior newline is part of the value and stays put."""
    out = _to_container_action({"name": "type", "arguments": {"text": text}})
    assert out["arguments"]["text"] == value
    assert out["arguments"]["press_enter"] is submits


def test_to_container_action_leaves_every_other_action_alone():
    """Only ``type`` carries an Enter spelling, so nothing else is rewritten --
    and the canonical list itself must survive untouched, because the judged v2
    trajectory and the error pairing both read it."""
    click = {"name": "click", "arguments": {"coordinate": [1, 2]}}
    assert _to_container_action(click) is click

    typed = {"name": "type", "arguments": {"text": "q\n"}}
    _to_container_action(typed)
    assert typed["arguments"] == {"text": "q\n"}


def test_action_description_still_reports_the_enter_after_projection():
    """The v2 trajectory renders from the CANONICAL action, so it must see the
    trailing newline. Projecting in place would strip it first and silently drop
    'and press Enter' from what the judge scores."""
    submitting, _ = _format_schema_action(
        "type", {"text": "shoes\n"}, status="SUCCESS", viewport=(1280, 720), description=None,
    )
    plain, _ = _format_schema_action(
        "type", {"text": "shoes"}, status="SUCCESS", viewport=(1280, 720), description=None,
    )
    assert "and press Enter" in submitting
    assert "shoes\\n" not in submitting
    assert "and press Enter" not in plain
