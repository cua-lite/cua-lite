"""Tests for the WebVoyager CUA-Lite gym environment (thin JSON RPC client).

Requires:
  - uv sync (no extra deps beyond the main venv)
  - For live tests (`-m live`): cua-lite/webharbor.webvoyager image built (install.sh) +
    a running docker daemon — WebVoyagerContainerServices.ensure() brings the
    container up automatically on gym.make().

Run:
    uv run python -m pytest \
        tests/gym/envs/webharbor/webvoyager/test_webharbor_webvoyager.py
    uv run python -m pytest \
        tests/gym/envs/webharbor/webvoyager/test_webharbor_webvoyager.py -m live
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lite.agents.core.agent.utils.annotations import coordinate_annotation_records
from lite.core.tools import make_tool_call
from lite.core.tools.schemas import tool_schema_name
from lite.gym.envs.webharbor.webvoyager.main import (
    _KNOWN_STANDALONE_TOOL_NAMES,
    RemoteWebVoyagerEnv,
    _pair_webvoyager_action_errors,
    _to_container_action,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(color: tuple[int, int, int] = (180, 90, 40)) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _call(name: str, arguments: dict | None = None, *, call_id: str | None = None) -> dict:
    return make_tool_call(name, arguments or {}, call_id=call_id)


def _raw_call(name: str, arguments, *, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _container_action_error_record(
    index: int,
    name: str,
    error: str,
    *,
    kind: str = "model_action",
    message: str | None = None,
) -> dict:
    return {
        "index": index,
        "name": name,
        "error": error,
        "message": message or f"{name}: {error}",
        "kind": kind,
    }


_PNG = _make_png()
_PNG_B64 = base64.b64encode(_PNG).decode("ascii")
_STEP_PNG = _make_png((30, 120, 210))
_STEP_PNG_B64 = base64.b64encode(_STEP_PNG).decode("ascii")
_REPO_ROOT = Path(__file__).resolve().parents[5]

_SAMPLE_TASK = {
    "instruction": "Find a vegetarian lasagna recipe with at least 4 stars.",
    "max_steps": 15,
    "reference_answer": "Vegetarian Four Cheese Lasagna",
    "reference_type": "possible",
    "split": "eval",
    "start_url": "https://www.allrecipes.com/",
    "upstream_id": "Allrecipes--0",
    "web_name": "Allrecipes",
    "source": "webharbor",  # matches the real committed manifest (NOT "webharbor.webvoyager")
}
_RESET_WEB_TEXT = '[0]: <a> "Recipes" @ (100, 50);\t[1]: <input> "Search" @ (500, 60);'
_STEP_WEB_TEXT = '[0]: <a> "Recipes" @ (100, 50);'
_RESET_MODEL_WEB_TEXT = (
    f"CURRENT URL: https://www.allrecipes.com/\nPAGE TITLE: Allrecipes\nDOM:\n{_RESET_WEB_TEXT}"
)
_STEP_MODEL_WEB_TEXT = (
    "CURRENT URL: https://www.allrecipes.com/search/\n"
    "PAGE TITLE: Allrecipes Search\n"
    f"DOM:\n{_STEP_WEB_TEXT}"
)
_RESET_SOM_MODEL_WEB_TEXT = (
    "CURRENT URL: https://www.allrecipes.com/\n"
    "PAGE TITLE: Allrecipes\n"
    'DOM:\n[0]: <a> "Recipes";\t[1]: <input> "Search";'
)


def _reset_resp(instance_id: str = "iid-abc123") -> dict:
    return {
        "instance_id": instance_id,
        "screenshot_b64": _PNG_B64,
        "instruction": _SAMPLE_TASK["instruction"],
        "web_text": _RESET_WEB_TEXT,
        "url": "https://www.allrecipes.com/",
        "title": "Allrecipes",
        "max_steps": 15,
    }


def _step_resp(
    *, terminated: bool = False, truncated: bool = False, answer: str | None = None
) -> dict:
    return {
        "screenshots_b64": [_PNG_B64],
        "reward": None,
        "terminated": terminated,
        "truncated": truncated,
        "executed": [{"call": "click", "args": {"coordinate": [500, 300]}}],
        "errors": [],
        "downloads": [],
        "web_text": _STEP_WEB_TEXT,
        "url": "https://www.allrecipes.com/search/",
        "title": "Allrecipes Search",
        "answer": answer,
    }


def _close_resp() -> dict:
    return {"ok": True, "closed": True}


def _route(path: str, _body: dict) -> dict:
    """Default side_effect for _post: route by path."""
    if path == "/reset":
        return _reset_resp()
    if path == "/step":
        return _step_resp()
    if path == "/close":
        return _close_resp()
    raise ValueError(f"unexpected path: {path}")


def _make_env(
    task: dict | None = None,
    max_steps: int = 10,
    extra_tools: list[str] | None = None,
    skip_eval: bool = True,
    eval_config: dict | None = None,
    cursor: bool = True,
    include_page_context_text: bool = True,
) -> RemoteWebVoyagerEnv:
    os.environ.setdefault("WEBHARBOR_WEBVOYAGER_RPC_URL", "http://localhost:7800")
    return RemoteWebVoyagerEnv(
        task_id="allrecipes.0",
        task=task or _SAMPLE_TASK,
        base_max_steps=15,
        max_steps=max_steps,
        extra_tools=extra_tools if extra_tools is not None else [],
        skip_eval=skip_eval,
        eval_config=eval_config,
        cursor=cursor,
        include_page_context_text=include_page_context_text,
    )


def test_webvoyager_default_disables_page_context_text():
    import inspect

    import lite.gym.envs.webharbor.webvoyager.main as m

    default = (
        inspect.signature(
            m.RemoteWebVoyagerEnv.__init__,
        )
        .parameters["include_page_context_text"]
        .default
    )

    assert m.CFG.env_kwargs["include_page_context_text"] is False
    assert m._INCLUDE_PAGE_CONTEXT_TEXT is False
    assert default is False


def test_container_services_launches_with_freshness_gated_image(monkeypatch):
    import lite.gym.envs.webharbor.webvoyager.main as m

    seen = {}
    monkeypatch.setattr(m, "_services_started", set())
    monkeypatch.delenv("WEBHARBOR_WEBVOYAGER_RPC_URL", raising=False)
    monkeypatch.setattr(m, "_healthz", lambda url: bool(url))
    monkeypatch.setattr(m, "docker_rm_f", lambda *args, **kwargs: None)
    monkeypatch.setattr("lite.gym.utils.backend.ports.allocate_ports", lambda **kwargs: [7801])

    def fake_docker_run(name, image, *, mem, port, env):
        seen["name"] = name
        seen["image"] = image
        seen["mem"] = mem
        seen["port"] = port
        seen["env"] = env

    monkeypatch.setattr(m, "docker_run", fake_docker_run)

    m.WebVoyagerContainerServices().ensure("webharbor.webvoyager")

    assert seen["image"].tag == m._WEBHARBOR_WEBVOYAGER_IMAGE
    assert seen["image"].sources == ("lite/gym/envs/webharbor/webvoyager/docker",)
    assert seen["port"] == (7801, 8000)
    assert seen["env"]["WEBHARBOR_WEBVOYAGER_INSTANCES"] == str(m._RESOLVED_INSTANCES)


def test_container_shutdown_evicts_singleton_caches(monkeypatch):
    import importlib

    import lite.gym.envs.webharbor.webvoyager.main as m

    registry_mod = importlib.import_module("lite.gym.registry")
    env_id = "webharbor.webvoyager"
    monkeypatch.setattr(m, "_services_started", {env_id})
    monkeypatch.setenv("WEBHARBOR_WEBVOYAGER_RPC_URL", "http://localhost:7802")
    registry_mod._services_started.add(env_id)
    monkeypatch.setattr(
        "lite.gym.remote.reaper.docker_rm_f",
        lambda *args, **kwargs: 1,
    )

    m.WebVoyagerContainerServices().shutdown(env_id, SimpleNamespace(server_port=1234))

    assert env_id not in m._services_started
    assert env_id not in registry_mod._services_started
    assert "WEBHARBOR_WEBVOYAGER_RPC_URL" not in os.environ


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_returns_valid_screenshot():
    env = _make_env()
    env._post = MagicMock(side_effect=_route)

    obs = await env.reset()

    assert obs.image == base64.b64decode(_PNG_B64)
    raw = obs.image
    assert raw[:4] == b"\x89PNG"
    assert env._instance_id == "iid-abc123"
    await env.close()


@pytest.mark.asyncio
async def test_reset_populates_observation_fields():
    env = _make_env()
    env._post = MagicMock(side_effect=_route)

    obs = await env.reset()

    assert obs.text == _SAMPLE_TASK["instruction"]
    assert obs.metadata["url"] == "https://www.allrecipes.com/"
    assert obs.metadata["title"] == "Allrecipes"
    assert obs.metadata["web_text"] == _RESET_WEB_TEXT
    assert obs.metadata["model_web_text"] == _RESET_MODEL_WEB_TEXT
    await env.close()


@pytest.mark.asyncio
async def test_page_context_text_can_be_disabled_without_losing_screenshot():
    env = _make_env(include_page_context_text=False)
    env._post = MagicMock(side_effect=_route)

    obs = await env.reset()
    r = await env.step([_call("click", {"coordinate": [500, 300]}, call_id="call-click")])

    assert obs.text == _SAMPLE_TASK["instruction"]
    assert obs.metadata["web_text"] == _RESET_WEB_TEXT
    result = r.results[0]
    assert result.tool_call_id == "call-click"
    assert result.images[-1] == _PNG
    assert result.text is None
    assert result.error is None
    assert result.metadata == {
        "url": "https://www.allrecipes.com/search/",
        "title": "Allrecipes Search",
        "web_text": _STEP_WEB_TEXT,
        "downloads": [],
    }
    await env.close()


@pytest.mark.asyncio
async def test_page_context_text_disabled_keeps_current_action_error():
    env = _make_env(include_page_context_text=False)
    env._post = MagicMock(side_effect=_route)

    await env.reset()
    r = await env.step([_raw_call("click", ["bad"], call_id="call-bad-click")])

    result = r.results[0]
    assert result.tool_call_id == "call-bad-click"
    assert result.images[-1] == _PNG
    assert result.text is None
    assert result.error == (
        "invalid tool call: tool_call.function.arguments must be an object, got list"
    )
    assert result.metadata == {
        "url": "https://www.allrecipes.com/",
        "title": "Allrecipes",
        "web_text": _RESET_WEB_TEXT,
        "downloads": [],
        "is_error": True,
    }
    await env.close()


@pytest.mark.asyncio
async def test_reset_closes_previous_instance():
    env = _make_env()
    calls: list[tuple[str, dict]] = []

    def _record(path: str, body: dict) -> dict:
        calls.append((path, body))
        return _route(path, body)

    env._post = MagicMock(side_effect=_record)

    await env.reset()  # first reset: no /close
    first_iid = env._instance_id
    await env.reset()  # second reset: must /close first

    paths = [c[0] for c in calls]
    assert paths == ["/reset", "/close", "/reset"]
    # The /close body must reference the first instance
    close_call = next(c for c in calls if c[0] == "/close")
    assert close_call[1]["instance_id"] == first_iid
    await env.close()


@pytest.mark.asyncio
async def test_step_before_reset_raises():
    env = _make_env()
    env._post = MagicMock(side_effect=_route)

    with pytest.raises(RuntimeError, match="reset"):
        await env.step([_call("click", {"coordinate": [500, 300]})])


@pytest.mark.asyncio
async def test_step_rejects_old_flat_lite_tool_call_shape():
    env = _make_env()
    env._post = MagicMock(side_effect=_route)
    await env.reset()

    with pytest.raises(TypeError, match="env\\.step expects canonical Lite tool calls"):
        await env.step(
            [
                {
                    "call_id": "old-call-id",
                    "name": "click",
                    "arguments": {"coordinate": [500, 300]},
                },
            ]
        )
    await env.close()


@pytest.mark.asyncio
async def test_close_without_reset_is_noop():
    env = _make_env()
    env._post = MagicMock(side_effect=_route)
    await env.close()  # should not raise
    env._post.assert_not_called()


@pytest.mark.asyncio
async def test_double_close_safe():
    env = _make_env()
    env._post = MagicMock(side_effect=_route)

    await env.reset()
    await env.close()
    await env.close()  # second close: _instance_id already None

    close_calls = [c for c in env._post.call_args_list if c[0][0] == "/close"]
    assert len(close_calls) == 1


@pytest.mark.asyncio
async def test_close_clears_instance_id():
    env = _make_env()
    env._post = MagicMock(side_effect=_route)

    await env.reset()
    assert env._instance_id is not None
    await env.close()
    assert env._instance_id is None


@pytest.mark.asyncio
async def test_close_swallows_server_error():
    env = _make_env()

    def _fail_close(path: str, body: dict) -> dict:
        if path == "/close":
            raise RuntimeError("container unreachable")
        return _route(path, body)

    env._post = MagicMock(side_effect=_fail_close)
    await env.reset()
    await env.close()  # must not propagate the error


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------


class TestActionSpace:
    """step() accepts every action in the env's action space; the server handles
    execution. The host env lowers canonical Lite calls to the WebVoyager
    ``/step`` action wire."""

    async def _step_one(
        self,
        name: str,
        args: dict,
        *,
        extra_tools: list[str] | None = None,
    ) -> dict:
        env = _make_env(extra_tools=extra_tools)
        captured: list[dict] = []

        def _capture(path: str, body: dict) -> dict:
            if path == "/step":
                captured.append(body)
            return _route(path, body)

        env._post = MagicMock(side_effect=_capture)
        await env.reset()
        await env.step([_call(name, args)])
        await env.close()
        assert captured, "no /step call recorded"
        return captured[0]

    @pytest.mark.asyncio
    async def test_click(self):
        body = await self._step_one("click", {"coordinate": [500, 300]})
        action = body["actions"][0]
        assert action["name"] == "click"
        assert action["arguments"]["coordinate"] == [500, 300]

    @pytest.mark.asyncio
    async def test_type(self):
        body = await self._step_one("type", {"text": "vegetarian lasagna"})
        action = body["actions"][0]
        assert action["name"] == "type"
        assert action["arguments"]["text"] == "vegetarian lasagna"

    @pytest.mark.asyncio
    async def test_type_with_a_trailing_newline_submits(self):
        """The trailing newline must reach the container as its ``press_enter``.

        Driven through ``step()`` on purpose: the projection lives in the
        ``/step`` payload literal, and the unit tests for
        ``_to_container_action`` pass whether or not it is still called there.
        Without this, deleting the call silently removes every Enter press.
        """
        body = await self._step_one("type", {"text": "vegetarian lasagna\n"})
        action = body["actions"][0]
        assert action["name"] == "type"
        assert action["arguments"]["text"] == "vegetarian lasagna"
        assert action["arguments"]["press_enter"] is True

    @pytest.mark.asyncio
    async def test_type_without_a_trailing_newline_does_not_submit(self):
        body = await self._step_one("type", {"text": "vegetarian lasagna"})
        assert body["actions"][0]["arguments"].get("press_enter") is not True

    @pytest.mark.asyncio
    async def test_scroll(self):
        body = await self._step_one(
            "scroll",
            {"coordinate": [500, 400], "direction": "down", "amount": 3},
        )
        action = body["actions"][0]
        assert action["name"] == "scroll"

    @pytest.mark.asyncio
    async def test_key(self):
        # The wire carries SELENIUM key values, not the canonical token: this
        # container executes through ActionChains, and `_selenium_key` passes an
        # unrecognized token through — so a canonical "enter" would have been
        # TYPED AS THE TEXT "enter". U+E007 is the W3C WebDriver Enter.
        body = await self._step_one("key", {"keys": ["enter"]})
        action = body["actions"][0]
        assert action["name"] == "key"
        assert action["arguments"]["keys"] == ["\ue007"]

    @pytest.mark.asyncio
    async def test_key_arrows_and_fkeys_reach_the_wire_projected(self):
        # Regression for the discarded-projection bug: canonical "down"/"meta"/"f5"
        # used to ship verbatim, and none of the three are in the container's
        # _KEY_MAP (it only knows "arrowdown"/"cmd"/"win") — they were typed as text.
        body = await self._step_one("key", {"keys": ["ctrl", "down"]})
        assert body["actions"][0]["arguments"]["keys"] == ["\ue009", "\ue015"]
        body = await self._step_one("key", {"keys": ["meta", "f5"]})
        assert body["actions"][0]["arguments"]["keys"] == ["\ue03d", "\ue035"]

    @pytest.mark.asyncio
    async def test_key_glyphs_reach_the_wire_as_glyphs(self):
        body = await self._step_one("key", {"keys": ["+", "-", "="]})
        assert body["actions"][0]["arguments"]["keys"] == ["+", "-", "="]

    @pytest.mark.asyncio
    async def test_wait(self):
        body = await self._step_one("wait", {"duration": 2.0})
        action = body["actions"][0]
        assert action["name"] == "wait"

    @pytest.mark.asyncio
    async def test_response(self):
        body = await self._step_one(
            "response",
            {"text": "Vegetarian Four Cheese Lasagna"},
            extra_tools=["response"],
        )
        action = body["actions"][0]
        assert action["name"] == "response"

    @pytest.mark.asyncio
    async def test_back_via_extra_tool(self):
        env = _make_env(extra_tools=["back"])
        captured: list[dict] = []

        def _capture(path: str, body: dict) -> dict:
            if path == "/step":
                captured.append(body)
            return _route(path, body)

        env._post = MagicMock(side_effect=_capture)
        await env.reset()
        await env.step([_call("back")])
        await env.close()

        assert captured[0]["actions"][0]["name"] == "back"

    @pytest.mark.asyncio
    async def test_goto_via_extra_tool(self):
        env = _make_env(extra_tools=["goto"])
        captured: list[dict] = []

        def _capture(path: str, body: dict) -> dict:
            if path == "/step":
                captured.append(body)
            return _route(path, body)

        env._post = MagicMock(side_effect=_capture)
        await env.reset()
        await env.step([_call("goto", {"url": "https://example.com"})])
        await env.close()

        assert captured[0]["actions"][0]["name"] == "goto"

    @pytest.mark.asyncio
    async def test_multiple_actions_per_step(self):
        env = _make_env()
        captured: list[dict] = []

        def _capture(path: str, body: dict) -> dict:
            if path == "/step":
                captured.append(body)
            return _route(path, body)

        env._post = MagicMock(side_effect=_capture)
        await env.reset()
        await env.step(
            [
                _call("click", {"coordinate": [500, 60]}),
                _call("type", {"text": "lasagna"}),
                _call("key", {"keys": ["enter"]}),
            ]
        )
        await env.close()

        assert len(captured[0]["actions"]) == 3


@pytest.mark.asyncio
async def test_click_scroll_same_names_route_by_argument_shape():
    env = RemoteWebVoyagerEnv(
        task_id="allrecipes.0",
        task=_SAMPLE_TASK,
        base_max_steps=15,
        max_steps=10,
        extra_tools=["click", "scroll"],
        valid_actions=["click", "scroll"],
        skip_eval=True,
        include_page_context_text=True,
    )
    captured: list[dict] = []

    def _capture(path: str, body: dict) -> dict:
        if path == "/step":
            captured.append(body)
            resp = _step_resp()
            resp.update(
                {
                    "screenshots_b64": [_STEP_PNG_B64],
                    "web_text": '[7]: <button> "Continue" @ (222, 333);',
                    "url": "https://www.allrecipes.com/results/",
                    "title": "Results",
                }
            )
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)
    await env.reset()
    result = await env.step(
        [
            _call("click", {"coordinate": [500, 300]}, call_id="coord_click"),
            _call("click", {"index": 7}, call_id="som_click"),
            _call(
                "scroll",
                {"coordinate": [500, 400], "direction": "down", "amount": 3},
                call_id="coord_scroll",
            ),
            _call("scroll", {"down": False, "pages": 1}, call_id="som_scroll"),
        ]
    )
    await env.close()

    assert [(a["name"], a["arguments"]) for a in captured[0]["actions"]] == [
        ("click", {"coordinate": [500, 300]}),
        ("click", {"index": 7}),
        ("scroll", {"coordinate": [500, 400], "direction": "down", "amount": 3}),
        ("scroll", {"down": False, "pages": 1}),
    ]
    by_id = {res.tool_call_id: res for res in result.results}
    assert set(by_id) == {"coord_click", "som_click", "coord_scroll", "som_scroll"}
    for tool_result in by_id.values():
        assert tool_result.images[-1] == _STEP_PNG
        assert 'DOM:\n[7]: <button> "Continue" @ (222, 333);' in (tool_result.text or "")
        assert tool_result.error is None
        assert tool_result.metadata == {
            "url": "https://www.allrecipes.com/results/",
            "title": "Results",
            "web_text": '[7]: <button> "Continue" @ (222, 333);',
            "model_web_text": (
                "CURRENT URL: https://www.allrecipes.com/results/\n"
                "PAGE TITLE: Results\n"
                'DOM:\n[7]: <button> "Continue" @ (222, 333);'
            ),
            "downloads": [],
        }


@pytest.mark.asyncio
async def test_som_mode_rejects_coordinate_shapes_but_sends_index_shapes():
    env = RemoteWebVoyagerEnv(
        task_id="allrecipes.0",
        task=_SAMPLE_TASK,
        base_max_steps=15,
        max_steps=10,
        extra_tools=["click", "scroll"],
        valid_actions=[],
        use_som=True,
        skip_eval=True,
        include_page_context_text=True,
    )
    captured: list[dict] = []

    def _capture(path: str, body: dict) -> dict:
        if path == "/step":
            captured.append(body)
            resp = _step_resp()
            resp.update(
                {
                    "screenshots_b64": [_STEP_PNG_B64],
                    "web_text": '[8]: <button> "Marked" @ (444, 555);',
                    "url": "https://www.allrecipes.com/marked/",
                    "title": "Marked",
                }
            )
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)
    await env.reset()
    result = await env.step(
        [
            _call("click", {"coordinate": [500, 300]}, call_id="coord_click"),
            _call("click", {"index": 8}, call_id="som_click"),
            _call(
                "scroll",
                {"coordinate": [500, 400], "direction": "down", "amount": 3},
                call_id="coord_scroll",
            ),
            _call("scroll", {"down": True, "pages": 0.5}, call_id="som_scroll"),
        ]
    )
    await env.close()

    assert [(a["name"], a["arguments"]) for a in captured[0]["actions"]] == [
        ("click", {"index": 8}),
        ("scroll", {"down": True, "pages": 0.5}),
    ]
    by_id = {res.tool_call_id: res for res in result.results}
    assert by_id["coord_click"].error == (
        "invalid action: click; choose an available action for this task"
    )
    assert by_id["coord_scroll"].error == (
        "invalid action: scroll; choose an available action for this task"
    )
    assert by_id["som_click"].error is None
    assert by_id["som_scroll"].error is None
    for tool_result in by_id.values():
        assert tool_result.images[-1] == _STEP_PNG
        assert 'DOM:\n[8]: <button> "Marked";' in (tool_result.text or "")
        assert "@ (" not in (tool_result.text or "")
        assert tool_result.metadata["web_text"] == '[8]: <button> "Marked" @ (444, 555);'
        assert tool_result.metadata["model_web_text"] == (
            "CURRENT URL: https://www.allrecipes.com/marked/\n"
            "PAGE TITLE: Marked\n"
            'DOM:\n[8]: <button> "Marked";'
        )
        assert "@ (" not in tool_result.metadata["model_web_text"]


@pytest.mark.asyncio
async def test_malformed_som_same_name_shapes_return_current_feedback_without_rpc_actions():
    env = RemoteWebVoyagerEnv(
        task_id="allrecipes.0",
        task=_SAMPLE_TASK,
        base_max_steps=15,
        max_steps=10,
        extra_tools=["click", "scroll"],
        valid_actions=["click", "scroll"],
        skip_eval=True,
        include_page_context_text=True,
    )
    captured: list[dict] = []

    def _capture(path: str, body: dict) -> dict:
        if path == "/step":
            captured.append(body)
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)
    await env.reset()
    result = await env.step(
        [
            _call("click", {"index": "7"}, call_id="bad_click_type"),
            _call(
                "click",
                {"coordinate": [500, 300], "index": 7},
                call_id="bad_click_mixed",
            ),
            _call("scroll", {"down": True, "pages": "one"}, call_id="bad_scroll_type"),
        ]
    )
    await env.close()

    assert captured == []
    by_id = {res.tool_call_id: res for res in result.results}
    assert by_id["bad_click_type"].error == (
        "invalid arguments for click: click.arguments.index must be an integer"
    )
    assert by_id["bad_click_mixed"].error == (
        "invalid arguments for click: click.arguments do not match its tool schema"
    )
    assert by_id["bad_scroll_type"].error == (
        "invalid arguments for scroll: scroll.arguments.pages must be a number"
    )
    for tool_result in by_id.values():
        assert tool_result.images[-1] == _PNG
        assert "CURRENT URL: https://www.allrecipes.com/" in (tool_result.text or "")
        assert tool_result.metadata == {
            "url": "https://www.allrecipes.com/",
            "title": "Allrecipes",
            "web_text": _RESET_WEB_TEXT,
            "model_web_text": _RESET_MODEL_WEB_TEXT,
            "downloads": [],
            "is_error": True,
        }


def test_som_same_name_extra_shapes_do_not_create_coordinate_annotations():
    annotations = coordinate_annotation_records(
        [
            _call("click", {"coordinate": [500, 300]}, call_id="coord_click"),
            _call("click", {"index": 7}, call_id="som_click"),
            _call(
                "scroll",
                {"coordinate": [500, 400], "direction": "down", "amount": 3},
                call_id="coord_scroll",
            ),
            _call("scroll", {"down": False, "pages": 1}, call_id="som_scroll"),
        ],
    )

    assert [action["result_call_id"] for action in annotations] == [
        "coord_click",
        "coord_scroll",
    ]
    assert [action["name"] for action in annotations] == ["click", "scroll"]
    assert all(set(action) == {"name", "arguments", "result_call_id"} for action in annotations)
    assert all("id" not in action for action in annotations)


# ---------------------------------------------------------------------------
# Termination and truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_action_terminates():
    env = _make_env(extra_tools=["response"])

    def _term_on_response(path: str, body: dict) -> dict:
        if path == "/step":
            name = body["actions"][0]["name"]
            return _step_resp(terminated=(name == "response"), answer="some answer")
        return _route(path, body)

    env._post = MagicMock(side_effect=_term_on_response)
    await env.reset()

    r = await env.step([_call("response", {"text": "some answer"})])
    assert r.terminated is True
    assert r.truncated is False
    assert r.info["answer"] == "some answer"
    await env.close()


@pytest.mark.asyncio
async def test_response_with_call_id_returns_no_tool_result():
    env = _make_env(extra_tools=["response"])

    def _term_on_response(path: str, body: dict) -> dict:
        if path == "/step":
            assert body["actions"] == [
                {
                    "name": "response",
                    "call_id": "call-response",
                    "arguments": {"text": "some answer"},
                }
            ]
            return _step_resp(terminated=True, answer="some answer")
        return _route(path, body)

    env._post = MagicMock(side_effect=_term_on_response)
    await env.reset()

    r = await env.step(
        [
            _call(
                "response",
                {"text": "some answer"},
                call_id="call-response",
            ),
        ]
    )
    await env.close()

    assert r.terminated is True
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    # What this test defends is unchanged: the exact ``/step`` body the container
    # received, asserted in ``_term_on_response`` above.
    assert r.results == []


@pytest.mark.asyncio
async def test_terminate_action_terminates():
    env = _make_env(extra_tools=["terminate"])

    def _term(path: str, body: dict) -> dict:
        if path == "/step":
            assert body["actions"] == [{"name": "terminate", "arguments": {"status": "success"}}]
            return _step_resp(terminated=True)
        return _route(path, body)

    env._post = MagicMock(side_effect=_term)
    await env.reset()

    r = await env.step([_call("terminate", {"status": "success"})])
    assert r.terminated is True
    await env.close()


@pytest.mark.asyncio
async def test_done_boundary_alias_is_unsupported_without_public_schema():
    env = _make_env(extra_tools=[])

    def _capture(path: str, body: dict) -> dict:
        if path == "/step":
            assert body["actions"] == []
            return _step_resp()
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)
    await env.reset()

    r = await env.step(
        [
            _call(
                "done",
                {"text": "some answer", "success": True},
                call_id="call_done",
            ),
        ]
    )

    assert r.terminated is False
    assert r.truncated is False
    assert r.info["answer"] is None
    assert len(r.results) == 1
    assert r.results[0].tool_call_id == "call_done"
    assert r.results[0].images[-1] == _PNG
    assert "CURRENT URL: https://www.allrecipes.com/" in (r.results[0].text or "")
    assert r.results[0].error == "done is not available in this task."
    assert r.results[0].metadata == {
        "url": "https://www.allrecipes.com/",
        "title": "Allrecipes",
        "web_text": _RESET_WEB_TEXT,
        "model_web_text": _RESET_MODEL_WEB_TEXT,
        "downloads": [],
        "is_error": True,
    }
    await env.close()


@pytest.mark.asyncio
async def test_truncation_at_max_steps():
    """Server signals truncated=True at max_steps; host env passes it through.

    The container derives that flag from the shipped ``step_count``, so the fake
    mirrors that rather than counting POSTs itself.
    """
    env = _make_env(max_steps=2)

    def _count_steps(path: str, body: dict) -> dict:
        if path == "/step":
            return _step_resp(truncated=(body["step_count"] >= 2))
        return _route(path, body)

    env._post = MagicMock(side_effect=_count_steps)
    await env.reset()

    r1 = await env.step([_call("click", {"coordinate": [500, 300]})])
    assert not r1.truncated
    assert not r1.terminated

    r2 = await env.step([_call("click", {"coordinate": [500, 300]})])
    assert r2.truncated is True
    assert r2.terminated is False
    await env.close()


@pytest.mark.asyncio
async def test_step_returns_screenshot_after_action():
    env = _make_env()
    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step([_call("scroll", {"direction": "down", "amount": 3})])
    assert r.results[0].images[-1] == base64.b64decode(_PNG_B64)
    raw = r.results[0].images[-1]
    assert raw[:4] == b"\x89PNG"
    await env.close()


@pytest.mark.asyncio
async def test_step_exposes_executed_actions_in_info():
    env = _make_env()
    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step([_call("click", {"coordinate": [500, 300]})])
    assert "executed_actions" in r.info
    assert isinstance(r.info["executed_actions"], list)
    await env.close()


@pytest.mark.asyncio
async def test_step_exposes_downloads_in_info():
    env = _make_env()

    def _with_download(path: str, body: dict) -> dict:
        if path == "/step":
            resp = _step_resp()
            resp["downloads"] = ["file.pdf"]
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_with_download)
    await env.reset()

    r = await env.step([_call("click", {"coordinate": [100, 200]})])
    assert r.info["downloads"] == ["file.pdf"]
    assert r.results[0].metadata["downloads"] == ["file.pdf"]
    await env.close()


@pytest.mark.asyncio
async def test_step_errors_passed_through_in_info():
    env = _make_env()

    def _with_error(path: str, body: dict) -> dict:
        if path == "/step":
            resp = _step_resp()
            resp["errors"] = ["click: element not found at coordinate (0, 0)"]
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_with_error)
    await env.reset()

    r = await env.step([_call("click", {"coordinate": [0, 0]})])
    assert r.info["errors"] == ["click: element not found at coordinate (0, 0)"]
    await env.close()


@pytest.mark.asyncio
async def test_action_error_pairs_to_originating_call_id():
    """A failed action is PER-CALL feedback on the call that emitted it.

    It used to be spliced into the shared ``web_text`` blob, which is sent to
    every call of the turn — the agent could not attribute the failure.
    """
    env = _make_env()

    def _with_error(path: str, body: dict) -> dict:
        if path == "/step":
            resp = _step_resp()
            resp["errors"] = ["type: element not interactable"]
            resp["action_errors"] = [
                _container_action_error_record(
                    1,
                    "type",
                    "element not interactable",
                )
            ]
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_with_error)
    await env.reset()

    r = await env.step(
        [
            _call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [500, 300]}]},
                call_id="call-a",
            ),
            _call(
                "computer",
                {
                    "actions": [
                        {"action": "type", "coordinate": [500, 300], "text": "hi"},
                    ]
                },
                call_id="call-b",
            ),
        ]
    )

    by_id = {res.tool_call_id: res for res in r.results}
    assert not (by_id["call-a"].metadata or {}).get("is_error")
    # The healthy call still gets the page observation, not the error.
    assert "[ACTION ERROR]" not in (by_id["call-a"].text or "")
    assert by_id["call-b"].metadata["is_error"] is True
    assert "element not interactable" not in (by_id["call-b"].text or "")
    assert "element not interactable" in by_id["call-b"].error
    assert r.info["errors"] == ["type: element not interactable"]
    await env.close()


@pytest.mark.asyncio
async def test_page_context_disabled_preserves_container_action_error():
    env = _make_env(include_page_context_text=False)

    def _with_error(path: str, body: dict) -> dict:
        if path == "/step":
            resp = _step_resp()
            resp["errors"] = ["click: stale element"]
            resp["action_errors"] = [
                _container_action_error_record(
                    0,
                    "click",
                    "stale element",
                )
            ]
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_with_error)
    await env.reset()

    r = await env.step(
        [
            _call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [500, 300]}]},
                call_id="call-click",
            ),
        ]
    )

    result = r.results[0]
    assert result.tool_call_id == "call-click"
    assert result.images[-1] == _PNG
    assert result.text is None
    assert result.error == "invalid arguments for click: stale element"
    assert result.metadata == {
        "url": "https://www.allrecipes.com/search/",
        "title": "Allrecipes Search",
        "web_text": _STEP_WEB_TEXT,
        "downloads": [],
        "is_error": True,
    }
    assert r.info["errors"] == ["click: stale element"]
    await env.close()


@pytest.mark.asyncio
async def test_page_context_disabled_hides_current_context_on_valid_action_rejection():
    env = _make_env(
        include_page_context_text=False,
        max_steps=10,
        extra_tools=["back", "response"],
    )
    paths: list[str] = []

    def _capture(path: str, body: dict) -> dict:
        paths.append(path)
        if path == "/step":
            raise AssertionError("rejected action should not POST /step")
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)

    await env.reset()
    r = await env.step(
        [
            _call(
                "computer",
                {"actions": [{"action": "screenshot"}]},
                call_id="call-screenshot",
            ),
        ]
    )
    await env.close()

    assert paths == ["/reset", "/close"]
    assert len(r.results) == 1
    result = r.results[0]
    assert result.tool_call_id == "call-screenshot"
    assert result.images[-1] == _PNG
    assert result.text is None
    assert result.error == "invalid action: screenshot; choose an available action for this task"
    assert result.metadata == {
        "url": "https://www.allrecipes.com/",
        "title": "Allrecipes",
        "web_text": _RESET_WEB_TEXT,
        "downloads": [],
        "is_error": True,
    }


def test_action_error_pairing_uses_record_name_not_action_fallback():
    paired = _pair_webvoyager_action_errors(
        [{"arguments": {"coordinate": [500, 300]}}],
        ["call-click"],
        [_container_action_error_record(0, "click", "bad coordinate")],
        [],
    )

    assert paired == {"call-click": "invalid arguments for click: bad coordinate"}


def test_unsupported_action_record_pairs_as_execution_failure():
    paired = _pair_webvoyager_action_errors(
        [{"name": "cursor_position", "arguments": {}}],
        ["call-cursor"],
        [
            _container_action_error_record(
                0,
                "cursor_position",
                "unsupported action: cursor_position",
                kind="unsupported_action",
                message="unsupported action: cursor_position",
            )
        ],
        [],
    )

    assert paired == {"call-cursor": "cursor_position failed: execution failed"}


@pytest.mark.asyncio
async def test_structured_action_error_pairs_same_named_calls_by_index():
    env = _make_env()

    def _with_error(path: str, body: dict) -> dict:
        if path == "/step":
            resp = _step_resp()
            resp["errors"] = ["click: second click failed"]
            resp["action_errors"] = [
                _container_action_error_record(1, "click", "second click failed")
            ]
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_with_error)
    await env.reset()

    r = await env.step(
        [
            _call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [500, 300]}]},
                call_id="call-a",
            ),
            _call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [700, 300]}]},
                call_id="call-b",
            ),
        ]
    )
    await env.close()

    by_id = {res.tool_call_id: res for res in r.results}
    assert not (by_id["call-a"].metadata or {}).get("is_error")
    assert by_id["call-b"].metadata["is_error"] is True
    assert by_id["call-b"].error == ("invalid arguments for click: second click failed")


@pytest.mark.asyncio
async def test_scalar_key_returns_current_feedback_without_rpc():
    env = _make_env()
    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            _call(
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
    assert result.images[-1] == _PNG
    assert result.metadata["is_error"] is True
    assert result.error == (
        "invalid arguments for key: key.keys must be a list of strings, not a string"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("keys", "expected_error"),
    [
        (["plus"], "invalid arguments for key: unknown key token 'plus'"),
        ([" "], "invalid arguments for key: unknown key token ' '"),
    ],
)
async def test_noncanonical_key_tokens_return_current_feedback_without_rpc(
    keys,
    expected_error,
):
    env = _make_env()
    env._post = MagicMock(side_effect=_route)
    await env.reset()

    r = await env.step(
        [
            _call(
                "computer",
                {"actions": [{"action": "key", "keys": keys}]},
                call_id="call-key",
            ),
        ]
    )
    await env.close()

    paths = [call.args[0] for call in env._post.call_args_list]
    assert paths == ["/reset", "/close"]
    result = r.results[0]
    assert result.tool_call_id == "call-key"
    assert result.images[-1] == _PNG
    assert result.metadata["is_error"] is True
    assert result.error == expected_error


@pytest.mark.asyncio
async def test_structured_tool_execution_error_uses_typed_generic_feedback():
    env = _make_env()

    def _with_error(path: str, body: dict) -> dict:
        if path == "/step":
            resp = _step_resp()
            raw = "selenium WebDriver container target closed"
            resp["errors"] = [f"click: {raw}"]
            resp["action_errors"] = [
                _container_action_error_record(
                    0,
                    "click",
                    raw,
                    kind="tool_execution",
                )
            ]
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_with_error)
    await env.reset()

    r = await env.step(
        [
            _call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [500, 300]}]},
                call_id="call-click",
            ),
        ]
    )
    await env.close()

    result = r.results[0]
    assert result.tool_call_id == "call-click"
    assert result.metadata["is_error"] is True
    assert result.error == "click failed: execution failed"
    assert "selenium" not in result.error
    assert "container" not in result.error


@pytest.mark.asyncio
async def test_legacy_errors_pair_when_no_structured_record_claims_the_action():
    env = _make_env()

    def _with_error(path: str, body: dict) -> dict:
        if path == "/step":
            resp = _step_resp()
            resp["errors"] = [
                "click: first click failed",
                "type: unstructured type failed",
            ]
            resp["action_errors"] = [
                _container_action_error_record(0, "click", "first click failed")
            ]
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_with_error)
    await env.reset()

    r = await env.step(
        [
            _call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [500, 300]}]},
                call_id="call-a",
            ),
            _call(
                "computer",
                {
                    "actions": [
                        {"action": "type", "coordinate": [500, 300], "text": "hi"},
                    ]
                },
                call_id="call-b",
            ),
        ]
    )
    await env.close()

    by_id = {res.tool_call_id: res for res in r.results}
    assert by_id["call-a"].error == ("invalid arguments for click: first click failed")
    assert by_id["call-b"].metadata["is_error"] is True
    assert by_id["call-b"].error == ("invalid arguments for type: unstructured type failed")
    assert r.info["errors"] == [
        "click: first click failed",
        "type: unstructured type failed",
    ]


@pytest.mark.asyncio
async def test_malformed_structured_action_error_falls_back_to_legacy_errors():
    env = _make_env()

    def _with_error(path: str, body: dict) -> dict:
        if path == "/step":
            resp = _step_resp()
            resp["errors"] = ["type: unstructured type failed"]
            resp["action_errors"] = [
                {
                    "index": "not-an-index",
                    "name": "type",
                    "error": "unusable structured entry",
                    "kind": "model_action",
                }
            ]
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_with_error)
    await env.reset()

    r = await env.step(
        [
            _call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [500, 300]}]},
                call_id="call-a",
            ),
            _call(
                "computer",
                {
                    "actions": [
                        {"action": "type", "coordinate": [500, 300], "text": "hi"},
                    ]
                },
                call_id="call-b",
            ),
        ]
    )
    await env.close()

    by_id = {res.tool_call_id: res for res in r.results}
    assert not (by_id["call-a"].metadata or {}).get("is_error")
    assert by_id["call-b"].metadata["is_error"] is True
    assert by_id["call-b"].error == ("invalid arguments for type: unstructured type failed")
    assert r.info["errors"] == ["type: unstructured type failed"]


@pytest.mark.asyncio
async def test_malformed_tool_call_with_call_id_returns_current_feedback():
    env = _make_env()

    def _capture(path: str, body: dict) -> dict:
        if path == "/step":
            assert body["actions"] == []
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)
    await env.reset()

    r = await env.step(
        [
            _raw_call("click", ["bad"], call_id="call-bad-click"),
        ]
    )
    await env.close()

    assert len(r.results) == 1
    result = r.results[0]
    assert result.tool_call_id == "call-bad-click"
    assert result.error == (
        "invalid tool call: tool_call.function.arguments must be an object, got list"
    )
    assert result.images[-1] == _PNG
    assert "CURRENT URL: https://www.allrecipes.com/" in (result.text or "")
    assert result.metadata == {
        "url": "https://www.allrecipes.com/",
        "title": "Allrecipes",
        "web_text": _RESET_WEB_TEXT,
        "model_web_text": _RESET_MODEL_WEB_TEXT,
        "downloads": [],
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_validation_only_malformed_known_action_skips_the_remote_but_consumes_budget():
    """A fully-rejected step costs no REMOTE round-trip but still costs a turn.

    The two are different budgets and were conflated. Not POSTing ``/step`` is
    correct -- there is nothing to execute. But ``_step_count`` must still
    advance, or a model emitting only invalid calls never truncates; the
    sibling ``online_mind2web`` increments before the same branch.

    And because this rejected step is the one that EXHAUSTS the budget, it must
    also be SCORED. Returning ``truncated=True, reward=None`` here threw away a
    complete trajectory; ``_evaluate`` is host-side for this env, so the fast
    path can score without the round-trip it correctly skips.
    """
    env = _make_env(max_steps=1)
    paths: list[str] = []
    evaluated: list[bool] = []

    async def _evaluate() -> tuple[float, dict]:
        evaluated.append(True)
        return 0.75, {"eval_reason": "judged after a fully-rejected final step"}

    def _capture(path: str, body: dict) -> dict:
        paths.append(path)
        if path == "/step":
            raise AssertionError("validation-only malformed action should not POST /step")
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)
    env._evaluate = _evaluate
    await env.reset()

    r = await env.step(
        [
            _raw_call("click", ["bad"], call_id="call-bad-click"),
        ]
    )

    assert env._step_count == 1, "a rejected step still costs the model a turn"
    await env.close()

    assert paths == ["/reset", "/close"], "but it must not cost a remote round-trip"
    assert r.terminated is False
    # max_steps=1, and this rejected step consumed it -- so the episode
    # truncates HERE. Before the budget fix it never could, which is exactly
    # how a model emitting only invalid calls ran unbounded.
    assert r.truncated is True
    # ...and truncating HERE must still score. This is the whole point: the
    # episode is over, so a reward of None means the trajectory was discarded.
    assert evaluated == [True], "the truncating step must trigger evaluation"
    assert r.reward == 0.75
    assert r.info["eval_reason"] == "judged after a fully-rejected final step"
    result = r.results[0]
    assert result.error == (
        "invalid tool call: tool_call.function.arguments must be an object, got list"
    )
    assert result.images[-1] == _PNG
    assert "CURRENT URL: https://www.allrecipes.com/" in (result.text or "")
    assert result.metadata == {
        "url": "https://www.allrecipes.com/",
        "title": "Allrecipes",
        "web_text": _RESET_WEB_TEXT,
        "model_web_text": _RESET_MODEL_WEB_TEXT,
        "downloads": [],
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_fully_rejected_step_mid_episode_does_not_evaluate():
    """Scoring happens at the END, not on every rejected step.

    Complement to the test above: with budget left, a fully-rejected step is a
    wasted turn, not a finished episode. Evaluating here would score a partial
    trajectory and burn a judge call per malformed turn.
    """
    env = _make_env(max_steps=5)

    async def _should_not_evaluate() -> tuple[float, dict]:
        raise AssertionError("a mid-episode rejected step must not evaluate")

    env._post = MagicMock(side_effect=_route)
    env._evaluate = _should_not_evaluate
    await env.reset()

    r = await env.step([_raw_call("click", ["bad"], call_id="call-bad-click")])

    assert env._step_count == 1
    assert r.truncated is False, "budget remains -- the episode is not over"
    assert r.reward is None, "an unfinished episode has no score yet"
    await env.close()


@pytest.mark.asyncio
async def test_rejected_steps_still_truncate_when_the_container_counter_lags():
    """The host budget is authoritative on the EXECUTED path too.

    The container never sees the turns we reject client-side, so a counter of its
    own could only lag. Here 2 of 3 turns are rejected and the container is POSTed
    exactly once -- it truncates on time only because the host ships the count.
    """
    env = _make_env(max_steps=3)
    evaluated: list[bool] = []
    step_bodies: list[dict] = []

    async def _evaluate() -> tuple[float, dict]:
        evaluated.append(True)
        return 1.0, {}

    def _mirror_container(path: str, body: dict) -> dict:
        if path == "/step":
            step_bodies.append(body)
            # docker/server.py: ``inst.step_count = body["step_count"]``.
            return _step_resp(truncated=body["step_count"] >= 3)
        return _route(path, body)

    env._post = MagicMock(side_effect=_mirror_container)
    env._evaluate = _evaluate
    await env.reset()

    # Two fully-rejected turns: host count 2, container count still 0.
    for _ in range(2):
        rejected = await env.step([_raw_call("click", ["bad"], call_id="call-bad")])
        assert rejected.truncated is False

    # A real third turn: the container's FIRST POST, and it already carries
    # the host's count of 3 -- so it truncates here rather than two turns late.
    r = await env.step([_call("click", {"coordinate": [10, 10]}, call_id="ok")])

    assert env._step_count == 3
    assert [b["step_count"] for b in step_bodies] == [3]
    assert r.truncated is True, (
        "host budget exhausted -- a container counting its own POSTs would "
        "run the episode past max_steps"
    )
    assert evaluated == [True] and r.reward == 1.0
    await env.close()


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_only_feedback():
    env = _make_env()
    captured: list[dict] = []

    def _capture(path: str, body: dict) -> dict:
        if path == "/step":
            captured.append(body)
            assert body["actions"] == []
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)
    await env.reset()

    r = await env.step(
        [
            _call("foo", {}, call_id="call-foo"),
        ]
    )
    await env.close()

    assert len(captured) == 0
    assert len(r.results) == 1
    result = r.results[0]
    assert result.tool_call_id == "call-foo"
    assert result.error == "unknown tool: foo"
    assert result.images == []
    assert result.text is None
    assert result.metadata == {"is_error": True}
    assert r.terminated is False
    assert r.truncated is False


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [-1, "NaN", "Infinity", 31])
async def test_bad_wait_duration_returns_current_feedback_without_rpc(duration):
    env = _make_env()
    captured: list[dict] = []

    def _capture(path: str, body: dict) -> dict:
        if path == "/step":
            captured.append(body)
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)
    await env.reset()

    r = await env.step(
        [
            _call("wait", {"duration": duration}, call_id="call-wait"),
        ]
    )
    await env.close()

    assert captured == []
    assert len(r.results) == 1
    result = r.results[0]
    assert result.tool_call_id == "call-wait"
    assert result.images[-1] == _PNG
    assert result.metadata["is_error"] is True
    assert result.error.startswith("invalid arguments for wait: wait.duration")


@pytest.mark.asyncio
async def test_container_infra_failure_still_raises():
    """A dead container / RPC failure must keep hard-raising out of step()."""
    env = _make_env()

    def _boom(path: str, body: dict) -> dict:
        if path == "/step":
            raise RuntimeError("webharbor.webvoyager container /step returned 500: boom")
        return _route(path, body)

    env._post = MagicMock(side_effect=_boom)
    await env.reset()

    with pytest.raises(RuntimeError, match="returned 500"):
        await env.step([_call("click", {"coordinate": [500, 300]})])
    await env.close()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_platform_and_type():
    env = _make_env()
    m = env.metadata
    assert m.platform == "browser"
    assert m.task_type == "use"


def test_metadata_valid_actions_grounded_subset():
    env = _make_env()
    m = env.metadata
    assert "click" in m.valid_actions
    assert "type" in m.valid_actions
    assert "scroll" in m.valid_actions
    assert "key" in m.valid_actions
    assert "response" not in m.valid_actions
    assert "goto" not in m.valid_actions
    assert "back" not in m.valid_actions


def test_metadata_others_fields():
    env = _make_env()
    m = env.metadata
    # Identity is framework-owned (metadata contract): a DIRECTLY-constructed env carries
    # no task_id; gym.make-built envs get it from the registry key (pinned by
    # tests/gym/metadata/test_metadata_invariant.py).
    assert "task_id" not in m.others
    assert m.others["start_url"] == "https://www.allrecipes.com/"
    assert m.others["web_name"] == "Allrecipes"
    assert m.others["source"] == "webharbor"


def test_registered_manifest_source_is_webharbor():
    """T4: the committed manifest's source is "webharbor", not "webharbor.webvoyager".
    Read it off the real registry (not the injected sample) so the fact can't
    silently drift."""
    import lite.gym as gym

    ids = gym.registry.task_ids("webharbor.webvoyager", split="eval")
    meta = gym.registry.task_metadata("webharbor.webvoyager", ids[0])
    assert meta.others["source"] == "webharbor"


def test_docker_asset_fetch_is_pinned_and_scoped_to_pinned_sites(tmp_path):
    docker_dir = _REPO_ROOT / "lite/gym/envs/webharbor/webvoyager/docker"
    dockerfile = (docker_dir / "Dockerfile").read_text()
    patch = (docker_dir / "patches/fetch_assets-no-same-owner.patch").read_text()

    assert re.search(r"^ARG WEBHARBOR_ASSETS_REVISION=[0-9a-f]{40}$", dockerfile, re.M)
    assert 'ASSETS_REVISION="${WEBHARBOR_ASSETS_REVISION}" ./scripts/fetch_assets.sh' in dockerfile
    assert "find sites -mindepth 1 -maxdepth 1 -type d ! -name .cache | sort" in patch
    assert 'INCLUDE_ARGS+=(--include "$site.tar.gz")' in patch

    webharbor = tmp_path / "webharbor"
    scripts = webharbor / "scripts"
    scripts.mkdir(parents=True)
    sites = webharbor / "sites"
    for site in ("amazon", "apple"):
        (sites / site).mkdir(parents=True)
    (webharbor / ".assets-revision").write_text(
        "repo: ChilleD/WebHarbor\nrevision: old-revision\n",
        encoding="utf-8",
    )
    fetch_assets = scripts / "fetch_assets.sh"
    fetch_assets.write_text(
        """#!/usr/bin/env bash
# Pull per-site asset tarballs from the Hugging Face dataset and extract
# them into sites/.
#
# The dataset stores assets as <site>.tar.gz (one tarball per site) to
# dodge the small-file tax that previously made `hf download` stall on
# 4000+ tiny image files. Each tarball extracts back to
# sites/<site>/{instance_seed,static/images,static/external_cache}.
#
# Usage:
#   ./scripts/fetch_assets.sh                 # fetch all sites at pinned rev
#   ./scripts/fetch_assets.sh google_search   # fetch one site only
#   ASSETS_REVISION=abc123 ./scripts/fetch_assets.sh   # override pin
#
# Requires:
#   - hf CLI  (pip install -U "huggingface_hub[cli]")
#   - (optional) HF auth if the dataset becomes gated: hf auth login  (or set HF_TOKEN env)
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=$(awk '/^repo:/ {print $2}' .assets-revision)
REVISION="${ASSETS_REVISION:-$(awk '/^revision:/ {print $2}' .assets-revision)}"
ONLY_SITE="${1:-}"
CACHE_DIR="sites/.cache/tarballs"

if ! command -v hf >/dev/null 2>&1; then
    echo "fetch_assets: 'hf' CLI not found. Install with: pip install -U \\"huggingface_hub[cli]\\"" >&2
    exit 1
fi

mkdir -p "$CACHE_DIR"
echo "[fetch] huggingface.co/datasets/$REPO @ $REVISION -> sites/"

if [[ -n "$ONLY_SITE" ]]; then
    INCLUDE="$ONLY_SITE.tar.gz"
    echo "[fetch] scope: $ONLY_SITE only"
else
    INCLUDE="*.tar.gz"
fi

hf download "$REPO" --repo-type dataset --revision "$REVISION" \\
    --include "$INCLUDE" --local-dir "$CACHE_DIR"

shopt -s nullglob
extracted=0
for tarball in "$CACHE_DIR"/*.tar.gz; do
    site=$(basename "$tarball" .tar.gz)
    if [[ -n "$ONLY_SITE" && "$site" != "$ONLY_SITE" ]]; then continue; fi
    echo "[fetch] extracting $site"
    tar -xzf "$tarball" -C sites/
    extracted=$((extracted + 1))
done

echo "[fetch] done — $extracted site(s) extracted into sites/"
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["patch", "-p1"],
        cwd=webharbor,
        input=patch,
        text=True,
        check=True,
        capture_output=True,
    )
    patched = fetch_assets.read_text()
    assert 'INCLUDE="*.tar.gz"' not in patched
    assert 'tar --no-same-owner -xzf "$tarball" -C sites/' in patched
    subprocess.run(["bash", "-n", str(fetch_assets)], check=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    include_log = tmp_path / "hf-includes.txt"
    fake_hf = fake_bin / "hf"
    fake_hf.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "download" ]]; then
    echo "fake hf only supports download" >&2
    exit 2
fi
shift

declare -a includes=()
local_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --include)
            includes+=("$2")
            shift 2
            ;;
        --local-dir)
            local_dir="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

: "${local_dir:?}"
: "${HF_INCLUDE_LOG:?}"
mkdir -p "$local_dir"
printf "%s\\n" "${includes[@]}" > "$HF_INCLUDE_LOG"

for include in "${includes[@]}"; do
    if [[ "$include" == ".cache.tar.gz" ]]; then
        echo "cache dir must not be fetched as a site" >&2
        exit 42
    fi
    site="${include%.tar.gz}"
    tmp="$(mktemp -d)"
    mkdir -p "$tmp/$site/instance_seed"
    printf "seed\\n" > "$tmp/$site/instance_seed/seed.txt"
    tar -czf "$local_dir/$include" -C "$tmp" "$site"
    rm -rf "$tmp"
done
""",
        encoding="utf-8",
    )
    fake_hf.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["HF_INCLUDE_LOG"] = str(include_log)
    env["ASSETS_REVISION"] = "0123456789abcdef0123456789abcdef01234567"

    result = subprocess.run(
        ["bash", str(fetch_assets)],
        cwd=tmp_path,
        env=env,
        text=True,
        check=True,
        capture_output=True,
    )

    assert "scope: 2 site(s) declared by the pinned WebHarbor checkout" in result.stdout
    assert include_log.read_text(encoding="utf-8").splitlines() == [
        "amazon.tar.gz",
        "apple.tar.gz",
    ]
    assert (sites / "amazon/instance_seed/seed.txt").is_file()
    assert (sites / "apple/instance_seed/seed.txt").is_file()


def test_task_manifest_generator_keeps_env_id_out_of_task_ids():
    generator = (
        _REPO_ROOT
        / "lite/gym/envs/webharbor/webvoyager/scripts/utils/tasks.sh"
    ).read_text()

    assert 'task_id = f"{site_slug}.{slug(suffix)}"' in generator
    assert "task_id = f\"webvoyager." not in generator


def test_max_steps_from_constructor():
    env = _make_env(max_steps=7)
    assert env.max_steps == 7


def test_max_steps_falls_back_to_base():
    env = RemoteWebVoyagerEnv(
        task_id="t",
        task=_SAMPLE_TASK,
        base_max_steps=20,
        max_steps=None,
        extra_tools=[],
    )
    assert env.max_steps == 20


# ---------------------------------------------------------------------------
# Extra tools
# ---------------------------------------------------------------------------


class TestExtraTools:
    def test_default_no_extra_tool_schemas(self):
        env = _make_env(extra_tools=[])
        assert env.metadata.extra_tool_schemas == []

    def test_back_schema_advertised(self):
        env = _make_env(extra_tools=["back"])
        names = [tool_schema_name(t) for t in env.metadata.extra_tool_schemas]
        assert "back" in names

    def test_goto_schema_advertised(self):
        env = _make_env(extra_tools=["goto"])
        names = [tool_schema_name(t) for t in env.metadata.extra_tool_schemas]
        assert "goto" in names

    def test_back_and_goto_together(self):
        env = _make_env(extra_tools=["back", "goto"])
        names = {tool_schema_name(t) for t in env.metadata.extra_tool_schemas}
        assert names == {"back", "goto"}

    def test_schemas_are_canonical_lite_tool_dicts(self):
        env = _make_env(extra_tools=["back", "goto"])
        for schema in env.metadata.extra_tool_schemas:
            assert schema["type"] == "function"
            assert set(schema) == {"type", "function"}
            assert {"name", "description", "parameters"} <= set(schema["function"])

    def test_unsupported_nav_tool_raises(self):
        with pytest.raises(ValueError, match="unknown extra_tools"):
            _make_env(extra_tools=["forward"])

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError):
            _make_env(extra_tools=["nonsense_tool"])

    def test_declared_extra_tools_plus_finish(self):
        # Host-requestable extra_tools: coord-mode nav (back/goto) + WebHarbor SoM
        # tools. click_elem/fill are NOT host-exposed — the SoM index interface is
        # the WebHarbor click/input tools; click_elem/fill stay a dormant server
        # capability (advertising them resolved to an empty schema — D3 footgun).
        assert "goto" in _KNOWN_STANDALONE_TOOL_NAMES
        assert "back" in _KNOWN_STANDALONE_TOOL_NAMES
        assert {
            "click",
            "input",
            "scroll",
            "navigate",
            "go_back",
            "response",
        } <= _KNOWN_STANDALONE_TOOL_NAMES
        assert "done" not in _KNOWN_STANDALONE_TOOL_NAMES
        assert "click_elem" not in _KNOWN_STANDALONE_TOOL_NAMES
        assert "fill" not in _KNOWN_STANDALONE_TOOL_NAMES
        assert "forward" not in _KNOWN_STANDALONE_TOOL_NAMES

    def test_done_is_not_public_extra_tool(self):
        with pytest.raises(ValueError, match="unknown extra_tools"):
            _make_env(extra_tools=["done"])

    def test_click_elem_fill_not_advertised(self):
        # D3: these used to resolve to an empty schema via the nav resolver
        # (silent footgun). They are not canonical WebHarbor schema names, so the
        # WebHarbor extra-tool resolver rejects them at the config boundary.
        for tool in ("click_elem", "fill"):
            with pytest.raises(ValueError, match="unknown extra_tools"):
                _make_env(extra_tools=[tool])

    def test_som_tools_together(self):
        env = _make_env(extra_tools=["click", "input", "scroll", "navigate", "go_back", "response"])
        names = {tool_schema_name(t) for t in env.metadata.extra_tool_schemas}
        assert names == {"click", "input", "scroll", "navigate", "go_back", "response"}


def test_som_last_turn_prompt_names_response_text_not_answer():
    server_py = _REPO_ROOT / "lite/gym/envs/webharbor/webvoyager/docker/server.py"
    text = server_py.read_text(encoding="utf-8")
    assert "call response(text) NOW with what you found" in text
    assert "call answer() NOW" not in text


def test_container_wait_duration_uses_bounded_local_mirror():
    server_py = _REPO_ROOT / "lite/gym/envs/webharbor/webvoyager/docker/server.py"
    text = server_py.read_text(encoding="utf-8")
    assert "def _coerce_model_duration(" in text
    assert "_MODEL_DURATION_CAPS_SECONDS" in text
    assert 'label=f"{action_name}.{field}"' in text
    assert 'time.sleep(float(args.get("duration", 1.0) or 1.0))' not in text
    assert "post_action_delay = _coerce_bounded_seconds(" in text


def test_container_duration_caps_match_host_table_by_value():
    """The in-container copy must agree with the host table on VALUES, not just
    on the identifier's spelling.

    The container script may not import ``lite.gym`` (it runs inside the image),
    so the duplication is forced -- but the test runs on the host and can read
    both. The sibling above asserts only that ``_MODEL_DURATION_CAPS_SECONDS``
    appears in the source text, which stays green while a cap silently drifts.
    """
    from lite.core.tools.action_space.duration import (
        ACTION_SCHEMA_DURATION_CAPS_SECONDS,
    )
    from lite.gym.envs.webharbor.webvoyager.docker import server
    from lite.gym.utils.backend.model_inputs import DEFAULT_MODEL_DURATION_CAP_SECONDS

    assert server._MODEL_DURATION_CAPS_SECONDS == dict(ACTION_SCHEMA_DURATION_CAPS_SECONDS)
    assert server._DEFAULT_MODEL_DURATION_CAP_SECONDS == DEFAULT_MODEL_DURATION_CAP_SECONDS


def test_container_norm_coord_rejects_nonfinite_values():
    import lite.gym.envs.webharbor.webvoyager.docker.server as server

    class _Driver:
        def execute_script(self, _script: str):
            return [1280, 720]

    with pytest.raises(ValueError, match="finite numbers"):
        server._norm_coord_to_viewport(_Driver(), ["Infinity", 500])
    with pytest.raises(ValueError, match="finite numbers"):
        server._norm_coord_to_viewport(_Driver(), [500, "NaN"])


def test_container_key_paths_validate_list_shape_before_selenium_projection():
    server_py = _REPO_ROOT / "lite/gym/envs/webharbor/webvoyager/docker/server.py"
    text = server_py.read_text(encoding="utf-8")
    assert "def _model_key_list(" in text
    assert "must be a list of strings, not a string" in text
    assert '_model_key_list(args.get("keys", []), action_name=name)' in text


def test_container_step_rejects_non_list_body_actions_as_infra_not_model():
    """A non-list ``body.actions`` is an INFRA fault and must not be model-blamed.

    It used to be recorded as an ``action_errors`` entry with
    ``kind="model_action"`` and returned **200**, so the host reported "your
    arguments were invalid" for a body the host client serialized
    (``main.py`` builds ``actions_to_send: list[EnvAction]``) and nothing
    retried. ``ActionErrorKind`` has no infra member and adding a fourth would
    extend a model-blame vocabulary to cover an infra fault, so the shape is
    unrepresentable instead: ``StepBody`` types the field and FastAPI answers 422.
    """
    from fastapi.testclient import TestClient

    from lite.gym.envs.webharbor.webvoyager.docker import server

    client = TestClient(server.app)
    r = client.post("/step", json={"instance_id": "i1", "actions": "click", "step_count": 1})
    assert r.status_code == 422, r.text
    assert "actions" in r.text
    # No model-blame vocabulary anywhere in the response.
    assert "model_action" not in r.text

    # A LIST still reaches the handler (this instance_id is unknown -> 404), so
    # the 422 above is about the type and not about rejecting every /step.
    r = client.post("/step", json={"instance_id": "i1", "actions": [], "step_count": 1})
    assert r.status_code == 404, r.text


def test_container_unknown_action_is_executed_warning_not_action_error(monkeypatch, tmp_path):
    from lite.gym.envs.webharbor.webvoyager.docker import server

    class _Driver:
        current_url = "https://example.com/"
        title = "Example"

        def get_screenshot_as_png(self):
            return base64.b64decode(_PNG_B64)

        def execute_script(self, _script, *args):
            if "window.innerWidth" in _script:
                return [1280, 720]
            return None

    monkeypatch.setattr(
        server,
        "_screenshot_observation",
        lambda _inst, *, cursor=True: {
            "screenshot_b64": _PNG_B64,
            "web_text": "",
            "url": "https://example.com/",
            "title": "Example",
        },
    )
    iid = "iid-unsupported-action"
    inst = server._Instance(
        driver=_Driver(),
        task_id="allrecipes.0",
        instruction=_SAMPLE_TASK["instruction"],
        start_url="https://example.com/",
        download_dir=tmp_path,
        max_steps=5,
    )
    monkeypatch.setitem(server._instances, iid, inst)

    resp = server._step_sync(
        {
            "instance_id": iid,
            "post_action_delay": 0,
            "step_count": 1,
            "actions": [{"name": "cursor_position", "call_id": "unsupported", "arguments": {}}],
        }
    )

    assert resp["executed"] == [
        {
            "call": "cursor_position",
            "args": {},
            "warning": "unknown action cursor_position",
        }
    ]
    assert resp["errors"] == []
    assert resp["action_errors"] == []


def test_container_action_errors_echo_call_id_for_older_hosts():
    from lite.gym.envs.webharbor.webvoyager.docker import server

    action_errors: list[dict] = []
    server._append_action_error(
        action_errors,
        index=0,
        kind="model_action",
        name="click",
        error=ValueError("bad coordinate"),
        action={
            "call_id": "internal-call-id",
            "name": "click",
            "arguments": {"coordinate": ["bad"]},
        },
    )

    assert action_errors == [
        {
            "index": 0,
            "name": "click",
            "error": "bad coordinate",
            "message": "click: bad coordinate",
            "kind": "model_action",
            "call_id": "internal-call-id",
            "action": {
                "call_id": "internal-call-id",
                "name": "click",
                "arguments": {"coordinate": ["bad"]},
            },
        }
    ]


def test_container_drag_without_start_uses_tracked_cursor_not_end_point():
    server_py = _REPO_ROOT / "lite/gym/envs/webharbor/webvoyager/docker/server.py"
    text = server_py.read_text(encoding="utf-8")

    assert 'raise ValueError("drag requires start_coordinate or a tracked cursor")' in text
    assert "start = inst.last_cursor" in text
    assert "start = end" not in text
    assert "no persistent pixel cursor" not in text


def test_install_health_timeout_matches_runtime_readiness_timeout():
    main_py = _REPO_ROOT / "lite/gym/envs/webharbor/webvoyager/main.py"
    install_sh = _REPO_ROOT / "lite/gym/envs/webharbor/webvoyager/scripts/install.sh"
    assert "time.monotonic() + 180" in main_py.read_text(encoding="utf-8")
    assert "SECONDS + 180" in install_sh.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Reset body sent to server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_sends_correct_body():
    env = _make_env(max_steps=12, cursor=False)
    captured: list[dict] = []

    def _capture(path: str, body: dict) -> dict:
        if path == "/reset":
            captured.append(body)
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)
    await env.reset()

    assert len(captured) == 1
    body = captured[0]
    assert body["task_id"] == "allrecipes.0"
    assert body["instruction"] == _SAMPLE_TASK["instruction"]
    assert "web" not in body
    assert "url" not in body
    assert body["start_url"] == "https://www.allrecipes.com/"
    assert body["max_steps"] == 12
    assert body["cursor"] is False
    await env.close()


@pytest.mark.asyncio
async def test_step_sends_instance_id():
    env = _make_env(cursor=False)
    captured: list[dict] = []

    def _capture(path: str, body: dict) -> dict:
        if path == "/step":
            captured.append(body)
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)
    await env.reset()
    await env.step([_call("click", {"coordinate": [100, 100]})])
    await env.close()

    assert captured[0]["instance_id"] == "iid-abc123"
    assert captured[0]["cursor"] is False


@pytest.mark.asyncio
async def test_valid_actions_rejection_keeps_current_observation_feedback():
    env = RemoteWebVoyagerEnv(
        task_id="allrecipes.0",
        task=_SAMPLE_TASK,
        base_max_steps=15,
        max_steps=10,
        extra_tools=["click", "input", "scroll", "go_back", "response"],
        valid_actions=[],
        skip_eval=True,
        include_page_context_text=True,
    )
    paths: list[str] = []

    def _capture(path: str, body: dict) -> dict:
        paths.append(path)
        return _route(path, body)

    env._post = MagicMock(side_effect=_capture)

    await env.reset()
    result = await env.step(
        [
            _call(
                "computer",
                {"actions": [{"action": "type", "text": "sentiment analysis"}]},
                call_id="call_0",
            ),
        ]
    )
    await env.close()

    assert paths == ["/reset", "/close"]
    assert len(result.results) == 1
    tool_result = result.results[0]
    assert tool_result.tool_call_id == "call_0"
    assert tool_result.images[-1] == _PNG
    assert "CURRENT URL: https://www.allrecipes.com/" in (tool_result.text or "")
    assert 'DOM:\n[0]: <a> "Recipes"' in (tool_result.text or "")
    assert tool_result.error == ("invalid action: type; choose an available action for this task")
    assert tool_result.metadata == {
        "url": "https://www.allrecipes.com/",
        "title": "Allrecipes",
        "web_text": _RESET_WEB_TEXT,
        "model_web_text": _RESET_SOM_MODEL_WEB_TEXT,
        "downloads": [],
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_valid_actions_rejection_keeps_separate_error_when_page_text_empty():
    env = RemoteWebVoyagerEnv(
        task_id="allrecipes.0",
        task=_SAMPLE_TASK,
        base_max_steps=15,
        max_steps=10,
        extra_tools=["click", "input", "scroll", "go_back", "response"],
        valid_actions=[],
        skip_eval=True,
        include_page_context_text=True,
    )

    def _blank_page(path: str, body: dict) -> dict:
        if path == "/reset":
            resp = _reset_resp()
            resp.update({"web_text": "", "url": "", "title": ""})
            return resp
        if path == "/step":
            resp = _step_resp()
            resp.update({"web_text": "", "url": "", "title": ""})
            return resp
        return _route(path, body)

    env._post = MagicMock(side_effect=_blank_page)

    await env.reset()
    result = await env.step(
        [
            _call(
                "computer",
                {"actions": [{"action": "type", "text": "sentiment analysis"}]},
                call_id="call_0",
            ),
        ]
    )
    await env.close()

    tool_result = result.results[0]
    assert tool_result.images[-1] == _PNG
    assert tool_result.text == "Current page context is unchanged."
    assert tool_result.error == ("invalid action: type; choose an available action for this task")
    assert tool_result.metadata == {
        "url": "",
        "title": "",
        "web_text": "",
        "downloads": [],
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_valid_actions_rejection_requires_active_instance():
    env = RemoteWebVoyagerEnv(
        task_id="allrecipes.0",
        task=_SAMPLE_TASK,
        base_max_steps=15,
        max_steps=10,
        extra_tools=["click", "input", "scroll", "go_back", "response"],
        valid_actions=[],
        skip_eval=True,
    )
    invalid_action = [
        _call(
            "computer",
            {"actions": [{"action": "type", "text": "sentiment analysis"}]},
            call_id="call_0",
        )
    ]

    with pytest.raises(RuntimeError, match="step\\(\\) called before reset"):
        await env.step(invalid_action)

    env._post = MagicMock(side_effect=lambda path, body: _route(path, body))
    await env.reset()
    await env.close()
    with pytest.raises(RuntimeError, match="step\\(\\) called before reset"):
        await env.step(invalid_action)


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_tasks_registered_in_registry(self):
        import lite.gym as gym

        ids = gym.registry.task_ids("webharbor.webvoyager")
        if isinstance(ids, dict):
            ids = [tid for lst in ids.values() for tid in lst]
        assert len(ids) > 0, "expected webharbor.webvoyager tasks to be registered"

    def test_task_ids_have_correct_format(self):
        import lite.gym as gym

        ids = gym.registry.task_ids("webharbor.webvoyager")
        if isinstance(ids, dict):
            ids = [tid for lst in ids.values() for tid in lst]
        # "{site}.{index}" — no redundant "webvoyager." prefix (the env_id carries it)
        assert ids and all(
            "." in tid and not tid.startswith("webvoyager.") and tid.rsplit(".", 1)[1].isdigit()
            for tid in ids
        )

    def test_known_task_present(self):
        import lite.gym as gym

        ids = gym.registry.task_ids("webharbor.webvoyager")
        if isinstance(ids, dict):
            ids = [tid for lst in ids.values() for tid in lst]
        assert "allrecipes.0" in ids

    def test_task_metadata_platform(self):
        import lite.gym as gym

        ids = gym.registry.task_ids("webharbor.webvoyager")
        if isinstance(ids, dict):
            ids = [tid for lst in ids.values() for tid in lst]
        meta = gym.registry.task_metadata("webharbor.webvoyager", ids[0])
        assert meta is not None
        assert meta.platform == "browser"
        assert meta.task_type == "use"

    def test_eval_split_tasks_present(self):
        import lite.gym as gym

        ids = gym.registry.task_ids("webharbor.webvoyager", split="eval")
        if isinstance(ids, dict):
            ids = [tid for lst in ids.values() for tid in lst]
        assert len(ids) > 0

    def test_all_registered_env_kwargs_are_wired(self):
        """Systemic guard: every env_kwargs key in default.yaml must be an
        EXPLICIT RemoteWebVoyagerEnv.__init__ param. The ``**_`` base-kwarg sink
        (which absorbs gym.make's resolution/post_action_delay) must NOT count as
        "wired" — otherwise a per-run override of an unwired key is silently
        swallowed and the guard is fake (T5)."""
        import inspect

        from lite.gym.envs.webharbor.webvoyager.main import CFG, RemoteWebVoyagerEnv

        sig = inspect.signature(RemoteWebVoyagerEnv.__init__)
        explicit = {
            name
            for name, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        registration_layer = {"step_timeout"}  # resolved by registry wrapper
        for key in CFG.env_kwargs:
            assert key in explicit or key in registration_layer, (
                f"env_kwarg '{key}' is neither an explicit RemoteWebVoyagerEnv.__init__ "
                f"param nor a known registration-layer key — a per-run override would be "
                f"silently swallowed by **_."
            )


# ---------------------------------------------------------------------------
# Live tests (require the cua-lite/webharbor.webvoyager container — `pytest -m live`)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestWebVoyagerLive:
    """Integration tests against a real in-container server.

    gym.make() brings the container up automatically via
    WebVoyagerContainerServices.ensure(). Build the image once first:
        bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh
    """

    def _make_live(
        self,
        max_steps: int = 5,
        *,
        extra_tools: list[str] | None = None,
        use_som: bool | None = None,
    ) -> RemoteWebVoyagerEnv:
        import lite.gym as gym

        ids = gym.registry.task_ids("webharbor.webvoyager", split="eval")
        if isinstance(ids, dict):
            ids = [tid for lst in ids.values() for tid in lst]
        kwargs = {}
        if use_som is not None:
            kwargs["use_som"] = use_som
        return gym.make(
            f"webharbor.webvoyager@{ids[0]}",
            max_steps=max_steps,
            extra_tools=extra_tools or [],
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_reset_returns_screenshot(self):
        env = self._make_live()
        try:
            obs = await env.reset()
            assert obs.image
            raw = obs.image
            assert raw[:4] == b"\x89PNG"
            assert obs.text  # instruction
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_click(self):
        env = self._make_live()
        try:
            await env.reset()
            r = await env.step(
                [
                    _call("click", {"coordinate": [500, 300]}),
                ]
            )
            assert r.results[0].images
            assert not r.terminated
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_type(self):
        env = self._make_live()
        try:
            await env.reset()
            r = await env.step(
                [
                    _call("click", {"coordinate": [500, 60]}),
                    _call("type", {"text": "vegetarian lasagna"}),
                ]
            )
            assert r.results[0].images
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_scroll(self):
        env = self._make_live()
        try:
            await env.reset()
            r = await env.step(
                [
                    _call(
                        "scroll",
                        {
                            "coordinate": [500, 400],
                            "direction": "down",
                            "amount": 3,
                        },
                    ),
                ]
            )
            assert r.results[0].images
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_key(self):
        env = self._make_live()
        try:
            await env.reset()
            r = await env.step(
                [
                    _call("key", {"keys": ["escape"]}),
                ]
            )
            assert r.results[0].images
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_response_terminates(self):
        env = self._make_live(extra_tools=["response"])
        try:
            await env.reset()
            r = await env.step(
                [
                    _call("response", {"text": "test answer"}),
                ]
            )
            assert r.terminated is True
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_web_text_enriched_with_coords(self):
        """web_text returned by the server includes @ (cx, cy) coordinates."""
        env = self._make_live(use_som=True)
        try:
            obs = await env.reset()
            wt = obs.metadata.get("web_text", "")
            if wt:
                assert "@ (" in wt, f"expected coord annotations in web_text; got: {wt[:200]}"
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_truncation_at_max_steps(self):
        env = self._make_live(max_steps=2)
        try:
            await env.reset()
            r1 = await env.step(
                [
                    _call("click", {"coordinate": [500, 300]}),
                ]
            )
            assert not r1.truncated
            r2 = await env.step(
                [
                    _call("click", {"coordinate": [500, 300]}),
                ]
            )
            assert r2.truncated is True
            assert r2.terminated is False
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        env = self._make_live()
        await env.reset()
        iid = env._instance_id
        assert iid is not None
        await env.close()
        assert env._instance_id is None

    @pytest.mark.asyncio
    async def test_full_episode(self):
        env = self._make_live(max_steps=5, extra_tools=["response"])
        try:
            obs = await env.reset()
            assert obs.image

            r = await env.step(
                [
                    _call("click", {"coordinate": [500, 60]}),
                ]
            )
            assert not r.terminated

            r = await env.step(
                [
                    _call("type", {"text": "vegetarian"}),
                ]
            )
            assert not r.terminated

            r = await env.step(
                [
                    _call(
                        "scroll",
                        {
                            "coordinate": [500, 400],
                            "direction": "down",
                            "amount": 2,
                        },
                    ),
                ]
            )
            assert not r.terminated

            r = await env.step(
                [
                    _call("response", {"text": "found it"}),
                ]
            )
            assert r.terminated is True
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_concurrent_resets(self):
        """Multiple envs can reset concurrently (pool permits it)."""
        import lite.gym as gym

        ids = gym.registry.task_ids("webharbor.webvoyager", split="eval")
        if isinstance(ids, dict):
            ids = [tid for lst in ids.values() for tid in lst]

        async def _run(tid: str) -> bool:
            env = gym.make(f"webharbor.webvoyager@{tid}", max_steps=2)
            try:
                obs = await env.reset()
                assert obs.image
                await env.close()
                return True
            except Exception:
                await env.close()
                raise

        results = await asyncio.gather(
            *[_run(ids[i % len(ids)]) for i in range(3)],
            return_exceptions=True,
        )
        successes = sum(1 for r in results if r is True)
        assert successes >= 2, f"expected >=2 concurrent resets to succeed; got {results}"


# ---------------------------------------------------------------------------
# Judge infra-error propagation (B2 / T2 — webharbor.webvoyager half)
#
# T2 flags webharbor.webvoyager's "errors propagate" guarantee as untested. The
# judge runs in-process (host side), so we can drive _vlm_evaluate_sync with an
# unroutable base_url and assert it RAISES (so the rollout ERRs + retries) rather
# than coalescing an infra error into reward=0.0.
# ---------------------------------------------------------------------------


def test_vlm_judge_propagates_infra_error_not_zero_reward(monkeypatch):
    # WEBHARBOR_WEBVOYAGER_EVAL_BASE_URL wins in _build_eval_config, so the judge
    # deterministically targets an unroutable endpoint regardless of ambient
    # OPENAI_BASE_URL.
    monkeypatch.setenv("WEBHARBOR_WEBVOYAGER_EVAL_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-bad-test-key")
    env = _make_env(skip_eval=False, eval_config={"model": "gpt-4o"})
    env._agent_response = "the final answer"
    env._screenshots = [_PNG]
    with pytest.raises(Exception):  # APIConnectionError — NOT swallowed into (0.0, ...)
        env._vlm_evaluate_sync()


@pytest.mark.asyncio
async def test_evaluate_skip_and_no_answer_return_none():
    # The two legitimate None paths (must NOT be 0.0): skip_eval, and no answer.
    env = _make_env(skip_eval=True)
    reward, _info = await env._evaluate()
    assert reward is None

    env2 = _make_env(skip_eval=False)
    env2._agent_response = None
    reward2, info2 = await env2._evaluate()
    assert reward2 is None and info2.get("eval_skip_reason") == "no_agent_response"


def test_container_reset_seeds_cursor_at_viewport_centre(monkeypatch, tmp_path):
    """A fresh instance starts with the pointer at the viewport CENTRE.

    Regression: ``last_cursor`` used to stay ``None`` until the first pointer
    action, so ``_show_capture_cursor`` returned ``False`` and the turn-0 frame
    carried NO cursor — structurally unlike every later frame, and unlike the
    retired ``CursorOverlayWrapper`` which seeded ``(500, 500)`` normalized
    (= centre) on ``reset()``.

    Drives the real ``_reset_sync`` with Selenium stubbed out, so the assertion
    covers the shipped construction site rather than a re-derived constant.
    """
    import lite.gym.envs.webharbor.webvoyager.docker.server as server

    class _Driver:
        def __init__(self, size):
            self._size = size

        def execute_script(self, _script, *_args):
            return list(self._size)

        def quit(self):
            pass

    size = [1280, 720]

    monkeypatch.setattr(server, "_new_driver", lambda _dir, _w, _h: _Driver(size))
    monkeypatch.setattr(server, "_initial_page_setup", lambda _driver, _url: None)
    monkeypatch.setattr(
        server,
        "_screenshot_observation",
        lambda _inst, *, cursor=True: {
            "screenshot_b64": "",
            "web_text": "",
            "url": "",
            "title": "",
        },
    )
    monkeypatch.setattr(server, "_BASE_DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(server, "_instances", {})

    out = server._reset_sync({"viewport": [1280, 720]})

    assert server._instances[out["instance_id"]].last_cursor == (640, 360)

    size[:] = [1920, 1080]
    wide = server._reset_sync({"viewport": [1920, 1080]})

    assert server._instances[wide["instance_id"]].last_cursor == (960, 540)


# ---------------------------------------------------------------------------
# Per-action result frames
#
# webharbor.webvoyager POSTs the WHOLE batch to the container in one ``/step``,
# so the action loop -- and therefore the only place that can see the page
# BETWEEN two actions of one batch -- lives in ``docker/server.py``. The
# container returns ``screenshots_b64`` (one frame per executed action, in
# action order) and the host turns it into the per-action ``images`` of the
# tool result.
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


def test_action_batch_result_carries_one_image_per_executed_action():
    """N executed actions -> N DISTINCT frames on the batch's single result.

    Distinctness is the assertion that matters: re-emitting one frame N times
    would satisfy a length check while carrying no information at all.
    """
    frames = _distinct_frames("wv-frame", 3)
    env = _make_env()

    def _routed(path: str, body: dict) -> dict:
        if path == "/step":
            assert [action["name"] for action in body["actions"]] == ["click"] * 3
            return {
                **_step_resp(),
                "screenshots_b64": frames,
                "screenshot_b64": frames[-1],
            }
        return _route(path, body)

    env._post = MagicMock(side_effect=_routed)
    asyncio.run(env.reset())
    r = asyncio.run(env.step([_batch_call("call-batch", 3)]))

    assert len(r.results) == 1
    images = list(r.results[0].images)
    assert images == [base64.b64decode(frame) for frame in frames]
    assert len(set(images)) == 3


def test_container_step_returns_one_frame_per_executed_action(monkeypatch, tmp_path):
    """The container-side half: the frame count follows the ACTION count.

    The post-loop observation SUPERSEDES the last per-action frame rather than
    extending the list -- it is the same action's state, re-taken after the
    settle delay and carrying the SoM overlay + web_text. A batch that executed
    nothing still owes exactly one observation.
    """
    from lite.gym.envs.webharbor.webvoyager.docker import server

    frames = iter([b"wv-1", b"wv-2", b"wv-3"])

    class _Driver:
        current_url = "https://www.allrecipes.com/"
        title = "Allrecipes"

        def get_screenshot_as_png(self):
            return next(frames)

        def execute_script(self, script, *_args):
            if "innerWidth" in script or "innerHeight" in script:
                return [1280, 720]
            return None

    monkeypatch.setattr(
        server,
        "_screenshot_observation",
        lambda _inst, *, cursor=True: {
            "screenshot_b64": base64.b64encode(b"wv-observation").decode("ascii"),
            "web_text": "",
            "url": "https://www.allrecipes.com/",
            "title": "Allrecipes",
        },
    )
    iid = "iid-per-action-frames"
    inst = server._Instance(
        driver=_Driver(),
        task_id="allrecipes.0",
        instruction=_SAMPLE_TASK["instruction"],
        start_url="https://www.allrecipes.com/",
        download_dir=tmp_path,
        max_steps=5,
    )
    monkeypatch.setitem(server._instances, iid, inst)

    resp = server._step_sync(
        {
            "instance_id": iid,
            "post_action_delay": 0,
            "step_count": 1,
            "cursor": False,
            "actions": [
                {"name": "screenshot", "call_id": "c1", "arguments": {}},
                {"name": "screenshot", "call_id": "c2", "arguments": {}},
                {"name": "screenshot", "call_id": "c3", "arguments": {}},
            ],
        }
    )
    assert [base64.b64decode(f) for f in resp["screenshots_b64"]] == [
        b"wv-1",
        b"wv-2",
        b"wv-observation",
    ]

    resp = server._step_sync(
        {
            "instance_id": iid,
            "post_action_delay": 0,
            "step_count": 2,
            "cursor": False,
            "actions": [],
        }
    )
    assert [base64.b64decode(f) for f in resp["screenshots_b64"]] == [b"wv-observation"]


def test_a_terminating_batch_keeps_the_previous_actions_frame(monkeypatch, tmp_path):
    """``terminate`` breaks the loop owing a frame; the observation EXTENDS.

    ``terminate`` executes but never drives the page, so unlike a normal action
    it appends no frame of its own. If the post-loop observation superseded here
    it would overwrite the PRECEDING action's frame, and a two-action batch would
    report one frame -- losing a frame and miscounting executed actions at once.
    """
    from lite.gym.envs.webharbor.webvoyager.docker import server

    frames = iter([b"wv-1", b"wv-2"])

    class _Driver:
        current_url = "https://www.allrecipes.com/"
        title = "Allrecipes"

        def get_screenshot_as_png(self):
            return next(frames)

        def execute_script(self, script, *_args):
            if "innerWidth" in script or "innerHeight" in script:
                return [1280, 720]
            return None

    monkeypatch.setattr(
        server,
        "_screenshot_observation",
        lambda _inst, *, cursor=True: {
            "screenshot_b64": base64.b64encode(b"wv-observation").decode("ascii"),
            "web_text": "",
            "url": "https://www.allrecipes.com/",
            "title": "Allrecipes",
        },
    )
    iid = "iid-terminating-batch"
    inst = server._Instance(
        driver=_Driver(),
        task_id="allrecipes.0",
        instruction=_SAMPLE_TASK["instruction"],
        start_url="https://www.allrecipes.com/",
        download_dir=tmp_path,
        max_steps=5,
    )
    monkeypatch.setitem(server._instances, iid, inst)

    resp = server._step_sync(
        {
            "instance_id": iid,
            "post_action_delay": 0,
            "step_count": 1,
            "cursor": False,
            "actions": [
                {"name": "screenshot", "call_id": "c1", "arguments": {}},
                {"name": "terminate", "call_id": "c2", "arguments": {}},
            ],
        }
    )

    assert [c["call"] for c in resp["executed"]] == ["screenshot", "terminate"]
    assert [base64.b64decode(f) for f in resp["screenshots_b64"]] == [
        b"wv-1",
        b"wv-observation",
    ]
    assert resp["terminated"] is True


def test_container_type_without_press_enter_types_only(monkeypatch):
    """Absent ``press_enter`` means TYPE ONLY, on every env that reads it.

    The container flag is host-derived: canonical spells Enter as a trailing
    newline, and the env strips it onto ``press_enter``. Absent means the model
    did not ask to submit, and the error is asymmetric: a missing Enter costs
    one turn, a spurious Enter
    irreversibly submits a form or navigates away. ``fill`` is the DOM-index
    extra, a DIFFERENT action, and already agreed.
    """
    from lite.gym.envs.webharbor.webvoyager.docker import server

    seen: list[bool] = []

    def _typed(_inst, _element, _text, *, press_enter: bool) -> str:
        seen.append(press_enter)
        return ""

    monkeypatch.setattr(server, "_exec_action_type", _typed)
    monkeypatch.setattr(server, "_active_or_last_element", lambda _inst: object())
    inst = SimpleNamespace(driver=None, web_eles=[object()], last_element=None)

    server._execute_action(inst, "type", {"text": "query"})
    server._execute_action(inst, "type", {"text": "query", "press_enter": True})
    server._execute_action(inst, "fill", {"text": "query", "index": 0})

    assert seen == [False, True, False]


def test_container_type_with_nothing_focused_matches_the_focused_branch(monkeypatch):
    """The unfocused branch is the FIRST type of every episode, not a corner.

    ``_initial_page_setup`` clicks ``<body>`` at reset, so ``_active_or_last_element``
    returns None until the model clicks something. That branch bypasses
    ``_exec_action_type`` entirely, so it has to reproduce all three of its
    behaviours itself: honour ``press_enter`` (the host already stripped the
    canonical trailing newline into that flag), reinstall the spacebar guard a
    navigation wipes, and settle as long as a submit needs.
    """
    from lite.gym.envs.webharbor.webvoyager.docker import server

    sent: list = []
    performed: list = []
    guarded: list = []
    slept: list = []

    class _Chain:
        def __init__(self, _driver):
            pass

        def send_keys(self, value):
            sent.append(value)
            return self

        def pause(self, _seconds):
            return self

        def perform(self):
            performed.append(tuple(sent))

    monkeypatch.setattr(server, "ActionChains", _Chain)
    monkeypatch.setattr(server, "_active_or_last_element", lambda _inst: None)
    monkeypatch.setattr(server, "_install_spacebar_guard", lambda _d: guarded.append(True))
    monkeypatch.setattr(server.time, "sleep", lambda s: slept.append(s))
    inst = SimpleNamespace(driver=None, web_eles=[], last_element=None)

    server._execute_action(inst, "type", {"text": "query", "press_enter": True})
    assert sent == ["query", server.Keys.ENTER]
    assert performed == [("query", server.Keys.ENTER)], "the chain must be dispatched"
    assert guarded == [True], "a navigation wipes the reset-time guard; reinstall it"
    assert slept == [10], "a submit navigates and needs the long settle"

    sent.clear()
    performed.clear()
    guarded.clear()
    slept.clear()
    server._execute_action(inst, "type", {"text": "query"})
    assert sent == ["query"], "a plain type must not submit"
    assert performed == [("query",)]
    assert slept == [3], "a plain type uses the short settle"


def test_container_goto_requires_canonical_url_argument():
    from lite.gym.envs.webharbor.webvoyager.docker import server

    inst = SimpleNamespace(driver=None, web_eles=[], last_element=None)

    for args in ({}, {"web": "https://example.com/"}, {"page": "https://example.com/"}):
        with pytest.raises(ValueError, match="requires url"):
            server._execute_action(inst, "goto", args)


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
    """``type`` is a CANONICAL action here (``valid_actions`` in the env config
    lists it), and the container types through Selenium ``send_keys``, where a
    newline is an ordinary character rather than the Return key. Without this
    projection a submitting ``type`` types its text and never submits -- and the
    container's longer post-navigation settle never runs either."""
    out = _to_container_action({"name": "type", "arguments": {"text": text}})
    assert out["arguments"]["text"] == value
    assert out["arguments"]["press_enter"] is submits


def test_to_container_action_leaves_the_canonical_action_untouched():
    """Error pairing and the recorded trajectory both read ``actions_to_send``,
    so the projection must not mutate it in place."""
    typed = {"name": "type", "arguments": {"text": "q\n"}}
    _to_container_action(typed)
    assert typed["arguments"] == {"text": "q\n"}

    scroll = {"name": "scroll", "arguments": {"direction": "down", "amount": 3}}
    assert _to_container_action(scroll) is scroll
