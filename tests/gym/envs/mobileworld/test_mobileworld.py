"""Tests for the mobileworld CUA-Lite gym environment.

Offline only — registry contents, bind contract, action translation, and
extra-tool resolution. No docker container is booted (the live episode path
is covered by the env-server live suite / manual smoke).

Run:
    uv run pytest tests/gym/envs/mobileworld/test_mobileworld.py -v
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

import lite.gym as gym
import lite.gym.envs.mobileworld.main as M
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters
from lite.core.utils.filters import parse_filter
from lite.gym.envs.mobileworld.main import _TASKS, MobileWorldEnv
from lite.infer.rollout import collect_tasks


def _tool_schema(env: MobileWorldEnv, name: str) -> dict:
    return next(s for s in env.metadata.extra_tool_schemas if tool_schema_name(s) == name)


def _mobile_action_schema_enum(metadata) -> list[str]:
    from lite.agents.core.action_space.base import LiteMobileActionSpace

    schemas = LiteMobileActionSpace.get_tool_schemas()
    if metadata.valid_actions is not None:
        schemas = LiteMobileActionSpace.filter_child_action_enum(
            schemas, metadata.valid_actions,
        )
    mobile = next(s for s in schemas if tool_schema_name(s) == "mobile")
    action_schema = tool_schema_parameters(mobile)["properties"]["actions"]["items"]
    return action_schema["properties"]["action"]["enum"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_task_registration_counts():
    """All non-MCP tasks register in the eval split; MCP tasks are excluded."""
    ids = gym.registry.task_ids("mobileworld")
    n_mcp = sum(1 for m in _TASKS.values() if "agent-mcp" in m["tags"])
    assert set(ids) == {"eval"}
    assert len(ids["eval"]) == len(_TASKS) - n_mcp
    assert not any(
        "agent-mcp" in _TASKS[t]["tags"] for t in ids["eval"]
    )


def test_interaction_tasks_have_exclude_reason_in_registered_and_live_metadata():
    ids = gym.registry.task_ids("mobileworld", split="eval")
    excluded = []
    for task_id in ids:
        expected = "ask_user" if "agent-user-interaction" in _TASKS[task_id]["tags"] else None
        registered = gym.registry.task_metadata("mobileworld", task_id)
        live = MobileWorldEnv(task_id=task_id, extra_tools=["ask_user"]).metadata
        assert registered.others.get("exclude_reason") == expected
        assert live.others.get("exclude_reason") == expected
        if expected:
            excluded.append(task_id)
    assert len(ids) == 161
    assert len(excluded) == 44


def test_eval_runner_filter_selects_all_gui_only_tasks():
    runner = Path(__file__).resolve().parents[4] / "devs/exps/eval/mobileworld/run.sh"
    command = runner.read_text().split("exec uv run python scripts/rollout.py", 1)[1]
    args = shlex.split(command.replace("\\\n", ""))
    keep = parse_filter(args[args.index("--filter") + 1])
    _, selected = collect_tasks("mobileworld", splits=["eval"], filter_fn=keep)
    expected = {
        task_id for task_id, task in _TASKS.items()
        if not {"agent-mcp", "agent-user-interaction"}.intersection(task["tags"])
    }
    assert set(selected) == expected
    assert len(selected) == 117


def test_services_ensure_uses_dependency_preflight(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(M, "_ensure_services", lambda env_id: calls.append(env_id))

    svc = M.MobileWorldServices()
    svc.ensure("mobileworld")

    assert calls == ["mobileworld"]


def test_services_health_uses_cached_dependency_preflight(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(M, "_HEALTH_CHECK", lambda env_id: calls.append(env_id))

    svc = M.MobileWorldServices()
    svc.health("mobileworld")

    assert calls == ["mobileworld"]


def test_registered_metadata():
    # Construct directly (like the other offline tests) rather than via
    # ``gym.make`` — the latter trips the image-freshness gate, so the metadata
    # assertions would fail on any checkout without the built image.
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    m = env.metadata
    assert m.platform == "mobile"
    assert m.task_type == "use"
    assert m.others["task_name"] == "AcceptMeetingTask"
    assert m.others["task_apps"] == _TASKS["AcceptMeetingTask"]["apps"]
    assert m.extra_tool_schemas == []  # extra tools are opt-in


@pytest.mark.parametrize("dedicated_credentials", [False, True])
def test_simulated_user_credentials_reach_container(monkeypatch, dedicated_credentials):
    import lite.gym.envs.mobileworld.container as C

    monkeypatch.setenv("OPENAI_API_KEY", "evaluated-model-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://evaluated.example/v1")
    monkeypatch.delenv("USER_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("USER_AGENT_BASE_URL", raising=False)
    if dedicated_credentials:
        monkeypatch.setenv("USER_AGENT_API_KEY", "simulated-user-key")
        monkeypatch.setenv("USER_AGENT_BASE_URL", "https://simulator.example/v1")

    captured = {}
    monkeypatch.setattr(C, "docker_run_detached", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(C, "_kvm_gid", lambda: None)
    container = C.MobileWorldContainer(name="test-simulated-user", api_port=10600)
    monkeypatch.setattr(container, "_register", lambda: None)
    monkeypatch.setattr(container, "_wait_until_ready", lambda: None)
    container.start()

    assert captured["env"] == {
        "USER_AGENT_API_KEY": (
            "simulated-user-key" if dedicated_credentials else "evaluated-model-key"
        ),
        "USER_AGENT_BASE_URL": (
            "https://simulator.example/v1"
            if dedicated_credentials else "https://evaluated.example/v1"
        ),
        "USER_AGENT_MODEL": C.CFG.server_kwargs["user_agent_model"],
    }
    assert "USER_AGENT_API_KEY" in captured["redact"]


# ---------------------------------------------------------------------------
# bind() contract
# ---------------------------------------------------------------------------

def test_bind_unknown_task_raises():
    with pytest.raises(ValueError, match="unknown mobileworld task_id"):
        MobileWorldEnv(task_id="NoSuchTask")


def test_bind_unknown_extra_tool_raises():
    with pytest.raises(ValueError, match="unknown extra_tools"):
        MobileWorldEnv(task_id="AcceptMeetingTask", extra_tools=["teleport"])


def test_extra_tools_opt_in():
    env = MobileWorldEnv(
        task_id="AcceptMeetingTask",
        extra_tools=["open_app", "ask_user", "response", "terminate"],
    )
    names = [tool_schema_name(t) for t in env.metadata.extra_tool_schemas]
    assert names == ["open_app", "ask_user", "response", "terminate"]


def test_open_app_catalog_metadata_matches_schema_enum():
    env = MobileWorldEnv(task_id="AcceptMeetingTask", extra_tools=["open_app"])

    assert env.metadata.others["apps"] == M._MOBILE_WORLD_APPS
    assert (
        tool_schema_parameters(_tool_schema(env, "open_app"))["properties"]["app_name"]["enum"]
        == env.metadata.others["apps"]
    )


def test_default_mobile_schema_does_not_advertise_pinch():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    enum = _mobile_action_schema_enum(env.metadata)
    assert "pinch" not in enum
    assert "system_button" in enum


def test_default_construction_metadata_shape_is_valid():
    env = MobileWorldEnv()
    # Identity is framework-owned (metadata contract): a directly-constructed env carries
    # no task_id; the default bind surfaces as an empty task_name.
    assert "task_id" not in env.metadata.others
    assert env.metadata.others["task_name"] == ""


# ---------------------------------------------------------------------------
# Action translation (no container: exercise the pure mapper)
# ---------------------------------------------------------------------------

@pytest.fixture()
def env() -> MobileWorldEnv:
    e = MobileWorldEnv(task_id="AcceptMeetingTask")
    e._screen_w, e._screen_h = 1080, 2400
    return e


def test_tap_translates_to_click_pixels(env):
    a = env._translate_action("tap", {"coordinate": [500, 500]})
    assert a == {"action_type": "click", "x": 540, "y": 1200}


def test_tap_preserves_no_clamp_max_edge(env):
    a = env._translate_action("tap", {"coordinate": [1000, 1000]})
    assert a == {"action_type": "click", "x": 1080, "y": 2400}


def test_double_tap_via_clicks(env):
    a = env._translate_action("tap", {"coordinate": [0, 0], "clicks": 2})
    assert a["action_type"] == "double_tap"


def test_bare_double_tap_is_not_an_action_name(env):
    """``double_tap`` is a JSONAction *action_type*, never a cua-lite action.

    The shared gate cannot reject it
    (``is_lite_action_name_or_action_batch_tool_name("double_tap")`` is False), so
    ``unsupported_env_action_message`` permits the name. Translation must decline
    it; the emit side above (``clicks>=2``) is the only place the string may appear.
    """
    from lite.gym.utils.feedback.surface import android_supported_actions

    assert "double_tap" not in android_supported_actions()
    assert env._translate_action("double_tap", {"coordinate": [500, 500]}) is None


def test_swipe_translates_to_precise_drag(env):
    a = env._translate_action(
        "swipe", {"start_coordinate": [500, 800], "coordinate": [500, 200]},
    )
    assert a == {
        "action_type": "drag",
        "start_x": 540, "start_y": 1920, "end_x": 540, "end_y": 480,
    }


def test_duration_fields_are_validate_only_for_upstream_json_actions(env):
    assert env._translate_action(
        "long_press",
        {"coordinate": [500, 500], "duration": 2.0},
    ) == {"action_type": "long_press", "x": 540, "y": 1200}

    swipe = env._translate_action(
        "swipe",
        {
            "start_coordinate": [500, 800],
            "coordinate": [500, 200],
            "duration": 1.25,
        },
    )
    assert swipe == {
        "action_type": "drag",
        "start_x": 540,
        "start_y": 1920,
        "end_x": 540,
        "end_y": 480,
    }
    assert "duration" not in swipe

    drag = env._translate_action(
        "drag",
        {
            "start_coordinate": [500, 800],
            "coordinate": [500, 200],
            "duration": 1.25,
        },
    )
    assert drag == swipe
    assert env._translate_action("wait", {"duration": 2.0}) == {
        "action_type": "wait"
    }


@pytest.mark.parametrize("button,expected", [
    ("Home", "navigate_home"),
    ("Back", "navigate_back"),
    ("Enter", "keyboard_enter"),
])
def test_system_buttons(env, button, expected):
    assert env._translate_action("system_button", {"button": button}) == {
        "action_type": expected,
    }


def test_unsupported_system_button_raises_model_error(env):
    with pytest.raises(ValueError, match="unknown button 'Bogus'"):
        env._translate_action("system_button", {"button": "Bogus"})


def test_type_and_open_app(env):
    assert env._translate_action("type", {"text": "hi"}) == {
        "action_type": "input_text", "text": "hi",
    }
    assert env._translate_action("open_app", {"app_name": "Settings"}) == {
        "action_type": "open_app", "app_name": "Settings",
    }


def test_screenshot_and_unknown_are_noops(env):
    assert env._translate_action("screenshot", {}) is None
    assert env._translate_action("frobnicate", {}) is None


@pytest.mark.asyncio
async def test_response_step_uses_canonical_tool_call():
    env = MobileWorldEnv(task_id="AcceptMeetingTask", extra_tools=["response"])
    env._post_action_delay = 0.0
    env._client = _StubClient()

    result = await env.step([
        make_tool_call("response", {"text": "done"}, call_id="call_response")
    ])

    assert env._client.calls[0] == ("step", {"action_type": "answer", "text": "done"})
    assert env._client.calls[-1] == ("eval", "AcceptMeetingTask")
    assert result.terminated is True
    assert result.info["executed_actions"] == [
        {"call": "json_action.answer", "args": {"text": "done"}}
    ]
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    assert result.results == []


@pytest.mark.asyncio
async def test_step_rejects_flat_lite_boundary_call():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._client = _StubClient()

    with pytest.raises(TypeError, match="env.step expects canonical Lite tool calls"):
        await env.step([
            {"name": "tap", "arguments": {"coordinate": [500, 500]}},
        ])


@pytest.mark.asyncio
async def test_content_only_final_text_uses_internal_response_no_tool_result():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._client = _StubClient()

    actions = make_no_tool_call_final_actions("final answer")
    result = await env.step(actions)

    assert tool_call_name(actions[0]) == "response"
    assert tool_call_arguments(actions[0]) == {"text": "final answer"}
    assert result.terminated is True
    assert result.results == []
    assert result.info["executed_actions"] == [
        {"call": "json_action.answer", "args": {"text": "final answer"}}
    ]


@pytest.mark.asyncio
async def test_response_side_effect_failure_returns_error_with_current_obs():
    class _FailingAnswerClient(_StubClient):
        def step(self, action):
            self.calls.append(("step", action))
            if action.get("action_type") == "answer":
                raise RuntimeError("answer write failed")
            return {"result": "ok"}

    env = MobileWorldEnv(task_id="AcceptMeetingTask", extra_tools=["response"])
    env._post_action_delay = 0.0
    env._client = _FailingAnswerClient()

    result = await env.step([
        make_tool_call(
            "response",
            {"text": "done"},
            call_id="call_response",
        )
    ])

    assert result.terminated is True
    assert result.results[0].tool_call_id == "call_response"
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].error == "response failed: execution failed"
    assert "answer write failed" not in result.results[0].error
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args,expected_note", [
    ("open_app", {"app_name": "Settings"}, None),
    ("ask_user", {"text": "what should I choose?"}, 'The user replied: "ok"'),
])
async def test_active_nonterminal_extra_tools_execute(name, args, expected_note):
    env = MobileWorldEnv(task_id="AcceptMeetingTask", extra_tools=[name])
    env._post_action_delay = 0.0
    env._client = _StubClient()

    result = await env.step([make_tool_call(name, args, call_id="call_1")])

    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_1"
    assert result.results[0].metadata != {"is_error": True}
    assert result.results[0].images[-1] == b"screenshot"
    assert result.results[0].text == expected_note
    assert env._client.calls[0][0] == "step"


@pytest.mark.asyncio
async def test_inactive_open_app_returns_error_only_feedback():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._client = _StubClient()

    result = await env.step([
        make_tool_call("open_app", {"app_name": "Settings"}, call_id="call_open")
    ])

    assert env._client.calls == []
    assert result.info["executed_actions"] == [{
        "call": "noop",
        "args": {"name": "open_app", "reason": "inactive extra tool"},
    }]
    assert result.results[0].tool_call_id == "call_open"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "open_app is not available in this task."
    assert result.results[0].images == []
    assert env._step_count == 1
    assert result.truncated is False


@pytest.mark.asyncio
async def test_invalid_open_app_is_rejected_before_dispatch():
    env = MobileWorldEnv(task_id="AcceptMeetingTask", extra_tools=["open_app"])
    env._post_action_delay = 0.0
    env._client = _StubClient()

    result = await env.step([
        make_tool_call(
            "open_app",
            {"app_name": "Not A Real App"},
            call_id="call_open",
        )
    ])

    assert env._client.calls == []
    assert result.info["executed_actions"] == []
    assert result.results[0].tool_call_id == "call_open"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error.startswith("invalid arguments for open_app: ")
    assert "Not A Real App" in result.results[0].error
    assert result.results[0].images[-1] == b"screenshot"
    assert env._step_count == 1
    assert result.truncated is False


@pytest.mark.asyncio
async def test_inactive_ask_user_returns_error_only_feedback():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._client = _StubClient()

    result = await env.step([
        make_tool_call("ask_user", {"text": "which one?"}, call_id="call_ask")
    ])

    assert env._client.calls == []
    assert result.info["executed_actions"] == [{
        "call": "noop",
        "args": {"name": "ask_user", "reason": "inactive extra tool"},
    }]
    assert result.results[0].tool_call_id == "call_ask"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "ask_user is not available in this task."
    assert result.results[0].images == []
    assert env._step_count == 1
    assert result.truncated is False


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments,expected", [
    ([], "tool_call.function.arguments must be an object, got list"),
    ({}, "ask_user.arguments.text is required"),
    ({"text": 123}, "ask_user.arguments.text must be a string"),
])
async def test_active_ask_user_wrong_shape_returns_paired_current_error(
    arguments, expected,
):
    env = MobileWorldEnv(task_id="AcceptMeetingTask", extra_tools=["ask_user"])
    env._post_action_delay = 0.0
    env._client = _StubClient()

    action = make_tool_call(
        "ask_user",
        arguments if isinstance(arguments, dict) else {},
        call_id="call_ask",
    )
    if not isinstance(arguments, dict):
        action["function"]["arguments"] = arguments

    result = await env.step([action])

    assert env._client.calls == []
    assert result.info["executed_actions"] == []
    assert result.results[0].tool_call_id == "call_ask"
    assert result.results[0].metadata == {"is_error": True}
    assert expected in result.results[0].error
    assert result.results[0].images[-1] == b"screenshot"
    assert env._step_count == 1
    assert result.truncated is False


@pytest.mark.asyncio
async def test_model_action_error_uses_canonical_error_text():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._screen_w, env._screen_h = 1080, 2400
    env._client = _StubClient()

    result = await env.step([
        make_tool_call(
            "tap",
            {"coordinate": [500, 500], "clicks": "two"},
            call_id="call_bad",
        )
    ])

    assert result.results[0].tool_call_id == "call_bad"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error.startswith("invalid arguments for tap: ")
    assert "'>='" in result.results[0].error


@pytest.mark.asyncio
async def test_execution_failure_returns_error_with_current_obs():
    class _FailingStepClient(_StubClient):
        def step(self, action):
            self.calls.append(("step", action))
            raise RuntimeError("backend rejected tap")

    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._screen_w, env._screen_h = 1080, 2400
    env._client = _FailingStepClient()

    result = await env.step([
        make_tool_call(
            "tap",
            {"coordinate": [500, 500]},
            call_id="call_tap",
        )
    ])

    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_tap"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "tap failed: execution failed"
    assert "backend rejected tap" not in result.results[0].error
    assert result.results[0].images[-1] == b"screenshot"


@pytest.mark.asyncio
async def test_malformed_tap_coordinate_returns_error_with_current_obs():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._screen_w, env._screen_h = 1080, 2400
    env._client = _StubClient()

    result = await env.step([
        make_tool_call("tap", {}, call_id="call_tap")
    ])

    assert env._client.calls == []
    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_tap"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == (
        "invalid arguments for tap: coordinate is required"
    )
    assert result.results[0].images[-1] == b"screenshot"


@pytest.mark.asyncio
async def test_unsupported_action_returns_error_only_feedback():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._client = _StubClient()

    result = await env.step([
        make_tool_call(
            "pinch",
            {"coordinate": [500, 500], "direction": "in", "amount": 25},
            call_id="call_pinch",
        )
    ])

    assert env._client.calls == []
    assert result.info["executed_actions"] == [{
        "call": "noop",
        "args": {"name": "pinch", "reason": "unsupported action: pinch"},
    }]
    assert result.results[0].tool_call_id == "call_pinch"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "unsupported action: pinch"
    assert result.results[0].images == []


@pytest.mark.asyncio
async def test_unknown_system_button_returns_model_error_with_current_obs():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._screen_w, env._screen_h = 1080, 2400
    env._client = _StubClient()

    result = await env.step([
        make_tool_call(
            "system_button",
            {"button": "Bogus"},
            call_id="call_bogus",
        )
    ])

    assert env._client.calls == []
    assert result.info["executed_actions"][0]["call"] == "noop"
    assert result.results[0].tool_call_id == "call_bogus"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error.startswith("invalid arguments for system_button: ")
    assert "unknown button 'Bogus'" in result.results[0].error
    assert result.results[0].images[-1] == b"screenshot"
    assert env._step_count == 1
    assert result.truncated is False


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args,expected", [
    (
        "long_press",
        {"coordinate": [500, 500], "duration": 0},
        "long_press.duration must be greater than 0",
    ),
    (
        "swipe",
        {
            "start_coordinate": [500, 700],
            "coordinate": [500, 300],
            "duration": "slow",
        },
        "swipe.duration must be a finite number",
    ),
    (
        "swipe",
        {
            "start_coordinate": [500, 700],
            "coordinate": [500, 300],
            "duration": float("inf"),
        },
        "swipe.duration must be finite",
    ),
    (
        "wait",
        {"duration": -1},
        "wait.duration must be non-negative",
    ),
    (
        "wait",
        {"duration": float("nan")},
        "wait.duration must be finite",
    ),
    (
        "wait",
        {"duration": 31},
        "wait.duration must be <= 30",
    ),
    (
        "long_press",
        {"coordinate": [500, 500], "duration": 6},
        "long_press.duration must be <= 5",
    ),
])
async def test_bad_duration_returns_model_error_with_current_obs(
    name, args, expected,
):
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._screen_w, env._screen_h = 1080, 2400
    env._client = _StubClient()

    result = await env.step([
        make_tool_call(name, args, call_id=f"call_{name}")
    ])

    assert result.info["executed_actions"][0]["call"] == "noop"
    assert result.results[0].tool_call_id == f"call_{name}"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == f"invalid arguments for {name}: {expected}"
    assert result.results[0].images[-1] == b"screenshot"
    assert ("step",) not in [call[:1] for call in env._client.calls]


@pytest.mark.asyncio
async def test_recent_system_button_executes_raw_keyevent(monkeypatch):
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._screen_w, env._screen_h = 1080, 2400
    env._client = _StubClient()

    def fake_keyevent(button: str) -> dict:
        return {
            "call": "adb.input.keyevent",
            "args": {"button": button, "keycode": 187},
        }

    monkeypatch.setattr(env, "_exec_system_button_adb", fake_keyevent)

    result = await env.step([
        make_tool_call(
            "system_button",
            {"button": "Recent"},
            call_id="call_recent",
        )
    ])

    assert result.info["executed_actions"][0] == {
        "call": "adb.input.keyevent",
        "args": {"button": "Recent", "keycode": 187},
    }
    assert result.results[0].tool_call_id == "call_recent"
    assert result.results[0].metadata != {"is_error": True}
    assert result.results[0].error is None
    assert result.results[0].images[-1] == b"screenshot"


@pytest.mark.asyncio
async def test_system_button_container_unavailable_is_sanitized():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._post_action_delay = 0.0
    env._screen_w, env._screen_h = 1080, 2400
    env._client = _StubClient()
    env._current_container = None

    result = await env.step([
        make_tool_call(
            "system_button",
            {"button": "Recent"},
            call_id="call_recent",
        )
    ])

    assert result.info["executed_actions"][0] == {
        "call": "noop",
        "args": {
            "name": "system_button",
            "reason": "mobileworld container is not available",
        },
    }
    assert result.results[0].tool_call_id == "call_recent"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "system_button failed: execution failed"
    assert "container" not in result.results[0].error
    assert result.results[0].images[-1] == b"screenshot"


@pytest.mark.asyncio
async def test_unknown_standalone_tool_returns_error_only():
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._max_steps = 1
    env._post_action_delay = 0.0
    env._client = _StubClient()

    result = await env.step([
        make_tool_call("foo", {}, call_id="call_foo")
    ])

    assert result.results[0].tool_call_id == "call_foo"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "unknown tool: foo"
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.info["executed_actions"] == [{
        "call": "noop",
        "args": {"name": "foo", "reason": "unknown tool"},
    }]
    assert env._step_count == 1
    assert result.truncated is True
    assert result.reward == 1.0
    assert env._client.calls == [("eval", "AcceptMeetingTask")]


# ── recycle cap: framework reset_with_recycle wiring ───────────────────────────────

class _StubClient:
    def __init__(self):
        self.calls = []

    def task_tear_down(self, task):
        self.calls.append(("tear_down", task))

    def task_init(self, task):
        self.calls.append(("init", task))

    def task_goal(self, task):
        return "goal"

    def task_eval(self, task):
        self.calls.append(("eval", task))
        return 1.0, "ok"

    def step(self, action):
        self.calls.append(("step", action))
        return {"result": "ok"}

    def screenshot_png(self):
        # Raw PNG bytes now: the client method was
        # renamed screenshot_b64()->screenshot_png() and returns bytes, not b64.
        return b"screenshot"


class _StubContainer:
    name = "mobileworld-test-stub"
    destroyed = 0

    def destroy(self):
        _StubContainer.destroyed += 1


def _reset_once(env):
    import asyncio
    return asyncio.run(env.reset())


def test_recycle_cap_destroys_and_respawns_at_cap(monkeypatch):
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._max_resets_per_container = 2
    _StubContainer.destroyed = 0

    booted = []

    async def stub_boot():
        booted.append(1)
        env._current_container = _StubContainer()
        env._client = _StubClient()

    monkeypatch.setattr(env, "boot", stub_boot)

    # Warm container at the cap: reset must recycle (destroy → fresh boot).
    env._current_container = _StubContainer()
    env._client = _StubClient()
    env._recycle_first_done = True
    env._recycle_count = 2
    _reset_once(env)
    assert _StubContainer.destroyed == 1, "cap reached → old container destroyed"
    assert booted == [1], "recycle cold-boots a fresh container"
    assert env._recycle_count == 0, "fresh container's reuse budget restarts at 0"


def test_task_init_retry_restarts_reuse_budget(monkeypatch):
    """Audit item: init_task's wedged-device retry (destroy + fresh boot
    OUTSIDE reset_with_recycle) must restart _recycle_count at 0 — else the
    fresh container inherits the old budget and recycles up to cap early."""
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._max_resets_per_container = 30
    _StubContainer.destroyed = 0

    class _WedgedOnceClient(_StubClient):
        wedged = True
        def task_init(self, task):
            if _WedgedOnceClient.wedged:
                _WedgedOnceClient.wedged = False
                raise RuntimeError("device wedged")
            super().task_init(task)

    async def stub_boot():
        env._current_container = _StubContainer()
        env._client = _WedgedOnceClient()

    monkeypatch.setattr(env, "boot", stub_boot)
    env._current_container = _StubContainer()
    env._client = _WedgedOnceClient()
    env._recycle_first_done = True
    env._recycle_count = 17                 # mid-life backend
    _reset_once(env)
    assert _StubContainer.destroyed == 1, "wedged /task/init → destroy + respawn"
    assert env._recycle_count == 0, (
        "retry-path fresh boot must restart the reuse budget at 0"
    )


def test_first_reset_never_recycles_a_fresh_backend():
    """A freshly booted env (_recycle_first_done=False) must not
    destroy its container on the first reset even at/above the cap."""
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._max_resets_per_container = 0
    _StubContainer.destroyed = 0

    env._current_container = _StubContainer()
    env._client = _StubClient()
    _reset_once(env)
    assert _StubContainer.destroyed == 0, "first reset on a fresh backend must not recycle"
    assert env._recycle_first_done is True
    assert env._recycle_count == 1, "the first reset counts as one in-place reuse"


# ── one frame per executed action ───────────────────────────────────────────

class _DistinctFramesClient(_StubClient):
    """``screenshot_png`` returns a DIFFERENT frame on every call.

    A stub that returns one constant frame cannot tell "N real captures" from
    "one cached frame emitted N times", which is the failure this test exists
    to catch.
    """

    def __init__(self):
        super().__init__()
        self.shots = 0

    def screenshot_png(self):
        self.shots += 1
        return f"frame-{self.shots}".encode()


@pytest.mark.asyncio
async def test_action_batch_returns_one_distinct_frame_per_action():
    """An N-action batch returns N frames, one per action, in action order."""
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._max_steps = 10
    env._post_action_delay = 0.0
    client = _DistinctFramesClient()
    env._client = client

    result = await env.step([
        make_tool_call(
            "mobile",
            {"actions": [
                {"action": "tap", "coordinate": [10, 10]},
                {"action": "tap", "coordinate": [20, 20]},
                {"action": "tap", "coordinate": [30, 30]},
            ]},
            call_id="call_batch",
        ),
    ])

    images = result.results[0].images
    assert len(images) == 3, "one frame per executed action"
    assert images == [b"frame-1", b"frame-2", b"frame-3"], (
        "frames must be captured per action, in action order"
    )
    assert client.shots == 3, "no extra capture beyond the per-action ones"


@pytest.mark.asyncio
async def test_read_only_screenshot_action_still_gets_its_own_frame():
    """No exception list: ``screenshot`` executes and therefore owes a frame,
    so the frame count stays a pure function of how many actions ran."""
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._max_steps = 10
    env._post_action_delay = 0.0
    env._client = _DistinctFramesClient()

    result = await env.step([
        make_tool_call(
            "mobile",
            {"actions": [
                {"action": "tap", "coordinate": [10, 10]},
                {"action": "screenshot"},
            ]},
            call_id="call_batch",
        ),
    ])

    assert result.results[0].images == [b"frame-1", b"frame-2"]


@pytest.mark.asyncio
async def test_zero_executed_actions_still_returns_one_frame():
    """Nothing ran, but the turn still owes the model a current observation."""
    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    env._max_steps = 10
    env._post_action_delay = 0.0
    client = _DistinctFramesClient()
    env._client = client

    result = await env.step([
        make_tool_call("mobile", {"actions": []}, call_id="call_empty"),
    ])

    assert result.results[0].images == [b"frame-1"]
    assert client.shots == 1
