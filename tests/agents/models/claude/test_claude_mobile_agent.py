"""Characterization tests for ClaudeMobileUseAgent.

Covers mobile-only kwargs and the mobile path's critical differences from desktop:
  - No computer-use beta flag in anthropic-beta header
  - Only function tools in tool list (no computer_XXXX tool)
  - system_prompt_suffix default
  - API images use Claude's provider-safe resize unless an explicit target is requested

Run:
    uv run pytest tests/agents/models/claude/test_claude_mobile_agent.py -v
"""

from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agents._support.valid_actions_gating import RESPONSE_SCHEMA
from agents.models._support.provider_fakes import png_bytes
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from PIL import Image

from lite.agents.core.agent import AgentRegistry
from lite.agents.models.claude.action_space import ClaudeMobileActionSpace
from lite.agents.models.claude.agent import (
    CLAUDE_MOBILE_API_KWARGS_DEFAULTS,
    ClaudeMobileUseAgent,
)
from lite.agents.models.claude.utils.parse import parse_mobile_response_with_provenance
from lite.core import LiteCUAMetadata
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_schema
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.extra_tools import BASH_TOOL_NAME, LiteFinishToolSet, LiteShellToolSet
from lite.core.tools.results import LiteToolResult
from lite.core.tools.schemas import tool_schema_name
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

# The production bash schema, so env-side admission in these tests is the real one:
# it requires ``command``.
_BASH_SCHEMA = LiteShellToolSet.get_tool_schema(BASH_TOOL_NAME)
_EMPTY_SCHEMA = make_tool_schema(
    "empty",
    description="Accepts no arguments.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
)

PIXEL_6 = (1080, 2400)


def _colored_png_bytes(
    w: int = 1080,
    h: int = 2400,
    color: tuple[int, int, int] = (32, 64, 96),
) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _provider_tool_name(tool: dict[str, Any]) -> str | None:
    return tool.get("name") or (tool.get("function") or {}).get("name")


def _provider_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {name for tool in tools if (name := _provider_tool_name(tool))}


def _provider_tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    return tool.get("parameters") or (tool.get("function") or {}).get("parameters", {})


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _fake_mobile_response(tool_calls: list | None = None) -> Any:
    """Minimal liteLLM-shape mobile response."""
    msg = SimpleNamespace(
        content="" if tool_calls else "done",
        tool_calls=tool_calls or [],
        role="assistant",
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model_dump=lambda: {"choices": []},
    )


def _fake_mobile_text_response(text: str) -> Any:
    msg = SimpleNamespace(
        content=text,
        tool_calls=[],
        role="assistant",
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model_dump=lambda: {"choices": []},
    )


def _fake_mobile_tool_call(name: str, arguments: str | None, id_: str) -> Any:
    return ChatCompletionMessageToolCall(
        id=id_,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _fake_tap_tool_call(x: int, y: int, id_: str = "toolu_tap_1") -> Any:
    return _fake_mobile_tool_call(
        "tap",
        json.dumps({"coordinate": [x, y]}),
        id_=id_,
    )


def _sent_image_size(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """Size of the newest screenshot a request actually put on the wire."""
    urls = [
        block["image_url"]["url"]
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "image_url"
    ]
    return Image.open(io.BytesIO(base64.b64decode(urls[-1].split("base64,", 1)[1]))).size


def _centre_tap_then(follow_up: Any, *, provider_id: str = "toolu_tap_1"):
    """``litellm.acompletion`` side effect: tap the centre of the frame the
    request sent, then answer with ``follow_up``.

    Reading the frame off the wire keeps the model's coordinate space the one
    Claude was actually shown, instead of re-deriving Claude's resize here.
    """
    calls: list[int] = []

    def respond(*_args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) > 1:
            return follow_up
        width, height = _sent_image_size(kwargs["messages"])
        return _fake_mobile_response(
            tool_calls=[_fake_tap_tool_call(width // 2, height // 2, id_=provider_id)]
        )

    return respond


def _fake_env_action_name(action: dict[str, Any]) -> str | None:
    return tool_call_name(action)


def _fake_env_action_id(action: dict[str, Any]) -> str | None:
    return tool_call_id(action)


class _FakeMobileEnv:
    def __init__(self, terminate_after: int = 1):
        self.metadata = LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
            others={"resolution": [1080, 2400]},
        )
        self._shot = png_bytes(1080, 2400)
        self._step_count = 0
        self._terminate_after = terminate_after

    async def reset(self):
        return LiteEnvObservation(image=self._shot, text="open settings")

    async def step(self, actions):
        self._step_count += 1
        finish = any(
            _fake_env_action_name(action) in {"response", "terminate"} for action in actions
        )
        done = finish or self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=_fake_env_action_id(action),
                    images=[self._shot],
                    text="open settings",
                )
                for action in actions
            ],
        )

    async def close(self):
        pass


async def _tools_sent(agent, monkeypatch) -> list[dict[str, Any]]:
    """The provider tool list ``agent.sample()`` actually puts on the wire.

    The mobile sample loop passes its assembled tool list to
    ``litellm.acompletion`` unmodified, so the mocked request payload is the
    public read of the agent's advertised tool surface — no private tool-assembly
    helper needed.
    """
    mock = AsyncMock(return_value=_fake_mobile_response())
    monkeypatch.setattr("litellm.acompletion", mock)
    await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=2)
    return mock.call_args.kwargs["tools"]


class _RecordingFakeMobileEnv(_FakeMobileEnv):
    def __init__(self, terminate_after: int = 1):
        super().__init__(terminate_after=terminate_after)
        self.actions_seen: list[list[dict[str, Any]]] = []

    async def step(self, actions):
        self.actions_seen.append(actions)
        return await super().step(actions)


class _RejectEmptyActionsMobileEnv(_RecordingFakeMobileEnv):
    async def step(self, actions):
        if actions == []:
            raise AssertionError("content-only final must not call env.step([])")
        result = await super().step(actions)
        return result


class _PerCallTextMobileEnv(_RecordingFakeMobileEnv):
    async def step(self, actions):
        self.actions_seen.append(actions)
        self._step_count += 1
        finish = any(
            _fake_env_action_name(action) in {"response", "terminate"} for action in actions
        )
        done = finish or self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=_fake_env_action_id(action),
                    images=[self._shot],
                    text=f"result {_fake_env_action_id(action)}",
                )
                for action in actions
            ],
        )


class _MultiImageMobileEnv(_FakeMobileEnv):
    def __init__(self, terminate_after: int = 1):
        super().__init__(terminate_after=terminate_after)
        self._batch_shots = [
            _colored_png_bytes(color=(10, 20, 30)),
            _colored_png_bytes(color=(80, 120, 160)),
        ]

    async def step(self, actions):
        self._step_count += 1
        finish = any(
            _fake_env_action_name(action) in {"response", "terminate"} for action in actions
        )
        done = finish or self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=_fake_env_action_id(action),
                    images=list(self._batch_shots),
                    text=f"result {_fake_env_action_id(action)}",
                )
                for action in actions
            ],
        )


# -----------------------------------------------------------------------------
# computer_use_beta_enabled is not a mobile runtime knob
# -----------------------------------------------------------------------------


class TestComputerUseBeta:
    def test_default_header_has_no_computer_use_beta(self):
        a = ClaudeMobileUseAgent(model_id="claude-opus-4-6")
        header = a._build_beta_header()
        assert header is None

    def test_caching_off_yields_no_header(self):
        a = ClaudeMobileUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={
                "prompt_caching": False,
                "token_efficient_tools_beta": False,
            },
        )
        assert a._build_beta_header() is None

    @pytest.mark.parametrize("enabled", [False, True])
    def test_computer_use_beta_enabled_is_rejected(self, enabled):
        with pytest.raises(ValueError, match="computer_use_beta_enabled"):
            ClaudeMobileUseAgent(
                model_id="claude-opus-4-6",
                api_kwargs={"computer_use_beta_enabled": enabled},
            )


# -----------------------------------------------------------------------------
# Mobile tool inventory
# -----------------------------------------------------------------------------


class TestMobileToolInventory:
    """Tool inventory is asserted on the request payload ``sample()`` sends, which
    is the agent's assembled tool list verbatim."""

    async def test_tools_are_function_type_only_no_computer_tool(self, monkeypatch):
        tools = await _tools_sent(ClaudeMobileUseAgent(), monkeypatch)
        # Every tool is a function tool (wrapped by liteLLM schema); none claim
        # Claude's native computer_XXXX type.
        for t in tools:
            assert t.get("type") == "function", t
            name = _provider_tool_name(t)
            assert name is not None
            assert not name.startswith("computer_"), name

    async def test_mobile_tool_names_are_provider_flat_gui_tools(self, monkeypatch):
        tools = await _tools_sent(ClaudeMobileUseAgent(), monkeypatch)
        names = _provider_tool_names(tools)
        expected = {
            "tap",
            "long_press",
            "swipe",
            "drag",
            "pinch",
            "type",
            "system_button",
            "wait",
            "screenshot",
        }
        assert names == expected

    def test_action_space_key_is_not_a_claude_mobile_runtime_knob(self):
        with pytest.raises(TypeError, match="action_space_key"):
            ClaudeMobileUseAgent(action_space_key="nonsense")

    async def test_valid_actions_filters_mobile_tools(self, monkeypatch):
        """Mobile exposes provider-flat function tools; valid_actions filters by tool name."""
        all_names = _provider_tool_names(await _tools_sent(ClaudeMobileUseAgent(), monkeypatch))
        assert {"tap", "swipe"} <= all_names  # None → full native mobile surface
        assert "terminate" not in all_names  # finish tools are schema-backed extras

        tools = await _tools_sent(
            ClaudeMobileUseAgent(metadata=LiteCUAMetadata(valid_actions=["tap", "swipe"])),
            monkeypatch,
        )

        assert _provider_tool_names(tools) == {"tap", "swipe"}

    async def test_empty_valid_actions_drops_mobile_but_preserves_extra_tools(self, monkeypatch):
        a = ClaudeMobileUseAgent(
            metadata=LiteCUAMetadata(
                valid_actions=[],
                extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")],
            )
        )

        tools = await _tools_sent(a, monkeypatch)

        assert _provider_tool_names(tools) == {"response"}

    async def test_open_app_is_extra_tool_not_native(self, monkeypatch):
        """open_app is surfaced via env extra_tools (make_open_app_tool → app-name
        enum), NOT a native mobile action (mirrors qwen). Removing it from the native
        set keeps it out of the declared action schemas, so extra_tools=["open_app"] doesn't
        trip the collision guard."""
        from lite.core.tools.extra_tools import make_open_app_tool

        # not a native tool
        native = await _tools_sent(ClaudeMobileUseAgent(), monkeypatch)
        assert "open_app" not in _provider_tool_names(native)
        # via extra_tools: appears with the env catalog as the app_name enum, no collision
        apps = ["Chrome", "Settings", "SMS"]
        a = ClaudeMobileUseAgent(
            metadata=LiteCUAMetadata(
                others={"apps": apps},
                extra_tool_schemas=[make_open_app_tool(apps)],
            )
        )

        tools = await _tools_sent(a, monkeypatch)

        oa = next(t for t in tools if _provider_tool_name(t) == "open_app")
        assert _provider_tool_parameters(oa)["properties"]["app_name"]["enum"] == apps

    async def test_ask_user_is_schema_backed_extra_tool(self, monkeypatch):
        ask_user = make_tool_schema(
            "ask_user",
            description="Ask the simulated user.",
            parameters={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        )
        a = ClaudeMobileUseAgent(metadata=LiteCUAMetadata(extra_tool_schemas=[ask_user]))

        tools = await _tools_sent(a, monkeypatch)

        au = next(t for t in tools if _provider_tool_name(t) == "ask_user")
        assert _provider_tool_parameters(au)["required"] == ["question"]
        assert "ask_user" not in ClaudeMobileActionSpace.get_tool_names()

    async def test_tools_map_through_litellm_anthropic_transform(self, monkeypatch):
        """The tools the loop actually sends survive liteLLM's Anthropic transform."""
        config_mod = pytest.importorskip("litellm.llms.anthropic.chat.transformation")
        agent = ClaudeMobileUseAgent(
            metadata=LiteCUAMetadata(
                extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")]
            )
        )

        tools = await _tools_sent(agent, monkeypatch)

        mapped, mcp_servers = config_mod.AnthropicConfig()._map_tools(tools)

        assert mcp_servers == []
        assert {tool["name"] for tool in mapped} >= {"tap", "response"}


# -----------------------------------------------------------------------------
# base prompt lives in system_prompt; suffix defaults to ""
# -----------------------------------------------------------------------------


class TestMobileSystemPromptDefault:
    def test_default_base_prompt_nonempty(self):
        """Mobile base prompt lives in ``system_prompt`` (NOT the suffix);
        ``system_prompt_suffix`` defaults to "" so a config APPENDS rather than
        clobbering the base prompt."""
        a = ClaudeMobileUseAgent()
        assert a.system_prompt_suffix == ""
        sp = a._effective_system_prompt()
        assert "mobile device" in sp
        assert "terminate" not in sp and "response" not in sp

    def test_suffix_appends_to_user_system_prompt(self):
        a = ClaudeMobileUseAgent(
            system_prompt="BASE",
            system_prompt_suffix="APPENDED",
        )
        sys_prompt = a._effective_system_prompt()
        assert sys_prompt is not None
        assert "BASE" in sys_prompt
        assert "APPENDED" in sys_prompt
        assert sys_prompt.index("BASE") < sys_prompt.index("APPENDED")


# -----------------------------------------------------------------------------
# __post_init__ merges partial config into defaults
# -----------------------------------------------------------------------------


class TestPostInitMerge:
    def test_partial_api_kwargs_merged_with_defaults(self):
        a = ClaudeMobileUseAgent(
            api_kwargs={"max_tokens": 8192, "temperature": 0.3},
        )
        # User-supplied keys kept
        assert a.api_kwargs["max_tokens"] == 8192
        assert a.api_kwargs["temperature"] == 0.3
        assert "mobile device" in a._effective_system_prompt()

    def test_empty_api_kwargs_all_defaults(self):
        a = ClaudeMobileUseAgent(api_kwargs={})
        for k, v in CLAUDE_MOBILE_API_KWARGS_DEFAULTS.items():
            assert a.api_kwargs[k] == v


class TestMobileConfigRejection:
    """Reject stale or unknown Claude mobile config instead of silent no-ops."""

    def test_preserve_raw_response_is_not_a_claude_runtime_knob(self):
        with pytest.raises(TypeError, match="preserve_raw_response"):
            ClaudeMobileUseAgent(preserve_raw_response=True)

    def test_registry_rejects_stale_preserve_raw_response(self):
        with pytest.raises(TypeError, match="preserve_raw_response"):
            AgentRegistry.get("claude@mobile@use", preserve_raw_response=True)

    def test_registry_rejects_stale_action_space_key(self):
        with pytest.raises(TypeError, match="action_space_key"):
            AgentRegistry.get("claude@mobile@use", action_space_key="claude@mobile")

    def test_registry_rejects_stale_computer_use_beta_enabled(self):
        with pytest.raises(ValueError, match="computer_use_beta_enabled"):
            AgentRegistry.get(
                "claude@mobile@use",
                api_kwargs={"computer_use_beta_enabled": True},
            )

    def test_registry_rejects_unknown_claude_mobile_config(self):
        with pytest.raises(TypeError, match="unknown_mobile_config"):
            AgentRegistry.get("claude@mobile@use", unknown_mobile_config=True)


# -----------------------------------------------------------------------------
# End-to-end: sample loop sends function-only tools, no computer-use header
# -----------------------------------------------------------------------------


class TestSampleLoopMobilePath:
    @pytest.mark.parametrize("model_id", ["claude-opus-4-6", "claude-opus-5", "claude-sonnet-5"])
    async def test_max_steps_exhaustion_marks_truncated_with_paired_feedback(
        self, monkeypatch, model_id
    ):
        tool_call = _fake_tap_tool_call(540, 1200)
        mock = AsyncMock(return_value=_fake_mobile_response(tool_calls=[tool_call]))
        monkeypatch.setattr("litellm.acompletion", mock)
        monkeypatch.setattr("lite.agents.models.claude.agent.acompletion_with_messages", mock)

        agent = ClaudeMobileUseAgent(model_id=model_id)
        result = await agent.sample(_RecordingFakeMobileEnv(terminate_after=99), max_steps=1)

        assert result.terminated is False
        assert result.truncated is True
        assert result.steps[-1].status == "truncated"
        assert mock.call_count == 1
        if model_id in {"claude-opus-5", "claude-sonnet-5"}:
            assert mock.call_args.kwargs["thinking"] == {"type": "adaptive"}
            assert mock.call_args.kwargs["output_config"] == {"effort": "medium"}
            assert "temperature" not in mock.call_args.kwargs
            assert all(tool["type"] == "function" for tool in mock.call_args.kwargs["tools"])
        assert [m["role"] for m in result.lite_sample.messages] == ["user", "assistant", "tool"]
        tool_msg = result.lite_sample.messages[-1]
        assert tool_msg["tool_call_id"] == "call_0000"
        assert tool_msg["content"] == [
            {"type": "image", "index": 1},
            {"type": "text", "text": "open settings"},
        ]

    def test_mobile_parser_preserves_extra_native_relative_order(self):
        response_tool = _fake_mobile_tool_call(
            "response",
            '{"text": "done"}',
            id_="toolu_response_1",
        )
        response = _fake_mobile_response(
            tool_calls=[
                response_tool,
                _fake_tap_tool_call(540, 1200, id_="toolu_tap_1"),
            ]
        )

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
            extra_tool_names=frozenset({"response"}),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["response", "mobile"]
        assert tool_call_arguments(calls[1])["actions"][0]["action"] == "tap"
        assert [tool_call_id(call) for call in calls] == ["call_0000", "call_0001"]
        assert all("tool_call_id" not in call for call in calls)

    def test_mobile_parser_merges_adjacent_provider_flat_native_calls_without_crossing_extras(self):
        response_tool = _fake_mobile_tool_call(
            "response",
            '{"text": "done"}',
            id_="toolu_response_1",
        )
        swipe_tool = _fake_mobile_tool_call(
            "swipe",
            json.dumps(
                {
                    "start_coordinate": [540, 1600],
                    "coordinate": [540, 800],
                }
            ),
            id_="toolu_swipe_1",
        )
        response = _fake_mobile_response(
            tool_calls=[
                _fake_tap_tool_call(108, 240, id_="toolu_tap_1"),
                swipe_tool,
                response_tool,
                _fake_tap_tool_call(216, 480, id_="toolu_tap_2"),
            ]
        )

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
            extra_tool_names=frozenset({"response"}),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["mobile", "response", "mobile"]
        assert [tool_call_id(call) for call in calls] == [
            "call_0000",
            "call_0001",
            "call_0002",
        ]
        assert [action["action"] for action in tool_call_arguments(calls[0])["actions"]] == [
            "tap",
            "swipe",
        ]
        assert [action["action"] for action in tool_call_arguments(calls[2])["actions"]] == ["tap"]

    def test_mobile_parser_drops_malformed_and_empty_arguments(self, caplog):
        bad_tool = _fake_mobile_tool_call(
            "response",
            "{not json",
            id_="toolu_bad_1",
        )
        none_tool = _fake_mobile_tool_call(
            "response",
            None,
            id_="toolu_none_1",
        )
        response = _fake_mobile_response(tool_calls=[bad_tool, none_tool])

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
            extra_tool_names=frozenset({"response"}),
        ).message

        # BOTH are dropped, and neither reaches the env: ``{not json`` fails to parse, and
        # LiteLLM normalizes ``arguments=None`` to ``""``, which reaches the same
        # malformed-JSON branch. The provider never handed over an argument object, so
        # there is nothing for the env to answer.
        assert msg["tool_calls"] == []
        assert "malformed arguments: response" in caplog.text

    def test_mobile_parser_marks_undeclared_tool_call_as_model_output_error(self, caplog):
        bad_tool = _fake_mobile_tool_call(
            "double_tap",
            json.dumps({"coordinate": [540, 1200]}),
            id_="toolu_double_tap_1",
        )
        response = _fake_mobile_response(tool_calls=[bad_tool])

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared tool_call double_tap"
        assert "Ignoring undeclared Claude mobile tool_call: double_tap" in caplog.text

    def test_mobile_parser_keeps_active_ask_user_extra(self):
        ask_user_tool = _fake_mobile_tool_call(
            "ask_user",
            json.dumps({"question": "Continue?"}),
            id_="toolu_ask_user_1",
        )
        response = _fake_mobile_response(tool_calls=[ask_user_tool])

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
            extra_tool_names=frozenset({"ask_user"}),
        ).message

        assert [tool_call_id(call) for call in msg["tool_calls"]] == ["call_0000"]
        assert [tool_call_name(call) for call in msg["tool_calls"]] == ["ask_user"]
        assert tool_call_arguments(msg["tool_calls"][0]) == {"question": "Continue?"}

    def test_mobile_parser_routes_invalid_active_extra_to_env_feedback(self):
        """A missing required argument is env ingress's answer, not a drop.

        The valid GUI call and the malformed extra both survive parsing; the
        model learns what was wrong from the env-owned feedback keyed to the
        extra's call id.
        """
        from lite.gym.utils.feedback.ingress import prepare_env_tool_calls

        ask_user_schema = make_tool_schema(
            "ask_user",
            description="Ask the simulated user.",
            parameters={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        )
        response = _fake_mobile_response(
            tool_calls=[
                _fake_mobile_tool_call(
                    "tap",
                    json.dumps({"coordinate": [540, 1200]}),
                    id_="toolu_tap",
                ),
                _fake_mobile_tool_call(
                    "ask_user",
                    json.dumps({}),
                    id_="toolu_ask_user_bad",
                ),
            ]
        )

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
            extra_tool_names=frozenset({"ask_user"}),
        ).message

        assert [tool_call_name(call) for call in msg["tool_calls"]] == ["mobile", "ask_user"]
        assert pop_model_output_error(msg) is None

        routed, errors = prepare_env_tool_calls(
            msg["tool_calls"],
            LiteCUAMetadata(extra_tool_schemas=[ask_user_schema]),
        )
        assert all(call["name"] != "ask_user" for call, _ in routed)
        assert set(errors) == {"call_0001"}
        assert "ask_user" in errors["call_0001"].message

    @pytest.mark.parametrize(
        "name,arguments",
        [
            ("tap", {}),
            ("tap", {"coordinate": [None, 7]}),
            ("swipe", {"start_coordinate": [10, 20]}),
        ],
    )
    def test_mobile_parser_marks_malformed_native_as_model_output_error(self, name, arguments):
        bad_tool = _fake_mobile_tool_call(
            name,
            json.dumps(arguments),
            id_="toolu_bad_native",
        )
        response = _fake_mobile_response(tool_calls=[bad_tool])

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
        ).message

        assert msg["tool_calls"] == []
        assert "malformed Claude mobile arguments" in pop_model_output_error(msg)

    def test_mobile_parser_rejects_undeclared_provider_wrapper(self):
        bad_tool = _fake_mobile_tool_call(
            "mobile",
            json.dumps(
                {
                    "actions": [{"action": "open_app", "app_name": "Settings"}],
                }
            ),
            id_="toolu_bad_native",
        )
        response = _fake_mobile_response(tool_calls=[bad_tool])

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared tool_call mobile"

    def test_mobile_parser_rejects_undeclared_provider_wrapper_batch(self):
        bad_tool = _fake_mobile_tool_call(
            "mobile",
            json.dumps(
                {
                    "actions": [
                        {"action": "tap", "coordinate": [540, 1200]},
                        {"action": "open_app", "app_name": "Settings"},
                    ],
                }
            ),
            id_="toolu_bad_native",
        )
        response = _fake_mobile_response(tool_calls=[bad_tool])

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared tool_call mobile"

    def test_mobile_parser_rejects_replayed_native_when_request_hid_native(self):
        native_tool = _fake_mobile_tool_call(
            "tap",
            json.dumps({"coordinate": [100, 200]}),
            id_="toolu_hidden_native",
        )
        response = _fake_mobile_response(tool_calls=[native_tool])

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
            active_provider_tool_names=frozenset(),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared tool_call tap"

    def test_mobile_merge_reports_malformed_native_provider_error(self):
        bad_tool = _fake_mobile_tool_call(
            "tap",
            json.dumps({}),
            id_="toolu_bad_native",
        )
        response = _fake_mobile_response(tool_calls=[bad_tool])

        parsed = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
            extra_tool_names=frozenset(),
        )

        assert tuple(p.canonical_call_id for p in parsed.provider_call_provenance) == (None,)
        assert "toolu_bad_native" in parsed.provider_errors
        assert (
            "malformed Claude mobile arguments for tap"
            in parsed.provider_errors["toolu_bad_native"]
        )

    def test_mobile_merge_reports_undeclared_provider_wrapper_error(self):
        bad_tool = _fake_mobile_tool_call(
            "mobile",
            json.dumps(
                {
                    "actions": [{"action": "tap", "coordinate": [100, 200]}],
                }
            ),
            id_="toolu_mobile_wrapper",
        )
        response = _fake_mobile_response(tool_calls=[bad_tool])

        parsed = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
            extra_tool_names=frozenset(),
        )

        assert tuple(p.canonical_call_id for p in parsed.provider_call_provenance) == (None,)
        assert "undeclared tool_call: mobile" in parsed.provider_errors["toolu_mobile_wrapper"]

    def test_mobile_parser_prefers_tool_calls_view_over_content_tool_use(self):
        response = _fake_mobile_response(
            tool_calls=[
                _fake_tap_tool_call(10, 20, id_="toolu_calls"),
            ],
        )
        response.choices[0].message.content = [
            {
                "type": "tool_use",
                "id": "toolu_content",
                "name": "tap",
                "input": {"coordinate": [30, 40]},
            }
        ]

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
        ).message

        action = tool_call_arguments(msg["tool_calls"][0])["actions"][0]
        assert action["coordinate"] == [9, 8]

    def test_mobile_provider_tool_uses_preserve_falsy_malformed_content_input(self):
        response = _fake_mobile_response()
        response.choices[0].message.content = [
            {
                "type": "tool_use",
                "id": "toolu_empty",
                "name": "empty",
                "input": [],
            }
        ]

        parsed = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
            extra_tool_names=frozenset({"empty"}),
        )

        assert tuple(p.canonical_call_id for p in parsed.provider_call_provenance) == (None,)
        assert "toolu_empty" in parsed.provider_errors
        assert "malformed tool_use input for empty" in parsed.provider_errors["toolu_empty"]

    def test_mobile_parser_scales_resized_frame_coordinates(self):
        response = _fake_mobile_response(
            tool_calls=[
                _fake_tap_tool_call(360, 640, id_="toolu_tap_1"),
            ]
        )

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.5,
            scale_y=1.875,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
        ).message

        action = tool_call_arguments(msg["tool_calls"][0])["actions"][0]
        assert action["action"] == "tap"
        assert 495 <= action["coordinate"][0] <= 505, action
        assert 495 <= action["coordinate"][1] <= 505, action

    async def test_request_has_no_computer_use_beta_header(self, monkeypatch):
        """Critical mobile guarantee: the mobile header never contains computer-use beta."""
        mock = AsyncMock(return_value=_fake_mobile_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeMobileUseAgent()
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        assert mock.called
        headers = mock.call_args.kwargs.get("headers", {})
        anthropic_beta = headers.get("anthropic-beta", "") if headers else ""
        assert "computer-use" not in anthropic_beta, f"unexpected: {anthropic_beta!r}"

    async def test_request_tools_are_all_function_type(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_mobile_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeMobileUseAgent()
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        tools = mock.call_args.kwargs.get("tools", [])
        assert tools, "tools list must not be empty"
        for t in tools:
            assert t.get("type") == "function", f"non-function tool leaked: {t}"

    async def test_system_prompt_sent_by_default(self, monkeypatch):
        """The provider request carries the mobile base prompt, without finish
        guidance unless the env declares finish tools."""
        mock = AsyncMock(return_value=_fake_mobile_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeMobileUseAgent()
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        messages = mock.call_args.kwargs["messages"]
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        assert sys_msgs, "expected system message with mobile prompt"
        sys_content = str(sys_msgs[0]["content"])
        # The prompt must report the frame the model was actually given.
        sent_w, sent_h = _sent_image_size(messages)
        assert "mobile device" in sys_content
        assert f"{sent_w}x{sent_h}" in sys_content
        assert "{w}" not in sys_content and "{h}" not in sys_content
        assert "terminate" not in sys_content
        assert "response" not in sys_content

    async def test_finish_extra_tools_restore_mobile_completion_guidance(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_mobile_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        metadata = LiteCUAMetadata(extra_tool_schemas=[
            LiteFinishToolSet.get_tool_schema("response"),
            LiteFinishToolSet.get_tool_schema("terminate"),
        ])
        agent = ClaudeMobileUseAgent(metadata=metadata)
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        messages = mock.call_args.kwargs["messages"]
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        assert sys_msgs
        assert (
            "When the task is done, call `terminate` "
            "(use `response` to return an answer the task asks for)."
        ) in str(sys_msgs[0]["content"])

    async def test_custom_system_prompt_is_augmented_by_finish_tools(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_mobile_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        metadata = LiteCUAMetadata(extra_tool_schemas=[
            LiteFinishToolSet.get_tool_schema("response"),
            LiteFinishToolSet.get_tool_schema("terminate"),
        ])
        agent = ClaudeMobileUseAgent(
            metadata=metadata,
            system_prompt="CUSTOM {w}x{h}",
            system_prompt_suffix="APPENDED",
        )
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        messages = mock.call_args.kwargs["messages"]
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        assert sys_msgs
        sent_w, sent_h = _sent_image_size(messages)
        content_blocks = sys_msgs[0]["content"]
        assert isinstance(content_blocks, list)
        text = "".join(
            block["text"]
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert text.startswith(f"CUSTOM {sent_w}x{sent_h}\nAPPENDED")
        assert "\n\nWhen the task is done, call `terminate`" in text
        assert "use `response` to return an answer" in text

    async def test_mobile_completion_guidance_matches_active_finish_surface(self, monkeypatch):
        async def first_prompt_for(*schema_names: str) -> str:
            mock = AsyncMock(return_value=_fake_mobile_response())
            monkeypatch.setattr("litellm.acompletion", mock)
            metadata = LiteCUAMetadata(extra_tool_schemas=[
                LiteFinishToolSet.get_tool_schema(name) for name in schema_names
            ])
            await ClaudeMobileUseAgent(metadata=metadata).sample(
                _FakeMobileEnv(terminate_after=1), max_steps=3
            )
            messages = mock.call_args.kwargs["messages"]
            sys_msgs = [m for m in messages if m.get("role") == "system"]
            return str(sys_msgs[0]["content"])

        response_only = await first_prompt_for("response")
        assert "call `response`" in response_only
        assert "call `terminate`" not in response_only

        terminate_only = await first_prompt_for("terminate")
        assert "When the task is done, call `terminate`." in terminate_only
        assert "call `response`" not in terminate_only

    async def test_finish_guidance_stands_alone_without_base_system_prompt(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_mobile_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        metadata = LiteCUAMetadata(extra_tool_schemas=[
            LiteFinishToolSet.get_tool_schema("terminate")
        ])
        agent = ClaudeMobileUseAgent(metadata=metadata, system_prompt=None)
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        messages = mock.call_args.kwargs["messages"]
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        assert sys_msgs
        assert "When the task is done, call `terminate`." in str(sys_msgs[0]["content"])
        assert "mobile device" not in str(sys_msgs[0]["content"])

    async def test_tap_coords_from_resized_frame_scale_to_original(self, monkeypatch):
        """Model pixel coords are relative to the frame sent to Claude."""
        # FakeMobileEnv renders 1080x2400. Claude receives the largest
        # aspect-preserving frame allowed by Anthropic's standard vision tier;
        # the sent-frame center scales back to the original screenshot before
        # CUA-lite normalizes to [0, 1000].
        mock = AsyncMock(side_effect=_centre_tap_then(_fake_mobile_response()))
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeMobileUseAgent()
        result = await agent.sample(_FakeMobileEnv(terminate_after=2), max_steps=5)

        asst_messages = [m for m in result.lite_sample.messages if m.get("role") == "assistant"]
        assert asst_messages
        assert "raw_response" not in asst_messages[0]
        tc = asst_messages[0]["tool_calls"][0]
        assert tool_call_id(tc) == "call_0000"
        assert "tool_call_id" not in tc
        assert tool_call_name(tc) == "mobile"
        action = tool_call_arguments(tc)["actions"][0]
        assert action["action"] == "tap"
        coord = action.get("coordinate")
        assert coord is not None
        # Center of the frame sent to Claude -> 500 in CUA-lite normalized.
        assert 495 <= coord[0] <= 505, coord
        assert 495 <= coord[1] <= 505, coord

        second_messages = mock.call_args_list[1].kwargs["messages"]
        provider_tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert len(provider_tool_msgs) == 1
        assert provider_tool_msgs[0]["tool_call_id"] == "toolu_tap_1"

    async def test_second_turn_provider_history_uses_provider_id_but_canonical_result_lookup(
        self, monkeypatch
    ):
        provider_id = "toolu_provider_tap_1"
        mock = AsyncMock(
            side_effect=_centre_tap_then(
                _fake_mobile_text_response("done"),
                provider_id=provider_id,
            )
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeMobileUseAgent()
        result = await agent.sample(_PerCallTextMobileEnv(terminate_after=2), max_steps=3)

        canonical_id = "call_0000"
        assert provider_id != canonical_id
        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert [tool_call_id(call) for call in assistant["tool_calls"]] == [canonical_id]
        assert [tool_call_name(call) for call in assistant["tool_calls"]] == ["mobile"]

        second_messages = mock.call_args_list[1].kwargs["messages"]
        provider_tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in provider_tool_msgs] == [provider_id]
        provider_content = provider_tool_msgs[0]["content"]
        assert isinstance(provider_content, list)
        provider_texts = [
            block["text"]
            for block in provider_content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        assert provider_texts == [f"result {canonical_id}"]

        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == canonical_id
        assert {"type": "text", "text": f"result {canonical_id}"} in tool_msg["content"]

    async def test_multi_image_mobile_result_logs_all_images_but_sends_latest(
        self,
        monkeypatch,
    ):
        provider_id = "toolu_provider_tap_batch_1"
        mock = AsyncMock(
            side_effect=_centre_tap_then(
                _fake_mobile_text_response("done"),
                provider_id=provider_id,
            )
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _MultiImageMobileEnv(terminate_after=2)
        agent = ClaudeMobileUseAgent()

        class _Hook:
            def __init__(self):
                self.current_image_indices = []

            def on_step(self, data):
                self.current_image_indices.append(data.current_image_index)

            def on_complete(self, result):
                del result

        hook = _Hook()
        result = await agent.sample(env, max_steps=3, hooks=[hook])

        assert len(result.lite_sample.images) == 3
        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_0000"
        assert [
            block["index"]
            for block in tool_msg["content"]
            if isinstance(block, dict) and block.get("type") == "image"
        ] == [2]
        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 2)]
        assert hook.current_image_indices == [0, 2]
        assert len(result.processed_images) == len(result.lite_sample.images)
        assert result.processed_images[0] is result.lite_sample.images[0]
        assert result.processed_images[1] is None
        assert result.processed_images[2] is result.lite_sample.images[2]

        second_messages = mock.call_args_list[1].kwargs["messages"]
        provider_tool_msg = next(m for m in second_messages if m.get("role") == "tool")
        assert provider_tool_msg["tool_call_id"] == provider_id
        image_blocks = [
            block
            for block in provider_tool_msg["content"]
            if isinstance(block, dict) and block.get("type") == "image_url"
        ]
        assert len(image_blocks) == 1
        payload = image_blocks[0]["image_url"]["url"].split("base64,", 1)[1]
        sent_image = Image.open(io.BytesIO(base64.b64decode(payload)))
        assert sent_image.getpixel((0, 0)) == (80, 120, 160)
        sent_payloads = [
            json.dumps(call.kwargs["messages"], ensure_ascii=False, default=str)
            for call in mock.call_args_list
        ]
        assert all("_cua_lite_image_index" not in payload for payload in sent_payloads)
        assert all("_cua_lite_image_index" not in step.prompt for step in result.steps)

    async def test_multi_image_extra_tool_result_keeps_all_model_visible_images(
        self,
        monkeypatch,
    ):
        extra = make_tool_schema(
            "visual_extra",
            description="Return visual feedback.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        provider_id = "toolu_visual_batch_1"
        tool_call = _fake_mobile_tool_call(
            "visual_extra",
            "{}",
            id_=provider_id,
        )
        mock = AsyncMock(
            side_effect=[
                _fake_mobile_response(tool_calls=[tool_call]),
                _fake_mobile_text_response("done"),
            ]
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _MultiImageMobileEnv(terminate_after=2)
        result = await ClaudeMobileUseAgent(
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        ).sample(env, max_steps=3)

        assert len(result.lite_sample.images) == 3
        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert [
            block["index"]
            for block in tool_msg["content"]
            if isinstance(block, dict) and block.get("type") == "image"
        ] == [1, 2]
        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 1, 2)]
        assert result.processed_images == result.lite_sample.images

        second_messages = mock.call_args_list[1].kwargs["messages"]
        provider_tool_msg = next(m for m in second_messages if m.get("role") == "tool")
        assert provider_tool_msg["tool_call_id"] == provider_id
        image_blocks = [
            block
            for block in provider_tool_msg["content"]
            if isinstance(block, dict) and block.get("type") == "image_url"
        ]
        assert len(image_blocks) == 2
        sent_images = [
            Image.open(io.BytesIO(base64.b64decode(block["image_url"]["url"].split(",", 1)[1])))
            for block in image_blocks
        ]
        assert [image.getpixel((0, 0)) for image in sent_images] == [(10, 20, 30), (80, 120, 160)]

    async def test_content_block_tool_use_gets_provider_tool_result(self, monkeypatch):
        r1 = _fake_mobile_response()
        r1.choices[0].message.content = [
            {
                "type": "tool_use",
                "id": "toolu_tap_1",
                "name": "tap",
                "input": {"coordinate": [540, 1200]},
            }
        ]
        r2 = _fake_mobile_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeMobileUseAgent()
        await agent.sample(_FakeMobileEnv(terminate_after=2), max_steps=3)

        second_messages = mock.call_args_list[1].kwargs["messages"]
        assistant_msgs = [m for m in second_messages if m.get("role") == "assistant"]
        provider_tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert any(
            (m.get("tool_calls") or [{}])[0].get("id") == "toolu_tap_1" for m in assistant_msgs
        )
        assert provider_tool_msgs[-1]["tool_call_id"] == "toolu_tap_1"

    def test_content_block_tool_use_nonobject_input_is_model_output_error(self):
        response = _fake_mobile_response()
        response.choices[0].message.content = [
            {
                "type": "tool_use",
                "id": "toolu_bad",
                "name": "tap",
                "input": ["not", "object"],
            }
        ]

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "malformed tool_use input for tap"

    def test_content_block_tool_use_missing_provider_id_is_model_output_error(self):
        response = _fake_mobile_response()
        response.choices[0].message.content = [
            {
                "type": "tool_use",
                "name": "tap",
                "input": {"coordinate": [540, 1200]},
            }
        ]

        msg = parse_mobile_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "missing provider id for tap"

    async def test_provider_flat_mobile_calls_all_get_tool_results(self, monkeypatch):
        open_app_tool = make_tool_schema(
            "open_app",
            description="Open an app.",
            parameters={
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
            },
        )
        provider_flat_tool_calls = [
            _fake_mobile_tool_call(
                "tap",
                json.dumps({"coordinate": [540, 1200]}),
                id_="toolu_tap",
            ),
            _fake_mobile_tool_call(
                "swipe",
                json.dumps(
                    {
                        "start_coordinate": [540, 1600],
                        "coordinate": [540, 800],
                    }
                ),
                id_="toolu_swipe",
            ),
            _fake_mobile_tool_call(
                "open_app",
                json.dumps({"app_name": "Settings"}),
                id_="toolu_open",
            ),
            _fake_mobile_tool_call(
                "open_app",
                json.dumps({"app_name": 42}),
                id_="toolu_open_bad",
            ),
        ]
        mock = AsyncMock(
            side_effect=[
                _fake_mobile_response(tool_calls=provider_flat_tool_calls),
                _fake_mobile_text_response("done"),
            ]
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _PerCallTextMobileEnv(terminate_after=99)
        env.metadata.extra_tool_schemas = [open_app_tool]
        agent = ClaudeMobileUseAgent(metadata=env.metadata)
        result = await agent.sample(env, max_steps=3)

        second_messages = mock.call_args_list[1].kwargs["messages"]
        provider_tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        provider_tool_msgs = provider_tool_msgs[-4:]
        assert [m["tool_call_id"] for m in provider_tool_msgs] == [
            "toolu_tap",
            "toolu_swipe",
            "toolu_open",
            "toolu_open_bad",
        ]

        def output_text(message):
            if isinstance(message["content"], str):
                return message["content"]
            return next(
                block["text"] for block in message["content"] if block.get("type") == "text"
            )

        # ``app_name: 42`` violates the extra's schema, but argument admission is
        # the env's: the call reaches it and is answered like any other.
        output_texts = [output_text(message) for message in provider_tool_msgs]
        assert output_texts == [
            "result call_0000",
            "result call_0000",
            "result call_0001",
            "result call_0002",
        ]

        def has_output_image(message):
            if isinstance(message["content"], str):
                return False
            return any(block.get("type") == "image_url" for block in message["content"])

        assert has_output_image(provider_tool_msgs[0]) is False
        assert has_output_image(provider_tool_msgs[1]) is True
        assert has_output_image(provider_tool_msgs[3]) is True
        image_block = next(
            block for block in provider_tool_msgs[1]["content"] if block.get("type") == "image_url"
        )
        # A result image rides the same provider-safe resize as the observation
        # screenshot it mirrors — same source bytes in, same payload out.
        first_messages = mock.call_args_list[0].kwargs["messages"]
        observation_block = next(
            block
            for message in first_messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "image_url"
        )
        assert image_block["image_url"]["url"] == observation_block["image_url"]["url"]
        assert _sent_image_size(first_messages) != Image.open(io.BytesIO(env._shot)).size
        assistant = result.lite_sample.messages[1]
        assert [tool_call_name(call) for call in assistant["tool_calls"]] == [
            "mobile",
            "open_app",
            "open_app",
        ]
        tool_messages = [
            message for message in result.lite_sample.messages if message.get("role") == "tool"
        ]
        assert [message["tool_call_id"] for message in tool_messages] == [
            "call_0000",
            "call_0001",
            "call_0002",
        ]

    async def test_content_only_final_text_is_not_saved_as_response_tool(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_mobile_text_response("  done  "))
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _RejectEmptyActionsMobileEnv(terminate_after=99)
        agent = ClaudeMobileUseAgent()
        result = await agent.sample(env, max_steps=3)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert "raw_response" not in assistant
        assert assistant["content"] == [{"type": "text", "text": "  done  "}]
        assert not assistant.get("tool_calls")
        assert len(env.actions_seen) == 1
        assert tool_call_name(env.actions_seen[0][0]) == "response"
        assert "call_id" not in env.actions_seen[0][0]
        assert tool_call_arguments(env.actions_seen[0][0]) == {"text": "done"}
        assert result.terminated is True
        assert result.truncated is False
        assert result.episode_return == 1.0
        assert result.steps[0].reward == 1.0

    async def test_content_only_final_uses_runtime_response_when_enabled(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_mobile_text_response("final text"))
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _RejectEmptyActionsMobileEnv(terminate_after=99)
        env.metadata.extra_tool_schemas = [LiteFinishToolSet.get_tool_schema("response")]
        agent = ClaudeMobileUseAgent()
        result = await agent.sample(env, max_steps=3)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert assistant["content"] == [{"type": "text", "text": "final text"}]
        assert not assistant.get("tool_calls")
        assert len(env.actions_seen) == 1
        assert tool_call_name(env.actions_seen[0][0]) == "response"
        assert "call_id" not in env.actions_seen[0][0]
        assert tool_call_arguments(env.actions_seen[0][0]) == {"text": "final text"}
        assert result.terminated is True
        assert result.truncated is False


async def test_claude_mobile_finish_exposure_uses_extra_tool_schemas_not_valid_actions(
    monkeypatch,
) -> None:
    valid_only = await _tools_sent(
        ClaudeMobileUseAgent(
            metadata=LiteCUAMetadata(dims=("mobile", "use"), valid_actions=["tap"])
        ),
        monkeypatch,
    )
    assert _provider_tool_names(valid_only).isdisjoint({"response", "terminate"})

    with_schema = await _tools_sent(
        ClaudeMobileUseAgent(
            metadata=LiteCUAMetadata(
                dims=("mobile", "use"),
                extra_tool_schemas=[RESPONSE_SCHEMA],
            )
        ),
        monkeypatch,
    )
    assert "response" in _provider_tool_names(with_schema)
    response_tool = next(t for t in with_schema if tool_schema_name(t) == "response")
    assert response_tool["type"] == "function"
    assert response_tool["function"]["name"] == "response"
