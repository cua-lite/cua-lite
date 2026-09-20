"""Characterization tests for ClaudeDesktopUseAgent kwargs.

Each kwarg added during the harness-optimization pass gets a non-default-path
assertion here, so regressions on the contract (not just default behavior) show
up in CI.

Run:
    uv run pytest tests/agents/models/claude/test_claude_agent.py -v
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agents._support.valid_actions_gating import (
    BASH_SCHEMA,
    RESPONSE_SCHEMA,
    anthropic_provider_tool_names,
    anthropic_tools_sent,
)
from agents.models._support.provider_fakes import png_bytes
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from PIL import Image

from lite.agents.core.agent import AgentRegistry
from lite.agents.models.claude.action_space import (
    ClaudeDesktopActionSpace,
    ClaudeDesktopGroundingPointActionSpace,
)
from lite.agents.models.claude.agent import (
    ClaudeDesktopGroundingPointAgent,
    ClaudeDesktopUseAgent,
)
from lite.agents.models.claude.utils.history import (
    append_canonical_step_feedback_messages,
    append_provider_assistant_message,
    filter_to_n_most_recent_images,
    inject_prompt_caching,
    strip_images,
)
from lite.agents.models.claude.utils.parse import (
    parse_response_with_provenance,
)
from lite.agents.models.claude.utils.toolset import _messages_for_anthropic
from lite.core import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_TRUNCATED,
    LiteCUAMetadata,
    LiteGenericMetadata,
    LiteSample,
)
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.extra_tools import BASH_TOOL_NAME, LiteFinishToolSet, LiteShellToolSet
from lite.core.tools.results import LiteToolResult
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

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _make_sample(w: int = 1024, h: int = 768) -> LiteSample:
    meta = LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
        others={"resolution": [w, h]},
    )
    img = Image.new("RGB", (w, h), color="white")
    sample = LiteSample(metadata=meta, images=[img])
    sample.messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "do the thing"},
                {"type": "image", "index": 0},
            ],
        }
    ]
    return sample


def _colored_png_bytes(
    w: int = 800,
    h: int = 600,
    color: tuple[int, int, int] = (32, 64, 96),
) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _fake_completion_response(
    content: Any = "done",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
) -> Any:
    """Build a minimal ``litellm.acompletion`` response shape."""
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        role="assistant",
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model_dump=lambda: {"choices": []},
    )
    return resp


def _fake_tool_call(name: str, arguments: str | None, id_: str = "tc1") -> Any:
    return ChatCompletionMessageToolCall(
        id=id_,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


async def _tools_sent(agent, monkeypatch, env: Any = None) -> list[dict[str, Any]]:
    """The provider tool list ``agent.sample()`` actually puts on the wire.

    The sample loop passes its assembled tool list to ``litellm.acompletion``
    unmodified, so the mocked request payload is the public read of the agent's
    advertised tool surface — no private tool-assembly helper needed. The
    computer-tool version/beta flag is pinned through the public
    ``api_kwargs.computer_tool_version`` / ``computer_use_beta_flag`` override, and
    the display dimensions come from the screenshot ``env`` serves.
    """
    mock = AsyncMock(return_value=_fake_completion_response())
    monkeypatch.setattr("litellm.acompletion", mock)
    monkeypatch.setattr("lite.agents.models.claude.agent.acompletion_with_messages", mock)
    await agent.sample(env if env is not None else _FakeEnv(terminate_after=1), max_steps=2)
    return mock.call_args.kwargs["tools"]


class _FakeEnv:
    def __init__(self, *, terminate_after: int = 1):
        self.metadata = LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
            others={"resolution": [800, 600]},
        )
        self._shot = png_bytes(800, 600)
        self._step_count = 0
        self._terminate_after = terminate_after
        self.closed = False

    async def reset(self):
        return LiteEnvObservation(image=self._shot, text="instr")

    async def step(self, actions):
        self._step_count += 1
        finish = any(tool_call_name(action) in {"response", "terminate"} for action in actions)
        done = finish or self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(tool_call_id=tool_call_id(action), images=[self._shot], text="instr")
                for action in actions
            ],
        )

    async def close(self):
        self.closed = True


class _CloseRaisesEnv(_FakeEnv):
    async def close(self):
        self.closed = True
        raise RuntimeError("claude close exploded")


class _CloseCancelledEnv(_FakeEnv):
    async def close(self):
        self.closed = True
        raise asyncio.CancelledError


class _RecordingFakeEnv(_FakeEnv):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.actions_seen: list[list[dict[str, Any]]] = []

    async def step(self, actions):
        self.actions_seen.append(actions)
        return await super().step(actions)


class _RejectEmptyActionsEnv(_RecordingFakeEnv):
    async def step(self, actions):
        if actions == []:
            raise AssertionError("content-only final must not call env.step([])")
        result = await super().step(actions)
        return result


class _ToolResultsEnv(_FakeEnv):
    async def step(self, actions):
        result = await super().step(actions)
        if actions:
            call = actions[0]
            result.results = [
                LiteToolResult(tool_call_id=tool_call_id(call), text="per-call stdout"),
            ]
        return result


class _EmptyToolResultsEnv(_FakeEnv):
    async def step(self, actions):
        result = await super().step(actions)
        if actions:
            call = actions[0]
            result.results = [
                LiteToolResult(tool_call_id=tool_call_id(call), text=""),
            ]
        return result


class _ImageToolResultsEnv(_FakeEnv):
    async def step(self, actions):
        result = await super().step(actions)
        if actions:
            call = actions[0]
            result.results = [
                LiteToolResult(
                    tool_call_id=tool_call_id(call), images=[self._shot], text="visual obs"
                ),
            ]
        return result


class _MultiImageToolResultsEnv(_FakeEnv):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._batch_shots = [
            _colored_png_bytes(color=(10, 20, 30)),
            _colored_png_bytes(color=(80, 120, 160)),
        ]

    async def step(self, actions):
        self._step_count += 1
        finish = any(tool_call_name(action) in {"response", "terminate"} for action in actions)
        done = finish or self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=tool_call_id(action),
                    images=list(self._batch_shots),
                    text="visual batch",
                )
                for action in actions
            ],
        )


class _MixedComputerAndBashResultsEnv(_FakeEnv):
    async def step(self, actions):
        result = await super().step(actions)
        result.results = []
        for call in actions:
            if tool_call_name(call) == "computer":
                result.results.append(
                    LiteToolResult(
                        tool_call_id=tool_call_id(call),
                        images=[self._shot],
                        text="computer screen text",
                    )
                )
            elif tool_call_name(call) == "bash":
                result.results.append(
                    LiteToolResult(
                        tool_call_id=tool_call_id(call),
                        text="bash stdout",
                    )
                )
        return result


class _TerminalNoResultsEnv(_RecordingFakeEnv):
    async def step(self, actions):
        self.actions_seen.append(actions)
        self._step_count += 1
        return LiteEnvStepResult(reward=1.0, terminated=True, results=[])


class _ReversedResultOrderEnv(_FakeEnv):
    """Answers a two-call turn in the OPPOSITE order the calls were emitted.

    Also exercises all three per-call text channels: plain text, text plus an
    ``error``, and ``metadata``.
    """

    async def step(self, actions):
        result = await super().step(actions)
        per_call: list[LiteToolResult] = []
        for call in actions:
            if tool_call_name(call) == "computer":
                per_call.append(
                    LiteToolResult(
                        tool_call_id=tool_call_id(call),
                        images=[self._shot],
                        text="computer screen text",
                        metadata={"source": "screen"},
                    )
                )
            elif tool_call_name(call) == "bash":
                per_call.append(
                    LiteToolResult(
                        tool_call_id=tool_call_id(call),
                        text="bash stdout",
                        error="exit status 1",
                    )
                )
        result.results = list(reversed(per_call))
        return result


class _DropsOneResultEnv(_FakeEnv):
    """Non-terminal env that answers only the FIRST call of a multi-call turn."""

    async def step(self, actions):
        result = await super().step(actions)
        result.results = [
            LiteToolResult(tool_call_id=tool_call_id(actions[0]), text="only one result")
        ]
        return result


class _FreshImageErrorEnv(_FakeEnv):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._fresh_shot = _colored_png_bytes()

    async def step(self, actions):
        self._step_count += 1
        finish = any(tool_call_name(action) in {"response", "terminate"} for action in actions)
        done = finish or self._step_count >= self._terminate_after
        if finish:
            return LiteEnvStepResult(reward=1.0, terminated=True, results=[])
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=tool_call_id(action),
                    images=[self._fresh_shot],
                    text="## AXTree:\nbutton Search",
                    error="invalid action: screenshot",
                    metadata={"is_error": True},
                )
                for action in actions
            ],
        )


async def test_claude_unpaired_text_feedback_writes_one_trajectory_message():
    trajectory = LiteSample(metadata=LiteCUAMetadata())
    step_result = LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id=None,
                text="stdout",
                metadata={"source": "env"},
            )
        ]
    )

    (
        image_index,
        image_indices_by_call_id,
    ) = await append_canonical_step_feedback_messages(
        trajectory,
        step_result,
        [],
    )

    assert image_index is None
    assert image_indices_by_call_id == {}
    assert trajectory.images == []
    assert trajectory.messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "stdout"},
                {"type": "metadata", "data": {"source": "env"}},
            ],
        }
    ]


async def test_claude_unpaired_image_feedback_uses_result_ordered_indices():
    trajectory = LiteSample(metadata=LiteCUAMetadata())
    step_result = LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id=None,
                images=[_colored_png_bytes(color=(10, 20, 30))],
                text="first",
            ),
            LiteToolResult(
                tool_call_id=None,
                images=[_colored_png_bytes(color=(80, 120, 160))],
                metadata={"source": "env"},
            ),
        ]
    )

    (
        image_index,
        image_indices_by_call_id,
    ) = await append_canonical_step_feedback_messages(
        trajectory,
        step_result,
        [],
    )

    assert image_index == 1
    assert image_indices_by_call_id == {}
    assert len(trajectory.images) == 2
    assert trajectory.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "first"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 1},
                {"type": "metadata", "data": {"source": "env"}},
            ],
        },
    ]


# -----------------------------------------------------------------------------
# prompt-caching beta flag
# -----------------------------------------------------------------------------


class TestPromptCachingBeta:
    """Prompt caching works without its obsolete beta header."""

    async def test_default_beta_omits_prompt_caching(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        header = mock.call_args.kwargs["headers"]["anthropic-beta"]
        assert "prompt-caching-2024-07-31" not in header
        assert "computer-use-" in header  # computer-use beta still present

    async def test_prompt_caching_false_omits_beta(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={
                "max_tokens": 4096,
                "temperature": 0.7,
                "thinking_budget": 0,
                "prompt_caching": False,
            },
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        header = mock.call_args.kwargs["headers"]["anthropic-beta"]
        assert "prompt-caching-2024-07-31" not in header

    async def test_grounding_cache_control_needs_no_beta(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopGroundingPointAgent(model_id="claude-opus-4-6")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert "headers" not in mock.call_args.kwargs
        assert mock.call_args.kwargs["messages"][0]["content"][0]["cache_control"] == {
            "type": "ephemeral"
        }


class TestModelToolMapping:
    """The model id decides the computer-use tool version and beta flag, and both
    are only observable in the request Anthropic receives."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-sonnet-4-6",
            "anthropic/claude-opus-4-8",
            "anthropic/claude-sonnet-5",
        ],
    )
    async def test_current_models_use_current_computer_use_tool(self, model_id, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        if model_id.removeprefix("anthropic/") in {"claude-opus-5", "claude-sonnet-5"}:
            monkeypatch.setattr(
                "lite.agents.models.claude.agent.acompletion_with_messages", mock
            )
        else:
            monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id=model_id,
            api_kwargs={"thinking_budget": 0, "prompt_caching": False},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        if model_id.removeprefix("anthropic/") in {"claude-opus-5", "claude-sonnet-5"}:
            assert mock.call_args.kwargs["tools"][0] == {
                "type": "computer_toolset_20260801",
                "configs": {
                    "zoom": {"enabled": False},
                    "cursor_position": {"enabled": False},
                },
            }
            assert "computer-use-2025-11-24" not in mock.call_args.kwargs.get(
                "headers", {}
            ).get("anthropic-beta", "")
        else:
            assert mock.call_args.kwargs["tools"][0]["type"] == "computer_20251124"
            assert mock.call_args.kwargs["headers"]["anthropic-beta"] == "computer-use-2025-11-24"

    @pytest.mark.parametrize("model_id", ["claude-opus-5", "claude-sonnet-5"])
    async def test_toolset_replays_screenshot_with_toolset_name(self, monkeypatch, model_id):
        mock = AsyncMock(side_effect=[
            _fake_completion_response(
                content=[{
                    "type": "tool_use", "id": "toolu_screenshot", "name": "screenshot",
                    "input": {}, "toolset_name": "computer",
                }],
                finish_reason="tool_calls",
            ),
            _fake_completion_response(content="done"),
        ])
        monkeypatch.setattr(
            "lite.agents.models.claude.agent.acompletion_with_messages", mock
        )
        agent = ClaudeDesktopUseAgent(model_id=model_id)
        await agent.sample(_FakeEnv(terminate_after=2), max_steps=2)

        _, wire_messages = _messages_for_anthropic(mock.call_args_list[1].kwargs["messages"])
        tool_use = next(b for m in wire_messages for b in m["content"]
                        if isinstance(m["content"], list) and b.get("type") == "tool_use")
        tool_result = next(b for m in wire_messages for b in m["content"]
                           if isinstance(m["content"], list) and b.get("type") == "tool_result")
        assert (tool_use["name"], tool_use["toolset_name"]) == ("screenshot", "computer")
        assert tool_result["tool_use_id"] == "toolu_screenshot"
        assert tool_result["toolset_name"] == "computer"
        assert any(b["type"] == "image" for b in tool_result["content"])

    @pytest.mark.parametrize("model_id", ["claude-opus-5", "claude-sonnet-5"])
    async def test_toolset_invalid_action_feedback_keeps_toolset_name(self, monkeypatch, model_id):
        mock = AsyncMock(side_effect=[
            _fake_completion_response(
                content=[{
                    "type": "tool_use", "id": "toolu_bad", "name": "left_click",
                    "input": {}, "toolset_name": "computer",
                }, {
                    "type": "tool_use", "id": "toolu_good", "name": "screenshot",
                    "input": {}, "toolset_name": "computer",
                }],
                finish_reason="tool_calls",
            ),
            _fake_completion_response(content="done"),
        ])
        monkeypatch.setattr(
            "lite.agents.models.claude.agent.acompletion_with_messages", mock
        )
        await ClaudeDesktopUseAgent(model_id=model_id).sample(
            _FakeEnv(terminate_after=2), max_steps=2
        )
        _, wire_messages = _messages_for_anthropic(mock.call_args_list[1].kwargs["messages"])
        result = next(b for m in wire_messages if isinstance(m["content"], list)
                      for b in m["content"] if b.get("type") == "tool_result")
        assert result["tool_use_id"] == "toolu_bad"
        assert result["toolset_name"] == "computer"
        assert result["is_error"] is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-future-model",
            "claude-opus-4-9",
            "claude-sonnet-4-7",
            "anthropic/claude-opus-4-9",
        ],
    )
    async def test_unknown_model_requires_mapping_or_explicit_override(self, model_id):
        """Resolved before the env is touched, so an unmapped id cannot reach the
        provider as a silently different tool version."""
        agent = ClaudeDesktopUseAgent(model_id=model_id)

        with pytest.raises(ValueError, match="Unsupported Claude model id"):
            await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

    @pytest.mark.parametrize(
        "api_kwargs",
        [
            {"computer_tool_version": "computer_20990101"},
            {"computer_use_beta_flag": "computer-use-2099-01-01"},
        ],
    )
    async def test_explicit_tool_override_requires_both_fields(self, api_kwargs):
        agent = ClaudeDesktopUseAgent(model_id="claude-future-model", api_kwargs=api_kwargs)

        with pytest.raises(ValueError, match="requires both"):
            await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

    async def test_unknown_model_override_is_used_in_paid_request_shape(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-future-model",
            api_kwargs={
                "computer_tool_version": "computer_20990101",
                "computer_use_beta_flag": "computer-use-2099-01-01",
                "thinking_budget": 0,
                "prompt_caching": False,
            },
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["tools"][0]["type"] == "computer_20990101"
        assert mock.call_args.kwargs["headers"]["anthropic-beta"] == "computer-use-2099-01-01"


class TestClaudeConfigRejection:
    """Reject stale or unknown Claude config instead of accepting silent no-ops."""

    def test_preserve_raw_response_is_not_a_claude_runtime_knob(self):
        with pytest.raises(TypeError, match="preserve_raw_response"):
            ClaudeDesktopUseAgent(
                model_id="claude-opus-4-6",
                preserve_raw_response=True,
            )

    def test_registry_rejects_stale_preserve_raw_response(self):
        with pytest.raises(TypeError, match="preserve_raw_response"):
            AgentRegistry.get(
                "claude@desktop@use",
                model_id="claude-opus-4-6",
                preserve_raw_response=True,
            )

    def test_registry_rejects_unknown_claude_config(self):
        with pytest.raises(TypeError, match="unknown_claude_config"):
            AgentRegistry.get(
                "claude@desktop@use",
                model_id="claude-opus-4-6",
                unknown_claude_config=True,
            )


# -----------------------------------------------------------------------------
# cache_breakpoints (rolling 3 on user + 1 on system)
# -----------------------------------------------------------------------------


class TestCacheBreakpoints:
    """up to 4 breakpoints total, rolling across recent user turns."""

    def test_cap_4_system_plus_rolling_3(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "u1"}]},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": [{"type": "text", "text": "u2"}]},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": [{"type": "text", "text": "u3"}]},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": [{"type": "text", "text": "u4"}]},
        ]
        inject_prompt_caching(messages, cap=4)

        # System wrapped into content list with cache_control
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0].get("cache_control") == {"type": "ephemeral"}
        # Last 3 user messages: last content block has cache_control
        for i in (3, 5, 7):
            last_block = messages[i]["content"][-1]
            assert last_block.get("cache_control") == {"type": "ephemeral"}, (
                f"msg {i} missing cache_control"
            )
        # Oldest user (index 1) should NOT have cache_control
        assert "cache_control" not in messages[1]["content"][-1]

    def test_cap_1_only_system(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "u1"}]},
            {"role": "user", "content": [{"type": "text", "text": "u2"}]},
        ]
        inject_prompt_caching(messages, cap=1)
        assert messages[0]["content"][0].get("cache_control") == {"type": "ephemeral"}
        assert "cache_control" not in messages[1]["content"][-1]
        assert "cache_control" not in messages[2]["content"][-1]

    def test_tool_messages_are_cacheable_body_history(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "u1"}]},
            {"role": "assistant", "content": "a1"},
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": [{"type": "text", "text": "tool result"}],
            },
        ]
        inject_prompt_caching(messages, cap=4)

        assert messages[0]["content"][0].get("cache_control") == {"type": "ephemeral"}
        assert messages[1]["content"][-1].get("cache_control") == {"type": "ephemeral"}
        assert messages[3]["content"][-1].get("cache_control") == {"type": "ephemeral"}

    def test_cache_breakpoints_clamp_to_anthropic_limit(self):
        messages = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": [{"type": "text", "text": f"u{i}"}]} for i in range(6)
        ]

        inject_prompt_caching(messages, cap=99)

        marked = [
            block
            for message in messages
            for block in (message["content"] if isinstance(message["content"], list) else [])
            if isinstance(block, dict) and block.get("cache_control")
        ]
        assert len(marked) == 4

    def test_cap_0_strips_cache_control(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "u", "cache_control": {"type": "ephemeral"}}],
            },
        ]
        inject_prompt_caching(messages, cap=0)
        assert "cache_control" not in messages[0]["content"][-1]

    def test_strip_images_preserves_cache_control_metadata(self):
        messages = [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

        redacted = strip_images(messages)

        image_block = redacted[0]["content"][0]
        assert image_block["image_url"] == {"url": "[base64 image omitted]"}
        assert image_block["cache_control"] == {"type": "ephemeral"}

    async def test_default_kwarg_injects_4_breakpoints(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6", system_prompt="SYS")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        messages = mock.call_args.kwargs["messages"]
        # System wrapped with cache_control
        sys_msg = messages[0]
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][0].get("cache_control") == {"type": "ephemeral"}
        # User msg's last content block has cache_control
        user_msg = messages[1]
        assert user_msg["content"][-1].get("cache_control") == {"type": "ephemeral"}


# -----------------------------------------------------------------------------
# token_efficient_tools_beta
# -----------------------------------------------------------------------------


class TestTokenEfficientToolsBeta:
    """token-efficient-tools-2025-02-19 beta is opt-in."""

    async def test_default_off_no_beta(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        header = mock.call_args.kwargs["headers"]["anthropic-beta"]
        assert "token-efficient-tools" not in header

    async def test_kwarg_true_adds_beta(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={
                "max_tokens": 4096,
                "temperature": 0.7,
                "thinking_budget": 0,
                "prompt_caching": False,
                "token_efficient_tools_beta": True,
            },
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        header = mock.call_args.kwargs["headers"]["anthropic-beta"]
        assert "token-efficient-tools-2025-02-19" in header


# -----------------------------------------------------------------------------
# only_n_most_recent_images + image_truncation_threshold
# -----------------------------------------------------------------------------


class TestImageHistoryTruncation:
    """drop old image_url blocks keeping only N most recent."""

    def test_default_none_no_truncation(self):
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "a"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "b"}}]},
        ]
        out = filter_to_n_most_recent_images(messages, images_to_keep=10)
        assert out == messages

    def test_keeps_last_n(self):
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "a"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "b"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "c"}}]},
        ]
        out = filter_to_n_most_recent_images(messages, images_to_keep=2)
        urls = [
            b["image_url"]["url"]
            for m in out
            for b in (m.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "image_url"
        ]
        assert urls == ["b", "c"]

    def test_threshold_rounds_down(self):
        # 3 total images, keep 1 → 2 to remove; threshold=3 rounds to 0 remove
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "a"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "b"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "c"}}]},
        ]
        out = filter_to_n_most_recent_images(messages, images_to_keep=1, min_removal_threshold=3)
        assert out == messages  # 2 % 3 → 2 - 2 = 0 remove

    def test_tool_message_is_preserved_when_its_image_is_pruned(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": [{"type": "image_url", "image_url": {"url": "old"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_2",
                "content": [{"type": "image_url", "image_url": {"url": "new"}}],
            },
        ]

        out = filter_to_n_most_recent_images(messages, images_to_keep=1)

        assert [m["tool_call_id"] for m in out] == ["toolu_1", "toolu_2"]
        assert out[0]["content"] == [{"type": "text", "text": "[Image omitted]"}]
        assert out[1]["content"] == [{"type": "image_url", "image_url": {"url": "new"}}]


# -----------------------------------------------------------------------------
# Tool schema (wrapped — liteLLM path; references-aligned sole path)
# -----------------------------------------------------------------------------


class TestToolSchema:
    def test_provider_native_rejects_generic_metadata_for_tool_surface(self):
        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            metadata=LiteGenericMetadata(dims=()),
        )

        with pytest.raises(TypeError, match="ClaudeDesktopUseAgent requires LiteCUAMetadata"):
            agent._build_tools(agent._computer_tool_config(), display_w=800, display_h=600)

    async def test_wrapped_schema_sent(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        computer = tools[0]
        assert "function" in computer
        assert computer["function"]["parameters"]["display_width_px"] == 800
        assert computer["function"]["parameters"]["display_height_px"] == 600

    async def test_hd_screenshot_display_matches_sent_image_size(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _FakeEnv(terminate_after=1)
        env._shot = png_bytes(1920, 1080)
        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        await agent.sample(env, max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        params = tools[0]["function"]["parameters"]
        assert (params["display_width_px"], params["display_height_px"]) == (1456, 819)

        first_user = next(m for m in mock.call_args.kwargs["messages"] if m["role"] == "user")
        image_url = next(
            block["image_url"]["url"]
            for block in first_user["content"]
            if block.get("type") == "image_url"
        )
        sent_png = base64.b64decode(image_url.split(",", 1)[1])
        assert Image.open(io.BytesIO(sent_png)).size == (1456, 819)

    async def test_desktop_tools_map_through_litellm_anthropic_transform(self, monkeypatch):
        """The tools the loop actually sends survive liteLLM's Anthropic transform.

        Reads the request payload rather than the assembly helper, so the
        versioned computer type and display dims are checked on what Anthropic
        would really receive.
        """
        config_mod = pytest.importorskip("litellm.llms.anthropic.chat.transformation")
        agent = ClaudeDesktopUseAgent(
            metadata=LiteCUAMetadata(
                extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")]
            ),
            api_kwargs={
                "computer_tool_version": "computer_20251124",
                "computer_use_beta_flag": "computer-use-2025-11-24",
            },
        )
        hd_env = _FakeEnv(terminate_after=1)
        hd_env._shot = png_bytes(1920, 1080)  # -> 1456x819 after Claude API resize

        tools = await _tools_sent(agent, monkeypatch, hd_env)

        mapped, mcp_servers = config_mod.AnthropicConfig()._map_tools(tools)

        assert mcp_servers == []
        computer = next(tool for tool in mapped if tool.get("name") == "computer")
        assert computer["type"] == "computer_20251124"
        assert computer["display_width_px"] == 1456
        assert computer["display_height_px"] == 819
        assert {tool["name"] for tool in mapped} >= {"computer", "response"}

    async def test_valid_actions_computer_verb_keeps_native_tool(self, monkeypatch):
        """Non-empty ``valid_actions`` can't be honored per-action for Claude —
        Anthropic's native ``computer_*`` tool's action enum is hard-coded
        server-side. The native tool is kept whole; finish tools surface only as
        schema-backed extras. ``[]`` drops the native tool; ``None`` keeps it."""
        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6", metadata=LiteCUAMetadata(valid_actions=["left_click"])
        )

        tools = await _tools_sent(agent, monkeypatch)

        assert any(str(t.get("type", "")).startswith("computer_") for t in tools)

    @pytest.mark.parametrize("model_id", ["claude-opus-4-6", "claude-opus-5", "claude-sonnet-5"])
    async def test_grounding_left_click_schema_is_action_space_owned(self, monkeypatch, model_id):
        """Grounding declares only the action-space-owned left_click function."""
        agent = ClaudeDesktopGroundingPointAgent(model_id=model_id)
        owner_schema = type(agent.action_space).get_tool_schema("left_click")
        assert owner_schema is not None

        tools = await _tools_sent(agent, monkeypatch)

        # Exactly the owner schema, wrapped in liteLLM's OpenAI-style envelope.
        assert tools == [
            {
                "type": "function",
                "function": {
                    "name": "left_click",
                    "description": owner_schema["function"].get("description", ""),
                    "parameters": owner_schema["function"]["parameters"],
                },
            }
        ]
        assert ClaudeDesktopGroundingPointActionSpace.get_tool_names() == frozenset({"left_click"})
        assert ClaudeDesktopGroundingPointActionSpace.get_declared_action_schema_names() == (
            frozenset({"left_click"})
        )
        assert all(not str(tool.get("type", "")).startswith("computer_") for tool in tools)
        coordinate = tools[0]["function"]["parameters"]["properties"]["coordinate"]
        assert coordinate["minItems"] == 2
        assert coordinate["maxItems"] == 2

    def test_grounding_parser_rejects_inherited_desktop_actions(self):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("screenshot", "{}", id_="toolu_screenshot_1"),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopGroundingPointActionSpace(),
            resolution=(1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared tool_call screenshot"

    def test_grounding_parser_rejects_bool_left_click_coordinates(self):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call(
                    "left_click", '{"coordinate": [false, 10]}', id_="toolu_left_click_1"
                ),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopGroundingPointActionSpace(),
            resolution=(1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "left_click requires valid coordinate"

    def test_grounding_action_space_unknown_action_raises_by_default(self):
        with pytest.raises(ValueError, match="unknown Claude desktop grounding action: move"):
            ClaudeDesktopGroundingPointActionSpace().convert_tool_calls_from_agent(
                [{"action": "move", "coordinate": [10, 20]}],
                resolution=(1024, 768),
            )

    async def test_valid_actions_empty_drops_native_tool(self, monkeypatch):
        """``valid_actions=[]`` (used by browsergym text+bid configs)
        suppresses the native ``computer_*`` tool — only env-supplied
        function tools surface. Mirrors GPT's ``valid_actions: []``
        convention through the action space's public
        ``filter_tool_schemas_for_valid_actions`` hook."""
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            metadata=LiteCUAMetadata(
                valid_actions=[],
                extra_tool_schemas=[
                    make_tool_schema(
                        "click",
                        description="Click an element by bid.",
                        parameters={
                            "type": "object",
                            "properties": {"bid": {"type": "string"}},
                            "required": ["bid"],
                        },
                    ),
                ],
            ),
            model_id="claude-opus-4-6",
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        # No native computer tool present (any ``computer_*`` versioned type).
        assert all(not str(t.get("type", "")).startswith("computer_") for t in tools)
        # Only the env-supplied function tool surfaces.
        assert len(tools) == 1
        assert tools[0]["name"] == "click"

    async def test_valid_actions_none_withholds_finish_tools_osworld_parity(self, monkeypatch):
        """OSWORLD PARITY / DRIFT GUARD (mirrors GPT). Finish tools
        (``response``/``terminate``) are a STRICT env-gated opt-in layer, NOT part
        of the native-enum ``None``=expose-all contract — the opaque computer tool
        can't carry them. So ``valid_actions=None`` desktop envs (osworld, which
        terminates via an empty action) get ONLY the native computer tool; offering
        a ``terminate`` tool would drift behavior + recorded trajectory format.
        Browser envs opt in by resolving a schema-backed ``response`` extra tool."""
        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")  # valid_actions=None

        tools = await _tools_sent(agent, monkeypatch)

        fn_names = {
            t.get("name") for t in tools if not str(t.get("type", "")).startswith("computer_")
        }
        assert "response" not in fn_names and "terminate" not in fn_names, fn_names
        assert any(str(t.get("type", "")).startswith("computer_") for t in tools)

    async def test_valid_actions_does_not_opt_in_finish_tools(self, monkeypatch):
        """``valid_actions`` is GUI-only for Claude; finish tools are schema-backed
        extras and must not appear without extra_tool_schemas."""
        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6", metadata=LiteCUAMetadata(valid_actions=["click"])
        )

        tools = await _tools_sent(agent, monkeypatch)

        fn_names = {
            t.get("name") for t in tools if not str(t.get("type", "")).startswith("computer_")
        }
        assert "response" not in fn_names and "terminate" not in fn_names
        assert any(str(t.get("type", "")).startswith("computer_") for t in tools)


class TestToolChoice:
    """``api_kwargs.tool_choice`` plumbing — string forms pass through,
    dict forms pass through. Mirrors GPT's ``tool_choice`` test."""

    async def test_tool_choice_required_passes_through(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={"max_tokens": 4096, "thinking_budget": 0, "tool_choice": "required"},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["tool_choice"] == "required"

    async def test_tool_choice_omitted_by_default(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert "tool_choice" not in mock.call_args.kwargs

    @pytest.mark.parametrize(
        "tool_choice",
        ["required", "any", {"type": "any"}, {"type": "tool", "name": "response"}],
    )
    async def test_forced_tool_choice_rejects_thinking_budget(self, monkeypatch, tool_choice):
        """Anthropic API rejects ``thinking`` + forced ``tool_choice`` together.
        The Claude loop fails before making the paid request."""
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={
                "max_tokens": 8192,
                "thinking_budget": 4096,
                "tool_choice": tool_choice,
            },
        )
        with pytest.raises(ValueError, match="thinking_budget cannot be used"):
            await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_count == 0


class TestEffort:
    """``api_kwargs.effort`` -> top-level ``output_config``."""

    async def test_effort_rides_top_level_output_config(self, monkeypatch):
        """``effort`` must reach the API as a TOP-LEVEL ``output_config``.

        The two obvious alternatives both fail silently or loudly: litellm drops
        an unknown key out of ``extra_body``, and its own ``reasoning_effort``
        param rewrites to ``thinking:{type:"enabled"}``, which Claude 4.7+
        rejects with a 400.
        """
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-7")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        kwargs = mock.call_args.kwargs
        assert kwargs["output_config"] == {"effort": "medium"}
        assert "reasoning_effort" not in kwargs
        assert "extra_body" not in kwargs

    @pytest.mark.parametrize("model_id", ["claude-opus-4-6", "claude-opus-5", "claude-sonnet-5"])
    async def test_grounding_drops_to_low_effort(self, monkeypatch, model_id):
        """Grounding is one click with no reasoning wanted, so it must not
        inherit the family's ``medium``."""
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)
        monkeypatch.setattr("lite.agents.models.claude.agent.acompletion_with_messages", mock)

        agent = ClaudeDesktopGroundingPointAgent(model_id=model_id)
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["output_config"] == {"effort": "low"}
        assert mock.call_args.kwargs["max_tokens"] == 1024
        assert "thinking" not in mock.call_args.kwargs

    async def test_user_supplied_effort_reaches_the_wire(self, monkeypatch):
        """A yaml-supplied override must win over the family default.

        Without this, nothing proves the one non-default effort in the config
        tree (``claude/default/osworld_2.yaml``: ``max``) is doing anything --
        the default test alone would pass on a hardcoded "medium".
        """
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-7",
            api_kwargs={"max_tokens": 64000, "effort": "max", "thinking_budget": 32000},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["output_config"] == {"effort": "max"}

    async def test_effort_omitted_when_unset(self, monkeypatch):
        """``effort=0``/None must not emit an empty ``output_config``."""
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-7",
            api_kwargs={"max_tokens": 4096, "effort": None},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert "output_config" not in mock.call_args.kwargs


class TestModelRejectsTemperature:
    """The agent requires omitted temperature for adaptive-only models."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
            "anthropic/claude-opus-4-7", "anthropic/claude-sonnet-5",
        ],
    )
    async def test_adaptive_model_rejects_temperature(self, monkeypatch, model_id):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)
        monkeypatch.setattr(
            "lite.agents.models.claude.agent.acompletion_with_messages", mock
        )

        agent = ClaudeDesktopUseAgent(
            model_id=model_id,
            api_kwargs={"max_tokens": 4096, "temperature": 0.7},
        )
        with pytest.raises(ValueError, match="rejects explicit temperature"):
            await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_count == 0

    @pytest.mark.parametrize("model_id", ["claude-opus-4-6", "claude-sonnet-4-6"])
    async def test_temperature_compatible_models_keep_temperature(self, monkeypatch, model_id):
        """Some current Claude 4 models accept temperature; explicitly disable thinking
        (defaults set ``thinking_budget=2048`` which would force
        ``temperature=1.0``)."""
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id=model_id,
            api_kwargs={"max_tokens": 4096, "temperature": 0.7, "thinking_budget": 0},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["temperature"] == 0.7

    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
            "anthropic/claude-opus-4-8", "anthropic/claude-sonnet-5",
        ],
    )
    async def test_adaptive_model_with_thinking_omits_temperature(self, monkeypatch, model_id):
        """Adaptive thinking must not add fixed-budget temperature policy."""
        mock = AsyncMock(return_value=_fake_completion_response())
        if model_id.removeprefix("anthropic/") in {"claude-opus-5", "claude-sonnet-5"}:
            monkeypatch.setattr(
                "lite.agents.models.claude.agent.acompletion_with_messages", mock
            )
        else:
            monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id=model_id,
            api_kwargs={"max_tokens": 8192, "thinking_budget": 4096},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        # Thinking is on in adaptive mode; no temperature is emitted.
        assert mock.call_args.kwargs["thinking"] == {"type": "adaptive"}
        assert "temperature" not in mock.call_args.kwargs

    async def test_fixed_budget_thinking_requires_enough_max_tokens(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={"max_tokens": 2048, "thinking_budget": 2048},
        )
        with pytest.raises(ValueError, match="max_tokens must be greater than thinking_budget"):
            await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_count == 0

    async def test_fixed_budget_thinking_requires_temperature_one(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={"max_tokens": 4096, "thinking_budget": 2048, "temperature": 0.7},
        )
        with pytest.raises(ValueError, match="requires temperature=1.0"):
            await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_count == 0

    async def test_fixed_budget_thinking_payload_is_explicit(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={"max_tokens": 4096, "thinking_budget": 2048},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["max_tokens"] == 4096
        assert mock.call_args.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert mock.call_args.kwargs["temperature"] == 1.0


class TestStepStatusFromFinishReason:
    """The provider's token-budget stop signal must reach ``LiteRLStep.status``.

    The loop used to hardcode ``STATUS_COMPLETED``, so a response cut off at
    ``max_tokens`` was recorded as a clean finish and the rollout segmenter's
    ``truncated`` metric never saw it.
    """

    async def test_length_finish_reason_marks_the_step_truncated(self, monkeypatch):
        tool_call = _fake_tool_call(
            "computer",
            '{"action": "screenshot"}',
            id_="tc_desktop_1",
        )
        mock = AsyncMock(
            return_value=_fake_completion_response(
                tool_calls=[tool_call],
                finish_reason="length",
            )
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(_FakeEnv(terminate_after=1), max_steps=3)

        assert result.terminated is True
        assert [s.status for s in result.steps] == [STATUS_TRUNCATED]

    async def test_stop_finish_reason_stays_completed(self, monkeypatch):
        tool_call = _fake_tool_call(
            "computer",
            '{"action": "screenshot"}',
            id_="tc_desktop_1",
        )
        mock = AsyncMock(return_value=_fake_completion_response(tool_calls=[tool_call]))
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(_FakeEnv(terminate_after=1), max_steps=3)

        assert [s.status for s in result.steps] == [STATUS_COMPLETED]

    async def test_empty_choices_is_a_failed_step_not_a_completed_one(self, monkeypatch):
        """An empty ``choices`` carries no assistant message, so it is a FAILURE.

        This test used to be parametrized over two shapes, both asserting
        ``STATUS_COMPLETED``:

        * ``"empty choices list"`` -- REACHABLE. ``ModelResponse(choices=[])``
          constructs, i.e. a proxy answering ``{"choices": []}`` produces it.
          Recording it as completed was a silent wrong status feeding the rollout
          segmenter's severity ranking and the engine's validity test, so the
          parameter survives here with the corrected expectation.
        * ``"no choices attribute"`` -- NOT reachable, and deleted with the
          ``hasattr(response, "choices")`` arm it pinned: ``choices`` is a
          required field on ``litellm.types.utils.ModelResponse`` and a
          default-constructed instance already carries a non-empty list, so only
          a hand-built ``SimpleNamespace`` ever had that shape.

        A real ``ModelResponse`` is used rather than a fake precisely so the
        reachability claim above is exercised, not asserted.
        """
        from litellm.types.utils import ModelResponse

        response = ModelResponse(choices=[])
        assert response.choices == []
        monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=response))

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(_FakeEnv(terminate_after=1), max_steps=3)

        assert [s.status for s in result.steps] == [STATUS_FAILED]


class TestCanonicalPersistence:
    async def test_step_prompt_logs_actual_api_messages_after_image_pruning(self, monkeypatch):
        tool_call = _fake_tool_call(
            "computer",
            '{"action": "screenshot"}',
            id_="toolu_computer_1",
        )
        mock = AsyncMock(
            side_effect=[
                _fake_completion_response(tool_calls=[tool_call]),
                _fake_completion_response("final"),
            ]
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            only_n_most_recent_images=1,
            image_truncation_threshold=1,
            api_kwargs={
                "max_tokens": 4096,
                "thinking_budget": 0,
                "prompt_caching": False,
            },
        )
        result = await agent.sample(_FakeEnv(terminate_after=99), max_steps=3)

        second_api_messages = mock.call_args_list[1].kwargs["messages"]
        assert result.steps[1].prompt == json.dumps(
            strip_images(second_api_messages),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        assert "[Image omitted]" in result.steps[1].prompt
        assert result.steps[1].prompt.count("[base64 image omitted]") == 1

    async def test_max_steps_exhaustion_marks_truncated_with_paired_feedback(self, monkeypatch):
        tool_call = _fake_tool_call(
            "computer",
            '{"action": "screenshot"}',
            id_="tc_desktop_1",
        )
        mock = AsyncMock(return_value=_fake_completion_response(tool_calls=[tool_call]))
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(_RecordingFakeEnv(terminate_after=99), max_steps=1)

        assert result.terminated is False
        assert result.truncated is True
        assert result.steps[-1].status == "truncated"
        assert mock.call_count == 1
        assert [m["role"] for m in result.lite_sample.messages] == ["user", "assistant", "tool"]
        tool_msg = result.lite_sample.messages[-1]
        assert tool_msg["tool_call_id"] == "call_0000"
        assert tool_msg["content"] == [
            {"type": "image", "index": 1},
            {"type": "text", "text": "instr"},
        ]

    async def test_terminal_step_with_no_results_ends_on_assistant_tool_call(self, monkeypatch):
        tool_call = _fake_tool_call(
            "computer",
            '{"action": "screenshot"}',
            id_="toolu_computer_1",
        )
        mock = AsyncMock(return_value=_fake_completion_response(tool_calls=[tool_call]))
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _TerminalNoResultsEnv()
        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(env, max_steps=2)

        assert result.terminated is True
        assert result.truncated is False
        assert mock.call_count == 1
        assert len(env.actions_seen) == 1
        assert env.actions_seen[0] == [
            make_tool_call(
                "computer",
                {"actions": [{"action": "screenshot"}]},
                call_id="call_0000",
            )
        ]

        assert [m["role"] for m in result.lite_sample.messages] == ["user", "assistant"]
        assistant = result.lite_sample.messages[-1]
        assert assistant["tool_calls"] == [
            make_tool_call(
                "computer",
                {"actions": [{"action": "screenshot"}]},
                call_id="call_0000",
            )
        ]

    @pytest.mark.parametrize(
        "terminate_after,terminal",
        [
            pytest.param(99, False, id="mid-episode-history-feedback"),
            pytest.param(1, True, id="terminal-feedback"),
        ],
    )
    async def test_out_of_order_env_results_persist_in_tool_call_order(
        self,
        terminate_after,
        terminal,
        monkeypatch,
    ):
        """Multi-result ``role:"tool"`` ordering plus text/error/metadata projection.

        The env answers the two canonical calls in the opposite order. The
        env-result boundary re-pairs them once, so the persisted messages follow
        the assistant's own call order on both the mid-episode feedback path and
        the terminal-feedback path.
        """
        extra = make_tool_schema(
            "bash",
            description="Run a command.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        first_resp = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("computer", '{"action": "screenshot"}', id_="toolu_computer_1"),
                _fake_tool_call("bash", "{}", id_="toolu_bash_1"),
            ]
        )
        responses = [first_resp] if terminal else [first_resp, _fake_completion_response()]
        mock = AsyncMock(side_effect=responses)
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        result = await agent.sample(
            _ReversedResultOrderEnv(terminate_after=terminate_after),
            max_steps=3,
        )

        # A terminal first step persists feedback through the terminal appender
        # and never asks the provider again; the mid-episode step goes through
        # the next-turn history appender instead.
        assert mock.call_count == (1 if terminal else 2)
        assert result.terminated is True
        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert [tool_call_name(call) for call in assistant["tool_calls"]] == ["computer", "bash"]

        tool_messages = [m for m in result.lite_sample.messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_messages] == [
            tool_call_id(call) for call in assistant["tool_calls"]
        ]
        # Images stay indexed against the aligned order, and metadata/error are
        # projected into the same canonical message.
        assert tool_messages[0]["content"] == [
            {"type": "image", "index": 1},
            {"type": "text", "text": "computer screen text"},
            {"type": "metadata", "data": {"source": "screen"}},
        ]
        assert tool_messages[1]["content"] == [
            {
                "type": "text",
                "text": "bash stdout\n\n## Error from previous action:\nexit status 1",
            },
        ]

    async def test_non_terminal_step_missing_a_result_fails_loudly(self, monkeypatch):
        """A non-terminal step owes one result per canonical call.

        The env-result boundary is the single place this is checked, so a
        dropped result must raise there instead of silently persisting a turn
        with fewer ``role:"tool"`` messages than tool calls.
        """
        extra = make_tool_schema(
            "bash",
            description="Run a command.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        first_resp = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("computer", '{"action": "screenshot"}', id_="toolu_computer_1"),
                _fake_tool_call("bash", "{}", id_="toolu_bash_1"),
            ]
        )
        monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=first_resp))

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        env = _DropsOneResultEnv(terminate_after=99)
        with pytest.raises(RuntimeError, match="do not match tool_calls"):
            await agent.sample(env, max_steps=3)
        assert env.closed is True

    def test_desktop_parser_preserves_extra_native_relative_order(self):
        response = _fake_completion_response(
            tool_calls=[
                # A REAL bash call: this test is about relative order, not validation.
                _fake_tool_call("bash", '{"command": "ls"}', id_="toolu_bash_1"),
                _fake_tool_call("computer", '{"action": "screenshot"}', id_="toolu_computer_1"),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
            extra_tool_names=frozenset({"bash"}),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["bash", "computer"]
        assert tool_call_arguments(calls[1]) == {"actions": [{"action": "screenshot"}]}
        assert [tool_call_id(call) for call in calls] == ["call_0000", "call_0001"]
        assert all("tool_call_id" not in call for call in calls)

    def test_desktop_parser_prefers_tool_calls_view_over_content_tool_use(self):
        response = _fake_completion_response(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_content",
                    "name": "computer",
                    "input": {"action": "screenshot"},
                }
            ],
            tool_calls=[
                _fake_tool_call("computer", '{"action": "screenshot"}', id_="toolu_calls"),
            ],
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        ).message

        actions = tool_call_arguments(msg["tool_calls"][0])["actions"]
        assert actions == [{"action": "screenshot"}]

    def test_desktop_parser_drops_malformed_and_empty_arguments(self, caplog):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("bash", "{not json", id_="toolu_bad_1"),
                _fake_tool_call("bash", None, id_="toolu_none_1"),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
            extra_tool_names=frozenset({"bash"}),
        ).message

        # BOTH are dropped, and neither reaches the env: ``{not json`` fails to parse, and
        # LiteLLM normalizes ``arguments=None`` to ``""``, which reaches the same
        # malformed-JSON branch. The provider never handed over an argument object, so
        # there is nothing for the env to answer.
        assert msg["tool_calls"] == []
        assert "malformed arguments" in caplog.text

    def test_desktop_parser_marks_undeclared_tool_call_as_model_output_error(self, caplog):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("goto", '{"url": "https://example.com"}', id_="toolu_goto_1"),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared tool_call goto"
        assert "Ignoring undeclared Claude tool_call: goto" in caplog.text

    def test_desktop_parser_routes_invalid_active_extra_to_env_feedback(self):
        """A malformed active extra survives the parser and reaches env ingress.

        Mixed output — one valid GUI call plus one bad-argument extra — must not
        lose the extra: the parser routes an advertised env tool by name, and
        ``prepare_env_tool_calls`` is what names the bad argument back to the
        model. Dropping it here would delete that answer.
        """
        from lite.gym.utils.feedback.ingress import prepare_env_tool_calls

        report_schema = make_tool_schema(
            "report_infeasible",
            description="Report infeasible.",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        )
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("computer", '{"action": "screenshot"}', id_="toolu_computer_1"),
                _fake_tool_call("report_infeasible", "{}", id_="toolu_report_1"),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
            extra_tool_names=frozenset({"report_infeasible"}),
        ).message

        assert [tool_call_name(call) for call in msg["tool_calls"]] == [
            "computer",
            "report_infeasible",
        ]
        assert pop_model_output_error(msg) is None

        routed, errors = prepare_env_tool_calls(
            msg["tool_calls"],
            LiteCUAMetadata(extra_tool_schemas=[report_schema]),
        )
        assert all(call["name"] != "report_infeasible" for call, _ in routed)
        assert set(errors) == {"call_0001"}
        assert "report_infeasible" in errors["call_0001"].message

    def test_desktop_parser_merges_adjacent_computer_tool_items(self):
        """Claude's native computer tool carries ONE action per ``tool_use``.

        A turn's parallel blocks are therefore a run of consecutive GUI actions,
        which merges into a single canonical ``computer{actions:[...]}`` batch —
        one env dispatch, one screenshot. ``sample()`` maps the N provider blocks
        back onto that one canonical call by run (see
        ``test_desktop_parallel_blocks_share_one_canonical_call_and_one_image``).
        """
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("computer", '{"action": "screenshot"}', id_="toolu_computer_1"),
                _fake_tool_call(
                    "computer", '{"action": "type", "text": "hello"}', id_="toolu_computer_2"
                ),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["computer"]
        assert [tool_call_id(call) for call in calls] == ["call_0000"]
        assert tool_call_arguments(calls[0])["actions"] == [
            {"action": "screenshot"},
            {"action": "type", "text": "hello"},
        ]

    def test_desktop_parser_rejects_replayed_computer_when_request_hid_native(self):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("computer", '{"action": "screenshot"}', id_="toolu_computer_1"),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
            active_provider_tool_names=frozenset(),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared tool_call computer"

    @pytest.mark.parametrize(
        "payload",
        [
            {"action": "left_click"},
            {"action": "left_click", "coordinate": [False, 10]},
            {"action": "move"},
            {"action": "triple_click"},
            {"action": "drag", "start_coordinate": [1, 2]},
            {"action": "drag", "start_coordinate": [None, 2], "coordinate": [10, 10]},
        ],
    )
    def test_desktop_parser_marks_malformed_native_coordinates_as_model_output_error(self, payload):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("computer", json.dumps(payload), id_="toolu_computer_1"),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg)

    @pytest.mark.parametrize(
        "tool_name,payload",
        [
            ("computer", {"action": "key", "text": "Down shift"}),
            ("key", {"text": "Down shift"}),
        ],
    )
    def test_desktop_parser_reports_malformed_key_to_model(self, tool_name, payload):
        response = _fake_completion_response(
            tool_calls=[_fake_tool_call(tool_name, json.dumps(payload), id_="toolu_bad_key")]
        )

        parsed = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        )

        assert parsed.message["tool_calls"] == []
        assert "unknown key token" in parsed.provider_errors["toolu_bad_key"]
        assert pop_model_output_error(parsed.message)

    def test_desktop_parser_marks_malformed_scaled_coordinate_as_model_output_error(self):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call(
                    "computer",
                    '{"action": "left_click", "coordinate": ["x", 5]}',
                    id_="toolu_computer_1",
                ),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=2.0,
            scale_y=2.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg)

    def test_desktop_parser_marks_malformed_provider_scaled_coordinate_as_model_output_error(self):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call(
                    "left_click",
                    '{"coordinate": ["x", 5]}',
                    id_="toolu_left_click_1",
                ),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=2.0,
            scale_y=2.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg)

    def test_desktop_merge_marks_malformed_provider_scaled_coordinate_as_error(self):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call(
                    "left_click",
                    '{"coordinate": ["x", 5]}',
                    id_="toolu_left_click_1",
                ),
            ],
        )

        parsed = parse_response_with_provenance(
            response,
            scale_x=2.0,
            scale_y=2.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
            extra_tool_names=frozenset(),
        )

        assert tuple(p.canonical_call_id for p in parsed.provider_call_provenance) == (None,)
        provider_error = parsed.provider_errors["toolu_left_click_1"]
        assert "requires valid coordinate" in provider_error
        assert "This tool_use was not executed" in provider_error
        assert "No action was executed" not in provider_error

    def test_desktop_parallel_blocks_share_one_canonical_call_and_one_image(self):
        """The N provider blocks of a merged run map onto ONE canonical call.

        The Anthropic API still requires a ``tool_result`` per ``tool_use``, so
        all N get one — but only the LAST carries the turn's screenshot, since
        they all describe the same canonical call and the same post-batch screen.
        """
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call(
                    "computer",
                    '{"action": "left_click", "coordinate": [10, 10]}',
                    id_="toolu_click",
                ),
                _fake_tool_call(
                    "computer",
                    '{"action": "type", "text": "hello"}',
                    id_="toolu_type",
                ),
            ]
        )
        parsed = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        )

        provenance = parsed.provider_call_provenance
        assert tuple(p.canonical_call_id for p in provenance) == ("call_0000", "call_0000")
        assert [p.is_final_for_canonical for p in provenance] == [False, True]

    def test_desktop_dropped_middle_provider_call_does_not_get_screenshot(self):
        """R22 red: provider drop mask must compose with action-batch provenance.

        Provider emits ``computer, undeclared_tool, computer``. The parser drops
        the middle provider call, then the two surviving GUI calls become
        adjacent and merge into one canonical ``computer`` action-batch call. The third
        provider call did execute inside ``call_0000``, so only it should carry
        the screenshot.
        """
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call(
                    "computer",
                    '{"action": "left_click", "coordinate": [10, 10]}',
                    id_="toolu_click",
                ),
                _fake_tool_call("goto", '{"url": "https://example.com"}', id_="toolu_goto"),
                _fake_tool_call(
                    "computer",
                    '{"action": "type", "text": "hello"}',
                    id_="toolu_type",
                ),
            ]
        )
        parsed = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        )

        provenance = parsed.provider_call_provenance
        assert tuple(p.canonical_call_id for p in provenance) == (
            "call_0000",
            None,
            "call_0000",
        )
        assert [p.is_final_for_canonical for p in provenance] == [False, False, True]

    def test_desktop_extra_between_computer_runs_gets_its_own_canonical_call(self):
        """A standalone extra breaks the run: batch, extra, batch — never merged
        across, and each segment keeps its own ``call_id`` and its own image."""
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call(
                    "computer",
                    '{"action": "left_click", "coordinate": [10, 10]}',
                    id_="toolu_click",
                ),
                _fake_tool_call(
                    "computer",
                    '{"action": "type", "text": "hello"}',
                    id_="toolu_type_1",
                ),
                _fake_tool_call("bash", '{"command": "ls"}', id_="toolu_bash"),
                _fake_tool_call(
                    "computer",
                    '{"action": "type", "text": "again"}',
                    id_="toolu_type_2",
                ),
            ]
        )
        parsed = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
            extra_tool_names=frozenset({"bash"}),
        )

        provenance = parsed.provider_call_provenance
        assert tuple(p.canonical_call_id for p in provenance) == (
            "call_0000",
            "call_0000",
            "call_0001",
            "call_0002",
        )
        assert [i for i, p in enumerate(provenance) if p.is_final_for_canonical] == [
            1,
            2,
            3,
        ]

    def test_desktop_parser_preserves_extra_between_computer_tools(self):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("computer", '{"action": "screenshot"}', id_="toolu_computer_1"),
                _fake_tool_call("bash", '{"command": "pwd"}', id_="toolu_bash_1"),
                _fake_tool_call(
                    "computer", '{"action": "type", "text": "hello"}', id_="toolu_computer_2"
                ),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
            extra_tool_names=frozenset({"bash"}),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["computer", "bash", "computer"]
        assert tool_call_arguments(calls[1]) == {"command": "pwd"}
        assert [tool_call_id(call) for call in calls] == [
            "call_0000",
            "call_0001",
            "call_0002",
        ]

    def test_desktop_parser_does_not_promote_unknown_native_computer_action_to_extra(self):
        response = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("computer", '{"action": "goto", "url": "https://example.com"}'),
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
            extra_tool_names=frozenset({"goto"}),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "unknown Claude native computer action: goto"

    def test_desktop_action_space_unknown_action_raises_by_default(self):
        with pytest.raises(ValueError, match="unknown Claude native computer action: goto"):
            ClaudeDesktopActionSpace().convert_tool_calls_from_agent(
                [{"action": "goto", "url": "https://example.com"}],
                resolution=(1024, 768),
            )

    def test_desktop_content_tool_use_nonobject_input_is_model_output_error(self):
        response = _fake_completion_response(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_bad",
                    "name": "computer",
                    "input": ["not", "object"],
                }
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "malformed tool_use input for computer"

    def test_desktop_content_tool_use_missing_provider_id_is_model_output_error(self):
        response = _fake_completion_response(
            content=[
                {
                    "type": "tool_use",
                    "name": "computer",
                    "input": {"action": "screenshot"},
                }
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "missing provider id for computer"

    def test_provider_tool_call_without_model_dump_is_rejected(self):
        """A tool_call record that is not a liteLLM object has no readable shape;
        the parser refuses it instead of guessing at attributes."""
        response = _fake_completion_response(
            content=[],
            tool_calls=[
                SimpleNamespace(
                    id="toolu_fake",
                    function=SimpleNamespace(
                        name="computer",
                        arguments='{"action": "screenshot"}',
                    ),
                )
            ],
        )

        with pytest.raises(ValueError, match="LiteLLM tool_call with model_dump"):
            parse_response_with_provenance(
                response,
                scale_x=1.0,
                scale_y=1.0,
                action_space=ClaudeDesktopActionSpace(),
                resolution=(1024, 768),
            )

    def test_provider_tool_call_with_non_object_function_is_rejected(self):
        """A ``function`` that is not an object is a broken tool_call envelope.

        LiteLLM builds this shape from a proxy response carrying
        ``{"function": null}``; the model never authors the envelope. So it stays
        a loud ``ValueError`` (a wrong-object-type signal) instead of becoming
        ``ModelToolCallParseError`` model-visible feedback: the parse boundary
        must not tell the model its output was malformed when the transport
        broke. The malformed-model-output path is
        ``test_desktop_parser_drops_malformed_and_empty_arguments`` above.
        """
        response = _fake_completion_response(
            content=[],
            tool_calls=[
                ChatCompletionMessageToolCall(id="toolu_1", type="function", function=None)
            ],
        )

        with pytest.raises(ValueError, match="id, function.name, and function.arguments"):
            parse_response_with_provenance(
                response,
                scale_x=1.0,
                scale_y=1.0,
                action_space=ClaudeDesktopActionSpace(),
                resolution=(1024, 768),
            )

    def test_desktop_provider_tool_uses_preserve_falsy_malformed_content_input(self):
        response = _fake_completion_response(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_empty",
                    "name": "empty",
                    "input": [],
                }
            ]
        )

        parsed = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
            extra_tool_names=frozenset({"empty"}),
        )

        assert tuple(p.canonical_call_id for p in parsed.provider_call_provenance) == (None,)
        assert "toolu_empty" in parsed.provider_errors
        assert "malformed tool_use input for empty" in parsed.provider_errors["toolu_empty"]

    def test_desktop_content_tool_use_unknown_native_action_is_model_output_error(self):
        response = _fake_completion_response(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_bad",
                    "name": "computer",
                    "input": {"action": "goto", "url": "https://example.com"},
                }
            ]
        )

        msg = parse_response_with_provenance(
            response,
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "unknown Claude native computer action: goto"

    async def test_provider_call_persists_canonical_id_no_raw_response(self, monkeypatch):
        tool_call = _fake_tool_call(
            "computer",
            '{"action": "screenshot"}',
            id_="tc_desktop_1",
        )
        mock = AsyncMock(return_value=_fake_completion_response(tool_calls=[tool_call]))
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert "raw_response" not in assistant
        assert assistant["tool_calls"] == [
            make_tool_call(
                "computer",
                {"actions": [{"action": "screenshot"}]},
                call_id="call_0000",
            ),
        ]
        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_0000"
        assert tool_msg["content"] == [
            {"type": "image", "index": 1},
            {"type": "text", "text": "instr"},
        ]

    async def test_second_turn_computer_feedback_uses_provider_id_but_canonical_result_lookup(
        self, monkeypatch
    ):
        class _CanonicalTextEnv(_FakeEnv):
            async def step(self, actions):
                result = await super().step(actions)
                result.results = [
                    LiteToolResult(
                        tool_call_id=tool_call_id(action),
                        images=[self._shot],
                        text=f"result {tool_call_id(action)}",
                    )
                    for action in actions
                ]
                return result

        provider_id = "toolu_provider_computer_1"
        tool_call = _fake_tool_call(
            "computer",
            '{"action": "screenshot"}',
            id_=provider_id,
        )
        mock = AsyncMock(
            side_effect=[
                _fake_completion_response(tool_calls=[tool_call]),
                _fake_completion_response("done"),
            ]
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(_CanonicalTextEnv(terminate_after=2), max_steps=3)

        canonical_id = "call_0000"
        assert provider_id != canonical_id
        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert assistant["tool_calls"] == [
            make_tool_call(
                "computer",
                {"actions": [{"action": "screenshot"}]},
                call_id=canonical_id,
            )
        ]

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

    def test_stamp_rejects_legacy_tool_call_id(self):
        from lite.core.errors import ToolCallValidationError
        from lite.core.tools.calls import stamp_tool_call_list_ids

        calls = [{"tool_call_id": "legacy_1", "name": "computer", "arguments": {}}]

        with pytest.raises(ToolCallValidationError, match="tool_call_id"):
            stamp_tool_call_list_ids(calls, preserve=False)

    async def test_function_tool_result_uses_per_call_text_and_provider_id(self, monkeypatch):
        extra = make_tool_schema(
            "bash",
            description="Run a command.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        tool_call = _fake_tool_call("bash", "{}", id_="toolu_bash_1")
        r1 = _fake_completion_response(tool_calls=[tool_call])
        r2 = _fake_completion_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        result = await agent.sample(_ToolResultsEnv(terminate_after=2), max_steps=3)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert assistant["tool_calls"] == [
            make_tool_call("bash", {}, call_id="call_0000"),
        ]

        second_messages = mock.call_args_list[1].kwargs["messages"]
        provider_tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert provider_tool_msgs[-1]["tool_call_id"] == "toolu_bash_1"
        assert provider_tool_msgs[-1]["content"] == "per-call stdout"

        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_0000"
        assert {"type": "text", "text": "per-call stdout"} in tool_msg["content"]

    async def test_function_tool_errors_for_malformed_and_undeclared_siblings(
        self,
        monkeypatch,
    ):
        extra = make_tool_schema(
            "bash",
            description="Run a command.",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        )
        r1 = _fake_completion_response(
            tool_calls=[
                _fake_tool_call("bash", '{"command": "pwd"}', id_="toolu_bash_1"),
                _fake_tool_call("bash", "{not json", id_="toolu_bad_json_1"),
                _fake_tool_call("unknown_tool", "{}", id_="toolu_unknown_1"),
                _fake_tool_call(
                    "report_infeasible",
                    '{"reason": "blocked"}',
                    id_="toolu_inactive_report_1",
                ),
            ]
        )
        r2 = _fake_completion_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        result = await agent.sample(_ToolResultsEnv(terminate_after=2), max_steps=3)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert assistant["tool_calls"] == [
            make_tool_call("bash", {"command": "pwd"}, call_id="call_0000"),
        ]

        second_messages = mock.call_args_list[1].kwargs["messages"]
        outputs = {
            msg["tool_call_id"]: msg["content"]
            for msg in second_messages
            if msg.get("role") == "tool"
        }
        assert outputs["toolu_bash_1"] == "per-call stdout"
        assert "malformed tool_call arguments for bash" in outputs["toolu_bad_json_1"]
        assert "undeclared tool_call: unknown_tool" in outputs["toolu_unknown_1"]
        assert "undeclared tool_call: report_infeasible" in outputs["toolu_inactive_report_1"]
        assert outputs["toolu_bad_json_1"] != "ok"
        assert outputs["toolu_unknown_1"] != "ok"
        assert outputs["toolu_inactive_report_1"] != "ok"
        assert "per-call stdout" not in outputs["toolu_bad_json_1"]
        assert "per-call stdout" not in outputs["toolu_unknown_1"]
        assert "per-call stdout" not in outputs["toolu_inactive_report_1"]

    async def test_function_tool_empty_text_uses_provider_placeholder(self, monkeypatch):
        extra = make_tool_schema(
            "bash",
            description="Run a command.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        tool_call = _fake_tool_call("bash", "{}", id_="toolu_bash_1")
        r1 = _fake_completion_response(tool_calls=[tool_call])
        r2 = _fake_completion_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        result = await agent.sample(_EmptyToolResultsEnv(terminate_after=2), max_steps=3)

        second_messages = mock.call_args_list[1].kwargs["messages"]
        provider_tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert provider_tool_msgs[-1]["content"] == "ok"
        assert result.lite_sample.messages[-2] == {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "text", "text": ""}],
        }

    async def test_mixed_computer_and_function_outputs_keep_per_call_text(
        self,
        monkeypatch,
    ):
        extra = make_tool_schema(
            "bash",
            description="Run a command.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        r1 = _fake_completion_response(
            tool_calls=[
                _fake_tool_call(
                    "computer",
                    '{"action": "screenshot"}',
                    id_="toolu_computer_1",
                ),
                _fake_tool_call("bash", "{}", id_="toolu_bash_1"),
            ]
        )
        r2 = _fake_completion_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        result = await agent.sample(
            _MixedComputerAndBashResultsEnv(terminate_after=2),
            max_steps=3,
        )

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert [tool_call_name(call) for call in assistant["tool_calls"]] == [
            "computer",
            "bash",
        ]

        second_messages = mock.call_args_list[1].kwargs["messages"]
        provider_tool_msgs = {
            msg["tool_call_id"]: msg["content"]
            for msg in second_messages
            if msg.get("role") == "tool"
        }
        computer_content = provider_tool_msgs["toolu_computer_1"]
        assert isinstance(computer_content, list)
        computer_texts = [
            block.get("text", "")
            for block in computer_content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        assert computer_texts == ["computer screen text"]
        assert provider_tool_msgs["toolu_bash_1"] == "bash stdout"

        tool_messages = {
            message["tool_call_id"]: message
            for message in result.lite_sample.messages
            if message.get("role") == "tool"
        }
        assert {"type": "text", "text": "computer screen text"} in tool_messages["call_0000"][
            "content"
        ]
        assert {"type": "text", "text": "bash stdout"} in tool_messages["call_0001"]["content"]

    async def test_function_tool_image_result_is_model_visible(self, monkeypatch):
        extra = make_tool_schema(
            "visual_extra",
            description="Return visual feedback.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        tool_call = _fake_tool_call("visual_extra", "{}", id_="toolu_visual_1")
        r1 = _fake_completion_response(tool_calls=[tool_call])
        r2 = _fake_completion_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        await agent.sample(_ImageToolResultsEnv(terminate_after=2), max_steps=3)

        second_messages = mock.call_args_list[1].kwargs["messages"]
        provider_tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert provider_tool_msgs[-1]["content"] == "visual obs"
        assert any(
            m.get("role") == "user"
            and any(
                isinstance(block, dict) and block.get("type") == "image_url"
                for block in (m.get("content") or [])
            )
            for m in second_messages
        ), f"image-bearing extra result was not model-visible: {second_messages}"

    async def test_multi_image_function_tool_result_keeps_all_model_visible_images(
        self,
        monkeypatch,
    ):
        extra = make_tool_schema(
            "visual_extra",
            description="Return visual feedback.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        tool_call = _fake_tool_call("visual_extra", "{}", id_="toolu_visual_1")
        mock = AsyncMock(
            side_effect=[
                _fake_completion_response(tool_calls=[tool_call]),
                _fake_completion_response("done"),
            ]
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _MultiImageToolResultsEnv(terminate_after=2)
        result = await ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
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
        provider_tool_index = next(
            i
            for i, message in enumerate(second_messages)
            if message.get("role") == "tool" and message.get("tool_call_id") == "toolu_visual_1"
        )
        image_blocks = [
            block
            for message in second_messages[provider_tool_index + 1 :]
            if message.get("role") == "user"
            for block in (message.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "image_url"
        ]
        assert len(image_blocks) == 2
        sent_images = [
            Image.open(io.BytesIO(base64.b64decode(block["image_url"]["url"].split(",", 1)[1])))
            for block in image_blocks
        ]
        assert [image.getpixel((0, 0)) for image in sent_images] == [(10, 20, 30), (80, 120, 160)]

    async def test_multi_image_computer_result_logs_all_images_but_sends_latest(
        self,
        monkeypatch,
    ):
        provider_id = "toolu_computer_batch_1"
        tool_call = _fake_tool_call(
            "computer",
            '{"action": "screenshot"}',
            id_=provider_id,
        )
        mock = AsyncMock(
            side_effect=[
                _fake_completion_response(tool_calls=[tool_call]),
                _fake_completion_response("done"),
            ]
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _MultiImageToolResultsEnv(terminate_after=2)
        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")

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

    async def test_action_batch_image_markers_are_private_to_claude_loop(
        self,
        monkeypatch,
    ):
        provider_id = "toolu_computer_batch_1"
        tool_call = _fake_tool_call(
            "computer",
            '{"action": "screenshot"}',
            id_=provider_id,
        )
        mock = AsyncMock(
            side_effect=[
                _fake_completion_response(tool_calls=[tool_call]),
                _fake_completion_response("done"),
            ]
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        result = await ClaudeDesktopUseAgent(model_id="claude-opus-4-6").sample(
            _MultiImageToolResultsEnv(terminate_after=2),
            max_steps=3,
        )

        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 2)]
        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_0000"
        assert [
            block["index"]
            for block in tool_msg["content"]
            if isinstance(block, dict) and block.get("type") == "image"
        ] == [2]

        sent_payloads = [
            json.dumps(call.kwargs["messages"], ensure_ascii=False, default=str)
            for call in mock.call_args_list
        ]
        assert all("_cua_lite_image_index" not in payload for payload in sent_payloads)
        assert all("_cua_lite_image_index" not in step.prompt for step in result.steps)

    async def test_terminal_multi_image_computer_result_logs_all_images(
        self,
        monkeypatch,
    ):
        tool_call = _fake_tool_call(
            "computer",
            '{"action": "screenshot"}',
            id_="toolu_terminal_batch_1",
        )
        mock = AsyncMock(return_value=_fake_completion_response(tool_calls=[tool_call]))
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(_MultiImageToolResultsEnv(terminate_after=1), max_steps=2)

        assert len(result.lite_sample.images) == 3
        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert [
            block["index"]
            for block in tool_msg["content"]
            if isinstance(block, dict) and block.get("type") == "image"
        ] == [2]
        assert [tuple(step.image_indices) for step in result.steps] == [(0,)]

    async def test_content_block_tool_use_gets_provider_tool_result(self, monkeypatch):
        r1 = _fake_completion_response(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_computer_1",
                    "name": "computer",
                    "input": {"action": "screenshot"},
                }
            ]
        )
        r2 = _fake_completion_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        await agent.sample(_FakeEnv(terminate_after=2), max_steps=3)

        second_messages = mock.call_args_list[1].kwargs["messages"]
        assistant_msgs = [m for m in second_messages if m.get("role") == "assistant"]
        provider_tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert any(
            (m.get("tool_calls") or [{}])[0].get("id") == "toolu_computer_1" for m in assistant_msgs
        )
        assert provider_tool_msgs[-1]["tool_call_id"] == "toolu_computer_1"

    async def test_computer_tool_result_carries_fresh_image_and_projected_error_text(
        self,
        monkeypatch,
    ):
        r1 = _fake_completion_response(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_computer_1",
                    "name": "computer",
                    "input": {"action": "screenshot"},
                }
            ]
        )
        r2 = _fake_completion_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _FreshImageErrorEnv(terminate_after=2)
        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            resolution=(800, 600),
        )
        await agent.sample(env, max_steps=3)

        second_messages = mock.call_args_list[1].kwargs["messages"]
        provider_tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert len(provider_tool_msgs) == 1
        assert provider_tool_msgs[-1]["tool_call_id"] == "toolu_computer_1"
        content = provider_tool_msgs[-1]["content"]
        assert isinstance(content, list)
        texts = [block.get("text", "") for block in content if block.get("type") == "text"]
        joined = "\n".join(texts)
        assert joined.count("## Error from previous action:") == 1
        assert joined.count("invalid action: screenshot") == 1
        assert "## AXTree:\nbutton Search" in joined
        assert "## Error from previous action:\ninvalid action: screenshot" in joined
        image_blocks = [block for block in content if block.get("type") == "image_url"]
        assert len(image_blocks) == 1
        payload = image_blocks[0]["image_url"]["url"].split("base64,", 1)[1]
        assert base64.b64decode(payload) == env._fresh_shot

    async def test_thinking_only_output_terminates_through_the_env(self, monkeypatch):
        """N3: reasoning-only output is a final turn, not a RuntimeError.

        It used to raise ``model returned no tool calls and no final text``,
        discarding the whole episode (and, being retryable, burning every
        remaining attempt). Real published data contains exactly this shape.
        """
        mock = AsyncMock(
            return_value=_fake_completion_response(
                content=[{"type": "thinking", "thinking": "checking", "signature": "sig_1"}],
            )
        )
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _RecordingFakeEnv(terminate_after=1)
        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")

        rl = await agent.sample(env, max_steps=2)
        assert len(env.actions_seen) == 1
        assert tool_call_name(env.actions_seen[0][0]) == "response"
        assert tool_call_arguments(env.actions_seen[0][0]) == {"text": ""}
        assert rl.terminated is True
        assert not rl.lite_sample.messages[-1].get("tool_calls")

    async def test_content_only_final_text_is_not_saved_as_response_tool(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response(content="  18 x 24  "))
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _RejectEmptyActionsEnv(terminate_after=99)
        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(env, max_steps=2)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert "raw_response" not in assistant
        assert assistant["content"] == [{"type": "text", "text": "  18 x 24  "}]
        assert not assistant.get("tool_calls")
        assert len(env.actions_seen) == 1
        assert tool_call_name(env.actions_seen[0][0]) == "response"
        assert tool_call_id(env.actions_seen[0][0]) is None
        assert tool_call_arguments(env.actions_seen[0][0]) == {"text": "18 x 24"}
        assert result.terminated is True
        assert result.truncated is False
        assert result.episode_return == 1.0
        assert result.steps[0].reward == 1.0

    async def test_content_only_final_uses_runtime_response_when_enabled(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response(content="Final answer"))
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _RejectEmptyActionsEnv(terminate_after=99)
        env.metadata.extra_tool_schemas = [LiteFinishToolSet.get_tool_schema("response")]
        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(env, max_steps=2)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert "raw_response" not in assistant
        assert assistant["content"] == [{"type": "text", "text": "Final answer"}]
        assert not assistant.get("tool_calls")
        assert len(env.actions_seen) == 1
        assert tool_call_name(env.actions_seen[0][0]) == "response"
        assert tool_call_id(env.actions_seen[0][0]) is None
        assert tool_call_arguments(env.actions_seen[0][0]) == {"text": "Final answer"}
        assert result.terminated is True
        assert result.truncated is False


# -----------------------------------------------------------------------------
# Object-shaped Anthropic content blocks
# -----------------------------------------------------------------------------


class _ObjectContentBlock:
    """A LiteLLM/Anthropic-SDK content block: not a dict, but ``model_dump()``-able."""

    def __init__(self, **data: Any) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)


class TestObjectShapedContentBlocks:
    """Anthropic content blocks reach CUA-Lite either as plain JSON dicts (proxy
    responses) or as LiteLLM/SDK objects. ``claude_content_blocks()`` reads both
    once, so parse and provider replay never re-sniff raw block shape."""

    @staticmethod
    def _parsed(content: list[Any]) -> Any:
        return parse_response_with_provenance(
            _fake_completion_response(content=content),
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        )

    def test_object_blocks_survive_parse_and_provider_replay(self):
        parsed = self._parsed(
            [
                _ObjectContentBlock(
                    type="thinking",
                    thinking="I will screenshot",
                    signature="sig_object",
                ),
                _ObjectContentBlock(type="text", text="Taking a screenshot."),
                _ObjectContentBlock(
                    type="tool_use",
                    id="toolu_object",
                    name="computer",
                    input={"action": "screenshot"},
                ),
            ]
        )

        # Parse: nothing was dropped as unreadable, and each block landed on its
        # canonical Lite field.
        assert pop_model_output_error(parsed.message) is None
        assert parsed.message["content"] == [
            {"type": "action_description", "text": "Taking a screenshot."}
        ]
        assert parsed.message["reasoning_content"] == "I will screenshot"
        assert [tool_call_name(call) for call in parsed.message["tool_calls"]] == ["computer"]
        assert tool_call_arguments(parsed.message["tool_calls"][0]) == {
            "actions": [{"action": "screenshot"}]
        }

        # Provenance: the object-shaped tool_use keeps its provider id and maps
        # to the canonical call the env will execute.
        assert [tc.provider_id for tc in parsed.provider_tool_uses] == ["toolu_object"]
        assert [p.canonical_call_id for p in parsed.provider_call_provenance] == [
            tool_call_id(parsed.message["tool_calls"][0])
        ]
        assert parsed.provider_errors == {}

        # Replay: the next request carries the thinking signature verbatim and
        # replays the tool_use through ``tool_calls``, not through content.
        completion_messages: list[dict[str, Any]] = []
        append_provider_assistant_message(
            completion_messages,
            replay_content=parsed.replay_content,
            provider_tool_uses=parsed.provider_tool_uses,
        )
        assert completion_messages == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I will screenshot",
                        "signature": "sig_object",
                    },
                    {"type": "text", "text": "Taking a screenshot."},
                ],
                "tool_calls": [
                    {
                        "id": "toolu_object",
                        "type": "function",
                        "function": {
                            "name": "computer",
                            "arguments": '{"action": "screenshot"}',
                        },
                    }
                ],
            }
        ]

    def test_unreadable_block_becomes_model_output_error(self):
        parsed = self._parsed([object()])

        assert parsed.message["tool_calls"] == []
        assert pop_model_output_error(parsed.message) == (
            "unreadable Claude content block of type object"
        )
        # An unreadable block cannot be replayed, so it is omitted rather than
        # forwarded in an unusable shape.
        assert parsed.replay_content is None


# -----------------------------------------------------------------------------
# Thinking signature preservation (always-on; matches claude-quickstarts)
# -----------------------------------------------------------------------------


class TestReplayContent:
    """``replay_content`` is produced by the one parse pass over the response."""

    @staticmethod
    def _replay_content(content: Any) -> Any:
        return parse_response_with_provenance(
            _fake_completion_response(content=content),
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        ).replay_content

    def test_replay_keeps_signature(self):
        out = self._replay_content(
            [
                {"type": "thinking", "thinking": "I will click", "signature": "sig_abc"},
                {"type": "text", "text": "Clicking button."},
            ]
        )
        assert isinstance(out, list)
        thinking = next(b for b in out if b["type"] == "thinking")
        assert thinking["signature"] == "sig_abc"
        assert thinking["thinking"] == "I will click"

    def test_replay_pass_through_str(self):
        assert self._replay_content("hello") == "hello"

    def test_replay_pass_through_none(self):
        assert self._replay_content(None) is None

    def test_response_without_a_choice_replays_no_assistant_message(self):
        """An empty ``choices`` carries no assistant turn, so replay stays empty."""
        from litellm.types.utils import ModelResponse

        parsed = parse_response_with_provenance(
            ModelResponse(choices=[]),
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
        )
        completion_messages: list[dict[str, Any]] = []

        append_provider_assistant_message(
            completion_messages,
            replay_content=parsed.replay_content,
            provider_tool_uses=parsed.provider_tool_uses,
        )

        assert completion_messages == []


# -----------------------------------------------------------------------------
# system_prompt_suffix
# -----------------------------------------------------------------------------


class TestSystemPromptSuffix:
    """system_prompt_suffix appends to base system_prompt (not replaces)."""

    async def test_suffix_alone_becomes_system(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={
                "max_tokens": 4096,
                "temperature": 0.7,
                "thinking_budget": 0,
                "prompt_caching": False,
            },
            system_prompt_suffix="CUSTOM_ANCHOR",
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        messages = mock.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        # Could be str or list (prompt_caching off → str); check for substring
        sys_content = messages[0]["content"]
        if isinstance(sys_content, list):
            sys_content = sys_content[0].get("text", "")
        assert "CUSTOM_ANCHOR" in sys_content

    async def test_suffix_appends_to_base_system_prompt(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            system_prompt="BASE",
            api_kwargs={
                "max_tokens": 4096,
                "temperature": 0.7,
                "thinking_budget": 0,
                "prompt_caching": False,
            },
            system_prompt_suffix="APPENDED",
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        messages = mock.call_args.kwargs["messages"]
        sys_content = messages[0]["content"]
        if isinstance(sys_content, list):
            sys_content = sys_content[0].get("text", "")
        assert "BASE" in sys_content and "APPENDED" in sys_content
        assert sys_content.index("BASE") < sys_content.index("APPENDED")


# -----------------------------------------------------------------------------
# api_retry_max / api_retry_base_delay
# -----------------------------------------------------------------------------


class TestAPIRetry:
    """exponential-backoff retry wrapping litellm.acompletion."""

    async def test_default_retries_on_transient_error(self, monkeypatch):
        calls = {"n": 0}

        async def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient boom")
            return _fake_completion_response()

        monkeypatch.setattr("litellm.acompletion", flaky)
        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={
                "max_tokens": 4096,
                "temperature": 0.7,
                "thinking_budget": 0,
                "prompt_caching": False,
            },
            api_retry_base_delay=0.0,
        )
        result = await agent._acompletion_with_retry(model="x", messages=[])
        assert calls["n"] == 3
        assert result is not None

    async def test_max_zero_disables_retry(self, monkeypatch):
        calls = {"n": 0}

        async def once_failing(**kwargs):
            calls["n"] += 1
            raise RuntimeError("boom")

        monkeypatch.setattr("litellm.acompletion", once_failing)
        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_kwargs={
                "max_tokens": 4096,
                "temperature": 0.7,
                "thinking_budget": 0,
                "prompt_caching": False,
            },
            api_retry_max=0,
            api_retry_base_delay=0.0,
        )
        with pytest.raises(RuntimeError):
            await agent._acompletion_with_retry(model="x", messages=[])
        assert calls["n"] == 1


class TestEnvCloseTeardown:
    async def test_close_failure_logged_and_swallowed_after_success(self, monkeypatch, caplog):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)
        caplog.set_level(logging.WARNING, logger="lite.agents.models.claude.agent")

        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
        result = await agent.sample(_CloseRaisesEnv(terminate_after=1), max_steps=2)

        assert result.terminated is True
        assert mock.call_count == 1
        assert "env.close() failed: claude close exploded" in caplog.text

    async def test_close_runs_when_on_complete_raises(self, monkeypatch):
        class _CompleteRaises:
            def on_complete(self, result):
                raise RuntimeError("claude hook complete exploded")

        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _FakeEnv(terminate_after=1)
        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")

        with pytest.raises(RuntimeError, match="claude hook complete exploded"):
            await agent.sample(env, max_steps=2, hooks=[_CompleteRaises()])

        assert env.closed is True

    async def test_close_cancellation_propagates(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_completion_response())
        monkeypatch.setattr("litellm.acompletion", mock)

        env = _CloseCancelledEnv(terminate_after=1)
        agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")

        with pytest.raises(asyncio.CancelledError):
            await agent.sample(env, max_steps=2)

        assert env.closed is True

    async def test_provider_error_is_not_masked_by_close_failure(self, monkeypatch, caplog):
        async def provider_boom(**kwargs):
            raise RuntimeError("claude provider exploded")

        monkeypatch.setattr("litellm.acompletion", provider_boom)
        caplog.set_level(logging.WARNING, logger="lite.agents.models.claude.agent")

        agent = ClaudeDesktopUseAgent(
            model_id="claude-opus-4-6",
            api_retry_max=0,
            api_retry_base_delay=0.0,
        )

        with pytest.raises(RuntimeError, match="claude provider exploded"):
            await agent.sample(_CloseRaisesEnv(terminate_after=1), max_steps=2)

        assert "env.close() failed: claude close exploded" in caplog.text


async def test_claude_finish_exposure_uses_extra_tool_schemas_not_valid_actions(
    monkeypatch,
) -> None:
    valid_only = await anthropic_tools_sent(
        ClaudeDesktopUseAgent(
            metadata=LiteCUAMetadata(dims=("desktop", "use"), valid_actions=["click"])
        ),
        monkeypatch,
    )
    assert anthropic_provider_tool_names(valid_only).isdisjoint({"response", "terminate"})

    with_schema = await anthropic_tools_sent(
        ClaudeDesktopUseAgent(
            metadata=LiteCUAMetadata(dims=("desktop", "use"), extra_tool_schemas=[RESPONSE_SCHEMA])
        ),
        monkeypatch,
    )
    assert "response" in anthropic_provider_tool_names(with_schema)


async def test_claude_grounding_point_valid_actions_point_controls_left_click_schema(
    monkeypatch,
) -> None:
    async def names(valid_actions, extra_tool_schemas=None):
        return anthropic_provider_tool_names(
            await anthropic_tools_sent(
                ClaudeDesktopGroundingPointAgent(
                    metadata=LiteCUAMetadata(
                        dims=("desktop", "grounding.point"),
                        valid_actions=valid_actions,
                        extra_tool_schemas=extra_tool_schemas or [],
                    )
                ),
                monkeypatch,
            )
        )

    assert "left_click" in await names(None)
    assert "left_click" in await names(["point"])
    assert "left_click" not in await names([])
    assert "left_click" not in await names(["click"])
    assert await names([], [BASH_SCHEMA]) == {"bash"}
