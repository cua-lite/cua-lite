"""Claude transport wire checks; every request uses an offline HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest

from lite.agents.models.claude.utils.toolset import acompletion_with_computer_toolset


@pytest.mark.parametrize(
    "choice,expected_choice",
    [
        ("required", {"type": "any"}),
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        (
            {"type": "function", "function": {"name": "response"}},
            {"type": "tool", "name": "response"},
        ),
        ({"type": "tool", "name": "response"}, {"type": "tool", "name": "response"}),
    ],
)
@pytest.mark.parametrize("model_id", ["claude-opus-5", "claude-sonnet-5"])
async def test_toolset_sdk_wire_round_trip(monkeypatch, choice, expected_choice, model_id):
    seen = []
    provider_body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_next",
                "name": "screenshot",
                "toolset_name": "computer",
                "input": {},
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=provider_body)

    original_init = httpx.AsyncClient.__init__

    def init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    tools = [
        {"type": "computer_toolset_20260801", "configs": {"zoom": {"enabled": False}}},
        {"name": "response", "input_schema": {"type": "object", "properties": {}}},
    ]
    messages = [
        {"role": "system", "content": "Use the screen."},
        {"role": "user", "content": "Continue."},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "test", "signature": "sig"}],
            "tool_calls": [
                {
                    "id": "toolu_prev",
                    "toolset_name": "computer",
                    "function": {"name": "screenshot", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_prev",
            "toolset_name": "computer",
            "is_error": True,
            "content": [
                {"type": "text", "text": "retry"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,cGl4ZWw="},
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
    ]
    response = await acompletion_with_computer_toolset(
        model=f"anthropic/{model_id}",
        api_key="test-key",
        api_base="https://claude.test",
        max_tokens=32,
        messages=messages,
        tools=tools,
        tool_choice=choice,
        output_config={"effort": "medium"},
        thinking={"type": "adaptive"},
        headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    assert len(seen) == 1
    assert seen[0].url.path == "/v1/messages"
    assert seen[0].headers["x-api-key"] == "test-key"
    assert seen[0].headers["anthropic-beta"] == "prompt-caching-2024-07-31"
    wire = json.loads(seen[0].content)
    assert wire["model"] == model_id
    assert wire["tools"] == tools
    assert wire["tool_choice"] == expected_choice
    assert wire["output_config"] == {"effort": "medium"}
    assert wire["thinking"] == {"type": "adaptive"}
    assert wire["system"] == "Use the screen."
    assistant = wire["messages"][1]["content"]
    assert assistant[0]["signature"] == "sig"
    assert assistant[1]["toolset_name"] == "computer"
    result = wire["messages"][2]["content"][0]
    assert (result["tool_use_id"], result["toolset_name"], result["is_error"]) == (
        "toolu_prev",
        "computer",
        True,
    )
    image = result["content"][1]
    assert image["source"] == {"type": "base64", "media_type": "image/png", "data": "cGl4ZWw="}
    assert image["cache_control"] == {"type": "ephemeral"}
    assert response.choices[0].message.content[0]["toolset_name"] == "computer"
    assert response.choices[0].finish_reason == "tool_use"
    assert response.model_dump()["usage"]["input_tokens"] == 10


@pytest.mark.parametrize("model_id", ["claude-opus-5", "claude-sonnet-5"])
@pytest.mark.parametrize("kind", ["grounding", "mobile"])
async def test_function_tools_work_without_litellm_model_metadata(monkeypatch, model_id, kind):
    import litellm
    from agents.models.claude.test_claude_agent import _FakeEnv
    from agents.models.claude.test_claude_mobile_agent import _FakeMobileEnv

    from lite.agents.models.claude.agent import (
        ClaudeDesktopGroundingPointAgent,
        ClaudeMobileUseAgent,
    )

    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(litellm, "anthropic_models", set())
    monkeypatch.setattr(litellm, "telemetry", False)
    seen = []
    tool_name = "left_click" if kind == "grounding" else "tap"

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": model_id,
            "content": [{
                "type": "tool_use", "id": "toolu_test", "name": tool_name,
                "input": {"coordinate": [10, 10]},
            }],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 4},
        })

    original_init = httpx.AsyncClient.__init__

    def init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    agent_class, env_class = (
        (ClaudeDesktopGroundingPointAgent, _FakeEnv)
        if kind == "grounding" else (ClaudeMobileUseAgent, _FakeMobileEnv)
    )
    agent = agent_class(
        model_id=model_id, api_key="test-key",
        api_base=f"https://{kind}-{model_id}.test", api_retry_max=0,
    )
    result = await agent.sample(env_class(), max_steps=1)

    assert result.terminated
    assert len(seen) == 1
    wire = json.loads(seen[0].content)
    assert wire["model"] == model_id
    assert all("input_schema" in tool for tool in wire["tools"])
    assert "custom_llm_provider" not in wire
    assert "allowed_openai_params" not in wire
    if kind == "grounding":
        assert "thinking" not in wire
        assert wire["output_config"] == {"effort": "low"}
        assert wire["max_tokens"] == 1024
    else:
        assert wire["thinking"] == {"type": "adaptive"}
        assert wire["output_config"] == {"effort": "medium"}
        assert wire["max_tokens"] == 4096
