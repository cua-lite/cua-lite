"""Claude transport wire checks; every request uses an offline HTTP transport."""

from __future__ import annotations

import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from lite.agents.models.claude.utils.history import inject_prompt_caching
from lite.agents.models.claude.utils.toolset import (
    _messages_for_anthropic,
    acompletion_with_messages,
)


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
async def test_toolset_messages_wire_round_trip(monkeypatch, choice, expected_choice, model_id):
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
        "stop_reason": {"auto": "max_tokens", "none": "refusal"}.get(
            choice if isinstance(choice, str) else "", "tool_use"
        ),
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10, "output_tokens": 4,
            "cache_creation_input_tokens": 20, "cache_read_input_tokens": 30,
        },
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
    # Exercise both summarized and signature-only thinking without changing defaults.
    thinking_text = "" if choice == "auto" else "test"
    messages = [
        {"role": "system", "content": "Use the screen."},
        {"role": "user", "content": "Continue."},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": thinking_text, "signature": "sig"}],
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
    response = await acompletion_with_messages(
        model=f"anthropic/{model_id}",
        api_key="test-key",
        api_base="https://claude.test",
        max_tokens=32,
        messages=messages,
        tools=tools,
        tool_choice=choice,
        output_config={"effort": "medium"},
        thinking={"type": "adaptive"},
        headers={"x-client-test": "toolset-round-trip"},
    )

    assert len(seen) == 1
    assert seen[0].url.path == "/v1/messages"
    assert seen[0].headers["x-api-key"] == "test-key"
    assert seen[0].headers["x-client-test"] == "toolset-round-trip"
    wire = json.loads(seen[0].content)
    assert wire["model"] == model_id
    assert wire["tools"] == tools
    assert wire["tool_choice"] == expected_choice
    assert wire["output_config"] == {"effort": "medium"}
    assert wire["thinking"] == {"type": "adaptive"}
    assert wire["system"] == "Use the screen."
    assistant = wire["messages"][1]["content"]
    assert assistant[0]["signature"] == "sig"
    assert assistant[0]["thinking"] == (thinking_text or "[omitted]")
    assert messages[2]["content"][0]["thinking"] == thinking_text
    assert assistant[1]["toolset_name"] == "computer"
    result = wire["messages"][2]["content"][0]
    assert (result["tool_use_id"], result["toolset_name"], result["is_error"]) == (
        "toolu_prev",
        "computer",
        True,
    )
    image = result["content"][1]
    assert image["source"] == {"type": "base64", "media_type": "image/png", "data": "cGl4ZWw="}
    assert "cache_control" not in image
    assert result["cache_control"] == {"type": "ephemeral"}
    assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert response.choices[0].message.content[0]["toolset_name"] == "computer"
    assert response.choices[0].finish_reason == provider_body["stop_reason"]
    assert response.model_dump()["usage"] == provider_body["usage"]


@pytest.mark.parametrize("last_block", [
    {"type": "text", "text": "screenshot unavailable"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,cGl4ZWw="}},
])
def test_rolling_cache_marks_whole_tool_results_without_changing_history(last_block):
    messages = [{"role": "system", "content": "Use the screen."}]
    for i in range(4):
        messages.append({
            "role": "tool", "tool_call_id": f"toolu_{i}",
            "content": [copy.deepcopy(last_block)],
        })
    inject_prompt_caching(messages, cap=4)
    before = copy.deepcopy(messages)

    system, native = _messages_for_anthropic(messages)

    assert messages == before
    assert json.dumps([system, native]).count('"cache_control"') == 4
    results = native[0]["content"]
    assert "cache_control" not in results[0]
    for result in results[1:]:
        assert result["cache_control"] == {"type": "ephemeral"}
        assert all("cache_control" not in block for block in result["content"])


async def test_messages_unwraps_explicit_legacy_computer_tool(monkeypatch):
    call = AsyncMock(return_value={"content": [], "stop_reason": "end_turn"})
    monkeypatch.setattr("litellm.anthropic.messages.acreate", call)
    display = {"display_width_px": 1920, "display_height_px": 1080, "display_number": 1}
    await acompletion_with_messages(
        model="anthropic/claude-opus-5", max_tokens=32,
        messages=[{"role": "user", "content": "Take a screenshot."}],
        tools=[{
            "type": "computer_20251124",
            "function": {"name": "computer", "parameters": display},
        }],
    )
    assert call.call_args.kwargs["tools"] == [
        {"type": "computer_20251124", "name": "computer", **display},
    ]


@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_messages_http_errors_propagate_without_transport_retries(monkeypatch, status):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(status, json={
            "type": "error", "error": {"type": "api_error", "message": "provider failure"},
        })

    original_init = httpx.AsyncClient.__init__

    def init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    with pytest.raises(Exception) as exc:
        await acompletion_with_messages(
            model="anthropic/claude-opus-5", api_key="test-key",
            api_base=f"https://error-{status}.test", max_tokens=32,
            messages=[{"role": "user", "content": "Take a screenshot."}],
            tools=[{"type": "computer_toolset_20260801"}],
        )
    assert exc.value.status_code == status
    assert len(seen) == 1


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
            "content": ([{"type": "thinking", "thinking": "", "signature": "sig"}]
                        if kind == "mobile" else []) + [{
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
    steps = 2 if kind == "mobile" else 1
    result = await agent.sample(env_class(terminate_after=steps), max_steps=steps)

    assert result.terminated
    assert len(seen) == steps
    wire = json.loads(seen[-1].content)
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
        assistant = next(m for m in wire["messages"] if m["role"] == "assistant")
        assert assistant["content"][0] == {
            "type": "thinking", "thinking": "[omitted]", "signature": "sig",
        }
