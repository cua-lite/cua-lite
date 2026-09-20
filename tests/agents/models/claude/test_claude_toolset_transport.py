"""Native SDK wire checks; every request uses an offline HTTP transport."""

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
async def test_toolset_sdk_wire_round_trip(monkeypatch, choice, expected_choice):
    seen = []
    provider_body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
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
        model="anthropic/claude-opus-5",
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
    assert wire["model"] == "claude-opus-5"
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
