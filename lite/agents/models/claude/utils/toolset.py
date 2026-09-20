"""Claude Opus 5 computer toolset transport for LiteLLM-style agent history."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def _content_block(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") != "image_url":
        return block
    media_type, image_data = block["image_url"]["url"].split(";base64,", 1)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type.removeprefix("data:"),
            "data": image_data,
        },
        **({"cache_control": block["cache_control"]} if "cache_control" in block else {}),
    }


def _messages_for_anthropic(messages: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    system = None
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content")
        if role == "system":
            system = content
            continue
        if role == "tool":
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": (
                    [_content_block(block) for block in content]
                    if isinstance(content, list)
                    else content
                ),
            }
            if msg.get("toolset_name"):
                result["toolset_name"] = msg["toolset_name"]
            if msg.get("is_error"):
                result["is_error"] = True
            role, content = "user", [result]
        else:
            content = (
                [_content_block(block) for block in content]
                if isinstance(content, list)
                else content
            )
            if role == "assistant" and msg.get("tool_calls"):
                blocks = (
                    content
                    if isinstance(content, list)
                    else ([{"type": "text", "text": content}] if content else [])
                )
                content = [*blocks]
                for call in msg["tool_calls"]:
                    tool_use = {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": json.loads(call["function"]["arguments"]),
                    }
                    if call.get("toolset_name"):
                        tool_use["toolset_name"] = call["toolset_name"]
                    content.append(tool_use)
        if role == "user" and converted and converted[-1]["role"] == "user":
            if isinstance(content, list) and isinstance(converted[-1]["content"], list):
                converted[-1]["content"].extend(content)
                continue
        converted.append({"role": role, "content": content})
    return system, converted


async def acompletion_with_computer_toolset(**kwargs: Any) -> Any:
    """Use the native SDK: installed LiteLLM maps toolsets as legacy computer tools."""
    import anthropic

    system, messages = _messages_for_anthropic(kwargs["messages"])
    request: dict[str, Any] = {
        "model": kwargs["model"].removeprefix("anthropic/"),
        "max_tokens": kwargs["max_tokens"],
        "messages": messages,
        "tools": kwargs["tools"],
    }
    if system is not None:
        request["system"] = system
    if kwargs.get("headers"):
        request["extra_headers"] = kwargs["headers"]
    if kwargs.get("tool_choice"):
        choice = kwargs["tool_choice"]
        if isinstance(choice, str):
            choice = {"type": "any" if choice == "required" else choice}
        elif choice.get("type") == "function":
            choice = {"type": "tool", "name": choice["function"]["name"]}
        request["tool_choice"] = choice
    extra_body = {k: kwargs[k] for k in ("thinking", "output_config") if k in kwargs}
    if extra_body:
        request["extra_body"] = extra_body

    async with anthropic.AsyncAnthropic(
        api_key=kwargs.get("api_key"),
        base_url=kwargs.get("api_base"),
        max_retries=0,
    ) as client:
        response = await client.messages.create(**request)

    message = SimpleNamespace(
        content=[block.model_dump(exclude_none=True) for block in response.content],
        tool_calls=[],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=response.stop_reason)],
        model_dump=response.model_dump,
    )
