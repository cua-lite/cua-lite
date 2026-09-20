"""Native Claude Messages transport for LiteLLM-style agent history."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def _content_block(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") == "thinking" and block.get("thinking") == "" and block.get("signature"):
        # LiteLLM 1.102 drops empty thinking, including valid omitted blocks.
        # Anthropic ignores this text and restores thinking from the unchanged signature.
        # https://platform.claude.com/docs/en/build-with-claude/thinking#controlling-thinking-display
        return {**block, "thinking": "[omitted]"}
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
            # Follow Anthropic's format: cache the outer tool_result block.
            if isinstance(result["content"], list) and result["content"]:
                last = result["content"][-1]
                if "cache_control" in last:
                    last = result["content"][-1] = dict(last)
                    result["cache_control"] = last.pop("cache_control")
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


async def acompletion_with_messages(**kwargs: Any) -> Any:
    """Use LiteLLM's Messages entrypoint to preserve native toolsets and effort."""
    import litellm

    system, messages = _messages_for_anthropic(kwargs["messages"])
    tools = []
    for tool in kwargs["tools"]:
        if "function" in tool:
            function = tool["function"]
            if tool["type"] == "function":
                tool = {
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "input_schema": function["parameters"],
                }
            else:
                tool = {"type": tool["type"], "name": function["name"], **function["parameters"]}
        tools.append(tool)
    request: dict[str, Any] = {
        "model": kwargs["model"],
        "max_tokens": kwargs["max_tokens"],
        "messages": messages,
        "tools": tools,
    }
    if system is not None:
        request["system"] = system
    if kwargs.get("headers"):
        request["headers"] = kwargs["headers"]
    if kwargs.get("tool_choice"):
        choice = kwargs["tool_choice"]
        if isinstance(choice, str):
            choice = {"type": "any" if choice == "required" else choice}
        elif choice.get("type") == "function":
            choice = {"type": "tool", "name": choice["function"]["name"]}
        request["tool_choice"] = choice
    request.update({k: kwargs[k] for k in ("thinking", "output_config") if k in kwargs})

    response = await litellm.anthropic.messages.acreate(
        **request,
        api_key=kwargs.get("api_key"),
        api_base=kwargs.get("api_base"),
        custom_llm_provider="anthropic",
        num_retries=0,
    )

    message = SimpleNamespace(content=response["content"], tool_calls=[])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=response["stop_reason"])],
        model_dump=lambda: response,
    )
