"""Claude provider-history helpers.

Most functions operate on LiteLLM/Anthropic request messages before the API call
or before diagnostic logging. The feedback appender also writes Claude's
bespoke-loop canonical Lite trajectory messages.
"""

from __future__ import annotations

import asyncio
from typing import Any

from lite.agents.core.agent.utils.image_markers import mark_provider_visible_image_index
from lite.agents.core.agent.utils.messages import build_tool_result_message
from lite.agents.core.agent.utils.provenance import ProviderCallProvenance
from lite.agents.core.agent.utils.tool_results import (
    append_tool_result_images,
    canonical_tool_result_messages,
    canonical_unpaired_response_feedback_messages,
    latest_step_feedback,
    tool_result_text_for_call_id,
)
from lite.agents.models.claude.utils.image_io import resize_for_claude_api
from lite.agents.models.claude.utils.parse import ClaudeProviderToolUse
from lite.core import LiteSample
from lite.core.tools.action_space import LITE_ACTION_BATCH_TOOL_NAMES
from lite.core.tools.calls import tool_call_id, tool_call_name
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvStepResult


def strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace base64 image payloads with placeholders for logs."""
    stripped = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    redacted = dict(item)
                    redacted["image_url"] = {"url": "[base64 image omitted]"}
                    new_content.append(redacted)
                else:
                    new_content.append(item)
            stripped.append({**msg, "content": new_content})
        else:
            stripped.append(msg)
    return stripped


def inject_prompt_caching(messages: list[dict[str, Any]], cap: int) -> None:
    """Set ``cache_control`` on system + rolling recent request-history blocks."""
    cap = max(0, min(int(cap), 4))
    body_breakpoints_budget = max(0, cap - 1)  # one slot is reserved for system

    if cap > 0 and messages and messages[0].get("role") == "system":
        sys_msg = messages[0]
        sys_content = sys_msg.get("content")
        if isinstance(sys_content, str):
            sys_msg["content"] = [
                {
                    "type": "text",
                    "text": sys_content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(sys_content, list) and sys_content:
            last = sys_content[-1]
            if isinstance(last, dict):
                last["cache_control"] = {"type": "ephemeral"}

    remaining = body_breakpoints_budget
    for msg in reversed(messages):
        if msg.get("role") not in {"user", "tool"}:
            continue
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            continue
        last = content[-1]
        if not isinstance(last, dict):
            continue
        if remaining > 0:
            last["cache_control"] = {"type": "ephemeral"}
            remaining -= 1
        elif "cache_control" in last:
            del last["cache_control"]


def append_provider_assistant_message(
    completion_messages: list[dict[str, Any]],
    *,
    replay_content: str | list[dict[str, Any]] | None,
    provider_tool_uses: tuple[ClaudeProviderToolUse, ...],
) -> None:
    """Append Claude's assistant turn to provider history for the next request.

    Both inputs come from the single parse pass over this turn's response, so
    replay never re-reads or re-normalizes raw provider output. A turn with
    neither replay content nor an identified provider tool call has no assistant
    message to replay and appends nothing.
    """
    provider_tool_calls = [
        {
            "id": tc.provider_id,
            "type": "function",
            **({"toolset_name": tc.toolset_name} if tc.toolset_name else {}),
            "function": {
                "name": tc.name,
                "arguments": tc.replay_arguments,
            },
        }
        for tc in provider_tool_uses
        if tc.provider_id
    ]
    if replay_content is None and not provider_tool_calls:
        return
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": replay_content}
    if provider_tool_calls:
        assistant_msg["tool_calls"] = provider_tool_calls
    completion_messages.append(assistant_msg)


def filter_to_n_most_recent_images(
    messages: list[dict[str, Any]],
    images_to_keep: int,
    *,
    min_removal_threshold: int = 1,
) -> list[dict[str, Any]]:
    """Replace non-retained ``image_url`` blocks, keeping the N most recent."""
    if images_to_keep < 0:
        return messages

    def _is_image(block: Any) -> bool:
        return isinstance(block, dict) and block.get("type") == "image_url"

    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            total += sum(1 for b in content if _is_image(b))

    to_remove = total - images_to_keep
    if min_removal_threshold > 0:
        to_remove -= to_remove % min_removal_threshold
    if to_remove <= 0:
        return messages

    removed = 0
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for block in content:
                if _is_image(block) and removed < to_remove:
                    removed += 1
                    new_content.append({"type": "text", "text": "[Image omitted]"})
                    continue
                new_content.append(block)
            new_msg = dict(msg)
            new_msg["content"] = new_content
            out.append(new_msg)
        else:
            out.append(msg)
    return out


def _provider_visible_result_image_pairs(
    *,
    results_by_call_id: dict[str, LiteToolResult],
    result_image_indices_by_call_id: dict[str, tuple[int, ...]],
    canonical_call_id: str | None,
    is_action_batch: bool,
    is_final_provider_call: bool = True,
) -> tuple[tuple[bytes, int], ...]:
    """Images from one env result that Claude may see in provider history."""
    if canonical_call_id is None:
        return ()
    if is_action_batch and not is_final_provider_call:
        return ()

    result = results_by_call_id.get(canonical_call_id)
    if result is None:
        return ()
    if not result.images:
        return ()

    image_indices = result_image_indices_by_call_id.get(canonical_call_id, ())
    pairs = tuple(
        zip(
            result.images,
            image_indices,
            strict=True,
        )
    )
    # Durable Lite storage keeps every returned frame. Provider replay keeps all
    # images for non-action tool results, but action batches expose only the
    # final frame because intermediate frames were never a separate model turn.
    return pairs[-1:] if is_action_batch else pairs


async def _image_url_blocks(
    images: tuple[bytes, ...],
    *,
    model_id: str,
    resize_target: tuple[int, int] | None,
    image_indices: tuple[int, ...],
    many_image: bool,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for image_bytes, image_index in zip(images, image_indices, strict=True):
        image_b64, _, _ = await asyncio.to_thread(
            resize_for_claude_api,
            image_bytes,
            model_id,
            resize_target,
            many_image=many_image,
        )
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/webp;base64,{image_b64}"},
        }
        blocks.append(
            mark_provider_visible_image_index(
                block,
                image_index,
            )
        )
    return blocks


def _provider_text_block(text: str | None) -> dict[str, str] | None:
    if not text:
        return None
    return {"type": "text", "text": text}


def _provider_text_or_ok(text: str | None) -> str:
    return text if text else "ok"


def _provider_tool_text(per_call_text: str | None, fallback_text: str | None) -> str:
    if per_call_text is not None:
        return _provider_text_or_ok(per_call_text)
    return _provider_text_or_ok(fallback_text)


async def append_canonical_step_feedback_messages(
    trajectory: LiteSample,
    step_result: LiteEnvStepResult,
    lite_tool_calls: list[dict[str, Any]],
) -> tuple[int | None, dict[str, tuple[int, ...]]]:
    """Append canonical Lite feedback while preserving every result image.

    ``step_result.results`` arrives already aligned to ``lite_tool_calls`` from
    the shared loop's env-result boundary. Claude provider history sends only
    the latest screenshot for an action-batch, but the durable trajectory keeps
    every frame the env returned. The returned per-call map contains all stored
    result image indices so provider history can pair images with indices before
    selecting the visible subset.
    """
    tool_results = step_result.results
    stored_result_images = await append_tool_result_images(
        trajectory,
        tool_results,
    )
    latest_image_index = next(
        (
            image_indices[-1]
            for image_indices in reversed(stored_result_images.by_position)
            if image_indices
        ),
        None,
    )

    if lite_tool_calls:
        tool_result_messages = canonical_tool_result_messages(
            tool_results,
            lite_tool_calls,
            result_image_indices_by_call_id=stored_result_images.by_call_id,
        )
    else:
        # Unpaired env results have no assistant call to pair with, so each one
        # becomes its own user feedback message.
        tool_result_messages = canonical_unpaired_response_feedback_messages(
            tool_results,
            result_image_indices_by_position=stored_result_images.by_position,
        )
    trajectory.messages.extend(tool_result_messages)

    return latest_image_index, stored_result_images.by_call_id


async def append_desktop_provider_feedback(
    *,
    completion_messages: list[dict[str, Any]],
    response: Any,
    trajectory: LiteSample,
    step_result: LiteEnvStepResult,
    lite_tool_calls: list[dict[str, Any]],
    model_id: str,
    resize_target: tuple[int, int] | None,
    many_image: bool,
    provider_tool_uses: tuple[ClaudeProviderToolUse, ...],
    provider_call_provenance: tuple[ProviderCallProvenance, ...],
    provider_errors: dict[str, str],
) -> tuple[bytes | None, int | None]:
    """Append Claude desktop tool results for the next provider turn.

    Mutates ``completion_messages`` and ``trajectory``. Returns the next
    observation image bytes and its stored index.
    """
    image_bytes, feedback_text, _ = latest_step_feedback(step_result)
    next_text = feedback_text if feedback_text is not None else ""
    if image_bytes:
        next_sent_image_b64, _, _ = await asyncio.to_thread(
            resize_for_claude_api,
            image_bytes,
            model_id,
            resize_target,
            many_image=many_image,
        )
    else:
        next_sent_image_b64 = None

    (
        image_index,
        result_image_indices_by_call_id,
    ) = await append_canonical_step_feedback_messages(
        trajectory,
        step_result,
        lite_tool_calls,
    )

    if response.choices:
        if provider_tool_uses:
            lite_action_batch_call_ids = frozenset(
                call_id
                for call in lite_tool_calls
                if (call_id := tool_call_id(call))
                and tool_call_name(call) in LITE_ACTION_BATCH_TOOL_NAMES
            )
            results_by_call_id = {
                result.tool_call_id: result
                for result in step_result.results
                if result.tool_call_id is not None
            }
            for tc, provenance in zip(provider_tool_uses, provider_call_provenance):
                if not tc.provider_id:
                    continue
                canonical_call_id = provenance.canonical_call_id
                provider_error = provider_errors.get(tc.provider_id)
                if provider_error is not None:
                    completion_messages.append(
                        {
                            "role": "tool",
                            "name": tc.name,
                            **({"toolset_name": tc.toolset_name} if tc.toolset_name else {}),
                            "tool_call_id": tc.provider_id,
                            "content": provider_error,
                            **({"is_error": True} if tc.toolset_name else {}),
                        }
                    )
                    continue

                if tc.name == "computer" or tc.toolset_name == "computer":
                    provider_visible_result_images = _provider_visible_result_image_pairs(
                        results_by_call_id=results_by_call_id,
                        result_image_indices_by_call_id=result_image_indices_by_call_id,
                        canonical_call_id=canonical_call_id,
                        is_action_batch=True,
                        is_final_provider_call=provenance.is_final_for_canonical,
                    )
                    provider_visible_images = tuple(
                        image for image, _ in provider_visible_result_images
                    )
                    provider_visible_image_indices = tuple(
                        image_index for _, image_index in provider_visible_result_images
                    )
                    image_blocks = await _image_url_blocks(
                        provider_visible_images,
                        model_id=model_id,
                        resize_target=resize_target,
                        image_indices=provider_visible_image_indices,
                        many_image=many_image,
                    )
                    per_call_text = tool_result_text_for_call_id(
                        step_result,
                        call_id=canonical_call_id,
                        default=None,
                    )
                    if image_blocks:
                        content: Any = image_blocks
                        if text_block := _provider_text_block(per_call_text):
                            content.insert(0, text_block)
                    elif not provenance.is_final_for_canonical:
                        content = "ok"
                    else:
                        content = _provider_text_or_ok(per_call_text)
                    completion_messages.append(
                        {
                            "role": "tool",
                            "name": tc.name,
                            **({"toolset_name": tc.toolset_name} if tc.toolset_name else {}),
                            "tool_call_id": tc.provider_id,
                            "content": content,
                        }
                    )
                else:
                    per_call_text = tool_result_text_for_call_id(
                        step_result,
                        call_id=canonical_call_id,
                        default=None,
                    )
                    tool_result = _provider_tool_text(per_call_text, next_text)
                    is_action_batch = canonical_call_id in lite_action_batch_call_ids
                    provider_visible_result_images = _provider_visible_result_image_pairs(
                        results_by_call_id=results_by_call_id,
                        result_image_indices_by_call_id=result_image_indices_by_call_id,
                        canonical_call_id=canonical_call_id,
                        is_action_batch=is_action_batch,
                        is_final_provider_call=provenance.is_final_for_canonical,
                    )
                    image_blocks = await _image_url_blocks(
                        tuple(image for image, _ in provider_visible_result_images),
                        model_id=model_id,
                        resize_target=resize_target,
                        image_indices=tuple(
                            image_index for _, image_index in provider_visible_result_images
                        ),
                        many_image=many_image,
                    )
                    completion_messages.append(
                        {
                            "role": "tool",
                            "name": tc.name,
                            "tool_call_id": tc.provider_id,
                            "content": str(tool_result),
                        }
                    )
                    if canonical_call_id is not None and image_blocks:
                        completion_messages.append(
                            {
                                "role": "user",
                                "content": image_blocks,
                            }
                        )
        else:
            if next_sent_image_b64 is not None:
                content = []
                if text_block := _provider_text_block(feedback_text):
                    content.append(text_block)
                content.append(
                    mark_provider_visible_image_index(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{next_sent_image_b64}"
                            },
                        },
                        image_index,
                    )
                )
                completion_messages.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )
            elif feedback_text:
                completion_messages.append({"role": "user", "content": next_text})

    return image_bytes, image_index


async def append_mobile_provider_feedback(
    *,
    completion_messages: list[dict[str, Any]],
    response: Any,
    trajectory: LiteSample,
    step_result: LiteEnvStepResult,
    lite_tool_calls: list[dict[str, Any]],
    model_id: str,
    resolution: tuple[int, int] | None,
    many_image: bool,
    provider_tool_uses: tuple[ClaudeProviderToolUse, ...],
    provider_call_provenance: tuple[ProviderCallProvenance, ...],
    provider_errors: dict[str, str],
) -> tuple[bytes, int]:
    """Append Claude mobile tool results for the next provider turn.

    Mutates ``completion_messages`` and ``trajectory``. Returns the next
    observation image bytes and its stored index.
    """
    image_bytes, next_text, next_metadata = latest_step_feedback(step_result)
    if not image_bytes:
        raise RuntimeError("No image returned after env.step()")

    next_sent_image_b64, _, _ = await asyncio.to_thread(
        resize_for_claude_api,
        image_bytes,
        model_id,
        resolution,
        many_image=many_image,
    )

    (
        image_index,
        result_image_indices_by_call_id,
    ) = await append_canonical_step_feedback_messages(
        trajectory,
        step_result,
        lite_tool_calls,
    )

    result_content: list[dict[str, Any]] = [
        mark_provider_visible_image_index(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/webp;base64,{next_sent_image_b64}"},
            },
            image_index,
        ),
    ]
    if text_block := _provider_text_block(next_text):
        result_content.insert(0, text_block)

    if response.choices:
        if provider_tool_uses:
            lite_action_batch_call_ids = frozenset(
                call_id
                for call in lite_tool_calls
                if (call_id := tool_call_id(call))
                and tool_call_name(call) in LITE_ACTION_BATCH_TOOL_NAMES
            )
            results_by_call_id = {
                result.tool_call_id: result
                for result in step_result.results
                if result.tool_call_id is not None
            }
            for tc, provenance in zip(provider_tool_uses, provider_call_provenance):
                if not tc.provider_id:
                    continue
                canonical_call_id = provenance.canonical_call_id
                provider_error = provider_errors.get(tc.provider_id)
                if provider_error is not None:
                    completion_messages.append(
                        {
                            "role": "tool",
                            "name": tc.name,
                            "tool_call_id": tc.provider_id,
                            "content": provider_error,
                        }
                    )
                    continue

                per_call_text = tool_result_text_for_call_id(
                    step_result,
                    call_id=canonical_call_id,
                    default=None,
                )
                is_action_batch = canonical_call_id in lite_action_batch_call_ids
                provider_visible_result_images = _provider_visible_result_image_pairs(
                    results_by_call_id=results_by_call_id,
                    result_image_indices_by_call_id=result_image_indices_by_call_id,
                    canonical_call_id=canonical_call_id,
                    is_action_batch=is_action_batch,
                    is_final_provider_call=provenance.is_final_for_canonical,
                )
                image_blocks = await _image_url_blocks(
                    tuple(image for image, _ in provider_visible_result_images),
                    model_id=model_id,
                    resize_target=resolution,
                    image_indices=tuple(
                        image_index for _, image_index in provider_visible_result_images
                    ),
                    many_image=many_image,
                )
                content: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": _provider_tool_text(per_call_text, next_text),
                    }
                ]
                if image_blocks:
                    content.extend(image_blocks)
                completion_messages.append(
                    {
                        "role": "tool",
                        "name": tc.name,
                        "tool_call_id": tc.provider_id,
                        "content": content,
                    }
                )
        else:
            completion_messages.append({"role": "user", "content": result_content})

    if image_index is None:
        message = build_tool_result_message(
            None,
            (),
            next_text,
            next_metadata,
        )
        if message is not None:
            trajectory.messages.append(message)
        image_index = len(trajectory.images) - 1
    return image_bytes, image_index
