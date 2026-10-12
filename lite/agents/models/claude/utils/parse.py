"""Claude provider response parsing helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.agent.utils.provenance import (
    ProviderCallProvenance,
    provider_call_provenance_from_merge,
)
from lite.agents.models.claude.action_space import (
    ClaudeDesktopActionSpace,
    ClaudeMobileActionSpace,
)
from lite.core import LiteMessage
from lite.core.messages.final import mark_model_output_error
from lite.core.tools.action_space.batches import (
    merge_adjacent_lite_action_batches_with_provenance,
)
from lite.core.tools.calls import make_tool_call, stamp_tool_call_list_ids

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaudeProviderToolUse:
    """Normalized Claude tool-use record for parsing and provider replay."""

    provider_id: str
    name: str
    arguments: dict[str, Any] | None
    replay_arguments: str
    source: Any
    source_type: str
    parse_error: str | None = None
    toolset_name: str | None = None


@dataclass(frozen=True)
class ClaudeProviderParseResult:
    """Claude response parse output plus provider-call provenance.

    ``message`` is the canonical Lite assistant message. The remaining fields
    describe the provider call records from the same parse pass, so provider
    history can acknowledge the original provider ids without reparsing raw
    Claude output. ``replay_content`` is the assistant content Claude must see
    again on the next request, with thinking signatures preserved and
    ``tool_use`` blocks removed because they replay through ``tool_calls``.
    ``finish_reason`` is the chat-completions stop signal from the same pass, so
    the loop never re-normalizes the response to record it.
    """

    message: LiteMessage
    provider_tool_uses: tuple[ClaudeProviderToolUse, ...]
    provider_call_provenance: tuple[ProviderCallProvenance, ...]
    provider_errors: dict[str, str]
    replay_content: str | list[dict[str, Any]] | None
    finish_reason: str | None


@dataclass
class _ClaudeToolCallParse:
    unmerged_lite_tool_calls: list[dict[str, Any]]
    provider_lite_source_groups: list[list[int] | None]
    provider_errors: dict[str, str]
    parse_errors: list[str]


@dataclass(frozen=True)
class _ClaudeResponseRecord:
    assistant_message: Any | None
    finish_reason: str | None


@dataclass(frozen=True)
class ClaudeContentBlocks:
    """Claude assistant content blocks read into dict form, plus unread blocks.

    ``blocks`` preserves provider order. ``errors`` names blocks that were
    neither a dict nor a ``model_dump()`` object, so a malformed block becomes
    parse feedback instead of a silent drop.
    """

    blocks: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class _ClaudeAssistantMessageRecord:
    content_parts: tuple[dict[str, Any], ...]
    thinking_parts: tuple[str, ...]
    provider_tool_uses: tuple[ClaudeProviderToolUse, ...]
    parse_errors: tuple[str, ...]
    replay_content: str | list[dict[str, Any]] | None


def _tool_name_for_error(tool_name: str) -> str:
    return tool_name or "<unknown>"


def _parse_tool_call_args(raw: Any, tool_name: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse liteLLM tool_call arguments without repairing malformed JSON."""
    error = f"malformed tool_call arguments for {_tool_name_for_error(tool_name)}"
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Ignoring Claude tool_call with malformed arguments: %s",
                tool_name,
            )
            return None, error
        if isinstance(parsed, dict):
            return parsed, None
    logger.warning(
        "Ignoring Claude tool_call with non-object arguments: %s",
        tool_name,
    )
    return None, error


def _claude_active_provider_tool_names(tools: list[dict[str, Any]]) -> frozenset[str]:
    """Top-level provider tool names declared in this request.

    Covers both surfaces Claude uses: the native ``computer`` wrapper tool on
    desktop and provider-flat function tools on mobile/grounding.
    """
    names: set[str] = set()
    for tool in tools:
        tool_type = str(tool.get("type", ""))
        function = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if tool_type == "computer_toolset_20260801":
            names.update(ClaudeDesktopActionSpace.get_declared_action_schema_names())
            names.update({"triple_click", "hold_key"})
        elif tool_type.startswith("computer_"):
            names.add(str((function or {}).get("name") or "computer"))
        elif tool_type == "function" and function and function.get("name"):
            names.add(str(function["name"]))
        elif tool.get("name"):
            names.add(str(tool["name"]))
    return frozenset(names)


def _normalize_claude_response(response: Any) -> _ClaudeResponseRecord:
    """Normalize the Claude completion wrapper before parser decisions."""
    if not response.choices:
        return _ClaudeResponseRecord(assistant_message=None, finish_reason=None)
    choice = response.choices[0]
    return _ClaudeResponseRecord(
        assistant_message=choice.message,
        finish_reason=choice.finish_reason,
    )


#: Empty choices carry no assistant message; loops treat them as model-output errors.
_NO_CHOICES_ERROR = "provider returned no choices"


def _empty_choices_message() -> LiteMessage:
    """The assistant message for a response that carries no choice at all.

    ``hasattr(response, "choices")`` is deliberately NOT part of the test:
    ``choices`` is a required field on ``litellm.types.utils.ModelResponse`` and
    a default-constructed instance already carries a non-empty list, so only a
    hand-built fake can omit the attribute. ``choices == []`` IS reachable
    (``ModelResponse(choices=[])``, i.e. a proxy returning ``{"choices": []}``).
    """
    message: dict[str, Any] = {"role": "assistant", "content": [], "tool_calls": []}
    mark_model_output_error(message, _NO_CHOICES_ERROR)
    return message


def _normalize_tool_call_record(tool_call: Any) -> ClaudeProviderToolUse:
    """Read one LiteLLM ``tool_calls`` entry into a provider tool-use record.

    Both raises below are wrong-object-type signals, not malformed model output,
    so they stay bare ``ValueError`` rather than the parse boundary's
    ``ModelToolCallParseError``. The model authors only the tool name and the
    argument payload; the ``model_dump()`` / ``function`` envelope around them is
    built by LiteLLM (``litellm.Message`` validates every entry into a
    ``ChatCompletionMessageToolCall``). Reaching either raise therefore means a
    non-LiteLLM response object or a proxy that broke the tool_call envelope, and
    telling the model its output was malformed would misattribute that. Malformed
    model output rides on ``ClaudeProviderToolUse.parse_error`` instead, which the
    parsers below turn into model-visible feedback.
    """
    model_dump = getattr(tool_call, "model_dump", None)
    if not callable(model_dump):
        raise ValueError(
            "Malformed Claude provider tool_call record: expected "
            "LiteLLM tool_call with model_dump()"
        )
    data = model_dump()
    function = data.get("function") if isinstance(data, dict) else None
    if not isinstance(function, dict):
        raise ValueError(
            "Malformed Claude provider tool_call record: expected "
            "id, function.name, and function.arguments"
        )
    raw_id = data.get("id")
    raw_name = function.get("name")
    raw_arguments = function.get("arguments")
    name = str(raw_name or "")
    replay_arguments = raw_arguments if isinstance(raw_arguments, str) else ""
    arguments, parse_error = _parse_tool_call_args(raw_arguments, name)
    provider_id = str(raw_id or "")
    if not provider_id:
        parse_error = f"missing provider id for {_tool_name_for_error(name)}"
    return ClaudeProviderToolUse(
        provider_id=provider_id,
        name=name,
        arguments=arguments,
        replay_arguments=replay_arguments,
        source=tool_call,
        source_type="tool_call",
        parse_error=parse_error,
    )


def _normalize_content_tool_use(block: dict[str, Any]) -> ClaudeProviderToolUse:
    name = str(block.get("name") or "")
    raw_input = block.get("input", {})
    replay_arguments = json.dumps({} if raw_input is None else raw_input)
    parse_error: str | None = None
    if raw_input is None:
        arguments: dict[str, Any] | None = {}
    elif isinstance(raw_input, dict):
        arguments = dict(raw_input)
    else:
        arguments = None
        parse_error = f"malformed tool_use input for {_tool_name_for_error(name)}"
    provider_id = str(block.get("id") or "")
    if not provider_id:
        parse_error = f"missing provider id for {_tool_name_for_error(name)}"
    return ClaudeProviderToolUse(
        provider_id=provider_id,
        name=name,
        arguments=arguments,
        replay_arguments=replay_arguments,
        source=block,
        source_type="content_tool_use",
        parse_error=parse_error,
        toolset_name=block.get("toolset_name"),
    )


def claude_content_blocks(content: list[Any]) -> ClaudeContentBlocks:
    """Read Claude assistant content blocks into dict form once.

    Anthropic content blocks reach CUA-Lite either as plain JSON dicts (proxy
    responses) or as LiteLLM/SDK objects exposing ``model_dump()``. This is the
    only place Claude code resolves that difference, so parse and provider
    replay never re-sniff raw block shape. Anything else is malformed provider
    output and is reported through ``ClaudeContentBlocks.errors``.
    """
    blocks: list[dict[str, Any]] = []
    errors: list[str] = []
    for block in content:
        if isinstance(block, dict):
            blocks.append(block)
            continue
        model_dump = getattr(block, "model_dump", None)
        dumped = model_dump() if callable(model_dump) else None
        if isinstance(dumped, dict):
            blocks.append(dumped)
            continue
        error = f"unreadable Claude content block of type {type(block).__name__}"
        logger.warning("Ignoring %s", error)
        errors.append(error)
    return ClaudeContentBlocks(blocks=tuple(blocks), errors=tuple(errors))


def _normalize_claude_assistant_message(message: Any) -> _ClaudeAssistantMessageRecord:
    """Normalize Claude assistant-message content and tool-use shape once.

    One pass over the assistant message produces everything downstream needs:
    canonical Lite content/thinking parts, the provider tool-use records, and
    the content Claude replays on the next request. Provider history therefore
    never re-reads the raw response.
    """
    content = message.content
    tool_calls = message.tool_calls or []
    tool_uses = tuple(_normalize_tool_call_record(tool_call) for tool_call in tool_calls)
    content_tool_uses: list[ClaudeProviderToolUse] = []
    content_parts: list[dict[str, Any]] = []
    thinking_parts: list[str] = []
    parse_errors: tuple[str, ...] = ()
    replay_content: str | list[dict[str, Any]] | None = None

    if isinstance(content, str):
        if content.strip():
            content_parts.append({"type": "action_description", "text": content})
        replay_content = content or None
    elif isinstance(content, list):
        read = claude_content_blocks(content)
        parse_errors = read.errors
        # Unreadable blocks cannot be replayed; ``read.errors`` already turns
        # them into model output errors, so replay silently omits them.
        replay_blocks: list[dict[str, Any]] = []
        for block in read.blocks:
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if text.strip():
                    content_parts.append({"type": "action_description", "text": text})
                if text:
                    replay_blocks.append({"type": "text", "text": text})
            elif block_type == "thinking":
                thinking = block.get("thinking", "")
                if thinking.strip():
                    thinking_parts.append(thinking)
                # Anthropic rejects a replayed thinking block without its
                # original signature, so the signature rides along untouched.
                replay_block: dict[str, Any] = {"type": "thinking", "thinking": thinking}
                signature = block.get("signature")
                if signature is not None:
                    replay_block["signature"] = signature
                replay_blocks.append(replay_block)
            elif block_type == "tool_use":
                # Replayed through the assistant ``tool_calls`` field, not content.
                if not tool_uses:
                    content_tool_uses.append(_normalize_content_tool_use(block))
            else:
                replay_blocks.append(block)
        replay_content = replay_blocks or None

    return _ClaudeAssistantMessageRecord(
        content_parts=tuple(content_parts),
        thinking_parts=tuple(thinking_parts),
        provider_tool_uses=tool_uses or tuple(content_tool_uses),
        parse_errors=parse_errors,
        replay_content=replay_content,
    )


def _record_provider_error(
    provider_errors: dict[str, str],
    tc: ClaudeProviderToolUse,
    error: str,
) -> None:
    if tc.provider_id:
        provider_errors[tc.provider_id] = (
            f"model output error: {error}. This tool_use was not executed. "
            "Return a valid tool_use in the required format, or plain final text if finished."
        )


def _parse_desktop_provider_tool_calls(
    provider_tool_uses: tuple[ClaudeProviderToolUse, ...],
    *,
    action_space: ClaudeDesktopActionSpace,
    resolution: tuple[int, int],
    scale_x: float,
    scale_y: float,
    extra_tool_names: frozenset[str],
    active_provider_tool_names: frozenset[str] | None = None,
) -> _ClaudeToolCallParse:
    # Desktop Claude's GUI surface is Anthropic's opaque native computer tool.
    # The tool schema list is therefore empty even though native GUI actions are
    # valid, so desktop admission must read the action layer, not the tool layer.
    # A declared provider-flat tool name (grounding ``left_click``) doubles as the
    # native ``action=`` value here, so the active provider tool names are read as
    # action values. Mobile uses provider-flat function tools and has its own path.
    native_action_values = (
        set(type(action_space).get_declared_action_schema_names())
        if active_provider_tool_names is None
        else set(active_provider_tool_names)
    )
    computer_tool_allowed = (
        active_provider_tool_names is None or "computer" in active_provider_tool_names
    )
    provider_lite_source_groups: list[list[int] | None] = []
    unmerged_lite_tool_calls: list[dict[str, Any]] = []
    provider_errors: dict[str, str] = {}
    parse_errors: list[str] = []

    for tc in provider_tool_uses:
        source_start = len(unmerged_lite_tool_calls)
        if tc.parse_error is not None:
            logger.warning("Ignoring malformed Claude provider tool_use: %s", tc.parse_error)
            parse_errors.append(tc.parse_error)
            _record_provider_error(provider_errors, tc, tc.parse_error)
            provider_lite_source_groups.append(None)
            continue

        args = dict(tc.arguments or {})
        if tc.name == "computer":
            if not computer_tool_allowed:
                logger.warning("Ignoring undeclared Claude tool_call: computer")
                error = "undeclared tool_call computer"
                parse_errors.append(error)
                _record_provider_error(provider_errors, tc, "undeclared tool_call: computer")
                provider_lite_source_groups.append(None)
                continue
            try:
                action = args
                _scale_action_coords(action, scale_x, scale_y)
                unmerged_lite_tool_calls.extend(
                    action_space.convert_tool_calls_from_agent(
                        [action],
                        resolution=resolution,
                    )
                )
            # Only the action-space parse boundary's own malformed-model-output
            # signal becomes model-visible feedback. An incidental TypeError or
            # ValueError here is a converter bug and stays loud.
            except ModelToolCallParseError as e:
                logger.warning("Ignoring malformed Claude tool_call: %s", e)
                error = str(e)
                parse_errors.append(error)
                _record_provider_error(provider_errors, tc, error)
        elif tc.name in extra_tool_names and tc.toolset_name != "computer":
            # An advertised env tool routes to the env by name. Argument
            # admission belongs to env ingress (``prepare_env_tool_calls``),
            # which answers a bad value with feedback keyed to this call id;
            # validating the schema here would delete the call before the model
            # could be told why.
            unmerged_lite_tool_calls.append(make_tool_call(tc.name, args))
        elif tc.name in native_action_values:
            try:
                action = {"action": tc.name, **args}
                _scale_action_coords(action, scale_x, scale_y)
                unmerged_lite_tool_calls.extend(
                    action_space.convert_tool_calls_from_agent(
                        [action],
                        resolution=resolution,
                    )
                )
            except ModelToolCallParseError as e:
                logger.warning("Ignoring malformed Claude tool_call: %s", e)
                error = str(e)
                parse_errors.append(error)
                _record_provider_error(provider_errors, tc, error)
        else:
            logger.warning("Ignoring undeclared Claude tool_call: %s", tc.name)
            error = f"undeclared tool_call {_tool_name_for_error(tc.name)}"
            parse_errors.append(error)
            _record_provider_error(
                provider_errors,
                tc,
                f"undeclared tool_call: {_tool_name_for_error(tc.name)}",
            )

        provider_lite_source_groups.append(
            list(range(source_start, len(unmerged_lite_tool_calls))) or None
        )

    return _ClaudeToolCallParse(
        unmerged_lite_tool_calls=unmerged_lite_tool_calls,
        provider_lite_source_groups=provider_lite_source_groups,
        provider_errors=provider_errors,
        parse_errors=parse_errors,
    )


def _parse_mobile_provider_tool_calls(
    provider_tool_uses: tuple[ClaudeProviderToolUse, ...],
    *,
    action_space: ClaudeMobileActionSpace,
    resolution: tuple[int, int],
    scale_x: float,
    scale_y: float,
    extra_tool_names: frozenset[str],
    active_provider_tool_names: frozenset[str] | None = None,
) -> _ClaudeToolCallParse:
    if active_provider_tool_names is None:
        provider_tool_names = type(action_space).get_tool_names()
    else:
        provider_tool_names = set(active_provider_tool_names)
    provider_lite_source_groups: list[list[int] | None] = []
    unmerged_lite_tool_calls: list[dict[str, Any]] = []
    provider_errors: dict[str, str] = {}
    parse_errors: list[str] = []

    def append_lite_call_from_provider_mobile_tool(name: str, tool_input: dict[str, Any]) -> None:
        args = dict(tool_input)
        _scale_action_coords(args, scale_x, scale_y)
        unmerged_lite_tool_calls.extend(
            action_space.convert_tool_calls_from_agent(
                [{"name": name, "arguments": args}],
                resolution=resolution,
            )
        )

    for tc in provider_tool_uses:
        source_start = len(unmerged_lite_tool_calls)
        if tc.parse_error is not None:
            logger.warning("Ignoring malformed Claude mobile provider tool_use: %s", tc.parse_error)
            parse_errors.append(tc.parse_error)
            _record_provider_error(provider_errors, tc, tc.parse_error)
            provider_lite_source_groups.append(None)
            continue

        args = dict(tc.arguments or {})
        if tc.name in extra_tool_names:
            # Env extras route to the env by name; see the desktop parser.
            unmerged_lite_tool_calls.append(make_tool_call(tc.name, args))
        elif tc.name in provider_tool_names:
            try:
                append_lite_call_from_provider_mobile_tool(tc.name, args)
            # Malformed model output only; a converter bug stays loud.
            except ModelToolCallParseError as e:
                logger.warning(
                    "Ignoring malformed Claude mobile tool arguments for %s: %s",
                    tc.name,
                    e,
                )
                error = str(e)
                parse_errors.append(error)
                _record_provider_error(provider_errors, tc, error)
        else:
            logger.warning("Ignoring undeclared Claude mobile tool_call: %s", tc.name)
            error = f"undeclared tool_call {_tool_name_for_error(tc.name)}"
            parse_errors.append(error)
            _record_provider_error(
                provider_errors,
                tc,
                f"undeclared tool_call: {_tool_name_for_error(tc.name)}",
            )

        provider_lite_source_groups.append(
            list(range(source_start, len(unmerged_lite_tool_calls))) or None
        )

    return _ClaudeToolCallParse(
        unmerged_lite_tool_calls=unmerged_lite_tool_calls,
        provider_lite_source_groups=provider_lite_source_groups,
        provider_errors=provider_errors,
        parse_errors=parse_errors,
    )


def _finish_provider_parse(
    *,
    content_parts: list[dict[str, Any]],
    thinking_parts: list[str],
    content_parse_errors: tuple[str, ...],
    call_parse: _ClaudeToolCallParse,
    provider_tool_uses: tuple[ClaudeProviderToolUse, ...],
    replay_content: str | list[dict[str, Any]] | None,
    finish_reason: str | None,
    call_id_start: int,
) -> ClaudeProviderParseResult:
    # Claude's native GUI surfaces carry one action per provider call. Adjacent
    # GUI calls merge into one canonical action-batch call, and the returned
    # provenance is the only provider-id -> canonical-id map history should use.
    merge = merge_adjacent_lite_action_batches_with_provenance(call_parse.unmerged_lite_tool_calls)
    lite_tool_calls = merge.tool_calls
    stamp_tool_call_list_ids(lite_tool_calls, start=call_id_start, preserve=False)
    provider_call_provenance = provider_call_provenance_from_merge(
        call_parse.provider_lite_source_groups,
        merge,
    )

    if not lite_tool_calls:
        for part in content_parts:
            if part.get("type") == "action_description":
                part["type"] = "text"

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content_parts,
        "tool_calls": lite_tool_calls,
    }
    if thinking_parts:
        message["reasoning_content"] = "\n".join(thinking_parts)
    parse_errors = [*content_parse_errors, *call_parse.parse_errors]
    if parse_errors and not lite_tool_calls:
        mark_model_output_error(message, "; ".join(parse_errors))

    return ClaudeProviderParseResult(
        message=message,
        provider_tool_uses=provider_tool_uses,
        provider_call_provenance=provider_call_provenance,
        provider_errors=call_parse.provider_errors,
        replay_content=replay_content,
        finish_reason=finish_reason,
    )


def parse_response_with_provenance(
    response: Any,
    scale_x: float,
    scale_y: float,
    action_space: ClaudeDesktopActionSpace,
    resolution: tuple[int, int],
    extra_tool_names: frozenset[str] = frozenset(),
    call_id_start: int = 0,
    *,
    active_provider_tool_names: frozenset[str] | None = None,
) -> ClaudeProviderParseResult:
    """Parse a Claude desktop response into canonical Lite output and provenance.

    Extracts text content and tool_use blocks, converts Claude computer actions
    to CUA-lite format with coordinate upscaling, and routes calls naming an
    advertised env extra tool to the env verbatim.
    """
    normalized_response = _normalize_claude_response(response)
    if normalized_response.assistant_message is None:
        return ClaudeProviderParseResult(
            message=_empty_choices_message(),
            provider_tool_uses=(),
            provider_call_provenance=(),
            provider_errors={},
            replay_content=None,
            finish_reason=normalized_response.finish_reason,
        )

    assistant = _normalize_claude_assistant_message(normalized_response.assistant_message)
    call_parse = _parse_desktop_provider_tool_calls(
        assistant.provider_tool_uses,
        action_space=action_space,
        resolution=resolution,
        scale_x=scale_x,
        scale_y=scale_y,
        extra_tool_names=extra_tool_names,
        active_provider_tool_names=active_provider_tool_names,
    )
    return _finish_provider_parse(
        content_parts=[dict(part) for part in assistant.content_parts],
        thinking_parts=list(assistant.thinking_parts),
        content_parse_errors=assistant.parse_errors,
        call_parse=call_parse,
        provider_tool_uses=assistant.provider_tool_uses,
        replay_content=assistant.replay_content,
        finish_reason=normalized_response.finish_reason,
        call_id_start=call_id_start,
    )


def _scale_action_coords(action: dict[str, Any], scale_x: float, scale_y: float):
    """Upscale coordinates in a Claude action dict in-place."""
    if scale_x == 1.0 and scale_y == 1.0:
        return

    for key in ("coordinate", "start_coordinate", "end_coordinate"):
        if key not in action:
            continue
        coord = action.get(key)
        if coord is None:
            continue
        if not (
            isinstance(coord, list)
            and len(coord) >= 2
            and isinstance(coord[0], (int, float))
            and isinstance(coord[1], (int, float))
            and not isinstance(coord[0], bool)
            and not isinstance(coord[1], bool)
        ):
            action_name = action.get("action", "action")
            raise ModelToolCallParseError(f"{action_name} requires valid {key}")
        action[key] = [
            int(round(coord[0] * scale_x)),
            int(round(coord[1] * scale_y)),
        ]


def parse_mobile_response_with_provenance(
    response: Any,
    scale_x: float,
    scale_y: float,
    action_space: ClaudeMobileActionSpace,
    resolution: tuple[int, int],
    extra_tool_names: frozenset[str] = frozenset(),
    call_id_start: int = 0,
    *,
    active_provider_tool_names: frozenset[str] | None = None,
) -> ClaudeProviderParseResult:
    """Parse a Claude mobile response into canonical Lite output and provenance.

    Env extra tools pass through as standalone top-level functions, with
    argument admission left to env ingress. Native mobile GUI actions are
    provider-flat function tools and normalize to canonical Lite
    ``mobile(actions=[...])`` batches.
    """
    normalized_response = _normalize_claude_response(response)
    if normalized_response.assistant_message is None:
        return ClaudeProviderParseResult(
            message=_empty_choices_message(),
            provider_tool_uses=(),
            provider_call_provenance=(),
            provider_errors={},
            replay_content=None,
            finish_reason=normalized_response.finish_reason,
        )

    assistant = _normalize_claude_assistant_message(normalized_response.assistant_message)
    call_parse = _parse_mobile_provider_tool_calls(
        assistant.provider_tool_uses,
        action_space=action_space,
        resolution=resolution,
        scale_x=scale_x,
        scale_y=scale_y,
        extra_tool_names=extra_tool_names,
        active_provider_tool_names=active_provider_tool_names,
    )
    return _finish_provider_parse(
        content_parts=[dict(part) for part in assistant.content_parts],
        thinking_parts=list(assistant.thinking_parts),
        content_parse_errors=assistant.parse_errors,
        call_parse=call_parse,
        provider_tool_uses=assistant.provider_tool_uses,
        replay_content=assistant.replay_content,
        finish_reason=normalized_response.finish_reason,
        call_id_start=call_id_start,
    )
