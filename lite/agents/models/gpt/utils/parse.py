"""GPT Responses-API output parsing helpers.

Owns the one pass that turns Responses ``output`` items into a canonical Lite
assistant message plus the provider-call provenance the GPT loops replay:
``_normalized_gpt_output_records`` normalizes item shape, and the two parsers
route each call through one declared surface. The response record, failure
surface, and cache/usage telemetry live in
``lite.agents.models.gpt.utils.responses``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.agent.utils.provenance import (
    provider_call_provenance_from_merge,
)
from lite.agents.models.gpt.utils.convert import (
    lite_calls_from_gpt_computer_actions,
    lite_calls_from_gpt_mobile_function_calls,
    parse_provider_function_args,
)
from lite.core import (
    LiteMessage,
)
from lite.core.messages.final import mark_model_output_error
from lite.core.tools.action_space.batches import (
    merge_adjacent_lite_action_batches_with_provenance,
)
from lite.core.tools.calls import make_tool_call, stamp_tool_call_list_ids

if TYPE_CHECKING:  # annotations only — avoids any import cycle at runtime
    from lite.agents.models.gpt.action_space import (
        GPTDesktopActionSpace,
        GPTMobileActionSpace,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPTProviderCall:
    """One GPT provider call plus its canonical Lite call provenance."""

    item: dict[str, Any]
    item_type: str
    provider_call_id: str
    canonical_call_id: str | None
    error: str | None
    is_final_for_canonical: bool


@dataclass(frozen=True)
class GPTParsedOutput:
    """GPT provider output parsed once for Lite storage and provider feedback."""

    message: LiteMessage
    provider_calls: tuple[GPTProviderCall, ...]


@dataclass(frozen=True)
class _GPTOutputRecord:
    """Normalized GPT Responses output item used by one parse pass."""

    item: dict[str, Any]
    item_type: str
    provider_call_id: str
    name: str
    arguments: dict[str, Any] | None
    actions: tuple[dict[str, Any], ...]
    message_texts: tuple[str, ...]
    reasoning_texts: tuple[str, ...]
    parse_error: str | None


def _provider_parse_error(error: str) -> str:
    if error.startswith("undeclared function_call "):
        error = error.replace("undeclared function_call ", "undeclared function_call: ", 1)
    return (
        f"model output error: {error}. This tool call was not executed. "
        "Return a valid tool_call in the required format, or plain final text if finished."
    )


def _normalized_gpt_output_records(
    output_items: list[dict[str, Any]],
) -> tuple[_GPTOutputRecord, ...]:
    """Normalize Responses output-item shape once before provider routing."""
    records: list[_GPTOutputRecord] = []
    for item in output_items:
        item_type = str(item.get("type", ""))
        provider_call_id = ""
        name = ""
        arguments: dict[str, Any] | None = None
        actions: tuple[dict[str, Any], ...] = ()
        message_texts: tuple[str, ...] = ()
        reasoning_texts: tuple[str, ...] = ()
        parse_error: str | None = None

        if item_type in {"computer_call", "function_call"}:
            provider_call_id = str(item.get("call_id") or item.get("id") or "")

        if item_type == "message":
            texts: list[str] = []
            for content_item in item.get("content", []) or []:
                if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                    text = content_item.get("text", "")
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
            message_texts = tuple(texts)
        elif item_type == "computer_call":
            name = "computer"
            raw_actions = item.get("actions")
            if isinstance(raw_actions, list) and raw_actions:
                actions = tuple(action for action in raw_actions if isinstance(action, dict))
        elif item_type == "function_call":
            name = str(item.get("name") or "")
            arguments = parse_provider_function_args(item)
            if arguments is None:
                parse_error = f"malformed function_call arguments for {name or '<unknown>'}"
        elif item_type == "reasoning":
            texts = []
            for summary in item.get("summary", []) or []:
                if not isinstance(summary, dict):
                    continue
                text = summary.get("text", "").strip()
                if text:
                    texts.append(text)
            reasoning_texts = tuple(texts)

        # A call the provider did not identify cannot be executed or answered:
        # an empty ``call_id`` would be echoed back verbatim in
        # ``computer_call_output`` / ``function_call_output``. Record it as a
        # parse error so the model sees the failure instead.
        if item_type in {"computer_call", "function_call"} and not provider_call_id:
            parse_error = f"missing provider id for {name or '<unknown>'}"

        # A call the provider marked ``incomplete`` carries no action to run: the
        # Responses API emits this shape when the item is cut off at
        # ``max_output_tokens`` mid-emission. Without a parse error the turn parses
        # to ZERO tool calls and no feedback, so the loop reads it as a DELIBERATE
        # content-only final and ends the episode -- a truncated turn is scored as
        # an agent that chose to stop, on an env it never touched.
        if (
            item_type in {"computer_call", "function_call"}
            and parse_error is None
            and str(item.get("status", "")) == "incomplete"
        ):
            parse_error = (
                f"truncated {item_type} for {name or '<unknown>'}: the provider "
                "cut the item off (max_output_tokens); emit a shorter action"
            )

        records.append(
            _GPTOutputRecord(
                item=item,
                item_type=item_type,
                provider_call_id=provider_call_id,
                name=name,
                arguments=arguments,
                actions=actions,
                message_texts=message_texts,
                reasoning_texts=reasoning_texts,
                parse_error=parse_error,
            )
        )
    return tuple(records)


def parse_output_items_with_provenance(
    output_items: list[dict[str, Any]],
    action_space: GPTDesktopActionSpace,
    resolution: tuple[int, int],
    extra_tool_names: frozenset[str] = frozenset(),
    declared_agent_tool_names: frozenset[str] = frozenset(),
    call_id_start: int = 0,
    *,
    active_provider_tool_names: frozenset[str] | None = None,
) -> GPTParsedOutput:
    """Parse Responses API output items into a Lite message and provider provenance.

    The parser routes each call through one declared surface: native computer,
    env-supplied extra tools, or agent-declared function tools. Text output stays
    as plain ``text`` content.
    """
    content_parts: list[dict[str, Any]] = []
    unmerged_lite_tool_calls: list[dict[str, Any]] = []
    # One entry per provider call item, in provider order, paired with the
    # feedback text the model must see when that call was not executed.
    provider_calls: list[tuple[_GPTOutputRecord, str | None]] = []
    provider_lite_source_groups: list[list[int] | None] = []
    parse_errors: list[str] = []
    for record in _normalized_gpt_output_records(output_items):
        item_type = record.item_type

        # Text output → a plain ``text`` part (the model's prose, e.g. a
        # Memory/Progress block).
        if item_type == "message":
            part_type = "text"
            for text in record.message_texts:
                content_parts.append({"type": part_type, "text": text})

        # Native computer_call returns batched `actions`
        elif item_type == "computer_call":
            source_start = len(unmerged_lite_tool_calls)
            error = record.parse_error
            if error is not None:
                logger.warning("Ignoring malformed GPT computer_call: %s", error)
                parse_errors.append(error)
            elif (
                active_provider_tool_names is not None
                and "computer" not in active_provider_tool_names
            ):
                logger.warning("Ignoring undeclared GPT computer_call")
                error = "undeclared computer_call"
                parse_errors.append(error)
            else:
                if record.actions:
                    try:
                        unmerged_lite_tool_calls.extend(
                            lite_calls_from_gpt_computer_actions(
                                action_space,
                                list(record.actions),
                                resolution,
                            )
                        )
                    # Only the action-space parse boundary's own
                    # malformed-model-output signal becomes model-visible
                    # feedback. An incidental TypeError or ValueError here is a
                    # converter bug and stays loud.
                    except ModelToolCallParseError as e:
                        logger.warning("Ignoring malformed GPT computer_call: %s", e)
                        error = str(e)
                        parse_errors.append(error)
            provider_calls.append((record, _provider_parse_error(error) if error else None))
            provider_lite_source_groups.append(
                list(range(source_start, len(unmerged_lite_tool_calls))) or None
            )

        # Function call: env extra OR agent-declared function tool. Native
        # computer actions come via computer_call, not here.
        elif item_type == "function_call":
            source_start = len(unmerged_lite_tool_calls)
            error = record.parse_error
            fn_name = record.name
            fn_args = record.arguments if record.arguments is not None else {}
            if error is not None:
                logger.warning("Ignoring GPT function_call with malformed arguments: %s", fn_name)
                parse_errors.append(error)
            elif fn_name in extra_tool_names:
                # An advertised env tool routes to the env by name. Argument
                # admission belongs to env ingress
                # (``prepare_env_tool_calls``), which answers a bad value with
                # feedback keyed to this call id; validating the schema here
                # would delete the call before the model could be told why.
                unmerged_lite_tool_calls.append(make_tool_call(fn_name, fn_args))
            elif fn_name in declared_agent_tool_names:
                # Agent-declared function tools route through the action space.
                try:
                    unmerged_lite_tool_calls.extend(
                        lite_calls_from_gpt_computer_actions(
                            action_space,
                            [{"type": fn_name, **fn_args}],
                            resolution,
                        )
                    )
                except ModelToolCallParseError as e:
                    logger.warning("Ignoring malformed GPT function_call: %s", e)
                    error = str(e)
                    parse_errors.append(error)
            else:
                logger.warning("Ignoring undeclared GPT function_call: %s", fn_name)
                error = f"undeclared function_call {fn_name or '<unknown>'}"
                parse_errors.append(error)
            provider_calls.append((record, _provider_parse_error(error) if error else None))
            provider_lite_source_groups.append(
                list(range(source_start, len(unmerged_lite_tool_calls))) or None
            )

        # Reasoning summary → InlineReasoningContent part. The OpenAI Responses
        # API exposes ``reasoning.summary`` (concise, user-visible). It is a
        # *summary* of the model's native reasoning trace, not the trace
        # itself, so per the LiteMessage type contract it belongs in
        # ``InlineReasoningContent`` inside ``content`` rather than the
        # top-level ``reasoning_content`` field.
        elif item_type == "reasoning":
            for text in record.reasoning_texts:
                content_parts.append({"type": "inline_reasoning", "text": text})

    merge = merge_adjacent_lite_action_batches_with_provenance(unmerged_lite_tool_calls)
    lite_tool_calls = merge.tool_calls
    stamp_tool_call_list_ids(lite_tool_calls, start=call_id_start, preserve=False)
    provider_provenance = provider_call_provenance_from_merge(provider_lite_source_groups, merge)
    result: LiteMessage = {
        "role": "assistant",
        "content": content_parts,
        "tool_calls": lite_tool_calls,
    }
    if parse_errors and not result["tool_calls"]:
        mark_model_output_error(result, "; ".join(parse_errors))
    return GPTParsedOutput(
        message=result,
        provider_calls=tuple(
            GPTProviderCall(
                item=record.item,
                item_type=record.item_type,
                provider_call_id=record.provider_call_id,
                canonical_call_id=provenance.canonical_call_id,
                error=error,
                is_final_for_canonical=provenance.is_final_for_canonical,
            )
            for (record, error), provenance in zip(provider_calls, provider_provenance)
        ),
    )


def parse_gpt_mobile_output_items_with_provenance(
    output_items: list[dict[str, Any]],
    action_space: GPTMobileActionSpace,
    resolution: tuple[int, int],
    extra_tool_names: frozenset[str] = frozenset(),
    declared_agent_tool_names: frozenset[str] = frozenset(),
    call_id_start: int = 0,
    *,
    active_provider_tool_names: frozenset[str] | None = None,
) -> GPTParsedOutput:
    """Parse mobile Responses output into Lite output and provider provenance.

    Mobile actions arrive as function tools. The parser routes provider-flat
    mobile actions, env extras, and agent-declared functions through their
    declared surfaces. Text output stays as plain ``text`` content.
    """
    content_parts: list[dict[str, Any]] = []
    unmerged_lite_tool_calls: list[dict[str, Any]] = []
    # One entry per provider call item, in provider order, paired with the
    # feedback text the model must see when that call was not executed.
    provider_calls: list[tuple[_GPTOutputRecord, str | None]] = []
    provider_lite_source_groups: list[list[int] | None] = []
    parse_errors: list[str] = []
    if active_provider_tool_names is None:
        provider_tool_names = {
            *action_space.get_tool_names(),
            *declared_agent_tool_names,
        }
    else:
        provider_tool_names = set(active_provider_tool_names)

    for record in _normalized_gpt_output_records(output_items):
        item_type = record.item_type

        if item_type == "message":
            part_type = "text"
            for text in record.message_texts:
                content_parts.append({"type": part_type, "text": text})
        elif item_type == "function_call":
            source_start = len(unmerged_lite_tool_calls)
            error = record.parse_error
            fn_name = record.name
            fn_args = record.arguments if record.arguments is not None else {}
            if error is not None:
                logger.warning(
                    "Ignoring GPT mobile function_call with malformed arguments: %s",
                    fn_name,
                )
                parse_errors.append(error)
            elif fn_name in extra_tool_names:
                # Env extras route to the env by name; see the desktop parser.
                unmerged_lite_tool_calls.append(make_tool_call(fn_name, fn_args))
            elif fn_name in provider_tool_names:
                try:
                    unmerged_lite_tool_calls.extend(
                        lite_calls_from_gpt_mobile_function_calls(
                            action_space,
                            [{"name": fn_name, "arguments": fn_args}],
                            resolution,
                        )
                    )
                # Malformed model output only; a converter bug stays loud.
                except ModelToolCallParseError as e:
                    logger.warning(
                        "Ignoring GPT mobile function_call with malformed GUI arguments: %s",
                        fn_name,
                    )
                    error = str(e)
                    parse_errors.append(error)
            else:
                logger.warning("Ignoring undeclared GPT mobile function_call: %s", fn_name)
                error = f"undeclared function_call {fn_name or '<unknown>'}"
                parse_errors.append(error)
            provider_calls.append((record, _provider_parse_error(error) if error else None))
            provider_lite_source_groups.append(
                list(range(source_start, len(unmerged_lite_tool_calls))) or None
            )
        elif item_type == "reasoning":
            for text in record.reasoning_texts:
                content_parts.append({"type": "inline_reasoning", "text": text})

    merge = merge_adjacent_lite_action_batches_with_provenance(unmerged_lite_tool_calls)
    lite_tool_calls = merge.tool_calls
    stamp_tool_call_list_ids(lite_tool_calls, start=call_id_start, preserve=False)
    provider_provenance = provider_call_provenance_from_merge(provider_lite_source_groups, merge)
    result: LiteMessage = {
        "role": "assistant",
        "content": content_parts,
        "tool_calls": lite_tool_calls,
    }
    if parse_errors and not result["tool_calls"]:
        mark_model_output_error(result, "; ".join(parse_errors))
    return GPTParsedOutput(
        message=result,
        provider_calls=tuple(
            GPTProviderCall(
                item=record.item,
                item_type=record.item_type,
                provider_call_id=record.provider_call_id,
                canonical_call_id=provenance.canonical_call_id,
                error=error,
                is_final_for_canonical=provenance.is_final_for_canonical,
            )
            for (record, error), provenance in zip(provider_calls, provider_provenance)
        ),
    )
