"""Surface-aware validators for canonical Lite tool calls.

Run:
    uv run pytest tests/data/utils/test_rows_canonical.py -q
"""

from __future__ import annotations

import json
from typing import Any

from lite.core.errors import LiteContractError
from lite.core.messages.content import (
    ASSISTANT_ROLE,
    TEXT_PART,
    TOOL_ROLE,
    validate_message_content_parts,
    validate_message_roles,
)
from lite.core.messages.final import (
    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY,
    MODEL_OUTPUT_ERROR_KEY,
)
from lite.core.messages.image_refs import validate_image_references
from lite.core.metadata import (
    SINGLE_TURN_TASK_TYPES,
    LiteBaseMetadata,
    LiteCUAMetadata,
    metadata_from_dict,
)
from lite.core.tools.action_space import (
    COORDINATE_ARGUMENT_NAMES,
    LITE_ACTION_BATCH_TOOL_NAMES,
    LITE_ACTION_SET_TOOL_NAMES,
    LITE_DESKTOP_KEY_ACTION_NAMES,
    action_coordinate_arguments_out_of_range,
    lite_builtin_tool_names_for_metadata,
    norm_coord_out_of_range,
    validate_lite_action_batch_structure,
)
from lite.core.tools.action_space.batches import (
    LiteActionBatchValidationError,
    LiteActionBatchValidationKind,
    lite_action_batch_child_name_errors,
    validate_lite_action_batch_child_arguments,
)
from lite.core.tools.action_space.keys import is_canonical_key_token
from lite.core.tools.calls import (
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
    validate_lite_tool_call,
)
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.results import extract_projected_tool_result_error, project_tool_result_text
from lite.core.tools.schemas import (
    tool_call_satisfies_schema,
    tool_name_and_arguments_match_schema_route_keys,
    tool_schema_name,
)
from lite.data.utils import messages as message_utils

#: Remediation appended by the publication gate when a row still carries the
#: retired flat Lite call shape (``call_id``/``name``/``arguments``), the one
#: failure ``devs/migration`` can actually rewrite.
_MIGRATION_HINT = (
    "; run devs/migration on the source tree "
    "(uv run python -m devs.migration.run <src> -o <out> --verify) "
    "before export/runtime consumption"
)
#: Alias of the core-owned grounding cohort, under the module-private name this
#: layer's callers and tests use.


#: Model-family spellings of a canonical finish tool -> the finish tool(s) each
#: stands for. A row arrives ANONYMOUS -- Lite metadata names no model family --
#: so the union over all families is the only answer available here.
#:
#: Written out rather than derived because the row validators must not import
#: ``lite.agents``; ``tests/data/utils/test_rows_dialect_census.py`` recomputes it from the family
#: declarations and reddens on drift.
#:
#: The value is a SET because one native spelling can mean either finish tool
#: depending on its arguments, or on which family emitted it: ui_tars
#: ``finished(content=…)`` is a ``response`` and bare ``finished()`` a
#: ``terminate``; ``call_user`` carries the answer text for qwen3_8 and is a
#: bare give-up for ui_tars_15_v1.
#:
#: Canonical spellings are absent (``LiteFinishToolSet.get_tool_names()`` covers them),
#: and so are dialect spellings of NON-finish tools -- every question here is
#: about termination.
_NATIVE_FINISH_TOOL_ALIASES: dict[str, frozenset[str]] = {
    "ABORT": frozenset({"terminate"}),
    "COMPLETE": frozenset({"terminate"}),
    "INFO": frozenset({"response"}),
    "answer": frozenset({"response"}),
    "call_user": frozenset({"response", "terminate"}),
    "finished": frozenset({"response", "terminate"}),
}
#: The raw-rollout contract name. Also user-facing prose: ``row_contract`` is
#: interpolated into ``f"{label}: row {i} is not {row_contract}"``.
_RAW = "raw rollout"
_CANONICAL_METADATA_KEYS = frozenset({
    "metadata_kind",
    "dims",
    "extra_tool_schemas",
    "valid_actions",
    "others",
})


class _RowValidationError(ValueError, LiteContractError):
    """Row-boundary validation failure with the row label already attached."""


def _join_names(names: list[str]) -> str:
    return " or ".join(repr(name) for name in names)


def clean_nones(item: Any) -> Any:
    """Recursively remove ``None`` values from dict/list parquet cells."""
    if isinstance(item, dict):
        return {key: clean_nones(value) for key, value in item.items() if value is not None}
    if isinstance(item, list):
        return [clean_nones(value) for value in item]
    return item


def validate_tool_calls(messages: list) -> list:
    """Validate new canonical Lite tool calls without repairing old rows.

    Runtime/export paths operate on nested ``{id, type, function}`` calls.
    Legacy flat envelopes and missing/null arguments are migration inputs, not
    canonical data. Return ``messages`` unchanged on success.
    """
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{mi}] must be a dict")
        if "tool_calls" not in msg:
            continue
        tool_calls = msg["tool_calls"]
        if msg.get("role") != ASSISTANT_ROLE:
            if tool_calls in (None, []):
                continue
            raise ValueError(
                f"messages[{mi}].tool_calls must be empty on non-assistant messages"
            )
        if not isinstance(tool_calls, list):
            raise ValueError(f"messages[{mi}].tool_calls must be a list")
        for ti, call in enumerate(tool_calls):
            # The canonical-call SHAPE is core's fact, not a row policy: this
            # module states only that a published row raises on it, and what an
            # operator should do about the one failure that has a remedy.
            reason = validate_lite_tool_call(
                call,
                f"messages[{mi}].tool_calls[{ti}]",
                require_id=True,
            )
            if reason is None:
                continue
            if (
                isinstance(call, dict)
                and {"call_id", "name", "arguments"} & set(call)
            ):
                reason += _MIGRATION_HINT
            raise ValueError(reason)
    return messages


def _reject_dialect_only_tool_name(name: str, where: str) -> None:
    canonical_names = _NATIVE_FINISH_TOOL_ALIASES.get(name)
    if canonical_names is None:
        return
    raise ValueError(
        f"{where} {name!r} is dialect-only; "
        f"use canonical {_join_names(sorted(canonical_names))}"
    )


def _has_unknown_tool_error_only_result(
    tool_call: dict[str, Any],
    tool_result_text_by_call_id: dict[str, str],
) -> bool:
    call_id = tool_call_id(tool_call)
    if not isinstance(call_id, str) or not call_id:
        return False
    text = tool_result_text_by_call_id.get(call_id)
    expected = project_tool_result_text(None, f"unknown tool: {tool_call_name(tool_call)}")
    return text == expected


def _has_invalid_arguments_tool_result(
    tool_call: dict[str, Any],
    tool_result_text_by_call_id: dict[str, str],
) -> bool:
    call_id = tool_call_id(tool_call)
    if not isinstance(call_id, str) or not call_id:
        return False
    error = extract_projected_tool_result_error(tool_result_text_by_call_id.get(call_id))
    return isinstance(error, str) and error.startswith(
        f"invalid arguments for {tool_call_name(tool_call)}: "
    )


def _iter_action_batch_children(
    name: str,
    arguments: Any,
    where: str,
) -> list[tuple[str, dict[str, Any]]]:
    call_where = f"{where}.{name}" if where else name
    child_calls, error = validate_lite_action_batch_structure(name, arguments)
    if error is not None:
        raise ValueError(_action_batch_value_error_message(error, call_where))
    # Offline row validation stays STRICT where env ingress degrades: a stored
    # row naming an action its batch tool does not carry is malformed data, not
    # a live model mistake to hand back as feedback.
    name_errors = lite_action_batch_child_name_errors(name, child_calls)
    if name_errors:
        first = name_errors[min(name_errors)]
        raise ValueError(_action_batch_value_error_message(first, call_where))
    error = validate_lite_action_batch_child_arguments(name, child_calls)
    if error is not None:
        raise ValueError(_action_batch_value_error_message(error, call_where))
    return [(call["name"], call["arguments"]) for call in child_calls]


def _action_batch_value_error_message(
    error: LiteActionBatchValidationError,
    call_where: str,
) -> str:
    if error.kind == LiteActionBatchValidationKind.CHILD_ACTION_UNKNOWN:
        return (
            f"{call_where}.arguments.actions action "
            f"{error.child_action_name!r} is not valid for {error.action_batch_tool_name}: "
            f"{error.reason}"
        )
    action_batch_prefix = f"{error.action_batch_tool_name}."
    if error.reason.startswith(action_batch_prefix):
        return f"{call_where}.{error.reason.removeprefix(action_batch_prefix)}"
    return f"{call_where}: {error.reason}"


def validate_action_batches(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate canonical ``computer/mobile(arguments.actions)`` structure."""
    for mi, msg in enumerate(messages):
        for ti, call in enumerate(msg.get("tool_calls") or []):
            name = tool_call_name(call)
            if name not in LITE_ACTION_BATCH_TOOL_NAMES:
                continue
            where = f"messages[{mi}].tool_calls[{ti}]"
            _iter_action_batch_children(name, tool_call_arguments(call), where)
    return messages


def _reject_nested_extra_tool_action_batch_children(
    messages: list[dict[str, Any]],
    extra_schemas_by_name: dict[str, list[dict[str, Any]]],
) -> None:
    for mi, msg in enumerate(messages):
        for ti, call in enumerate(msg.get("tool_calls") or []):
            name = tool_call_name(call)
            if name not in LITE_ACTION_BATCH_TOOL_NAMES:
                continue
            where = f"messages[{mi}].tool_calls[{ti}]"
            arguments = tool_call_arguments(call)
            actions = arguments.get("actions") if isinstance(arguments, dict) else None
            if not isinstance(actions, list):
                continue
            for ai, action in enumerate(actions):
                if not isinstance(action, dict):
                    continue
                action_name = action.get("action")
                if not isinstance(action_name, str):
                    continue
                action_arguments = {k: v for k, v in action.items() if k != "action"}
                if any(
                    tool_name_and_arguments_match_schema_route_keys(
                        name=action_name,
                        arguments=action_arguments,
                        schema=schema,
                    )
                    for schema in extra_schemas_by_name.get(action_name, [])
                ):
                    raise ValueError(
                        f"{where}.{name}.arguments.actions[{ai}] must not nest "
                        f"standalone extra tool {action_name!r}"
                    )


def extra_tool_schemas_by_name(
    metadata: LiteBaseMetadata,
) -> dict[str, list[dict[str, Any]]]:
    """Index ``metadata.extra_tool_schemas`` by name, rejecting illegal names.

    Two names are illegal before any tool call is looked at: a dialect-only
    finish spelling (:data:`_NATIVE_FINISH_TOOL_ALIASES`), and a canonical
    top-level GUI tool, which is reserved for schema-free calls and may not be
    redeclared.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for si, schema in enumerate(metadata.extra_tool_schemas):
        name = tool_schema_name(schema)
        _reject_dialect_only_tool_name(
            name,
            f"metadata.extra_tool_schemas[{si}].name",
        )
        by_name.setdefault(name, []).append(schema)
    redeclared = set(by_name) & LITE_ACTION_SET_TOOL_NAMES
    if redeclared:
        raise ValueError(
            "metadata.extra_tool_schemas must not redeclare canonical "
            "top-level GUI tools reserved for schema-free Lite GUI calls: "
            f"{sorted(redeclared)}"
        )
    return by_name


def validate_tool_schema_calls(
    messages: list[dict[str, Any]],
    metadata: LiteBaseMetadata,
    *,
    schema_free_names: frozenset[str],
    extra_schemas_by_name: dict[str, list[dict[str, Any]]],
    model_visible_error_results: dict[str, str],
) -> None:
    """THE owner of "may this row call this tool, with these arguments".

    Every caller asks the same three questions -- is a canonical top-level GUI tool
    schema-free on this surface, is a standalone call declared, do its arguments
    fit the declaration -- so the two inputs a caller may legitimately differ on
    are parameters rather than a second copy of the walk:

    * ``schema_free_names`` is the caller's accepted schema-free surface. A
      raw/boundary checker may accept a wider one than the runtime declares
      without widening the runtime catalog.
    * ``model_visible_error_results`` is ``call_id -> role:tool text`` for the
      raw-only contract that keeps env-attributed model mistakes as ordinary
      rollout rows once the error reached the model. Pass ``{}`` where no raw
      env-error exemption applies.
    """
    _reject_nested_extra_tool_action_batch_children(messages, extra_schemas_by_name)
    for msg in messages:
        for tool_call in msg.get("tool_calls") or []:
            name = tool_call_name(tool_call)
            if name in LITE_ACTION_SET_TOOL_NAMES and name not in schema_free_names:
                raise ValueError(
                    f"tool_call {name!r} is not valid for "
                    f"metadata_kind={metadata.metadata_kind} dims={metadata.dims}"
                )
            if name in schema_free_names:
                continue
            if name not in extra_schemas_by_name:
                if _has_unknown_tool_error_only_result(
                    tool_call,
                    model_visible_error_results,
                ):
                    continue
                raise ValueError(
                    f"tool_call {name!r} is standalone but missing from "
                    "metadata.extra_tool_schemas"
                )
            if not any(
                tool_call_satisfies_schema(tool_call, schema)
                for schema in extra_schemas_by_name[name]
            ):
                if _has_invalid_arguments_tool_result(
                    tool_call,
                    model_visible_error_results,
                ):
                    continue
                raise ValueError(
                    f"tool_call {name!r} arguments do not match "
                    "metadata.extra_tool_schemas"
                )


def iter_canonical_actions(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return top-level tools and action-batch child actions as ``(name, arguments)`` pairs."""
    out: list[tuple[str, dict[str, Any]]] = []
    for call in message.get("tool_calls") or []:
        name = tool_call_name(call)
        arguments = tool_call_arguments(call)
        if name in LITE_ACTION_BATCH_TOOL_NAMES:
            out.extend(_iter_action_batch_children(name, arguments, ""))
            continue
        out.append((name, arguments))
    return out


def _first_oob_coordinate(messages: list[dict[str, Any]]) -> Any | None:
    for message in messages:
        for _name, args in iter_canonical_actions(message):
            if not action_coordinate_arguments_out_of_range(args):
                continue
            for key in COORDINATE_ARGUMENT_NAMES:
                coord = args.get(key)
                if norm_coord_out_of_range(coord):
                    return coord
            return True
    return None


_NO_NONCANONICAL_KEY = object()


def _first_empty_key_action(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        for name, args in iter_canonical_actions(message):
            if name not in LITE_DESKTOP_KEY_ACTION_NAMES:
                continue
            if args.get("keys") == []:
                return name
    return None


def _first_noncanonical_key(messages: list[dict[str, Any]]) -> object:
    for message in messages:
        for name, args in iter_canonical_actions(message):
            if name not in LITE_DESKTOP_KEY_ACTION_NAMES:
                continue
            for key in args.get("keys", []):
                if not is_canonical_key_token(key):
                    return key
    return _NO_NONCANONICAL_KEY


def _assistant_tool_call_id(call: Any, where: str) -> str:
    if not isinstance(call, dict):
        raise ValueError(f"{where}: assistant tool_call must be a dict")
    call_id = tool_call_id(call)
    if not isinstance(call_id, str) or not call_id:
        raise ValueError(
            f"{where}: assistant tool_call missing non-empty string id"
        )
    return call_id


def _finish_call_ids(messages: list[dict[str, Any]]) -> frozenset[str]:
    """Call ids of the finish calls the row actually ENDED on.

    A finish NAME is not enough. If the env answered the call with a
    ``role:"tool"`` result and the row then took another assistant decision, the
    episode did not end there: the env did not honour the call, so it is an
    ordinary standalone tool call and is judged by the ordinary rules.
    """
    answered_then_continued: set[str] = set()
    answered_since_last_assistant: set[str] = set()
    # ``_validate_rows`` runs ``validate_message_roles`` and ``validate_tool_calls``
    # before pairing, but this helper is also directly tested/called. Validate the
    # assistant ids here so direct callers cannot sneak malformed finish ids into
    # the finish set.
    for msg in messages:
        role = msg["role"]
        if role == TOOL_ROLE:
            call_id = msg.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                answered_since_last_assistant.add(call_id)
        elif role == ASSISTANT_ROLE:
            answered_then_continued |= answered_since_last_assistant
            answered_since_last_assistant.clear()
    out: set[str] = set()
    for mi, msg in enumerate(messages):
        if msg["role"] != ASSISTANT_ROLE:
            continue
        for ti, call in enumerate(msg.get("tool_calls") or []):
            call_id = _assistant_tool_call_id(
                call, f"messages[{mi}].tool_calls[{ti}]"
            )
            if (
                tool_call_name(call) in LiteFinishToolSet.get_tool_names()
                and call_id not in answered_then_continued
            ):
                out.add(call_id)
    return frozenset(out)


def validate_role_tool_pairing(
    messages: list[dict[str, Any]],
    *,
    is_rollout: bool,
    episode_ended: bool = False,
    row_contract: str = "canonical",
) -> None:
    """Reject invalid ``role:"tool"`` pairing without inventing EOF results.

    * ``is_rollout`` -- does this row have an observation loop at all? Only
      ``use`` rows do.
    * ``episode_ended`` -- does the row record live env feedback as OVER at EOF?
      (``metadata.others.terminated``/``metadata.others.truncated``.)
    * ``row_contract`` -- is this a local raw rollout artifact rather than a
      publishable canonical row?

    **1. A non-final rollout call needs a result iff it can produce an observation.**

        requires_result = is_rollout and call not in finish calls

    Acting on the screen produces the next screen when the trajectory continues.
    At EOF, an offline SFT row may legitimately end on an unobserved final
    action: the action is still the supervised label, and no synthetic
    ``role:"tool"`` result or ``Done.`` final should be invented for it.
    A non-rollout row (``grounding.*`` / ``understanding``) is a single-turn SFT
    label with no env behind it, so none of its calls require a result.

    **2. A finish call ends the row.** Nothing may follow it except its own
    ``role:"tool"`` result. Which calls are finish calls is
    :func:`_finish_call_ids`, not a name test.

    **3. "Not required" is not "anything goes".** A ``role:"tool"`` that answers
    no emitted call is an orphan in EVERY row. Two shapes answer an emitted call
    that did not require one: a result for a finish call (any row), and env
    feedback on a grounding call, which is admissible only in a RAW row whose
    metadata records the episode as ended, and only in the trailing EOF block
    written by the final assistant turn.

    **4. EOF may carry a final label, but not a dangling middle.** A pending
    call may survive only when it belongs to the final assistant turn. Any later
    assistant/user message hits the pending check above. A trailing
    ``role:"tool"`` result with no later assistant decision still needs
    ``episode_ended`` evidence because it is env feedback, not a supervised
    target.
    """
    finish_call_ids = _finish_call_ids(messages)
    terminal_unrequired_call_ids = (
        _terminal_paired_tool_result_call_ids_at_eof(messages)
        if row_contract == _RAW and episode_ended
        else frozenset()
    )
    pending: set[Any] = set()
    finished_at: int | None = None
    completed: set[Any] = set()
    emitted: set[Any] = set()
    trailing_tool_where: str | None = None
    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == TOOL_ROLE and "call_id" in msg:
            raise ValueError(
                f"messages[{mi}]: role:tool must use tool_call_id, not call_id"
            )
        if role == ASSISTANT_ROLE:
            if pending:
                raise ValueError(
                    f"messages[{mi}]: assistant message arrived before role:tool "
                    f"result(s) for tool_call_id(s) {sorted(pending)}"
                )
            if finished_at is not None:
                raise ValueError(
                    f"messages[{mi}]: assistant message after the finish call at "
                    f"messages[{finished_at}]; a finish call ends the row"
                )
            trailing_tool_where = None
            next_pending: set[Any] = set()
            seen_this_turn: set[Any] = set()
            for ti, call in enumerate(msg.get("tool_calls") or []):
                call_id = _assistant_tool_call_id(
                    call, f"messages[{mi}].tool_calls[{ti}]"
                )
                # Two different authoring faults, named apart: one turn emitting
                # the same id twice is a decode/parse bug, the same id coming
                # back a turn later is an id-allocation bug.
                if call_id in seen_this_turn:
                    raise ValueError(
                        f"messages[{mi}].tool_calls[{ti}]: duplicate assistant "
                        f"tool_call id {call_id!r}"
                    )
                if call_id in emitted:
                    raise ValueError(
                        f"messages[{mi}].tool_calls[{ti}]: reused assistant "
                        f"tool_call id {call_id!r}"
                    )
                seen_this_turn.add(call_id)
                # ``is_rollout`` gates the finish arm too: "a finish call ends the
                # row" is a statement about an EPISODE, and a single-turn label row
                # has none. Grounding adapters emit ``response`` as a deliberate
                # non-terminator, and ``response`` is a finish tool name.
                if is_rollout and call_id in finish_call_ids:
                    finished_at = mi
                elif is_rollout:
                    next_pending.add(call_id)
                emitted.add(call_id)
            pending = next_pending
            continue
        if role != TOOL_ROLE:
            if finished_at is not None:
                raise ValueError(
                    f"messages[{mi}]: {role!r} message after the finish call at "
                    f"messages[{finished_at}]; a finish call ends the row"
                )
            if pending:
                raise ValueError(
                    f"messages[{mi}]: {role!r} message arrived before role:tool "
                    f"result(s) for tool_call_id(s) {sorted(pending)}"
                )
            continue

        trailing_tool_where = f"messages[{mi}]"
        call_id = msg.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"messages[{mi}]: role:tool message missing tool_call_id")
        if call_id in completed:
            raise ValueError(
                f"messages[{mi}]: duplicate role:tool result for "
                f"tool_call_id {call_id!r}"
            )
        if call_id not in pending:
            # A call that required no result never entered `pending`, so both
            # admissible unrequired results land here: an env that DOES
            # screenshot after `terminate`, and terminal env feedback on a
            # grounding call. Everything else unpaired is orphan feedback.
            # No ``and call_id in emitted``: ``terminal_unrequired_call_ids`` is
            # built by intersecting the FINAL assistant turn's call ids with the
            # trailing results, and the main loop already added that turn's ids
            # to ``emitted``. The conjunct could never be false.
            if call_id in finish_call_ids or call_id in terminal_unrequired_call_ids:
                completed.add(call_id)
                continue
            raise ValueError(
                f"messages[{mi}]: orphan role:tool result for "
                f"tool_call_id {call_id!r}; "
                "no previous assistant tool_call"
            )
        pending.discard(call_id)
        completed.add(call_id)

    # EOF is allowed to be an unobserved final action label. Intermediate
    # unpaired calls have already been rejected when the next non-tool message
    # arrived while ``pending`` was non-empty.
    if trailing_tool_where is not None and not episode_ended:
        raise ValueError(
            f"{trailing_tool_where}: trailing role:tool result has no later assistant decision"
        )


def _terminal_paired_tool_result_call_ids_at_eof(
    messages: list[dict[str, Any]],
) -> frozenset[str]:
    """Return final-assistant call IDs that have trailing EOF tool feedback."""
    if (
        not messages
        or not isinstance(messages[-1], dict)
        or messages[-1].get("role") != TOOL_ROLE
    ):
        return frozenset()
    first_trailing_tool = len(messages) - 1
    while (
        first_trailing_tool > 0
        and isinstance(messages[first_trailing_tool - 1], dict)
        and messages[first_trailing_tool - 1].get("role") == TOOL_ROLE
    ):
        first_trailing_tool -= 1
    if first_trailing_tool == 0:
        return frozenset()
    assistant = messages[first_trailing_tool - 1]
    if not isinstance(assistant, dict) or assistant.get("role") != ASSISTANT_ROLE:
        return frozenset()

    assistant_call_ids = {
        _assistant_tool_call_id(call, "final assistant tool_call")
        for call in assistant.get("tool_calls") or []
    }
    trailing_result_call_ids = {
        msg["tool_call_id"]
        for msg in messages[first_trailing_tool:]
        if isinstance(msg, dict)
        and isinstance(msg.get("tool_call_id"), str)
        and msg.get("tool_call_id")
    }
    return frozenset(assistant_call_ids & trailing_result_call_ids)


def _tool_result_single_text_by_call_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != TOOL_ROLE:
            continue
        call_id = msg.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        text = _single_text_part_content(msg.get("content"))
        if text is not None:
            out[call_id] = text
    return out


def _single_text_part_content(content: Any) -> str | None:
    if not isinstance(content, list):
        return None
    text_parts = [
        item["text"]
        for item in content
        if (
            isinstance(item, dict)
            and item.get("type") == TEXT_PART
            and isinstance(item.get("text"), str)
        )
    ]
    if len(text_parts) != 1:
        return None
    return text_parts[0]


def _reject_understanding_tool_calls(
    messages: list[dict[str, Any]],
    metadata: LiteBaseMetadata,
) -> None:
    if (
        not isinstance(metadata, LiteCUAMetadata)
        or metadata.task_type != LiteCUAMetadata.TaskType.UNDERSTANDING
    ):
        return
    for mi, msg in enumerate(messages):
        if (
            isinstance(msg, dict)
            and msg.get("role") == ASSISTANT_ROLE
            and msg.get("tool_calls")
        ):
            raise ValueError(
                "understanding rows are plain QA/caption and must not include "
                f"assistant tool_calls (messages[{mi}])"
            )


def _reject_dialect_tool_call_names(messages: list[dict[str, Any]]) -> None:
    """Reject every family's dialect finish spelling in ``tool_calls``.

    Runs BEFORE pairing so the row reports the real fault -- the spelling -- and
    not the pairing consequence of a name the pairing vocabulary does not know.
    Row-contract-independent: see :data:`_NATIVE_FINISH_TOOL_ALIASES`.
    """
    for msg in messages:
        for tool_call in msg.get("tool_calls") or []:
            _reject_dialect_only_tool_name(tool_call_name(tool_call), "tool_call")


def validate_canonical_rows(
    rows: list[dict],
    label: str,
) -> None:
    """Reject noncanonical Lite rows before staging, HF publication, or export.

    Canonical rows have no private rollout sidecars, no unknown-tool feedback
    exemptions, and no local diagnostic metadata such as
    ``metadata.others.content_only_final``. Call/result pairing is
    :func:`validate_role_tool_pairing` with ``row_contract="canonical"``:
    non-final rollout calls must have matching ``role:"tool"`` results, while
    the final assistant turn may be an unobserved SFT target at EOF. Do not
    fabricate an empty result or a ``Done.`` final to close that label.
    """
    _validate_rows(rows, label, row_contract="canonical")


def validate_raw_rollout_rows(
    rows: list[dict],
    label: str,
) -> None:
    """Validate local raw rollout/log rows without treating them as canonical.

    Raw rows may carry adapter replay sidecars, local final-turn diagnostics,
    unknown-tool error-only feedback, out-of-range coordinates, and empty live
    content-only finals, since a raw row is a same-family local export/debugging
    artifact. Tool-call names must still be canonical: raw rows preserve private
    diagnostics, not producer-family dialect spellings. Pairing adds one
    relaxation of its own, gated on the row's own record that the episode ENDED
    (``metadata.others.terminated`` / ``metadata.others.truncated``): the final
    turn may carry env feedback on a call that required no result
    (terminal/error feedback on a grounding call). Final unpaired tool calls
    are not raw-only: they are valid offline SFT targets for both row contracts.
    """
    _validate_rows(rows, label, row_contract=_RAW)


def _validate_rows(
    rows: list[dict],
    label: str,
    *,
    row_contract: str,
) -> None:
    """Walk ``rows`` under ``row_contract``, the ONE thing the two gates differ on.

    ``row_contract`` is the whole difference between
    :func:`validate_canonical_rows` and :func:`validate_raw_rollout_rows`: every
    relaxation below is that same question asked again. The raw contract allows
    private ``raw_response`` sidecars, out-of-range coordinates, terminal EOF
    env feedback, unknown-tool error-only results, ``others.content_only_final``,
    and empty live content-only finals. They are derived here rather than passed
    as six booleans the two callers could only ever set in lockstep. Keeping the
    CONTRACT as the parameter — not a pre-reduced ``bool`` — is deliberate:
    rules further down that need the raw-vs-canonical distinction should be
    handed ``row_contract`` itself instead of growing one more flag per rule.
    """
    is_raw = row_contract == _RAW
    for i, row in enumerate(rows):
        try:
            if "image" in row:
                raise ValueError("row.image is retired; use images")
            if "images" not in row:
                raise ValueError("images is required")
            images = row["images"]
            if not isinstance(images, list):
                raise ValueError("images must be a list")

            msgs = row["messages"]
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            # THE dataset-row boundary: roles and content parts arrive as data
            # here, so they are checked against the closed vocabularies BEFORE
            # any walk over the list gets to meet a value it cannot name.
            validate_message_roles(msgs)
            validate_message_content_parts(msgs)
            for mi, msg in enumerate(msgs):
                if is_raw or not isinstance(msg, dict):
                    continue
                if "raw_response" in msg:
                    raise ValueError(
                        f"messages[{mi}].raw_response is a private rollout sidecar "
                        "and must not be published"
                    )
                for key in (
                    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY,
                    MODEL_OUTPUT_ERROR_KEY,
                ):
                    if key in msg:
                        raise ValueError(
                            f"messages[{mi}].{key} is a private final-message "
                            "sidecar and must not be published"
                        )
            validate_tool_calls(msgs)
            _reject_dialect_tool_call_names(msgs)

            if "metadata" not in row:
                raise ValueError("metadata is required")
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            if not isinstance(meta, dict):
                raise ValueError("metadata must be a dict")
            if "split" in meta:
                raise ValueError(
                    "metadata.split must not be present; split lives in the partition path"
                )
            if not is_raw:
                unknown_metadata_keys = set(meta) - _CANONICAL_METADATA_KEYS
                if unknown_metadata_keys:
                    raise ValueError(
                        "metadata has unknown top-level keys "
                        f"{sorted(unknown_metadata_keys)}; put rollout identity "
                        "and outcome facts under metadata.others"
                    )

            # THE metadata boundary of this validator: everything above reads
            # the raw wire dict and must check; everything below holds tagged
            # Lite metadata from ``metadata_from_dict`` -- ``others`` is a dict,
            # ``extra_tool_schemas`` is a list, and every entry in it has a
            # typed canonical function declaration. One standard for all reads.
            lite_meta = metadata_from_dict(meta)
            _reject_understanding_tool_calls(msgs, lite_meta)
            if not is_raw and "content_only_final" in lite_meta.others:
                raise ValueError(
                    "metadata.others.content_only_final is a private local "
                    "diagnostic and must not be published"
                )
            extra_schemas_by_name = extra_tool_schemas_by_name(lite_meta)
            _reject_nested_extra_tool_action_batch_children(msgs, extra_schemas_by_name)
            validate_action_batches(msgs)
            if not is_raw and (example := _first_oob_coordinate(msgs)) is not None:
                raise ValueError(
                    f"out-of-range coordinates (e.g. {example}); "
                    "coords must be normalized to [0, 1000]"
                )
            empty_key_action = _first_empty_key_action(msgs) if not is_raw else None
            if empty_key_action is not None:
                raise ValueError(f"{empty_key_action}.keys must not be empty")
            key = _first_noncanonical_key(msgs) if not is_raw else _NO_NONCANONICAL_KEY
            if key is not _NO_NONCANONICAL_KEY:
                raise ValueError(
                    f"noncanonical or unsupported key {key!r}; use one literal "
                    "character or a canonical special key"
                )
            validate_image_references(msgs, images)

            tool_result_text_by_call_id = _tool_result_single_text_by_call_id(msgs)
            validate_role_tool_pairing(
                msgs,
                is_rollout=not (
                    isinstance(lite_meta, LiteCUAMetadata)
                    and lite_meta.task_type in SINGLE_TURN_TASK_TYPES
                ),
                episode_ended=(
                    lite_meta.others.get("terminated") is True
                    or lite_meta.others.get("truncated") is True
                ),
                row_contract=row_contract,
            )
            message_utils.validate_content_only_finals(
                msgs,
                lite_meta,
                require_visible_text=not is_raw,
            )
            validate_tool_schema_calls(
                msgs,
                lite_meta,
                schema_free_names=(
                    lite_builtin_tool_names_for_metadata(lite_meta)
                    if isinstance(lite_meta, LiteCUAMetadata)
                    else frozenset()
                ),
                extra_schemas_by_name=extra_schemas_by_name,
                model_visible_error_results=(
                    tool_result_text_by_call_id if is_raw else {}
                ),
            )

        except (KeyError, TypeError, ValueError, LiteContractError) as exc:
            raise _RowValidationError(
                f"{label}: row {i} is not {row_contract}: {exc}"
            ) from exc
