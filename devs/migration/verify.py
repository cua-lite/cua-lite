"""Invariant checks for one-time LiteSample migration outputs.

This module is intentionally local to ``devs/migration``. It accepts parsed
rows or parquet-style rows whose ``messages`` / ``metadata`` fields are JSON
strings, but it verifies only post-migration nested Lite tool calls with
optional action-batch payloads. The action vocabulary it checks against is
imported from ``lite.core.tools.action_space`` rather than re-declared, so
verification can never drift from the canonical surface.

Verification is not a repair path and does not reindex images. It checks that
retained image indices point into the preserved ``images`` list, and that each
``role:"tool"`` result is owned by exactly one assistant call via
``tool_call_id``.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from lite.core.errors import LiteContractError
from lite.core.messages.image_refs import validate_image_references
from lite.core.metadata import LiteCUAMetadata, metadata_from_dict
from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    LITE_DESKTOP_KEY_ACTION_NAMES,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    lite_action_batch_tool_name_for_platform,
    lite_action_names_by_action_batch_tool,
    lite_builtin_tool_names_for_metadata,
)
from lite.core.tools.action_space.keys import is_canonical_key_token
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.extra_tools import (
    BASH_TOOL_NAME,
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
    make_open_app_tool,
)
from lite.core.tools.schemas import tool_schema_name
from lite.data.utils.messages import validate_content_only_finals
from lite.data.utils.rows import (
    extra_tool_schemas_by_name,
    validate_action_batches,
    validate_tool_calls,
    validate_tool_schema_calls,
)

FINISH_TOOL_NAMES = LiteFinishToolSet.get_tool_names()
# Migration verifies migrated outputs from env surfaces that had resultless
# terminal-like extras outside the canonical finish catalog. Keep those
# compatibility names explicit and local to this raw-boundary checker.
MIGRATION_RESULTLESS_STANDALONE_TOOLS = frozenset(
    {
        "done",
        "report_infeasible",
        "send_msg_to_user",
    }
)
TERMINAL_STANDALONE_TOOLS = FINISH_TOOL_NAMES | MIGRATION_RESULTLESS_STANDALONE_TOOLS
BROWSER_NAV_TOOLS = LiteBrowserNavToolSet.get_tool_names()
APP_LAUNCH_TOOL_NAME = tool_schema_name(make_open_app_tool())
APP_LAUNCH_TOOLS = frozenset({APP_LAUNCH_TOOL_NAME})
NAV_TOOLS = BROWSER_NAV_TOOLS | APP_LAUNCH_TOOLS
TEXT_RESULT_ONLY_TOOLS = frozenset({BASH_TOOL_NAME, "ask_user"})
STANDALONE_EXTRA_TOOLS = TERMINAL_STANDALONE_TOOLS | NAV_TOOLS | TEXT_RESULT_ONLY_TOOLS
TOP_LEVEL_DURABLE_METADATA_KEYS = frozenset(
    {
        "env_id",
        "task_id",
        "episode_return",
        "terminated",
        "truncated",
    }
)
#: Model-family spellings of a canonical finish tool that a MIGRATED row may not
#: carry -- migration's job is to normalize them away, so their survival is a
#: migration bug, not a data variant.
#:
#: This is the same vocabulary ``lite/data/utils/rows.py``'s
#: ``_NATIVE_FINISH_TOOL_ALIASES`` declares, and it is re-declared here rather
#: than imported ON PURPOSE: ``tests/static/test_data_utils_rows_owner.py::
#: test_migration_does_not_import_private_data_validator_hooks`` fails any
#: ``devs/migration/*`` import of a private name from ``lite.data.utils.rows``,
#: and that module's own comment refuses to publish an accessor. So the copy is
#: forced -- but it is TIED to its owners by ``tests/data/utils/test_rows_dialect_census.py::
#: test_migration_verifier_dialect_census_matches_every_model_family``, which
#: recomputes the union from the family action spaces and reddens if this table
#: is short. It used to read ``{"answer": "response"}`` -- ONE family's ONE row,
#: exactly the 1-of-6 defect the core copy was retired for, with ``INFO``,
#: ``ABORT``, ``COMPLETE``, ``call_user`` and ``finished`` unknown to it.
#:
#: The value is a frozenset because one native spelling can mean either finish
#: tool depending on its arguments (ui_tars ``finished(content=…)`` is a
#: ``response``, bare ``finished()`` a ``terminate``).
_DIALECT_ONLY_TOOL_ALIASES: dict[str, frozenset[str]] = {
    "ABORT": frozenset({"terminate"}),
    "COMPLETE": frozenset({"terminate"}),
    "INFO": frozenset({"response"}),
    "answer": frozenset({"response"}),
    "call_user": frozenset({"response", "terminate"}),
    "finished": frozenset({"response", "terminate"}),
}
#: Migrated rows carry *Lite* action names only: provider-native spellings
#: (``double_click``/``hotkey``/``move``/``double_tap``) are normalized away by
#: ``BaseActionSpace.convert_tool_calls_from_agent`` before storage, so there
#: are no legacy aliases to accept and these are exactly the upstream catalogs.
DESKTOP_GUI_ACTIONS = LiteDesktopActionSet.get_action_names()
MOBILE_GUI_ACTIONS = LiteMobileActionSet.get_action_names()
ACTION_NAMES_BY_ACTION_BATCH_TOOL = lite_action_names_by_action_batch_tool()
BROWSER_PLATFORM = LiteCUAMetadata.Platform.BROWSER.value
LEGACY_WEB_PLATFORM = "web"
DESKTOP_PLATFORM = LiteCUAMetadata.Platform.DESKTOP.value
MOBILE_PLATFORM = LiteCUAMetadata.Platform.MOBILE.value


class VerificationError(ValueError):
    """Raised when a migrated row violates the canonical migration contract."""


def _verify_no_top_level_durable_metadata(metadata: dict[str, Any]) -> None:
    leaked = sorted(TOP_LEVEL_DURABLE_METADATA_KEYS & set(metadata))
    if leaked:
        raise VerificationError(
            "durable rollout metadata belongs in metadata.others, not top level: "
            + ", ".join(leaked)
        )


def _metadata_platform(metadata: dict[str, Any]) -> str | None:
    dims = metadata.get("dims")
    if isinstance(dims, (list, tuple)) and dims:
        return str(dims[0])
    platform = metadata.get("platform")
    if platform is None:
        return None
    return str(platform)


def _canonical_platform(platform: str | None) -> str:
    value = str(platform or DESKTOP_PLATFORM)
    if value == LEGACY_WEB_PLATFORM:
        return BROWSER_PLATFORM
    return value


def _action_batch_tool_name_for_platform(platform: str) -> str | None:
    return lite_action_batch_tool_name_for_platform(_canonical_platform(platform))


def _metadata_from_migration_output(metadata: dict[str, Any]) -> LiteCUAMetadata:
    platform = _metadata_platform(metadata)
    if platform == LEGACY_WEB_PLATFORM:
        raise VerificationError("migration output metadata.dims[0] must be 'browser', not 'web'")

    try:
        lite_meta = metadata_from_dict(metadata)
    except LiteContractError as exc:
        raise VerificationError(str(exc)) from exc

    if not isinstance(lite_meta, LiteCUAMetadata):
        raise VerificationError("migration output metadata must be tagged CUA metadata")
    return lite_meta


def verify_lite_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Verify one migrated LiteSample-shaped row and return a parsed copy.

    The returned row has ``messages`` and ``metadata`` parsed into Python
    objects. The input is not mutated.
    """
    row = normalize_lite_sample(sample)
    messages = _as_list(row.get("messages"), field="messages")
    metadata = _as_dict(row.get("metadata"), field="metadata")
    if "images" not in row:
        raise VerificationError("images is required")
    images = _coerce_images(row["images"])

    _verify_valid_actions(metadata)
    _verify_no_top_level_durable_metadata(metadata)
    lite_meta = _metadata_from_migration_output(metadata)
    platform = _canonical_platform(_metadata_platform(metadata) or lite_meta.platform.value)
    task_type = lite_meta.task_type.value

    try:
        validate_image_references(messages, images)
    except LiteContractError as exc:
        raise VerificationError(str(exc)) from exc

    try:
        validate_tool_calls(messages)
    except ValueError as exc:
        raise VerificationError(str(exc)) from exc

    call_names: dict[str, str] = {}
    # Result ownership pairs assistant ``tool_calls[].id`` with
    # ``role:"tool".tool_call_id``; no assistant call may receive two results.
    result_messages: dict[str, dict[str, Any]] = {}
    #: Calls in the last assistant turn that HAS tool calls are allowed to remain
    #: unpaired: the episode ran out of step budget, so the last action's
    #: observation was never published and migration has none to move into
    #: ``role:"tool"``. This matches the current final-EOF relaxation in
    #: ``validate_canonical_rows``; publish policy may still filter these rows
    #: because the final observation is absent.
    #:
    #: LOAD-BEARING, measured 2026-08-08 over every locally cached row of the
    #: allow-listed datasets: disabling it refuses 242/636 migrated rows in 39/46
    #: partitions (Lite.CUAWorld 235/409, Lite.CUAGym 7/21, Lite.OSWorld 0/133,
    #: Lite.ScaleCUA 0/73). All 242 are ``truncated=True`` step-budget
    #: truncations ending on a bare ``computer`` call; 0 folded a ``terminate``
    #: onto the last action turn, which an earlier revision of this comment gave
    #: as the reason. Unlike the raw relaxation it copies, this exemption is not
    #: gated on the row's own ``terminated``/``truncated`` evidence -- all 242
    #: carry it, so gating it would be free here, but that is a data-owner call.
    #: The Lite.ScaleCUA figure above is a locally cached sample: over the full
    #: published dataset, 17,953/17,953 rows migrate and ``--verify`` passes, but
    #: 460/17,953 rows are excluded by the incomplete-row publish filter; the
    #: filtered staged output is 17,493 rows.
    #: See the terminal-turn note in ``devs/migration/AGENTS.md``.
    final_turn_call_ids: set[str] = set()

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            turn_call_ids: set[str] = set()
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name = tool_call_name(call)
                if _is_top_level_use_action(
                    name,
                    platform=platform,
                    task_type=task_type,
                ):
                    action_batch_tool_name = _action_batch_tool_name_for_platform(platform)
                    raise VerificationError(
                        f"top-level GUI action {name!r} is not canonical "
                        f"for platform={platform!r} task_type='use'; use "
                        f"{action_batch_tool_name!r}.arguments.actions[]"
                    )
                call_id = tool_call_id(call)
                if not isinstance(call_id, str) or not call_id:
                    continue
                if call_id in call_names:
                    raise VerificationError(f"duplicate tool call id {call_id!r}")
                call_names[call_id] = name
                turn_call_ids.add(call_id)
            if msg.get("tool_calls"):
                final_turn_call_ids = turn_call_ids
        elif role == "tool":
            if "call_id" in msg:
                raise VerificationError("tool result must use tool_call_id, not call_id")
            call_id = msg.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise VerificationError("tool result is missing tool_call_id")
            if call_id in result_messages:
                raise VerificationError(f"duplicate tool result tool_call_id {call_id!r}")
            result_messages[call_id] = msg

    for call_id, msg in result_messages.items():
        if call_id not in call_names:
            raise VerificationError(f"tool result references unknown tool_call_id {call_id!r}")
        name = call_names[call_id]
        if name in TERMINAL_STANDALONE_TOOLS:
            raise VerificationError(f"terminal call {name!r} must not have a tool result")
        if name in TEXT_RESULT_ONLY_TOOLS and _content_has_image(msg.get("content") or []):
            raise VerificationError(
                f"text-result-only tool {name!r} must not receive an image/screenshot result"
            )

    # Only ``use`` rows are rollout turns. ``grounding.*`` / ``understanding``
    # rows are SFT *labels* -- an aguvis grounding row packs N independent
    # single-step samples as [user, assistant] x N and has no observations by
    # construction -- so the pairing rule must not be applied to them.
    if task_type == "use":
        for call_id, name in call_names.items():
            if call_id in final_turn_call_ids:
                continue
            if (
                _is_screen_producing_call(name, platform=platform)
                and call_id not in result_messages
            ):
                raise VerificationError(
                    f"screen-producing call {name!r} with id {call_id!r} is missing a tool result"
                )

    _validate_shared_publish_schema_contract(messages, metadata, images)

    row["messages"] = messages
    row["metadata"] = metadata
    return row


def normalize_lite_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with JSON-string ``messages`` / ``metadata`` parsed."""
    row = copy.deepcopy(sample)
    if isinstance(row.get("messages"), str):
        row["messages"] = json.loads(row["messages"])
    if isinstance(row.get("metadata"), str):
        row["metadata"] = json.loads(row["metadata"])
    return row


def _verify_valid_actions(metadata: dict[str, Any]) -> None:
    valid_actions = metadata.get("valid_actions")
    if valid_actions is not None:
        if not isinstance(valid_actions, list):
            raise VerificationError("metadata.valid_actions must be a list when present")
        leaked = [name for name in valid_actions if name in STANDALONE_EXTRA_TOOLS]
        if leaked:
            raise VerificationError(
                "metadata.valid_actions must contain GUI actions only; found "
                + ", ".join(map(str, leaked))
            )


def _validate_shared_publish_schema_contract(
    messages: list[dict[str, Any]],
    metadata: dict[str, Any],
    _images: list[Any],
) -> None:
    """Validate post-migration tool/schema shape at the migration boundary.

    The WALK is the production one -- ``validate_tool_schema_calls`` owns it, so
    migration cannot drift from the gate it is preparing rows for. What is local
    here is only the two inputs migration is entitled to differ on: the wider
    schema-free catalog (:func:`_migration_builtin_tool_names_for_metadata`) and the fact
    that a raw boundary row gets no unknown-tool exemption. Production row
    validators additionally reject partial trajectories, which migration does
    not check here.
    """
    try:
        if "split" in metadata:
            raise ValueError(
                "metadata.split must not be present; split lives in the partition path"
            )
        _verify_no_top_level_durable_metadata(metadata)
        lite_meta = _metadata_from_migration_output(metadata)
        validate_tool_calls(messages)
        validate_action_batches(messages)
        _verify_canonical_key_tokens(messages)
        validate_content_only_finals(messages, lite_meta)
        _reject_dialect_only_tool_call_names(messages)
        validate_tool_schema_calls(
            messages,
            lite_meta,
            schema_free_names=_migration_builtin_tool_names_for_metadata(lite_meta),
            extra_schemas_by_name=extra_tool_schemas_by_name(lite_meta),
            model_visible_error_results={},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(str(exc)) from exc


def _reject_dialect_only_tool_call_names(messages: list[dict[str, Any]]) -> None:
    """The ``tool_calls`` side of the dialect check, against the LOCAL census.

    The schema-name side is the owner's (inside ``extra_tool_schemas_by_name``),
    which is why this pass exists separately: :data:`_DIALECT_ONLY_TOOL_ALIASES`
    is deliberately migration's own copy.
    """
    for msg in messages:
        for tool_call in msg.get("tool_calls") or []:
            _reject_dialect_only_tool_name(tool_call_name(tool_call), "tool_call")


def _verify_canonical_key_tokens(messages: list[dict[str, Any]]) -> None:
    for mi, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        for ti, call in enumerate(msg.get("tool_calls") or []):
            name = tool_call_name(call)
            args = tool_call_arguments(call)
            where = f"messages[{mi}].tool_calls[{ti}]"
            if name in LITE_ACTION_BATCH_TOOL_NAMES:
                actions = args.get("actions")
                if not isinstance(actions, list):
                    continue
                for ai, action in enumerate(actions):
                    if not isinstance(action, dict):
                        continue
                    action_name = action.get("action")
                    if action_name in LITE_DESKTOP_KEY_ACTION_NAMES:
                        _verify_key_argument_tokens(
                            action.get("keys"),
                            f"{where}.arguments.actions[{ai}].keys",
                        )
                continue
            if name in LITE_DESKTOP_KEY_ACTION_NAMES:
                _verify_key_argument_tokens(args.get("keys"), f"{where}.function.arguments.keys")


def _verify_key_argument_tokens(keys: Any, where: str) -> None:
    if not isinstance(keys, list):
        raise ValueError(f"{where} must be a non-empty list[str]")
    if not keys:
        raise ValueError(f"{where} must not be empty")
    for index, key in enumerate(keys):
        if not isinstance(key, str):
            raise ValueError(f"{where}[{index}] must be str, got {type(key).__name__}")
        if not is_canonical_key_token(key):
            raise ValueError(f"{where}[{index}] has noncanonical key token {key!r}")


def _reject_dialect_only_tool_name(name: str, where: str) -> None:
    canonical = _DIALECT_ONLY_TOOL_ALIASES.get(name)
    if canonical is None:
        return
    raise ValueError(
        f"{where} {name!r} is dialect-only; use canonical {_join_canonical(canonical)}"
    )


def _join_canonical(names: frozenset[str]) -> str:
    return " or ".join(repr(name) for name in sorted(names))


def _migration_builtin_tool_names_for_metadata(metadata: LiteCUAMetadata) -> frozenset[str]:
    """Schema-free names accepted by migration's raw-boundary verifier.

    Runtime/current rows accept only the wrapper-native ``grounding.action``
    surface from ``lite_builtin_tool_names_for_metadata``. Published legacy label rows may still
    carry bare top-level GUI actions, so migration keeps that compatibility
    local instead of widening the runtime catalog.
    """
    names = set(lite_builtin_tool_names_for_metadata(metadata))
    if metadata.task_type == LiteCUAMetadata.TaskType.GROUNDING_ACTION:
        if metadata.platform == LiteCUAMetadata.Platform.MOBILE:
            names.update(MOBILE_GUI_ACTIONS)
        else:
            names.update(DESKTOP_GUI_ACTIONS)
    return frozenset(names)


def schema_free_names(metadata: dict[str, Any]) -> frozenset[str]:
    """Tool names valid without schemas for migration input/output checks."""
    lite_meta = _metadata_from_migration_output(metadata)
    return _migration_builtin_tool_names_for_metadata(lite_meta)


def _is_top_level_use_action(name: str, *, platform: str, task_type: str) -> bool:
    return (
        task_type == "use"
        and _action_batch_tool_name_for_platform(platform) is not None
        and _is_gui_action(name, platform=platform)
    )


def _is_screen_producing_call(name: str, *, platform: str) -> bool:
    return (
        name in LITE_ACTION_BATCH_TOOL_NAMES
        or name in NAV_TOOLS
        or _is_gui_action(name, platform=platform)
    )


def _is_gui_action(name: str, *, platform: str) -> bool:
    if name in STANDALONE_EXTRA_TOOLS:
        return False
    if platform == "mobile":
        return name in MOBILE_GUI_ACTIONS
    return name in DESKTOP_GUI_ACTIONS


def _content_has_image(content: list[Any]) -> bool:
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image":
            return True
    return False


def _as_dict(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError(f"{field} must be an object")
    return value


def _as_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise VerificationError(f"{field} must be a list")
    return value


def _coerce_images(value: Any) -> list[Any]:
    """Parse an image list for validation only; never reorder or reindex it."""
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise VerificationError("images must be a list")
    return value


__all__ = [
    "VerificationError",
    "normalize_lite_sample",
    "verify_lite_sample",
]
