"""Provider-independent tool-result policies used by agent loops.

Every helper here is production-shared by at least two loop or provider-history
owners:

- ``align_tool_results_to_tool_calls`` is the single env-result alignment
  boundary. ``AdapterBasedAgent.sample()`` and ``record_lite_env_result()``
  call it once per step and write the aligned order back onto
  ``step_result.results``; nothing downstream re-aligns.
- ``latest_step_feedback`` and ``tool_result_text_for_call_id`` project
  observation and per-call text for provider-native history appenders.
- ``require_model_visible_unpaired_response_feedback`` owns the shared check for
  no-tool-call response attempts that continue to another model turn.
- ``append_tool_result_images`` is durable trajectory image storage for
  provider-native loops, and owns both views of that append record
  (``StoredResultImages``) so provider history never rebuilds the call-id map.
- ``canonical_tool_result_messages`` assembles canonical ``role:"tool"``
  messages from already-aligned results for those same provider-native loops.
- ``canonical_unpaired_response_feedback_messages`` assembles canonical
  ``role:"user"`` feedback for no-tool-call response attempts.

The adapter loop keeps its image visibility decisions inline, and provider
history modules still decide which stored images they send back to the
provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from lite.agents.core.agent.utils.messages import build_tool_result_message
from lite.core.tools.action_space import LITE_ACTION_BATCH_TOOL_NAMES
from lite.core.tools.calls import tool_call_id, tool_call_name
from lite.core.tools.results import (
    LiteToolResult,
    project_tool_result_text,
)
from lite.gym.types import LiteEnvStepResult
from lite.utils.image import decode_image


def align_tool_results_to_tool_calls(
    actions: list[dict[str, Any]],
    results: list[LiteToolResult],
    *,
    require_all: bool = True,
) -> list[LiteToolResult]:
    """Order env results by the previous assistant tool calls.

    Non-terminal env steps must return one paired result per model-emitted tool
    call. Terminal env steps may omit results; in that case the trajectory
    naturally ends on the assistant tool_call.

    A step may also carry result frames that no assistant call addresses. Envs
    build "one per EXECUTED action" (``build_tool_results_from_decisions``), and
    a wrapper may expand one assistant call into several env actions — e.g.
    ``LoopDetectWrapper`` forwards the truncated prefix of a ``computer`` batch
    before its internal ``terminate``, and only the terminate answers the
    assistant's call id. Those extra frames are unpaired by construction, so
    they are skipped here rather than treated as a malformed env response.
    Skipping them does not weaken the contract where the contract applies:
    ``missing`` below still fails loudly for a call the assistant *did* make,
    whenever ``require_all`` is set — which both callers do except on a terminal
    step, where a step carrying no paired result at all was already legal.

    This handles the expansion's *result* frames, not the case where the wrapper
    stops forwarding: ``LoopDetectWrapper`` breaks out of the action loop once a
    loop trips, so a later call in the same assistant turn is never executed and
    gets no frame. That still raises here, as ``missing``.
    """
    expected = [action_id for action in actions if (action_id := tool_call_id(action))]
    if not expected and all(not result.tool_call_id for result in results):
        return []

    expected_set = set(expected)

    by_call_id: dict[str, LiteToolResult] = {}
    duplicate: set[str] = set()
    for result in results:
        call_id = result.tool_call_id
        if not call_id:
            # Unaddressed frame from a wrapper-expanded action (see docstring).
            continue
        if call_id in by_call_id:
            duplicate.add(call_id)
        by_call_id[call_id] = result

    result_set = set(by_call_id)
    missing = [call_id for call_id in expected if call_id not in result_set]
    orphan = sorted(result_set - expected_set)
    if duplicate or (require_all and missing) or orphan:
        raise RuntimeError(
            "Env tool results do not match tool_calls "
            f"(missing={missing}, duplicate={sorted(duplicate)}, orphan={orphan})"
        )

    return [by_call_id[call_id] for call_id in expected if call_id in by_call_id]


def latest_step_feedback(
    step_result: LiteEnvStepResult,
) -> tuple[bytes | None, str | None, dict[str, Any] | None]:
    """Latest unpaired-style observation channels across a step's env results."""
    image: bytes | None = None
    text: str | None = None
    metadata: dict[str, Any] | None = None
    for result in step_result.results:
        if result.images:
            image = result.images[-1]
        projected_text = project_tool_result_text(result.text, result.error)
        if projected_text is not None:
            text = projected_text
        if result.metadata:
            metadata = dict(result.metadata)
    return image, text, metadata


def has_model_visible_feedback(result: LiteToolResult) -> bool:
    """Return whether an unpaired env result can be shown to the next model turn."""
    return bool(result.images) or bool(project_tool_result_text(result.text, result.error))


def require_unpaired_response_feedback(results: Sequence[LiteToolResult]) -> None:
    """Reject call-paired results from a no-tool response attempt."""
    paired_ids = sorted(
        str(result.tool_call_id)
        for result in results
        if result.tool_call_id is not None
    )
    if paired_ids:
        raise RuntimeError(
            "Env returned paired tool results after a no-tool response attempt "
            f"(tool_call_id={paired_ids}); response feedback must be unpaired"
        )


def require_model_visible_unpaired_response_feedback(
    results: Sequence[LiteToolResult],
) -> None:
    """Validate feedback that must become the next no-tool response turn.

    A deliberate no-tool response attempt has no assistant call id to pair
    against. If the env wants another model turn, it must therefore return
    unpaired feedback that the next model can actually see.
    """
    if not results:
        raise RuntimeError("Env returned no feedback after non-terminal response attempt")
    require_unpaired_response_feedback(results)
    if not any(has_model_visible_feedback(result) for result in results):
        raise RuntimeError(
            "Env returned no model-visible feedback after non-terminal response attempt"
        )


def tool_result_text_for_call_id(
    step_result: LiteEnvStepResult,
    *,
    call_id: str | None,
    default: str | None = None,
) -> str | None:
    """Return canonical per-call tool result text by call_id."""
    if not call_id:
        return default
    for result in step_result.results:
        if result.tool_call_id == call_id:
            if result.text is not None or result.error is not None:
                return project_tool_result_text(result.text, result.error)
            return default

    return default


class StoredResultImages(NamedTuple):
    """Durable trajectory image indices for one step's env results.

    Both views describe the same append record and always hold *every* stored
    index; deciding which of them the model sees belongs to
    ``canonical_tool_result_messages()`` and provider history, not here.

    - ``by_position`` is the storage record itself: one index tuple per entry of
      ``results``, in the same order, including results that stored no image.
    - ``by_call_id`` is the routing map provider history looks up by canonical
      call id. Results without a ``tool_call_id`` or without images are absent,
      so a lookup miss means "nothing stored for this call".
    """

    by_position: tuple[tuple[int, ...], ...]
    by_call_id: dict[str, tuple[int, ...]]


async def append_tool_result_images(
    trajectory: Any,
    results: Sequence[LiteToolResult],
) -> StoredResultImages:
    """Append every result image to ``trajectory`` and index what was stored."""
    by_position: list[tuple[int, ...]] = []
    by_call_id: dict[str, tuple[int, ...]] = {}
    for result in results:
        result_image_indices: list[int] = []
        for image_bytes in result.images:
            trajectory.images.append(await asyncio.to_thread(decode_image, image_bytes))
            image_index = len(trajectory.images) - 1
            result_image_indices.append(image_index)
        by_position.append(tuple(result_image_indices))
        if result.tool_call_id is not None and result_image_indices:
            by_call_id[result.tool_call_id] = tuple(result_image_indices)
    return StoredResultImages(tuple(by_position), by_call_id)


def canonical_tool_result_messages(
    results: Sequence[LiteToolResult],
    tool_calls: list[dict[str, Any]],
    *,
    result_image_indices_by_call_id: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    """Build canonical ``role:"tool"`` messages from aligned env results.

    ``results`` must already be paired and ordered against ``tool_calls`` by the
    loop's ``align_tool_results_to_tool_calls()`` boundary; this builder only
    applies the storage-vs-visibility policy and never re-aligns.
    """
    if not tool_calls:
        return []
    action_batch_call_ids = {
        call_id
        for tool_call in tool_calls
        if (
            (call_id := tool_call_id(tool_call))
            and tool_call_name(tool_call) in LITE_ACTION_BATCH_TOOL_NAMES
        )
    }
    messages: list[dict[str, Any]] = []
    for result in results:
        call_id = result.tool_call_id
        stored_image_indices = (
            tuple(result_image_indices_by_call_id.get(call_id, ())) if call_id else ()
        )
        # Storage vs visibility: every env-returned frame stays in Lite storage, but
        # an action-batch shows only its final frame to the next model turn.
        model_visible_image_indices = (
            stored_image_indices[-1:] if call_id in action_batch_call_ids else stored_image_indices
        )
        message = build_tool_result_message(
            call_id,
            model_visible_image_indices,
            result.text,
            result.metadata,
            error=result.error,
        )
        if message is not None:
            messages.append(message)
    return messages


def canonical_unpaired_response_feedback_messages(
    results: Sequence[LiteToolResult],
    *,
    result_image_indices_by_position: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Build canonical ``role:"user"`` feedback after a no-tool response.

    The env result has no assistant call id to answer. Store every image on the
    trajectory before this call, then expose only each result's latest stored
    image to the next model turn; older frames remain stored but were not part
    of the next prompt.
    """
    require_unpaired_response_feedback(results)
    messages: list[dict[str, Any]] = []
    for result, image_indices in zip(
        results,
        result_image_indices_by_position,
        strict=True,
    ):
        message = build_tool_result_message(
            None,
            tuple(image_indices[-1:]),
            result.text,
            result.metadata,
            error=result.error,
        )
        if message is not None:
            messages.append(message)
    return messages


__all__ = [
    "StoredResultImages",
    "align_tool_results_to_tool_calls",
    "append_tool_result_images",
    "canonical_tool_result_messages",
    "canonical_unpaired_response_feedback_messages",
    "has_model_visible_feedback",
    "require_model_visible_unpaired_response_feedback",
    "require_unpaired_response_feedback",
    "latest_step_feedback",
    "tool_result_text_for_call_id",
]
