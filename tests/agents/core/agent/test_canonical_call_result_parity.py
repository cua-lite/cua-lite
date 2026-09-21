"""Core canonical call/result parity tests.

The core agent result helpers own the invariant that each canonical tool call
gets at most one aligned env result and one canonical ``role:"tool"`` message.
Model-family tests cover how native provider calls become canonical calls and
how those calls render back into each family's prompt dialect.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/agent/test_canonical_call_result_parity.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from lite.agents.core.agent.utils.tool_results import (
    align_tool_results_to_tool_calls,
    canonical_tool_result_messages,
)
from lite.core.tools import make_tool_call
from lite.core.tools.calls import (
    stamp_tool_call_list_ids,
    tool_call_id,
    tool_call_name,
)
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvStepResult

_CLICK_A = {"action": "click", "coordinate": [10, 20]}
_CLICK_B = {"action": "click", "coordinate": [30, 40]}


def _merged_canonical_calls() -> list[dict[str, Any]]:
    return stamp_tool_call_list_ids(
        [
            make_tool_call(
                "computer",
                {"actions": [_CLICK_A, _CLICK_B]},
            )
        ],
        preserve=False,
    )


def _unmerged_canonical_calls() -> list[dict[str, Any]]:
    return stamp_tool_call_list_ids(
        [
            make_tool_call("computer", {"actions": [_CLICK_A]}),
            make_tool_call("goto", {"url": "https://example.com"}),
            make_tool_call("computer", {"actions": [_CLICK_B]}),
        ],
        preserve=False,
    )


def _env_results(canonical_calls: list[dict[str, Any]]) -> LiteEnvStepResult:
    """A well-behaved env: exactly one paired result per canonical call."""
    return LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id=tool_call_id(call),
                images=[b"shot"],
                text=f"obs for {tool_call_name(call)}",
            )
            for call in canonical_calls
        ]
    )


def _stored_image_indices_by_call_id(
    step_result: LiteEnvStepResult,
) -> dict[str, tuple[int, ...]]:
    """Stand-in for ``append_tool_result_images(...).by_call_id``.

    Every env-returned frame is stored, in result order, so a result carrying
    two frames owns two consecutive trajectory indices. The real appender
    decodes bytes into a trajectory, which these synthetic ``b"shot"`` payloads
    cannot survive, so only the index allocation is reproduced here.
    """
    by_call_id: dict[str, tuple[int, ...]] = {}
    next_index = 0
    for result in step_result.results:
        indices = tuple(range(next_index, next_index + len(result.images)))
        next_index += len(result.images)
        if result.tool_call_id is not None and indices:
            by_call_id[result.tool_call_id] = indices
    return by_call_id


def _role_tool_messages(
    step_result: LiteEnvStepResult,
    canonical_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align at the boundary, then build through the core result-message owner."""
    aligned = align_tool_results_to_tool_calls(canonical_calls, step_result.results)
    return canonical_tool_result_messages(
        aligned,
        canonical_calls,
        result_image_indices_by_call_id=_stored_image_indices_by_call_id(step_result),
    )


def _role_tool_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get("role") == "tool")


@pytest.mark.parametrize(
    "canonical_calls_factory,expected_canonical",
    [
        pytest.param(_merged_canonical_calls, 1, id="merged-2-native-1-canonical"),
        pytest.param(_unmerged_canonical_calls, 3, id="unmerged-3-native-3-canonical"),
    ],
)
def test_role_tool_message_count_tracks_canonical_calls_not_native_calls(
    canonical_calls_factory: Callable[[], list[dict[str, Any]]], expected_canonical: int
) -> None:
    """The core invariant is counted over canonical calls, not provider-native calls."""
    canonical = canonical_calls_factory()
    assert len(canonical) == expected_canonical

    step_result = _env_results(canonical)
    aligned = align_tool_results_to_tool_calls(canonical, step_result.results)
    assert len(aligned) == expected_canonical

    messages = _role_tool_messages(step_result, canonical)
    assert len(messages) == expected_canonical
    assert _role_tool_count(messages) == expected_canonical
    assert [message["tool_call_id"] for message in messages] == [
        tool_call_id(call) for call in canonical
    ]


def test_unaddressed_expansion_frame_is_skipped_not_rejected() -> None:
    """A wrapper-expanded action leaves an unpaired frame; pairing must survive it.

    Shape taken from a live capture: ``LoopDetectWrapper`` truncated a
    ``computer`` batch, so the env executed two actions -- the prefix batch and
    the wrapper's internal ``terminate`` -- and built one frame each. Only the
    terminate answers the assistant's call, because the loop can also trigger on
    the batch's first action, where no prefix is forwarded at all and the
    terminate is the call's only possible result.
    """
    # A merged (two-action) batch on purpose: `prefix_len > 0` needs at least two
    # children, so a single-action batch cannot produce the prefix frame below.
    canonical = _merged_canonical_calls()
    call_id = tool_call_id(canonical[0])
    expanded = LiteEnvStepResult(
        results=[
            LiteToolResult(tool_call_id=None, images=[b"prefix"], text="prefix obs"),
            LiteToolResult(tool_call_id=call_id, images=[b"shot"], text="obs"),
        ]
    )

    aligned = align_tool_results_to_tool_calls(canonical, expanded.results)
    assert [result.tool_call_id for result in aligned] == [call_id]
    assert aligned[0].text == "obs"


def test_unaddressed_frames_do_not_satisfy_a_call_that_got_no_result() -> None:
    """Skipping unpaired frames must not paper over a genuinely missing result."""
    canonical = _unmerged_canonical_calls()[:1]
    only_unaddressed = LiteEnvStepResult(
        results=[LiteToolResult(tool_call_id=None, images=[b"prefix"], text="prefix obs")]
    )

    with pytest.raises(RuntimeError, match="do not match tool_calls"):
        align_tool_results_to_tool_calls(canonical, only_unaddressed.results)


def test_env_that_merges_results_for_distinct_canonical_calls_is_rejected() -> None:
    """One result answering three canonical calls must not silently pass."""
    canonical = _unmerged_canonical_calls()
    merged = LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id=tool_call_id(canonical[0]),
                images=[b"shot"],
                text="obs",
            )
        ]
    )

    with pytest.raises(RuntimeError, match="do not match tool_calls"):
        align_tool_results_to_tool_calls(canonical, merged.results)

    with pytest.raises(RuntimeError, match="do not match tool_calls"):
        _role_tool_messages(merged, canonical)


def test_merged_action_batch_shows_only_its_last_frame_to_the_next_turn() -> None:
    """Action-batch results store every frame but expose only the final one."""
    canonical = _unmerged_canonical_calls()
    assert [tool_call_name(call) for call in canonical] == ["computer", "goto", "computer"]

    step_result = LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id=tool_call_id(call),
                images=[b"shot-a", b"shot-b"],
                text=f"obs for {tool_call_name(call)}",
            )
            for call in canonical
        ]
    )
    assert list(_stored_image_indices_by_call_id(step_result).values()) == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]

    messages = _role_tool_messages(step_result, canonical)

    assert _role_tool_count(messages) == 3
    visible = [
        [part["index"] for part in message["content"] if part["type"] == "image"]
        for message in messages
    ]
    assert visible == [[1], [2, 3], [5]]
