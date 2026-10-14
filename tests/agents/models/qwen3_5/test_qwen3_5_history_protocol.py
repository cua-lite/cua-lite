"""Static validation gate for the Qwen3.5 history protocol default.

These tests deliberately avoid GPU servers, Docker, and real env-server calls.
They live with the model family because the history window is owned by
``Qwen3_5HistoryProtocol``.

Run:
    uv run pytest tests/agents/models/qwen3_5/test_qwen3_5_history_protocol.py -q
"""

from __future__ import annotations

import re

import pytest

from lite.agents.models.qwen3_5.protocol import Qwen3_5HistoryProtocol

_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}


def _trajectory(num_turns: int) -> list[dict]:
    """A GUI trajectory of ``num_turns`` turns whose last turn awaits an action.

    Turn k is (observation_k, assistant_k) except the last, which is the pending
    observation the model is about to answer -- the shape ``process_messages``
    sees at generation time.
    """
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": "SYS"}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Instruction: do the thing"}, dict(_IMAGE)]},
    ]
    for k in range(1, num_turns):
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": f"a{k}", "action_description": f"act{k}"}]})
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"obs{k + 1}"}, dict(_IMAGE)]})
    return messages


def _summarized_steps(num_turns: int, history_n: int | None = None) -> list[int]:
    """The ``Step N`` numbers that got evicted into the ``Previous actions:`` text.

    ``history_n=None`` builds the protocol with NO override, so the caller exercises
    whatever the shipped default is rather than a value it named itself.
    """
    protocol = (
        Qwen3_5HistoryProtocol()
        if history_n is None
        else Qwen3_5HistoryProtocol(history_n=history_n)
    )
    rendered = protocol.process_messages(_trajectory(num_turns))
    for message in rendered:
        for part in message.get("content") or []:
            text = part.get("text", "") if isinstance(part, dict) else ""
            if "Previous actions:" in text:
                return [int(n) for n in re.findall(
                    r"Step (\d+):", text.split("Previous actions:")[1])]
    raise AssertionError("no summary block rendered")


def test_qwen3_5_default_history_contract() -> None:
    protocol = Qwen3_5HistoryProtocol()

    assert protocol.history_n == 100
    assert protocol.image_max == 4
    assert protocol.fold_size == 4
    assert protocol._compute_folded_count(4) == 0
    assert protocol._compute_folded_count(5) == 4
    assert protocol._compute_folded_count(9) == 8


@pytest.mark.parametrize(
    ("history_n", "num_turns", "expected_evicted"),
    [
        (4, 3, 0),      # under the window: nothing evicted
        (4, 4, 0),      # exactly at it: still nothing -- window_start_idx is 0
        (4, 5, 1),      # one past: exactly one action moves to the summary
        (4, 10, 6),
        (100, 100, 0),
        (100, 101, 1),
        (100, 150, 50),
    ],
)
def test_eviction_boundary_falls_at_history_n(
    history_n: int, num_turns: int, expected_evicted: int
) -> None:
    """Turns leave the full-render window at ``history_n``, not near it.

    The window keeps the last ``history_n`` turns; everything older is rendered as
    ``Step N: <action>`` lines under ``Previous actions:``. So the count evicted is
    exactly ``max(0, num_turns - history_n)`` and the evicted steps are the OLDEST
    ones, contiguously from 1.

    Every row here names its own ``history_n``, so this test says what the window
    DOES, not what the shipped default is;
    ``test_shipped_default_evicts_at_turn_101`` below covers that.

    Known deviation from the reference agent (xlang-ai/OSWorld#448), pinned here on
    purpose rather than left to be rediscovered: the reference computes
    ``start_step = max(1, total_steps - history_n)`` as a 1-BASED first-kept step
    and renders ``range(start_step, total_steps + 1)``, so it keeps
    ``history_n + 1`` turns. We keep ``history_n``. The divergence begins at
    ``num_turns == history_n + 1`` -- the ``(100, 101, 1)`` row here is ``(…, 0)``
    upstream.
    """
    evicted = _summarized_steps(num_turns, history_n=history_n)

    assert len(evicted) == expected_evicted
    assert evicted == list(range(1, expected_evicted + 1)), "evicted the wrong steps"


def test_window_keeps_history_n_turns_not_history_n_plus_one() -> None:
    """The off-by-one above, asserted directly on the kept window.

    Separate from the parametrized case so that a future alignment with the
    reference fails HERE, with a name that says what to change, instead of
    failing as an off-by-one in a row of eviction counts.
    """
    # 150 turns, window 100 -> turns 1..50 evicted, 51..150 kept in full.
    evicted = _summarized_steps(150, history_n=100)

    assert evicted[-1] == 50, "we keep history_n turns; the reference keeps history_n + 1"


def test_shipped_default_evicts_at_turn_101() -> None:
    """The DEFAULT window, exercised through the default constructor.

    Every other default-protocol test in this package tops out around 12 turns, and
    the packing tests that reach 100 pass ``history_n`` explicitly -- so before this
    test the shipped default was guarded by one integer-equality assertion, which
    says what the value is and not what it does. Lowering the default to 50 would
    turn the 101-turn case from 1 evicted step into 51 and fail here.
    """
    assert _summarized_steps(100) == []
    assert _summarized_steps(101) == [1]
    assert _summarized_steps(150) == list(range(1, 51))
