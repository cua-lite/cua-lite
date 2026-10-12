"""Unit tests for env wrappers in lite/gym/wrappers.py.

Focus today: LoopDetectWrapper. StepTimeout already has indirect coverage via
the env-server tests.

The wrapper's contract is: on N identical consecutive actions, replace the
triggering action with an internal ``terminate`` and drop the rest of the batch.
The default terminal status is ``success``; tests also pin the explicit status
parameter so scoring policy stays outside loop detection.

Run: uv run pytest -n 0 tests/gym/wrappers/test_wrappers.py -v
"""
from __future__ import annotations

import pytest

from lite.core.messages.final import CONTENT_ONLY_FINAL_REASON
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import (
    RUNTIME_INTERNAL_STOP_REASON_KEY,
    RUNTIME_RESULT_CALL_ID_KEY,
    LiteToolCall,
    make_tool_call,
    tool_call_arguments,
    with_runtime_internal_stop_reason,
)
from lite.gym.base import LiteBaseEnv
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.feedback.ingress import make_internal_terminate_action
from lite.gym.wrappers import LoopDetectWrapper

# ---------------------------------------------------------------------------
# Stub env — records the actions it RECEIVED (post-wrapper), returns vanilla obs.
# ---------------------------------------------------------------------------

class _StubEnv(LiteBaseEnv):
    def __init__(self):
        self.stepped: list[list[LiteToolCall]] = []
        self.reset_count = 0

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(dims=("mobile", "use"))

    async def reset(self) -> LiteEnvObservation:
        self.reset_count += 1
        return LiteEnvObservation(image=None, text="reset")

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.stepped.append(list(actions))
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


def _click(x: int, y: int) -> LiteToolCall:
    return make_tool_call("click", {"x": x, "y": y})


def _fp(action: LiteToolCall) -> str:
    return LoopDetectWrapper._fingerprint(LoopDetectWrapper._fingerprint_actions(action)[0])


def _terminate_success() -> LiteToolCall:
    return make_internal_terminate_action()


def _terminate_success_for(call_id: str) -> LiteToolCall:
    return make_internal_terminate_action(result_call_id=call_id)


# ---------------------------------------------------------------------------
# Init / fingerprint / contract
# ---------------------------------------------------------------------------

def test_init_rejects_threshold_below_two():
    """Threshold of 1 is meaningless (every action is its own loop);
    0 / negative are non-sensical."""
    inner = _StubEnv()
    for bad in (0, 1, -3):
        with pytest.raises(ValueError, match="loop_threshold"):
            LoopDetectWrapper(inner, loop_threshold=bad)


def test_fingerprint_is_canonical_json():
    """Same action with reordered kwargs MUST collapse to same fp —
    mirrors upstream bench_env/runner/base.py:_action_fingerprint use
    of sort_keys=True so the wire ordering of dict items doesn't
    accidentally evade loop detection."""
    a1 = make_tool_call("click", {"x": 1, "y": 2})
    a2 = make_tool_call("click", {"y": 2, "x": 1})
    assert _fp(a1) == _fp(a2)
    a3 = make_tool_call("click", {"x": 1, "y": 3})
    assert _fp(a1) != _fp(a3)


# ---------------------------------------------------------------------------
# Step behavior — across batches
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_trigger_when_actions_differ():
    """Actions with distinct fingerprints never trigger; the env sees the
    actions exactly as submitted."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    for x in range(10):
        result = await env.step([_click(x, x)])
        assert "stop_reason" not in result.info, f"unexpected trigger at x={x}"
    # Inner saw each batch verbatim — no rewrites when below threshold.
    assert inner.stepped == [[_click(x, x)] for x in range(10)]


@pytest.mark.asyncio
async def test_triggers_across_batches_replaces_with_terminate():
    """Three single-action batches, all identical → 3rd action gets
    rewritten to terminate(success); info carries the original fingerprint."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    orig_fp = _fp(_click(50, 50))

    r1 = await env.step([_click(50, 50)])
    r2 = await env.step([_click(50, 50)])
    r3 = await env.step([_click(50, 50)])

    # First two pass through; third is hijacked.
    assert "stop_reason" not in r1.info
    assert "stop_reason" not in r2.info
    assert r3.info["stop_reason"] == "REPETITIVE_LOOP"
    assert r3.info["loop_action"] == orig_fp

    # Inner env's view: 3rd batch's action replaced with terminate.
    assert inner.stepped == [
        [_click(50, 50)],
        [_click(50, 50)],
        [_terminate_success()],
    ]


@pytest.mark.asyncio
async def test_loopdetect_forwards_private_result_id_for_intercepted_call():
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)
    await env.reset()
    first = make_tool_call("click", {"x": 1, "y": 1}, call_id="call_0000")
    second = make_tool_call("click", {"x": 1, "y": 1}, call_id="call_0001")

    await env.step([first])
    result = await env.step([second])

    assert inner.stepped[-1] == [_terminate_success_for("call_0001")]
    assert inner.stepped[-1][0][RUNTIME_RESULT_CALL_ID_KEY] == "call_0001"
    assert result.info["stop_reason"] == "REPETITIVE_LOOP"
    assert result.results == []


@pytest.mark.asyncio
async def test_loopdetect_terminal_status_is_explicit_policy_parameter():
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2, terminal_status="failure")
    await env.reset()

    await env.step([_click(1, 1)])
    result = await env.step([_click(1, 1)])

    forwarded = inner.stepped[-1][0]
    assert tool_call_arguments(forwarded)["status"] == "failure"
    assert tool_call_arguments(forwarded)["reason"] == "REPETITIVE_LOOP"
    assert result.info["stop_reason"] == "REPETITIVE_LOOP"


@pytest.mark.asyncio
async def test_loopdetect_forwards_public_reserved_result_id_to_env_ingress():
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)
    await env.reset()
    first = make_tool_call("click", {"x": 1, "y": 1}, call_id="call_0000")
    second = make_tool_call("click", {"x": 1, "y": 1}, call_id="call_0001")
    second[RUNTIME_RESULT_CALL_ID_KEY] = "call_private"

    await env.step([first])
    result = await env.step([second])

    assert "stop_reason" not in result.info
    assert inner.stepped[-1] == [second]
    assert list(env._recent_fps) == [_fp(first)]


@pytest.mark.asyncio
async def test_loopdetect_forwards_public_internal_stop_reason_to_env_ingress():
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)
    await env.reset()
    first = make_tool_call("click", {"x": 1, "y": 1}, call_id="call_0000")
    second = make_tool_call("click", {"x": 1, "y": 1}, call_id="call_0001")
    second[RUNTIME_INTERNAL_STOP_REASON_KEY] = "REPETITIVE_LOOP"

    await env.step([first])
    result = await env.step([second])

    assert "stop_reason" not in result.info
    assert inner.stepped[-1] == [second]
    assert list(env._recent_fps) == [_fp(first)]


@pytest.mark.asyncio
async def test_no_trigger_when_one_action_differs_in_window():
    """A different action inside the window resets the streak."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    await env.step([_click(50, 50)])
    await env.step([_click(50, 50)])
    r = await env.step([_click(99, 99)])  # different — fp diverges
    assert "stop_reason" not in r.info
    r = await env.step([_click(50, 50)])  # streak restarts from 1
    assert "stop_reason" not in r.info


# ---------------------------------------------------------------------------
# Step behavior — within a single batch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_triggers_within_single_batch():
    """Loop wholly inside one batch triggers — cua-lite step() takes a
    list, an agent could emit [click, click, click] in one go."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    r = await env.step([_click(7, 7), _click(7, 7), _click(7, 7)])

    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    assert r.info["loop_action"] == _fp(_click(7, 7))
    # Env sees [click, click, terminate] — 3rd action replaced.
    assert inner.stepped == [[_click(7, 7), _click(7, 7), _terminate_success()]]


@pytest.mark.asyncio
async def test_within_batch_drops_subsequent_actions():
    """When trigger fires mid-batch, every action after the trigger position
    is dropped — terminate is the last useful action of any episode, so
    forwarding the rest just wastes env-side dispatch."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    # Five identical actions; trigger at index 2 (3rd).
    r = await env.step([_click(1, 1)] * 5)

    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    # Env sees only three (first two clicks + terminate); last two dropped.
    assert inner.stepped == [[_click(1, 1), _click(1, 1), _terminate_success()]]


@pytest.mark.asyncio
async def test_within_batch_preserves_actions_before_trigger():
    """Actions before the trigger position are forwarded as-is — only the
    triggering action itself is rewritten."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    # Mixed batch: first action is different, then 3 identical → trigger at idx 3.
    actions = [_click(99, 99), _click(1, 1), _click(1, 1), _click(1, 1)]
    r = await env.step(actions)

    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    assert actions == [_click(99, 99), _click(1, 1), _click(1, 1), _click(1, 1)]
    assert inner.stepped == [[
        _click(99, 99), _click(1, 1), _click(1, 1), _terminate_success(),
    ]]


# ---------------------------------------------------------------------------
# Reset clears state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_clears_streak():
    """After reset(), the deque must be empty — a new episode starts fresh
    even if a previous episode left wrapper state behind."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    await env.step([_click(5, 5)])
    await env.step([_click(5, 5)])
    # one more click(5,5) would trigger; reset instead.
    await env.reset()
    r = await env.step([_click(5, 5)])
    assert "stop_reason" not in r.info, "reset() must clear the streak deque"
    # Inner saw the post-reset click verbatim — no rewrite.
    assert inner.stepped[-1] == [_click(5, 5)]


# ---------------------------------------------------------------------------
# Wrapper never overrides terminated/truncated/reward
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_carries_loop_period_length_one_for_single_action_repeat():
    """L=1 (pure AAAAA) is the most common loop — info should report it
    so downstream metrics can distinguish stuck-tap from ping-pong."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    await env.step([_click(1, 1)])
    await env.step([_click(1, 1)])
    r = await env.step([_click(1, 1)])
    assert r.info["loop_period_length"] == 1


# ---------------------------------------------------------------------------
# Multi-action periodicity — ABABAB, ABCABCABC, ...
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_triggers_on_period_2_pingpong():
    """ABABABABAB (period 2, 5 reps) must trigger at the 10th action —
    real failure mode: agent oscillates between two screens / two taps."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=5)
    await env.reset()
    A, B = _click(1, 1), _click(2, 2)
    # First 9 actions — buffer fills toward 10, no trigger yet.
    for action in (A, B, A, B, A, B, A, B, A):
        r = await env.step([action])
        assert "stop_reason" not in r.info
    # 10th action closes the ABABABABAB period — trigger fires.
    r = await env.step([B])
    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    assert r.info["loop_period_length"] == 2
    assert r.info["loop_action"] == _fp(B)
    # Inner saw the 10th action rewritten to terminate.
    assert inner.stepped[-1] == [_terminate_success()]


@pytest.mark.asyncio
async def test_triggers_on_period_3_abcabc():
    """ABCABCABCABCABC (period 3, 5 reps) — least common but cheap to
    detect and documents the contract for arbitrary L within max_period."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=5)
    await env.reset()
    A, B, C = _click(1, 1), _click(2, 2), _click(3, 3)
    # 14 actions — one shy of 15 = 3 × 5.
    for action in [A, B, C] * 4 + [A, B]:
        r = await env.step([action])
        assert "stop_reason" not in r.info, "premature trigger"
    # 15th closes the period.
    r = await env.step([C])
    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    assert r.info["loop_period_length"] == 3


@pytest.mark.asyncio
async def test_period_4_within_max_period():
    """L=4 with threshold=3 → need 12 actions. Default max_period=threshold
    is too small (3 < 4), so test explicitly bumps max_period to enable
    L=4 detection. Documents the orthogonality between threshold and
    max_period."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3, max_period=4)
    await env.reset()
    A, B, C, D = _click(1, 1), _click(2, 2), _click(3, 3), _click(4, 4)
    pattern = [A, B, C, D]
    # 11 actions — one shy of 12 = 4 × 3.
    for action in (pattern * 2 + pattern[:3]):
        r = await env.step([action])
        assert "stop_reason" not in r.info
    # Closing the 3rd repeat triggers.
    r = await env.step([D])
    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    assert r.info["loop_period_length"] == 4


@pytest.mark.asyncio
async def test_period_above_max_does_not_trigger():
    """Period 6 with default max_period=5 must not trigger even at high
    repetition counts — the wrapper has to bound its check space."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3, max_period=5)
    await env.reset()
    a1, a2, a3, a4, a5, a6 = (_click(i, i) for i in range(1, 7))
    pattern = [a1, a2, a3, a4, a5, a6]  # length 6 > max_period
    for action in pattern * 3:  # 18 actions, 3 reps of length-6 pattern
        r = await env.step([action])
        assert "stop_reason" not in r.info


@pytest.mark.asyncio
async def test_custom_max_period_extends_detection():
    """Bumping max_period catches longer patterns. threshold=2 + 6 distinct
    actions in the pattern avoids any sub-period false-trigger (no two
    adjacent actions share a fingerprint → L=1 can't fire)."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2, max_period=6)
    await env.reset()
    a1, a2, a3, a4, a5, a6 = (
        _click(1, 1),
        _click(2, 2),
        _click(3, 3),
        _click(4, 4),
        _click(5, 5),
        _click(6, 6),
    )
    pattern = [a1, a2, a3, a4, a5, a6]
    # 11 actions — one short of 12 = 6 × 2.
    feed = pattern + pattern[:5]
    assert len(feed) == 11
    for action in feed:
        r = await env.step([action])
        assert "stop_reason" not in r.info
    # 12th closes the 2nd repeat.
    r = await env.step([a6])
    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    assert r.info["loop_period_length"] == 6


@pytest.mark.asyncio
async def test_init_rejects_max_period_below_one():
    inner = _StubEnv()
    for bad in (0, -3):
        with pytest.raises(ValueError, match="max_period"):
            LoopDetectWrapper(inner, loop_threshold=3, max_period=bad)


@pytest.mark.asyncio
async def test_l2_substring_match_is_not_l2_period_match():
    """Sanity: AABB must NOT trigger as a length-2 period (block [A,A] is
    not equal to block [B,B]). Locks the distinction between a length-2
    *substring* and a length-2 *period* repeated N times. Threshold=3 so
    L=1 streak of 'A' (only 2 As) also doesn't fire."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3, max_period=2)
    await env.reset()
    A, B = _click(1, 1), _click(2, 2)
    for action in [A, A, B, B]:
        r = await env.step([action])
        assert "stop_reason" not in r.info


@pytest.mark.asyncio
async def test_priority_l1_over_l2_on_pure_repeat():
    """Docstring promises ascending-L iteration: a pure AAAA repeat must
    be reported as L=1 (single-action stuck), not L=2 with period AA.
    The buffer is checked in ascending L order; first match wins."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2, max_period=4)
    await env.reset()
    A = _click(1, 1)
    await env.step([A])
    r = await env.step([A])  # 2nd A triggers — L=1 should win over L=2.
    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    assert r.info["loop_period_length"] == 1


@pytest.mark.asyncio
async def test_empty_actions_passes_through_no_trigger():
    """env.step([]) is rare but legal — no actions to fingerprint, no
    trigger possible. Inner env sees empty list verbatim. Locks the
    iteration's "no actions means no work" invariant."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    r = await env.step([])
    assert "stop_reason" not in r.info
    assert inner.stepped == [[]]


@pytest.mark.asyncio
async def test_standalone_extra_tools_are_loop_detected_without_metadata_gate():
    """LoopDetectWrapper fingerprints canonical Lite calls without consulting
    the env metadata or trying to decide whether a name is GUI or extra."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)
    await env.reset()
    goto = make_tool_call("goto", {"url": "https://example.com"})

    r1 = await env.step([goto])
    r2 = await env.step([goto])

    assert "stop_reason" not in r1.info
    assert r2.info["stop_reason"] == "REPETITIVE_LOOP"
    assert inner.stepped == [[goto], [_terminate_success()]]


@pytest.mark.asyncio
async def test_first_action_of_batch_triggers_solo_terminate():
    """Edge: deque is pre-filled to threshold-1 from prior batches, then
    a new batch's FIRST action closes the loop. Forwarded must be just
    [terminate] — no original action leaks through, none of the rest of
    the batch is forwarded either."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    A = _click(1, 1)
    await env.step([A])
    await env.step([A])
    # Now the deque is [A, A]. New batch's first action closes L=1.
    r = await env.step([A, _click(2, 2), _click(3, 3)])
    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    # Inner saw ONE action: terminate. The other batch members dropped.
    assert inner.stepped[-1] == [_terminate_success()]


@pytest.mark.asyncio
async def test_inner_step_exception_restores_deque():
    """If env.step raises (timeout / network / cancellation), the deque
    must NOT retain fingerprints from a batch the env never observed.
    Without this guarantee, a retry-on-exception caller would see stale
    fps and potentially trigger spurious loop detection."""
    class _RaisingEnv(_StubEnv):
        def __init__(self):
            super().__init__()
            self.raise_on_next = False

        async def step(self, actions):
            self.stepped.append(list(actions))
            if self.raise_on_next:
                raise RuntimeError("simulated env timeout")
            return await super().step(actions)

    inner = _RaisingEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3)
    await env.reset()
    # Fill deque to threshold-1.
    await env.step([_click(1, 1)])
    await env.step([_click(1, 1)])
    # Snapshot what the deque should restore to.
    before_exc = list(env._recent_fps)
    # Next step raises. Wrapper should NOT keep the new fingerprint.
    inner.raise_on_next = True
    with pytest.raises(RuntimeError):
        await env.step([_click(1, 1)])
    assert list(env._recent_fps) == before_exc


@pytest.mark.asyncio
async def test_fingerprint_tolerates_non_json_args_instead_of_crashing_step():
    """A non-JSON argument must not take down a live rollout.

    ScaleCUA's generic-kwarg branch runs ``ast.literal_eval`` on model output,
    so ``write(message=b'hello')`` reaches the wrapper as ``bytes`` and
    ``button={'left'}`` as a ``set``. Two committed configs pin
    ``loop_detect: 5`` on ScaleCUA rows, so this path is live. The fingerprint
    only has to be stable and comparable — ``default=str`` is the baseline's
    answer, and the alternative is killing the episode mid-turn.

    Repetition detection must keep working on such arguments, which is why the
    second half asserts a loop still trips rather than merely that no exception
    escaped: a fingerprint that silently collapsed every action to one string
    would pass the first half alone.
    """
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)
    await env.reset()
    weird_action: LiteToolCall = {
        **make_tool_call("type", {"text": b"raw bytes"}),
    }
    weird_action["function"]["arguments"]["modifiers"] = {"ctrl", "alt"}
    result = await env.step([weird_action])
    assert result.info.get("stop_reason") is None

    other: LiteToolCall = make_tool_call("type", {"text": b"different"})
    assert _fp(weird_action) != _fp(other)

    # Same action twice more → the trailing-repetition window closes.
    await env.step([weird_action])
    looped = await env.step([weird_action])
    assert looped.info.get("stop_reason") == "REPETITIVE_LOOP"


@pytest.mark.asyncio
async def test_loopdetect_forwards_invalid_envelope_to_env_ingress():
    """LoopDetect may observe raw model calls, but env ingress owns errors."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)
    await env.reset()

    bad = {"call_id": "bad_0", "arguments": {"coordinate": [1, 2]}}
    await env.step([bad])

    assert inner.stepped == [[bad]]


@pytest.mark.asyncio
async def test_loopdetect_does_not_read_env_metadata():
    """Metadata/valid_actions/extra_tools belong to env ingress, not the wrapper."""

    class _NoMetadataEnv(_StubEnv):
        def _runtime_metadata(self) -> LiteCUAMetadata:
            raise AssertionError("LoopDetectWrapper must not read metadata")

    inner = _NoMetadataEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)

    await env.step([_click(8, 8)])
    result = await env.step([_click(8, 8)])

    assert result.info["stop_reason"] == "REPETITIVE_LOOP"
    assert inner.stepped == [[_click(8, 8)], [_terminate_success()]]


@pytest.mark.asyncio
async def test_loopdetect_ignores_canonical_id_in_fingerprint():
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)

    first = make_tool_call("click", {"x": 3, "y": 4}, call_id="call_a")
    second = make_tool_call("click", {"x": 3, "y": 4}, call_id="call_b")
    await env.step([first])
    result = await env.step([second])

    assert result.info["stop_reason"] == "REPETITIVE_LOOP"
    assert inner.stepped[-1] == [_terminate_success_for("call_b")]


@pytest.mark.asyncio
async def test_loopdetect_does_not_fingerprint_runtime_internal_finish_calls():
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)
    internal_response = with_runtime_internal_stop_reason(
        make_tool_call("response", {"text": "done"}),
        CONTENT_ONLY_FINAL_REASON,
    )

    await env.step([internal_response])
    result = await env.step([internal_response])

    assert "stop_reason" not in result.info
    assert inner.stepped == [[internal_response], [internal_response]]
    assert list(env._recent_fps) == []


@pytest.mark.asyncio
async def test_wrapper_does_not_touch_terminated_truncated_reward():
    """The new design routes through env's terminate handler, so the wrapper
    itself must NOT modify ``terminated`` / ``truncated`` / ``reward`` — those
    come from the env's natural step path (and _evaluate). This isolates the
    fix from the prior bug where setting ``truncated=True`` bypassed
    ``_evaluate`` and zeroed the reward."""
    class _RewardingEnv(_StubEnv):
        async def step(self, actions):
            self.stepped.append(list(actions))
            # Simulate an env whose terminate handler returns real reward.
            return LiteEnvStepResult(
                reward=0.73,
                terminated=True,
                truncated=False,
                info={"executed": "yes"},
            )

    inner = _RewardingEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)
    await env.reset()
    await env.step([_click(1, 1)])
    r = await env.step([_click(1, 1)])  # 2nd identical → triggers

    # Wrapper added its annotations but left numeric/flag fields alone.
    assert r.reward == 0.73
    assert r.terminated is True
    assert r.truncated is False
    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    # Wrapper additions don't clobber existing info keys.
    assert r.info["executed"] == "yes"
