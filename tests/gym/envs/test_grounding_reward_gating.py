"""What the grounding reward is allowed to score, and what must not zero it.

The rule these envs implement: **the reward scores the first usable ANSWER.**
It is 0.0 when no usable answer is present, or when the first usable answer is
wrong. Rejected or malformed calls are reported to the model as feedback but do
not poison a surviving first-valid answer.

Nothing distinguished these cases before: the gate was
``had_model_action_error or action_errors or unsupported_reasons``, so ANY
errored call zeroed a correct first-valid click. Both halves are pinned here
because both are silent -- a wrong reward looks exactly like a right one in
aggregate.

These fixtures construct the env classes directly and seed the screenshot bytes
instead of calling ``reset()``, so they need no ScreenSpot-Pro / OSWorld-G data.

This file stays shallow because it asserts the shared pure-grounding reward
contract across ``screenspot_pro`` and ``osworld_g``; each test name keeps the
env-specific half explicit.

Run:
    uv run pytest tests/gym/envs/test_grounding_reward_gating.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lite.core.tools import make_tool_call

# bbox [10, 10, 20, 20] on a 100x100 image -> normalised [0.1, 0.1, 0.2, 0.2].
# cua-lite points are [0, 1000], so 150 -> 0.15 (inside) and 900 -> 0.9 (outside).
_INSIDE = [150, 150]
_OUTSIDE = [900, 900]


def _screenspot_env():
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation={
            "img_filename": "unused.png",
            "instruction": "click the target",
            "bbox": [10, 10, 20, 20],
            "img_size": [100, 100],
        },
        images_dir=Path("/tmp"),
    )
    env._screenshot = b"shot"
    return env


def _osworld_g_env(box_type: str = "bbox"):
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": box_type,
        "box_coordinates": [10, 10, 20, 20],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
    )
    env._screenshot = b"shot"
    return env


# ---------------------------------------------------------------------------
# Baseline: a lone correct / incorrect point still scores the obvious way.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_screenspot_single_point_scores_normally() -> None:
    inside = await _screenspot_env().step(
        [make_tool_call("point", {"coordinate": _INSIDE}, call_id="a")]
    )
    outside = await _screenspot_env().step(
        [make_tool_call("point", {"coordinate": _OUTSIDE}, call_id="a")]
    )
    assert inside.reward == 1.0
    assert outside.reward == 0.0


# ---------------------------------------------------------------------------
# Half A -- first-valid scoring. Later answers do not change the score.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_screenspot_first_point_scores_when_followed_by_miss() -> None:
    result = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="first"),
        make_tool_call("point", {"coordinate": _OUTSIDE}, call_id="second"),
    ])
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_screenspot_first_point_miss_stays_zero_when_followed_by_hit() -> None:
    result = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": _OUTSIDE}, call_id="first"),
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="second"),
    ])
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_osworld_g_first_point_scores_when_followed_by_miss() -> None:
    result = await _osworld_g_env().step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="first"),
        make_tool_call("point", {"coordinate": _OUTSIDE}, call_id="second"),
    ])
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_osworld_g_first_point_miss_stays_zero_when_followed_by_hit() -> None:
    result = await _osworld_g_env().step([
        make_tool_call("point", {"coordinate": _OUTSIDE}, call_id="first"),
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="second"),
    ])
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_screenspot_no_point_scores_zero() -> None:
    """A model that only calls an unadvertised finish tool has not answered."""
    result = await _screenspot_env().step(
        [make_tool_call("terminate", {"status": "success"}, call_id="t")]
    )
    assert result.reward == 0.0


# ---------------------------------------------------------------------------
# Half B -- a NON-answer error must not zero a correct first-valid click.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_screenspot_inactive_finish_tool_does_not_zero_a_correct_point() -> None:
    """``terminate`` is not advertised here (``extra_tools`` defaults to []).

    A model that clicks correctly AND emits a stray ``terminate`` gets feedback
    about the terminate, but the click is still the single well-formed answer.
    Zeroing it grades a correct click as wrong because of an unrelated call.
    """
    env = _screenspot_env()
    result = await env.step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="answer"),
        make_tool_call("terminate", {"status": "success"}, call_id="stray"),
    ])

    assert result.reward == 1.0, (
        "an unrelated rejected call must not zero a correct first-valid answer"
    )
    errors = {r.tool_call_id: r.error for r in result.results}
    assert "terminate is not available" in (errors["stray"] or ""), (
        "the stray call is still reported to the model"
    )


@pytest.mark.asyncio
async def test_screenspot_unknown_tool_does_not_zero_a_correct_point() -> None:
    env = _screenspot_env()
    result = await env.step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="answer"),
        make_tool_call("not_a_real_tool", {}, call_id="bogus"),
    ])

    assert result.reward == 1.0
    errors = {r.tool_call_id: r.error for r in result.results}
    assert "unknown tool" in (errors["bogus"] or "")


@pytest.mark.asyncio
async def test_osworld_g_inactive_finish_tool_does_not_zero_a_correct_point() -> None:
    env = _osworld_g_env()
    result = await env.step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="answer"),
        make_tool_call("terminate", {"status": "success"}, call_id="stray"),
    ])

    assert result.reward == 1.0
    errors = {r.tool_call_id: r.error for r in result.results}
    assert "terminate is not available" in (errors["stray"] or "")


@pytest.mark.asyncio
async def test_screenspot_non_answer_error_still_zeroes_an_INCORRECT_point() -> None:
    """The relaxation must not accidentally reward a miss."""
    result = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": _OUTSIDE}, call_id="answer"),
        make_tool_call("terminate", {"status": "success"}, call_id="stray"),
    ])
    assert result.reward == 0.0


# ---------------------------------------------------------------------------
# ...and malformed/rejected attempts are ignored once a valid answer survives.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_screenspot_malformed_point_alongside_a_good_one_scores_first_valid() -> None:
    result = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": "abc"}, call_id="bad"),
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="good"),
    ])
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_osworld_g_malformed_point_alongside_a_good_one_scores_first_valid() -> None:
    result = await _osworld_g_env().step([
        make_tool_call("point", {"coordinate": "abc"}, call_id="bad"),
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="good"),
    ])
    assert result.reward == 1.0


# ---------------------------------------------------------------------------
# The crux: a correct first valid point remains the answer even when another
# rejected call is present.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rejected_calls_do_not_poison_the_first_valid_answer() -> None:
    stray_finish = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="answer"),
        make_tool_call("terminate", {"status": "success"}, call_id="stray"),
    ])
    wrong_wrapper = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="answer"),
        make_tool_call(
            "computer",
            {"actions": [{"action": "point", "coordinate": _OUTSIDE}]},
            call_id="second_attempt",
        ),
    ])

    assert stray_finish.reward == 1.0, "a non-answer call must not poison the score"
    assert wrong_wrapper.reward == 1.0, "a rejected wrapper must not poison it"
