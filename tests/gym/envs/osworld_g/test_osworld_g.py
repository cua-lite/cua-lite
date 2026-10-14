"""Tests for the osworld_g CUA-Lite gym environment.

Requires the OSWorld-G upstream repo to be cloned locally:
    uv run python lite/gym/envs/osworld_g/scripts/utils/download_tasks.py

Run:
    uv run pytest tests/gym/envs/osworld_g/test_osworld_g.py -v
"""

from __future__ import annotations

import asyncio

import pytest

import lite.gym as gym
from lite.core.tools import make_tool_call
from lite.core.tools.schemas import tool_schema_name
from lite.gym.envs.osworld_g.main import (
    OSWorldGEnv,
    _all_tasks,
    _data_present,
    _is_point_in_polygon,
)

pytestmark = pytest.mark.skipif(not _data_present(), reason="OSWorld-G data not cloned")

# Trigger lazy registration so ``_all_tasks`` is populated for the tests
# below that iterate it directly. (Skipped by ``pytestmark`` if data is
# absent; safe to call either way.)
if _data_present():
    gym.registry.task_ids("osworld_g")


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

def test_tasks_registered():
    task_ids = gym.registry.task_ids("osworld_g", split="eval")
    assert len(task_ids) == 564, f"expected 564 osworld_g tasks, got {len(task_ids)}"


def test_task_id_format():
    task_ids = gym.registry.task_ids("osworld_g", split="eval")
    for tid in task_ids[:5]:
        # IDs look like ``0FOB4CLBT2-0``
        assert "-" in tid


def test_box_type_distribution():
    """Confirm we have 470 bbox + 40 polygon + 54 refusal — matches reference."""
    counts = {"bbox": 0, "polygon": 0, "refusal": 0}
    for _tid, orig, _ref in _all_tasks:
        counts[orig["box_type"]] = counts.get(orig["box_type"], 0) + 1
    assert counts == {"bbox": 470, "polygon": 40, "refusal": 54}


def test_box_type_and_exclude_reason():
    """``box_type`` is descriptive (one of bbox / polygon / refusal);
    ``exclude_reason`` aligns with the OSWorld convention — set ONLY for
    refusal tasks (the ones users typically want to skip), matching how
    OSWorld tags ``infeasible`` / ``google_auth`` only on tasks the user
    wants to skip. Bbox + polygon stay untagged so the default filter
    ``--filter "lambda m: not m.others.get('exclude_reason')"`` keeps
    both as the "default grounding" surface; mode-specific filtering
    uses ``m.others.get('box_type')`` instead.
    """
    counts = {"bbox": 0, "polygon": 0, "refusal": 0}
    for tid in gym.registry.task_ids("osworld_g", split="eval"):
        env = gym.make(f"osworld_g@{tid}", split="eval")
        meta = env.metadata
        box_type = meta.others.get("box_type")
        excl = meta.others.get("exclude_reason")
        counts[box_type] += 1
        if box_type == "refusal":
            assert excl == "refusal", f"refusal task {tid} missing exclude_reason"
        else:
            assert excl is None, f"non-refusal task {tid} unexpectedly tagged: {excl}"
    assert counts == {"bbox": 470, "polygon": 40, "refusal": 54}


def test_extra_tools_includes_report_infeasible():
    """All osworld_g tasks can opt in to the ``report_infeasible`` extra_tool —
    needed for refusal scoring + for false-refusal on bbox/polygon. Default
    is now ``[]`` (opt-in); yaml must explicitly request the tool."""
    tid = gym.registry.task_ids("osworld_g", split="eval")[0]
    env = gym.make(f"osworld_g@{tid}", split="eval", extra_tools=["report_infeasible"])
    extras = env.metadata.extra_tool_schemas or []
    names = {tool_schema_name(t) for t in extras}
    assert "report_infeasible" in names


# ---------------------------------------------------------------------------
# bbox eval
# ---------------------------------------------------------------------------

def _bbox_env():
    """Return an env for a known bbox task: ``0FOB4CLBT2-0``.

    bbox = [1422.9, 326.4, 26.68, 28.4] in 1920×1080
    bbox center pixel = (1436.24, 340.6); cua-lite norm = (748, 315).
    """
    return gym.make("osworld_g@0FOB4CLBT2-0", split="eval")


def test_bbox_click_inside_returns_1():
    async def go():
        env = gym.make(
            "osworld_g@0FOB4CLBT2-0",
            split="eval",
            extra_tools=["report_infeasible"],
        )
        await env.reset()
        # point center, in cua-lite norm
        actions = [make_tool_call("point", {"coordinate": [748, 315]})]
        res = await env.step(actions)
        return res
    res = asyncio.run(go())
    assert res.reward == 1.0
    assert res.terminated is True
    # Same shape as lite/data/preproc's single-turn label rows: no observation
    # UNROUNDED de-normalization: 748/1000*1920 and 315/1000*1080 exactly. The log
    # used to round to (1436, 340) while ``_evaluate`` scored the float, so a
    # log-only reward audit disagreed with the recorded reward at box boundaries
    # (see test_boundary_point_logs_the_float_the_reward_used).
    assert res.info["executed_actions"] == [
        {"call": "point", "args": {"x": 748 / 1000 * 1920, "y": 315 / 1000 * 1080}}
    ]


def test_boundary_point_logs_the_float_the_reward_used():
    """A log-only reward audit must agree with the recorded reward.

    ``_evaluate`` scores the UNROUNDED de-normalized pixel, so if
    ``info.executed_actions`` rounds to int, an auditor replaying the log
    disagrees with the reward whenever the click lands within half a pixel of a
    box edge. Measured on the campaign corpus: 2 of 60 sampled grounding tasks.

    This one is that case exactly: point [44, 355] on 1920x1080 → y = 355/1000 *
    1080 = 383.4, which IS the top edge of box ``[75, 383.4, 176.2, 17.8]`` — so
    the reward is 1.0, while the old int log (383) replayed as OUTSIDE → 0.0.
    """
    async def go():
        env = gym.make("osworld_g@1YgyNsIUQY-0", split="eval")
        await env.reset()
        res = await env.step([make_tool_call("point", {"coordinate": [44, 355]})])
        return res

    res = asyncio.run(go())
    ann = res.info["annotation"]
    assert ann["box_type"] == "bbox"
    x, y, w, h = ann["box_coordinates"]
    assert y == 383.4, "task fixture changed — pick another exact-boundary task"

    # Replay the env's own bbox rule using ONLY what the log carries.
    logged = res.info["executed_actions"][0]["args"]
    audited = 1.0 if (x <= logged["x"] <= x + w and y <= logged["y"] <= y + h) else 0.0
    assert audited == res.reward == 1.0


def test_bbox_click_outside_returns_0():
    async def go():
        env = _bbox_env()
        await env.reset()
        actions = [make_tool_call("point", {"coordinate": [50, 50]})]
        res = await env.step(actions)
        return res
    res = asyncio.run(go())
    assert res.reward == 0.0


def test_unknown_non_action_tool_returns_unknown_tool_result():
    async def go():
        env = _bbox_env()
        await env.reset()
        return await env.step([
            make_tool_call("frobnicate", {}, call_id="call_frob"),
        ])

    res = asyncio.run(go())
    assert res.terminated is True
    assert res.results[0].tool_call_id == "call_frob"
    assert res.results[0].images == []
    assert res.results[0].text is None
    assert res.results[0].error == "unknown tool: frobnicate"
    assert res.results[0].metadata == {"is_error": True}
    assert res.info["executed_actions"][0] == {
        "call": "noop",
        "args": {
            "name": "frobnicate",
            "reason": "unknown tool",
        },
    }


def test_bbox_report_infeasible_returns_0():
    """report_infeasible on a real-target task is wrong → 0."""
    async def go():
        env = gym.make(
            "osworld_g@0FOB4CLBT2-0",
            split="eval",
            extra_tools=["report_infeasible"],
        )
        await env.reset()
        actions = [make_tool_call("report_infeasible", {"reason": "..."})]
        return await env.step(actions)
    res = asyncio.run(go())
    assert res.reward == 0.0


def test_bbox_point_then_report_infeasible_scores_first_point():
    async def go():
        env = gym.make(
            "osworld_g@0FOB4CLBT2-0",
            split="eval",
            extra_tools=["report_infeasible"],
        )
        await env.reset()
        actions = [
            make_tool_call("point", {"coordinate": [748, 315]}),
            make_tool_call("report_infeasible", {"reason": "hedge"}),
        ]
        return await env.step(actions)

    res = asyncio.run(go())
    assert res.reward == 1.0


def test_bbox_report_infeasible_then_point_scores_first_refusal_as_wrong():
    async def go():
        env = gym.make(
            "osworld_g@0FOB4CLBT2-0",
            split="eval",
            extra_tools=["report_infeasible"],
        )
        await env.reset()
        actions = [
            make_tool_call("report_infeasible", {"reason": "hedge"}),
            make_tool_call("point", {"coordinate": [748, 315]}),
        ]
        return await env.step(actions)

    res = asyncio.run(go())
    assert res.reward == 0.0


# ---------------------------------------------------------------------------
# polygon eval
# ---------------------------------------------------------------------------

def test_polygon_pure_function():
    """``_is_point_in_polygon`` ray-casting matches reference behavior."""
    # Unit square polygon: (0,0) (10,0) (10,10) (0,10)
    poly = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
    assert _is_point_in_polygon(5, 5, poly) is True
    assert _is_point_in_polygon(15, 5, poly) is False
    assert _is_point_in_polygon(5, 15, poly) is False
    # Triangle
    tri = [0.0, 0.0, 10.0, 0.0, 5.0, 10.0]
    assert _is_point_in_polygon(5, 1, tri) is True
    assert _is_point_in_polygon(5, 11, tri) is False
    # Degenerate (< 3 vertices)
    assert _is_point_in_polygon(5, 5, [0.0, 0.0, 10.0, 10.0]) is False


def test_polygon_click_outside_polygon_returns_0():
    """Pick the first polygon task, click [0,0] which is the image
    top-left corner — almost certainly outside any UI element."""
    poly_tid = next(tid for tid, o, _ in _all_tasks if o["box_type"] == "polygon")
    async def go():
        env = gym.make(f"osworld_g@{poly_tid}", split="eval")
        await env.reset()
        actions = [make_tool_call("point", {"coordinate": [0, 0]})]
        return await env.step(actions)
    res = asyncio.run(go())
    assert res.reward == 0.0


# ---------------------------------------------------------------------------
# refusal eval
# ---------------------------------------------------------------------------

def _refusal_tid() -> str:
    return next(tid for tid, o, _ in _all_tasks if o["box_type"] == "refusal")


def test_refusal_report_infeasible_returns_1():
    async def go():
        env = gym.make(
            f"osworld_g@{_refusal_tid()}",
            split="eval",
            extra_tools=["report_infeasible"],
        )
        await env.reset()
        actions = [make_tool_call("report_infeasible", {"reason": "no such element"})]
        return await env.step(actions)
    res = asyncio.run(go())
    assert res.reward == 1.0
    assert res.terminated is True


def test_refusal_click_returns_0():
    async def go():
        env = gym.make(f"osworld_g@{_refusal_tid()}", split="eval")
        await env.reset()
        actions = [make_tool_call("point", {"coordinate": [500, 500]})]
        return await env.step(actions)
    res = asyncio.run(go())
    assert res.reward == 0.0


def test_refusal_point_plus_report_infeasible_returns_0():
    async def go():
        env = gym.make(
            f"osworld_g@{_refusal_tid()}",
            split="eval",
            extra_tools=["report_infeasible"],
        )
        await env.reset()
        actions = [
            make_tool_call("point", {"coordinate": [500, 500]}),
            make_tool_call("report_infeasible", {"reason": "hedge"}),
        ]
        return await env.step(actions)

    res = asyncio.run(go())
    assert res.reward == 0.0


def test_refusal_report_infeasible_then_point_scores_first_refusal():
    async def go():
        env = gym.make(
            f"osworld_g@{_refusal_tid()}",
            split="eval",
            extra_tools=["report_infeasible"],
        )
        await env.reset()
        actions = [
            make_tool_call("report_infeasible", {"reason": "no such element"}),
            make_tool_call("point", {"coordinate": [500, 500]}),
        ]
        return await env.step(actions)

    res = asyncio.run(go())
    assert res.reward == 1.0


# ---------------------------------------------------------------------------
# instruction_style env_kwarg
# ---------------------------------------------------------------------------

def test_instruction_style_default_is_original():
    """Default instruction_style returns the short ``OSWorld-G.json`` text."""
    async def go():
        env = gym.make("osworld_g@0FOB4CLBT2-0", split="eval")
        return (await env.reset()).text
    text = asyncio.run(go())
    assert "Open the filter function" in text  # original instruction


def test_instruction_style_refined_returns_long_form():
    """Refined style returns the longer ``OSWorld-G_refined.json`` text."""
    async def go():
        env = gym.make(
            "osworld_g@0FOB4CLBT2-0",
            split="eval",
            instruction_style="refined",
        )
        return (await env.reset()).text
    text = asyncio.run(go())
    # refined instruction is longer + describes funnel icon explicitly
    assert "funnel" in text.lower()


def test_instruction_style_invalid_raises():
    with pytest.raises(ValueError, match="instruction_style"):
        OSWorldGEnv(
            annotation_original={},
            annotation_refined={},
            images_dir=None,
            task_id="x",
            instruction_style="bogus",
        )
