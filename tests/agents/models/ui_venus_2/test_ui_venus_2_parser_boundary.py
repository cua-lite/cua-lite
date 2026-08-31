"""Malformed model output must reach the env as feedback, never as a crash.

Only :class:`ModelToolCallParseError` is caught by
``AdapterBasedAgent._parse_generation_response`` and turned into a terminal
parse-failure final — a bare ``ValueError`` from a parser propagates and kills
the rollout. Every argument reachable from ``<action>`` is model-chosen, so
every rejection on this side has to use the named error.

Run:
    uv run pytest tests/agents/models/ui_venus_2 -p no:cacheprovider -q
"""

from __future__ import annotations

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.ui_venus_2.action_space import (
    UIVenus2BrowserActionSpace,
    UIVenus2DesktopActionSpace,
    UIVenus2GroundingPointActionSpace,
    UIVenus2MobileActionSpace,
    parse_action_text,
)
from lite.agents.models.ui_venus_2.adapter import _parse_grounding_answer
from lite.core.tools.action_space import LiteDesktopActionSet

register_all()


@pytest.mark.parametrize(
    ("space", "agent_call", "match"),
    [
        # Coordinates.
        (
            UIVenus2DesktopActionSpace(),
            {"name": "Hover", "arguments": {}},
            "box is required",
        ),
        (
            UIVenus2DesktopActionSpace(),
            {"name": "Drag", "arguments": {"start": [1, 2]}},
            "end is required",
        ),
        (
            UIVenus2DesktopActionSpace(),
            {"name": "Click", "arguments": {"box": [1, 2, 3]}},
            "exactly 2",
        ),
        (
            UIVenus2MobileActionSpace(),
            {"name": "Click", "arguments": {"point": ["bad", 2]}},
            "finite numeric",
        ),
        (
            UIVenus2MobileActionSpace(),
            {"name": "Swipe", "arguments": {"start": [1, 2]}},
            "end is required",
        ),
        (
            UIVenus2BrowserActionSpace(),
            {"name": "Click", "arguments": {}},
            "point is required",
        ),
        (
            UIVenus2GroundingPointActionSpace(),
            {"name": "point", "arguments": {"box": [1]}},
            "exactly 2",
        ),
        # Key lists.
        (
            UIVenus2DesktopActionSpace(),
            {"name": "Hotkey", "arguments": {"keys": []}},
            "non-empty keys list",
        ),
        (
            UIVenus2DesktopActionSpace(),
            {"name": "KeyDown", "arguments": {}},
            "non-empty keys list",
        ),
        (
            UIVenus2BrowserActionSpace(),
            {"name": "Hotkey", "arguments": {"keys": 5}},
            "non-empty keys list",
        ),
        # Swipe geometry.
        (
            UIVenus2DesktopActionSpace(),
            {"name": "Swipe", "arguments": {"amount": "lots", "axis": "vertical"}},
            "integer amount",
        ),
        (
            UIVenus2DesktopActionSpace(),
            {"name": "Swipe", "arguments": {"amount": -5, "axis": "sideways"}},
            "vertical or horizontal",
        ),
    ],
)
def test_parse_boundaries_raise_the_named_error(space, agent_call, match) -> None:
    with pytest.raises(ModelToolCallParseError, match=match):
        space.convert_tool_calls_from_agent([agent_call])


@pytest.mark.parametrize(
    "space", [UIVenus2DesktopActionSpace(), UIVenus2BrowserActionSpace()],
)
def test_a_bare_string_keys_value_is_accepted_as_one_key(space) -> None:
    """Tolerance, not a rejection: the model sometimes drops the list around a
    single key, and the meaning is unambiguous. (The mobile grammar has no
    Hotkey, so it is not in this sweep.)"""
    (call,) = space.convert_tool_calls_from_agent(
        [{"name": "Hotkey", "arguments": {"keys": "enter"}}]
    )
    (child,) = call["function"]["arguments"]["actions"]
    assert child == {"action": "key", "keys": ["enter"]}


# ---------------------------------------------------------------------------
# Answer-vs-reasoning, truncation, and multi-statement boundaries.
#
# The grounding cases use the shape that motivates the split: reasoning that
# names a candidate, rejects it, then states the real answer.
# ---------------------------------------------------------------------------


def test_grounding_answer_comes_from_after_the_last_think_close() -> None:
    """The reasoning names candidates it rejects; only the tail is the answer."""
    raw = (
        "The output handle is at approximately [479, 614]. Wait -- that is the "
        "input. The output handle is at [512, 634].\n</think>\n\n[512, 634]"
    )
    (call,) = UIVenus2GroundingPointActionSpace().convert_tool_calls_from_agent(
        [_parse_grounding_answer(raw)]
    )
    assert call["function"]["arguments"]["coordinate"] == [512, 634]


def test_a_second_think_close_still_answers_from_the_LAST_one() -> None:
    """A ``</think>`` inside the reasoning must not end the span early.

    The model quotes the tag while narrating, so the first close is not the
    boundary. Scanning to the FIRST one answers with the rejected candidate the
    span exists to skip.
    """
    raw = (
        "I should not emit </think> until I am sure. The input handle is at "
        "[479, 614] -- not the target. The output handle is at [512, 634]."
        "\n</think>\n\n[512, 634]"
    )
    (call,) = UIVenus2GroundingPointActionSpace().convert_tool_calls_from_agent(
        [_parse_grounding_answer(raw)]
    )
    assert call["function"]["arguments"]["coordinate"] == [512, 634]


def test_a_refusal_marker_inside_the_reasoning_is_not_a_refusal() -> None:
    """``[-1,-1]`` only refuses when the model ANSWERS with it."""
    raw = "It is clearly not [-1,-1]; the icon is visible.\n</think>\n[500, 500]"
    parsed = _parse_grounding_answer(raw)
    assert parsed["name"] != "report_infeasible"
    assert parsed["arguments"]["box"] == [500, 500]


def test_a_grounding_answer_with_no_think_close_still_parses() -> None:
    """Truncation mid-reasoning: the whole text is all the answer there is."""
    assert _parse_grounding_answer("the icon sits at [100, 200]") is not None


def test_swipe_without_amount_is_a_parse_error_not_a_default_direction() -> None:
    """``amount`` carries the DIRECTION, so defaulting it silently inverts it."""
    with pytest.raises(ModelToolCallParseError, match="requires an amount"):
        UIVenus2DesktopActionSpace().convert_tool_calls_from_agent(
            [{"name": "Swipe", "arguments": {"axis": "vertical"}}]
        )


@pytest.mark.parametrize(
    ("text", "content"),
    [
        ("Type(content='hello wor", "hello wor"),
        # A ``)`` the model typed is part of the text, not the call's closer.
        ("Type(content='see fig 3) and then the next ste", "see fig 3) and then the next ste"),
        ("Type(content=':) hello wor", ":) hello wor"),
    ],
)
def test_a_truncated_string_argument_is_still_recovered(text, content) -> None:
    """The case the recovery path exists for stays recovered."""
    assert parse_action_text(text) == {"name": "Type", "arguments": {"content": content}}


def test_browser_click_without_a_coordinate_fails_instead_of_emitting_bad_wire() -> None:
    """Canonical ``click`` allows click-at-cursor; this grammar's Click does not.
    Rendering it anyway emits ``Click()``, which this parser then rejects."""
    with pytest.raises(ValueError, match="without a coordinate"):
        UIVenus2BrowserActionSpace().convert_tool_calls_to_agent(
            [LiteDesktopActionSet.click()]
        )


# ---------------------------------------------------------------------------
# Boundaries found by replaying the fixes above against truncation prefixes and
# hand-built tag variants.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("close", ["</think>", "</think >", "</THINK>"])
def test_the_grounding_split_accepts_every_tag_spelling(close) -> None:
    """A literal ``"</think>"`` would miss these and silently fall back to the
    whole-text scan -- i.e. to the rejected candidate this split exists to skip."""
    parsed = _parse_grounding_answer(f"the wrong one is [479, 614]{close}[512, 634]")
    assert parsed["arguments"]["box"] == [512, 634]


