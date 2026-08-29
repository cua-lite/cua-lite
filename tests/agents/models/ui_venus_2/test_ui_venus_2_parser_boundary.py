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
)

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
