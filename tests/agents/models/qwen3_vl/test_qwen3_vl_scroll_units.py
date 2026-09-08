"""The scroll magnitude contract: one boundary, one coefficient, both 100.

A qwen-style ``pixels`` field carries two units -- a raw wheel-notch count or
thousandths of a 1000x1000 screen -- because the prompt names neither. Only the
magnitude discriminates them:

  * from-agent is shared: below ``PIXELS_PER_CLICK`` the value is notches, at
    or above it screen units,
  * to-agent is per-family (``SCROLL_WIRE_UNIT``), so a notch writer's target
    reads ``3`` and not ``300`` -- switching to screen units at the boundary,
    where the notch spelling stops being readable.

The expected values here are LITERAL. A test that recomputes the rule it is
checking pins nothing: an earlier version of this file did that and stayed green
with the boundary set to 10.
"""
from __future__ import annotations

import json

import pytest

from lite.agents.core.action_space.utils.geometry import PIXELS_PER_CLICK
from lite.agents.models.qwen3_5.action_space import Qwen3_5DesktopActionSpace
from lite.agents.models.qwen3_8.action_space import Qwen3_8DesktopActionSpace
from lite.agents.models.qwen3_vl.action_space import Qwen3VLDesktopActionSpace
from lite.core.tools.action_space import LiteDesktopActionSet

RESOLUTION = (1000, 1000)


def _amount(space, pixels: int) -> int:
    """Canonical click count a family reads out of a wire magnitude."""
    calls = space.convert_tool_calls_from_agent(
        [{"name": "computer_use", "arguments": {"action": "scroll", "pixels": pixels}}],
        resolution=RESOLUTION,
    )
    return calls[0]["function"]["arguments"]["actions"][0]["amount"]


def _pixels(space, amount: int) -> int:
    """Wire magnitude a family renders for a canonical click count."""
    wire = space.convert_tool_calls_to_agent(
        [LiteDesktopActionSet.scroll(direction="down", amount=amount)],
        resolution=RESOLUTION,
    )
    return wire[0]["arguments"]["pixels"]


#: Wire magnitude -> click count, as literals. These fix the boundary at 100:
#: change ``PIXELS_PER_CLICK`` and every row below moves.
@pytest.mark.parametrize(
    "pixels,amount",
    [(-3, 3), (-10, 10), (-50, 50), (-99, 99),
     (-100, 1), (-200, 2), (-301, 3), (-500, 5), (-2000, 20)],
)
def test_from_agent_splits_notches_from_screen_units_at_100(pixels, amount):
    assert _amount(Qwen3VLDesktopActionSpace(), pixels) == amount


def test_the_boundary_is_exclusive():
    """99 is 99 notches; 100 is one click. An inclusive test would send a
    rendered click back multiplied by a hundred."""
    space = Qwen3_8DesktopActionSpace()
    assert _amount(space, -99) == 99
    assert _amount(space, -100) == 1


#: Every scroll magnitude these families were observed writing, from their own
#: rollouts and OSWorld evals, deduplicated per trajectory. Enumerating the real
#: space matters because the last regression hid in a value nobody parametrized.
OBSERVED_WIRE_VALUES = {
    Qwen3_8DesktopActionSpace: [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50,
                                100, 300, 500, 3000, 20000],
    Qwen3_5DesktopActionSpace: [3, 5, 10, 15, 20, 50, 100, 196, 300, 500,
                                1000, 2000, 100000000],
    Qwen3VLDesktopActionSpace: [3, 5, 10, 11, 19, 44, 49, 99, 100, 196,
                                301, 499, 501, 3000],
}


@pytest.mark.parametrize(
    "space,wire",
    [(sp, w) for sp, ws in OBSERVED_WIRE_VALUES.items() for w in ws],
    ids=lambda v: v.__name__ if isinstance(v, type) else str(v),
)
def test_every_observed_wire_value_reads_back_as_at_least_one_click(space, wire):
    """No value the model actually wrote may decode to zero or negative."""
    assert _amount(space(), -wire) >= 1


#: Screen-unit values that are NOT multiples of the coefficient. These are the
#: only rows that tell ``round`` apart from ``//`` or ``int``, and 196 is one the
#: models actually wrote -- the observed-value test above asserts only ``>= 1``,
#: so without these the rounding rule is unpinned.
@pytest.mark.parametrize("pixels,clicks", [(150, 2), (196, 2), (250, 2), (350, 4), (449, 4)])
def test_a_screen_unit_value_rounds_rather_than_truncating(pixels, clicks):
    assert _amount(Qwen3VLDesktopActionSpace(), -pixels) == clicks


@pytest.mark.parametrize("amount,expected", [(2.6, 300), (2.4, 200), (1.5, 200)])
def test_a_float_amount_rounds_before_it_is_rendered(amount, expected):
    """``int(round(...))``, not ``int(...)``: truncation would render 2.6 as 200.

    The float-leak test only checks that no ``.`` reaches the wire, which
    truncation also satisfies.
    """
    call = LiteDesktopActionSet.scroll(direction="down", amount=amount)
    wire = Qwen3VLDesktopActionSpace().convert_tool_calls_to_agent(
        [call], resolution=RESOLUTION)
    assert abs(wire[0]["arguments"]["pixels"]) == expected


#: Click counts straddling the boundary. ``1`` is what an inclusive boundary
#: would send back multiplied by a hundred; the large ones catch saturation.
ROUND_TRIP_CLICKS = [1, 2, 3, 10, 50, 99, 100, 101, 500, 1000]


@pytest.mark.parametrize("space", sorted(OBSERVED_WIRE_VALUES, key=lambda c: c.__name__),
                         ids=lambda c: c.__name__)
@pytest.mark.parametrize("clicks", ROUND_TRIP_CLICKS)
def test_rendering_a_click_count_reads_back_as_the_same_count(space, clicks):
    """to-agent and from-agent are mutually inverse over the whole range.

    The expectation is the INPUT, so a change to either direction alone fails.
    This broke when the render unit became per-family while the read boundary
    stayed shared: 100 clicks rendered as ``100`` and read back as ONE.
    """
    sp = space()
    call = LiteDesktopActionSet.scroll(direction="down", amount=clicks)
    wire = sp.convert_tool_calls_to_agent([call], resolution=RESOLUTION)
    back = sp.convert_tool_calls_from_agent(wire, resolution=RESOLUTION)
    restored = back[0]["function"]["arguments"]["actions"][0]
    assert restored["amount"] == clicks
    assert restored["direction"] == "down"


#: The spelling each family writes, as a measured per-family fact. The round-trip
#: test below cannot pin this -- a click count survives under either unit -- so a
#: literal expected wire value is the only thing holding a family to its dialect.
WIRE_UNIT_BY_KEY = {
    "qwen3_vl@desktop": PIXELS_PER_CLICK,
    "qwen3_5@desktop": 1,
    "qwen3_8@desktop": 1,
    "evocua@desktop": 1,
    "fara@desktop": PIXELS_PER_CLICK,
    "qwen2_5_vl@desktop": PIXELS_PER_CLICK,
}


@pytest.mark.parametrize("key,unit", sorted(WIRE_UNIT_BY_KEY.items()))
def test_to_agent_renders_the_unit_that_family_writes(key, unit):
    from lite.agents.bootstrap import register_all
    from lite.agents.core.action_space.base import ActionSpaceRegistry

    register_all()
    space = ActionSpaceRegistry.get(key)
    assert abs(_pixels(space, 3)) == 3 * unit


def test_every_family_declaring_the_policy_is_pinned():
    """A new family cannot join the policy without a measured unit here."""
    from lite.agents.bootstrap import register_all
    from lite.agents.core.action_space.base import ActionSpaceRegistry

    register_all()
    declared = {k for k in ActionSpaceRegistry.list_expanded()
                if k.endswith("@desktop")
                and hasattr(type(ActionSpaceRegistry.get(k)), "SCROLL_WIRE_UNIT")}
    assert declared == set(WIRE_UNIT_BY_KEY), (
        f"unpinned: {sorted(declared - set(WIRE_UNIT_BY_KEY))}; "
        f"stale: {sorted(set(WIRE_UNIT_BY_KEY) - declared)}")


@pytest.mark.parametrize("space", sorted(OBSERVED_WIRE_VALUES, key=lambda c: c.__name__),
                         ids=lambda c: c.__name__)
def test_a_float_amount_never_leaks_onto_an_integer_wire(space):
    """``amount`` is integer-typed on the wire; source rows can carry floats."""
    call = LiteDesktopActionSet.scroll(direction="down", amount=2.6)
    wire = space().convert_tool_calls_to_agent([call], resolution=RESOLUTION)
    assert "." not in json.dumps(wire), wire


@pytest.mark.parametrize("space", sorted(OBSERVED_WIRE_VALUES, key=lambda c: c.__name__),
                         ids=lambda c: c.__name__)
@pytest.mark.parametrize("clicks", [1, 3, 99, 100, 200])
def test_the_magnitude_helpers_ignore_the_sign_of_the_click_count(space, clicks):
    """``direction`` carries the heading; the helpers take magnitudes only.

    A signed comparison against the boundary let a negative click count skip the
    screen-unit switch and decode a hundred times small. Nothing else in this
    file exercises a negative, so this is the only guard against that.
    """
    from lite.agents.core.action_space.utils.geometry import scroll_wire_magnitude

    unit = type(space()).SCROLL_WIRE_UNIT
    assert (scroll_wire_magnitude(-clicks, wire_unit=unit)
            == scroll_wire_magnitude(clicks, wire_unit=unit))
