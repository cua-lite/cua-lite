"""Every family that divides a DEFINED-unit wire by ``PIXELS_PER_CLICK`` rounds.

Two different reads live in this tree and only one of them is a plain division:

* ``gpt``, ``gemini`` and ``ui_venus_2`` carry a wire whose unit its own spec
  defines -- OpenAI documents ``scroll_x``/``scroll_y`` as pixel deltas and ships
  ``Math.round(delta / 100)`` as the conversion, Gemini documents
  ``magnitude_in_pixels`` on the same 0-999 grid as its coordinates. There is
  nothing to disambiguate, so the read is ``amount / PIXELS_PER_CLICK``.
* the qwen family, ``evocua`` and ``fara`` carry a wire whose spec never states a
  unit (Qwen's own schema says only "the amount of scrolling" and its reference
  leaves ``_scroll`` unimplemented), so those go through
  :func:`scroll_clicks`, which discriminates notches from screen units by
  magnitude. Feeding a defined-unit pixel value to that reader would misread a
  50 px scroll as 50 clicks.

This pins the FIRST group's rounding only. ``gpt`` and ``gemini`` used floor
division, which silently understates every wire value that is not a multiple of
100 -- a 150 px scroll became one click, not two -- and no test noticed the
difference. The expected values are literal: recomputing ``round(px / 100)``
here would pin nothing.
"""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.utils.geometry import PIXELS_PER_CLICK, scroll_clicks
from lite.agents.models.gemini.action_space import GeminiDesktopActionSpace
from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
from lite.agents.models.ui_venus_2.action_space import UIVenus2DesktopActionSpace

# (wire magnitude, clicks). 149/150 and 349/350 straddle a half-click, which is
# exactly where floor and round disagree; 50 is below one click and must still
# reach the caller as a real scroll rather than zero.
# 250 pins the repo's banker's rounding (round(2.5) == 2), which the qwen scroll-unit
# tests already assume; JS Math.round would give 3, so the choice must be explicit.
ROUNDING_CASES = [(50, 1), (99, 1), (100, 1), (149, 1), (150, 2), (199, 2),
                  (250, 2), (349, 3), (350, 4)]


def _gpt_clicks(wire: int) -> int:
    calls = GPTDesktopActionSpace()._convert_single_from_agent(
        {"type": "scroll", "x": 10, "y": 10, "scroll_x": 0, "scroll_y": wire},
        (1000, 1000),
    )
    arguments = calls[0]["function"]["arguments"]
    return (arguments.get("actions") or [arguments])[0]["amount"]


def _gemini_clicks(wire: int) -> int:
    calls = GeminiDesktopActionSpace()._convert_single_from_agent(
        {"name": "scroll",
         "arguments": {"direction": "down", "magnitude_in_pixels": wire, "x": 10, "y": 10}},
    )
    arguments = calls[0]["function"]["arguments"]
    return (arguments.get("actions") or [arguments])[0]["amount"]


@pytest.mark.parametrize(("wire", "clicks"), ROUNDING_CASES)
def test_gpt_rounds_its_pixel_delta_to_the_nearest_click(wire, clicks):
    assert _gpt_clicks(wire) == clicks


@pytest.mark.parametrize(("wire", "clicks"), ROUNDING_CASES)
def test_gemini_rounds_its_grid_distance_to_the_nearest_click(wire, clicks):
    assert _gemini_clicks(wire) == clicks


def _ui_venus_2_clicks(wire: int) -> int:
    calls = UIVenus2DesktopActionSpace()._convert_single_from_agent(
        {"name": "Swipe", "arguments": {"amount": -wire, "axis": "vertical"}},
    )
    arguments = calls[0]["function"]["arguments"]
    return (arguments.get("actions") or [arguments])[0]["amount"]


@pytest.mark.parametrize(("wire", "clicks"), ROUNDING_CASES)
def test_ui_venus_2_rounds_its_swipe_amount_to_the_nearest_click(wire, clicks):
    assert _ui_venus_2_clicks(wire) == clicks


def test_the_defined_unit_families_agree_with_each_other():
    """One rule, not three. The split this pins cost a click per non-round wire."""
    for wire, _clicks in ROUNDING_CASES:
        assert _gpt_clicks(wire) == _gemini_clicks(wire) == _ui_venus_2_clicks(wire), wire


def test_the_undefined_unit_reader_is_not_this_rule():
    """`scroll_clicks` must NOT be reused here: below one click it means notches.

    A defined-unit family reading 50 through it would scroll fifty clicks where
    the model asked for half of one, so the two readers stay separate.
    """
    assert scroll_clicks(50) == 50
    assert _gpt_clicks(50) == 1
    assert _ui_venus_2_clicks(50) == 1
