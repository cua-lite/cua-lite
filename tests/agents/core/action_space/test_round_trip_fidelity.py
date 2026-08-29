"""The action-space round-trip invariant, for core families x canonical action.

    parse(serialize(x)) == x   OR   serialize(x) RAISED   OR   parse(serialize(x)) RAISED

A silent ``[]``, a substituted action, or a re-invented default means the model's
action reaches the environment as something else than it asked for. A loud raise
is an acceptable outcome for both legs: this repo's unknown-model-action policy
makes GPT/Claude strict-mode families that fail loudly rather than silently drop.

Four declared outcomes per ``(action-space key, canonical action)`` cell:

* absent from all tables -> the round trip must be EXACT (the default);
* in :data:`CANNOT_EXPRESS` -> ``convert_tool_calls_to_agent`` (the *to_agent*/
  serialize leg) must raise ``ValueError``, with a reason naming what the wire
  lacks;
* in :data:`DECLARED_GAPS` -> both legs succeed but the round trip diverges, a
  known and pinned lossy conversion. The test asserts the cell still diverges,
  so closing a gap forces a table edit;
* in :data:`RAISES_ON_PARSE` -> the *to_agent* leg succeeds (the wire can carry
  the action, e.g. as an opaque pass-through), but the *from_agent*/parse leg
  raises ``ValueError`` because the provider wire format it produced has no recognized
  native verb. This is GPT/Claude's strict unknown-action policy: an action the
  family's wire cannot spell fails loudly on parse instead of vanishing.

The structured ``convert_tool_calls_{to,from}_agent`` pair is not always the real
wire; the text-rendering families (ui_tars) are not covered here yet.

Provider-family splits live beside their model tests.

Run:
    uv run pytest tests/agents/core/action_space/test_round_trip_fidelity.py
"""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import (
    ActionSpaceRegistry,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
)

# The identity frame keeps canonical coordinates already in [0, 1000], so
# reported coordinate drift is real loss and not rounding against an arbitrary
# size.
RESOLUTION = (1000, 1000)


@pytest.fixture(autouse=True, scope="module")
def _registered():
    from lite.agents.bootstrap import register_all

    register_all()


# =============================================================================
# The canonical cells
# =============================================================================
#
# One representative call per canonical action, plus the argument variants whose
# loss is invisible at the action level (a dropped ``button``/``clicks`` still
# leaves a ``click``).

DESKTOP_CASES: dict[str, dict] = {
    "click": LiteDesktopActionSpace.click(coordinate=[500, 300]),
    "click_right": LiteDesktopActionSpace.click(coordinate=[500, 300], button="right"),
    "click_double": LiteDesktopActionSpace.click(coordinate=[500, 300], clicks=2),
    "type": LiteDesktopActionSpace.type(text="hello"),
    "key": LiteDesktopActionSpace.key(keys=["ctrl", "c"]),
    "key_down": LiteDesktopActionSpace.key_down(keys=["shift"]),
    "key_up": LiteDesktopActionSpace.key_up(keys=["shift"]),
    "hold_key": LiteDesktopActionSpace.hold_key(keys=["shift"], duration=2.0),
    "scroll_down": LiteDesktopActionSpace.scroll(direction="down", amount=3, coordinate=[400, 400]),
    "scroll_up": LiteDesktopActionSpace.scroll(direction="up", amount=3, coordinate=[400, 400]),
    "scroll_left": LiteDesktopActionSpace.scroll(direction="left", amount=3, coordinate=[400, 400]),
    "scroll_right": LiteDesktopActionSpace.scroll(
        direction="right", amount=3, coordinate=[400, 400]
    ),
    "drag": LiteDesktopActionSpace.drag(coordinate=[600, 700], start_coordinate=[100, 200]),
    "mouse_move": LiteDesktopActionSpace.mouse_move(coordinate=[123, 456]),
    "mouse_down": LiteDesktopActionSpace.mouse_down(),
    "mouse_up": LiteDesktopActionSpace.mouse_up(),
    "screenshot": LiteDesktopActionSpace.screenshot(),
    "cursor_position": LiteDesktopActionSpace.cursor_position(),
    "wait": LiteDesktopActionSpace.wait(duration=2),
}

MOBILE_CASES: dict[str, dict] = {
    "tap": LiteMobileActionSpace.tap(coordinate=[500, 300]),
    "tap_double": LiteMobileActionSpace.tap(coordinate=[500, 300], clicks=2),
    "long_press": LiteMobileActionSpace.long_press(coordinate=[500, 300], duration=2),
    "swipe": LiteMobileActionSpace.swipe(start_coordinate=[100, 200], coordinate=[300, 400]),
    "drag": LiteMobileActionSpace.drag(start_coordinate=[100, 200], coordinate=[300, 400]),
    "pinch": LiteMobileActionSpace.pinch(coordinate=[500, 300], direction="out", amount=25),
    "type": LiteMobileActionSpace.type(text="hello"),
    "system_button": LiteMobileActionSpace.system_button(button="Back"),
    "screenshot": LiteMobileActionSpace.screenshot(),
    "wait": LiteMobileActionSpace.wait(duration=2),
}

# The ``@use`` spaces -- one key per (class, platform). ``@point`` grounding
# spaces are absent: their vocabulary is ``point``, not the sets above.
# ``<family>@browser`` resolves to the same class as ``<family>@desktop``.
SPACE_KEYS: tuple[str, ...] = (
    "lite@desktop",
    "lite@mobile",
    "evocua@desktop",
    "fara@desktop",
    "mai_ui@mobile",
    "qwen2_5_vl@desktop",
    "qwen2_5_vl@mobile",
    "qwen3_5@desktop",
    "qwen3_5@mobile",
    "qwen3_vl@desktop",
    "qwen3_vl@mobile",
    "step_gui@mobile",
    "ui_tars@desktop",
    "ui_tars@mobile",
    "ui_tars_15_v1@desktop",
    "ui_tars_15_v1@mobile",
    "ui_venus_2@desktop",
    "ui_venus_2@browser",
    "ui_venus_2@mobile",
)


# =============================================================================
# Declaration 1: the per-family "cannot express" set
# =============================================================================
#
# Serialising one of these MUST raise ``ValueError``. The value names the missing
# native verb.

CANNOT_EXPRESS: dict[str, dict[str, str]] = {
    "evocua@desktop": {
        "scroll_left": "EvoCUA dropped Qwen3-VL's 'hscroll'; 'pixels' is vertical-only",
        "scroll_right": "EvoCUA dropped Qwen3-VL's 'hscroll'; 'pixels' is vertical-only",
    },
    "fara@desktop": {
        "cursor_position": "no query verb",
        "drag": "no drag verb; only left_click and mouse_move",
        "hold_key": "no half-press or hold verb in Fara's enum",
        "key_down": "no half-press or hold verb in Fara's enum",
        "key_up": "no half-press or hold verb in Fara's enum",
        "mouse_down": "no half-press mouse verb",
        "mouse_up": "no half-press mouse verb",
        "screenshot": "Fara screenshots implicitly; no native verb",
        "click_right": "only 'left_click'; the wire carries no button",
        "click_double": "only 'left_click'; the wire carries no repeat count",
        "scroll_left": "no horizontal scroll; 'pixels' is vertical-only",
        "scroll_right": "no horizontal scroll; 'pixels' is vertical-only",
    },
    "qwen2_5_vl@desktop": {
        "cursor_position": "no query verb",
        "hold_key": "the trained computer_use enum has no half-press or hold",
        "key_down": "the trained computer_use enum has no half-press or hold",
        "key_up": "the trained computer_use enum has no half-press or hold",
        "mouse_down": "no half-press mouse verb",
        "mouse_up": "no half-press mouse verb",
        "screenshot": "screenshots are implicit; no native verb",
        "scroll_left": "no horizontal scroll; 'pixels' is vertical-only",
        "scroll_right": "no horizontal scroll; 'pixels' is vertical-only",
    },
    "ui_tars@desktop": {
        "key_down": "the UI-TARS desktop grammar has only 'hotkey'",
        "key_up": "the UI-TARS desktop grammar has only 'hotkey'",
        "hold_key": "the UI-TARS desktop grammar has only 'hotkey'",
        "mouse_move": "the UI-TARS desktop grammar has no hover verb",
        "mouse_down": "the UI-TARS desktop grammar has no half-press mouse verb",
        "mouse_up": "the UI-TARS desktop grammar has no half-press mouse verb",
        "screenshot": "UI-TARS takes screenshots implicitly; no native verb",
        "cursor_position": "the UI-TARS desktop grammar has no query verb",
    },
    "ui_tars@mobile": {
        "drag": "the UI-TARS mobile grammar has only 'drag'-less swipe",
        "tap_double": "the mobile grammar has only 'click' (no 'left_double' as on "
        "desktop); the wire carries no repeat count",
        "pinch": "the UI-TARS mobile grammar has no pinch verb",
        "screenshot": "UI-TARS takes screenshots implicitly; no native verb",
        "wait": "the UI-TARS mobile grammar has no wait verb",
    },
    # The 1.5 grammar is IDENTICAL to the original ui_tars grammar above, so the
    # cells are keyed identically: same actions, same missing wire verbs. Only
    # ``drag`` differs on mobile (this family parses a two-endpoint ``drag`` back
    # to canonical ``drag``, so it round-trips and is not declared).
    "ui_tars_15_v1@desktop": {
        "key_down": "the UI-TARS 1.5 desktop grammar has only 'hotkey'",
        "key_up": "the UI-TARS 1.5 desktop grammar has only 'hotkey'",
        "hold_key": "the UI-TARS 1.5 desktop grammar has only 'hotkey'",
        "mouse_move": "the UI-TARS 1.5 desktop grammar has no hover verb",
        "mouse_down": "the UI-TARS 1.5 desktop grammar has no half-press mouse verb",
        "mouse_up": "the UI-TARS 1.5 desktop grammar has no half-press mouse verb",
        "screenshot": "UI-TARS 1.5 takes screenshots implicitly; no native verb",
        "cursor_position": "the UI-TARS 1.5 desktop grammar has no query verb",
    },
    "ui_tars_15_v1@mobile": {
        "tap_double": "the mobile grammar has only 'click' (no 'left_double' as on "
        "desktop); the wire carries no repeat count",
        "pinch": "the UI-TARS 1.5 mobile grammar has no pinch verb",
        "screenshot": "UI-TARS 1.5 takes screenshots implicitly; no native verb",
        "wait": "the UI-TARS 1.5 mobile grammar has no wait verb",
    },
    "qwen2_5_vl@mobile": {
        "drag": "only 'swipe', which parses back as swipe",
        "tap_double": "only 'click' in the mobile enum; the wire carries no repeat count",
        "pinch": "no pinch verb",
        "screenshot": "no screenshot verb",
    },
    "qwen3_vl@desktop": {
        "cursor_position": "no query verb",
        "hold_key": "the trained computer_use enum has no half-press or hold",
        "key_down": "the trained computer_use enum has no half-press or hold",
        "key_up": "the trained computer_use enum has no half-press or hold",
        "mouse_down": "no half-press mouse verb",
        "mouse_up": "no half-press mouse verb",
        "screenshot": "screenshots are implicit; no native verb",
    },
    "qwen3_vl@mobile": {
        "drag": "not in the trained mobile enum; only 'swipe'",
        "tap_double": "only 'click' in the mobile enum; the wire carries no repeat count",
        "pinch": "no pinch verb",
        "screenshot": "no screenshot verb",
    },
    "qwen3_5@desktop": {
        "cursor_position": "inherited from qwen3_vl -- same computer_use enum",
        "hold_key": "inherited from qwen3_vl -- same computer_use enum",
        "key_down": "inherited from qwen3_vl -- same computer_use enum",
        "key_up": "inherited from qwen3_vl -- same computer_use enum",
        "mouse_down": "inherited from qwen3_vl -- same computer_use enum",
        "mouse_up": "inherited from qwen3_vl -- same computer_use enum",
        "screenshot": "inherited from qwen3_vl -- same computer_use enum",
    },
    "qwen3_5@mobile": {
        "drag": "not in the trained mobile enum; only 'swipe'",
        "tap_double": "only 'click' in the mobile enum; the wire carries no repeat count",
        "pinch": "inherited from qwen3_vl -- same mobile_use enum",
        "screenshot": "inherited from qwen3_vl -- same mobile_use enum",
    },
    "step_gui@mobile": {
        "drag": "no drag verb; 'SLIDE' parses back as swipe",
        "tap_double": "only 'CLICK' in the enum; the wire carries no repeat count",
        "pinch": "no pinch verb",
        "screenshot": "no screenshot verb",
        "system_button": "the enum spells NO system button, not even Back",
    },
    "mai_ui@mobile": {
        "pinch": "no pinch verb",
        "tap_double": "only 'click' in the enum; the wire carries no repeat count",
        "screenshot": "no screenshot verb",
    },
    # UI-Venus-2's Computer grammar is the widest in this table: it spells
    # right / middle / triple click, KeyDown/KeyUp and MouseDown/MouseUp, so the
    # only desktop verbs it cannot carry are the two query-ish ones and the
    # timed hold.
    "ui_venus_2@desktop": {
        "hold_key": "KeyDown/KeyUp hold across turns; the wire carries no duration",
        "screenshot": "UI-Venus-2 screenshots implicitly; no native verb",
        "cursor_position": "no query verb",
    },
    "ui_venus_2@browser": {
        "click_right": "the browser grammar has only Click and DoubleClick",
        "key_down": "the browser grammar has only Hotkey",
        "key_up": "the browser grammar has only Hotkey",
        "hold_key": "the browser grammar has only Hotkey",
        "mouse_down": "no half-press mouse verb",
        "mouse_up": "no half-press mouse verb",
        "screenshot": "UI-Venus-2 screenshots implicitly; no native verb",
        "cursor_position": "no query verb",
    },
    "ui_venus_2@mobile": {
        "pinch": "no pinch verb",
    },
}


# =============================================================================
# Declaration 2: the known gaps, pinned
# =============================================================================
#
# Each row is a divergence the conversion currently has; the reason names the
# lost fact. The end state for every row is either an exact round trip or a move
# into ``CANNOT_EXPRESS``.

DECLARED_GAPS: dict[str, dict[str, str]] = {
    "evocua@desktop": {
        "drag": "start_coordinate fans out to a separate mouse_move; batch has 2 actions",
    },
    "fara@desktop": {
        "scroll_down": "Fara's scroll carries no coordinate",
        "scroll_up": "Fara's scroll carries no coordinate",
    },
    "mai_ui@mobile": {
        "tap": "coordinates go through smart_resize pixel space and re-quantize",
        "long_press": "coordinates re-quantize through smart_resize pixel space",
        "swipe": "the wire carries a direction, not an end point",
    },
    "qwen2_5_vl@desktop": {
        "drag": "start_coordinate fans out to a separate mouse_move; batch has 2 actions",
    },
    "qwen3_5@desktop": {
        "scroll_down": "the wire's scroll carries pixels only, no coordinate",
        "scroll_up": "the wire's scroll carries pixels only, no coordinate",
        "scroll_left": "the wire's hscroll carries pixels only, no coordinate",
        "scroll_right": "the wire's hscroll carries pixels only, no coordinate",
        "drag": "start_coordinate fans out to a separate mouse_move; batch has 2 actions",
    },
    "qwen3_vl@desktop": {
        "scroll_down": "the wire's scroll carries pixels only, no coordinate",
        "scroll_up": "the wire's scroll carries pixels only, no coordinate",
        "scroll_left": "the wire's hscroll carries pixels only, no coordinate",
        "scroll_right": "the wire's hscroll carries pixels only, no coordinate",
        "drag": "start_coordinate fans out to a separate mouse_move; batch has 2 actions",
    },
    "step_gui@mobile": {
        "long_press": "STEP-GUI's LONGPRESS carries no duration",
    },
    "ui_tars@desktop": {
        "scroll_down": "the wire's scroll carries no amount; parse re-invents 5",
        "scroll_up": "the wire's scroll carries no amount; parse re-invents 5",
        "scroll_left": "the wire's scroll carries no amount; parse re-invents 5",
        "scroll_right": "the wire's scroll carries no amount; parse re-invents 5",
        "wait": "the wire's wait carries no duration; parse re-invents 5",
    },
    "ui_tars_15_v1@desktop": {
        "scroll_down": "the wire's scroll carries no amount; parse re-invents 5",
        "scroll_up": "the wire's scroll carries no amount; parse re-invents 5",
        "scroll_left": "the wire's scroll carries no amount; parse re-invents 5",
        "scroll_right": "the wire's scroll carries no amount; parse re-invents 5",
        "wait": "the wire's wait carries no duration; parse re-invents 5",
    },
    "ui_venus_2@desktop": {
        # Swipe always scrolls at the cursor, so the anchor has no wire slot.
        # The MAGNITUDE survives -- it goes out and comes back through
        # PIXELS_PER_CLICK -- so only ``coordinate`` is lost.
        "scroll_down": "Swipe scrolls at the cursor; the anchor has no wire slot",
        "scroll_up": "Swipe scrolls at the cursor; the anchor has no wire slot",
        "scroll_left": "Swipe scrolls at the cursor; the anchor has no wire slot",
        "scroll_right": "Swipe scrolls at the cursor; the anchor has no wire slot",
        "wait": "the wire's Wait carries no duration; parse re-invents 1s",
    },
    "ui_venus_2@browser": {
        "scroll_down": "Scroll keeps the anchor but carries no amount; parse re-invents 5",
        "scroll_up": "Scroll keeps the anchor but carries no amount; parse re-invents 5",
        "scroll_left": "Scroll keeps the anchor but carries no amount; parse re-invents 5",
        "scroll_right": "Scroll keeps the anchor but carries no amount; parse re-invents 5",
        "wait": "the wire's Wait carries no duration; parse re-invents 1s",
    },
    "ui_venus_2@mobile": {
        "long_press": "LongPress takes only a point; the wire carries no duration",
        "wait": "the wire's Wait carries no duration; parse re-invents 1s",
    },
    "ui_tars_15_v1@mobile": {
        # UI-TARS 1.5's mobile wire has ONE drag-shaped verb, `swipe`, so canonical
        # `drag` serializes to it and parses back as `swipe`. This mapping is
        # deliberate -- it is origin/main's behavior, kept under the standing
        # "ui_tars follows origin/main" ruling -- so the gap is declared, not fixed.
        # What is lost is env-visible on exactly one env: mobilegym SPLITS
        # ActionType.SWIPE (a fling) from ActionType.DRAG (press-move-release),
        # while androidworld/androidlab/mobileworld/cua-bench merge both into one
        # handler. So a canonical `drag` becomes a fling on mobilegym and is
        # indistinguishable everywhere else.
        "drag": "the wire has only `swipe`; press-move-release becomes a fling on mobilegym",
    },
}


# =============================================================================
# Declaration 3: parse-side failures, pinned
# =============================================================================
#
# Strict provider-family parse failures live in the provider split files.

RAISES_ON_PARSE: dict[str, dict[str, str]] = {}


# =============================================================================
# Helpers
# =============================================================================


def _cases_for(space) -> dict[str, dict]:
    return MOBILE_CASES if space.platform == "mobile" else DESKTOP_CASES


def _cells() -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for key in SPACE_KEYS:
        space = ActionSpaceRegistry.get(key)
        cells.extend((key, case) for case in _cases_for(space))
    return cells


def _round_trip(space, call: dict) -> list[dict]:
    wire = space.convert_tool_calls_to_agent([call], resolution=RESOLUTION)
    return space.convert_tool_calls_from_agent(wire, resolution=RESOLUTION)


# Built at import time (not inside the fixture) because parametrisation needs the
# ids at collection.
from lite.agents.bootstrap import register_all as _register_all  # noqa: E402

_register_all()
_CELLS = _cells()


# =============================================================================
# The invariant
# =============================================================================


@pytest.mark.parametrize(
    ("key", "case"),
    _CELLS,
    ids=[f"{k}-{c}" for k, c in _CELLS],
)
def test_round_trip_is_exact_or_the_serializer_raised(key: str, case: str) -> None:
    space = ActionSpaceRegistry.get(key)
    call = _cases_for(space)[case]

    if case in CANNOT_EXPRESS.get(key, {}):
        with pytest.raises(ValueError):
            space.convert_tool_calls_to_agent([call], resolution=RESOLUTION)
        return

    if case in RAISES_ON_PARSE.get(key, {}):
        wire = space.convert_tool_calls_to_agent([call], resolution=RESOLUTION)
        with pytest.raises(ValueError):
            space.convert_tool_calls_from_agent(wire, resolution=RESOLUTION)
        return

    restored = _round_trip(space, call)

    if case in DECLARED_GAPS.get(key, {}):
        assert restored != [call], (
            f"{key}/{case} now round-trips exactly, but it is still listed in "
            f"DECLARED_GAPS ({DECLARED_GAPS[key][case]!r}). Delete the row."
        )
        return

    assert restored == [call], (
        f"{key}/{case} does not round-trip and is not declared. Either fix the "
        f"conversion, make the serializer raise and declare it in "
        f"CANNOT_EXPRESS, make the parser raise and declare it in "
        f"RAISES_ON_PARSE, or (last resort) pin it in DECLARED_GAPS with the "
        f"fact that is lost.\n  sent {call}\n  got  {restored}"
    )


# =============================================================================
# Table hygiene
# =============================================================================


@pytest.mark.parametrize("table_name", ["CANNOT_EXPRESS", "DECLARED_GAPS", "RAISES_ON_PARSE"])
def test_declaration_tables_name_only_real_cells(table_name: str) -> None:
    table = {
        "CANNOT_EXPRESS": CANNOT_EXPRESS,
        "DECLARED_GAPS": DECLARED_GAPS,
        "RAISES_ON_PARSE": RAISES_ON_PARSE,
    }[table_name]
    known = set(_CELLS)
    unknown = [
        (key, case) for key, cases in table.items() for case in cases if (key, case) not in known
    ]
    assert not unknown, (
        f"{table_name} declares cells that do not exist (stale key or case): {unknown}"
    )


def test_no_cell_is_declared_twice() -> None:
    tables = {
        "CANNOT_EXPRESS": CANNOT_EXPRESS,
        "DECLARED_GAPS": DECLARED_GAPS,
        "RAISES_ON_PARSE": RAISES_ON_PARSE,
    }
    both = [
        (name_a, name_b, key, case)
        for name_a, table_a in tables.items()
        for name_b, table_b in tables.items()
        if name_a < name_b
        for key, cases in table_a.items()
        for case in cases
        if case in table_b.get(key, {})
    ]
    assert not both, f"a cell cannot be declared in more than one outcome table at once: {both}"


def test_the_canonical_vocabulary_is_fully_covered() -> None:
    """The cases are hand-written, so a canonical action added to
    ``LiteDesktopActionSet``/``LiteMobileActionSet`` would otherwise be silently untested."""
    from lite.core.tools.action_space import LiteDesktopActionSet, LiteMobileActionSet
    from lite.core.tools.calls import tool_call_arguments

    for actions, cases, label in (
        (LiteDesktopActionSet.get_action_names(), DESKTOP_CASES, "desktop"),
        (LiteMobileActionSet.get_action_names(), MOBILE_CASES, "mobile"),
    ):
        covered = {tool_call_arguments(call)["actions"][0]["action"] for call in cases.values()}
        assert covered == set(actions), (
            f"{label} case table is out of sync with the canonical action set: "
            f"missing {sorted(set(actions) - covered)}, "
            f"unknown {sorted(covered - set(actions))}"
        )
