"""A coordinate click lands on the pixel the model named.

Run:
    uv run pytest tests/gym/envs/webharbor/webvoyager/test_webvoyager_coordinate_click.py -q

``element.click()`` implements the W3C element-click algorithm, which discards
the caller's coordinate, re-derives the element's own in-view centre point, and
refuses with ElementClickIntercepted if anything covers THAT pixel. So a model
that picked an uncovered pixel off the screenshot could be refused because a
different part of the element it landed on was covered.

Measured on ``google_flights.8`` (40-task gpt-5.5 rollout, viewport 1280x720):
a ``<span class="material-symbols-outlined">`` icon sits over the centre of a
300px-wide search field. The model correctly walked its click x from 684 to 530
across five turns to get out from under it, and every attempt was re-aimed at
the same covered centre (642, 544) and refused:

    turn  model asked (viewport)   actually clicked
    0004      (588, 618)              (642, 544)
    0005      (684, 618)              (642, 544)
    0008      (540, 618)              (642, 544)
    0014      (535, 618)              (641, 543)
    0017      (530, 618)              (641, 543)

The interception was manufactured by the round-trip, not present on the page.

Index clicks keep ``element.click()``: there the [N] id IS the target and no
pixel was chosen, so the element's own centre is the right thing to aim at.

The behavioural tests below call ``_execute_action`` against a recording fake of
the driver and of selenium's pointer API, so each assertion is about what the
code did -- which calls, in which order, with which arguments. An earlier
version of this file asserted on the parse tree instead, and four wrong
implementations passed every one of its checks: x and y transposed, the
``.click()`` deleted so the pointer moved and never clicked, ``mouse_move``
reverted to element-centering through a differently-named API, and
``mouse_move``'s pointer move deleted outright. The source-shape guard at the
bottom is kept only for the one question a call cannot answer -- whether some
OTHER coordinate action has since acquired the same defect.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

SERVER_PY = pathlib.Path("lite/gym/envs/webharbor/webvoyager/docker/server.py")

VIEWPORT = (1280, 720)
#: A normalized coordinate whose viewport pixel has x != y, so a transposed
#: (y, x) is visible rather than accidentally correct.
COORDINATE = [500, 600]
PIXEL = (640, 432)  # round(1280 * 0.500), round(720 * 0.600)
ELEMENT_CENTRE = (642, 544)  # the covered centre from the measurement above
STALE_CURSOR = (11, 22)


# --- a recording fake of the selenium surface --------------------------------


class _Log(list):
    def names(self) -> list[str]:
        return [entry[0] for entry in self]


class _PointerActions:
    """selenium.webdriver.common.actions.pointer_actions.PointerActions."""

    def __init__(self, log: _Log) -> None:
        self._log = log

    def move_to_location(self, x: int, y: int, **_: Any) -> _PointerActions:
        self._log.append(("move_to_location", x, y))
        return self

    def move_to(self, element: Any, **_: Any) -> _PointerActions:  # element origin
        self._log.append(("move_to", element))
        return self

    def move_by(self, dx: int, dy: int, **_: Any) -> _PointerActions:  # pointer origin
        self._log.append(("move_by", dx, dy))
        return self

    def click(self, element: Any = None, **_: Any) -> _PointerActions:
        self._log.append(("pointer_click", element))
        return self

    def pointer_down(self, **_: Any) -> _PointerActions:
        self._log.append(("pointer_down",))
        return self

    def pointer_up(self, **_: Any) -> _PointerActions:
        self._log.append(("pointer_up",))
        return self


class _ActionBuilder:
    def __init__(self, log: _Log, fail: BaseException | None = None) -> None:
        self._log = log
        self._fail = fail
        self.pointer_action = _PointerActions(log)

    def perform(self) -> None:
        self._log.append(("perform",))
        if self._fail is not None:
            raise self._fail


class _Element:
    def __init__(self, log: _Log, name: str) -> None:
        self._log = log
        self.name = name

    def click(self) -> None:
        self._log.append(("element_click", self.name))

    def __repr__(self) -> str:  # pragma: no cover - failure-message aid
        return f"<{self.name}>"


class _Driver:
    """Answers the scripts the coordinate path runs, and records element clicks."""

    def __init__(self, log: _Log, element_at_point: Any) -> None:
        self._log = log
        self._element_at_point = element_at_point

    def execute_script(self, script: str, *args: Any) -> Any:
        if "window.innerWidth" in script:
            return list(VIEWPORT)
        if "elementFromPoint" in script:
            self._log.append(("element_from_point", args[0], args[1]))
            return self._element_at_point
        if "getBoundingClientRect" in script:
            return list(ELEMENT_CENTRE)
        if "setAttribute" in script:
            return None
        raise AssertionError(f"unexpected script: {script[:80]}")


class _Clock:
    """``_exec_action_click`` sleeps 3s after acting; never actually wait."""

    def __init__(self, log: _Log) -> None:
        self._log = log

    def sleep(self, seconds: float) -> None:
        self._log.append(("sleep", seconds))

    def monotonic(self) -> float:
        return 0.0


def _setup(monkeypatch, tmp_path, *, element_at_point: Any = None, perform_fails=None):
    """A webvoyager instance whose driver and pointer API only record."""
    from lite.gym.envs.webharbor.webvoyager.docker import server

    log = _Log()
    element = _Element(log, "under-the-pixel") if element_at_point is None else element_at_point
    driver = _Driver(log, element or None)

    monkeypatch.setattr(server, "ActionBuilder", lambda _driver: _ActionBuilder(log, perform_fails))
    monkeypatch.setattr(server, "time", _Clock(log))

    inst = server._Instance(
        driver=driver,
        task_id="google_flights.8",
        instruction="find a flight",
        start_url="https://example.com/",
        download_dir=tmp_path,
        max_steps=5,
    )
    inst.last_cursor = STALE_CURSOR
    return server, inst, log, element


# --- what a coordinate click does --------------------------------------------


def test_a_named_pixel_is_clicked_at_that_pixel(monkeypatch, tmp_path) -> None:
    """The pointer goes to (x, y) as given, and then clicks.

    Transposing x and y, or moving without clicking, fails here.
    """
    server, inst, log, _ = _setup(monkeypatch, tmp_path)

    server._execute_action(inst, "click", {"coordinate": COORDINATE})

    assert ("move_to_location", PIXEL[0], PIXEL[1]) in log, (
        f"expected a pointer move to x={PIXEL[0]} y={PIXEL[1]}, got {list(log)}"
    )
    assert "pointer_click" in log.names(), (
        f"the pointer moved to the pixel but never clicked it: {list(log)}"
    )
    assert log.names().index("move_to_location") < log.names().index("pointer_click")
    assert "perform" in log.names(), "a built action sequence is inert until perform()"
    assert "element_click" not in log.names(), (
        "a named pixel must not go through element.click(): it re-derives the "
        "element centre and can refuse a pixel the model saw was clear"
    )


def test_an_index_click_uses_the_element_click_algorithm(monkeypatch, tmp_path) -> None:
    """No pixel was chosen, so the element's own centre is the right target."""
    server, inst, log, _ = _setup(monkeypatch, tmp_path)
    target = _Element(log, "by-index")
    inst.web_eles = [target]

    server._execute_action(inst, "click", {"index": 0})

    assert ("element_click", "by-index") in log, f"index clicks aim at the element: {list(log)}"
    assert "move_to_location" not in log.names(), (
        "an index click has no pixel to move to; issuing one would move the "
        "pointer somewhere the model did not ask for"
    )
    assert inst.last_cursor == ELEMENT_CENTRE


def test_the_overlay_shows_the_pixel_that_was_clicked(monkeypatch, tmp_path) -> None:
    """``last_cursor`` draws the cursor the model reads next turn."""
    server, inst, log, _ = _setup(monkeypatch, tmp_path)

    server._execute_action(inst, "click", {"coordinate": COORDINATE})

    assert inst.last_cursor == PIXEL, (
        f"the overlay must sit on the clicked pixel, not {inst.last_cursor}"
    )


def test_the_overlay_survives_a_coordinate_click_that_could_not_run(monkeypatch, tmp_path) -> None:
    """Regression guard.

    The overlay used to be written only after a successful click, so when the
    coordinate resolved to nothing the next screenshot drew the cursor at the
    PREVIOUS position -- contradicting the action the model had just issued,
    which is the same defect this file exists to prevent.
    """
    server, inst, log, _ = _setup(monkeypatch, tmp_path, element_at_point=False)

    with pytest.raises(ValueError, match="no element at coordinate"):
        server._execute_action(inst, "click", {"coordinate": COORDINATE})

    assert inst.last_cursor == PIXEL, (
        "a coordinate click that failed must still show the model where it "
        f"aimed; the overlay is at {inst.last_cursor}"
    )


# --- the sibling action with the same rule -----------------------------------


def test_a_hover_goes_to_the_pixel_not_the_element_centre(monkeypatch, tmp_path) -> None:
    """``mouse_move`` had the same defect through ``move_to_element``.

    Measured: a hover meant for the left end of a 600px nav item landed 250px
    away, at the item's centre, and the overlay disagreed with the real pointer.
    """
    server, inst, log, element = _setup(monkeypatch, tmp_path)

    server._execute_action(inst, "mouse_move", {"coordinate": COORDINATE})

    assert ("move_to_location", PIXEL[0], PIXEL[1]) in log, (
        f"the hover must address the viewport pixel, got {list(log)}"
    )
    assert "perform" in log.names(), f"the hover was built but never performed: {list(log)}"
    assert "move_to" not in log.names(), (
        "move_to takes an element origin -- its centre, not the named pixel"
    )
    assert inst.last_cursor == PIXEL
    assert inst.last_element is element


def test_a_hover_records_where_it_aimed_even_when_the_move_fails(monkeypatch, tmp_path) -> None:
    server, inst, log, _ = _setup(
        monkeypatch, tmp_path, perform_fails=RuntimeError("move target out of bounds")
    )

    with pytest.raises(RuntimeError):
        server._execute_action(inst, "mouse_move", {"coordinate": COORDINATE})

    assert inst.last_cursor == PIXEL


# --- the invariant, checked across every action that takes a coordinate ------
#
# The tests above run the two actions that were fixed. They would not notice a
# THIRD action acquiring the same defect, so the guard below is written against
# the rule rather than the symptom: any branch that consumes a model-supplied
# coordinate must address the viewport directly. It reads the source because the
# question is about code no test calls yet, not about a call's behaviour.
#
# Two things the first version of this guard got wrong, both fixed here:
#   - it scanned only the dispatch branch, so it could not see `click`, whose
#     pointer call lives in `_exec_action_click`. It now follows calls into the
#     helpers server.py defines, and on the unfixed source it flags `click`.
#   - it matched API names as substrings, so `pointer_action.move_to(element)`
#     -- element-centering, and exactly the defect -- slipped past a blocklist
#     containing `move_to_element`. It now compares attribute names exactly.

#: Coordinate actions this rule does NOT yet hold for, each with a measured
#: reason. Exempting them here is deliberate: the alternative is dropping the
#: guard entirely and losing it for the actions that ARE correct.
#:
#: ``drag`` -- ``move_by_offset`` is relative to the current pointer, so a drag
#: from (250, 120) with the pointer at (700, 400) starts at (950, 520), or
#: raises MoveTargetOutOfBoundsException. Addressing both endpoints absolutely
#: fixes the coordinates but NOT the drag: after any earlier pointer move the
#: slider receives mousedown/mousemove/mouseup on itself, at the right pixels,
#: and never fires `input` -- pointer capture does not survive between two
#: `perform()` calls, and releasing it first (W3C_CLEAR_ACTIONS) worked in one
#: scenario and not the next. That trades a loud failure for a silent one.
#: `drag` is in no shipped config's `valid_actions`, so it waits for a fix that
#: can be verified end to end.
_KNOWN_NOT_PIXEL_ACCURATE = frozenset({"drag"})

#: Compared against ``ast.Attribute.attr`` exactly, never as substrings.
_ELEMENT_CENTERING_APIS = frozenset(
    {
        "move_to_element",  # ActionChains: the element's in-view centre
        "move_to",  # PointerActions: element origin, i.e. its centre
        "move_by_offset",  # ActionChains: relative to the current pointer
        "move_by",  # PointerActions: same, relative
    }
)


def _server_tree() -> ast.Module:
    return ast.parse(SERVER_PY.read_text())


def _coordinate_branches(tree: ast.Module) -> dict[str, ast.If]:
    """``_execute_action`` branches that consume a model-supplied coordinate.

    Detection is by behaviour, not by spelling: a branch qualifies if it reads
    the ``coordinate`` argument or converts one. Keying off the literal string
    alone let a branch hide behind ``args.get(_COORD_KEY)``.
    """
    execute = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_execute_action"
        ),
        None,
    )
    assert execute is not None, f"_execute_action() not found in {SERVER_PY}"

    found: dict[str, ast.If] = {}
    for node in ast.walk(execute):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "name ==" not in test:
            continue
        body = ast.unparse(node.body)
        if "coordinate" not in body and "_norm_coord_to_viewport" not in body:
            continue
        found[test.split("==")[-1].strip().strip("'\"")] = node
    return found


def _reachable_source(tree: ast.Module, body: list[ast.stmt]) -> list[ast.AST]:
    """``body`` plus the bodies of any server.py function it calls.

    Takes the branch BODY, never the ``If`` node: ``ast.walk`` on an ``If``
    descends into ``orelse``, which for an if/elif chain is every branch that
    follows -- so walking the ``click`` node would charge it with ``drag``'s
    ``move_by_offset`` and the guard would fail on correct code.
    """
    defined = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called = {
        n.func.id
        for stmt in body
        for n in ast.walk(stmt)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defined
    }
    return [*body, *(defined[name] for name in sorted(called))]


def test_every_coordinate_action_addresses_the_viewport() -> None:
    tree = _server_tree()
    branches = _coordinate_branches(tree)
    assert branches, "no coordinate-taking action found -- has the dispatch moved?"

    offenders = {}
    for name, node in branches.items():
        if name in _KNOWN_NOT_PIXEL_ACCURATE:
            continue
        used = {
            n.attr
            for scope in _reachable_source(tree, node.body)
            for n in ast.walk(scope)
            if isinstance(n, ast.Attribute) and n.attr in _ELEMENT_CENTERING_APIS
        }
        if used:
            offenders[name] = sorted(used)
    assert not offenders, (
        f"these coordinate actions snap to an element instead of the named pixel: "
        f"{offenders}. Use ActionBuilder(...).pointer_action.move_to_location(x, y)."
    )


def test_every_coordinate_action_is_covered_by_this_file() -> None:
    """A new coordinate action must be named here, so the rule is not silently
    outgrown -- the first version of this fix covered ``click`` alone while
    ``mouse_move`` and ``drag`` had the same defect."""
    assert set(_coordinate_branches(_server_tree())) == {
        "click",
        "scroll",
        "drag",
        "mouse_move",
    }, (
        "the set of coordinate-taking actions changed; check the new one honours "
        "the named pixel and add it here"
    )
    # ``scroll`` is in that set but only partly honours its coordinate, and the
    # guard above cannot see the difference: it uses the pixel for last_cursor
    # and to focus the element there, then scrolls the WINDOW regardless, so a
    # coordinate aimed at an inner scrollable pane scrolls the page instead
    # (measured: window.scrollY 0->300 while pane.scrollTop stayed 0). Steering
    # the scroll by coordinate needs a nearest-scrollable-ancestor walk and
    # would change behaviour for every env that ships `scroll`, so it is called
    # out here rather than silently folded into this change.
