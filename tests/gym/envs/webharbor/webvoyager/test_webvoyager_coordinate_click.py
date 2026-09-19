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

Read with ``ast`` rather than imported, like the sibling image-side tests --
``docker/server.py`` runs in-container and has no host-side selenium.
"""

from __future__ import annotations

import ast
import pathlib

SERVER = pathlib.Path("lite/gym/envs/webharbor/webvoyager/docker/server.py")


def _tree() -> ast.Module:
    return ast.parse(SERVER.read_text())


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {SERVER}")


def test_the_executor_accepts_the_pixel_the_model_named() -> None:
    fn = _func(_tree(), "_exec_action_click")
    args = [a.arg for a in fn.args.args]
    assert args[:2] == ["inst", "element"], f"unexpected leading params: {args}"
    assert "point" in args, (
        "the executor must be able to click a chosen pixel; without it every "
        "coordinate click is silently re-aimed at the element's centre"
    )


def test_a_chosen_pixel_is_clicked_as_a_pointer_event_not_an_element_click() -> None:
    """``element.click()`` may only run on the no-pixel branch.

    If it ran unconditionally the W3C hit test would come back and with it the
    manufactured interception this file exists to prevent.
    """
    fn = _func(_tree(), "_exec_action_click")
    branch = next(
        (n for n in ast.walk(fn) if isinstance(n, ast.If) and "point" in ast.unparse(n.test)),
        None,
    )
    assert branch is not None, "the executor must branch on whether a pixel was named"

    body, orelse = ast.unparse(branch.body), ast.unparse(branch.orelse)
    no_pixel, pixel = (body, orelse) if "is None" in ast.unparse(branch.test) else (orelse, body)

    assert "element.click()" in no_pixel, "index clicks still aim at the element"
    assert "element.click()" not in pixel, (
        "a named pixel must not go through element.click(): it re-derives the "
        "element centre and can refuse a pixel the model saw was clear"
    )
    assert "move_to_location" in pixel, (
        "a named pixel is dispatched as a real pointer event at that location"
    )


def test_the_cursor_overlay_shows_where_the_click_actually_went() -> None:
    """``last_cursor`` draws the cursor the model sees next turn.

    It used to be overwritten with the element centre even on the coordinate
    path, so the overlay pointed somewhere the model had not clicked.
    """
    fn = _func(_tree(), "_exec_action_click")
    branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and "point" in ast.unparse(n.test))
    body, orelse = ast.unparse(branch.body), ast.unparse(branch.orelse)
    pixel = orelse if "is None" in ast.unparse(branch.test) else body

    assert "inst.last_cursor = point" in pixel, (
        "on the pixel branch the overlay must be the clicked pixel"
    )
    assert "_element_center_viewport" not in pixel, (
        "the element centre must not overwrite the pixel the model clicked"
    )


def test_only_the_coordinate_call_site_passes_a_pixel() -> None:
    """``click_elem`` is SoM's index action -- it names an element, not a pixel."""
    tree = _tree()
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_exec_action_click"
    ]
    assert calls, "_exec_action_click is never called"

    by_arity: dict[int, list[str]] = {}
    for c in calls:
        by_arity.setdefault(len(c.args), []).append(ast.unparse(c))

    assert 3 in by_arity, "the coordinate click must forward its pixel"
    assert all("_elem_by_index" not in c for c in by_arity[3]), (
        "an index-resolved element has no pixel to forward"
    )
    for c in by_arity.get(2, []):
        assert "_elem_by_index" in c, f"a pixel-less click must be index-resolved: {c}"
