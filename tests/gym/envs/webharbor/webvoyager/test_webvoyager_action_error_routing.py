"""Selenium interaction failures are per-action feedback, and the webdriver
deadline must not reach session creation.

Run:
    uv run pytest tests/gym/envs/webharbor/webvoyager/test_webvoyager_action_error_routing.py -q

Two regression guards over the in-image ``docker/server.py``, which cannot be
imported host-side (no selenium, no container ``utils``), so both are read with
``ast`` the way the sibling image-side tests are.

1. ``_execute_action`` used to be guarded by ``_MODEL_ACTION_ERROR_TYPES``
   alone, so a stale element or an intercepted click escaped to
   ``except Exception: raise`` and became an HTTP 500 -- one unlucky click
   destroyed the whole trajectory. Over 35 minutes of one training run every
   exception that reached that branch was one of the four pinned below.
   The narrow tuple itself must stay a byte-for-byte copy of the host contract
   (see tests/gym/utils/backend/test_docker_copy_parity.py), so the selenium
   types live in their own tuple and only the action call site widens.

2. The webdriver client deadline must be installed on the driver's OWN urllib3
   pool, after the session exists. ``RemoteConnection.set_timeout`` is a
   classmethod read at request time: it also bounds SESSION CREATION, which
   legitimately outlasts any single command when many browsers start at once
   (it failed 29 resets in a live run), and one thread tightening it races
   against another thread's concurrent session creation.
"""

from __future__ import annotations

import ast
import pathlib

SERVER = pathlib.Path("lite/gym/envs/webharbor/webvoyager/docker/server.py")


def _tree() -> ast.Module:
    return ast.parse(SERVER.read_text())


def _assigned_names(tree: ast.Module, target: str) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == target for t in node.targets
        ):
            assert isinstance(node.value, ast.Tuple), f"{target} is not a tuple literal"
            return [e.id for e in node.value.elts if isinstance(e, ast.Name)]
    raise AssertionError(f"{target} not found in {SERVER}")


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {SERVER}")


def test_selenium_interaction_failures_are_model_feedback() -> None:
    tree = _tree()
    selenium_types = _assigned_names(tree, "_SELENIUM_ACTION_ERROR_TYPES")
    assert selenium_types == [
        "ElementClickInterceptedException",
        "ElementNotInteractableException",
        "StaleElementReferenceException",
        "TimeoutException",
    ], "the measured 500-producing exceptions must stay model-attributable"

    # The host-parity copy must NOT absorb them: selenium is not a host dep.
    assert not set(selenium_types) & set(_assigned_names(tree, "_MODEL_ACTION_ERROR_TYPES"))


def test_only_the_action_call_site_widens() -> None:
    step = _func(_tree(), "_step_sync")
    widened, narrow = [], []
    for node in ast.walk(step):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if not isinstance(handler.type, ast.Name):
                continue
            if handler.type.id not in {"_ACTION_ERROR_TYPES", "_MODEL_ACTION_ERROR_TYPES"}:
                continue  # the sibling `except Exception: raise` is the infra branch
            called = [
                n.func.id
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            ]
            (widened if handler.type.id == "_ACTION_ERROR_TYPES" else narrow).append(called)

    assert any("_execute_action" in c for c in widened), (
        "_execute_action must be guarded by the widened tuple"
    )
    assert all("_execute_action" not in c for c in narrow), (
        "only the action call site widens; envelope parsing cannot raise a selenium error"
    )


def test_the_deadline_does_not_reach_session_creation() -> None:
    # Checked on the AST, not the text: the explanation of why that knob is
    # wrong is itself a comment naming it.
    tree = _tree()
    calls = {
        ast.unparse(n.func)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "RemoteConnection.set_timeout" not in calls, (
        "a process-global deadline also bounds session creation and races across "
        "concurrent resets; bound the driver's own pool instead"
    )

    body = _func(tree, "_new_driver").body
    def line_of(fragment: str) -> int:
        for node in body:
            if fragment in ast.unparse(node):
                return node.lineno
        raise AssertionError(f"{fragment!r} not found in _new_driver")

    assert line_of("webdriver.Chrome") < line_of("connection_pool_kw"), (
        "the deadline must be installed AFTER the session exists"
    )
