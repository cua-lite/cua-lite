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

2. Both constants the deadline depends on are load-bearing and neither is
   guarded by anything but a comment: dropping ``retries=0`` restores urllib3's
   default 3 retries (a measured 3.0s deadline became 12.0s -- at the shipped
   25s that is 100s per call, ~500s for a wedged step, i.e. the symptom this
   file exists to prevent), and the deadline itself only works inside a measured
   band.

3. The webdriver client deadline must be installed on the driver's OWN urllib3
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


def _pool_kw_assignments() -> dict[str, ast.AST]:
    """``_pool.connection_pool_kw["<key>"] = <value>`` inside ``_new_driver``."""
    out: dict[str, ast.AST] = {}
    for node in ast.walk(_func(_tree(), "_new_driver")):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "connection_pool_kw"
            and isinstance(target.slice, ast.Constant)
        ):
            out[target.slice.value] = node.value
    return out


def test_retries_are_switched_off_with_the_deadline() -> None:
    """Deleting ``retries=0`` reads like tidiness and silently undoes the fix.

    urllib3 retries idempotent commands three times by default, so the bound
    becomes 4x the deadline -- measured, a 3.0s deadline took 12.0s. A retry
    cannot help a chromedriver that is not answering; it only multiplies the
    wait back past the host client's 180s.
    """
    kw = _pool_kw_assignments()
    retries = kw.get("retries")
    assert retries is not None, "the pool must switch retries off alongside the deadline"
    assert isinstance(retries, ast.Call), "retries must be an explicit urllib3.Retry(...)"
    zeroed = {
        k.arg: k.value.value
        for k in retries.keywords
        if isinstance(k.value, ast.Constant)
    }
    assert zeroed.get("total") == 0, f"urllib3.Retry(total=...) must be 0, got {zeroed.get('total')}"

    timeout = kw.get("timeout")
    assert timeout is not None, "the deadline must be installed on the pool, not per request"


def test_the_deadline_default_stays_inside_the_measured_band() -> None:
    """The default is pinned between two measured bounds, not chosen for taste.

    Below chromedriver's own 15s page-load timeout it would abort legitimate
    navigations. Above ~36s it stops helping: a wedged step makes about five
    webdriver calls that each must time out, and N x the deadline has to stay
    under the host client's 180s.
    """
    for node in ast.walk(_tree()):
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_WEBDRIVER_CLIENT_TIMEOUT_S" for t in node.targets)
        ):
            continue
        defaults = [
            a.value
            for call in ast.walk(node.value)
            if isinstance(call, ast.Call)
            for a in call.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        assert defaults, "the env-var default must be a literal so it can be checked"
        value = float(defaults[-1])
        assert 15.0 <= value <= 36.0, (
            f"deadline default {value}s is outside the measured band: below 15s it "
            "aborts legitimate navigations, above ~36s N x deadline exceeds the "
            "host client's 180s"
        )
        return
    raise AssertionError("_WEBDRIVER_CLIENT_TIMEOUT_S not found")
