"""Static guard: no NEW zero-variance ``ActionSpace`` / ``Tools`` subclass.

A *zero-variance subclass* is a class that inherits from a project base and
whose body — after the docstring — is empty (or bare ``pass``). It registers a
distinct key but contributes no behaviour, so the key could have been an extra
alternative in the parent's ``key=`` regex instead of a whole class.

This guards an architectural invariant rather than a shim, which is why it is
the one new guard test the convergence plan sanctions.

Scope is deliberately ``ActionSpace`` / ``Tools`` classes only. Registration-only
``*Agent`` classes (``class FooDesktopUseAgent(AutoAdapterAgent, key=...)``, 49
of them) are the intended shape of the agent registry, not a defect — including
them would make this gate meaningless.

The exemption list is checked in BOTH directions:

* a zero-variance class that is not exempt fails, so no new one can land;
* an exempt class that has since grown a body fails, so the list cannot rot.

Run:
    uv run pytest tests/static/test_zero_variance_subclass.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SKIPPED_PARTS = {
    "__pycache__",
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "_vendor",
    "build",
    "docker",
    "node_modules",
    "slime",
}

CLASS_NAME_SUFFIXES = ("ActionSpace", "Tools")

# Classes the plan knowingly leaves zero-variance. Cutting the
# ``ActionSpace`` -> ``BaseTools`` inheritance edge, which is what would let
# these collapse into their parents' key regexes, is out of scope by the plan's
# own scope test — so an unqualified "no zero-variance subclasses" would be a
# box that can never be checked.
#
# Each entry is ``module path -> reason``. Removing a class from this set must
# be accompanied by deleting the class or giving it a body; adding to it needs a
# reason of the same kind.
EXEMPT: dict[str, str] = {
    "lite/agents/models/evocua/action_space.py::EvoCUADesktopGroundingPointActionSpace": (
        "EvoCUA reuses the Qwen3-VL desktop grounding surface verbatim; only the "
        "registry key differs. Collapsing it into the parent's key regex would "
        "put a second vendor's name on a Qwen3-VL class."
    ),
    "lite/agents/models/qwen3_8/action_space.py::Qwen3_8MobileActionSpace": (
        "The expanded harness Qwen3.8 is served with declares only the desktop "
        "computer_use tool, so mobile has no delta to project. It inherits "
        "Qwen3-VL rather than the Qwen3.5 subclass on purpose: that subclass's "
        "left_click repair was justified by measured 3.5-family leakage, and a "
        "Qwen3.8 mobilegym rollout emitted zero such turns."
    ),
    "lite/agents/models/qwen3_8/action_space.py::Qwen3_8DesktopGroundingPointActionSpace": (
        "A grounding surface exposes one click action, which the expanded enum "
        "does not change. The class exists to own the qwen3_8@(desktop|browser)@point "
        "key so the qwen3_8 grounding adapter resolves."
    ),
    "lite/agents/models/qwen3_8/action_space.py::Qwen3_8MobileGroundingPointActionSpace": (
        "Mobile counterpart of the above, for the qwen3_8@mobile@point key."
    ),
}


def _iter_source_files() -> list[Path]:
    """Discovery. The non-empty assert is the SANCTIONED CANARY shape for every
    discovery-based gate in ``tests/static/``: a gate whose in-scope set is
    walked out of the tree passes VACUOUSLY once the walk comes back empty, so
    a directory rename would turn the gate green instead of red. Assert the
    walk found something, so the rename reddens here.
    """
    paths = sorted(
        path
        for path in (REPO / "lite").rglob("*.py")
        if not SKIPPED_PARTS & set(path.relative_to(REPO).parts)
    )
    assert paths, f"discovery found no .py files under {REPO / 'lite'}"
    return paths


def _body_is_empty(node: ast.ClassDef) -> bool:
    """True when the class body is only a docstring and/or ``pass``."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return all(isinstance(stmt, ast.Pass) for stmt in body)


def _collect() -> tuple[dict[str, str], set[str]]:
    """Return (zero-variance classes -> ``file:line``, all in-scope class ids)."""
    zero_variance: dict[str, str] = {}
    in_scope: set[str] = set()
    for path in _iter_source_files():
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not node.bases:
                continue
            if not node.name.endswith(CLASS_NAME_SUFFIXES):
                continue
            identifier = f"{rel}::{node.name}"
            in_scope.add(identifier)
            if _body_is_empty(node):
                zero_variance[identifier] = f"{rel}:{node.lineno}"
    return zero_variance, in_scope


def test_no_unexempted_zero_variance_subclass() -> None:
    """A zero-variance subclass must be named in ``EXEMPT`` with a reason."""
    zero_variance, _ = _collect()
    offenders = sorted(set(zero_variance) - set(EXEMPT))
    assert not offenders, (
        "Zero-variance ActionSpace/Tools subclass(es) with an empty body. Either "
        "give the class a body, fold its key into the parent's key= regex, or add "
        "it to EXEMPT with a reason:\n"
        + "\n".join(f"  {zero_variance[name]}  {name.split('::')[1]}" for name in offenders)
    )


def test_exemptions_are_still_zero_variance() -> None:
    """An exemption that has grown a body must be removed, not left to rot."""
    zero_variance, in_scope = _collect()
    stale = sorted(name for name in EXEMPT if name in in_scope and name not in zero_variance)
    assert not stale, (
        "EXEMPT entries whose class now has a real body — drop them from EXEMPT:\n"
        + "\n".join(f"  {name}" for name in stale)
    )


def test_exemptions_all_exist() -> None:
    """An exemption naming a class that no longer exists must be removed.

    Doubles as the discovery canary this module has always carried by
    accident: it asserts every ``EXEMPT`` name is IN scope, so an empty walk
    reddens it. ``_iter_source_files`` now states that guard outright — see
    its docstring for the shape the other ``tests/static/`` gates copy.
    """
    _, in_scope = _collect()
    missing = sorted(name for name in EXEMPT if name not in in_scope)
    assert not missing, (
        "EXEMPT entries naming a class that no longer exists — drop them:\n"
        + "\n".join(f"  {name}" for name in missing)
    )
