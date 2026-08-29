"""Lints for the two action-space DECLARATION tables, over every registry key.

A family declares two tables on its action-space class, each spelled with the
family's own name::

    LITE_ACTION_NAME_TO_<FAMILY>_ACTION_VALUES        (wrapper families)
    LITE_ACTION_NAME_TO_<FAMILY>_PROVIDER_FLAT_TOOL_NAMES / _FUNCTION_NAMES
    <FAMILY>_ACTION_VALUE_TO_EXTRA_TOOL_NAMES         (wrapper families)
    <FAMILY>_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES

They answer two different questions in the SAME native namespace: which native
spellings are canonical ACTIONS, and which native spellings are canonical EXTRA
TOOLS. That split is not derivable from the emitted schemas — on ``ui_tars@desktop``
the schemas for ``wait`` (an action) and ``finished``/``call_user`` (finish tools)
are byte-identical modulo name and description, and ``step_gui`` packs ``WAIT``
and ``ABORT`` into ONE ``Literal[...]`` on ONE ``@tool`` — so each family
DECLARES it. A declaration nothing checks rots, hence these four lints:

  * **lint 0** — every action-table KEY is a canonical action of the core set
    that registry key serves (desktop / mobile / point / bbox).
  * **lint 1** — every action-table VALUE appears in the family's emitted
    wrapper ``action`` enum OR in its declared ``_SCHEMAS``. A REAL disjunction:
    the qwen-style wrapper families satisfy only the first arm, the
    ``ui_tars`` / ``ui_tars_15_v1`` keys only the second.
  * **lint 2** — every extra-tool-table KEY satisfies the same disjunction.
  * **lint 3** — the two tables are DISJOINT in the NATIVE namespace: a native
    entry is either an action spelling or an extra-tool spelling, never both.

The tool-LAYER invariants (``get_tool_names`` vs ``get_tool_schemas``) are a
different subject and live in
``tests/agents/core/action_space/test_tool_layer_invariants.py``; this module
is the only registry-wide check of the declaration tables themselves.

Hermetic: registry construction and schema generation are pure Python (no model
download, no network).

Run:
    uv run pytest \
        tests/agents/core/action_space/test_action_space_declaration_tables.py -q
"""

from __future__ import annotations

from typing import Any

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import ActionSpaceRegistry
from lite.core.tools.action_space import (
    LiteBBoxActionSet,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
)
from lite.core.tools.schemas import tool_schema_parameters

register_all()

REGISTRY_KEYS = sorted(ActionSpaceRegistry.list_expanded())

# Enumeration canary: an empty parametrize passes vacuously.
assert len(REGISTRY_KEYS) > 50, f"registry enumeration found only {REGISTRY_KEYS}"

#: The grounding spaces whose native spelling exists in NEITHER arm of lint 1,
#: because their wire carries no tool schemas at all — the answer format lives
#: entirely in the prompt, so the synthetic name is declared and nowhere else.
#: MAI-UI's is ``<answer>{"coordinate":[x,y]}</answer>``; UI-Venus-2 answers with
#: a bare ``[x,y]`` list.
_LINT1_EXEMPT_NATIVE_ENTRIES = {
    "MAIUIGroundingPointActionSpace": frozenset({"answer"}),
    "UIVenus2GroundingPointActionSpace": frozenset({"point"}),
}


def _lite_action_name_table(cls: type) -> dict[str, list[str]]:
    """A family's Lite-action-name -> native-spelling table, by its declared name.

    Wrapper families declare ``LITE_ACTION_NAME_TO_<FAMILY>_ACTION_VALUES`` and
    provider-flat families ``LITE_ACTION_NAME_TO_<FAMILY>_PROVIDER_FLAT_TOOL_NAMES``
    / ``_FUNCTION_NAMES``. Every spelling shares the prefix, so this reads the
    prefix rather than naming each family here. A family that declares no table
    (the black-box provider spaces and core ``lite@*``) reports an empty one.
    """
    for name in dir(cls):
        if name.startswith("LITE_ACTION_NAME_TO_"):
            return getattr(cls, name) or {}
    return {}


def _extra_tool_names_table(cls: type) -> dict[str, frozenset[str]]:
    """A family's native-spelling -> canonical-extra-tool table, by its declared name.

    Spelled ``<FAMILY>_ACTION_VALUE_TO_EXTRA_TOOL_NAMES`` for wrapper families
    and ``<FAMILY>_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES`` for
    provider-flat ones.
    """
    for name in dir(cls):
        if name.endswith("_TO_EXTRA_TOOL_NAMES"):
            return getattr(cls, name) or {}
    return {}


def _core_action_names(key: str) -> frozenset[str]:
    """The canonical core action set the registry ``key`` serves.

    The ``@point`` / ``@bbox`` format axis wins over the platform axis: a
    grounding space carries exactly the one coordinate-output action.
    """
    segments = set(key.split("@")[1:])
    # The key segment is a TOOL name (``@point`` / ``@bbox``); what this helper
    # RETURNS is the ACTION set. The two coincide on these N=1 surfaces, so the
    # accessors are spelled apart rather than relying on the coincidence.
    if LitePointActionSet.get_tool_names() & segments:
        return LitePointActionSet.get_action_names()
    if LiteBBoxActionSet.get_tool_names() & segments:
        return LiteBBoxActionSet.get_action_names()
    if "mobile" in segments:
        return LiteMobileActionSet.get_action_names()
    if {"desktop", "browser"} & segments:
        return LiteDesktopActionSet.get_action_names()
    raise AssertionError(f"registry key {key!r} names no core action set")


def _emitted_enum_entries(cls: type) -> frozenset[str]:
    """Entries of every emitted wrapper schema's flat ``action`` enum."""
    entries: set[str] = set()
    for schema in cls.get_tool_schemas():
        action_prop: dict[str, Any] = tool_schema_parameters(schema).get("properties", {}).get(
            "action",
            {},
        )
        entries.update(action_prop.get("enum", ()))
    return frozenset(entries)


def _declared_native_names(cls: type) -> frozenset[str]:
    """The disjunction lint 1 / lint 2 test: emitted enum OR declared schemas."""
    return _emitted_enum_entries(cls) | frozenset(getattr(cls, "_SCHEMAS", {}) or {})


def _native_action_spellings(cls: type) -> frozenset[str]:
    """Every native spelling the family declares as a canonical ACTION."""
    return frozenset(
        entry for entries in _lite_action_name_table(cls).values() for entry in entries
    )


@pytest.mark.parametrize("key", REGISTRY_KEYS)
def test_lint0_action_table_keys_are_canonical_actions(key: str):
    """Every action-table key names an action of the core set it serves.

    A key outside the core set is a canonical name nothing will ever hand the
    table, so the mapping is dead and the real Lite action it was meant to serve
    silently has no native spelling ``[regression: qwen2_5_vl@mobile declared
    ``key: ["key"]`` while canonical MOBILE has NO ``key`` action — and its own
    parser routes the native ``key`` keyevent to ``system_button``.]``
    """
    cls = type(ActionSpaceRegistry.get(key))
    offenders = sorted(set(_lite_action_name_table(cls)) - _core_action_names(key))
    assert not offenders, (
        f"{key} ({cls.__name__}) declares action-table keys that are not "
        f"canonical actions of its core set: {offenders}"
    )


@pytest.mark.parametrize("key", REGISTRY_KEYS)
def test_lint1_action_table_values_are_native_names(key: str):
    """Every declared native action spelling exists on the family's own wire.

    A value absent from both arms is a spelling the model is never offered, so
    ``valid_actions`` gating and the renderer disagree about what the family can
    emit.
    """
    cls = type(ActionSpaceRegistry.get(key))
    exempt = _LINT1_EXEMPT_NATIVE_ENTRIES.get(cls.__name__, frozenset())
    offenders = sorted(_native_action_spellings(cls) - _declared_native_names(cls) - exempt)
    assert not offenders, (
        f"{key} ({cls.__name__}) declares action-table values absent from both "
        f"its emitted action enum and its _SCHEMAS: {offenders}"
    )


@pytest.mark.parametrize("key", REGISTRY_KEYS)
def test_lint2_extra_tool_table_keys_are_native_names(key: str):
    """Every declared native extra-tool spelling exists on the family's wire.

    Same failure as lint 1 on the other table: an extra tool the env activates
    would open an enum entry the model never sees.
    """
    cls = type(ActionSpaceRegistry.get(key))
    exempt = _LINT1_EXEMPT_NATIVE_ENTRIES.get(cls.__name__, frozenset())
    offenders = sorted(set(_extra_tool_names_table(cls)) - _declared_native_names(cls) - exempt)
    assert not offenders, (
        f"{key} ({cls.__name__}) declares extra-tool-table keys absent from both "
        f"its emitted action enum and its _SCHEMAS: {offenders}"
    )


@pytest.mark.parametrize("key", REGISTRY_KEYS)
def test_lint3_tables_are_disjoint_in_the_native_namespace(key: str):
    """A native entry is an ACTION spelling or an EXTRA-TOOL spelling, not both.

    Both tables name entries in the SAME namespace — the family's own wire
    vocabulary — on opposite sides: the action table's VALUES and the extra-tool
    table's KEYS. An overlap would make the same spelling gated by
    ``valid_actions`` and by ``extra_tool_schemas`` at once, which are ORTHOGONAL
    knobs.
    """
    cls = type(ActionSpaceRegistry.get(key))
    overlap = sorted(_native_action_spellings(cls) & set(_extra_tool_names_table(cls)))
    assert not overlap, (
        f"{key} ({cls.__name__}) declares {overlap} as BOTH a native action "
        f"spelling and a native extra-tool spelling"
    )
