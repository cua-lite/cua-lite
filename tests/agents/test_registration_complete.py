"""Lock agent/env REGISTRATION + registry slug surface against the refactor.

The refactor renames module paths (``lite.agents.models`` → ``models``,
``lite.agents.extensions`` → ``envs``, etc.) but MUST NOT change any registry
*slug*: keys are import-path-independent. These tests freeze the registered
slug set + resolution behavior for all four ``key=`` registries, the protocol
registry, and the gym env-id catalog, so a re-export that silently drops a
registration (a class that no longer imports → never runs ``__init_subclass__``)
fails loudly instead of at rollout time.

Goldens here were COMPUTED from the live registries (not hand-typed); see the
inline "golden:" comments. Slugs are import-path-independent registry contracts.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/test_registration_complete.py -p no:cacheprovider -q
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lite.gym as gym  # noqa: E402
from lite.agents.bootstrap import (  # noqa: E402
    _OPTIONAL_REGISTRATION_MODULES,
    _REGISTRATION_MODULES,
    register_all,
)
from lite.agents.core.action_space.base import ActionSpaceRegistry  # noqa: E402
from lite.agents.core.adapter.base import AgentAdapterRegistry  # noqa: E402
from lite.agents.core.agent.base import AgentRegistry  # noqa: E402
from lite.agents.core.protocol.base import ProtocolRegistry  # noqa: E402

# Register every built-in agent, adapter, action space, and protocol (plus the
# env bridges) via each submodule's ``key=`` declarations.
register_all()

# =============================================================================
# Family coverage matrix
# =============================================================================
# The agent families, DISCOVERED from the ``lite/agents/models/*/`` package
# directories. Deliberately NOT derived from the registries this test asserts
# against: the whole point is that a dropped registration fails loudly, and an
# input read off the registries would just SHRINK the parametrize when one goes
# missing — a gate that cannot fail. The directory listing is the independent
# source. (A hand-written list drifts the other way: it stood at 15 while
# ``bootstrap.register_all`` registered 16, so ``fara`` was never exercised.)
MODELS_DIR = Path(__file__).parents[2] / "lite" / "agents" / "models"


def _discover_families() -> list[str]:
    families = sorted(
        path.name
        for path in MODELS_DIR.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    # Discovery canary: a rename/move of ``lite/agents/models`` must redden
    # here, not silently empty the parametrize below.
    assert families, f"discovery found no agent-family packages under {MODELS_DIR}"
    return families


ALL_FAMILIES = _discover_families()

# Mobile-ONLY families: only ``@mobile@use`` is registered; a desktop
# lookup legitimately misses (make(..., desktop) raises KeyError).
MOBILE_ONLY_FAMILIES = ["mai_ui", "step_gui"]

# Registries where a family legitimately has NO entry (discovered from code):
#   - "lite":  the cua-lite reference DIALECT (adapter + action-space +
#              protocol) has no concrete Agent class — there is no LiteAgent,
#              so AgentRegistry has no ``lite@...`` key.
#   - "claude"/"gpt": API agents talk to the provider directly and DON'T use a
#              BaseAgentAdapter, so AgentAdapterRegistry has no claude/gpt key
#              (they still have an Agent class and an action space).
_NO_AGENT = {"lite"}
_NO_ADAPTER = {"claude", "gpt", "gemini"}


def _platform(family: str) -> str:
    return "mobile" if family in MOBILE_ONLY_FAMILIES else "desktop"


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_all_families_register_in_each_registry(family: str) -> None:
    """Each family RESOLVES its representative key (no instantiation) in every
    registry where it legitimately has an entry.

    Uses ``contains`` / ``_find_item`` (lookup only — never ``get()``), so a
    refactor that drops a registration trips here without needing a processor,
    generate_fn, or live env.
    """
    plat = _platform(family)
    adapter_agent_key = f"{family}@{plat}@use"   # adapter + agent grain
    action_space_key = f"{family}@{plat}"               # action-space grain

    # ActionSpace: every family has one (API families included).
    assert ActionSpaceRegistry.contains(action_space_key), (
        f"ActionSpaceRegistry missing {action_space_key!r}"
    )
    # ``_find_item`` returns the (class, matched_pattern) tuple — assert the
    # class side is non-None to prove a real resolution (not an empty hit).
    assert ActionSpaceRegistry._find_item(action_space_key)[0] is not None

    # Adapter: all but the API families (claude/gpt have no adapter).
    if family not in _NO_ADAPTER:
        assert AgentAdapterRegistry.contains(adapter_agent_key), (
            f"AgentAdapterRegistry missing {adapter_agent_key!r}"
        )
        assert AgentAdapterRegistry._find_item(adapter_agent_key)[0] is not None
    else:
        assert not AgentAdapterRegistry.contains(adapter_agent_key), (
            f"{family} unexpectedly gained an adapter — update _NO_ADAPTER"
        )

    # Agent: all but the reference dialect ``lite`` (no agent class).
    if family not in _NO_AGENT:
        assert AgentRegistry.contains(adapter_agent_key), (
            f"AgentRegistry missing {adapter_agent_key!r}"
        )
        assert AgentRegistry._find_item(adapter_agent_key)[0] is not None
    else:
        assert not AgentRegistry.contains(adapter_agent_key), (
            f"{family} unexpectedly gained an agent — update _NO_AGENT"
        )


def test_mobile_only_families_miss_on_desktop() -> None:
    """The mobile-only families have NO desktop agent — a desktop lookup
    misses (this is what makes ``make(..., desktop)`` raise KeyError)."""
    for family in MOBILE_ONLY_FAMILIES:
        assert not AgentRegistry.contains(f"{family}@desktop@use"), (
            f"{family} unexpectedly registered a desktop agent"
        )


# =============================================================================
# bootstrap._REGISTRATION_MODULES gate
# =============================================================================
# ``register_all`` imports a HAND-WRITTEN leaf list on purpose, and that design
# stays: registration is a class-declaration side effect, so *someone* must
# import the leaves; the explicit list keeps family package roots
# side-effect-free (importing ``lite.agents`` must not drag in every
# family), keeps the import order visible, and lets the optional block tell
# "litellm is not installed" apart from a real internal ImportError. Deriving
# the list in bootstrap would destroy the optional/required split and the
# order — so the list is GATED here instead.
#
# The expected set is derived from the DIRECTORY, per the same rule as
# ``_discover_families`` above (G7: never derive from the registries the list
# populates, or a dropped registration merely shrinks the expectation). A
# module is a registration leaf iff its SOURCE declares a class with a ``key=``
# keyword — ``class X(Base, key="...")`` is exactly the declaration that runs
# ``__init_subclass__`` and populates a registry.
#
# That AST predicate is deliberately used instead of a fixed
# {action_space, protocol, adapter, agent} name pattern:
#   - families legitimately lack leaves (evocua / qwen2_5_vl / ui_tars_15_v1
#     have no ``protocol.py``; ``lite`` has only ``adapter.py``), and
#   - a family may hold a pure helper that must stay OUT of the list
#     (``models/gpt/utils/`` declares no ``key=``).
# Both fall out of "what is on disk and declares a key", with no name list to
# keep in sync.
#
# Recorded private-surface exception: ``_REGISTRATION_MODULES`` and
# ``_OPTIONAL_REGISTRATION_MODULES`` are the only ``lite.agents.bootstrap``
# privates any test imports, and there is no public equivalent by design. A
# public accessor would either hand back one merged sequence — losing the
# required/optional split that the disjointness gate below asserts — or be a
# second spelling of the same two tuples, which is exactly the aliasing this
# cleanup removes elsewhere. Deletion criterion: drop these two imports once
# ``register_all()`` itself reports what it imported (e.g. returns the required
# and optional dotted paths it walked), because the gates below can then read
# the RESULT of registration instead of reaching for its input table.
AGENTS_DIR = MODELS_DIR.parent
_LEAF_PACKAGES = ("models", "extensions")


def _declares_registry_key(path: Path) -> bool:
    """True if *path* declares ``class X(..., key=...)`` — i.e. it registers."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(
        any(keyword.arg == "key" for keyword in node.keywords)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    )


def _discover_registration_modules() -> set[str]:
    """Dotted paths of every on-disk registration leaf under models/ + extensions/."""
    found = {
        "lite.agents."
        + ".".join(path.relative_to(AGENTS_DIR).with_suffix("").parts)
        for package in _LEAF_PACKAGES
        for path in (AGENTS_DIR / package).rglob("*.py")
        if path.name != "__init__.py" and _declares_registry_key(path)
    }
    # Discovery canary: a rename/move of the family trees must redden here,
    # not silently empty the expected set (which would pass vacuously).
    assert found, f"discovery found no registration leaves under {AGENTS_DIR}"
    return found


def _listed_registration_modules() -> set[str]:
    """Union of the two hand-written bootstrap tables (see the section note above)."""
    return set(_REGISTRATION_MODULES) | set(_OPTIONAL_REGISTRATION_MODULES)


def test_bootstrap_imports_every_registration_leaf() -> None:
    """Every on-disk leaf that declares ``key=`` is imported by ``register_all``.

    Catches the silent-drop direction: add a family (or a leaf to one) and
    forget the bootstrap lines, and its registrations never happen — every
    registry lookup for it misses at rollout time instead of failing here.
    """
    missing = _discover_registration_modules() - _listed_registration_modules()
    assert not missing, (
        "lite/agents/bootstrap.py does not import these registration leaves "
        f"(they declare `key=` but never get imported): {sorted(missing)}"
    )


def test_bootstrap_lists_no_unknown_module() -> None:
    """Every listed module exists on disk and actually registers something.

    Catches the stale direction: a leaf that was deleted, renamed, moved, or
    stripped of its ``key=`` declaration but left behind in the list.
    """
    stale = _listed_registration_modules() - _discover_registration_modules()
    assert not stale, (
        "lite/agents/bootstrap.py lists modules that are not registration "
        f"leaves on disk (deleted, renamed, or no longer declare `key=`): {sorted(stale)}"
    )


def test_bootstrap_lists_are_disjoint_and_duplicate_free() -> None:
    """The required/optional split is a partition — order stays meaningful.

    A module in both tuples would be imported twice (harmless but confusing);
    a module duplicated inside one tuple hides the real import order.
    """
    assert len(set(_REGISTRATION_MODULES)) == len(_REGISTRATION_MODULES)
    assert len(set(_OPTIONAL_REGISTRATION_MODULES)) == len(
        _OPTIONAL_REGISTRATION_MODULES
    )
    overlap = set(_REGISTRATION_MODULES) & set(_OPTIONAL_REGISTRATION_MODULES)
    assert not overlap, f"modules listed as both required and optional: {sorted(overlap)}"


# =============================================================================
# Protocol registry golden
# =============================================================================
# golden: slugs are import-path-independent — exactly the 13 protocol keys.
# Includes the env-bridge protocol browsergym.generic +
# the browsergym goal_image history protocols and the webharbor.webvoyager SoM
# history protocols (one per agent family).
PROTOCOL_GOLDEN = {
    "browsergym.generic",
    "browsergym.goal_image.qwen3_5.history",
    "browsergym.goal_image.qwen3_vl.history",
    "fara.history",
    "lite.history",
    "mai_ui.history",
    "qwen3_5.history",
    "qwen3_vl.history",
    "step_gui.history",
    "ui_tars.history",
    "ui_venus_2.history",
    "webharbor.webvoyager.qwen3_5.history",
    "webharbor.webvoyager.qwen3_vl.history",
}


def test_protocol_registry_golden() -> None:
    """Every golden protocol key is present AND resolves via ``get()``."""
    registered = set(ProtocolRegistry.list())
    missing = PROTOCOL_GOLDEN - registered
    assert not missing, f"ProtocolRegistry lost protocols: {sorted(missing)}"
    # Bridge protocols explicitly (their registration rides on importing the
    # bridges package, the most likely casualty of a rename).
    assert "browsergym.generic" in registered
    for key in sorted(PROTOCOL_GOLDEN):
        assert ProtocolRegistry.get(key) is not None, f"{key} failed to .get()"


# =============================================================================
# Registry slug golden (exact + patterns) for all 4 key= registries
# =============================================================================
# golden: slugs are import-path-independent — the refactor moves MODULE PATHS,
# not registry keys, so the sets below MUST be byte-identical afterward.
# Each was computed from the live registries (sorted(R.list()) /
# sorted(R.list_patterns())).

ACTION_SPACE_LIST = [
    "claude@mobile",
    "gemini@mobile",
    "gpt@mobile",
    "lite@bbox",
    "lite@browser",
    "lite@desktop",
    "lite@mobile",
    "lite@point",
    "mai_ui@mobile",
    "qwen2_5_vl@mobile",
    "qwen2_5_vl@mobile@point",
    "qwen3_5@mobile",
    "qwen3_5@mobile@point",
    "qwen3_8@mobile",
    "qwen3_8@mobile@point",
    "qwen3_vl@mobile",
    "qwen3_vl@mobile@point",
    "step_gui@mobile",
    "ui_tars@mobile",
    "ui_tars@mobile@point",
    "ui_tars_15_v1@mobile",
    "ui_tars_15_v1@mobile@point",
    "ui_venus_2@browser",
    "ui_venus_2@desktop",
    "ui_venus_2@mobile",
]
ACTION_SPACE_PATTERNS = [
    "claude@(desktop|browser)",
    "claude@(desktop|browser)@point",
    "evocua@(desktop|browser)",
    "evocua@(desktop|browser)@point",
    "fara@(desktop|browser)",
    "fara@(desktop|browser)@point",
    "gemini@(desktop|browser)",
    "gpt@(desktop|browser)",
    "gpt@(desktop|browser)@point",
    "mai_ui@(desktop|browser|mobile)@point",
    "qwen2_5_vl@(desktop|browser)",
    "qwen2_5_vl@(desktop|browser)@point",
    "qwen3_5@(desktop|browser)",
    "qwen3_5@(desktop|browser)@point",
    "qwen3_8@(desktop|browser)",
    "qwen3_8@(desktop|browser)@point",
    "qwen3_vl@(desktop|browser)",
    "qwen3_vl@(desktop|browser)@point",
    "ui_tars@(desktop|browser)",
    "ui_tars@(desktop|browser)@point",
    "ui_tars_15_v1@(desktop|browser)",
    "ui_tars_15_v1@(desktop|browser)@point",
    "ui_venus_2@(desktop|browser|mobile)@point",
]

ADAPTER_LIST = [
    "as_is",
    "lite@mobile@grounding.action",
    "lite@mobile@use",
    "mai_ui@mobile@use",
    "qwen2_5_vl@mobile@grounding.action",
    "qwen2_5_vl@mobile@grounding.point",
    "qwen2_5_vl@mobile@use",
    "qwen3_5@mobile@grounding.action",
    "qwen3_5@mobile@grounding.point",
    "qwen3_5@mobile@use",
    "qwen3_8@mobile@grounding.action",
    "qwen3_8@mobile@grounding.point",
    "qwen3_8@mobile@use",
    "qwen3_vl@mobile@grounding.action",
    "qwen3_vl@mobile@grounding.point",
    "qwen3_vl@mobile@use",
    "step_gui@mobile@use",
    "ui_tars@mobile@grounding.action",
    "ui_tars@mobile@grounding.point",
    "ui_tars@mobile@use",
    "ui_tars_15_v1@mobile@grounding.action",
    "ui_tars_15_v1@mobile@grounding.point",
    "ui_tars_15_v1@mobile@use",
    "ui_venus_2@browser@use",
    "ui_venus_2@desktop@use",
    "ui_venus_2@mobile@use",
]
ADAPTER_PATTERNS = [
    "evocua@(desktop|browser)@grounding\\.action",
    "evocua@(desktop|browser)@grounding\\.bbox",
    "evocua@(desktop|browser)@grounding\\.point",
    "evocua@(desktop|browser)@understanding",
    "evocua@(desktop|browser)@use",
    "fara@(desktop|browser)@grounding\\.action",
    "fara@(desktop|browser)@grounding\\.point",
    "fara@(desktop|browser)@use",
    "fara\\.base(@(desktop|browser)@(use|grounding\\.action|grounding\\.point))?",
    "lite@(desktop|browser)@grounding\\.action",
    "lite@(desktop|browser)@use",
    "lite@(desktop|browser|mobile)@grounding\\.bbox",
    "lite@(desktop|browser|mobile)@grounding\\.point",
    "lite@(desktop|browser|mobile)@understanding",
    "mai_ui@(desktop|browser|mobile)@grounding\\.point",
    "qwen2_5_vl@(desktop|browser)@grounding\\.action",
    "qwen2_5_vl@(desktop|browser)@grounding\\.point",
    "qwen2_5_vl@(desktop|browser)@use",
    "qwen2_5_vl@(desktop|browser|mobile)@grounding\\.bbox",
    "qwen2_5_vl@(desktop|browser|mobile)@understanding",
    "qwen2_5_vl\\.base(@(desktop|browser|mobile)@(use|understanding|grounding\\.action|grounding\\.point|grounding\\.bbox))?",
    "qwen3_5@(desktop|browser)@grounding\\.action",
    "qwen3_5@(desktop|browser)@grounding\\.point",
    "qwen3_5@(desktop|browser)@use",
    "qwen3_5@(desktop|browser|mobile)@grounding\\.bbox",
    "qwen3_5@(desktop|browser|mobile)@understanding",
    "qwen3_5\\.base(@(desktop|browser|mobile)@(use|understanding|grounding\\.action|grounding\\.point|grounding\\.bbox))?",
    "qwen3_8@(desktop|browser)@grounding\\.action",
    "qwen3_8@(desktop|browser)@grounding\\.point",
    "qwen3_8@(desktop|browser)@use",
    "qwen3_8@(desktop|browser|mobile)@grounding\\.bbox",
    "qwen3_8@(desktop|browser|mobile)@understanding",
    "qwen3_8\\.base(@(desktop|browser|mobile)@(use|understanding|grounding\\.action|grounding\\.point|grounding\\.bbox))?",
    "qwen3_vl@(desktop|browser)@grounding\\.action",
    "qwen3_vl@(desktop|browser)@grounding\\.point",
    "qwen3_vl@(desktop|browser)@use",
    "qwen3_vl@(desktop|browser|mobile)@grounding\\.bbox",
    "qwen3_vl@(desktop|browser|mobile)@understanding",
    "qwen3_vl\\.base(@(desktop|browser|mobile)@(use|understanding|grounding\\.action|grounding\\.point|grounding\\.bbox))?",
    "ui_tars@(desktop|browser)@grounding\\.action",
    "ui_tars@(desktop|browser)@grounding\\.point",
    "ui_tars@(desktop|browser)@use",
    "ui_tars@(desktop|browser|mobile)@grounding\\.bbox",
    "ui_tars@(desktop|browser|mobile)@understanding",
    "ui_tars_15_v1@(desktop|browser)@grounding\\.action",
    "ui_tars_15_v1@(desktop|browser)@grounding\\.point",
    "ui_tars_15_v1@(desktop|browser)@use",
    "ui_tars_15_v1@(desktop|browser|mobile)@grounding\\.bbox",
    "ui_tars_15_v1@(desktop|browser|mobile)@understanding",
    "ui_venus_2@(desktop|browser|mobile)@grounding\\.point",
]

AGENT_LIST = [
    "claude@mobile@use",
    "gemini@mobile@use",
    "gpt@mobile@use",
    "mai_ui@mobile@use",
    "qwen2_5_vl@mobile@grounding.action",
    "qwen2_5_vl@mobile@grounding.point",
    "qwen2_5_vl@mobile@use",
    "qwen3_5@mobile@grounding.action",
    "qwen3_5@mobile@grounding.point",
    "qwen3_5@mobile@use",
    "qwen3_8@mobile@grounding.action",
    "qwen3_8@mobile@grounding.point",
    "qwen3_8@mobile@use",
    "qwen3_vl@mobile@grounding.action",
    "qwen3_vl@mobile@grounding.point",
    "qwen3_vl@mobile@use",
    "step_gui@mobile@use",
    "ui_tars@mobile@grounding.action",
    "ui_tars@mobile@grounding.point",
    "ui_tars@mobile@use",
    "ui_tars_15_v1@mobile@grounding.action",
    "ui_tars_15_v1@mobile@grounding.point",
    "ui_tars_15_v1@mobile@use",
    "ui_venus_2@browser@use",
    "ui_venus_2@desktop@use",
    "ui_venus_2@mobile@use",
]
AGENT_PATTERNS = [
    "claude@(desktop|browser)@grounding\\.point",
    "claude@(desktop|browser)@use",
    "evocua@(desktop|browser)@grounding\\.action",
    "evocua@(desktop|browser)@grounding\\.point",
    "evocua@(desktop|browser)@use",
    "fara@(desktop|browser)@grounding\\.action",
    "fara@(desktop|browser)@grounding\\.point",
    "fara@(desktop|browser)@use",
    "fara\\.base(@(desktop|browser)@(use|grounding\\.action|grounding\\.point))?",
    "gemini@(desktop|browser)@use",
    "gpt@(desktop|browser)@grounding\\.point",
    "gpt@(desktop|browser)@use",
    "gpt\\.teacher(@(desktop|browser)@use)?",
    "mai_ui@(desktop|browser|mobile)@grounding\\.point",
    "qwen2_5_vl@(desktop|browser)@grounding\\.action",
    "qwen2_5_vl@(desktop|browser)@grounding\\.point",
    "qwen2_5_vl@(desktop|browser)@use",
    "qwen2_5_vl\\.base(@(desktop|browser|mobile)@(use|understanding|grounding\\.action|grounding\\.point|grounding\\.bbox))?",
    "qwen3_5@(desktop|browser)@grounding\\.action",
    "qwen3_5@(desktop|browser)@grounding\\.point",
    "qwen3_5@(desktop|browser)@use",
    "qwen3_5\\.base(@(desktop|browser|mobile)@(use|understanding|grounding\\.action|grounding\\.point|grounding\\.bbox))?",
    "qwen3_8@(desktop|browser)@grounding\\.action",
    "qwen3_8@(desktop|browser)@grounding\\.point",
    "qwen3_8@(desktop|browser)@use",
    "qwen3_8\\.base(@(desktop|browser|mobile)@(use|understanding|grounding\\.action|grounding\\.point|grounding\\.bbox))?",
    "qwen3_vl@(desktop|browser)@grounding\\.action",
    "qwen3_vl@(desktop|browser)@grounding\\.point",
    "qwen3_vl@(desktop|browser)@use",
    "qwen3_vl\\.base(@(desktop|browser|mobile)@(use|understanding|grounding\\.action|grounding\\.point|grounding\\.bbox))?",
    "ui_tars@(desktop|browser)@grounding\\.action",
    "ui_tars@(desktop|browser)@grounding\\.point",
    "ui_tars@(desktop|browser)@use",
    "ui_tars_15_v1@(desktop|browser)@grounding\\.action",
    "ui_tars_15_v1@(desktop|browser)@grounding\\.point",
    "ui_tars_15_v1@(desktop|browser)@use",
    "ui_venus_2@(desktop|browser|mobile)@grounding\\.point",
    "visualwebarena\\.goal_image(@browser@use)?",
]

PROTOCOL_LIST = sorted(PROTOCOL_GOLDEN)
PROTOCOL_PATTERNS: list[str] = []  # golden: ProtocolRegistry registers no patterns

_EXTERNAL_REGISTRY_PATTERNS = {
    AgentAdapterRegistry: {
        r"qwen3_5\.regionfocus@(desktop|browser)@grounding\.point",
        r"qwen3_vl\.regionfocus@(desktop|browser)@grounding\.point",
    },
    AgentRegistry: {
        r"qwen3_5\.regionfocus@(desktop|browser)@grounding\.point",
        r"qwen3_vl\.regionfocus@(desktop|browser)@grounding\.point",
    },
}


def _builtin_patterns(registry) -> list[str]:
    patterns = sorted(registry.list_patterns())
    external = _EXTERNAL_REGISTRY_PATTERNS.get(registry, set())
    if external:
        patterns = [pattern for pattern in patterns if pattern not in external]
    return patterns


@pytest.mark.parametrize(
    "registry, golden_list, golden_patterns",
    [
        (ActionSpaceRegistry, ACTION_SPACE_LIST, ACTION_SPACE_PATTERNS),
        (AgentAdapterRegistry, ADAPTER_LIST, ADAPTER_PATTERNS),
        (AgentRegistry, AGENT_LIST, AGENT_PATTERNS),
        (ProtocolRegistry, PROTOCOL_LIST, PROTOCOL_PATTERNS),
    ],
    ids=["action_space", "adapter", "agent", "protocol"],
)
def test_registry_slug_golden(registry, golden_list, golden_patterns) -> None:
    """Freeze the exact + pattern slug surface of each ``key=`` registry."""
    assert sorted(registry.list()) == golden_list
    assert _builtin_patterns(registry) == golden_patterns


def test_gpt_claude_action_space_grounding_key_axis_is_point_format() -> None:
    """Action-space format keys stay separate from agent/task keys."""
    action_space_keys = set(ActionSpaceRegistry.list()) | set(
        ActionSpaceRegistry.list_patterns()
    )
    adapter_agent_keys = (
        set(AgentAdapterRegistry.list())
        | set(AgentAdapterRegistry.list_patterns())
        | set(AgentRegistry.list())
        | set(AgentRegistry.list_patterns())
    )

    for family in ("gpt", "claude"):
        assert f"{family}@(desktop|browser)@point" in action_space_keys
        assert f"{family}@(desktop|browser)@grounding\\.point" not in action_space_keys
        assert f"{family}@(desktop|browser)@grounding\\.point" in adapter_agent_keys


# =============================================================================
# Gym discoverable namespace golden
# =============================================================================
# ``env_ids()`` is the cheap directory-scan surface. Multi-variant families such
# as ``lite.cuaworld`` appear here as namespaces; use ``registered_env_ids()``
# for concrete makeable IDs such as ``lite.cuaworld.pymol``.
ENV_IDS_GOLDEN = {
    "androidlab",
    "androidworld",
    "browsergym",
    "captcha",
    "cua.bench",
    "lite.demo",
    "lite.cuagym",
    "lite.cuaworld",
    "lite.osworld",
    "lite.scalecua",
    "mobileworld",
    "online_mind2web",
    "mobilegym",
    "osworld",
    "osworld_2",
    "osworld_g",
    "screenspot_pro",
    "waa",
    "webgym",
    "webharbor.webvoyager",
}


def test_env_ids_golden() -> None:
    assert set(gym.registry.env_ids()) == ENV_IDS_GOLDEN


# =============================================================================
# Import surface smoke
# =============================================================================
# Bare-import each family submodule + the package re-export points. A broken
# re-export after a rename (e.g. a missing shim) fails loudly here, BEFORE the
# registry-content tests above (which would mask the cause behind a missing key).
# claude / gpt are gated on litellm (optional); skip if it isn't installed.
_OPTIONAL_FAMILIES = ("claude", "gpt")
_FAMILY_MODULES = [
    f"lite.agents.models.{fam}"
    for fam in ALL_FAMILIES
    if fam not in _OPTIONAL_FAMILIES
]
_OPTIONAL_FAMILY_MODULES = [f"lite.agents.models.{fam}" for fam in _OPTIONAL_FAMILIES]

_PACKAGE_SURFACE = [
    "lite.agents.models",
    "lite.agents.extensions",
    "lite.agents.core.action_space",
    "lite.core",
    "lite.gym",
]

_PACKAGE_ROOT_INIT_FILES = [
    "lite/agents/extensions/__init__.py",
    "lite/agents/extensions/browsergym/__init__.py",
    "lite/agents/extensions/teacher/__init__.py",
    "lite/agents/extensions/webharbor/__init__.py",
    "lite/agents/extensions/webharbor/webvoyager/__init__.py",
    "lite/agents/models/claude/__init__.py",
    "lite/agents/models/evocua/__init__.py",
    "lite/agents/models/fara/__init__.py",
    "lite/agents/models/gemini/__init__.py",
    "lite/agents/models/gpt/__init__.py",
    "lite/agents/models/lite/__init__.py",
    "lite/agents/models/mai_ui/__init__.py",
    "lite/agents/models/qwen2_5_vl/__init__.py",
    "lite/agents/models/qwen3_5/__init__.py",
    "lite/agents/models/qwen3_vl/__init__.py",
    "lite/agents/models/step_gui/__init__.py",
    "lite/agents/models/ui_tars/__init__.py",
    "lite/agents/models/ui_tars_15_v1/__init__.py",
    "lite/agents/models/ui_venus_2/__init__.py",
]


@pytest.mark.parametrize("module", _FAMILY_MODULES + _PACKAGE_SURFACE)
def test_import_surface(module: str) -> None:
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize("module", _OPTIONAL_FAMILY_MODULES)
def test_import_surface_optional(module: str) -> None:
    pytest.importorskip("litellm")
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize("path", _PACKAGE_ROOT_INIT_FILES)
def test_family_package_roots_do_not_import_registration_leaves(path: str) -> None:
    """Built-in registration belongs to ``lite.agents.bootstrap.register_all``."""
    source_path = Path(__file__).parents[2] / path
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    offenders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        )
    ]
    assert not offenders, f"{path} imports registration leaves"
