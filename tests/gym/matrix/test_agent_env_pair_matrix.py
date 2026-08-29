"""The (agent × env) CONFIG MATRIX — every shipped yaml pair, by construction.

WHY THIS FILE EXISTS
--------------------
``tests/gym/metadata/test_metadata_invariant.py`` gates the *source yaml strings*
(which rows name ``response`` / ``terminate`` / ``open_app`` / nav) against
frozen expectation sets, and
``tests/agents/core/action_space/test_valid_actions_gating.py`` gates the
*adapter* against hand-built :class:`LiteCUAMetadata`. Neither one ever
joins the two ends: no test took a shipped yaml, let the REAL env resolve its
``env_kwargs`` into ``LiteCUAMetadata``, and then built the REAL agent named by
that yaml's ``agent_id``. A yaml that merely parses is not coverage — shipped
configs have parsed fine and still raised ``KeyError`` the moment
``lite.agents.factory.make`` was called. Rows that cannot resolve are pinned in
``_KNOWN_UNRESOLVABLE_PAIRS`` (currently empty); everything else rides the
normal sweep.

Everything here is **table-driven off the yaml tree**, so a new
``scripts/configs/<agent>/<variant>/<env>.yaml`` is covered the day it lands —
no per-pair test to remember to add.

Layers, weakest to strongest:
  1. registry containment — dep-light, covers ALL rows (envs whose deps are
     missing fall back to pinned routing ``dims``, and the pins
     self-verify against the live registry wherever the env IS importable).
  2. protocol_key / adapter_key resolution — the ``agents/extensions`` rows
     (browsergym.generic / webharbor SoM / goal_image).
  3. live construction — the row's ``env_kwargs`` through the REAL env
     (``spec.entry_point`` + ``_runtime_metadata``, i.e. ``gym.make`` minus the
     container-booting ``ensure_services`` step) → ``factory.make(model_id,
     env=…, agent_id=…)``. Skips rows whose env deps are genuinely absent, with
     a floor so a mass-skip fails loudly. Most rows resolve on a
     mid-provisioned dev host; the skipped ones are browsergym / cua-bench /
     lite.cuagym, whose env MODULE cannot import without their packages.
  4. boundary conditions — ``extra_tools`` / ``valid_actions`` edge values,
     and the single-step-grounding terminal-channel exemption (proved by
     stepping the env, not by reading it).

Run:
    uv run pytest tests/gym/matrix/test_agent_env_pair_matrix.py -p no:cacheprovider -q
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

import lite.gym as gym
from lite.agents.bootstrap import register_all
from lite.agents.core.adapter.base import AgentAdapterRegistry
from lite.agents.core.agent.base import AgentRegistry, BaseAgent
from lite.agents.core.protocol.base import ProtocolRegistry
from lite.agents.factory import AGENTS, LOCAL_AGENTS, make
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
)
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name
from lite.gym.errors import EnvDepsMissingError
from lite.gym.registry import _specs, registry
from lite.gym.utils.config.defaults import finalize_env_kwargs
from lite.utils.path import project_root
from lite.utils.registry import compose_key

register_all()

# ``lite.gym.registry`` the MODULE (the package re-exports the singleton under
# the same attribute name, so ``lite.gym.registry`` on the package is the
# Registry instance). The module holds the private make-path helpers this file
# reuses to resolve metadata without booting anything.
_registry_module = importlib.import_module("lite.gym.registry")


@pytest.fixture(autouse=True)
def _direct_mode(monkeypatch):
    """Registry/make probes only mean anything in direct mode."""
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)


# ---------------------------------------------------------------------------
# The matrix, enumerated from the yaml tree
# ---------------------------------------------------------------------------

_CONFIG_ROOTS = (Path("scripts/configs"),)

# Floors, not equalities: these are yaml COUNTS, which grow as benches land.
# A floor catches the failure this file guards against — the enumeration
# silently going empty (bad glob / moved directory) and every assertion below
# vacuously passing.
_MIN_CONFIG_YAMLS = 150
_MIN_DISTINCT_PAIRS = 110
_EXPECTED_AGENT_CONFIG_ROWS = 196
_EXPECTED_ENV_CONFIG_ROWS = 191
_EXPECTED_DISTINCT_PAIRS = 153


class ConfigRow:
    """One agent-config yaml, flattened to what the matrix cares about."""

    __slots__ = ("rel", "agent_id", "env_id", "agent_kwargs", "env_kwargs")

    def __init__(self, rel: str, data: dict):
        self.rel = rel
        self.agent_id = data.get("agent_id")
        self.env_id = data.get("env_id")
        self.agent_kwargs = data.get("agent_kwargs") or {}
        self.env_kwargs = data.get("env_kwargs") or {}

    @property
    def pair(self) -> tuple[str, str]:
        return (self.agent_id, self.env_id)

    def __repr__(self) -> str:  # parametrize ids
        return self.rel


def _config_rows() -> list[ConfigRow]:
    """Every agent-config yaml under the config roots (mapping-shaped only)."""
    root = project_root()
    rows: list[ConfigRow] = []
    for rel_root in _CONFIG_ROOTS:
        base = root / rel_root
        assert base.exists(), f"missing agent config root: {rel_root}"
        for path in sorted(base.rglob("*.yaml")):
            posix = path.as_posix()
            if "build/lib" in posix or "/.tmp/" in posix:
                continue
            data = yaml.safe_load(path.read_text()) or {}
            if not isinstance(data, dict) or "agent_id" not in data:
                continue
            rows.append(ConfigRow(path.relative_to(root).as_posix(), data))
    return rows


ROWS = _config_rows()
# Rows that name an env — the (agent × env) matrix proper. The remainder are
# training recipes (SFT/GRPO), which carry an agent_id but no env_id.
PAIR_ROWS = [r for r in ROWS if r.env_id]


# ---------------------------------------------------------------------------
# Env dimension table
# ---------------------------------------------------------------------------
# ``lite.agents.factory.make`` composes the agent key from the ENV's
# ``metadata.dims``. Most envs answer that from
# the registry with no docker image and no VM; the few whose *module import*
# is hard-gated on an uninstalled third-party package (browsergym, cua-bench)
# or a provisioned task cache (lite.cuagym) are pinned here so the containment
# sweep still covers their rows on a partially-provisioned host.
#
# The pins are NOT trusted blindly: ``test_pinned_env_dims_match_live_registry``
# re-derives every pin that IS importable and fails on drift, so a fully
# provisioned host verifies them automatically.
_PINNED_ENV_DIMS: dict[str, tuple[str, ...]] = {
    "browsergym.miniwob": ("browser", "use"),
    "browsergym.visualwebarena": ("browser", "use"),
    "browsergym.webarena": ("browser", "use"),
    "cua.bench.local.basic": ("desktop", "use"),
    "cua.bench.local.kicad": ("desktop", "use"),
    "cua.bench.local.workflows": ("desktop", "use"),
    "lite.cuagym": ("browser", "use"),
}

# Envs that end the episode on the FIRST step (single-step grounding). They are
# exempt from the terminal-channel expectations that multi-step ``use`` envs
# carry — proved by execution in
# ``test_single_step_grounding_envs_terminate_on_first_step``.
_SINGLE_STEP_GROUNDING_ENVS = {"osworld_g", "screenspot_pro"}


def _live_env_dims(env_id: str) -> tuple[str, ...] | None:
    """Routing dims from the registry, or None when unavailable."""
    try:
        task_ids = registry.task_ids(env_id)
        first = next((v[0] for v in task_ids.values() if v), None)
        if first is None:
            return None
        md = registry.task_metadata(env_id, first)
    except (EnvDepsMissingError, KeyError):
        return None
    assert md is not None, f"{env_id}@{first} has no registered metadata"
    return tuple(md.dims)


_ENV_DIMS_CACHE: dict[str, tuple[str, ...]] = {}


def _env_dims(env_id: str) -> tuple[str, ...]:
    if env_id not in _ENV_DIMS_CACHE:
        dims = _live_env_dims(env_id)
        if dims is None:
            assert env_id in _PINNED_ENV_DIMS, (
                f"env {env_id!r} is neither importable on this host nor pinned in "
                "_PINNED_ENV_DIMS — add a routing dims pin so its config "
                "rows stay covered"
            )
            dims = _PINNED_ENV_DIMS[env_id]
        _ENV_DIMS_CACHE[env_id] = dims
    return _ENV_DIMS_CACHE[env_id]


# ---------------------------------------------------------------------------
# Known holes — NEGATIVE pins, so a fix forces this list to shrink
# ---------------------------------------------------------------------------
# Listed as an EXPECTED-to-fail set rather than skipped: the tests below assert
# these still fail, so whoever fixes one is forced to delete the entry.
#
# CURRENTLY EMPTY: every shipped row resolves, so all of them are covered as
# ordinary members of the live-construction sweep below. A row belongs here only
# while its family lacks a concrete agent + adapter + action space for the
# dims its env resolves to.
_KNOWN_UNRESOLVABLE_PAIRS: set[tuple[str, str]] = set()

# ---------------------------------------------------------------------------
# agent_id → a representative model_id from the factory catalog
# ---------------------------------------------------------------------------
# A yaml names ``agent_id`` (the registry family) but never a checkpoint; the
# checkpoint comes from ``rollout.py --model``. For construction we need SOME
# catalog entry whose family matches. ``.base`` / ``.passthrough`` / ``.teacher``
# / ``.regionfocus`` modifiers bind to the name, so strip at the first dot.
_FAMILY_TO_MODEL: dict[str, str] = {}
for _model_id, _cfg in AGENTS.items():
    _FAMILY_TO_MODEL.setdefault(_cfg["agent_id"], _model_id)

# ``visualwebarena.goal_image`` is a model-AGNOSTIC bridge agent: it wraps
# whatever ``agent_kwargs.adapter_key`` names. Any catalog model constructs it.
_BRIDGE_AGENT_FAMILIES = {"visualwebarena"}

# agent_ids a config names that no ``AGENTS`` checkpoint routes to.
# Every entry is a release-visible quarantine: keep the owner/date/reason here
# so the gap is explicit instead of becoming an ambiguous V16 failure.
#
#   * ``lite`` — BY DESIGN. The cua-lite reference DIALECT has no Agent class at
#     all (see ``_NO_AGENT`` in tests/agents/test_registration_complete.py); its
#     single config is an SFT recipe (``scripts/configs/lite/recipes/sft/``),
#     which trains on saved trajectories and never constructs an agent.
#     Owner: config/catalog owner. Date: 2026-08-02.
#
# Pinned so adding the catalog entry (or dropping the configs) forces an update.
_AGENT_IDS_WITHOUT_CATALOG_MODEL = {"lite"}


def _model_for(row: ConfigRow) -> str | None:
    family = row.agent_id.split(".")[0]
    if family in _FAMILY_TO_MODEL:
        return _FAMILY_TO_MODEL[family]
    if family in _BRIDGE_AGENT_FAMILIES:
        # Pick the family named by the bridge's adapter_key.
        adapter_key = (row.agent_kwargs or {}).get("adapter_key") or ""
        inner = adapter_key.split("@")[0].split(".")[0]
        return _FAMILY_TO_MODEL.get(inner)
    return None


# ---------------------------------------------------------------------------
# 0. Enumeration floors
# ---------------------------------------------------------------------------


def test_config_matrix_enumeration_floor() -> None:
    """The table-driven sweeps below are only as good as this enumeration."""
    assert len(ROWS) >= _MIN_CONFIG_YAMLS, (
        f"only {len(ROWS)} agent-config yamls found under {_CONFIG_ROOTS} — "
        "the glob broke, or the config tree moved"
    )
    pairs = {r.pair for r in PAIR_ROWS}
    assert len(pairs) >= _MIN_DISTINCT_PAIRS, (
        f"only {len(pairs)} distinct (agent_id, env_id) pairs enumerated"
    )
    # Every row names an agent; rows without an env are training recipes only.
    assert all(r.agent_id for r in ROWS)
    envless = sorted(r.rel for r in ROWS if not r.env_id)
    assert all("recipes/sft" in rel for rel in envless), (
        f"an env-less config that is not an SFT recipe appeared: {envless}"
    )


def test_config_matrix_exact_count_report_is_current() -> None:
    """Freeze the exact report used by the release checklist."""
    assert (
        len(ROWS),
        len(PAIR_ROWS),
        len({r.pair for r in PAIR_ROWS}),
    ) == (
        _EXPECTED_AGENT_CONFIG_ROWS,
        _EXPECTED_ENV_CONFIG_ROWS,
        _EXPECTED_DISTINCT_PAIRS,
    )


def test_pinned_env_dims_match_live_registry() -> None:
    """Every ``_PINNED_ENV_DIMS`` entry that IS importable on this host must
    agree with the registry — so the pins can never silently rot into a wrong
    agent key on a fully-provisioned machine."""
    checked = 0
    for env_id, pinned in sorted(_PINNED_ENV_DIMS.items()):
        live = _live_env_dims(env_id)
        if live is None:
            continue
        checked += 1
        assert live == pinned, (
            f"_PINNED_ENV_DIMS[{env_id!r}] = {pinned} but the live registry says "
            f"{live} — update the pin"
        )
    # A floor is not assertable here: an env is pinned PRECISELY because its
    # deps may be absent, so on a bare host `checked == 0` is the correct
    # outcome, not a defect. But "ran and verified nothing" must not report as
    # a pass — skip, so the run says so out loud instead of going green on an
    # `assert checked >= 0` that no state can falsify.
    if checked == 0:
        pytest.skip(
            f"none of the {len(_PINNED_ENV_DIMS)} pinned envs is importable on "
            "this host — the pins are unverified here (install env deps, or run "
            "this on a provisioned host, to give this test teeth)"
        )


def test_pinned_env_dims_carries_no_dead_entries() -> None:
    """A pin for an env no config references is dead weight."""
    referenced = {r.env_id for r in PAIR_ROWS}
    dead = sorted(set(_PINNED_ENV_DIMS) - referenced)
    assert not dead, f"_PINNED_ENV_DIMS pins envs no config uses: {dead}"


# ---------------------------------------------------------------------------
# 1. Registry containment — every pair resolves an Agent class
# ---------------------------------------------------------------------------


def test_every_config_pair_resolves_an_agent_class() -> None:
    """For every shipped yaml, the key ``factory.make`` will compose from that
    env's metadata must resolve in ``AgentRegistry``.

    This is the cheap, dep-light half of the matrix: pure key composition +
    ``contains``, no env boot, no processor. It covers rows whose env deps are
    absent (browsergym / cua-bench) via ``_PINNED_ENV_DIMS``.
    """
    unresolvable: list[str] = []
    for row in PAIR_ROWS:
        dims = _env_dims(row.env_id)
        key = compose_key(row.agent_id, *dims)
        if not AgentRegistry.contains(key):
            unresolvable.append(f"{row.rel}: {key}")

    expected = sorted(
        f"{row.rel}: {compose_key(row.agent_id, *_env_dims(row.env_id))}"
        for row in PAIR_ROWS
        if row.pair in _KNOWN_UNRESOLVABLE_PAIRS
    )
    # Only meaningful while the pin is non-empty: an entry that matches no
    # shipped config row is stale and must be deleted. An EMPTY pin (the
    # current state — every known hole is fixed) is the goal, not a failure.
    assert expected or not _KNOWN_UNRESOLVABLE_PAIRS, (
        "_KNOWN_UNRESOLVABLE_PAIRS matched no config row — stale entries?"
    )

    new = sorted(set(unresolvable) - set(expected))
    assert not new, "config rows whose (agent_id, env) pair resolves NO agent class:\n" + "\n".join(
        new
    )
    fixed = sorted(set(expected) - set(unresolvable))
    assert not fixed, (
        "these rows now resolve — delete them from _KNOWN_UNRESOLVABLE_PAIRS:\n" + "\n".join(fixed)
    )


def test_every_config_agent_id_has_a_catalog_model() -> None:
    """Every ``agent_id`` a config names must be reachable from some
    ``AGENTS`` checkpoint — otherwise ``rollout.py --model`` cannot run it."""
    missing = sorted({r.agent_id for r in ROWS if _model_for(r) is None})
    assert missing == sorted(_AGENT_IDS_WITHOUT_CATALOG_MODEL), (
        f"agent_ids with no catalog model changed: got {missing}, "
        f"pinned {sorted(_AGENT_IDS_WITHOUT_CATALOG_MODEL)}"
    )


# ---------------------------------------------------------------------------
# 2. Extension plumbing — protocol_key / adapter_key
# ---------------------------------------------------------------------------


def test_every_config_protocol_key_resolves() -> None:
    """``agent_kwargs.protocol_key`` is the env-bridge seam (browsergym.generic,
    webharbor.webvoyager.*). A rename that drops a registration
    would otherwise only surface at rollout time."""
    offenders = []
    seen = 0
    for row in ROWS:
        key = (row.agent_kwargs or {}).get("protocol_key")
        if not key:
            continue
        seen += 1
        if not ProtocolRegistry.contains(key):
            offenders.append(f"{row.rel}: protocol_key={key!r}")
    assert not offenders, "unresolvable protocol_key:\n" + "\n".join(offenders)
    assert seen >= 15, f"only {seen} rows carry a protocol_key — enumeration broke"


def test_every_config_adapter_key_resolves() -> None:
    """``agent_kwargs.adapter_key`` is the goal_image bridge's pointer at the
    real model adapter. A bare slug auto-completes with the ENV's dims exactly
    the way ``VisualWebArenaGoalImageAgent.__post_init__`` does."""
    offenders = []
    seen = 0
    for row in PAIR_ROWS:
        key = (row.agent_kwargs or {}).get("adapter_key")
        if not key:
            continue
        seen += 1
        dims = _env_dims(row.env_id)
        full = key if "@" in key else compose_key(key, *dims)
        if not AgentAdapterRegistry.contains(full):
            offenders.append(f"{row.rel}: adapter_key={key!r} → {full!r}")
    assert not offenders, "unresolvable adapter_key:\n" + "\n".join(offenders)
    assert seen >= 5, f"only {seen} rows carry an adapter_key — enumeration broke"


# ---------------------------------------------------------------------------
# 3. Live construction — the real env, the real agent
# ---------------------------------------------------------------------------

# How many rows must actually construct before the sweep is meaningful. On a
# host with no env provisioned at all the sweep would otherwise skip its way to
# a green run. Set below what a mid-provisioned dev host reaches (16 of the 24
# envs at the time of writing) so the floor tracks a REGRESSION, not the host.
_MIN_LIVE_CONSTRUCTED_ROWS = 80


@pytest.fixture
def _ignore_image_staleness(monkeypatch):
    """Neutralize the docker-image FRESHNESS gate for metadata-only probes.

    ``ContainerImage.ensure_runnable`` refuses construction when an image's
    ``lite.src_hash`` label lags its build sources. That gate protects RUNNING
    a stale container; it says nothing about whether an env turns its
    ``env_kwargs`` into the right ``LiteCUAMetadata``. Nothing here resets or
    steps, so no container is ever launched (see ``_resolve_metadata``, which
    deliberately bypasses ``ensure_services`` — the layer that DOES boot the
    shared-backend containers). Without the patch, a dev branch that edited any
    env source silently drops most of the matrix from the sweep.
    """
    from lite.gym.utils.backend.freshness import ContainerImage

    monkeypatch.setattr(ContainerImage, "ensure_runnable", lambda self: None)


def _resolve_metadata(row: ConfigRow):
    """The row's resolved Lite metadata, or None when its env deps are missing.

    Mirrors ``lite.gym.factory.make``'s config chain
    (``env_make_kwargs`` < ``spec.kwargs`` < caller kwargs, minus the
    framework-only wrapper/identity/routing keys) and then calls the spec's
    ``entry_point`` directly instead of ``gym.make``.

    WHY NOT ``gym.make``: it first runs ``registry.ensure_services(env_id)``,
    which for the shared-backend envs (webgym / mobilegym / online_mind2web /
    webharbor) BOOTS a docker container. A metadata sweep over ~170 configs
    must not spawn containers. The substitution is safe because the wrapper
    layer ``gym.make`` adds is metadata-transparent — pinned independently by
    wrapper passthrough and registered-vs-constructed parity in
    ``tests/gym/metadata/test_metadata_invariant.py`` — and spot-checked here by
    ``test_entry_point_metadata_matches_gym_make``.
    """
    try:
        task_ids = registry.task_ids(row.env_id)
    except (EnvDepsMissingError, KeyError):
        return None
    first = next((v[0] for v in task_ids.values() if v), None)
    if first is None:
        return None
    spec = _specs[f"{row.env_id}@{first}"]
    merged = {
        **registry.env_make_kwargs(row.env_id),
        **spec.kwargs,
        **finalize_env_kwargs(row.env_kwargs),
    }
    for key in _registry_module._WRAPPER_KWARG_DEFAULTS:
        merged.pop(key, None)
    for key in ("session_id", "token_hash", "server_port", "env_server_url"):
        merged.pop(key, None)
    try:
        env = spec.entry_point(**merged)
    except EnvDepsMissingError:
        return None
    return env._runtime_metadata()


def _build_agent(row: ConfigRow, metadata) -> BaseAgent:
    """Construct exactly the way ``lite.infer.rollout.run_rollout`` does.

    ``factory.make`` reads nothing off ``env`` but ``env.metadata``, so a
    namespace carrying the REAL resolved metadata is the whole contract
    (same shim as ``tests/agents/test_make_all.py``).
    """
    model_id = _model_for(row)
    kwargs: dict = {}
    if model_id in LOCAL_AGENTS:
        # Never called at construction — mirrors tests/agents/test_make_all.py.
        kwargs = {"processor": MagicMock(), "generate_fn": (lambda *a, **k: None)}
    agent_kwargs = dict(row.agent_kwargs)
    # rollout.py consumes sampling_kwargs itself; it is not an agent field.
    agent_kwargs.pop("sampling_kwargs", None)
    env = SimpleNamespace(metadata=metadata)
    return make(model_id, env=env, agent_id=row.agent_id, **kwargs, **agent_kwargs)


@pytest.mark.parametrize("env_id", ["osworld_g", "screenspot_pro", "lite.demo"])
def test_entry_point_metadata_matches_gym_make(env_id, _ignore_image_staleness) -> None:
    """Justify the ``_resolve_metadata`` substitution on the envs where
    ``gym.make`` is free (no container / no VM): the two paths must agree."""
    try:
        task_ids = registry.task_ids(env_id)
    except EnvDepsMissingError as exc:
        pytest.skip(f"{env_id} unavailable: {exc}")
    first = next(v[0] for v in task_ids.values() if v)
    row = ConfigRow(f"<synthetic>/{env_id}.yaml", {"agent_id": "x", "env_id": env_id})
    via_entry_point = _resolve_metadata(row)
    made = gym.make(f"{env_id}@{first}")
    try:
        assert via_entry_point.valid_actions == made.metadata.valid_actions
        assert via_entry_point.extra_tool_schemas == made.metadata.extra_tool_schemas
        assert tuple(via_entry_point.dims) == tuple(made.metadata.dims)
    finally:
        asyncio.run(made.close())


def test_every_config_pair_constructs_a_live_agent(_ignore_image_staleness) -> None:
    """END TO END over the matrix: real env metadata → real agent.

    This is the layer that no existing test covered. It is what catches an
    ``agent_kwargs`` key the agent class does not accept, an ``env_kwargs``
    value the env rejects, and routing dims the agent family never
    registered — the whole class of "the yaml parses, the run dies".
    """
    failures: list[str] = []
    constructed = 0
    skipped: list[str] = []

    for row in PAIR_ROWS:
        if row.pair in _KNOWN_UNRESOLVABLE_PAIRS:
            continue
        if _model_for(row) is None:
            continue  # pinned by test_every_config_agent_id_has_a_catalog_model
        try:
            metadata = _resolve_metadata(row)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append(f"{row.rel}: env build raised {type(exc).__name__}: {exc}")
            continue
        if metadata is None:
            skipped.append(row.rel)
            continue
        try:
            agent = _build_agent(row, metadata)
            assert isinstance(agent, BaseAgent), type(agent).__name__
            constructed += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{row.rel}: agent build raised {type(exc).__name__}: {exc}")

    assert not failures, "config rows that do not construct end to end:\n" + "\n".join(failures)
    assert constructed >= _MIN_LIVE_CONSTRUCTED_ROWS, (
        f"only {constructed} of {len(PAIR_ROWS)} rows constructed "
        f"({len(skipped)} skipped for missing env deps) — below the floor of "
        f"{_MIN_LIVE_CONSTRUCTED_ROWS}. Either an env regressed, or this host "
        "lost provisioning it used to have."
    )


def test_live_metadata_tool_surface_is_yaml_attributable(_ignore_image_staleness) -> None:
    """Every name a row puts in ``env_kwargs.extra_tools`` must come back out
    of the resolved ``metadata.extra_tool_schemas`` — the env may ADD (its
    own defaults) but must never silently DROP a requested tool.

    Any env that violates this would be pinned in
    ``_ENVS_THAT_IGNORE_TOOL_SURFACE_ENV_KWARGS`` (see the boundary section);
    that set is currently empty.
    """
    offenders: list[str] = []
    checked = 0
    for row in PAIR_ROWS:
        requested = list(row.env_kwargs.get("extra_tools") or [])
        if not requested:
            continue
        metadata = _resolve_metadata(row)
        if metadata is None:
            continue
        checked += 1
        got = {tool_schema_name(s) for s in (metadata.extra_tool_schemas or [])}
        dropped = sorted(set(requested) - got)
        if dropped:
            offenders.append(f"{row.rel}: requested {requested}, dropped {dropped}")
    assert not offenders, "env silently dropped a yaml-requested extra tool:\n" + "\n".join(
        offenders
    )
    assert checked >= 40, f"only {checked} rows with extra_tools resolved"


# ---------------------------------------------------------------------------
# 4. Boundary conditions
# ---------------------------------------------------------------------------

# The canonical GUI-inner vocabulary a source yaml may name in
# ``env_kwargs.valid_actions``. This is a STATIC yaml lint (it cannot know
# which env a row points at, so it accepts the union of every platform's
# actions plus ``point``); the per-env runtime check is the authority and
# lives in ``lite/gym/utils/feedback/surface.py:resolve_valid_actions``, which raises on
# any name outside the TARGET env's own surface. Both layers exist because a
# yaml row can be wrong in two ways: a name no env would accept (caught here,
# at lint time, across all 176 rows) and a name the specific env doesn't have
# (caught at construction). See tests/gym/utils/feedback/test_valid_actions_contract.py.
_ALLOWED_YAML_VALID_ACTIONS = (
    LiteDesktopActionSet.get_action_names()
    | LiteMobileActionSet.get_action_names()
    | LitePointActionSet.get_action_names()
)


def test_config_valid_actions_are_known_actions() -> None:
    """Whitelist source-yaml ``valid_actions`` values for typo protection."""
    offenders: list[str] = []
    for row in ROWS:
        value = row.env_kwargs.get("valid_actions")
        if value is None:
            continue
        assert isinstance(value, list), f"{row.rel}: valid_actions must be list|null"
        unknown = sorted(set(value) - _ALLOWED_YAML_VALID_ACTIONS)
        if unknown:
            offenders.append(f"{row.rel}: unknown GUI actions {unknown}")
    assert not offenders, (
        "valid_actions may only name GUI actions; standalone tools (finish / "
        "nav / open_app / bash) belong in env_kwargs.extra_tools:\n" + "\n".join(offenders)
    )


def test_unknown_valid_actions_raise_instead_of_being_dropped() -> None:
    """REGRESSION for the footgun this file used to merely characterize.

    The old desktop-only filter silently kept only ``LiteDesktopActionSet.get_action_names()``
    and dropped the rest without a word, so a ``"clik"`` typo quietly shrank the
    agent's action enum and ``["terminate"]`` collapsed to ``[]`` — which does
    not mean "no filtering" but DELETES the action-batch tool entirely. Both are
    now hard config-boundary errors.
    """
    from lite.gym.utils.feedback.surface import resolve_valid_actions

    assert resolve_valid_actions(None, env_name="x", platform="desktop") is None
    # A deliberate empty list is still honored verbatim (bid-only rows need it).
    assert resolve_valid_actions([], env_name="x", platform="desktop") == []
    # A typo no longer vanishes.
    with pytest.raises(ValueError, match="clik"):
        resolve_valid_actions(["clik", "type"], env_name="x", platform="desktop")
    # A finish tool name no longer collapses the whole GUI surface to empty.
    with pytest.raises(ValueError, match="terminate"):
        resolve_valid_actions(["terminate"], env_name="x", platform="desktop")


# Envs whose ``env_kwargs`` are validated: an unknown ``extra_tools`` name must
# raise rather than resolve to something the agent then cannot use. Restricted
# to envs constructible without a container/VM boot.
_TOOL_SURFACE_VALIDATING_ENVS = (
    "lite.osworld",
    "lite.demo",
    "lite.scalecua",
    "osworld",
    "osworld_g",
    "screenspot_pro",
    "waa",
    "androidworld",
    "androidlab",
    "mobileworld",
    "webgym",
    "webharbor.webvoyager",
    "online_mind2web",
)

# CURRENTLY EMPTY. Envs that accept ``extra_tools`` / ``valid_actions`` must
# resolve them through the shared helpers and amend ``_runtime_metadata``; those
# envs belong in ``_TOOL_SURFACE_VALIDATING_ENVS`` instead of this denylist.
_ENVS_THAT_IGNORE_TOOL_SURFACE_ENV_KWARGS: tuple[str, ...] = ()


def _construct_raw_env(env_id: str, **env_kwargs):
    """The row-free sibling of :func:`_resolve_metadata` — build the raw env
    through its registered ``entry_point``.

    Same reason for skipping ``gym.make``: its ``ensure_services`` step BOOTS a
    shared-backend container (webgym / mobilegym / online_mind2web /
    webharbor), and it runs BEFORE the env ctor, so even a probe that is
    supposed to raise would leak one.
    """
    task_ids = registry.task_ids(env_id)
    first = next(v[0] for v in task_ids.values() if v)
    spec = _specs[f"{env_id}@{first}"]
    merged = {
        **registry.env_make_kwargs(env_id),
        **spec.kwargs,
        **finalize_env_kwargs(env_kwargs),
    }
    for key in _registry_module._WRAPPER_KWARG_DEFAULTS:
        merged.pop(key, None)
    return spec.entry_point(**merged)


@pytest.mark.parametrize("env_id", _TOOL_SURFACE_VALIDATING_ENVS)
def test_unknown_extra_tool_name_is_rejected(env_id, _ignore_image_staleness) -> None:
    """``extra_tools`` naming a tool the env cannot execute must fail LOUDLY at
    ``gym.make`` — never resolve to an empty/partial surface the agent then
    advertises to the model."""
    try:
        registry.task_ids(env_id)
    except EnvDepsMissingError as exc:
        pytest.skip(f"{env_id} unavailable: {exc}")
    with pytest.raises(ValueError):
        _construct_raw_env(env_id, extra_tools=["definitely_not_a_tool"])


def test_no_config_row_sets_tool_surface_on_an_ignoring_env() -> None:
    """Make the silent-drop unreachable from configuration.

    Until the envs in ``_ENVS_THAT_IGNORE_TOOL_SURFACE_ENV_KWARGS`` validate
    their tool-surface env_kwargs, no shipped yaml may set them there — the
    author would get silence instead of the surface they asked for.
    """
    offenders = [
        f"{row.rel}: env_kwargs.{key}"
        for row in PAIR_ROWS
        if row.env_id in _ENVS_THAT_IGNORE_TOOL_SURFACE_ENV_KWARGS
        for key in ("extra_tools", "valid_actions")
        if key in row.env_kwargs
    ]
    assert not offenders, (
        "these envs DISCARD tool-surface env_kwargs without error, so a config "
        "that sets them is silently a no-op:\n" + "\n".join(offenders)
    )


def test_extra_tools_omitted_equals_empty_list(_ignore_image_staleness) -> None:
    """The omitted-vs-``[]``-vs-``None`` boundary: all three must resolve to
    the same surface. (``finalize_env_kwargs`` strips top-level ``None``s, so
    ``extra_tools: null`` is the omitted case by the time gym.make sees it.)"""
    checked = 0
    for env_id in ("lite.osworld", "osworld_g", "webgym", "mobileworld"):
        try:
            registry.task_ids(env_id)
        except EnvDepsMissingError:
            continue
        checked += 1
        omitted = _construct_raw_env(env_id)._runtime_metadata()
        empty = _construct_raw_env(env_id, extra_tools=[])._runtime_metadata()
        nulled = _construct_raw_env(
            env_id, **finalize_env_kwargs({"extra_tools": None})
        )._runtime_metadata()
        assert (
            omitted.extra_tool_schemas == empty.extra_tool_schemas == nulled.extra_tool_schemas
        ), (
            env_id,
            omitted.extra_tool_schemas,
            empty.extra_tool_schemas,
            nulled.extra_tool_schemas,
        )
    assert checked >= 2, "no env available for the omitted/[]/None boundary"


# ---------------------------------------------------------------------------
# 5. Terminal channel — who needs one, who is exempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("env_id", sorted(_SINGLE_STEP_GROUNDING_ENVS))
async def test_single_step_grounding_envs_terminate_on_first_step(env_id) -> None:
    """PROOF (by stepping the env, not by reading it) that the two grounding
    envs are legitimately exempt from the terminal-tool expectations the
    multi-step ``use`` rows carry: the env itself terminates the episode on
    step 1, so no agent-side finish channel is ever needed.

    This is what makes the 29 screenspot_pro + osworld_g config rows correct
    despite carrying no ``response`` / ``terminate`` in ``extra_tools``.
    """
    from lite.core.tools import make_tool_call

    try:
        registry.task_ids(env_id)
    except EnvDepsMissingError as exc:
        pytest.skip(f"{env_id} unavailable: {exc}")

    first = next(v[0] for v in registry.task_ids(env_id).values() if v)
    env = gym.make(f"{env_id}@{first}")
    try:
        await env.reset()
        result = await env.step([make_tool_call("point", {"x": 0.5, "y": 0.5})])
        assert result.terminated is True, f"{env_id} did not terminate on step 1"
        assert result.truncated is False
    finally:
        await env.close()


def test_grounding_config_rows_are_single_step_envs_only() -> None:
    """The exemption above is scoped by env, so pin that no config row routes a
    ``grounding.*`` agent at a MULTI-step env (which would need the finish
    channel the grounding rows deliberately omit)."""
    offenders = []
    for row in PAIR_ROWS:
        dims = _env_dims(row.env_id)
        task_dim = dims[1] if len(dims) > 1 else ""
        if task_dim.startswith("grounding") and row.env_id not in _SINGLE_STEP_GROUNDING_ENVS:
            offenders.append(f"{row.rel}: {row.env_id} is grounding but not single-step")
    assert not offenders, "\n".join(offenders)


def test_content_only_final_gives_every_pair_a_terminal_channel(
    _ignore_image_staleness,
) -> None:
    """Universal content-only-final invariant across the whole live matrix.

    An episode can always end even when a row declares NO finish extra tool: the
    rollout loop turns a tool-call-free assistant turn into an internal
    ``response(text=...)`` action, then marks the step result terminal after the
    env has had a chance to score it. The answer text is never dropped.
    """
    from lite.core.messages.final import make_no_tool_call_final_actions

    checked = 0
    for row in PAIR_ROWS:
        if row.pair in _KNOWN_UNRESOLVABLE_PAIRS:
            continue
        md = _resolve_metadata(row)
        if md is None:
            continue
        checked += 1
        names = {tool_schema_name(s) for s in (md.extra_tool_schemas or [])}
        actions = make_no_tool_call_final_actions("the answer")
        assert len(actions) == 1, (row.rel, actions)
        action = actions[0]
        # N4: the terminal channel no longer depends on what the env declared.
        # Every pair gets response(text=...) and the answer text is never dropped.
        assert tool_call_name(action) == "response", (row.rel, action)
        assert tool_call_arguments(action)["text"] == "the answer", (row.rel, names)
    assert checked >= _MIN_LIVE_CONSTRUCTED_ROWS, f"only {checked} rows resolved live metadata"


def test_multi_step_rows_expose_response_when_the_env_offers_one(
    _ignore_image_staleness,
) -> None:
    """Freeze CUA multi-step rows whose model-visible schema omits ``response``.

    Content-only final prose still has the internal ``response(text=...)``
    backstop. This check is about the explicit tool surface: a NEW multi-step
    CUA row that hides ``response`` should be justified rather than appearing
    by omission. Generic no-tools envs answer through direct text and are not
    part of the CUA extra-tool surface.

    Rows that legitimately do this today are frozen below; a NEW one has to
    be justified explicitly rather than appearing by omission.
    """
    without_response: list[str] = []
    checked = 0
    for row in PAIR_ROWS:
        if row.env_id in _SINGLE_STEP_GROUNDING_ENVS:
            continue
        md = _resolve_metadata(row)
        if md is None:
            continue
        if not isinstance(md, LiteCUAMetadata):
            continue
        checked += 1
        names = {tool_schema_name(s) for s in (md.extra_tool_schemas or [])}
        if "response" not in names:
            without_response.append(row.rel)

    assert checked >= 40, f"only {checked} multi-step rows resolved"
    unexpected = sorted(set(without_response) - _MULTI_STEP_ROWS_WITHOUT_RESPONSE)
    assert not unexpected, (
        "new multi-step config rows with no ``response`` channel — a "
        "model-visible answer tool is absent from the row:\n" + "\n".join(unexpected)
    )


# Frozen set: multi-step rows that expose no ``response`` extra tool. Every
# entry is a DESKTOP automation bench (osworld / lite.osworld / lite.demo /
# lite.scalecua / lite.cuaworld / waa / cua.bench) scored on VM state, not on returned
# text — losing the text channel costs nothing there. Computed live; a browser /
# mobile / QA row appearing here is a bug, which is why the sweep freezes it.
_MULTI_STEP_ROWS_WITHOUT_RESPONSE = {
    "scripts/configs/claude/default/lite.osworld.yaml",
    "scripts/configs/claude/default/osworld_2.yaml",
    "scripts/configs/claude/default/osworld.yaml",
    "scripts/configs/evocua/default/lite.osworld.yaml",
    "scripts/configs/evocua/default/osworld.yaml",
    "scripts/configs/gemini/default/lite.osworld.yaml",
    "scripts/configs/gemini/default/osworld.yaml",
    "scripts/configs/gemini/default/osworld_2.yaml",
    "scripts/configs/gpt/default/lite.cuaworld.yaml",
    "scripts/configs/gpt/default/lite.cuagym.yaml",
    "scripts/configs/gpt/default/lite.demo.yaml",
    "scripts/configs/gpt/default/cua.bench/basic.yaml",
    "scripts/configs/gpt/default/cua.bench/kicad.yaml",
    "scripts/configs/gpt/default/cua.bench/workflows.yaml",
    "scripts/configs/gpt/default/lite.osworld.bash.yaml",
    "scripts/configs/gpt/default/lite.osworld.yaml",
    "scripts/configs/gpt/default/lite.scalecua.yaml",
    "scripts/configs/gpt/default/osworld.yaml",
    "scripts/configs/gpt/default/osworld_2.yaml",
    "scripts/configs/gpt/default/waa.yaml",
    "scripts/configs/gpt/recipes/collect/lite.cuagym.yaml",
    "scripts/configs/gpt/recipes/collect/lite.cuaworld.yaml",
    "scripts/configs/gpt/recipes/collect/lite.osworld.yaml",
    "scripts/configs/gpt/recipes/collect/lite.scalecua.yaml",
    "scripts/configs/qwen2_5_vl/default/lite.osworld.yaml",
    "scripts/configs/qwen3_5/compact/lite.osworld.yaml",
    "scripts/configs/qwen3_5/default/cua.bench/basic.yaml",
    "scripts/configs/qwen3_5/default/cua.bench/kicad.yaml",
    "scripts/configs/qwen3_5/default/cua.bench/workflows.yaml",
    "scripts/configs/qwen3_5/default/lite.cuaworld.yaml",
    "scripts/configs/qwen3_5/default/lite.cuagym.yaml",
    "scripts/configs/qwen3_5/default/lite.osworld.bash.yaml",
    "scripts/configs/qwen3_5/default/lite.osworld.yaml",
    "scripts/configs/qwen3_5/default/osworld.yaml",
    "scripts/configs/qwen3_5/default/waa.yaml",
    "scripts/configs/qwen3_8/default/cua.bench/basic.yaml",
    "scripts/configs/qwen3_8/default/cua.bench/kicad.yaml",
    "scripts/configs/qwen3_8/default/cua.bench/workflows.yaml",
    "scripts/configs/qwen3_8/default/lite.cuagym.yaml",
    "scripts/configs/qwen3_8/default/lite.cuaworld.yaml",
    "scripts/configs/qwen3_8/default/osworld.yaml",
    "scripts/configs/qwen3_8/default/waa.yaml",
    "scripts/configs/qwen3_vl/compact/lite.osworld.yaml",
    "scripts/configs/qwen3_vl/default/cua.bench/basic.yaml",
    "scripts/configs/qwen3_vl/default/cua.bench/kicad.yaml",
    "scripts/configs/qwen3_vl/default/cua.bench/workflows.yaml",
    "scripts/configs/qwen3_vl/default/lite.osworld.bash.yaml",
    "scripts/configs/qwen3_vl/default/lite.osworld.yaml",
    "scripts/configs/qwen3_vl/default/osworld.yaml",
    "scripts/configs/qwen3_vl/default/osworld_2.yaml",
    "scripts/configs/qwen3_vl/default/waa.yaml",
    "scripts/configs/ui_tars/default/lite.osworld.yaml",
    "scripts/configs/ui_tars/default/osworld.yaml",
    "scripts/configs/ui_tars_15_v1/default/lite.osworld.yaml",
    "scripts/configs/ui_tars_15_v1/default/osworld.yaml",
}
