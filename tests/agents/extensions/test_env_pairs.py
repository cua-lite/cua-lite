"""``lite/agents/extensions/*`` × env boundary conditions, for the unusual yaml rows.

``tests/gym/matrix/test_agent_env_pair_matrix.py`` proves every shipped ``(agent, env)``
yaml CONSTRUCTS. Construction is not the whole contract: the extension rows also
have to survive the round trip a rollout actually drives — parsing a model reply
back into canonical tool calls, shaping the prompt through the extension protocol,
and honoring the seams the bespoke API loops call. This file gates those, per
extension family:

  * ``extensions/webharbor/webvoyager`` + ``extensions/browsergym`` — an env may
    advertise a STANDALONE extra tool whose NAME equals a canonical GUI action
    but whose signature is completely different (webvoyager SoM ``click(index)``,
    browsergym bid ``click(bid)``). ``metadata.extra_tool_schemas`` is the single
    source of truth for the standalone surface, so it must win that collision —
    the same precedence the Claude / GPT loops already implement. Before the fix,
    ``qwen3_5``/``qwen3_vl`` fed such a call to ``LiteDesktopActionSpace.click``
    and raised ``TypeError`` mid-rollout on every SoM click.
  * ``extensions/browsergym`` (``browsergym.generic``) — ``tool_call_format`` picks
    the wire format the ``# History`` section is rendered in. It MUST match the
    paired adapter's live ``<tool_call>`` format: the mismatch is not cosmetic,
    the other family's parser returns NO tool calls for that block.
  * ``extensions/browsergym`` (``visualwebarena.goal_image``) — goal images are
    injected AFTER the protocol windows history, so they must survive every
    protocol the shipped goal_image / som / mixed rows pair with.
  * ``extensions/teacher`` (``gpt.teacher``) — it overrides the GPT loop's parse
    seam, so its signature must track the base seam exactly (the loop always
    passes ``call_id_start=``), and it must not swallow the shared
    termination classifier's channel.

Hermetic + dep-light: no model download, no network, no browser, no container.
Rows whose env cannot register on this host are skipped individually.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/extensions/test_env_pairs.py -p no:cacheprovider -q
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from PIL import Image

import lite.gym  # noqa: F401  (installs the registry module)
from lite.agents.core.adapter.base import AgentAdapterRegistry
from lite.agents.extensions.browsergym.protocol import (
    render_tool_call_json,
    render_tool_call_xml,
)
from lite.agents.extensions.teacher.agent import (
    GPTMobileTeacherAgent,
    GPTTeacherAgent,
    _GPTTeacherMixin,
)
from lite.agents.factory import AGENTS, API_AGENTS, LOCAL_AGENTS
from lite.agents.models import AgentRegistry
from lite.agents.models.gpt.agent import GPTDesktopUseAgent, GPTMobileUseAgent
from lite.core import (
    LiteCUAMetadata,
    LiteSample,
)
from lite.core.messages import no_tool_call_final_text
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.action_space import LiteDesktopActionSet, LiteMobileActionSet
from lite.core.tools.calls import (
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters
from lite.gym.errors import EnvDepsMissingError
from lite.utils.path import project_root
from lite.utils.registry import compose_key

_GUI_VALID_ACTION_NAMES = (
    LiteDesktopActionSet.get_action_names() | LiteMobileActionSet.get_action_names()
)

_CONFIGS = project_root() / "scripts" / "configs"
_registry_module = sys.modules["lite.gym.registry"]


def _register_extension_pair_test_modules() -> None:
    for module_name in (
        "lite.agents.models.qwen3_vl.action_space",
        "lite.agents.models.qwen3_vl.protocol",
        "lite.agents.models.qwen3_vl.adapter",
        "lite.agents.models.qwen3_vl.agent",
        "lite.agents.models.qwen2_5_vl.action_space",
        "lite.agents.models.qwen2_5_vl.adapter",
        "lite.agents.models.qwen2_5_vl.agent",
        "lite.agents.models.qwen3_5.action_space",
        "lite.agents.models.qwen3_5.protocol",
        "lite.agents.models.qwen3_5.adapter",
        "lite.agents.models.qwen3_5.agent",
        "lite.agents.models.qwen3_8.action_space",
        "lite.agents.models.qwen3_8.adapter",
        "lite.agents.models.qwen3_8.agent",
        "lite.agents.models.fara.action_space",
        "lite.agents.models.fara.protocol",
        "lite.agents.models.fara.adapter",
        "lite.agents.models.fara.agent",
        "lite.agents.models.gpt.action_space",
        "lite.agents.models.gpt.agent",
        "lite.agents.extensions.browsergym.protocol",
        "lite.agents.extensions.browsergym.goal_image",
        "lite.agents.extensions.webharbor.webvoyager.protocol",
        "lite.agents.extensions.teacher.agent",
    ):
        importlib.import_module(module_name)


_register_extension_pair_test_modules()


# =============================================================================
# Helpers
# =============================================================================


def _load(rel: str) -> dict[str, Any]:
    path = _CONFIGS / rel
    if not path.exists():
        pytest.skip(f"config not present: {rel}")
    return yaml.safe_load(path.read_text())


def _env_metadata(env_id: str, env_kwargs: dict[str, Any]) -> LiteCUAMetadata:
    """Resolve a yaml row's ``env_kwargs`` through the REAL env — i.e. what
    ``gym.make`` does minus the container-booting ``ensure_services`` step."""
    try:
        _registry_module._import_env(env_id)
    except Exception as exc:  # optional deps / assets absent on this host
        pytest.skip(f"{env_id} not available here: {type(exc).__name__}")
    keys = [k for k in _registry_module._specs if k.startswith(f"{env_id}@")]
    if not keys:
        pytest.skip(f"{env_id}: nothing registered on this host")
    spec = _registry_module._specs[sorted(keys)[0]]
    merged = {
        **_registry_module.registry.env_make_kwargs(env_id),
        **(spec.kwargs or {}),
        **env_kwargs,
    }
    for framework_only in (
        *_registry_module._WRAPPER_KWARG_DEFAULTS,
        "session_id",
        "token_hash",
        "server_port",
        "env_server_url",
    ):
        merged.pop(framework_only, None)
    try:
        return spec.entry_point(**merged).metadata
    except EnvDepsMissingError as exc:
        # The env's image is missing or STALE on this host. That is a HOST state,
        # not the wiring these tests assert, and the freshness hash is over raw
        # source bytes — so a comment-only edit to a baked source reddens this
        # file with nothing wrong in it. Skip like every sibling image-dependent
        # test (cf. tests/gym/metadata/test_metadata_invariant.py), which is what this
        # helper's docstring already promises ("minus the container-booting step").
        pytest.skip(f"{env_id}: {exc.what}")


def _agent_kwargs_for_construction(cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(cfg.get("agent_kwargs") or {})
    # Production rollout consumes sampling_kwargs before adapter construction;
    # local serving uses it as generation-only config.
    kwargs.pop("sampling_kwargs", None)
    return kwargs


def _agent_make_for_test(
    model_id: str,
    *,
    env: Any,
    agent_id: str | None = None,
    **kwargs: Any,
) -> Any:
    """Factory-equivalent construction with this file's bounded registrations."""
    if model_id not in AGENTS:
        raise KeyError(f"Unknown model '{model_id}'. Available: {list(AGENTS)}")
    _register_extension_pair_test_modules()
    meta = env.metadata
    kwargs.setdefault("metadata", meta)
    if model_id in API_AGENTS:
        kwargs.setdefault("model_id", model_id)
    key = compose_key(
        agent_id or AGENTS[model_id]["agent_id"],
        *meta.dims,
    )
    return AgentRegistry.get(key, **kwargs)


def _agent_for(rel: str, model_id: str, metadata: LiteCUAMetadata):
    cfg = _load(rel)
    kwargs = _agent_kwargs_for_construction(cfg)
    if model_id in LOCAL_AGENTS:
        kwargs.setdefault("processor", MagicMock())
        kwargs.setdefault("generate_fn", lambda *a, **k: None)
    return _agent_make_for_test(
        model_id,
        env=SimpleNamespace(metadata=metadata),
        agent_id=cfg["agent_id"],
        **kwargs,
    )


def _agent_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Bare agent-wire call, used only to simulate parsed model output."""
    call = {"name": name, "arguments": arguments}
    call["call_id"] = "c0"
    return call


def _lite_call(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str | None = None,
) -> dict[str, Any]:
    return make_tool_call(name, arguments, call_id=call_id)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


# The two shipped SoM rows: the FULL (non-``.base``) qwen adapters paired with an
# env whose standalone extras are named ``click`` / ``scroll``.
_WEBVOYAGER_SOM_ROWS = [
    ("qwen3_5/default/webharbor.webvoyager/som.yaml", "Qwen/Qwen3.5-9B"),
    ("qwen3_vl/default/webharbor.webvoyager/som.yaml", "Qwen/Qwen3-VL-8B-Instruct"),
]


# =============================================================================
# 1. Env extras win a name collision with the canonical GUI actions
# =============================================================================


@pytest.mark.parametrize("rel,model_id", _WEBVOYAGER_SOM_ROWS)
def test_webvoyager_som_extras_collide_with_action_names(rel, model_id) -> None:
    """Premise check: these rows really do advertise GUI-action NAMES as
    standalone extras while blocking the GUI enum. If the env ever renames them
    the collision rows below become vacuous, so pin the premise explicitly."""
    cfg = _load(rel)
    md = _env_metadata(cfg["env_id"], dict(cfg.get("env_kwargs") or {}))
    names = {tool_schema_name(s) for s in (md.extra_tool_schemas or [])}
    assert md.valid_actions == [], "row is supposed to block the GUI enum"
    assert names & _GUI_VALID_ACTION_NAMES == {"click", "scroll"}, names
    # …and their signatures are NOT the canonical GUI ones (index / page-scroll).
    by_name = {tool_schema_name(s): s for s in md.extra_tool_schemas}
    assert set(tool_schema_parameters(by_name["click"])["properties"]) == {"index"}
    assert set(tool_schema_parameters(by_name["scroll"])["properties"]) == {"down", "pages"}


@pytest.mark.parametrize("rel,model_id", _WEBVOYAGER_SOM_ROWS)
def test_webvoyager_som_extras_survive_conversion(rel, model_id) -> None:
    """The SoM tool calls the model is TOLD to emit must round-trip as standalone
    canonical calls.

    Regression: the qwen action space routed any bare name in
    ``LiteDesktopActionSpace.get_tool_names()`` into the GUI space, so
    ``click(index=3)`` raised ``TypeError: LiteDesktopActionSpace.click() got an
    unexpected keyword argument 'index'`` — a hard crash on the FIRST SoM click of
    every rollout. The construction-time guard in ``BaseAgentAdapter.__post_init__``
    could not see it: it compares extras against
    ``action_space.get_tool_schemas()``, which here is just ``{"computer_use"}``.
    """
    cfg = _load(rel)
    md = _env_metadata(cfg["env_id"], dict(cfg.get("env_kwargs") or {}))
    adapter = _agent_for(rel, model_id, md).adapter

    for name, args in (
        ("click", {"index": 3}),
        ("scroll", {"down": True, "pages": 1}),
        ("input", {"index": 2, "text": "hi"}),
        ("go_back", {}),
        ("response", {"text": "42"}),
    ):
        out = adapter.convert_message_from_agent(
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [_agent_call(name, args)],
            },
            resolution=(1000, 1000),
        )
        assert [tool_call_name(c) for c in out["tool_calls"]] == [name], (
            f"{name} did not survive as a standalone extra: {out['tool_calls']}"
        )
        assert tool_call_arguments(out["tool_calls"][0]) == args
        assert tool_call_id(out["tool_calls"][0]) == "c0"


@pytest.mark.parametrize("rel,model_id", _WEBVOYAGER_SOM_ROWS)
def test_action_wrapper_path_unaffected_by_the_extras_route(rel, model_id) -> None:
    """Extras winning the name collision must NOT shadow the family's own wrapper:
    ``computer_use`` still converts, and its native finish spelling still maps onto
    the canonical ``response`` the env advertises."""
    cfg = _load(rel)
    md = _env_metadata(cfg["env_id"], dict(cfg.get("env_kwargs") or {}))
    adapter = _agent_for(rel, model_id, md).adapter

    out = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [_agent_call("computer_use", {"action": "answer", "text": "42"})],
        },
        resolution=(1000, 1000),
    )
    assert [tool_call_name(c) for c in out["tool_calls"]] == ["response"]
    assert tool_call_arguments(out["tool_calls"][0]) == {"text": "42"}


@pytest.mark.parametrize("rel,model_id", _WEBVOYAGER_SOM_ROWS)
def test_valid_actions_empty_keeps_exactly_one_terminal_channel(rel, model_id) -> None:
    """``valid_actions: []`` drops the provider-native wrapper but keeps canonical finish extras.

    Finish tools are prompt-surface priors, just like GUI valid_actions. An
    otherwise empty native wrapper must not hide ``response``/``terminate``;
    the terminal channel remains explicit and singular.
    """
    cfg = _load(rel)
    md = _env_metadata(cfg["env_id"], dict(cfg.get("env_kwargs") or {}))
    adapter = _agent_for(rel, model_id, md).adapter

    lines = adapter._build_tools_section().splitlines()
    rendered = [
        json.loads(line)
        for line in lines[lines.index("<tools>") + 1 : lines.index("</tools>")]
        if line.strip()
    ]
    # Qwen renders canonical nested Lite function-tool schemas.
    by_name = {tool_schema_name(s): s for s in rendered}
    assert "computer_use" not in by_name
    assert "response" in by_name
    # The SoM surface itself is intact.
    assert {"click", "input", "scroll", "go_back"} <= set(by_name)


# Every shipped row whose ``env_kwargs.extra_tools`` names a canonical GUI
# action. These are the non-canonical action surfaces: browsergym's bid tools
# (``click(bid)`` / ``scroll(delta_x, delta_y)``) and webvoyager's SoM tools
# (``click(index)`` / ``scroll(down, pages)``). Frozen because the collision has
# two consequences a new row must be forced to think about:
#   1. the adapter must route the extra, not the GUI action (tested above);
#   2. publish validation must remain schema-aware, so ``click(index=...)`` is
#      accepted while ``click(coordinate=...)`` cannot masquerade as that extra.
# Adding a row here is a deliberate act, not an accident.
_KNOWN_ACTION_NAMED_EXTRA_TOOL_ROWS = {
    "qwen3_5/default/browsergym.miniwob/text_only.yaml",
    "qwen3_5/default/browsergym.visualwebarena/mixed.yaml",
    "qwen3_5/default/browsergym.visualwebarena/som.yaml",
    "qwen3_5/default/browsergym.webarena/som.yaml",
    "qwen3_5/default/browsergym.webarena/text_only.yaml",
    "qwen3_5/default/webharbor.webvoyager/som.yaml",
    "qwen3_8/default/browsergym.miniwob/text_only.yaml",
    "qwen3_8/default/browsergym.visualwebarena/mixed.yaml",
    "qwen3_8/default/browsergym.visualwebarena/som.yaml",
    "qwen3_8/default/browsergym.webarena/som.yaml",
    "qwen3_8/default/browsergym.webarena/text_only.yaml",
    "qwen3_8/default/webharbor.webvoyager/som.yaml",
    "qwen3_vl/default/browsergym.miniwob/text_only.yaml",
    "qwen3_vl/default/browsergym.visualwebarena/mixed.yaml",
    "qwen3_vl/default/browsergym.visualwebarena/som.yaml",
    "qwen3_vl/default/browsergym.webarena/som.yaml",
    "qwen3_vl/default/browsergym.webarena/text_only.yaml",
    "qwen3_vl/default/webharbor.webvoyager/som.yaml",
}


def test_action_named_extra_tool_rows_are_the_known_set() -> None:
    """Source-yaml gate (dep-light: no env import needed). A NEW row naming a
    canonical GUI action in ``extra_tools`` must be added here consciously —
    see the comment above for the two consequences it inherits."""
    found = set()
    for path in sorted(_CONFIGS.rglob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(cfg, dict):
            continue
        requested = set((cfg.get("env_kwargs") or {}).get("extra_tools") or [])
        if requested & _GUI_VALID_ACTION_NAMES:
            found.add(str(path.relative_to(_CONFIGS)))
    assert found == _KNOWN_ACTION_NAMED_EXTRA_TOOL_ROWS, (
        f"new: {sorted(found - _KNOWN_ACTION_NAMED_EXTRA_TOOL_ROWS)}; "
        f"gone: {sorted(_KNOWN_ACTION_NAMED_EXTRA_TOOL_ROWS - found)}"
    )


def test_browsergym_base_adapter_passes_bid_tools_through_unchanged() -> None:
    """The browsergym text+bid rows point at the ``.base`` adapters (empty native
    action space), so their bid tools were never intercepted — the extras route
    must preserve the same Lite call ids, names, and arguments."""
    schemas = [
        make_tool_schema(
            "click",
            description="bid click",
            parameters={
                "type": "object",
                "properties": {"bid": {"type": "string"}},
                "required": ["bid"],
            },
        ),
        make_tool_schema(
            "scroll",
            description="bid scroll",
            parameters={
                "type": "object",
                "properties": {"delta_x": {"type": "number"}, "delta_y": {"type": "number"}},
                "required": ["delta_x", "delta_y"],
            },
        ),
    ]
    md = LiteCUAMetadata(dims=("browser", "use"), valid_actions=[], extra_tool_schemas=schemas)
    for key in ("qwen3_5.base@browser@use", "qwen3_vl.base@browser@use"):
        adapter = AgentAdapterRegistry.get(key, metadata=md)
        calls = [
            _agent_call("click", {"bid": "a23"}),
            _agent_call("scroll", {"delta_x": 0, "delta_y": 200}),
        ]
        out = adapter.convert_message_from_agent(
            {"role": "assistant", "content": [], "tool_calls": calls},
            resolution=(1280, 720),
        )
        assert [tool_call_id(call) for call in out["tool_calls"]] == ["c0", "c0"], key
        assert [tool_call_name(call) for call in out["tool_calls"]] == ["click", "scroll"], key
        assert [tool_call_arguments(call) for call in out["tool_calls"]] == [
            {"bid": "a23"},
            {"delta_x": 0, "delta_y": 200},
        ], key


# =============================================================================
# 2. browsergym.generic `tool_call_format` must match the adapter's wire format
# =============================================================================

# Adapter family prefix → (its `tool_call_format`, the OTHER format).
# Qwen3.8 shares Qwen3.5's XML tool_call chat_template.
_FAMILY_WIRE_FORMAT = {"qwen3_5": "xml", "qwen3_8": "xml", "qwen3_vl": "json"}
_RENDERERS = {"json": render_tool_call_json, "xml": render_tool_call_xml}


def _browsergym_generic_rows() -> list[tuple[Path, str, str]]:
    """(path, adapter family, effective tool_call_format) for every shipped row
    that selects the ``browsergym.generic`` protocol."""
    rows: list[tuple[Path, str, str]] = []
    for path in sorted(_CONFIGS.rglob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(cfg, dict):
            continue
        ak = cfg.get("agent_kwargs") or {}
        if (ak.get("protocol_key") or "") != "browsergym.generic":
            continue
        key = ak.get("adapter_key") or cfg.get("agent_id") or ""
        family = str(key).split(".")[0].split("@")[0]
        fmt = (ak.get("protocol_kwargs") or {}).get("tool_call_format", "json")
        rows.append((path, family, fmt))
    return rows


def test_browsergym_generic_rows_exist() -> None:
    """Floor: if this ever returns nothing the pairing test below is vacuous."""
    assert len(_browsergym_generic_rows()) >= 6


@pytest.mark.parametrize("family,fmt", sorted(_FAMILY_WIRE_FORMAT.items()))
def test_history_render_format_is_load_bearing(family: str, fmt: str) -> None:
    """A mismatched ``tool_call_format`` is NOT cosmetic: the other family's
    ``<tool_call>`` block yields NO tool calls from this adapter's own parser, so
    every ``# History`` entry teaches the model a format its own decoder drops."""
    adapter = AgentAdapterRegistry.get(
        f"{family}.base@browser@use",
        metadata=LiteCUAMetadata(dims=("browser", "use"), valid_actions=[]),
    )
    lite_tc = _lite_call("click", {"bid": "a23"})
    agent_tc = {"name": "click", "arguments": {"bid": "a23"}}
    other = "json" if fmt == "xml" else "xml"

    matching = adapter.parse_raw_assistant_response(_RENDERERS[fmt](lite_tc))
    assert matching.get("tool_calls") == [agent_tc], (
        f"{family}: its own {fmt} render must parse back"
    )
    mismatched = adapter.parse_raw_assistant_response(_RENDERERS[other](lite_tc))
    assert not mismatched.get("tool_calls"), (
        f"{family}: {other} render unexpectedly parsed — the pairing rule below "
        "would no longer be load-bearing"
    )


def test_every_browsergym_generic_row_pairs_the_right_tool_call_format() -> None:
    """Every shipped ``browsergym.generic`` row must render ``# History`` in the
    wire format its own adapter emits and parses."""
    wrong = [
        (str(path.relative_to(_CONFIGS)), family, fmt, _FAMILY_WIRE_FORMAT[family])
        for path, family, fmt in _browsergym_generic_rows()
        if family in _FAMILY_WIRE_FORMAT and fmt != _FAMILY_WIRE_FORMAT[family]
    ]
    assert not wrong, (
        "browsergym.generic rows whose protocol_kwargs.tool_call_format does not "
        f"match their adapter's live wire format: {wrong}"
    )


def test_browsergym_generic_rows_use_a_known_adapter_family() -> None:
    """Guard the guard: a new family reaching ``browsergym.generic`` must be added
    to ``_FAMILY_WIRE_FORMAT`` (otherwise the pairing test silently skips it)."""
    unknown = sorted(
        {family for _, family, _ in _browsergym_generic_rows() if family not in _FAMILY_WIRE_FORMAT}
    )
    assert not unknown, (
        f"unmapped adapter families on browsergym.generic: {unknown} — add their "
        "live <tool_call> wire format to _FAMILY_WIRE_FORMAT"
    )


# =============================================================================
# 3. goal_image survives every protocol the shipped rows pair it with
# =============================================================================

_GOAL_IMAGE_ROWS = [
    ("qwen3_vl/default/browsergym.visualwebarena/goal_image.yaml", "Qwen/Qwen3-VL-8B-Instruct"),
    ("qwen3_5/default/browsergym.visualwebarena/goal_image.yaml", "Qwen/Qwen3.5-9B"),
    ("fara/default/browsergym.visualwebarena/goal_image.yaml", "microsoft/Fara-7B"),
    ("qwen3_vl/default/browsergym.visualwebarena/som.yaml", "Qwen/Qwen3-VL-8B-Instruct"),
    ("qwen3_5/default/browsergym.visualwebarena/som.yaml", "Qwen/Qwen3.5-9B"),
    ("qwen3_vl/default/browsergym.visualwebarena/mixed.yaml", "Qwen/Qwen3-VL-8B-Instruct"),
    ("qwen3_5/default/browsergym.visualwebarena/mixed.yaml", "Qwen/Qwen3.5-9B"),
]


def _goal_image_trajectory(n_goal: int, n_turns: int):
    """A LiteSample shaped like a VWA rollout: turn-0 user carries the reset
    metadata goal-image refs that ``BrowserGymEnv.reset`` emits, plus a
    page screenshot; each later turn adds a fresh screenshot on a role:tool obs."""
    sample = LiteSample(metadata=None)
    sample.images = [Image.new("RGB", (8, 8), (1, 1, 1))]
    sample.messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Sell this item.\n## AXTree:\n[1] button 'Go'"},
                {
                    "type": "metadata",
                    "data": {
                        "goal_images_b64": [
                            base64.b64encode(_png_bytes((10 * i, 0, 0))).decode()
                            for i in range(n_goal)
                        ]
                    },
                },
            ],
        }
    ]
    processed = list(sample.images)
    for turn in range(n_turns):
        call_id = f"c{turn}"
        sample.images.append(Image.new("RGB", (8, 8), (turn + 2,) * 3))
        processed.append(sample.images[-1])
        sample.messages.append(
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [_lite_call("click", {"bid": "a1"}, call_id=call_id)],
            }
        )
        sample.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": [
                    {"type": "image", "index": len(sample.images) - 1},
                    {"type": "text", "text": f"obs{turn}\n## AXTree:\n[1] button 'Go'"},
                ],
            }
        )
    return sample, processed


@pytest.mark.parametrize("rel,model_id", _GOAL_IMAGE_ROWS)
def test_goal_images_survive_the_paired_protocols_history_window(rel, model_id) -> None:
    """The goal-image agent re-surfaces the task images AFTER the protocol windows
    history, so they must reach the prompt for EVERY protocol the shipped rows use
    — the chat-style history protocols (which drop old turns) and
    ``browsergym.generic`` (which collapses everything into one user message)."""
    cfg = _load(rel)
    if cfg.get("agent_id") != "visualwebarena.goal_image":
        pytest.skip(f"{rel} is not a goal_image row")
    md = LiteCUAMetadata(dims=("browser", "use"), extra_tool_schemas=[], valid_actions=None)
    agent = _agent_for(rel, model_id, md)
    protocol = agent.adapter.protocol

    sample, processed = _goal_image_trajectory(n_goal=2, n_turns=6)
    asyncio.run(agent._ingest_goal_images(sample, processed))
    data = next(c for c in sample.messages[0]["content"] if c["type"] == "metadata")["data"]
    indices = [c["index"] for c in sample.messages[0]["content"] if c.get("type") == "image"][:2]

    assert len(indices) == 2
    assert "goal_images_b64" not in data, "goal-image refs must be consumed at ingest"
    assert len(sample.images) == len(processed), "image arrays must stay in lockstep"

    out = protocol.process_messages(sample.messages)
    surviving = [
        c.get("index") for m in out for c in m.get("content", []) if c.get("type") == "image"
    ]
    assert set(indices) <= set(surviving), (
        f"{type(protocol).__name__} dropped the goal images: {surviving}"
    )
    # They lead the first user message, ahead of the per-turn page screenshot.
    first_user = next(m for m in out if m.get("role") == "user")
    head_images = [c.get("index") for c in first_user["content"] if c.get("type") == "image"]
    assert head_images[: len(indices)] == indices


def test_goal_images_are_additive_to_the_protocol_image_budget() -> None:
    """``protocol_kwargs.max_n_images`` bounds the HISTORY screenshots only; the
    goal images are injected afterwards, so the effective per-turn image count is
    ``max_n_images + n_goal_images``. Pinned because the fara row sets
    ``max_n_images: 3`` to match the upstream budget — with 2 goal images the model
    actually sees 5."""
    rel, model_id = "fara/default/browsergym.visualwebarena/goal_image.yaml", "microsoft/Fara-7B"
    md = LiteCUAMetadata(dims=("browser", "use"), extra_tool_schemas=[], valid_actions=None)
    agent = _agent_for(rel, model_id, md)
    protocol = agent.adapter.protocol
    assert protocol.max_n_images == 3

    sample, processed = _goal_image_trajectory(n_goal=2, n_turns=4)
    asyncio.run(agent._ingest_goal_images(sample, processed))
    out = protocol.process_messages(sample.messages)
    n_images = sum(1 for m in out for c in m.get("content", []) if c.get("type") == "image")
    assert n_images == protocol.max_n_images + 2


# =============================================================================
# 4. gpt.teacher must not diverge from the shared GPT loop seams
# =============================================================================


def _teacher_metadata() -> LiteCUAMetadata:
    return LiteCUAMetadata(dims=("desktop", "use"), extra_tool_schemas=[
            make_tool_schema(
                "response",
                description="finish",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ])


def _teacher_agent():
    return _agent_make_for_test(
        "gpt-5.5", env=SimpleNamespace(metadata=_teacher_metadata()), agent_id="gpt.teacher"
    )


_TEACHER_ACTION_OUTPUT = [
    {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Thought: The Sheet1 tab is visible.\nAction: Right-click the Sheet1 tab.",
            }
        ],
    },
    {
        "type": "function_call",
        "name": "response",
        "call_id": "provider-x",
        "arguments": json.dumps({"text": "42"}),
    },
]


@pytest.mark.parametrize(
    "base_cls, teacher_cls",
    [(GPTDesktopUseAgent, GPTTeacherAgent), (GPTMobileUseAgent, GPTMobileTeacherAgent)],
    ids=["desktop", "mobile"],
)
def test_teacher_parse_seam_signature_tracks_the_base(base_cls, teacher_cls) -> None:
    """Each use-agent's ``sample()`` is the ONLY caller of its own seam and always
    passes ``call_id_start=``. A narrower override raises ``TypeError`` on the first
    model reply of every ``gpt.teacher`` run, so pin the keyword set.

    Both platforms are checked because ONE override in ``_GPTTeacherMixin`` serves
    both: the mobile seam exists only so that override applies there too, and a
    signature drift on either side breaks it silently at import time.
    """
    import inspect

    base = inspect.signature(base_cls._parse_output_items)
    teacher = inspect.signature(teacher_cls._parse_output_items)
    missing = sorted(set(base.parameters) - set(teacher.parameters))
    assert not missing, (
        f"{teacher_cls.__name__} drops base seam parameters {missing} — sample() calls "
        "it with them and would raise TypeError"
    )


def test_mobilegym_teacher_surface_is_renderable_by_the_qwen_student() -> None:
    """Every action the mobilegym collect recipe offers the teacher must be one the
    Qwen mobile student can render back.

    This is a DATA-LOSS gate, not a style one. `export_sft` runs `--no-strict` in the
    campaign runbook, so an unrenderable action does not fail the export -- it prints
    one `Skipped N rows` line and discards that trajectory whole. One `drag` in a
    30-step episode costs all 30 steps, and the runbook's i1-vs-i4 row-count check
    cannot see it because both cells drop the same row. Measured once for real: 1 of 4
    trajectories in the first smoke collection carried a `drag`.
    """
    import yaml as _yaml

    from lite.agents.models.qwen3_5.action_space import Qwen3_5MobileActionSpace

    cfg = _yaml.safe_load((_CONFIGS / "gpt/recipes/collect/mobilegym.yaml").read_text())
    md = _env_metadata("mobilegym", dict(cfg.get("env_kwargs") or {}))
    agent = _agent_make_for_test(
        "gpt-5.5",
        env=SimpleNamespace(metadata=md),
        agent_id=cfg["agent_id"],
        **_agent_kwargs_for_construction(cfg),
    )
    student = Qwen3_5MobileActionSpace()
    # A minimal well-formed call per offered GUI action, in canonical Lite shape.
    probes = {
        "tap": {"coordinate": [300, 400], "clicks": 1},
        "long_press": {"coordinate": [300, 400], "duration": 1},
        "swipe": {"start_coordinate": [1, 2], "end_coordinate": [3, 4]},
        "drag": {"start_coordinate": [1, 2], "end_coordinate": [3, 4]},
        "pinch": {"coordinate": [1, 2], "direction": "in", "amount": 25},
        "type": {"text": "x"},
        "system_button": {"button": "Back"},
        "wait": {"duration": 1},
        "screenshot": {},
    }
    offered = {
        (t["function"]["name"] if "function" in t else t["name"]) for t in agent._build_tools()
    }
    unrenderable = []
    for name in sorted(offered & probes.keys()):
        try:
            student.convert_tool_calls_to_agent(
                [{"id": "c0", "type": "function",
                  "function": {"name": name, "arguments": probes[name]}}]
            )
        except Exception as exc:  # noqa: BLE001 -- any failure is the finding
            unrenderable.append(f"{name}: {type(exc).__name__}")
    assert not unrenderable, (
        "mobilegym collect recipe offers the teacher actions the Qwen student cannot "
        f"render, so export silently drops those trajectories: {unrenderable}. "
        "Pin them out via env_kwargs.valid_actions in the recipe."
    )


def test_mobile_teacher_appends_the_contract_at_its_own_prompt_seam() -> None:
    """Mobile assembles its system prompt in ``_system_prompt_for_mobile_request``,
    not ``_effective_system_prompt``, so that is where the contract has to land.

    Getting this wrong is SILENT: the run still collects, the Thought/Action
    requirement just never reaches the model, and the rows come back with no
    ``inline_reasoning`` — the one thing the teacher exists to produce. Asserting
    against the plain base agent pins BOTH halves: the contract is added, and the
    ``{w}``/``{h}`` substitution the seam owns still happens.
    """
    prompt = "MOBILE BASE {w}x{h}"
    teacher = GPTMobileTeacherAgent(system_prompt=prompt)
    plain = GPTMobileUseAgent(system_prompt=prompt)

    base_out = plain._system_prompt_for_mobile_request(720, 1600)
    teacher_out = teacher._system_prompt_for_mobile_request(720, 1600)

    assert base_out == "MOBILE BASE 720x1600"
    assert teacher_out == base_out + "\n" + teacher._structured_output_instruction()


def test_teacher_stamps_call_ids_from_the_loops_counter() -> None:
    """Tool-result pairing depends on the loop's running counter reaching the
    parse seam, not on the provider's own ids."""
    agent = _teacher_agent()
    parsed = agent._parse_output_items(
        _TEACHER_ACTION_OUTPUT,
        resolution=(1280, 720),
        call_id_start=7,
    )
    assert [tool_call_id(c) for c in parsed.message["tool_calls"]] == ["call_0007"]
    assert [tool_call_name(c) for c in parsed.message["tool_calls"]] == ["response"]
    # The relabel must not cost the loop its provider->canonical provenance.
    assert [(c.provider_call_id, c.canonical_call_id) for c in parsed.provider_calls] == [
        ("provider-x", "call_0007")
    ]


def test_teacher_relabels_only_action_turns() -> None:
    """The Thought/Action relabel applies to a turn that CARRIES tool calls; a
    no-tool-call turn's prose must stay plain ``text`` so the shared
    ``no_tool_call_final_text`` classifier still sees the deliberate termination."""
    agent = _teacher_agent()

    action_turn = agent._parse_output_items(
        _TEACHER_ACTION_OUTPUT,
        resolution=(1280, 720),
        call_id_start=0,
    ).message
    assert [c["type"] for c in action_turn["content"]] == [
        "inline_reasoning",
        "action_description",
    ]
    assert no_tool_call_final_text(action_turn) == "", (
        "an action turn's narration must not read as a termination"
    )

    final_turn = agent._parse_output_items(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "The answer is 42."}],
            }
        ],
        resolution=(1280, 720),
        call_id_start=0,
    ).message
    assert not final_turn.get("tool_calls")
    assert [c["type"] for c in final_turn["content"]] == ["text"]
    assert no_tool_call_final_text(final_turn) == "The answer is 42."


def _teacher_recipe_rows() -> list[Path]:
    """The shipped ``gpt.teacher`` collect recipes, from source yaml only — no env
    import, so the parametrization itself cannot depend on host state."""
    return [
        p
        for p in sorted(_CONFIGS.glob("gpt/recipes/collect/*.yaml"))
        if (yaml.safe_load(p.read_text()) or {}).get("agent_id") == "gpt.teacher"
    ]


def test_teacher_recipe_rows_are_the_known_six() -> None:
    """Premise, dep-light: asserted where no host state can skip it. It used to
    live inside the construction loop below, where the first unavailable env
    skipped the whole function and took this count with it."""
    assert [p.name for p in _teacher_recipe_rows()] == [
        "lite.cuagym.yaml",
        "lite.cuaworld.yaml",
        "lite.osworld.yaml",
        "lite.scalecua.yaml",
        "mobilegym.yaml",
        "webgym.yaml",
    ]


@pytest.mark.parametrize("path", _teacher_recipe_rows(), ids=lambda p: p.stem)
def test_teacher_rows_construct_against_their_real_envs(path: Path) -> None:
    """Each shipped ``gpt.teacher`` collect recipe still resolves the teacher
    agent (not the plain GPT agent).

    ONE PYTEST ROW PER RECIPE on purpose. ``_env_metadata`` skips when an env is
    unregistered or its image is missing/STALE on this host, and a single loop let
    that skip swallow every remaining recipe: with ``lite.cuagym`` STALE (it sorts
    first) all five went unasserted while the file reported ``s``. Per-row, the
    healthy envs keep asserting and pytest's own skip count reports how many rows
    went unchecked.
    """
    cfg = yaml.safe_load(path.read_text())
    md = _env_metadata(cfg["env_id"], dict(cfg.get("env_kwargs") or {}))
    agent = _agent_make_for_test(
        "gpt-5.5",
        env=SimpleNamespace(metadata=md),
        agent_id="gpt.teacher",
        **_agent_kwargs_for_construction(cfg),
    )
    # Pin the LEAF, not just the mixin. Asserting only ``_GPTTeacherMixin`` would
    # still pass if the desktop pattern were ever broadened to cover ``@mobile@use``
    # and a mobile env silently got a desktop-based agent -- which would use the
    # desktop action space and coordinate frame on a phone.
    expected = GPTMobileTeacherAgent if "mobile" in md.dims else GPTTeacherAgent
    assert type(agent) is expected, f"{path.name}: dims={md.dims} -> {type(agent).__name__}"
