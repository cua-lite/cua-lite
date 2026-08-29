"""Lock the agent FACTORY (``lite.agents.factory.make``) against the refactor.

Construct every entry in ``AGENTS`` (the full local + API catalog) through the
factory with no live model / no network: a refactor that breaks the
agent-key composition or drops a registration trips here. Also pins the
``{agent_id}@<metadata dims>`` key grammar used by the factory and goal-image
bridge.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/test_make_all.py -p no:cacheprovider -q
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.agent.base import BaseAgent
from lite.agents.factory import AGENTS, API_AGENTS, LOCAL_AGENTS, make
from lite.core import LiteCUAMetadata
from lite.utils.registry import compose_key

register_all()

# The LOCAL model_ids whose family is mobile-ONLY (agent_id in
# {mai_ui, step_gui}) — these must render at platform="mobile";
# any desktop lookup raises KeyError. Discovered from LOCAL_AGENTS.
_MOBILE_ONLY_AGENT_IDS = {"mai_ui", "step_gui"}
MOBILE_ONLY_MODEL_IDS = {
    model_id
    for model_id, cfg in AGENTS.items()
    if cfg["agent_id"] in _MOBILE_ONLY_AGENT_IDS
}


def _make_env(platform: str):
    """Minimal env stub exposing a REAL LiteCUAMetadata.

    A real LiteCUAMetadata (not SimpleNamespace) is required: adapters read
    ``self.metadata.others`` (and ``extra_tool_schemas`` / ``valid_actions``)
    in ``__post_init__``; a SimpleNamespace lacks those defaults and breaks.
    """
    meta = LiteCUAMetadata(dims=(LiteCUAMetadata.Platform(platform), LiteCUAMetadata.TaskType.USE))
    return SimpleNamespace(metadata=meta)


@pytest.mark.parametrize("model_id", sorted(AGENTS.keys()))
def test_make_agent_constructs_all(model_id: str) -> None:
    """Every AGENTS entry constructs a BaseAgent subclass without raising."""
    platform = "mobile" if model_id in MOBILE_ONLY_MODEL_IDS else "desktop"
    env = _make_env(platform)

    kwargs: dict = {}
    if model_id in LOCAL_AGENTS:
        # Local agents need a processor (apply_chat_template) + generate_fn.
        # Neither is called at construction, so mocks suffice.
        kwargs = {
            "processor": MagicMock(),
            "generate_fn": (lambda *a, **k: None),
        }
    # API agents (claude/gpt) need no extras — model_id is auto-forwarded.

    agent = make(model_id, env=env, **kwargs)
    assert isinstance(agent, BaseAgent), (
        f"make({model_id!r}) returned {type(agent).__name__}, "
        f"not a BaseAgent subclass"
    )


def test_agents_dicts_nonempty() -> None:
    """The catalog dicts are populated and AGENTS is exactly LOCAL ∪ API."""
    assert AGENTS, "AGENTS is empty"
    assert LOCAL_AGENTS, "LOCAL_AGENTS is empty"
    assert API_AGENTS, "API_AGENTS is empty"
    union = set(LOCAL_AGENTS) | set(API_AGENTS)
    assert set(AGENTS) >= union, (
        f"AGENTS missing entries from LOCAL ∪ API: {union - set(AGENTS)}"
    )
    # Sanity on the audited counts (32 = 23 LOCAL + 9 API).
    assert len(LOCAL_AGENTS) == 23
    assert len(API_AGENTS) == 9
    assert len(AGENTS) == 32


def test_compose_agent_key_format() -> None:
    """Pin the public key grammar used by factory and bridge callers."""
    dims = ("desktop", "use")

    assert compose_key("qwen3_vl", *dims) == "qwen3_vl@desktop@use"
    assert (
        compose_key("qwen3_vl.base", *dims)
        == "qwen3_vl.base@desktop@use"
    )

    def _compose_if_needed(adapter_key: str, dims: tuple[str, ...]) -> str:
        if "@" not in adapter_key:
            return compose_key(adapter_key, *dims)
        return adapter_key

    assert _compose_if_needed("qwen3_vl", dims) == "qwen3_vl@desktop@use"
    assert _compose_if_needed("qwen3_vl.base", dims) == "qwen3_vl.base@desktop@use"
    assert _compose_if_needed("qwen3_vl@browser@use", dims) == "qwen3_vl@browser@use"
