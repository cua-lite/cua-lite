"""Tests for lite.agents.factory.

Run:
    uv run pytest tests/agents/test_agents.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lite.agents.factory import AGENTS, API_AGENTS, LOCAL_AGENTS, make
from lite.agents.models.claude.agent import ClaudeDesktopUseAgent
from lite.agents.models.gemini.agent import GeminiDesktopUseAgent
from lite.agents.models.gpt.agent import GPTDesktopUseAgent
from lite.core import LiteCUAMetadata
from lite.core.tools.schemas import make_tool_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_env(
    platform: str = "desktop",
    task_type: str = "use",
    extra_tools: list | None = None,
    valid_actions: list | None = None,
):
    """Create a minimal mock env with metadata."""
    metadata = LiteCUAMetadata(
        dims=(platform, task_type),
        extra_tool_schemas=extra_tools or [],
        valid_actions=valid_actions,
    )
    env = MagicMock()
    env.metadata = metadata
    return env


# ---------------------------------------------------------------------------
# make — API path
# ---------------------------------------------------------------------------


class TestMakeAgentAPI:
    @pytest.mark.parametrize(
        "model", ["claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5"]
    )
    def test_claude_model_returns_claude_agent(self, model):
        env = _mock_env()
        agent = make(model, env=env)
        assert isinstance(agent, ClaudeDesktopUseAgent)
        assert agent.model_id == model

    @pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.5"])
    def test_gpt_model_returns_gpt_agent(self, model):
        env = _mock_env()
        agent = make(model, env=env)
        assert isinstance(agent, GPTDesktopUseAgent)
        assert agent.model_id == model

    def test_extra_tools_forwarded(self):
        # ``make`` forwards env.metadata wholesale into the agent
        # (no per-field copy) — adapters read ``self.metadata.extra_tool_schemas``
        # rather than a direct attribute on the agent.
        tools = [make_tool_schema("goto", parameters={})]
        env = _mock_env(extra_tools=tools)
        agent = make("claude-opus-4-6", env=env)
        assert agent.metadata.extra_tool_schemas == tools

    def test_valid_actions_absorbed_not_crash(self):
        """valid_actions reaches API agents through env metadata."""
        env = _mock_env(valid_actions=["click", "type"])
        agent = make("claude-opus-4-6", env=env)
        assert isinstance(agent, ClaudeDesktopUseAgent)

    def test_valid_actions_none_not_crash(self):
        env = _mock_env(valid_actions=None)
        agent = make("claude-opus-4-6", env=env)
        assert isinstance(agent, ClaudeDesktopUseAgent)

    def test_api_kwargs_forwarded(self):
        env = _mock_env()
        agent = make(
            "claude-opus-4-6", env=env, api_kwargs={"max_tokens": 2048, "temperature": 0.5}
        )
        assert agent.api_kwargs["max_tokens"] == 2048
        assert agent.api_kwargs["temperature"] == 0.5

    def test_unknown_model_raises(self):
        env = _mock_env()
        with pytest.raises(KeyError, match="Unknown model"):
            make("nonexistent-model", env=env)

    def test_api_agents_dict_coverage(self):
        """All API_AGENTS entries should produce valid agent instances."""
        for model in API_AGENTS:
            env = _mock_env()
            agent = make(model, env=env)
            assert isinstance(
                agent, (ClaudeDesktopUseAgent, GPTDesktopUseAgent, GeminiDesktopUseAgent)
            )

    def test_agents_includes_local_and_api(self):
        """AGENTS should be a superset of both LOCAL_AGENTS and API_AGENTS."""
        for key in LOCAL_AGENTS:
            assert key in AGENTS
        for key in API_AGENTS:
            assert key in AGENTS


# ---------------------------------------------------------------------------
# agent_kwargs.resolution — the declared tuple must survive a yaml list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["claude-opus-4-6", "gpt-5.5"])
def test_api_agent_resolution_is_a_tuple_after_a_yaml_list(model):
    """``agent_kwargs: {resolution: [W, H]}`` reaches a field declared
    ``tuple[int, int]``. It must be a tuple by the end of construction, or
    ``img.size == self.resolution`` is a tuple/list compare that never holds and
    every screenshot pays a same-size resample."""
    agent = make(model, env=_mock_env(), resolution=[1280, 720])
    assert type(agent.resolution) is tuple
    assert agent.resolution == (1280, 720)


def test_adapter_resolution_is_a_tuple_after_a_yaml_list():
    """Same for the adapter-based families, where the field is declared on
    ``BaseAgentAdapter`` and set by ``_apply_kwargs``."""
    from PIL import Image

    from lite.agents.bootstrap import register_all
    from lite.agents.core.adapter.base import AgentAdapterRegistry

    register_all()
    adapter = AgentAdapterRegistry.get("lite@desktop@use", resolution=[1280, 720])
    assert type(adapter.resolution) is tuple
    # The fast path the type lie disabled: an already-correct image is returned
    # unchanged instead of being LANCZOS-resampled to its own size.
    img = Image.new("RGB", (1280, 720))
    assert adapter.process_image(img) is img
