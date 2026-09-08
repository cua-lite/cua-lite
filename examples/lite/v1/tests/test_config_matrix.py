from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest
import yaml

from lite.agents.bootstrap import register_all
from lite.agents.core.agent.base import AgentRegistry, BaseAgent
from lite.agents.factory import AGENTS, LOCAL_AGENTS, make
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
)
from lite.utils.registry import compose_key

register_all()

_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_ROOT = _ROOT / "examples" / "lite" / "v1" / "configs"

# Every Lite v1 config is an env-less desktop.use recipe: two screenshot profiles
# (default = native, compact = 1280x720 agent-side downsample) x two reasoning modes
# (Action-only, native <think>). The env is chosen at rollout by --env-id / ENV_ID, and
# export_sft picks the adapter per row from the data's metadata.dims + agent_id.
_EXPECTED_CONFIGS = (
    "examples/lite/v1/configs/qwen3_5/desktop.use.compact.reasoning.yaml",
    "examples/lite/v1/configs/qwen3_5/desktop.use.compact.yaml",
    "examples/lite/v1/configs/qwen3_5/desktop.use.default.reasoning.yaml",
    "examples/lite/v1/configs/qwen3_5/desktop.use.default.yaml",
)

_DESKTOP_USE_DIMS = ("desktop", "use")

_ALLOWED_YAML_VALID_ACTIONS = (
    LiteDesktopActionSet.get_action_names()
    | LiteMobileActionSet.get_action_names()
    | LitePointActionSet.get_action_names()
)

_FAMILY_TO_MODEL: dict[str, str] = {}
for _model_id, _cfg in AGENTS.items():
    _FAMILY_TO_MODEL.setdefault(_cfg["agent_id"], _model_id)


class ConfigRow(NamedTuple):
    rel: str
    agent_id: str
    env_id: str | None
    agent_kwargs: dict
    env_kwargs: dict


def _rows() -> list[ConfigRow]:
    rows: list[ConfigRow] = []
    for path in sorted(_CONFIG_ROOT.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        assert isinstance(data, dict), path
        rows.append(
            ConfigRow(
                rel=path.relative_to(_ROOT).as_posix(),
                agent_id=data["agent_id"],
                env_id=data.get("env_id"),
                agent_kwargs=data.get("agent_kwargs") or {},
                env_kwargs=data.get("env_kwargs") or {},
            )
        )
    return rows


ROWS = _rows()


def _model_for(agent_id: str) -> str:
    family = agent_id.split(".")[0]
    return _FAMILY_TO_MODEL[family]


def _build_agent(row: ConfigRow) -> BaseAgent:
    model_id = _model_for(row.agent_id)
    kwargs = {}
    if model_id in LOCAL_AGENTS:
        kwargs = {"processor": MagicMock(), "generate_fn": lambda *a, **k: None}
    agent_kwargs = dict(row.agent_kwargs)
    # ``sampling_kwargs`` configures local serving/generate_fn, never the adapter.
    # Both real consumers drop it before construction -- rollout at
    # ``lite/infer/rollout.py`` and export at ``export_sft._adapter_kwargs_for_export``
    # -- so a config carrying it is valid and this builder must mirror them.
    agent_kwargs.pop("sampling_kwargs", None)
    env = SimpleNamespace(metadata=LiteCUAMetadata(dims=_DESKTOP_USE_DIMS))
    return make(model_id, env=env, agent_id=row.agent_id, **kwargs, **agent_kwargs)


def test_lite_v1_config_matrix_enumerates_every_example_yaml() -> None:
    assert [row.rel for row in ROWS] == list(_EXPECTED_CONFIGS)
    # env-less by design: one desktop.use recipe drives every desktop env.
    assert [row.rel for row in ROWS if row.env_id is None] == list(_EXPECTED_CONFIGS)


def test_lite_v1_config_valid_actions_are_known_actions() -> None:
    offenders: list[str] = []
    for row in ROWS:
        value = row.env_kwargs.get("valid_actions")
        if value is None:
            continue
        assert isinstance(value, list), f"{row.rel}: valid_actions must be list|null"
        unknown = sorted(set(value) - _ALLOWED_YAML_VALID_ACTIONS)
        if unknown:
            offenders.append(f"{row.rel}: unknown GUI actions {unknown}")
    assert not offenders, "\n".join(offenders)


def test_lite_v1_compact_configs_differ_from_default_only_by_resolution() -> None:
    """``compact`` is the ``default`` recipe plus one agent-side downsample.

    The four configs are a 2x2 (screenshot profile x reasoning mode), so a compact
    row must match its default twin everywhere except ``agent_kwargs.resolution``.
    Any other drift means the pair no longer isolates the VRAM knob.
    """
    by_rel = {row.rel: row for row in ROWS}
    for suffix in ("", ".reasoning"):
        default = by_rel[f"examples/lite/v1/configs/qwen3_5/desktop.use.default{suffix}.yaml"]
        compact = by_rel[f"examples/lite/v1/configs/qwen3_5/desktop.use.compact{suffix}.yaml"]
        assert compact.agent_id == default.agent_id
        assert compact.env_kwargs == default.env_kwargs
        assert "resolution" not in default.agent_kwargs
        assert compact.agent_kwargs["resolution"] == [1280, 720]
        stripped = {k: v for k, v in compact.agent_kwargs.items() if k != "resolution"}
        assert stripped == default.agent_kwargs


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.rel)
def test_lite_v1_config_matrix_constructs_desktop_use_agent(row: ConfigRow) -> None:
    key = compose_key(row.agent_id, *_DESKTOP_USE_DIMS)
    assert AgentRegistry.contains(key)
    assert isinstance(_build_agent(row), BaseAgent)
