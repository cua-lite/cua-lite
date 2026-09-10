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
from lite.utils.registry import compose_key

register_all()

_ROOT = Path(__file__).resolve().parents[5]
_REL = "devs/exps/train/desktop/configs/qwen3_5"
_CONFIG_ROOT = _ROOT / "devs" / "exps" / "train" / "desktop" / "configs"

# Every config here is an env-less desktop.use recipe varying screenshot size,
# history depth, and reasoning mode. ``default`` is native resolution with the
# protocol's default 4-image window; ``lowr.h4`` downsamples to 1280x720 and keeps
# the same 4 images; ``highr.h1`` keeps native resolution and one image. Each of the
# three has a ``.reasoning`` twin that adds ``enable_thinking`` and nothing else --
# reading a reasoning run against its twin is what isolates the <think> channel. The
# env is chosen at rollout by --env-id / ENV_ID, and export_sft picks the adapter per
# row from the data's metadata.dims + agent_id.
_PROFILES = ("default", "highr.h1", "lowr.h4")
_EXPECTED_CONFIGS = tuple(
    f"devs/exps/train/desktop/configs/qwen3_5/desktop.use.{_p}{_suffix}.yaml"
    for _p in _PROFILES
    for _suffix in (".reasoning", "")
)

_DESKTOP_USE_DIMS = ("desktop", "use")

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


def test_no_desktop_config_trims_the_action_space() -> None:
    """Every config exposes the FULL action space; trimming one is a prompt-surface change.

    `valid_actions` is what narrows the rendered action enum, so a config that sets it trains
    and evaluates on a different surface than its siblings — which would silently break the
    twin comparisons this suite exists to protect. Nothing sets it today; this fails the
    moment something does, so the change has to be argued for rather than slipped in.
    """
    trimmed = {row.rel: row.env_kwargs["valid_actions"]
               for row in ROWS if row.env_kwargs.get("valid_actions") is not None}
    assert trimmed == {}, (
        f"a config now trims valid_actions: {trimmed}. If that is intended, replace this test "
        f"with one checking each trimmed list against the action sets in "
        f"lite.core.tools.action_space, and say in the config header why that cell's surface "
        f"differs from its twin's."
    )


def test_every_desktop_config_declares_the_finish_tools() -> None:
    """The env_kwargs every config shares, none of which was pinned here before.

    ``extra_tools``'s own yaml comment calls it "Required, not cosmetic: without it the env
    rejects both as inactive extras and the model can never end an episode." ``max_steps``
    and ``loop_detect`` must match too, or the twin comparisons run different episodes.
    """
    for row in ROWS:
        assert row.env_kwargs.get("extra_tools") == ["response", "terminate"], row.rel
        assert row.env_kwargs.get("max_steps") == 30, row.rel
        assert row.env_kwargs.get("loop_detect") == 5, row.rel


def test_lowr_configs_differ_from_default_only_by_resolution() -> None:
    """``lowr.h4`` is the ``default`` recipe plus one agent-side downsample.

    It must match ``default`` everywhere except ``agent_kwargs.resolution``. Any other
    drift means the pair no longer isolates the VRAM knob.
    """
    by_rel = {row.rel: row for row in ROWS}
    default = by_rel[f"{_REL}/desktop.use.default.yaml"]
    lowr = by_rel[f"{_REL}/desktop.use.lowr.h4.yaml"]
    assert lowr.agent_id == default.agent_id
    assert lowr.env_kwargs == default.env_kwargs
    assert "resolution" not in default.agent_kwargs
    assert lowr.agent_kwargs["resolution"] == [1280, 720]
    stripped = {k: v for k, v in lowr.agent_kwargs.items() if k != "resolution"}
    assert stripped == default.agent_kwargs


def test_highr_h1_differs_from_default_only_by_history_depth() -> None:
    """``highr.h1`` is the ``default`` recipe with a one-turn history window.

    The counterpart of the ``lowr.h4`` pairing above, on the other axis: same native
    resolution, same tool surface, only ``protocol_kwargs.history_n`` moves. That is
    what makes a highr.h1-vs-lowr.h4 comparison isolate "detail" against "history"
    rather than confounding both with a third difference.
    """
    by_rel = {row.rel: row for row in ROWS}
    default = by_rel[f"{_REL}/desktop.use.default.yaml"]
    highr = by_rel[f"{_REL}/desktop.use.highr.h1.yaml"]
    assert highr.agent_id == default.agent_id
    assert highr.env_kwargs == default.env_kwargs
    assert "resolution" not in highr.agent_kwargs, "highr keeps the env's native size"
    assert highr.agent_kwargs["protocol_kwargs"] == {"history_n": 1}
    assert "protocol_kwargs" not in default.agent_kwargs
    stripped = {k: v for k, v in highr.agent_kwargs.items() if k != "protocol_kwargs"}
    assert stripped == default.agent_kwargs


@pytest.mark.parametrize("profile", _PROFILES)
def test_each_reasoning_twin_differs_only_by_enable_thinking(profile: str) -> None:
    """A ``.reasoning`` config is its Action-only twin plus ``enable_thinking``.

    The campaign reads a reasoning checkpoint against the Action-only cell of the SAME
    profile, so that pair is the whole measurement. A second difference -- a resolution,
    a history depth, one extra tool -- would silently turn the <think> result into a
    two-variable one, and nothing downstream would notice.
    """
    by_rel = {row.rel: row for row in ROWS}
    plain = by_rel[f"{_REL}/desktop.use.{profile}.yaml"]
    thinking = by_rel[f"{_REL}/desktop.use.{profile}.reasoning.yaml"]
    assert thinking.agent_id == plain.agent_id
    assert thinking.env_kwargs == plain.env_kwargs
    assert thinking.agent_kwargs.get("enable_thinking") is True
    assert "enable_thinking" not in plain.agent_kwargs, "the twin carries the only flag"
    stripped = {k: v for k, v in thinking.agent_kwargs.items() if k != "enable_thinking"}
    assert stripped == plain.agent_kwargs


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.rel)
def test_lite_v1_config_matrix_constructs_desktop_use_agent(row: ConfigRow) -> None:
    key = compose_key(row.agent_id, *_DESKTOP_USE_DIMS)
    assert AgentRegistry.contains(key)
    assert isinstance(_build_agent(row), BaseAgent)
