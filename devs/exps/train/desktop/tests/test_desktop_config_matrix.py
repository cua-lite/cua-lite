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
from lite.agents.models.qwen3_5.protocol import Qwen3_5HistoryProtocol
from lite.core.metadata import LiteCUAMetadata
from lite.utils.registry import compose_key

register_all()

_ROOT = Path(__file__).resolve().parents[5]
_REL = "devs/exps/train/desktop/configs/qwen3_5"
_CONFIG_ROOT = _ROOT / "devs" / "exps" / "train" / "desktop" / "configs"

# Every config here is an env-less desktop.use recipe varying screenshot size, how
# much history reaches the model, and reasoning mode. Two DIFFERENT caps produce the
# "one image" profiles and they are not interchangeable:
#   ``hN``  caps TURNS   (``history_n``)  -- older turns collapse into the
#           ``Previous actions:`` prose summary, so their tool calls and <think> are gone.
#   ``iN``  caps IMAGES  (``image_max``, ``fold_size`` tracking it) -- history_n stays 100,
#           so every past turn is still rendered in full, screenshots aside.
# ``default`` is native resolution on the protocol's own defaults (100 / 4 / 4). Six of the
# twelve are NOT campaign cells: ``default`` and both ``h1`` profiles, with their twins. Each of the six profiles has a
# ``.reasoning`` twin that adds ``enable_thinking`` and nothing else -- reading a reasoning
# run against its twin is what isolates the <think> channel. The env is chosen at rollout
# by --env-id / ENV_ID, and export_sft picks the adapter per row from metadata.dims.
_PROFILES = ("default", "highr.h1", "highr.i1", "lowr.h1", "lowr.i1", "lowr.i4")
# sorted(), not the _PROFILES order: ROWS comes from sorted(rglob(...)), and matching it by
# construction order would silently require _PROFILES to be lexicographic -- an invariant
# nothing states, that a new profile appended to the tuple breaks, and whose failure reads
# as "wrong config filename at index N" rather than "tuple out of order".
_EXPECTED_CONFIGS = tuple(sorted(
    f"devs/exps/train/desktop/configs/qwen3_5/desktop.use.{_p}{_suffix}.yaml"
    for _p in _PROFILES
    for _suffix in (".reasoning", "")
))

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


def test_lowr_i4_differs_from_default_only_by_resolution() -> None:
    """``lowr.i4`` is the ``default`` recipe plus one agent-side downsample.

    ``i4`` names ``image_max``, so the config PINS ``image_max``/``fold_size`` rather
    than inheriting them -- otherwise a change to the protocol default would silently
    rename the profile. The pin must therefore restate the defaults exactly: pinning a
    DIFFERENT value would make this a two-knob change against ``default`` while still
    reading as "default plus a downsample".
    """
    by_rel = {row.rel: row for row in ROWS}
    default = by_rel[f"{_REL}/desktop.use.default.yaml"]
    lowr = by_rel[f"{_REL}/desktop.use.lowr.i4.yaml"]
    assert lowr.agent_id == default.agent_id
    assert lowr.env_kwargs == default.env_kwargs
    assert "resolution" not in default.agent_kwargs
    assert lowr.agent_kwargs["resolution"] == [1280, 720]

    # The pin equals what ``default`` inherits, so the two run the same protocol.
    proto = Qwen3_5HistoryProtocol()
    pinned = lowr.agent_kwargs["protocol_kwargs"]
    assert pinned == {"image_max": 4, "fold_size": 4}
    assert pinned["image_max"] == proto.image_max, "pin drifted from the protocol default"
    assert pinned["fold_size"] == proto.fold_size, "pin drifted from the protocol default"
    assert "history_n" not in pinned, "i4 caps images, not turns -- history_n stays default"
    assert "protocol_kwargs" not in default.agent_kwargs

    stripped = {k: v for k, v in lowr.agent_kwargs.items()
                if k not in ("resolution", "protocol_kwargs")}
    assert stripped == default.agent_kwargs


def test_highr_h1_differs_from_default_only_by_history_depth() -> None:
    """``highr.h1`` is the ``default`` recipe with a one-turn history window.

    The counterpart of the ``lowr.i4`` pairing above, on the other axis: same native
    resolution, same tool surface, only ``protocol_kwargs.history_n`` moves. Its
    one-axis partner is ``lowr.h1``, not ``lowr.i4`` -- highr.h1-vs-lowr.i4 moves
    resolution AND history together, which is the confound ``lowr.h1`` was added to
    remove.
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


@pytest.mark.parametrize("res", ["lowr", "highr"])
def test_h1_and_i1_differ_only_by_which_cap_is_set(res: str) -> None:
    """At one resolution, ``h1`` and ``i1`` show ONE image each and differ only in cap.

    Both render a single screenshot, so this pair does not move the image axis at all --
    it moves what the model is told about everything OLDER: ``h1`` gets prose summary
    lines built from ``action_description``, ``i1`` gets each past turn rendered in full
    (its literal ``<tool_call>``, and on the reasoning arm its ``<think>``). If a second
    field ever drifts, that measurement silently becomes a two-variable one.
    """
    by_rel = {row.rel: row for row in ROWS}
    h1 = by_rel[f"{_REL}/desktop.use.{res}.h1.yaml"]
    i1 = by_rel[f"{_REL}/desktop.use.{res}.i1.yaml"]
    assert h1.agent_id == i1.agent_id
    assert h1.env_kwargs == i1.env_kwargs
    assert h1.agent_kwargs.get("resolution") == i1.agent_kwargs.get("resolution")
    assert h1.agent_kwargs["protocol_kwargs"] == {"history_n": 1}
    assert i1.agent_kwargs["protocol_kwargs"] == {"image_max": 1, "fold_size": 1}
    strip = lambda r: {k: v for k, v in r.agent_kwargs.items() if k != "protocol_kwargs"}
    assert strip(h1) == strip(i1)


@pytest.mark.parametrize("cap", ["h1", "i1"])
def test_lowr_and_highr_of_one_cap_differ_only_by_resolution(cap: str) -> None:
    """The resolution axis, measured twice -- once under each history cap.

    Two independent clean pairs are the point: a resolution effect that shows up under
    ``h1`` but not ``i1`` (or vice versa) is a real interaction, not noise, and that is
    only readable while each pair moves resolution ALONE.
    """
    by_rel = {row.rel: row for row in ROWS}
    lowr = by_rel[f"{_REL}/desktop.use.lowr.{cap}.yaml"]
    highr = by_rel[f"{_REL}/desktop.use.highr.{cap}.yaml"]
    assert lowr.agent_id == highr.agent_id
    assert lowr.env_kwargs == highr.env_kwargs
    assert lowr.agent_kwargs["resolution"] == [1280, 720]
    assert "resolution" not in highr.agent_kwargs, "highr keeps the env's native size"
    strip = lambda r: {k: v for k, v in r.agent_kwargs.items() if k != "resolution"}
    assert strip(lowr) == strip(highr)


def test_lowr_i1_and_i4_differ_only_by_the_image_cap() -> None:
    """The image axis: same uncapped text history, 1 screenshot against up to 4.

    Both leave ``history_n`` at the protocol default, so the text the model sees is the
    same shape in each; only ``image_max`` moves. ``fold_size`` must track ``image_max``
    -- below it the folded prefix advances EVERY step, so adjacent steps stop being
    prefix-compatible and ``build_segment_samples`` can no longer pack them. (The visible
    count gets steadier, not deeper: at ``fold_size: 1`` it pins to ``image_max``. Steadier
    is what costs more -- every segment then carries the full image_max.)
    """
    by_rel = {row.rel: row for row in ROWS}
    i1 = by_rel[f"{_REL}/desktop.use.lowr.i1.yaml"]
    i4 = by_rel[f"{_REL}/desktop.use.lowr.i4.yaml"]
    assert i1.agent_kwargs.get("resolution") == i4.agent_kwargs.get("resolution")
    assert i1.env_kwargs == i4.env_kwargs
    for row, n in ((i1, 1), (i4, 4)):
        pk = row.agent_kwargs["protocol_kwargs"]
        assert pk == {"image_max": n, "fold_size": n}, row.rel
        assert "history_n" not in pk, f"{row.rel}: an i-profile caps images, not turns"


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
