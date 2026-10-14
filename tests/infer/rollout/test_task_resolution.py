"""Characterization tests for the rollout task-resolution layer (``TaskSpec``).

Pins the EXACT behavior of task discovery + on-disk layout + resume across the
three sources (parquet / registry / single-task). The expected values document
the current ``TaskSpec`` contract for task identity, split handling, limits,
and result paths.

Run: uv run pytest tests/infer/rollout/test_task_resolution.py -q
"""
from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import pytest

from lite.core import (
    LiteCUAMetadata,
    LiteRLSample,
)
from lite.gym.base import LiteBaseEnv
from lite.gym.registry import _env_supported_kwargs, _imported, _specs, _splits, register, registry
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.infer.rollout import (
    TaskSpec,
    _make_result,
    _resolve_run_tasks,
    collect_tasks,
    get_pending,
    print_results,
    rebuild_results,
    resolve_prompt_data_tasks,
    run_rollout,
)
from lite.utils.parquet import write_records_to_parquet

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _parquet(tmp_path: Path, rows: list[dict]) -> str:
    p = tmp_path / "tasks.parquet"
    write_records_to_parquet([{"problem": "x", "metadata": m} for m in rows], p)
    return str(p)


def _units(specs: list[TaskSpec]) -> list[tuple]:
    return [(s.task_id, s.env_id, s.split, dict(s.env_kwargs)) for s in specs]


def _resolve(env_id="webgym", **kw):
    kw.setdefault("prompt_data", None)
    kw.setdefault("splits", None)
    kw.setdefault("head", None)
    kw.setdefault("sample", None)
    kw.setdefault("filter_expr", None)
    kw.setdefault("rng", random.Random(1))
    kw.setdefault("task_id", None)
    return _resolve_run_tasks(env_id, **kw)


# --------------------------------------------------------------------------- #
# parquet source — split honoring + fallback (the (b) behavior, pinned)
# --------------------------------------------------------------------------- #

def test_parquet_split_mix(tmp_path):
    specs = _resolve(prompt_data=_parquet(tmp_path, [
        {"env_key": "webgym@a", "split": "train"},
        {"env_key": "webgym@b", "split": "eval"},
        {"env_key": "webgym@c", "split": "train"},
    ]))
    assert _units(specs) == [
        ("a", "webgym", "train", {}),
        ("b", "webgym", "eval", {}),
        ("c", "webgym", "train", {}),
    ]


def test_parquet_no_split_falls_back_to_parquet(tmp_path):
    specs = _resolve(prompt_data=_parquet(tmp_path, [
        {"env_key": "webgym@a"}, {"env_key": "webgym@b"},
    ]))
    assert _units(specs) == [
        ("a", "webgym", "parquet", {}),
        ("b", "webgym", "parquet", {}),
    ]


def test_parquet_partial_split_is_per_row(tmp_path):
    specs = _resolve(prompt_data=_parquet(tmp_path, [
        {"env_key": "webgym@a", "split": "train"},
        {"env_key": "webgym@b"},
    ]))
    assert _units(specs) == [
        ("a", "webgym", "train", {}),
        ("b", "webgym", "parquet", {}),  # only the split-less row falls back
    ]


def test_parquet_head_slices_raw_rows_before_dedup(tmp_path):
    # head=2 keeps the first 2 RAW rows [a, a] → dedup → [a]. (Not "first 2
    # distinct tasks".) env_kwargs survives (int 7 → float via pyarrow).
    specs = _resolve(head=2, prompt_data=_parquet(tmp_path, [
        {"env_key": "webgym@a", "split": "train", "env_kwargs": {"max_steps": 7}},
        {"env_key": "webgym@a", "split": "train", "env_kwargs": {"max_steps": 7}},
        {"env_key": "webgym@b", "split": "train"},
    ]))
    assert _units(specs) == [("a", "webgym", "train", {"max_steps": 7.0})]


def test_parquet_sample_is_seed_deterministic(tmp_path):
    rows = [{"env_key": f"webgym@t{i}", "split": "train"} for i in range(10)]
    p = _parquet(tmp_path, rows)
    a = _resolve(sample=4, prompt_data=p, rng=random.Random(1))
    b = _resolve(sample=4, prompt_data=p, rng=random.Random(1))
    assert [s.task_id for s in a] == [s.task_id for s in b]
    assert len(a) == 4


def test_parquet_sample_equal_to_len_shuffles_whole_set(tmp_path):
    # sample == len(rows) shuffles the full set (same semantics as the
    # registry path in collect_tasks) instead of silently no-opping.
    rows = [{"env_key": f"webgym@t{i}", "split": "train"} for i in range(10)]
    p = _parquet(tmp_path, rows)
    shuffled = _resolve(sample=10, prompt_data=p, rng=random.Random(1))
    plain = _resolve(prompt_data=p)
    assert sorted(s.task_id for s in shuffled) == sorted(s.task_id for s in plain)
    assert [s.task_id for s in shuffled] != [s.task_id for s in plain]


def test_parquet_dedup_conflict_env_id_raises(tmp_path):
    with pytest.raises(ValueError, match="both env_ids"):
        _resolve(env_id="x", prompt_data=_parquet(tmp_path, [
            {"env_key": "x@dup"}, {"env_key": "y@dup"},
        ]))


def test_parquet_dedup_conflict_env_kwargs_raises(tmp_path):
    with pytest.raises(ValueError, match="conflicting env_kwargs"):
        _resolve(prompt_data=_parquet(tmp_path, [
            {"env_key": "webgym@a", "env_kwargs": {"max_steps": 5}},
            {"env_key": "webgym@a", "env_kwargs": {"max_steps": 9}},
        ]))


def test_parquet_env_divergence_warns(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        resolve_prompt_data_tasks(
            _parquet(tmp_path, [{"env_key": "otherenv@a", "split": "train"}]),
            effective_env_id="webgym", rng=random.Random(1),
        )
    assert any("prompt_data carries" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# single-task source
# --------------------------------------------------------------------------- #

def test_single_task_uses_task_split(tmp_path):
    specs = _resolve(task_id="solo")
    assert _units(specs) == [("solo", "webgym", "task", {})]


def test_single_task_rejects_other_task_sources(tmp_path):
    with pytest.raises(ValueError, match="--task-id cannot be combined"):
        _resolve(task_id="solo", prompt_data=_parquet(tmp_path, [{"env_key": "webgym@a"}]))
    with pytest.raises(ValueError, match="--task-id pins one task"):
        _resolve(task_id="solo", splits=["train"])


def test_prompt_data_rejects_registry_filters(tmp_path):
    prompt_data = _parquet(tmp_path, [{"env_key": "webgym@a"}])
    with pytest.raises(ValueError, match="--splits cannot be used"):
        _resolve(prompt_data=prompt_data, splits=["train"])
    with pytest.raises(ValueError, match="--filter cannot be used"):
        _resolve(prompt_data=prompt_data, filter_expr="lambda m: True")


# --------------------------------------------------------------------------- #
# registry source (reuse the _test env fixture pattern from test_task_filter)
# --------------------------------------------------------------------------- #

@pytest.fixture
def reg_env():
    for k in [k for k in _specs if k.startswith("_tr@")]:
        del _specs[k]
    _splits.pop("_tr", None)
    tasks = [
        ("t1", "train", {"difficulty": 1}),
        ("t2", "train", {"difficulty": 3}),
        ("t3", "train", {"difficulty": 5}),
        ("e1", "eval", {"difficulty": 1}),
    ]
    for tid, split, others in tasks:
        register(key=f"_tr@{tid}", entry_point=lambda **kw: None, split=split,
                 metadata=LiteCUAMetadata(dims=("browser", "use"), others=others))
    yield "_tr"
    for k in [k for k in _specs if k.startswith("_tr@")]:
        del _specs[k]
    _splits.pop("_tr", None)


def test_registry_specs_carry_split(reg_env):
    specs = _resolve(env_id=reg_env, splits=["train"])
    assert {s.task_id for s in specs} == {"t1", "t2", "t3"}
    assert all(s.split == "train" and s.env_id == "_tr" and s.env_kwargs == {} for s in specs)


def test_registry_filter_then_head(reg_env):
    specs = _resolve(env_id=reg_env, splits=["train"], head=2,
                     filter_expr="lambda m: m.others.get('difficulty', 0) <= 3")
    # filter → {t1, t2}, then head=2 → both (filter runs BEFORE head)
    assert {s.task_id for s in specs} == {"t1", "t2"}


def test_registry_filter_that_removes_every_task_returns_empty(reg_env):
    _, tasks = collect_tasks(
        reg_env,
        splits=["train"],
        filter_fn=lambda _metadata: False,
    )
    assert tasks == []
    assert _resolve(
        env_id=reg_env,
        splits=["train"],
        filter_expr="lambda _metadata: False",
    ) == []


def test_run_rollout_empty_filter_writes_summary(reg_env, tmp_path):
    all_done, log_root = asyncio.run(run_rollout(
        model_id="gpt-5.5",
        model_path="gpt-5.5",
        env_id=reg_env,
        agent_kwargs={},
        env_kwargs={},
        seed=1,
        splits=["train"],
        filter_expr="lambda _metadata: False",
        log_root=tmp_path / "empty-filter",
    ))

    assert all_done is True
    summary = json.loads((log_root / "summary.json").read_text())
    assert summary["stats"]["num_tasks"] == 0
    assert summary["stats"]["num_samples"] == 0
    assert summary["stats"]["num_valid"] == 0
    assert summary["tasks"] == []


def test_run_rollout_rejects_head_and_sample_together(reg_env, tmp_path):
    with pytest.raises(ValueError, match="--head and --sample"):
        asyncio.run(run_rollout(
            model_id="gpt-5.5",
            model_path="gpt-5.5",
            env_id=reg_env,
            agent_kwargs={},
            env_kwargs={},
            seed=1,
            head=1,
            sample=1,
            log_root=tmp_path / "bad-limit",
        ))


class _DoneAgent:
    async def sample(self, env, hooks=()):
        from lite.core import LiteSample

        sample = LiteRLSample(
            lite_sample=LiteSample(
                metadata=env.metadata,
                images=[],
                messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            ),
            processed_images=[],
            steps=[],
            episode_return=1.0,
            terminated=True,
            truncated=False,
        )
        for hook in hooks:
            hook.on_complete(sample)
        return sample


class _SeedProbeEnv(LiteBaseEnv):
    seen_kwargs: list[dict] = []
    reject_seed = False

    def __init__(self, **kwargs):
        if self.reject_seed and "seed" in kwargs:
            raise TypeError("seed is not accepted")
        type(self).seen_kwargs.append(dict(kwargs))

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(dims=("browser", "use"), others={})

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(text="ready")

    async def step(self, actions) -> LiteEnvStepResult:
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


def _cleanup_seed_env(env_id: str) -> None:
    for key in [k for k in _specs if k.startswith(f"{env_id}@")]:
        del _specs[key]
    _splits.pop(env_id, None)
    _imported.pop(env_id, None)
    _env_supported_kwargs.pop(env_id, None)


def test_group_shared_seed_is_only_injected_for_seed_capable_env(tmp_path, monkeypatch):
    import lite.agents.factory as agent_factory

    env_id = "_seedcap"
    _cleanup_seed_env(env_id)
    _SeedProbeEnv.seen_kwargs = []
    _SeedProbeEnv.reject_seed = False
    _imported[env_id] = "local"
    registry.set_env_supported_kwargs(env_id, {"seed"})
    register(
        key=f"{env_id}@a",
        entry_point=_SeedProbeEnv,
        split="train",
        metadata=LiteCUAMetadata(dims=("browser", "use"), others={}),
    )
    monkeypatch.setattr(agent_factory, "make", lambda model_id, *, env, **kw: _DoneAgent())

    try:
        all_done, _ = asyncio.run(run_rollout(
            model_id="gpt-5.5",
            model_path="gpt-5.5",
            env_id=env_id,
            agent_kwargs={},
            env_kwargs={},
            seed=123,
            splits=["train"],
            group_size=2,
            log_root=tmp_path / "seedcap",
        ))
    finally:
        _cleanup_seed_env(env_id)

    assert all_done is True
    seeds = [kwargs["seed"] for kwargs in _SeedProbeEnv.seen_kwargs]
    assert len(seeds) == 2
    assert len(set(seeds)) == 1


def test_sampling_kwargs_are_not_forwarded_to_agent_factory(tmp_path, monkeypatch):
    import lite.agents.factory as agent_factory

    env_id = "_samplingkw"
    seen_agent_kwargs: list[dict] = []
    _cleanup_seed_env(env_id)
    _SeedProbeEnv.seen_kwargs = []
    _SeedProbeEnv.reject_seed = False
    _imported[env_id] = "local"
    register(
        key=f"{env_id}@a",
        entry_point=_SeedProbeEnv,
        split="train",
        metadata=LiteCUAMetadata(dims=("browser", "use"), others={}),
    )

    def _make(_model_id, *, env, **kwargs):
        del env
        seen_agent_kwargs.append(dict(kwargs))
        return _DoneAgent()

    monkeypatch.setattr(agent_factory, "make", _make)

    try:
        all_done, log_root = asyncio.run(run_rollout(
            model_id="gpt-5.5",
            model_path="gpt-5.5",
            env_id=env_id,
            agent_kwargs={
                "sampling_kwargs": {"temperature": 0.6},
                "api_kwargs": {"reasoning_effort": "low"},
            },
            env_kwargs={},
            seed=123,
            splits=["train"],
            group_size=1,
            log_root=tmp_path / "samplingkw",
        ))
    finally:
        _cleanup_seed_env(env_id)

    assert all_done is True
    assert seen_agent_kwargs == [{
        "agent_id": None,
        "api_kwargs": {"reasoning_effort": "low"},
    }]
    info = (log_root / "run_info.txt").read_text()
    agent_kwargs_line = next(line for line in info.splitlines() if line.startswith("agent_kwargs:"))
    assert "sampling_kwargs" not in agent_kwargs_line


def test_group_shared_seed_skips_strict_env_without_seed_capability(tmp_path, monkeypatch):
    import lite.agents.factory as agent_factory

    env_id = "_noseedcap"
    _cleanup_seed_env(env_id)
    _SeedProbeEnv.seen_kwargs = []
    _SeedProbeEnv.reject_seed = True
    _imported[env_id] = "local"
    register(
        key=f"{env_id}@a",
        entry_point=_SeedProbeEnv,
        split="train",
        metadata=LiteCUAMetadata(dims=("browser", "use"), others={}),
    )
    monkeypatch.setattr(agent_factory, "make", lambda model_id, *, env, **kw: _DoneAgent())

    try:
        all_done, _ = asyncio.run(run_rollout(
            model_id="gpt-5.5",
            model_path="gpt-5.5",
            env_id=env_id,
            agent_kwargs={},
            env_kwargs={},
            seed=123,
            splits=["train"],
            group_size=1,
            log_root=tmp_path / "noseedcap",
        ))
    finally:
        _SeedProbeEnv.reject_seed = False
        _cleanup_seed_env(env_id)

    assert all_done is True
    assert "seed" not in _SeedProbeEnv.seen_kwargs[0]


def test_registry_eval_split(reg_env):
    specs = _resolve(env_id=reg_env, splits=["eval"])
    assert _units(specs) == [("e1", "_tr", "eval", {})]


# --------------------------------------------------------------------------- #
# TaskSpec on-disk layout
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("split,expected", [
    ("train", "lr/train/a/sample_00/summary.json"),
    ("", "lr/a/sample_01/summary.json"),
    ("parquet", "lr/parquet/a/sample_00/summary.json"),
])
def test_summary_path_layout(split, expected):
    idx = 1 if split == "" else 0
    got = TaskSpec("a", "webgym", split).summary_path(Path("lr"), idx)
    assert got == Path(expected)


def test_env_key():
    assert TaskSpec("123", "webgym").env_key == "webgym@123"


# --------------------------------------------------------------------------- #
# rollout summary reporting
# --------------------------------------------------------------------------- #

def test_grpo_summary_does_not_count_all_error_groups_as_all_zero_and_all_one(capsys):
    specs = [
        TaskSpec("err", "webgym", "train"),
        TaskSpec("zero", "webgym", "train"),
        TaskSpec("one", "webgym", "train"),
        TaskSpec("mixed", "webgym", "train"),
    ]
    results = [
        _make_result("err", 0, 0, error="reset failed"),
        _make_result("err", 0, 1, error="reset failed"),
        _make_result("zero", 1, 0, episode_return=0.0, stop_reason="parse_failure"),
        _make_result("zero", 1, 1, episode_return=0.0, stop_reason="content_only_final"),
        _make_result("one", 2, 0, episode_return=1.0),
        _make_result("one", 2, 1, episode_return=1.0),
        _make_result("mixed", 3, 0, episode_return=0.0),
        _make_result("mixed", 3, 1, episode_return=1.0),
    ]

    stats = print_results(results, specs, group_size=2)

    out = capsys.readouterr().out
    assert "All-zero groups: 1/4" in out
    assert "All-one groups:  1/4" in out
    assert "Mixed groups:    1/4" in out
    assert stats["num_valid"] == 6
    assert stats["groups_with_variance"] == 1
    assert stats["stop_reasons"] == {"content_only_final": 1, "parse_failure": 1}
    assert "Stop reasons: content_only_final=1, parse_failure=1" in out


# --------------------------------------------------------------------------- #
# resume / rebuild over specs
# --------------------------------------------------------------------------- #

def _write_summary(
    spec: TaskSpec,
    log_root: Path,
    sample_idx: int,
    ret: float,
    *,
    stop_reason: str | None = None,
) -> None:
    d = spec.sample_dir(log_root, sample_idx)
    d.mkdir(parents=True, exist_ok=True)
    summary = {"n_turns": 3, "episode_return": ret, "terminated": True, "truncated": False}
    if stop_reason is not None:
        summary["stop_reason"] = stop_reason
    (d / "summary.json").write_text(json.dumps(summary))


def test_run_rollout_completed_resume_writes_top_level_summary(reg_env, tmp_path):
    log_root = tmp_path / "completed"
    _write_summary(TaskSpec("t1", reg_env, "task"), log_root, 0, 0.75)

    all_done, resolved = asyncio.run(run_rollout(
        model_id="gpt-5.5",
        model_path="gpt-5.5",
        env_id=reg_env,
        task_id="t1",
        agent_kwargs={},
        env_kwargs={},
        seed=1,
        log_root=log_root,
    ))

    assert all_done is True
    assert resolved == log_root
    summary = json.loads((log_root / "summary.json").read_text())
    assert summary["stats"]["num_tasks"] == 1
    assert summary["stats"]["num_valid"] == 1
    assert summary["tasks"] == [{
        "task": "t1",
        "env_id": reg_env,
        "num_samples": 1,
        "num_valid": 1,
        "episode_returns": [0.75],
        "mean_episode_return": 0.75,
        "has_variance": False,
    }]


def test_get_pending_and_rebuild(tmp_path):
    specs = [TaskSpec("a", "webgym", "train"), TaskSpec("b", "webgym", "train")]
    _write_summary(specs[0], tmp_path, 0, 1.0, stop_reason="parse_failure")  # a done, b pending

    pending = get_pending(tmp_path, specs, group_size=1)
    assert pending == [(specs[1], 0)]  # only b

    results = rebuild_results(tmp_path, specs, group_size=1)
    by_task = {r["task"]: r for r in results}
    assert by_task["a"]["episode_return"] == 1.0
    # A parse failure is a measurement, not an error: the env scored the turn,
    # so the summary's episode_return stands and only ``error`` marks a row bad.
    assert by_task["a"]["error"] is None
    assert by_task["a"]["env_id"] == "webgym" and by_task["a"]["turns"] == 3
    assert by_task["a"]["stop_reason"] == "parse_failure"
    assert by_task["b"]["error"] == "unresolved"


def test_run_rollout_existing_completed_root_rewrites_top_level_summary(
    reg_env, tmp_path, monkeypatch,
):
    import lite.agents.factory as agent_factory

    spec = TaskSpec("t1", reg_env, "task")
    log_root = tmp_path / "completed"
    _write_summary(spec, log_root, 0, 0.75)
    monkeypatch.setattr(
        agent_factory,
        "make",
        lambda *args, **kwargs: pytest.fail("agent should not be constructed"),
    )

    all_done, resolved = asyncio.run(run_rollout(
        model_id="gpt-5.5",
        model_path="gpt-5.5",
        env_id=reg_env,
        task_id="t1",
        agent_kwargs={},
        env_kwargs={},
        seed=1,
        log_root=log_root,
    ))

    assert all_done is True
    assert resolved == log_root
    summary = json.loads((log_root / "summary.json").read_text())
    assert summary["stats"]["num_tasks"] == 1
    assert summary["stats"]["num_samples"] == 1
    assert summary["stats"]["num_valid"] == 1
    assert summary["tasks"][0]["episode_returns"] == [0.75]


def test_terminal_error_summary_is_resolved_not_pending(tmp_path):
    """A non-retryable error writes a terminal ``summary.json`` carrying an
    ``error`` key. ``get_pending`` must then treat it as RESOLVED (never re-run by
    the ``--max-attempts`` loop), and ``rebuild_results`` must surface that error
    (excluded from ``valid``) — distinct from a *missing* summary (``unresolved``)."""
    spec = TaskSpec("blocked", "webgym", "train")
    d = spec.sample_dir(tmp_path, 0)
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "n_turns": 0, "episode_return": 0.0, "terminated": False,
        "truncated": False, "error": "lite.gym.errors.EnvBlocked: site blocked",
    }))
    # Resolved → NOT pending → the retry loop won't re-run it.
    assert get_pending(tmp_path, [spec], group_size=1) == []
    # Errored, with the terminal message (NOT the missing-summary "unresolved").
    r = rebuild_results(tmp_path, [spec], group_size=1)[0]
    assert r["error"] == "lite.gym.errors.EnvBlocked: site blocked"
