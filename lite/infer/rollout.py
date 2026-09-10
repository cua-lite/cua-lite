"""Rollout driver: run agents on tasks at scale for evaluation or training-data
collection.

ENTRYPOINT-tier module (``lite.infer``): orchestrates the ``gym`` and
``agents`` subsystems. Task discovery, the async run loop, resume/reporting,
the shared CLI parser, and CI gates live here. Pure helpers it leans on
live below it: ``load_config`` / ``deep_merge`` in :mod:`lite.utils.config`,
``parse_filter`` in :mod:`lite.core.utils.filters`, and
``finalize_env_kwargs`` on the :mod:`lite.gym` facade.

Usage::

    from lite.infer.rollout import make_rollout_parser, run_rollout

    # Build CLI with shared rollout args + script-specific args
    parser = argparse.ArgumentParser(parents=[make_rollout_parser()])
    parser.add_argument("--model-id", ...)
    parser.set_defaults(concurrency=4)

    # Run rollout loop
    asyncio.run(run_rollout(
        model_id=args.model_id, env_id=args.env_id,
        agent_kwargs=agent_kwargs, env_kwargs=args.env_kwargs, ...
    ))
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import shlex
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

import lite.gym as gym
from lite.agents.core.adapter import tool_surface_agent_kwarg_names
from lite.core.messages.final import PARSE_FAILURE_FINAL_REASON, STOP_REASON_INFO_KEY
from lite.core.utils.filters import parse_filter
from lite.gym import finalize_env_kwargs, routing_server_url
from lite.utils.config import deep_merge, load_config
from lite.utils.git import git_commit, run_git

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSpec:
    """One unit of rollout work: a task plus everything needed to run and locate
    it on disk.

    The single currency between task discovery (registry / parquet / single
    task) and the run loop — it replaces the old parallel ``task_to_env_id`` /
    ``task_to_env_kwargs`` / ``task_to_split`` dicts + ``all_splits``, so adding
    a per-task field never reshapes a call site. On-disk layout (split subdir,
    sample dir, summary path) lives here as the one source of truth.
    """

    task_id: str
    env_id: str
    split: str = ""  # "" → no split subdir (e.g. registry tasks with no split)
    env_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def env_key(self) -> str:
        """``"<env_id>@<task_id>"`` — the key ``gym.make`` expects."""
        return f"{self.env_id}@{self.task_id}"

    def task_dir(self, log_root: Path) -> Path:
        """Directory that owns all samples for this task under ``log_root``."""
        return (log_root / self.split if self.split else log_root) / self.task_id

    def sample_dir(self, log_root: Path, sample_idx: int) -> Path:
        """Directory for one sample attempt of this task."""
        return self.task_dir(log_root) / f"sample_{sample_idx:02d}"

    def summary_path(self, log_root: Path, sample_idx: int) -> Path:
        """Path to the sample-level ``summary.json``."""
        return self.sample_dir(log_root, sample_idx) / "summary.json"


@dataclass(frozen=True)
class RolloutPlan:
    """Derived run state after task resolution, before any env is started."""

    specs: list[TaskSpec]
    group_idx_of: dict[str, int]
    splits_seen: list[str]
    total_runs: int
    log_root: Path
    resuming: bool
    pending: list[tuple[TaskSpec, int]]


# =============================================================================
# Task discovery — registry / parquet
# =============================================================================

def collect_tasks(
    env_id: str,
    *,
    splits: list[str] | None = None,
    head: int | None = None,
    sample: int | None = None,
    filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    rng: random.Random | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Collect deduplicated task_ids from the registry.

    Args:
        env_id: Environment name (e.g. ``"lite.demo"``).
        splits: If given, only include these splits. None → all.
        head: Keep only the first N tasks.
        sample: Randomly sample N tasks. Ignored if head is set.
        filter_fn: Optional filter on task metadata. Use
            :func:`lite.core.utils.filters.parse_filter` to build one
            from a CLI string.
        rng: Dedicated RNG for ``sample``. Must be supplied by callers
            that need cross-process determinism — the global ``random``
            module is polluted by intervening calls (transformers init,
            sglang launch, agent setup), so seeding it once at CLI entry
            does **not** guarantee a stable ``--sample`` subset. Pass
            ``random.Random(args.seed)`` to lock it in. ``None`` → falls
            back to global ``random`` (legacy callers / tests).

    Returns:
        ``(all_splits, tasks)`` — splits dict mapping split→task_ids and the
        flattened deduplicated task list.
    """
    all_splits = cast(dict[str, list[str]], gym.registry.task_ids(env_id))
    if splits is not None:
        unknown = set(splits) - set(all_splits)
        if unknown:
            raise ValueError(
                f"Unknown splits {unknown}. Available: {list(all_splits)}"
            )
        all_splits = {s: all_splits[s] for s in splits}

    seen: set[str] = set()
    tasks: list[str] = []
    for ids in all_splits.values():
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                tasks.append(tid)
    if not tasks:
        raise ValueError(f"No tasks found for env '{env_id}'")

    if filter_fn is not None:
        before = len(tasks)
        tasks = [t for t in tasks if filter_fn(gym.registry.task_metadata(env_id, t))]
        log.info("Filter: %d → %d tasks", before, len(tasks))

    if head is not None:
        tasks = tasks[:head]
    elif sample is not None and sample <= len(tasks):
        # ``<=`` (not ``<``): ``sample == len(tasks)`` is the "shuffle the whole
        # set" case — ``random.sample(tasks, len(tasks))`` returns every task in
        # random order. A ``sample > len(tasks)`` still no-ops (avoids the
        # ValueError from over-sampling). Deterministic under ``--seed``.
        tasks = (rng or random).sample(tasks, sample)

    return all_splits, tasks


# -----------------------------------------------------------------------------
# agent_kwargs normalization
# -----------------------------------------------------------------------------

def _reject_tool_surface_agent_kwargs(agent_kwargs: dict[str, Any] | None, *, source: str) -> None:
    surface_keys = tool_surface_agent_kwarg_names(agent_kwargs)
    if surface_keys:
        raise ValueError(
            f"{source} agent_kwargs contains tool-surface settings {sorted(surface_keys)}; "
            "put run-time selectors in env_kwargs, and use saved Lite task metadata for "
            "offline/render surfaces"
        )


def _merge_agent_kwargs(file_cfg: dict[str, Any], cli: dict[str, Any]) -> dict[str, Any]:
    """Merge yaml ``agent_kwargs`` (base) with CLI ``agent_kwargs`` (override),
    CLI-wins, **deep-merging every nested dict per key** (``api_kwargs``,
    ``protocol_kwargs``, ``sampling_kwargs``, ...).

    Footgun fix: a shallow ``{**file, **cli}`` lets a CLI override (e.g.
    ``--api-kwargs '{"reasoning_summary": ""}'``, which arrives nested under
    ``cli["api_kwargs"]``) REPLACE the entire yaml sub-dict, silently dropping
    config keys like ``reasoning_effort`` / ``max_output_tokens``. A
    recursive deep-merge overrides only the named leaf keys; the same protection
    extends to ``protocol_kwargs`` / ``sampling_kwargs``. Non-dict values (e.g.
    ``resolution``, ``system_prompt``) still replace wholesale (CLI wins).

    Tool-surface keys are rejected here, before rollout/export can silently
    reinterpret env-owned surface as model kwargs.

    ``... or {}``: a yaml ``agent_kwargs:`` with only comments parses to None,
    and callers pass ``file_cfg.get("agent_kwargs", {})`` which returns that None
    (key present). Guard here so EVERY call site is covered."""
    _reject_tool_surface_agent_kwargs(file_cfg, source="yaml")
    _reject_tool_surface_agent_kwargs(cli, source="CLI")
    return deep_merge(file_cfg or {}, cli or {})


# -----------------------------------------------------------------------------
# Prompt-data parquet loader (called via ``resolve_prompt_data_tasks``)
# -----------------------------------------------------------------------------

def _collect_tasks_from_parquet(
    prompt_data: str | Path,
) -> list[tuple[str, str, dict[str, Any], str]]:
    """Load ``(env_id, task_id, env_kwargs, split)`` tuples from a prompt-data
    parquet (e.g. produced by :mod:`lite.train.export.export_tasks`,
    or hand-constructed for per-task tuning).

    Each row's ``metadata`` dict carries:
      * ``env_key`` (required) — fully-qualified ``env_id@task_id``.
      * ``split`` (optional) — split name (``"train"`` / ``"eval"`` / ...).
        Returned verbatim (``""`` when absent); the caller buckets the
        on-disk layout by it, falling back to a synthetic ``"parquet"``
        split only when every row omits it.
      * ``env_kwargs`` (optional, default ``{}``) — per-row env_kwargs
        override for fine-grained per-task control (e.g. pinning ``seed``
        / ``max_steps`` per task without touching yaml or CLI). When set,
        deep-merges over args/yaml env_kwargs per-leaf (``prompt_data > args > yaml``).

    Note on parquet schema: when ``env_kwargs`` columns differ across rows,
    PyArrow's struct inference unifies the schema (missing keys become
    None on read, ints get promoted to float). Nones are stripped by
    :func:`lite.gym.finalize_env_kwargs`, and gym.make typically accepts
    int/float interchangeably.
    """
    import pandas as pd
    df = pd.read_parquet(prompt_data)
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    for idx, row in df.iterrows():
        meta = row["metadata"]
        # ``write_records_to_parquet`` stores metadata as a PyArrow struct →
        # dict on read, matching the rest of the codebase's convention
        # (``sample.metadata["env_key"]`` etc.). Catch the mistake here with
        # a typed error rather than letting it surface downstream as an
        # opaque ``string indices must be integers``.
        if not isinstance(meta, dict):
            raise TypeError(
                f"prompt_data row {idx}: metadata must be a dict (parquet struct "
                f"format), got {type(meta).__name__}={meta!r}"
            )
        env_key = meta.get("env_key")
        if not env_key or "@" not in env_key:
            raise ValueError(
                f"prompt_data row {idx}: env_key={env_key!r} is missing or malformed "
                f"(expected 'env_id@task_id')"
            )
        env_id, task_id = env_key.split("@", 1)
        if not env_id or not task_id:
            raise ValueError(
                f"prompt_data row {idx}: env_key={env_key!r} has empty env_id or task_id"
            )
        split = str(meta.get("split") or "")
        rows.append((env_id, task_id, finalize_env_kwargs(meta.get("env_kwargs")), split))
    return rows


# -----------------------------------------------------------------------------
# Config resolution — prompt_data > args > yaml (flexibility ladder)
# -----------------------------------------------------------------------------
#
# Three sources, in decreasing priority, applied uniformly to ``env_id``
# AND per-row ``env_kwargs``:
#   1. ``prompt_data`` row's ``metadata`` (per-row — ``env_key`` carries
#      env_id, optional ``env_kwargs`` carries per-task overrides; most
#      flexible source, gets the final word).
#   2. CLI args (``--env-id`` is argparse-``required=True`` → always
#      explicit; per-invocation ad-hoc override).
#   3. yaml config (static baseline / default).
#
# The ordering matches the universal Unix convention (CLI overrides config)
# and the "flexibility wins" intuition: prompt_data is the most expressive
# source; yaml sits at the bottom as the default. For ``env_id``, conflicts
# between two actively-set sources emit ``log.warning`` so silent overrides
# never happen. For ``env_kwargs``, the per-leaf deep merge is silent (each
# leaf is a distinct override — fan-out warnings would be noisy).
#
# Helpers here are the single source of truth so both :func:`run_rollout`
# and external entrypoints (custom ``examples/`` rollout shims) share
# identical resolution + warning behavior.


def resolve_effective_env_id(cli_env_id: str, yaml_env_id: str | None) -> str:
    """Resolve CLI-vs-yaml env_id with CLI priority; warn on real conflict.

    Always returns ``cli_env_id`` — CLI is argparse-``required=True`` so
    it is the authoritative source. ``yaml_env_id`` is consulted only to
    detect conflict: if it is set AND differs from ``cli_env_id``, emit
    ``log.warning`` so a silently-overridden yaml never goes unnoticed.
    A missing yaml field is silent (no conflict, no override).
    """
    if yaml_env_id is not None and yaml_env_id != cli_env_id:
        log.warning(
            "env_id: CLI --env-id=%r overrides yaml=%r (args > yaml)",
            cli_env_id, yaml_env_id,
        )
    return cli_env_id


def resolve_prompt_data_tasks(
    prompt_data: str | Path,
    *,
    effective_env_id: str,
    head: int | None = None,
    sample: int | None = None,
    rng: random.Random | None = None,
) -> list[TaskSpec]:
    """Load a prompt-data parquet into an ordered, deduplicated ``list[TaskSpec]``.

    Applies ``head`` / ``sample`` slicing **on the raw rows, before dedup**
    (a row subset, not a task subset), deduplicates task_ids (first-seen
    order), and warns when prompt_data's env_ids differ from
    ``effective_env_id`` (the resolved CLI-or-yaml baseline). prompt_data is
    authoritative (most flexible source wins) — the warning only surfaces an
    unintended override of an explicitly-set CLI / yaml env_id.

    Each row's ``split`` is used verbatim; a row with no split falls back to
    the synthetic split ``"parquet"`` (so the on-disk layout has a subdir
    rather than dumping under the log-root). Per-key env_kwargs overrides are
    silent (normal behavior, not a "conflict").

    Raises :class:`ValueError` when:
    * the same task_id appears under multiple env_ids — disk paths are keyed
      by ``task_id`` alone, so a collision would silently collapse two
      distinct tasks. Standard exports usually namespace task IDs, but arbitrary
      multi-env prompt_data can still trigger this guard.
    * the same task_id appears with conflicting ``env_kwargs`` — likely a
      data-prep bug (duplicate rows with different overrides).

    ``rng`` is used for ``--sample`` selection; the caller must supply a
    dedicated :class:`random.Random` for cross-process determinism (the
    global ``random`` module is unsafe — see :func:`collect_tasks`).
    """
    rows = _collect_tasks_from_parquet(prompt_data)
    if head is not None:
        rows = rows[:head]
    elif sample is not None and sample <= len(rows):
        # ``<=`` matches collect_tasks: ``sample == len(rows)`` shuffles the
        # whole set (same --sample flag, same semantics on both task sources).
        if rng is None:
            raise ValueError("rng is required when sample is set")
        rows = rng.sample(rows, sample)

    prompt_env_ids = sorted({eid for eid, _, _, _ in rows})
    if prompt_env_ids and prompt_env_ids != [effective_env_id]:
        log.warning(
            "env_id: prompt_data carries %s, overriding effective=%r "
            "(prompt_data > args > yaml)",
            prompt_env_ids, effective_env_id,
        )

    specs: list[TaskSpec] = []
    by_id: dict[str, TaskSpec] = {}
    for eid, tid, ekw, split in rows:
        if tid in by_id:
            prev = by_id[tid]
            if prev.env_id != eid:
                raise ValueError(
                    f"task_id {tid!r} appears under both env_ids "
                    f"{prev.env_id!r} and {eid!r} in prompt_data — "
                    f"task_ids must be globally unique across env_ids."
                )
            if prev.env_kwargs != ekw:
                raise ValueError(
                    f"task_id {tid!r} appears with conflicting env_kwargs in "
                    f"prompt_data: {prev.env_kwargs!r} vs {ekw!r}."
                )
            continue
        spec = TaskSpec(tid, eid, split or "parquet", ekw)
        by_id[tid] = spec
        specs.append(spec)
    return specs


def _registry_specs(
    env_id: str, *,
    splits: list[str] | None,
    head: int | None,
    sample: int | None,
    filter_fn: Callable[[dict[str, Any]], bool] | None,
    rng: random.Random | None,
) -> list[TaskSpec]:
    """Registry task source → ``list[TaskSpec]`` (uniform env_id, no per-row
    env_kwargs). Splits come from the registry buckets; a task in multiple
    splits takes its last bucket (matches the prior ``task_to_split``
    inversion)."""
    all_splits, task_ids = collect_tasks(
        env_id, splits=splits, head=head, sample=sample, filter_fn=filter_fn, rng=rng,
    )
    split_of = {tid: s for s, ids in all_splits.items() for tid in ids}
    return [TaskSpec(tid, env_id, split_of.get(tid, "")) for tid in task_ids]


def _resolve_run_tasks(
    env_id: str, *,
    prompt_data: str | None,
    splits: list[str] | None,
    head: int | None,
    sample: int | None,
    filter_expr: str | None,
    rng: random.Random,
    yaml_env_id: str | None = None,
    task_id: str | None = None,
) -> list[TaskSpec]:
    """Resolve an ordered, deduplicated ``list[TaskSpec]`` from one of three
    mutually-exclusive sources:

    * ``task_id`` — a single pinned task (degenerate rollout), split ``"task"``.
    * ``prompt_data`` — exported parquet; each row's ``env_key = env_id@task_id``
      carries its own env_id + optional per-row ``env_kwargs`` + ``split``, so one
      parquet can drive multi-env / per-task-tuned rollout. head/sample apply;
      splits/filter are rejected (filter at export time).
    * Otherwise — the env registry under the effective env_id, optionally
      filtered. Uniform env_id, no per-row env_kwargs.

    Config precedence is **prompt_data > args > yaml** (env_id, env_kwargs) —
    "flexibility wins", matching Unix CLI-overrides-config. ``rng`` drives
    ``--sample`` and must be a dedicated :class:`random.Random` for
    cross-process determinism (see :func:`collect_tasks`).
    """
    effective_env_id = resolve_effective_env_id(env_id, yaml_env_id)

    if task_id is not None:
        if prompt_data is not None:
            raise ValueError("--task-id cannot be combined with --prompt-data")
        if filter_expr is not None or splits is not None:
            raise ValueError("--task-id pins one task; --filter/--splits don't apply")
        return [TaskSpec(task_id, effective_env_id, "task")]

    if prompt_data:
        if splits is not None:
            raise ValueError("--splits cannot be used with --prompt-data")
        if filter_expr is not None:
            raise ValueError(
                "--filter cannot be used with --prompt-data "
                "(filter at export time instead)"
            )
        return resolve_prompt_data_tasks(
            prompt_data, effective_env_id=effective_env_id,
            head=head, sample=sample, rng=rng,
        )

    filter_fn = parse_filter(filter_expr) if filter_expr else None
    return _registry_specs(
        effective_env_id, splits=splits, head=head, sample=sample,
        filter_fn=filter_fn, rng=rng,
    )


# =============================================================================
# Per-sample result records — uniform shape across success / failure / disk
# =============================================================================

def _make_result(
    task_id: str, group_idx: int, sample_idx: int,
    *, episode_return: float = 0.0, turns: int = 0,
    terminated: bool = False, truncated: bool = False,
    stop_reason: str | None = None,
    error: str | None = None,
    env_id: str | None = None,
) -> dict[str, Any]:
    """Per-sample result dict — same shape for in-memory results and
    on-disk ``summary.json`` rebuilds.

    ``env_id`` is the resolved env_id for this task; included so multi-env
    prompt_data runs can disambiguate which env each row came from.
    """
    return {
        "task": task_id, "env_id": env_id,
        "group_idx": group_idx, "sample_idx": sample_idx,
        "turns": turns, "episode_return": episode_return,
        "terminated": terminated, "truncated": truncated,
        STOP_REASON_INFO_KEY: stop_reason,
        "error": error,
    }


def _load_summary(path: Path) -> dict[str, Any] | None:
    """Load one ``sample_*/summary.json``. Returns ``None`` if the file
    is missing, unreadable, or contains malformed JSON (e.g. process
    kill mid-``write_text``). Single source of truth for "is this
    sample resolved (done)?" — used by both :func:`get_pending` (resume
    gate) and :func:`rebuild_results` (final results). A corrupt file
    consistently registers as "not resolved" in both views, so the
    surrounding retry loop will re-run it instead of leaving an orphan
    that ``get_pending`` skips but ``rebuild_results`` can't read.

    A present summary is ``resolved`` whether it succeeded (no ``error`` key) or
    terminally errored (an ``error`` key, written by ``_run_one`` for a
    non-retryable error) — both are "done", neither is re-run."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("ignoring corrupt %s (%s); sample will be re-run", path, e)
        return None


def get_pending(
    log_root: Path,
    specs: list[TaskSpec],
    group_size: int,
) -> list[tuple[TaskSpec, int]]:
    """Return ``(spec, sample_idx)`` units whose ``summary.json`` is missing or
    unreadable on disk — drives resume."""
    return [
        (spec, i)
        for spec in specs
        for i in range(group_size)
        if _load_summary(spec.summary_path(log_root, i)) is None
    ]


def rebuild_results(
    log_root: Path,
    specs: list[TaskSpec],
    group_size: int,
) -> list[dict[str, Any]]:
    """Rebuild the per-sample results list from every ``sample_*/summary.json``
    on disk. A summary with no file (or corrupt) becomes an ``error="unresolved"``
    record — same gate as :func:`get_pending`, so a sample that is unresolved here
    is also ``pending`` there and will be re-run by the next attempt. A present
    summary carrying an ``error`` key (a terminal non-retryable error) is read as
    that error — excluded from ``valid`` but NOT re-run. Each result is tagged with
    its spec's env_id (multi-env prompt_data disambiguation)."""
    results: list[dict[str, Any]] = []
    for group_idx, spec in enumerate(specs):
        for sample_idx in range(group_size):
            data = _load_summary(spec.summary_path(log_root, sample_idx))
            if data is not None:
                results.append(_make_result(
                    spec.task_id, group_idx, sample_idx,
                    turns=data.get("n_turns", 0),
                    episode_return=data.get("episode_return", 0.0),
                    terminated=data.get("terminated", False),
                    truncated=data.get("truncated", False),
                    stop_reason=data.get(STOP_REASON_INFO_KEY),
                    error=_summary_error(data),
                    env_id=spec.env_id,
                ))
            else:
                results.append(_make_result(
                    spec.task_id, group_idx, sample_idx, error="unresolved",
                    env_id=spec.env_id,
                ))
    return results


def _summary_error(data: dict[str, Any]) -> str | None:
    """Return the terminal error represented by one per-sample summary.

    Agent-side unpairable model-output failures are resolved terminal samples:
    they persist a summary (so resume must not retry) and carry a durable
    stop_reason instead of an exception. Treat them as invalid results for
    rollout stats without fabricating role:tool feedback or re-running them.
    """
    error = data.get("error")
    if error is not None:
        return str(error)
    stop_reason = data.get(STOP_REASON_INFO_KEY)
    if stop_reason == PARSE_FAILURE_FINAL_REASON:
        return f"terminal model_output_error: {stop_reason}"
    return None


# =============================================================================
# Reporting — console table, JSON summary, run_info
# =============================================================================

# -----------------------------------------------------------------------------
# Private — JSON encoder that keeps flat scalar lists on a single line
# -----------------------------------------------------------------------------

class _CompactListEncoder(json.JSONEncoder):
    """JSON encoder that keeps flat lists (e.g. ``episode_returns``) on
    one line while still pretty-printing nested objects."""

    def encode(self, o: Any) -> str:
        return self._encode(o, indent_level=0)

    def _encode(self, o: Any, indent_level: int) -> str:
        if isinstance(o, dict):
            if not o:
                return "{}"
            indent = "  " * (indent_level + 1)
            closing = "  " * indent_level
            items = [
                f"{indent}{json.dumps(k, ensure_ascii=False)}: "
                f"{self._encode(v, indent_level + 1)}"
                for k, v in o.items()
            ]
            return "{\n" + ",\n".join(items) + f"\n{closing}}}"
        if isinstance(o, list):
            # Flat list of scalars → single line
            if all(not isinstance(x, (dict, list)) for x in o):
                return "[" + ", ".join(
                    json.dumps(x, ensure_ascii=False, default=str) for x in o
                ) + "]"
            # List of dicts / lists → expand
            indent = "  " * (indent_level + 1)
            closing = "  " * indent_level
            items = [f"{indent}{self._encode(x, indent_level + 1)}" for x in o]
            return "[\n" + ",\n".join(items) + f"\n{closing}]"
        return json.dumps(o, ensure_ascii=False, default=str)


def _compact_json(obj: Any) -> str:
    return _CompactListEncoder().encode(obj)


def _format_exception_for_error_file(exc: BaseException) -> str:
    """Format an exception for saved rollout artifacts.

    `httpx.HTTPStatusError` messages omit the response body, but the body is the
    only place remote envs put typed reset/step failures such as cuagym's
    `phase`/`kind`. Typed ``LiteGymError`` exceptions carry the same details in
    ``to_payload()``. Keep the traceback first and append the available bounded
    diagnostic envelope.
    """
    report = traceback.format_exc()
    try:
        from lite.gym.errors import LiteGymError

        if isinstance(exc, LiteGymError):
            payload = exc.to_payload()
            tagged = " ".join(
                f"{key}={payload[key]!r}"
                for key in (
                    "error_type",
                    "phase",
                    "kind",
                    "returncode",
                    "task",
                    "retry_after_s",
                    "layer",
                )
                if payload.get(key) is not None
            )
            what = str(payload.get("what") or "")
            return f"{report}\nLiteGymError payload:\n{tagged} {what}".rstrip() + "\n"
    except Exception as format_exc:  # noqa: BLE001 - diagnostic fallback only
        return (
            f"{report}\n"
            "LiteGymError payload: "
            f"<unavailable: {type(format_exc).__name__}: {format_exc}>\n"
        )

    response = getattr(exc, "response", None)
    if response is None:
        return report
    try:
        from lite.gym.remote.client import _error_body_summary

        status = getattr(response, "status_code", "?")
        request = getattr(response, "request", None)
        method = getattr(request, "method", "?")
        url = getattr(request, "url", "?")
        body = _error_body_summary(response)
        return (
            f"{report}\n"
            f"HTTP response body ({method} {url} -> {status}):\n"
            f"{body}\n"
        )
    except Exception as format_exc:  # noqa: BLE001 - diagnostic fallback only
        return (
            f"{report}\n"
            "HTTP response body: "
            f"<unavailable: {type(format_exc).__name__}: {format_exc}>\n"
        )


# -----------------------------------------------------------------------------
# Public — print / save summary, save run metadata
# -----------------------------------------------------------------------------

def print_results(
    results: list[dict[str, Any]],
    specs: list[TaskSpec],
    *,
    group_size: int = 1,
) -> dict[str, Any]:
    """Print a per-task result table + aggregate stats, return summary dict.

    Args:
        results: per-sample dicts from ``_make_result``.
        specs: task specs in display order.
        group_size: rollouts per task (drives GRPO variance breakdown).

    Returns:
        Summary stats dict (mean episode_return, valid count, etc.).
    """
    n_tasks = len(specs)
    total_runs = n_tasks * group_size
    valid = [r for r in results if r["error"] is None]
    all_returns = [r["episode_return"] for r in valid]
    avg_return = sum(all_returns) / len(all_returns) if all_returns else 0.0
    stop_reason_counts: dict[str, int] = {}
    for r in valid:
        reason = r.get(STOP_REASON_INFO_KEY)
        if isinstance(reason, str) and reason:
            stop_reason_counts[reason] = stop_reason_counts.get(reason, 0) + 1

    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["task"], []).append(r)

    # Per-task table
    print(f"\n{'task':<40} | {'episode_returns (per sample)':>40} | {'mean':>6} | {'var>0':>5}")
    print("-" * 100)
    n_groups_with_variance = 0
    for spec in specs:
        group = groups.get(spec.task_id, [])
        returns = [r["episode_return"] for r in group if r["error"] is None]
        errors = sum(1 for r in group if r["error"] is not None)
        cells = [f"{r:.1f}" for r in returns] + ["ERR"] * errors
        mean = sum(returns) / len(returns) if returns else 0.0
        has_var = len(set(returns)) > 1
        if has_var:
            n_groups_with_variance += 1
        var_mark = "YES" if has_var else "no"
        print(f"{spec.task_id:<40} | {', '.join(cells):>40} | {mean:>6.2f} | {var_mark:>5}")

    # Aggregate
    print("-" * 100)
    print(f"Tasks: {n_tasks}  Samples/task: {group_size}  Total runs: {total_runs}")
    print(f"Valid: {len(valid)}/{total_runs}  Avg episode_return: {avg_return:.4f}")
    if stop_reason_counts:
        print(
            "Stop reasons: "
            + ", ".join(
                f"{name}={count}"
                for name, count in sorted(stop_reason_counts.items())
            )
        )
    if group_size > 1:
        pct = n_groups_with_variance * 100 // max(n_tasks, 1)
        valid_group_returns = [
            [r["episode_return"] for r in g if r["error"] is None]
            for g in groups.values()
        ]
        all_zero = sum(1 for returns in valid_group_returns
                       if returns and all(r == 0.0 for r in returns))
        all_one = sum(1 for returns in valid_group_returns
                      if returns and all(r == 1.0 for r in returns))
        print(f"Groups with episode_return variance: {n_groups_with_variance}/{n_tasks} ({pct}%)")
        print(f"\n--- GRPO Simulation (group_size={group_size}) ---")
        print(f"  All-zero groups: {all_zero}/{n_tasks} (advantage=0, no learning signal)")
        print(f"  All-one groups:  {all_one}/{n_tasks} (advantage=0, no learning signal)")
        print(f"  Mixed groups:    {n_groups_with_variance}/{n_tasks} (advantage!=0, can learn)")

    return {
        "num_tasks": n_tasks,
        "num_samples": total_runs,
        "num_valid": len(valid),
        "mean_episode_return": avg_return,
        "groups_with_variance": n_groups_with_variance,
        "stop_reasons": stop_reason_counts,
    }


def save_summary(
    path: Path,
    *,
    results: list[dict[str, Any]],
    stats: dict[str, Any],
    specs: list[TaskSpec],
    **extra: Any,
) -> None:
    """Write per-task ``summary.json`` files alongside sample dirs and
    a global ``summary.json`` at ``path``.

    Args:
        path: Output path for the global summary (``<log_root>/summary.json``).
        results: Per-run result dicts.
        stats: Aggregate stats from :func:`print_results`.
        specs: task specs — supply each task's split-aware on-disk dir.
        **extra: Extra fields embedded under ``config`` (e.g. model, env_id).
    """
    log_root = path.parent
    spec_of = {spec.task_id: spec for spec in specs}
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["task"], []).append(r)

    task_summaries: list[dict[str, Any]] = []
    for task_id, task_results in groups.items():
        returns = [r["episode_return"] for r in task_results if r["error"] is None]
        # All samples in a group share the same task → same env_id; the
        # rollout pipeline always populates it via _make_result.
        task_summary = {
            "task": task_id,
            "env_id": task_results[0]["env_id"],
            "num_samples": len(task_results),
            "num_valid": len(returns),
            "episode_returns": returns,
            "mean_episode_return": sum(returns) / len(returns) if returns else 0.0,
            "has_variance": len(set(returns)) > 1,
        }
        task_summaries.append(task_summary)
        task_dir = spec_of[task_id].task_dir(log_root)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "summary.json").write_text(_compact_json(task_summary))

    summary = {"config": extra, "stats": stats, "tasks": task_summaries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_compact_json(summary))
    print(f"\nSummary saved to {path}")


def _write_rollout_report(
    plan: RolloutPlan,
    *,
    group_size: int,
    model_id: str,
    model_path: str,
    env_id: str,
    config_path: str | None,
    runtime_stamp: dict[str, Any],
    agent_kwargs: dict[str, Any],
    env_kwargs: dict[str, Any],
) -> None:
    """Rebuild current disk state and write console + JSON rollout summaries."""
    results = rebuild_results(plan.log_root, plan.specs, group_size)
    stats = print_results(results, plan.specs, group_size=group_size)
    save_summary(
        plan.log_root / "summary.json",
        results=results,
        stats=stats,
        specs=plan.specs,
        model=model_id,
        model_path=model_path,
        env_id=env_id,
        config_path=config_path,
        **runtime_stamp,
        agent_kwargs=agent_kwargs,
        env_kwargs=env_kwargs,
        splits=plan.splits_seen,
        group_size=group_size,
    )


def _parent_cmdline_parts() -> list[str]:
    """Best-effort parent argv for preserving ``uv run`` flags in run_info."""
    try:
        raw = Path(f"/proc/{os.getppid()}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode() for part in raw.split(b"\0") if part]


def _cli_command() -> str:
    """The invocation that launched this process, in copy-paste-to-rerun form."""
    uv_run = ["uv", "run"]
    parent = _parent_cmdline_parts()
    if len(parent) >= 2 and Path(parent[0]).name == "uv" and parent[1] == "run":
        python_i = next(
            (i for i, part in enumerate(parent) if Path(part).name.startswith("python")),
            len(parent),
        )
        if "--no-sync" in parent[:python_i]:
            uv_run.append("--no-sync")
    return f"{shlex.join([*uv_run, 'python'])} {shlex.join(sys.argv)}"


def build_runtime_stamp() -> dict[str, Any]:
    """Runtime-mode provenance shared by run_info, summary, and rows.

    The raw env-server token and URL are deliberately not persisted. The boolean
    fields are enough to audit whether this was direct or server mode without
    leaking operator-specific connection details into artifacts.
    """
    env_server_url = os.environ.get("CUA_LITE_ENV_SERVER_URL")
    return {
        "runtime_mode": "server" if routing_server_url() else "direct",
        "env_server_url_present": bool(env_server_url),
        "env_server_url_redacted": "<set>" if env_server_url else None,
        "env_server_token_present": bool(os.environ.get("CUA_LITE_ENV_SERVER_TOKEN")),
        "session_id": os.environ.get("SESSION_ID"),
    }


def build_provenance(model_id: str, agent_id: str, config_path: str | None) -> dict[str, Any]:
    """Per-row run provenance stored in ``metadata.others`` of every
    trajectory.parquet row (see ``TrajectoryLogger``).

    These keys live in ``others`` because they are slicing dimensions
    (``m.others["agent_id"] == ...`` in a mixed multi-agent dataset; ``commit``
    doubles as the batch id under the pinned-COMMIT collection discipline, see
    devs/data/*/AGENTS.md). So do the identity/outcome fields ``env_id``,
    ``task_id``, ``episode_return``, ``terminated`` and ``truncated``
    (``ROLLOUT_METADATA_OTHER_KEYS``, :mod:`lite.agents.core.agent.logger`):
    :class:`~lite.core.metadata.LiteBaseMetadata` has no top-level slot for
    either group -- only ``dims``, ``extra_tool_schemas`` and ``others``.

    The log-root-level ``run_info.txt`` records the same facts per
    *invocation* — a root that accumulates resumes (each possibly at a
    different commit/command) can't say which invocation produced a given
    row, and the file doesn't survive stage/merge. This dict can: ``command``
    (with its CLI overrides) + ``commit`` pin the exact recipe;
    ``config_path`` alone wouldn't, since CLI args override the yaml.

    :func:`build_runtime_stamp` is deliberately NOT spread in here: its five
    fields are per-*invocation* connection facts, which is what ``run_info.txt``
    and ``summary.json``'s ``config`` already carry.
    """
    return {
        "model_id": model_id,
        "agent_id": agent_id,
        "config_path": config_path,
        "commit": git_commit(),
        "command": _cli_command(),
    }


def save_run_info(log_root: Path, **extra: Any) -> None:
    """Save run metadata to ``{log_root}/run_info.txt`` for reproducibility.

    Captures the CLI command (verbatim + a resume-ready variant), git state,
    and caller-provided config. Idempotent within one process — re-invoking
    with the same ``sys.argv`` is a no-op (the retry loop calls this once
    per attempt; we don't want a "Resumed" section per attempt). A fresh
    process resuming the same ``log_root`` with a different argv appends a
    new section, preserving the original invocation as audit trail.
    """
    info_path = log_root / "run_info.txt"
    argv_str = _cli_command()

    # Idempotency guard: skip if this exact argv already recorded. Match whole
    # lines, not substrings — a recorded ``--sample 10`` must not swallow a
    # later ``--sample 1`` invocation (prefix collision).
    if info_path.exists() and argv_str in info_path.read_text().splitlines():
        return

    dirty = run_git("status --porcelain --short")
    resume_cmd = (
        argv_str if "--log-root" in argv_str
        else f"{argv_str} --log-root {shlex.quote(str(log_root))}"
    )
    is_resume = info_path.exists()
    lines: list[str] = []
    if is_resume:
        lines.append("")  # blank line separator from prior content
        lines.append(f"# === Resumed at {datetime.now().isoformat(timespec='seconds')} ===")
        lines.append(argv_str)
    else:
        lines.extend(["# Command", argv_str, ""])
        lines.extend(["# Command for resume", resume_cmd, ""])
    lines.extend([
        "# Git",
        f"commit: {run_git('rev-parse --short HEAD')}",
        f"branch: {run_git('rev-parse --abbrev-ref HEAD')}",
    ])
    if dirty:
        lines.append("dirty:")
        lines.extend(f"  {line}" for line in dirty.splitlines())
    if extra:
        lines.append("")
        lines.append("# Config")
        lines.extend(f"{k}: {v}" for k, v in extra.items())

    log_root.mkdir(parents=True, exist_ok=True)
    mode = "a" if is_resume else "w"
    with info_path.open(mode) as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# CLI parser
# =============================================================================


def _parse_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def make_rollout_parser() -> argparse.ArgumentParser:
    """Parent parser with shared rollout arguments.

    Usage::

        parser = argparse.ArgumentParser(parents=[make_rollout_parser()], ...)
        parser.add_argument("--model-id", ...)     # script-specific
        parser.set_defaults(concurrency=4)         # override default
    """
    p = argparse.ArgumentParser(add_help=False)

    task = p.add_argument_group("task selection")
    task.add_argument(
        "--env-id",
        required=True,
        help=f"Env name. Available: {gym.registry.env_ids()}",
    )
    task.add_argument(
        "--prompt-data",
        default=None,
        help=(
            "Parquet file with task list (from export_tasks). Compatible "
            "with --head and --sample for subsetting."
        ),
    )
    task.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Splits to evaluate (default: all)",
    )
    task.add_argument(
        "--head",
        type=int,
        default=None,
        help="Keep first N tasks only",
    )
    task.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N tasks",
    )
    task.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Master seed (default 42). Drives --sample task selection AND "
            "per-group env seeds via a dedicated random.Random passed into "
            "run_rollout — same seed → same subset + same per-task group "
            "seeds, regardless of process state or async scheduling order. "
            "Vary across sweeps to redraw."
        ),
    )
    task.add_argument(
        "--filter",
        type=str,
        default=None,
        dest="filter_expr",
        help=(
            "Python lambda on Lite task metadata to filter tasks. Examples: "
            '"lambda m: m.others.get(\'difficulty\', 0) <= 6", '
            '"lambda m: m.others.get(\'domain\') in (\'gimp\', \'vlc\')"'
        ),
    )

    run = p.add_argument_group("run shape")
    run.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Rollouts per task (for GRPO variance analysis)",
    )
    run.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max parallel environments",
    )
    run.add_argument(
        "--log-root",
        default=None,
        help=(
            "Log directory. If it already exists, resumes by re-running "
            "only missing samples."
        ),
    )

    knobs = p.add_argument_group("config and overrides")
    knobs.add_argument(
        "--config-path",
        default=None,
        help=(
            "YAML config file (e.g. scripts/configs/qwen3_vl/default/webgym/"
            "memory.yaml). Provides base agent_kwargs and env_kwargs, "
            "overridden by CLI flags."
        ),
    )
    knobs.add_argument(
        "--agent-kwargs",
        type=json.loads,
        default={},
        help=(
            "JSON overrides for agent, e.g. "
            '\'{"protocol_kwargs": {"full_history_size": 2}}\''
        ),
    )
    knobs.add_argument(
        "--env-kwargs",
        type=json.loads,
        default={},
        help=(
            "JSON overrides for env, e.g. "
            '\'{"max_steps": 15, "resolution": [1280, 720]}\''
        ),
    )

    artifacts = p.add_argument_group("artifacts")
    # nargs="?"/const=True: accept both a bare flag (--save-gif) and an explicit
    # value (--save-gif false), so the README's bare-flag form runs as-is.
    artifacts.add_argument(
        "--save-data",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="[BOOL]",
        help="Save per-turn directories + trajectory.parquet (default: True)",
    )
    artifacts.add_argument(
        "--save-video",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="[BOOL]",
        help="Save trajectory.mp4 (requires ffmpeg; default: True)",
    )
    artifacts.add_argument(
        "--save-gif",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="[BOOL]",
        help="Save trajectory.gif (PIL, no ffmpeg, downscaled; default: False)",
    )
    artifacts.add_argument(
        "--render-instruction-banner",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="[BOOL]",
        help="Burn the task instruction into each mp4/gif frame (default: True)",
    )
    artifacts.add_argument(
        "--group-shared-seed",
        type=_parse_bool,
        default=True,
        metavar="BOOL",
        help=(
            "Share one per-group env seed across all samples in a group "
            "(same task_params / noise). Each per-group seed is drawn from "
            "the global RNG (see --seed), so reruns with the same --seed are "
            "reproducible; vary --seed across sweeps to draw fresh group "
            "seeds. Requires env to accept a `seed` kwarg (androidworld, "
            "lite.osworld). Pass `--group-shared-seed false` to make each "
            "sample independent. Default: True."
        ),
    )
    retry = p.add_argument_group("resume and retry")
    retry.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help=(
            "Max rollout attempts. Each attempt resumes from where the last "
            "left off, re-running only missing/failed samples "
            "(already-succeeded tasks are skipped). Raise the cap only when "
            "persistent retries are known-safe for the target env."
        ),
    )

    gates = p.add_argument_group("quality gates")
    # CI gates: when set, the rollout script exits nonzero if the final
    # top-level summary.json fails any of these. Useful for regression CI:
    # a baseline measurement on main + the same gate on a feature branch.
    # All gates default to "no check" so the args are opt-in.
    gates.add_argument(
        "--min-valid-frac",
        type=float,
        default=None,
        help=(
            "CI gate. Require num_valid / num_samples ≥ F (0.0–1.0). Exit "
            "nonzero if violated. Default: no check."
        ),
    )
    gates.add_argument(
        "--min-mean-return",
        type=float,
        default=None,
        help=(
            "CI gate. Require mean_episode_return ≥ F. Exit nonzero if "
            "violated. Default: no check."
        ),
    )
    gates.add_argument(
        "--debug",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="[BOOL]",
        help=(
            "Emit debug-only prompt image artifacts and run rollout artifact "
            "checks after collection. Default: False."
        ),
    )
    return p


def check_ci_gates(
    summary_path: Path,
    *,
    min_valid_frac: float | None = None,
    min_mean_return: float | None = None,
) -> list[str]:
    """Return list of CI-gate violation reasons (empty list = all gates pass).

    Reads the top-level ``summary.json`` produced by :func:`save_summary`
    and checks each declared gate. Caller (rollout entry-points) decides
    whether to ``sys.exit(1)`` on any reason.

    Designed for CI use: a stable baseline measurement on ``main`` plus
    these gates on a feature branch catches regressions before they merge.
    Gates are independent — pass multiple to require AND semantics.
    """
    reasons: list[str] = []
    if not summary_path.is_file():
        reasons.append(f"summary.json not found at {summary_path}")
        return reasons
    data = json.loads(summary_path.read_text())
    stats = data.get("stats", {})
    if min_valid_frac is not None:
        # ``num_samples`` (not ``num_tasks``) is the valid-fraction denominator:
        # grouped rollout has num_samples = num_tasks * group_size, and
        # :func:`print_results` always writes it, so there is no old-shape
        # fallback to read here.
        total = stats.get("num_samples", 0)
        done = stats.get("num_valid", 0)
        frac = (done / total) if total else 0.0
        if frac < min_valid_frac:
            reasons.append(
                f"valid fraction {frac:.3f} < required {min_valid_frac:.3f} "
                f"({done}/{total})"
            )
    if min_mean_return is not None:
        mean = stats.get("mean_episode_return", 0.0)
        if mean < min_mean_return:
            reasons.append(
                f"mean_episode_return {mean:.4f} < required {min_mean_return:.4f}"
            )
    return reasons


# =============================================================================
# Main rollout loop
# =============================================================================


# -----------------------------------------------------------------------------
# Rollout orchestration helpers
# -----------------------------------------------------------------------------

def _resolve_log_root(
    *,
    model_id: str,
    env_id: str,
    log_root: Path | str | None,
) -> Path:
    """Resolve the rollout log root; auto-generated paths are collision-safe."""
    if log_root:
        return Path(log_root)
    model_slug = model_id.replace("/", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(f".logs/rollout/{model_slug}/{env_id}/{stamp}")


def _build_rollout_plan(
    *,
    specs: list[TaskSpec],
    model_id: str,
    env_id: str,
    log_root: Path | str | None,
    group_size: int,
) -> RolloutPlan:
    """Build derived rollout state without starting envs or agents."""
    resolved_log_root = _resolve_log_root(
        model_id=model_id,
        env_id=env_id,
        log_root=log_root,
    )
    return RolloutPlan(
        specs=specs,
        group_idx_of={spec.task_id: gi for gi, spec in enumerate(specs)},
        splits_seen=list(dict.fromkeys(spec.split for spec in specs)),
        total_runs=len(specs) * group_size,
        log_root=resolved_log_root,
        resuming=resolved_log_root.exists(),
        pending=get_pending(resolved_log_root, specs, group_size),
    )


def _print_rollout_banner(
    *,
    model_id: str,
    model_path: str,
    env_id: str,
    plan: RolloutPlan,
    concurrency: int,
    group_size: int,
    group_shared_seed: bool,
) -> None:
    """Print the operator-facing run summary before workers start."""
    print(f"=== Rollout: {model_id} on {env_id} ===")
    print(f"  model_path: {model_path}")
    # Surface the actually-used env_ids when prompt_data introduces extra
    # env_ids beyond the CLI ``--env-id`` — silent for the common single-env
    # case. (yaml.env_id alone never diverges from CLI since args > yaml,
    # so divergence here implies multi-env prompt_data.)
    env_ids_in_use = sorted({spec.env_id for spec in plan.specs})
    if env_ids_in_use and env_ids_in_use != [env_id]:
        print(f"  env_ids in use: {env_ids_in_use}  (CLI --env-id={env_id})")
    print(
        f"  splits: {plan.splits_seen} | tasks: {len(plan.specs)} | "
        f"group_size: {group_size} | total: {plan.total_runs}"
    )
    print(
        f"  to run: {len(plan.pending)}/{plan.total_runs}"
        + (" (resuming)" if plan.resuming else "")
    )
    print(f"  concurrency: {concurrency}")
    print(f"  log_root: {plan.log_root}")
    if not group_shared_seed and "eval" in plan.splits_seen:
        print(
            "  WARNING: --group-shared-seed false has no effect on eval splits "
            "(they use registered deterministic seeds, e.g. seed=42)."
        )


# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------

async def run_rollout(
    *,
    model_id: str,
    model_path: str,
    env_id: str,
    agent_kwargs: dict,
    env_kwargs: dict,
    seed: int,
    group_size: int = 1,
    concurrency: int = 16,
    log_root: Path | str | None = None,
    prompt_data: str | None = None,
    splits: list[str] | None = None,
    head: int | None = None,
    sample: int | None = None,
    filter_expr: str | None = None,
    task_id: str | None = None,
    config_path: str | None = None,
    save_data: bool = True,
    save_video: bool = False,
    save_gif: bool = False,
    debug: bool = False,
    render_instruction_banner: bool = True,
    group_shared_seed: bool = True,
) -> tuple[bool, Path]:
    """Run a resumable rollout batch and write its artifacts.

    This is the async entrypoint behind ``scripts/rollout.py``. The flow is
    intentionally linear: resolve config + tasks, merge kwargs, build the
    pending-sample plan, run workers, then rebuild summaries from disk.

    Public contracts to preserve:
    - precedence is ``prompt_data > CLI args > yaml`` for task/env inputs;
    - ``seed`` drives task sampling and, on training splits, any supported
      unpinned group env seeds plus generate_fn-backed per-sample sampling
      seeds, without relying on global ``random`` state;
    - return ``(all_done, log_root)`` for retry wrappers and CI gates.
    """
    import lite.gym as gym
    from lite.agents.core.agent.logger import TrajectoryLogger
    from lite.agents.factory import AGENTS, make

    if head is not None and sample is not None:
        raise ValueError("--head and --sample are mutually exclusive")
    if model_id not in AGENTS:
        raise KeyError(f"Unknown model '{model_id}'. Available: {list(AGENTS)}")

    # === Resolve config + task list ===
    file_cfg = load_config(config_path) if config_path else {}
    # Env_id precedence (highest → lowest): prompt_data row > CLI args > yaml.
    # _resolve_run_tasks handles all three sources + warning on conflict.
    yaml_env_id = file_cfg.get("env_id")
    # Dedicated RNG so ``--sample`` and per-group seeds are independent of
    # global ``random`` state — transformers init / sglang launch /
    # asyncio scheduling all consume the global module, so seeding it once
    # at CLI entry is *not* sufficient for cross-process determinism.
    specs = _resolve_run_tasks(
        env_id, prompt_data=prompt_data, splits=splits,
        head=head, sample=sample, filter_expr=filter_expr,
        rng=random.Random(seed),
        yaml_env_id=yaml_env_id,
        task_id=task_id,
    )

    # === Merge kwargs (args > yaml, per-leaf deep) — per-model defaults live on the adapter ===
    # Deep-merge (like agent_kwargs): a CLI ``--env-kwargs '{"computer": {"image": …}}'``
    # overrides only the named nested leaf, not the whole ``computer`` sub-dict — a shallow
    # ``{**yaml, **cli}`` would silently drop siblings (memory/cpu/display). ``or {}`` guards
    # a comments-only ``env_kwargs:`` yaml that parses to None.
    merged_env_kwargs = deep_merge(file_cfg.get("env_kwargs") or {}, env_kwargs or {})
    merged_agent_kwargs = _merge_agent_kwargs(file_cfg.get("agent_kwargs", {}), agent_kwargs)
    # ``sampling_kwargs`` configures local serving/generate_fn and is consumed
    # before rollout starts; adapters should never see it as a model kwarg.
    merged_agent_kwargs.pop("sampling_kwargs", None)
    # Yaml-level ``agent_id`` overrides the family-level routing in
    # :func:`make` — e.g. browsergym text+bid yamls set
    # ``agent_id: "qwen3_vl.base"`` to opt into :class:`Qwen3VLBaseAdapter`
    # (workflow-agnostic) instead of the navigation default for the same
    # model checkpoint.
    agent_id = file_cfg.get("agent_id")
    # Per-row run provenance, persisted under ``metadata.others``. agent_id is
    # resolved the same way :func:`lite.agents.factory.make` does (yaml
    # override or the family default), so the persisted value is the agent
    # actually used.
    provenance = build_provenance(
        model_id, agent_id or AGENTS[model_id]["agent_id"], config_path,
    )
    runtime_stamp = build_runtime_stamp()

    # === Build derived run plan (log-root, split view, pending samples) ===
    plan = _build_rollout_plan(
        specs=specs,
        model_id=model_id,
        env_id=env_id,
        log_root=log_root,
        group_size=group_size,
    )
    _print_rollout_banner(
        model_id=model_id,
        model_path=model_path,
        env_id=env_id,
        plan=plan,
        concurrency=concurrency,
        group_size=group_size,
        group_shared_seed=group_shared_seed,
    )

    save_run_info(
        plan.log_root,
        model=model_id, model_path=model_path, env_id=env_id,
        config_path=config_path,
        **runtime_stamp,
        group_size=group_size, concurrency=concurrency,
        agent_kwargs=merged_agent_kwargs, env_kwargs=merged_env_kwargs,
    )

    if not plan.pending:
        print("  nothing to run.")
        _write_rollout_report(
            plan,
            group_size=group_size,
            model_id=model_id,
            model_path=model_path,
            env_id=env_id,
            config_path=config_path,
            runtime_stamp=runtime_stamp,
            agent_kwargs=merged_agent_kwargs,
            env_kwargs=merged_env_kwargs,
        )
        return True, plan.log_root

    # === Run loop ===
    sem = asyncio.Semaphore(concurrency)
    # Per-group seed cache. Each task's seed is derived from
    # ``random.Random(f"{seed}:{task_id}")`` — fully determined by
    # ``(seed, task_id)`` and independent of asyncio scheduling order.
    group_seeds: dict[str, int] = {}
    env_seed_support: dict[str, bool] = {}

    def _resolve_env_kwargs(spec: TaskSpec) -> dict:
        """Compose env_kwargs for one sample, applying **prompt_data > args
        > yaml** per-leaf (deep):

            yaml.env_kwargs  <  args.env_kwargs  <  prompt_data row env_kwargs

        Then normalize the merged dict (strip Nones, ``resolution`` → tuple)
        via :func:`lite.gym.finalize_env_kwargs`, and optionally inject a
        group-shared seed when nothing pinned one.
        """
        effective = finalize_env_kwargs(
            # Deep-merge (matches the args>yaml site above): a prompt_data row's
            # nested override (e.g. ``computer.image``) wins only at the named leaf.
            deep_merge(merged_env_kwargs, spec.env_kwargs or {})
        )
        # Skip seed injection when (a) eval split — registered spec seed is
        # authoritative; (b) anything upstream already pinned a seed; (c) the
        # env does not declare seed as an accepted soft kwarg. This keeps the
        # generic rollout loop out of env-id/action-space allowlists.
        if (
            group_shared_seed
            and spec.split != "eval"
            and "seed" not in effective
        ):
            seed_supported = env_seed_support.setdefault(
                spec.env_id, gym.registry.env_supports_kwarg(spec.env_id, "seed")
            )
            if seed_supported:
                if spec.task_id not in group_seeds:
                    group_seeds[spec.task_id] = random.Random(
                        f"{seed}:{spec.task_id}"
                    ).randint(0, 2**31 - 1)
                effective["seed"] = group_seeds[spec.task_id]
        return effective

    def _resolve_agent_kwargs(spec: TaskSpec, sample_idx: int) -> dict:
        """Return agent kwargs, wrapping ``generate_fn`` with a training seed when possible.

        Only generate_fn-backed model paths receive this per-sample
        ``sampling_seed``. API agents have no ``generate_fn`` in this path and
        are left unchanged; eval splits are also left unchanged.
        """
        base_generate_fn = merged_agent_kwargs.get("generate_fn")
        if base_generate_fn is None or spec.split == "eval":
            return merged_agent_kwargs
        sampling_seed = random.Random(
            f"{seed}:{spec.task_id}:{sample_idx}"
        ).randint(0, 2**31 - 1)

        async def _seeded_generate_fn(**kwargs):
            return await base_generate_fn(sampling_seed=sampling_seed, **kwargs)

        return {**merged_agent_kwargs, "generate_fn": _seeded_generate_fn}

    def _prepare_sample_dir(spec: TaskSpec, sample_idx: int) -> Path:
        """Resolve sample dir and wipe stale per-turn data from a prior
        (longer) attempt — otherwise resume keeps orphan ``turn_NNNN`` dirs."""
        sample_dir = spec.sample_dir(plan.log_root, sample_idx)
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        return sample_dir

    async def _run_one(spec: TaskSpec, sample_idx: int) -> dict:
        group_idx = plan.group_idx_of[spec.task_id]
        tag = f"[group={group_idx} sample={sample_idx}]"
        sample_dir = _prepare_sample_dir(spec, sample_idx)
        # Pass the per-task env_id/task_id as AUTHORITATIVE durable metadata:
        # the logger writes them as top-level row fields (overriding any
        # env-supplied value — see TrajectoryLogger._save_trajectory_parquet),
        # so a mixed multi-env dataset is filterable by env_id after rollout.
        traj_logger = TrajectoryLogger(
            sample_dir, save_data=save_data, save_video=save_video, save_gif=save_gif,
            debug_artifacts=debug,
            render_instruction_banner=render_instruction_banner,
            env_id=spec.env_id, task_id=spec.task_id, provenance=provenance,
        )
        async with sem:
            log.info("START %s %s", spec.task_id, tag)
            env = None
            try:
                # Direct mode: ``gym.make`` may spend substantial time in
                # synchronous container/service startup. Offload to a thread so
                # this rollout worker's event loop doesn't block siblings.
                # Server mode: ``gym.make`` just constructs a
                # ``LiteEnvClient`` — ``to_thread`` is near-free.
                env = await asyncio.to_thread(
                    gym.make, spec.env_key, **_resolve_env_kwargs(spec),
                )
                agent = make(
                    model_id, env=env, agent_id=agent_id,
                    **_resolve_agent_kwargs(spec, sample_idx),
                )
                lite_rl_sample = await agent.sample(env, hooks=[traj_logger])
            except Exception as exc:
                tb = _format_exception_for_error_file(exc)
                log.error("FAIL  %s %s", spec.task_id, tag, exc_info=True)
                sample_dir.mkdir(parents=True, exist_ok=True)
                (sample_dir / "error.txt").write_text(tb)
                # Terminal (non-retryable) error → write a summary so get_pending
                # treats it as resolved (NOT re-run by the --max-attempts loop) and
                # rebuild_results reads its `error` (excluded from `valid`). A
                # retryable error writes no summary → re-run next attempt, as before.
                from lite.gym.errors import is_retryable
                if not is_retryable(exc):
                    from lite.agents.core.agent.logger import build_trajectory_summary
                    (sample_dir / "summary.json").write_text(json.dumps(build_trajectory_summary(
                        n_turns=0, episode_return=0.0,
                        terminated=False, truncated=False, error=tb,
                    )))
                return _make_result(
                    spec.task_id, group_idx, sample_idx, error=tb, env_id=spec.env_id,
                )
            finally:
                # ``agent.sample()`` closes env in its own ``finally`` once it runs,
                # but if ``make()`` raises BEFORE ``sample()`` the env is never
                # closed — and in DIRECT mode its booted pid-scoped container leaks
                # (nothing boot-reaps it). Close here too; ``env.close()`` is
                # idempotent, so the double-close on the happy path is harmless.
                if env is not None:
                    try:
                        await env.close()
                    except Exception:
                        log.warning("env.close() leak-guard failed for %s",
                                    spec.task_id, exc_info=True)
            log.info("DONE  %s %s episode_return=%s",
                     spec.task_id, tag, lite_rl_sample.episode_return)
            return _make_result(
                spec.task_id, group_idx, sample_idx,
                turns=len(lite_rl_sample.steps),
                episode_return=lite_rl_sample.episode_return,
                terminated=lite_rl_sample.terminated,
                truncated=lite_rl_sample.truncated,
                env_id=spec.env_id,
            )

    await asyncio.gather(*[_run_one(spec, si) for spec, si in plan.pending])

    # === Report ===
    _write_rollout_report(
        plan,
        group_size=group_size,
        model_id=model_id,
        model_path=model_path,
        env_id=env_id,
        config_path=config_path,
        runtime_stamp=runtime_stamp,
        agent_kwargs=merged_agent_kwargs,
        env_kwargs=merged_env_kwargs,
    )
    remaining = get_pending(plan.log_root, plan.specs, group_size)
    return len(remaining) == 0, plan.log_root
