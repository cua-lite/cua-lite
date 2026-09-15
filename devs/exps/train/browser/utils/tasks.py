"""Build and calibrate browser-local WebGym GRPO task manifests.

This is experiment-local policy for
``/devs/exps/train/browser/TASKS.md``. It intentionally lives beside the browser
training runbook instead of teaching generic export code about one WebGym ->
WebVoyager transfer experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from datasets import load_dataset

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lite.data.staging import coerce_meta  # noqa: E402
from lite.utils.parquet import write_records_to_parquet  # noqa: E402

DEFAULT_SFT_SRC = (
    _REPO_ROOT
    / ".data/huggingface-webgym-nogoto/gpt5_5/cua-lite/WebGym/browser/use/train/"
    "browser.use.gpt5_5.parquet"
)
DEFAULT_POPULAR = _REPO_ROOT / "lite/gym/envs/webgym/data/webgym_popular_2102.parquet"
DEFAULT_WEBVOYAGER_TASKS = (
    _REPO_ROOT / "lite/gym/envs/webharbor/webvoyager/data/tasks.json"
)
DEFAULT_CANDIDATE = (
    _REPO_ROOT
    / "devs/exps/train/browser/data/"
    "webgym.train.gpt55_anchor.nogoto.site.candidates.seed42.parquet"
)
DEFAULT_CALIBRATED = (
    _REPO_ROOT
    / "devs/exps/train/browser/data/"
    "webgym.train.gpt55_anchor.nogoto.site.calibrated.head1024.g4.seed42.parquet"
)

SEARCH_HOSTS = {"google.com", "bing.com", "duckduckgo.com"}
EXPECTED_CANDIDATE_COUNTS = {
    "sft_source_rows": 2389,
    "duplicate_rows": 15,
    "unique_success_nogoto": 2374,
    "search_start": 8,
    "eligible": 2366,
    "popular_eligible": 666,
    "nonpopular_eligible": 1700,
}
EXPECTED_CANDIDATE_HASH = "c50569db5c0c17dfe38499c4de79ebcb44986f42968998e21bdad8c74e0bdf35"


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _host(url: str | None) -> str:
    value = url or ""
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return parsed.netloc.lower().removeprefix("www.")


def _env_key_digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def _task_id_digest(task_ids: list[str]) -> str:
    return _env_key_digest([f"webgym@{task_id}" for task_id in task_ids])


def _load_webgym_catalog(split: str = "train") -> dict[str, dict[str, Any]]:
    return {
        str(row["task_id"]): dict(row)
        for row in load_dataset("microsoft/webgym_tasks", split=split)
    }


def _load_webvoyager_instructions(path: Path) -> set[str]:
    tasks = json.loads(path.read_text())
    return {
        _norm(task.get("instruction"))
        for task in tasks.values()
        if isinstance(task, dict) and isinstance(task.get("instruction"), str)
    }


def _load_popular_task_ids(path: Path) -> set[str]:
    return {
        coerce_meta(row["metadata"])["env_key"].split("@", 1)[1]
        for _, row in pd.read_parquet(path).iterrows()
    }


def _candidate_records(task_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "problem": f"Complete the task: {task_id}",
            "metadata": {"env_key": f"webgym@{task_id}", "split": "train"},
        }
        for task_id in task_ids
    ]


def _candidate_summary(
    *,
    source: Path,
    task_ids: list[str],
    counts: Counter[str],
    catalog: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    hosts = Counter(_host(catalog[task_id].get("website")) for task_id in task_ids)
    difficulties = Counter(int(catalog[task_id]["difficulty"]) for task_id in task_ids)
    sources = Counter(str(catalog[task_id].get("benchmark_name")) for task_id in task_ids)
    return {
        "source": str(source),
        "selection": "all eligible no-goto successes; popular bucket first; "
        "shuffle within popular/nonpopular buckets",
        "seed": seed,
        "counts": dict(sorted(counts.items())),
        "rows": len(task_ids),
        "env_key_sha256": _task_id_digest(task_ids),
        "difficulty": dict(sorted(difficulties.items())),
        "top_sources": dict(sources.most_common(10)),
        "top_hosts": dict(hosts.most_common(25)),
    }


def build_candidates(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f"keep existing fixed candidate manifest: {out}")
        return

    catalog = _load_webgym_catalog("train")
    webvoyager_instructions = _load_webvoyager_instructions(Path(args.webvoyager_tasks))
    popular_ids = _load_popular_task_ids(Path(args.popular))

    counts: Counter[str] = Counter()
    seen: set[str] = set()
    eligible: list[str] = []
    for _, row in pd.read_parquet(args.sft_src).iterrows():
        counts["sft_source_rows"] += 1
        meta = coerce_meta(row["metadata"])
        others = meta.get("others") or {}
        task_id = str(others.get("task_id") or "")
        if not task_id:
            raise SystemExit("SFT source row missing metadata.others.task_id")

        if (others.get("episode_return") or 0) <= 0.5:
            counts["low_return_rows"] += 1
            continue

        if task_id in seen:
            counts["duplicate_rows"] += 1
            continue
        seen.add(task_id)
        counts["unique_success_nogoto"] += 1

        task = catalog[task_id]
        if _norm(task.get("task_name")) in webvoyager_instructions:
            counts["exact_webvoyager_instruction_matches"] += 1
        if _host(task.get("website")) in SEARCH_HOSTS:
            counts["search_start"] += 1
            continue

        counts["eligible"] += 1
        eligible.append(task_id)

    popular = [task_id for task_id in eligible if task_id in popular_ids]
    nonpopular = [task_id for task_id in eligible if task_id not in popular_ids]
    counts["popular_eligible"] = len(popular)
    counts["nonpopular_eligible"] = len(nonpopular)

    actual = {key: counts[key] for key in EXPECTED_CANDIDATE_COUNTS}
    if actual != EXPECTED_CANDIDATE_COUNTS and not args.allow_count_drift:
        raise SystemExit(
            "Anchor-pool counts changed; inspect the no-goto SFT source or "
            "WebGym catalog before writing a new fixed manifest. "
            f"expected={EXPECTED_CANDIDATE_COUNTS} actual={actual}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(popular)
    rng.shuffle(nonpopular)
    ordered = popular + nonpopular

    write_records_to_parquet(_candidate_records(ordered), out)
    summary = _candidate_summary(
        source=Path(args.sft_src),
        task_ids=ordered,
        counts=counts,
        catalog=catalog,
        seed=args.seed,
    )
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def verify_candidates(args: argparse.Namespace) -> None:
    catalog = _load_webgym_catalog("train")
    webvoyager_instructions = _load_webvoyager_instructions(Path(args.webvoyager_tasks))

    df = pd.read_parquet(args.candidate)
    keys = [coerce_meta(row["metadata"])["env_key"] for _, row in df.iterrows()]
    task_ids = [key.split("@", 1)[1] for key in keys]
    digest = _env_key_digest(keys)
    difficulty: Counter[int] = Counter()
    exact_matches = 0

    assert len(df) == args.expected_rows, len(df)
    assert len(keys) == len(set(keys))
    assert digest == args.expected_hash, digest

    for key, task_id in zip(keys, task_ids, strict=True):
        env_id, _ = key.split("@", 1)
        task = catalog[task_id]
        assert env_id == "webgym", key
        assert _host(task["website"]) not in SEARCH_HOSTS, key
        difficulty[int(task["difficulty"])] += 1
        exact_matches += int(_norm(task.get("task_name")) in webvoyager_instructions)

    print(
        f"{args.candidate}: rows={len(df)} env_key_sha256={digest} "
        f"exact_webvoyager_instruction_matches={exact_matches} "
        f"difficulty={dict(sorted(difficulty.items()))} OK"
    )


def _error_kind(error: str | None) -> str | None:
    if not error:
        return None
    lowered = error.lower()
    if "envblocked" in lowered or "blocked" in lowered:
        return "blocked"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "blank" in lowered or "screenshot" in lowered:
        return "screen"
    if "reset" in lowered or "gym.make" in lowered:
        return "reset"
    if "parse" in lowered:
        return "parse"
    return "other"


def _sample_summaries(task_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted(task_dir.glob("sample_*/summary.json"))
    ]


def _classify(
    summaries: list[dict[str, Any]],
    *,
    group_size: int,
    min_valid: int,
    success_threshold: float,
) -> tuple[str, dict[str, Any]]:
    errors = Counter(_error_kind(item.get("error")) for item in summaries if item.get("error"))
    returns = [
        float(item.get("episode_return") or 0.0)
        for item in summaries
        if not item.get("error")
    ]
    num_samples = len(summaries)
    num_valid = len(returns)
    success = sum(value > success_threshold for value in returns)
    infra_errors = sum(errors[kind] for kind in ("blocked", "timeout", "screen", "reset"))

    detail = {
        "num_samples": num_samples,
        "num_valid": num_valid,
        "success": success,
        "mean_return": sum(returns) / num_valid if returns else 0.0,
        "returns": returns,
        "errors": dict(errors),
    }
    if num_samples < group_size:
        return "incomplete", detail
    if num_valid < min_valid or infra_errors:
        return "infra_bad", detail
    if 0 < success < num_valid:
        return "mixed_success", detail
    if success == num_valid:
        return "all_success", detail
    if len(set(returns)) > 1 or max(returns, default=0.0) > 0.0:
        return "all_fail_with_reward_variance", detail
    return "all_fail", detail


def calibrate(args: argparse.Namespace) -> None:
    catalog = _load_webgym_catalog("train")
    candidates = pd.read_parquet(args.candidate)
    if args.head is not None:
        candidates = candidates.head(args.head)
    min_valid = math.ceil(args.group_size * args.min_valid_frac)

    rows: list[dict[str, Any]] = []
    for idx, row in candidates.iterrows():
        meta = coerce_meta(row["metadata"])
        env_key = meta["env_key"]
        _, task_id = env_key.split("@", 1)
        split = str(meta.get("split") or "")
        task_dir = Path(args.log_root) / split / task_id if split else Path(args.log_root) / task_id
        status, detail = _classify(
            _sample_summaries(task_dir),
            group_size=args.group_size,
            min_valid=min_valid,
            success_threshold=args.success_threshold,
        )
        task = catalog[task_id]
        rows.append(
            {
                "idx": idx,
                "env_key": env_key,
                "task_id": task_id,
                "status": status,
                "host": _host(task.get("website")),
                "difficulty": int(task.get("difficulty") or 0),
                "source": str(task.get("benchmark_name")),
                "num_samples": detail["num_samples"],
                "num_valid": detail["num_valid"],
                "success": detail["success"],
                "mean_return": detail["mean_return"],
                "returns_json": json.dumps(detail["returns"]),
                "errors_json": json.dumps(detail["errors"], sort_keys=True),
            }
        )

    priority = {
        "mixed_success": 0,
        "all_fail_with_reward_variance": 1,
        "all_success": 2,
    }
    host_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    all_success_kept = 0
    for row in sorted(rows, key=lambda item: (priority.get(item["status"], 99), item["idx"])):
        if row["status"] not in priority:
            continue
        if host_counts[row["host"]] >= args.max_per_host:
            continue
        if row["status"] == "all_success":
            if all_success_kept >= args.all_success_cap:
                continue
            all_success_kept += 1
        selected.append(row)
        host_counts[row["host"]] += 1
        if len(selected) >= args.target:
            break

    if len(selected) < args.min_final:
        raise SystemExit(
            f"only selected {len(selected)} tasks; inspect calibration health before GRPO"
        )

    selected_indices = [row["idx"] for row in selected]
    records = [
        {
            "problem": candidates.loc[idx, "problem"],
            "metadata": coerce_meta(candidates.loc[idx, "metadata"]),
        }
        for idx in selected_indices
    ]

    out = Path(args.out)
    write_records_to_parquet(records, out)
    selected_keys = [coerce_meta(record["metadata"])["env_key"] for record in records]
    summary = {
        "candidate": str(args.candidate),
        "calibration_log": str(args.log_root),
        "group_size": args.group_size,
        "target": args.target,
        "min_final": args.min_final,
        "rows": len(records),
        "env_key_sha256": _env_key_digest(selected_keys),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "selected_status_counts": dict(Counter(row["status"] for row in selected)),
        "selected_difficulty": dict(Counter(row["difficulty"] for row in selected)),
        "selected_top_sources": dict(Counter(row["source"] for row in selected).most_common(10)),
        "selected_top_hosts": dict(Counter(row["host"] for row in selected).most_common(25)),
    }
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(rows).to_parquet(out.with_suffix(".calibration.parquet"), index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


def write_smoke_log(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.candidate).head(args.head)
    patterns: list[list[float | str]] = [
        [1.0, 0.0, 1.0, 0.0],
        [0.25, 0.0, 0.5, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        ["EnvBlocked", "EnvBlocked", "EnvBlocked", "EnvBlocked"],
        [0.25, 0.0, 0.5, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    if len(df) > len(patterns):
        raise SystemExit(f"smoke head {len(df)} exceeds built-in patterns {len(patterns)}")

    log_root = Path(args.log_root)
    for (_, row), returns in zip(df.iterrows(), patterns, strict=True):
        meta = coerce_meta(row["metadata"])
        _, task_id = meta["env_key"].split("@", 1)
        split = str(meta.get("split") or "")
        task_dir = log_root / split / task_id if split else log_root / task_id
        for sample_idx, value in enumerate(returns[: args.group_size]):
            sample_dir = task_dir / f"sample_{sample_idx:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            if value == "EnvBlocked":
                summary = {
                    "n_turns": 0,
                    "episode_return": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "error": "lite.gym.errors.EnvBlocked: blocked",
                }
            else:
                summary = {
                    "n_turns": 3,
                    "episode_return": float(value),
                    "terminated": True,
                    "truncated": False,
                    "error": None,
                }
            (sample_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    print(f"wrote smoke summaries under {log_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-candidates")
    build.add_argument("--sft-src", default=str(DEFAULT_SFT_SRC))
    build.add_argument("--popular", default=str(DEFAULT_POPULAR))
    build.add_argument("--webvoyager-tasks", default=str(DEFAULT_WEBVOYAGER_TASKS))
    build.add_argument("--out", default=str(DEFAULT_CANDIDATE))
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--allow-count-drift", action="store_true")
    build.set_defaults(func=build_candidates)

    verify = sub.add_parser("verify-candidates")
    verify.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    verify.add_argument("--webvoyager-tasks", default=str(DEFAULT_WEBVOYAGER_TASKS))
    verify.add_argument("--expected-rows", type=int, default=2366)
    verify.add_argument("--expected-hash", default=EXPECTED_CANDIDATE_HASH)
    verify.set_defaults(func=verify_candidates)

    smoke = sub.add_parser("write-smoke-log")
    smoke.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    smoke.add_argument("--log-root", required=True)
    smoke.add_argument("--head", type=int, default=8)
    smoke.add_argument("--group-size", type=int, default=4)
    smoke.set_defaults(func=write_smoke_log)

    cal = sub.add_parser("calibrate")
    cal.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    cal.add_argument("--log-root", required=True)
    cal.add_argument("--out", default=str(DEFAULT_CALIBRATED))
    cal.add_argument("--head", type=int, default=None)
    cal.add_argument("--group-size", type=int, default=4)
    cal.add_argument("--success-threshold", type=float, default=0.5)
    cal.add_argument("--min-valid-frac", type=float, default=0.75)
    cal.add_argument("--target", type=int, default=1024)
    cal.add_argument("--min-final", type=int, default=500)
    cal.add_argument("--max-per-host", type=int, default=128)
    cal.add_argument("--all-success-cap", type=int, default=128)
    cal.set_defaults(func=calibrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
