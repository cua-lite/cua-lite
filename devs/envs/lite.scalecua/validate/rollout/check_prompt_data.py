#!/usr/bin/env python3
"""Validate a lite.scalecua rollout prompt-data parquet against current catalog.

Run after generating prompt-data, and rerun after any importer/filter change
before using an older prompt file for clean rollout accounting.

Run:
    uv run python devs/envs/lite.scalecua/validate/rollout/check_prompt_data.py \
        --prompt-data .exps/validate/lite.scalecua/batch/<run-id>.prompt.parquet
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from common import catalog_index, payload_for_exclusion

from lite.gym.envs.lite.scalecua.src.osworld import judges
from lite.gym.envs.lite.scalecua.src.utils import dataset


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    try:
        return dict(value)
    except Exception:
        return {}


def _task_id_from_row(row: dict[str, Any]) -> str | None:
    metadata = _metadata_dict(row.get("metadata"))
    env_key = metadata.get("env_key")
    if isinstance(env_key, str) and "@" in env_key:
        return env_key.split("@", 1)[1]
    task_id = metadata.get("task_id") or metadata.get("id")
    return task_id if isinstance(task_id, str) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--max-examples", type=int, default=40)
    args = parser.parse_args()

    prompt_path = Path(args.prompt_data)
    rows = pd.read_parquet(prompt_path).to_dict(orient="records")
    catalog = catalog_index()
    broken_metrics = dataset._metrics_calling_undefined_helpers(judges.overlay_root())

    missing: list[str] = []
    excluded: list[tuple[str, str]] = []
    split_mismatch: list[tuple[str, str | None, str | None]] = []
    counts = Counter()

    for row in rows:
        task_id = _task_id_from_row(row)
        if not task_id or task_id not in catalog:
            missing.append(task_id or "<missing-task-id>")
            continue
        current = catalog[task_id]
        current_md = current["metadata"]
        current_split = current_md["scalecua"]["runtime_split"]
        prompt_md = _metadata_dict(row.get("metadata"))
        prompt_split = prompt_md.get("split")
        if prompt_split and prompt_split != current_split:
            split_mismatch.append((task_id, str(prompt_split), str(current_split)))
        others = current_md.get("others") or {}
        domain = others.get("domain")
        counts[(current_split, domain)] += 1
        reason = dataset._exclude_reason(
            payload_for_exclusion(current),
            task_id=task_id,
            inherited_exclusion=others.get("exclude_reason"),
            unsupported=[],
            runtime_split=current_split,
            broken_metrics=broken_metrics,
        )
        if reason:
            excluded.append((task_id, str(reason)))

    print(f"prompt={prompt_path}")
    print(f"rows={len(rows)}")
    for (split, domain), count in sorted(counts.items()):
        print(f"count {split}/{domain}: {count}")

    failures = []
    if missing:
        failures.append(f"missing task ids: {len(missing)}")
        for task_id in missing[: args.max_examples]:
            print(f"MISSING {task_id}")
    if split_mismatch:
        failures.append(f"split mismatches: {len(split_mismatch)}")
        for task_id, prompt_split, current_split in split_mismatch[: args.max_examples]:
            print(f"SPLIT_MISMATCH {task_id} prompt={prompt_split} current={current_split}")
    if excluded:
        failures.append(f"current excluded tasks in prompt: {len(excluded)}")
        reason_counts = Counter(reason for _, reason in excluded)
        for reason, count in reason_counts.most_common():
            print(f"excluded_reason {reason}: {count}")
        for task_id, reason in excluded[: args.max_examples]:
            print(f"EXCLUDED {reason} {task_id}")

    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    print("prompt-data validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
