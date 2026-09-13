"""Create a stratified lite.scalecua rollout prompt-data parquet.

Default gate: train/rl, 10 domains, 50 tasks per split-domain cell = 1000 tasks.
Rows with metadata.others.exclude_reason are excluded before sampling.

Run:
    uv run python devs/envs/lite.scalecua/validate/rollout/make_batch_prompt_data.py \
        --per-domain 50 --seed 20260714 \
        --output .exps/validate/lite.scalecua/batch/<run-id>.prompt.parquet \
        --manifest .exps/validate/lite.scalecua/batch/<run-id>.manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from lite.utils.parquet import write_records_to_parquet

os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
os.environ.pop("CUA_LITE_ENV_SERVER_TOKEN", None)

from common import catalog_index, payload_for_exclusion  # noqa: E402

import lite.gym as gym  # noqa: E402
from lite.gym.envs.lite.scalecua.src.osworld import judges  # noqa: E402
from lite.gym.envs.lite.scalecua.src.utils import dataset  # noqa: E402


def _task_domain(env_id: str, task_id: str) -> str:
    meta = gym.registry.task_metadata(env_id, task_id)
    domain = meta.others.get("domain")
    if not domain:
        raise RuntimeError(f"{env_id}@{task_id} is missing metadata.others.domain")
    return str(domain)


def _is_runnable(
    env_id: str,
    task_id: str,
    catalog: dict[str, dict[str, Any]],
    *,
    broken_metrics: frozenset[str],
) -> bool:
    meta = gym.registry.task_metadata(env_id, task_id)
    if meta.others.get("exclude_reason"):
        return False
    current = catalog.get(task_id)
    if current is None:
        return True
    current_md = current.get("metadata") or {}
    current_split = (current_md.get("scalecua") or {}).get("runtime_split")
    if not isinstance(current_split, str):
        return True
    reason = dataset._exclude_reason(
        payload_for_exclusion(current),
        task_id=task_id,
        inherited_exclusion=(current_md.get("others") or {}).get("exclude_reason"),
        unsupported=[],
        runtime_split=current_split,
        broken_metrics=broken_metrics,
    )
    return not bool(reason)


def _sample_split(
    *,
    env_id: str,
    split: str,
    per_domain: int,
    rng: random.Random,
    catalog: dict[str, dict[str, Any]],
    broken_metrics: frozenset[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for task_id in gym.registry.task_ids(env_id, split=split):
        if not _is_runnable(
            env_id,
            task_id,
            catalog,
            broken_metrics=broken_metrics,
        ):
            continue
        buckets[_task_domain(env_id, task_id)].append(task_id)

    records: list[dict[str, Any]] = []
    manifest_domains: dict[str, Any] = {}
    for domain in sorted(buckets):
        candidates = sorted(buckets[domain])
        if len(candidates) < per_domain:
            raise SystemExit(
                f"{split}/{domain} has only {len(candidates)} runnable tasks; "
                f"need {per_domain}"
            )
        chosen = sorted(rng.sample(candidates, per_domain))
        manifest_domains[domain] = {"available": len(candidates), "sampled": chosen}
        for task_id in chosen:
            records.append(
                {
                    "problem": f"Complete the task: {task_id}",
                    "metadata": {
                        "env_key": f"{env_id}@{task_id}",
                        "split": split,
                    },
                }
            )

    return records, {"domains": manifest_domains}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="lite.scalecua")
    parser.add_argument("--splits", nargs="+", default=["train", "rl"])
    parser.add_argument("--per-domain", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    catalog = catalog_index()
    broken_metrics = dataset._metrics_calling_undefined_helpers(judges.overlay_root())
    all_records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "env_id": args.env_id,
        "splits": args.splits,
        "per_domain": args.per_domain,
        "seed": args.seed,
        "filter": "not metadata.others.exclude_reason",
        "split_reports": {},
    }
    for split in args.splits:
        records, split_manifest = _sample_split(
            env_id=args.env_id,
            split=split,
            per_domain=args.per_domain,
            rng=rng,
            catalog=catalog,
            broken_metrics=broken_metrics,
        )
        all_records.extend(records)
        manifest["split_reports"][split] = split_manifest

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_records_to_parquet(all_records, out)

    manifest["rows"] = len(all_records)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(all_records)} records to {out}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
