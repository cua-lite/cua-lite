#!/usr/bin/env python
"""Export the WebArena 241-template eval subset as prompt-data parquet files.

Usage:
    WebArena_SRC=/path/to/BrowserGym uv run python \
      devs/exps/eval/browsergym.webarena/export_template_tasks.py

`WebArena_SRC` is the upstream BrowserGym checkout. BrowserGym's wrapper does
not vendor `webarena/test.raw.json`; at runtime it reads that JSON from the
installed `webarena` package, so this exporter does the same when the source
checkout does not contain a raw-task JSON copy.

The all-task parquet and manifest stay next to this exporter for auditability.
The read/write split parquets live under `lite/gym/envs/browsergym/data/`, next
to other env-owned prompt-data fixtures.
"""

from __future__ import annotations

import argparse
import csv
import importlib.resources as ir
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from lite.gym.envs.browsergym import isolation
from lite.utils.parquet import write_records_to_parquet

ENV_VAR = "WebArena_SRC"
ENV_ID = "browsergym.webarena"
EXPECTED_TEMPLATES = 241


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_out_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_data_dir() -> Path:
    return _repo_root() / "lite/gym/envs/browsergym/data"


def _resolve_browsergym_src() -> Path:
    raw = os.environ.get(ENV_VAR)
    if not raw:
        raise SystemExit(
            f"{ENV_VAR} is required, e.g. "
            f"{ENV_VAR}=~/ref/BrowserGym uv run python {Path(__file__).as_posix()}"
        )
    src = Path(raw).expanduser().resolve()
    metadata = _metadata_csv_path(src)
    if not metadata.is_file():
        raise SystemExit(
            f"{ENV_VAR}={src} does not look like a BrowserGym checkout: "
            f"missing {metadata}"
        )
    return src


def _metadata_csv_path(browsergym_src: Path) -> Path:
    return (
        browsergym_src
        / "browsergym"
        / "experiments"
        / "src"
        / "browsergym"
        / "experiments"
        / "benchmark"
        / "metadata"
        / "webarena.csv"
    )


def _load_browsergym_metadata(browsergym_src: Path) -> dict[str, dict[str, str]]:
    with _metadata_csv_path(browsergym_src).open(encoding="utf-8", newline="") as handle:
        return {row["task_id"]: row for row in csv.DictReader(handle)}


def _raw_json_candidates(browsergym_src: Path) -> list[Path]:
    return [
        browsergym_src / "webarena" / "test.raw.json",
        browsergym_src / "test.raw.json",
        browsergym_src / "browsergym" / "webarena" / "src" / "webarena" / "test.raw.json",
        (
            browsergym_src
            / "browsergym"
            / "webarena"
            / "src"
            / "browsergym"
            / "webarena"
            / "test.raw.json"
        ),
    ]


def _load_webarena_configs(browsergym_src: Path) -> tuple[list[dict[str, Any]], str]:
    for candidate in _raw_json_candidates(browsergym_src):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8")), str(candidate)

    resource = ir.files("webarena").joinpath("test.raw.json")
    return json.loads(resource.read_text(encoding="utf-8")), str(resource)


def _template_groups(
    configs: list[dict[str, Any]],
) -> list[tuple[tuple[int, str], list[dict[str, Any]]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(configs, key=lambda item: item["task_id"]):
        grouped[(row["intent_template_id"], row["intent_template"])].append(row)
    return sorted(grouped.items(), key=lambda item: item[1][0]["task_id"])


def _eval_types(row: dict[str, Any]) -> str:
    return "|".join((row.get("eval") or {}).keys())


def _record(
    row: dict[str, Any],
    rows_for_template: list[dict[str, Any]],
    browsergym_meta: dict[str, str],
    *,
    mutating: bool,
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    template_task_ids = [str(item["task_id"]) for item in rows_for_template]
    return {
        # Rollout resolves the real BrowserGym prompt from metadata.env_key at
        # env reset time; this column is only the prompt-data table placeholder.
        "problem": f"Complete the task: {task_id}",
        "metadata": {
            "env_key": f"{ENV_ID}@{task_id}",
            "split": "eval",
            "intent_template_id": row["intent_template_id"],
            "intent_template": row["intent_template"],
            "source_task_id": task_id,
            "template_task_ids": template_task_ids,
            "mutating": mutating,
            "browsergym_split": browsergym_meta.get("browsergym_split", ""),
            "depends_on": browsergym_meta.get("depends_on", ""),
        },
    }


def _manifest_row(
    row: dict[str, Any],
    rows_for_template: list[dict[str, Any]],
    browsergym_meta: dict[str, str],
    *,
    mutating: bool,
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    return {
        "env_key": f"{ENV_ID}@{task_id}",
        "task_id": task_id,
        "intent_template_id": str(row["intent_template_id"]),
        "intent_template": row["intent_template"],
        "intent": row["intent"],
        "mutating": str(mutating),
        "sites": "|".join(row.get("sites") or []),
        "eval_types": _eval_types(row),
        "browsergym_split": browsergym_meta.get("browsergym_split", ""),
        "depends_on": browsergym_meta.get("depends_on", ""),
        "template_task_ids": "|".join(str(item["task_id"]) for item in rows_for_template),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(out_dir: Path, data_dir: Path, *, expected_templates: int) -> None:
    browsergym_src = _resolve_browsergym_src()
    metadata_by_task = _load_browsergym_metadata(browsergym_src)
    configs, raw_source = _load_webarena_configs(browsergym_src)

    grouped = _template_groups(configs)
    if len(configs) != 812:
        raise SystemExit(f"expected 812 WebArena instantiated intents, found {len(configs)}")
    if len(grouped) != expected_templates:
        raise SystemExit(
            f"expected {expected_templates} exact templates, found {len(grouped)}. "
            "Check the installed webarena package version."
        )

    all_records: list[dict[str, Any]] = []
    read_records: list[dict[str, Any]] = []
    write_records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for (_, _), rows_for_template in grouped:
        row = rows_for_template[0]
        task_id = str(row["task_id"])
        meta = metadata_by_task.get(task_id, {})
        mutating = isolation.is_mutating(row)
        record = _record(row, rows_for_template, meta, mutating=mutating)
        all_records.append(record)
        if mutating:
            write_records.append(record)
        else:
            read_records.append(record)
        manifest.append(_manifest_row(row, rows_for_template, meta, mutating=mutating))

    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    write_records_to_parquet(
        all_records,
        out_dir / "webarena_241_templates.all.prompt_data.parquet",
    )
    write_records_to_parquet(
        read_records,
        data_dir / "webarena_241_templates.read.prompt_data.parquet",
    )
    write_records_to_parquet(
        write_records,
        data_dir / "webarena_241_templates.write.prompt_data.parquet",
    )
    _write_jsonl(out_dir / "webarena_241_templates.manifest.jsonl", manifest)
    _write_csv(out_dir / "webarena_241_templates.manifest.csv", manifest)

    print(f"BrowserGym source: {browsergym_src}")
    print(f"WebArena raw configs: {raw_source}")
    print(f"instantiated intents: {len(configs)}")
    print(f"exact templates: {len(grouped)}")
    print(f"read/non-mutating: {len(read_records)}")
    print(f"write/mutating: {len(write_records)}")
    print(f"wrote audit files: {out_dir}")
    print(f"wrote read/write prompt-data: {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the WebArena 241-template prompt-data subset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_default_out_dir(),
        help="Output directory for all-task parquet + manifest files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Output directory for read/write split prompt-data files.",
    )
    parser.add_argument(
        "--expected-templates",
        type=int,
        default=EXPECTED_TEMPLATES,
        help="Fail if the exact-template count differs from this value.",
    )
    args = parser.parse_args()
    export(args.out_dir, args.data_dir, expected_templates=args.expected_templates)


if __name__ == "__main__":
    main()
