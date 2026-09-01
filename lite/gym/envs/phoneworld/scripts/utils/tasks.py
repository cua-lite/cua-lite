#!/usr/bin/env python
"""Normalize a pinned PhoneWorld checkout into cua-lite's static task manifest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

SOURCE_REVISION = "7d3868adf4163aebb9a3f51babc124a81994d729"
DB_NAMES = {
    "m12306": "railway",
    "malipay": "alipay",
    "mbeike": "beike",
    "mbilibili": "bilibili",
    "mctrip": "ctrip",
    "mdewu": "dewu",
    "mdianping": "dianping",
    "mdidi": "didi",
    "mdouban": "douban",
    "mdouyin": "douyin",
    "meleme": "eleme",
    "mfanqie": "fanqie",
    "mgaode": "gaode",
    "mhema": "hema",
    "miqiyi": "iqiyi",
    "mjd": "jd",
    "mkeep": "keep",
    "mkfc": "kfc",
    "mmcdonalds": "mcdonalds",
    "mmeituan": "meituan",
    "mmeituan_waimai": "meituan_waimai",
    "mnetease_music": "netease_music",
    "mqq": "qq",
    "mqqmusic": "qqmusic",
    "mtaobao": "taobao",
    "mtengxunshipin": "tengxunshipin",
    "mths": "ths",
    "mtoutiao": "toutiao",
    "mweibo": "weibo",
    "mweread": "weread",
    "mxianyu": "xianyu",
    "mxiaohongshu": "xiaohongshu",
    "mximalaya": "ximalaya",
    "mzhihu": "zhihu",
}
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
OPS = {"==", ">", "<", ">=", "<=", "LIKE", "contains"}
DB_PATH = re.compile(
    r"^/data/data/com\.phoneuse\.(?P<app>[A-Za-z][A-Za-z0-9_]*)/databases/[A-Za-z0-9_.-]+$"
)


def normalize_verification(raw: dict, app: str, apps: list[str]) -> dict:
    kind = raw.get("type", "sqlite")
    if kind == "answer_contains":
        return {
            "type": kind,
            "keywords": list(raw.get("keywords", [])),
            "min_length": int(raw.get("min_length", 0)),
        }
    if kind != "sqlite":
        raise ValueError(f"unknown verification type {kind!r}")
    db_path = raw.get("database_path") or raw.get("db_path") or raw.get("database")
    db_path = db_path or f"/data/data/com.phoneuse.{app}/databases/{DB_NAMES[app]}.db"
    match = DB_PATH.fullmatch(db_path)
    if match is None or match.group("app") not in apps:
        raise ValueError(f"database path targets an undeclared app: {db_path!r}")
    checks = raw.get("checks") or [
        {
            "table": raw["table"],
            "rules": raw.get("rules", []),
            "count_min": raw.get("count_min", 1),
        }
    ]
    normalized_checks = []
    for check in checks:
        if not IDENT.fullmatch(check["table"]):
            raise ValueError(f"invalid table {check['table']!r}")
        rules = []
        for rule in check.get("rules", []):
            if not IDENT.fullmatch(rule["field"]) or rule["operator"] not in OPS:
                raise ValueError(f"invalid rule {rule!r}")
            rules.append({k: rule[k] for k in ("field", "operator", "value")})
        normalized_checks.append(
            {
                "table": check["table"],
                "rules": rules,
                "count_min": int(check.get("count_min", 1)),
            }
        )
    result = {"type": kind, "database_path": db_path, "checks": normalized_checks}
    if "answer_contains" in raw:
        result["answer_contains"] = normalize_verification(
            {"type": "answer_contains", **raw["answer_contains"]},
            app,
            apps,
        )
    return result


def load_split(root: Path, split: str) -> list[dict]:
    source = root / ("tasks" if split == "eval" else "tasks_train")
    tasks = []
    for path in sorted(source.rglob("*.json")):
        if "example" in path.relative_to(source).parts:
            continue
        raw = json.loads(path.read_text())
        task_id = raw.get("task_id") or raw.get("id") or path.stem
        apps = list(raw.get("apps") or [raw["app"]])
        if raw["app"] not in apps:
            apps.insert(0, raw["app"])
        if raw.get("app2") and raw["app2"] not in apps:
            apps.append(raw["app2"])
        unknown = set(apps) - DB_NAMES.keys()
        if unknown:
            raise ValueError(f"{path}: unknown apps {sorted(unknown)}")
        tasks.append(
            {
                "task_id": task_id,
                "goal": raw["goal"],
                "apps": apps,
                "difficulty": raw.get("difficulty", "unknown"),
                "verification": normalize_verification(
                    raw["verification"], raw["app"], apps
                ),
            }
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "tasks.json",
    )
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=args.source,
        text=True,
    ).strip()
    if revision != SOURCE_REVISION:
        raise SystemExit(f"PhoneWorld checkout must be at {SOURCE_REVISION}, got {revision}")
    splits = {name: load_split(args.source, name) for name in ("eval", "train")}
    if [len(splits["eval"]), len(splits["train"])] != [120, 300]:
        raise SystemExit("expected exactly 120 eval and 300 train tasks")
    ids = [task["task_id"] for tasks in splits.values() for task in tasks]
    if len(ids) != len(set(ids)):
        raise SystemExit("task ids must be unique across splits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "source_revision": SOURCE_REVISION,
                "splits": splits,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
