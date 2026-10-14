#!/usr/bin/env python3
"""Verify that every upstream WAA task can run setup and evaluation.

Run: uv run python lite/gym/envs/waa/scripts/utils/verify_all_tasks.py \\
       --resume   # add --task-id <id> / --domain <d> / --concurrency N to scope
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from lite.core.tools.calls import make_tool_call
from lite.gym.envs.waa.main import CFG, WindowsAgentArenaEnv

ENV_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = ENV_DIR / "data" / "tasks.json"
DEFAULT_OUTPUT = Path(".data/waa/task-smoke")


def _runtime_signature(row: dict[str, Any]) -> str:
    config = row["config"]
    return json.dumps(
        {
            "snapshot": config.get("snapshot"),
            "setup": config.get("config"),
            "evaluator": config.get("evaluator"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _load_tasks(
    path: Path,
    domains: set[str],
    task_ids: set[str],
    variants: set[str],
) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    selected = [
        row
        for row in rows
        if (not domains or row["domain"] in domains)
        and (not task_ids or row["task_id"] in task_ids)
        and (not variants or row["variant"] in variants)
    ]
    missing = task_ids - {row["task_id"] for row in selected}
    if missing:
        raise ValueError("Unknown task IDs: " + ", ".join(sorted(missing)))

    representatives: list[dict[str, Any]] = []
    standard_by_id: dict[str, dict[str, Any]] = {}
    for row in selected:
        copied = {**row, "covered_task_ids": [row["task_id"]]}
        if row["variant"] == "standard":
            standard_by_id[row["task_id"]] = copied
            representatives.append(copied)
            continue

        standard_id = row["task_id"].removeprefix("noctxt_")
        standard = standard_by_id.get(standard_id)
        if standard and _runtime_signature(standard) == _runtime_signature(row):
            standard["covered_task_ids"].append(row["task_id"])
        else:
            representatives.append(copied)
    return representatives


def _result_path(output: Path, task_id: str) -> Path:
    return output / "tasks" / f"{task_id}.json"


def _previously_passed(output: Path, task_id: str) -> bool:
    path = _result_path(output, task_id)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("ok") is True
    except (OSError, ValueError):
        return False


async def _run_attempt(
    row: dict[str, Any],
    *,
    base_disk: Path,
    assets_dir: Path,
    reset_timeout: float,
    evaluate_timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    stage = "construct"
    env = WindowsAgentArenaEnv(
        task_config=row["config"],
        domain=row["domain"],
        variant=row["variant"],
        base_disk=str(base_disk),
        assets_dir=str(assets_dir),
        extra_tools=["terminate"],
    )
    try:
        stage = "setup"
        obs = await asyncio.wait_for(env.reset(), timeout=reset_timeout)
        screenshot = obs.image or b""
        if not screenshot:
            raise RuntimeError("reset returned no screenshot")

        stage = "evaluate"
        result = await asyncio.wait_for(
            env.step([make_tool_call("terminate", {"status": "success"})]),
            timeout=evaluate_timeout,
        )
        if not result.terminated or result.reward is None or not 0.0 <= result.reward <= 1.0:
            raise RuntimeError(f"WAA evaluation did not finish with a valid reward: {result}")
        return {
            "ok": True,
            "task_id": row["task_id"],
            "covered_task_ids": row["covered_task_ids"],
            "domain": row["domain"],
            "setup_ok": True,
            "evaluate_ok": True,
            "reward": result.reward,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        return {
            "ok": False,
            "task_id": row["task_id"],
            "covered_task_ids": row["covered_task_ids"],
            "domain": row["domain"],
            "setup_ok": stage == "evaluate",
            "evaluate_ok": False,
            "failed_stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    finally:
        await env.close()


async def _run_task(
    row: dict[str, Any],
    *,
    output: Path,
    base_disk: Path,
    assets_dir: Path,
    attempts: int,
    reset_timeout: float,
    evaluate_timeout: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        history: list[dict[str, Any]] = []
        for attempt in range(1, attempts + 1):
            result = await _run_attempt(
                row,
                base_disk=base_disk,
                assets_dir=assets_dir,
                reset_timeout=reset_timeout,
                evaluate_timeout=evaluate_timeout,
            )
            result["attempt"] = attempt
            history.append(result)
            if result["ok"]:
                break

        final = {**history[-1], "attempts": history}
        path = _result_path(output, row["task_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(final, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        status = "PASS" if final["ok"] else "FAIL"
        print(
            f"[{status}] {row['domain']}/{row['task_id']} "
            f"attempts={len(history)} elapsed={final['elapsed_s']}s",
            flush=True,
        )
        return final


async def _main(args: argparse.Namespace) -> int:
    base_disk = args.base_disk.expanduser().resolve()
    assets_dir = args.assets_dir.expanduser().resolve()
    if not base_disk.is_file() or base_disk.stat().st_size == 0:
        raise FileNotFoundError(f"WAA base disk is missing or empty: {base_disk}")

    rows = _load_tasks(
        args.tasks,
        set(args.domain),
        set(args.task_id),
        set(args.variant),
    )
    skipped = [
        row["task_id"]
        for row in rows
        if args.resume and _previously_passed(args.output, row["task_id"])
    ]
    selected = [row for row in rows if row["task_id"] not in set(skipped)]
    args.output.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(
        *(
            _run_task(
                row,
                output=args.output,
                base_disk=base_disk,
                assets_dir=assets_dir,
                attempts=args.attempts,
                reset_timeout=args.reset_timeout,
                evaluate_timeout=args.evaluate_timeout,
                semaphore=semaphore,
            )
            for row in selected
        )
    )

    all_results = []
    for row in rows:
        path = _result_path(args.output, row["task_id"])
        if path.is_file():
            all_results.append(json.loads(path.read_text(encoding="utf-8")))
    failures = [result for result in all_results if not result["ok"]]
    covered_tasks = sum(len(row["covered_task_ids"]) for row in rows)
    summary = {
        "covered_tasks": covered_tasks,
        "runtime_cases": len(rows),
        "selected_runtime_cases": len(selected),
        "resumed_runtime_cases": len(skipped),
        "passed": sum(result["ok"] for result in all_results),
        "failed": len(failures),
        "base_disk": str(base_disk),
        "assets_dir": str(assets_dir),
        "failures": [
            {
                "task_id": result["task_id"],
                "covered_task_ids": result.get("covered_task_ids", [result["task_id"]]),
                "domain": result["domain"],
                "failed_stage": result.get("failed_stage"),
                "error_type": result.get("error_type"),
                "error": result.get("error"),
            }
            for result in failures
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failures or len(all_results) != len(rows) else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-disk",
        type=Path,
        default=Path("~/.cache/cua-lite/waa/images/windows11-waa.qcow2"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(CFG.env_kwargs["assets_dir"]),
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--reset-timeout", type=float, default=1200)
    parser.add_argument("--evaluate-timeout", type=float, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--variant",
        action="append",
        choices=("standard", "no_context"),
        default=[],
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
