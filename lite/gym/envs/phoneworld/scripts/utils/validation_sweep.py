#!/usr/bin/env python3
"""Verify every PhoneWorld task against one untouched pristine device."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "pyproject.toml").is_file():
        sys.path.insert(0, str(_parent))
        break

from lite.gym.envs.phoneworld import main as phoneworld  # noqa: E402


async def sweep(splits: list[str]) -> dict:
    tasks = [task for split in splits for task in phoneworld._load_tasks()[split]]
    if not tasks:
        raise ValueError("at least one split is required")
    first_id = tasks[0]["task_id"]
    config, _ = phoneworld._TASK_CONFIGS[first_id]
    env = phoneworld._make_env(config=config, task_id=first_id)
    nonzero: list[dict] = []
    errors: list[dict] = []
    try:
        await env.reset()
        loop = asyncio.get_running_loop()
        for task in tasks:
            task_id = task["task_id"]
            env.bind(task_id)
            await env.init_task()
            try:
                reward = await env._evaluate_episode(True, False, loop)
            except Exception as exc:
                errors.append({"task_id": task_id, "error": repr(exc)})
            else:
                if reward != 0.0:
                    nonzero.append({"task_id": task_id, "reward": reward})
    finally:
        await env.close()
    return {
        "splits": splits,
        "tasks": len(tasks),
        "zero_reward": len(tasks) - len(nonzero) - len(errors),
        "nonzero": nonzero,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=("eval", "train"), default=("eval", "train"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = asyncio.run(sweep(list(args.splits)))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n")
    return int(bool(report["nonzero"] or report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
