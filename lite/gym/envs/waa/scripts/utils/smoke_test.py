#!/usr/bin/env python3
"""Cold-boot one prepared WAA image through the public CUA-Lite API.

Run: uv run python lite/gym/envs/waa/scripts/utils/smoke_test.py \\
       ~/.cache/cua-lite/waa/images/windows11-waa.qcow2 --assets-dir ~/.cache/cua-lite/waa/assets
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import lite.gym as gym
from lite.core.tools.calls import make_tool_call
from lite.gym.envs.waa.main import CFG


async def _run(base_disk: Path, assets_dir: Path) -> None:
    task_id = gym.registry.task_ids("waa", split="eval")[0]
    env = gym.make(
        f"waa@{task_id}",
        base_disk=str(base_disk),
        assets_dir=str(assets_dir),
        extra_tools=["terminate"],
        reset_timeout=1200,
        step_timeout=600,
    )
    try:
        obs = await env.reset()
        screenshot = obs.image or b""
        if not screenshot.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("WAA smoke test reset did not return a PNG screenshot")
        evaluated = await env.step([make_tool_call("terminate", {"status": "success"})])
        if evaluated.reward is None or not 0.0 <= evaluated.reward <= 1.0:
            raise RuntimeError(f"WAA smoke test returned invalid reward: {evaluated.reward}")
        print(
            "WAA smoke test passed: "
            f"task={task_id} screenshot_bytes={len(screenshot)} reward={evaluated.reward}"
        )
    finally:
        await env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_disk", type=Path)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(CFG.env_kwargs["assets_dir"]),
    )
    args = parser.parse_args()
    base_disk = args.base_disk.expanduser().resolve()
    assets_dir = args.assets_dir.expanduser().resolve()
    if not base_disk.is_file():
        raise SystemExit(f"Prepared WAA disk does not exist: {base_disk}")
    asyncio.run(_run(base_disk, assets_dir))


if __name__ == "__main__":
    main()
