"""A bounded, screenshot-driven controller bridge; no model credentials required.

Run via ``python -m examples.not_a_robot.smoke --mode fixture --campaign``.
Each stdin line contains ``actions`` in canonical Lite GUI format and an
optional public ``decision_summary``. ``{"finish": true}`` ends the attempt.
This bridge records supplied decisions, not inaccessible private reasoning.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import select
import sys
import time

from lite import gym
from lite.core.tools.calls import make_tool_call

from . import registration  # noqa: F401
from .env import NealAccessBlocked
from .local_tasks import LOCAL_TASKS


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixture", "live", "local"), default="fixture")
    parser.add_argument("--task", choices=tuple(LOCAL_TASKS), default="click")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--browser-executable")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=180)
    args = parser.parse_args()
    if args.mode == "local" and args.campaign:
        parser.error("Local tasks run independently; --campaign is for the Neal experiment")
    return args


async def run(args) -> int:
    """Use gym.make, return observations, and archive external controller turns."""
    if os.environ.get("CUA_LITE_ENV_SERVER_URL") or os.environ.get("CUA_LITE_ENV_SERVER_PORT"):
        raise RuntimeError("Run this direct-mode smoke without env-server configuration.")
    key = "not_a_robot_campaign@full_game" if args.campaign else "not_a_robot@level_001"
    kwargs = {"mode": args.mode}
    if args.mode == "local":
        key = f"visual_tasks@{args.task}"
        kwargs = {"seed": args.seed}
    env = gym.make(
        key,
        **kwargs,
        artifact_root=args.artifact_root,
        browser_executable=args.browser_executable,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
    )
    raw = env.unwrapped
    exit_code = 0
    try:
        # Reset and expose exactly the model-visible image, not DOM-derived targets.
        observation = await env.reset()
        raw.recorder.emit(
            "controller_started",
            controller="external_stdin",
            instruction=observation.text,
            slurm_job_id=os.getenv("SLURM_JOB_ID"),
        )
        print(
            json.dumps(
                {
                    "type": "ready",
                    "env_key": key,
                    "instruction": observation.text,
                    "image": str(raw.attempt_dir / raw.last_observation["path"]),
                    "viewport": list(raw.display_resolution),
                    "coordinate_space": "normalized_0_1000",
                }
            ),
            flush=True,
        )
        deadline = time.monotonic() + args.max_seconds
        turn = 0
        while time.monotonic() < deadline:
            if not select.select([sys.stdin], [], [], 0)[0]:
                await asyncio.sleep(0.1)
                continue
            line = sys.stdin.readline()
            if not line:
                raw.outcome = "controller_disconnected"
                break
            # Preserve malformed input, retries and public decision summaries.
            raw.recorder.emit("controller_input", raw_text=line, turn=turn)
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Input must be a JSON object")
                turn += 1
                call = make_tool_call(
                    "terminate" if request.get("finish") else "computer",
                    {"status": request.get("status", "failure")}
                    if request.get("finish")
                    else {"actions": request["actions"]},
                    call_id=f"smoke_{turn}",
                )
            except (ValueError, KeyError) as error:
                raw.recorder.emit("controller_input_error", error=str(error), turn=turn)
                print(json.dumps({"type": "input_error", "error": str(error)}), flush=True)
                continue
            raw.recorder.emit(
                "model_decision",
                controller=request.get("controller", "external"),
                summary=request.get("decision_summary"),
                based_on=raw.last_observation,
                tool_call=call,
                private_reasoning_available=False,
            )
            result = await env.step([call])
            # Store returned feedback and image references, not lossy stdout previews.
            results = []
            for item in result.results:
                refs = [raw.recorder.image(png, variant="tool_result") for png in item.images]
                results.append(
                    {
                        "tool_call_id": item.tool_call_id,
                        "images": refs,
                        "text": item.text,
                        "error": item.error,
                        "metadata": item.metadata,
                    }
                )
            raw.recorder.emit(
                "controller_result",
                results=results,
                info=result.info,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
            )
            print(
                json.dumps(
                    {
                        "type": "step",
                        "reward": result.reward,
                        "terminated": result.terminated,
                        "truncated": result.truncated,
                        "outcome": raw.outcome,
                        "image": str(raw.attempt_dir / raw.last_observation["path"]),
                        "errors": [item.error for item in result.results if item.error],
                    }
                ),
                flush=True,
            )
            if result.terminated or result.truncated:
                if result.truncated:
                    exit_code = 1
                break
        else:
            raw.outcome = "controller_timeout"
            exit_code = 1
    except NealAccessBlocked as error:
        exit_code = 2
        print(json.dumps({"type": "access_blocked", "error": str(error)}), flush=True)
    except Exception as error:
        exit_code = 1
        if raw.recorder is not None:
            raw.outcome = "infra_error"
            raw.recorder.emit("error", phase="controller", error=repr(error))
        print(json.dumps({"type": "error", "error": repr(error)}), flush=True)
    finally:
        await env.close()
        print(
            json.dumps(
                {
                    "type": "closed",
                    "outcome": raw.outcome,
                    "state": dataclasses.asdict(raw.state) if raw.state else None,
                    "attempt_dir": str(raw.attempt_dir),
                }
            ),
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse_args())))
