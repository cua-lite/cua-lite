"""Screenshot-only stdio MCP controller for one local visual-task attempt.

Run ``python -m examples.not_a_robot.codex_bridge --task neal_01
--artifact-root /path/to/run --model gpt-6-astra --reasoning-effort xhigh``.
The client owns model inference; these flags record requested configuration,
not proof of which model the provider actually served. Only screenshots and
canonical GUI actions cross the MCP boundary. No DOM or source-reading tool
is exposed. Stdout is exclusively newline-delimited JSON-RPC.
``finish`` closes the browser and finalizes the archive before replying. Later
calls return the cached final response and are retained only in client logs.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import math
import os
import signal
import sys
import time
from typing import TextIO

from .stdio import ControllerInput

SEQUENCE_MAX_FRAMES = 64
SEQUENCE_MAX_PNG_BYTES = 8 * 1024 * 1024
OBSERVATION_PROTOCOL_VERSION = "1.1.0"


def _parse_args(argv=None):
    from .local_tasks import LOCAL_TASKS, task_reference

    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--task", choices=tuple(LOCAL_TASKS))
    target.add_argument(
        "--campaign", action="store_true", help="Attempt local levels 1 through 48 in order"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-instance", default="default")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--browser-executable")
    parser.add_argument("--record-video", action="store_true", help="Save silent owned-page video")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.max_seconds <= 0 or not math.isfinite(args.max_seconds):
        parser.error("--max-steps and --max-seconds must be positive")
    if not 0 <= args.seed <= 4294967295:
        parser.error("--seed must be an integer from 0 through 4294967295")
    if args.campaign and args.reference_instance != "default":
        parser.error("--reference-instance requires an independent local task")
    if not args.campaign:
        try:
            task_reference(args.task, args.reference_instance)
        except ValueError as error:
            parser.error(str(error))
    return args


class CodexBridge:
    """Own one environment; preserve controller provenance beside its actions."""

    def __init__(self, args):
        if os.environ.get("CUA_LITE_ENV_SERVER_URL") or os.environ.get("CUA_LITE_ENV_SERVER_PORT"):
            raise RuntimeError("Run the Codex bridge without env-server configuration.")
        self.args = args
        self.env = None
        self.instruction = None
        self.turn = 0
        self.terminal = False
        self.reward = None
        self.truncated = False
        self.finish_claim = None
        self.pending_inputs = []
        self.closed = False
        self._close_task = None
        self._final_response = None
        self.deadline = time.monotonic() + args.max_seconds
        self.last_zoom = None
        self.last_returned_observation = None
        self.last_sequence = None
        self.campaign = None

    @staticmethod
    def tool_definitions() -> list[dict]:
        """Project the owning Lite action schema into the small MCP surface."""
        from lite.core.tools.action_space.base import LiteDesktopActionSet

        from .catalog import VALID_ACTIONS

        sequence = {
            "type": "object",
            "properties": {
                "duration_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
                "interval_seconds": {"type": "number", "minimum": 0.1, "maximum": 10},
            },
            "required": ["duration_seconds", "interval_seconds"],
            "additionalProperties": False,
            "description": (
                "Request real-time screenshots after this call's normal observation or actions. "
                "The initial result is frame zero; later screenshots each consume one step. "
                "Duration is the planned window; interval is a target, not a guaranteed rate. "
                "Late sampling slots are skipped. At most 64 frames and 8 MiB of PNGs are "
                "returned, with actual timestamps and explicit partial-result reasons. "
                "No input is sent while sampling. Cannot be combined with region."
            ),
        }
        computer = LiteDesktopActionSet.get_tool_schemas(include=VALID_ACTIONS)[0]["function"]
        computer["parameters"]["properties"]["sequence"] = sequence
        computer["parameters"]["properties"]["decision_summary"] = {
            "type": "string",
            "maxLength": 1000,
            "description": "A short public action rationale; do not supply private reasoning.",
        }
        computer["parameters"]["required"].append("decision_summary")
        computer["parameters"]["additionalProperties"] = False
        return [
            {
                "name": "get_observation",
                "description": (
                    "Start the task and return its screenshot, or refresh the current screenshot. "
                    "Call this first. Later refreshes consume one step. Coordinates in computer "
                    "are normalized 0..1000 over the complete returned image. An optional "
                    "region [left, top, right, bottom] in that same coordinate space adds "
                    "a magnified crop of the screenshot (up to 4x, longest edge at most "
                    "1600 pixels). Use this to inspect small text or ambiguous boundaries. "
                    "Actions always use full-screenshot coordinates, never crop coordinates."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sequence": sequence,
                        "region": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {"type": "number", "minimum": 0, "maximum": 1000},
                            "description": (
                                "[left, top, right, bottom], normalized to the full screenshot."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "computer",
                "description": (
                    computer["description"]
                    + " Use normalized 0..1000 coordinates across the complete screenshot. "
                    'Keys must be lowercase canonical tokens, e.g. ["ctrl", "a"] or '
                    '["enter"], with separate tokens for a chord. '
                    'Arrow tokens are "up", "down", "left", "right"; for ArrowDown use ["down"]. '
                    "Wait/hold duration is in seconds, at most 10. Observe before acting. "
                    "Actions execute in order without an added settle sleep. Use explicit "
                    "wait actions when needed. Real-time animation continues during inference."
                ),
                "inputSchema": computer["parameters"],
            },
            {
                "name": "finish",
                "description": (
                    "End the attempt with a public result summary. Claiming success cannot "
                    "award success: the game itself must have completed."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["success", "failure"]},
                        "summary": {"type": "string", "maxLength": 1000},
                    },
                    "required": ["status", "summary"],
                    "additionalProperties": False,
                },
            },
        ]

    async def call_tool(self, name: str, arguments: dict, raw_request: str) -> dict:
        """Apply a model request and prepare only its explicitly returned screenshots."""
        if self.closed:
            # Episode files are immutable after finish. The outer client retains
            # these subsequent requests; they cannot reopen or change the game.
            return self._final_response or {
                "content": [{"type": "text", "text": "The attempt is already closed."}],
                "isError": True,
            }
        from lite import gym
        from lite.core.tools.calls import make_tool_call

        from . import registration  # noqa: F401

        self.turn += 1
        request = {
            "raw_text": raw_request,
            "tool": name,
            "arguments": arguments,
            "turn": self.turn,
        }
        if self.env is not None and self.env.unwrapped.recorder is not None:
            self.env.unwrapped.recorder.emit("controller_input", **request)
        else:
            self.pending_inputs.append(request)
        # Validate only the MCP boundary; Lite remains the GUI-argument owner.
        definitions = {tool["name"]: tool["inputSchema"] for tool in self.tool_definitions()}
        if name not in definitions:
            raise ValueError(f"Unknown tool: {name}")
        schema = definitions[name]
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        if set(arguments) - set(schema["properties"]) or set(schema.get("required", [])) - set(
            arguments
        ):
            raise ValueError(f"Arguments do not match {name}'s input schema")
        summary = arguments.get("decision_summary", arguments.get("summary"))
        if summary is not None and (not isinstance(summary, str) or len(summary) > 1000):
            raise ValueError("Public summary must be a string of at most 1000 characters")
        if name == "finish" and arguments["status"] not in ("success", "failure"):
            raise ValueError("finish status must be success or failure")
        sequence = arguments.get("sequence")
        if "sequence" in arguments:
            if (
                not isinstance(sequence, dict)
                or set(sequence) != {"duration_seconds", "interval_seconds"}
                or any(type(value) not in (int, float) for value in sequence.values())
                or not 0 < sequence["duration_seconds"] <= 10
                or not 0.1 <= sequence["interval_seconds"] <= 10
                or not all(math.isfinite(value) for value in sequence.values())
            ):
                raise ValueError(
                    "sequence requires finite numeric duration_seconds in (0, 10] and "
                    "interval_seconds in [0.1, 10]"
                )
            if "region" in arguments:
                raise ValueError("sequence and region cannot be combined")
        region = arguments.get("region")
        if name == "get_observation" and "region" in arguments:
            if (
                not isinstance(region, list)
                or len(region) != 4
                or any(
                    type(value) not in (int, float) or not 0 <= value <= 1000 for value in region
                )
                or region[0] >= region[2]
                or region[1] >= region[3]
            ):
                raise ValueError(
                    "region must be [left, top, right, bottom] with positive area in 0..1000"
                )
        if self.env is None and name != "get_observation":
            raise ValueError("Call get_observation before taking actions or finishing")

        result = None
        step_results = []
        if self.env is None:
            # Prepare the first observation only when the model requests it.
            self.env = gym.make(
                "visual_tasks_campaign@full_game"
                if self.args.campaign
                else f"visual_tasks@{self.args.task}",
                seed=self.args.seed,
                **(
                    {"reference_instance": self.args.reference_instance}
                    if not self.args.campaign
                    else {}
                ),
                artifact_root=self.args.artifact_root,
                browser_executable=self.args.browser_executable,
                record_video=self.args.record_video,
                max_steps=self.args.max_steps,
                max_seconds=self.args.max_seconds,
                post_action_delay=0,
            )
            observation = await self.env.reset()
            raw = self.env.unwrapped
            self.instruction = observation.text
            if self.args.campaign:
                from .env import LOCAL_CAMPAIGN_TASKS

                self.campaign = {
                    "completed_tasks": [],
                    "displayed_task": raw.state.task_id,
                    "total_tasks": len(LOCAL_CAMPAIGN_TASKS),
                    "next_task": None,
                    "notice": None,
                    "unexecuted_actions": 0,
                }
            raw.recorder.emit(
                "controller_started",
                controller="codex_exec_mcp",
                observation_protocol_version=OBSERVATION_PROTOCOL_VERSION,
                requested_model=self.args.model,
                requested_reasoning_effort=self.args.reasoning_effort,
                actual_model_verified=False,
                private_reasoning_available=False,
                post_action_delay=0,
                instruction=self.instruction,
                slurm_job_id=os.getenv("SLURM_JOB_ID"),
            )
            for pending in self.pending_inputs:
                raw.recorder.emit("controller_input", **pending)
            self.pending_inputs.clear()
        else:
            raw = self.env.unwrapped
            call = make_tool_call(
                "terminate" if name == "finish" else "computer",
                {"status": arguments["status"]}
                if name == "finish"
                else {
                    "actions": [{"action": "screenshot"}]
                    if name == "get_observation"
                    else arguments["actions"]
                },
                call_id=f"codex_mcp_{self.turn}",
            )
            raw.recorder.emit(
                "model_decision",
                controller="codex_exec_mcp",
                summary=summary,
                based_on=self.last_returned_observation,
                based_on_zoom=self.last_zoom,
                based_on_sequence=self.last_sequence,
                sequence=sequence,
                tool_call=call,
                private_reasoning_available=False,
                ignored_after_terminal=self.terminal,
            )
            if name == "finish":
                self.finish_claim = {"status": arguments["status"], "summary": summary}
            if not self.terminal:
                # Execute through the normal Lite ingress, evaluator and recorder.
                result = await self.env.step([call])
                step_results.append(result)
                self.terminal = result.terminated or result.truncated
                self.reward = result.reward
                self.truncated = result.truncated
                if self.args.campaign:
                    self.campaign = result.info["campaign"]

        # Sample only on this explicit request. The environment remains the owner
        # of GUI steps, terminal grading, campaign transitions and step accounting.
        raw = self.env.unwrapped
        errors = [item.error for entry in step_results for item in entry.results if item.error]
        returned_frames = []
        returned_pngs = []
        sampling = None
        if sequence is not None:
            if raw._terminal and not self.terminal:
                self.terminal = True
                self.truncated = raw.outcome not in ("success", "failure")
                self.reward = float(raw.outcome == "success")
            window_started = time.monotonic()
            window_started_unix = time.time()
            duration = sequence["duration_seconds"]
            interval = sequence["interval_seconds"]
            final_slot = math.ceil(duration / interval)
            slot = 0
            target = 0.0
            request_relative = None
            request_lag = None
            sampling = {
                "requested": sequence,
                "window_started_unix_s": window_started_unix,
                "capture_time_source": "page_screenshot_unix_seconds",
                "schedule_time_source": "controller_monotonic_seconds",
                "first_frame_source": "normal_call_result",
                "status": "partial",
                "stop_reason": None,
                "limit_reached": None,
                "frame_count": 0,
                "png_bytes": 0,
                "skipped_slots": [],
                "frames": [],
            }
            raw.recorder.emit(
                "controller_sequence_started",
                turn=self.turn,
                requested=sequence,
                window_started_unix_s=window_started_unix,
                capture_time_source=sampling["capture_time_source"],
                schedule_time_source=sampling["schedule_time_source"],
            )
        while True:
            reference = raw.last_observation
            png = (raw.attempt_dir / reference["path"]).read_bytes()
            if sampling is not None:
                descriptor = {
                    "index": len(returned_frames),
                    "sha256": reference["sha256"],
                    "viewport": list(raw.display_resolution),
                    "capture_started_unix_s": reference["capture_started_unix_s"],
                    "capture_completed_unix_s": reference["capture_completed_unix_s"],
                    "capture_relative_seconds": (
                        reference["capture_started_unix_s"] - window_started_unix
                    ),
                    "target_seconds": target,
                    "request_relative_seconds": request_relative,
                    "lag_seconds": request_lag,
                }
                raw.recorder.emit(
                    "controller_sequence_frame",
                    turn=self.turn,
                    slot=slot,
                    observation=reference,
                    timing=descriptor,
                    response_delivery="not_prepared",
                )
                if sampling["png_bytes"] + len(png) > SEQUENCE_MAX_PNG_BYTES:
                    sampling["limit_reached"] = "byte_limit"
                    sampling["stop_reason"] = (
                        "infra_error"
                        if raw.outcome == "infra_error"
                        else "tool_error"
                        if errors
                        else "byte_limit"
                    )
                    break
                sampling["frames"].append(descriptor)
                sampling["png_bytes"] += len(png)
            returned_frames.append(reference)
            returned_pngs.append(png)
            if sampling is None:
                break
            sampling["frame_count"] = len(returned_frames)
            if raw.outcome == "infra_error":
                sampling["stop_reason"] = "infra_error"
                break
            if errors:
                sampling["stop_reason"] = "tool_error"
                break
            if result and result.info.get("campaign", {}).get("notice"):
                sampling["stop_reason"] = "campaign_boundary"
                break
            if self.terminal:
                sampling["stop_reason"] = (
                    "step_budget"
                    if raw.outcome == "budget_exhausted" and raw._steps >= self.args.max_steps
                    else "time_budget"
                    if raw.outcome in ("timeout", "budget_exhausted")
                    else "terminal"
                )
                break
            if slot == final_slot:
                sampling["stop_reason"] = "duration_elapsed"
                break
            if len(returned_frames) >= SEQUENCE_MAX_FRAMES:
                sampling["stop_reason"] = "frame_limit"
                sampling["limit_reached"] = "frame_limit"
                break
            if raw._steps >= self.args.max_steps:
                sampling["stop_reason"] = "step_budget"
                break
            # Do not replay elapsed slots after a slow screenshot or disk sync.
            # Sleeping and capturing are serial; no sampler survives this call.
            slot += 1
            while slot <= final_slot:
                target = min(slot * interval, duration)
                if window_started + target < time.monotonic():
                    sampling["skipped_slots"].append(slot)
                    slot += 1
                    continue
                if window_started + target >= self.deadline:
                    sampling["stop_reason"] = "time_budget"
                    break
                await asyncio.sleep(max(0, window_started + target - time.monotonic()))
                request_started = time.monotonic()
                if request_started >= self.deadline:
                    sampling["stop_reason"] = "time_budget"
                    break
                if request_started - (window_started + target) >= interval:
                    # The scheduler itself may wake after an entire sampling
                    # period. Skip that expired slot too, rather than backfill it.
                    sampling["skipped_slots"].append(slot)
                    slot += 1
                    continue
                break
            if sampling["stop_reason"] is not None:
                break
            if slot > final_slot:
                sampling["stop_reason"] = "duration_elapsed"
                break
            request_relative = request_started - window_started
            request_lag = request_relative - target
            result = await self.env.step(
                [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "screenshot"}]},
                        call_id=f"codex_mcp_{self.turn}_sequence_{slot}",
                    )
                ]
            )
            step_results.append(result)
            errors.extend(item.error for item in result.results if item.error)
            self.terminal = result.terminated or result.truncated
            self.reward = result.reward
            self.truncated = result.truncated
            if self.args.campaign:
                self.campaign = result.info["campaign"]
        if sampling is not None:
            sampling["status"] = (
                "complete"
                if sampling["stop_reason"] == "duration_elapsed"
                and not sampling["skipped_slots"]
                and len(returned_frames) == final_slot + 1
                else "partial"
            )
            sampling["elapsed_seconds"] = time.monotonic() - window_started
            raw.recorder.emit("controller_sequence_stopped", turn=self.turn, sequence=sampling)

        # Archive-only frames and evaluator state are never promoted to model input.
        observation = returned_frames[-1] if returned_frames else None
        image = returned_pngs[-1] if returned_pngs else None
        zoom = None
        zoom_bytes = None
        if region is not None:
            from PIL import Image

            with Image.open(io.BytesIO(image)) as frame:
                box = (
                    math.floor(region[0] * frame.width / 1000),
                    math.floor(region[1] * frame.height / 1000),
                    math.ceil(region[2] * frame.width / 1000),
                    math.ceil(region[3] * frame.height / 1000),
                )
                crop = frame.crop(box)
                factor = min(4, 1600 / max(crop.size))
                output_size = [max(1, round(value * factor)) for value in crop.size]
                crop = crop.resize(tuple(output_size), Image.Resampling.NEAREST)
                buffer = io.BytesIO()
                crop.save(buffer, format="PNG")
                zoom_bytes = buffer.getvalue()
            zoom = raw.recorder.image(
                zoom_bytes,
                variant="model_visible_zoom",
                source_sha256=observation["sha256"],
                region=region,
                source_pixel_box=list(box),
                output_size=output_size,
                capture_started_unix_s=observation["capture_started_unix_s"],
                capture_completed_unix_s=observation["capture_completed_unix_s"],
            )
        feedback_time = time.time()
        feedback = {
            "instruction": self.instruction,
            "task": self.args.task,
            "viewport": list(raw.display_resolution),
            "coordinate_space": "normalized_0_1000",
            "outcome": raw.outcome,
            "terminal": self.terminal,
            "truncated": self.truncated,
            "reward": self.reward,
            "clock": "real_time",
            "remaining_seconds": max(0, self.deadline - time.monotonic()),
            "remaining_steps": max(0, self.args.max_steps - raw._steps),
            "observation": {
                "sha256": observation["sha256"],
                "capture_started_unix_s": observation["capture_started_unix_s"],
                "capture_completed_unix_s": observation["capture_completed_unix_s"],
                "feedback_at_unix_s": feedback_time,
                "age_seconds_at_feedback": max(
                    0, feedback_time - observation["capture_completed_unix_s"]
                ),
            }
            if observation is not None
            else None,
            "errors": errors,
        }
        if sampling is not None:
            feedback["sequence"] = sampling
        if zoom is not None:
            feedback["zoom"] = {
                key: zoom[key] for key in ("sha256", "region", "source_pixel_box", "output_size")
            }
        if self.args.campaign:
            feedback["task"] = raw.state.task_id
            feedback["campaign"] = self.campaign
        if self.terminal:
            feedback["notice"] = "The attempt is terminal. Further calls cannot send GUI input."
        if name == "finish":
            feedback["notice"] += (
                " Finish returns only after resources are closed and the archive is finalized."
                " Subsequent calls remain only in the outer client log."
            )
        raw.recorder.emit(
            "controller_result",
            turn=self.turn,
            response_delivery="prepared",
            model_visible_image=observation,
            model_visible_frames=returned_frames,
            model_visible_zoom=zoom,
            feedback=feedback,
            info=result.info if result else None,
            results=[
                {
                    "tool_call_id": item.tool_call_id,
                    "images": [
                        raw.recorder.image(png, variant="tool_result") for png in item.images
                    ],
                    "text": item.text,
                    "error": item.error,
                    "metadata": item.metadata,
                }
                for entry in step_results
                for item in entry.results
            ],
        )
        response = {
            "content": [
                {"type": "text", "text": json.dumps(feedback)},
                *(
                    {
                        "type": "image",
                        "data": base64.b64encode(png).decode(),
                        "mimeType": "image/png",
                    }
                    for png in returned_pngs
                ),
            ],
            "isError": bool(feedback["errors"]),
        }
        if zoom_bytes is not None:
            response["content"].append(
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": base64.b64encode(zoom_bytes).decode(),
                }
            )
        if returned_frames:
            self.last_returned_observation = observation
            self.last_zoom = zoom
            self.last_sequence = sampling
        if name == "finish":
            # Codex can terminate its MCP child immediately after receiving this
            # reply. No cleanup or manifest publication may depend on later EOF.
            await self.close("controller_finished")
            self._final_response = response
        return response

    async def close(self, reason: str) -> None:
        """Finalize once; repeated cancellation cannot interrupt owned cleanup."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_environment(reason))
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                # A client may send SIGTERM more than once during shutdown.
                # Keep the same owned cleanup task alive until it completes.
                continue
        await self._close_task

    async def _close_environment(self, reason: str) -> None:
        if self.env is None:
            self.closed = True
            return
        raw = self.env.unwrapped
        if raw.recorder is not None:
            if raw.outcome == "in_progress":
                raw.outcome = reason
            raw.recorder.emit("controller_closed", reason=reason, finish_claim=self.finish_claim)
        await self.env.close()
        if raw.attempt_dir is not None:
            report = {
                "outcome": raw.outcome,
                "close_reason": reason,
                "finish_claim": self.finish_claim,
                "requested_model": self.args.model,
                "requested_reasoning_effort": self.args.reasoning_effort,
                "actual_model_verified": False,
                "turns": self.turn,
                "manifest": "manifest.json",
            }
            with (raw.attempt_dir / "bridge_result.json").open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        self.closed = True


async def serve(args, *, input_fd: int, output: TextIO) -> int:
    """Run newline JSON-RPC with a wall-clock limit even while stdin is idle."""
    bridge = CodexBridge(args)
    initialized = False
    pending = b""
    close_reason = "controller_disconnected"
    exit_code = 0
    controller_input = ControllerInput(input_fd)
    try:
        while time.monotonic() < bridge.deadline:
            # Read available bytes, not buffered readline: partial or coalesced
            # requests must neither block deadlines nor strand buffered lines.
            if b"\n" not in pending:
                chunk = controller_input.read()
                if chunk is None:
                    await asyncio.sleep(0.05)
                    continue
                if not chunk:
                    break
                pending += chunk
                continue
            line, pending = pending.split(b"\n", 1)
            request = None
            request_id = None
            response_turn = None
            response = {"jsonrpc": "2.0", "id": None}
            try:
                request = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                response["error"] = {"code": -32700, "message": "Invalid JSON"}
            else:
                if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                    response["error"] = {"code": -32600, "message": "Invalid JSON-RPC request"}
                elif "id" not in request:
                    # MCP notifications, including initialized/cancelled, have no reply.
                    continue
                else:
                    request_id = request["id"]
                    response["id"] = request_id
                    method = request.get("method")
                    params = request.get("params", {})
                    try:
                        if not isinstance(params, dict):
                            raise ValueError("params must be an object")
                        if method == "initialize":
                            version = params["protocolVersion"]
                            versions = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
                            response["result"] = {
                                "protocolVersion": version if version in versions else versions[-1],
                                "capabilities": {"tools": {"listChanged": False}},
                                "serverInfo": {
                                    "name": "cua-lite-local-game",
                                    "version": OBSERVATION_PROTOCOL_VERSION,
                                },
                            }
                            initialized = True
                        elif method in ("ping", "shutdown"):
                            response["result"] = {}
                        elif not initialized:
                            response["error"] = {"code": -32002, "message": "Initialize first"}
                        elif method == "tools/list":
                            response["result"] = {"tools": bridge.tool_definitions()}
                        elif method == "tools/call":
                            try:
                                async with asyncio.timeout(
                                    max(0, bridge.deadline - time.monotonic())
                                ):
                                    response["result"] = await bridge.call_tool(
                                        params["name"], params.get("arguments", {}), line.decode()
                                    )
                                    response_turn = bridge.turn
                            except (ValueError, KeyError, TypeError) as error:
                                if bridge.env and bridge.env.unwrapped.recorder:
                                    bridge.env.unwrapped.recorder.emit(
                                        "controller_input_error", error=str(error), turn=bridge.turn
                                    )
                                response["result"] = {
                                    "content": [{"type": "text", "text": str(error)}],
                                    "isError": True,
                                }
                        else:
                            response["error"] = {"code": -32601, "message": "Method not found"}
                    except (ValueError, KeyError, TypeError) as error:
                        response["error"] = {"code": -32602, "message": str(error)}
            output.write(json.dumps(response, allow_nan=False) + "\n")
            output.flush()
            if (
                response_turn is not None
                and bridge.env is not None
                and bridge.env.unwrapped.recorder is not None
            ):
                # A successful write is transport evidence, not proof of model
                # receipt or understanding. Finish has already closed its archive.
                bridge.env.unwrapped.recorder.emit(
                    "controller_response_sent", turn=response_turn, response_id=request_id
                )
            if isinstance(request, dict) and request.get("method") == "shutdown":
                close_reason = "controller_shutdown"
                break
        else:
            close_reason, exit_code = "controller_timeout", 1
    except TimeoutError:
        close_reason, exit_code = "controller_timeout", 1
    except asyncio.CancelledError:
        close_reason, exit_code = "controller_cancelled", 1
    except BrokenPipeError:
        close_reason, exit_code = "controller_disconnected", 1
    except Exception as error:
        close_reason, exit_code = "infra_error", 1
        print(f"Codex bridge failed: {error!r}", file=sys.stderr)
        if bridge.env and bridge.env.unwrapped.recorder:
            bridge.env.unwrapped.outcome = "infra_error"
            bridge.env.unwrapped.recorder.emit("error", phase="codex_bridge", error=repr(error))
    finally:
        controller_input.close()
        await bridge.close(close_reason)
    return exit_code


async def _main(args, protocol_output):
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    previous_handlers = {}
    signals = (signal.SIGTERM, signal.SIGINT)
    if os.name == "nt":
        signals += (signal.SIGBREAK,)
    for signum in signals:
        if os.name == "nt":
            previous_handlers[signum] = signal.signal(
                signum, lambda *_: loop.call_soon_threadsafe(task.cancel)
            )
        else:
            loop.add_signal_handler(signum, task.cancel)
    try:
        return await serve(args, input_fd=sys.stdin.fileno(), output=protocol_output)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    protocol_output = sys.stdout
    # Imported libraries and browser diagnostics cannot corrupt the MCP stream.
    with contextlib.redirect_stdout(sys.stderr):
        sys.exit(asyncio.run(_main(_parse_args(), protocol_output)))
