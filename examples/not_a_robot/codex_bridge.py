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
import json
import math
import os
import select
import signal
import sys
import time
from typing import TextIO


def _parse_args(argv=None):
    from .local_tasks import LOCAL_TASKS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(LOCAL_TASKS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--browser-executable")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.max_seconds <= 0 or not math.isfinite(args.max_seconds):
        parser.error("--max-steps and --max-seconds must be positive")
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

    @staticmethod
    def tool_definitions() -> list[dict]:
        """Project the owning Lite action schema into the small MCP surface."""
        from lite.core.tools.action_space.base import LiteDesktopActionSet

        from .catalog import VALID_ACTIONS

        computer = LiteDesktopActionSet.get_tool_schemas(include=VALID_ACTIONS)[0]["function"]
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
                    "are normalized 0..1000 over the complete returned image."
                ),
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "computer",
                "description": (
                    computer["description"]
                    + " Use normalized 0..1000 coordinates across the complete screenshot. "
                    "Wait/hold duration is in seconds, at most 10. Observe before acting."
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
        """Apply a model request and return exactly its model-visible PNG."""
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
        if self.env is None and name != "get_observation":
            raise ValueError("Call get_observation before taking actions or finishing")

        result = None
        if self.env is None:
            # Prepare the first observation only when the model requests it.
            self.env = gym.make(
                f"visual_tasks@{self.args.task}",
                seed=self.args.seed,
                artifact_root=self.args.artifact_root,
                browser_executable=self.args.browser_executable,
                max_steps=self.args.max_steps,
                max_seconds=self.args.max_seconds,
            )
            observation = await self.env.reset()
            raw = self.env.unwrapped
            self.instruction = observation.text
            raw.recorder.emit(
                "controller_started",
                controller="codex_exec_mcp",
                requested_model=self.args.model,
                requested_reasoning_effort=self.args.reasoning_effort,
                actual_model_verified=False,
                private_reasoning_available=False,
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
                based_on=raw.last_observation,
                tool_call=call,
                private_reasoning_available=False,
                ignored_after_terminal=self.terminal,
            )
            if name == "finish":
                self.finish_claim = {"status": arguments["status"], "summary": summary}
            if not self.terminal:
                # Execute through the normal Lite ingress, evaluator and recorder.
                result = await self.env.step([call])
                self.terminal = result.terminated or result.truncated
                self.reward = result.reward
                self.truncated = result.truncated

        # Do not expose evaluation-only state, paths, answers or intermediate frames.
        raw = self.env.unwrapped
        image = (raw.attempt_dir / raw.last_observation["path"]).read_bytes()
        feedback = {
            "instruction": self.instruction,
            "task": self.args.task,
            "viewport": list(raw.display_resolution),
            "coordinate_space": "normalized_0_1000",
            "outcome": raw.outcome,
            "terminal": self.terminal,
            "truncated": self.truncated,
            "reward": self.reward,
            "errors": [item.error for item in result.results if item.error] if result else [],
        }
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
            model_visible_image=raw.last_observation,
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
                for item in result.results
            ]
            if result
            else [],
        )
        response = {
            "content": [
                {"type": "text", "text": json.dumps(feedback)},
                {
                    "type": "image",
                    "data": base64.b64encode(image).decode(),
                    "mimeType": "image/png",
                },
            ],
            "isError": bool(feedback["errors"]),
        }
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
    deadline = time.monotonic() + args.max_seconds
    initialized = False
    pending = b""
    close_reason = "controller_disconnected"
    exit_code = 0
    try:
        while time.monotonic() < deadline:
            # Read available bytes, not buffered readline: partial or coalesced
            # requests must neither block deadlines nor strand buffered lines.
            if b"\n" not in pending:
                if not select.select([input_fd], [], [], 0)[0]:
                    await asyncio.sleep(0.05)
                    continue
                chunk = os.read(input_fd, 65536)
                if not chunk:
                    break
                pending += chunk
                continue
            line, pending = pending.split(b"\n", 1)
            request = None
            request_id = None
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
                                "serverInfo": {"name": "cua-lite-local-game", "version": "1.0.0"},
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
                                async with asyncio.timeout(max(0, deadline - time.monotonic())):
                                    response["result"] = await bridge.call_tool(
                                        params["name"], params.get("arguments", {}), line.decode()
                                    )
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
        await bridge.close(close_reason)
    return exit_code


async def _main(args, protocol_output):
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, task.cancel)
    return await serve(args, input_fd=sys.stdin.fileno(), output=protocol_output)


if __name__ == "__main__":
    protocol_output = sys.stdout
    # Imported libraries and browser diagnostics cannot corrupt the MCP stream.
    with contextlib.redirect_stdout(sys.stderr):
        sys.exit(asyncio.run(_main(_parse_args(), protocol_output)))
