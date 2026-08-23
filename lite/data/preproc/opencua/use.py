"""OpenCUA (AgentNet) use trajectory preprocessor.

Reads xlangai/AgentNet JSONL files, converts pyautogui/computer API
actions to CUA-lite tool calls, and writes the canonical local layout
consumed by ``lite.data.hf.upload`` and ``lite.train.export.export_sft``.

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/datasets
    export CUA_LITE_DATASETS_ROOT=/path/to/output
    uv run python lite/data/preproc/opencua/use.py [--dry-run] [--verbose] [--head N]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from lite.core.messages import make_assistant_content
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    MAX_NORM,
    LiteDesktopActionSet,
    clamp_norm,
    merge_adjacent_lite_action_batches,
    normalize_keys,
)
from lite.core.tools.action_space.keys import is_canonical_key_token
from lite.core.tools.calls import make_tool_call
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate, prestage_images
from lite.data.preproc.opencua.utils import (
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.data.utils.messages import (
    extra_tool_schemas_for_messages,
    finalize_use_messages,
    pop_terminal_terminate,
    structural_final_message,
    terminate_outcome_others,
)
from lite.utils.path import resolve_path

# =============================================================================
# Error Classes
# =============================================================================

class SkipTrajectoryError(Exception):
    """Raised when a trajectory should be skipped due to ambiguous or malformed data."""
    pass

class AgentNetCodeParseError(RuntimeError):
    """Raised when AgentNet code cannot be parsed."""
    pass


def _canonical_key(value: Any, code: str) -> str:
    """Validate one source key before it reaches the string-only JSON schema."""
    if not isinstance(value, str):
        raise AgentNetCodeParseError(f"Key must be a string, got {value!r}.\ncode=\n{code}")
    # One successful AgentNet trajectory spells the numpad multiply key this way.
    value = {"num*": "*"}.get(value, value)
    try:
        normalized = normalize_keys([value])
    except (TypeError, ValueError) as exc:
        raise AgentNetCodeParseError(f"Unsupported key token {value!r}.\ncode=\n{code}") from exc
    if len(normalized) != 1:
        raise AgentNetCodeParseError(
            f"press/hotkey expected one key token, got {value!r}.\ncode=\n{code}"
        )
    key = normalized[0]
    if not is_canonical_key_token(key):
        raise AgentNetCodeParseError(
            f"Unsupported key token {value!r}.\ncode=\n{code}"
        )
    return key

_RESULT_BOUNDARY_TOOLS = frozenset({
    "click",
    "drag",
    "key",
    "mouse_move",
    "scroll",
    "type",
    "wait",
})


def _step_sidecar_metadata(value: dict[str, Any]) -> dict[str, Any] | None:
    """Canonical metadata content for AgentNet per-step analysis sidecars."""
    sidecar: dict[str, Any] = {}
    observation = value.get("observation")
    reflection = value.get("reflection")
    if isinstance(observation, str) and observation:
        sidecar["observation"] = observation
    if isinstance(reflection, str) and reflection:
        sidecar["reflection"] = reflection
    if "last_step_correct" in value and value.get("last_step_correct") is not None:
        sidecar["last_step_correct"] = value.get("last_step_correct")
    if "last_step_redundant" in value and value.get("last_step_redundant") is not None:
        sidecar["last_step_redundant"] = value.get("last_step_redundant")
    return {"opencua_step": sidecar} if sidecar else None

# =============================================================================
# AgentNet Code Parsing Utilities
# =============================================================================

def _dotted_name(expr: ast.AST) -> str:
    """Return dotted name for ast.Name / ast.Attribute chains, else empty string."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _dotted_name(expr.value)
        return f"{base}.{expr.attr}" if base else expr.attr
    return ""

def _literal_eval(node: ast.AST) -> Any:
    """Safely evaluate an AST node to a Python literal."""
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError) as e:
        raise AgentNetCodeParseError(
            f"Failed literal_eval on node={ast.dump(node)}"
        ) from e

def _get_kw(call: ast.Call, name: str) -> ast.AST | None:
    """Get keyword argument value from a call node."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None

def _extract_xy(call: ast.Call) -> tuple[float, float]:
    """Extract x, y from a pyautogui-like call.
    
    Supports keyword x=, y= (preferred), or positional (x, y).
    """
    x_node = _get_kw(call, "x")
    y_node = _get_kw(call, "y")

    if x_node is None and len(call.args) >= 1:
        x_node = call.args[0]
    if y_node is None and len(call.args) >= 2:
        y_node = call.args[1]

    if x_node is None or y_node is None:
        raise AgentNetCodeParseError(
            f"Expected x and y arguments, got args={len(call.args)} keywords={[kw.arg for kw in call.keywords]}"
        )

    x = float(_literal_eval(x_node))
    y = float(_literal_eval(y_node))
    return x, y

def _norm01_to_0_1000(x: float, y: float) -> list[int]:
    """Convert normalized [0, 1] floats -> int [0, 1000] with rounding."""
    eps = 1e-6
    if x < -eps or x > 1 + eps or y < -eps or y > 1 + eps:
        raise AgentNetCodeParseError(
            f"Coordinates out of normalized range [0, 1]: x={x}, y={y}"
        )
    xi = clamp_norm(int(round(x * MAX_NORM)))
    yi = clamp_norm(int(round(y * MAX_NORM)))
    return [xi, yi]

# =============================================================================
# AgentNet Code to CUA-Lite Tool Calls Conversion
# =============================================================================

def agentnet_code_to_tool_calls(code: str) -> list[dict[str, Any]]:
    """Convert AgentNet pyautogui/computer code string into CUA-lite tool calls.
    
    Raises:
        AgentNetCodeParseError: If code cannot be parsed
    """
    if not isinstance(code, str) or not code.strip():
        raise AgentNetCodeParseError(f"Expected non-empty code string. Got: {code!r}")

    try:
        module = ast.parse(code)
    except (TypeError, ValueError, SyntaxError) as e:
        raise AgentNetCodeParseError(f"ast.parse failed for code:\n{code}") from e

    tool_calls: list[dict[str, Any]] = []

    for stmt in module.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            raise AgentNetCodeParseError(
                f"Unsupported statement type: {type(stmt).__name__}. Only expression calls are supported.\ncode=\n{code}"
            )

        call: ast.Call = stmt.value
        fname = _dotted_name(call.func)

        # ---- Mouse clicks ----
        if fname == "pyautogui.click":
            x, y = _extract_xy(call)
            tool_calls.append(
                LiteDesktopActionSet.click(coordinate=_norm01_to_0_1000(x, y))
            )
            continue

        if fname == "pyautogui.rightClick":
            x, y = _extract_xy(call)
            tool_calls.append(
                LiteDesktopActionSet.click(coordinate=_norm01_to_0_1000(x, y), button="right")
            )
            continue

        if fname == "pyautogui.middleClick":
            x, y = _extract_xy(call)
            tool_calls.append(
                LiteDesktopActionSet.click(coordinate=_norm01_to_0_1000(x, y), button="middle")
            )
            continue

        if fname == "pyautogui.doubleClick":
            x, y = _extract_xy(call)
            tool_calls.append(
                LiteDesktopActionSet.click(coordinate=_norm01_to_0_1000(x, y), clicks=2)
            )
            continue

        if fname in {"pyautogui.tripleClick", "computer.tripleClick"}:
            x, y = _extract_xy(call)
            tool_calls.append(
                LiteDesktopActionSet.click(coordinate=_norm01_to_0_1000(x, y), clicks=3)
            )
            continue

        # ---- Mouse movement / drag ----
        if fname == "pyautogui.moveTo":
            x, y = _extract_xy(call)
            tool_calls.append(
                LiteDesktopActionSet.mouse_move(coordinate=_norm01_to_0_1000(x, y))
            )
            continue

        if fname == "pyautogui.dragTo":
            btn_node = _get_kw(call, "button")
            btn = "left" if btn_node is None else str(_literal_eval(btn_node))
            if btn not in {"left", "right", "middle"}:
                raise AgentNetCodeParseError(
                    f"Unsupported dragTo button={btn!r}.\ncode=\n{code}"
                )
            x, y = _extract_xy(call)
            tool_calls.append(
                LiteDesktopActionSet.drag(coordinate=_norm01_to_0_1000(x, y), button=btn)
            )
            continue

        # ---- Scroll ----
        if fname == "pyautogui.scroll":
            if len(call.args) < 1:
                raise AgentNetCodeParseError(
                    f"scroll requires a pixels argument.\ncode=\n{code}"
                )
            pixels = int(_literal_eval(call.args[0]))
            # Positive pixels = scroll up, negative = scroll down
            direction = "up" if pixels > 0 else "down"
            amount = abs(pixels)
            tool_calls.append(
                LiteDesktopActionSet.scroll(direction=direction, amount=amount)
            )
            continue

        if fname == "pyautogui.hscroll":
            if len(call.args) < 1:
                raise AgentNetCodeParseError(
                    f"hscroll requires a pixels argument.\ncode=\n{code}"
                )
            pixels = int(_literal_eval(call.args[0]))
            # Positive pixels = scroll right, negative = scroll left
            direction = "right" if pixels > 0 else "left"
            amount = abs(pixels)
            tool_calls.append(
                LiteDesktopActionSet.scroll(direction=direction, amount=amount)
            )
            continue

        # ---- Keyboard ----
        if fname == "pyautogui.hotkey":
            if len(call.args) != 1:
                raise AgentNetCodeParseError(
                    f"hotkey expected a single list argument.\ncode=\n{code}"
                )
            keys_val = _literal_eval(call.args[0])
            if not isinstance(keys_val, (list, tuple)) or not keys_val:
                raise AgentNetCodeParseError(
                    f"hotkey arg must be a non-empty list/tuple. Got: {keys_val!r}"
                )
            keys = [_canonical_key(k, code) for k in keys_val]
            tool_calls.append(
                LiteDesktopActionSet.key(keys=keys)
            )
            continue

        if fname == "pyautogui.press":
            keys_node = _get_kw(call, "keys")
            if len(call.args) == 1 and keys_node is None:
                keys_node = call.args[0]
            elif len(call.args) != 0 or keys_node is None:
                raise AgentNetCodeParseError(
                    f"press expected a single key argument.\ncode=\n{code}"
                )
            key_val = _literal_eval(keys_node)
            presses_node = _get_kw(call, "presses")
            presses_value = _literal_eval(presses_node) if presses_node is not None else 1
            if isinstance(presses_value, bool) or not isinstance(presses_value, int):
                raise AgentNetCodeParseError(
                    f"presses must be an integer, got {presses_value!r}.\ncode=\n{code}"
                )
            presses = presses_value
            if presses < 1:
                raise AgentNetCodeParseError(
                    f"presses must be positive, got {presses}.\ncode=\n{code}"
                )
            key_values = list(key_val) if isinstance(key_val, (list, tuple)) else [key_val]
            if not key_values:
                raise AgentNetCodeParseError(f"press keys cannot be empty.\ncode=\n{code}")
            keys = [_canonical_key(k, code) for k in key_values]
            for _ in range(presses):
                for key in keys:
                    tool_calls.append(LiteDesktopActionSet.key(keys=[key]))
            continue

        if fname in {"pyautogui.write", "pyautogui.typewrite"}:
            msg_node = _get_kw(call, "message")
            if msg_node is None and len(call.args) == 1:
                msg_node = call.args[0]
            if msg_node is None:
                raise AgentNetCodeParseError(
                    f"write/typewrite requires message argument.\ncode=\n{code}"
                )
            text = str(_literal_eval(msg_node))
            tool_calls.append(
                LiteDesktopActionSet.type(text=text)
            )
            continue

        # ---- Wait / Terminate ----
        if fname == "computer.wait":
            if len(call.args) == 0 and len(call.keywords) == 0:
                t = 1.0
            elif len(call.args) == 1 and len(call.keywords) == 0:
                t = float(_literal_eval(call.args[0]))
            else:
                raise AgentNetCodeParseError(
                    f"Unsupported wait signature.\ncode=\n{code}"
                )
            tool_calls.append(
                LiteDesktopActionSet.wait(duration=t)
            )
            continue

        if fname == "computer.terminate":
            status_node = _get_kw(call, "status")
            if status_node is None:
                raise AgentNetCodeParseError(
                    f"terminate requires status='success'|'failure'.\ncode=\n{code}"
                )
            status = str(_literal_eval(status_node))
            # Normalize status: 'fail' -> 'failure'
            if status == "fail":
                status = "failure"
            if status not in {"success", "failure"}:
                raise AgentNetCodeParseError(
                    f"Unsupported terminate status={status!r}.\ncode=\n{code}"
                )
            tool_calls.append(make_tool_call("terminate", {"status": status}))
            continue

        raise AgentNetCodeParseError(
            f"Unsupported function call: {fname!r}.\ncode=\n{code}"
        )

    return merge_adjacent_lite_action_batches(tool_calls)

# =============================================================================
# Platform/OS
# =============================================================================

#: AgentNet's per-task metadata sidecar, part of the same HF snapshot
#: ``scripts/download_raw_data.sh`` fetches, keyed on the same ``task_id`` the
#: two trajectory files carry.
OS_METADATA_JSONL = "xlangai/AgentNet/meta_data_merged.jsonl"

#: The sidecar's ``system`` spellings -> the canonical ``others.os`` vocabulary
#: shared with the other adapters (``gui360``: ``"windows"``; ``scalecua``:
#: ``"windows"`` / ``"macos"`` / ``"ubuntu"``).
_SYSTEM_TO_OS = {"Windows": "windows", "Darwin": "macos", "Ubuntu": "ubuntu"}


def load_os_by_task_id() -> dict[str, str]:
    """``task_id`` -> canonical ``os``, read from AgentNet's own sidecar.

    The OS is a field the dataset publishes, so this adapter reads it instead of
    inferring one: the ``win_mac`` subset genuinely mixes two systems and its
    records carry no other signal that separates them. Both lookups here are
    total on purpose — an unmapped ``system`` spelling or a ``task_id`` the
    sidecar does not cover means the snapshot and its sidecar disagree, and
    raising says so where a default would publish a guess.
    """
    path = resolve_path(OS_METADATA_JSONL, "CUA_LITE_RAW_DATASETS_ROOT")
    os_by_task_id: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON parse error at line {line_no} in {path}"
                ) from e
            os_by_task_id[record["task_id"]] = _SYSTEM_TO_OS[record["system"]]
    return os_by_task_id

# =============================================================================
# Image Loading
# =============================================================================

def get_image_resolution(image_path: Path) -> list[int] | None:
    """Get image resolution [width, height] from image file."""
    try:
        with PILImage.open(image_path) as im:
            return [im.width, im.height]
    except (OSError, SyntaxError, ValueError):
        return None

# =============================================================================
# Record Processing
# =============================================================================

def record_to_example(
    record: dict[str, Any],
    images_dir: Path,
    relative_root: str,
    dataset_type: str,
    record_idx: int,
    os_by_task_id: dict[str, str],
    duplicate_task_id: bool = False,
) -> dict[str, Any]:
    """Convert one JSONL record into one dataset row.

    Args:
        record: The raw record from JSONL
        images_dir: Absolute path to images directory
        relative_root: Relative path prefix for images (from CUA_LITE_RAW_DATASETS_ROOT)
        dataset_type: 'ubuntu' or 'win_mac'
        record_idx: Index of the record for generating unique ID
        os_by_task_id: :func:`load_os_by_task_id`'s table, read once per run
        duplicate_task_id: whether this is a later occurrence of an upstream id.

    Returns:
        Processed record in CUA-lite format
        
    Raises:
        ValueError: If record is malformed
        SkipTrajectoryError: If trajectory should be skipped
    """
    instruction = record.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(
            f"Missing/invalid 'instruction' in record. Keys={list(record.keys())}"
        )

    traj = record.get("traj")
    if not isinstance(traj, list) or len(traj) == 0:
        raise ValueError(
            f"Missing/invalid 'traj' in record. task_id={record.get('task_id')}"
        )

    # Some AgentNet trajectories emit a spurious terminate mid-stream and then
    # continue (e.g. [..., terminate, scroll, terminate]) or repeat it
    # consecutively ([..., terminate, terminate]). Drop non-final terminate
    # marker steps before images/messages are built so they stay in sync. A
    # trajectory with no terminal terminate is still valid; its final executable
    # action stays as the EOF SFT label below.
    def _step_terminates(step: dict[str, Any]) -> bool:
        code = (step.get("value") or {}).get("code") or ""
        return "computer.terminate" in code

    if traj:
        traj = [s for s in traj[:-1] if not _step_terminates(s)] + [traj[-1]]

    # Collect images and validate
    images: list[str] = []
    resolution: list[int] | None = None

    for i, step in enumerate(traj):
        img_name = step.get("image")
        if not isinstance(img_name, str) or not img_name:
            raise ValueError(
                f"Missing/invalid step.image at traj[{i}]. task_id={record.get('task_id')} step={step}"
            )

        rel_path = os.path.join(relative_root, img_name)
        try:
            abs_path = resolve_path(rel_path, "CUA_LITE_RAW_DATASETS_ROOT")
        except FileNotFoundError as e:
            raise SkipTrajectoryError(
                f"Missing or unresolvable image: {rel_path}"
            ) from e

        # Get resolution from first image
        if resolution is None:
            resolution = get_image_resolution(Path(abs_path))

        # Store absolute path — stage_entry will hash into ImageStore
        images.append(abs_path)

    assert len(images) == len(traj), "images and traj must have same length"

    # Build messages
    messages: list[dict[str, Any]] = []

    # First user message with instruction and first image
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": instruction},
        ],
    })

    # Process each trajectory step
    for i, step in enumerate(traj):
        value = step.get("value")
        if not isinstance(value, dict):
            raise ValueError(
                f"Missing/invalid step.value at traj[{i}] task_id={record.get('task_id')}"
            )

        thought = value.get("thought")
        action_text = value.get("action")
        code = value.get("code")

        if not isinstance(thought, str):
            raise ValueError(
                f"Missing/invalid value.thought at traj[{i}] task_id={record.get('task_id')}"
            )
        if not isinstance(action_text, str):
            raise ValueError(
                f"Missing/invalid value.action at traj[{i}] task_id={record.get('task_id')}"
            )
        if not isinstance(code, str):
            raise ValueError(
                f"Missing/invalid value.code at traj[{i}] task_id={record.get('task_id')}"
            )

        try:
            tool_calls = agentnet_code_to_tool_calls(code)
        except AgentNetCodeParseError as e:
            raise SkipTrajectoryError(
                f"Failed to parse code at step {i}: {e}"
            )

        content = make_assistant_content(
            inline_reasoning=thought or "",
            action_description=action_text or "",
        )
        sidecar = _step_sidecar_metadata(value)
        if sidecar:
            content.append({"type": "metadata", "data": sidecar})

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        if content:
            assistant_msg["content"] = content
        messages.append(assistant_msg)

        # Stage the next screenshot as a user image; finalize_use_messages()
        # rewrites it into a role:"tool" result for the preceding call.
        if i != len(traj) - 1:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "index": i + 1},
                ],
            })

    # No ``terminate`` call is ever persisted: when present, the terminate turn
    # is dropped whole (its reflection and ``last_step_*`` sidecars with it) and
    # any source-asserted failure status/reason moves to ``others``. A trajectory
    # that simply ends on an executable action keeps that action as its EOF SFT
    # label instead of inventing a tool result or ``Done.`` turn.
    terminate_call = pop_terminal_terminate(messages)
    ends_on_action = bool(
        messages
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("tool_calls")
    )
    if not ends_on_action:
        messages.append(structural_final_message())
    messages = finalize_use_messages(messages, result_boundary_tools=_RESULT_BOUNDARY_TOOLS)

    # Build final entry
    task_id = record.get("task_id", f"{dataset_type}_{record_idx}")
    row_id = f"agentnet_{dataset_type}_{task_id}"
    if duplicate_task_id:
        row_id = f"{row_id}_record_{record_idx}"
    entry = {
        "images": images,
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=extra_tool_schemas_for_messages(messages),
            valid_actions=None,
            others={
                "id": row_id,
                "resolution": resolution,
                "os": os_by_task_id[task_id],
                "source": "xlangai/AgentNet",
                "source_id": task_id,
                "domain": record.get("domain"),
                "task_completed": record.get("task_completed"),
                "alignment_score": record.get("alignment_score"),
                "efficiency_score": record.get("efficiency_score"),
                "task_difficulty": record.get("task_difficulty"),
                # The demonstrator's self-report at termination, kept SEPARATE
                # from the externally-judged ``task_completed`` above: on
                # published rows the two disagree (``task_completed=False``
                # alongside ``status="success"``), and the judgement is the more
                # reliable label, so neither may overwrite the other.
                **terminate_outcome_others(terminate_call),
            },
        ).to_dict(),
    }

    return entry


def iter_examples(
    jsonl_path: Path,
    images_dir: Path,
    relative_root: str,
    dataset_type: str,
    os_by_task_id: dict[str, str],
    verbose: bool = False,
    head: int | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield dataset rows from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file
        images_dir: Path to images directory
        relative_root: Relative path prefix for images
        dataset_type: 'ubuntu' or 'win_mac'
        os_by_task_id: :func:`load_os_by_task_id`'s table, read once per run
        verbose: Whether to print progress
        head: Stop after reading this many trajectories (smoke test).

    Yields:
        Processed records in CUA-lite format
    """
    skipped_count = 0
    processed_count = 0
    seen_task_ids: set[str] = set()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if head is not None and line_no > head:
                break
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON parse error at line {line_no} in {jsonl_path}"
                ) from e

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at line {line_no}"
                )

            try:
                task_id = record.get("task_id")
                repeated = isinstance(task_id, str) and task_id in seen_task_ids
                yield record_to_example(
                    record,
                    images_dir=images_dir,
                    relative_root=relative_root,
                    dataset_type=dataset_type,
                    record_idx=line_no,
                    os_by_task_id=os_by_task_id,
                    duplicate_task_id=repeated,
                )
                if isinstance(task_id, str):
                    seen_task_ids.add(task_id)
                processed_count += 1
            except SkipTrajectoryError as e:
                skipped_count += 1
                if verbose:
                    print(f"  Skipping trajectory at line {line_no}: {e}")
                continue

    # Unconditional, like main()'s out-of-bound line: a dropped trajectory is
    # never silent, only its per-line reason is behind --verbose.
    print(f"  Processed: {processed_count}, Skipped: {skipped_count}")

# =============================================================================
# Dataset Configuration
# =============================================================================

DATASETS = {
    "ubuntu": {
        "jsonl": "xlangai/AgentNet/agentnet_ubuntu_5k.jsonl",
        "images": "xlangai/AgentNet/ubuntu_images",
        "description": "Ubuntu 5k trajectories",
    },
    "win_mac": {
        "jsonl": "xlangai/AgentNet/agentnet_win_mac_18k.jsonl",
        "images": "xlangai/AgentNet/win_mac_images",
        "description": "Windows and macOS 18k trajectories",
    },
}

def main():
    parser = argparse.ArgumentParser(
        description="Process AgentNet trajectory data into canonical cua-lite layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed progress information")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate source files without writing output")
    parser.add_argument("--head", type=int, default=None,
                        help="Process at most N trajectories per subset (smoke test)")
    args = parser.parse_args()

    CUA_LITE_RAW_DATASETS_ROOT = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not CUA_LITE_RAW_DATASETS_ROOT:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT environment variable must be set")
        return 1

    # Validate source files exist
    for dataset_type, config in DATASETS.items():
        jsonl_path = Path(CUA_LITE_RAW_DATASETS_ROOT) / config["jsonl"]
        images_dir = Path(CUA_LITE_RAW_DATASETS_ROOT) / config["images"]
        if not jsonl_path.exists():
            print(f"Error: JSONL file not found: {jsonl_path}")
            return 1
        if not images_dir.exists():
            print(f"Error: Images directory not found: {images_dir}")
            return 1
        print(f"  {dataset_type}: {config['description']}")

    if args.dry_run:
        print("\n=== DRY RUN — no files will be written ===")
        return 0

    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT environment variable must be set")
        return 1

    os_by_task_id = load_os_by_task_id()
    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict]] = defaultdict(list)

    print(f"Output directory: {out_dir}")

    n_oob = n_corrupt = 0
    for dataset_type, config in DATASETS.items():
        jsonl_path = Path(CUA_LITE_RAW_DATASETS_ROOT) / config["jsonl"]
        images_dir = Path(CUA_LITE_RAW_DATASETS_ROOT) / config["images"]
        relative_root = config["images"]

        print(f"\nProcessing {dataset_type}...")
        pending: list[dict[str, Any]] = []

        def stage_pending() -> None:
            nonlocal n_corrupt
            if not pending:
                return
            corrupt_paths = prestage_images(
                store, (path for entry in pending for path in entry["images"])
            )
            for entry in pending:
                if any(Path(path) in corrupt_paths for path in entry["images"]):
                    n_corrupt += 1
                    continue
                key, staged = stage_entry(
                    entry, store=store, splitter=splitter, variant=dataset_type,
                )
                buffers[key].append(staged)
            pending.clear()

        for entry in iter_examples(
            jsonl_path=jsonl_path,
            images_dir=images_dir,
            relative_root=relative_root,
            dataset_type=dataset_type,
            os_by_task_id=os_by_task_id,
            verbose=args.verbose,
            head=args.head,
        ):
            if has_oob_coordinate(entry):
                n_oob += 1
                continue
            pending.append(entry)
            if len(pending) == 256:
                stage_pending()
        stage_pending()

    staging.flush_buffers(out_dir, buffers)

    n_rows = sum(len(rs) for rs in buffers.values())
    print(f"Dropped {n_oob} trajectories with out-of-bound coordinates")
    print(f"Dropped {n_corrupt} trajectories with corrupt images")
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    print(f"Image store: {store.count()} unique images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
