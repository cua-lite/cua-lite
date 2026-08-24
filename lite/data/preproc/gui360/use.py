"""GUI-360 Use Data Preprocessor.

Reconstructs multi-step use trajectories from GUI-360's raw, successful
``train`` trajectories into the canonical ``use`` cohort:

    ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360/desktop/use/<split>/use.parquet

Each raw trajectory is one JSONL file under
``vyokky/GUI-360/train/data/<app>/<category>/success/<exec>.jsonl`` with one
step record per line. A step carries the user ``request`` (goal), the agent's
``thought`` / ``subtask``, the executed ``action`` (structured), the
``screenshot_clean`` path, and a ``status``. Screenshots live in
``train/image.tar.gz`` (extracted by ``scripts/process_raw_data.sh``).

Per the GUI-360 paper, only successful trajectories feed SFT, so this reads the
``train`` split exclusively (``test`` is the held-out benchmark, ``fail`` is for
error analysis). A trajectory is **dropped** if it contains any Office-API
action with no GUI equivalent (e.g. ``set_cell_value``, ``insert_excel_table``);
within kept trajectories, no-op "sub-task complete" steps and ``summary`` steps
(neither changes screen state) are skipped. A terminal ``OVERALL_FINISH`` step
with a GUI action becomes the final EOF action label; a structural terminal
record with no GUI action becomes the content-only ``Done.`` final.

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount
    export CUA_LITE_DATASETS_ROOT=/path/to/canonical-output
    uv run python lite/data/preproc/gui360/use.py \
        [--dry-run] [--verbose] [--head N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any

from PIL import Image

from lite.core.messages import make_assistant_content
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    merge_adjacent_lite_action_batches,
    normalize_keys,
)
from lite.core.tools.action_space.keys import is_canonical_key_token
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate
from lite.data.preproc.gui360.utils import (
    OS,
    PLATFORM,
    SOURCE,
    VK_KEY_MAP,
    make_image_store,
    make_splitter,
    normalize_coordinate,
    out_dir_for,
    stage_entry,
)
from lite.data.utils.messages import (
    extra_tool_schemas_for_messages,
    finalize_use_messages,
    structural_final_message,
)
from lite.utils.path import resolve_path

VARIANT = "use"
TRAIN_REL = "vyokky/GUI-360/train"
APPS = ("excel", "word", "ppt")

# GUI actions with a faithful cua-lite desktop equivalent.
GUI_FUNCTIONS = {
    "click",
    "type",
    "drag",
    "wheel_mouse_input",
    "set_focus",
    "double_click_input",
    "double_click_on_coordinates",
}


class SkipTrajectory(Exception):
    """Raised when a whole trajectory must be dropped (e.g. Office-API action)."""


def parse_keys_string(keys: str, exec_id: str) -> list[dict[str, Any]]:
    """Parse SendKeys text in source order, preserving repeats and hold state."""
    standalone_modifier = re.fullmatch(
        r"\{(?:VK_)?(CONTROL|CTRL|SHIFT|MENU|ALT)\}", keys, re.IGNORECASE
    )
    if standalone_modifier:
        name = standalone_modifier.group(1).lower()
        return [LiteDesktopActionSet.key(keys=[{"control": "ctrl", "menu": "alt"}.get(name, name)])]

    # UFO escapes bare SendKeys modifier glyphs before expanding explicit
    # ``{VK_*}`` markers. Reproduce that order: ``^a`` types the literal text,
    # while ``{VK_CONTROL}a`` is Ctrl+A.
    parts = re.split(r"(\{[^}]*\})", keys)
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("+", "{+}").replace("^", "{^}").replace("%", "{%}")
    keys = "".join(parts)

    for pattern, prefix in (
        (r"\{(?:VK_)?(?:CONTROL|CTRL)\}", "^"),
        (r"\{(?:VK_)?SHIFT\}", "+"),
        (r"\{(?:VK_)?(?:MENU|ALT)\}", "%"),
    ):
        keys = re.sub(pattern, prefix, keys, flags=re.IGNORECASE)

    def canonical(name: str) -> list[str]:
        values: list[str] = []
        for part in [name] if name == "+" else name.split("+"):
            lower = part.lower()
            if lower.startswith("vk_"):
                vk_name = lower[3:]
                if vk_name in VK_KEY_MAP:
                    mapped = VK_KEY_MAP[vk_name]
                elif re.fullmatch(r"[a-z0-9]", vk_name):
                    mapped = vk_name
                else:
                    raise SkipTrajectory(f"Unknown VK token {name!r} in {exec_id}")
            else:
                mapped = VK_KEY_MAP.get(lower, part)
            try:
                value = normalize_keys([mapped])[0]
            except (TypeError, ValueError) as exc:
                raise SkipTrajectory(f"Unknown VK token {name!r} in {exec_id}") from exc
            if not is_canonical_key_token(value):
                raise SkipTrajectory(f"Unknown VK token {name!r} in {exec_id}")
            values.append(value)
        return values

    modifier_keys = {"^": "ctrl", "+": "shift", "%": "alt"}

    def with_modifiers(modifiers: list[str], names: list[str]) -> list[str]:
        return list(dict.fromkeys([*modifiers, *names]))

    def token_calls(token: str, modifiers: list[str]) -> list[dict[str, Any]]:
        parts = token.strip().rsplit(maxsplit=1)
        suffix = parts[1].lower() if len(parts) == 2 else ""
        names = canonical(
            parts[0] if suffix in {"down", "up"} or suffix.isdigit() else token.strip()
        )
        names = with_modifiers(modifiers, names)
        if suffix == "down":
            return [LiteDesktopActionSet.key_down(keys=names)]
        if suffix == "up":
            return [LiteDesktopActionSet.key_up(keys=names)]
        repeat = int(suffix) if suffix.isdigit() else 1
        return [LiteDesktopActionSet.key(keys=names) for _ in range(repeat)]

    def parse(segment: str, held: list[str] | None = None) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        held = held or []
        text = ""

        def flush_text() -> None:
            nonlocal text
            if not text:
                return
            if held:
                calls.extend(
                    LiteDesktopActionSet.key(keys=with_modifiers(held, canonical(char)))
                    for char in text
                )
            else:
                calls.append(LiteDesktopActionSet.type(text=text))
            text = ""

        i = 0
        while i < len(segment):
            if segment[i] == "{":
                flush_text()
                end = segment.find("}", i + 1)
                if end < 0:
                    raise SkipTrajectory(f"Unclosed SendKeys token in {exec_id}")
                calls.extend(token_calls(segment[i + 1:end], held))
                i = end + 1
                continue

            if segment[i] in modifier_keys:
                flush_text()
                modifiers: list[str] = []
                while i < len(segment) and segment[i] in modifier_keys:
                    modifiers.append(modifier_keys[segment[i]])
                    i += 1
                active = with_modifiers(held, modifiers)
                whitespace_start = i
                while i < len(segment) and segment[i] == " ":
                    i += 1
                if i >= len(segment):
                    if i > whitespace_start:
                        calls.append(
                            LiteDesktopActionSet.key(keys=with_modifiers(active, ["space"]))
                        )
                        continue
                    raise SkipTrajectory(f"dangling SendKeys modifier in {exec_id}")
                if segment[i] == "(":
                    depth = 1
                    end = i + 1
                    while end < len(segment) and depth:
                        depth += (segment[end] == "(") - (segment[end] == ")")
                        end += 1
                    if depth:
                        raise SkipTrajectory(f"Unclosed SendKeys group in {exec_id}")
                    calls.extend(parse(segment[i + 1:end - 1], active))
                    i = end
                elif segment[i] == "{":
                    end = segment.find("}", i + 1)
                    if end < 0:
                        raise SkipTrajectory(f"Unclosed SendKeys token in {exec_id}")
                    calls.extend(token_calls(segment[i + 1:end], active))
                    i = end + 1
                else:
                    calls.append(LiteDesktopActionSet.key(
                        keys=with_modifiers(active, canonical(segment[i]))
                    ))
                    i += 1
                continue

            text += segment[i]
            i += 1

        flush_text()
        return calls

    return parse(keys)


def _coord(action: dict[str, Any]) -> tuple[float, float] | None:
    """Action-level click/type coordinate, or None when absent."""
    cx, cy = action.get("coordinate_x"), action.get("coordinate_y")
    if cx is None or cy is None:
        args = action.get("args") or {}
        cx, cy = args.get("x"), args.get("y")
    if cx is None or cy is None:
        return None
    return float(cx), float(cy)


def map_action(
    action: dict[str, Any], width: int, height: int, exec_id: str
) -> tuple[str, list[dict[str, Any]]]:
    """Map one raw step action to ``(kind, tool_calls)``.

    kind ∈ {"emit", "skip"}. Raises SkipTrajectory for Office-API actions and
    malformed GUI actions.
    """
    fn = action.get("function") or ""
    args = action.get("args") or {}

    if fn == "":
        # Empty terminal OVERALL_FINISH and mid-trajectory "sub-task complete"
        # steps are structural no-ops. The trajectory-level final policy appends
        # the content-only final after all real actions are emitted.
        return ("skip", [])

    if fn == "summary":
        return ("skip", [])  # textual state report; no screen change

    if fn not in GUI_FUNCTIONS:
        raise SkipTrajectory(f"Office-API action {fn!r} in {exec_id}")

    def norm() -> list[int]:
        c = _coord(action)
        if c is None:
            raise SkipTrajectory(f"{fn} without coordinate in {exec_id}")
        return normalize_coordinate(c[0], c[1], width, height)

    if fn in ("click", "set_focus"):
        button = args.get("button", "left")
        if button not in {"left", "right", "middle"}:
            raise SkipTrajectory(f"unsupported click button {button!r} in {exec_id}")
        clicks = 2 if args.get("double") else 1
        return ("emit", [LiteDesktopActionSet.click(coordinate=norm(), button=button, clicks=clicks)])

    if fn in ("double_click_input", "double_click_on_coordinates"):
        return ("emit", [LiteDesktopActionSet.click(coordinate=norm(), clicks=2)])

    if fn == "type":
        calls: list[dict[str, Any]] = []
        coord = _coord(action)
        if coord is not None:
            calls.append(LiteDesktopActionSet.click(coordinate=normalize_coordinate(coord[0], coord[1], width, height)))
        if args.get("clear_current_text"):
            calls.append(LiteDesktopActionSet.key(keys=["ctrl", "a"]))
        text = args.get("text")
        keys = args.get("keys")
        if text is not None and text != "":
            calls.append(LiteDesktopActionSet.type(text=text))
        elif keys:
            calls.extend(parse_keys_string(keys, exec_id))
        else:
            raise SkipTrajectory(f"type without text/keys in {exec_id}")
        return ("emit", calls)

    if fn == "drag":
        sx, sy = args.get("start_x"), args.get("start_y")
        ex, ey = args.get("end_x"), args.get("end_y")
        if None in (sx, sy, ex, ey):
            sc, ec = args.get("start_coordinate"), args.get("end_coordinate")
            if not (sc and ec):
                raise SkipTrajectory(f"drag without endpoints in {exec_id}")
            sx, sy = sc
            ex, ey = ec
        drag = LiteDesktopActionSet.drag(
            start_coordinate=normalize_coordinate(float(sx), float(sy), width, height),
            coordinate=normalize_coordinate(float(ex), float(ey), width, height),
            button=args.get("button", "left"),
        )
        key_hold = args.get("key_hold")
        if not key_hold:
            return ("emit", [drag])
        try:
            keys = normalize_keys([key_hold])
        except (TypeError, ValueError) as exc:
            raise SkipTrajectory(f"unsupported drag key_hold {key_hold!r} in {exec_id}") from exc
        if not all(is_canonical_key_token(key) for key in keys):
            raise SkipTrajectory(f"unsupported drag key_hold {key_hold!r} in {exec_id}")
        return (
            "emit",
            [LiteDesktopActionSet.key_down(keys=keys), drag, LiteDesktopActionSet.key_up(keys=keys)],
        )

    if fn == "wheel_mouse_input":
        dist = args.get("wheel_dist")
        if isinstance(dist, bool) or not isinstance(dist, int):
            raise SkipTrajectory(
                f"wheel_mouse_input without integer wheel_dist in {exec_id}"
            )
        if dist == 0:
            return ("skip", [])
        direction = "up" if dist > 0 else "down"
        c = _coord(action)
        coord = normalize_coordinate(c[0], c[1], width, height) if c else None
        return ("emit", [LiteDesktopActionSet.scroll(direction=direction, amount=abs(dist), coordinate=coord)])

    raise SkipTrajectory(f"Unhandled GUI function {fn!r} in {exec_id}")


def load_steps(path: str) -> list[dict[str, Any]]:
    """Read a trajectory JSONL, keeping only the light per-step fields we need.

    Drops the multi-KB ``ui_tree`` / ``control_infos`` blobs immediately to keep
    memory bounded.
    """
    steps = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid GUI-360 JSON in {path} line {line_no}") from e
            step = rec.get("step", {})
            steps.append(
                {
                    "execution_id": rec.get("execution_id"),
                    "request": rec.get("request"),
                    "complete": (rec.get("evaluation") or {}).get("complete"),
                    "status": step.get("status"),
                    "thought": step.get("thought") or "",
                    "subtask": step.get("subtask") or "",
                    "screenshot": step.get("screenshot_clean"),
                    "action": step.get("action") or {},
                }
            )
    return steps


def build_trajectory(path: str, app: str, category: str) -> dict[str, Any]:
    """Build one use entry, raising ``SkipTrajectory`` with an auditable reason."""
    steps = load_steps(path)
    if not steps:
        raise SkipTrajectory(f"empty trajectory {path}")

    exec_id = steps[0].get("execution_id") or os.path.splitext(os.path.basename(path))[0]

    # Only successful, properly-terminated trajectories feed SFT.
    if steps[0].get("complete") != "yes":
        raise SkipTrajectory(f"evaluation is not complete in {exec_id}")
    if steps[-1].get("status") != "OVERALL_FINISH":
        raise SkipTrajectory(f"missing final OVERALL_FINISH in {exec_id}")

    all_images: list[str] = []
    messages: list[dict[str, Any]] = []
    resolution: list[int] | None = None
    goal = steps[0].get("request")
    if not goal:
        raise SkipTrajectory(f"missing request in {exec_id}")
    goal_emitted = False

    for i, step in enumerate(steps):
        screenshot = step["screenshot"]
        if not screenshot:
            raise SkipTrajectory(f"missing screenshot_clean in {exec_id}")
        rel = f"{TRAIN_REL}/image/{app}/{category}/{screenshot}"
        try:
            abs_image = resolve_path(rel, "CUA_LITE_RAW_DATASETS_ROOT")
        except FileNotFoundError as e:
            raise SkipTrajectory(f"missing image {rel}: {e}") from e

        with Image.open(abs_image) as image:
            width, height = image.size
        if step.get("status") == "OVERALL_FINISH":
            if i != len(steps) - 1:
                raise SkipTrajectory(f"OVERALL_FINISH before final step in {exec_id}")
            kind, tool_calls = map_action(step["action"], width, height, exec_id)
            if kind == "emit":
                tool_calls = merge_adjacent_lite_action_batches(tool_calls)
                if resolution is None:
                    resolution = [width, height]
                all_images.append(abs_image)
                image_index = len(all_images) - 1
                user_content: list[dict[str, Any]] = [{"type": "image", "index": image_index}]
                if not goal_emitted:
                    user_content.append({"type": "text", "text": goal})
                    goal_emitted = True
                messages.append({"role": "user", "content": user_content})
                messages.append(
                    {
                        "role": "assistant",
                        "content": make_assistant_content(
                            inline_reasoning=step["thought"],
                            action_description=step["subtask"],
                        ),
                        "tool_calls": tool_calls,
                    }
                )
            elif messages:
                all_images.append(abs_image)
                messages.append({
                    "role": "user",
                    "content": [{"type": "image", "index": len(all_images) - 1}],
                })
            break

        kind, tool_calls = map_action(step["action"], width, height, exec_id)
        if kind == "skip":
            continue
        tool_calls = merge_adjacent_lite_action_batches(tool_calls)

        if resolution is None:
            resolution = [width, height]

        all_images.append(abs_image)
        image_index = len(all_images) - 1
        user_content: list[dict[str, Any]] = [{"type": "image", "index": image_index}]
        if not goal_emitted:
            user_content.append({"type": "text", "text": goal})
            goal_emitted = True
        messages.append({"role": "user", "content": user_content})
        messages.append(
            {
                "role": "assistant",
                "content": make_assistant_content(
                    inline_reasoning=step["thought"],
                    action_description=step["subtask"],
                ),
                "tool_calls": tool_calls,
            }
        )

    ends_on_action = bool(
        messages
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("tool_calls")
    )
    if messages and not ends_on_action:
        messages.append(structural_final_message())
        messages = finalize_use_messages(messages)
    elif messages:
        messages = finalize_use_messages(messages)

    # Need at least one real action step.
    n_assistant = sum(1 for msg in messages if msg.get("tool_calls"))
    if n_assistant < 1:
        raise SkipTrajectory(f"no executable GUI actions in {exec_id}")
    assert len(all_images) >= n_assistant

    return {
        "images": all_images,
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=(PLATFORM, "use"),
            # Derived, not hardcoded: same as every sibling ``use.py``. GUI-360
            # emits no standalone extra tools today, so this is ``[]`` — but if
            # one ever appears the schema follows it automatically instead of
            # silently failing the publish gate.
            extra_tool_schemas=extra_tool_schemas_for_messages(messages),
            valid_actions=None,
            others={
                "id": exec_id,
                "resolution": resolution,
                "os": OS,
                "source": SOURCE,
                "source_id": f"{app}_{category}",
            },
        ).to_dict(),
    }


def iter_trajectory_files(raw_root: str):
    """Yield (path, app, category) for every train trajectory JSONL."""
    base = os.path.join(raw_root, TRAIN_REL, "data")
    for app in APPS:
        for path in sorted(glob.glob(os.path.join(base, app, "*", "success", "*.jsonl"))):
            # .../data/<app>/<category>/success/<exec>.jsonl
            category = path.split(os.sep)[-3]
            yield path, app, category


def main():
    parser = argparse.ArgumentParser(
        description="Process GUI-360 use trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan, write nothing")
    parser.add_argument("--verbose", action="store_true", help="Per-trajectory skip logging")
    parser.add_argument("--head", type=int, default=None, help="Process at most N trajectory files (smoke test)")
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT must be set")
        return 1
    if not os.path.isdir(os.path.join(raw_root, TRAIN_REL, "data")):
        print(f"Error: train data not found under {os.path.join(raw_root, TRAIN_REL, 'data')}")
        return 1

    if args.dry_run:
        files = list(iter_trajectory_files(raw_root))
        print(f"=== DRY RUN ===\nFound {len(files)} train trajectory files across {APPS}")
        print("Would write desktop/use/<split>/use.parquet")
        return 0

    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set")
        return 1

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)

    n_files = 0
    n_office = 0
    n_skip_other = 0
    n_oob = 0
    n_corrupt = 0
    n_kept = 0
    for path, app, category in iter_trajectory_files(raw_root):
        if args.head is not None and n_files >= args.head:
            break
        n_files += 1
        try:
            entry = build_trajectory(path, app, category)
        except SkipTrajectory as e:
            if "Office-API" in str(e):
                n_office += 1
            else:
                n_skip_other += 1
            if args.verbose:
                print(f"  Skip trajectory {os.path.basename(path)}: {e}")
            continue
        if has_oob_coordinate(entry):
            n_oob += 1
            continue
        try:
            bk, e = stage_entry(entry, store=store, splitter=splitter, variant=VARIANT)
        except staging.CorruptImageError as img_err:
            n_corrupt += 1
            if args.verbose:
                print(f"  Skip corrupt trajectory {os.path.basename(path)}: {img_err}")
            continue
        buffers[bk].append(e)
        n_kept += 1

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(
        f"Scanned {n_files} trajectories: kept {n_kept}, "
        f"dropped {n_office} (Office-API), {n_skip_other} (incomplete/empty/other), "
        f"{n_oob} (OOB coords), {n_corrupt} (corrupt images)"
    )
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
