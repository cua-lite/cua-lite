"""UI-Genie-Agent → cua-lite mobile ``use`` preprocessor.

Converts ``HanXiao1999/UI-Genie-Agent-16k`` (two subsets) into the canonical
``use`` cohort, with the subset as the variant:

    cua-lite/UI-Genie-Agent/mobile/use/<split>/{ui_genie,amex}.parquet

Each JSONL line is one *step*: ``messages`` = [system (carries "resolution is
WxH"), user (carries "The user query: <goal>"), assistant (one
``<tool_call>`` with a ``mobile_use`` action dict)] and ``images=[path]``. Steps
are grouped into trajectories by uid (parsed from the image path) and sorted by
step index.

``ui_genie_agent_16k.jsonl`` ships **two annotation modalities** under one
filename, and only one of them is expressible in the canonical vocabulary; see
:func:`step_resolution`, which is where a trajectory is assigned to one.

Action mapping (``mobile_use`` action → ``LiteMobileActionSet``):
  click→tap, long_press→long_press, swipe(coordinate,coordinate2 | direction)→swipe,
  type→type, system_button→system_button, open→open_app, wait→wait,
  terminate→terminate. ``som``-referenced coords / ``key`` / unknown action →
  skip the whole trajectory. Coordinates are pixels in the per-step screen
  resolution → normalized to [0,1000]; truly out-of-bounds coordinates skip the
  trajectory before endpoint clamping handles rounding noise. Each non-terminate step's
  ``action_desc`` → ``action_description`` content part. A trajectory without a final
  ``terminate`` keeps its last executable action as the EOF label. A ``terminate`` step is never
  persisted as a tool call: it is replaced by the content-only final, whose text is the
  task's ANSWER when ``terminate``'s ``action_info`` carries one (``terminal_answer_text``
  -- ``ui_genie`` hangs QA answers off ``terminate``, which the canonical vocabulary
  reserves for ``response``) and the plain ``Done.`` marker otherwise (matching
  ``opencua/use.py``). Source-asserted non-success status/reason payloads move to
  ``metadata.others`` via ``terminate_outcome_others``.

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount
    export CUA_LITE_DATASETS_ROOT=/path/to/canonical-output
    uv run python lite/data/preproc/ui_genie_agent/use.py \
        [--subset all|ui_genie|amex] [--dry-run] [--verbose] [--head N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.messages import make_assistant_content
from lite.core.tools.action_space import (
    MAX_NORM,
    LiteMobileActionSet,
    clamp_norm,
    merge_adjacent_lite_action_batches,
)
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate
from lite.data.preproc.ui_genie_agent.utils import (
    OS,
    PLATFORM,
    SOURCE,
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

# Synthetic swipe coordinates for direction-only swipes (already in [0,1000]).
DIRECTION_SWIPES: dict[str, tuple[list[int], list[int]]] = {
    "up": ([500, 700], [500, 300]),
    "down": ([500, 300], [500, 700]),
    "left": ([700, 500], [300, 500]),
    "right": ([300, 500], [700, 500]),
}
_COORD_EPS = 1e-6

SUBSETS: dict[str, dict[str, str]] = {
    "ui_genie": {
        "jsonl_rel": "HanXiao1999/UI-Genie-Agent-16k/ui_genie_agent_16k.jsonl",
        "image_root_rel": "HanXiao1999/UI-Genie-Agent-16k",
        "image_dir_check_rel": "HanXiao1999/UI-Genie-Agent-16k/data/screenshots",
        "id_prefix": "ui_genie_16k",
    },
    "amex": {
        "jsonl_rel": "HanXiao1999/UI-Genie-Agent-16k/AMEX_Agent_34K.jsonl",
        "image_root_rel": "HanXiao1999/UI-Genie-Agent-16k",
        "image_dir_check_rel": "HanXiao1999/UI-Genie-Agent-16k/AMEX/screenshot",
        "id_prefix": "ui_genie_amex",
    },
}

_UI_GENIE_STEP_RE = re.compile(r"^screenshot-(\d+)\.png$")
_AMEX_STEP_RE = re.compile(r"^(.+)-(\d+)\.png$")
_RES_RE = re.compile(r"resolution is (\d+)x(\d+)")
# The set-of-mark modality's own words. See ``step_resolution``: this is the
# discriminator between UI-Genie's two annotation modalities, and it is checked
# BEFORE the resolution, because "no resolution" is a CONSEQUENCE of being the
# SOM variant rather than a defect of the record.
_SOM_PROMPT_MARK = "labeled with numeric tags"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_PROGRESS_STEP_RE = re.compile(r"\bStep\s*(\d+):")


class SkipTrajectoryError(Exception):
    """A trajectory dropped for unpublishable data, with its ledger key.

    ``reason`` is a required positional argument, not an optional label: the
    catch site counts drops by it, so a drop nobody can count is
    unconstructible, and a ``raise`` added later brings its own bucket to the
    run's ``Skip reasons:`` line. Same shape as ``scalecua/use.py``'s.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def _parse_uid_step(image_path: str, subset: str) -> tuple[str, int] | None:
    try:
        parts = image_path.split("/")
        if subset == "ui_genie":
            if len(parts) < 4:
                return None
            m = _UI_GENIE_STEP_RE.match(parts[3])
            return (parts[2], int(m.group(1))) if m else None
        if len(parts) < 3:
            return None
        m = _AMEX_STEP_RE.match(parts[-1])
        return (m.group(1), int(m.group(2))) if m else None
    except (ValueError, IndexError):
        return None


def step_resolution(system_msg: str) -> tuple[int, int]:
    """The step's coordinate frame, or a skip naming the modality that has none.

    ``ui_genie_agent_16k.jsonl`` interleaves **two annotation modalities** whose
    system prompts are disjoint, and the coordinate frame only exists in one of
    them:

    * **point** (14,282 of 16,698 steps) -- "The screen's resolution is WxH",
      and ``click``/``long_press``/``swipe`` carry ``coordinate`` in that frame.
    * **set-of-mark** (2,416 steps, 417 of 2,208 trajectories) -- "The
      interactive UI elements on the screenshot are labeled with numeric tags
      starting from 1", the screenshots have those tags **burned into the
      pixels**, and the actions carry ``som: <int>`` instead of a coordinate.
      It declares no resolution *by design*: there is nothing to normalize.

    The modality is therefore decided FIRST and the resolution read second. That
    ordering is the whole point of this function existing: read the other way
    round -- which is what this adapter used to do -- a set-of-mark trajectory is
    reported as ``missing/malformed resolution in system prompt``, so 417
    trajectories (18.9% of the subset, 65% of all UI-Genie loss) blame the source
    for a malformed prompt when the truth is that the corpus holds a second
    modality the canonical ``use`` vocabulary cannot express. Measured over the
    whole subset, the two sets coincide exactly in both directions: every one of
    the 2,416 steps lacking a resolution carries the SOM marker, and none of the
    14,282 resolution-declaring steps does.

    The residual ``malformed_resolution`` bucket is the third outcome, not a
    guard: it is 0 on today's snapshot for both subsets, and it is what a
    genuinely broken point-modality prompt would be counted as.
    """
    if _SOM_PROMPT_MARK in system_msg:
        raise SkipTrajectoryError(
            "som_annotation_variant",
            "set-of-mark prompt: actions reference burned-in numeric element "
            "tags, not coordinates, and the record carries no tag->coordinate map",
        )
    m = _RES_RE.search(system_msg)
    if m is None:
        raise SkipTrajectoryError(
            "malformed_resolution",
            "point-annotation prompt declares no 'resolution is WxH'",
        )
    return int(m.group(1)), int(m.group(2))


def _parse_tool_call(content: str) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    m = _TOOL_CALL_RE.search(content)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    args = obj.get("arguments") if isinstance(obj, dict) else None
    return args if isinstance(args, dict) else None


def _parse_query(content: str) -> str | None:
    if not isinstance(content, str) or "The user query:" not in content:
        return None
    s = content.split("The user query:", 1)[1]
    if "\nTask progress" in s:
        s = s.split("\nTask progress", 1)[0]
    return s.strip() or None


def _logical_step(record: dict[str, Any]) -> int:
    """Current action index from the source's accumulated Task progress."""
    content = record["messages"][1]["content"]
    history = content.partition("Task progress")[2]
    steps = [int(value) for value in _PROGRESS_STEP_RE.findall(history)]
    return max(steps) if steps else 0


def terminal_answer_text(action_info: str | None) -> str | None:
    """Preserve source-authored successful terminal text without language guessing."""
    if not isinstance(action_info, str):
        return None
    stripped = action_info.strip()
    inner = (
        stripped[1:-1]
        if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"'
        else action_info
    )
    if inner and inner[-1].isspace():
        raise SkipTrajectoryError(
            "truncated_terminal_text", f"terminal action_info ends mid-phrase: {action_info!r}"
        )
    return stripped or None


def _norm_xy(x: float, y: float, width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0:
        raise SkipTrajectoryError(
            "non_positive_resolution", f"{width}x{height}")
    if (
        x < -_COORD_EPS
        or x > width + _COORD_EPS
        or y < -_COORD_EPS
        or y > height + _COORD_EPS
    ):
        raise SkipTrajectoryError(
            "coordinate_outside_resolution",
            f"coordinate ({x}, {y}) outside resolution {width}x{height}",
        )
    return [clamp_norm(int(round(x / width * MAX_NORM))),
            clamp_norm(int(round(y / height * MAX_NORM)))]


def _coerce_xy(coord: Any) -> tuple[float, float] | None:
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None
    try:
        return float(coord[0]), float(coord[1])
    except (TypeError, ValueError):
        return None


def _duration(args: dict[str, Any], *, default: float | None) -> float | None:
    if "time" not in args:
        return default
    if isinstance(args["time"], bool):
        raise SkipTrajectoryError("malformed_duration", repr(args["time"]))
    try:
        duration = float(args["time"])
    except (TypeError, ValueError) as e:
        raise SkipTrajectoryError("malformed_duration", repr(args["time"])) from e
    if not math.isfinite(duration) or duration <= 0:
        raise SkipTrajectoryError("malformed_duration", repr(args["time"]))
    return duration


def args_to_tool_call(args: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """One mobile_use ``arguments`` dict → one cua-lite tool_call (raises to skip).

    There is deliberately no ``som`` branch here. A ``som``-referenced action is
    reachable only from a set-of-mark prompt, and :func:`step_resolution` has
    already skipped the whole trajectory by then -- exhaustively, not by sample:
    all 1,768 ``som``-bearing steps sit under a SOM prompt, and 0 of the 14,282
    point-modality steps carry the key. A branch here would restate that
    classification, and it is the pair of them that produced the
    mis-attribution this function's caller documents.
    """
    action = args.get("action")
    if not isinstance(action, str):
        raise SkipTrajectoryError("missing_action_field", repr(args))

    if action == "click":
        xy = _coerce_xy(args.get("coordinate"))
        if xy is None:
            raise SkipTrajectoryError("missing_coordinate", f"click: {args!r}")
        return LiteMobileActionSet.tap(coordinate=_norm_xy(*xy, width, height))

    if action == "long_press":
        xy = _coerce_xy(args.get("coordinate"))
        if xy is None:
            raise SkipTrajectoryError("missing_coordinate", f"long_press: {args!r}")
        return LiteMobileActionSet.long_press(
            coordinate=_norm_xy(*xy, width, height),
            duration=_duration(args, default=None),
        )

    if action == "swipe":
        c1, c2 = _coerce_xy(args.get("coordinate")), _coerce_xy(args.get("coordinate2"))
        if c1 is not None and c2 is not None:
            return LiteMobileActionSet.swipe(
                start_coordinate=_norm_xy(*c1, width, height),
                coordinate=_norm_xy(*c2, width, height))
        direction = args.get("direction")
        if isinstance(direction, str) and direction in DIRECTION_SWIPES:
            start, end = DIRECTION_SWIPES[direction]
            return LiteMobileActionSet.swipe(start_coordinate=list(start), coordinate=list(end))
        raise SkipTrajectoryError(
            "swipe_without_endpoints", f"no usable coordinates / direction: {args!r}")

    if action == "type":
        text = args.get("text")
        if not isinstance(text, str) or text == "":
            raise SkipTrajectoryError("empty_text", f"type: {args!r}")
        return LiteMobileActionSet.type(text=text)

    if action == "system_button":
        button = args.get("button")
        if button not in {"Back", "Home", "Enter", "Menu", "Recent"}:
            raise SkipTrajectoryError(
                "unsupported_system_button", f"button={button!r}")
        return LiteMobileActionSet.system_button(button=button)

    if action == "open":
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SkipTrajectoryError("empty_text", f"open: {args!r}")
        return make_tool_call("open_app", {"app_name": text})

    if action == "wait":
        return LiteMobileActionSet.wait(duration=_duration(args, default=1.0))

    if action == "terminate":
        status = args.get("status")
        if status not in {"success", "failure"}:
            raise SkipTrajectoryError(
                "unsupported_terminate_status", f"status={status!r}")
        reason = args.get("action_info")
        if not isinstance(reason, str) or not reason.strip():
            reason = None
        return make_tool_call("terminate", {"status": status, "reason": reason})

    if action == "key":
        raise SkipTrajectoryError(
            "key_action", "ADB keyevent not supported in lite:mobile")
    raise SkipTrajectoryError("unmapped_action", f"action={action!r}")


def group_records_by_uid(jsonl_path: str, subset: str) -> dict[str, list[tuple[int, dict]]]:
    uid_records: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON in {jsonl_path} line {line_no}") from e
            if not isinstance(rec, dict):
                raise ValueError(f"expected object in {jsonl_path} line {line_no}")
            images = rec.get("images")
            if not isinstance(images, list) or not images or not isinstance(images[0], str):
                raise ValueError(f"invalid images in {jsonl_path} line {line_no}")
            parsed = _parse_uid_step(images[0], subset)
            if parsed is None:
                raise ValueError(
                    f"unrecognized image path in {jsonl_path} line {line_no}: {images[0]!r}"
                )
            uid, step = parsed
            uid_records[uid].append((step, rec))
    return uid_records


def build_trajectory(uid: str, records: list[tuple[int, dict]], cfg: dict) -> dict:
    """Build one canonical use entry from a uid's grouped step records."""
    first_system = records[0][1]["messages"][0]["content"]
    if _SOM_PROMPT_MARK in first_system:
        step_resolution(first_system)
    records = [(_logical_step(record), record) for _, record in records]
    steps_seen = [step for step, _ in records]
    if len(steps_seen) != len(set(steps_seen)):
        raise SkipTrajectoryError(
            "duplicate_step_numbers", f"{sorted(steps_seen)}")
    records = sorted(records, key=lambda x: x[0])
    steps_seen = [s for s, _ in records]
    if any(right != left + 1 for left, right in zip(steps_seen, steps_seen[1:])):
        raise SkipTrajectoryError(
            "non_contiguous_step_numbers", f"{steps_seen}")

    instruction = _parse_query(records[0][1]["messages"][1]["content"])
    if not instruction:
        raise SkipTrajectoryError("no_instruction", "could not parse 'The user query:'")

    images: list[str] = []
    assistant_turns: list[dict] = []
    # ``others.resolution`` is the FIRST step's frame, read from the same call
    # that reads every other step's -- there is one owner of the question "what
    # frame is this step in?", so the trajectory cannot be classified one way for
    # its metadata and another way for its coordinates.
    res_first: tuple[int, int] | None = None
    for record_pos, (step_idx, rec) in enumerate(records):
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 3:
            raise SkipTrajectoryError(
                "malformed_messages", f"step {step_idx}")
        res = step_resolution(msgs[0]["content"])
        if res_first is None:
            res_first = res
        args = _parse_tool_call(msgs[2].get("content"))
        if args is None:
            raise SkipTrajectoryError("malformed_tool_call", f"step {step_idx}")
        tool_call = args_to_tool_call(args, width=res[0], height=res[1])

        full_rel = os.path.join(cfg["image_root_rel"], rec["images"][0])
        try:
            abs_path = resolve_path(full_rel, "CUA_LITE_RAW_DATASETS_ROOT")
        except FileNotFoundError as e:
            raise SkipTrajectoryError(
                "image_absent_on_host", f"step {step_idx}: {full_rel}") from e

        tool_calls = merge_adjacent_lite_action_batches([tool_call])
        content = make_assistant_content(action_description=str(args.get("action_desc") or "").strip())
        turn: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        if content:
            turn["content"] = content
        images.append(abs_path)
        assistant_turns.append(turn)
        if len(tool_calls) == 1 and tool_call_name(tool_calls[0]) == "terminate":
            if record_pos != len(records) - 1:
                raise SkipTrajectoryError(
                    "post_terminal_steps", f"terminal logical step {step_idx} is not final"
                )
            break

    if not images:
        raise SkipTrajectoryError("no_usable_steps", uid)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "image", "index": 0}, {"type": "text", "text": instruction}]},
    ]
    for i, turn in enumerate(assistant_turns):
        messages.append(turn)
        if i != len(assistant_turns) - 1:
            messages.append({"role": "user", "content": [{"type": "image", "index": i + 1}]})
    # No ``terminate`` call is ever persisted: when present, it is dropped and
    # any source-asserted failure status/reason moves to ``others``. Without a
    # terminate, the final executable action remains the EOF label.
    terminate_call = pop_terminal_terminate(messages)
    answer_text = (
        terminal_answer_text(tool_call_arguments(terminate_call).get("reason"))
        if terminate_call is not None
        and tool_call_arguments(terminate_call).get("status") == "success"
        else None
    )
    ends_on_action = bool(
        messages
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("tool_calls")
    )
    if ends_on_action:
        if answer_text:
            content = list(messages[-1].get("content") or [])
            content.append({"type": "text", "text": answer_text})
            messages[-1]["content"] = content
    else:
        messages.append(
            structural_final_message(answer_text) if answer_text
            else structural_final_message()
        )
    messages = finalize_use_messages(messages, result_boundary_tools={"open_app"})

    assert len(images) == len(assistant_turns)

    return {
        "images": images,
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=(PLATFORM, "use"),
            extra_tool_schemas=extra_tool_schemas_for_messages(messages),
            valid_actions=None,
            others={
                "id": f"{cfg['id_prefix']}_{uid}",
                "resolution": list(res_first),
                "os": OS,
                "source": SOURCE,
                "source_id": uid,
                **terminate_outcome_others(terminate_call),
            },
        ).to_dict(),
    }


def process_subset(
    name: str, cfg: dict, raw_root: str, *, head: int | None, verbose: bool
) -> tuple[list[dict[str, Any]], int, int, Counter[str]]:
    """Build every publishable trajectory of one subset.

    Returns:
        ``(rows, n_trajectories, n_step_records, skips)`` — the accounting travels
        WITH the rows so a caller cannot report the rows without it. The ledger is
        denominated in **trajectories**, because that is the unit in which this
        adapter reconciles: every skip is trajectory-granular and one trajectory
        yields at most one row, so::

            n_trajectories == len(rows) + sum(skips.values())

        and this function closes it before returning, folding any remainder into
        ``skips["unaccounted"]`` so a ``raise`` or ``continue`` added later
        without its own bucket surfaces by name instead of vanishing.

        ``n_step_records`` is the other denominator — source JSONL lines behind
        those trajectories — and is returned separately rather than mixed into
        the ledger, because a count of steps and a count of trajectories are
        different claims about the same run. ``skips`` is keyed by
        :class:`SkipTrajectoryError`'s ``reason``; ``image_absent_on_host`` is
        named apart from the format reasons because it is the host lacking data,
        not the adapter refusing it.
    """
    jsonl_path = os.path.join(raw_root, cfg["jsonl_rel"])
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Annotation file not found: {jsonl_path}")
    image_dir = os.path.join(raw_root, cfg["image_dir_check_rel"])
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}. For subset={name!r}, the "
            f"screenshots must be present before running.")

    uid_records = group_records_by_uid(jsonl_path, name)
    uids = sorted(uid_records)
    if head is not None:
        uids = uids[:head]
    rows: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    n_step_records = sum(len(uid_records[uid]) for uid in uids)
    for uid in uids:
        try:
            rows.append(build_trajectory(uid, uid_records[uid], cfg))
        except SkipTrajectoryError as e:
            skips[e.reason] += 1
            if verbose:
                print(f"  [skip] {name}/{uid}: {e}", file=sys.stderr)

    # Close the ledger: every trajectory read is either behind a row or named by
    # a reason.
    unaccounted = len(uids) - len(rows) - sum(skips.values())
    if unaccounted:
        skips["unaccounted"] += unaccounted
    return rows, len(uids), n_step_records, skips


def main() -> int:
    parser = argparse.ArgumentParser(
        description="UI-Genie-Agent → cua-lite mobile use",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--head", type=int, default=None, help="Process at most N trajectories per subset")
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT must be set", file=sys.stderr)
        return 1
    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]

    if args.dry_run:
        print("=== DRY RUN === would write mobile/use/<split>/{%s}.parquet" % ",".join(subsets))
        return 0
    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set", file=sys.stderr)
        return 1

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    n_traj_total = n_steps_total = 0
    skips: Counter[str] = Counter()
    for name in subsets:
        rows, n_traj, n_steps, subset_skips = process_subset(
            name, SUBSETS[name], raw_root, head=args.head, verbose=args.verbose)
        n_traj_total += n_traj
        n_steps_total += n_steps
        n_kept = 0
        for entry in rows:
            if has_oob_coordinate(entry):
                subset_skips["oob_coordinate"] += 1
                continue
            try:
                bk, e = stage_entry(entry, store=store, splitter=splitter, variant=name)
            except staging.CorruptImageError as img_err:
                subset_skips["image_corrupt_on_host"] += 1
                if args.verbose:
                    print(f"  [skip-corrupt] {name}: {img_err}", file=sys.stderr)
                continue
            buffers[bk].append(e)
            n_kept += 1
        skips.update(subset_skips)
        print(
            f"  {name}: {n_traj} trajectories read ({n_steps} step records) → "
            f"{n_kept} rows, {n_traj - n_kept} skipped"
        )
        print(f"  {name} skip reasons (trajectories): {dict(subset_skips)}")

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    # Denominated in TRAJECTORIES, and the identity is printed so a reader does
    # not have to trust it: trajectories read == rows + sum(skip reasons).
    print(
        f"Trajectories read: {n_traj_total}  kept: {n_rows}  "
        f"skipped: {sum(skips.values())}  (from {n_steps_total} source step records)"
    )
    print("Skip reasons (trajectories):", dict(skips))
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
