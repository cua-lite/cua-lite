"""MobileGym quality-ANNOTATION pass — the single step before stage/export_sft.

Used by every MobileGym teacher: it runs on canonical Lite rows, after the adapter has
already projected each model family's wire format, so it carries no model-family branch.

Pipeline: collect (scripts/rollout.py) → **annotate** (this) → stage (lite.data.hf.stage)
→ upload; consumers download the published canonical dataset before ``export_sft``. Read
every ``trajectory.parquet`` under ``--log-root``, clean + tag each, and write them to
``--out``. Quality gates are recorded in ``metadata.others.exclude_reason`` (comma-joined;
the key is omitted when clean) and the consumer decides — with three hard-drop
exceptions: a trajectory naming an action that does not exist, one with an
out-of-range GUI coordinate, and one naming a tool the row never declared. All three
fail the staging row-format check and should never enter the canonical dataset, so they
are physically excluded rather than left for a downstream threshold.
Everything else is kept. Downstream selects the training set with
``not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) >= 0.30``
— the same ``exclude_reason`` idiom used for task-level exclusion. See
[devs/data/mobilegym/AGENTS.md](/devs/data/mobilegym/AGENTS.md#shared-filter) for why the
gate is 0.30 and not "successes only".

This is the MOBILE sibling of ``devs/data/lite.osworld/filter.py`` and deliberately does
NOT inherit its desktop rules. MobileGym exposes no terminal and no filesystem, so
``dependency_install``, ``complex_shell``, the ``/opt/env`` leak hard-drop and the
Ctrl+S / Ctrl+Z (undo-storm) machinery have no referent here and are absent rather than
dormant. What mobile adds instead is ``teacher_gave_up``.

Three layers:

  1. STRIP no-op *actions* from inside a trajectory (keeps the trajectory, shortens it):
       --actions screenshot   (default) — ``screenshot`` is a verified dispatch no-op on
         this env: ``_translate_action`` maps it to ``None`` and the container only
         captures the frame it would have captured anyway
         (``lite/gym/envs/mobilegym/docker/server.py``), and the teacher recipe tells the
         model never to call it. It changes no state and carries zero signal, so teaching
         the student to emit it is pure harm. Strip.
         For canonical action-batch calls, strip only the no-op child actions and keep the
         original batch id when any non-no-op child remains; remove the following tool
         result only when the whole call/turn is stripped.

     ``wait`` is NOT in the default set, unlike the desktop filter. On mobile it is a real
     executed action (``ActionType.WAIT``), not a dispatch no-op, and a phone UI that is
     still loading is exactly when waiting is the correct move. Pass
     ``--actions screenshot,wait`` only for an ablation that deliberately trains the
     student never to wait.

  2. TAG quality gates in ``metadata.others.exclude_reason`` (the trajectory is KEPT):
       incomplete        metadata.others.terminated != true — the episode ran out of
         steps without ever calling ``terminate``/``response``.
       teacher_gave_up   the teacher ended on ``terminate(status="failure")``. That is an
         ABORT: the container sets ``terminated=True`` but ``completed=False``, so the
         Success-Rate gate ``terminated and success and clean`` cannot fire and the row
         falls through to the shaped branch. It can therefore carry a perfectly
         respectable ``episode_return`` (up to 0.5) while its terminal act teaches the
         student to give up — which is why reward alone does not catch it.
       footgun:loop      >=3 consecutive identical (name,args) actions — enabled by
         --drop-loops (a genuine stall: re-tapping a control that never responds).
     The raw reward is NOT a tag: ``metadata.others.episode_return`` is already available
     for the consumer to threshold. The ``--drop-loops`` flag name is the desktop filter's
     spelling, kept so the two annotate passes read the same; it gates whether the tag is
     EMITTED, it does not drop the trajectory.

     ``TRAJECTORY_EXCLUDE_REASONS`` below is the closed vocabulary for this namespace —
     separate from the task-level env catalogs, same ``category(:detail)?`` format
     grammar. Unlike the desktop filter it ships no runtime validator: three tags are
     emitted from three sites, each a literal from that dict, so an out-of-vocabulary
     value is not representable in the first place.

  3. NORMALIZE the content-only final turn to one clean ``text`` part. A final assistant
     turn with NO ``tool_calls`` has its content replaced wholesale by
     ``structural_final_message()`` — ``[{"type": "text", "text": "Done."}]`` — no matter
     what it held before. This is unconditional and has no opt-out; see
     ``lite.data.utils.messages.normalize_content_only_final``. On MobileGym every real
     ending goes through a TOOL (``terminate`` for an operate task, ``response`` for a
     query one), so a content-only final is an abandoned ending, and a turn that DOES
     carry tool_calls is untouched.

Usage:  # <teacher>/<commit> per the per-teacher runbook, e.g.
        # devs/data/mobilegym/gpt5_5/AGENTS.md
    uv run python devs/data/mobilegym/filter.py \
        --log-root .data/rollout/mobilegym/<teacher>/<commit>/train \
        --out      .data/rollout/mobilegym/<teacher>/<commit>/train_annotated \
        --drop-loops
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    action_coordinate_arguments_out_of_range,
    is_lite_action_name_or_action_batch_tool_name,
)
from lite.core.tools.calls import tool_call_id, tool_call_name
from lite.core.tools.schemas import tool_schema_name
from lite.data.staging import (
    coerce_messages,
    coerce_meta,
    prepare_output_dir,
    write_partition,
)
from lite.data.utils.messages import (
    normalize_content_only_final,
    strip_raw_response_if_message_changed,
)

# ``devs`` is not an installed package (``pyproject.toml`` ships ``lite*`` only),
# and ``python <script>.py`` puts only the SCRIPT's directory on ``sys.path`` —
# so the repo root must be added before ``devs.data.utils`` can be imported.
# Depth is per-file: this file is ``<repo>/devs/data/mobilegym/filter.py``, so the
# root is ``parents[3]``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from devs.data.utils import (  # noqa: E402  (needs _REPO_ROOT on sys.path)
    _action_name_args,
    _args_of,
    _iter_action_items,
    _with_args,
    carry_content_without_observation_images,
    compact_row_images,
    has_invalid_action_batch,
    rebase_images_for_output,
)

# The one action MobileGym executes as a pure dispatch no-op (the frame is captured
# either way), and which the teacher recipe forbids. ``wait`` is deliberately NOT here
# — see the module docstring; add it with --actions only for an ablation.
DEFAULT_NOOP_ACTIONS = ("screenshot",)

# Any run of whitespace (newlines included) → used to flatten inline_reasoning to one line.
_WS_RUN = re.compile(r"\s+")


def collapse_inline_reasoning(messages: list[dict]) -> tuple[list[dict], int]:
    """Flatten each assistant turn's ``inline_reasoning`` into a single-line paragraph:
    collapse newlines + whitespace runs to single spaces. GPT-5.5 emits the reasoning as
    a multi-line body; the distilled student is trained to produce a one-line Thought, so
    we normalize it at filter time. Idempotent, does NOT mutate the input. Returns
    ``(messages, n_blocks_collapsed)``."""
    out: list[dict] = []
    n = 0
    for m in messages:
        parts = m.get("content") if m.get("role") == "assistant" else None
        if not parts:
            out.append(m)
            continue
        new_parts: list[Any] = []
        changed = False
        for p in parts:
            if (isinstance(p, dict) and p.get("type") == "inline_reasoning"
                    and isinstance(p.get("text"), str)):
                flat = _WS_RUN.sub(" ", p["text"]).strip()
                if flat != p["text"]:
                    p = {**p, "text": flat}
                    changed = True
                    n += 1
            new_parts.append(p)
        if changed:
            m = strip_raw_response_if_message_changed(m, {**m, "content": new_parts})
        out.append(m)
    return out, n


def has_oob_coordinate(messages: list[dict]) -> bool:
    """True if any tool-call carries a coordinate outside the normalized [0, 1000].

    Model edge over-prediction (a tap predicted just past the screen edge) or screenshot-
    resolution corruption. MobileGym and CUA-Lite share the [0, 1000] frame, so no
    conversion sits between the model and this check. The whole trajectory is
    hard-dropped before staging, which rejects such a row. ``messages`` must already be
    ``to_plain``-ed (ndarray args → lists)."""
    for m in messages:
        if not isinstance(m, dict):
            continue
        for _, args in _iter_action_items(m):
            if action_coordinate_arguments_out_of_range(args):
                return True
    return False


def has_undeclared_tool_call(messages: list[dict], metadata: dict) -> bool:
    """True if a tool call names a tool the row never declared.

    The model occasionally hallucinates a tool NAME while emitting real arguments.
    Staging rejects the row — ``tool_call 'x' is standalone but missing from
    metadata.extra_tool_schemas`` — so, like an OOB coordinate, it is hard-dropped here
    rather than left to fail the publish.

    "Known" is the canonical vocabulary plus what the row declares. Every MobileGym
    collect config opts into ``extra_tools: ["open_app", "response", "terminate"]``, and
    the live instance writes those resolved schemas onto the row
    (``RemoteMobileGymEnv.metadata``), so the three standalone extras this env's teachers
    actually use are declared and kept.
    """
    declared = {
        tool_schema_name(t)
        for t in (metadata.get("extra_tool_schemas") or [])
    }
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name")
            if is_lite_action_name_or_action_batch_tool_name(name):
                continue
            if name not in declared:
                return True
    return False


def strip_noop_actions(
    messages: list[dict], noop: frozenset[str]
) -> tuple[list[dict], int, int]:
    """Return (cleaned_messages, n_actions_stripped, n_turns_dropped).

    ``messages`` is a ``[user, assistant, tool, assistant, tool, ...]`` sequence (already
    ``to_plain``-ed) — the observation for turn N+1 is the ``role:"tool"`` result that
    FOLLOWS assistant turn N, paired by the assistant call's ``id`` and the result's
    ``tool_call_id``. For each assistant turn: drop no-op ``tool_calls`` or no-op children
    of an action-batch call; if nothing remains, drop the turn AND the ``role:"tool"``
    results carrying those ids.

    Rows without ``role:"tool"`` results put the observation in a ``role:"user"`` message
    BEFORE the turn; for those the preceding user is popped instead and a leading goal is
    carried forward. The branch is selected by result layout — does the sequence contain a
    ``role:"tool"`` message — not by whether the assistant call has an id: preceding-
    observation rows can carry stamped ids too, so that test misroutes them.
    """
    msgs = messages
    out: list[dict] = []
    n_stripped = n_dropped = 0
    # Non-observation content carried from a dropped LEADING user (the goal/instruction,
    # including indexed reference image parts): when the trajectory opens with a
    # no-op-only assistant turn, popping its paired user obs would strand the goal (→ two
    # consecutive user msgs, breaking chat-template alternation).
    carried: list[dict] = []
    has_tool_results = any(m.get("role") == "tool" for m in msgs)
    # Assistant call ids whose turn was dropped; their role:"tool" results go too.
    orphaned_call_ids: set[str] = set()
    for m in msgs:
        if m.get("role") != "assistant":
            if m.get("role") == "tool" and m.get("tool_call_id") in orphaned_call_ids:
                orphaned_call_ids.discard(m.get("tool_call_id"))
                continue
            if m.get("role") == "user" and carried:
                m = dict(m)
                m["content"] = carried + list(m.get("content") or [])
                carried = []
            out.append(m)
            continue
        tcs = m.get("tool_calls") or []
        kept: list[dict] = []
        for tc in tcs:
            name = tool_call_name(tc)
            args = _args_of(tc)
            actions = args.get("actions")
            if name in LITE_ACTION_BATCH_TOOL_NAMES and isinstance(actions, list):
                kept_actions = []
                for action in actions:
                    action_name, action_args = _action_name_args(action, name)
                    if action_name in noop:
                        n_stripped += 1
                    else:
                        kept_actions.append({"action": action_name, **action_args})
                if kept_actions:
                    kept.append(_with_args(tc, {**args, "actions": kept_actions}))
                elif has_tool_results and tool_call_id(tc):
                    orphaned_call_ids.add(tool_call_id(tc) or "")
                continue
            if name in noop:
                n_stripped += 1
                if has_tool_results and tool_call_id(tc):
                    orphaned_call_ids.add(tool_call_id(tc) or "")
                continue
            kept.append(tc)
        if not kept and tcs:
            # No-op-only turn → drop it and its paired observation so alternation is
            # preserved (the no-op didn't change state, so the next obs already matches).
            if has_tool_results:
                orphaned_call_ids |= {
                    call_id for tc in tcs if (call_id := tool_call_id(tc))
                }
            elif out and out[-1].get("role") == "user":
                prev = out.pop()
                if not any(o.get("role") == "user" for o in out):
                    carried = carry_content_without_observation_images(
                        prev.get("content") or []
                    ) + carried
            n_dropped += 1
            continue
        updated = dict(m)
        updated["tool_calls"] = kept
        out.append(strip_raw_response_if_message_changed(m, updated))
    return out, n_stripped, n_dropped


def teacher_gave_up(messages: list[dict]) -> bool:
    """True if the teacher ended the episode with ``terminate(status="failure")``.

    That call is MobileGym's ABORT. The container sets ``terminated=True`` but
    ``completed=False``, and the score is computed as
    ``_evaluate(inst, terminated and completed)`` — so the Success-Rate branch
    (``terminated and success and clean``) cannot fire and the episode falls through to
    the shaped branch, where it still collects ``0.5 * progress``. A give-up can
    therefore sit ABOVE this dataset's quality gate on reward alone while its terminal
    act is precisely the behaviour the student must not learn; hence a tag of its own.

    No false positive from the loop detector: ``LoopDetectWrapper`` injects an
    env-private ``terminate`` below the model tool surface (it never lands in an
    assistant turn), and its status is ``"success"`` in any case.
    """
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for name, args in _iter_action_items(m):
            if name == "terminate" and str(args.get("status") or "") == "failure":
                return True
    return False


def _traj_footguns(messages: list[dict], drop_loops: bool) -> set[str]:
    """Footgun labels for a trajectory's assistant actions — patterns that hurt the
    distilled student even when the trajectory SCORED WELL:

      * ``loop`` — >=3 consecutive identical ``(name, args)`` actions (a genuine stall:
        re-tapping a control that never responds). 2-in-a-row is legitimate.

    The collect configs also run the env-side ``loop_detect: 5``, but that fires on 5
    repetitions and ends the episode with an env-private terminate that is never
    recorded as an assistant turn, so this tag is the only per-row record of a stall.

    ``messages`` must already be ``to_plain``-ed."""
    found: set[str] = set()
    prev_key = None
    run = 1
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for name, args in _iter_action_items(m):
            key = (name, json.dumps(args, sort_keys=True, default=str))
            run = run + 1 if key == prev_key else 1
            if drop_loops and run >= 3:
                found.add("loop")
            prev_key = key
    return found


# --- Canonical TRAJECTORY-LEVEL exclude_reason vocabulary ---------------------
# A property of a single ROLLOUT (not the task). Written comma-joined to
# metadata.others.exclude_reason by ``_exclude_reasons`` below; a clean trajectory omits
# the key. This is a SEPARATE namespace from the task-level env catalogs — same
# ``category(":" detail)?`` format grammar, different closed set. The dict below IS the
# spec for this namespace.
TRAJECTORY_EXCLUDE_REASONS: dict[str, str] = {
    "incomplete": "trajectory did not terminate (metadata.others.terminated != True)",
    "teacher_gave_up": 'ended on terminate(status="failure") — MobileGym ABORT, no Success Rate',
    "footgun": "a rollout footgun; detail in {loop}",
}
TRAJECTORY_DETAIL_ALLOWED: dict[str, frozenset[str]] = {
    "footgun": frozenset({"loop"}),
}


def _exclude_reasons(
    msgs: list[dict], metadata: dict[str, Any], check_loops: bool
) -> list[str]:
    """Ordered quality-exclusion tags for a trajectory (ANNOTATE, not drop).

    Written comma-joined to ``metadata.others.exclude_reason``; a clean trajectory omits
    the key entirely. Downstream filters with ``not m.others.get('exclude_reason')`` —
    identical to the task-level idiom.

    The raw reward is deliberately NOT a tag: ``metadata.others.episode_return`` is
    already the field for the consumer to threshold. The tags cover the gates that are
    NOT otherwise a field."""
    reasons: list[str] = []
    if (metadata.get("others") or {}).get("terminated") is not True:
        reasons.append("incomplete")
    if teacher_gave_up(msgs):
        reasons.append("teacher_gave_up")
    footguns = _traj_footguns(msgs, check_loops)
    reasons.extend(f"footgun:{fg}" for fg in sorted(TRAJECTORY_DETAIL_ALLOWED["footgun"])
                   if fg in footguns)
    return reasons


def _metadata(row: Any) -> dict[str, Any]:
    return dict(coerce_meta(row["metadata"]) or {})


def _annotate_metadata(md_raw: Any, reasons: list[str]) -> Any:
    """Return ``md_raw`` with ``others.exclude_reason`` set to the comma-joined
    ``reasons`` (or the key removed when clean), preserving the original encoding: a
    JSON-string metadata (hf.unstage rows) stays a string; a struct/dict stays a dict.
    Per-file metadata is homogeneous, so the parquet column stays uniform."""
    was_str = isinstance(md_raw, str)
    md = dict(coerce_meta(md_raw) or {})
    others = dict(md.get("others") or {})
    if reasons:
        others["exclude_reason"] = ",".join(reasons)
    else:
        others.pop("exclude_reason", None)
    md["others"] = others
    return json.dumps(md) if was_str else md


def _process_one(task: tuple) -> tuple:
    """Pool worker: unpack, run, and return the source path with the counts."""
    src, dst, noop, out_root, drop_loops, collapse_reasoning, dry_run = task
    return src, _process_file(
        src, dst, noop, out_root, drop_loops, collapse_reasoning, dry_run
    )


def _process_file(
    src: Path, dst: Path, noop: frozenset[str], output_root: Path,
    drop_loops: bool, collapse_reasoning: bool, dry_run: bool = False,
) -> tuple[int, int, int, int, Counter[str], bool]:
    """ANNOTATE mode — keep EVERY trajectory that survives the hard drops, tagging quality
    gates in ``metadata.others.exclude_reason`` instead of dropping. Returns
    ``(n_stripped, n_turns_dropped, n_reasoning_collapsed,
    n_content_only_finals_normalized, reason_counts, wrote_file)``.
    ``reason_counts['_trajectories']`` = number of trajectories carrying ANY
    exclude_reason; ``reason_counts['_total']`` = total trajectories kept. ``drop_loops``
    gates whether the loop check CONTRIBUTES A TAG; it drops nothing."""
    df = pd.read_parquet(src)
    tot_strip = tot_drop = tot_collapse = tot_final_norm = 0
    reason_counts: Counter[str] = Counter()
    new_messages: list[Any] = []
    new_metadata: list[Any] = []
    kept_idx: list[int] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        msgs = coerce_messages(row["messages"])
        # HARD DROP (not a tag): these trajectories fail the staging row-format check,
        # so remove them from the output.
        #
        # The invented-action-name check runs FIRST and must stay there. Every other pass
        # below walks the batch through ``_iter_action_items``, whose ``_action_name_args``
        # RAISES on a child name that is not in the tool's action set -- so a single such
        # row aborts the whole log-root before any later drop can remove it, taking every
        # clean sibling with it.
        if has_invalid_action_batch(msgs):
            reason_counts["_dropped_invalid_action"] += 1
            continue
        if has_oob_coordinate(msgs):
            reason_counts["_dropped_oob"] += 1
            continue
        metadata = _metadata(row)
        if has_undeclared_tool_call(msgs, metadata):
            reason_counts["_dropped_undeclared_tool"] += 1
            continue
        reasons = _exclude_reasons(msgs, metadata, drop_loops)
        cleaned, ns, nd = strip_noop_actions(msgs, noop)
        if collapse_reasoning:
            cleaned, nc = collapse_inline_reasoning(cleaned)
            tot_collapse += nc
        # Unconditional, no flag: a no-tool-call final turn becomes one clean ``text``
        # part. A MobileGym answer is submitted through the ``response`` TOOL and a
        # completion through ``terminate``, so an ending turn that finished the task has
        # tool_calls and is untouched.
        cleaned, normalized = normalize_content_only_final(cleaned)
        tot_final_norm += int(normalized)
        kept_idx.append(pos)
        new_messages.append(cleaned)
        new_metadata.append(_annotate_metadata(row["metadata"], reasons))
        reason_counts["_total"] += 1
        if reasons:
            reason_counts["_trajectories"] += 1
            reason_counts.update(reasons)
        tot_strip += ns
        tot_drop += nd
    # All rows in this parquet were hard-dropped → write nothing, so the sample is
    # physically absent from the annotated output.
    if not new_messages:
        return (tot_strip, tot_drop, tot_collapse, tot_final_norm, reason_counts, False)
    out = df.iloc[kept_idx].copy().reset_index(drop=True)
    out["messages"] = new_messages
    out["metadata"] = new_metadata
    if "images" in out.columns:
        # Dropping a turn drops no picture, so a filtered row would otherwise keep images
        # nothing references and leave its indices non-contiguous. Compact BEFORE
        # rebasing: rebase copies files positionally off this column, so the two must see
        # the same list. compact_row_images owns the renumbering (and asserts the result
        # is dense); rebase renumbers nothing.
        compacted_images, compacted_messages = [], []
        for row_images, row_messages in zip(out["images"], out["messages"], strict=True):
            imgs, msgs = compact_row_images(row_images, row_messages)
            compacted_images.append(imgs)
            compacted_messages.append(msgs)
        out["images"] = compacted_images
        out["messages"] = compacted_messages
        out["images"] = rebase_images_for_output(
            out,
            source_parquet=src,
            output_parquet=dst,
            image_path_root=output_root,
            dry_run=dry_run,
        )
    if not dry_run:  # dry-run: compute every tag + counter, but write nothing
        write_partition(out.to_dict("records"), dst)
    return (tot_strip, tot_drop, tot_collapse, tot_final_norm, reason_counts, True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log-root", required=True, help="input rollout log-root")
    ap.add_argument(
        "--out",
        default=None,
        help="output (annotated) log-root (required unless --dry-run)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="compute + report the clean/tagged counts WITHOUT writing --out: "
        "'if I annotated now, how many would be tagged?'. --out is ignored.",
    )
    ap.add_argument(
        "--actions",
        default=None,
        help="comma-separated action names to strip (default: screenshot). `wait` is a "
        "REAL executed action on this env, not a no-op — add it only for an ablation.",
    )
    ap.add_argument(
        "--drop-loops",
        action="store_true",
        help="tag exclude_reason=footgun:loop on >=3 consecutive identical actions (stall)",
    )
    ap.add_argument(
        "--collapse-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="flatten each assistant turn's inline_reasoning into a single-line "
        "paragraph (ON by default; GPT-5.5 emits a multi-line body, the student is "
        "trained to produce a one-line Thought)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing non-empty --out root. Without this, filter requires a "
        "fresh output root so stale trajectory.parquet files cannot survive a rerun.",
    )
    ap.add_argument(
        "--jobs", type=int, default=16,
        help="worker processes (default 16). Each trajectory is independent, so this "
             "scales nearly linearly; use 1 to debug a traceback.",
    )
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")
    noop = frozenset(a.strip() for a in args.actions.split(",")) if args.actions \
        else frozenset(DEFAULT_NOOP_ACTIONS)
    src_root = Path(args.log_root)
    out_root = Path(args.out) if args.out else src_root  # dry-run never writes
    traj_files = sorted(src_root.rglob("trajectory.parquet"))
    if not traj_files:
        raise SystemExit(f"no trajectory.parquet under {src_root}")
    if not args.dry_run:
        try:
            prepare_output_dir(
                out_root,
                overwrite=args.overwrite,
                label="filter output root",
                protected_roots=(src_root,),
            )
        except (FileExistsError, ValueError) as e:
            raise SystemExit(str(e)) from e

    if args.dry_run:
        print("=== DRY RUN — computing clean/tagged stats, writing nothing ===")
    print(f"stripping {sorted(noop)} from {len(traj_files)} trajectories | "
          f"drop_loops={args.drop_loops} collapse_reasoning={args.collapse_reasoning}")
    ts = td = tcollapse = tfinal = nw = 0
    reason_counts: Counter[str] = Counter()
    tasks = [
        (src, out_root / src.relative_to(src_root), noop, out_root,
         args.drop_loops, args.collapse_reasoning, args.dry_run)
        for src in traj_files
    ]
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            results = list(pool.imap_unordered(_process_one, tasks, chunksize=8))
    else:
        results = [_process_one(t) for t in tasks]

    for src, (ns, nd, ncol, nfinal, npolicy, wrote) in results:
        dst = out_root / src.relative_to(src_root)
        ts += ns
        td += nd
        tcollapse += ncol
        tfinal += nfinal
        reason_counts.update(npolicy)
        if wrote:
            nw += 1
            if not args.dry_run:
                sidecar = src.parent / "summary.json"
                if sidecar.exists():
                    shutil.copy2(sidecar, dst.parent / "summary.json")

    total = reason_counts["_total"]
    annotated = reason_counts["_trajectories"]
    clean = total - annotated
    verb = "WOULD be" if args.dry_run else ""
    dest = "(dry run — nothing written)" if args.dry_run else f"→ {out_root}"
    print(
        f"done (ANNOTATE mode — HARD-DROPPED {reason_counts['_dropped_invalid_action']} "
        f"trajectories with an invalid action name, {reason_counts['_dropped_oob']} with "
        f"OOB coordinates and {reason_counts['_dropped_undeclared_tool']} with an "
        f"undeclared tool call; all {total} trajectories kept after hard drops): "
        f"stripped {ts} no-op actions, dropped {td} no-op-only turns, collapsed "
        f"{tcollapse} inline_reasoning blocks, normalized {tfinal} content-only final "
        f"turns to text 'Done.' {dest} — {clean} clean (no exclude_reason) + {annotated} "
        f"{verb} tagged = {100*clean//max(total,1)}% clean "
        f"({nw}/{len(traj_files)} trajectory files written)"
    )
    # Bookkeeping keys are underscore-prefixed and real exclude_reasons never are, so the
    # convention filters them -- a hard-drop counter listed here would read as a tag on
    # rows still in the output, when those rows were removed.
    print("exclude_reason tag counts: " + ", ".join(
        f"{reason}={count}" for reason, count in sorted(reason_counts.items())
        if not reason.startswith("_")
    ))


if __name__ == "__main__":
    main()
