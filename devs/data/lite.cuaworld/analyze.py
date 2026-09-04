"""Trajectory analyzer for the cuaworld GPT collection (pre-exploration).

Forked from devs/data/lite.osworld/analyze.py. Reports the signals we use to design the
collection plan, with two CUAWorld-specific additions:

  * WAIT-PATTERN breakdown — how often the teacher emits a standalone ``wait`` turn vs a
    trailing/among-others ``wait``. Informational only: cuaworld reuses the shared
    devs/data/lite.osworld/filter.py, which strips ``wait`` unconditionally like the rest of
    the family (keeps a bare Ctrl+S); this does not imply ``wait`` is invalid at runtime.
  * PROGRAMMATIC vs VLM-judged yield — detector-dependent VLM-judged tasks (vlm_tasks.json)
    score via the vlm.py shim; a big yield gap or implausibly-high VLM yield flags noisy judge
    labels.

Groups by software (env_id ``lite.cuaworld.<software>``), not osworld's per-domain. Point it at one
software's log-root, or a parent dir holding several ``<software>/`` roots.

Memory-safe: reads one sample's summary.json + trajectory.parquet at a time.

    uv run python devs/data/lite.cuaworld/analyze.py \
        --log-root .data/rollout/lite.cuaworld/gpt5_5/<commit>
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from lite.core.tools.action_space import LITE_ACTION_BATCH_TOOL_NAMES
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.data.staging import coerce_messages, coerce_meta

_HERE = Path(__file__).parent


def _vlm_task_ids() -> set[str]:
    """Bare task ids (no software prefix) that are VLM-judged (vlm_tasks.json)."""
    doc = json.loads((_HERE / "vlm_tasks.json").read_text())
    ids: set[str] = set()
    for tasks in doc["vlm_tasks"].values():
        ids.update(tasks)
    return ids


_ENV_PREFIX = "lite.cuaworld."


def _software(env_id: str, task_id: str, log_root: Path, sample_dir: Path) -> str:
    """lite.cuaworld.<software> → <software>; fall back to the sub-dir under log_root."""
    if env_id.startswith(_ENV_PREFIX):
        return env_id[len(_ENV_PREFIX):]
    try:
        rel = sample_dir.relative_to(log_root)
        return rel.parts[0] if rel.parts else "?"
    except ValueError:
        return "?"


def _wait_pattern(msgs: list[dict]) -> tuple[int, int]:
    """(standalone_wait_turns, trailing_wait_actions) over a trajectory's assistant turns.

    standalone = a turn whose only action(s) are ``wait`` (the shared filter strips
      the whole no-op-only turn).
    trailing   = a ``wait`` sharing a turn with ≥1 real (non-wait/screenshot)
      action (the shared filter strips just the child action)."""
    standalone = trailing = 0
    for m in msgs:
        if not (isinstance(m, dict) and m.get("role") == "assistant"):
            continue
        tcs = m.get("tool_calls") or []
        names: list[str] = []
        for tc in tcs:
            name = tool_call_name(tc)
            args = tool_call_arguments(tc)
            if name in LITE_ACTION_BATCH_TOOL_NAMES:
                names.extend(action["action"] for action in args["actions"])
            else:
                names.append(name)
        n_wait = names.count("wait")
        if not n_wait:
            continue
        real = [n for n in names if n not in ("wait", "screenshot")]
        if real:
            trailing += n_wait
        else:
            standalone += 1
    return standalone, trailing


def analyze(log_root: Path) -> None:
    samples = sorted(log_root.rglob("sample_*/summary.json"))
    if not samples:
        raise SystemExit(f"no sample_*/summary.json under {log_root}")

    vlm_ids = _vlm_task_ids()
    n = succ = 0
    by_sw: dict[str, list[int]] = defaultdict(list)          # software -> [0/1 success]
    by_verifier: dict[str, list[int]] = defaultdict(list)    # 'programmatic'|'vlm' -> [0/1]
    turns_all: list[int] = []
    fail_kind: Counter = Counter()
    looped = 0
    desc_steps = desc_with = 0
    wait_standalone = wait_trailing = 0
    trajs_with_standalone_wait = 0
    fail_examples: list[tuple] = []

    for sj in samples:
        s = json.loads(sj.read_text())
        task_id = sj.parent.parent.name
        ret = float(s.get("episode_return", 0.0) or 0.0)
        n_turns = int(s.get("n_turns", 0) or 0)
        terminated = bool(s.get("terminated", False))
        truncated = bool(s.get("truncated", False))
        ok = ret >= 1.0
        n += 1
        succ += ok
        turns_all.append(n_turns)

        traj = sj.parent / "trajectory.parquet"
        env_id, msgs = "", []
        if traj.exists():
            df = pd.read_parquet(traj)
            md = coerce_meta(df.iloc[0]["metadata"])
            others = md.get("others") or {}
            env_id = str(others.get("env_id") or "")
            msgs = coerce_messages(df.iloc[0]["messages"])
            del df
            gc.collect()
        sw = _software(env_id, task_id, log_root, sj.parent.parent)
        by_sw[sw].append(int(ok))
        by_verifier["vlm" if task_id in vlm_ids else "programmatic"].append(int(ok))

        if not ok:
            kind = (
                "truncated"
                if truncated
                else ("term_fail" if terminated else ("error" if s.get("error") else "other"))
            )
            fail_kind[kind] += 1
            if len(fail_examples) < 8:
                verifier = "vlm" if task_id in vlm_ids else "prog"
                fail_examples.append((sw, task_id, kind, verifier, n_turns, round(ret, 2)))

        # wait-pattern (the conditional-filter evidence)
        st, tr = _wait_pattern(msgs)
        wait_standalone += st
        wait_trailing += tr
        if st:
            trajs_with_standalone_wait += 1

        # action-loop + action_description coverage
        acts: list[str] = []
        for m in msgs:
            if not (isinstance(m, dict) and m.get("role") == "assistant"):
                continue
            parts = m.get("content") or []
            if not isinstance(parts, list):
                continue
            ad = next(
                (
                    p.get("text")
                    for p in parts
                    if isinstance(p, dict) and p.get("type") == "action_description"
                ),
                None,
            )
            desc_steps += 1
            if ad and ad.strip():
                desc_with += 1
                acts.append(ad.strip().lower())
        run = 1
        for i in range(1, len(acts)):
            run = run + 1 if acts[i] == acts[i - 1] else 1
            if run >= 3:
                looped += 1
                break

    # ---- report ----
    print(f"\n=== cuaworld GPT collection — {log_root} ===")
    print(f"samples: {n}   success(ret>=1.0): {succ} ({100*succ//max(n,1)}%)   fail: {n-succ}")
    print(f"\nfailure breakdown: {dict(fail_kind)}")
    print(f"action-loop (>=3 identical consecutive): {looped}/{n}")
    cov = 100 * desc_with // max(desc_steps, 1)
    print(f"action_description coverage: {desc_with}/{desc_steps} assistant steps ({cov}%)")

    print("\n-- WAIT patterns (conditional-filter evidence) --")
    print(
        f"  standalone-wait turns : {wait_standalone}  "
        f"(in {trajs_with_standalone_wait}/{n} trajectories) → STRIPPED as no-op-only turns"
    )
    print(f"  trailing-wait actions : {wait_trailing}  (bundled with a real action) → STRIPPED")

    print("\n-- verifier type (programmatic vs VLM-judged) --")
    for v, arr in sorted(by_verifier.items()):
        print(f"  {v:12s} {sum(arr):3d}/{len(arr):<3d} ({100*sum(arr)//max(len(arr),1)}%)")

    print("\nturns: min/median/max =",
          (min(turns_all), sorted(turns_all)[len(turns_all)//2], max(turns_all)),
          " | hit max(40):", sum(1 for t in turns_all if t >= 40))

    print("\nper-software success (sorted by count):")
    for sw, v in sorted(by_sw.items(), key=lambda kv: -len(kv[1])):
        print(f"  {sw:20s} {sum(v):3d}/{len(v):<3d} ({100*sum(v)//max(len(v),1)}%)")

    if fail_examples:
        print("\nfailure examples (software, task, kind, verifier, turns, return):")
        for ex in fail_examples:
            print("  ", ex)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log-root", required=True, type=Path)
    args = ap.parse_args()
    analyze(args.log_root)


if __name__ == "__main__":
    main()
