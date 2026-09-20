#!/usr/bin/env python3
"""Build and calibrate the desktop GRPO train task manifest.

Runbook: /devs/exps/train/desktop/TASKS.md

Four subcommands:
    build-candidates   corpus -> candidate parquet (provenance screened)
    verify-candidates  re-check a candidate parquet against eval
    calibrate          candidate + grouped rollout logs -> final manifest
    write-smoke-log    synthetic calibration log, for testing the aggregator

Why this is not `export_tasks --filter`: the policy needs per-task rollout
evidence (does a group of samples come back mixed?), which registry metadata
does not carry.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

def _repo_root() -> Path:
    """Walk up to the checkout root; counting parents breaks if the file moves."""
    for d in Path(__file__).resolve().parents:
        if (d / "pyproject.toml").exists() and (d / "lite").is_dir():
            return d
    raise SystemExit("cannot locate the cua-lite checkout root from " + __file__)


REPO = _repo_root()
OSW = REPO / "lite/gym/envs/lite/osworld/data"
SCA = REPO / "lite/gym/envs/lite/scalecua/data"

FILE_FAMILY = "compare-artifact"
STATE_FAMILY = "check-state"
OTHER_FAMILY = "other"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _family(func: str) -> str:
    if func.startswith("compare_") or "structure_sim" in func or func.startswith("check_image"):
        return FILE_FAMILY
    if func.startswith(("check_", "is_")) or func == "exact_match":
        return STATE_FAMILY
    return OTHER_FAMILY


def family_of(rec: dict[str, Any]) -> str:
    """A task's family: artifact-comparison wins if any verifier is one."""
    evaluator = (rec.get("metadata") or {}).get("evaluator") or {}
    func = evaluator.get("func")
    funcs = [str(x) for x in func] if isinstance(func, list) else ([str(func)] if func else [])
    if not funcs:
        return OTHER_FAMILY
    fams = [_family(f) for f in funcs]
    return FILE_FAMILY if FILE_FAMILY in fams else fams[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()]


def eval_rows(scored_only: bool = True) -> list[dict[str, Any]]:
    """osworld eval tasks. ``scored_only`` applies the same predicate the eval
    protocol uses (``--filter "not m.others.get('exclude_reason')"``), i.e. the
    328 rows actually graded, not all 369."""
    rows = _read_jsonl(OSW / "eval.jsonl")
    if not scored_only:
        return rows
    return [r for r in rows
            if not ((r.get("metadata") or {}).get("others") or {}).get("exclude_reason")]


def eval_stems() -> set[str]:
    """8-hex stems of eval tasks. Uses ALL 369 rows on purpose: screening should
    be conservative, and a training task derived from an excluded eval task is
    still eval-derived even though that task is never graded."""
    return {r["task_id"].rsplit("_", 1)[-1] for r in eval_rows(scored_only=False)}


def eval_instruction_index() -> dict[str, str]:
    """instruction text -> eval task_id, for corpora that declare no provenance.

    `synth` ids encode no origin, so the id-based screen is a structural no-op
    for it — it can only ever return zero. Measured: 5 synth rows carry an eval
    task's exact instruction (3 of them with an identical evaluator too). Content
    is the only handle on that corpus."""
    return {r["instruction"].strip(): r["task_id"] for r in eval_rows(scored_only=False)}


def pool_of(task_id: str) -> str:
    """Which corpus a task came from. Needed because calibration now runs over a
    combined pool and the bucket rates per corpus are the point of the exercise."""
    if task_id.startswith("scalecua_"):
        return "scalecua_rl"
    if task_id.startswith("perturb_"):
        return "perturb"
    return "synth"


def domain_of(rec: dict[str, Any]) -> str:
    return ((rec.get("metadata") or {}).get("others") or {}).get("domain", "unknown")


def base_task_of(task_id: str, rec: dict[str, Any]) -> str:
    """The underlying osworld task, used for provenance screening and capping.

    Every corpus hides the origin differently, and getting any one of them wrong
    produces a silent FALSE NEGATIVE on the leak check (RL.md rule 22) -- the raw
    id sets never intersect, so a missed decode looks like "clean".
    """
    oid = (rec.get("metadata") or {}).get("osworld_id")
    if oid:                                   # scalecua declares it outright
        return oid[:8]
    if task_id.startswith("perturb_"):        # perturb_<base task>_<hash8>
        return task_id[len("perturb_"):].rsplit("_", 1)[0].rsplit("_", 1)[-1]
    # synth ids are <template>_NNNN; the template is the unit worth capping
    return re.sub(r"_\d{4}$", "", task_id)


# --------------------------------------------------------------------------- #
# build-candidates
# --------------------------------------------------------------------------- #
POOLS = {
    "synth": (OSW / "train.synth.jsonl", "lite.osworld", "train.synth"),
    "perturb": (OSW / "train.perturb.jsonl", "lite.osworld", "train.perturb"),
    "scalecua_rl": (SCA / "rl.jsonl", "lite.scalecua", "rl"),
    "scalecua_train": (SCA / "train.jsonl", "lite.scalecua", "train"),
}

# Two different kinds of eval contamination, both gated behind --allow-env-reuse
# but not equally severe:
#   perturb  -- TASK-level. Its 256 base tasks are 100% eval tasks and its tasks
#               are rewrites of the eval instructions themselves (76% of the
#               filtered eval set). SFT on its successes measured -6.77pp at 8
#               updates and -14.49pp at 26 (RL.md 1.1-AJ).
#   scalecua -- ENVIRONMENT-level. Same starting screens, but the goals and
#               verifiers are new (96% different func, 100% different expected).
LEAK_KIND = {"perturb": "task-level (rewrites of eval instructions)",
             "scalecua_rl": "environment-level (same setup, new goals)",
             "scalecua_train": "environment-level (same setup, new goals)"}


def build_candidates(args: argparse.Namespace) -> None:
    stems = eval_stems()
    ev_text = eval_instruction_index()
    rows, notes = [], []
    for name in args.pools:
        path, env_id, split = POOLS[name]
        if not path.exists():
            raise SystemExit(f"missing corpus for pool {name}: {path}")
        src = _read_jsonl(path)
        # Same predicate the eval protocol uses. Without it the env rejects the
        # row at setup (ScaleCuaTaskError kind='excluded_task') and the whole
        # group is burned -- 240 scalecua rows carry a reason (proxy_required,
        # upstream_generated_eval_bug, live_site_drift, flakes).
        excluded = [r for r in src
                    if ((r.get("metadata") or {}).get("others") or {}).get("exclude_reason")]
        src = [r for r in src
               if not ((r.get("metadata") or {}).get("others") or {}).get("exclude_reason")]
        kept, leaked, by_text = [], 0, 0
        for r in src:
            stem = base_task_of(r["task_id"], r)
            # Two independent screens. The id screen is blind on `synth` (its ids
            # encode no origin, so it can only ever return zero); the instruction
            # screen catches what that structurally cannot see.
            hit_id = stem in stems
            hit_text = r["instruction"].strip() in ev_text
            if hit_text and not hit_id:
                by_text += 1
            if hit_id or hit_text:
                leaked += 1
                if not args.allow_env_reuse:
                    continue
            kept.append((name, env_id, split, r))
        if by_text:
            print(f"  {name:<16} !! {by_text} row(s) carry an EVAL INSTRUCTION verbatim "
                  "(id screen is blind to these)")
        notes.append({"pool": name, "rows": len(src) + len(excluded),
                      "excluded_unrunnable": len(excluded),
                      "leaked_rows": leaked, "leaked_by_instruction_only": by_text,
                      "kept": len(kept)})
        rows.extend(kept)
        kind = LEAK_KIND.get(name, "")
        print(f"  {name:<16} {len(src)+len(excluded):>6} rows, {len(excluded):>4} unrunnable,"
              f" {leaked:>6} eval-derived, kept {len(kept):>6}"
              + (f"   <= {kind}" if leaked and kind else ""))

    if not rows:
        raise SystemExit("  no candidates -- every pool was fully eval-derived; "
                         "pass --allow-env-reuse only if you accept that label")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    out = pd.DataFrame({
        "problem": [f"Complete the task: {r['task_id']}" for _, _, _, r in rows],
        "metadata": [{"env_key": f"{env}@{r['task_id']}", "split": split}
                     for _, env, split, r in rows],
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    fam = collections.Counter(family_of(r) for _, _, _, r in rows)
    dom = collections.Counter(domain_of(r) for _, _, _, r in rows)
    leak_pools = [n["pool"] for n in notes if n["leaked_rows"]]
    summary = {"rows": len(rows), "seed": args.seed, "pools": args.pools,
               "allow_env_reuse": args.allow_env_reuse, "per_pool": notes,
               "EVAL_CONTAMINATION": {p: LEAK_KIND.get(p, "unknown") for p in leak_pools}
               or "none",
               "family_counts": dict(fam), "domain_counts": dict(dom)}
    Path(_sidecar(args.out, ".summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  wrote {len(rows)} candidates -> {args.out}")
    _print_family(fam)


def _print_family(fam: collections.Counter) -> None:
    ev = collections.Counter(family_of(r) for r in eval_rows())
    n, m = sum(fam.values()), sum(ev.values())
    print(f"  {'family':<20}{'pool':>16}{'eval':>14}")
    for k in sorted(set(fam) | set(ev)):
        print(f"  {k:<20}{fam.get(k,0):>8} ({100*fam.get(k,0)/max(n,1):>4.1f}%)"
              f"{ev.get(k,0):>6} ({100*ev.get(k,0)/max(m,1):>4.1f}%)")


# --------------------------------------------------------------------------- #
# verify-candidates
# --------------------------------------------------------------------------- #
def verify_candidates(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.candidate)
    ids = [m["env_key"].split("@", 1)[1] for m in df["metadata"]]
    stems = eval_stems()
    # A missing corpus must be fatal, not silent: base_task_of falls back to the
    # raw id when it has no record, which turns the leak gate off without a word.
    byid, missing = {}, []
    for name, (path, _, _) in POOLS.items():
        if path.exists():
            byid.update({r["task_id"]: r for r in _read_jsonl(path)})
        else:
            missing.append(f"{name} ({path})")
    if missing:
        raise SystemExit("  FAIL: corpus file(s) absent, leak screening would be "
                         "a false negative: " + "; ".join(missing))

    ev_text = eval_instruction_index()
    leaked = [t for t in ids
              if base_task_of(t, byid.get(t, {})) in stems
              or (byid.get(t, {}).get("instruction") or "").strip() in ev_text]
    perturb = [t for t in ids if t.startswith("perturb_")]
    dup = len(ids) - len(set(ids))

    print(f"  rows                 {len(ids)}")
    print(f"  unique task ids      {len(set(ids))}  (duplicates: {dup})")
    print(f"  eval-derived rows    {len(leaked)}")
    print(f"  train.perturb rows   {len(perturb)}")
    _print_family(collections.Counter(family_of(byid[t]) for t in ids if t in byid))

    bad = []
    if dup:
        bad.append("duplicate task ids")
    if perturb and not args.allow_env_reuse:
        bad.append(f"{len(perturb)} train.perturb rows (task-level leak) "
                   "without --allow-env-reuse")
    if leaked and not args.allow_env_reuse:
        bad.append(f"{len(leaked)} eval-derived rows without --allow-env-reuse")
    if bad:
        raise SystemExit("  FAIL: " + "; ".join(bad))
    print("  OK" + ("  (eval-derived rows present by explicit opt-in)" if leaked else ""))


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
def _read_group(log_root: str) -> dict[str, list[dict[str, Any]]]:
    per: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for f in glob.glob(os.path.join(log_root, "*", "*", "sample_*", "summary.json")):
        task = os.path.basename(os.path.dirname(os.path.dirname(f)))
        try:
            per[task].append(json.load(open(f)))
        except Exception:
            per[task].append({"error": "unreadable summary"})
    return per


def bucket(samples: list[dict[str, Any]], group_size: int) -> str:
    """Classify a task by what its grouped rollout would give GRPO.

    Variance is checked BEFORE the infra threshold: a group like
    [error, error, 1.0, 0.0] is the single most valuable shape there is, and an
    earlier version discarded it as infra_bad. The threshold only decides what to
    do with groups that carry no signal anyway.

    ⚠ ``incomplete`` is not the same as healthy-but-unfinished. Under
    ``--max-attempts 1`` a RETRYABLE error writes error.txt and no summary.json
    (rollout.py writes a summary carrying ``error`` only for non-retryable ones,
    and lite/gym/errors.py defaults unknown exceptions to retryable), so ordinary
    infra noise lands here rather than in ``infra_bad``. Watch the incomplete rate
    as the real infra-health number.
    """
    rs = [float(s.get("episode_return") or 0) for s in samples if not s.get("error")]
    wins = sum(1 for r in rs if r >= 1.0)
    if len(rs) > 1 and 0 < wins < len(rs):
        return "mixed_success"                      # signal regardless of errors
    if len(samples) < group_size:
        return "incomplete"
    if sum(1 for s in samples if s.get("error")) * 2 >= len(samples):
        return "infra_bad"
    if not rs:
        return "infra_bad"
    if wins == len(rs):
        return "all_success"
    return "all_fail_with_reward_variance" if len(set(rs)) > 1 else "all_fail_flat"


# Preference order: only the first two carry real advantage signal. all_success
# is kept in small numbers as a behaviour-retention anchor -- PTB showed that a
# narrow success-only corpus degrades the policy monotonically (-6.77pp at 8
# updates, -14.49pp at 26).
ORDER = ["mixed_success", "all_fail_with_reward_variance", "all_success"]


def base_eval_outcome(log_root: str) -> tuple[set[str], set[str]]:
    """(solved, failed) eval stems from a base-checkpoint eval run.

    DIAGNOSTIC ONLY -- not a ranking signal. An earlier version ranked on it
    ("headroom lives in the tasks the base gets wrong"); see the note in
    calibrate() for the five reasons that was removed. Chiefly: the label is one
    Bernoulli draw per eval task, and where it would actually be consulted
    (mixed groups) it demotes the marginal-success tasks whose retention decided
    every prior run.
    """
    solved: set[str] = set()
    failed: set[str] = set()
    for f in glob.glob(os.path.join(log_root, "*", "*", "sample_*", "summary.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("error"):
            continue
        stem = os.path.basename(os.path.dirname(os.path.dirname(f))).rsplit("_", 1)[-1]
        (solved if float(d.get("episode_return") or 0) >= 1.0 else failed).add(stem)
    return solved, failed - solved


def group_spread(samples: list[dict[str, Any]]) -> float:
    """Population std of the group's returns.

    ⚠ This is NOT "the advantage scale". `grpo.py` computes
    ``adv = (r - mean) / (adv.std() + 1e-6)`` with ``grpo_std_normalization``
    defaulting True and nothing in this repo disabling it, so the std is exactly
    the quantity DIVIDED OUT: ``||adv||_2 == sqrt(n-1)`` for every p, and
    ``max|adv|`` is in fact LARGER for a 1/4 group (1.50) than a 2/4 one (0.87).
    An earlier version of this file claimed a "~1.5x advantage scale" for p=0.5 --
    that was wrong, and refuted by reading grpo.py.

    What survives is weaker and different: a balanced group spreads the same total
    advantage mass over more trajectories (L1 mass 3.00 -> 3.46 at g=4,
    4.95 -> 7.48 at g=8), so no single rollout dominates the estimate. The effect
    is ~15% at g=4 and only becomes substantial at g>=8. Direction only.

    Population std (not |success_rate - 0.5|) because the latter is identically
    0.5 for every all_fail_with_reward_variance group and discards partial credit:
    [0.998, 0.998, 0.998, 0.0] is real usable variance and ranked worst under it.
    """
    rs = [float(s.get("episode_return") or 0) for s in samples if not s.get("error")]
    return statistics.pstdev(rs) if len(rs) > 1 else 0.0


def _sidecar(out: str, suffix: str) -> str:
    """Sidecar path. str.replace('.parquet', ...) silently collapses all three
    outputs onto one path when --out has no .parquet suffix, and rewrites any
    directory component containing '.parquet'."""
    return str(Path(out).with_suffix("")) + suffix


def calibrate(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.candidate)
    ids = [m["env_key"].split("@", 1)[1] for m in df["metadata"]]
    byid, missing = {}, []
    for name, (path, _, _) in POOLS.items():
        if path.exists():
            byid.update({r["task_id"]: r for r in _read_jsonl(path)})
        else:
            missing.append(f"{name} ({path})")
    if missing:
        raise SystemExit("  FAIL: corpus file(s) absent, provenance ranking would "
                         "silently degrade: " + "; ".join(missing))

    per = _read_group(args.log_root)
    buckets = {t: bucket(per[t], args.group_size) for t in ids if t in per}
    counts = collections.Counter(buckets.values())
    print(f"  calibrated tasks {len(buckets)} / {len(ids)} candidates")
    for k, v in counts.most_common():
        print(f"    {k:<32}{v:>6}")

    # Per-corpus bucket rates: the whole reason to calibrate a combined pool.
    by_pool: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for t, bk in buckets.items():
        by_pool[pool_of(t)][bk] += 1
    if len(by_pool) > 1:
        print()
        print("  === bucket rate by corpus ===")
        all_b = sorted({b for c in by_pool.values() for b in c})
        print(f"  {'corpus':<12}{'tasks':>7}" + "".join(f"{b[:15]:>17}" for b in all_b))
        for pool, c in sorted(by_pool.items()):
            n = sum(c.values())
            cells = "".join(f"{c[b]:>9} ({100*c[b]/n:>4.1f}%)" for b in all_b)
            print(f"  {pool:<12}{n:>7}{cells}")

    # Ranking, in order of precedence:
    #   1. bucket          -- mixed groups are the only ones with advantage signal
    #   2. group spread    -- see group_spread(); direction only, small effect at g=4
    #   3. a seeded shuffle
    #
    # ⚠ Ties must NOT fall back to task_id. Ids sort perturb_ < scalecua_ < synth_
    # and then by application, so an id tie-break clusters selection by corpus and
    # by app -- structurally the same failure as --head N, which this runbook
    # warns about elsewhere.
    #
    # There is deliberately NO "prefer tasks the base fails" tier. It was in an
    # earlier version and an audit took it apart:
    #   * its evidence is four runs of ONE method (rejection-sampling SFT), whose
    #     flat 8.7-12.5% recovery measures that sampler's coverage ceiling
    #     (it cannot produce an example for a task it never solves), not a limit
    #     on learning;
    #   * where it is right (p ~ 1) bucket ranking already demotes those tasks,
    #     so it adds nothing;
    #   * where it actually gets consulted (mixed groups, 0 < p < 1) it ranks LAST
    #     exactly the marginal-success tasks whose retention decided every prior
    #     run -- it deprioritizes the one lever the evidence identifies;
    #   * 70% of the rows it promoted are scalecua, which shares only the starting
    #     screen with its origin eval task (87% different verifier func), so the
    #     "training on X raises eval task E" mechanism is absent there;
    #   * the label comes from ONE Bernoulli draw per eval task. At this project's
    #     measured 13.3% repeat-disagreement, ~44 of 328 tasks flip on a re-run and
    #     >=26% of labels carry no information -- and the noise is maximal on the
    #     mid-p tasks that bucket ranking hands it.
    # The base-eval split is still REPORTED (--base-eval-root) as a diagnostic.
    solved, failed = (base_eval_outcome(args.base_eval_root)
                      if args.base_eval_root else (set(), set()))
    if failed:
        print(f"  base eval reference: solves {len(solved)}, fails {len(failed)}"
              "   (reported only -- not used for ranking)")

    def upside(t: str) -> int:
        b = base_task_of(t, byid.get(t, {}))
        if b in failed:
            return 0
        if b in solved:
            return 2
        return 1

    jitter = {t: random.Random(f"{args.seed}:{t}").random() for t in ids}
    eligible = [t for t in ids if buckets.get(t) in ORDER]
    spread = {t: group_spread(per[t]) for t in ids if t in per}
    eligible.sort(key=lambda t: (ORDER.index(buckets[t]), -spread[t], jitter[t]))

    ev_fam = collections.Counter(family_of(r) for r in eval_rows())
    ev_n = sum(ev_fam.values())
    quota = ({k: round(args.target * v / ev_n) for k, v in ev_fam.items()}
             if args.match_eval_families else None)
    if quota:
        pool_fam = collections.Counter(family_of(byid.get(t, {})) for t in eligible)
        short = {k: (pool_fam.get(k, 0), q) for k, q in quota.items() if pool_fam.get(k, 0) < q}
        if short:
            print("  ⚠ family quota exceeds the pool; selection will fall short of "
                  f"--target {args.target}:")
            for k, (have, want) in short.items():
                print(f"      {k:<20} have {have:>5}  quota {want:>5}")

    # all_success as a genuine anchor needs a RESERVED slot, not a ceiling: it is
    # last in ORDER, so on real bucket rates (mixed ~39% of 4220) the first bucket
    # alone exceeds --target and the anchor is never reached. Measured: 1500 mixed,
    # 0 all_fail_var, 0 all_success -- the cap never fired.
    reserve = min(args.all_success_cap, sum(1 for t in eligible
                                            if buckets[t] == "all_success"))
    budget = max(args.target - reserve, 0)
    picked: list[str] = []
    per_base: collections.Counter[str] = collections.Counter()
    per_dom: collections.Counter[str] = collections.Counter()
    per_fam: collections.Counter[str] = collections.Counter()
    n_all_success = 0
    for t in eligible:
        # Leave room for the reserved anchors; they are appended below.
        if len(picked) >= budget and buckets[t] != "all_success":
            continue
        if len(picked) >= args.target:
            break
        rec = byid.get(t, {})
        base, dom, fam = base_task_of(t, rec), domain_of(rec), family_of(rec)
        if per_base[base] >= args.max_per_base:
            continue
        if per_dom[dom] >= args.max_per_domain:
            continue
        if buckets[t] == "all_success" and n_all_success >= args.all_success_cap:
            continue
        if quota is not None and per_fam[fam] >= quota.get(fam, 0):
            continue
        picked.append(t)
        per_base[base] += 1
        per_dom[dom] += 1
        per_fam[fam] += 1
        n_all_success += buckets[t] == "all_success"

    if len(picked) < args.min_final or not picked:
        raise SystemExit(
            f"  FAIL: only {len(picked)} selected, need >= {args.min_final}. "
            + ("A family quota can be unfillable -- the pool holds "
               f"{sum(1 for t in ids if family_of(byid.get(t, {})) == OTHER_FAMILY)} "
               f"'{OTHER_FAMILY}' rows against a quota of {quota.get(OTHER_FAMILY, 0)}. "
               if quota else "")
            + "Inspect calibration health before training.")

    keep = set(picked)
    out = df[[m["env_key"].split("@", 1)[1] in keep for m in df["metadata"]]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    side = pd.DataFrame({"task_id": list(buckets), "bucket": list(buckets.values()),
                         "selected": [t in keep for t in buckets]})
    side.to_parquet(_sidecar(args.out, ".calibration.parquet"), index=False)
    summary = {"selected": len(picked), "target": args.target,
               "bucket_counts": dict(counts),
               "selected_bucket_counts": dict(collections.Counter(buckets[t] for t in picked)),
               "family_counts": dict(per_fam), "domain_counts": dict(per_dom),
               "bucket_by_corpus": {k: dict(v) for k, v in by_pool.items()},
               "selected_by_corpus": dict(collections.Counter(pool_of(t) for t in picked)),
               "match_eval_families": args.match_eval_families,
               "max_per_base": args.max_per_base, "max_per_domain": args.max_per_domain,
               "all_success_cap": args.all_success_cap}
    Path(_sidecar(args.out, ".summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  selected {len(picked)} -> {args.out}")
    for k, v in collections.Counter(buckets[t] for t in picked).most_common():
        print(f"    {k:<32}{v:>6}")
    if failed:
        up = collections.Counter(upside(t) for t in picked)
        print(f"  upside: base-fails {up[0]}   no-provenance {up[1]}   base-solves {up[2]}")
        print(f"  median group std {statistics.median([group_spread(per[t]) for t in picked]):.3f}"
              f"  (higher = more usable gradient)")
    sel_pool = collections.Counter(pool_of(t) for t in picked)
    if len(sel_pool) > 1:
        print("  by corpus: " + "  ".join(f"{k} {v}" for k, v in sorted(sel_pool.items())))
    _print_family(per_fam)


# --------------------------------------------------------------------------- #
# write-smoke-log
# --------------------------------------------------------------------------- #
def write_smoke_log(args: argparse.Namespace) -> None:
    """Synthetic grouped rollout log covering every bucket the aggregator knows."""
    df = pd.read_parquet(args.candidate)
    ids = [m["env_key"].split("@", 1)[1] for m in df["metadata"]][:args.head]
    shapes = [
        ("mixed_success", lambda i: 1.0 if i % 2 else 0.0, False),
        ("all_success", lambda i: 1.0, False),
        ("all_fail_with_reward_variance", lambda i: 0.1 * i, False),
        ("all_fail_flat", lambda i: 0.0, False),
        ("infra_bad", lambda i: 0.0, True),
    ]
    root = Path(args.log_root)
    for n, task in enumerate(ids):
        _, score, err = shapes[n % len(shapes)]
        for i in range(args.group_size):
            d = root / "train" / task / f"sample_{i:02d}"
            d.mkdir(parents=True, exist_ok=True)
            rec = ({"error": "synthetic infra failure"} if err else
                   {"episode_return": score(i), "n_turns": 5 + i,
                    "terminated": True, "truncated": False})
            (d / "summary.json").write_text(json.dumps(rec))
    print(f"  wrote synthetic calibration log for {len(ids)} tasks -> {root}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-candidates")
    b.add_argument("--out", required=True)
    b.add_argument("--pools", nargs="+", default=["synth"], choices=sorted(POOLS),
                   help="default: synth only (the one corpus with zero eval provenance)")
    b.add_argument("--allow-env-reuse", action="store_true",
                   help="keep rows whose base task is an eval task. lite.scalecua is "
                        "100%% eval-derived at the environment level; the tasks "
                        "themselves are new. Recorded in the summary sidecar.")
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(fn=build_candidates)

    v = sub.add_parser("verify-candidates")
    v.add_argument("--candidate", required=True)
    v.add_argument("--allow-env-reuse", action="store_true")
    v.set_defaults(fn=verify_candidates)

    c = sub.add_parser("calibrate")
    c.add_argument("--candidate", required=True)
    c.add_argument("--log-root", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--group-size", type=int, default=4)
    c.add_argument("--target", type=int, default=1000,
                   help="GRPO will not train on more than ~1000 tasks; "
                        "selecting more is self-deception.")
    c.add_argument("--min-final", type=int, default=500)
    c.add_argument("--max-per-base", type=int, default=8)
    c.add_argument("--max-per-domain", type=int, default=384)
    c.add_argument("--all-success-cap", type=int, default=64,
                   help="Reserved slots (a floor, not a ceiling) for "
                        "behaviour-retention anchors. 64 of a 1000-task "
                        "budget; the number is a guess.")
    c.add_argument("--seed", type=int, default=42,
                   help="Seeds the tie-break shuffle. Ties must NOT fall back to "
                        "task_id: ids sort perturb_ < scalecua_ < synth_ and then "
                        "by application, which reproduces the --head footgun.")
    c.add_argument("--match-eval-families", action="store_true")
    c.add_argument("--base-eval-root", default=None,
                   help="A base-checkpoint eval run's log root. REPORTED ONLY -- prints "
                        "the base's solve/fail split and how the selected manifest "
                        "lands against it. It does NOT rank; an earlier version ranked "
                        "on it and the reasoning was refuted (see calibrate()).")
    c.set_defaults(fn=calibrate)

    s = sub.add_parser("write-smoke-log")
    s.add_argument("--candidate", required=True)
    s.add_argument("--log-root", required=True)
    s.add_argument("--head", type=int, default=8)
    s.add_argument("--group-size", type=int, default=4)
    s.set_defaults(fn=write_smoke_log)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
