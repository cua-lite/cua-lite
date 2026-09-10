# WebGym — Collect With `gpt5_5`

Teacher runbook for the `gpt-5.5` collection of WebGym. Dataset-level targets,
the shared filter policy, install/serve setup, staging, upload, and export live
in [`../AGENTS.md`](/devs/data/webgym/AGENTS.md); run its
[§1 Install And Configure](/devs/data/webgym/AGENTS.md#1-install-and-configure)
first — it builds the image, starts the env-server, and pins `$COMMIT`.

This runbook ends at the INTERNALIZED clean log roots. They are the only thing the
dataset runbook consumes:

    .data/rollout/webgym/gpt5_5/$COMMIT/{d1..d7,popular}_clean.think

The bare `_clean` roots are an intermediate: they still carry `inline_reasoning`, and
`## Internalize Reasoning` below turns them into the `.think` siblings that stage reads.

## Prompt Design

Canonical recipe:
[/scripts/configs/gpt/recipes/collect/webgym.yaml](/scripts/configs/gpt/recipes/collect/webgym.yaml).

**Teacher prompt — balanced:** default to the open page; `goto`/search **only when the answer isn't
reachable on the current site**; **search on DuckDuckGo** (Google/Bing return blank pages in the
webgym browser) and **open+read a result** (don't answer from snippets); don't fabricate deep URLs.
(Both extremes are bad — see [Evidence](#evidence-prompt-exploration-n32tier): a *strict* no-search
prompt can't do research tasks, a *permissive* one flails search on easy tasks and tanks success.)

Prompt changes alter the training distribution. The dataset's KEEP/DROP policy for search behaviour
is dataset-level — see
[Search & navigation policy](/devs/data/webgym/AGENTS.md#search--navigation-policy).

## Attempt sizing — flat per-tier `--sample`

This batch used a **flat per-tier sample**, NOT a band-budget split: each **easy** tier (d1–3)
`--sample 500`, each **medium** tier (d4–6) `--sample 1000`, **hard** (d7) `--sample 2000`. The
curated popular pool runs the full ~2,102 tasks (no `--sample`). **Clean-yield** = success demos
surviving the [shared filter](/devs/data/webgym/AGENTS.md#shared-filter) (`--drop-failed
--drop-loops --drop-serp-only` + captcha/unsubmitted/illposed). Measured this batch (`a2cad60b`,
seed 1, n=1):

| tier | `--sample` | ran | clean | yield |
|------|-----------:|----:|------:|------:|
| d1 | 500  | 500  | 316 | 63% |
| d2 | 500  | 499  | 255 | 51% |
| d3 | 500  | 477  | 200 | 42% |
| d4 | 1000 | 975  | 470 | 48% |
| d5 | 1000 | 988  | 356 | 36% |
| d6 | 1000 | 997  | 376 | 38% |
| d7 | 2000 | 1,392 | 350 | 25% |
| popular | full (~2,102) | 2,065 | 928 | 45% |
| **total** | | **7,893** | **3,251** | **41%** |

Notes:
- **Yield falls steadily with difficulty** (d1 63% → d7 25%), as expected — harder tasks fail or
  loop more, so the same `--sample` buys fewer clean demos up the ladder.
- **d7 `--sample 2000` only ran ~1,392** — that's the available d7 site-start pool (the pool cap
  binds before 2000), giving **350 clean** at n=1. d8+ remain uncollected (`difficulty==7` filter
  only); adding them needs a separate `difficulty>=8` pass
  ([Scale to 5k](/devs/data/webgym/AGENTS.md#scale-to-5k--additive)).
- **Concurrency — keep GLOBAL ≈16** (throughput scales sub-linearly past ~16, see
  [Cost / time](#cost--time); ≈16 also stays well within the browser pool —
  `WEBGYM_INSTANCES`, dataset §1). Collect **one tier at a time** at
  `--concurrency 16` — never several tiers in parallel. Run the popular pool the same way (its own
  single phase at `--concurrency 16`), so the global concurrency is always ~16, one phase at a time.
- Re-confirm per-tier yields on the next scaled batch.

## Collect

Real collection writes to `.data/` (curated). Loop `d` over each difficulty with its `--sample N_d`.
`$COMMIT` and `CUA_LITE_ENV_SERVER_URL` come from the dataset runbook's
[§1 Install And Configure](/devs/data/webgym/AGENTS.md#1-install-and-configure).

```bash
# --- host ---

# step 1: collect (per difficulty d). Balanced prompt handles on-page vs search-when-needed.
# --filter is SITE-START ONLY for every tier (see dataset Collection targets): difficulty==d AND
# the start site is not a search-engine root (their SERP is blank → ~0 yield). Only the *roots* are
# excluded — real product sites like accounts.google.com / translate.google.com stay in.
# Per-tier --sample (flat, see Attempt sizing above): d1,d2,d3 → 500 each; d4,d5,d6 → 1000 each;
# d7 → 2000.
# Collect ONE tier at a time (loop d over the difficulties) at --concurrency 16 — never several
# tiers in parallel (that would push global concurrency past ~16, see Attempt sizing / Cost).
D=7; N=2000          # one tier per run: set D to the tier and N to its budget above
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id webgym \
  --splits train --sample "$N" --seed 1 --concurrency 16 --max-attempts 2 \
  --filter "lambda m: m.others.get('difficulty',0)==$D and m.others.get('website','').split('//')[-1].split('/')[0].removeprefix('www.') not in ('google.com','bing.com','duckduckgo.com')" \
  --config-path scripts/configs/gpt/recipes/collect/webgym.yaml \
  --log-root ".data/rollout/webgym/gpt5_5/$COMMIT/d$D"

# step 1.5: PRIORITY — the curated "popular" pool (high-value, do this first / weight it heavily).
# `webgym_popular_2102.parquet` is OpenWebRL's filtered+cleaned popular subset (2102 train tasks).
# It is already pre-filtered, so it is driven by --prompt-data (NOT --splits/--filter, which are
# mutually exclusive with it — the parquet IS the task list). Because these tasks matter more, give
# them a LARGER attempt budget: --max-attempts 5 (vs 2 for the bulk difficulty tiers) so a transient
# judge/nav hiccup gets retried instead of lost. Run the FULL pool (no --sample) — all 2102 tasks.
# Use --group-size 1 (same as the bulk tiers): one rollout per task — coverage over diversity, and
# consistent with the rest of the corpus. Resume re-runs only the unfinished tasks.
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id webgym \
  --prompt-data lite/gym/envs/webgym/data/webgym_popular_2102.parquet \
  --seed 1 --concurrency 16 --max-attempts 5 --group-size 1 \
  --config-path scripts/configs/gpt/recipes/collect/webgym.yaml \
  --log-root .data/rollout/webgym/gpt5_5/$COMMIT/popular
# then filter it exactly like a tier (step 2 with --log-root .../popular --out .../popular_clean).
```

**Flat per-tier `--sample`**: 500 each for d1–3, 1000 each for d4–6, 2000 for d7 — NOT a band
budget split. Medium is 92% site-start and hard 45% (3,007 tasks) — the SITE-START filter just drops
the google·bing-blank remainder (and d7 site-start runs out at ~1,392 < 2000).

**On failure (crash / stall / host-overload kill): just re-launch the SAME command — do NOT wipe
`--log-root`.** A fixed `--log-root` makes `api.py` *resume*: it re-runs only the missing samples and
skips already-succeeded tasks (`lite/infer/rollout.py` `get_pending`; see `--max-attempts` help).
Deleting the log-root throws away good trajectories and re-rolls the whole tier — only do that for a
genuinely corrupt run. A stall looks like: the tier log stops growing (`mtime` frozen) for minutes
with the process parked in `do_poll`, usually the aftermath of a step-timeout wave when the host is
oversubscribed (other users' jobs spiking load) — kill ONLY your own driver process group
(`kill -KILL -<pgid>`, never others' procs and never the shared OmniBoxes backend container) and
relaunch; resume continues from where it left off.

## Filter and quality check

```bash
# step 2: filter — success filter (--drop-failed) + strip no-ops + drop stalls + drop snippet-scrape
#   (keep productive research) + drop captcha-wall / unsubmitted / ill-posed-task trajectories.
#   This is the ONLY filtering step; stage (dataset step 3) stages everything. No-op stripping leaves
#   surviving image indices untouched and preserves text, metadata, and earlier indexed reference
#   image parts when carrying a dropped leading goal forward.
uv run python devs/data/webgym/filter.py \
  --log-root ".data/rollout/webgym/gpt5_5/$COMMIT/d$D" --out ".data/rollout/webgym/gpt5_5/$COMMIT/d${D}_clean" \
  --drop-failed --drop-loops --drop-serp-only --drop-captcha --drop-unsubmitted --drop-illposed-task
```

`quality_check.py` (co-located with `filter.py`) is the canonical quality battery on the success
demos — action histogram, `goto` breakdown, footgun prevalence (`serp_only` / `searched` /
`search_recover` / `captcha` / `back_bounce` / `scroll_hunt` / `loop` / `deepx` / `unsubmitted`),
`action_description` coverage, and the per-tier difficulty histogram. Run it on a tier, or on the
whole batch, before handing the cleaned roots to the dataset runbook:

```bash
uv run python devs/data/webgym/quality_check.py .data/rollout/webgym/gpt5_5/$COMMIT/d7
uv run python devs/data/webgym/quality_check.py .data/rollout/webgym/gpt5_5/$COMMIT   # all tiers
```

## Internalize Reasoning

The last step before staging. This teacher is PROMPTED for a `Thought:` line — see
`inline_reasoning_instruction` in
[the collect recipe](/scripts/configs/gpt/recipes/collect/webgym.yaml) — which the
`gpt.teacher` agent parses into an `inline_reasoning` CONTENT PART.
[`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py) moves it into the
`reasoning_content` FIELD, the same one a teacher sampled with `enable_thinking` writes
natively. Same fact, one shape, so the PUBLISHED rows do not make every consumer ask which
config produced them.

```bash
# ROOT, not D: `D` is the difficulty tier set above and is still needed if you loop back.
for ROOT in .data/rollout/webgym/gpt5_5/$COMMIT/*_clean; do
  uv run python devs/data/internalize_cot.py --in "$ROOT" --out "$ROOT.think"
done
```

Run it on a root rebuilt by `unstage` too
([Add A Config To A Published Dataset](/devs/data/AGENTS.md#add-a-config-to-a-published-dataset)),
without checking first. Rows published BEFORE this step existed still carry
`inline_reasoning`, and re-staging one of those beside a `.think` root would put two
reasoning shapes in one repo — silently, since nothing downstream rejects either. The pass
is idempotent in CONTENT: on already-canonical rows it finds no `inline_reasoning` and
reports `0 assistant turns`. It is not idempotent in PLACEMENT — a non-empty `--out`
raises `FileExistsError` rather than merging a stale tree into a fresh one, so re-running
over a surviving `.think` root needs `--overwrite`.

The `.think` roots hold parquet only: image refs are rewritten to absolute, so the copy is
small and stages the same image bytes. It also pops `raw_response`, whose saved provider
payload no longer matches the mutated message.
[The dataset runbook](/devs/data/webgym/AGENTS.md) stages THESE roots.

## Cost / time

This batch ran **~5,830 difficulty-tier attempts** (`--sample` 500/1000/2000, see Attempt sizing)
**plus ~2,065 for the popular pool** (the ~2,102-task pool × `--group-size 1`, step 1.5) ⇒ **~7,900
total** → 3,251 clean. `gpt-5.5` latency-bound. Measured throughput at concurrency 56 ≈ **~1.2
traj/min** (~26 turns/min) ⇒ on the order of **days** single-stream (the popular pool adds ~35% to
the tier run). That 56 figure is the prior batch; the **recommended global 16** (Attempt sizing) is
slower in wall-clock but trades throughput for pool stability (gains past ~16 are sharply
diminishing anyway — see below).

**Throughput root-cause (2026-06-17 investigation; full reproducible analysis + commands in
[/devs/envs/webgym.md](/devs/envs/webgym.md)):**
- Per-turn nameable cost ≈ **15–20 s**: LLM ~8–12 s (dominated by **input/context** processing
  of the chained multi-screenshot history — NOT output; ~94 out-tok/turn measured
  when `reasoning_effort` was `low`) + env ~2–5 s. Plus a **per-trajectory VLM reward judge ~30 s**. × 20+ turns
  on hard tiers ⇒ trajectories are inherently minutes long.
- **Throughput scales strongly sub-linearly** with concurrency (clean isolated-pool sweep:
  conc-16 per-slot 3.2× worse than conc-1; production conc-56 ≈ 126 s/turn/slot vs ~15–20 s of
  nameable work). The rollout process is **CPU-idle** (not loop/GIL-bound); ruled out by direct
  measurement: provider concurrency, every single env op, litellm, memory/GC, wedged instances. The
  residual is consistent with combined-load I/O contention but was not definitively pinned
  (py-spy blocked by ptrace).
- **Levers (in order):** reduce turns (`max_steps`) on failing tiers · cheaper/shared reward
  judge · **horizontal scale across hosts/independent pools** (NOT more processes on one host —
  multiprocessing won't help since it's not CPU-bound) · keep the pool stable (it crashes under
  load, which also drags throughput). Output length was **not** a lever at the
  `reasoning_effort: low` these numbers were taken at; the collect recipe now pins
  `medium` ([/scripts/configs/gpt/recipes/collect/webgym.yaml](/scripts/configs/gpt/recipes/collect/webgym.yaml)), so re-measure
  out-tok/turn before reusing this cost model.
- Never exceed concurrency 64 (pool size); adding concurrency past ~16 gives sharply diminishing
  returns.

## Evidence (prompt exploration, n=32/tier)

| prompt | easy succ | med succ | hard succ | goto behavior |
|--------|----------:|---------:|----------:|---------------|
| strict (no search) | ~42% | ~23% | ~12% | can't do research tasks |
| **permissive** (over-loose) | 13% | 14% | **0%** | 86–96% of attempts goto; flails search on easy |
| **balanced** (current) | **41%** | **34%** | **10%** | easy demos on-page; search only when needed |

- Per-task A/B: `191272` (arxiv) — permissive `goto google`; balanced stays on arxiv, uses its own
  search, succeeds.
- Click-through (productive research vs snippet-scrape): medium 18/18, hard 23/24 search-trajectories
  click through → the `serp_only` filter keeps them and drops only the snippet-scrapers.
- **Bing/Google blank-SERP** confirmed by screenshot → forced engine-bouncing. The ddg-prompt fix
  (the +ddg prompt) made search single-engine: medium 20/20 search demos use only DuckDuckGo (the prior prompt bounced).
- Post-filter clean demos (per 32): balanced easy 12 / med 11 / hard 3 → balanced+ddg
  easy 13 / med 15 / hard 2–3. ddg lifts easy+medium and removes the bounce; hard stays low/noisy.
- **Site-start hard demos come in two good shapes, both kept:** (a) solved **on-page** —
  e.g. `172888` Amazon search, `177424` Apple spec comparison (scroll), `251269` BBC search; and
  (b) solved by **ddg → click a result → read** (research) when the open site lacks the answer.
  `serp_only` keeps both and drops only snippet-scraping. (Search-*start* hard — opening on
  google.com — is excluded at collection: its SERP is blank, ~0 yield. See dataset Collection
  targets and Attempt sizing above.)
- **End-to-end chain validated:** collect→clean→stage→`export_sft` on the +ddg demos → 30 success rows →
  valid Qwen SFT parquet (GPT `action_description` → Qwen `Action:`+`<tool_call>` targets).

## Current status

- The ddg prompt, site-start restriction, and clean→stage→export_sft chain have already been
  validated on small batches.
- Current collection should use the env-server path (dataset §1) and the
  [Collection targets](/devs/data/webgym/AGENTS.md#collection-targets) / Attempt sizing tier mix.
  Keep hard near the documented share unless a fresh scaled batch proves the site-start yield is
  materially higher.
- For any scaled batch, record per-tier clean yield before publishing. The small-batch hard yield is
  noisy enough that it should not by itself justify raising the hard share.
