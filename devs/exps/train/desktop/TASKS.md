# Desktop Anchor Tasks For osworld GRPO

Runbook for building the GRPO train task manifest that raises the fixed
`lite.osworld` `eval` score. Companion to
`devs/exps/train/browser/TASKS.md` (on branch `tmp/browser-webgym-anchor-20260915`, not on `dev`).

**Objective, stated plainly: produce a checkpoint that scores higher on
`lite.osworld` eval.** Not "demonstrate transfer". That choice is what makes
eval-derived corpora usable here — see [§0](#0-objective-and-what-it-licenses).

---

## 0. North star, and the budget it has to fit in

**Raise the `lite.osworld` eval score.** Not "demonstrate transfer" — the two
objectives pull opposite ways and the pipeline can only serve one:

| objective | eval-derived training data | verdict |
|---|---|---|
| **raise the eval number** | helps | **this runbook** |
| show GRPO transfers | destroys the claim | needs a held-out design, not here |

Consequences, made explicit so nothing is implied by silence:

- `train.perturb` (task-level leak) and `lite.scalecua rl` (environment-level)
  are both **in**, behind `--allow-env-reuse`.
- Every manifest carries an `EVAL_CONTAMINATION` field naming the leak kind per
  corpus. **Any number quoted from a checkpoint trained on it must carry that
  label.** It is not a generalization result and never becomes one.
- `lite.scalecua rl` covers **316 of the 328** scored eval tasks after its 240
  unrunnable rows are dropped, so once it is in the pool **essentially no eval
  task is held out**. A gain cannot be attributed to transfer even in principle.
  Accepted, not overlooked.

### The budget, which decides the whole design

```
GRPO will train on at most      ~1,000 tasks
measured calibration throughput  886-990 trajectories/hour  (dp=8, conc 96)
largest GRPO run in this project 1,536 trajectories
ScaleCUA's reported run          256,000 trajectories, no calibration
```

**Calibration is not free relative to training.** Classifying the full 4,220-task
pool at g=4 costs 16,880 trajectories — **eleven times the largest training run
ever done here**. That arithmetic drives two rules:

1. **Never calibrate more tasks than the training budget can use.** We need
   ~1,000; classifying 4,220 and discarding 3,200 spends 76% of the budget on
   tasks that will never be selected.
2. **Spend the saved budget on depth, not breadth.** At a fixed trajectory cost,
   fewer tasks at higher `g` beats more tasks at lower `g` — see §3.

⚠ **And if the follow-on training budget is itself ~1,500 trajectories, skip
calibration entirely and spend it on training.** Calibration buys roughly 2x
gradient utilization; paying 16,880 to double 1,500 is a bad trade. It only pays
when the training run that follows is much larger than the calibration.

### 🔴 Break-even, computed — decide the training budget BEFORE calibrating

The rule above is now quantified with this project's own measurements:

```
calibration cost                  8,440 trajectories (2110 x g4)
live-group rate WITHOUT selection   48.4%   (measured, 22 steps across 4 runs)
live-group rate WITH selection      87.5%   (p~0.5 at n=4)

break-even training volume = 8440 / (0.875 - 0.484) = 21,585 trajectories
                           = 168 GRPO steps at RBS=32 x n=4
                           = ~24 hours at the measured 900 traj/hour
```

| follow-on training | trajectories | net gain from calibrating |
|---|---:|---:|
| 8 h | 7,200 | **−5,625 effective** |
| 14 h | 12,600 | **−3,513** |
| 24 h | 21,600 | +6 |
| 48 h | 43,200 | +8,451 |

**The largest GRPO run in this project is 1,536 trajectories — 12 steps.** At
that scale calibration is a large net loss; the budget belongs in training.

⟹ **Calibrating is only justified by a commitment to ≥24 h of training that
follows.** The manifest is reusable across runs, which amortizes it — but only
partially: `mixed` is measured against the *base* checkpoint and decays as the
policy moves.



## 1. ❌ The "headroom" ranking, and why it is not in the selector

An earlier version of this runbook made **"prefer training tasks derived from
eval tasks the base FAILS"** the second ranking tier, on this evidence:

| run | of the base's 144 successes, kept | of its 184 failures, recovered |
|---|---:|---:|
| SYN | 86.8% | 10.9% |
| SYN3 | 83.3% | 12.5% |
| RFT | 78.5% | 10.3% |
| PTB | 71.5% | 8.7% |

The observation is real: recovery was flat at 8.7–12.5% and every score
difference came from retention. **The inference from it was not.** An adversarial
audit took it apart on five counts, each verified against the code or the data:

1. **Four runs of one method, not four probes of a law.** All four are
   rejection-sampling SFT on successful trajectories. Recovery is bounded by the
   *sampler's* ability to find a success at all — a task with `p_base = 0` never
   produces a training example. The flat band measures that procedure's coverage
   ceiling.
2. **Redundant where it is right.** At `p ≈ 1` "no upside" is true — and the
   bucket tier already demotes those tasks to `all_success`.
3. **Self-undermining where it is actually consulted.** It only changes anything
   inside `mixed_success`, i.e. `0 < p < 1`. Many of the base's 144 successes sit
   at `p ≈ 0.5–0.8` and come back *mixed*; consolidating those is precisely the
   retention lever the table says decided every prior run. The tier ranked them
   **last**.
4. **No mechanism for 70% of the rows it promoted.** 857 of the 1,216
   base-failure-derived rows are `scalecua`, which shares only the starting screen
   with its origin eval task (87.4% different verifier `func`). "Train on X ⟹ eval
   task E improves" has no route there.
5. **The label is one Bernoulli draw.** `base_eval_outcome` reads a single
   eval run. At this project's measured 13.3% repeat-disagreement, **~44 of 328
   tasks flip on a re-run** and **≥26% of labels carry no information** — and the
   noise peaks on exactly the mid-`p` tasks the bucket tier hands it.

⟹ **The base-eval split is reported by `--base-eval-root`, and used for nothing.**
It stays in the output because it is a useful diagnostic, not a ranking signal.

For the record, the split itself reproduces exactly: 1,216 rows derived from eval
tasks the base fails, 1,122 from ones it solves, 1,882 with no usable provenance.

## 2. Corpora

| corpus | rows | unrunnable | usable | eval provenance |
|---|---:|---:|---:|---|
| `lite.osworld train.synth` | 1722 | 18 | **1704** | **id screen is structurally blind here** — synth ids encode no origin, so it can only ever return zero. The instruction screen finds **5 rows carrying an eval instruction verbatim** (3 with an identical evaluator). Not a clean corpus, just an unprovable one |
| `lite.osworld train.perturb` | 707 | 0 | **707** | **task-level** — 707/707 decode to eval base tasks; instructions are rewrites of eval instructions (76% of eval) |
| `lite.scalecua rl` | 2049 | 240 | **1809** | **environment-level** — same starting screens, new goals/verifiers (**87.4%** different `func`, **97.9%** different `expected`; 27 rows share both with their origin eval task, 5 of them runnable) |
| `lite.scalecua train` | 20289 | 4282 | 16007 | environment-level; **not used, but the usual reason given is wrong** — its 95.6% state-check mix barely differs from `rl`'s 95.4%, which *is* used. The honest reason is cost: calibrating it at g=4 is 64k trajectories (~59 h) |

**Two screening rules the builder enforces, both learned by getting them wrong:**

1. **`others.exclude_reason` must be filtered.** The registry lists all 2049
   scalecua rows as present; the env rejects the excluded ones at setup
   (`ScaleCuaTaskError kind='excluded_task'`), burning the whole group. 258 rows
   across the pools carry a reason.
2. **Two independent screens, because one corpus defeats the other.** The id
   screen joins on the decoded base task, never the raw id — Each
   each corpus hides the origin differently (`scalecua` declares `osworld_id`,
   `perturb` encodes `perturb_<base>_<hash8>`, `synth` declares nothing). A missed
   decode is a **silent false negative**: raw id sets never intersect, so leaked
   data looks clean. That bug shipped once here and reported
   `perturb 707 rows, 0 eval-derived`. And because synth declares nothing, the id
   screen is a structural no-op on it — so the builder also screens on
   **instruction text**, which is what surfaced synth's 5 verbatim eval rows.

---

## 3. ⚠ Group size: depth beats breadth once the target is ~1,000

`--group-size` is not a quality knob layered on top of the selection — **it
decides whether the bucket tier is even correct.** A task's true success rate is
`p`; the calibration sees `g` draws and calls it `mixed` only if it observes both
outcomes:

```
P(classified mixed) = 1 - p^g - (1-p)^g

  true p     g=4      g=8     missed by g=4
    0.2     58.9%    83.2%        41.1%
    0.5     87.5%    99.2%        12.5%
    0.8     58.9%    83.2%        41.1%
```

A missed task is not merely ranked lower — it is misclassified as `all_success`
or `all_fail_flat` and **dropped from the candidate set entirely**. At `g=4`,
**41% of genuinely learnable `p≈0.2` tasks never make it in.**

Under a "collect as many as possible" objective that is tolerable. Under
**"select the best 1,000"** it is not: the manifest then contains tasks that are
not really mixed, while real ones were silently discarded.

### ✅ Resolved by measurement: `g=4` over a ~2,110-task subset

The argument above initially pointed at `g=8` over 2,110 tasks, on an estimated
**38.8%** `mixed` rate taken from a synth-only, n=49 sample. **Live measurement
on the combined pool gives 47.3%** (n=167, all three corpora), which changes the
arithmetic:

```
2,110 tasks x 47.3% mixed  =  ~998 tasks        <- already the --target 1000
```

and reveals a property the earlier reasoning missed:

> **A `mixed` verdict at `g=4` is PROOF that `0 < p < 1` — there are no false
> positives. `g=4` only produces false NEGATIVES, and those are concentrated on
> extreme-`p` tasks, which are exactly the ones the spread tier ranks last.**
> The bias points the same way the ranking does.

| plan | g | tasks | expected `mixed` | cost | wall clock |
|---|---:|---:|---:|---:|---:|
| **chosen** | **4** | **2,110** | **~998** | **8,440** | **~8 h** |
| deeper | 8 | 2,110 | ~1,265 | 16,880 | ~17 h |

The extra 9 hours buys finer spread estimates (worth ~15%, see §4) and ~267
additional `mixed` tasks that a 1,000-task budget cannot use. Not worth it.

**Uniform sampling, not stratified.** Early per-corpus `mixed` rates are synth
40.6% (28/69), perturb 50.0% (13/26), scalecua 52.8% (38/72) — a 12.2pp
scalecua−synth gap at **SE 8.3pp, z=+1.46, p≈0.14: not significant**.
Stratifying on it now would be treating noise as signal. Separating a 12pp gap
needs ~126 completed tasks per corpus; the full pass yields ~870/355/885, so the
question answers itself at the end.

⟹ **`g=4` over a random 2,110-task subset**, drawn with `--sample` under a fixed
`--seed` so it is reproducible and resumable; the candidate parquet is already
seed-shuffled, so per-corpus rates stay unbiased. A later `G=8` pass resumes into
the same log root if the spread tier ever needs the precision.

⚠ At `--target 1000` against ~998 `mixed`, the selector is **not really
selecting** — it takes essentially every mixed task. That is acceptable because
after the audit tiers 2 and 3 are worth little (≈15% and anti-clustering
respectively); tier 1 is the whole product.

### ⟹ Train with `--n-samples-per-prompt 4`, matching the calibration

Calibrating at `g=4` and training at `n=8` is safe but inconsistent: the label
says "this task gives a mixed group in 4 draws", and training then takes 8. The
matched choice removes the extrapolation entirely —

```
calibration verdict `mixed`  <=>  0 < p < 1 observed in 4 draws
training at n=4              <=>  the same 4 draws, every step
```

— and it is also the better use of a fixed environment budget. For the `p` range
the calibration selects for:

| p | live-group rate n=4 | n=8 | **live groups per trajectory** n=4 | n=8 |
|---|---:|---:|---:|---:|
| 0.3 | 75.2% | 94.2% | **0.188** | 0.118 |
| 0.5 | 87.5% | 99.2% | **0.219** | 0.124 |
| 0.7 | 75.2% | 94.2% | **0.188** | 0.118 |

**`n=4` yields ~1.8x the live groups per trajectory** — the same env budget buys
twice the prompt diversity and twice the gradient steps.

⚠ The cost is a noisier baseline: the group mean and std come from 4 samples
instead of 8. **No measurement here supports or refutes that trade** — it is the
ordinary "fewer samples, noisier estimate" argument, nothing more. Prior runs in
this project and ScaleCUA both used `n=8`, so this is a deliberate departure and
should be labelled as one wherever a result from it is quoted.

### Targets follow the same logic

| flag | value | why |
|---|---|---|
| `--target` | **1000** | GRPO will not use more. Selecting 1,500 is self-deception. |
| `--all-success-cap` | **64** | 128 reserved slots is 12.8% of a 1,000-task budget spent on an anchor whose only evidence (PTB) is the same confounded observation already used to justify `--max-per-base` — double-dipping. Halved, and treat the number as a guess. |
| `--max-per-base` | 8 | Near-inert on a 4,220 pool, but **binding when only 1,000 are picked**: 1,000 rows concentrated on 125 base tasks is exactly PTB's ~2-rows-per-base shape, which degraded the policy monotonically. |

## 4. Selection

```
1. bucket         mixed_success > all_fail_with_reward_variance > all_success
2. group spread   population std of the group's returns, descending
3. seeded shuffle
```

**Tier 1 is arithmetic and does essentially all the work.** GRPO's advantage is
`(r − group_mean) / group_std`; an all-0 or all-1 group contributes exactly zero,
and `grpo.py` replaces zero-std groups with a dummy sample *after* the rollout is
already spent. Across the five 32-task runs actually logged, **mean live-group
rate was 48.4%** — i.e. about half the rollout budget computed zeros. (An earlier
draft quoted "69% zero-variance", which is the worst step of the worst run.
~52% is the supported figure.)

**Tier 2 is direction-only, and small.** The obvious justification — "group std
is the advantage scale, so p≈0.5 gives 1.5× the gradient" — **is wrong, and
refuted by the code**: `grpo.py` divides by `adv.std()`
(`grpo_std_normalization` defaults True, nothing here disables it), so
`‖adv‖₂ = sqrt(n−1)` for *every* p and `max|adv|` is actually larger for a 1/4
group (1.50) than a 2/4 one (0.87).

What survives is weaker: a balanced group spreads the same advantage mass over
more trajectories (L1 mass 3.00 → 3.46 at g=4, **4.95 → 7.48 at g=8**), so no
single rollout dominates. **~15% at g=4; it only matters at g ≥ 8.**

Population std, not `|success_rate − 0.5|`: the latter is identically 0.5 for
every `all_fail_with_reward_variance` group and discards partial credit, ranking
`[0.998, 0.998, 0.998, 0.0]` — real usable variance — worst.

**Tier 3 exists to prevent a footgun, not to rank.** Ties must not fall back to
`task_id`: ids sort `perturb_ < scalecua_ < synth_` and then by application, so an
id tie-break clusters selection by corpus and by app — structurally the same
failure as `--head N` that §5 warns about. Measured under alphabetical order:
147 domain runs across 4,220 rows where random ordering gives ~3,700.

### ⚠ What tier 1 leaves undone

On the observed g=4 bucket rates (mixed 38.8%, all_fail_var **2.0%**,
all_success 26.5%, all_fail_flat 32.7%), replaying the selector over the real
4,220 candidates gives **only `mixed_success` picks and nothing else** (measured
at the then-current `--target 1500`; the same holds at 1000):

- **`all_fail_with_reward_variance` is ~2% of tasks** on this env — osworld
  verifiers are near-binary (only 4 of 328 base-eval rows carry partial credit).
  It is a real bucket but an almost empty one. This is also why "GRPO can learn
  from graded failure where RFT cannot" buys nearly nothing here.
- **`all_success` would never be reached**, so the cap could never fire. It is now
  a **reserved quota** (`min(--all-success-cap, available)` slots held back before
  the main pass), because a ceiling is not an anchor.

### Caps

| cap | default | why |
|---|---|---|
| `--max-per-base` | 8 | Direction evidenced, **magnitude arbitrary and nearly inert**: only 110 bases exceed 8 rows, holding 236 excess rows = 5.6% of the pool. The PTB evidence comes from a corpus at ~2 rows/base; a cap of 8 recreates nothing like it. Its *actual* effect here is different and more useful: all 110 over-cap bases are perturb+scalecua pairs on the same eval stem, so it functions as a **cross-corpus cap on rows sharing one eval task**. |
| `--all-success-cap` | **64** (code default, was 128) | Now a **floor** (reserved slots), not a ceiling. Rationale: training only on successes teaches "terminate early" — successful trajectories average **9.0 turns** vs **16.2** for failures across all three corpora, and every one ends with the identical `</think>\n\nDone.<\|im_end\|>`. ⚠ The PTB datum is used to justify this cap *and* `--max-per-base`; PTB was simultaneously narrow, task-leaked, success-only and over-trained, so one confounded observation cannot separately evidence two caps. Treat both magnitudes as guesses. |
| `--max-per-domain` | 384 | Keeps one application family from dominating. |
| `--match-eval-families` | off | Reweights toward eval's **53.7 / 42.1 / 4.3** family mix (the scored 328; using all 369 rows gives 50/38/12 and over-allocates ~3x to `other`). Off by default: the one time a family-based prediction was tested it was **refuted** (damage landed on the family the corpus covered *best*), so this is an untested lever, not a known good. |

---

## 5. Pipeline

### Step 1 — candidates

```bash
T=devs/exps/train/desktop/utils/tasks.py
D=devs/exps/train/desktop/data
CAND=$D/desktop.combined3.candidates.seed42.parquet

uv run python $T build-candidates \
  --pools synth perturb scalecua_rl --allow-env-reuse --out "$CAND"
uv run python $T verify-candidates --candidate "$CAND" --allow-env-reuse
```

Expected: **4,220 rows** (1704 + 707 + 1809), 258 unrunnable dropped, a
`!! 5 row(s) carry an EVAL INSTRUCTION verbatim` line for synth, and the
`EVAL_CONTAMINATION` sidecar naming both leak kinds.

`verify` hard-fails when eval-derived rows are present **without**
`--allow-env-reuse`, and also when any corpus jsonl is **absent** — the corpora
are gitignored asset downloads, and a missing one would otherwise make
`base_task_of` fall back to the raw id and report a clean pool.

### Step 2 — calibration rollout

```bash
G=4 bash .logs/run_CALIB2.sh   # 2110 x 4 = 8,440 trajectories, ~8 h
# the script's own defaults are G=8 SAMPLE=2110; pass G=4 explicitly.
# to deepen later:  G=8 bash .logs/run_CALIB2.sh   (same log root, resumes 4->8)
#
# NOT run_CALIB.sh -- that is the earlier g=4 full-coverage runner, kept only
# because its log root is the same and a pass from it resumes into this one.
```

`--group-size 4`, `--sample 2110 --seed 42`, `--group-shared-seed false`,
`--concurrency 96`, `--max-attempts 1`.

**Why 96 and not more.** Measured while running: host load **1.26 per core on 96
cores** (over-subscribed), RAM 13% used, and the env-server at **96 / 384
in-flight — only 25% of its cap**. The limit is host CPU, not server capacity or
memory; the server's cumulative `emergency` 503 counter (9,647) is exactly the
host-load/RAM trigger. In-flight also sits at ~70/96, i.e. env container churn,
not the ceiling, is what throttles throughput. Raising `--concurrency` buys 503s.

**Measured throughput: 886–990 trajectories/hour** once concurrency fills
(dp_size=8, tp_size=1 on 8 GPUs; GPU utilisation is uneven and often 0 — this
workload is environment-bound, not GPU-bound).

**Resumable.** The resume unit is `(task, sample_idx)` over `range(group_size)`,
so a later pass at a higher `G` on the **same `--log-root`** re-runs only the
missing indices. Requires a fixed `--seed` so `--sample` reproduces the subset.
The extra draws are genuinely independent: `sampling_seed` is derived from
`random.Random(f"{seed}:{task_id}:{sample_idx}")`, so `sample_idx` enters it.

⚠ Under `--max-attempts 1`, a *retryable* error writes `error.txt` and **no**
`summary.json`, so it does not land in `infra_bad` — the task falls below
`group_size` and is classified `incomplete` and dropped. **Watch the incomplete
rate as the real infra-health number**, not the `infra_bad` row. A resumed pass
will also re-run those indices (harmless, but the trajectory estimate is a lower
bound).

⚠ **`--sample`, never `--head`.** `--head N` takes the first N rows in file
order, which on these corpora is one application family at a time; a smoke test
that used `--head 8` read 81.2% success where the full set was 51.9%.

### Step 3 — select

```bash
uv run python $T calibrate \
  --candidate "$CAND" \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/calibrate.combined3.seed42 \
  --out $D/desktop.combined3.calibrated.g4.sample2110.seed42.parquet \
  --group-size 4 --target 1000 --min-final 500 \
  --all-success-cap 64 --seed 42 \
  --base-eval-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/base.gpt5_5.highr.i1
```

`--base-eval-root` is **diagnostic only** — it prints the base's solve/fail split
and the selected manifest's breakdown against it. It does not rank — see §1.

Prints per corpus: bucket rates, selected counts, the upside split, and the
median group std.

### Step 4 — GRPO

```bash
PROMPT_DATA=.../desktop.combined3.calibrated.g4.sample2110.seed42.parquet
EVAL_PROMPT_DATA=lite.osworld_eval
CONFIG_PATH=devs/exps/train/desktop/configs/qwen3_5/desktop.use.highr.i1.reasoning.yaml
N_SAMPLES_PER_PROMPT=4     # matches the calibration -- see the end of §3
```

From the `gpt5_5` + `<think>` SFT checkpoint. Promotion gate: `lite.osworld`
eval, n=328, `--concurrency 64`, judged with the **paired McNemar** test in
`.logs/paired.py` — the unpaired mean difference discards the pairing and
inflates the detection floor from ~4pp to ~7.6pp.

**Keep `max_steps: 30`.** Re-measuring the base at 50 gave **+0.73pp, z=+0.16**
— undetectable. Of the 60 tasks that exhausted 30 turns, only 6 converted, and
**32 of them did not even reach 30 turns on a re-run**: "hits the turn cap" is a
per-run noise realization, not a task property.

**Scale is the unaddressed variable.** Our largest GRPO run was **1,536
trajectories**; ScaleCUA reports 1,000 iterations × 32 tasks × 8 rollouts =
**256,000**, on 600 parallel VMs. Task selection raises gradient utilization; it
does not substitute for two orders of magnitude of rollout.

## 6. Smoke test

```bash
# Everything under /tmp. The smoke test must NOT rebuild the production manifest
# (a calibration run may be reading it) and must NOT rm -rf anything under .logs.
CAND=/tmp/smoke_candidates.parquet
SL=/tmp/smoke_calib_log
OUT=/tmp/smoke3.parquet
rm -rf "$SL" "$OUT" "$CAND" /tmp/smoke_candidates.summary.json /tmp/smoke3.*

uv run python $T build-candidates --pools synth perturb scalecua_rl --allow-env-reuse --out "$CAND"
uv run python $T verify-candidates --candidate "$CAND" --allow-env-reuse
uv run python $T write-smoke-log --candidate "$CAND" --log-root "$SL" --head 40 --group-size 4
uv run python $T calibrate --candidate "$CAND" --log-root "$SL" --out "$OUT" \
  --group-size 4 --target 20 --min-final 1 --all-success-cap 4 --seed 42 \
  --base-eval-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/base.gpt5_5.highr.i1
uv run python -c "
from lite.infer.rollout import resolve_prompt_data_tasks
s = resolve_prompt_data_tasks('$OUT', effective_env_id='lite.osworld')
print('resolver ok:', len(s), 'tasks')"
```

The synthetic log covers all five buckets. It validates format, screening,
ranking and the resolver — **it does not show eval will improve.**
