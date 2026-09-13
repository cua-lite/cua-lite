# desktop.use — teacher x screenshot-profile x reasoning campaign (SFT + one RL run)

Train **fifteen checkpoints** — three screenshot profiles x three teachers, plus a `<think>` arm
on the two teachers that emit reasoning — and score them on one eval. A single GRPO run from the
same base weights closes the file (**RL** at the end), scored on that same eval.

| | `lowr.i4` | `lowr.i1` | `highr.i1` |
|---|---|---|---|
| | 1280x704, 1-4 img | 1280x704, 1 img | 1920x1088, 1 img |
| **`gpt5_5`** | ✓ | ✓ | ✓ |
| **`gpt5_5` + `<think>`** | ✓ | ✓ | ✓ |
| **`qwen3_5_27b`** | ✓ | ✓ | ✓ |
| **`qwen3_5_27b` + `<think>`** | ✓ | ✓ | ✓ |
| **`qwen3_8_27b`** | ✓ | ✓ | ✓ |

All three cap SCREENSHOTS and nothing else: `iN` sets `image_max: N` (with `fold_size` tracking
it) and leaves `history_n` at the protocol default of 100, so every past turn is still rendered in
full — its text, its literal `<tool_call>`, and on the reasoning arm its `<think>`. Only the old
*pixels* become `"This screenshot has been collapsed."`, which is what the reference agent
effectively does ([xlang-ai/OSWorld#448](https://github.com/xlang-ai/OSWorld/pull/448)). Sizes
above are post-`smart_resize` (x32): the yamls ask for 1280x720, and `highr.i1` takes the envs'
native 1920x1080.

The three form an L with `lowr.i1` at the corner: `lowr.i4` vs `lowr.i1` moves **image count**,
`lowr.i1` vs `highr.i1` moves **resolution**, and the diagonal moves both — not a profile effect.

> **Train and eval must use the same profile yaml.** `$P` — the config stem — selects both the
> checkpoint and `--config-path`, in every SFT block below. A `highr.i1` checkpoint scored under
> `lowr.i4` is measuring a prompt surface it never saw.

**What compares to what.** Within a ROW only the profile moves: every cell draws the same 5000
rows from **Lite.ScaleCUA** at the same seed. Within a COLUMN, a `+ <think>` row differs from the
row above by `enable_thinking` alone — so read a reasoning cell against **its own teacher's**
Action-only row, never the other teacher's. **Across teacher rows is NOT a clean contrast**: each
teacher's pool is its own successes (`episode_return > 0.5`), so the pools differ in membership
and the shared seed buys nothing — measured, `gpt5_5` and `qwen3_8_27b` share only **43%** of
their task ids. To make it clean, intersect the pools first and draw every teacher's 5000 from
that intersection. (The third pool has not been measured against either.)

`qwen3_8_27b` has no `<think>` arm: thinking was off when it was sampled, so the reasoning config
would train an empty `<think>` — and the empty-vs-filled mismatch drops packing from 3.5
steps/segment to 1.0. The other two reach the field by different routes: `gpt5_5` is prompted for
a `Thought:` line that [`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py)
canonicalizes before staging; `qwen3_5_27b` was sampled with `enable_thinking` and writes it
natively.

Three profiles on disk are deliberately NOT cells. `desktop.use.default.yaml` is the fourth
corner `highr.i4` — 8160 peak vision tokens per step, 2.3x `lowr.i4`, excluded on that memory
ceiling (which is what sets MBS), not on total cost. The `hN` pair (`lowr.h1`, `highr.h1`) caps
TURNS instead, collapsing older ones into a `Previous actions:` summary: measured, `h1` carries
**0** historical `<think>` blocks where `i1` carries all of them — wrong shape for a reasoning
campaign, kept for a future history-representation study.

**Naming.** A cell is a `(config stem, teacher)` pair; the SFT blocks enumerate the fifteen from
a `cells()` function that each block redefines (four copies — paste the one in the block you are
running). RL uses none of it. `$P` is the stem taken whole off the filename and
`$DS` the dataset recipe (`scalecua_5k` = source + row cap). Both are threaded verbatim into every
artifact — parquet `$P.$DS.$T.parquet`, checkpoint `sft.$P.$DS.$T`, HF repo
`ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T`, W&B group — so nothing downstream re-derives them and two
recipes never collide. To ablate the dataset, change `DS=` **and** `--sample` together.

### SFT

#### Export

One parquet per cell — the config decides what the model sees, so no two cells can share one.

```bash
# --- TRAIN HOST ---  (the Slime container mounts the repo root, so its .data/ is what Train
# reads; exporting on the eval host leaves Train with no parquet to read)
DS=scalecua_5k                           # dataset recipe: source + row cap
DL=.data/huggingface                     # per-teacher roots: $DL/$T/cua-lite/...
OUT=.data/sft/qwen3_5/desktop.use
CFG=devs/exps/train/desktop/configs/qwen3_5
FILTER="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

# The fifteen cells as "<config stem> <teacher>". The <think> arm skips qwen3_8_27b.
cells() {
  for T in gpt5_5 qwen3_5_27b qwen3_8_27b; do
    for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do echo "$P $T"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in desktop.use.lowr.i4.reasoning desktop.use.lowr.i1.reasoning \
             desktop.use.highr.i1.reasoning; do
      echo "$P $T"
    done
  done
}

# One root per teacher, because export_sft reads whatever --data-paths names and a shared root
# would pool them. --allow-patterns bounds the walk as well as the fetch, so a warm HF cache
# cannot drag in the `rl` variant (which lives in this same directory). --overwrite makes this
# re-runnable. KEEP THIS LIST IN SYNC WITH cells(): a teacher in cells() but not here exports
# against a data root that was never downloaded -- and cells() is defined FOUR times below, so a
# teacher edit touches all four. How each teacher is collected: devs/data/lite.scalecua/<T>/AGENTS.md
for T in gpt5_5 qwen3_5_27b qwen3_8_27b; do
  uv run python -m lite.data.hf.download Lite.ScaleCUA \
    --allow-patterns "desktop/use/train/desktop.use.train.$T/*" \
    --out "$DL/$T/cua-lite/Lite.ScaleCUA" --overwrite
done

# --filter runs before --sample, so the 5000 cap lands on kept rows. The fixed --seed makes the
# shuffle identical, but --sample is a guarantee on the OUTPUT: under --no-strict a dropped row
# is topped up from further down that same order, so two cells that drop DIFFERENT rows both
# reach 5000 with different row sets (export_sft.py: "a superset-prefix ... not the identical
# set"). The counts below cannot see that -- compare each cell's `Skipped N rows` line too.
# All-zero means every cell drew the same rows, which is what every cell here assumes.
while read -r P T; do
  uv run python -m lite.train.export.export_sft \
    --config "$CFG/$P.yaml" \
    --model-id Qwen/Qwen3.5-4B \
    --data-paths "$DL/$T/cua-lite/Lite.ScaleCUA" \
    --image-root "$DL/$T" \
    --filter "$FILTER" --sample 5000 --seed 42 --no-strict \
    -o "$OUT/$P.$DS.$T.parquet" < /dev/null   # or the child eats the rest of the cell list
done <<< "$(cells)"

# `--no-strict` makes a conversion failure a SKIP, not an error: a wrong --image-root
# prints "Skipped N rows ... Wrote 0 trajectory rows" and still exits 0, and run_sft.sh
# would then train on an empty parquet. `--sample` is a contract on the OUTPUT -- export
# converts in rounds until 5000 SURVIVE -- so anything but 5000 means it ran out of
# convertible rows, and export said so on its own line. Count them before trusting them.
while read -r P T; do
  uv run python -c "
import sys, pyarrow.parquet as pq
n = pq.read_metadata(sys.argv[1]).num_rows
print(('OK   ' if n == 5000 else 'SHORT'), n, sys.argv[1])
" "$OUT/$P.$DS.$T.parquet" < /dev/null
done <<< "$(cells)"
```

#### Train

```bash
# --- Slime container, TRAIN HOST ---  (DS must match the Export block)
# SFT at TP=2 (8 GPUs -> DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't
# THD-pack). Serial: each run takes all 8 GPUs.
#
# SAVE_DIR / SAVE_HF_DIR / WANDB_GROUP_SUFFIX are all MANDATORY here. run_sft.sh keys the two
# checkpoint dirs AND the W&B group off PROMPT_DATA's parent dir (DATA_SLUG), which is
# `desktop.use` for all fifteen cells -- so unset, the runs overwrite each other's checkpoints
# and land in one W&B group. WANDB_GROUP_SUFFIX has no default; separating them is its job.
#
# Leave SAVE_INTERVAL unset: run_sft.sh defaults it to 1000 (run_sft.sh:135), far past the ~312
# steps a 5000-row/GBS-32/2-epoch cell takes, so the step-interval save never fires. What writes
# the two iter_* dirs is slime's epoch-boundary save -- `step % num_rollout_per_epoch == 0` in
# slime/slime/utils/misc.py (num_rollout = num_rollout_per_epoch * NUM_EPOCH is set in
# slime/slime/ray/placement_group.py) -- so NUM_EPOCH=2
# gives exactly 2. Ship gates on finding exactly $EPOCHS of them; SKIP on every cell means that
# premise broke (a slime bump can move it), and the checkpoints are still on disk.
DS=scalecua_5k
# The fifteen cells as "<config stem> <teacher>". The <think> arm skips qwen3_8_27b.
cells() {
  for T in gpt5_5 qwen3_5_27b qwen3_8_27b; do
    for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do echo "$P $T"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in desktop.use.lowr.i4.reasoning desktop.use.lowr.i1.reasoning \
             desktop.use.highr.i1.reasoning; do
      echo "$P $T"
    done
  done
}

while read -r P T; do
  TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
    MODEL_ID=Qwen/Qwen3.5-4B \
    SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=2 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
    PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5/desktop.use/$P.$DS.$T.parquet \
    SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-4b/sft.$P.$DS.$T/iter_{rollout_id} \
    SAVE_DIR=/root/checkpoints/qwen3_5-4b/sft.$P.$DS.$T/megatron \
    WANDB_GROUP_SUFFIX=".$P.$DS.$T" \
    bash /workspaces/cua-lite/scripts/train/run_sft.sh < /dev/null   # else it eats the cell list
done <<< "$(cells)"
```

- `MBS=1` is the safe start, and **tune it on `lowr.i4`** — that column has the longest sequence
  so it binds first (peak vision tokens per step: `lowr.i4` 3520, `highr.i1` 2040, `lowr.i1` 880).
  Total training COST inverts that ordering: `lowr.i4` packs adjacent steps into one sequence and
  the one-image profiles cannot — measured 0.52x of `lowr.i1` and 0.37x of `highr.i1`. Longest
  sequences, smallest bill. TP=2 fits all three at 4B; `TP_SIZE=4` if it OOMs.
- `NO_SAVE_OPTIM=1` keeps weights only — these checkpoints are for eval, not for resuming.
- **Export `WANDB_API_KEY` before the first cell.** `run_sft.sh` builds its W&B arguments inside
  `if [ -n "${WANDB_API_KEY:-}" ]`, so without it fifteen multi-hour runs train with no logging
  at all.

#### Ship the checkpoints

Train and eval usually run on different machines, so checkpoints travel through the Hub:
**repo** = the cell (`ZHZisZZ/qwen3_5-4b.sft.<config-stem>.<dataset>.<teacher>`),
**subdir** = `epoch_1/`, `epoch_2/`, **tag** = the producing commit.

The tag is provenance, not the entry point: **eval pulls `main`**, so a re-run replaces the epoch
dirs and the next campaign picks them up with no sha to carry between hosts. Use `--revision
<tag>` only to re-run an OLD campaign; a re-run at the same commit *moves* the tag. Epoch dirs are
named by rank at upload time — training writes `iter_<N>` with slime's 0-based index, a number to
read off disk, never predict.

```bash
# --- TRAIN HOST ---  (run from the repo root; commit FIRST -- `git rev-parse` reports a sha
# for a dirty tree just as happily, so an uncommitted edit tags fifteen public repos with a sha
# that does not describe the weights, silently.)
# `uv run hf`, NOT bare `hf`: `hf repos` needs huggingface_hub >= 1.17 and the system hf may be
# older. Needs a WRITE-scoped token (`uv run hf auth login`, or HF_TOKEN). Repos are public,
# so the eval host needs no auth at all.
COMMIT="$(git rev-parse --short HEAD)"   # commit message + tag; eval still pulls main
DS=scalecua_5k
EPOCHS=2                                 # must match NUM_EPOCH in the Train block
CKPTS=.ckpts/qwen3_5-4b
# The fifteen cells as "<config stem> <teacher>". The <think> arm skips qwen3_8_27b.
cells() {
  for T in gpt5_5 qwen3_5_27b qwen3_8_27b; do
    for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do echo "$P $T"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in desktop.use.lowr.i4.reasoning desktop.use.lowr.i1.reasoning \
             desktop.use.highr.i1.reasoning; do
      echo "$P $T"
    done
  done
}

while read -r P T; do
  REPO="ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T"
  # `ls` first, and gate on the count here: creating the repo up front would leave an empty
  # public one behind for a cell that never trained, and a half-trained cell must not reach
  # the eval host, which pulls whatever is on main.
  DIRS=$(ls -d "$CKPTS/sft.$P.$DS.$T"/iter_* 2>/dev/null | sort -V)
  if [ "$(echo "$DIRS" | grep -c .)" -ne "$EPOCHS" ]; then
    echo "SKIP $REPO: $(echo "$DIRS" | grep -c .) iter dir(s), expected $EPOCHS -- not uploading"
    continue
  fi
  # NOTE: this runs before the per-file check below, so a cell whose FIRST epoch dir is
  # incomplete still leaves an empty public repo. The count gate above is what prevents that
  # for a cell that never trained at all.
  uv run hf repos create "$REPO" --repo-type model --exist-ok < /dev/null

  ep=0
  for D in $DIRS; do
    ep=$((ep + 1))
    # eval loads the tokenizer/processor from --model-path, not --model-id, so the epoch dir
    # must carry slime's FULL HF export. Check before spending 8 GB of upload on it:
    # a weights-only dir only fails on the eval host, after upload AND download.
    for f in config.json tokenizer_config.json preprocessor_config.json; do
      [ -e "$D/$f" ] && continue
      echo "SKIP $REPO: $D has no $f -- AutoProcessor would fail at eval; stopping this cell"
      ep=-1; break
    done
    [ "$ep" = "-1" ] && break
    uv run hf upload "$REPO" "$D" "epoch_$ep" \
      --repo-type model --commit-message "$COMMIT: epoch $ep (from $(basename "$D"))" \
      < /dev/null \
      || { echo "SKIP $REPO: upload of $D failed -- not tagging"; ep=-1; break; }
  done
  # Both breaks above escape the `for D` loop ONLY; without this, a half-uploaded cell falls
  # through and gets tagged as if complete. Pushed epoch dirs stay on main, untagged.
  [ "$ep" = "-1" ] && continue

  # Tag the commit that produced these weights. Eval pulls `main` and never needs this; it is
  # here so an old campaign stays reproducible -- `--revision <tag>` on the download.
  # create-or-move; the Hub has no move, so it is delete-then-create.
  uv run hf repos tag delete "$REPO" "$COMMIT" --repo-type model --yes < /dev/null \
    || echo "note: no existing '$COMMIT' tag on $REPO (expected on a first upload)"
  uv run hf repos tag create "$REPO" "$COMMIT" --repo-type model -m "$COMMIT" < /dev/null \
    || echo "ERROR $REPO has NO '$COMMIT' tag now; weights are on main -- re-tag by hand"
done <<< "$(cells)"
```

`hf upload` is single-commit and not resumable; `upload-large-folder` is, but takes no
path-in-repo and so cannot express `epoch_<k>/`.

#### Eval

Eighteen runs on the `lite.osworld` eval split: the rows left after `--filter` drops
`exclude_reason`, **328 of the 369** `catalog.lock.json` pins. The 41 exclusions are
recorded in the tracked catalog lock generated from `eval.jsonl`; count them there
rather than restating denominators by hand. Env
setup: [`lite/gym/envs/lite/osworld/README.md`](/lite/gym/envs/lite/osworld/README.md).

Eighteen, not fifteen: **the base model runs once per screenshot profile.** A checkpoint must be
scored against a baseline that saw the same screenshot surface, and the fifteen cells use three.

```bash
# --- EVAL HOST ---
# RUN names this campaign's artifacts. It is NOT a code version -- this pulls each repo's main,
# whatever the train host last pushed. To score an OLD campaign instead, add `--revision <tag>`
# to the download below; the tags are there. Any label works; a date is easiest.
# It only has to CHANGE between campaigns: rollout resumes a log-root sample by sample and
# never checks what produced it, so a slug that did not rotate silently re-reports the
# previous campaign's numbers -- including the base run every cell is read against.
# No default on purpose: a `date`-derived one does not rotate on a same-day re-run, and
# ${RUN:-...} in a reused shell keeps the OLD value -- both silently blend two campaigns'
# samples into one summary.json. Under `bash script.sh` the line below exits 1; PASTED into an
# interactive shell it only prints and the rest runs on with RUN empty -- same caveat as the
# MISSING check further down. Read the error and stop by hand.
: "${RUN:?set RUN to a fresh slug for THIS campaign, e.g. 20260910a}"

# MUST be unset. `--sglang-server-url` defaults to $SGLANG_SERVER_URL (lite/infer/cli.py), and
# with a URL in hand serving.py never starts a server -- `--model-path` then only picks the
# tokenizer/processor, so all eighteen runs GENERATE from whatever model that server holds and
# every summary.json still looks normal. Nothing warns.
unset SGLANG_SERVER_URL

# The env-server is REQUIRED, and it is a scoring condition, not an implementation detail:
# unset, rollout runs envs in-process and owns the containers itself, which changes container
# lifecycle and timing. The size of the effect is NOT established -- an earlier note here claimed
# direct mode scored 0.017-0.035 lower, from a single pass; the six paired diffs behind it were
# all within +/-0.016 with mixed sign, which this campaign's noise floor (half-range up to
# +/-0.020) covers entirely. Treat the mode as a condition to hold fixed, not as a known offset:
# every number in the Results table must come from the same mode, so a campaign that mixes them
# cannot be read across columns.
export CUA_LITE_ENV_SERVER_URL="http://$(hostname -I | awk '{print $1}'):30100"
export CUA_LITE_ENV_SERVER_TOKEN=desktop-eval   # passthrough auth; any value scopes your envs
EPOCH=epoch_2                            # epoch_1 = after 1 epoch
DS=scalecua_5k
CFG=devs/exps/train/desktop/configs/qwen3_5
PULL=.ckpts/pulled
LOGS=.logs/rollout/Qwen_Qwen3.5-4B/lite.osworld
# The fifteen cells as "<config stem> <teacher>". The <think> arm skips qwen3_8_27b.
cells() {
  for T in gpt5_5 qwen3_5_27b qwen3_8_27b; do
    for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do echo "$P $T"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in desktop.use.lowr.i4.reasoning desktop.use.lowr.i1.reasoning \
             desktop.use.highr.i1.reasoning; do
      echo "$P $T"
    done
  done
}

MISSING=
while read -r P T; do
  uv run hf download "ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T" \
    --include "$EPOCH/*" \
    --local-dir "$PULL/sft.$P.$DS.$T@$RUN" < /dev/null
  # snapshot_download has no empty-match guard: a wrong $EPOCH yields an empty dir, silently.
  # All three, not just config.json: a dir with config.json but no processor files passes a
  # one-file check, then dies inside a backgrounded score job at AutoProcessor.from_pretrained.
  for f in config.json tokenizer_config.json preprocessor_config.json; do
    [ -e "$PULL/sft.$P.$DS.$T@$RUN/$EPOCH/$f" ] || MISSING="$MISSING $P.$T:$f"
  done
done <<< "$(cells)"
# A missing checkpoint must not reach the score loop: it would take a GPU and fail as a crashed
# background job long after `wait`. Like every check in this file it only PRINTS — a pasted
# block cannot abort itself — so read the line and stop by hand before running the rest.
[ -z "$MISSING" ] && echo "checkpoints OK" \
  || echo "STOP -- do not run the score loop; missing under $EPOCH:$MISSING"

# $1 = config stem, $2 = --model-path ("" = base model), $3 = log slug. --config-path always
# follows $1, so a checkpoint is only ever scored on the surface it was trained on.
# CONCURRENCY IS A BUDGET ON THE HOST, not a per-run knob: every run in flight adds
# `--concurrency` desktop containers to the same docker daemon. Pick the host total first,
# then divide. 96 containers is what a 208-vCPU / 1.8 TB host carries with ~25% idle left;
# with NGPU runs in flight that is `--concurrency $((96 / NGPU))`. Raising the per-run number
# without lowering the number of runs is what oversubscribes the host.
NGPU=8   # cards this host will use -- eighteen runs no longer fit one per card
CONC=12  # 8 x 12 = 96 containers
gpu=0
score() {
  # Drain the batch before wrapping back to card 0, or two rollouts land on one GPU and
  # both OOM. `wait` blocks on every score backgrounded so far, which is the batch.
  if [ "$gpu" -ge "$NGPU" ]; then wait; gpu=0; fi
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.5-4B ${2:+--model-path "$2"} \
    --env-id lite.osworld --splits eval --concurrency "$CONC" \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --config-path "$CFG/$1.yaml" \
    --log-root "$LOGS/$3" < /dev/null &
  gpu=$((gpu + 1))
}

# 3 BASE runs -- empty $2 drops --model-path, so rollout serves --model-id's own weights.
# One per screenshot profile, not one total: a checkpoint is only comparable to a baseline
# that saw the same screenshots. The .reasoning cells reuse their profile's base run too.
for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do
  score "$P" "" "base.$P@$RUN"
done

# 15 CHECKPOINT runs -- $2 is the pulled epoch dir, so that is what gets served.
while read -r P T; do
  score "$P" "$PULL/sft.$P.$DS.$T@$RUN/$EPOCH" "sft.$P.$DS.$T@$RUN.$EPOCH"
done <<< "$(cells)"

wait
```

`--model-id` picks the adapter and action space; the weights, tokenizer and chat template all
come from `--model-path`.

Eighteen runs over `NGPU` cards, in batches: `score` drains with `wait` before reusing card 0 —
without it `$gpu` keeps counting past the last card onto ordinals that do not exist. Set `NGPU` to
what the host has FREE; `NGPU=1` serializes all eighteen, slow but correct. Both drains are bare
`wait`s, so anything else left backgrounded in this shell delays them.

Score each run from `<log-root>/summary.json` -> `stats.mean_episode_return` (denominator
`num_valid`). Which cells compare to which is settled at the top of this file. `base` runs
thinking OFF, so it is the wrong baseline for a reasoning checkpoint — give that arm its own base
run under the `.reasoning` config if you want its did-SFT-help number.

#### Record the scores

Collect all eighteen — nineteen with the RL run — and commit as
`devs/exps/train/desktop/logs/$RUN.md`: one file per campaign, edited as runs land, not a wrap-up
from memory.

```bash
# --- EVAL HOST, same shell as the Eval block ---
# Reuses $RUN, $DS, $EPOCH, $LOGS and cells(). In a fresh shell set all five again -- $RUN needs
# an ASSIGNMENT (`RUN=<the same slug>`): the Eval block's line is `: "${RUN:?...}"`, an assertion,
# so re-pasting THAT leaves RUN empty and `show` prints MISSING for every slug.
# The n it prints is a result too, not just bookkeeping -- see "Reading the table" in Results.
show() {  # $1 = log slug
  uv run python -c "
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'summary.json'
if not p.exists(): print(f'{sys.argv[2]:78s} MISSING'); raise SystemExit
d = json.loads(p.read_text()); s = d['stats']
# summary.json has no solved count: derive it. 'fully-solved' is episode_return == 1.0.
solved = sum(r == 1.0 for t in d['tasks'] for r in t['episode_returns'])
pf = (s.get('stop_reasons') or {}).get('parse_failure', 0)
print(f\"{sys.argv[2]:70s} {s['mean_episode_return']:.4f} ({solved}/{s['num_valid']})\"
      f\"  err={s['num_samples'] - s['num_valid']} parse_fail={pf}\")
" "$LOGS/$1" "$1"
}
for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do
  show "base.$P@$RUN"
done
while read -r P T; do show "sft.$P.$DS.$T@$RUN.$EPOCH"; done <<< "$(cells)"
# The RL run, if it was scored (see the RL section): 19th line, not part of cells().
show "grpo.lowr.i4@$RUN"
```

Paste the numbers into a snapshot file, reusing the two tables' LAYOUT from **Results** below
(column meanings — `±`, `n/3`, `(solved/num_valid)`, `MER (solved) parse_failure` — are defined
there; the pass labels and the retired column are this campaign's, not a template). Front matter:

```markdown
# desktop.use @ <run>

- **Checkpoints**: each cell repo's `main` as of `<date>`, tag `<sha>`. Ship can SKIP a cell
  and re-run it later, so the fifteen repos need not share a sha — if they diverge, list the odd
  ones out here (`uv run hf repos tag list <repo>`), or a later `--revision` re-run pulls the
  wrong weights.
- **Dataset**: `scalecua_5k` (`--sample 5000 --seed 42`, `episode_return > 0.5`)
- **Eval**: `lite.osworld` eval, 328/369 after the `exclude_reason` filter, `epoch_2`
- **Host / GPUs**: `<host>` / `0-7`
- **Last updated**: `<date>`

## Results

<the two tables from the Results section of this file>
```

### Results

Mean episode return, (fully-solved / `num_valid`) in parentheses. Campaign `20260912b/c/d`: every TRAINED
cell scored three times, interleaved (all cells once, then all again) so host drift spreads across
cells instead of landing on one. `±` is half the range of the three.

| | `lowr.i4` | `lowr.i1` | `highr.i1` | `highr.h1` (retired) |
|---|---:|---:|---:|---:|
| **base** | 0.2597 ±0.0091 (82/328) 3/3 | 0.1592 ±0.0031 (49/328) 3/3 | 0.2329 ±0.0090 (74/328) 3/3 | 0.2070 (65/328) |
| **`gpt5_5`** | 0.3673 ±0.0154 (115/328) 3/3 | 0.3420 ±0.0201 (107/328) 3/3 | 0.3883 ±0.0120 (122/328) 3/3 | 0.3836 (118/320) |
| **`gpt5_5` + `<think>`** | 0.4165 ±0.0050 (132/328) 3/3 | 0.3828 ±0.0127 (121/328) 3/3 | **0.4665** ±0.0011 (147/328) 3/3 | — |
| **`qwen3_5_27b`** | 0.3421 ±0.0031 (108/328) 3/3 | 0.3209 ±0.0071 (101/328) 3/3 | 0.3582 ±0.0150 (113/328) 3/3 | — |
| **`qwen3_5_27b` + `<think>`** | 0.3349 ±0.0207 (106/328) 3/3 | 0.3574 ±0.0096 (112/328) 3/3 | 0.3649 ±0.0046 (116/328) 3/3 | — |
| **`qwen3_8_27b`** | 0.4090 ±0.0095 (130/328) 3/3 | 0.4019 ±0.0092 (128/328) 3/3 | 0.4069 ±0.0122 (129/328) 3/3 | 0.4318 (131/318) |
| **GRPO from base** | not run | — | — | — |

Every individual run — `MER (solved) parse_failure`, one column per pass:

| | | `b` | `c` | `d` |
|---|---|---:|---:|---:|
| **base** | `lowr.i4` | 0.2618 (83) 0 | 0.2496 (79) 0 | 0.2679 (85) 0 |
|  | `lowr.i1` | 0.1612 (50) 0 | 0.1551 (48) 0 | 0.1612 (50) 0 |
|  | `highr.i1` | 0.2256 (72) 0 | 0.2435 (77) 0 | 0.2296 (73) 0 |
| **`gpt5_5`** | `lowr.i4` | 0.3520 (110) 2 | 0.3672 (116) 5 | 0.3828 (120) 1 |
|  | `lowr.i1` | 0.3369 (106) 3 | 0.3645 (115) 0 | 0.3244 (101) 0 |
|  | `highr.i1` | 0.3742 (118) 3 | 0.3923 (123) 0 | 0.3983 (126) 0 |
| **`gpt5_5`+`<think>`** | `lowr.i4` | 0.4187 (133) 6 | 0.4203 (133) 4 | 0.4103 (129) 5 |
|  | `lowr.i1` | 0.3938 (124) 4 | 0.3683 (117) 3 | 0.3864 (123) 3 |
|  | `highr.i1` | 0.4674 (146) 4 | 0.4670 (149) 4 | 0.4652 (147) 5 |
| **`qwen3_5_27b`** | `lowr.i4` | 0.3402 (108) 0 | 0.3461 (109) 0 | 0.3399 (107) 0 |
|  | `lowr.i1` | 0.3150 (99) 0 | 0.3292 (103) 0 | 0.3184 (101) 0 |
|  | `highr.i1` | 0.3747 (118) 0 | 0.3552 (112) 0 | 0.3447 (109) 0 |
| **`qwen3_5_27b`+`<think>`** | `lowr.i4` | 0.3365 (106) 0 | 0.3133 (99) 2 | 0.3547 (112) 1 |
|  | `lowr.i1` | 0.3676 (115) 1 | 0.3562 (111) 1 | 0.3484 (109) 0 |
|  | `highr.i1` | 0.3616 (116) 2 | 0.3708 (118) 1 | 0.3622 (115) 0 |
| **`qwen3_8_27b`** | `lowr.i4` | 0.4175 (132) 7 | 0.3985 (127) 5 | 0.4110 (131) 5 |
|  | `lowr.i1` | 0.4142 (133) 8 | 0.3959 (126) 3 | 0.3957 (126) 6 |
|  | `highr.i1` | 0.4200 (134) 16 | 0.4050 (129) 3 | 0.3956 (125) 4 |

Reading the table:

- **Compare against the larger of the two cells' own `±`.** Across the eighteen cells the
  half-range spans ±0.0011 to ±0.0207 (median ±0.0094), so one global threshold is either too
  strict or too loose. Six cells sit at ±0.012 or worse: the whole `gpt5_5` row (±0.012-0.020),
  `gpt5_5`+`<think>` x `lowr.i1`, `qwen3_5_27b` x `highr.i1`, and the noisiest in the table,
  `qwen3_5_27b`+`<think>` x `lowr.i4` (±0.0207). It is not a property of any one column — the
  tightest cell is `gpt5_5`+`<think>` x `highr.i1` (±0.0011), and `base` x `lowr.i1` ties
  `qwen3_5_27b` x `lowr.i4` for second at ±0.0031.
- **Only within a column.** `highr.h1` is a retired profile from the 2026-09-10 campaign, scored
  before the env-server was named as a condition; it is kept because it was measured, and its two
  SFT cells carry their own sub-328 denominators (320, 318) — that campaign lost samples to
  errors. (Every `lowr.i4` cell lands within 0.016 of the 2026-09-10 run it replaces — the same
  surface under its old name `lowr.h4`, which renders byte-identical prompts here — inside those
  cells' noise floor.)
- **`mean_episode_return` averages over `num_valid`, not 328.** The `err=` that `show` prints is
  `num_samples - num_valid`: non-zero means that pass came up short and must be RE-RUN, not
  averaged in. Every run in these tables is a full 328. Record the mean WITH its denominator.
- **`parse_failure` does not move the denominator** — a malformed final turn is scored by the env
  like any other episode. It rises with SFT and again with `<think>`; the single 16 was one run,
  not the cell (the next pass was 3).

#### What the profiles actually do

The profile effect is something SFT creates, not a property of the eval:

- **base prefers more images to more pixels** — `lowr.i4` best, `highr.i1` 0.027 lower, 3x its
  noise floor. Both `gpt5_5` rows reverse that; `qwen3_8_27b` does not move either way.
- **SFT flips it, `<think>` widens it** — `lowr.i4` to `highr.i1` gains +0.021 for `gpt5_5`,
  +0.050 with `<think>`.
- **`qwen3_8_27b` is flat** — its three cells span 0.007 under a ±0.009-0.012 floor. The one row
  where the cheap profile costs nothing.
- **Best cell `gpt5_5` + `<think>` x `highr.i1` (0.4665)** leads by 0.050 = 10x noise, but it is
  0.019 BEHIND `qwen3_8_27b` at `lowr.i1`. The win is specific to high resolution.
- **`<think>` pays off for `gpt5_5` everywhere, for `qwen3_5_27b` in one cell.** Against each
  teacher's own Action-only row: `gpt5_5` gains +0.049 / +0.041 / +0.078 (`lowr.i4` / `lowr.i1` /
  `highr.i1`), every one of them past that pair's noise. `qwen3_5_27b` gains +0.037 at `lowr.i1`
  — 3.8x its ±0.0096 — but −0.007 at `lowr.i4` and +0.007 at `highr.i1`, both inside the noise.
  And `lowr.i1` is exactly where its Action-only row was weakest (0.3209), so `<think>` reads as
  lifting a floor there rather than helping across the board.
- **Reading a `<think>` effect**, mind the empty-`<think>` rate: a `.reasoning` cell trains an
  empty one on any terminal turn the teacher ended with prose. `gpt5_5` does that every
  trajectory (11.8% of its 42427 exported steps), `qwen3_5_27b` almost never (0.01%). Compare
  within a teacher, not across.

### RL

One GRPO run, from the **base model** under `desktop.use.lowr.i4.yaml` — the same surface as the
`base` / `lowr.i4` cell above, so that row's 0.2597 is this run's step-0 reference. Trains on
**Lite.ScaleCUA's `rl` split**, scores on **`lite.osworld` eval**. Read
[docs/grpo.md](/docs/grpo.md) first: env-server prerequisite, sync-vs-async, and the knobs this
block does not repeat.

> **Two envs, one run.** Training tasks are `lite.scalecua`, eval tasks are `lite.osworld`. Each
> parquet row carries its own `env_key` (`<env_id>@<task_id>`) and the engine resolves the env per
> row. `ENV_ID` never picks the env — it selects the W&B group, the default config path, which env
> preflight probes and reaps, and the eval dataset LABEL. That last one bites: the eval curve will
> be keyed `lite.scalecua_eval` even though the tasks are `lite.osworld`.
> **The env-server must serve BOTH** — start it with
> `--env-ids lite.scalecua lite.osworld` or the first eval task fails.

<details>
<summary>Data</summary>

```bash
# --- TRAIN HOST ---  (into .data/, which the container sees because launch.sh binds the repo
# root at /workspaces/cua-lite; the only other bind is an optional read-only HF cache.
# docs/grpo.md writes to /root/datasets because it runs its Data step INSIDE the container; that path
# does not survive the boundary, so do not copy it here.)
# Same `exclude_reason` filter on both sides: those tasks cannot be solved (infeasible,
# proxy_required, broken upstream evaluator), so they add zero-reward noise to training and a
# fixed penalty to eval.
#   rl   split: 2049 -> 1809 kept (240 excluded, mostly proxy_required + evaluator bugs)
#   eval split:  369 ->  328 kept (41 excluded, 29 of them literally `infeasible`)
RL=.data/rl/qwen3_5/desktop.use

uv run python -m lite.train.export.export_tasks --env-id lite.scalecua --split rl \
  -o "$RL/scalecua.rl.parquet" \
  --filter "lambda m: not m.others.get('exclude_reason')"

# ONE parquet: the 128-task subset slime reads for the in-training curve. The final 328-task
# score does NOT come from a parquet -- it goes through scripts/rollout.py on the eval host,
# which reads the env registry directly (see "Eval budget" below).
uv run python -m lite.train.export.export_tasks --env-id lite.osworld --split eval --sample 128 \
  -o "$RL/osworld.eval128.parquet" \
  --filter "lambda m: not m.others.get('exclude_reason')"
```

</details>

```bash
# --- Slime container ---
# sync, 8 GPUs colocated, TP=4 (-> DP=2). No HF_CKPT: this starts from BASE weights, which is the
# point -- starting from an SFT checkpoint answers a different question.
W=/workspaces/cua-lite
CELL=grpo.desktop.use.lowr.i4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_TRAIN_GPUS=8 TP_SIZE=4 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  ENV_ID=lite.scalecua \
  PROMPT_DATA="$W/.data/rl/qwen3_5/desktop.use/scalecua.rl.parquet" \
  EVAL_PROMPT_DATA="$W/.data/rl/qwen3_5/desktop.use/osworld.eval128.parquet" \
  ENV_CONCURRENCY=64 \
  ROLLOUT_BATCH_SIZE=16 \
  ROLLOUT_MAX_RESPONSE_LEN=2048 \
  CONFIG_PATH="$W/devs/exps/train/desktop/configs/qwen3_5/desktop.use.lowr.i4.yaml" \
  SAVE=1 SAVE_HF_DIR="$W/.ckpts/qwen3_5-4b/$CELL/iter_{rollout_id}" \
  WANDB_GROUP_SUFFIX=".$CELL" \
  bash "$W/scripts/train/run_grpo.sh"
```

- **`CONFIG_PATH` is mandatory.** Unset, `run_grpo.sh` derives a path that does not exist and
  exits 1 before Ray starts. Pointing it at another `compact/*.yaml` would be SILENT and would
  stop the run being comparable to the `lowr.i4` column.
- **`lowr.i4` is heavier than the `compact` profile RL normally uses** (compact pins
  `history_n: 1` to save VRAM). That cost is the price of scoring on the SFT cells' surface.
- **`TP_SIZE=4`, not SFT's 2 — measured, TP=2 OOMs.** RL is colocated: 8 sglang engines hold
  `--sglang-mem-fraction-static 0.6` ≈ 47.5 GB/card all run, so training gets ~31 GB where SFT
  gets ~79. The backward dies on fp32 vocab-parallel logits (3.91 GiB needed, 1.38 free). TP=4
  halves the vocab shard; it costs DP 4→2 and ~+17% per step. Do **not** instead lower
  `MEM_FRACTION` (trades sglang KV cache for the phase that is already 65% of a step) or touch
  `image_max`/`history_n` (makes the run incomparable). Per-run override only — the 1-2 GPU
  examples in [docs/grpo.md](/docs/grpo.md) pass no `TP_SIZE` and `resolve_tp` hard-fails when it
  does not divide `NUM_TRAIN_GPUS`. `ASYNC=1` is the one lever that would make TP=2 viable again,
  by giving train and rollout their own cards.
- **`ROLLOUT_MAX_RESPONSE_LEN=2048`** — the 512 default is a real ceiling here: measured at step 0
  under `lowr.i4`, mean response 258 tokens, max 710, and **35% of episodes hit the cap at least
  once**. A truncated turn is a reward ceiling RL cannot train past, and it binds harder on the
  `.reasoning` arm. This is a GENERATION budget: it leaves the prompt untouched, so comparability
  holds, and it does not touch the final 328 score. It DOES shift the in-training curve, so keep
  it fixed for the whole run.
- **Eval budget: 128 during training, 328 once at the end.** One eval pass = one rollout step, and
  `EVAL_INTERVAL=5` puts it at 20% of training — **leave it there**: cheap enough, and frequent
  enough to catch a reward collapse early rather than five steps late. (It defaults to 5 like
  `SAVE_INTERVAL`, but the advice is the opposite — raise that one, not this one.) 128 over 64
  because at p≈0.3 the 95% interval is ±0.079 against ±0.112, and ±0.112 is wider than the
  improvement being looked for.
- **The final number must be the full 328**, via `scripts/rollout.py` on the eval host. The Eval
  block will not do it unmodified (its `cells()` lists SFT cells only) and Ship never uploads
  `grpo.*`: move the chosen `iter_*` to the eval host yourself, then add one
  `score desktop.use.lowr.i4 <that dir> grpo.lowr.i4@$RUN` line.
- **Step 0 is the run's own baseline**, unpaired against the table's 0.2597 (different task set),
  so expect ~±0.08. Landing well outside that means the prompt surface drifted.
- Group size is `N_SAMPLES_PER_PROMPT` (default 8): 16 x 8 = 128 trajectories per rollout step,
  which is what `ENV_CONCURRENCY=64` feeds.
- **Export `WANDB_API_KEY`** or the every-5-steps curve this section is built around silently
  does not exist.
- **Set `NUM_ROLLOUT` and raise `SAVE_INTERVAL`.** The default 5 writes a full 4B checkpoint every
  5 steps into the bind-mounted repo on a shared volume. Then pick one `iter_*` to score rather
  than whatever is last.
- Compare against the `base` / `lowr.i4` cell, never an SFT cell — RL-from-base and
  SFT-from-a-teacher answer different questions.
