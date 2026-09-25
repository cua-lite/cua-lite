# desktop.use — teacher x screenshot-profile x reasoning campaign (SFT + one RL run)

Train **fifteen checkpoints** — three screenshot profiles x three teachers, plus a `<think>` arm
on the two teachers that emit reasoning — and score them on one eval. A single GRPO run from the
best `gpt5_5` + `<think>` SFT cell closes the file (**RL** at the end), scored on that same eval.

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
running). RL pins one of those cells explicitly. `$P` is the stem taken whole off the filename and
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

Twenty-one runs on the `lite.osworld` eval split: the rows left after `--filter` drops
`exclude_reason`, **328 of the 369** `catalog.lock.json` pins. The 41 exclusions are
recorded in the tracked catalog lock generated from `eval.jsonl`; count them there
rather than restating denominators by hand. Env
setup: [`lite/gym/envs/lite/osworld/README.md`](/lite/gym/envs/lite/osworld/README.md).

Twenty-one, not fifteen: **the base model runs once per screenshot profile and thinking mode.** A
checkpoint must be scored against a baseline that saw the same prompt surface: the fifteen cells
use three screenshot profiles, each with and without `<think>`.

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
# tokenizer/processor, so all twenty-one runs GENERATE from whatever model that server holds and
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
# background job long after `wait`.
if [ -n "$MISSING" ]; then
  echo "STOP -- do not run the score loop; missing under $EPOCH:$MISSING"
  return 1 2>/dev/null || exit 1
fi
echo "checkpoints OK"

# $1 = config stem, $2 = --model-path ("" = base model), $3 = log slug. --config-path always
# follows $1, so a checkpoint is only ever scored on the surface it was trained on.
# CONCURRENCY IS A BUDGET ON THE HOST, not a per-run knob: every run in flight adds
# `--concurrency` desktop containers to the same docker daemon. Pick the host total first,
# then divide. 96 containers is what a 208-vCPU / 1.8 TB host carries with ~25% idle left;
# with NGPU runs in flight that is `--concurrency $((96 / NGPU))`. Raising the per-run number
# without lowering the number of runs is what oversubscribes the host.
NGPU=8   # cards this host will use -- twenty-one runs no longer fit one per card
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

# 6 BASE runs -- empty $2 drops --model-path, so rollout serves --model-id's own weights.
# One per screenshot profile and thinking mode, not one total: a checkpoint is only comparable
# to a baseline that saw the same screenshots and the same <think> channel.
for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1 \
         desktop.use.lowr.i4.reasoning desktop.use.lowr.i1.reasoning \
         desktop.use.highr.i1.reasoning; do
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

Twenty-one runs over `NGPU` cards, in batches: `score` drains with `wait` before reusing card 0 —
without it `$gpu` keeps counting past the last card onto ordinals that do not exist. Set `NGPU` to
what the host has FREE; `NGPU=1` serializes all twenty-one, slow but correct. Both drains are bare
`wait`s, so anything else left backgrounded in this shell delays them.

Score each run from `<log-root>/summary.json` -> `stats.mean_episode_return` (denominator
`num_valid`). Which cells compare to which is settled at the top of this file. A reasoning
checkpoint's did-SFT-help number reads against `base` + `<think>` under the same `.reasoning`
config, never against the thinking-off `base`: turning `<think>` on moves the base model by up
to +0.076 on its own (Results).

#### Record the scores

Collect all twenty-one — twenty-two with the RL run — and commit as
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
for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1 \
         desktop.use.lowr.i4.reasoning desktop.use.lowr.i1.reasoning \
         desktop.use.highr.i1.reasoning; do
  show "base.$P@$RUN"
done
while read -r P T; do show "sft.$P.$DS.$T@$RUN.$EPOCH"; done <<< "$(cells)"
# The RL run, if it was scored (see the RL section): 19th line, not part of cells().
show "grpo.desktop.use.highr.i1.reasoning.scalecua_5k.gpt5_5.scalecua_rl_osworld128.from_sft@$RUN.<iter>"
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
cells instead of landing on one. `±` is half the range of the three. The `base` + `<think>` row is
campaign `20260924b/c/d`: same eval split, env-server mode and three-pass interleave, scored later
on a 208-vCPU host with three runs in flight at `--concurrency 21` (63 containers).

| | `lowr.i4` | `lowr.i1` | `highr.i1` | `highr.h1` (retired) |
|---|---:|---:|---:|---:|
| **base** | 0.2597 ±0.0091 (82/328) 3/3 | 0.1592 ±0.0031 (49/328) 3/3 | 0.2329 ±0.0090 (74/328) 3/3 | 0.2070 (65/328) |
| **base + `<think>`** | 0.2683 ±0.0040 (85/328) 3/3 | 0.2348 ±0.0175 (74/328) 3/3 | 0.2850 ±0.0122 (89/328) 3/3 | — |
| **`gpt5_5`** | 0.3673 ±0.0154 (115/328) 3/3 | 0.3420 ±0.0201 (107/328) 3/3 | 0.3883 ±0.0120 (122/328) 3/3 | 0.3836 (118/320) |
| **`gpt5_5` + `<think>`** | 0.4165 ±0.0050 (132/328) 3/3 | 0.3828 ±0.0127 (121/328) 3/3 | **0.4665** ±0.0011 (147/328) 3/3 | — |
| **`qwen3_5_27b`** | 0.3421 ±0.0031 (108/328) 3/3 | 0.3209 ±0.0071 (101/328) 3/3 | 0.3582 ±0.0150 (113/328) 3/3 | — |
| **`qwen3_5_27b` + `<think>`** | 0.3349 ±0.0207 (106/328) 3/3 | 0.3574 ±0.0096 (112/328) 3/3 | 0.3649 ±0.0046 (116/328) 3/3 | — |
| **`qwen3_8_27b`** | 0.4090 ±0.0095 (130/328) 3/3 | 0.4019 ±0.0092 (128/328) 3/3 | 0.4069 ±0.0122 (129/328) 3/3 | 0.4318 (131/318) |
| **GRPO from `gpt5_5` + `<think>`** | — | — | TBD | — |

Every individual run — `MER (solved) parse_failure`, one column per pass:

| | | `b` | `c` | `d` |
|---|---|---:|---:|---:|
| **base** | `lowr.i4` | 0.2618 (83) 0 | 0.2496 (79) 0 | 0.2679 (85) 0 |
|  | `lowr.i1` | 0.1612 (50) 0 | 0.1551 (48) 0 | 0.1612 (50) 0 |
|  | `highr.i1` | 0.2256 (72) 0 | 0.2435 (77) 0 | 0.2296 (73) 0 |
| **base+`<think>`** | `lowr.i4` | 0.2649 (82) 4 | 0.2728 (88) 5 | 0.2671 (84) 3 |
|  | `lowr.i1` | 0.2481 (78) 10 | 0.2432 (76) 2 | 0.2130 (67) 4 |
|  | `highr.i1` | 0.2769 (86) 3 | 0.3012 (95) 1 | 0.2768 (87) 1 |
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
| **GRPO from `gpt5_5`+`<think>`** | `highr.i1` | TBD | TBD | TBD |

Reading the table:

- **Compare against the larger of the two cells' own `±`.** Across the eighteen `20260912` cells the
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

The profile effect is something the prompt surface creates, not a property of the eval:

- **base prefers more images to more pixels** — `lowr.i4` best, `highr.i1` 0.027 lower, 3x its
  noise floor. Both `gpt5_5` rows reverse that; `qwen3_8_27b` does not move either way.
- **`<think>` alone already moves base, most where it sees one image.** Against the thinking-off
  `base` row: +0.009 at `lowr.i4` (inside its ±0.009), +0.076 at `lowr.i1` (4.3x its ±0.0175),
  +0.052 at `highr.i1` (4.3x its ±0.0122). With `<think>` on, base already ranks `highr.i1` first
  (0.2850 vs 0.2683 at `lowr.i4`, 1.4x the larger `±`), so part of the SFT flip below is present
  before any training.
- **Read a reasoning cell's SFT gain against `base` + `<think>`.** For `gpt5_5` + `<think>` that
  is +0.148 / +0.148 / +0.182 (`lowr.i4` / `lowr.i1` / `highr.i1`); against the thinking-off
  `base` the same cells would read +0.157 / +0.224 / +0.234, which credits SFT with what enabling
  `<think>` did on its own.
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

One GRPO run from the local **`gpt5_5` + `<think>` SFT checkpoint** trained with
[`desktop.use.highr.i1.reasoning.yaml`](/devs/exps/train/desktop/configs/qwen3_5/desktop.use.highr.i1.reasoning.yaml).
This is the best measured `gpt5_5` reasoning cell in the table above, so that row's `highr.i1`
score is this run's step-0 reference. Trains on **Lite.ScaleCUA's `rl` split**, scores on
**`lite.osworld` eval**. Read [docs/grpo.md](/docs/grpo.md) first: env-server prerequisite,
sync-vs-async, and the knobs this block does not repeat.

> **Two envs, one run.** Training tasks are `lite.scalecua`, eval tasks are `lite.osworld`. Each
> parquet row carries its own `env_key` (`<env_id>@<task_id>`) and the engine resolves the env per
> row. `ENV_ID` never picks the env — it selects the W&B group, the default config path, which env
> preflight probes and reaps, and the eval dataset LABEL. That last one bites: the eval curve will
> be keyed `lite.scalecua_eval` even though the tasks are `lite.osworld`.
> **The env-server must serve BOTH** — start it with
> `--env-ids lite.scalecua lite.osworld` or the first eval task fails.

**Which training tasks.** The Data block below builds the whole `rl` split minus
`exclude_reason`. That is the fallback, not the intended input:
[`TASKS.md`](/devs/exps/train/desktop/TASKS.md) calibrates a candidate pool at `g=4` and keeps only
the tasks whose group still carries a gradient, which is 48.4% of an uncalibrated pool — the rest
spends rollout budget computing advantages that are identically zero. Its
[`utils/tasks.py`](/devs/exps/train/desktop/utils/tasks.py) writes the manifest; point `PROMPT_DATA`
at that instead. Calibrated manifests are tracked under
[`data/`](/devs/exps/train/desktop/data); the candidate pools they are built from are not (see
`.gitignore` — they rebuild in seconds, the calibration does not).

<details>
<summary>Data</summary>

```bash
# --- ONE-TIME DATA BUILD; skip generation for any file that already exists ---
# These parquet files are fixed experiment manifests under the repo, so Slime sees them at
# /workspaces/cua-lite/devs/exps/train/desktop/data.
# Same `exclude_reason` filter on both sides: those tasks cannot be solved (infeasible,
# proxy_required, broken upstream evaluator), so they add zero-reward noise to training and a
# fixed penalty to eval.
#   rl   split: 2049 -> 1809 kept (240 excluded, mostly proxy_required + evaluator bugs)
#   eval split:  369 ->  328 kept (41 excluded, 29 of them literally `infeasible`)
DATA=devs/exps/train/desktop/data
TRAIN="$DATA/scalecua.rl.no_exclude.parquet"
EVAL="$DATA/osworld.eval128.no_exclude.seed42.parquet"
mkdir -p "$DATA"

if [ -e "$TRAIN" ]; then
  echo "keep existing fixed train manifest: $TRAIN"
else
  uv run python -m lite.train.export.export_tasks --env-id lite.scalecua --split rl \
    -o "$TRAIN" \
    --filter "lambda m: not m.others.get('exclude_reason')"
fi

# ONE parquet: the 128-task subset slime reads for the in-training curve. The final 328-task
# score does NOT come from a parquet -- it goes through scripts/rollout.py on the eval host,
# which reads the env registry directly (see "Eval budget" below).
if [ -e "$EVAL" ]; then
  echo "keep existing fixed eval manifest: $EVAL"
else
  uv run python -m lite.train.export.export_tasks --env-id lite.osworld --split eval --sample 128 \
    --seed 42 \
    -o "$EVAL" \
    --filter "lambda m: not m.others.get('exclude_reason')"
fi

uv run python - "$TRAIN" "$EVAL" <<'PY'
import hashlib
import sys

import pandas as pd

import lite.gym as gym
from lite.data.staging import coerce_meta

train = pd.read_parquet(sys.argv[1])
eval_ = pd.read_parquet(sys.argv[2])
train_keys = [coerce_meta(row["metadata"])["env_key"] for _, row in train.iterrows()]
eval_keys = [coerce_meta(row["metadata"])["env_key"] for _, row in eval_.iterrows()]
train_hash = hashlib.sha256("\n".join(train_keys).encode()).hexdigest()
eval_hash = hashlib.sha256("\n".join(eval_keys).encode()).hexdigest()

assert len(train) == 1809
assert len(eval_) == 128
assert all(key.startswith("lite.scalecua@") for key in train_keys)
assert all(key.startswith("lite.osworld@") for key in eval_keys)
assert all(
    not gym.registry.task_metadata("lite.scalecua", key.split("@", 1)[1]).others.get("exclude_reason")
    for key in train_keys
)
assert all(
    not gym.registry.task_metadata("lite.osworld", key.split("@", 1)[1]).others.get("exclude_reason")
    for key in eval_keys
)
print(
    "desktop RL data ok: 1809 Lite.ScaleCUA train tasks, 128 OSWorld eval tasks, "
    f"train_sha256={train_hash} eval_sha256={eval_hash}"
)
PY
```

</details>

```bash
# --- Slime container ---
# sync, 8 GPUs colocated, TP=4 (-> DP=2). Start from the gpt5_5 + <think> SFT checkpoint for the
# same desktop.use.highr.i1.reasoning surface.
# train == eval == the same 128 lite.osworld tasks. Deliberate: it asks whether this recipe can
# move the eval number AT ALL. Measured 0.3604 -> 0.5223 over one epoch, which at nspr=4 is 16
# optimizer steps, not 40. Read that against the noise bar below before concluding anything from
# it: +16pp is about three times a single eval's spread, a transfer run's +8.6pp was one.
# For the transfer run, point PROMPT_DATA at a calibrated manifest from TASKS.md and set
# ENV_ID=lite.scalecua; leave everything else alone, above all the eval set.
W=/workspaces/cua-lite
P=desktop.use.highr.i1.reasoning
DS=scalecua_5k
T=gpt5_5
RLDS=osworld128_overfit
CELL=grpo.$P.$DS.$T.$RLDS.from_sft
CKPT=${CKPT:-$(ls -d "$W/.ckpts/qwen3_5-4b/sft.$P.$DS.$T"/iter_* 2>/dev/null | sort -V | tail -1)}

: "${CUA_LITE_ENV_SERVER_URL:?paste export line from env-server shell}"
: "${CUA_LITE_ENV_SERVER_TOKEN:?paste export line from env-server shell}"
[ -d "$CKPT" ] || { echo "MISSING CKPT=$CKPT"; exit 1; }
for f in config.json tokenizer_config.json preprocessor_config.json; do
  [ -e "$CKPT/$f" ] || { echo "MISSING $CKPT/$f"; exit 1; }
done

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_TRAIN_GPUS=8 TP_SIZE=4 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  HF_CKPT="$CKPT" \
  CUA_LITE_MULTIMODAL_LAZY_EXPAND=1 \
  ENV_ID=lite.osworld \
  PROMPT_DATA="$W/devs/exps/train/desktop/data/osworld.eval128.no_exclude.seed42.parquet" \
  EVAL_PROMPT_DATA="$W/devs/exps/train/desktop/data/osworld.eval128.no_exclude.seed42.parquet" \
  ENV_CONCURRENCY=96 \
  ROLLOUT_BATCH_SIZE=32 \
  N_SAMPLES_PER_PROMPT=8 \
  NUM_STEPS_PER_ROLLOUT=4 \
  ROLLOUT_MAX_RESPONSE_LEN=2048 \
  LR=3e-6 \
  CONFIG_PATH="$W/devs/exps/train/desktop/configs/qwen3_5/$P.yaml" \
  EVAL_TEMPERATURE=1 N_SAMPLES_PER_EVAL_PROMPT=1 \
  SAVE=1 NO_SAVE_OPTIM=1 SAVE_INTERVAL=4 EVAL_INTERVAL=4 NUM_ROLLOUT=20 \
  SAVE_HF_DIR="$W/.ckpts/qwen3_5-4b/$CELL/iter_{rollout_id}" \
  SAVE_DIR="/root/checkpoints/qwen3_5-4b/$CELL/megatron" \
  WANDB_GROUP_SUFFIX=".$CELL" \
  bash "$W/scripts/train/run_grpo.sh"
```

- **`CONFIG_PATH` is mandatory.** Unset, `run_grpo.sh` derives a path that does not exist and
  exits 1 before Ray starts. Pointing it at another `compact/*.yaml` would be SILENT and would
  stop the run being comparable to the `gpt5_5` + `<think>` / `highr.i1` parent.
- **`HF_CKPT` is mandatory for this RL run.** It must point at the local
  `sft.desktop.use.highr.i1.reasoning.scalecua_5k.gpt5_5/iter_*` HF export; omitting it silently
  starts from base Qwen3.5-4B and answers a different question.
- **`desktop.use.highr.i1.reasoning` is heavier than the `compact` profile RL normally uses**
  (compact pins `history_n: 1` to save VRAM). That cost is the price of matching the chosen SFT
  parent exactly.
- **`TP_SIZE=4`, not SFT's 2, is the conservative colocated-RL default.** The measured
  `lowr.i4` RL run OOMed at TP=2 because sglang engines hold memory for the whole run; keep TP=4
  until this `highr.i1.reasoning` run has its own memory trace. Do **not** change
  `image_max`/`history_n` to save memory, because that makes the run incomparable to its SFT
  parent. `ASYNC=1` is the one structural lever that would make lower TP more plausible.
- **`ROLLOUT_MAX_RESPONSE_LEN=2048`** — the 512 default is a real ceiling on desktop RL, and the
  `.reasoning` arm can only make the tail longer. This is a GENERATION budget: it leaves the prompt
  untouched, so comparability holds. It DOES shift the in-training curve, so keep it fixed for the
  whole run.
- **Eval budget: 128 during training, 328 once at the end.** An eval pass costs about half a
  rollout (measured: 11 min at 1 sample/task, 19.8 at 2), and
  `EVAL_INTERVAL=5` puts it at 20% of training — **leave it there**: cheap enough, and frequent
  enough to catch a reward collapse early rather than five steps late. (It defaults to 5 like
  `SAVE_INTERVAL`, but the advice is the opposite — raise that one, not this one.) 128 over 64
  because at p≈0.3 the 95% interval is ±0.079 against ±0.112, and ±0.112 is wider than the
  improvement being looked for.
- **The final number must be the full 328**, via `scripts/rollout.py` on the eval host. The Eval
  block will not do it unmodified (its `cells()` lists SFT cells only) and Ship never uploads
  `grpo.*`: move the chosen `iter_*` to the eval host yourself, then add one
  `score desktop.use.highr.i1.reasoning "$GRPO_CKPT" "grpo.desktop.use.highr.i1.reasoning.scalecua_5k.gpt5_5.scalecua_rl_osworld128.from_sft@$RUN.$(basename "$GRPO_CKPT")"`.
- **Step 0 is the run's own baseline**, paired against the `gpt5_5` + `<think>` / `highr.i1` SFT
  cell up to eval-subset noise. Landing far below that usually means the wrong `P`, `CKPT`, or
  config was used.
- **`32 x 8 = 256` trajectories per rollout, split into 4 optimizer steps** (`256 / nspr 4 = 64`
  GBS -- slime derives it, do NOT also pass `--global-batch-size`). 128 tasks / `RBS` 32 = one
  epoch every 4 rollouts, which is what `EVAL_INTERVAL` and `SAVE_INTERVAL` are set to.

  WARNING: **`nspr` must divide `ROLLOUT_BATCH_SIZE`, not just `RBS x n`**, and `run_grpo.sh` only
  preflights the weaker one. `slime/utils/dp_schedule.py` slices by *trajectory* in task-contiguous
  order, so a task's `n` samples share an optimizer step only when `RBS % nspr == 0` (32 % 4 = 0).
  A split group makes advantages sum non-zero inside each step it lands in. `RBS=12, n=4, nspr=8`
  passes the script's check and still splits.

  `n=8` over `n=4`: these tasks sit low on the p axis (0.36 at t=1), where `1 - p^n - (1-p)^n`
  separates sharply -- 57% vs 34% live groups at p=0.10, 90% vs 68% at p=0.25. Measured here,
  38-66% mixed per rollout. On an easier pool `n=4` wins per *trajectory*, so this is a property of
  the pool, not a default. [`TASKS.md`](/devs/exps/train/desktop/TASKS.md) has the arithmetic.

  Watch `grad_norm`, the entropy slope, and `train_rollout_logprob_abs_diff` (train/inference
  agreement, threshold 0.1) rather than arguing `nspr`: steps 2..4 are off-policy w.r.t. the sampled
  data. Measured on this run, 0.006-0.014 and 1.4-5.3.
- **`CUA_LITE_MULTIMODAL_LAZY_EXPAND=1` is not optional at this batch.** Default 0 expands every
  screenshot into `pixel_values` inside the rollout and ships the float tensors through plasma; at
  `32 x 8` that took the node to 972 GB of Ray's 1024 and a worker was killed mid-training. The lazy
  path carries PNG bytes and rebuilds them on the trainer: same numbers (`logprob_abs_diff`
  unchanged at 0.013), peak 632 GB. `--multimodal-lazy-expand-fn-path` alone does nothing -- it
  registers the hook, this env var gates the payloads.
- **`EVAL_TEMPERATURE=1` scores what GRPO optimises.** `run_grpo.sh` defaults it to 0, which is a
  deterministic, deployment-shaped number; slime's own default is the rollout temperature. On this
  128-task set the same checkpoint scores 0.5012 greedy against a sampled mean of 0.4337 (three
  draws) -- about 7pp, which compounds over a 12-20 turn trajectory, so the two are not
  interchangeable. Pick one per campaign and say which; mixing them across a table makes the rows
  unreadable. A single sampled draw of that checkpoint read 0.3604, which is why the gap was first
  written down as 10.4pp.
- **One eval pass over these 128 tasks has a standard deviation of 5.5pp.** Measured: the SFT
  checkpoint scored 0.3721 / 0.4790 / 0.4499 on three independent draws at t=1 (`--group-size 3
  --group-shared-seed false`, 384 trajectories). Three quarters of the tasks are deterministic
  across draws; the spread comes from the quarter that flip. So a single-draw difference under
  ~8pp says nothing, and differences that size HAVE been read as signal here: a transfer run's
  apparent +8.6pp came back as +0.48pp under a paired 384-trajectory re-measurement of the same
  two checkpoints. Either raise `N_SAMPLES_PER_EVAL_PROMPT`, or re-measure the checkpoints the
  conclusion rests on before reporting a number.
- **Read `return_mean`, not the bare `eval/{ds}` scalar.** That one averages over the DENSE reward
  list, so every errored rollout dilutes it as a 0.0. The engine logs the valid-only mean beside the
  counts: `Eval <ds>: return_mean=..., N valid / M errored / ...`. They agree only while `M == 0`.
- **Export `WANDB_API_KEY`** or the every-5-steps curve this section is built around silently
  does not exist.
- **Set `NUM_ROLLOUT` and raise `SAVE_INTERVAL`.** The default 5 writes a full 4B checkpoint every
  5 steps into the bind-mounted repo on a shared volume. Then pick one `iter_*` to score rather
  than whatever is last.
- Compare against the `gpt5_5` + `<think>` / `highr.i1` SFT cell, not the base row: this run is
  RL-from-SFT, not RL-from-base.

#### LibreOffice family transfer

A second RL experiment, separate from the overfit run above: train on **Lite.ScaleCUA `rl`**
tasks from the three LibreOffice applications, score on the **`lite.osworld` eval** tasks of those
same three. It exists because seven earlier single-domain runs disagreed with each other, and the
unit "one domain" turned out to be too small to read.

**Why the application family and not one domain.** Picking `libreoffice_calc` or
`libreoffice_impress` after the fact is selection: eight domains splitting a null result guarantee
one reads positive. LibreOffice is an a-priori unit — one office suite, shared UI idioms, shared
save/verify semantics — and it covers **115 of the 328** scored eval tasks, so the denominator is
fixed before any number is looked at.

**What the earlier runs measured.** All at `LR=3e-6`, `RBS=32`, `nspr=8`, 4 steps per rollout,
from the same `gpt5_5` + `<think>` / `highr.i1` SFT checkpoint. The delta is against each run's
own step-0 eval:

| training pool | pool size | eval | step | delta |
|---|---:|---|---:|---:|
| impress, full | 116 | impress 47, n=8 | 16 | **−2.24pp** |
| impress, eval-instruction rewrites | 50 | impress 47, n=8 | 16 | **+4.71pp** |
| impress, same, continued | 50 | impress 47, n=8 | 36 | **+7.45pp** |
| calc, eval-instruction rewrites | — | calc 46, n=8 | 20 / 40 | **+1.63pp** / +1.63pp |
| calc, calibrated | 141 | calc 46, n=4 | 16 | **+6.52pp** |
| 4 domains mixed | — | 124, n=8 | 24 / 48 | **+3.97pp** / +1.97pp |
| 10 domains, 1000 tasks | 1000 | 328, n=4 | 12 | **−0.25pp** |

Mean of the four single-domain runs is +2.66pp with a spread of 3.8pp, i.e. **t ≈ 1.4 — not
significant**. No factor separates them: pool size, per-rollout pool coverage (64% positive, 27.6%
negative, 22.7% positive), contamination level and pool provenance were each checked and none
orders the outcomes. Treat a single run's number as uninformative until the repeat spread below
has been measured.

**Result: a clean in-domain corpus moves the eval number by ~+10pp; eval-derived rows cost only
time.** Five runs from the same SFT checkpoint. Two on `lite.scalecua rl`, read on the 115-task
LibreOffice eval (2 sigma = 2.2pp from a baseline measured four times). Three on
`lite.osworld train` — two single-domain arms on impress read at five checkpoints each on the
47-task impress eval (2 sigma = 3.7pp from five baseline measurements 0.4701 / 0.4521 / 0.4775 /
0.4615 / 0.4303; the shared 0.4303 is the contaminated arm's own step-0 pass, and the clean arm
skipped its own because the two start from the same weights), and one two-domain run on
impress + calc read on the 93-task impress + calc eval:

| training corpus | pool | delta |
|---|---|---:|
| `lite.scalecua rl` | 214, environment-level only | +0.39pp at 16 steps |
| `lite.scalecua rl` | 214 + 64 `perturb` | +4.61pp at 16 steps |

| `lite.osworld train` impress, 200 steps | 40 | 80 | 120 | 160 | 200 | mean |
|---|---:|---:|---:|---:|---:|---:|
| `train.synth` 287, **zero** eval provenance | **+11.57** | **+11.18** | +8.92 | +9.47 | +8.24 | **+9.88pp** |
| the same 287 plus 108 `perturb` | +4.80 | +8.55 | +8.97 | +9.14 | +9.78 | +8.25pp |
| gap | +6.77 | +2.63 | −0.06 | +0.33 | −1.53 | |

**The corpus is what decides.** `train.synth` is authored template tasks in the SAME env registry
as the eval split — same starting documents, same verifier family, and verified zero base-task
overlap and zero identical instructions with any eval task. Five readings, none below +8.2pp,
against a 3.7pp threshold; the curve peaks at 40 steps and settles around +9pp rather than
collapsing. `lite.scalecua rl` is a different registry whose rows are labelled environment-level,
and it gives +0.39pp. A clean in-domain corpus is the ingredient — not proximity to the eval tasks.

**Eval-derived rows cost time, not accuracy — compare at matched data, not matched steps.** The two
impress arms look 6.77pp apart at 40 steps, which is where an earlier version of this file stopped
and called the added rows harmful. They are not: `train395` spends 27% of every rollout on its 108
`perturb` rows, so at a given step it has seen far less `synth`. The gap closes monotonically and
is gone by 120 steps (−0.06, +0.33, −1.53pp — all inside the threshold, and the last one has the
contaminated arm ahead). `utils/tasks.py` records SFT on those same rows measuring −6.77pp at 8
updates and −14.49pp at 26; these RL runs reproduce no such damage. What they cost is a quarter of
the rollout budget.

Both impress arms are n=1 on a 47-task eval. The five-point curves, not any single reading, are
what the claim rests on.

**Two domains at once: +12.24pp on 93 tasks.** The single-domain arms are 47 tasks wide, which is
where this campaign has burned itself before. The same recipe on `libreoffice_impress` +
`libreoffice_calc` — pools unioned, nothing else changed, `n=3` per eval task instead of 4 —
doubles the denominator to 93 of the 328 scored eval tasks:

| `lite.osworld train` impress + calc, 200 steps | 0 | 40 | 80 | 120 | 160 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| `train.synth` 565 + `perturb` 214, pass rate | 0.3219 | 0.3374 | 0.3229 | 0.4292 | 0.4191 | **0.4443** |
| delta | — | +1.55 | +0.10 | **+10.73** | +9.72 | **+12.24pp** |

The last three readings sit at 0.42-0.44 and none of them is within reach of the first two, so this
is a level change rather than one lucky checkpoint. **Nothing before 120 steps predicts it**: at 40
and 80 steps the run reads +1.55pp and +0.10pp, i.e. flat, and the first live reading of this run
was reported as two-domain training being worse than single-domain on exactly those two points.
Every curve in this section has the same shape — read a 200-step run at 80 steps and it says
nothing.

**This eval set has no noise floor of its own.** The measured one above is 115 tasks at `n=4`
(threshold ~5pp); 93 tasks at `n=3` is fewer tasks and fewer samples per task, so its threshold is
wider than 5pp, not narrower. +12.24pp and +10.73pp clear it either way; +1.55pp and +0.10pp clear
nothing, which is the point. Measure it before reading a smaller delta off this set.

The step-0 rate is lower here (0.3219 vs 0.4303 on impress alone) because calc is the harder half:
the calc pilots in the table above start from 0.1902. The delta, not the level, is what compares.

**Contamination, per run.** `utils/tasks.py` grades the three corpora and every run above is
labelled by which ones it drew from. Report the label with the number, always:

| corpus | level | overlap with its eval set |
|---|---|---|
| `synth` | none — authored templates | 0 of 47 impress, 0 of 93 impress + calc |
| `scalecua_rl` | environment-level: same starting screens, new goals and verifiers | 99 of 115 LibreOffice |
| `perturb` | task-level: rewrites of the eval instructions | 40 of 47 impress, 77 of 93 impress + calc |

The one clean-corpus run (`synth` 287, zero overlap) is also the one with the largest mean delta,
which is why the claim is about the corpus and not about proximity. The two runs that include
`perturb` are labelled in every table above; the +12.24pp two-domain run is one of them, and its
clean-arm control has not been run.

<details>
<summary>Data</summary>

Both manifests are pure functions of tracked files, so two builds on two clusters are
byte-identical — verified. The train pool reads the **committed calibration sidecar**
(`*.calibration.parquet`, one `bucket` per task from a `g=4` pass), not a rollout log, so
rebuilding it needs no GPU. Regenerating that sidecar from scratch is the `build-candidates` →
`scripts/rollout.py` → `calibrate` path in [`TASKS.md`](/devs/exps/train/desktop/TASKS.md); it is
stochastic, so the committed sidecar is the pinned artifact, not the recipe.

```bash
# --- ONE-TIME DATA BUILD; skip generation for any file that already exists ---
DATA=devs/exps/train/desktop/data
TRAIN="$DATA/grpo.scalecua_rl.libreoffice.g4.n214.usable.parquet"
EVAL="$DATA/osworld.eval115.libreoffice.parquet"
ARMB="$DATA/grpo.libreoffice.armB.scalecua214_perturb64.parquet"

if [ -e "$TRAIN" ] && [ -e "$EVAL" ] && [ -e "$ARMB" ]; then
  echo "keep existing fixed manifests: $TRAIN $EVAL $ARMB"
else
  uv run python - "$DATA" <<'PY'
import sys
sys.path.insert(0, "devs/exps/train/desktop")

import pandas as pd

from lite.utils.parquet import write_records_to_parquet
from utils.tasks import OSW, SCA, _read_jsonl, domain_of, eval_rows, pool_of

D = sys.argv[1]
LO = {"libreoffice_calc", "libreoffice_impress", "libreoffice_writer"}
# The only two buckets that carry a gradient; the rest give identically-zero advantage.
USABLE = {"mixed_success", "all_fail_with_reward_variance"}
row = lambda env, t, split: {"problem": f"Complete the task: {t}",
                             "metadata": {"env_key": f"{env}@{t}", "split": split}}

cal = pd.read_parquet(f"{D}/desktop.combined3.calibrated.g4.sample2110.seed42.calibration.parquet")
bucket = dict(zip(cal["task_id"], cal["bucket"]))
rl = {r["task_id"]: r for r in _read_jsonl(SCA / "rl.jsonl")}
# sorted() is load-bearing: it is what makes two independent builds byte-identical.
train = sorted(t for t, b in bucket.items()
               if b in USABLE and pool_of(t) == "scalecua_rl"
               and rl.get(t) is not None and domain_of(rl[t]) in LO)
ev = sorted(r["task_id"] for r in eval_rows(scored_only=True) if domain_of(r) in LO)
write_records_to_parquet([row("lite.scalecua", t, "rl") for t in train],
                         f"{D}/grpo.scalecua_rl.libreoffice.g4.n{len(train)}.usable.parquet")
write_records_to_parquet([row("lite.osworld", t, "eval") for t in ev],
                         f"{D}/osworld.eval{len(ev)}.libreoffice.parquet")
# Arm B adds the task-level-contaminated rows back and changes nothing else. It exists to be
# compared against the pool above; never report a number trained on it without that label.
pt = {r["task_id"]: r for r in _read_jsonl(OSW / "train.perturb.jsonl")}
pb = sorted(t for t, b in bucket.items()
            if b in USABLE and pool_of(t) == "perturb"
            and pt.get(t) is not None and domain_of(pt[t]) in LO)
write_records_to_parquet([row("lite.scalecua", t, "rl") for t in train]
                         + [row("lite.osworld", t, "train.perturb") for t in pb],
                         f"{D}/grpo.libreoffice.armB.scalecua{len(train)}_perturb{len(pb)}.parquet")
print(f"wrote {len(train)} train, {len(ev)} eval, arm-B pool {len(train) + len(pb)}")
PY
fi

uv run python - "$TRAIN" "$EVAL" <<'PY'
import hashlib
import sys

import pandas as pd

import lite.gym as gym
from lite.data.staging import coerce_meta

LO = {"libreoffice_calc", "libreoffice_impress", "libreoffice_writer"}
keys = lambda p: [coerce_meta(r["metadata"])["env_key"] for _, r in pd.read_parquet(p).iterrows()]
train_keys, eval_keys = keys(sys.argv[1]), keys(sys.argv[2])

assert len(train_keys) == 214 and len(eval_keys) == 115
assert all(k.startswith("lite.scalecua@") for k in train_keys)
assert all(k.startswith("lite.osworld@") for k in eval_keys)
# No task-level-contaminated corpus in the training pool.
assert all(k.split("@", 1)[1].startswith("scalecua_") for k in train_keys)
for env, ks in (("lite.scalecua", train_keys), ("lite.osworld", eval_keys)):
    for k in ks:
        others = gym.registry.task_metadata(env, k.split("@", 1)[1]).others
        assert others.get("domain") in LO, k
        assert not others.get("exclude_reason"), k
sha = lambda ks: hashlib.sha256("\n".join(ks).encode()).hexdigest()
print("LibreOffice RL data ok: 214 Lite.ScaleCUA train tasks (calc 93 / impress 55 / writer 66), "
      "115 OSWorld eval tasks (calc 46 / impress 47 / writer 22), "
      f"train_sha256={sha(train_keys)} eval_sha256={sha(eval_keys)}")
PY
```

The committed manifests are pinned at

    train  214 tasks, 3 domains, calc 93 / impress 55 / writer 66
           env_key_sha256 5a076dfccde4808c16849bac9a4495709b3344c10bdf6942f9c9ae1bdbcdbef1
    eval   115 tasks, 3 domains, calc 46 / impress 47 / writer 22
           env_key_sha256 08bd93883004b4e747fbf04eeda73d47708b572bbde4c5c1c26144069ddfc66a

</details>

**Measure the eval's own spread before reading any delta.** This campaign has already published
one apparent +8.6pp that came back as +0.48pp under re-measurement, and the three noise figures on
record disagree in the wrong direction (128 tasks at n=1 gave sd 5.5pp; 47 tasks at n=4 gave a
±0.77pp half-range; 47 tasks at n=8, i.e. MORE trajectories, gave two draws 1.8pp apart). None of
them is this eval set. Score the step-0 checkpoint on the 115 twice, independently, before running anything. **Measured
on this eval set** (same SFT checkpoint, `n=4`, two draws):

    p1  0.3594 (stderr 0.0223)      p2  0.3386 (stderr 0.0220)      whole-set difference  -2.08pp
    paired over 115 tasks: mean -2.17pp, sd 26.61pp, sem 2.48pp  ->  threshold ~5pp at n=4
    57.4% of tasks score identically across the two draws; all of the spread is the other 42.6%
    per domain: calc +1.35pp, impress -1.60pp, writer -10.28pp (22 tasks cannot be read alone)

That killed the `n=4` design outright: the four pilots average +2.66pp per domain, which spread
over the family is about +4.5pp — **below its own 5pp threshold**. The run block below therefore
uses `N_SAMPLES_PER_EVAL_PROMPT=8`, which halves the per-task rate variance (sd 26.61 -> ~18.8pp,
sem -> ~1.75pp, threshold ~3.5pp), and `EVAL_INTERVAL=4` to keep the cost down by dropping the
step-8 reading — the one point where every pilot still showed nothing. Budget moves from 5.2h to
6.6h: 4 x (20 min sampling + 25 min training) + 2 x ~108 min eval.

Re-measure this if the eval set, the checkpoint, or `ENV_CONCURRENCY` changes; the number is a
property of all three, not of the task count. The command:

```bash
W=/workspaces/cua-lite
P=desktop.use.highr.i1.reasoning
CKPT=$W/.ckpts/pulled/sft.highr.i1.reasoning.gpt5_5/epoch_2
for R in p1 p2; do
  uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.5-4B --model-path "$CKPT" \
    --env-id lite.osworld \
    --prompt-data "$W/devs/exps/train/desktop/data/osworld.eval115.libreoffice.parquet" \
    --group-size 4 --group-shared-seed false \
    --sampling-kwargs '{"temperature": 1.0}' \
    --concurrency 24 --max-attempts 1 --save-video false --save-gif false \
    --config-path "$W/devs/exps/train/desktop/configs/qwen3_5/$P.yaml" \
    --log-root "$W/.logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/lo115.base.$R"
done
```

Only `temperature` is overridden: the config yaml pins it to `0.0` for greedy rollout, and this
measurement has to match the sampled eval it is the noise floor for. Everything else stays at
`DEFAULT_SAMPLING_KWARGS` (`top_p` 1.0, `max_new_tokens` 2048) — `extract_sampling_kwargs`
merges default < yaml < CLI rather than replacing, so omitting a key keeps the default.

```bash
# --- Slime container ---
# Train on LibreOffice Lite.ScaleCUA rl tasks, score on the LibreOffice lite.osworld eval tasks.
# Every knob below is what the four single-domain pilots ran; only the two manifests change, so a
# result here is comparable to that table rather than a new point in a new space.
W=/workspaces/cua-lite
P=desktop.use.highr.i1.reasoning
CELL=grpo.$P.libreoffice.n214.from_sft
CKPT=$W/.ckpts/pulled/sft.highr.i1.reasoning.gpt5_5/epoch_2

: "${CUA_LITE_ENV_SERVER_URL:?paste export line from env-server shell}"
: "${CUA_LITE_ENV_SERVER_TOKEN:?paste export line from env-server shell}"
[ -d "$CKPT" ] || { echo "MISSING CKPT=$CKPT"; exit 1; }

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_TRAIN_GPUS=8 TP_SIZE=4 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  HF_CKPT="$CKPT" \
  CUA_LITE_MULTIMODAL_LAZY_EXPAND=1 \
  ENV_ID=lite.osworld \
  PROMPT_DATA="$W/devs/exps/train/desktop/data/grpo.scalecua_rl.libreoffice.g4.n214.usable.parquet" \
  EVAL_PROMPT_DATA="$W/devs/exps/train/desktop/data/osworld.eval115.libreoffice.parquet" \
  CONFIG_PATH="$W/devs/exps/train/desktop/configs/qwen3_5/$P.yaml" \
  ENV_CONCURRENCY=24 \
  ROLLOUT_BATCH_SIZE=32 N_SAMPLES_PER_PROMPT=8 NUM_STEPS_PER_ROLLOUT=4 \
  ROLLOUT_TEMPERATURE=1.0 ROLLOUT_MAX_RESPONSE_LEN=2048 \
  LR=3e-6 \
  EVAL_TEMPERATURE=1 N_SAMPLES_PER_EVAL_PROMPT=8 \
  EVAL_INTERVAL=4 SKIP_EVAL_BEFORE_TRAIN=0 \
  SAVE=1 NO_SAVE_OPTIM=1 SAVE_INTERVAL=2 NUM_ROLLOUT=4 \
  SAVE_HF_DIR="$W/.ckpts/qwen3_5-4b/$CELL/iter_{rollout_id}" \
  SAVE_DIR="/root/checkpoints/qwen3_5-4b/$CELL/megatron" \
  WANDB_GROUP_SUFFIX=".$CELL" \
  bash "$W/scripts/train/run_grpo.sh"
```

- **The env-server must serve both envs** (`--env-ids lite.scalecua lite.osworld`): training rows
  carry `lite.scalecua@...`, eval rows carry `lite.osworld@...`, and the engine resolves per row.
- **`ENV_ID=lite.osworld` even though training is `lite.scalecua`, and this contradicts the
  overfit section above on purpose.** That one says to set `ENV_ID=lite.scalecua` for a transfer
  run; it is buying preflight coverage of the training env and paying for it with an eval curve
  keyed `lite.scalecua_eval` (its own blockquote flags the confusion). This section makes the
  opposite trade: the seven runs in the table above are all keyed `eval/lite.osworld_eval`, and a
  result that cannot be grepped alongside them is worth less than the preflight.

  **Neither setting covers both envs, so probe the other one by hand.** `ENV_ID` picks the W&B
  group, the eval dataset label, and the two env-scoped preflight steps — `GET /envs/${ENV_ID}`
  (availability) and `DELETE /instances?session_id=&env_id=${ENV_ID}` (prior-session leftovers).
  `GET /host_status` is env-independent and still covers the server. With `lite.osworld` set, the
  env that serves all 1024 TRAINING trajectories is neither probed nor drained, and a missing or
  dirty `lite.scalecua` surfaces at the first training rollout instead of before Ray starts.
  Run this first — `SESSION_ID` is the value preflight prints on its own cleanup line:

  ```bash
  curl -sf -H "Authorization: Bearer $CUA_LITE_ENV_SERVER_TOKEN" \
    "$CUA_LITE_ENV_SERVER_URL/envs/lite.scalecua" | grep -q '"available"[[:space:]]*:[[:space:]]*true' \
    && echo "lite.scalecua available" || { echo "lite.scalecua NOT available -- fix before launching"; }
  curl -sf -X DELETE -G -H "Authorization: Bearer $CUA_LITE_ENV_SERVER_TOKEN" \
    --data-urlencode "session_id=${SESSION_ID:?}" --data-urlencode "env_id=lite.scalecua" \
    "$CUA_LITE_ENV_SERVER_URL/instances"
  ```
- **`NUM_ROLLOUT=4` is 4 rollouts = 16 optimizer steps = 1024 trajectories, with 3 evals** (step 0,
  8, 16 — `EVAL_INTERVAL=2` plus the step-0 pass). 16 is where both positive pilots were read, and
  there is no evidence more helps: the 4-domain run peaked at 24 steps (+3.97pp) and fell back by 48
  (+1.97pp), and the calc rewrite pool was flat from 20 to 40. With `EVAL_INTERVAL=4` that is two
  evals, at step 0 and step 16.
- **Run it more than once.** The four pilots spread 3.8pp, but all four used DIFFERENT pools, so
  that number mixes recipe effect with run noise and bounds neither. No two runs in this campaign
  have ever shared a pool AND a config, so the repeat spread of a fixed recipe is unmeasured. Until
  it is, a single run's delta cannot be reported as an effect.
- **`CUA_LITE_NORM_BY_TURNS` is deliberately unset.** It switches the per-trajectory loss
  denominator from masked tokens to turn count; the impress pilots set it and the calc pilot did
  not, and the calc pilot is the one that moved furthest. One fewer knob between this run and the
  pilots it is read against.

##### The three `lite.osworld train` runs

These are the +9.88pp / +8.25pp / +12.24pp rows above: same base checkpoint, same render config,
same hyperparameters, and only the two manifests change between them. The shape is the browser and
mobile campaigns' shape (`LR=1e-6`, `RBS=16`, `nspr=8`, 8 steps per rollout), not the `LR=3e-6`,
`RBS=32`, 4-steps-per-rollout shape of the `lite.scalecua` block above — so that one number is
quotable across all three campaigns. **Compare learning rates per optimizer step, not nominally**:
`RBS x nspr / steps_per_rollout` is 16 trajectories per gradient step here against the pilots' 64,
so `1e-6` here is 6.3e-8 per trajectory and the pilots' `3e-6` is 4.7e-8 — this run's effective
rate is the higher of the two despite the smaller nominal value.

<details>
<summary>Data</summary>

All six manifests are plain domain filters over two tracked corpora — no calibration sidecar, no
rollout log, no GPU. `sorted()` is what makes two builds on two clusters byte-identical.

```bash
# --- ONE-TIME DATA BUILD; skip generation for any file that already exists ---
DATA=devs/exps/train/desktop/data
MISSING=0
for m in impress.synth287 impress.train395 impress.eval47 \
         impress_calc.synth565 impress_calc.train779 impress_calc.eval93; do
  [ -e "$DATA/$m.parquet" ] || MISSING=1
done
if [ "$MISSING" = 0 ]; then
  echo "keep existing fixed manifests"
else
  uv run python - "$DATA" <<'PY'
import sys
sys.path.insert(0, "devs/exps/train/desktop")

from lite.utils.parquet import write_records_to_parquet
from utils.tasks import OSW, _read_jsonl, domain_of, eval_rows

D = sys.argv[1]
row = lambda t, split: {"problem": f"Complete the task: {t}",
                        "metadata": {"env_key": f"lite.osworld@{t}", "split": split}}
syn = {r["task_id"]: r for r in _read_jsonl(OSW / "train.synth.jsonl")}
pt = {r["task_id"]: r for r in _read_jsonl(OSW / "train.perturb.jsonl")}

for stem, doms in (("impress", {"libreoffice_impress"}),
                   ("impress_calc", {"libreoffice_impress", "libreoffice_calc"})):
    # sorted() is load-bearing: it is what makes two independent builds byte-identical.
    s = sorted(t for t, r in syn.items() if domain_of(r) in doms)
    p = sorted(t for t, r in pt.items() if domain_of(r) in doms)
    ev = sorted(r["task_id"] for r in eval_rows(scored_only=True) if domain_of(r) in doms)
    train = sorted([(t, "train.synth") for t in s] + [(t, "train.perturb") for t in p])
    write_records_to_parquet([row(t, "train.synth") for t in s],
                             f"{D}/{stem}.synth{len(s)}.parquet")
    write_records_to_parquet([row(t, sp) for t, sp in train],
                             f"{D}/{stem}.train{len(train)}.parquet")
    write_records_to_parquet([row(t, "eval") for t in ev], f"{D}/{stem}.eval{len(ev)}.parquet")
    print(f"{stem}: synth {len(s)}, train {len(train)}, eval {len(ev)}")
PY
fi

uv run python - "$DATA" <<'PY'
import hashlib
import sys

import pandas as pd

from lite.data.staging import coerce_meta

D = sys.argv[1]
EXPECT = {  # (rows, sha256 over the env_key column in file order)
    "impress.synth287":     (287, "246fc3b18e501ab01ac9a04c5e92427b3324dbc58810849f443b32cbd6585c03"),
    "impress.train395":     (395, "b7fa1005057575e612510c9f2adf669ade92c905cc963596b87e9a3e0b6b7355"),
    "impress.eval47":       (47,  "c194503155ddd7b221401a236718e5a95a765128498efb66d2a3e03ab0ed47fc"),
    "impress_calc.train779": (779, "1a83e8e5d9a1bfe561d97be55866a2ba11a5fed6efa7e7e816549fdaea1b83b1"),
    "impress_calc.eval93":  (93,  "56b1ffcfcea45176bed7ed7276e591d022a638ad2ed7f576541724716cb274da"),
}
for stem, (n, want) in EXPECT.items():
    keys = [coerce_meta(r["metadata"])["env_key"]
            for _, r in pd.read_parquet(f"{D}/{stem}.parquet").iterrows()]
    got = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    assert len(keys) == n and got == want, f"{stem}: {len(keys)} rows, sha {got}"
    print(f"{stem} ok: {n} rows, env_key_sha256={got}")
PY
```

Composition, and the provenance label that has to travel with every number read off them:

    impress.synth287      287 synth           overlaps  0 of the 47 impress eval tasks
    impress.train395      287 synth + 108 perturb       40 of 47  (task-level)
    impress.eval47         47 eval tasks, libreoffice_impress
    impress_calc.train779 565 synth (calc 278 / impress 287)
                        + 214 perturb (calc 106 / impress 108)   77 of 93  (task-level)
    impress_calc.eval93    93 eval tasks, calc 46 / impress 47

`impress_calc.synth565.parquet` also falls out of the build. It is the clean-arm control for the
two-domain run and has not been trained yet; the `+12.24pp` above is the `perturb`-including arm.

</details>

```bash
# --- Slime container ---
# One of the three runs; pick the row below and change nothing else.
W=/workspaces/cua-lite
P=desktop.use.highr.i1.reasoning
CKPT=$W/.ckpts/pulled/sft.highr.i1.reasoning.gpt5_5/epoch_2

# CELL                         TRAIN                 EVAL                NSPE SKIP0 GPUS CONC
# grpo.impress.synth287.clean  impress.synth287      impress.eval47       4    1    0-3   32
# grpo.impress.train395.contam impress.train395      impress.eval47       4    0    4-7   32
# grpo.impress_calc.train779   impress_calc.train779 impress_calc.eval93  3    0    0-7   48
CELL=grpo.impress_calc.train779
TRAIN=impress_calc.train779; EVAL=impress_calc.eval93; NSPE=3; SKIP0=0
GPUS=0,1,2,3,4,5,6,7; NGPU=8; CONC=48

: "${CUA_LITE_ENV_SERVER_URL:?paste export line from env-server shell}"
: "${CUA_LITE_ENV_SERVER_TOKEN:?paste export line from env-server shell}"
[ -d "$CKPT" ] || { echo "MISSING CKPT=$CKPT"; exit 1; }

CUDA_VISIBLE_DEVICES=$GPUS NUM_TRAIN_GPUS=$NGPU TP_SIZE=4 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  HF_CKPT="$CKPT" \
  CUA_LITE_MULTIMODAL_LAZY_EXPAND=1 \
  NCCL_NVLS_ENABLE=0 \
  ENV_ID=lite.osworld \
  PROMPT_DATA="$W/devs/exps/train/desktop/data/$TRAIN.parquet" \
  EVAL_PROMPT_DATA="$W/devs/exps/train/desktop/data/$EVAL.parquet" \
  CONFIG_PATH="$W/devs/exps/train/desktop/configs/qwen3_5/$P.yaml" \
  ENV_CONCURRENCY=$CONC \
  ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=8 NUM_STEPS_PER_ROLLOUT=8 \
  ROLLOUT_TEMPERATURE=1.0 ROLLOUT_MAX_RESPONSE_LEN=2048 \
  LR=1e-6 \
  EVAL_TEMPERATURE=1 N_SAMPLES_PER_EVAL_PROMPT=$NSPE \
  EVAL_INTERVAL=5 SKIP_EVAL_BEFORE_TRAIN=$SKIP0 \
  SAVE=1 NO_SAVE_OPTIM=1 SAVE_INTERVAL=5 NUM_ROLLOUT=25 \
  SAVE_DIR="/root/checkpoints/qwen3_5-4b/$CELL/megatron" \
  WANDB_GROUP_SUFFIX=".$CELL" \
  bash "$W/scripts/train/run_grpo.sh"
```

- **Both manifests are `lite.osworld`, so `ENV_ID=lite.osworld` covers preflight outright.** The
  two-env caveat above applies only to the `lite.scalecua` training block; nothing here needs a
  hand-run probe of a second env.
- **The two impress arms ran side by side on one 8-GPU host, 4 GPUs each; the two-domain run had
  all 8.** That is the whole reason the pair is a clean A/B — same host, same env-server, same
  hour, so a host-level confound moves both or neither. It also means their wall clocks
  (~13.5h and ~11.8h) are co-tenant numbers, not the solo cost; the two-domain run took 10.2h on 8
  GPUs.
- **`NUM_ROLLOUT=25` is 25 rollouts = 200 optimizer steps = 3200 trajectories, evaluated every 40
  steps.** Not a budget guess. In two of the three runs the largest delta is the last reading, and
  the two-domain run reads +1.55pp then +0.10pp at 40 and 80 steps before +10.73pp at 120 — stopped
  at 80 it would have been reported dead. Only the clean impress arm peaks early (40 steps) and
  drifts down.
- **`SKIP_EVAL_BEFORE_TRAIN=1` only for the clean impress arm**, which starts from the same weights
  as the contaminated arm and borrows its 0.4303. Any run that is not paired with an identically
  seeded step-0 pass must measure its own.
- **`ENV_CONCURRENCY` is 32 per 4-GPU run and 48 for the 8-GPU run** on a 96-vCPU host, i.e. 8 and
  6 containers per rollout engine. The env-server admits up to 2.5 load per cpu; 48 concurrent
  desktop containers settle around 40-55 load. `run_grpo.sh` records only
  `SERVER_CONCURRENCY = ceil(ENV_CONCURRENCY / NUM_ENGINES)` in the launched command line, so that
  is where the value has to be read back from, multiplied by `NUM_ENGINES`.
- **No `SAVE_HF_DIR`.** The three runs saved Megatron checkpoints only; add it back if a
  downstream eval needs HF weights, and expect the extra wall clock at every `SAVE_INTERVAL`.

#### All three LibreOffice domains, three seeds

The three runs above are each **n=1**, and two of them argued with each other for two commits
before the curves settled it. This one fixes a recipe and varies only the seed — **three arms,
three seeds, one pool** — so the repeat spread of a fixed recipe is measured instead of assumed,
and adds `libreoffice_writer` to make the unit the whole application family. In flight; numbers
below are through optimizer step 112 of 200.

Two knobs differ from the block above, both to buy resolution on the seed question:

- **All three domains, `train.synth` only: 814 tasks** (impress 287 / calc 278 / writer 249).
  The corpus, not the contamination label, is what decides whether the eval number moves —
  `train.synth` produced every gain in this section — so the clean pool is the one worth
  replicating.
- **Eval is the family's full eval split, 117 tasks** (impress 47 / calc 47 / writer 23) at
  `N_SAMPLES_PER_EVAL_PROMPT=4`, `EVAL_TEMPERATURE=1` — 468 trajectories per point, and
  `EVAL_INTERVAL=2` so a reading lands every 16 steps rather than every 40. Denser because the
  per-arm point-to-point swing turns out to be ~6pp and a 40-step grid cannot tell a dip from a
  trend. It costs: 13 x 468 eval episodes against 3200 training ones, **65% of all episodes**.

**The eval's own spread, measured on this 117-task set.** Five step-0 passes over the same
`sft.highr.i1.reasoning.gpt5_5/epoch_2` checkpoint, on five different pods — the three arms below
plus two the campaign later dropped:
**.3164 .3453 .3229 .3253 .3262** — mean .3272, **sd 1.08pp**, so **2 sigma = 2.2pp** for a single
arm and **1.25pp** for the three-arm mean. (The impress-47 threshold quoted above is 3.7pp from
five passes; 117 tasks at n=4 is the tighter measurement, as the task count predicts.)

That bounds the *measurement*. It does not bound the *run*: once training starts the three arms
diverge for real, and at step 32 they were 8.6pp apart (+5.08 / −3.56 / +4.38). Nothing below is
read one arm at a time.

**Eval** — `eval/lite.osworld_eval`, 117 tasks, T=1, 4 draws per task. Step = rollout x 8,
`EVAL_INTERVAL=2`, so the grid is every 16 steps from 0 to 192 (rollout 24, the last of 25).
Values are **pp against each arm's own step-0**; `—` is a reading the run has not reached yet.

| step | 501/7001 | 502/7002 | 503/7003 | **mean** |
|---:|---:|---:|---:|---:|
| 0 *(absolute)* | .3164 | .3453 | .3229 | **.3282** |
| 16 | −1.81 | −1.05 | +3.25 | **+0.13** |
| 32 | +5.08 | −3.56 | +4.38 | **+1.98** |
| 48 | +1.97 | +2.87 | +3.60 | **+2.81** |
| 64 | +2.96 | +1.35 | +7.33 | **+3.88** |
| 80 | +1.86 | +3.08 | +6.06 | **+3.67** |
| 96 | +7.20 | +1.29 | +5.05 | **+4.51** |
| 112 | +8.95 | +2.21 | +7.37 | **+6.18** |
| 128 | — | — | — | — |
| 144 | — | — | — | — |
| 160 | — | — | — | — |
| 176 | — | — | — | — |
| 192 | — | — | — | — |
| **best so far** | **+8.95** | **+3.08** | **+7.37** | **+6.18** |

Absolute values behind the deltas, for the arms' own records:

    501/7001  .3164 .2983 .3672 .3361 .3460 .3350 .3884 .4059
    502/7002  .3453 .3348 .3097 .3740 .3588 .3761 .3582 .3674
    503/7003  .3229 .3554 .3667 .3589 .3962 .3835 .3734 .3966

The mean is monotone from 16 to 112 except for the step-80 reading, and all three arms were
positive at every reading from 48 on. Individual arms are not: 502 ran −3.56 at step 32 and
+2.87 sixteen steps later, and 501 went +1.86 → +7.20 in one interval. **Do not quote a single
arm's point.**

This also re-reads the early-flat pattern the two-domain run hit. Its 40- and 80-step readings were
+1.55 and +0.10 before +10.73 at 120; these three average +0.13 at step 16 and +1.98 at 32 before
+6.18 at 112. Same shape, and it is now four runs plus three seeds saying the first two points of a
200-step curve do not predict it.

**Train** — `rollout/raw_reward`, one value per rollout from rollout 0, T=1. At
`ROLLOUT_BATCH_SIZE=16` over 814 tasks no prompt repeats inside 50 rollouts and the run is 25, so
this is a rolling held-out score on the training distribution, not a convergence curve.

It rises, and the rise is the cleaner of the two signals. Ordinary least squares on the sixteen
rollouts, no smoothing: slope **+2.02 / +0.87 / +1.83 pp per rollout** (t = 5.04 / 1.84 / 4.59),
**+1.58pp per rollout pooled** (t = 5.95), residual sd 7.4-8.7pp per arm. The t-statistics assume
independent residuals and the model state drifts, so treat them as optimistic; the load-bearing
evidence is that **three different `rollout_seed` values draw three different prompt orders and
produce the same slope**, which a lucky easy-tasks-last ordering cannot do. Over sixteen rollouts
that is **~+24pp on the training distribution against +6.18pp on the eval** — the generalization
gap — and the same arm (502) is the weakest on both.

```python
# devs/exps/train/desktop -- GRPO seed replication, libreoffice.synth814 / libreoffice.eval117
# Live through rollout 16 of 25.
EVAL = {  # optimizer step -> mean reward, 117 tasks x 4 draws, T=1
  "501/7001": {0:.3164, 16:.2983, 32:.3672, 48:.3361, 64:.3460, 80:.3350, 96:.3884, 112:.4059},
  "502/7002": {0:.3453, 16:.3348, 32:.3097, 48:.3740, 64:.3588, 80:.3761, 96:.3582, 112:.3674},
  "503/7003": {0:.3229, 16:.3554, 32:.3667, 48:.3589, 64:.3962, 80:.3835, 96:.3734, 112:.3966},
}

TRAIN = {  # rollout/raw_reward, index = rollout
  "501/7001": [0.3281,0.2656,0.2812,0.1641,0.4297,0.5000,0.3359,0.4531,
               0.3750,0.3547,0.3984,0.4766,0.5469,0.5625,0.5938,0.5391],
  "502/7002": [0.3047,0.2812,0.5312,0.3984,0.4688,0.4609,0.3047,0.3906,
               0.4430,0.4062,0.4859,0.3203,0.3750,0.5234,0.6406,0.4453],
  "503/7003": [0.3359,0.3984,0.3047,0.2969,0.3750,0.4453,0.3516,0.3906,
               0.5547,0.4219,0.4609,0.6172,0.6797,0.5547,0.5625,0.4688],
}
```



- **`ROLLOUT_SEED` / `SEED` need a pod-local `run_grpo.sh` patch.** The committed launcher has no
  seed knob, so without it all three arms land on slime's defaults and the "three seeds" are one
  seed three times — the same trap the browser campaign documents. Verify against slime's own
  printed argument table, never the launcher's echo.
- **A pooled eval hides which domain moved.** One `--eval-prompt-data` pair yields one curve;
  slime accepts repeated `name path` pairs but `run_grpo.sh` hardcodes a single pair. The
  per-domain split is recoverable offline from the `SAVE_INTERVAL=2` checkpoints rather than by
  patching the launcher.
- **The family eval dilutes a single-domain effect, on purpose.** A real +6pp confined to impress
  reads as `6 x 47/117 = +2.4pp` here while the noise floor only improves by `sqrt(117/47) = 1.6x`.
  This design is more sensitive to an effect shared across the three applications and *less*
  sensitive to any one domain's — the trade taken after seven single-domain runs failed to agree.
- **Three arms on three hosts.** No two arms share a pod, so a host-level confound cannot move the
  whole condition — the complement of the impress A/B's design, which put both arms on one host to
  hold the host fixed. It also means a host effect is inside the seed spread reported here, not
  separable from it.

```bash
# --- Slime container; one arm. The three clean arms differ only in the two seeds. ---
W=/workspaces/cua-lite
P=desktop.use.highr.i1.reasoning
CKPT=$W/.ckpts/pulled/sft.highr.i1.reasoning.gpt5_5/epoch_2
DATA=$W/devs/exps/train/desktop/data
CELL=grpo.libreoffice.synth814.rs501s7001

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_TRAIN_GPUS=8 TP_SIZE=4 \
  MODEL_ID=Qwen/Qwen3.5-4B HF_CKPT="$CKPT" \
  CUA_LITE_MULTIMODAL_LAZY_EXPAND=1 ENV_ID=lite.osworld \
  PROMPT_DATA="$DATA/libreoffice.synth814.parquet" \
  EVAL_PROMPT_DATA="$DATA/libreoffice.eval117.parquet" \
  CONFIG_PATH="$W/devs/exps/train/desktop/configs/qwen3_5/$P.yaml" \
  ENV_CONCURRENCY=24 \
  ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=8 NUM_STEPS_PER_ROLLOUT=8 \
  ROLLOUT_TEMPERATURE=1.0 ROLLOUT_MAX_RESPONSE_LEN=2048 \
  ROLLOUT_SEED=501 SEED=7001 LR=1e-6 \
  EVAL_TEMPERATURE=1 N_SAMPLES_PER_EVAL_PROMPT=4 \
  EVAL_INTERVAL=2 SKIP_EVAL_BEFORE_TRAIN=0 \
  SAVE=1 NO_SAVE_OPTIM=1 SAVE_INTERVAL=2 NUM_ROLLOUT=25 \
  SAVE_HF_DIR="$W/.ckpts/qwen3_5-4b/$CELL/iter_{rollout_id}" \
  SAVE_DIR="/root/checkpoints/qwen3_5-4b/$CELL/megatron" \
  WANDB_GROUP_SUFFIX=".$CELL" \
  bash "$W/scripts/train/run_grpo.sh"
```

<details>
<summary>Data</summary>

The three manifests are committed and rebuildable. They are **not** built by the `sorted()` block
above: these runs consumed per-domain `export_tasks` output concatenated impress -> calc ->
writer, and that row order decides which 16 prompts each rollout draws, so reproducing the runs
means reproducing the order. The build below was verified to regenerate all three files
byte-identically.

```bash
# --- ONE-TIME DATA BUILD; skip generation if the files already exist ---
DATA=devs/exps/train/desktop/data
MISSING=0
for m in libreoffice.synth814 libreoffice.eval117; do
  [ -e "$DATA/$m.parquet" ] || MISSING=1
done
if [ "$MISSING" = 0 ]; then
  echo "keep existing fixed manifests"
else
  TMP=$(mktemp -d)
  for dom in impress calc writer; do
    for split in train.synth eval; do
      uv run python -m lite.train.export.export_tasks \
        --env-id lite.osworld --split "$split" \
        --filter "lambda m: m.others.get('domain') == 'libreoffice_$dom'" \
        -o "$TMP/$dom.$split.parquet"
    done
  done
  uv run python - "$DATA" "$TMP" <<'BUILD'
import sys

import pandas as pd

from lite.data.staging import coerce_meta
from lite.utils.parquet import write_records_to_parquet

D, T = sys.argv[1], sys.argv[2]
DOMS = ("impress", "calc", "writer")   # concat order is load-bearing, see above


def rows(path, force=None):
    out = []
    for _, r in pd.read_parquet(path).iterrows():
        m = dict(coerce_meta(r["metadata"]))
        # Every training row is tagged "train". main.py registers each train.synth /
        # train.perturb task under BOTH its own split name and "train", so the tag is an
        # alias rather than a second task -- but it is the tag these runs consumed.
        if force:
            m["split"] = force
        out.append({"problem": r["problem"], "metadata": m})
    return out


syn = [x for d in DOMS for x in rows(f"{T}/{d}.train.synth.parquet", force="train")]
ev = [x for d in DOMS for x in rows(f"{T}/{d}.eval.parquet")]
write_records_to_parquet(syn, f"{D}/libreoffice.synth{len(syn)}.parquet")
write_records_to_parquet(ev, f"{D}/libreoffice.eval{len(ev)}.parquet")
print(f"synth {len(syn)}, eval {len(ev)}")
BUILD
fi

uv run python - "$DATA" <<'CHECK'
import hashlib
import sys

import pandas as pd

from lite.data.staging import coerce_meta

D = sys.argv[1]
EXPECT = {  # (rows, sha256 over "env_key<TAB>split<TAB>problem" per row, in file order)
    "libreoffice.synth814":
        (814, "aec9e1581d9a2768e827bfceb877f46d6a29680f0842e85eda25bc9781032162"),
    "libreoffice.eval117":
        (117, "903fc60adefdbd7055eb77ea5b0fb7747df424b772caa966dc520cf2fe105ecd"),
}
for stem, (n, want) in EXPECT.items():
    df = pd.read_parquet(f"{D}/{stem}.parquet")
    lines = []
    for _, r in df.iterrows():
        m = coerce_meta(r["metadata"])
        lines.append("\t".join((m["env_key"], m["split"], r["problem"])))
    got = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    assert len(df) == n and got == want, f"{stem}: {len(df)} rows, sha {got}"
    print(f"{stem} ok: {n} rows, content_sha256={got}")
CHECK
```

Composition, and the one deviation that has to travel with every number in this section:

    libreoffice.synth814    287 impress + 278 calc + 249 writer synth
    libreoffice.eval117     47 impress + 47 calc + 23 writer

**`eval117` is not the scored eval split.** `utils.tasks.eval_rows(scored_only=True)` returns
**115** for these three domains; the two extra rows here are
`osworld_libreoffice_calc_2bd59342` and `osworld_libreoffice_writer_bb8ccc78`, both
`exclude_reason='infeasible'`. They are scorable — `report_infeasible(reason)` is a terminal tool
the OSWorld checker grades — but the rest of this file excludes them, so **these deltas are not
directly comparable to the `impress_calc.eval93` numbers above**. Re-score the 115 subset offline
from the `SAVE_INTERVAL=2` checkpoints before putting the two sections in one table; worst case the
two rows are worth `2/117 = 1.7pp`.

</details>
