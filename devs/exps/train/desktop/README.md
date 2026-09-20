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
cells instead of landing on one. `±` is half the range of the three.

| | `lowr.i4` | `lowr.i1` | `highr.i1` | `highr.h1` (retired) |
|---|---:|---:|---:|---:|
| **base** | 0.2597 ±0.0091 (82/328) 3/3 | 0.1592 ±0.0031 (49/328) 3/3 | 0.2329 ±0.0090 (74/328) 3/3 | 0.2070 (65/328) |
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
