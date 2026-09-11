# desktop.use — teacher x screenshot-profile x reasoning SFT campaign

Train **nine checkpoints** — three screenshot profiles x two teachers, plus a reasoning arm on
`gpt5_5` — and score them on one eval.

| | `lowr.i4` | `lowr.i1` | `highr.i1` |
|---|---|---|---|
| | 1280x704, 1-4 img | 1280x704, 1 img | 1920x1088, 1 img |
| **`gpt5_5`** | ✓ | ✓ | ✓ |
| **`gpt5_5` + `<think>`** | ✓ | ✓ | ✓ |
| **`qwen3_8_27b`** | ✓ | ✓ | ✓ |

All three cap SCREENSHOTS and nothing else: `iN` sets `image_max: N` (with `fold_size` tracking
it) and leaves `history_n` at the protocol default of 100, so every past turn is still rendered in
full — its text, its literal `<tool_call>`, and on the reasoning arm its `<think>`. Only the old
*pixels* become `"This screenshot has been collapsed."` That is what the reference agent
effectively does: [xlang-ai/OSWorld#448](https://github.com/xlang-ai/OSWorld/pull/448) ships
`history_n=100` against far shorter episodes, so image folding is the only truncation that fires.

The three form an L with `lowr.i1` at the corner, so each arm moves one thing:

- **image count** — `lowr.i4` vs `lowr.i1`: same resolution, same text history, 1-4 img vs 1.
- **resolution** — `lowr.i1` vs `highr.i1`: same one image, same text history, 880 vs 2040 tokens.
- `lowr.i4` vs `highr.i1` is the diagonal and moves both. Not a profile effect.

Three profiles exist on disk and are deliberately NOT cells (six files with their twins).
`desktop.use.default.yaml` is the fourth corner `highr.i4`: 8160 peak vision tokens per step,
2.3x `lowr.i4`'s peak, which is what sets MBS — excluded for that memory ceiling, not for total
cost (being an i4 profile it packs, so its BILL would land under `highr.i1`). The `hN` pair
(`lowr.h1`, `highr.h1`, twins)
caps TURNS instead, collapsing older ones into a `Previous actions:` prose summary: measured, `h1`
carries **0** historical `<think>` blocks where `i1` carries all of them, so it is the wrong shape
for a reasoning campaign. Kept for a future history-representation study.

Every cell draws 5000 rows from **Lite.ScaleCUA** at the same seed, so within a row only the
profile moves, and within a column rows 1 and 2 differ only by the `<think>` channel — same
teacher, same rows.

The TEACHER axis is not that clean, and the seed does not make it so. Each teacher's pool is its
own successes (`episode_return > 0.5`), so the two pools differ in size and membership and the
shared seed buys nothing across them: measured, the two 5000-row draws share **43%** of their task
ids, which is what independent sampling would give. Reading one teacher row against another
therefore mixes the teacher with a mostly-disjoint training task set. To make it a clean contrast,
intersect the two pools first and draw both teachers' 5000 from that intersection.

The middle row is the reasoning arm. Its config is `desktop.use.<profile>.reasoning.yaml`, which
differs from its Action-only twin by `enable_thinking` alone, so reading a reasoning cell against
the `gpt5_5` cell directly above it isolates Qwen3.5's native `<think>` channel. It is
**`gpt5_5`-only**: that teacher is prompted for a `Thought:` line, which
[`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py) canonicalizes into
`reasoning_content` before staging, so the published rows carry it. `qwen3_8_27b` runs thinking
off, so the same config would train an empty `<think>` block on it.

A cell is a `(config stem, teacher)` pair, and every block below enumerates the nine from one
`cells()` definition. Two names carry them: `$P`, the config stem taken whole off the filename,
and `$DS`, the dataset recipe (`scalecua_5k` = source + row cap). Both are threaded verbatim into
every artifact — parquet `$P.$DS.$T.parquet`, checkpoint `sft.$P.$DS.$T`, HF repo
`ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T`, W&B group suffix the same — so nothing downstream re-derives
`desktop.use`, `.reasoning`, or the row cap on its own, and two recipes never collide. To ablate
the dataset, change `DS=` **and** `--sample` together; the cap is in the name.

> **Train and eval must use the same profile yaml.** `$P` — the config stem — selects both the checkpoint and
> `--config-path`, in every block below. A `highr.i1` checkpoint scored under `lowr.i4` is
> measuring a prompt surface it never saw. The sizes in the matrix above are post-`smart_resize`
> (x32): the yamls ask for 1280x720, and `highr.i1` just takes the envs' native 1920x1080.

### SFT

#### Export

One parquet per cell — the config decides what the model sees, so no two cells can share one.

```bash
# --- TRAIN HOST ---  (its .data/ is what the Slime container mounts; exporting on
# the eval host leaves Train with no parquet to read)
DS=scalecua_5k                           # dataset recipe: source + row cap
DL=.data/huggingface                     # per-teacher roots: $DL/$T/cua-lite/...
OUT=.data/sft/qwen3_5/desktop.use
CFG=devs/exps/train/desktop/configs/qwen3_5
FILTER="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

# The nine cells, as "<config stem> <teacher>" lines. Reasoning arm is gpt5_5-only (see above).
cells() {
  for T in gpt5_5 qwen3_8_27b; do
    for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do echo "$P $T"; done
  done
  for P in desktop.use.lowr.i4.reasoning desktop.use.lowr.i1.reasoning \
           desktop.use.highr.i1.reasoning; do
    echo "$P gpt5_5"
  done
}

# One root per teacher, because export_sft reads whatever --data-paths names and a root with
# both teachers would pool them. --allow-patterns bounds the walk as well as the fetch, so a
# warm HF cache cannot drag in the `rl` variant. --overwrite makes this re-runnable.
for T in gpt5_5 qwen3_8_27b; do
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
    -o "$OUT/$P.$DS.$T.parquet"
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
" "$OUT/$P.$DS.$T.parquet"
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
# `desktop.use` for all nine cells -- so unset, the nine runs overwrite each other's checkpoints
# and land in one W&B group. WANDB_GROUP_SUFFIX has no default; separating them is its job.
#
# Leave SAVE_INTERVAL unset: run_sft.sh defaults it to 1000 (run_sft.sh:135), far past the ~312
# steps a 5000-row/GBS-32/2-epoch cell takes, so the step-interval save never fires. What writes
# the two iter_* dirs is slime's epoch-boundary save -- `step % num_rollout_per_epoch == 0` in
# slime/slime/utils/misc.py, with num_rollout = num_rollout_per_epoch * NUM_EPOCH -- so NUM_EPOCH=2
# gives exactly 2. Ship gates on finding exactly $EPOCHS of them; SKIP on every cell means that
# premise broke (a slime bump can move it), and the checkpoints are still on disk.
#
# CUA_LITE_TRAIN_BROAD_CLEANUP=1 is what makes the LOOP safe: run_sft.sh starts a Ray head and
# never stops it, and its only teardown is utils/cleanup.sh, sourced at startup and opt-in. So
# each cell tears down the previous cell's cluster instead of stacking a second one on top.
# Safe here precisely because the Slime container is dedicated.
DS=scalecua_5k
# The nine cells, as "<config stem> <teacher>" lines. Reasoning arm is gpt5_5-only (see above).
cells() {
  for T in gpt5_5 qwen3_8_27b; do
    for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do echo "$P $T"; done
  done
  for P in desktop.use.lowr.i4.reasoning desktop.use.lowr.i1.reasoning \
           desktop.use.highr.i1.reasoning; do
    echo "$P gpt5_5"
  done
}

while read -r P T; do
  TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
    CUA_LITE_TRAIN_BROAD_CLEANUP=1 \
    MODEL_ID=Qwen/Qwen3.5-4B \
    SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=2 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
    PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5/desktop.use/$P.$DS.$T.parquet \
    SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-4b/sft.$P.$DS.$T/iter_{rollout_id} \
    SAVE_DIR=/root/checkpoints/qwen3_5-4b/sft.$P.$DS.$T/megatron \
    WANDB_GROUP_SUFFIX=".$P.$DS.$T" \
    bash /workspaces/cua-lite/scripts/train/run_sft.sh
done <<< "$(cells)"
```

- `MBS=1` is the safe start, and **tune it on `lowr.i4`** — that column carries the longest
  sequence, so it is what binds. All three columns run the same number of micro-batches per step --
  `GLOBAL_BATCH_SIZE` counts trajectories and each one's segments come along (`run_sft.sh`), and
  all three draw the same rows. Peak vision tokens per step decide whether a micro-batch fits:
  `lowr.i4` 3520, `highr.i1` 2040, `lowr.i1` 880. Total TRAINING cost inverts that ordering,
  because `lowr.i4` packs adjacent steps into one sequence and the one-image profiles cannot --
  measured 0.52x of `lowr.i1` and 0.37x of `highr.i1` on 20 qwen3_8_27b trajectories.
  Longest sequences, smallest bill.
  TP=2 fits all three at 4B; `TP_SIZE=4` if it OOMs.
- `NO_SAVE_OPTIM=1` keeps weights only — these checkpoints are for eval, not for resuming.
- **Export `WANDB_API_KEY` before the first cell.** `run_sft.sh` builds its whole W&B argument
  list inside `if [ -n "${WANDB_API_KEY:-}" ]`, so without it every run trains with no logging
  at all and `WANDB_GROUP_SUFFIX` above does nothing — discovered only after nine multi-hour
  runs have finished.

#### Ship the checkpoints

Train and eval usually run on different machines, so checkpoints travel through the Hub:
**repo** = the cell (`ZHZisZZ/qwen3_5-4b.sft.<config-stem>.<dataset>.<teacher>`),
**subdir** = `epoch_1/`, `epoch_2/`, **tag** = the producing commit.

The tag is provenance, not the entry point: **eval pulls `main`**, so a re-run replaces the epoch
dirs and the next campaign picks them up with no sha to carry between hosts. Reach for the tag
only to re-run an OLD campaign — `--revision <tag>` on the download. Tagging by commit is the
dataset runbooks' convention ([`devs/data/lite.scalecua/AGENTS.md`](/devs/data/lite.scalecua/AGENTS.md)),
and a re-run at the same commit *moves* the tag.

Epoch dirs are named by rank at upload time, because training writes `iter_<N>` with slime's
0-based rollout index — a number you should read off disk, never predict.

```bash
# --- TRAIN HOST ---  (run from the repo root; commit FIRST -- `git rev-parse` reports a sha
# for a dirty tree just as happily, so an uncommitted edit tags nine public repos with a sha
# that does not describe the weights, silently.)
# `uv run hf`, NOT bare `hf`: `hf repos` needs huggingface_hub >= 1.17 and the system hf may be
# older. Needs a WRITE-scoped token (`uv run hf auth login`, or HF_TOKEN). Repos are public,
# so the eval host needs no auth at all.
COMMIT="$(git rev-parse --short HEAD)"   # commit message + tag; eval still pulls main
DS=scalecua_5k
EPOCHS=2                                 # must match NUM_EPOCH in the Train block
CKPTS=.ckpts/qwen3_5-4b
# The nine cells, as "<config stem> <teacher>" lines. Reasoning arm is gpt5_5-only (see above).
cells() {
  for T in gpt5_5 qwen3_8_27b; do
    for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do echo "$P $T"; done
  done
  for P in desktop.use.lowr.i4.reasoning desktop.use.lowr.i1.reasoning \
           desktop.use.highr.i1.reasoning; do
    echo "$P gpt5_5"
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
  uv run hf repos create "$REPO" --repo-type model --exist-ok

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
      --repo-type model --commit-message "$COMMIT: epoch $ep (from $(basename "$D"))"
  done
  # The break above escapes the `for D` loop ONLY; without this, a half-uploaded cell falls
  # through and gets tagged as if complete. Pushed epoch dirs stay on main, untagged.
  [ "$ep" = "-1" ] && continue

  # Tag the commit that produced these weights. Eval pulls `main` and never needs this; it is
  # here so an old campaign stays reproducible -- `--revision <tag>` on the download.
  # create-or-move; the Hub has no move, so it is delete-then-create.
  uv run hf repos tag delete "$REPO" "$COMMIT" --repo-type model --yes \
    || echo "note: no existing '$COMMIT' tag on $REPO (expected on a first upload)"
  uv run hf repos tag create "$REPO" "$COMMIT" --repo-type model -m "$COMMIT" \
    || echo "ERROR $REPO has NO '$COMMIT' tag now; weights are on main -- re-tag by hand"
done <<< "$(cells)"
```

`hf upload` is single-commit and not resumable; `upload-large-folder` is, but takes no
path-in-repo and so cannot express `epoch_<k>/`.

#### Eval

Twelve runs on the `lite.osworld` eval split: the rows left after `--filter` drops
`exclude_reason`, **328 of the 369** `catalog.lock.json` pins. The 41 exclusions live in the
generated `eval.jsonl`, which is gitignored — count them there, not in the lock, and not in
[docs/eval.md](/docs/eval.md#osworld--liteosworld), whose Lite.OSWorld row still says 332. Env
setup: [`lite/gym/envs/lite/osworld/README.md`](/lite/gym/envs/lite/osworld/README.md).

Twelve, not nine: **the base model runs once per screenshot profile.** A checkpoint must be
scored against a baseline that saw the same screenshot surface, and the nine cells use three.

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
# tokenizer/processor, so all twelve runs GENERATE from whatever model that server holds and
# every summary.json still looks normal. Nothing warns.
unset SGLANG_SERVER_URL
EPOCH=epoch_2                            # epoch_1 = after 1 epoch
DS=scalecua_5k
CFG=devs/exps/train/desktop/configs/qwen3_5
PULL=.ckpts/pulled
MISSING=
LOGS=.logs/rollout/Qwen_Qwen3.5-4B/lite.osworld
# The nine cells, as "<config stem> <teacher>" lines. Reasoning arm is gpt5_5-only (see above).
cells() {
  for T in gpt5_5 qwen3_8_27b; do
    for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do echo "$P $T"; done
  done
  for P in desktop.use.lowr.i4.reasoning desktop.use.lowr.i1.reasoning \
           desktop.use.highr.i1.reasoning; do
    echo "$P gpt5_5"
  done
}

while read -r P T; do
  uv run hf download "ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T" \
    --include "$EPOCH/*" \
    --local-dir "$PULL/sft.$P.$DS.$T@$RUN"
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
NGPU=8   # cards this host will use -- twelve runs no longer fit one per card
gpu=0
score() {
  # Drain the batch before wrapping back to card 0, or two rollouts land on one GPU and
  # both OOM. `wait` blocks on every score backgrounded so far, which is the batch.
  if [ "$gpu" -ge "$NGPU" ]; then wait; gpu=0; fi
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.5-4B ${2:+--model-path "$2"} \
    --env-id lite.osworld --splits eval --concurrency 8 \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --config-path "$CFG/$1.yaml" \
    --log-root "$LOGS/$3" &
  gpu=$((gpu + 1))
}

# 3 BASE runs -- empty $2 drops --model-path, so rollout serves --model-id's own weights.
# One per screenshot profile, not one total: a checkpoint is only comparable to a baseline
# that saw the same screenshots. The .reasoning cells reuse their profile's gpt5_5 cell.
for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do
  score "$P" "" "base.$P@$RUN"
done

# 9 CHECKPOINT runs -- $2 is the pulled epoch dir, so that is what gets served.
while read -r P T; do
  score "$P" "$PULL/sft.$P.$DS.$T@$RUN/$EPOCH" "sft.$P.$DS.$T@$RUN.$EPOCH"
done <<< "$(cells)"

wait
```

`--model-id` picks the adapter and action space; the weights, tokenizer and chat template all
come from `--model-path`.

Twelve runs over `NGPU` cards, in batches: `score` drains with `wait` before reusing card 0 —
without it `$gpu` keeps counting past the last card onto ordinals that do not exist. Set `NGPU` to
what the host has FREE; `NGPU=1` serializes all twelve, slow but correct. Both drains are bare
`wait`s, so anything else left backgrounded in this shell delays them.

Score each run from `<log-root>/summary.json` -> `stats.mean_episode_return` (denominator
`num_valid`). Within a column, row 1 vs row 3 is the teacher effect and row 1 vs row 2 is the
`<think>` effect; across a row, between ADJACENT columns, is the profile effect (the L at the top
of this file). Compare an Action-only checkpoint only
against the base run of its own column, and a reasoning checkpoint only against the `gpt5_5`
cell of its own column. `base` runs thinking OFF, so it is the wrong baseline for a reasoning
checkpoint — give that arm its own base run under the `.reasoning` config if you want its
did-SFT-help number too.

#### Record the scores

A campaign that is not written down cannot answer the only question it was run to answer —
did SFT beat the base model. Collect all twelve, then commit them as
`devs/exps/train/desktop/logs/$RUN.md`, the same running-snapshot convention `devs/exps/eval/`
uses ([`/devs/exps/eval/AGENTS.md`](/devs/exps/eval/AGENTS.md#snapshot-template)). One file per
campaign, edited as runs land — not a wrap-up written from memory.

```bash
# --- EVAL HOST, same shell as the Eval block ---
# Reuses $RUN, $DS, $EPOCH, $LOGS and cells(). In a fresh shell set all five again -- $RUN needs
# an ASSIGNMENT (`RUN=<the same slug>`): the Eval block's line is `: "${RUN:?...}"`, an assertion,
# so re-pasting THAT leaves RUN empty and `show` prints MISSING twelve times.
# The n it prints is a result too, not just bookkeeping -- see the num_valid bullet below.
show() {  # $1 = log slug
  uv run python -c "
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'summary.json'
if not p.exists(): print(f'{sys.argv[2]:72s} MISSING'); raise SystemExit
s = json.loads(p.read_text())['stats']
print(f\"{sys.argv[2]:72s} {s['mean_episode_return']:.4f}  n={s['num_valid']}\")
" "$LOGS/$1" "$1"
}
for P in desktop.use.lowr.i4 desktop.use.lowr.i1 desktop.use.highr.i1; do
  show "base.$P@$RUN"
done
while read -r P T; do show "sft.$P.$DS.$T@$RUN.$EPOCH"; done <<< "$(cells)"
```

Paste the numbers into the snapshot's `Results` table, laid out so each contrast the block
above defines reads off one row or one adjacent column pair:

```markdown
# desktop.use @ <run>

- **Checkpoints**: each cell repo's `main` as of `<date>`, tag `<sha>`. Ship can SKIP a cell
  and re-run it later, so the nine repos need not share a sha — if they diverge, list the odd
  ones out here (`uv run hf repos tag list <repo>`), or a later `--revision` re-run pulls the
  wrong weights.
- **Dataset**: `scalecua_5k` (`--sample 5000 --seed 42`, `episode_return > 0.5`)
- **Eval**: `lite.osworld` eval, 328/369 after the `exclude_reason` filter, `epoch_2`
- **Host / GPUs**: `<host>` / `0-7`
- **Last updated**: `<date>`

## Results

Mean episode return, (fully-solved / `num_valid`) in parentheses. Same axes as the matrix at the
top of this file, plus a `base` row: a cell is only comparable to `base` of its own COLUMN.
`highr.h1` is a retired profile kept here because it was scored; it is not a cell to re-run.
The `lowr.i4` column reuses runs made under its old name `lowr.h4`: measured, the two render
byte-identical prompts on these episodes, because `history_n` 50 vs 100 cannot bind at
`max_steps: 30`.

| | `lowr.i4` | `lowr.i1` | `highr.i1` | `highr.h1` (retired) |
|---|---:|---:|---:|---:|
| **base** | 0.2557 (81/328) | | | 0.2070 (65/328) |
| **`gpt5_5`** | 0.3670 (115/328) | | | 0.3743 (118/328) |
| **`gpt5_5` + `<think>`** | | | | |
| **`qwen3_8_27b`** | **0.4052** (130/328) | | | **0.4187** (131/328) |

`num_valid` is 328 in every scored run so far — no run lost samples to errors, so those means
share one denominator. Each sits +0.009 to +0.019 above its solved count (the partial-credit
tail); the ordering reads the same either way. Blanks are cells that have not been trained and
scored yet — the two `i1` profiles, and the whole `<think>` row. Read `lowr.i4` vs `highr.h1` as
a prior only: that pair moves resolution, image count AND text history all at once, which is the
three-knob confound the current profile set was reshaped to remove.

- **Teacher effect** — `gpt5_5` vs `qwen3_8_27b`, within a column.
- **`<think>` effect** — `gpt5_5` + `<think>` vs `gpt5_5`, within a column. Never against
  `base`, which runs thinking off.
- **Profile effect** — the same row, across columns, and only between ADJACENT ones:
  `lowr.i4` vs `lowr.i1` is image count at fixed resolution, `lowr.i1` vs `highr.i1` is
  resolution at fixed image count. `lowr.i4` vs `highr.i1` moves both and is the diagonal.
- **Did SFT help at all** — each cell vs a base run on ITS OWN surface. The two Action-only
  rows use the `base` row above. The `<think>` row needs its own: Qwen3.5's chat template has a
  native reasoning channel that `factory.py` merely pins off for the eval matrix, so a base run
  under the `.reasoning` config is a real baseline, not a model filling a slot it never learned.
  Score it by adding `score "$P.reasoning" "" "base.$P.reasoning@$RUN"` for each profile — three
  more runs, and the campaign answers this for all nine cells instead of six.
- **`num_samples - num_valid`** — record it per run. `num_valid` counts the samples that
  finished with no error (`valid = [r for r in results if r["error"] is None]`,
  `lite/infer/rollout.py`), and MER averages over exactly those, so an arm that errored more is
  a mean over fewer episodes. A malformed FINAL turn is not one of them: it is scored by the env
  like any other episode and stays in the denominator, surfacing instead as
  `stats.stop_reasons.parse_failure` in `summary.json`. Record that separately — it is a
  model-quality number and the reasoning arm is the one to watch on it. If two arms differ on
  either count, say so before reading their MER gap.
```

Note what the reasoning row measures here: every trajectory's terminal turn is a bare
`Done.` with no `reasoning_content`, so a `.reasoning` cell trains an empty `<think>` on its
last step. That is one step per trajectory — **11.8%** of steps on the real 5000-row export
(42427 steps, 8.49 per trajectory), not the ~29% a short-trajectory sample suggests. It is a
property of the data, not of the config. Read the `<think>` effect with it in mind.
