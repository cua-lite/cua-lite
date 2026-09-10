# desktop.use — teacher x screenshot-profile x reasoning SFT campaign

Train **six checkpoints** — a 2x2 of two teachers x two screenshot profiles, plus a reasoning
arm on `gpt5_5` — and score them on one eval.

| | `lowr.h4` | `highr.h1` |
|---|---|---|
| | 1280x720, up to 4 screenshots | native 1920x1080, 1 screenshot |
| **`gpt5_5`** | ✓ | ✓ |
| **`gpt5_5` + `<think>`** | ✓ | ✓ |
| **`qwen3_8_27b`** | ✓ | ✓ |

Teacher axis: whose trajectories teach better. Profile axis: how to spend a fixed VRAM budget —
four small screenshots, or one big one. Every cell draws 5000 rows from **Lite.ScaleCUA** at the
same seed, so within a row only the profile moves, and within a column rows 1 and 2 differ only
by the `<think>` channel — same teacher, same rows.

The TEACHER axis is not that clean, and the seed does not make it so. Each teacher's pool is its
own successes (`episode_return > 0.5`), so the two pools differ in size and membership and the
shared seed buys nothing across them: measured, the two 5000-row draws share **43%** of their
task ids, which is what independent sampling would give. A column difference therefore mixes the
teacher with a mostly-disjoint training task set. To make it a clean contrast, intersect the two
pools first and draw both teachers' 5000 from that intersection.

The middle row is the reasoning arm. Its config is `desktop.use.<profile>.reasoning.yaml`, which
differs from its Action-only twin by `enable_thinking` alone, so reading a reasoning cell against
the `gpt5_5` cell directly above it isolates Qwen3.5's native `<think>` channel. It is
**`gpt5_5`-only**: that teacher is prompted for a `Thought:` line, which
[`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py) canonicalizes into
`reasoning_content` before staging, so the published rows carry it. `qwen3_8_27b` runs thinking
off, so the same config would train an empty `<think>` block on it.

A cell is a `(config stem, teacher)` pair, and every block below enumerates the six from one
`cells()` definition. Two names carry them: `$P`, the config stem taken whole off the filename,
and `$DS`, the dataset recipe (`scalecua_5k` = source + row cap). Both are threaded verbatim into
every artifact — parquet `$P.$DS.$T.parquet`, checkpoint `sft.$P.$DS.$T`, HF repo
`ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T`, W&B group suffix the same — so nothing downstream re-derives
`desktop.use`, `.reasoning`, or the row cap on its own, and two recipes never collide. To ablate
the dataset, change `DS=` **and** `--sample` together; the cap is in the name.

> **Train and eval must use the same profile yaml.** `$P` — the config stem — selects both the checkpoint and
> `--config-path`, in every block below. A `highr.h1` checkpoint scored under `lowr.h4` is
> measuring a prompt surface it never saw. Resolutions, for the record: `lowr.h4` downsamples to
> 1280x720 and Qwen `smart_resize` rounds it to **1280x704**; `highr.h1` leaves the env's native
> **1920x1080**, rounded to **1920x1088**. Both envs (`lite.osworld`, `lite.cuagym`) declare
> `display_resolution: [1920, 1080]`.

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

# The six cells, as "<config stem> <teacher>" lines. The reasoning arm is gpt5_5-only:
# qwen3_8_27b runs thinking off, so a thinking-on config would train an empty <think> block.
cells() {
  for T in gpt5_5 qwen3_8_27b; do
    for P in desktop.use.lowr.h4 desktop.use.highr.h1; do echo "$P $T"; done
  done
  for P in desktop.use.lowr.h4.reasoning desktop.use.highr.h1.reasoning; do echo "$P gpt5_5"; done
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
# All-zero means every cell drew the same rows, which is what the 2x2 assumes.
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
# `desktop.use` for all six cells -- so unset, the six runs overwrite each other's checkpoints
# and land in one W&B group. WANDB_GROUP_SUFFIX has no default; separating them is its job.
#
# Leave SAVE_INTERVAL unset: run_sft.sh defaults it to 1000 (run_sft.sh:135), far past the ~312
# steps a 5000-row/GBS-32/2-epoch cell takes, so the step-interval save never fires. What writes
# the two iter_* dirs is slime's epoch-boundary save -- `step % num_rollout_per_epoch == 0` in
# slime/utils/misc.py, with num_rollout = num_rollout_per_epoch * NUM_EPOCH -- so NUM_EPOCH=2
# gives exactly 2. Ship gates on finding exactly $EPOCHS of them; SKIP on every cell means that
# premise broke (a slime bump can move it), and the checkpoints are still on disk.
#
# CUA_LITE_TRAIN_BROAD_CLEANUP=1 is what makes the LOOP safe: run_sft.sh starts a Ray head and
# never stops it, and its only teardown is utils/cleanup.sh, sourced at startup and opt-in. So
# each cell tears down the previous cell's cluster instead of stacking a second one on top.
# Safe here precisely because the Slime container is dedicated.
DS=scalecua_5k
# The six cells, as "<config stem> <teacher>" lines. The reasoning arm is gpt5_5-only:
# qwen3_8_27b runs thinking off, so a thinking-on config would train an empty <think> block.
cells() {
  for T in gpt5_5 qwen3_8_27b; do
    for P in desktop.use.lowr.h4 desktop.use.highr.h1; do echo "$P $T"; done
  done
  for P in desktop.use.lowr.h4.reasoning desktop.use.highr.h1.reasoning; do echo "$P gpt5_5"; done
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

- `MBS=1` is the safe start, and **tune it on `lowr.h4`** — that column carries the longer
  sequence (~4 x 880 vision tokens vs one 2040), so it is what binds. Both columns run the same
  number of micro-batches per step -- `GLOBAL_BATCH_SIZE` counts trajectories and each one's
  segments come along (`run_sft.sh`), and both columns draw the same rows -- so `highr.h1` is
  the lighter of the two, not merely the easier fit.
  TP=2 fits both at 4B; `TP_SIZE=4` if it OOMs.
- `NO_SAVE_OPTIM=1` keeps weights only — these checkpoints are for eval, not for resuming.
- **Export `WANDB_API_KEY` before the first cell.** `run_sft.sh` builds its whole W&B argument
  list inside `if [ -n "${WANDB_API_KEY:-}" ]`, so without it every run trains with no logging
  at all and `WANDB_GROUP_SUFFIX` above does nothing — discovered only after six multi-hour
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
# for a dirty tree just as happily, so an uncommitted edit tags six public repos with a sha
# that does not describe the weights, silently.)
# `uv run hf`, NOT bare `hf`: `hf repos` needs huggingface_hub >= 1.5 and the system hf may be
# older. Needs a WRITE-scoped token (`uv run hf auth login`, or HF_TOKEN). Repos are public,
# so the eval host needs no auth at all.
COMMIT="$(git rev-parse --short HEAD)"   # commit message + tag; eval still pulls main
DS=scalecua_5k
EPOCHS=2                                 # must match NUM_EPOCH in the Train block
CKPTS=.ckpts/qwen3_5-4b
# The six cells, as "<config stem> <teacher>" lines. The reasoning arm is gpt5_5-only:
# qwen3_8_27b runs thinking off, so a thinking-on config would train an empty <think> block.
cells() {
  for T in gpt5_5 qwen3_8_27b; do
    for P in desktop.use.lowr.h4 desktop.use.highr.h1; do echo "$P $T"; done
  done
  for P in desktop.use.lowr.h4.reasoning desktop.use.highr.h1.reasoning; do echo "$P gpt5_5"; done
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
  uv run hf repos create "$REPO" --repo-type model --exist-ok

  ep=0
  for D in $DIRS; do
    ep=$((ep + 1))
    # eval loads the tokenizer/processor from --model-path, not --model-id, so the epoch dir
    # must carry slime's FULL HF export. Check before spending 8 GB of upload on it:
    # a weights-only dir only fails on the eval host, after upload AND download.
    for f in config.json tokenizer_config.json preprocessor_config.json; do
      [ -e "$D/$f" ] && continue
      echo "SKIP $REPO: $D has no $f -- AutoProcessor would fail at eval; not uploading"
      ep=-1; break
    done
    [ "$ep" = "-1" ] && break
    uv run hf upload "$REPO" "$D" "epoch_$ep" \
      --repo-type model --commit-message "$COMMIT: epoch $ep (from $(basename "$D"))"
  done

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

Eight runs on the `lite.osworld` eval split: the rows left after `--filter` drops
`exclude_reason`, **328 of the 369** `catalog.lock.json` pins. The 41 exclusions live in the
generated `eval.jsonl`, which is gitignored — count them there, not in the lock, and not in
[docs/eval.md](/docs/eval.md#osworld--liteosworld), whose Lite.OSWorld row still says 332. Env
setup: [`lite/gym/envs/lite/osworld/README.md`](/lite/gym/envs/lite/osworld/README.md).

Eight, not six: **the base model runs once per screenshot profile.** A checkpoint must be scored
against a baseline that saw the same screenshot surface, and the six cells use only two.

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
# samples into one summary.json.
: "${RUN:?set RUN to a fresh slug for THIS campaign, e.g. 20260910a}"

# MUST be unset. `--sglang-server-url` defaults to $SGLANG_SERVER_URL (lite/infer/cli.py), and
# with a URL in hand serving.py never starts a server -- `--model-path` then only picks the
# tokenizer/processor, so all eight runs GENERATE from whatever model that server holds and
# every summary.json still looks normal. Nothing warns.
unset SGLANG_SERVER_URL
EPOCH=epoch_2                            # epoch_1 = after 1 epoch
DS=scalecua_5k
CFG=devs/exps/train/desktop/configs/qwen3_5
PULL=.ckpts/pulled
MISSING=
LOGS=.logs/rollout/Qwen_Qwen3.5-4B/lite.osworld
# The six cells, as "<config stem> <teacher>" lines. The reasoning arm is gpt5_5-only:
# qwen3_8_27b runs thinking off, so a thinking-on config would train an empty <think> block.
cells() {
  for T in gpt5_5 qwen3_8_27b; do
    for P in desktop.use.lowr.h4 desktop.use.highr.h1; do echo "$P $T"; done
  done
  for P in desktop.use.lowr.h4.reasoning desktop.use.highr.h1.reasoning; do echo "$P gpt5_5"; done
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
gpu=0
score() {
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.5-4B ${2:+--model-path "$2"} \
    --env-id lite.osworld --splits eval --concurrency 8 \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --config-path "$CFG/$1.yaml" \
    --log-root "$LOGS/$3" &
  gpu=$((gpu + 1))
}

# 2 BASE runs -- empty $2 drops --model-path, so rollout serves --model-id's own weights.
# One per screenshot profile, not one total: a checkpoint is only comparable to a baseline
# that saw the same screenshots. The .reasoning cells reuse their profile's gpt5_5 cell.
for P in desktop.use.lowr.h4 desktop.use.highr.h1; do score "$P" "" "base.$P@$RUN"; done

# 6 CHECKPOINT runs -- $2 is the pulled epoch dir, so that is what gets served.
while read -r P T; do
  score "$P" "$PULL/sft.$P.$DS.$T@$RUN/$EPOCH" "sft.$P.$DS.$T@$RUN.$EPOCH"
done <<< "$(cells)"

wait
```

`--model-id` picks the adapter and action space; the weights, tokenizer and chat template all
come from `--model-path`.

Needs 8 free GPUs as written: `score` hands out `$gpu` and increments, across both loops. With
fewer, insert a `wait` and reset `gpu=0` wherever you want a batch boundary; dropping `&`/`wait`
entirely serializes all eight and still names `CUDA_VISIBLE_DEVICES=7`.

Score each run from `<log-root>/summary.json` -> `stats.mean_episode_return` (denominator
`num_valid`). Within a column, row 1 vs row 3 is the teacher effect and row 1 vs row 2 is the
`<think>` effect; across a row is the profile effect. Compare an Action-only checkpoint only
against the base run of its own column, and a reasoning checkpoint only against the `gpt5_5`
cell of its own column. `base` runs thinking OFF, so it is the wrong baseline for a reasoning
checkpoint — give that arm its own base run under the `.reasoning` config if you want its
did-SFT-help number too.

#### Record the scores

A campaign that is not written down cannot answer the only question it was run to answer —
did SFT beat the base model. Collect all eight, then commit them as
`devs/exps/train/desktop/logs/$RUN.md`, the same running-snapshot convention `devs/exps/eval/`
uses ([`/devs/exps/eval/AGENTS.md`](/devs/exps/eval/AGENTS.md#snapshot-template)). One file per
campaign, edited as runs land — not a wrap-up written from memory.

```bash
# --- EVAL HOST, same shell as the Eval block ---
# Reuses $RUN, $DS, $EPOCH, $LOGS and cells() from it; in a fresh shell, re-paste those
# five lines first (same $RUN, or you will read a different campaign's log roots).
# num_valid is the DENOMINATOR of mean_episode_return, and it is model-dependent: a sample whose
# last turn was malformed is dropped from it rather than scored 0. So n below is a result too --
# a gap between two arms means their MERs are over different denominators.
show() {  # $1 = log slug
  uv run python -c "
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'summary.json'
if not p.exists(): print(f'{sys.argv[2]:52s} MISSING'); raise SystemExit
s = json.loads(p.read_text())['stats']
print(f\"{sys.argv[2]:52s} {s['mean_episode_return']:.4f}  n={s['num_valid']}\")
" "$LOGS/$1" "$1"
}
for P in desktop.use.lowr.h4 desktop.use.highr.h1; do show "base.$P@$RUN"; done
while read -r P T; do show "sft.$P.$DS.$T@$RUN.$EPOCH"; done <<< "$(cells)"
```

Paste the numbers into the snapshot's `Results` table, laid out so the three contrasts the
block above defines are each one comparison:

```markdown
# desktop.use @ <run>

- **Checkpoints**: each cell repo's `main` as of `<date>`, tag `<sha>`. Ship can SKIP a cell
  and re-run it later, so the six repos need not share a sha — if they diverge, list the odd
  ones out here (`uv run hf repos tag list <repo>`), or a later `--revision` re-run pulls the
  wrong weights.
- **Dataset**: `scalecua_5k` (`--sample 5000 --seed 42`, `episode_return > 0.5`)
- **Eval**: `lite.osworld` eval, 328/369 after the `exclude_reason` filter, `epoch_2`
- **Host / GPUs**: `<host>` / `0-7`
- **Last updated**: `<date>`

## Results

Mean episode return, `num_valid` in parentheses. Same axes as the matrix at the top of this
file, plus a `base` row: a cell is only comparable to `base` of its own COLUMN.

| | `lowr.h4` | `highr.h1` |
|---|---|---|
| **base** | | |
| **`gpt5_5`** | | |
| **`gpt5_5` + `<think>`** | | |
| **`qwen3_8_27b`** | | |

- **Teacher effect** — `gpt5_5` vs `qwen3_8_27b`, within a column.
- **`<think>` effect** — `gpt5_5` + `<think>` vs `gpt5_5`, within a column. Never against
  `base`, which runs thinking off.
- **Profile effect** — the same row, across the two columns.
- **Did SFT help at all** — each cell vs a base run on ITS OWN surface. The two Action-only
  rows use the `base` row above. The `<think>` row needs its own: Qwen3.5's chat template has a
  native reasoning channel that `factory.py` merely pins off for the eval matrix, so a base run
  under the `.reasoning` config is a real baseline, not a model filling a slot it never learned.
  Score it by adding `score "$P.reasoning" "" "base.$P.reasoning@$RUN"` for each profile — two
  more runs, and the campaign answers this for all six cells instead of four.
- **`num_samples - num_valid`** — record it per run. A sample whose last turn was malformed
  leaves BOTH sides of the mean (`_summary_error`, `lite/infer/rollout.py`), so MER flatters the
  arm that fails more. If two arms differ here, say so before reading their MER gap.
```

Note what the reasoning column measures here: every trajectory's terminal turn is a bare
`Done.` with no `reasoning_content`, so a `.reasoning` cell trains an empty `<think>` on its
last step. That is one step per trajectory — **11.8%** of steps on the real 5000-row export
(42427 steps, 8.49 per trajectory), not the ~29% a short-trajectory sample suggests. It is a
property of the data, not of the config. Read the `<think>` effect with it in mind.
