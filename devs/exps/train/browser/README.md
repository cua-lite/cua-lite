# browser.use — teacher x image-cap x reasoning SFT campaign

Train **eight checkpoints** — two image caps x two teachers, each with a `<think>` arm — and
score them on the held-out WebGym eval split.

| | `i4` | `i1` |
|---|---|---|
| | 1-4 screenshots | 1 screenshot |
| **`gpt5_5`** | ✓ | ✓ |
| **`gpt5_5` + `<think>`** | ✓ | ✓ |
| **`qwen3_5_27b`** | ✓ | ✓ |
| **`qwen3_5_27b` + `<think>`** | ✓ | ✓ |

**There is no resolution axis here.** WebGym renders at its own 1280x720 viewport
(`lite/gym/envs/webgym/configs/default.yaml`), so every profile sees the same pixels —
`smart_resize` rounds to 1280x704, 880 vision tokens. That is what the desktop campaign calls
`lowr`, and there is no `highr` to pair it with. The only knobs left are how many screenshots
survive and whether `<think>` is on, which is why these configs are named for `image_max` alone.

`iN` sets `image_max: N` with `fold_size` tracking it and leaves `history_n` at the protocol
default of 100, so every past turn is still rendered in full — its text, its literal
`<tool_call>`, and on the reasoning arm its `<think>`. Only the old *pixels* become
`"This screenshot has been collapsed."`

- **image count** — `i4` vs `i1`, same text history, 1-4 screenshots against 1. That is the
  campaign's one profile contrast.
- **teacher** — `gpt5_5` vs `qwen3_5_27b`, within a column.
- **`<think>`** — each `+ <think>` row against the Action-only row of the SAME teacher. Never
  against the other teacher's row.

`i4` costs about HALF of `i1` to train, which is not obvious: `fold_size == image_max` keeps four
adjacent steps token- and image-prefix compatible, so `build_segment_samples` packs them into one
sequence. A one-image window is never prefix-compatible with the next step, so `i1` emits one
sequence per step — 3.33 steps/segment against 1.00 (gpt5_5; 3.16 for qwen3_5_27b), at 2.36 / 2.31
visible images per step. Per trajectory, over 150 exported `webgym_1k` rows with the real
tokenizer and 880 vision tokens per 1280x704 screenshot:

| | seq/traj | text tok | image slots | vision tok | total |
|---|---:|---:|---:|---:|---:|
| `gpt5_5` `i4` | 2.58 | 5 570 | 8.9 | 7 814 | **13 385** |
| `gpt5_5` `i1` | 8.88 | 17 783 | 8.9 | 7 814 | **25 598** |

That is **0.52x** for `gpt5_5` and **0.57x** for `qwen3_5_27b`. Note the image slots are
IDENTICAL — every screenshot is encoded exactly once either way, so the saving is not in pixels.
It is all text: `i1` re-renders the full turn history once per step, `i4` once per segment. Tune
`MBS` on `i4` anyway — packing makes its sequences the longest.

**`qwen3_8_27b` is not in this campaign.** It emits no `reasoning_content`, so it has no
`<think>` arm at all, and its Action-only cells were not judged worth the four extra runs here.

## Data

`cua-lite/WebGym` publishes both teachers: **`browser.use.gpt5_5`** (3143 rows, 148 shards) and
**`browser.use.qwen3_5_27b`** (967 rows, 32 shards, tag `f50e58f`). All eight cells export today.

> **Adding the second teacher is not an append.** `lite.data.hf.upload` is a declarative full
> sync: it plans the whole repo from the local staging dir and deletes everything else, so
> staging only the new config and uploading DELETES the published `gpt5_5` shards. Follow
> [Add A Config To A Published Dataset](/devs/data/AGENTS.md#add-a-config-to-a-published-dataset).
> The two teachers also hand over different roots — `gpt5_5` stages its `.think` siblings
> (`internalize_cot.py` canonicalizes its prompted `Thought:` into `reasoning_content`),
> `qwen3_5_27b` stages the bare `_clean` roots because `enable_thinking` wrote the field
> natively. See [`/devs/data/webgym/AGENTS.md`](/devs/data/webgym/AGENTS.md) step 3.

### Export

```bash
# --- TRAIN HOST ---  (the Slime container mounts the repo root, so its .data/ is what Train reads)
DS=webgym_1k                             # dataset recipe: source + row cap
DL=.data/huggingface-webgym              # per-teacher roots: $DL/$T/cua-lite/...
OUT=.data/sft/qwen3_5/browser.use
CFG=devs/exps/train/browser/configs/qwen3_5
FILTER="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

# The eight cells as "<config stem> <teacher>". Both teachers get a <think> arm here --
# unlike desktop.use, neither of them is thinking-off.
cells() {
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4 browser.use.i1; do echo "$P $T"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4.reasoning browser.use.i1.reasoning; do echo "$P $T"; done
  done
}

# One root per teacher: export_sft reads whatever --data-paths names and a shared root would
# pool them. --allow-patterns bounds the walk as well as the fetch. KEEP THIS LIST IN SYNC
# WITH cells().
for T in gpt5_5 qwen3_5_27b; do
  uv run python -m lite.data.hf.download WebGym \
    --allow-patterns "browser/use/train/browser.use.$T/*" \
    --out "$DL/$T/cua-lite/WebGym" --overwrite
done

# WebGym rewards are BINARY (0 or 1.0, no partial credit), so `> 0.5` keeps exactly the solved
# episodes. --filter runs before --sample, so the cap lands on kept rows.
while read -r P T; do
  uv run python -m lite.train.export.export_sft \
    --config "$CFG/$P.yaml" \
    --model-id Qwen/Qwen3.5-4B \
    --data-paths "$DL/$T/cua-lite/WebGym" \
    --image-root "$DL/$T" \
    --filter "$FILTER" --sample 1000 --seed 42 --no-strict \
    -o "$OUT/$P.$DS.$T.parquet" < /dev/null   # or the child eats the rest of the cell list
done <<< "$(cells)"

# --no-strict makes a conversion failure a SKIP, not an error, so a wrong --image-root writes
# an empty parquet and still exits 0. Count the rows before trusting them.
# Expected rows = min(--sample, pool). qwen3_5_27b's pool IS 967, so its four cells have ZERO
# margin: any future unconvertible row shortens them silently. Check the exact number per
# teacher, not --sample.
while read -r P T; do
  case "$T" in gpt5_5) N=1000;; qwen3_5_27b) N=967;; esac
  uv run python -c "
import sys, pyarrow.parquet as pq
n, want = pq.read_metadata(sys.argv[1]).num_rows, int(sys.argv[2])
print(('OK   ' if n == want else 'SHORT'), n, f'(want {want})', sys.argv[1])
" "$OUT/$P.$DS.$T.parquet" "$N" < /dev/null
done <<< "$(cells)"
```

**Why 1000.** Pool after `$FILTER`: `browser.use.gpt5_5` **3143** rows, `browser.use.qwen3_5_27b`
**967**. Both configs are already success-filtered upstream — staged from `_clean` roots — so
`$FILTER` keeps 100% and the pool IS the published row count; the filter stays in the command
only to remain correct if a future config ever stages raw roots. So 1000 fills the four `gpt5_5`
cells and takes the ENTIRE qwen pool at 967: `export_sft` prints `Only 967 rows convertible` and
exits 0. That 3.3% row gap is deliberate — it is an order of magnitude smaller than the task-mix
gap below, and capping every cell at 967 to match exactly would discard 2176 `gpt5_5` rows to
close a difference that is already noise.

**900 matches row count, not difficulty.** The two teachers solved different tasks: at `--seed 42`
their 900-row samples share only **208** task ids, and the difficulty histograms diverge —
`gpt5_5` d1/d7 = 167/143 with a thin d8-d15 tail, `qwen3_5_27b` d1/d7 = 281/82 and nothing above
d7. That is not a sampling artifact; it is what each teacher could solve. So a teacher cell
compares teacher AND task mix together. Read the teacher rows that way, or re-sample stratified on
`others.difficulty` if you need the mix held fixed — the intersection is too small to train on.

### Train

```bash
# --- Slime container ---  (DS must match the Export block)
# SFT at TP=2 (8 GPUs -> DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN cannot
# THD-pack). Serial: each run takes all 8 GPUs.
#
# SAVE_DIR / SAVE_HF_DIR / WANDB_GROUP_SUFFIX are MANDATORY. run_sft.sh keys both checkpoint
# dirs AND the W&B group off PROMPT_DATA's parent dir, which is `browser.use` for every cell --
# so unset, the runs overwrite each other and land in one W&B group.
#
# Export WANDB_API_KEY first: run_sft.sh builds its whole W&B argument list only when that
# variable is set, so without it the runs train fine and log nowhere.
#
# CUA_LITE_TRAIN_BROAD_CLEANUP=1 makes the LOOP safe: run_sft.sh starts a Ray head and never
# stops it. Set it only in a dedicated training container.
DS=webgym_1k
cells() {
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4 browser.use.i1; do echo "$P $T"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4.reasoning browser.use.i1.reasoning; do echo "$P $T"; done
  done
}

while read -r P T; do
  TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
    CUA_LITE_TRAIN_BROAD_CLEANUP=1 \
    MODEL_ID=Qwen/Qwen3.5-4B \
    SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=2 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
    PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5/browser.use/$P.$DS.$T.parquet \
    SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-4b/sft.$P.$DS.$T/iter_{rollout_id} \
    SAVE_DIR=/root/checkpoints/qwen3_5-4b/sft.$P.$DS.$T/megatron \
    WANDB_GROUP_SUFFIX=".$P.$DS.$T" \
    bash /workspaces/cua-lite/scripts/train/run_sft.sh < /dev/null
done <<< "$(cells)"
```

- `MBS=1` is the safe start; tune it on `i4`, which carries the longest sequence.
- `NO_SAVE_OPTIM=1` keeps weights only — these checkpoints are for eval, not for resuming.

### Eval

WebGym's eval split is **1167 tasks**, and every one of them hits a live website. Two costs that
`lite.osworld` does not have:

- **Live-site blocking.** Measured over 7.6k training rollouts on this env, **18.8%** ended in
  `EnvBlocked` — a bot wall, not a model failure. Those are void episodes: out of the eval
  denominator, so they cost wall-clock without costing score. Budget for them.
- **Scale.** 1167 tasks x 10 runs is a lot of live browsing. `--sample` a fixed subset with a
  seed and score every checkpoint on the SAME subset; the seeded parquet is what makes the
  comparison paired.

```bash
# --- EVAL HOST ---
: "${RUN:?set RUN to a fresh slug for THIS campaign, e.g. 20260911a}"
unset SGLANG_SERVER_URL   # else every run silently attaches to that server
EPOCH=epoch_2
DS=webgym_1k
CFG=devs/exps/train/browser/configs/qwen3_5
PULL=.ckpts/pulled
LOGS=.logs/rollout/Qwen_Qwen3.5-4B/webgym
NGPU=8                    # cards this host will use
EVAL_N=256                # eval-split subset; the same 256 for every checkpoint
cells() {
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4 browser.use.i1; do echo "$P $T"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4.reasoning browser.use.i1.reasoning; do echo "$P $T"; done
  done
}

MISSING=
while read -r P T; do
  uv run hf download "ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T" \
    --include "$EPOCH/*" --local-dir "$PULL/sft.$P.$DS.$T@$RUN" < /dev/null
  for f in config.json tokenizer_config.json preprocessor_config.json; do
    [ -e "$PULL/sft.$P.$DS.$T@$RUN/$EPOCH/$f" ] || MISSING="$MISSING $P.$T:$f"
  done
done <<< "$(cells)"
[ -z "$MISSING" ] && echo "checkpoints OK" \
  || echo "STOP -- do not run the score loop; missing under $EPOCH:$MISSING"

gpu=0
score() {  # $1 = config stem, $2 = --model-path ("" = base), $3 = log slug
  if [ "$gpu" -ge "$NGPU" ]; then wait; gpu=0; fi
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.5-4B ${2:+--model-path "$2"} \
    --env-id webgym --splits eval --sample "$EVAL_N" --seed 42 --concurrency 16 \
    --config-path "$CFG/$1.yaml" \
    --log-root "$LOGS/$3" < /dev/null &
  gpu=$((gpu + 1))
}

# 2 BASE runs, one per image cap: a checkpoint is only comparable to a baseline that saw the
# same prompt surface. The .reasoning cells reuse their profile's base run.
for P in browser.use.i4 browser.use.i1; do score "$P" "" "base.$P@$RUN"; done

# 8 CHECKPOINT runs.
while read -r P T; do
  score "$P" "$PULL/sft.$P.$DS.$T@$RUN/$EPOCH" "sft.$P.$DS.$T@$RUN.$EPOCH"
done <<< "$(cells)"

wait
```

Ten runs over `NGPU` cards, in batches: `score` drains with `wait` before reusing card 0 —
without it `$gpu` keeps counting past the last card onto ordinals that do not exist.
`--concurrency 16` per card, not 8: WebGym steps are dominated by page load, not by the model.

Score each run from `<log-root>/summary.json` -> `stats.mean_episode_return` (denominator
`num_valid`, which excludes the `EnvBlocked` episodes). Within a column, the two Action-only rows
give the teacher effect and each `+ <think>` row against the Action-only row of the same teacher
gives the `<think>` effect; across the two columns is the image-count effect.

### Record the scores

Commit the numbers as `devs/exps/train/browser/logs/$RUN.md` — flat here, where
[`/devs/exps/eval/AGENTS.md`](/devs/exps/eval/AGENTS.md#snapshot-template) nests under a
per-commit directory, but the same idea: one file per campaign, edited as runs land.

```markdown
# browser.use @ <run>

- **Dataset**: `webgym_1k` (`--sample 1000 --seed 42`, `episode_return > 0.5`) — 1000 rows per
  `gpt5_5` cell, 967 per `qwen3_5_27b` cell (its whole pool).
- **Eval**: WebGym eval split, `--sample 256 --seed 42`, `epoch_2`
- **Last updated**: `<date>`

## Results

Mean episode return, `num_valid` in parentheses. `num_valid` matters more here than on
`lite.osworld`: `EnvBlocked` episodes leave the denominator, so two arms can be averaged over
different task counts. Record it per run and say so before reading a gap.

| | `i4` | `i1` |
|---|---:|---:|
| **base** | | |
| **`gpt5_5`** | | |
| **`gpt5_5` + `<think>`** | | |
| **`qwen3_5_27b`** | | |
| **`qwen3_5_27b` + `<think>`** | | |
```
