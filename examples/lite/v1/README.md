# Lite v1 — desktop.use teacher-comparison training walkthrough

Train **three checkpoints** — `gpt5_5` and `qwen3_8_27b` head to head, plus a reasoning arm on
`gpt5_5` — and score them on the same eval. Both draw from **Lite.ScaleCUA**, whose
`desktop.use.train` rows are published one shard per teacher, both run the same recipe, and both
are capped at 5000 train trajectories (as in
[README.md](/README.md#sft-any-cua-on-any-datasets)), so between those two the only moving part
is the teacher.
The two 5000-row draws are NOT the same tasks: `episode_return > 0.5` keeps a different subset
per teacher, which IS the teacher effect rather than a confound to control for.

A third checkpoint comes along for free: the **reasoning arm**, `gpt5_5` exported under
`desktop.use.compact.reasoning.yaml` instead of `desktop.use.compact.yaml`. The twin differs by
`enable_thinking` alone, so reading it against the `gpt5_5` Action-only checkpoint isolates
Qwen3.5's native `<think>` channel. It is `gpt5_5`-only — that teacher is prompted for a
`Thought:` line, which [`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py)
canonicalizes into `reasoning_content` before staging, so the published rows carry it;
`qwen3_8_27b` runs thinking off, so the same config would train an empty `<think>` block on it.
No extra preprocessing and no separate root: a thinking-off config strips `reasoning_content` at
the model boundary, so the Action-only exports above are byte-identical to exports from rows
that never carried a Thought.

> **Train and eval must use the same config.** A checkpoint scored under a config it was not
> trained on is measuring a prompt surface it never saw. `$CELL` names each run end to end,
> and the Eval block derives the matching config from it.

### SFT

#### Export

```bash
# --- host ---
OUT=.data/sft/qwen3_5/desktop.use
CFG=examples/lite/v1/configs/qwen3_5
FILTER="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

# One download + one export per teacher. The HF config name IS the shard directory on the Hub
# (`<platform>/<task_type>/<split>/<variant>/shard-*.parquet`), so `--allow-patterns` selects a
# teacher by that same name — the token you would pass to
# `load_dataset("cua-lite/Lite.ScaleCUA", "desktop.use.train.gpt5_5")`.
#
# Each teacher gets its own root, and that costs nothing: image bytes are embedded in the shards
# on the Hub, so a root only ever materializes the images its own shards carry. `download`
# reshapes the Hub layout to the local one and extracts those bytes back into `<root>/images/`,
# which is why the root must keep the canonical `cua-lite/Lite.ScaleCUA/` tail — rows reference
# images as `cua-lite/<Name>/images/...`, resolved against `--image-root`.
#
# `--filter` runs before `--sample`, so the 5000 cap lands on KEPT rows, not on rows read.
# `--overwrite` is what makes this block re-runnable: `download` refuses a non-empty
# `--out` so an earlier subset pull cannot survive into a later one.
for T in gpt5_5 qwen3_8_27b; do
  uv run python -m lite.data.hf.download Lite.ScaleCUA \
    --allow-patterns "desktop/use/train/desktop.use.train.$T/*" \
    --out ".data/huggingface-$T/cua-lite/Lite.ScaleCUA" --overwrite

  uv run python -m lite.train.export.export_sft \
    --config "$CFG/desktop.use.compact.yaml" \
    --model-id Qwen/Qwen3.5-4B \
    --data-paths ".data/huggingface-$T/cua-lite/Lite.ScaleCUA" \
    --image-root ".data/huggingface-$T" \
    --filter "$FILTER" --sample 5000 --seed 42 --no-strict \
    -o "$OUT/scalecua_5k.$T.parquet"
done

# Reasoning arm, gpt5_5 only: same root, filter, sample and seed as its Action-only twin above,
# so the two parquets differ by the <think> block alone.
uv run python -m lite.train.export.export_sft \
  --config "$CFG/desktop.use.compact.reasoning.yaml" \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths ".data/huggingface-gpt5_5/cua-lite/Lite.ScaleCUA" \
  --image-root ".data/huggingface-gpt5_5" \
  --filter "$FILTER" --sample 5000 --seed 42 --no-strict \
  -o "$OUT/scalecua_5k.gpt5_5.reasoning.parquet"

# `--no-strict` makes a conversion failure a SKIP, not an error: a wrong --image-root
# prints "Skipped N rows ... Wrote 0 trajectory rows" and still exits 0, and run_sft.sh
# would then train on an empty parquet. `--sample` is a contract on the OUTPUT -- export
# converts in rounds until 5000 SURVIVE -- so anything but 5000 means it ran out of
# convertible rows, and export said so on its own line. Count them before trusting them.
for CELL in gpt5_5 qwen3_8_27b gpt5_5.reasoning; do
  uv run python -c "
import sys, pyarrow.parquet as pq
n = pq.read_metadata(sys.argv[1]).num_rows
print(('OK   ' if n == 5000 else 'SHORT'), n, sys.argv[1])
" "$OUT/scalecua_5k.$CELL.parquet"
done
```


#### Train

```bash
# --- Slime container ---
# One run per cell, same recipe; only PROMPT_DATA and the three names differ. SFT at TP=2
# (8 GPUs → DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't THD-pack).
# One ckpt per epoch. Run them one after the other: each takes all 8 GPUs.
#
# SAVE_DIR and WANDB_GROUP_SUFFIX are as MANDATORY as SAVE_HF_DIR. run_sft.sh keys both
# checkpoint dirs AND the W&B group off PROMPT_DATA's parent dir, which is `desktop.use` for
# EVERY cell -- so unset, the second run overwrites the first's Megatron checkpoints and lands
# in its W&B group. WANDB_GROUP_SUFFIX has no default; separating them is its job.
#
# Export WANDB_API_KEY before the first cell: run_sft.sh builds its whole W&B argument
# list only when that variable is set, so without it the runs train fine and log
# nowhere -- and WANDB_GROUP_SUFFIX below silently does nothing.
#
# CUA_LITE_TRAIN_BROAD_CLEANUP=1 is what makes the LOOP safe: run_sft.sh starts a Ray head and
# never stops it, and its only teardown (utils/cleanup.sh) is sourced at startup and opt-in.
# Without it the second cell stacks a second cluster on the same 8 GPUs. Set it only in a
# dedicated training container, which is what this block assumes.
# $CELL names the run and every artifact derived from it: the parquet, both checkpoint dirs
# and the W&B group. gpt5_5, qwen3_8_27b, then gpt5_5.reasoning.
for CELL in gpt5_5 qwen3_8_27b gpt5_5.reasoning; do
  TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
    CUA_LITE_TRAIN_BROAD_CLEANUP=1 \
    MODEL_ID=Qwen/Qwen3.5-4B \
    SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
    PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5/desktop.use/scalecua_5k.$CELL.parquet \
    SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-4b/desktop.use/sft.$CELL/iter_{rollout_id} \
    SAVE_DIR=/root/checkpoints/qwen3_5-4b/desktop.use/sft.$CELL/megatron \
    WANDB_GROUP_SUFFIX=".$CELL" \
    bash /workspaces/cua-lite/scripts/train/run_sft.sh
done
```

- `MBS=1` is the safe start (4-image steps are heavy); raise to `MBS=2` only if it fits.
- TP=2 fits the 4-image step at 4B; fall back to `TP_SIZE=4` only if it OOMs.

#### Eval

Four runs — the base model plus the three checkpoints — on the full `lite.osworld` eval split
(328 scored tasks after `--filter`) — the [Lite.OSWorld row](/docs/eval.md#osworld--liteosworld)
of [docs/eval.md](/docs/eval.md). Env setup:
[`lite/gym/envs/lite/osworld/README.md`](/lite/gym/envs/lite/osworld/README.md).

```bash
# --- host ---
unset SGLANG_SERVER_URL                  # else every run silently attaches to that server
CFG=examples/lite/v1/configs/qwen3_5
CKPTS=.ckpts/qwen3_5-4b/desktop.use
LOGS=.logs/rollout/Qwen_Qwen3.5-4B/lite.osworld

# The LAST checkpoint slime wrote. `iter_<N>` is its 0-based rollout index, not the epoch,
# so read it off disk -- and never paste `iter_<N>` literally: bash reads `<N` as a redirect
# and the command silently never runs.
last_iter() { ls -d "$CKPTS/sft.$1"/iter_* 2>/dev/null | sort -V | tail -1; }

# A missing checkpoint must not reach the score loop: `last_iter` prints nothing, the
# ${2:+...} below drops --model-path, and the run would score BASE weights into a slug
# that says sft. Like every check in this file it only PRINTS -- a pasted block cannot
# abort itself -- so read the line and stop by hand before running the rest.
MISSING=
for CELL in gpt5_5 qwen3_8_27b gpt5_5.reasoning; do
  [ -n "$(last_iter "$CELL")" ] || MISSING="$MISSING $CELL"
done
[ -z "$MISSING" ] && echo "checkpoints OK" \
  || echo "STOP -- do not run the score loop: no $CKPTS/sft.<cell>/iter_* for:$MISSING"

gpu=0
score() {  # $1 = config stem, $2 = --model-path ("" = base model), $3 = log slug
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.5-4B ${2:+--model-path "$2"} \
    --env-id lite.osworld --splits eval --concurrency 8 \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --config-path "$CFG/$1.yaml" \
    --log-root "$LOGS/$3" &
  gpu=$((gpu + 1))
}

# The base model, once, under the Action-only config. The reasoning cell needs no base run of
# its own: its comparison partner is the gpt5_5 Action-only checkpoint, same screenshot surface.
score desktop.use.compact "" base

for CELL in gpt5_5 qwen3_8_27b gpt5_5.reasoning; do
  case "$CELL" in *.reasoning) C=desktop.use.compact.reasoning ;; *) C=desktop.use.compact ;; esac
  D="$(last_iter "$CELL")"
  # The slug carries the checkpoint on purpose: rollout RESUMES a log-root sample by sample
  # and never checks which weights wrote it, so scoring a second checkpoint into one slug
  # would silently re-report the first one's numbers.
  score "$C" "$D" "sft.$CELL.$(basename "$D")"
done

wait
```

Score each run from `$LOGS/<slug>/summary.json` → `stats.mean_episode_return` (denominator is
`num_valid`). Four runs: the base model plus the three cells. Read the two Action-only
checkpoints against each other — same config, tasks, seeds and step budget, so the only moving
part there is the teacher. Read `gpt5_5.reasoning` against `gpt5_5` — same teacher and rows, and
the config differs by `enable_thinking` alone, so the moving part is the `<think>` channel. Do
NOT read the reasoning cell against `base`: that would move both at once.
