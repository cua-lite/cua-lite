# Lite v1 — desktop.use teacher-comparison training walkthrough

## Desktop

Train **two checkpoints on the same tasks from two different teachers** — `gpt5_5` and
`qwen3_8_27b` — and compare them head to head on the same eval. Both draw from
**Lite.ScaleCUA**, whose `desktop.use.train` rows are published one shard per teacher, and both
are capped at 5000 train trajectories (as in
[README.md](/README.md#sft-any-cua-on-any-datasets)) so the only moving part is the teacher.

### SFT

#### Export

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
OUT=.data/sft/qwen3_5/desktop.use
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
# `--filter` runs before `--sample`, so the 5000 cap lands per teacher.
# `--overwrite` is what makes this block re-runnable: `download` refuses a non-empty
# `--out` so an earlier subset pull cannot survive into a later one.
for T in gpt5_5 qwen3_8_27b; do
  uv run python -m lite.data.hf.download Lite.ScaleCUA \
    --allow-patterns "desktop/use/train/desktop.use.train.$T/*" \
    --out ".data/huggingface-$T/cua-lite/Lite.ScaleCUA" --overwrite

  uv run python -m lite.train.export.export_sft \
    --config examples/lite/v1/configs/qwen3_5/desktop.use.compact.yaml \
    --model-id Qwen/Qwen3.5-4B \
    --data-paths ".data/huggingface-$T/cua-lite/Lite.ScaleCUA" \
    --image-root ".data/huggingface-$T" \
    --filter "$FILTER" --sample 5000 --seed 42 --no-strict \
    -o "$OUT/scalecua_5k.$T.parquet"
done
```


#### Train

```bash
# --- Slime container ---
# One run per teacher, same recipe, only PROMPT_DATA and SAVE_HF_DIR differ. SFT at TP=2
# (8 GPUs → DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't THD-pack).
# One ckpt per epoch. Run them one after the other: each takes all 8 GPUs.
for T in gpt5_5 qwen3_8_27b; do
  TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
    MODEL_ID=Qwen/Qwen3.5-4B \
    SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
    PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5/desktop.use/scalecua_5k.$T.parquet \
    SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-4b/desktop.use/sft.$T/iter_{rollout_id} \
    bash /workspaces/cua-lite/scripts/train/run_sft.sh
done
```

- `MBS=1` is the safe start (4-image steps are heavy); raise to `MBS=2` only if it fits.
- TP=2 fits the 4-image step at 4B; fall back to `TP_SIZE=4` only if it OOMs.

#### Eval

Three runs — base model and both SFT checkpoints — on the full `lite.osworld` eval split
(332 scored tasks after `--filter`) — the [Lite.OSWorld row](/docs/eval.md#osworld--liteosworld)
of [docs/eval.md](/docs/eval.md). Env setup:
[`lite/gym/envs/lite/osworld/README.md`](/lite/gym/envs/lite/osworld/README.md).

```bash
# --- host ---  (replace iter_<N> with the saved iter, e.g. iter_369 for epoch 3)

# base (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --env-id lite.osworld --splits eval --concurrency 8 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/desktop.use.compact.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/base &

# SFT on gpt5_5 data (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B --model-path .ckpts/qwen3_5-4b/desktop.use/sft.gpt5_5/iter_<N> \
  --env-id lite.osworld --splits eval --concurrency 8 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/desktop.use.compact.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/sft.gpt5_5 &

# SFT on qwen3_8_27b data (GPU 2)
CUDA_VISIBLE_DEVICES=2 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B --model-path .ckpts/qwen3_5-4b/desktop.use/sft.qwen3_8_27b/iter_<N> \
  --env-id lite.osworld --splits eval --concurrency 8 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/desktop.use.compact.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/sft.qwen3_8_27b &

wait
```

Score each run from `.logs/rollout/<model_slug>/<env_id>/<role>/summary.json` →
`stats.mean_episode_return` (denominator is `num_valid`). All three runs use the same tasks,
seeds, and step budget, so the only moving part is the teacher the checkpoint was trained on.

<!--
## Ablations

The reasoning ablation: train a checkpoint with the same recipe, only swapping
`desktop.use.compact.yaml` for `desktop.use.compact.reasoning.yaml` (reasoning in Qwen3.5's
native `<think>` channel). Everything below mirrors [SFT](#sft) step for step, so it yields one
more checkpoint comparable to its Action-only counterpart.

**This ablation applies to `gpt5_5` only.** It is the one teacher whose rows carry a Thought:
`gpt5_5` is prompted for a `Thought:` line and stores it as an `inline_reasoning` part, while
`qwen3_8_27b` runs with thinking off and stores only `action_description` + `tool_calls`
(see [the dataset runbook](/devs/data/lite.scalecua/AGENTS.md)). Running internalize-CoT on the
`qwen3_8_27b` root is a no-op, and exporting it under a thinking-enabled config would train an
empty `<think>` block.

One flow difference, and it is forced: `desktop.use.compact.reasoning.yaml` reasons in the
`<think>` channel, so the Thought must first be moved out of `inline_reasoning` and into
`reasoning_content` — an extra internalize-CoT step producing a `.think` copy, which this flow
exports from. Exporting `.think` data with thinking off fails fast in the other direction
(`SFT prompt/target boundary broke (enable_thinking=False)`): the internalized reasoning lives
in the target, so the prompt/target boundary no longer holds.

### SFT (Reasoning)

#### Export (Reasoning)

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
OUT=.data/sft/qwen3_5-reasoning/desktop.use
FILTER="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

# The gpt5_5 root is already downloaded by the SFT flow above. Internalize CoT into a sibling
# `.think` root, then export from that. Images stay path-referenced, so the `.think` copy is
# tiny — but its rows still resolve images against the ORIGINAL root, which is why
# `--image-root` keeps pointing there.
T=gpt5_5

uv run python examples/lite/v1/internalize_cot.py \
  --in  ".data/huggingface-$T/cua-lite/Lite.ScaleCUA" \
  --out ".data/huggingface-$T/cua-lite/Lite.ScaleCUA.think"

uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/desktop.use.compact.reasoning.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths ".data/huggingface-$T/cua-lite/Lite.ScaleCUA.think" \
  --image-root ".data/huggingface-$T" \
  --filter "$FILTER" --sample 5000 --seed 42 \
  -o "$OUT/scalecua_5k.$T.parquet"
```

#### Train (Reasoning)

```bash
# --- Slime container ---
# Same recipe as the Action-only runs, only PROMPT_DATA and SAVE_HF_DIR differ. SFT at TP=2
# (8 GPUs → DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't THD-pack).
# One ckpt per epoch. A single run: this ablation has one teacher.
TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5-reasoning/desktop.use/scalecua_5k.gpt5_5.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/desktop.use/sft.gpt5_5/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

- `MBS=1` is the safe start (4-image steps are heavy); raise to `MBS=2` only if it fits.
- TP=2 fits the 4-image step at 4B; fall back to `TP_SIZE=4` only if it OOMs.

#### Eval (Reasoning)

One run — `gpt5_5` is the only teacher this ablation has — on the same `lite.osworld` eval
split as [Eval](#eval). The base model is already scored there; it is not re-run.

```bash
# --- host ---  (replace iter_<N> with the saved iter, e.g. iter_369 for epoch 3)

# reasoning SFT on gpt5_5 data (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --model-path .ckpts/qwen3_5-reasoning-4b/desktop.use/sft.gpt5_5/iter_<N> \
  --env-id lite.osworld --splits eval --concurrency 8 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/desktop.use.compact.reasoning.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/sft.reasoning.gpt5_5
```

Compare its `stats.mean_episode_return` against the `gpt5_5` Action-only checkpoint from
[Eval](#eval). Both use the same tasks, seeds, and step budget, so the only moving part is the
`<think>` channel. Do not compare runs that use different task sets, sampling
configurations, or task budgets.
-->
