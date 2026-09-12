# GRPO Training

All commands run inside the Slime container. For container setup, see [docs/slime.md](/docs/slime.md).

## Contents

- **Grounding**
  - [ScreenSpot-Pro](#screenspot-pro)
- **Desktop**
  - [Lite.OSWorld](#liteosworld)
- **Browser**
  - [WebGym](#webgym)
- **Mobile**
  - [AndroidWorld](#androidworld)
  - [MobileGym](#mobilegym)

## Env-server prerequisite

Slime drives environments through a managed env-server. Before any command
below, the env-server must be running and the container must have
`CUA_LITE_ENV_SERVER_URL` + `CUA_LITE_ENV_SERVER_TOKEN` exported. For
per-environment setup, see [docs/envs.md#installation](/docs/envs.md#installation);
to start the server, see [docs/envs.md#env-server](/docs/envs.md#env-server).

---

## ScreenSpot-Pro

Full task set (1581 tasks) with proper train/eval split (128 random tasks held out for eval).

<details>
<summary>Data</summary>

```bash
# step 1: export all tasks
python -m lite.train.export.export_tasks --env-id screenspot_pro --split eval \
  -o /root/datasets/cua-lite/screenspot_pro/all.parquet

# step 2: split into train + eval
python -m lite.data.split \
  -i /root/datasets/cua-lite/screenspot_pro/all.parquet --eval-size 128 \
  --train-output /root/datasets/cua-lite/screenspot_pro/train.parquet \
  --eval-output /root/datasets/cua-lite/screenspot_pro/eval.parquet
```

</details>

<details>
<summary>Train (qwen3_vl)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=screenspot_pro \
  PROMPT_DATA=/root/datasets/cua-lite/screenspot_pro/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/screenspot_pro/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/screenspot_pro.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=screenspot_pro \
  PROMPT_DATA=/root/datasets/cua-lite/screenspot_pro/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/screenspot_pro/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/screenspot_pro.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh
```

</details>

<details>
<summary>Train (qwen3_5)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  ENV_ID=screenspot_pro \
  PROMPT_DATA=/root/datasets/cua-lite/screenspot_pro/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/screenspot_pro/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/screenspot_pro.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  ENV_ID=screenspot_pro \
  PROMPT_DATA=/root/datasets/cua-lite/screenspot_pro/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/screenspot_pro/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/screenspot_pro.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh
```

</details>

---

## Lite.OSWorld

Full train split (synth + perturb, 2,429 raw tasks; 2,411 after the
`exclude_reason` filter). Eval: 64 sampled from the 328 non-excluded eval tasks
using the same filter as training.

<details>
<summary>Data</summary>

```bash
python -m lite.train.export.export_tasks --env-id lite.osworld --split train \
  -o /root/datasets/cua-lite/lite.osworld/train.parquet \
  --filter "lambda m: not m.others.get('exclude_reason')"
python -m lite.train.export.export_tasks --env-id lite.osworld --split eval --sample 64 \
  -o /root/datasets/cua-lite/lite.osworld/eval.parquet \
  --filter "lambda m: not m.others.get('exclude_reason')"
```

</details>

<details>
<summary>Train (qwen3_vl)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=lite.osworld \
  PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=lite.osworld \
  PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

```

</details>

<details>
<summary>Train (qwen3_5)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  ENV_ID=lite.osworld \
  PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/lite.osworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  ENV_ID=lite.osworld \
  PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/lite.osworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

```

</details>

---

## WebGym

The HuggingFace dataset has native train (292k) / test (1,167 OOD, unseen websites) splits.
Difficulty 1-3 = easy, 4-6 = medium, 7+ = hard. Dev defaults to easy only.

<details>
<summary>Data</summary>

```bash
# easy tasks only (difficulty <= 3), sample 1024 for train
python -m lite.train.export.export_tasks --env-id webgym --split train --sample 1024 \
  --filter "lambda m: m.others.get('difficulty', 0) <= 3" \
  -o /root/datasets/cua-lite/webgym/train.parquet
python -m lite.train.export.export_tasks --env-id webgym --split eval --sample 64 \
  --filter "lambda m: m.others.get('difficulty', 0) <= 3" \
  -o /root/datasets/cua-lite/webgym/eval.parquet
```

</details>

<details>
<summary>Train (qwen3_vl)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=webgym \
  PROMPT_DATA=/root/datasets/cua-lite/webgym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/webgym/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/webgym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=webgym \
  PROMPT_DATA=/root/datasets/cua-lite/webgym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/webgym/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/webgym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

```

</details>

<details>
<summary>Train (qwen3_5)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  ENV_ID=webgym \
  PROMPT_DATA=/root/datasets/cua-lite/webgym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/webgym/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/webgym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  ENV_ID=webgym \
  PROMPT_DATA=/root/datasets/cua-lite/webgym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/webgym/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/webgym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

```

</details>

---

## AndroidWorld

Easy + medium tasks only (86 per split, filtering out hard tasks where small models
have near-zero success). Train split (`perturb_*`) gets random params each reset;
eval split gets deterministic params via `seed=42` (baked into eval task registration).

<details>
<summary>Data</summary>

```bash
# train (perturb_* task IDs, randomized) + eval (original IDs, seeded)
python -m lite.train.export.export_tasks --env-id androidworld --split train \
  --filter "lambda m: m.others.get('complexity') <= 2.0" \
  -o /root/datasets/cua-lite/androidworld/train.parquet
python -m lite.train.export.export_tasks --env-id androidworld --split eval \
  --filter "lambda m: m.others.get('complexity') <= 2.0" \
  -o /root/datasets/cua-lite/androidworld/eval.parquet
```

</details>

<details>
<summary>Train (qwen3_vl)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=androidworld \
  PROMPT_DATA=/root/datasets/cua-lite/androidworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/androidworld/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/androidworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=androidworld \
  PROMPT_DATA=/root/datasets/cua-lite/androidworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/androidworld/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/androidworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

```

</details>

<details>
<summary>Train (qwen3_5)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  ENV_ID=androidworld \
  PROMPT_DATA=/root/datasets/cua-lite/androidworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/androidworld/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/androidworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  ENV_ID=androidworld \
  PROMPT_DATA=/root/datasets/cua-lite/androidworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/androidworld/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/androidworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

```

</details>

<details>
<summary>Train (mai_ui)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Tongyi-MAI/MAI-UI-2B \
  ENV_ID=androidworld \
  PROMPT_DATA=/root/datasets/cua-lite/androidworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/androidworld/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/mai_ui/compact/androidworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Tongyi-MAI/MAI-UI-2B \
  ENV_ID=androidworld \
  PROMPT_DATA=/root/datasets/cua-lite/androidworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/androidworld/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/mai_ui/compact/androidworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

```

</details>

---

## MobileGym

Browser-simulated mobile env (24 apps, deterministic state-diff judge,
Docker, no KVM). Train split = 160 randomized tasks; eval split = 256
deterministic seeded tasks filtered to **L1+L2** (easy + medium) so
small models have a non-trivial baseline. Difficulty is in
`m.others['difficulty']` as `L1/L2/L3/L4`.

**Reward: `reward_shaping: true` in the training config's `env_kwargs`.** MobileGym's
Success Rate is `stop_reason == COMPLETE` AND `judge.success` AND `judge.clean`
(`bench_env/runner/base.py::EpisodeResult.success`) — a binary signal that is too
sparse to bootstrap from. With shaping on, an episode that misses the SR still earns
partial credit for the `check_goals` it passed, capped strictly below 1.0 so a real
success always outranks it. The cap is load-bearing: `progress` looks at neither
`clean` nor COMPLETE, so paying it at face value would let a policy max the reward by
hitting every goal while dirtying state or never calling terminate. Eval leaves the
flag off (`scripts/configs/*/default/mobilegym.yaml`) and reports the SR itself. One
known deviation remains: in `eval_mode: text` a `response` ends the episode and counts
as COMPLETE here, where upstream's `AnswerHandler` leaves `stop_reason` unset and lets
its own runner finish the episode — so Q&A-style tasks can still read slightly high.

<details>
<summary>Data</summary>

```bash
# train: L1+L2 only, randomized per reset (80 tasks)
python -m lite.train.export.export_tasks --env-id mobilegym --split train \
  --filter "lambda m: m.others.get('difficulty') in ('L1', 'L2')" \
  -o /root/datasets/cua-lite/mobilegym/train.parquet
# eval: L1+L2 only, deterministic seeds (93 tasks)
python -m lite.train.export.export_tasks --env-id mobilegym --split eval \
  --filter "lambda m: m.others.get('difficulty') in ('L1', 'L2')" \
  -o /root/datasets/cua-lite/mobilegym/eval.parquet
```

</details>

<details>
<summary>Train (qwen3_vl)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=mobilegym \
  PROMPT_DATA=/root/datasets/cua-lite/mobilegym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/mobilegym/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/mobilegym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=mobilegym \
  PROMPT_DATA=/root/datasets/cua-lite/mobilegym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/mobilegym/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/mobilegym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

```

</details>

<details>
<summary>Train (qwen3_5)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  ENV_ID=mobilegym \
  PROMPT_DATA=/root/datasets/cua-lite/mobilegym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/mobilegym/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/mobilegym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  ENV_ID=mobilegym \
  PROMPT_DATA=/root/datasets/cua-lite/mobilegym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/mobilegym/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/mobilegym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

```

</details>

<details>
<summary>Train (mai_ui)</summary>

```bash
# sync, 2 GPU (colocate), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Tongyi-MAI/MAI-UI-2B \
  ENV_ID=mobilegym \
  PROMPT_DATA=/root/datasets/cua-lite/mobilegym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/mobilegym/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/mai_ui/compact/mobilegym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 2 GPU (1 train + 1 rollout), 2B
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Tongyi-MAI/MAI-UI-2B \
  ENV_ID=mobilegym \
  PROMPT_DATA=/root/datasets/cua-lite/mobilegym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/mobilegym/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/mai_ui/compact/mobilegym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

```

</details>
