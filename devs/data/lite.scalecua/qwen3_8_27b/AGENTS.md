# Lite.ScaleCUA — Collect With `qwen3_8_27b`

Teacher runbook for the `Qwen/Qwen3.8-27B` half of Lite.ScaleCUA. Dataset-level
setup, staging, upload, and export live in
[`../AGENTS.md`](/devs/data/lite.scalecua/AGENTS.md); run its §1 first.

This runbook ends at the annotated log roots. They are the only thing the
dataset runbook consumes:

    .data/rollout/lite.scalecua/qwen3_8_27b/$COMMIT/{rl,train}_annotated

## Serve The Model

Unlike `gpt5_5` (an API model), this teacher is local weights and needs a
running sglang server first. `tp_size=2` comes from `LOCAL_AGENTS` in
`lite/agents/factory.py`; the launcher derives `dp_size = visible_gpus //
tp_size`, so two visible GPUs give one replica.

```bash
# GPU pair is illustrative — pick any two FREE devices (nvidia-smi). What matters
# is that exactly TWO are visible: tp_size=2 comes from LOCAL_AGENTS, and the
# launcher derives dp_size = visible // tp, so two GPUs give one replica.
# --port 0 picks a free port and prints PORT=<actual> on the first line.
CUDA_VISIBLE_DEVICES=<GPU_A>,<GPU_B> uv run python scripts/serve_sglang.py \
  --model-id Qwen/Qwen3.8-27B --host 127.0.0.1 --port 0
```

Export the address the launcher printed before collecting:

```bash
export SGLANG_URL="http://127.0.0.1:<the PORT it printed>"
```

Readiness is `GET /health` returning 200 — the same probe `lite/infer/serving.py`
polls when it launches a server for you. Bind `127.0.0.1` on shared hosts; the
launcher defaults to `0.0.0.0`.

## Reasoning Channel

Thinking stays OFF: run the rollout config below as-is, with no
`agent_kwargs` override. That is the upstream harness default
(`QwenAgent(enable_thinking=False)` in
`${CUA_LITE_REFERENCES_ROOT}/OSWorld/mm_agents/qwen/main.py`), and it is what a
matched A/B on this env measured as the better collection setting — turning it on
bought no success-rate improvement and raised the malformed-output rate.

So the rows this teacher publishes are ACTION trajectories: `action_description`
+ `tool_calls`, no `reasoning_content`. That is a different kind of data from
`gpt5_5`'s, which is why the two are separate configs rather than one pooled set.

## Task Set

Every teacher on this dataset collects the SAME tasks, and each run is
STANDALONE — the set is pinned by the same `--sample` count (`$TRAIN_N`, derived in
§1 of the dataset runbook) against the same `--seed` (`--seed` drives subset
selection, so an identical seed redraws an identical subset), not by replaying
another teacher's output. On `train` that count equals the runnable total, so both
teachers get every task and only their execution order is drawn. Run the teachers
in any order, or at the same time; neither waits on the other.

To CHECK coverage after the fact (retries, skips, or an interrupted batch can
still leave the two uneven), compare the task ids each run actually produced —
they are the sample directory names, and are also recorded in each trajectory's
`metadata.others.task_id`:

```bash
ids() { find "$1" -name summary.json -path "*/sample_*" \
  | xargs -n1 dirname | xargs -n1 dirname | xargs -n1 basename | sort -u; }
diff <(ids ".data/rollout/lite.scalecua/gpt5_5/$COMMIT") \
     <(ids ".data/rollout/lite.scalecua/qwen3_8_27b/$COMMIT")
```

A non-empty diff means one side is short; re-run that side's collect command
(it resumes) rather than narrowing the other.

## Collect

Collect each source into its own subfolder (`rl` / `train`), keeping HF config
names separable at stage time.

```bash
uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.8-27B \
  --sglang-server-url "$SGLANG_URL" \
  --env-id lite.scalecua \
  --splits rl \
  --filter "$TASK_FILTER" \
  --concurrency 32 \
  --max-attempts 2 \
  --save-data true \
  --save-video false \
  --save-gif false \
  --config-path scripts/configs/qwen3_8/default/lite.osworld.yaml \
  --log-root ".data/rollout/lite.scalecua/qwen3_8_27b/$COMMIT"

uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.8-27B \
  --sglang-server-url "$SGLANG_URL" \
  --env-id lite.scalecua \
  --splits train \
  --filter "$TASK_FILTER" \
  --sample "$TRAIN_N" \
  --concurrency 32 \
  --max-attempts 2 \
  --save-data true \
  --save-video false \
  --save-gif false \
  --config-path scripts/configs/qwen3_8/default/lite.osworld.yaml \
  --log-root ".data/rollout/lite.scalecua/qwen3_8_27b/$COMMIT"
```

There is no `scripts/configs/qwen3_8/default/lite.scalecua.yaml`, for the same
reason the Export section of the dataset runbook gives for `qwen3_5`:
`lite.scalecua` is an OSWorld-task adapter riding the `lite.osworld` desktop
substrate, so the family's `default/lite.osworld.yaml` is the matching rollout
config and `--env-id lite.scalecua` selects the task catalog.

`--sample "$TRAIN_N"` shuffles the whole runnable `train` set so any partial or
resumed run stays domain-balanced; the rationale is spelled out in
[`gpt5_5/AGENTS.md`](/devs/data/lite.scalecua/gpt5_5/AGENTS.md). (`rl` is small
enough to leave in catalog order.)

Re-run the identical command to resume.

## Annotate And Review

```bash
uv run python devs/data/lite.osworld/filter.py \
  --log-root ".data/rollout/lite.scalecua/qwen3_8_27b/$COMMIT/rl" \
  --out ".data/rollout/lite.scalecua/qwen3_8_27b/$COMMIT/rl_annotated" \
  --drop-loops --drop-undo-storm

uv run python devs/data/lite.osworld/filter.py \
  --log-root ".data/rollout/lite.scalecua/qwen3_8_27b/$COMMIT/train" \
  --out ".data/rollout/lite.scalecua/qwen3_8_27b/$COMMIT/train_annotated" \
  --drop-loops --drop-undo-storm
```

Same script, same flags as `gpt5_5` (see
[Shared Filter](/devs/data/lite.scalecua/AGENTS.md#shared-filter)). The filter
runs on canonical Lite rows — the adapter has already projected this family's
wire format — so it carries no model-family branch and needs none.

Review the `exclude_reason` tag counts and sample every tag class, plus a sample
of clean (untagged) and terminal trajectories, before publishing.
