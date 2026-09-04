# Lite.OSWorld — Collect With `qwen3_8_27b`

Teacher runbook for the `Qwen/Qwen3.8-27B` half of Lite.OSWorld. Dataset-level
setup, staging, upload, and export live in
[`../AGENTS.md`](/devs/data/lite.osworld/AGENTS.md); run its §1 first.

This runbook ends at the annotated log roots. They are the only thing the
dataset runbook consumes:

    .data/rollout/lite.osworld/qwen3_8_27b/$COMMIT/train.{synth,perturb}_annotated

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
polls when it launches a server for you. It answers 503 only while the server is
still starting. Bind `127.0.0.1` on shared hosts; the launcher defaults to
`0.0.0.0`.

## Reasoning Channel

Thinking stays OFF: run `scripts/configs/qwen3_8/default/lite.osworld.yaml`
as-is, with no `agent_kwargs` override. That is the upstream harness default
(`QwenAgent(enable_thinking=False)` in
`${CUA_LITE_REFERENCES_ROOT}/OSWorld/mm_agents/qwen/main.py`), and it is what a
matched A/B on this env measured as the better collection setting — turning it on
bought no success-rate improvement and raised the malformed-output rate.

So the rows this teacher publishes are ACTION trajectories: `action_description`
+ `tool_calls`, no `reasoning_content`. That is a different kind of data from
`gpt5_5`'s, which is why the two are separate configs rather than one pooled set.

## Task Set

Every teacher on this dataset collects the SAME tasks, and each run is
STANDALONE — the set is pinned by the whole registered split behind the same
`--filter`, no `--sample`, not by replaying another teacher's output. Run the
teachers in any order, or at the same time; neither waits on the other.

To CHECK coverage after the fact (retries, skips, or an interrupted batch can
still leave the two uneven), compare the task ids each run actually produced —
they are the sample directory names, and are also recorded in each trajectory's
`metadata.others.task_id`:

```bash
ids() { find "$1" -name summary.json -path "*/sample_*" \
  | xargs -n1 dirname | xargs -n1 dirname | xargs -n1 basename | sort -u; }
diff <(ids ".data/rollout/lite.osworld/gpt5_5/$COMMIT") \
     <(ids ".data/rollout/lite.osworld/qwen3_8_27b/$COMMIT")
```

A non-empty diff means one side is short; re-run that side's collect command
(it resumes) rather than narrowing the other.

## Collect

```bash
for SUB in synth perturb; do
  uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.8-27B \
    --sglang-server-url "$SGLANG_URL" \
    --env-id lite.osworld \
    --splits "train.$SUB" \
    --concurrency 32 \
    --max-attempts 3 \
    --save-data true \
    --save-video false \
    --save-gif false \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --config-path scripts/configs/qwen3_8/default/lite.osworld.yaml \
    --log-root ".data/rollout/lite.osworld/qwen3_8_27b/$COMMIT"
done
```

The config is used unmodified so a re-run reproduces this collection exactly: a
CLI-only `agent_kwargs` override lives nowhere but the shell history, while the
yaml is the frozen record of how these rows were produced. (Export does not read
this config at all -- `export_sft --config` takes the STUDENT's rollout config,
so that training prompts match what the student will see at inference.)

The rollout default of `max_new_tokens: 2048` is ample here: the longest reply
across 962 recorded turns was ~514 tokens, and none were cut off.

This covers all 2,429 registered train tasks (1,722 synth + 707 perturb); the
`--filter` runs the 2,411 runnable (1,704 synth + 707 perturb), skipping the
18 quarantined synth rows. Re-run the same command to resume.

Collect each source into its OWN subfolder (`train.synth` / `train.perturb` are
registered sub-splits): stage maps log-roots 1:1 to config names, so the
`desktop.use.synth.qwen3_8_27b` / `desktop.use.perturb.qwen3_8_27b` separation
depends on keeping them apart here.

## Annotate And Review

```bash
for SUB in synth perturb; do
  uv run python devs/data/lite.osworld/filter.py \
    --log-root ".data/rollout/lite.osworld/qwen3_8_27b/$COMMIT/train.$SUB" \
    --out ".data/rollout/lite.osworld/qwen3_8_27b/$COMMIT/train.${SUB}_annotated" \
    --drop-loops --drop-undo-storm
done
```

Same script, same flags as `gpt5_5`. The filter runs on canonical Lite rows —
the adapter has already projected this family's wire format — so it carries no
model-family branch and needs none.

Review the hard-drop counts, the `exclude_reason` tag counts, and sample every
tag class, plus a sample of clean (untagged) and terminal trajectories, before
publishing.
