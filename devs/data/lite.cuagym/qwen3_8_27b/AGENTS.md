# Lite.CUAGym — Collect With `qwen3_8_27b`

Teacher runbook for the `Qwen/Qwen3.8-27B` half of Lite.CUAGym. Dataset-level
setup, staging, upload, and export live in
[`../AGENTS.md`](/devs/data/lite.cuagym/AGENTS.md); run its §1 first.

This runbook ends at the annotated log root. It is the only thing the dataset
runbook consumes:

    .data/rollout/lite.cuagym/qwen3_8_27b/$COMMIT/{browser,desktop}/train_annotated

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

The CUA-Gym reward judge is a separate API route from the teacher policy: it
still needs the `OPENAI_API_KEY` / `LITE_CUAGYM_JUDGE_*` settings from the
dataset runbook's §1, even though the policy itself is served locally.

## Reasoning Channel

Thinking stays OFF: run `scripts/configs/qwen3_8/default/lite.cuagym.yaml`
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
STANDALONE — the set is pinned by the same frozen `--prompt-data` parquet, not
by replaying another teacher's output. Run the teachers in any order, or at the
same time; neither waits on the other.

To CHECK coverage after the fact (retries, skips, or an interrupted batch can
still leave the two uneven), compare the task ids each run actually produced —
they are the sample directory names, and are also recorded in each trajectory's
`metadata.others.task_id`:

```bash
ids() { find "$1" -name summary.json -path "*/sample_*" \
  | xargs -n1 dirname | xargs -n1 dirname | xargs -n1 basename | sort -u; }
diff <(ids ".data/rollout/lite.cuagym/gpt5_5/$COMMIT") \
     <(ids ".data/rollout/lite.cuagym/qwen3_8_27b/$COMMIT")
```

A non-empty diff means one side is short; re-run that side's collect command
(it resumes) rather than narrowing the other.

## Collect

`--prompt-data` and `--filter` are mutually exclusive (`lite/infer/rollout.py`
rejects the pair), so the per-platform split cannot happen at rollout time — it
has to be baked into the frozen inputs. A prompt-data row's `metadata` carries
only `env_key` / `split` / `env_kwargs` (see `_collect_tasks_from_parquet`), NOT
the routing `dims`, and this env's task ids are UUIDs that do not encode their
platform — so the platform has to come from the registry:

```bash
# once, before collection — split the frozen pool by registered platform
uv run python - "$CUAGYM_INPUT" <<'PY'
import sys
import pandas as pd
import lite.gym as gym

# task_ids() returns {split: [task_id, ...]}; task_metadata() gives the routing
# dims, whose first element is the platform.
plat_of = {
    tid: gym.registry.task_metadata("lite.cuagym", tid).dims[0]
    for tids in gym.registry.task_ids("lite.cuagym").values()
    for tid in tids
}
df = pd.read_parquet(sys.argv[1])
plat = df["metadata"].map(lambda m: plat_of[m["env_key"].split("@", 1)[1]])
for p in ("browser", "desktop"):
    out = sys.argv[1].replace(".parquet", f".{p}.parquet")
    df[plat == p].to_parquet(out)
    print(f"{p}: {(plat == p).sum()} rows -> {out}")
PY
```

Splitting the ALREADY-FROZEN pool keeps whatever task-level filtering went into
it (the `exclude_reason` gate above), so both halves stay publishable.


```bash
# One run per platform: stage needs a log root per config, and the two
# platforms are two configs.
for PLAT in browser desktop; do
  uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.8-27B \
    --sglang-server-url "$SGLANG_URL" \
    --env-id lite.cuagym \
    --prompt-data "${CUAGYM_INPUT%.parquet}.$PLAT.parquet" \
    --concurrency 15 \
    --max-attempts 3 \
    --save-data true \
    --save-video false \
    --save-gif false \
    --config-path scripts/configs/qwen3_8/default/lite.cuagym.yaml \
    --log-root ".data/rollout/lite.cuagym/qwen3_8_27b/$COMMIT/$PLAT"
done
```

The config is used unmodified so a re-run reproduces this collection exactly: a
CLI-only `agent_kwargs` override lives nowhere but the shell history, while the
yaml is the frozen record of how these rows were produced. (Export does not read
this config at all — `export_sft --config` takes the STUDENT's rollout config, so
that training prompts match what the student will see at inference.)

For a fresh sampled pool, freeze `--sample`, `--seed`, and the resolved task
IDs. Re-run the identical command to resume.

Collect this teacher into its OWN log root
(`.data/rollout/lite.cuagym/qwen3_8_27b/`): stage maps log-roots 1:1 to config
names, so the `gpt5_5` / `qwen3_8_27b` separation depends on keeping them apart
here.

## Annotate And Review

```bash
for PLAT in browser desktop; do
  uv run python devs/data/lite.osworld/filter.py \
    --log-root ".data/rollout/lite.cuagym/qwen3_8_27b/$COMMIT/$PLAT/train" \
    --out ".data/rollout/lite.cuagym/qwen3_8_27b/$COMMIT/$PLAT/train_annotated" \
    --drop-loops --drop-undo-storm
done
```

`split` is read PER ROW (`TaskSpec(tid, eid, split or "parquet", ekw)`), so
`$COMMIT/$PLAT/train` holds every frozen row carrying `split: "train"` and only
the rows that omit the field land under `$COMMIT/$PLAT/parquet/` (see
[/lite/infer/rollout.py](/lite/infer/rollout.py)), so freeze the parquet with an
explicit `train` split on every row.

Same script, same flags as `gpt5_5`. The filter runs on canonical Lite rows —
the adapter has already projected this family's wire format — so it carries no
model-family branch and needs none.

Review the `exclude_reason` tag counts and sample every tag class, plus a sample
of clean (untagged) and terminal trajectories, before publishing.
