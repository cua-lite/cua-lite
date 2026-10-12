# Lite.CUAGym — Collect With `qwen3_8_27b`

Teacher runbook for the `Qwen/Qwen3.8-27B` rows of Lite.CUAGym. Dataset-level
setup, staging, upload, and export live in
[`../AGENTS.md`](/devs/data/lite.cuagym/AGENTS.md); run its §1 first.

This runbook ends at the annotated log root. It is the only thing the dataset
runbook consumes:

    .data/rollout/lite.cuagym/qwen3_8_27b/$COMMIT/{browser,desktop}/train_annotated

## Serve The Model

Unlike `gpt5_5` (an API model), this teacher is local weights and needs a running
sglang server first. `tp_size` is pinned PER MODEL in `LOCAL_AGENTS`
([/lite/agents/factory.py](/lite/agents/factory.py)) — 2 for this one — and the
launcher derives `dp_size = visible_gpus // tp_size`.

**How many GPUs is your call, not this runbook's.** Any multiple of `tp_size`
works, and each extra `tp_size`-sized group is one more replica behind the same
port. Pick from what `nvidia-smi` actually shows free and what the host's other
tenants are doing. The collect step below starts at `--concurrency 15`, which
assumes ONE replica; raise it roughly in step with the replicas you actually got,
and watch the env-server / Docker error rates rather than trusting the number.

```bash
# Pick FREE devices; the COUNT must be a multiple of tp_size (2 here).
# --port 0 picks a free port and prints PORT=<actual> on the first line.
CUDA_VISIBLE_DEVICES=<free devices> uv run python scripts/serve_sglang.py \
  --model-id Qwen/Qwen3.8-27B --host 127.0.0.1 --port 0
```

Export the address the launcher printed before collecting:

```bash
export SGLANG_URL="http://127.0.0.1:<the PORT it printed>"
```

Readiness is `GET /health` returning 200 — the same probe `lite/infer/serving.py`
polls when it launches a server for you. Bind `127.0.0.1` on shared hosts; the
launcher defaults to `0.0.0.0`.

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
+ `tool_calls`, no `reasoning_content`. Action-only rows are a different kind of
data from the two reasoning teachers', which is why each teacher is its own config
rather than one pooled set.

## Task Set

Every teacher on this dataset collects the SAME tasks, and each run is
STANDALONE — the set is pinned by the same frozen `--prompt-data` parquet, not
by replaying another teacher's output. Run the teachers in any order, or at the
same time; none waits on the others.

To CHECK coverage after the fact (retries, skips, or an interrupted batch can
still leave them uneven), compare the task ids each run actually produced —
they are the sample directory names, and are also recorded in each trajectory's
`metadata.others.task_id`:

```bash
ids() { find "$1" -name summary.json -path "*/sample_*" \
  | xargs -n1 dirname | xargs -n1 dirname | xargs -n1 basename | sort -u; }
for T in gpt5_5 qwen3_8_27b qwen3_5_27b; do
  printf "%-12s %s\n" "$T" "$(ids ".data/rollout/lite.cuagym/$T/$COMMIT" | wc -l)"
done
# then diff any pair that disagrees, e.g.
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

Keep the config unmodified: then the yaml plus the row's `command` is the whole recipe.
Part of this dataset's published rows are the counterexample — see
[Step Timeout Under I/O Contention](#step-timeout-under-io-contention) for the override
they were collected with, and why it belongs on the command line. A CLI override is
still recorded — `rollout.py` writes the full argv into each run's
provenance `command` (`_cli_command()` prefixes the interpreter, then
`shlex.join(sys.argv)`) — but the yaml is the one place a reader finds the recipe
without digging through per-run metadata. (Export does not read this config at all —
`export_sft --config` takes the STUDENT's rollout config, so that training prompts
match what the student will see at inference.)

For a fresh sampled pool, freeze `--sample`, `--seed`, and the resolved task
IDs. Re-run the identical command to resume.

Collect this teacher into its OWN log root
(`.data/rollout/lite.cuagym/qwen3_8_27b/`): stage maps log-roots 1:1 to config
names, so the per-teacher config separation depends on keeping them apart here.

## Step Timeout Under I/O Contention

The effective ceiling is 120s: this env pins `step_timeout: 120.0` in its own
`make_kwargs` (`lite/gym/envs/lite/<env>/configs/default.yaml`), which is also what
`_WRAPPER_KWARG_DEFAULTS` in `lite/gym/registry.py` would give. The rollout config above
does not touch it. That is right for a quiet host. It is not right when a co-tenant saturates the array the
env containers live on: on 2026-09-06 `/srv` sat at 87-93% util, `docker exec`
stalled 75-326s, and every stall past 120s killed the whole episode rather than the
step -- stuck-step rate went 0.78% -> 8.5%. Raising the ceiling past the longest
observed stall fixed it (0.4%, and throughput 15 -> 360 traj/h):

```
--env-kwargs '{"step_timeout": 400}'
```

Pass it on the command line when `iostat` says the array is contended; do NOT bake
it into the shared yaml, and do not fork the yaml into a scratch directory. Part of
this dataset's published `qwen3_8_27b` rows were collected through such a fork, so
their `metadata.others.config_path` names a path that no longer exists -- the
override above is the only surviving record of what it changed. A CLI override
lands in `metadata.others.command` instead, which travels with the row.

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
