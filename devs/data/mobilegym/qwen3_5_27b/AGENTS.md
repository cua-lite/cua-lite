# MobileGym — Collect With `qwen3_5_27b`

Teacher runbook for the `Qwen/Qwen3.5-27B` rows of MobileGym. Dataset-level setup,
staging, upload, and export live in [`../AGENTS.md`](/devs/data/mobilegym/AGENTS.md); run
its §1 first.

This runbook ends at the annotated log root. It is the only thing the dataset runbook
consumes:

    .data/rollout/mobilegym/qwen3_5_27b/$COMMIT/train_annotated

## Serve The Model

Unlike `gpt5_5` (an API model), this teacher is local weights and needs a running sglang
server first. `tp_size` is pinned PER MODEL in `LOCAL_AGENTS`
([/lite/agents/factory.py](/lite/agents/factory.py)) — 2 for this one — and the launcher
derives `dp_size = visible_gpus // tp_size`.

**How many GPUs is your call, not this runbook's.** Any multiple of `tp_size` works, and
each extra `tp_size`-sized group is one more replica behind the same port. Pick from what
`nvidia-smi` actually shows free and what the host's other tenants are doing. The collect
step below starts at `--concurrency 128`, which is sized for the env container's browser
pool rather than for one sglang replica; raise or lower it in step with the replicas you
actually got, and watch the env-server error rate rather than trusting the number.

```bash
# Pick FREE devices; the COUNT must be a multiple of tp_size (2 here).
# --port 0 picks a free port and prints PORT=<actual> on the first line.
CUDA_VISIBLE_DEVICES=<free devices> uv run python scripts/serve_sglang.py \
  --model-id Qwen/Qwen3.5-27B --host 127.0.0.1 --port 0
```

Export the address the launcher printed before collecting:

```bash
export SGLANG_URL="http://127.0.0.1:<the PORT it printed>"
```

Readiness is `GET /health` returning 200 — the same probe `lite/infer/serving.py` polls
when it launches a server for you. Bind `127.0.0.1` on shared hosts; the launcher
defaults to `0.0.0.0`.

## Reasoning Channel

Thinking is ON, and it is the only `agent_kwargs` override this runbook makes:

```
--agent-kwargs '{"enable_thinking": true}'
```

The adapter defaults to thinking OFF (`enable_thinking: bool = False`,
[/lite/agents/models/qwen3_vl/adapter.py](/lite/agents/models/qwen3_vl/adapter.py),
inherited by the qwen3_5 adapters), so without the override this teacher would collect
the same Action-only rows `qwen3_8_27b` does.

With it on, the model's native `<think>...</think>` is parsed straight into the canonical
`reasoning_content` FIELD ([`lite/agents/models/qwen3_5/adapter.py`](/lite/agents/models/qwen3_5/adapter.py)),
so **this teacher needs no `internalize_cot.py` pass** — its published rows are already
in the one vocabulary. That is the whole difference from `gpt5_5`, which is PROMPTED for
a `Thought:` line and lands it in an `inline_reasoning` content part that must be moved
before staging.

Keep the override on the command line rather than forking a yaml: the rollout config
stays the shared, unmodified one, and `metadata.others.command` records the full argv
(`_cli_command()`, [`lite/infer/rollout.py`](/lite/infer/rollout.py)), so the override
reads straight off the row. A forked yaml's setting is not on the row — recovering it
means taking the row's `config_path` + `commit` back to the checkout and reading the
file.

## Task Set

Every teacher on this dataset collects the SAME tasks with the SAME sampling, and each
run is STANDALONE — the set is the whole registered `train` split, no `--filter` and no
`--sample`, not a replay of another teacher's output. Run the teachers in any order, or
at the same time; none waits on the others.

To CHECK coverage after the fact (retries, skips, or an interrupted batch can still leave
them uneven), compare the task ids each run actually produced — they are the sample
directory names, and are also recorded in each trajectory's `metadata.others.task_id`:

```bash
ids() { find "$1" -name summary.json -path "*/sample_*" \
  | xargs -n1 dirname | xargs -n1 dirname | xargs -n1 basename | sort -u; }
for T in gpt5_5 qwen3_8_27b qwen3_5_27b; do
  printf "%-12s %s\n" "$T" "$(ids ".data/rollout/mobilegym/$T/$COMMIT" | wc -l)"
done
# then diff any pair that disagrees, e.g.
diff <(ids ".data/rollout/mobilegym/gpt5_5/$COMMIT") \
     <(ids ".data/rollout/mobilegym/qwen3_5_27b/$COMMIT")
```

All three should print 160. That counts TASKS, not draws: with `--group-size 16` each
task directory holds up to 16 `sample_*` dirs, and a short run shows up as a low sample
count inside a task rather than a missing task. A non-empty diff means one side is short;
re-run that side's collect command (it resumes) rather than narrowing the other.

## Collect

```bash
uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-27B \
  --agent-kwargs '{"enable_thinking": true}' \
  --sglang-server-url "$SGLANG_URL" \
  --env-id mobilegym \
  --splits train \
  --group-size 16 \
  --group-shared-seed false \
  --env-kwargs '{"reward_shaping": true}' \
  --concurrency 128 \
  --max-attempts 3 \
  --save-data true \
  --save-video false \
  --save-gif false \
  --config-path scripts/configs/qwen3_5/default/mobilegym.yaml \
  --log-root ".data/rollout/mobilegym/qwen3_5_27b/$COMMIT"
```

160 train templates × 16 draws = 2,560 attempted trajectories. Re-run the same command to
resume.

`--group-shared-seed false` and `--env-kwargs '{"reward_shaping": true}'` are dataset
policy, identical across all three teachers — see
[Sampling](/devs/data/mobilegym/AGENTS.md#sampling-16-draws-per-template-all-different)
and [Reward And The Quality Gate](/devs/data/mobilegym/AGENTS.md#reward-and-the-quality-gate).
CLI `env_kwargs` deep-merge OVER the yaml's per leaf, so the config's `loop_detect` and
`extra_tools` survive.

**Check the token budget on the first batch.** The rollout default is
`max_new_tokens: 2048` (`lite/infer/serving.py`), and with thinking on the reply carries
the whole `<think>` body ahead of the action. Nothing has measured that distribution on
this env yet. A truncated reply is not silent — `lite/agents/core/agent/base.py` maps a
`finish_reason` of `length` / `max_tokens` / `context_length_exceeded` to a truncated
step — so check the first batch before collecting the rest. To raise it, extend the SAME
`--agent-kwargs` object; it is one flag, so a second one would drop `enable_thinking` and
silently collect Action-only rows:

```
--agent-kwargs '{"enable_thinking": true, "sampling_kwargs": {"max_new_tokens": 4096}}'
```

Record whatever you settle on here, and drop this paragraph once it is measured.

## Annotate And Review

```bash
uv run python devs/data/mobilegym/filter.py \
  --log-root ".data/rollout/mobilegym/qwen3_5_27b/$COMMIT/train" \
  --out ".data/rollout/mobilegym/qwen3_5_27b/$COMMIT/train_annotated" \
  --drop-loops
```

Same script, same flags as `gpt5_5`. The filter runs on canonical Lite rows — the adapter
has already projected this family's wire format — so it carries no model-family branch
and needs none.

Review the hard-drop counts, the `exclude_reason` tag counts, and sample every tag class,
plus a sample of clean (untagged) and terminal trajectories, before publishing. Also
record the reward histogram (how many rows at `1.0`, in `[0.30, 0.5]`, below the gate) —
that distribution is what the campaign's gate choice rests on, and it is the number most
likely to differ between teachers.
