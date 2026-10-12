# MobileGym — Collect With `qwen3_8_27b`

Teacher runbook for the `Qwen/Qwen3.8-27B` rows of MobileGym. Dataset-level setup,
staging, upload, and export live in [`../AGENTS.md`](/devs/data/mobilegym/AGENTS.md); run
its §1 first.

This runbook ends at the annotated log root. It is the only thing the dataset runbook
consumes:

    .data/rollout/mobilegym/qwen3_8_27b/$COMMIT/train_annotated

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
  --model-id Qwen/Qwen3.8-27B --host 127.0.0.1 --port 0
```

Export the address the launcher printed before collecting:

```bash
export SGLANG_URL="http://127.0.0.1:<the PORT it printed>"
```

Readiness is `GET /health` returning 200 — the same probe `lite/infer/serving.py` polls
when it launches a server for you. Bind `127.0.0.1` on shared hosts; the launcher
defaults to `0.0.0.0`.

## Reasoning Channel

Thinking stays OFF: run
[`scripts/configs/qwen3_8/default/mobilegym.yaml`](/scripts/configs/qwen3_8/default/mobilegym.yaml)
as-is, with no `agent_kwargs` override. That is the adapter default
(`enable_thinking: bool = False`,
[/lite/agents/models/qwen3_vl/adapter.py](/lite/agents/models/qwen3_vl/adapter.py)), and
it is what makes this teacher the ACTION arm of the three.

So the rows this teacher publishes are action trajectories: `action_description` +
`tool_calls`, no `reasoning_content`. Action-only rows are a different kind of data from
the two reasoning teachers', which is why each teacher is its own config rather than one
pooled set.

No A/B of thinking on-vs-off has been run on this env. The desktop campaign measured one
on `lite.osworld` and kept thinking off there; that result is about that env's prompt and
task shape, so do not read it as measured here. If you want the mobile number, run the
A/B and record it in this section.

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
     <(ids ".data/rollout/mobilegym/qwen3_8_27b/$COMMIT")
```

All three should print 160. That counts TASKS, not draws: with `--group-size 16` each
task directory holds up to 16 `sample_*` dirs, and a short run shows up as a low sample
count inside a task rather than a missing task. A non-empty diff means one side is short;
re-run that side's collect command (it resumes) rather than narrowing the other.

## Collect

```bash
uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.8-27B \
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
  --config-path scripts/configs/qwen3_8/default/mobilegym.yaml \
  --log-root ".data/rollout/mobilegym/qwen3_8_27b/$COMMIT"
```

160 train templates × 16 draws = 2,560 attempted trajectories. Re-run the same command to
resume.

Keep the config unmodified: then the yaml plus the row's `command` is the whole recipe.
The two overrides above are deliberately CLI-only —
`--group-shared-seed false` and `--env-kwargs '{"reward_shaping": true}'` are per-run
switches that all three teachers share, and both are recorded per row because
`rollout.py` writes the full argv into each run's provenance `command` (`_cli_command()`
prefixes the interpreter, then `shlex.join(sys.argv)`). CLI `env_kwargs` deep-merge OVER
the yaml's per leaf, so the config's `loop_detect` and `extra_tools` survive. See
[Sampling](/devs/data/mobilegym/AGENTS.md#sampling-16-draws-per-template-all-different)
and [Reward And The Quality Gate](/devs/data/mobilegym/AGENTS.md#reward-and-the-quality-gate).
(Export does not read this config at all — `export_sft --config` takes the STUDENT's
rollout config, so that training prompts match what the student will see at inference.)

**Check the token budget on the first batch.** The rollout default is
`max_new_tokens: 2048` (`lite/infer/serving.py`). Nothing has measured this model's reply
length on this env; the desktop campaign's ~514-token ceiling was measured on
`lite.osworld` and does not carry over. A truncated reply is not silent —
`lite/agents/core/agent/base.py` maps a `finish_reason` of `length` / `max_tokens` /
`context_length_exceeded` to a truncated step — so check the first batch, then record the
observed ceiling here and drop this paragraph.

## Annotate And Review

```bash
uv run python devs/data/mobilegym/filter.py \
  --log-root ".data/rollout/mobilegym/qwen3_8_27b/$COMMIT/train" \
  --out ".data/rollout/mobilegym/qwen3_8_27b/$COMMIT/train_annotated" \
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
