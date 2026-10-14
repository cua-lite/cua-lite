# Lite.CUAWorld — Collect With `qwen3_5_27b`

Teacher runbook for the `Qwen/Qwen3.5-27B` rows of Lite.CUAWorld. Dataset-level
setup, staging, upload, and export live in
[`../AGENTS.md`](/devs/data/lite.cuaworld/AGENTS.md); run its §1 first.

This runbook ends at the annotated log roots. They are the only thing the
dataset runbook consumes:

    .data/rollout/lite.cuaworld/qwen3_5_27b/$COMMIT/<software>/train_annotated

## Serve The Model

Unlike `gpt5_5` (an API model), this teacher is local weights and needs a running
sglang server first. `tp_size` is pinned PER MODEL in `LOCAL_AGENTS`
([/lite/agents/factory.py](/lite/agents/factory.py)) — 2 for this one — and the
launcher derives `dp_size = visible_gpus // tp_size`.

**How many GPUs is your call, not this runbook's.** Any multiple of `tp_size`
works, and each extra `tp_size`-sized group is one more replica behind the same
port. Pick from what `nvidia-smi` actually shows free and what the host's other
tenants are doing. The collect step below reads `CUAWORLD_CONCURRENCY` (default 24), which assumes
ONE replica; raise it roughly in step with the replicas you actually got, and watch
the env-server / Docker error rates rather than trusting the number.

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

Readiness is `GET /health` returning 200 — the same probe `lite/infer/serving.py`
polls when it launches a server for you. Bind `127.0.0.1` on shared hosts; the
launcher defaults to `0.0.0.0`.

## Reasoning Channel

Thinking is ON, and it is the only `agent_kwargs` override this runbook makes:

```
--agent-kwargs '{"enable_thinking": true}'
```

The adapter defaults to thinking OFF (`enable_thinking: bool = False`,
[/lite/agents/models/qwen3_vl/adapter.py](/lite/agents/models/qwen3_vl/adapter.py),
inherited by the qwen3_5 adapters), so without the override this teacher would
collect the same Action-only rows `qwen3_8_27b` does.

With it on, the model's native `<think>...</think>` is parsed straight into the
canonical `reasoning_content` FIELD (`lite/agents/models/qwen3_5/adapter.py`), so
**this teacher needs no `internalize_cot.py` pass** — its published rows are
already in the one vocabulary. That is the whole difference from `gpt5_5`, which
is PROMPTED for a `Thought:` line and lands it in an `inline_reasoning` content
part that must be moved before staging.

Keep the override on the command line rather than forking a yaml: the rollout
config stays the shared, unmodified one, and `metadata.others.command` records
the full argv (`_cli_command()`, `lite/infer/rollout.py`), so the override reads
straight off the row. A forked yaml's setting is not on the row — recovering it
means taking the row's `config_path` + `commit` back to the checkout and reading
the file.

## Task Set

Every teacher on this dataset collects the SAME tasks, and each run is
STANDALONE — the set is pinned by the whole registered split behind the same
`--filter`, no `--sample`, not by replaying another teacher's output. Run the
teachers in any order, or at the same time; none waits on the others.

To CHECK coverage after the fact (retries, skips, or an interrupted batch can
still leave them uneven), compare the task ids each run actually produced —
they are the sample directory names, and are also recorded in each trajectory's
`metadata.others.task_id`:

```bash
ids() { find "$1" -name summary.json -path "*/sample_*" \
  | xargs -n1 dirname | xargs -n1 dirname | xargs -n1 basename | sort -u; }
for T in gpt5_5 qwen3_8_27b qwen3_5_27b; do
  printf "%-12s %s\n" "$T" "$(ids ".data/rollout/lite.cuaworld/$T/$COMMIT" | wc -l)"
done
# then diff any pair that disagrees, e.g.
diff <(ids ".data/rollout/lite.cuaworld/gpt5_5/$COMMIT") \
     <(ids ".data/rollout/lite.cuaworld/qwen3_8_27b/$COMMIT")
```

A non-empty diff means one side is short; re-run that side's collect command
(it resumes) rather than narrowing the other.

## Collect

Loop the 38 rolloutable softwares on `train`, each into its own subfolder (stage maps log-roots 1:1
to config names). The rollout `--filter` excludes every task the engine flagged with
`exclude_reason`. Start from 24 for CUAWorld and raise only after the env-server, Docker, and
serving error rates are stable; the host-wide concurrency budget lives in the dataset runbook's
[cost / resources section](/devs/data/lite.cuaworld/AGENTS.md#cost--time-resources--disk-hygiene).

```bash
CUAWORLD_CONCURRENCY="${CUAWORLD_CONCURRENCY:-24}"
for SW in <the 38 rolloutable softwares>; do
  uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.5-27B \
    --agent-kwargs '{"enable_thinking": true}' \
    --sglang-server-url "$SGLANG_URL" \
    --env-id "lite.cuaworld.$SW" \
    --splits train \
    --concurrency "$CUAWORLD_CONCURRENCY" \
    --max-attempts 3 \
    --save-data true \
    --save-video false \
    --save-gif false \
    --config-path scripts/configs/qwen3_5/default/lite.cuaworld.yaml \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --log-root ".data/rollout/lite.cuaworld/qwen3_5_27b/$COMMIT/$SW"
done
```

**Check the token budget on the first batch.** The rollout default is
`max_new_tokens: 2048` (`lite/infer/serving.py`), and no ceiling has been measured for this
teacher: with thinking on the reply carries the whole `<think>` body ahead of the action, a
different and much longer distribution than `qwen3_8_27b`'s. A truncated reply is not silent
— `lite/agents/core/agent/base.py` maps a `finish_reason` of `length` / `max_tokens` /
`context_length_exceeded` to a truncated step — so check the first batch before collecting the
rest. To raise it, extend the SAME `--agent-kwargs` object; it is one flag, so a second one
would drop `enable_thinking` and silently collect Action-only rows:

```
--agent-kwargs '{"enable_thinking": true, "sampling_kwargs": {"max_new_tokens": 4096}}'
```

Record whatever you settle on here, and drop this paragraph once it is measured.

The config pins one software via `env_id`; `--env-id` selects the software per
iteration, exactly as in the `gpt5_5` loop.

Re-run the identical command to resume. If you collect `eval`/`long_horizon` as held-out reference
sets, use a separate explicit software list and a separate log root, and keep those splits out of the
SFT mix.

## Annotate And Review

`filter.py` keeps every trajectory except the `/opt/env/` and OOB-coordinate
hard-drops and writes `metadata.others.exclude_reason` (see
[Shared Filter](/devs/data/lite.cuaworld/AGENTS.md#shared-filter)).

```bash
for SW in <softwares>; do
  uv run python devs/data/lite.osworld/filter.py \
    --log-root ".data/rollout/lite.cuaworld/qwen3_5_27b/$COMMIT/$SW/train" \
    --out     ".data/rollout/lite.cuaworld/qwen3_5_27b/$COMMIT/$SW/train_annotated" \
    --drop-loops --drop-undo-storm
done
```

Same script, same flags as `gpt5_5`. The filter runs on canonical Lite rows —
the adapter has already projected this family's wire format — so it carries no
model-family branch and needs none.

Review the hard-drop counts, the `exclude_reason` tag counts, and sample each
tag class plus clean/terminal trajectories before publishing. The per-software
yield / failure / WAIT-pattern / programmatic-vs-VLM report described in
[`gpt5_5/AGENTS.md`](/devs/data/lite.cuaworld/gpt5_5/AGENTS.md#annotate-and-review)
reads any log root, so point it at this teacher's roots the same way.
