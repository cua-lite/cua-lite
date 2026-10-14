# WebGym — Collect With `qwen3_5_27b`

Teacher runbook for the `Qwen/Qwen3.5-27B` rows of WebGym. Dataset-level targets,
the shared filter policy, install/serve setup, staging, upload, and export live in
[`../AGENTS.md`](/devs/data/webgym/AGENTS.md); run its
[§1 Install And Configure](/devs/data/webgym/AGENTS.md#1-install-and-configure)
first — it builds the image, starts the env-server, and pins `$COMMIT`.

This runbook ends at the clean log roots. They are the only thing the dataset
runbook consumes:

    .data/rollout/webgym/qwen3_5_27b/$COMMIT/{d1..d7,popular}_clean

**No `.think` siblings.** `gpt5_5` ends one step later, at `*_clean.think`, because that
teacher is PROMPTED for a `Thought:` line that
[`internalize_cot.py`](/devs/data/internalize_cot.py) has to move into
`reasoning_content`. This teacher writes that field NATIVELY, so the bare `_clean` roots
are already canonical — see [Reasoning Channel](#reasoning-channel). Stage THESE.

## Serve The Model

Unlike `gpt5_5` (an API model), this teacher is local weights and needs a running
sglang server first. `tp_size` is pinned PER MODEL in `LOCAL_AGENTS`
([/lite/agents/factory.py](/lite/agents/factory.py) — line 81, 2 for this one) and the
launcher derives `dp_size = visible_gpus // tp_size`.

**How many GPUs is your call, not this runbook's.** Any multiple of `tp_size` works, and
each extra `tp_size`-sized group is one more replica behind the same port. Pick from what
`nvidia-smi` actually shows free and what the host's other tenants are doing. The collect
step below uses `--concurrency 16`, which is a WebGym browser-pool ceiling rather than a
model-throughput one (see [Concurrency](#concurrency)), so extra replicas buy less here
than they do on the desktop datasets.

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
when it launches a server for you. Bind `127.0.0.1` on shared hosts; the launcher defaults
to `0.0.0.0`.

## Reasoning Channel

Thinking is ON, and it is the only `agent_kwargs` override this runbook makes:

```
--agent-kwargs '{"enable_thinking": true}'
```

`LOCAL_AGENTS` pins `enable_thinking=False` for this model
([`factory.py:81`](/lite/agents/factory.py), `**_NO_THINK`), and the adapter default is OFF
too, so without the override this teacher would collect the same Action-only rows
`qwen3_8_27b` does.

With it on, the model's native `<think>...</think>` is parsed straight into the canonical
`reasoning_content` FIELD (`lite/agents/models/qwen3_5/adapter.py`), so **this teacher needs
no `internalize_cot.py` pass** — its published rows are already in the one vocabulary. That
is the whole difference from `gpt5_5`, which is PROMPTED for a `Thought:` line and lands it
in an `inline_reasoning` content part that must be moved before staging. `qwen3_8_27b`
also skips the pass, but for the opposite reason: it emits no reasoning at all.

Keep the override on the command line rather than forking a yaml: the rollout config stays
the shared, unmodified one, and `metadata.others.command` records the full argv
(`_cli_command()`, `lite/infer/rollout.py`), so the override reads straight off the row. A
forked yaml's setting is not on the row — recovering it means taking the row's
`config_path` + `commit` back to the checkout and reading the file.

Do NOT reuse [`gpt/recipes/collect/webgym.yaml`](/scripts/configs/gpt/recipes/collect/webgym.yaml):
it carries `inline_reasoning_instruction` and `action_description_instruction`, which the
`gpt.teacher` agent parses. Use `scripts/configs/qwen3_5/default/webgym.yaml`, which pins
the same env surface (`extra_tools: ["goto", "back", "response"]`, `loop_detect: 5`) so the
filter's `--drop-unsubmitted` and `--drop-loops` mean the same thing across teachers.

`max_steps` is omitted in that config on purpose, so WebGym applies its per-difficulty
budgets (train: easy 15 / medium 25 / hard 35). Leave it alone; the tier budgets are what
the `gpt5_5` batch was collected under, and changing them here would make the two teachers'
rows incomparable.

## Task Set

Same tiers and the same site-start restriction as every other teacher on this dataset —
see [Collection Targets](/devs/data/webgym/AGENTS.md#collection-targets). The `--filter`
and `--sample` below are byte-identical to
[`gpt5_5/AGENTS.md`](/devs/data/webgym/gpt5_5/AGENTS.md#collect) so the two teachers draw
the same task population at the same `--seed 1`.

To CHECK coverage after the fact (retries, skips, or an interrupted batch can still leave
them uneven), compare the task ids each run actually produced — they are the sample
directory names, and are also recorded in each trajectory's `metadata.others.task_id`:

```bash
ids() { find "$1" -name summary.json -path "*/sample_*" \
  | xargs -n1 dirname | xargs -n1 dirname | xargs -n1 basename | sort -u; }
for T in gpt5_5 qwen3_5_27b; do
  printf "%-12s %s\n" "$T" "$(ids ".data/rollout/webgym/$T/$COMMIT" | wc -l)"
done
diff <(ids ".data/rollout/webgym/gpt5_5/$COMMIT") \
     <(ids ".data/rollout/webgym/qwen3_5_27b/$COMMIT")
```

A non-empty diff means one side is short; re-run that side's collect command (it resumes)
rather than narrowing the other.

## Attempt sizing

Same flat per-tier `--sample` as `gpt5_5`: each **easy** tier (d1–3) 500, each **medium**
tier (d4–6) 1000, **hard** (d7) 2000, plus the full curated **popular** pool (~2,102, no
`--sample`).

**Clean yields for this teacher are NOT measured.** `gpt5_5`'s table (63% at d1 falling to
25% at d7, 41% overall) is that teacher's, and a 27B open-weights model on multi-step web
navigation should be expected to land below it — the dataset runbook's own paired
measurement puts a base 8B at 33% on medium where gpt hits 83%. Treat the `--sample`
numbers as attempt budgets, not as a promise of demo counts, and **record the realized
per-tier yield before publishing** (the dataset runbook's bookkeeping asks for exactly
this). If d1–3 come back near zero, that is a signal to re-read the prompt surface, not to
raise `--sample`.

### Concurrency

Keep GLOBAL ≈16 and collect **one tier at a time**, exactly as `gpt5_5` does. The ceiling
here is the OmniBoxes browser pool (`WEBGYM_INSTANCES`, dataset §1) and its sub-linear
throughput past ~16 — not the model server. A second replica behind the same `$SGLANG_URL`
will not raise the useful concurrency on this dataset; it only removes the model as the
bottleneck. Never exceed 64 (pool size).

## Collect

`$COMMIT` and `CUA_LITE_ENV_SERVER_URL` come from the dataset runbook's
[§1 Install And Configure](/devs/data/webgym/AGENTS.md#1-install-and-configure);
`$SGLANG_URL` from [Serve The Model](#serve-the-model) above.

```bash
# --- host ---

# step 1: collect (per difficulty d). One tier per run — set D and N, then re-run for the
# next tier. Never several tiers in parallel (that pushes global concurrency past ~16).
# --filter is SITE-START ONLY for every tier: difficulty==d AND the start site is not a
# search-engine root (their SERP is blank in this browser → ~0 yield). Only the *roots* are
# excluded — real product sites like accounts.google.com stay in.
# Per-tier --sample: d1,d2,d3 → 500 each; d4,d5,d6 → 1000 each; d7 → 2000.
D=7; N=2000
uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-27B \
  --agent-kwargs '{"enable_thinking": true}' \
  --sglang-server-url "$SGLANG_URL" \
  --env-id webgym \
  --splits train --sample "$N" --seed 1 --concurrency 16 --max-attempts 2 \
  --filter "lambda m: m.others.get('difficulty',0)==$D and m.others.get('website','').split('//')[-1].split('/')[0].removeprefix('www.') not in ('google.com','bing.com','duckduckgo.com')" \
  --save-data true --save-video false --save-gif false \
  --config-path scripts/configs/qwen3_5/default/webgym.yaml \
  --log-root ".data/rollout/webgym/qwen3_5_27b/$COMMIT/d$D"

# step 1.5: PRIORITY — the curated "popular" pool (high-value, do this first / weight it
# heavily). `webgym_popular_2102.parquet` is OpenWebRL's filtered+cleaned popular subset.
# It is already pre-filtered, so it is driven by --prompt-data (NOT --splits/--filter,
# which are mutually exclusive with it — the parquet IS the task list). Larger attempt
# budget (--max-attempts 5 vs 2) so a transient judge/nav hiccup gets retried instead of
# lost. Run the FULL pool (no --sample), --group-size 1 like the bulk tiers.
uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-27B \
  --agent-kwargs '{"enable_thinking": true}' \
  --sglang-server-url "$SGLANG_URL" \
  --env-id webgym \
  --prompt-data lite/gym/envs/webgym/data/webgym_popular_2102.parquet \
  --seed 1 --concurrency 16 --max-attempts 5 --group-size 1 \
  --save-data true --save-video false --save-gif false \
  --config-path scripts/configs/qwen3_5/default/webgym.yaml \
  --log-root ".data/rollout/webgym/qwen3_5_27b/$COMMIT/popular"
```

**On failure (crash / stall / host-overload kill): just re-launch the SAME command — do NOT
wipe `--log-root`.** A fixed `--log-root` makes the collector *resume*: it re-runs only the
samples with no `summary.json` and skips the rest (`lite/infer/rollout.py` `get_pending`).
Deleting the log-root throws away good trajectories and re-rolls the whole tier — only do
that for a genuinely corrupt run. Note the resume gate is summary PRESENCE, so a trajectory
that ended in a model parse failure is *resolved*, not pending: it will not be re-rolled,
and it is scored at whatever the judge awarded. A stall looks like the tier log's `mtime`
frozen for minutes — kill ONLY your own driver process group (`kill -KILL -<pgid>`, never
others' procs and never the shared OmniBoxes backend container) and relaunch.

## Filter and quality check

Identical to `gpt5_5` — same script, same flags. The filter runs on canonical Lite rows
(the adapter has already projected this family's wire format), so it carries no
model-family branch and needs none.

```bash
# step 2: the ONLY filtering step. stage applies no predicate of its own.
uv run python devs/data/webgym/filter.py \
  --log-root ".data/rollout/webgym/qwen3_5_27b/$COMMIT/d$D" \
  --out ".data/rollout/webgym/qwen3_5_27b/$COMMIT/d${D}_clean" \
  --drop-failed --drop-loops --drop-serp-only --drop-captcha --drop-unsubmitted --drop-illposed-task
# same for the popular pool: --log-root .../popular --out .../popular_clean
```

Run the quality battery on a tier, or on the whole batch, before handing the cleaned roots
to the dataset runbook:

```bash
uv run python devs/data/webgym/quality_check.py .data/rollout/webgym/qwen3_5_27b/$COMMIT/d7
uv run python devs/data/webgym/quality_check.py .data/rollout/webgym/qwen3_5_27b/$COMMIT
```

Two footgun classes deserve a closer look on THIS teacher than on `gpt5_5`, because they
are exactly what the dataset runbook measured a smaller open-weights model failing at:
`loop` / `back_bounce` / `scroll_hunt` (broken navigation — re-goto'ing its own homepage,
looping) and `serp_only` (snippet-scraping instead of clicking through). If either is a
large share of the *surviving* demos rather than the dropped ones, the filter flags are not
enough and the batch needs re-reading before it becomes training data.

Also spot-check that `reasoning_content` is actually populated and is not empty on the
surviving rows — a thinking-on run whose `<think>` block came back empty would publish rows
that look like reasoning data and train an empty channel:

```bash
uv run python - <<'PY'
import json, pathlib, pyarrow.parquet as pq
root = pathlib.Path(".data/rollout/webgym/qwen3_5_27b").glob("*/d7_clean")
n = empty = 0
for r in root:
    for p in r.rglob("*.parquet"):
        for msgs in pq.read_table(p, columns=["messages"]).column("messages").to_pylist():
            for m in json.loads(msgs):
                if m.get("role") != "assistant":
                    continue
                n += 1
                empty += not (m.get("reasoning_content") or "").strip()
print(f"assistant turns={n}  empty reasoning_content={empty}")
PY
```

## Staging

The dataset runbook's step 3 says the staged roots are the `.think` siblings. That sentence
is `gpt5_5`-specific. This teacher stages the bare `_clean` roots, under its own config
label:

    --config-names browser.use.qwen3_5_27b

Everything else about step 3 is unchanged — including that **upload is a declarative full
sync**, so adding this teacher to an already-published repo means staging *all* configs in
one call. Follow
[Add A Config To A Published Dataset](/devs/data/AGENTS.md#add-a-config-to-a-published-dataset)
rather than staging these roots alone.

## Cost / time

Different shape from `gpt5_5`'s. That teacher is API-latency-bound at ~1.2 traj/min; this
one owns its serving, so per-turn model latency is yours to control and the binding
constraint moves to the browser pool and the per-trajectory VLM reward judge (~30 s, still
an API call). Thinking-on also lengthens every turn's output relative to `qwen3_8_27b`, so
do not carry that teacher's numbers over either. The dataset's throughput investigation
([/devs/envs/webgym.md](/devs/envs/webgym.md)) found the rollout process CPU-idle and
throughput sub-linear in concurrency, so the levers are the same ones listed there:
reduce turns on failing tiers, share/cheapen the judge, scale horizontally across hosts —
not more processes on one host.

**No throughput numbers are recorded for this teacher yet.** Measure and add them on the
first scaled batch.

## Current status

- Runbook only. No scaled batch has been collected with this teacher.
- The commands above are the `gpt5_5` flow with the model, config, `enable_thinking`
  override and log-root swapped, and the `internalize_cot` step removed.
- Before a scaled run, do a small smoke (one tier at `--sample 2 --concurrency 2`) and
  confirm: trajectories carry `tool_calls` + `action_description`, carry a NON-EMPTY
  `reasoning_content`, and `filter.py` keeps the successes.
- Record realized per-tier yield and throughput here before publishing.
