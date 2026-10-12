# Lite.OSWorld Teacher-Data Pipeline

This directory owns Lite.OSWorld teacher-data collection and the shared desktop
trajectory filter used by Lite.OSWorld-family teacher-data workflows.

Lite.OSWorld publishes trajectories from THREE teachers into one HF repo, each as
its own set of configs. Per-teacher collection and annotation live in their own
runbooks; everything below the annotated log roots is dataset-level and covers
all of them at once.

| Teacher | Runbook | Model |
|---|---|---|
| `gpt5_5` | [`gpt5_5/AGENTS.md`](/devs/data/lite.osworld/gpt5_5/AGENTS.md) | `gpt-5.5` (API) |
| `qwen3_8_27b` | [`qwen3_8_27b/AGENTS.md`](/devs/data/lite.osworld/qwen3_8_27b/AGENTS.md) | `Qwen/Qwen3.8-27B` (local, sglang) |
| `qwen3_5_27b` | [`qwen3_5_27b/AGENTS.md`](/devs/data/lite.osworld/qwen3_5_27b/AGENTS.md) | `Qwen/Qwen3.5-27B` (local, sglang, **thinking on**) |

The handoff between a teacher runbook and this one is exactly:

    .data/rollout/lite.osworld/<teacher>/$COMMIT/train.{synth,perturb}_annotated
    .data/rollout/lite.osworld/gpt5_5/$COMMIT/train.{synth,perturb}_annotated.think

`gpt5_5` hands over the `.think` sibling: its Internalize Reasoning step runs after
`filter.py`, and the stage command below reads that root, not the plain `_annotated`
one. The other teachers have no `.think` root at all; their `_annotated` root is the handoff.

Lite.CUAGym and Lite.ScaleCUA have their own workflow documentation in
`devs/data/lite.cuagym/AGENTS.md` and `devs/data/lite.scalecua/AGENTS.md`.
They reuse `filter.py`; the datasets, prompts, rollout inputs, and published
artifacts remain separate.

## Collection Targets

Collect only the training sub-splits. Do not collect the eval split for SFT.

The synth sub-split currently includes rows with
`metadata.others.exclude_reason` (quarantined/unrunnable tasks such as OCR
verification gaps or upstream live-site drift); teacher-data collection must
filter those rows before rollout. Perturb currently has no task-level
exclusions, but every collect command still carries the same filter.

| Split | Registered rows | Runnable rows | HF configs |
|---|---:|---:|---|
| `train.synth` | 1,722 | 1,704 | `desktop.use.synth.gpt5_5`, `desktop.use.synth.qwen3_8_27b`, `desktop.use.synth.qwen3_5_27b` |
| `train.perturb` | 707 | 707 | `desktop.use.perturb.gpt5_5`, `desktop.use.perturb.qwen3_8_27b`, `desktop.use.perturb.qwen3_5_27b` |

`qwen3_5_27b` joins each row once that teacher is collected and staged.

Every collect command must include:

```bash
--filter "lambda m: not m.others.get('exclude_reason')"
```

`scripts/rollout.py` applies the filter before task execution. (This is the
same task-level idiom as Lite.ScaleCUA — see
`devs/data/lite.scalecua/AGENTS.md`.)

Collect, filter, and stage the two sources separately. `train.perturb` is
derived from eval setups, so it must stay identifiable for train/eval-leakage
review before mixing into any SFT set.

The config name carries the teacher, and the log-root directory uses the SAME
token (`.data/rollout/lite.osworld/gpt5_5/...` ↔ `desktop.use.*.gpt5_5`).
`--config-names` is positional and 1:1 with `--log-roots`, and stage only checks
that the two lists are the same LENGTH — mislabelling a teacher is otherwise
silent, so keeping the tokens identical is what makes the pairing checkable by
eye.

## Shared Filter

`filter.py` is the single quality-**annotation** pass before staging for desktop
datasets. It **keeps every ordinary quality-failed trajectory** and tags gates in
`metadata.others.exclude_reason` (comma-joined; the key is omitted when clean) —
downstream consumers filter with `not m.others.get('exclude_reason')`, exactly
like the task-level idiom. The physical drops are env-tool leaks, OOB
coordinates, undeclared tool calls and invalid action-batch children:
trajectories whose agent typed a `/opt/env/` path are not reproducible in the
faithful guest, and the other three fail the staging row-format check.

```bash
uv run python devs/data/lite.osworld/filter.py \
  --log-root <raw-log-root> \
  --out <annotated-log-root> \
  --drop-loops --drop-undo-storm
```

It tags, in `exclude_reason`:

- `incomplete` — `terminated != true`;
- `dependency_install` — apt/pip/conda/snap/flatpak installs;
- `complex_shell` — a non-teachable terminal *operation* (see below);
- `footgun:loop` / `footgun:undo_storm` — ≥3 identical consecutive actions / ≥4 Ctrl+Z, gated by
  `--drop-loops` / `--drop-undo-storm` (both passed by the canonical command below);
- `footgun:no_submit` — no submit action, gated by `--drop-no-submit`, which the canonical
  command does NOT pass, so this tag never appears in the published rows;
- `reward_vision_disagree` — SOFT tag, emitted UNCONDITIONALLY (no `--drop` flag gates it):
  the scalar checker reward and the multi-frame vision verdict disagree. It never overwrites
  `episode_return`, and stage publishes the row either way — but the export filter this runbook
  uses (`not m.others.get('exclude_reason')`) drops it, so a tagged row reaches the Hub and not
  the training set.

Reward is deliberately **not** a tag: `episode_return` is already in
`metadata.others.episode_return`, so a consumer thresholds it directly (`episode_return > 0.5`).
Ordinary quality gates are not dropped. `--drop-loops` / `--drop-undo-storm` are
**still required** —
in annotate mode they no longer drop, they gate whether `footgun:loop` /
`footgun:undo_storm` get **tagged**, so the canonical annotate command passes
them.

`complex_shell` is **operation-driven, not structure-driven**: for/while loops,
multi-line blocks, `;` / `&&` / single `|` of SIMPLE commands are kept (efficient
repetition). Tagged are operations a small GUI model can't ground from a
screenshot: nested/substituted execution (`$()` / `<()` / backtick / heredoc),
inline interpreters (`python -c` / `bash -c`) and running or authoring code
scripts, `sed -i` in-place file surgery, dotfile / `.desktop` authoring, and awk
state machines (`next` / `exit`).

On every (kept) trajectory it also: strips no-op `screenshot` and `wait` calls
or action-batch child actions, preserving the canonical action-batch call and any remaining
child actions; keeps bare Ctrl+S; flattens inline reasoning to one line; keeps
the content-only final channel, normalized to one plain `text` part — `"Done."`
is now MANDATORY, not merely common: whatever the turn held before
(`inline_reasoning`, `action_description`, both) is replaced. A final turn
carrying `tool_calls` (a QA answer submitted through `response`, a `terminate`)
is untouched.
Synthetic `terminate(status="success")` is opt-in only, and when enabled the
staged row must include the matching canonical nested `terminate` schema in
`metadata.extra_tool_schemas`.

There is intentionally no blanket terminal-command-count threshold. Some OS
tasks need several simple commands; tag unsafe operations, not an arbitrary
count.

Tests:

```bash
PYTHONPATH="$PWD" uv run pytest -n 0 \
  devs/data/lite.osworld/tests/test_lite_osworld_filter.py -q
```


## Complete Workflow

Run from the repository root. Pipeline: collect → filter/annotate → stage →
upload/download → `export_sft`. Freeze the code revision, input task set,
prompt, and log root for each batch. A resume must use the identical command.

### 1. Install And Configure

```bash
uv sync --locked --extra quick-start --extra gym
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh
uv run python scripts/serve_env.py --port 30200 --env-ids lite.osworld

HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:30200

COMMIT="$(git rev-parse --short HEAD)"
```

Use the install script rather than a plain Docker build; it stamps the source
freshness label required by env-server. Collection should use env-server mode;
the per-teacher commands assume a 32-ish rollout batch against that server.

Each teacher needs its own credentials or serving step — an API key for
`gpt5_5`, and an sglang server for each of `qwen3_8_27b` / `qwen3_5_27b`. Those live in the teacher
runbooks.

### 2. Collect And Annotate (per teacher)

Run [`gpt5_5/AGENTS.md`](/devs/data/lite.osworld/gpt5_5/AGENTS.md),
[`qwen3_8_27b/AGENTS.md`](/devs/data/lite.osworld/qwen3_8_27b/AGENTS.md) and
[`qwen3_5_27b/AGENTS.md`](/devs/data/lite.osworld/qwen3_5_27b/AGENTS.md). All
end with annotated log roots under
`.data/rollout/lite.osworld/<teacher>/$COMMIT/`.

Every teacher runs the SAME `filter.py` with the SAME flags. That is deliberate:
it makes the published subsets comparable, so a measured quality difference is a
property of the teacher rather than of the annotation pass.

### 3. Stage, Upload Transport, And Download

> **Upload is a declarative full sync, not an append.** It plans the whole repo
> from the LOCAL staging dir and deletes everything else, so staging one teacher
> and uploading DELETES the other's published shards. Adding a teacher (or any
> config) to an already-published repo therefore has one shared procedure —
> read what the repo actually holds, then stage all of it in one call:
> [Add A Config To A Published Dataset](/devs/data/AGENTS.md#add-a-config-to-a-published-dataset).

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

# The gpt5_5 roots below are the `.think` siblings its teacher runbook produced: reasoning
# canonicalized into `reasoning_content` before staging, so the PUBLISHED rows carry one
# reasoning shape. `qwen3_8_27b` runs thinking off and has none to move, so its roots are
# used as-is, and `qwen3_5_27b` needs none either — it is sampled with thinking ON, so its
# native <think> is already in `reasoning_content` at collection time. Only gpt5_5 has a
# step between filter and stage. See /devs/data/lite.osworld/gpt5_5/AGENTS.md.
# DELETE the qwen3_5_27b lines below until that teacher is actually collected — they are
# LIVE as written, and a shell comment cannot be used here (a `#` after a `\` continuation
# swallows the rest of the command). Once it IS published they are MANDATORY again, and
# NOTHING WILL TELL YOU IF YOU FORGET: stage rglobs each
# root and only errors when ALL of them are empty, so a missing one contributes zero rows
# and stage still exits 0 (verified) — then upload sweeps and deletes that teacher's
# published shards. Re-read the blockquote above before every re-stage.
uv run python -m lite.data.hf.stage \
  --log-roots ".data/rollout/lite.osworld/gpt5_5/$COMMIT/train.synth_annotated.think" \
              ".data/rollout/lite.osworld/gpt5_5/$COMMIT/train.perturb_annotated.think" \
              ".data/rollout/lite.osworld/qwen3_8_27b/$COMMIT/train.synth_annotated" \
              ".data/rollout/lite.osworld/qwen3_8_27b/$COMMIT/train.perturb_annotated" \
              ".data/rollout/lite.osworld/qwen3_5_27b/$COMMIT/train.synth_annotated" \
              ".data/rollout/lite.osworld/qwen3_5_27b/$COMMIT/train.perturb_annotated" \
  --config-names desktop.use.synth.gpt5_5       desktop.use.perturb.gpt5_5 \
                 desktop.use.synth.qwen3_8_27b  desktop.use.perturb.qwen3_8_27b \
                 desktop.use.synth.qwen3_5_27b  desktop.use.perturb.qwen3_5_27b \
  --name Lite.OSWorld \
  --repo-dir devs/data/lite.osworld \
  --overwrite   # the default out dir is $CUA_LITE_DATASETS_ROOT/cua-lite/Lite.OSWorld;
                # stage refuses a non-empty one, so a re-stage needs this

: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload Lite.OSWorld --org "$HF_ORG" --private --tag "$COMMIT"

# Consumer / verification: pull the uploaded revision into canonical local layout.
# NOTE: download verifies LAYOUT only; row content was already gated by stage above.
# upload/download are transport/layout steps, and export_sft below is a conversion smoke.
# See /devs/migration/AGENTS.md#1-download-the-source.
uv run python -m lite.data.hf.download Lite.OSWorld \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/Lite.OSWorld"
```

A consumer who wants one teacher pulls only that teacher's shards, either
through the HF config (`load_dataset("cua-lite/Lite.OSWorld",
"desktop.use.synth.qwen3_8_27b")`) or with
`hf.download --allow-patterns '*/*/*/desktop.use.*.qwen3_8_27b/*'`.

Record `stage`'s final `seen=... kept=... dropped_by_filter=...` line and the
per-config row lines as the publish gate. Upload is transport only; use the
release org only after the private upload/readback/export smoke is approved.

### 4. Export SFT Parquet

The three teachers publish DIFFERENT kinds of row, and a consumer that mixes them
should know which it is training on:

| Teacher | Rows carry | Reasoning |
|---|---|---|
| `gpt5_5` | `inline_reasoning` + `action_description` + `tool_calls` | prompted `Thought:` line |
| `qwen3_8_27b` | `action_description` + `tool_calls` | none — runs with thinking off, per its runbook |
| `qwen3_5_27b` | `reasoning_content` + `action_description` + `tool_calls` | native `<think>`, written straight to the canonical field |

The table is what each teacher COLLECTS. `gpt5_5`'s last step before staging
([Internalize Reasoning](/devs/data/lite.osworld/gpt5_5/AGENTS.md#internalize-reasoning))
moves its prompted `Thought:` out of the `inline_reasoning` content part and into
the `reasoning_content` FIELD — the same one a teacher sampled with
`enable_thinking` writes natively. So the PUBLISHED rows carry one vocabulary and
consumers run no extra step.

`qwen3_8_27b` has no reasoning to internalize (it runs thinking off and writes
prose into `action_description`), so a thinking-enabled config on those rows
would train an empty `<think>` block — pair them with a thinking-off config.

`qwen3_5_27b` needs no internalize pass either, for the opposite reason: it is
sampled with `enable_thinking: true`, so the adapter parses its native `<think>`
straight into `reasoning_content` at collection time. Three teachers, three
collection recipes, one published vocabulary — only `gpt5_5` has a step between
filter and stage.

One published root serves both recipes for a reasoning teacher: a thinking-on
config renders the reasoning, and a thinking-off one strips `reasoning_content`
at the model boundary (`lite/train/export/sft_tokenize.py`), which is
byte-identical to exporting from a root that never carried any. Nothing under
`scripts/configs/*/default/` sets `enable_thinking: true` for these envs, so the
command below is the thinking-off recipe; the thinking-on twins live with the
campaign that trains them
([`/devs/exps/train/desktop/configs/qwen3_5/`](/devs/exps/train/desktop/configs/qwen3_5/),
[`/examples/lite/v1/configs/qwen3_5/`](/examples/lite/v1/configs/qwen3_5/)).

```bash
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_5/default/lite.osworld.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --data-paths "${READBACK_ROOT}/cua-lite/Lite.OSWorld" \
  --image-root "${READBACK_ROOT}" \
  --filter "lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5" \
  --num-proc 16 \
  -o .data/sft/qwen3_5/lite-osworld/train.parquet
```

`--config` is the **rollout** config, not an SFT-only recipe under
`scripts/configs/*/recipes/sft/`: `export_sft` re-renders every step through the
agent adapter, so exporting under a different history window or resolution than
the rollout used trains the model on prompts it will never see at inference.

The processor/model ID must match training because tokenization and the chat
template are frozen during export. Keep fail-fast enabled; use `--no-strict`
only for an identified and recorded corrupt source row.
