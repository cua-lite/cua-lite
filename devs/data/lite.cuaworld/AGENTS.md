# Lite.CUAWorld Teacher-Data Pipeline

This directory owns the Lite.CUAWorld teacher-data workflow.

Lite.CUAWorld re-hosts the **gym-anything (CMU CUA-World)** desktop software task suite — **40
softwares, one Docker image each** (`lite.cuaworld.<software>`). Its desktop tasks use the same
screenshot, coordinate-action, and terminal substrate as Lite.OSWorld, so it
**reuses the shared quality filter at [`devs/data/lite.osworld/filter.py`](/devs/data/lite.osworld/filter.py)**
(like Lite.CUAGym and Lite.ScaleCUA). Its task pool, prompt, rollout logs, and published dataset
remain separate. Env content lives in the HF materials dataset `cua-lite/lite.cuaworld-assets`
(pinned by `data/assets.lock.yaml`); this repo is engine + pipeline only.

Lite.CUAWorld publishes trajectories from THREE teachers into one HF repo, each as
its own set of configs. Per-teacher collection and annotation live in their own
runbooks; everything below the annotated log roots is dataset-level and covers
all of them at once.

| Teacher | Runbook | Model |
|---|---|---|
| `gpt5_5` | [`gpt5_5/AGENTS.md`](/devs/data/lite.cuaworld/gpt5_5/AGENTS.md) | `gpt-5.5` (API) |
| `qwen3_8_27b` | [`qwen3_8_27b/AGENTS.md`](/devs/data/lite.cuaworld/qwen3_8_27b/AGENTS.md) | `Qwen/Qwen3.8-27B` (local, sglang) |
| `qwen3_5_27b` | [`qwen3_5_27b/AGENTS.md`](/devs/data/lite.cuaworld/qwen3_5_27b/AGENTS.md) | `Qwen/Qwen3.5-27B` (local, sglang, **thinking on**) |

The handoff between a teacher runbook and this one is exactly:

    .data/rollout/lite.cuaworld/<teacher>/$COMMIT/<software>/train_annotated
    .data/rollout/lite.cuaworld/gpt5_5/$COMMIT/<software>/train_annotated.think

`gpt5_5` hands over the `.think` sibling: its Internalize Reasoning step runs after
`filter.py`, and the stage command below reads that root, not the plain `_annotated`
one. The other teachers have no `.think` root at all; their `_annotated` root is the handoff.

## Collection Targets

Each software registers the splits present in the current materials: `train`
(rollout target for SFT), `eval` = **CUAWorld-Test**, and sometimes
`long_horizon` = **CUAWorld-Long** (1/software). For SFT, roll out `train` on the
38 rolloutable softwares; collect `eval`/`long_horizon` only as held-out reference
sets with an explicit software list, and keep them **identifiable and out of any
SFT mix** (mixing them into train is eval leakage). Current locked-materials snapshot
over the 40 softwares:

| split | registered | live after task excludes | HF config |
|---|---:|---:|---|
| `train` | 2,419 | 1,745 | `desktop.use.<software>.<teacher>` — one config per software per teacher |
| `eval` (CUAWorld-Test) | 626 | 440 | held-out reference, **not** SFT |
| `long_horizon` (CUAWorld-Long) | 38 | 18 | held-out reference, **not** SFT |

**Excluded softwares** (single-digit / incomplete onboarding): `knime` (0 train) and `freecad`
(5 train, all validation-excluded as `gameable_full`) → **38 softwares** carry a real train pool
(18–135 each after task excludes).

**Excluded tasks (task-level, pre-rollout).** A no-LLM validation sweep flagged 880 tasks
(across all splits) whose setup/verify pipeline is broken or gameable regardless of the agent.
They are recorded in
[`/lite/gym/envs/lite/cuaworld/data/validation_excludes.json`](/lite/gym/envs/lite/cuaworld/data/validation_excludes.json)
and the engine bakes each into `metadata.others['exclude_reason']` at registration
(`src/software.py::_exclude_reasons`). This is the **same `exclude_reason` idiom** the shared filter
uses at the trajectory level (below), so both compose under one downstream selector. Reason-code
breakdown: [`/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md`](/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md).

The config name carries BOTH axes: the software and the teacher. The software
comes from the log-root subfolder (`$SW`), and the teacher token is the same one
the log-root path uses (`.data/rollout/lite.cuaworld/gpt5_5/...` ↔
`desktop.use.*.gpt5_5`). `--config-names` is positional and 1:1 with
`--log-roots`, and stage only checks that the two lists are the same LENGTH —
mislabelling a software or a teacher is otherwise silent, so keeping the tokens
identical is what makes the pairing checkable by eye.

## Shared Filter

Lite.CUAWorld uses [`devs/data/lite.osworld/filter.py`](/devs/data/lite.osworld/filter.py). It is mostly an
**annotation** pass: it keeps trajectories and tags quality gates in
`metadata.others.exclude_reason` (comma-joined; the key is omitted when clean). Four
publish-invalid classes are physically dropped before staging: a trajectory whose
agent typed a `/opt/env/` path leaked an env-only tool tree and is
non-reproducible, and a trajectory with GUI coordinates outside normalized
`[0, 1000]`, one calling a tool it never declared, or one naming an action that
does not exist would all fail publish validation. Downstream selects the training set with
`not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5`.

```bash
uv run python devs/data/lite.osworld/filter.py \
  --log-root <raw-log-root> \
  --out <annotated-log-root> \
  --drop-loops --drop-undo-storm
```

It tags, in `exclude_reason`:

- `incomplete` — `terminated != true`;
- `dependency_install` — apt/pip/conda/snap/flatpak installs;
- `complex_shell` — a non-teachable terminal *operation* (operation-driven, not structure-driven:
  loops / `;` / `&&` / single `|` of simple commands are KEPT; tagged are `$()` / `<()` / backtick /
  heredoc, `python -c` / `bash -c`, running or authoring code scripts, `sed -i`, dotfile authoring,
  awk state machines);
- `footgun:loop` / `footgun:undo_storm` — ≥3 identical consecutive actions or
  ≥4 Ctrl+Z when `--drop-loops` / `--drop-undo-storm` are passed;
- `footgun:no_submit` — no explicit final submit tool (`terminate`/`response`),
  only when `--drop-no-submit` is passed; this is separate from the default
  content-only final turn policy (normalized to one plain `text` part, not preserved as emitted);
- `reward_vision_disagree` — SOFT tag, emitted UNCONDITIONALLY (no `--drop` flag gates it):
  the scalar checker reward and the multi-frame vision verdict disagree. It never overwrites
  `episode_return`, and stage publishes the row either way — but the export filter this runbook
  uses (`not m.others.get('exclude_reason')`) drops it, so a tagged row reaches the Hub and not
  the training set.

Reward is **not** a tag (`episode_return` is in `metadata.others.episode_return` for the consumer to threshold).
On every kept trajectory it strips `screenshot` and `wait` (keeps a bare
Ctrl+S), flattens inline reasoning, and normalizes the content-only final turn to one plain `text` part (`{"type": "text", "text": "Done."}`) — unconditionally, whatever it held before (`inline_reasoning`, `action_description`, both, or anything else). A final turn that DOES carry `tool_calls` is untouched. Synthetic
`terminate(status="success")` is opt-in only, and when enabled the staged row must include the matching
canonical nested `terminate` schema in `metadata.extra_tool_schemas`.

Tests: `uv run pytest devs/data/lite.osworld/tests/test_lite_osworld_filter.py`.

## Complete Workflow

Run from the repository root. Pipeline: collect → filter/annotate → stage →
upload/download → `export_sft`. Freeze the code revision, task set, prompt, and log root per batch; a
resume must use the identical command.

### 1. Install And Configure

```bash
uv sync --locked --extra quick-start --extra gym

# Default agent + VLM judge route. Set these before starting the env-server;
# the host-side CUAWorld judge reads the same process environment.
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before collection}"
# Optional; set only for a custom endpoint.
# export OPENAI_BASE_URL="..."

# Usually no VLM_* vars are needed: the default judge model is in
# /lite/gym/envs/lite/cuaworld/configs/default.yaml and litellm reads OPENAI_*.
# VLM_MODEL overrides LITE_CUAWORLD_VLM_MODEL, which overrides that default.
# Only set VLM_BASE_URL / VLM_API_KEY when overriding the judge route/key.

# Optional: point at a local materials checkout to avoid HF reads while validating
# local materials edits. Leave unset for the locked public HF materials.
# export LITE_CUAWORLD_MATERIALS_REPO=/path/to/lite.cuaworld-assets

COMMIT="$(git rev-parse --short HEAD)"

# Build the explicit software list you will collect/eval ONCE (stamps the
# lite.src_hash freshness label env-server needs).
# NOTE: images MUST be built by the current engine (stdio protocol v4 — the default-user revert
# bumped it from v3 so stale `:shim` images hard-fail instead of silently running the old
# contract); images from an older protocol are rejected at runtime. Full-suite
# validation can cover all 40, while collection usually uses the rolloutable subset.
uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build <software>   # per software
# uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh rebuild <software>
# uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh provision <software>  # no Docker build

# Start your own env-server on a free port:
CUAWORLD_ENVS="<space-separated lite.cuaworld.<software> ids you built>"
uv run python scripts/serve_env.py --port <PORT> --env-ids $CUAWORLD_ENVS &
HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:<PORT>
```

The `OPENAI_*` route above is required for the VLM judge whichever teacher you
collect with. Beyond that, each teacher needs its own credentials or serving
step — an API key for `gpt5_5`, and an sglang server for each of
`qwen3_8_27b` / `qwen3_5_27b`. Those live
in the teacher runbooks.

Before scaling collection, run the same one-task smoke command shown in the
env README and default GPT config, then inspect the saved trajectory shape:

```bash
uv run python scripts/rollout.py --model-id gpt-5.5 \
  --env-id lite.cuaworld.pymol --splits eval --head 1 \
  --concurrency 1 --max-attempts 1 --save-data true \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/gpt/default/lite.cuaworld.yaml
```

### 2. Collect And Annotate (per teacher)

Run [`gpt5_5/AGENTS.md`](/devs/data/lite.cuaworld/gpt5_5/AGENTS.md),
[`qwen3_8_27b/AGENTS.md`](/devs/data/lite.cuaworld/qwen3_8_27b/AGENTS.md) and
[`qwen3_5_27b/AGENTS.md`](/devs/data/lite.cuaworld/qwen3_5_27b/AGENTS.md). All
loop the 38 rolloutable softwares on `train`, one log-root subfolder per
software, and end with annotated log roots under
`.data/rollout/lite.cuaworld/<teacher>/$COMMIT/<software>/train_annotated` —
`.../gpt5_5/$COMMIT/<software>/train_annotated.think` for `gpt5_5`, which internalizes
after filtering.

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
>
> Lite.CUAWorld's wrinkle: the stage below globs its log-roots, one config per
> software. Rebuilt roots do not match that glob, so the flags that procedure
> prints REPLACE the glob rather than joining it.

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

# The gpt5_5 roots below are the `.think` siblings its teacher runbook produced: reasoning
# canonicalized into `reasoning_content` before staging, so the PUBLISHED rows carry one
# reasoning shape. `qwen3_8_27b` runs thinking off and has none to move, so its roots are
# used as-is, and `qwen3_5_27b` needs none either — it is sampled with thinking ON, so its
# native <think> is already in `reasoning_content` at collection time. Only gpt5_5 has a
# step between filter and stage. See /devs/data/lite.cuaworld/gpt5_5/AGENTS.md.
# DELETE the qwen3_5_27b lines below until that teacher is actually collected — they are
# LIVE as written, and a shell comment cannot be used here (a `#` after a `\` continuation
# swallows the rest of the command). Once it IS published they are MANDATORY again, and
# NOTHING WILL TELL YOU IF YOU FORGET: stage rglobs each
# root and only errors when ALL of them are empty, so a missing one contributes zero rows
# and stage still exits 0 (verified) — then upload sweeps and deletes that teacher's
# published shards. Re-read the blockquote above before every re-stage.
uv run python -m lite.data.hf.stage \
  --log-roots .data/rollout/lite.cuaworld/gpt5_5/$COMMIT/*/train_annotated.think \
              .data/rollout/lite.cuaworld/qwen3_8_27b/$COMMIT/*/train_annotated \
              .data/rollout/lite.cuaworld/qwen3_5_27b/$COMMIT/*/train_annotated \
  --config-names <one desktop.use.<software>.<teacher> per log-root, teacher block by teacher block, in the SAME order as the globs above> \
  --name Lite.CUAWorld \
  --repo-dir devs/data/lite.cuaworld \
  --overwrite   # the default out dir is $CUA_LITE_DATASETS_ROOT/cua-lite/Lite.CUAWorld;
                # stage refuses a non-empty one, so a re-stage needs this

# Private smoke upload; use the release org only for the approved final publish.
: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload Lite.CUAWorld --org "$HF_ORG" --private --tag "$COMMIT"

uv run python -m lite.data.hf.download Lite.CUAWorld \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/Lite.CUAWorld"
```

The `--log-roots` globs expand in sorted order, one per teacher, so build the
`--config-names` list in that same order, one teacher block after the other: the
software token in each label must be the `$SW` subfolder of the log-root at the
same position, and the teacher token must be the teacher directory of that same
path.

A consumer who wants one teacher pulls only that teacher's shards, either
through the HF config (`load_dataset("cua-lite/Lite.CUAWorld",
"desktop.use.pymol.qwen3_8_27b")`) or with
`hf.download --allow-patterns '*/*/*/desktop.use.*.qwen3_8_27b/*'`.

Record `stage`'s final `seen=... kept=... dropped_by_filter=...` line and the
per-config row lines as the publish gate. Upload/download are transport/layout
smokes only; row content was already gated by `stage`.

### 4. Export SFT Parquet

The three teachers publish DIFFERENT kinds of row, and a consumer that mixes them
should know which it is training on:

| Teacher | Rows carry | Reasoning |
|---|---|---|
| `gpt5_5` | `inline_reasoning` + `action_description` + `tool_calls` | prompted `Thought:` line |
| `qwen3_8_27b` | `action_description` + `tool_calls` | none — runs with thinking off, per its runbook |
| `qwen3_5_27b` | `reasoning_content` + `action_description` + `tool_calls` | native `<think>`, written straight to the canonical field |

The table is what each teacher COLLECTS. `gpt5_5`'s last step before staging
([Internalize Reasoning](/devs/data/lite.cuaworld/gpt5_5/AGENTS.md#internalize-reasoning))
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
  --config scripts/configs/qwen3_5/default/lite.cuaworld.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --data-paths "${READBACK_ROOT}/cua-lite/Lite.CUAWorld" \
  --image-root "${READBACK_ROOT}" \
  --filter "lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5" \
  --num-proc 16 \
  -o .data/sft/qwen3_5/lite-cuaworld/train.parquet
```

`--config` is the **rollout** config, not an SFT-only recipe under
`scripts/configs/*/recipes/sft/`: `export_sft` re-renders every step through the
agent adapter, so exporting under a different history window or resolution than
the rollout used trains the model on prompts it will never see at inference.
(`lite.cuaworld.yaml` pins one software via `env_id`, but `export_sft` reads
only `agent_id` and `agent_kwargs` from the config — the
agent-side render settings are identical across every `lite.cuaworld.*` software.)

The processor/model ID must match training (tokenization + chat template are frozen at export).
Keep fail-fast on; use `--no-strict` only for an identified, recorded corrupt source row.

## Cost / time, resources & disk hygiene

Desktop is slow (~2–3 min/task). On a shared host, a 40-wide run spins up to 40 desktop containers
at once — **run a resource watchdog** and throttle `--concurrency` down if free RAM drops (each
container is memory-heavy; the host is co-tenant with other jobs). Keep total live rollout
concurrency across CUAGym/CUAWorld at 48 or lower; when several are active, budget them explicitly
instead of letting one campaign monopolize the host. The ceiling is host-wide and covers every
teacher: three CUAWorld collections at the start-from-24 default would be 72, so run them in
sequence or divide the 48 among the ones you actually start.
`blender3d` needs `gpus=1` (the
only GPU env); `gmat`'s upstream download is dead (rescue the retagged image). After **stage**
captures a batch, delete its raw `$SW/<split>`, `$SW/<split>_annotated` and (for `gpt5_5`)
`$SW/<split>_annotated.think` roots. Clean up only your
own env-server containers (named by your port) afterward. Task/verifier defects:
[/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md](/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md).
