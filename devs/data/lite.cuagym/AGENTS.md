# Lite.CUAGym Teacher-Data Pipeline

This directory owns the Lite.CUAGym teacher-data workflow.

Lite.CUAGym imports CUA-Gym web, cross-app, and desktop task bundles. Its
Docker image extends `cua-lite/lite.osworld:latest`, and its desktop tasks use
the same screenshot, coordinate-action, terminal, and 30-turn interaction
substrate as Lite.OSWorld. It therefore reuses the shared quality filter at
`devs/data/lite.osworld/filter.py`; its task pool, prompt, rollout logs, and
published dataset remain separate.

Lite.CUAGym publishes trajectories from THREE teachers into one HF repo. It spans
two platforms, so each teacher lands in TWO configs — one per platform, six in all. Per-teacher collection and annotation live in their own
runbooks; everything below the annotated log roots is dataset-level and covers
all of them at once.

| Teacher | Runbook | Model |
|---|---|---|
| `gpt5_5` | [`gpt5_5/AGENTS.md`](/devs/data/lite.cuagym/gpt5_5/AGENTS.md) | `gpt-5.5` (API) |
| `qwen3_8_27b` | [`qwen3_8_27b/AGENTS.md`](/devs/data/lite.cuagym/qwen3_8_27b/AGENTS.md) | `Qwen/Qwen3.8-27B` (local, sglang) |
| `qwen3_5_27b` | [`qwen3_5_27b/AGENTS.md`](/devs/data/lite.cuagym/qwen3_5_27b/AGENTS.md) | `Qwen/Qwen3.5-27B` (local, sglang, **thinking on**) |

The handoff between a teacher runbook and this one is exactly:

    .data/rollout/lite.cuagym/<teacher>/$COMMIT/{browser,desktop}/train_annotated
    .data/rollout/lite.cuagym/gpt5_5/$COMMIT/{browser,desktop}/train_annotated.think

`gpt5_5` hands over the `.think` sibling: its Internalize Reasoning step runs after
`filter.py`, and the stage command below reads that root, not the plain `_annotated`
one. The other teachers have no `.think` root at all; their `_annotated` root is the handoff.

## Collection Targets

Lite.CUAGym exposes one registered `train` split of 10,910 tasks (1,505 web +
9,405 desktop — every pinned upstream row is registered; none is dropped). For
teacher-data collection, use either a frozen audited `--prompt-data` parquet or
an explicitly frozen sample/seed/task-id list. The full registered train split
includes known upstream setup/reward failures, so a seed alone does not define a
publishable collection.

**513 of those 10,910 rows are unusable as default training signals** and carry
a task-level `metadata.others.exclude_reason` from the closed vocabulary in
[/lite/gym/envs/lite/cuagym/src/utils/dataset.py](/lite/gym/envs/lite/cuagym/src/utils/dataset.py)
(broken/empty/no-sentinel/mismatched `reward.py`, `broken_mock:blank_render`
rows whose pinned mock builds but renders empty, and deterministic pinned setup
defects). GitHub and Trello are NOT among them — they build unchanged from the
pinned snapshot (`/devs/envs/lite.cuagym/UPSTREAM_ISSUES.md`).
Nothing is dropped — the rows are annotated and you filter them out, leaving
10,397 default-collectable tasks:

```bash
--filter "lambda m: not m.others.get('exclude_reason')"
```

`rollout.py --filter` applies only when tasks are selected from the registry.
If you collect from `--prompt-data`, apply the task-level filter before freezing
that parquet; `--filter` is rejected with `--prompt-data` by design. Freezing a
`--prompt-data` parquet without that filter costs container boots, not data:
`guard_excluded` in
[/lite/gym/envs/lite/cuagym/main.py](/lite/gym/envs/lite/cuagym/main.py) refuses
each such row at setup with `CuaGymTaskError(kind="excluded_task")` — a terminal
(non-retryable) error, so it is not re-run by `--max-attempts`, produces no
trajectory, and is excluded from the reported mean rather than averaged in as a
zero (see [/devs/envs/lite.cuagym/UPSTREAM_ISSUES.md](/devs/envs/lite.cuagym/UPSTREAM_ISSUES.md)).

Note the two independent namespaces that share the `exclude_reason` key: the
**task-level** one above (catalog rows, applied at import, filtered before
rollout) and the **trajectory-level** one written by the shared filter after
rollout ([Shared Filter](#shared-filter)). They never collide — the first lives
on registry task metadata, the second on collected trajectory metadata — but do
not read a count of one as a count of the other.

Lite.CUAGym spans TWO platforms — `register_jsonl_tasks` registers the web pool
as `browser` and the desktop pool as `desktop` under one split — so each teacher
collects into one log root PER PLATFORM (the teacher runbooks slice them with
`--filter "lambda m: m.dims[0] == ..."`). That gives four log roots and four
configs, `<platform>.use.<teacher>`, and keeps the log-root path
(`.../gpt5_5/$COMMIT/browser/...`) readable off its config name
(`browser.use.gpt5_5`). `--config-names` is positional and 1:1 with `--log-roots`, and stage
only checks that the two lists are the same LENGTH — mislabelling a teacher is
otherwise silent, so keeping the tokens identical is what makes the pairing
checkable by eye.

## Shared Filter

Lite.CUAGym uses [/devs/data/lite.osworld/filter.py](/devs/data/lite.osworld/filter.py).
It is mostly an **annotation** pass: it keeps trajectories and tags quality gates
in the *trajectory-level* `metadata.others.exclude_reason` (comma-joined; the key
is omitted when clean) — a separate namespace from the *task-level*
`exclude_reason` on catalog rows (see [Collection Targets](#collection-targets)).
Four publish-invalid classes are physically dropped before staging: trajectories
whose agent typed a `/opt/env/` path, trajectories with GUI coordinates outside
normalized `[0, 1000]`, a call naming a tool the row never declared, and an
action-batch child naming an action that does not exist. Downstream consumers filter with
`not m.others.get('exclude_reason')`.

```bash
uv run python devs/data/lite.osworld/filter.py \
  --log-root <raw-log-root> \
  --out <annotated-log-root> \
  --drop-loops --drop-undo-storm
```

It tags, in `exclude_reason`:

- `incomplete` — `terminated != true`;
- `dependency_install` — apt/pip/conda/snap/flatpak installs;
- `complex_shell` — a non-teachable terminal *operation* (**operation-driven, not
  structure-driven**: for/while loops, multi-line blocks, `;` / `&&` / single `|`
  of simple commands are KEPT; tagged are `$()` / `<()` / backtick / heredoc,
  `python -c` / `bash -c`, running or authoring code scripts, `sed -i` in-place
  edits, dotfile / `.desktop` authoring, and awk state machines);
- `footgun:loop` / `footgun:undo_storm` — repeated-action loops / undo storms
  when `--drop-loops` / `--drop-undo-storm` are passed;
- `footgun:no_submit` — no explicit final submit tool (`terminate`/`response`),
  only when `--drop-no-submit` is passed; this is separate from the default
  content-only final turn policy (normalized to one plain `text` part, not preserved as emitted);
- `reward_vision_disagree` — SOFT tag, emitted UNCONDITIONALLY (no `--drop` flag gates it):
  the scalar checker reward and the multi-frame vision verdict disagree. It never overwrites
  `episode_return`, and stage publishes the row either way — but the export filter this runbook
  uses (`not m.others.get('exclude_reason')`) drops it, so a tagged row reaches the Hub and not
  the training set.

Reward is deliberately **not** a tag: `episode_return` is already in
`metadata.others.episode_return`, so a consumer thresholds it directly.

On every kept trajectory it also strips `screenshot` and `wait` (keeps bare
Ctrl+S), flattens inline reasoning, and normalizes the content-only final turn to one plain `text` part (`{"type": "text", "text": "Done."}`) — unconditionally, whatever it held before (`inline_reasoning`, `action_description`, both, or anything else). A final turn that DOES carry `tool_calls` is untouched. Synthetic `terminate(status="success")` is opt-in only, and when enabled
the staged row must include the matching canonical nested `terminate` schema in
`metadata.extra_tool_schemas`.

There is no blanket terminal-command-count threshold because legitimate OS
tasks may need several simple commands.

Tests:

```bash
PYTHONPATH="$PWD" uv run pytest -n 0 \
  devs/data/lite.osworld/tests/test_lite_osworld_filter.py -q
```

## Complete Workflow

Run from the repository root. Pipeline: collect → filter/annotate → stage →
upload/download → `export_sft`. Freeze the code revision, audited task parquet,
prompt, and log root for each batch. A resume must use the identical command.

### 1. Install And Configure

```bash
uv sync --locked --extra quick-start --extra gym
uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh

# Reward-judge route, needed for EVERY teacher (it also doubles as the gpt5_5
# agent route). Set these before starting the env-server; the CUA-Gym reward
# judge defaults to the same endpoint settings.
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before collection}"
# Optional; set only for a custom endpoint.
# export OPENAI_BASE_URL="..."
# Set LITE_CUAGYM_JUDGE_* for judge-specific model/base URL/API key/retry/timeout;
# VLM_* are compatibility aliases with lower precedence.

# Start your own env-server on a free port:
uv run python scripts/serve_env.py --port <PORT> --env-ids lite.cuagym
HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:<PORT>

COMMIT="$(git rev-parse --short HEAD)"
CUAGYM_INPUT="<frozen-audited-prompt-data.parquet>"
```

Use the install script rather than a plain Docker build. It provisions pinned
task catalogs/assets, ensures the Lite.OSWorld-derived base when an image build
is needed, and stamps the env-server freshness label.

The image BAKES the web mocks that the imported catalog references. Normal
source/lock/importer changes are covered by image freshness and the mock build
stamp, but manually corrupted local caches are not a source change. After any
forced re-import or catalog repair, run
`uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh provision`;
after HF mirror/cache repair, run maintainer helper
`uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh assets`;
then run
`uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh rebuild`
if local mock dists or the image may be stale.
Sanity-check the import line before building: it must report ~1505 web tasks
across **31 apps** — `across 0 apps` means the catalog is broken.

Beyond the shared judge credentials above, each teacher needs its own agent
credentials or serving step — the `OPENAI_API_KEY` route for `gpt5_5`, and an sglang
server for each of `qwen3_8_27b` / `qwen3_5_27b`. Those live in the teacher runbooks.

### 2. Collect And Annotate (per teacher)

Run [`gpt5_5/AGENTS.md`](/devs/data/lite.cuagym/gpt5_5/AGENTS.md),
[`qwen3_8_27b/AGENTS.md`](/devs/data/lite.cuagym/qwen3_8_27b/AGENTS.md) and
[`qwen3_5_27b/AGENTS.md`](/devs/data/lite.cuagym/qwen3_5_27b/AGENTS.md). All
end with annotated log roots under
`.data/rollout/lite.cuagym/<teacher>/$COMMIT/`.

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
# step between filter and stage. See /devs/data/lite.cuagym/gpt5_5/AGENTS.md.
# DELETE the qwen3_5_27b lines below until that teacher is actually collected — they are
# LIVE as written, and a shell comment cannot be used here (a `#` after a `\` continuation
# swallows the rest of the command). Once it IS published they are MANDATORY again, and
# NOTHING WILL TELL YOU IF YOU FORGET: stage rglobs each
# root and only errors when ALL of them are empty, so a missing one contributes zero rows
# and stage still exits 0 (verified) — then upload sweeps and deletes that teacher's
# published shards. Re-read the blockquote above before every re-stage.
uv run python -m lite.data.hf.stage \
  --log-roots ".data/rollout/lite.cuagym/gpt5_5/$COMMIT/browser/train_annotated.think" \
              ".data/rollout/lite.cuagym/gpt5_5/$COMMIT/desktop/train_annotated.think" \
              ".data/rollout/lite.cuagym/qwen3_8_27b/$COMMIT/browser/train_annotated" \
              ".data/rollout/lite.cuagym/qwen3_8_27b/$COMMIT/desktop/train_annotated" \
              ".data/rollout/lite.cuagym/qwen3_5_27b/$COMMIT/browser/train_annotated" \
              ".data/rollout/lite.cuagym/qwen3_5_27b/$COMMIT/desktop/train_annotated" \
  --config-names browser.use.gpt5_5       desktop.use.gpt5_5 \
                 browser.use.qwen3_8_27b  desktop.use.qwen3_8_27b \
                 browser.use.qwen3_5_27b  desktop.use.qwen3_5_27b \
  --name Lite.CUAGym \
  --repo-dir devs/data/lite.cuagym \
  --overwrite   # the default out dir is $CUA_LITE_DATASETS_ROOT/cua-lite/Lite.CUAGym;
                # stage refuses a non-empty one, so a re-stage needs this

: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload Lite.CUAGym --org "$HF_ORG" --private --tag "$COMMIT"

# Consumer / verification: pull the uploaded revision into canonical local layout.
# NOTE: download verifies LAYOUT only; row content was already gated by stage above.
# upload/download are transport/layout steps, and export_sft below is a conversion smoke.
uv run python -m lite.data.hf.download Lite.CUAGym \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/Lite.CUAGym"
```

**`--config-names` changes the published config set.** With it, the card takes
the override path (`_build_configs_override` in
[/lite/data/hf/card.py](/lite/data/hf/card.py)): the emitted configs are
`default` plus each label VERBATIM, and the auto-derived
`<platform>.<task_type>` cohort configs are NOT emitted. Lite.CUAGym spans two
platforms, so the derived path gave it five configs — `default`, `browser`,
`desktop`, `browser.use`, `desktop.use` (the platform-only pair is redundant
here: each platform carries exactly one task_type, so `browser` and
`browser.use` select the same files). That switch has already happened — the
published repo carries `default` plus the teacher-suffixed labels below. Kept as
a migration record for consumers still pinning an old name:

| was | now (published) |
|---|---|
| `browser.use` | `browser.use.gpt5_5`, `browser.use.qwen3_8_27b` |
| `desktop.use` | `desktop.use.gpt5_5`, `desktop.use.qwen3_8_27b` |
| `browser`, `desktop` | gone — they duplicated the `.use` pair |

`qwen3_5_27b` joins each row once that teacher is collected and staged.

The labels keep the `<platform>.<task_type>` stem because collection is split by
platform (one log root each), so every label names exactly the cohort it covers.

A consumer who wants one teacher pulls only that teacher's shards, either
through the HF config (`load_dataset("cua-lite/Lite.CUAGym",
"desktop.use.qwen3_8_27b")`) or with
`hf.download --allow-patterns '*/*/*/*.use.qwen3_8_27b/*'`. A teacher spans TWO
configs here, one per platform, so no single config name means "all of this
teacher" — the allow-pattern is how you pull both platforms at once.

Record `stage`'s final `seen=... kept=... dropped_by_filter=...` line and the
per-config row lines as the publish gate. Upload/download are transport/layout
smokes only; row content was already gated by `stage`. Use the release org only
after the private upload/readback/export smoke is approved.

### 4. Export SFT Parquet

The three teachers publish DIFFERENT kinds of row, and a consumer that mixes them
should know which it is training on:

| Teacher | Rows carry | Reasoning |
|---|---|---|
| `gpt5_5` | `inline_reasoning` + `action_description` + `tool_calls` | prompted `Thought:` line |
| `qwen3_8_27b` | `action_description` + `tool_calls` | none — runs with thinking off, per its runbook |
| `qwen3_5_27b` | `reasoning_content` + `action_description` + `tool_calls` | native `<think>`, written straight to the canonical field |

The table is what each teacher COLLECTS. `gpt5_5`'s last step before staging
([Internalize Reasoning](/devs/data/lite.cuagym/gpt5_5/AGENTS.md#internalize-reasoning))
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
  --config scripts/configs/qwen3_5/default/lite.cuagym.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --data-paths "${READBACK_ROOT}/cua-lite/Lite.CUAGym" \
  --image-root "${READBACK_ROOT}" \
  --filter "lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5" \
  --num-proc 16 \
  -o .data/sft/qwen3_5/lite-cuagym/train.parquet
```

`--config` is the **rollout** config, not an SFT-only recipe under
`scripts/configs/*/recipes/sft/`: `export_sft` re-renders every step through the
agent adapter, so exporting under a different history window or resolution than
the rollout used trains the model on prompts it will never see at inference.

For a joint Lite.OSWorld + Lite.CUAGym run, pass both dataset paths to one
export command. The processor/model ID must match training because tokenization
and the chat template are frozen during export.

Keep fail-fast enabled; use `--no-strict` only for an identified and recorded
corrupt source row.
