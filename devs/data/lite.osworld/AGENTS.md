# Lite.OSWorld Teacher-Data Pipeline

This directory owns Lite.OSWorld teacher-data collection and the shared desktop
trajectory filter used by Lite.OSWorld-family teacher-data workflows.

Lite.OSWorld publishes trajectories from TWO teachers into one HF repo, each as
its own set of configs. Per-teacher collection and annotation live in their own
runbooks; everything below the annotated log roots is dataset-level and covers
both at once.

| Teacher | Runbook | Model |
|---|---|---|
| `gpt5_5` | [`gpt5_5/AGENTS.md`](/devs/data/lite.osworld/gpt5_5/AGENTS.md) | `gpt-5.5` (API) |
| `qwen3_8_27b` | [`qwen3_8_27b/AGENTS.md`](/devs/data/lite.osworld/qwen3_8_27b/AGENTS.md) | `Qwen/Qwen3.8-27B` (local, sglang) |

The handoff between a teacher runbook and this one is exactly:

    .data/rollout/lite.osworld/<teacher>/$COMMIT/train.{synth,perturb}_annotated

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
| `train.synth` | 1,722 | 1,704 | `desktop.use.synth.gpt5_5`, `desktop.use.synth.qwen3_8_27b` |
| `train.perturb` | 707 | 707 | `desktop.use.perturb.gpt5_5`, `desktop.use.perturb.qwen3_8_27b` |

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
- `footgun:loop` / `footgun:undo_storm` / `footgun:no_submit` — ≥3 identical consecutive actions / ≥4 Ctrl+Z / no submit action;

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
`gpt5_5`, an sglang server for `qwen3_8_27b`. Those live in the teacher
runbooks.

### 2. Collect And Annotate (per teacher)

Run [`gpt5_5/AGENTS.md`](/devs/data/lite.osworld/gpt5_5/AGENTS.md) and
[`qwen3_8_27b/AGENTS.md`](/devs/data/lite.osworld/qwen3_8_27b/AGENTS.md). Both
end with annotated log roots under
`.data/rollout/lite.osworld/<teacher>/$COMMIT/`.

Both teachers run the SAME `filter.py` with the SAME flags. That is deliberate:
it makes the published subsets comparable, so a measured quality difference is a
property of the teacher rather than of the annotation pass.

### 3. Stage, Upload Transport, And Download

> **Upload is a declarative full sync, not an append.** It plans the whole repo
> from the LOCAL staging dir and deletes everything else: `orphans = current -
> planned_paths - {.gitattributes}` are committed as deletions, and the rendered
> README (which defines the HF configs) is rebuilt from local stats alone.
> Staging one teacher and uploading would therefore DELETE the other teacher's
> published shards and drop its configs from the card. There is no flag that
> disables the sweep, and `--skip-existing` does not protect anything (it only
> skips re-uploading files this run already plans). **Every stage must list every
> teacher.**
>
> `--dry-run` does NOT report the orphan set — the whole sweep, including its
> logging, sits behind `if not dry_run`. A dry run only prints the paths it would
> push. The real pre-flight is to diff those planned paths against
> `HfApi().list_repo_files(repo_id=..., repo_type="dataset")` yourself.

To ADD a teacher to an already-published dataset, every stage must still list
EVERYTHING already published — the sweep above deletes whatever this stage
does not plan. Which means two cases, and only one needs `unstage`:

**You still have the other teacher's annotated log roots** (the usual case —
they are under `.data/rollout/lite.osworld/gpt5_5/$COMMIT/`). Nothing to
reconstruct: run the single stage below, listing every teacher's roots, and
upload. Skip the rest of this block.

**Those roots are gone** (a different machine, or the collection tree was
cleaned). Rebuild them from the published repo first. `unstage` writes a rollout
LOG-ROOT, not a staging layout, and it must run **once per config** into its own
directory — `stage` maps log roots to config names 1:1, so one call that pours
several configs into one directory cannot be relabelled afterwards. `stage` also
refuses a non-empty output dir (and with `--overwrite` deletes it), so there is
no "append into the same directory" path:

```bash
# 1. pull the published repo, then unstage ONE config per log-root
uv run python -m lite.data.hf.download Lite.OSWorld --org "$HF_ORG" \
  --out "${READBACK_ROOT}/cua-lite/Lite.OSWorld"
for C in synth perturb; do
  uv run python -m lite.data.hf.unstage \
    --dataset "${READBACK_ROOT}/cua-lite/Lite.OSWorld" \
    --config-names "desktop.use.$C.gpt5_5" --splits "train.$C" \
    --log-root ".data/rollout/lite.osworld/gpt5_5-published/$COMMIT"
done
```

Then run the stage below with the reconstructed roots substituted for the
missing teacher's — `.../gpt5_5-published/$COMMIT/train.$C` in place of
`.../gpt5_5/$COMMIT/train.${C}_annotated` — keeping the SAME config labels. It
still lists all four; only the rebuilt teacher's paths change.

Provenance note: after an unstage→re-stage cycle the card's `## Notes` names the
RECONSTRUCTED log-roots, not the original rollout roots.

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

uv run python -m lite.data.hf.stage \
  --log-roots ".data/rollout/lite.osworld/gpt5_5/$COMMIT/train.synth_annotated" \
              ".data/rollout/lite.osworld/gpt5_5/$COMMIT/train.perturb_annotated" \
              ".data/rollout/lite.osworld/qwen3_8_27b/$COMMIT/train.synth_annotated" \
              ".data/rollout/lite.osworld/qwen3_8_27b/$COMMIT/train.perturb_annotated" \
  --config-names desktop.use.synth.gpt5_5       desktop.use.perturb.gpt5_5 \
                 desktop.use.synth.qwen3_8_27b  desktop.use.perturb.qwen3_8_27b \
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

The two teachers publish DIFFERENT kinds of row, and a consumer that mixes them
should know which it is training on:

| Teacher | Rows carry | Reasoning |
|---|---|---|
| `gpt5_5` | `inline_reasoning` + `action_description` + `tool_calls` | prompted `Thought:` line |
| `qwen3_8_27b` | `action_description` + `tool_calls` | none — runs with thinking off, per its runbook |

Only `gpt5_5` needs `examples/lite/v1/internalize_cot.py`, which moves
`inline_reasoning` parts into `reasoning_content` (what the chat template renders
as `<think>`). Run it before exporting under a thinking-enabled config. The
`qwen3_8_27b` configs have no reasoning to internalize, so exporting them under a
thinking-enabled config would train an empty `<think>` block — pair them with a
config whose `enable_thinking` matches the rollout that produced them.

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
