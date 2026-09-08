# Lite.ScaleCUA Teacher-Data Pipeline

This directory owns the Lite.ScaleCUA teacher-data workflow.

Lite.ScaleCUA materializes ScaleCUA's OSWorld-shaped `rl` and `train` task
catalogs, then runs them on the Lite.OSWorld desktop runtime. It uses the same
screenshot, coordinate-action, terminal, and 30-turn interaction substrate as
Lite.OSWorld. It therefore reuses the shared quality filter at
`devs/data/lite.osworld/filter.py`; its task pool, prompt, rollout logs, and
published dataset remain separate.

Lite.ScaleCUA publishes trajectories from TWO teachers into one HF repo, each as
its own set of configs. Per-teacher collection and annotation live in their own
runbooks; everything below the annotated log roots is dataset-level and covers
both at once.

| Teacher | Runbook | Model |
|---|---|---|
| `gpt5_5` | [`gpt5_5/AGENTS.md`](/devs/data/lite.scalecua/gpt5_5/AGENTS.md) | `gpt-5.5` (API) |
| `qwen3_8_27b` | [`qwen3_8_27b/AGENTS.md`](/devs/data/lite.scalecua/qwen3_8_27b/AGENTS.md) | `Qwen/Qwen3.8-27B` (local, sglang) |

The handoff between a teacher runbook and this one is exactly:

    .data/rollout/lite.scalecua/<teacher>/$COMMIT/{rl,train}_annotated

## Collection Targets

Lite.ScaleCUA exposes two registered splits, `rl` and `train`. Both include rows
with `metadata.others.exclude_reason`; teacher-data collection must filter those
rows before rollout.

| Split | Registered rows | Runnable rows | HF configs |
|---|---:|---:|---|
| `rl` | 2,049 | 1,809 | `desktop.use.rl.gpt5_5`, `desktop.use.rl.qwen3_8_27b` |
| `train` | 20,289 | 16,139 | `desktop.use.train.gpt5_5`, `desktop.use.train.qwen3_8_27b` |

Runnable counts are a snapshot of the installed catalog, printed here for sizing
only. `$TRAIN_N` from §1 is the authority the commands use.

Every collect command must include:

```bash
--filter "lambda m: not m.others.get('exclude_reason')"
```

`scripts/rollout.py` applies the filter before task execution.

The config name carries the teacher, and the log-root directory uses the SAME
token (`.data/rollout/lite.scalecua/gpt5_5/...` ↔ `desktop.use.*.gpt5_5`).
`--config-names` is positional and 1:1 with `--log-roots`, and stage only checks
that the two lists are the same LENGTH — mislabelling a teacher is otherwise
silent, so keeping the tokens identical is what makes the pairing checkable by
eye.

## Shared Filter

Lite.ScaleCUA uses `devs/data/lite.osworld/filter.py`. It is an **annotation**
pass for ordinary quality gates: it keeps those trajectories and tags them in
`metadata.others.exclude_reason` (comma-joined; the key is omitted when clean).
Four publish-invalid classes are hard-dropped before staging: typed `/opt/env/`
tool leaks, out-of-range GUI coordinates, a call naming a tool the row never
declared, and an action-batch child naming an action that does not exist. Downstream consumers filter with
`not m.others.get('exclude_reason')` (the same idiom as the task-level
`exclude_reason` above, though the meaning differs: task-level marks unrunnable
tasks, trajectory-level marks quality gates).

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
- `footgun:loop` / `footgun:undo_storm` — repeated-action loops / undo storms.

It hard-drops:

- typed `/opt/env/` paths — env-only tool leaks that are not reproducible on the faithful guest;
- OOB coordinates — coordinates outside normalized `[0, 1000]`, which fail publish validation;
- undeclared tool calls — a hallucinated tool NAME (`command(pixels=-3)` for `computer(scroll)`);
- invalid action-batch children — an invented action name (`terminal`), or raw wire text landing in it.

Reward is deliberately **not** a tag: `episode_return` is already in
`metadata.others.episode_return`, so a consumer thresholds it directly (`episode_return > 0.5`).
`--drop-loops` / `--drop-undo-storm` are **still required** — in annotate mode they no longer drop,
they gate whether `footgun:loop` / `footgun:undo_storm` get **tagged**, so the
canonical annotate command must pass them.

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

Run from the repository root. Freeze the code revision, task catalog lock,
prompt, task filter, and log root for each batch. A resume must use the
identical command.

### 1. Install And Configure

```bash
uv sync --locked --extra quick-start --extra gym
uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh
uv run python scripts/serve_env.py --port 30250 --env-ids lite.scalecua

HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:30250

COMMIT="$(git rev-parse --short HEAD)"
TASK_FILTER="lambda m: not m.others.get('exclude_reason')"

# Runnable `train` rows, read from the catalog this host actually serves. The collect
# commands pass it as `--sample "$TRAIN_N"` to shuffle the whole set (see the teacher
# runbooks). DERIVE it, never pin a literal: `--sample` ABOVE the task count silently
# no-ops instead of failing, so a stale literal drops the shuffle without a word.
TRAIN_N=$(uv run python -c "import lite.gym as gym; ids = gym.registry.task_ids('lite.scalecua')['train']; print(sum(1 for t in ids if not gym.registry.task_metadata('lite.scalecua', t).others.get('exclude_reason')))")
```

Use the install script rather than manually importing catalogs. It ensures the
Lite.OSWorld base image is available and validates the ScaleCUA catalog lock.

Each teacher needs its own credentials or serving step — an API key for
`gpt5_5`, an sglang server for `qwen3_8_27b`. Those live in the teacher
runbooks.

### 2. Collect And Annotate (per teacher)

Run [`gpt5_5/AGENTS.md`](/devs/data/lite.scalecua/gpt5_5/AGENTS.md) and
[`qwen3_8_27b/AGENTS.md`](/devs/data/lite.scalecua/qwen3_8_27b/AGENTS.md). Both
end with annotated log roots under
`.data/rollout/lite.scalecua/<teacher>/$COMMIT/`.

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

**You still have the published rows' annotated log roots** (the usual case —
under `.data/rollout/lite.scalecua/<teacher>/$COMMIT/`). Nothing to reconstruct: run the single
stage below listing every root, and upload. Skip the rest of this block.

**Those roots are gone** (a different machine, or the tree was cleaned).
Rebuild them from the published repo first. `unstage` writes a rollout LOG-ROOT (not a
staging layout), and it must be run **once per config** into its own directory —
`stage` maps log-roots to config names 1:1, and one call that pours several
configs into one directory cannot be relabelled afterwards. `stage` also refuses
a non-empty output dir (and with `--overwrite` deletes it), so there is no
"append into the same directory" path:

```bash
# 1. pull the published repo, then unstage ONE config per log-root
uv run python -m lite.data.hf.download Lite.ScaleCUA --org "$HF_ORG" \
  --out "${READBACK_ROOT}/cua-lite/Lite.ScaleCUA"
for C in rl train; do
  uv run python -m lite.data.hf.unstage \
    --dataset "${READBACK_ROOT}/cua-lite/Lite.ScaleCUA" \
    --config-names "desktop.use.$C.gpt5_5" --splits "$C" \
    --log-root ".data/rollout/lite.scalecua/gpt5_5-published/$COMMIT"
done
```

Then run the stage below with the reconstructed roots substituted for the missing
teacher's — `$REC/rl` and `$REC/train` in place of that teacher's `rl_annotated`
and `train_annotated` — keeping the SAME config labels. It still lists every
root; only the rebuilt teacher's paths change.

Provenance note: after an unstage→re-stage cycle the card's `## Notes` names the
RECONSTRUCTED log-roots, not the original rollout roots.

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

uv run python -m lite.data.hf.stage \
  --log-roots ".data/rollout/lite.scalecua/gpt5_5/$COMMIT/rl_annotated" \
              ".data/rollout/lite.scalecua/gpt5_5/$COMMIT/train_annotated" \
              ".data/rollout/lite.scalecua/qwen3_8_27b/$COMMIT/rl_annotated" \
              ".data/rollout/lite.scalecua/qwen3_8_27b/$COMMIT/train_annotated" \
  --config-names desktop.use.rl.gpt5_5       desktop.use.train.gpt5_5 \
                 desktop.use.rl.qwen3_8_27b  desktop.use.train.qwen3_8_27b \
  --name Lite.ScaleCUA \
  --repo-dir devs/data/lite.scalecua \
  --overwrite   # the default out dir is $CUA_LITE_DATASETS_ROOT/cua-lite/Lite.ScaleCUA;
                # stage refuses a non-empty one, so a re-stage needs this

: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload Lite.ScaleCUA --org "$HF_ORG" --private --tag "$COMMIT"

uv run python -m lite.data.hf.download Lite.ScaleCUA \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/Lite.ScaleCUA"
```

A consumer who wants one teacher pulls only that teacher's shards, either
through the HF config (`load_dataset("cua-lite/Lite.ScaleCUA",
"desktop.use.rl.qwen3_8_27b")`) or with
`hf.download --allow-patterns '*/*/*/desktop.use.*.qwen3_8_27b/*'`.

Record `stage`'s final `seen=... kept=... dropped_by_filter=...` line and the
per-config row lines as the publish gate. Upload/download are transport/layout
smokes only; row content was already gated by `stage`.

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
  --data-paths "${READBACK_ROOT}/cua-lite/Lite.ScaleCUA" \
  --image-root "${READBACK_ROOT}" \
  --filter "lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5" \
  --num-proc 16 \
  -o .data/sft/qwen3_5/lite-scalecua/train.parquet
```

`--config` is the **rollout** config, not an SFT-only recipe under
`scripts/configs/*/recipes/sft/`: `export_sft` re-renders every step through the
agent adapter, so exporting under a different history window or resolution than
the rollout used trains the model on prompts it will never see at inference.
There is no `qwen3_5/default/lite.scalecua.yaml` — `lite.scalecua` is an
OSWorld-task adapter riding the `lite.osworld` desktop substrate, and the only
fields `export_sft` reads (`agent_id`, `agent_kwargs`)
are identical across the family's `default/*.yaml`.

For a joint Lite.OSWorld + Lite.ScaleCUA run, pass both dataset paths to one
export command. The processor/model ID must match training because tokenization
and the chat template are frozen during export.

Keep fail-fast enabled; use `--no-strict` only for an identified and recorded
corrupt source row.

### 5. Continue Collecting Across Machines (per-split resume)

Split the full collection across machines (or resume after an interruption) without
re-running what's already been *attempted*. Coordinate through a **throwaway temp HF
dataset** — NOT the canonical `Lite.ScaleCUA`, and **NOT filtered**.

This is per teacher: the example below uses `gpt5_5`, and its collect commands are
the ones from that teacher's runbook. For `qwen3_8_27b`, substitute the teacher
token in BOTH the log root and the config names (they must match) and use that
runbook's collect commands.

Why raw + temp: this round-trip's only job is to tell every machine "which
`(task, sample)` is already done" so it isn't redone. `scripts/rollout.py` resume keys
off `sample_*/summary.json` presence (`get_pending`); `hf.unstage` recreates those
summaries from any staged dataset. So stage the **RAW** log-root — **skip `filter.py`**:
failures carry a summary too, so resume then **skips every attempted sample (success OR
failure)** rather than re-running failures. `filter.py` (the annotation pass — it keeps
ordinary quality-failed trajectories, tags `exclude_reason`, and hard-drops the
four publish-invalid classes; see [Shared Filter](#shared-filter)) runs
**ONCE at the very end** on the final merged log-root to produce the canonical dataset
(the teacher runbook's annotate step → §3). Delete the temp dataset afterward.

**One temp dataset, both configs.** Stage `rl` and `train` into a SINGLE repo as the two
configs `desktop.use.rl.gpt5_5` / `desktop.use.train.gpt5_5` (the same layout the
canonical dataset uses), so the temp dataset IS the accumulating superset and whichever
machine finishes last can produce the complete canonical upload from its own disk.
Screenshots ride embedded in the parquet, so a downloaded config carries its own images.
Requirements: every machine shares the same catalog lock / `$COMMIT` (so `task_id`s
match); each `--config-names ↔ --splits` pair below uses the registry split name.

Machine A — publish raw progress, both configs, no `filter.py`. This WIP upload
is transport/resume coordination only; it is not canonical publish validation:
```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
RESUME_ROOT=".data/rollout/lite.scalecua/gpt5_5/$COMMIT"

uv run python -m lite.data.hf.stage \
  --log-roots "$RESUME_ROOT/rl" "$RESUME_ROOT/train" \
  --config-names desktop.use.rl.gpt5_5 desktop.use.train.gpt5_5 \
  --name Lite.ScaleCUA.wip --description "TEMP resume coordination (raw; delete after merge)"
: "${HF_ORG:?set HF_ORG to your Hub user/org for the private temp repo}"
uv run python -m lite.data.hf.upload Lite.ScaleCUA.wip --org "$HF_ORG" --private --tag "$COMMIT"
```

Machine B — ONE download → unstage EACH config into its own registry-split dir → resume.
`hf.unstage --config-names` reads ONLY that config's parquet, so `rl` rows land under
`rl/` and `train` rows under `train/` with no cross-contamination (a plain `unstage`
globs *all* parquet into the single `--splits` dir, misfiling the other config and
breaking its resume + the final config split). `--filter "$TASK_FILTER"` here is the
*task* exclude-reason gate, unrelated to `filter.py`:
```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
RESUME_ROOT=".data/rollout/lite.scalecua/gpt5_5/$COMMIT"

# one full download (both configs; images are embedded in the parquet)
uv run python -m lite.data.hf.download Lite.ScaleCUA.wip \
  --org "$HF_ORG" \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.wip"

# route each config to its registry split (no --allow-patterns needed)
uv run python -m lite.data.hf.unstage \
  --dataset "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.wip" \
  --log-root "$RESUME_ROOT" --splits rl    --config-names desktop.use.rl.gpt5_5
uv run python -m lite.data.hf.unstage \
  --dataset "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.wip" \
  --log-root "$RESUME_ROOT" --splits train --config-names desktop.use.train.gpt5_5

# resume each split — identical to the teacher runbook's collect commands
# (resume skips attempted)
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.scalecua --splits rl \
  --filter "$TASK_FILTER" --concurrency 32 --max-attempts 2 \
  --save-data true --save-video false --save-gif false \
  --config-path scripts/configs/gpt/recipes/collect/lite.scalecua.yaml --log-root "$RESUME_ROOT"
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.scalecua --splits train \
  --filter "$TASK_FILTER" --sample "$TRAIN_N" --concurrency 32 --max-attempts 2 \
  --save-data true --save-video false --save-gif false \
  --config-path scripts/configs/gpt/recipes/collect/lite.scalecua.yaml --log-root "$RESUME_ROOT"
```

Each machine republishes its grown log-root exactly as Machine A did (stage both configs
→ upload `Lite.ScaleCUA.wip`), so the temp dataset always holds the union. The machine
that finishes last holds the complete superset on disk; it runs `filter.py` (the teacher
runbook's annotate step) → stages the CANONICAL `Lite.ScaleCUA` with every teacher's
configs → uploads (§3) → exports SFT (§4), then deletes `Lite.ScaleCUA.wip`.

Verified: the unit round-trip
(`tests/data/hf/test_unstage.py::test_multi_config_unstage_routes_each_config_to_its_split`),
plus an offline unstage of the real staged dataset — `--config-names desktop.use.rl
--splits rl` / `desktop.use.train --splits train` routed **1839 `rl` + 3844 `train`**
trajectories into their own split dirs with **zero cross-contamination**, so per-split
resume is correct. Because RAW stages every attempted sample (success **or** failure),
resume treats *attempted* as done and never re-runs it; `filter.py` annotates
ordinary quality gates and hard-drops the four publish-invalid classes once at
the end. (That run predates the teacher suffix, so it names the configs as they
were spelled then; the mechanism it verifies is unchanged.)
