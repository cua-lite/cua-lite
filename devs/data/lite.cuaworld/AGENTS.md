# Lite.CUAWorld Teacher-Data Pipeline

This directory owns the Lite.CUAWorld teacher-data workflow.

Lite.CUAWorld re-hosts the **gym-anything (CMU CUA-World)** desktop software task suite — **40
softwares, one Docker image each** (`lite.cuaworld.<software>`). Its desktop tasks use the same
screenshot, coordinate-action, and terminal substrate as Lite.OSWorld, so it
**reuses the shared quality filter at [`devs/data/lite.osworld/filter.py`](/devs/data/lite.osworld/filter.py)**
(like Lite.CUAGym and Lite.ScaleCUA). Its task pool, prompt, rollout logs, and published dataset
remain separate. Env content lives in the HF materials dataset `cua-lite/lite.cuaworld-assets`
(pinned by `data/assets.lock.yaml`); this repo is engine + pipeline only.

Lite.CUAWorld publishes trajectories from TWO teachers into one HF repo, each as
its own set of configs. Per-teacher collection and annotation live in their own
runbooks; everything below the annotated log roots is dataset-level and covers
both at once.

| Teacher | Runbook | Model |
|---|---|---|
| `gpt5_5` | [`gpt5_5/AGENTS.md`](/devs/data/lite.cuaworld/gpt5_5/AGENTS.md) | `gpt-5.5` (API) |
| `qwen3_8_27b` | [`qwen3_8_27b/AGENTS.md`](/devs/data/lite.cuaworld/qwen3_8_27b/AGENTS.md) | `Qwen/Qwen3.8-27B` (local, sglang) |

The handoff between a teacher runbook and this one is exactly:

    .data/rollout/lite.cuaworld/<teacher>/$COMMIT/<software>/train_annotated

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
| `train` | 2,419 | 1,745 | `desktop.use.<software>.gpt5_5`, `desktop.use.<software>.qwen3_8_27b` (one config per software per teacher) |
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
`metadata.others.exclude_reason` (comma-joined; the key is omitted when clean). Two
publish-invalid classes are physically dropped before staging: a trajectory whose
agent typed a `/opt/env/` path leaked an env-only tool tree and is
non-reproducible, and any trajectory with GUI coordinates outside normalized
`[0, 1000]` would fail publish validation. Downstream selects the training set with
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
- `oob_coordinate` — a coordinate outside normalized `[0, 1000]`;
- `reward_vision_disagree` — scalar reward and multi-frame visual judgement disagree.

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
step — an API key for `gpt5_5`, an sglang server for `qwen3_8_27b`. Those live
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

Run [`gpt5_5/AGENTS.md`](/devs/data/lite.cuaworld/gpt5_5/AGENTS.md) and
[`qwen3_8_27b/AGENTS.md`](/devs/data/lite.cuaworld/qwen3_8_27b/AGENTS.md). Both
loop the 38 rolloutable softwares on `train`, one log-root subfolder per
software, and end with annotated log roots under
`.data/rollout/lite.cuaworld/<teacher>/$COMMIT/<software>/train_annotated`.

Both teachers run the SAME `filter.py` with the SAME flags. That is deliberate:
it makes the published subsets comparable, so a measured quality difference is a
property of the teacher rather than of the annotation pass.

### 3. Stage, Upload Transport, And Download

> **Upload is a declarative full sync, not an append.** It plans the whole repo
> from the LOCAL staging dir and deletes everything else: `orphans = current -
> planned_paths - {.gitattributes}` are committed as deletions, and the rendered
> README (which defines the HF configs) is rebuilt from local stats alone.
> Staging one teacher and uploading would therefore DELETE the other teacher's
> published shards and drop its configs from the card — and the same is true of
> any software left out of the staged set. There is no flag that disables the
> sweep, and `--skip-existing` does not protect anything (it only skips
> re-uploading files this run already plans). **Every stage must list every
> teacher and every published software.**
>
> `--dry-run` does NOT report the orphan set — the whole sweep, including its
> logging, sits behind `if not dry_run`. A dry run only prints the paths it would
> push. The real pre-flight is to diff those planned paths against
> `HfApi().list_repo_files(repo_id=..., repo_type="dataset")` yourself.

To ADD a teacher or a software to an already-published dataset, every stage must still list
EVERYTHING already published — the sweep above deletes whatever this stage
does not plan. Which means two cases, and only one needs `unstage`:

**You still have the published rows' annotated log roots** (the usual case —
under `.data/rollout/lite.cuaworld/<teacher>/$COMMIT/`). Nothing to reconstruct: run the single
stage below listing every root, and upload. Skip the rest of this block.

**Those roots are gone** (a different machine, or the tree was cleaned).
Rebuild them from the published repo first. `unstage` writes a rollout LOG-ROOT
(not a staging layout), and it must be run **once per config** into its own
directory — `stage` maps log-roots to config names 1:1, and one call that pours
several configs into one directory cannot be relabelled afterwards. `stage` also
refuses a non-empty output dir (and with `--overwrite` deletes it), so there is
no "append into the same directory" path:

```bash
# 1. pull the published repo, then unstage ONE config per log-root
uv run python -m lite.data.hf.download Lite.CUAWorld --org "$HF_ORG" \
  --out "${READBACK_ROOT}/cua-lite/Lite.CUAWorld"
for SW in <the softwares already published for this teacher>; do
  uv run python -m lite.data.hf.unstage \
    --dataset "${READBACK_ROOT}/cua-lite/Lite.CUAWorld" \
    --config-names "desktop.use.$SW.gpt5_5" --splits train \
    --log-root ".data/rollout/lite.cuaworld/gpt5_5-published/$SW"
done
# unstage writes <log-root>/train/..., so each reconstruction enters stage below
# as ".data/rollout/lite.cuaworld/gpt5_5-published/$SW/train".
# 2. then run ONE stage listing old + new roots, and ONE upload (below)
```

Provenance note: after an unstage→re-stage cycle the card's `## Notes` names the
RECONSTRUCTED log-roots, not the original rollout roots.

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

uv run python -m lite.data.hf.stage \
  --log-roots .data/rollout/lite.cuaworld/gpt5_5/$COMMIT/*/train_annotated \
              .data/rollout/lite.cuaworld/qwen3_8_27b/$COMMIT/*/train_annotated \
  --config-names <one desktop.use.<software>.gpt5_5 per gpt5_5 log-root, then one desktop.use.<software>.qwen3_8_27b per qwen3_8_27b log-root, same order> \
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

The two `--log-roots` globs expand in sorted order, so build the
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
concurrency across CUAGym/CUAWorld at 48 or lower; when both are active, budget them explicitly
(for example 24 + 24) instead of letting one campaign monopolize the host. The same ceiling covers
both teachers: two CUAWorld collections running at once share it, they do not each get their own.
`blender3d` needs `gpus=1` (the
only GPU env); `gmat`'s upstream download is dead (rescue the retagged image). After **stage**
captures a batch, delete its raw `$SW/<split>` and `$SW/<split>_annotated` roots. Clean up only your
own env-server containers (named by your port) afterward. Task/verifier defects:
[/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md](/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md).
