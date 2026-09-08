# Lite.CUAGym Teacher-Data Pipeline

This directory owns the Lite.CUAGym teacher-data workflow.

Lite.CUAGym imports CUA-Gym web, cross-app, and desktop task bundles. Its
Docker image extends `cua-lite/lite.osworld:latest`, and its desktop tasks use
the same screenshot, coordinate-action, terminal, and 30-turn interaction
substrate as Lite.OSWorld. It therefore reuses the shared quality filter at
`devs/data/lite.osworld/filter.py`; its task pool, prompt, rollout logs, and
published dataset remain separate.

Lite.CUAGym publishes trajectories from TWO teachers into one HF repo. It spans
two platforms, so each teacher lands in TWO configs — one per platform. Per-teacher collection and annotation live in their own
runbooks; everything below the annotated log roots is dataset-level and covers
both at once.

| Teacher | Runbook | Model |
|---|---|---|
| `gpt5_5` | [`gpt5_5/AGENTS.md`](/devs/data/lite.cuagym/gpt5_5/AGENTS.md) | `gpt-5.5` (API) |
| `qwen3_8_27b` | [`qwen3_8_27b/AGENTS.md`](/devs/data/lite.cuagym/qwen3_8_27b/AGENTS.md) | `Qwen/Qwen3.8-27B` (local, sglang) |

The handoff between a teacher runbook and this one is exactly:

    .data/rollout/lite.cuagym/<teacher>/$COMMIT/{browser,desktop}/train_annotated

## Collection Targets

Lite.CUAGym exposes one registered `train` split of 10,910 tasks (1,505 web +
9,405 desktop — every pinned upstream row is registered; none is dropped). For
teacher-data collection, use either a frozen audited `--prompt-data` parquet or
an explicitly frozen sample/seed/task-id list. The full registered train split
includes known upstream setup/reward failures, so a seed alone does not define a
publishable collection.

**494 of those 10,910 rows are unusable as default training signals** and carry
a task-level `metadata.others.exclude_reason` from the closed vocabulary in
[/lite/gym/envs/lite/cuagym/src/utils/dataset.py](/lite/gym/envs/lite/cuagym/src/utils/dataset.py)
(broken/empty/no-sentinel/mismatched `reward.py`, unbuildable GitHub/Trello
mocks, Google Drive blank-render rows, and deterministic pinned setup defects).
Nothing is dropped — the rows are annotated and you filter them out, leaving
10,416 default-collectable tasks:

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
- `oob_coordinate` — a coordinate outside normalized `[0, 1000]`;
- `reward_vision_disagree` — scalar reward and multi-frame visual judgement disagree.

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

# Reward-judge route, needed for BOTH teachers (it also doubles as the gpt5_5
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
credentials or serving step — the `OPENAI_API_KEY` route for `gpt5_5`, an sglang
server for `qwen3_8_27b`. Those live in the teacher runbooks.

### 2. Collect And Annotate (per teacher)

Run [`gpt5_5/AGENTS.md`](/devs/data/lite.cuagym/gpt5_5/AGENTS.md) and
[`qwen3_8_27b/AGENTS.md`](/devs/data/lite.cuagym/qwen3_8_27b/AGENTS.md). Both
end with annotated log roots under
`.data/rollout/lite.cuagym/<teacher>/$COMMIT/`.

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

**You still have the published rows' per-platform annotated log roots** (under
`.data/rollout/lite.cuagym/<teacher>/$COMMIT/{browser,desktop}/`). Nothing to
reconstruct: run the single stage below listing every root, and upload; skip the
rest of this block. Note this does NOT cover the rows published BEFORE the
per-platform split — those were collected into one mixed root, which cannot
supply the two labels the stage command needs. For those, use the rebuild path.

**Those roots are gone** (a different machine, or the tree was cleaned).
Rebuild them from the published repo first. `unstage` writes a rollout LOG-ROOT (not a
staging layout), and it must be run **once per config** into its own directory —
`stage` maps log-roots to config names 1:1, and one call that pours several
configs into one directory cannot be relabelled afterwards. `stage` also refuses
a non-empty output dir (and with `--overwrite` deletes it), so there is no
"append into the same directory" path. The already-published Lite.CUAGym rows
were staged BEFORE `--config-names` existed here, so their single config is the
default variant label `rollout`; unstage that one config and re-stage it under
the `gpt5_5` label:

```bash
# 1. pull the published repo
uv run python -m lite.data.hf.download Lite.CUAGym --org "$HF_ORG" \
  --out "${READBACK_ROOT}/cua-lite/Lite.CUAGym"

# 2. unstage the one published config, then SPLIT the reconstruction by platform:
#    the published `rollout` config predates the per-platform split, so a single
#    unstage yields ONE root holding both platforms. Staging that root as-is would
#    label every row with one config name — desktop rows published under
#    `browser.use.gpt5_5`, silently.
REC=".data/rollout/lite.cuagym/gpt5_5-published/$COMMIT"
uv run python -m lite.data.hf.unstage \
  --dataset "${READBACK_ROOT}/cua-lite/Lite.CUAGym" \
  --config-names rollout --splits train --log-root "$REC"

uv run python - "$REC" <<'PY'
import shutil, sys
from pathlib import Path
import lite.gym as gym

rec = Path(sys.argv[1])
plat_of = {
    tid: gym.registry.task_metadata("lite.cuagym", tid).dims[0]
    for tids in gym.registry.task_ids("lite.cuagym").values()
    for tid in tids
}
for task_dir in sorted((rec / "train").iterdir()):
    if not task_dir.is_dir():
        continue
    dst = rec.with_name(rec.name + f".{plat_of[task_dir.name]}") / "train"
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / task_dir.name
    # shutil.move onto an EXISTING dir moves the source INSIDE it, silently
    # nesting a level. Refuse instead: a populated target means this ran before.
    if target.exists():
        sys.exit(f"{target} already exists — clear the split roots and re-run")
    shutil.move(str(task_dir), str(target))
print("split ->", [str(p) for p in rec.parent.glob(rec.name + ".*")])
PY

# 3. then stage, substituting the two split roots for the missing teacher's:
#    "$REC.browser/train" and "$REC.desktop/train" in place of that teacher's
#    two annotated roots, keeping the SAME config labels.
```

Provenance note: after an unstage→re-stage cycle the card's `## Notes` names the
RECONSTRUCTED log-roots, not the original rollout roots.

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

uv run python -m lite.data.hf.stage \
  --log-roots ".data/rollout/lite.cuagym/gpt5_5/$COMMIT/browser/train_annotated" \
              ".data/rollout/lite.cuagym/gpt5_5/$COMMIT/desktop/train_annotated" \
              ".data/rollout/lite.cuagym/qwen3_8_27b/$COMMIT/browser/train_annotated" \
              ".data/rollout/lite.cuagym/qwen3_8_27b/$COMMIT/desktop/train_annotated" \
  --config-names browser.use.gpt5_5       desktop.use.gpt5_5 \
                 browser.use.qwen3_8_27b  desktop.use.qwen3_8_27b \
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
`browser.use` select the same files). After the switch only `default` and the
four labels below exist, so a consumer on `browser.use` or `desktop.use` moves
to the teacher-suffixed name:

| was | now |
|---|---|
| `browser.use` | `browser.use.gpt5_5`, `browser.use.qwen3_8_27b` |
| `desktop.use` | `desktop.use.gpt5_5`, `desktop.use.qwen3_8_27b` |
| `browser`, `desktop` | gone — they duplicated the `.use` pair |

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
