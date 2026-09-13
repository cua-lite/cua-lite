# Data Promotion Evidence

This directory owns real rollout-data collection runbooks. Keep synthetic adapter
contract tests separate from real dataset promotion evidence: a unit test can
prove shape, but it does not prove a real raw subset, skip count, visual sample,
or export path.

Before upload transport or downstream training, record one evidence row for each
promoted dataset batch. The row must include:

| Field | Required evidence |
|---|---|
| Dataset / cohort | Dataset name plus config/cohort names, e.g. `Lite.OSWorld desktop.use.synth.gpt5_5` |
| Producing commit | Pinned cua-lite commit or batch tag used for collection/filter/export |
| Command | Exact collect/preproc/filter/internalize/stage command, including filters and `--config-path`. `gpt5_5` batches internalize reasoning between filter and stage — record its `moved inline_reasoning → reasoning_content on N assistant turns` line |
| Raw subset | Source split, prompt-data parquet, task-id list, or log-root glob |
| Input / output rows | Raw attempted row count, output row count, and per-config counts |
| Skips / hard drops | Task skips, trajectory `exclude_reason` tag counts, `/opt/env`, OOB, undeclared-tool and invalid-action hard drops |
| Stage / publish gate | Exact `hf.stage` command plus its `seen=... kept=... dropped_by_filter=...` line and per-config row lines; this is the row-content validation gate |
| Strict validation | Exact `validate_canonical_rows`, `log_contract`, migration `--verify`, or repo-local pytest command and result |
| Visual/sample artifact | Path to a rendered prompt, inspected row JSON, screenshot sample, or QA note |
| Export smoke | Exact `export_sft` command, config/model id, output parquet path, row count, and result |
| Upload transport / readback | `hf.upload --dry-run` or private upload result, tag/revision, and `hf.download --revision` readback path/result |

Use one table or log block per batch; do not replace real evidence with the
synthetic smoke matrix in `lite/data/preproc/AGENTS.md`.

Run [devs/data/prestage_check.py](/devs/data/prestage_check.py) over the same
log-roots first. `stage` validates each row as it emits it and raises on the
first bad one — but only after copying that root's screenshots into the image
store, so one unpublishable trajectory ends the run tens of minutes and tens of
GB in, with nothing staged. The check calls `stage`'s own
`validate_canonical_rows` in parallel and names every row that would fail, so
the offenders can be set aside in one pass instead of one per restart:

```bash
uv run python devs/data/prestage_check.py --log-roots <the stage command's roots>
```

It owns no rules of its own — if it and `stage` ever disagree, `stage` is right.
A row it rejects is not necessarily a bad trajectory: a malformed tool call the
env rejected visibly, and the model then corrected, still makes the whole row
unpublishable.

Upload and download are transport/layout checks only. They prove the staged tree
can be packaged, pushed, tagged, and read back; they do not replace the stage
gate, migration `--verify`, filter tests, or `export_sft` conversion smoke.

`hf.download` fetches shards with `--max-workers` (default 16). These repos run
to hundreds of shards, so raise it — 32 is roughly 4x the `huggingface_hub`
default of 8 — and run independent datasets concurrently rather than in
sequence.

Row helpers shared by the cohort filters live in
[devs/data/utils.py](/devs/data/utils.py) — `compact_row_images` (the one place
allowed to renumber an image index) and `rebase_images_for_output`. They are
dev-side only and are deliberately NOT in `lite/`; importing them needs the repo
root on `sys.path` (see the `_REPO_ROOT` bootstrap in each filter).

## Add A Config To A Published Dataset

Adding a teacher, a software, or any other config to a repo that already has
some. This is the shared procedure; a dataset runbook adds only its own wrinkle.

**`hf.upload` sweeps.** `orphans = current - planned`: it deletes every published
file the current stage did not plan, and drops those configs from the card.
There is no flag that disables it, and `--skip-existing` protects nothing (it only
skips re-uploading files this run already plans). `--dry-run` does NOT report the
orphan set — the sweep and its logging both sit behind `if not dry_run`. So
**one stage must list every config the repo is to end up with**, old and new.

Which means the input is not what you remember publishing. Read it off the repo:

```bash
# Set these three yourself — the dataset runbook defines them only inside its own §1/§3.
HF_ORG="${HF_ORG:?the org the dataset is published under}"
COMMIT="${COMMIT:?the pinned batch id}"
READBACK_ROOT="$PWD/.data/huggingface-readback"

DL="${READBACK_ROOT}/cua-lite/<Name>"
REC=".data/rollout/<env>/published/$COMMIT"   # where the rebuilt roots go

uv run python -m lite.data.hf.download <Name> --org "$HF_ORG" --out "$DL" --max-workers 32
# config<TAB>dir, one per config. The second column is the DIRECTORY to unstage into, not a
# split: a config spanning train+validation prints `all`, because one unstage call pulls both.
uv run python devs/data/published_configs.py "$DL"
```

If you still have the annotated log roots for everything already published, stop
here: you need no reconstruction. Stage those roots plus the new one, under the
SAME config labels the listing shows, and upload.

Otherwise rebuild the missing ones from the repo. `unstage` writes a rollout
LOG-ROOT (not a staging layout) and must run **once per config** into its own
directory — `stage` maps log-roots to config names 1:1, so one call pouring
several configs into one directory cannot be relabelled afterwards:

```bash
while IFS=$'\t' read -r CFG DIR; do
  # `< /dev/null`: unstage inherits the loop's stdin and could otherwise eat the
  # remaining lines, silently rebuilding fewer configs than the repo has.
  uv run python -m lite.data.hf.unstage --dataset "$DL" \
    --config-names "$CFG" --splits "$DIR" --log-root "$REC/$CFG" < /dev/null
done <<< "$(uv run python devs/data/published_configs.py "$DL")"

# the matching --log-roots/--config-names, 1:1 and in order, for the re-stage
uv run python devs/data/published_configs.py "$DL" --stage-args "$REC"
```

Before pasting: a rebuilt root is raw published rows, so any teacher whose
runbook has a post-collection step still owes it here. For `gpt5_5` that is
`internalize_cot.py` — rows published before that step existed still carry
`inline_reasoning`, and re-staging one beside a `.think` root would put two
reasoning shapes in one repo, silently. Run the teacher's
`## Internalize Reasoning` section on the rebuilt root and stage the `.think`
output; the pass is idempotent, so it costs nothing when the rows are already
canonical.

Then paste that pair into the dataset's stage command, replacing any
`--config-name`/`--config-names` already there, append the NEW config's root and
label, and run ONE stage and ONE upload. `stage` refuses a non-empty output dir,
so a re-stage needs `--overwrite`.

Provenance note: after an unstage→re-stage cycle the card's `## Notes` names the
RECONSTRUCTED log-roots, not the original rollout roots.

## Route Table And Migration Scope

The dev-side uploaded rollout route table is exactly:

| HF dataset route | Owning route doc |
|---|---|
| `Lite.OSWorld` | [devs/data/lite.osworld/AGENTS.md](/devs/data/lite.osworld/AGENTS.md) |
| `Lite.CUAGym` | [devs/data/lite.cuagym/AGENTS.md](/devs/data/lite.cuagym/AGENTS.md) |
| `Lite.CUAWorld` | [devs/data/lite.cuaworld/AGENTS.md](/devs/data/lite.cuaworld/AGENTS.md) |
| `Lite.ScaleCUA` | [devs/data/lite.scalecua/AGENTS.md](/devs/data/lite.scalecua/AGENTS.md) |
| `WebGym` | [devs/data/webgym/AGENTS.md](/devs/data/webgym/AGENTS.md) |

Each route also owns exactly one `devs/data/<route>/repo.json` — the static HF card
fields (`description`, `original_urls`, `license`, `citation`). Pass its directory to `hf.stage`
as `--repo-dir`; `hf.upload` renders `## Origin` and `## License & citation` from it.
It is the single source for both this route's runbook and
[devs/migration/AGENTS.md](/devs/migration/AGENTS.md), so a re-stage from either side
publishes the same upstream attribution. Do not retype those values into a `--description`.

These five routes are also the entire user-defined migration whitelist for
HF-uploaded rollout datasets. The match is the exact canonical dataset route,
not a scratch alias, copy, or lookalike child path. Any other uploaded dataset is
intentionally retired as a migration input: regenerate it from the owning
`lite/data/preproc` raw-source pipeline, then stage/verify the regenerated
canonical rows. Do not add a migration branch for retired uploads.

## Current Real Collection Rows

| Dataset | Real evidence owner | Required collection/export evidence |
|---|---|---|
| Lite.OSWorld | `devs/data/lite.osworld/AGENTS.md` | synth + perturb commands, hard-drop/tag counts from `filter.py`, stage gate output, upload transport/readback result, SFT export parquet and sample inspection |
| Lite.CUAGym | `devs/data/lite.cuagym/AGENTS.md` | frozen prompt-data id, task-level exclude filter proof, hard-drop/tag counts, stage gate output, upload transport/readback result, SFT export parquet and sample inspection |
| Lite.CUAWorld | `devs/data/lite.cuaworld/AGENTS.md` | software list, one-task smoke, per-software counts, hard-drop/tag counts, stage gate output, upload transport/readback result, SFT export parquet and sample inspection |
| Lite.ScaleCUA | `devs/data/lite.scalecua/AGENTS.md` | `rl` + `train` counts, temp-resume evidence when used, hard-drop/tag counts, stage gate output, upload transport/readback result, SFT export parquet and sample inspection |
| WebGym | `devs/data/webgym/AGENTS.md` | per-tier/popular counts, filter drop counts, quality report, stage gate output, upload transport/readback result, SFT export parquet and sample inspection |
