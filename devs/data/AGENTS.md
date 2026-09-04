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
| Command | Exact collect/preproc/filter/stage command, including filters and `--config-path` |
| Raw subset | Source split, prompt-data parquet, task-id list, or log-root glob |
| Input / output rows | Raw attempted row count, output row count, and per-config counts |
| Skips / hard drops | Task skips, trajectory `exclude_reason` tag counts, `/opt/env` and OOB hard drops |
| Stage / publish gate | Exact `hf.stage` command plus its `seen=... kept=... dropped_by_filter=...` line and per-config row lines; this is the row-content validation gate |
| Strict validation | Exact `validate_canonical_rows`, `log_contract`, migration `--verify`, or repo-local pytest command and result |
| Visual/sample artifact | Path to a rendered prompt, inspected row JSON, screenshot sample, or QA note |
| Export smoke | Exact `export_sft` command, config/model id, output parquet path, row count, and result |
| Upload transport / readback | `hf.upload --dry-run` or private upload result, tag/revision, and `hf.download --revision` readback path/result |

Use one table or log block per batch; do not replace real evidence with the
synthetic smoke matrix in `lite/data/preproc/AGENTS.md`.

Upload and download are transport/layout checks only. They prove the staged tree
can be packaged, pushed, tagged, and read back; they do not replace the stage
gate, migration `--verify`, filter tests, or `export_sft` conversion smoke.

Row helpers shared by the cohort filters live in
[devs/data/utils.py](/devs/data/utils.py) — `compact_row_images` (the one place
allowed to renumber an image index) and `rebase_images_for_output`. They are
dev-side only and are deliberately NOT in `lite/`; importing them needs the repo
root on `sys.path` (see the `_REPO_ROOT` bootstrap in each filter).

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
