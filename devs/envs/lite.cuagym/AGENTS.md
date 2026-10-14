# lite.cuagym Developer Guide

This env integrates the pinned upstream CUA-Gym corpus into CUA-Lite. Keep the
scope on integration: import upstream tasks, provide their runtime dependencies,
execute official setup/reward scripts, and document upstream defects without
patching task semantics.

## Scope

- CUA-Lite owns the Docker image, mock serving, display bridging, task
  registration, setup/evaluator transport, env-server services, and integration
  tests.
- Upstream owns `task.json`, `initial_setup.*`, seed assets, mock business
  logic, and `reward.py`.
- Do not add task-specific semantic patches. Validation findings live in
  `data/validation_excludes.json`; review the full live-eligible sweep before running or
  writing it. Rows that are unusable as default training signals because of pinned
  upstream defects are ANNOTATED, never dropped: they carry a closed-vocabulary
  `metadata.others.exclude_reason` (see `EXCLUDE_REASONS` in
  [/lite/gym/envs/lite/cuagym/src/utils/dataset.py](/lite/gym/envs/lite/cuagym/src/utils/dataset.py))
  and are refused at setup. The caller filters with
  `--filter "lambda m: not m.others.get('exclude_reason')"`.

## Assets And Import

Runtime materials come from the pinned Hugging Face dataset
`cua-lite/lite.cuagym-assets`, locked in
`lite/gym/envs/lite/cuagym/data/assets.lock.yaml`.

```bash
uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh provision
# Maintainer/recovery only: force-refresh the pinned HF mirror/cache.
uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh assets
```

`provision` is the docker-free path: env-local Python deps, pinned task
catalogs, and the HF mirror/cache. Plain `install.sh` runs that first and then
builds `cua-lite/lite.cuagym:latest` only if the image is missing/stale; `pull`
checks the remote freshness label first, adopts a matching published image, then
provisions; `push` publishes only a fresh local image; `status` is read-only.
`rebuild` forces the local mock dist cache and Docker image to be rebuilt.
`assets` is a maintainer force-refresh helper, not part of the normal user lifecycle.

The importers register every selected upstream row:

- web and cross-app: 1,505 tasks;
- desktop and missing-platform: 9,405 tasks;
- total `train` split: 10,910 tasks.

Generated caches are ignored by git:

- `.cache/web/lite.cuagym_tasks/train.jsonl`
- `.cache/web/lite.cuagym_tasks/import_report.json`
- `.cache/web/lite.cuagym_tasks/.asset_revision` / `.asset_digest`
- `.cache/desktop/lite.cuagym_desktop_tasks/train.jsonl`
- `.cache/desktop/lite.cuagym_desktop_tasks/import_report.json`
- `.cache/desktop/lite.cuagym_desktop_tasks/.asset_revision` / `.asset_digest`

Import must fail loudly if an upstream row cannot produce a task ID,
instruction, or bundle path. Do not silently drop it. Runtime defects such as a
broken mock or reward remain in the catalog and surface as task-level rollout
errors.

## Backend Notes

- **Web / cross-app**: start each referenced CUA-Gym-Hub mock in the task
  container, materialize `__CUA_GYM_<APP>_(URL|HOST)__`, run upstream
  `initial_setup.py`, and expose cross-app targets as Chrome tabs.
- **Desktop-shaped**: run upstream `initial_setup.py`/`.sh`, or upload and open
  the seed `docx`/`pptx`/`xlsx`; run official `reward.py` at episode end.
- **Runtime compatibility**: bridge upstream `DISPLAY=:0` to the
  `lite.osworld` desktop on `:1`. Runtime-only path/interpreter selection is
  allowed; editing task bundles is not.

## Failure Policy

`CuaGymTaskError` marks deterministic task failures such as invalid bundles,
missing mocks, setup failures, or broken rewards. It is non-retryable: the
shared rollout paths record the failed task/sample and continue sibling tasks.
Training rollout converts failures into empty `FAILED` samples and drops them
from training signal.

Infrastructure failures use shared retryable error types such as
`CapacityExhausted` and `EnvDesktopCrashed`. Never convert one of those into an
`exclude_reason`: that vocabulary is only for deterministic pinned upstream
defects, not transient host/image/provider failures. An infrastructure failure
is transient and retryable; an `exclude_reason` is permanent and terminal, and
mislabelling the first as the second silently shrinks the corpus.

## PR Validation

Use focused unit tests plus representative live no-op samples. Cover web,
cross-app, document, desktop application, setup, GUI observation, and reward
paths. A full live-eligible pre-validation sweep is explicitly not required.

```bash
uv run pytest -q \
  tests/gym/envs/lite/cuagym/test_cuagym.py \
  tests/gym/envs/lite/cuagym/test_cuagym_image_contract.py

# Requires the built image / Docker and is deselected by default.
uv run pytest -q -m live tests/gym/envs/lite/cuagym/test_cuagym_parity.py
```

The full validator covers every currently eligible row (10,397 in the pinned
snapshot):

```bash
uv run python lite/gym/envs/lite/cuagym/scripts/utils/validation_sweep.py \
  --live --all --write --concurrency 8
```

Do not run it before the script, reason mapping, and sample report are reviewed.
Record confirmed upstream defects in `UPSTREAM_ISSUES.md`; do not patch task
semantics in the integration layer.
