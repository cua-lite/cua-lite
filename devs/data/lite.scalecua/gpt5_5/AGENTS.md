# Lite.ScaleCUA — Collect With `gpt5_5`

Teacher runbook for the `gpt-5.5` rows of Lite.ScaleCUA. Dataset-level setup,
staging, upload, and export live in [`../AGENTS.md`](/devs/data/lite.scalecua/AGENTS.md);
run its §1 first.

This runbook ends at the INTERNALIZED annotated log roots. They are the only thing
the dataset runbook consumes:

    .data/rollout/lite.scalecua/gpt5_5/$COMMIT/{rl,train}_annotated.think

The bare `_annotated` roots are an intermediate: they still carry `inline_reasoning`, and
`## Internalize Reasoning` below turns them into the `.think` siblings that stage reads.

## Prompt Design

Canonical recipe:
`scripts/configs/gpt/recipes/collect/lite.scalecua.yaml`.

The policy requires:

1. A concise `Thought` grounded in the current screenshot, prior action result,
   exact task values, and remaining global requirements.
2. At most three mechanically coupled tool calls per turn. Any action that
   changes the next UI surface ends the turn; do not predict unseen controls.
3. GUI-first execution. Terminal use is limited to explicitly command-line,
   inherently OS-level, or simple filesystem tasks with no suitable GUI.
   Interpreters, dependency installs, multiline scripts, complex shell logic,
   hidden config edits, and guessed internal paths are forbidden.
4. Exact custom-color hex values, clean temporary state, application-level
   saves, and committed settings.
5. At most one harmless reversible mistake, only when the task is simple,
   stable, and safely within budget. Never mention training or an intentional
   mistake in the inline reasoning.
6. Final completion only after the requested state is visibly verified and
   saved. Do not combine an ordinary state-changing action with termination.

The effective agent policy should remain aligned with the Lite.OSWorld recipe,
but environment-specific changes belong in this runbook.

The recipe emits prompt-controlled `Thought:` / `Action:` lines, which land in
`inline_reasoning` and `action_description` CONTENT PARTS rather than the
top-level `reasoning_content` field — see the Export section of the dataset
runbook for why that matters.

## Collect

Collect each source into its own subfolder (`rl` / `train`), keeping HF config
names separable at stage time.

```bash
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before collection}"
# Optional; set only for a custom endpoint.
# export OPENAI_BASE_URL="..."

uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --splits rl \
  --filter "$TASK_FILTER" \
  --concurrency 32 \
  --max-attempts 2 \
  --save-data true \
  --save-video false \
  --save-gif false \
  --config-path scripts/configs/gpt/recipes/collect/lite.scalecua.yaml \
  --log-root ".data/rollout/lite.scalecua/gpt5_5/$COMMIT"

uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --splits train \
  --filter "$TASK_FILTER" \
  --sample "$TRAIN_N" \
  --concurrency 32 \
  --max-attempts 2 \
  --save-data true \
  --save-video false \
  --save-gif false \
  --config-path scripts/configs/gpt/recipes/collect/lite.scalecua.yaml \
  --log-root ".data/rollout/lite.scalecua/gpt5_5/$COMMIT"
```

`--sample "$TRAIN_N"` shuffles the whole runnable `train` set. `$TRAIN_N` (§1 of the
dataset runbook) is the post-filter count, so this is the N==len case: every task, in
random order, deterministic under the default `--seed 42`. It must equal that count
exactly — BELOW it silently drops tasks, ABOVE it silently skips the shuffle
(`lite/infer/rollout.py` takes the branch only when `sample <= len(tasks)`), which is
why the count is derived rather than pinned. Without it, the catalog is grouped by domain, so tasks run
`chrome → multi_apps → vlc → gimp → …` in blocks; the shuffle interleaves domains so
any partial/resumed run stays domain-balanced. The final dataset is identical either
way — only execution order differs. (`rl` is small enough to leave in catalog order.)

Re-run the identical command to resume.

## Annotate And Review

`filter.py` keeps ordinary quality-failed trajectories and writes
`metadata.others.exclude_reason` (see
[Shared Filter](/devs/data/lite.scalecua/AGENTS.md#shared-filter)); typed
`/opt/env/` leaks and OOB coordinates are hard-dropped.

```bash
uv run python devs/data/lite.osworld/filter.py \
  --log-root ".data/rollout/lite.scalecua/gpt5_5/$COMMIT/rl" \
  --out ".data/rollout/lite.scalecua/gpt5_5/$COMMIT/rl_annotated" \
  --drop-loops --drop-undo-storm

uv run python devs/data/lite.osworld/filter.py \
  --log-root ".data/rollout/lite.scalecua/gpt5_5/$COMMIT/train" \
  --out ".data/rollout/lite.scalecua/gpt5_5/$COMMIT/train_annotated" \
  --drop-loops --drop-undo-storm
```

Every teacher runs the SAME filter with the SAME flags. That is what makes the
published subsets comparable: a quality difference between them is then a
property of the teacher, not of the annotation pass.

Review the `exclude_reason` tag counts and sample every tag class, plus a sample
of clean (untagged) and terminal trajectories, before publishing.

## Internalize Reasoning

The last `gpt5_5` step before staging, and a `gpt5_5`-only one. This teacher is PROMPTED for a
`Thought:` line — see `inline_reasoning_instruction` in
[the collect recipe](/scripts/configs/gpt/recipes/collect/lite.scalecua.yaml) — which the
`gpt.teacher` agent parses into an `inline_reasoning` CONTENT PART;
[`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py) moves it into the
`reasoning_content` FIELD — the same one a teacher sampled with `enable_thinking` writes
natively. Same fact, one shape, so the PUBLISHED rows do not make every consumer ask which
config produced them. The other two teachers skip this step for opposite reasons: `qwen3_8_27b` runs
thinking off and has nothing to move; `qwen3_5_27b` runs thinking ON and its native
`<think>` is already parsed into `reasoning_content` at collection time.

```bash
for C in rl train; do
  uv run python devs/data/internalize_cot.py \
    --in  ".data/rollout/lite.scalecua/gpt5_5/$COMMIT/${C}_annotated" \
    --out ".data/rollout/lite.scalecua/gpt5_5/$COMMIT/${C}_annotated.think"
done
```

Run it on a root rebuilt by `unstage` too
([Add A Config To A Published Dataset](/devs/data/AGENTS.md#add-a-config-to-a-published-dataset)),
without checking first. Rows published BEFORE this step existed still carry
`inline_reasoning`, and re-staging one of those beside a `.think` root would put two
reasoning shapes in one repo — silently, since nothing downstream rejects either. The pass
is idempotent in CONTENT: on already-canonical rows it finds no `inline_reasoning` and
reports `0 assistant turns`. It is not idempotent in PLACEMENT — a non-empty `--out`
raises `FileExistsError` rather than merging a stale tree into a fresh one, so re-running
over a surviving `.think` root needs `--overwrite`.

The `.think` roots hold parquet only: image refs are rewritten to absolute, so the copy is
small and stages the same image bytes. It also pops `raw_response`, whose saved provider
payload no longer matches the mutated message.
[The dataset runbook](/devs/data/lite.scalecua/AGENTS.md) stages THESE roots for `gpt5_5`.
