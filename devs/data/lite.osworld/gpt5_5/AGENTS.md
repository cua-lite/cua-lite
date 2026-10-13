# Lite.OSWorld — Collect With `gpt5_5`

Teacher runbook for the `gpt-5.5` rows of Lite.OSWorld. Dataset-level setup,
staging, upload, and export live in [`../AGENTS.md`](/devs/data/lite.osworld/AGENTS.md);
run its §1 first.

This runbook ends at the INTERNALIZED annotated log roots. They are the only thing the
dataset runbook consumes:

    .data/rollout/lite.osworld/gpt5_5/$COMMIT/train.{synth,perturb}_annotated.think

The bare `_annotated` root is an intermediate: it still carries `inline_reasoning`, and
`## Internalize Reasoning` below turns it into the `.think` sibling that stage reads.

## Prompt Design

Canonical recipe:
`scripts/configs/gpt/recipes/collect/lite.osworld.yaml`.

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

Prompt changes alter the training distribution. Run matched-task smoke tests
before a full collection.

The recipe uses `agent_id: gpt.teacher`
(`lite.agents.extensions.teacher.agent.GPTTeacherAgent`), which parses the model's
labelled `Thought:` / `Action:` lines into `inline_reasoning` and
`action_description` content parts. The reasoning therefore lands in an
`inline_reasoning` CONTENT PART, not the top-level `reasoning_content` field —
see the Export section of the dataset runbook for why that matters.

## Collect

```bash
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before collection}"
# Optional; set only for a custom endpoint.
# export OPENAI_BASE_URL="..."

for SUB in synth perturb; do
  uv run python scripts/rollout.py \
    --model-id gpt-5.5 \
    --env-id lite.osworld \
    --splits "train.$SUB" \
    --concurrency 32 \
    --max-attempts 3 \
    --save-data true \
    --save-video false \
    --save-gif false \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --config-path scripts/configs/gpt/recipes/collect/lite.osworld.yaml \
    --log-root ".data/rollout/lite.osworld/gpt5_5/$COMMIT"
done
```

This covers all 2,429 registered train tasks (1,722 synth + 707 perturb); the
`--filter` runs the 2,411 runnable (1,704 synth + 707 perturb), skipping the
18 quarantined synth rows. Re-run the same command to resume.

Collect each source into its OWN subfolder (`train.synth` / `train.perturb` are
registered sub-splits): stage maps log-roots 1:1 to config names, so the
`desktop.use.synth.gpt5_5` / `desktop.use.perturb.gpt5_5` separation depends on
keeping them apart here.

## Annotate And Review

`filter.py` writes `metadata.others.exclude_reason` for ordinary quality gates
(see [Shared Filter](/devs/data/lite.osworld/AGENTS.md#shared-filter)); only
`/opt/env` leaks and OOB coordinates are physically dropped.

```bash
for SUB in synth perturb; do
  uv run python devs/data/lite.osworld/filter.py \
    --log-root ".data/rollout/lite.osworld/gpt5_5/$COMMIT/train.$SUB" \
    --out ".data/rollout/lite.osworld/gpt5_5/$COMMIT/train.${SUB}_annotated" \
    --drop-loops --drop-undo-storm
done
```

Every teacher runs the SAME filter with the SAME flags. That is what makes the
published subsets comparable: a quality difference between them is then a
property of the teacher, not of the annotation pass.

Review the hard-drop counts, the `exclude_reason` tag counts, and sample every
tag class, plus a sample of clean (untagged) and terminal trajectories, before
publishing. Re-run into a fresh annotated root, or pass `--overwrite` only when
intentionally replacing the entire previous output tree.

## Internalize Reasoning

The last `gpt5_5` step before staging, and a `gpt5_5`-only one. This teacher is PROMPTED for a
`Thought:` line — see `inline_reasoning_instruction` in
[the collect recipe](/scripts/configs/gpt/recipes/collect/lite.osworld.yaml) — which the
`gpt.teacher` agent parses into an `inline_reasoning` CONTENT PART;
[`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py) moves it into the
`reasoning_content` FIELD — the same one a teacher sampled with `enable_thinking` writes
natively. Same fact, one shape, so the PUBLISHED rows do not make every consumer ask which
config produced them. The other two teachers skip this step for opposite reasons: `qwen3_8_27b` runs
thinking off and has nothing to move; `qwen3_5_27b` runs thinking ON and its native
`<think>` is already parsed into `reasoning_content` at collection time.

```bash
for C in synth perturb; do
  uv run python devs/data/internalize_cot.py \
    --in  ".data/rollout/lite.osworld/gpt5_5/$COMMIT/train.${C}_annotated" \
    --out ".data/rollout/lite.osworld/gpt5_5/$COMMIT/train.${C}_annotated.think"
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
[The dataset runbook](/devs/data/lite.osworld/AGENTS.md) stages THESE roots for `gpt5_5`.
