# Lite.CUAWorld — Collect With `gpt5_5`

Teacher runbook for the `gpt-5.5` rows of Lite.CUAWorld. Dataset-level setup,
staging, upload, and export live in
[`../AGENTS.md`](/devs/data/lite.cuaworld/AGENTS.md); run its §1 first.

This runbook ends at the INTERNALIZED annotated log roots. They are the only thing
the dataset runbook consumes:

    .data/rollout/lite.cuaworld/gpt5_5/$COMMIT/<software>/train_annotated.think

The bare `_annotated` roots are an intermediate: they still carry `inline_reasoning`, and
`## Internalize Reasoning` below turns them into the `.think` siblings that stage reads.

## Prompt Design

Canonical recipe: `scripts/configs/gpt/recipes/collect/lite.cuaworld.yaml` — **functionally
identical to `lite.osworld`'s collect recipe** except `env_id` (a nominal default; `rollout.py`
overrides it per software via `--env-id`). The policy requires:

1. A concise `Thought` grounded in the current screenshot, prior action result, exact task values,
   and remaining global requirements.
2. At most three mechanically coupled tool calls per turn. Any action that changes the next UI
   surface ends the turn; do not predict unseen controls.
3. GUI-first execution. Terminal use is limited to explicitly command-line, inherently OS-level, or
   simple filesystem tasks with no suitable GUI. Interpreters, dependency installs, multiline
   scripts, complex shell logic, hidden config edits, and guessed internal paths are forbidden.
4. Exact custom-color hex values, clean temporary state, application-level saves, committed settings.
5. At most one harmless reversible mistake, only when the task is simple, stable, and safely within
   budget. Never mention training or an intentional mistake in the inline reasoning.
6. Final completion only after the requested state is visibly verified and saved. Do not combine an
   ordinary state-changing action with termination.

The agent policy stays aligned with the Lite.OSWorld recipe; environment-specific changes belong in
this file. Prompt changes alter the training distribution — smoke-test a few softwares first.

The recipe uses `agent_id: gpt.teacher`
(`lite.agents.extensions.teacher.agent.GPTTeacherAgent`), which parses the model's
labelled `Thought:` / `Action:` lines into `inline_reasoning` and
`action_description` content parts. The reasoning therefore lands in an
`inline_reasoning` CONTENT PART, not the top-level `reasoning_content` field —
see the Export section of the dataset runbook for why that matters.

## Collect

Loop the 38 rolloutable softwares on `train`, each into its own subfolder (stage maps log-roots 1:1
to config names). The rollout `--filter` excludes every task the engine flagged with
`exclude_reason`. Start from 24 for CUAWorld and raise only after the env-server, Docker, and
provider error rates are stable; the host-wide concurrency budget lives in the dataset runbook's
[cost / resources section](/devs/data/lite.cuaworld/AGENTS.md#cost--time-resources--disk-hygiene).

```bash
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before collection}"
# Optional; set only for a custom endpoint.
# export OPENAI_BASE_URL="..."

CUAWORLD_CONCURRENCY="${CUAWORLD_CONCURRENCY:-24}"
for SW in <the 38 rolloutable softwares>; do
  uv run python scripts/rollout.py \
    --model-id gpt-5.5 \
    --env-id "lite.cuaworld.$SW" \
    --splits train \
    --concurrency "$CUAWORLD_CONCURRENCY" \
    --max-attempts 3 \
    --save-data true \
    --save-video false \
    --save-gif false \
    --config-path scripts/configs/gpt/recipes/collect/lite.cuaworld.yaml \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --log-root ".data/rollout/lite.cuaworld/gpt5_5/$COMMIT/$SW"
done
```

Re-run the identical command to resume. If you collect `eval`/`long_horizon` as held-out reference
sets, use a separate explicit software list and a separate log root, and keep those splits out of the
SFT mix.

## Annotate And Review

`filter.py` keeps every trajectory except the `/opt/env/` and OOB-coordinate
hard-drops and writes `metadata.others.exclude_reason` (see
[Shared Filter](/devs/data/lite.cuaworld/AGENTS.md#shared-filter)).

```bash
for SW in <softwares>; do
  uv run python devs/data/lite.osworld/filter.py \
    --log-root ".data/rollout/lite.cuaworld/gpt5_5/$COMMIT/$SW/train" \
    --out     ".data/rollout/lite.cuaworld/gpt5_5/$COMMIT/$SW/train_annotated" \
    --drop-loops --drop-undo-storm
done
```

Every teacher runs the SAME filter with the SAME flags. That is what makes the
published subsets comparable: a quality difference between them is then a
property of the teacher, not of the annotation pass.

Review the `exclude_reason` tag counts and sample each tag class + clean/terminal trajectories
before publishing. `devs/data/lite.cuaworld/analyze.py --log-root <root>` reports per-software yield,
failure breakdown, WAIT-patterns, and programmatic-vs-VLM yield (watch for an implausibly-high VLM
yield = garbage passing).

## Internalize Reasoning

The last `gpt5_5` step before staging, and a `gpt5_5`-only one. This teacher is PROMPTED for a
`Thought:` line — see `inline_reasoning_instruction` in
[the collect recipe](/scripts/configs/gpt/recipes/collect/lite.cuaworld.yaml) — which the
`gpt.teacher` agent parses into an `inline_reasoning` CONTENT PART;
[`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py) moves it into the
`reasoning_content` FIELD — the same one a teacher sampled with `enable_thinking` writes
natively. Same fact, one shape, so the PUBLISHED rows do not make every consumer ask which
config produced them. The other two teachers skip this step for opposite reasons: `qwen3_8_27b` runs
thinking off and has nothing to move; `qwen3_5_27b` runs thinking ON and its native
`<think>` is already parsed into `reasoning_content` at collection time.

```bash
for D in .data/rollout/lite.cuaworld/gpt5_5/$COMMIT/*/train_annotated; do
  uv run python devs/data/internalize_cot.py --in "$D" --out "$D.think"
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
[The dataset runbook](/devs/data/lite.cuaworld/AGENTS.md) stages THESE roots for `gpt5_5`.
