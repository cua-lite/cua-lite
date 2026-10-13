# MobileGym — Collect With `gpt5_5`

Teacher runbook for the `gpt-5.5` rows of MobileGym. Dataset-level setup, staging,
upload, and export live in [`../AGENTS.md`](/devs/data/mobilegym/AGENTS.md); run its §1
first.

This runbook ends at the INTERNALIZED annotated log root. It is the only thing the
dataset runbook consumes:

    .data/rollout/mobilegym/gpt5_5/$COMMIT/train_annotated.think

The bare `_annotated` root is an intermediate: it still carries `inline_reasoning`, and
`## Internalize Reasoning` below turns it into the `.think` sibling that stage reads.

## Prompt Design

Canonical recipe:
[`scripts/configs/gpt/recipes/collect/mobilegym.yaml`](/scripts/configs/gpt/recipes/collect/mobilegym.yaml).

It is the mobile sibling of the `lite.osworld` collect recipe — same distillation
mechanism, a PROMPT-CONTROLLED grounded reasoning channel rather than the OpenAI
reasoning summary — and the recipe's own header explains each mobile-specific clause.
The policy requires:

1. A concise `Thought` before every action, grounded in what is visible on the CURRENT
   screenshot, including the check that the previous action produced its intended visible
   effect.
2. Acting only on controls visible in the current screenshot, and observing the result of
   anything that navigates, scrolls, switches app, or submits.
3. `open_app` to launch by name rather than hunting icons across home screens — shorter
   and more reproducible for the student.
4. No unrequested state change. The judge scores `clean` as well as `success`, so an
   unrequested message sent, post published, item purchased, or setting toggled fails the
   episode even when the goal was met.
5. Finishing through a tool: `terminate(status="success")` for an operate task,
   `response` for a query one — and never `terminate(status="failure")`, which ABORTs and
   scores zero.
6. Verification before finishing, in a turn of its own: never combine the action that
   completed the work with the call that declares it done.

The step budget is PER TASK (L1 15 / L2 30 / L3 45 / L4 60), so unlike the desktop recipe
the prompt cannot name a number; it tells the model the budget is limited and varies.

Prompt changes alter the training distribution. Run matched-task smoke tests before a
full collection.

The recipe uses `agent_id: gpt.teacher`. `agent_id` is a slug and
`lite.agents.factory` composes the key as `compose_key(agent_id, *meta.dims)`, so
mobilegym's `("mobile", "use")` dims resolve it to
`lite.agents.extensions.teacher.agent.GPTMobileTeacherAgent` — the mobile leaf, not the
desktop one. It parses the model's labelled `Thought:` / `Action:` lines into
`inline_reasoning` and `action_description` content parts. The reasoning therefore lands
in an `inline_reasoning` CONTENT PART, not the top-level `reasoning_content` field — see
the Export section of the dataset runbook for why that matters.

## Collect

```bash
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before collection}"
# Optional; set only for a custom endpoint.
# export OPENAI_BASE_URL="..."

uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id mobilegym \
  --splits train \
  --group-size 16 \
  --group-shared-seed false \
  --env-kwargs '{"reward_shaping": true}' \
  --concurrency 128 \
  --max-attempts 3 \
  --save-data true \
  --save-video false \
  --save-gif false \
  --config-path scripts/configs/gpt/recipes/collect/mobilegym.yaml \
  --log-root ".data/rollout/mobilegym/gpt5_5/$COMMIT"
```

160 train templates × 16 draws = 2,560 attempted trajectories. Re-run the same command to
resume.

Three flags on that command are dataset policy, not tuning, and are identical for all
three teachers — see [Sampling](/devs/data/mobilegym/AGENTS.md#sampling-16-draws-per-template-all-different)
and [Reward And The Quality Gate](/devs/data/mobilegym/AGENTS.md#reward-and-the-quality-gate):

- `--group-shared-seed false` — train tasks are templates with `seed: null`, so each draw
  is a DIFFERENT instance. The default (`true`) would pin all 16 samples of a task to one
  shared seed and collect 16 near-duplicates.
- `--env-kwargs '{"reward_shaping": true}'` — on the command line, never in the yaml. CLI
  env_kwargs deep-merge OVER the yaml's per leaf (`lite/infer/rollout.py`), so the
  recipe's `loop_detect` and `extra_tools` survive untouched, and the override is
  recorded in the row's provenance `command`.
- `--group-size 16` — the pool is only 160 tasks.

`--concurrency 128` is a starting point, not a measurement: the env container is a shared
Chromium pool (`max_browsers` × `contexts_per_browser` from the env yaml), and it answers
HTTP 503 → `CapacityExhausted` when saturated rather than degrading quietly. Watch the
env-server error rate on the first batch and record what this host actually sustained.

No task-level `--filter` — the MobileGym manifest has no `exclude_reason` field, so all
160 train tasks are runnable.

## Annotate And Review

`filter.py` writes `metadata.others.exclude_reason` for ordinary quality gates (see
[Shared Filter](/devs/data/mobilegym/AGENTS.md#shared-filter)); only invalid action
names, OOB coordinates and undeclared tool calls are physically dropped.

```bash
uv run python devs/data/mobilegym/filter.py \
  --log-root ".data/rollout/mobilegym/gpt5_5/$COMMIT/train" \
  --out ".data/rollout/mobilegym/gpt5_5/$COMMIT/train_annotated" \
  --drop-loops
```

Every teacher runs the SAME filter with the SAME flags. That is what makes the published
subsets comparable: a quality difference between them is then a property of the teacher,
not of the annotation pass.

`--dry-run` (with no `--out`) prints the same tag counts without writing, which is the
cheap way to see how big the `teacher_gave_up` and `incomplete` classes are before
committing to a full annotate.

Review the hard-drop counts, the `exclude_reason` tag counts, and sample every tag class,
plus a sample of clean (untagged) and terminal trajectories, before publishing. Also
record the reward histogram — how many rows land at `1.0`, in `[0.30, 0.5]`, and below
the gate — since that distribution is what the campaign's gate choice rests on. Re-run
into a fresh annotated root, or pass `--overwrite` only when intentionally replacing the
entire previous output tree.

## Internalize Reasoning

The last `gpt5_5` step before staging, and a `gpt5_5`-only one. This teacher is PROMPTED
for a `Thought:` line — see `inline_reasoning_instruction` in
[the collect recipe](/scripts/configs/gpt/recipes/collect/mobilegym.yaml) — which the
`gpt.teacher` agent parses into an `inline_reasoning` CONTENT PART;
[`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py) moves it into the
`reasoning_content` FIELD — the same one a teacher sampled with `enable_thinking` writes
natively. Same fact, one shape, so the PUBLISHED rows do not make every consumer ask
which config produced them. The other two teachers skip this step for opposite reasons:
`qwen3_8_27b` runs thinking off and has nothing to move; `qwen3_5_27b` runs thinking ON
and its native `<think>` is already parsed into `reasoning_content` at collection time.

```bash
uv run python devs/data/internalize_cot.py \
  --in  ".data/rollout/mobilegym/gpt5_5/$COMMIT/train_annotated" \
  --out ".data/rollout/mobilegym/gpt5_5/$COMMIT/train_annotated.think"
```

Record its `moved inline_reasoning → reasoning_content on N assistant turns` line as part
of the batch's evidence row.

Run it on a root rebuilt by `unstage` too
([Add A Config To A Published Dataset](/devs/data/AGENTS.md#add-a-config-to-a-published-dataset)),
without checking first. Rows published BEFORE this step existed still carry
`inline_reasoning`, and re-staging one of those beside a `.think` root would put two
reasoning shapes in one repo — silently, since nothing downstream rejects either. The
pass is idempotent in CONTENT: on already-canonical rows it finds no `inline_reasoning`
and reports `0 assistant turns`. It is not idempotent in PLACEMENT — a non-empty `--out`
raises `FileExistsError` rather than merging a stale tree into a fresh one, so re-running
over a surviving `.think` root needs `--overwrite`.

The `.think` root holds parquet only: image refs are rewritten to absolute, so the copy
is small and stages the same image bytes. It also pops `raw_response`, whose saved
provider payload no longer matches the mutated message.
[The dataset runbook](/devs/data/mobilegym/AGENTS.md) stages THIS root for `gpt5_5`.
