# Lite.CUAGym — Collect With `gpt5_5`

Teacher runbook for the `gpt-5.5` rows of Lite.CUAGym. Dataset-level setup,
staging, upload, and export live in [`../AGENTS.md`](/devs/data/lite.cuagym/AGENTS.md);
run its §1 first.

This runbook ends at the INTERNALIZED annotated log root. It is the only thing the dataset
runbook consumes:

    .data/rollout/lite.cuagym/gpt5_5/$COMMIT/{browser,desktop}/train_annotated.think

The bare `_annotated` root is an intermediate: it still carries `inline_reasoning`, and
`## Internalize Reasoning` below turns it into the `.think` sibling that stage reads.

## Prompt Design

Canonical recipe:
`scripts/configs/gpt/recipes/collect/lite.cuagym.yaml`.

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
but environment-specific changes belong in this file.

The recipe uses `agent_id: gpt.teacher`
(`lite.agents.extensions.teacher.agent.GPTTeacherAgent`), which parses the model's
labelled `Thought:` / `Action:` lines into `inline_reasoning` and
`action_description` content parts. The reasoning therefore lands in an
`inline_reasoning` CONTENT PART, not the top-level `reasoning_content` field —
see the Export section of the dataset runbook for why that matters.

## Collect

`--prompt-data` and `--filter` are mutually exclusive (`lite/infer/rollout.py`
rejects the pair), so the per-platform split cannot happen at rollout time — it
has to be baked into the frozen inputs. A prompt-data row's `metadata` carries
only `env_key` / `split` / `env_kwargs` (see `_collect_tasks_from_parquet`), NOT
the routing `dims`, and this env's task ids are UUIDs that do not encode their
platform — so the platform has to come from the registry:

```bash
# once, before collection — split the frozen pool by registered platform
uv run python - "$CUAGYM_INPUT" <<'PY'
import sys
import pandas as pd
import lite.gym as gym

# task_ids() returns {split: [task_id, ...]}; task_metadata() gives the routing
# dims, whose first element is the platform.
plat_of = {
    tid: gym.registry.task_metadata("lite.cuagym", tid).dims[0]
    for tids in gym.registry.task_ids("lite.cuagym").values()
    for tid in tids
}
df = pd.read_parquet(sys.argv[1])
plat = df["metadata"].map(lambda m: plat_of[m["env_key"].split("@", 1)[1]])
for p in ("browser", "desktop"):
    out = sys.argv[1].replace(".parquet", f".{p}.parquet")
    df[plat == p].to_parquet(out)
    print(f"{p}: {(plat == p).sum()} rows -> {out}")
PY
```

Splitting the ALREADY-FROZEN pool keeps whatever task-level filtering went into
it (the `exclude_reason` gate above), so both halves stay publishable.


```bash
# One run per platform: stage needs a log root per config, and the two
# platforms are two configs.
for PLAT in browser desktop; do
  uv run python scripts/rollout.py \
    --model-id gpt-5.5 \
    --env-id lite.cuagym \
    --prompt-data "${CUAGYM_INPUT%.parquet}.$PLAT.parquet" \
    --concurrency 15 \
    --max-attempts 3 \
    --save-data true \
    --save-video false \
    --save-gif false \
    --config-path scripts/configs/gpt/recipes/collect/lite.cuagym.yaml \
    --log-root ".data/rollout/lite.cuagym/gpt5_5/$COMMIT/$PLAT"
done
```

For a fresh sampled pool, freeze `--sample`, `--seed`, and the resolved task
IDs. Re-run the identical command to resume.

Collect this teacher into its OWN log root (`.data/rollout/lite.cuagym/gpt5_5/`):
stage maps log-roots 1:1 to config names, so the per-teacher config separation
depends on keeping them apart here.

## Annotate And Review

`filter.py` keeps every trajectory except the `/opt/env/` and OOB-coordinate
hard-drop cases and writes `metadata.others.exclude_reason` (see
[Shared Filter](/devs/data/lite.cuagym/AGENTS.md#shared-filter)).

```bash
for PLAT in browser desktop; do
  uv run python devs/data/lite.osworld/filter.py \
    --log-root ".data/rollout/lite.cuagym/gpt5_5/$COMMIT/$PLAT/train" \
    --out ".data/rollout/lite.cuagym/gpt5_5/$COMMIT/$PLAT/train_annotated" \
    --drop-loops --drop-undo-storm
done
```

`split` is read PER ROW (`TaskSpec(tid, eid, split or "parquet", ekw)`), so
`$COMMIT/$PLAT/train` holds every frozen row carrying `split: "train"` and only
the rows that omit the field land under `$COMMIT/$PLAT/parquet/` (see
[/lite/infer/rollout.py](/lite/infer/rollout.py)), so freeze the parquet with an
explicit `train` split on every row.

Every teacher runs the SAME filter with the SAME flags. That is what makes the
published subsets comparable: a quality difference between them is then a
property of the teacher, not of the annotation pass.

Review the `exclude_reason` tag counts and sample every tag class, plus a sample
of clean (untagged) and terminal trajectories, before publishing.

## Internalize Reasoning

The last `gpt5_5` step before staging, and a `gpt5_5`-only one. This teacher is PROMPTED for a
`Thought:` line — see `inline_reasoning_instruction` in
[the collect recipe](/scripts/configs/gpt/recipes/collect/lite.cuagym.yaml) — which the
`gpt.teacher` agent parses into an `inline_reasoning` CONTENT PART;
[`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py) moves it into the
`reasoning_content` FIELD — the same one a teacher sampled with `enable_thinking` writes
natively. Same fact, one shape, so the PUBLISHED rows do not make every consumer ask which
config produced them. The other two teachers skip this step for opposite reasons: `qwen3_8_27b` runs
thinking off and has nothing to move; `qwen3_5_27b` runs thinking ON and its native
`<think>` is already parsed into `reasoning_content` at collection time.

```bash
for P in browser desktop; do
  uv run python devs/data/internalize_cot.py \
    --in  ".data/rollout/lite.cuagym/gpt5_5/$COMMIT/$P/train_annotated" \
    --out ".data/rollout/lite.cuagym/gpt5_5/$COMMIT/$P/train_annotated.think"
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
[The dataset runbook](/devs/data/lite.cuagym/AGENTS.md) stages THESE roots for `gpt5_5`.
