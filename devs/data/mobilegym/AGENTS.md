# MobileGym Teacher-Data Pipeline

This directory owns MobileGym teacher-data collection and the MobileGym trajectory
filter used before staging.

MobileGym publishes trajectories from THREE teachers into one HF repo, each as its own
config. Per-teacher collection and annotation live in their own runbooks; everything
below the annotated log roots is dataset-level and covers all of them at once.

| Teacher | Runbook | Model |
|---|---|---|
| `gpt5_5` | [`gpt5_5/AGENTS.md`](/devs/data/mobilegym/gpt5_5/AGENTS.md) | `gpt-5.5` (API) |
| `qwen3_8_27b` | [`qwen3_8_27b/AGENTS.md`](/devs/data/mobilegym/qwen3_8_27b/AGENTS.md) | `Qwen/Qwen3.8-27B` (local, sglang) |
| `qwen3_5_27b` | [`qwen3_5_27b/AGENTS.md`](/devs/data/mobilegym/qwen3_5_27b/AGENTS.md) | `Qwen/Qwen3.5-27B` (local, sglang, **thinking on**) |

The handoff between a teacher runbook and this one is exactly:

    .data/rollout/mobilegym/<teacher>/$COMMIT/train_annotated
    .data/rollout/mobilegym/gpt5_5/$COMMIT/train_annotated.think

`gpt5_5` hands over the `.think` sibling: its Internalize Reasoning step runs after
`filter.py`, and the stage command below reads that root, not the plain `_annotated`
one. The other teachers have no `.think` root at all; their `_annotated` root is the
handoff.

This route uses its own [`filter.py`](/devs/data/mobilegym/filter.py), not the shared
desktop one. MobileGym exposes no terminal and no filesystem, so the desktop pass's
`dependency_install` / `complex_shell` gates and its `/opt/env` leak hard-drop have no
referent here — see [Shared Filter](#shared-filter).

## Collection Targets

Collect the **train** split only. Do not collect `eval` for SFT: it is the held-out
Success-Rate split and its seeds are fixed, so collecting it both leaks eval into
training and reports a number that is no longer a baseline.

| Split | Registered rows | Seed | HF configs |
|---|---:|---|---|
| `train` | 160 | `null` — **re-randomized every reset** | `mobile.use.gpt5_5`, `mobile.use.qwen3_8_27b`, `mobile.use.qwen3_5_27b` |
| `eval` | 256 | `42`, fixed | — (held out) |

Difficulty (`metadata.others.difficulty`), from
[`lite/gym/envs/mobilegym/data/tasks.json`](/lite/gym/envs/mobilegym/data/tasks.json):

| Split | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|
| `train` | 20 | 60 | 54 | 26 |
| `eval` | 20 | 73 | 83 | 80 |

Difficulty also sets the per-task step budget (L1 15 / L2 30 / L3 45 / L4 60), which is
why no collect command pins `max_steps`.

**There is no task-level `--filter` on this route.** Every other dataset here carries
`--filter "lambda m: not m.others.get('exclude_reason')"` because its catalog quarantines
unrunnable rows; the MobileGym manifest has no `exclude_reason` field at all, so all 160
train tasks are runnable and the flag would be noise. `--filter` remains available for
slicing (difficulty, suite, `needs_answer_sheet`, …) — see
[the env README](/lite/gym/envs/mobilegym/README.md#metadata--filtering).

`qwen3_5_27b` joins each row once that teacher is collected and staged.

The config name carries the teacher, and the log-root directory uses the SAME token
(`.data/rollout/mobilegym/gpt5_5/...` ↔ `mobile.use.gpt5_5`). `--config-names` is
positional and 1:1 with `--log-roots`, and stage only checks that the two lists are the
same LENGTH — mislabelling a teacher is otherwise silent, so keeping the tokens identical
is what makes the pairing checkable by eye.

## Sampling: 16 Draws Per Template, All Different

160 tasks is a small pool, so every teacher samples each one 16 times:

```bash
--group-size 16 --group-shared-seed false
```

`--group-shared-seed false` is **load-bearing here, not a tuning knob.** MobileGym train
tasks are parameterized TEMPLATES with `seed: null`, so one `task_id` yields a different
instance on every reset — different contact, amount, search term, starting screen. The
rollout default is `--group-shared-seed true`, which draws ONE per-group seed and injects
it into every sample's `env_kwargs` for non-eval splits
([`lite/infer/rollout.py`](/lite/infer/rollout.py)); on an env that accepts `seed` — this
one does — that pins all 16 samples of a task to the SAME instance. That default exists
for GRPO, where a group must share an instance for the advantage to mean anything. For
teacher data it is the opposite of what we want: it would buy 16 near-duplicate
trajectories per template instead of 16 instances of the template.

With it off, 160 templates × 16 draws = 2,560 attempted trajectories per teacher.

## Reward And The Quality Gate

Collection runs with reward shaping **on the command line**:

```bash
--env-kwargs '{"reward_shaping": true}'
```

Not in the yaml. [`lite/gym/envs/mobilegym/configs/default.yaml`](/lite/gym/envs/mobilegym/configs/default.yaml)
keeps `reward_shaping: false` because eval must report MobileGym's own Success Rate; the
recipe is the shared, unmodified file and the flag is a per-run switch. It is still
recorded per row — `rollout.py` writes the full argv into each run's provenance
`command` (`_cli_command()`), so the override reads straight off the row. A forked yaml's
setting is not on the row.

What `metadata.others.episode_return` then means
([`docker/server.py::_evaluate`](/lite/gym/envs/mobilegym/docker/server.py)):

| value | meaning |
|---|---|
| `1.0` | MobileGym Success Rate: the episode ended COMPLETE **and** `judge.success` (goal met) **and** `judge.clean` (nothing unrequested changed) |
| `[0, 0.5]` | shaped partial credit, `0.5 * progress` — the fraction of `check_goals` that passed, paid only where the SR is 0 |

There is no value in `(0.5, 1.0)`. `progress == 1.0` with a dirty state or no COMPLETE
scores exactly `0.5`, so **`episode_return > 0.5` selects precisely the true successes.**

The campaign's main export filter is:

```
not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) >= 0.30
```

`>= 0.30` is `progress >= 0.60`, plus every true success. The reason it is not `> 0.5`:
an SFT set built only from full successes leaves the follow-on GRPO run no movable band —
the student starts already matching the only behaviour it was shown, and the group
advantage on those tasks collapses. Keeping the 0.60-progress shelf keeps partially
solved templates in the SFT mixture, so GRPO has somewhere to climb. **Run `> 0.5` as a
documented ABLATION on ONE cell of the campaign, not as the default**, and report the two
side by side.

Two things the reward does NOT catch, which is why the filter tags them separately: an
episode that ran out of steps still earns whatever progress it made (`incomplete`), and a
`terminate(status="failure")` ABORT keeps its shaped credit while its terminal act is
giving up (`teacher_gave_up`).

## Gotcha: Unset The Env-Server Vars For Anything That Enumerates Tasks

Task enumeration is routed, not local. `lite.gym.registry._import_env` fetches the task
list over HTTP whenever `CUA_LITE_ENV_SERVER_URL` is set — on EVERY enumeration, not only
as a fallback — and a server that does not serve this env answers 404, which the client
raises as a terminal `EnvUnavailable ... HTTP 404`. So a shell that still has a
co-tenant's (or a previous step's) server exported cannot count MobileGym tasks or export
its prompt data, even though the manifest is sitting on disk.

Prefix every such command:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run python -m lite.train.export.export_tasks --env-id mobilegym --split train \
  -o .data/tasks/mobilegym/train.parquet
```

Same prefix for any `gym.registry.task_ids('mobilegym')` / `task_metadata` one-liner used
for counting. Collection itself is the exception: it is *meant* to reach a server, and
that server is the one §1 starts with `--env-ids mobilegym`.

## Shared Filter

[`filter.py`](/devs/data/mobilegym/filter.py) is the single quality-**annotation** pass
before staging. It **keeps every ordinary quality-failed trajectory** and tags gates in
`metadata.others.exclude_reason` (comma-joined; the key is omitted when clean) —
downstream consumers filter with `not m.others.get('exclude_reason')`. The physical drops
are the three publish-invalid classes: an invented action name, an out-of-range
coordinate, and a call naming a tool the row never declared; all three fail the staging
row-format check.

```bash
uv run python devs/data/mobilegym/filter.py \
  --log-root <raw-log-root> \
  --out <annotated-log-root> \
  --drop-loops
```

It tags, in `exclude_reason`:

- `incomplete` — `metadata.others.terminated != true`: the episode ran out of steps
  without ever calling `terminate` / `response`;
- `teacher_gave_up` — the teacher ended on `terminate(status="failure")`. That is
  MobileGym's ABORT: the container sets `terminated=True` but `completed=False`, and the
  score is `_evaluate(inst, terminated and completed)`, so the Success-Rate branch cannot
  fire and the episode collects `0.5 * progress` instead. It can therefore sit ABOVE the
  0.30 gate on reward alone while its terminal act teaches the student to quit;
- `footgun:loop` — ≥3 identical consecutive actions, gated by `--drop-loops` (passed by
  the canonical command above). In annotate mode the flag does not drop; it gates whether
  the tag is EMITTED.

It hard-drops:

- invalid action-batch children — an invented action name, or raw wire text landing in
  one;
- OOB coordinates — outside the normalized `[0, 1000]` frame MobileGym and CUA-Lite
  share;
- undeclared tool calls — a hallucinated tool NAME the row's `extra_tool_schemas` does
  not declare.

Reward is deliberately **not** a tag: `episode_return` is already in
`metadata.others.episode_return`, so a consumer thresholds it directly (see
[Reward And The Quality Gate](#reward-and-the-quality-gate)).

What the desktop filter has and this one does NOT, deliberately: `dependency_install`,
`complex_shell` and the `/opt/env` leak hard-drop are terminal/filesystem rules and
MobileGym exposes neither; `footgun:undo_storm` counts Ctrl+Z, which a phone has no
equivalent of; `footgun:no_submit` would duplicate `incomplete`, since on this env
"never finished" and "never called a finish tool" are the same fact. They are absent
rather than dormant.

On every kept trajectory it also: strips no-op `screenshot` calls or action-batch child
actions, preserving the canonical action-batch call and any remaining child actions;
flattens inline reasoning to one line; and normalizes a content-only final turn to one
plain `text` part (`"Done."`) unconditionally, whatever it held before. A final turn
carrying `tool_calls` — the `terminate` of an operate task, the `response` of a query one
— is untouched.

`wait` is NOT stripped, unlike on desktop. MobileGym executes it as a real
`ActionType.WAIT`, not a dispatch no-op, and a phone UI that is still loading is exactly
when waiting is right. `--actions screenshot,wait` exists for an ablation that
deliberately trains the student never to wait.

This route ships no filter tests of its own yet. The shared row helpers it uses
(`has_invalid_action_batch`, `compact_row_images`, `rebase_images_for_output`) are covered
by `tests/data/staging/test_staging_contract.py`; the two mobile-specific rules
(`teacher_gave_up`, the `screenshot`-only no-op set) are not. Add
`devs/data/mobilegym/tests/` before the first published batch, modelled on
`devs/data/lite.osworld/tests/`.

## Complete Workflow

Run from the repository root. Pipeline: collect → filter/annotate → stage →
upload/download → `export_sft`. Freeze the code revision, task set, prompt, and log root
for each batch. A resume must use the identical command.

### 1. Install And Configure

```bash
uv sync --locked --extra quick-start --extra gym
uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh

# The container is a browser pool, not a VM per task, so ONE env-server backs every
# teacher. max_browsers/contexts_per_browser come from the env yaml and are forwarded
# into the container by MobileGymContainerServices.ensure.
uv run python scripts/serve_env.py --port 30350 --env-ids mobilegym

HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:30350

COMMIT="$(git rev-parse --short HEAD)"
```

Use the install script rather than a plain Docker build; it stamps the source freshness
label required by env-server, and the image bundles the pinned media dataset (~4.5 GB) so
collection needs no CDN egress.

Each teacher needs its own credentials or serving step — an API key for `gpt5_5`, and an
sglang server for each of `qwen3_8_27b` / `qwen3_5_27b`. Those live in the teacher
runbooks.

### 2. Collect And Annotate (per teacher)

Run [`gpt5_5/AGENTS.md`](/devs/data/mobilegym/gpt5_5/AGENTS.md),
[`qwen3_8_27b/AGENTS.md`](/devs/data/mobilegym/qwen3_8_27b/AGENTS.md) and
[`qwen3_5_27b/AGENTS.md`](/devs/data/mobilegym/qwen3_5_27b/AGENTS.md). All end with
annotated log roots under `.data/rollout/mobilegym/<teacher>/$COMMIT/`.

Every teacher runs the SAME `filter.py` with the SAME flags, and the same
`--group-size` / `--group-shared-seed` / `--env-kwargs`. That is deliberate: it makes the
published subsets comparable, so a measured quality difference is a property of the
teacher rather than of the collection or annotation pass.

### 3. Stage, Upload Transport, And Download

> **Upload is a declarative full sync, not an append.** It plans the whole repo from the
> LOCAL staging dir and deletes everything else, so staging one teacher and uploading
> DELETES the other's published shards. Adding a teacher (or any config) to an
> already-published repo therefore has one shared procedure — read what the repo actually
> holds, then stage all of it in one call:
> [Add A Config To A Published Dataset](/devs/data/AGENTS.md#add-a-config-to-a-published-dataset).

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

# The gpt5_5 root below is the `.think` sibling its teacher runbook produced: reasoning
# canonicalized into `reasoning_content` before staging, so the PUBLISHED rows carry one
# reasoning shape. `qwen3_8_27b` runs thinking off and has none to move, so its root is
# used as-is, and `qwen3_5_27b` needs none either — it is sampled with thinking ON, so its
# native <think> is already in `reasoning_content` at collection time. Only gpt5_5 has a
# step between filter and stage. See /devs/data/mobilegym/gpt5_5/AGENTS.md.
# DELETE the lines for any teacher not yet collected — they are LIVE as written, and a
# shell comment cannot be used here (a `#` after a `\` continuation swallows the rest of
# the command). Once a teacher IS published its lines are MANDATORY again, and NOTHING
# WILL TELL YOU IF YOU FORGET: stage rglobs each root and only errors when ALL of them
# are empty, so a missing one contributes zero rows and stage still exits 0 — then upload
# sweeps and deletes that teacher's published shards. Re-read the blockquote above before
# every re-stage.
uv run python -m lite.data.hf.stage \
  --log-roots ".data/rollout/mobilegym/gpt5_5/$COMMIT/train_annotated.think" \
              ".data/rollout/mobilegym/qwen3_8_27b/$COMMIT/train_annotated" \
              ".data/rollout/mobilegym/qwen3_5_27b/$COMMIT/train_annotated" \
  --config-names mobile.use.gpt5_5 mobile.use.qwen3_8_27b mobile.use.qwen3_5_27b \
  --name MobileGym \
  --repo-dir devs/data/mobilegym \
  --overwrite   # the default out dir is $CUA_LITE_DATASETS_ROOT/cua-lite/MobileGym;
                # stage refuses a non-empty one, so a re-stage needs this

: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload MobileGym --org "$HF_ORG" --private --tag "$COMMIT"

# Consumer / verification: pull the uploaded revision into canonical local layout.
# NOTE: download verifies LAYOUT only; row content was already gated by stage above.
uv run python -m lite.data.hf.download MobileGym \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/MobileGym"
```

A consumer who wants one teacher pulls only that teacher's shards, either through the HF
config (`load_dataset("cua-lite/MobileGym", "mobile.use.qwen3_8_27b")`) or with
`hf.download --allow-patterns '*/*/*/mobile.use.qwen3_8_27b/*'`.

Record `stage`'s final `seen=... kept=... dropped_by_filter=...` line and the per-config
row lines as the publish gate. Upload is transport only; use the release org only after
the private upload/readback/export smoke is approved.

### 4. Export SFT Parquet

The three teachers publish DIFFERENT kinds of row, and a consumer that mixes them should
know which it is training on:

| Teacher | Rows carry | Reasoning |
|---|---|---|
| `gpt5_5` | `inline_reasoning` + `action_description` + `tool_calls` | prompted `Thought:` line |
| `qwen3_8_27b` | `action_description` + `tool_calls` | none — runs with thinking off, per its runbook |
| `qwen3_5_27b` | `reasoning_content` + `action_description` + `tool_calls` | native `<think>`, written straight to the canonical field |

The table is what each teacher COLLECTS. `gpt5_5`'s last step before staging
([Internalize Reasoning](/devs/data/mobilegym/gpt5_5/AGENTS.md#internalize-reasoning))
moves its prompted `Thought:` out of the `inline_reasoning` content part and into the
`reasoning_content` FIELD — the same one a teacher sampled with `enable_thinking` writes
natively. So the PUBLISHED rows carry one vocabulary and consumers run no extra step.

`qwen3_8_27b` has no reasoning to internalize, so a thinking-enabled config on those rows
would train an empty `<think>` block — pair them with a thinking-off config.

One published root serves both recipes for a reasoning teacher: a thinking-on config
renders the reasoning, and a thinking-off one strips `reasoning_content` at the model
boundary ([`lite/train/export/sft_tokenize.py`](/lite/train/export/sft_tokenize.py)),
which is byte-identical to exporting from a root that never carried any.
`scripts/configs/qwen3_5/default/mobilegym.yaml` sets no `enable_thinking`, so the
command below is the thinking-off recipe; the thinking-on twins live with the campaign
that trains them
([`/devs/exps/train/mobile/configs/qwen3_5/`](/devs/exps/train/mobile/configs/qwen3_5/)).

```bash
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_5/default/mobilegym.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --data-paths "${READBACK_ROOT}/cua-lite/MobileGym" \
  --image-root "${READBACK_ROOT}" \
  --filter "lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) >= 0.30" \
  --num-proc 16 \
  -o .data/sft/qwen3_5/mobilegym/train.parquet
```

`--config` is the **rollout** config, not an SFT-only recipe under
`scripts/configs/*/recipes/sft/`: `export_sft` re-renders every step through the agent
adapter, so exporting under a different history window or resolution than the rollout
used trains the model on prompts it will never see at inference. The campaign profiles
in `/devs/exps/train/mobile/configs/qwen3_5/` pin `resolution: [720, 1600]` for exactly
this reason — the teacher saw the env-native 1080×2400 and the student's export resizes
once, at the model boundary.

The processor/model ID must match training because tokenization and the chat template are
frozen during export. Keep fail-fast enabled; use `--no-strict` only for an identified
and recorded corrupt source row.

For the `> 0.5` successes-only ablation, run the same command with that predicate into a
sibling output path and record both row counts in the batch's evidence row.
