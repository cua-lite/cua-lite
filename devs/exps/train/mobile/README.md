# mobile.use — screenshot-profile x reasoning campaign (SFT + one filter ablation + one RL run)

Train **five checkpoints** — two screenshot profiles on the `gpt5_5` teacher, plus a `<think>` arm
on each, plus one filter ablation — and score them on the full MobileGym
eval split. A single GRPO run from the `gpt5_5` + `<think>` SFT cell closes the file (**RL** at the
end), scored on that same eval.

| | `i1` | `i4` |
|---|---|---|
| | 720x1600, 1 img | 720x1600, 1-4 img |
| **`gpt5_5`** | ✓ | ✓ |
| **`gpt5_5` + `<think>`** | ✓ | ✓ |

Four cells, plus a fifth that is not in the grid: the GRPO cell's own
(`gpt5_5`, `mobile.use.i1.reasoning`) pair exported a second time under a **stricter row filter**.
See [Filter ablation](#filter-ablation).

Both profiles cap SCREENSHOTS and nothing else: `iN` sets `image_max: N` (with `fold_size` tracking
it) and leaves `history_n` at the protocol default of 100, so every past turn is still rendered in
full — its text, its literal `<tool_call>`, and on the reasoning arm its `<think>`. Only the old
*pixels* collapse.

**One resolution, 720x1600, across all four configs.** Unlike the desktop campaign there is no
`lowr`/`highr` axis here, so the profile name carries only the image budget and `i1` vs `i4` is the
campaign's single prompt-surface contrast. 720x1600 is the frame the screenshot is resized to, not
the tensor shape: the mobile adapter sets `smart_resize_enabled=False`, but the HF processor still
rounds to its own patch grid, so the model sees **704x1600** (`image_grid_thw` `[1,100,44]`) — the
same relationship desktop's 1280x720 has to 1280x704. Coordinates are unaffected by any of this:
canonical Lite coordinates are NORMALIZED to `[0, 1000]`
([`geometry.py`](/lite/core/tools/action_space/geometry.py)), so a teacher acting on the env-native
1080x2400 exports into this profile with its numbers unchanged (measured: 56/56 coordinate pairs).
The `smart_resize_enabled=False` rationale in `adapter.py` cites a coordinate hazard that belongs to
Qwen2.5-VL, which is pixel-in-resized-frame; it does not apply to this family.

> **Train and eval must use the same profile yaml.** `$P` — the config stem — selects both the
> checkpoint and `--config-path`, in every SFT block below. An `i1` checkpoint scored under `i4` is
> measuring a prompt surface it never saw.

> **`reward_shaping` is ON everywhere — collection, training and eval — and every manifest pins it
> explicitly.** The profile yamls leave it unset, so the env default (`false`) applies to anything a
> manifest or `--env-kwargs` does not pin, and that default costs resolution rather than buying
> correctness: shaping is a superset, `episode_return == 1.0` is still exactly MobileGym's Success
> Rate, and the mean additionally resolves partial progress that a 0/1 metric cannot. Pin it on both
> sides of every export. See [Reward shaping](#reward-shaping).

> **`CUA_LITE_ENV_SERVER_URL` must be UNSET for every export command.** With it set, the task
> registry resolves through that env-server, and a server that does not serve `mobilegym` fails the
> export with `EnvUnavailable ... (HTTP 404)` (verified in this session: with the var set the export
> dies; without it, train exports 160 rows and eval 256). Every export block below is prefixed with
> `env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN`; keep the prefix even when you think
> no server is running, because a stale `export` in a reused shell is exactly how this bites.

**What compares to what.** Within a ROW only the image budget moves: the two cells draw the same
filtered pool. Within a COLUMN, a `+ <think>` row differs from the row above by `enable_thinking`
alone. Every cell here draws from the one `gpt5_5` pool, so both contrasts are clean by
construction — there is no cross-teacher row to confound them. **Adding a second teacher would
reintroduce that confound**: each teacher's pool is its own filtered trajectories, differing in
membership and in size, and how much two pools overlap on task id is **not measured here** (desktop
measured 43% for its two extreme teachers; that number is desktop's, not mobile's). Intersect the
pools first and draw every teacher's rows from that intersection.

`gpt5_5` reaches the `<think>` field by prompting: it is asked for a `Thought:` line that
[`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py) canonicalizes before staging. A
teacher sampled with thinking OFF gets no `<think>` arm at all — the reasoning config would train an
empty `<think>` on its rows, the same exclusion the desktop campaign documents. The mobile configs
say the same thing in their own headers —
[`mobile.use.i1.reasoning.yaml`](/devs/exps/train/mobile/configs/qwen3_5/mobile.use.i1.reasoning.yaml).

**Naming.** A cell is a `(config stem, teacher, dataset recipe)` triple; the SFT blocks enumerate the
five from a `cells()` function that each block redefines (four copies — paste the one in the block
you are running). RL pins one of those cells explicitly. `$P` is the stem taken whole off the
filename and `$DS` the dataset recipe (source + row filter). Both are threaded verbatim into every
artifact — parquet `$P.$DS.$T.parquet`, checkpoint `sft.$P.$DS.$T`, HF repo
`ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T`, W&B group — so nothing downstream re-derives them and two recipes
never collide. To ablate the dataset, change `DS=` **and** `--filter` together.

### Teacher data

Collection, cleaning and publication of the teacher's MobileGym trajectories are owned by
[`/devs/data/mobilegym/AGENTS.md`](/devs/data/mobilegym/AGENTS.md). Read it before the Export block;
nothing about collection is repeated here. Two handoffs matter to this campaign:

- **The HF dataset name and the per-teacher config names** come from that runbook. The
  `mobile/use/train/` path prefix is fixed by the canonical staging layout
  ([`lite/data/hf/stage.py`](/lite/data/hf/stage.py)); the variant name after it is the runbook's.
  Check both there before the first `download`.
- **Collection runs with `reward_shaping: true`**
  ([`scripts/configs/gpt/recipes/collect/mobilegym.yaml`](/scripts/configs/gpt/recipes/collect/mobilegym.yaml)),
  which is why published rows carry `episode_return` values strictly between 0 and 1 at all. That is
  what makes the row filter below a real choice rather than a formality.

#### Reward shaping

One number, two meanings, and the difference is the whole reason the manifests below are exported
separately:

| `reward_shaping` | `episode_return` | used by |
|---|---|---|
| `false` | 1.0 iff COMPLETE **and** `judge.success` **and** `judge.clean`, else 0.0 — MobileGym's own Success Rate | nothing in this campaign; see below |
| `true` | the same 1.0 on success, otherwise `0.5 * progress` (fraction of `check_goals` passed), capped strictly below 1.0 so a real success always outranks a partial one | **everything** — teacher collection, RL training, and all eval |

**Shaping is a superset, so eval runs with it ON.** The 1.0 branch is reached by the SR
criterion and by nothing else — the shaped branch is capped at `0.5 * progress <= 0.5` — so in a
shaped run `episode_return == 1.0` is *exactly* SR success. Counting those gives the published
Success Rate back, unchanged, and the mean additionally carries the partial progress a binary metric
discards. One pass, two numbers, nothing lost.

Turning it off would cost real resolution. At `n = 256` and `p ~ 0.29`, the binomial standard error
on the SR alone is `sqrt(.29*.71/256) = 2.8pp` — wider than the entire move an RL run of this size
produces, so a 0/1 metric cannot see it. The shaped mean is continuous and moves when a trajectory
gets further without finishing, which is most of what GRPO changes first.

Two consequences the filters below depend on, both read off
[`lite/gym/envs/mobilegym/docker/server.py`](/lite/gym/envs/mobilegym/docker/server.py):

- `episode_return > 0.5` selects **exactly** the true successes, because the shaped branch can never
  exceed 0.5.
- `episode_return >= 0.30` additionally keeps partial trajectories that passed at least 60% of their
  `check_goals` (`0.30 / 0.5`). That is the campaign filter.

The mechanism that keeps the two sides apart, end to end:
[`export_tasks.py:81`](/lite/train/export/export_tasks.py) writes `--env-kwargs` into every row's
`metadata.env_kwargs`, and [`engine.py:310`](/lite/train/rollout/core/engine.py) deep-merges `args`
(which slime populates from `CONFIG_PATH`,
[`slime/utils/arguments.py`](/slime/slime/utils/arguments.py) `setattr` per top-level key) with that
row, **the row winning per leaf**. So the row is the per-side channel and the yaml is the shared
baseline.

`mobile.use.i1.reasoning.yaml` sets `loop_detect` and `extra_tools` and **leaves `reward_shaping`
unset**, so the env default (`false`) applies to anything the row does not pin. Both manifests
therefore pin it to `true` explicitly — the train one because otherwise GRPO would optimise the
binary SR, the sparse signal shaping exists to avoid; the eval one because otherwise the curve loses
the partial-progress resolution it is there to provide. The assertion after the export checks every
row of both. Pinning both sides to the SAME value looks redundant today and is not: the default is
`false`, so an unpinned manifest silently changes metric, and pinning is what makes the value a
property of this campaign rather than of the env registry.

### SFT

#### Export

One parquet per cell — the config decides what the model sees, so no two cells can share one.

```bash
# --- TRAIN HOST ---  (the Slime container mounts the repo root, so its .data/ is what Train
# reads; exporting on the eval host leaves Train with no parquet to read)
DS=mobilegym_r30                         # dataset recipe: source + row filter (episode_return >= 0.30)
DS_SR=mobilegym_sr                       # the ablation recipe: true successes only (> 0.5)
DL=.data/huggingface                     # per-teacher roots: $DL/$T/cua-lite/<dataset>
OUT=.data/sft/qwen3_5/mobile.use
CFG=devs/exps/train/mobile/configs/qwen3_5
HF_DS=MobileGym                          # published 2026-09-21 as PRIVATE ZHZisZZ/MobileGym, tag
                                         # `83e0629`, one config `mobile.use.gpt5_5`; export from it
                                         # with HF_ORG=ZHZisZZ until it moves to the release org.
                                         # CONFIRM against /devs/data/mobilegym/AGENTS.md before
VARIANT=mobile.use                       # the first run; this runbook does not own either name.
                                         # NOT `mobile.use.train`: the HF path already carries the
                                         # split (`mobile/use/train/`), and the config name is
                                         # `mobile.use.<teacher>` (/devs/data/mobilegym/AGENTS.md).
                                         # `--allow-patterns` ERRORS on a 0-file match, so a wrong
                                         # VARIANT aborts the very first command for all teachers.
FILTER_R30="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) >= 0.30"
FILTER_SR="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

# The five cells as "<config stem> <teacher> <dataset recipe>". The last line is the filter
# ablation, not a grid cell.
cells() {
  for T in gpt5_5; do
    for P in mobile.use.i1 mobile.use.i4; do echo "$P $T $DS"; done
  done
  for T in gpt5_5; do
    for P in mobile.use.i1.reasoning mobile.use.i4.reasoning; do echo "$P $T $DS"; done
  done
  echo "mobile.use.i1.reasoning gpt5_5 $DS_SR"
}

# One root per teacher, because export_sft reads whatever --data-paths names and a shared root
# would pool them. --allow-patterns bounds the walk as well as the fetch, so a warm HF cache
# cannot drag in a sibling split. --overwrite makes this re-runnable. KEEP THIS LIST IN SYNC WITH
# cells(): a teacher in cells() but not here exports against a data root that was never
# downloaded -- and cells() is defined FOUR times below, so a teacher edit touches all four.
for T in gpt5_5; do
  # --org defaults to `cua-lite`; the mobilegym runbook stages to a PRIVATE $HF_ORG first
  # (/devs/data/mobilegym/AGENTS.md), so pass it explicitly until the repo is public.
  env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
    uv run python -m lite.data.hf.download "$HF_DS" --org "${HF_ORG:-cua-lite}" \
      --allow-patterns "mobile/use/train/$VARIANT.$T/*" \
      --out "$DL/$T/cua-lite/$HF_DS" --overwrite
done

# No --sample: every cell takes the teacher's whole filtered pool, so the campaign has no row-cap
# axis. The pool size is MEASURED for gpt5_5 as of 2026-09-21: 1099 rows pass the `>= 0.30` filter
# and 1016
# convert; the other 83 are dropped by export_sft because Qwen3-VL mobile cannot render
# `tap(clicks=2)` (its action enum has only `click` and the wire carries no repeat count). That
# 1016/83 split is identical across all four profiles, so the image budget and `<think>` change
# the prompt, not which rows survive.
# The `exclude_reason` clause is the shared publish-time quality gate; whether these rows carry
# one is /devs/data/mobilegym/AGENTS.md's call. The clause is NOT decorative: filter.py does tag
# mobilegym rows (measured: incomplete / footgun:loop / teacher_gave_up), and dropping it changes
# which rows convert.
while read -r P T D; do
  case "$D" in
    "$DS")    F="$FILTER_R30" ;;
    "$DS_SR") F="$FILTER_SR" ;;
    *) echo "unknown dataset recipe $D for $P $T"; continue ;;
  esac
  env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
    uv run python -m lite.train.export.export_sft \
      --config "$CFG/$P.yaml" \
      --model-id Qwen/Qwen3.5-4B \
      --data-paths "$DL/$T/cua-lite/$HF_DS" \
      --image-root "$DL/$T" \
      --filter "$F" --no-strict \
      -o "$OUT/$P.$D.$T.parquet" < /dev/null   # or the child eats the rest of the cell list
done <<< "$(cells)"


# `--no-strict` makes a conversion failure a SKIP, not an error: a wrong --image-root prints
# "Skipped N rows ... Wrote 0 trajectory rows" and still exits 0, and run_sft.sh would then train
# on an empty parquet. With no --sample there is no target row count to check against, so check
# the two things that are still knowable: the file is non-empty, and the two cells of one
# (teacher, arm) ROW hold the SAME number of rows -- they draw the same pool and differ only in
# image budget, so a mismatch means one of them dropped rows during conversion.
while read -r P T D; do
  env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
    uv run python -c "
import sys, pyarrow.parquet as pq
n = pq.read_metadata(sys.argv[1]).num_rows
print(('EMPTY' if n == 0 else 'ok   '), n, sys.argv[1])
" "$OUT/$P.$D.$T.parquet" < /dev/null
done <<< "$(cells)"
```

The download + export pair is **idempotent and verified**. Run on 2026-09-21 against the published
`ZHZisZZ/MobileGym`, it reproduces byte-for-byte the parquet this campaign's checkpoints were
trained from -- 1016 rows, 83 skipped, the same SHA-256 over every step's prompt / response /
`image_indices` / reward, the same 8131 images -- and a second consecutive run reproduces itself.
`--overwrite` on both commands is what makes a re-run clean rather than additive. The three
published checkpoints predate the dataset's publication and were exported straight from the local
annotated root, so this check is what establishes that they and the Hub path carry the same rows.

#### Train

```bash
# --- Slime container, TRAIN HOST ---  (DS / DS_SR must match the Export block)
# SFT at TP=2 (8 GPUs -> DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't
# THD-pack). Serial: each run takes all 8 GPUs.
#
# SAVE_DIR / SAVE_HF_DIR / WANDB_GROUP_SUFFIX are all MANDATORY here. run_sft.sh keys the two
# checkpoint dirs AND the W&B group off PROMPT_DATA's parent dir (DATA_SLUG, run_sft.sh:131),
# which is `mobile.use` for all five cells -- so unset, the runs overwrite each other's
# checkpoints and land in one W&B group. WANDB_GROUP_SUFFIX has no default; separating them is
# its job.
#
# Leave SAVE_INTERVAL unset: run_sft.sh defaults it to 1000 (run_sft.sh:135), far past what a
# 2-epoch cell of this size takes, so the step-interval save never fires. What writes the two
# iter_* dirs is slime's epoch-boundary save -- `step % num_rollout_per_epoch == 0` in
# slime/slime/utils/misc.py -- so NUM_EPOCH=2 gives exactly 2. Ship gates on finding exactly
# $EPOCHS of them; SKIP on every cell means that premise broke (a slime bump can move it), and
# the checkpoints are still on disk.
DS=mobilegym_r30
DS_SR=mobilegym_sr
# The five cells as "<config stem> <teacher> <dataset recipe>".
cells() {
  for T in gpt5_5; do
    for P in mobile.use.i1 mobile.use.i4; do echo "$P $T $DS"; done
  done
  for T in gpt5_5; do
    for P in mobile.use.i1.reasoning mobile.use.i4.reasoning; do echo "$P $T $DS"; done
  done
  echo "mobile.use.i1.reasoning gpt5_5 $DS_SR"
}

while read -r P T D; do
  TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
    MODEL_ID=Qwen/Qwen3.5-4B \
    SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=2 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
    PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5/mobile.use/$P.$D.$T.parquet \
    SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-4b/sft.$P.$D.$T/iter_{rollout_id} \
    SAVE_DIR=/root/checkpoints/qwen3_5-4b/sft.$P.$D.$T/megatron \
    WANDB_GROUP_SUFFIX=".$P.$D.$T" \
    bash /workspaces/cua-lite/scripts/train/run_sft.sh < /dev/null   # else it eats the cell list
done <<< "$(cells)"
```

- `MBS=1` is the safe start, and **tune it on `i4`** — that column has the longer sequence, so it
  binds first. The peak vision tokens per step and the realized packing ratio are **not measured on
  mobile**: the shape is structural (`fold_size == image_max` keeps four consecutive `i4` steps
  prefix-compatible so `build_segment_samples` packs them; `i1` is never prefix-compatible with its
  successor and cannot pack), but the numbers desktop quotes for that trade are desktop's 1280x720
  frames, not a 720x1600 phone frame. Measure it here before repeating any of them.
- TP=2 is the starting point at 4B; `TP_SIZE=4` if it OOMs.
- `NO_SAVE_OPTIM=1` keeps weights only — these checkpoints are for eval, not for resuming.
- **Export `WANDB_API_KEY` before the first cell.** `run_sft.sh` builds its W&B arguments inside
  `if [ -n "${WANDB_API_KEY:-}" ]`, so without it five multi-hour runs train with no logging at
  all.

#### Ship the checkpoints

Train and eval usually run on different machines, so checkpoints travel through the Hub:
**repo** = the cell (`ZHZisZZ/qwen3_5-4b.sft.<config-stem>.<dataset>.<teacher>`),
**subdir** = `epoch_1/`, `epoch_2/`, **tag** = the producing commit.

The tag is provenance, not the entry point: **eval pulls `main`**, so a re-run replaces the epoch
dirs and the next campaign picks them up with no sha to carry between hosts. Use `--revision <tag>`
only to re-run an OLD campaign; a re-run at the same commit *moves* the tag. Epoch dirs are named by
rank at upload time — training writes `iter_<N>` with slime's 0-based index, a number to read off
disk, never predict.

```bash
# --- TRAIN HOST ---  (run from the repo root; commit FIRST -- `git rev-parse` reports a sha
# for a dirty tree just as happily, so an uncommitted edit tags five public repos with a sha
# that does not describe the weights, silently.)
# `uv run hf`, NOT bare `hf`: `hf repos` needs huggingface_hub >= 1.17 and the system hf may be
# older. Needs a WRITE-scoped token (`uv run hf auth login`, or HF_TOKEN). Repos are public,
# so the eval host needs no auth at all.
COMMIT="$(git rev-parse --short HEAD)"   # commit message + tag; eval still pulls main
DS=mobilegym_r30
DS_SR=mobilegym_sr
EPOCHS=2                                 # must match NUM_EPOCH in the Train block
CKPTS=.ckpts/qwen3_5-4b
# The five cells as "<config stem> <teacher> <dataset recipe>".
cells() {
  for T in gpt5_5; do
    for P in mobile.use.i1 mobile.use.i4; do echo "$P $T $DS"; done
  done
  for T in gpt5_5; do
    for P in mobile.use.i1.reasoning mobile.use.i4.reasoning; do echo "$P $T $DS"; done
  done
  echo "mobile.use.i1.reasoning gpt5_5 $DS_SR"
}

while read -r P T D; do
  REPO="ZHZisZZ/qwen3_5-4b.sft.$P.$D.$T"
  # `ls` first, and gate on the count here: creating the repo up front would leave an empty
  # public one behind for a cell that never trained, and a half-trained cell must not reach
  # the eval host, which pulls whatever is on main.
  DIRS=$(ls -d "$CKPTS/sft.$P.$D.$T"/iter_* 2>/dev/null | sort -V)
  if [ "$(echo "$DIRS" | grep -c .)" -ne "$EPOCHS" ]; then
    echo "SKIP $REPO: $(echo "$DIRS" | grep -c .) iter dir(s), expected $EPOCHS -- not uploading"
    continue
  fi
  # NOTE: this runs before the per-file check below, so a cell whose FIRST epoch dir is
  # incomplete still leaves an empty public repo. The count gate above is what prevents that
  # for a cell that never trained at all.
  uv run hf repos create "$REPO" --repo-type model --exist-ok < /dev/null

  ep=0
  for D_IN in $DIRS; do
    ep=$((ep + 1))
    # eval loads the tokenizer/processor from --model-path, not --model-id, so the epoch dir
    # must carry slime's FULL HF export. Check before spending 8 GB of upload on it:
    # a weights-only dir only fails on the eval host, after upload AND download.
    for f in config.json tokenizer_config.json preprocessor_config.json; do
      [ -e "$D_IN/$f" ] && continue
      echo "SKIP $REPO: $D_IN has no $f -- AutoProcessor would fail at eval; stopping this cell"
      ep=-1; break
    done
    [ "$ep" = "-1" ] && break
    uv run hf upload "$REPO" "$D_IN" "epoch_$ep" \
      --repo-type model --commit-message "$COMMIT: epoch $ep (from $(basename "$D_IN"))" \
      < /dev/null \
      || { echo "SKIP $REPO: upload of $D_IN failed -- not tagging"; ep=-1; break; }
  done
  # Both breaks above escape the `for D_IN` loop ONLY; without this, a half-uploaded cell falls
  # through and gets tagged as if complete. Pushed epoch dirs stay on main, untagged.
  [ "$ep" = "-1" ] && continue

  # Tag the commit that produced these weights. Eval pulls `main` and never needs this; it is
  # here so an old campaign stays reproducible -- `--revision <tag>` on the download.
  # create-or-move; the Hub has no move, so it is delete-then-create.
  uv run hf repos tag delete "$REPO" "$COMMIT" --repo-type model --yes < /dev/null \
    || echo "note: no existing '$COMMIT' tag on $REPO (expected on a first upload)"
  uv run hf repos tag create "$REPO" "$COMMIT" --repo-type model -m "$COMMIT" < /dev/null \
    || echo "ERROR $REPO has NO '$COMMIT' tag now; weights are on main -- re-tag by hand"
done <<< "$(cells)"
```

`hf upload` is single-commit and not resumable; `upload-large-folder` is, but takes no
path-in-repo and so cannot express `epoch_<k>/`.

#### Eval

Nine runs on the **full 256-task `mobilegym` eval split** — no filter. `mobilegym` registers no
`exclude_reason`, and this campaign deliberately does NOT take the L1+L2 subset that
[docs/grpo.md#mobilegym](/docs/grpo.md#mobilegym) uses for its 2B smoke runs, so the denominator here
is 256 and not 93. Eval tasks carry registered deterministic seeds (`seed=42`), so the task
instances are fixed across runs. Env setup:
[`lite/gym/envs/mobilegym/README.md`](/lite/gym/envs/mobilegym/README.md).

Nine, not five: **the base model runs once per config stem**, all four of them. A checkpoint
must be scored against a baseline that saw the same prompt surface, and `base` under an Action-only
config runs with thinking OFF — the wrong baseline for a reasoning checkpoint. The ablation cell
shares `mobile.use.i1.reasoning` with its parent, so it needs no base run of its own.

```bash
# --- EVAL HOST ---
# RUN names this campaign's artifacts. It is NOT a code version -- this pulls each repo's main,
# whatever the train host last pushed. To score an OLD campaign instead, add `--revision <tag>`
# to the download below; the tags are there. Any label works; a date is easiest.
# It only has to CHANGE between campaigns: rollout resumes a log-root sample by sample and
# never checks what produced it, so a slug that did not rotate silently re-reports the
# previous campaign's numbers -- including the base runs every cell is read against.
# No default on purpose: a `date`-derived one does not rotate on a same-day re-run, and
# ${RUN:-...} in a reused shell keeps the OLD value -- both silently blend two campaigns'
# samples into one summary.json. Under `bash script.sh` the line below exits 1; PASTED into an
# interactive shell it only prints and the rest runs on with RUN empty. Read the error and stop
# by hand.
: "${RUN:?set RUN to a fresh slug for THIS campaign, e.g. 20260920a}"

# MUST be unset. `--sglang-server-url` defaults to $SGLANG_SERVER_URL (lite/infer/cli.py), and
# with a URL in hand serving.py never starts a server -- `--model-path` then only picks the
# tokenizer/processor, so all fifteen runs GENERATE from whatever model that server holds and
# every summary.json still looks normal. Nothing warns.
unset SGLANG_SERVER_URL

# The env-server is a scoring CONDITION, not an implementation detail. mobilegym works in direct
# mode too, but there each rollout.py process owns its own `mobilegym-<port>` container, so N
# parallel runs mean N containers instead of one shared browser pool -- different lifecycle,
# different timing. The size of that effect is NOT MEASURED here. Hold the mode fixed for the
# whole campaign: every number in the Results table must come from the same one.
# It must be started FOR mobilegym: the shared lab server does not serve this env, and that is
# the same 404 the export blocks guard against.
#   uv run python scripts/serve_env.py --port 30107 --env-ids mobilegym --token mobile-eval
export CUA_LITE_ENV_SERVER_URL="http://$(hostname -I | awk '{print $1}'):30107"
export CUA_LITE_ENV_SERVER_TOKEN=mobile-eval   # must MATCH the server's --token above: passing
                                               # --token puts serve_env.py in STRICT auth, where a
                                               # mismatched bearer is rejected. Drop --token on the
                                               # server for passthrough, where any value just scopes.
EPOCH=epoch_2                            # epoch_1 = after 1 epoch
DS=mobilegym_r30
DS_SR=mobilegym_sr
CFG=devs/exps/train/mobile/configs/qwen3_5
PULL=.ckpts/pulled
LOGS=.logs/rollout/Qwen_Qwen3.5-4B/mobilegym
# The five cells as "<config stem> <teacher> <dataset recipe>".
cells() {
  for T in gpt5_5; do
    for P in mobile.use.i1 mobile.use.i4; do echo "$P $T $DS"; done
  done
  for T in gpt5_5; do
    for P in mobile.use.i1.reasoning mobile.use.i4.reasoning; do echo "$P $T $DS"; done
  done
  echo "mobile.use.i1.reasoning gpt5_5 $DS_SR"
}

MISSING=
while read -r P T D; do
  uv run hf download "ZHZisZZ/qwen3_5-4b.sft.$P.$D.$T" \
    --include "$EPOCH/*" \
    --local-dir "$PULL/sft.$P.$D.$T@$RUN" < /dev/null
  # snapshot_download has no empty-match guard: a wrong $EPOCH yields an empty dir, silently.
  # All three, not just config.json: a dir with config.json but no processor files passes a
  # one-file check, then dies inside a backgrounded score job at AutoProcessor.from_pretrained.
  for f in config.json tokenizer_config.json preprocessor_config.json; do
    [ -e "$PULL/sft.$P.$D.$T@$RUN/$EPOCH/$f" ] || MISSING="$MISSING $P.$D.$T:$f"
  done
done <<< "$(cells)"
# A missing checkpoint must not reach the score loop: it would take a GPU and fail as a crashed
# background job long after `wait`.
if [ -n "$MISSING" ]; then
  echo "STOP -- do not run the score loop; missing under $EPOCH:$MISSING"
  return 1 2>/dev/null || exit 1
fi
echo "checkpoints OK"

# $1 = config stem, $2 = --model-path ("" = base model), $3 = log slug. --config-path always
# follows $1, so a checkpoint is only ever scored on the surface it was trained on.
# CONCURRENCY IS A BUDGET ON THE ENV-SERVER, not a per-run knob: every run in flight adds
# `--concurrency` browser contexts to the SAME shared mobilegym container. Pick the host total
# first, then divide. 128 total is this campaign's number and it is not the container's ceiling
# -- the in-container pool is max_browsers 128 x contexts_per_browser 8 = 1024 contexts
# (lite/gym/envs/mobilegym/configs/default.yaml), so the binding limit is host CPU (96 vCPU
# here), not the pool. 256 is plausible on this host and is UNMEASURED; do not raise it mid-
# campaign, because a concurrency change is a timing change and these episodes have waits in
# them.
NGPU=8   # cards this host will use
CONC=16  # 8 x 16 = 128 contexts
gpu=0
score() {
  # Drain the batch before wrapping back to card 0, or two rollouts land on one GPU and
  # both OOM. `wait` blocks on every score backgrounded so far, which is the batch.
  if [ "$gpu" -ge "$NGPU" ]; then wait; gpu=0; fi
  # --env-kwargs, not the yaml: the profiles leave reward_shaping unset so the env default
  # (false) would apply, and a bare SR cannot resolve the moves this campaign measures. With it
  # on, `episode_return == 1.0` is still exactly SR success, so BOTH numbers come out of this
  # one pass -- see #reward-shaping.
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.5-4B ${2:+--model-path "$2"} \
    --env-id mobilegym --splits eval --concurrency "$CONC" \
    --env-kwargs '{"reward_shaping": true}' \
    --config-path "$CFG/$1.yaml" \
    --log-root "$LOGS/$3" < /dev/null &
  gpu=$((gpu + 1))
}

# 4 BASE runs -- empty $2 drops --model-path, so rollout serves --model-id's own weights.
# One per CONFIG STEM, including the two .reasoning ones: `base` under an Action-only config
# runs with thinking off and is the wrong baseline for a reasoning checkpoint.
for P in mobile.use.i1 mobile.use.i4 mobile.use.i1.reasoning mobile.use.i4.reasoning; do
  score "$P" "" "base.$P@$RUN"
done

# 11 CHECKPOINT runs -- $2 is the pulled epoch dir, so that is what gets served.
while read -r P T D; do
  score "$P" "$PULL/sft.$P.$D.$T@$RUN/$EPOCH" "sft.$P.$D.$T@$RUN.$EPOCH"
done <<< "$(cells)"

wait
```

`--model-id` picks the adapter and action space; the weights, tokenizer and chat template all come
from `--model-path`.

These runs report BOTH numbers from one pass: the shaped mean, and the Success Rate as the count of
`episode_return == 1.0`. `scripts/rollout.py` has no task parquet in the loop, so unlike the RL block
there is no per-row channel here — `--env-kwargs` on the command line is the only way to set the
flag, and it is the whole reason the flag is on the command line rather than in the profile yaml
(the yaml is shared with training, where the same value happens to be wanted, but for a different
reason).

Fifteen runs over `NGPU` cards, in batches: `score` drains with `wait` before reusing card 0 —
without it `$gpu` keeps counting past the last card onto ordinals that do not exist. Set `NGPU` to
what the host has FREE; `NGPU=1` serializes all fifteen, slow but correct. Both drains are bare
`wait`s, so anything else left backgrounded in this shell delays them.

Score each run from `<log-root>/summary.json` -> `stats.mean_episode_return` (denominator
`num_valid`). Which cells compare to which is settled at the top of this file.

#### Record the scores

Collect all fifteen — sixteen with the RL run — and commit as
`devs/exps/train/mobile/logs/$RUN.md`: one file per campaign, edited as runs land, not a wrap-up
from memory.

```bash
# --- EVAL HOST, same shell as the Eval block ---
# Reuses $RUN, $DS, $DS_SR, $EPOCH, $LOGS and cells(). In a fresh shell set them all again --
# $RUN needs an ASSIGNMENT (`RUN=<the same slug>`): the Eval block's line is `: "${RUN:?...}"`,
# an assertion, so re-pasting THAT leaves RUN empty and `show` prints MISSING for every slug.
# The n it prints is a result too, not just bookkeeping -- see "Reading the table" in Results.
show() {  # $1 = log slug
  uv run python -c "
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'summary.json'
if not p.exists(): print(f'{sys.argv[2]:78s} MISSING'); raise SystemExit
d = json.loads(p.read_text()); s = d['stats']
# summary.json has no solved count: derive it. Under shaping 1.0 is still reached ONLY by the
# SR criterion (the shaped branch caps at 0.5), so 'solved' IS the Success Rate numerator and
# 'partial' is the extra resolution shaping buys. partial == 0 over a whole pass means the flag
# did not reach the rollout -- that pass measured the bare SR; find it before recording.
solved = sum(r == 1.0 for t in d['tasks'] for r in t['episode_returns'])
partial = sum(0.0 < r < 1.0 for t in d['tasks'] for r in t['episode_returns'])
pf = (s.get('stop_reasons') or {}).get('parse_failure', 0)
print(f\"{sys.argv[2]:70s} {s['mean_episode_return']:.4f} ({solved}/{s['num_valid']})\"
      f\"  err={s['num_samples'] - s['num_valid']} parse_fail={pf} partial={partial}\")
" "$LOGS/$1" "$1"
}
for P in mobile.use.i1 mobile.use.i4 mobile.use.i1.reasoning mobile.use.i4.reasoning; do
  show "base.$P@$RUN"
done
while read -r P T D; do show "sft.$P.$D.$T@$RUN.$EPOCH"; done <<< "$(cells)"
# The RL run is NOT scored here. It uses its own eval manifests and its own in-training
# eval points (see the RL section); mixing it into this table would put two different
# protocols in one column.
```

Paste the numbers into a snapshot file, reusing the table LAYOUT from **Results** below (the cell
format — `mean (solved/num_valid)` — is defined there). Front matter:

```markdown
# mobile.use @ <run>

- **Checkpoints**: each cell repo's `main` as of `<date>`, tag `<sha>`. Ship can SKIP a cell and
  re-run it later, so the five repos need not share a sha — if they diverge, list the odd ones
  out here (`uv run hf repos tag list <repo>`), or a later `--revision` re-run pulls the wrong
  weights.
- **Dataset**: `mobilegym_r30` (`episode_return >= 0.30`), the teacher's whole filtered pool — record
  the row counts the Export block printed, they are part of the result
- **Ablation**: `mobilegym_sr` (`episode_return > 0.5`), same cell, record its row count too
- **Eval**: `mobilegym` eval, full 256 tasks, no filter, `reward_shaping` ON, `epoch_2` —
  record the shaped mean and the `solved/256` SR from the same pass
- **Host / GPUs**: `<host>` / `0-7`
- **Env mode / concurrency**: env-server / `<total contexts>`
- **Last updated**: `<date>`

## Results

<the table from the Results section of this file>
```

### Results

**Measured 2026-09-22, campaign `20260922`.** Every cell of the grid carries three full 256-task
passes. No desktop or browser number transfers into this table — different env, different teacher
pool, different reward definition — and none is used as a substitute for a missing measurement here.

Two numbers per cell, both from the SAME shaped pass: **shaped mean** first, then the **Success
Rate** as `(solved / num_valid)` — the count of `episode_return == 1.0`, which under shaping is still
exactly MobileGym's SR criterion. The shaped mean is the sensitive one and the SR is the comparable
one; report both, and say which you are comparing when you cite a number elsewhere.

**Three passes per cell**, as in the desktop and browser campaigns. Campaign `20260922`: every cell
scored three times, interleaved across `cc9c`/`cc9d`/`cc9e` so host drift spreads across cells
instead of landing on one. `±` is **half the range** of the three; with n=3 a range is what there is,
and it is not a confidence interval. An earlier version of this table carried one pass per cell and
said to "read only large moves"; those numbers are superseded by the ones below, and the reason is in
[Three passes is a floor, not a measurement](#three-passes-is-a-floor-not-a-measurement).

Cells read `shaped-mean ±half-range (solved/256) n/3`.

| | `i1` | `i4` |
|---|---:|---:|
| **base** | 0.1954 ±0.0086 (37.3/256) 3/3 | 0.2023 ±0.0051 (39.0/256) 3/3 |
| **base + `<think>`** | 0.1862 ±0.0260 (35.7/256) 3/3 | 0.2085 ±0.0152 (41.3/256) 3/3 |
| **`gpt5_5`** | 0.2727 ±0.0140 (54.7/256) 3/3 | 0.3023 ±0.0033 (61.7/256) 3/3 |
| **`gpt5_5` + `<think>`** | 0.3409 ±0.0157 (68.7/256) 3/3 | **0.3448** ±0.0056 (69.7/256) 3/3 |
| **`gpt5_5` + `<think>`, `> 0.5` rows** | not run | — |

Every individual run — `shaped-mean (solved) parse_failure`, one column per pass, pod in italics:

| | | p1 | p2 | p3 |
|---|---|---|---|---|
| **base** | `i1` | 0.2050 (40) 1 *cc9c* | 0.1879 (35) 0 *cc9c* | 0.1933 (37) 0 *cc9c* |
| | `i4` | 0.1988 (39) 0 *cc9c* | 0.1992 (38) 0 *cc9d* | 0.2090 (40) 0 *cc9c* |
| **base + `<think>`** | `i1` | 0.1651 (31) 0 *cc9c* | 0.2170 (44) 0 *cc9c* | 0.1766 (32) 0 *cc9c* |
| | `i4` | 0.2218 (46) 0 *cc9e* | 0.2122 (42) 0 *cc9e* | 0.1914 (36) 2 *cc9e* |
| **`gpt5_5`** | `i1` | 0.2892 (61) 0 *cc9e* | 0.2678 (54) 0 *cc9e* | 0.2612 (49) 1 *cc9d* |
| | `i4` | 0.3019 (62) 1 *cc9e* | 0.2993 (61) 1 *cc9e* | 0.3058 (62) 1 *cc9e* |
| **`gpt5_5` + `<think>`** | `i1` | 0.3453 (71) 0 *cc9d* | 0.3230 (63) 0 *cc9d* | 0.3543 (72) 0 *cc9e* |
| | `i4` | 0.3383 (67) 1 *cc9e* | 0.3494 (71) 0 *cc9e* | 0.3467 (71) 0 *cc9e* |

Every pass in both tables is a full 256-task pass with `err=0` and `partial>0`. Passes are assigned
to a cell by run slug in time order and the FIRST THREE are the ones aggregated — a rule fixed before
the numbers were read, because some cells have four or five passes (see the section below) and
picking which three after the fact would be picking the answer.

The `> 0.5` filter ablation was dropped as uninformative on this corpus: the gate removes only 7% of
the trajectories (1133 -> 1049) and 3% of the templates (158 -> 153), because MobileGym scores `1.0`
iff success and `0.5 x progress` otherwise, so `> 0.5` IS the success set and `>= 0.30` adds back
only the partial-progress runs above `progress 0.6` — of which this teacher produces few.

#### Three passes is a floor, not a measurement

Seven passes beyond the three aggregated above were run on five of the cells. **Two of them fall
outside their own cell's three-pass range**, and one of those is the cell the table calls tightest:

| cell | aggregated three | extra passes | verdict |
|---|---|---|---|
| `gpt5_5` / `i4` | 0.2993 .. 0.3058 (±0.0033) | **0.2862** | outside, by 1.3pp |
| `base` / `i4` | 0.1988 .. 0.2090 (±0.0051) | **0.2127** | outside, by 0.4pp |
| `base` + `<think>` / `i1` | 0.1651 .. 0.2170 (±0.0260) | 0.1825, 0.1901 | inside |
| `base` + `<think>` / `i4` | 0.1914 .. 0.2218 (±0.0152) | 0.1983 | inside |
| `gpt5_5` + `<think>` / `i1` | 0.3230 .. 0.3543 (±0.0157) | 0.3432 | inside |

So a small `±` can be three draws landing close together rather than a tight cell: `gpt5_5` / `i4`
reads ±0.0033 over its three and ±0.0098 over four. **Compare against the larger of the two cells'
own `±`, and treat any `±` under ~0.010 as un-measured rather than tight.** The moves this table
supports (+13.6pp for SFT, +6.8pp for `<think>` after SFT) are far outside that; the ones it does not
(`i4` vs `i1` at +3.0pp on the `gpt5_5` row, which is +2.6pp over four passes) are the ones this
caveat bites.

Reading the table:

- **Compare against the larger of the two cells' own `±`.** Across the eight cells the half-range
  spans ±0.0033 to ±0.0260 — an 8x spread — so one global threshold is either too strict or too
  loose. The widest cell is `base` + `<think>` / `i1` (±0.0260, wider than desktop's worst at
  ±0.0207); the tightest reading is not trustworthy (see above). The SR half also carries a binomial
  floor of about 2.8pp at `n=256, p~0.3` all by itself.
- **`mean_episode_return` averages over `num_valid`, not 256.** The `err=` that `show` prints is
  `num_samples - num_valid`: non-zero means that pass came up short and must be RE-RUN, not averaged
  in. Record the mean WITH its denominator.
- **`partial=` is expected to be NON-zero, and a zero is the alarm.** With shaping on the env returns
  1.0, 0.0, or a value in `(0, 0.5]`. `partial=0` across a whole 256-task pass means the shaping flag
  did NOT reach that rollout — the run silently measured the bare SR and its shaped mean is not
  comparable to the other cells. Check `--env-kwargs` before recording it. (This bullet said the
  opposite while eval ran unshaped; the inversion is the point.)
- **`parse_failure` does not move the denominator** — a malformed final turn is scored by the env
  like any other episode. Record it per run anyway; on desktop it rose with SFT and again with
  `<think>`, and whether it does so here is one of the things this table answers.
- **Only within a column.** `i1` and `i4` are different prompt surfaces; the base row is what makes a
  column readable, and each `.reasoning` column has its own base row for that reason.
- **The `> 0.5` row answers one question only** — see [Filter ablation](#filter-ablation) — and it is
  read against the `gpt5_5` + `<think>` / `i1` cell directly above it, never against another teacher.

#### The `i4` + `<think>` cell: six hangs on one host, then clean on another

`mobile.use.i4.reasoning` failed six times on 2026-09-21, every time the same way, and the cell stood
at `not run`. **It then trained to completion on 2026-09-22 on a different pod, at the same
`TP_SIZE=2` the failures used.** Read off that run's own log rather than its launcher:

    tensor_model_parallel_size = 2      <- slime's parsed table, not the launcher's echo
    data_parallel_size         = 4      <- the DP group the hang always named
    Watchdog caught collective   0 hits
    iter_30, iter_61 both written;  lm loss 1.06 -> 0.358;  SFT EXIT rc=0

The one traceback in that log is a wandb `atexit` `BrokenPipeError` raised AFTER the ray job reported
success — not a training event. The weights are on the Hub and the cell has a number in the table
above.

This matters because the same cell has also been written up as *requiring* `TP=8`. It does not: the
run above is `TP=2` at `DP=4`, the exact configuration that conclusion excludes, on a host that had
not run it before. `TP=8` does avoid the hang, but so does changing machines, which means the
parallelism was never shown to be the cause.

So the hang is **host-specific, not configuration-specific**. That is what the exclusion table below
could not settle: it ruled out every property of the data and the parallelism it could reach, but
"Environment drift" was excluded only by re-running `mobile.use.i4` on the SAME host. A second host
was the one control not available at the time. Keep the analysis below — the failure is real, it will
recur, and its signature is precise — but read it as a diagnosis of one machine.

**Signature.** Training reaches iteration 12 (iteration 5 at `DP=2`), completes that step's last
micro-batch, and then hangs. NCCL's watchdog aborts ~10 min later:

```
PG GUID 2 (DATA_PARALLEL_GROUP_WITH_CP)   x7
PG GUID 25 (TENSOR_MODEL_PARALLEL_GROUP)  x1
WorkNCCL(SeqNum=5, OpType=ALLREDUCE, NumelIn=1, Timeout(ms)=600000) ran for 600051 ms
```

The stalled collective carries ONE element, so this is a rank that never arrives, not a step that
runs long. `py-spy` on all eight ranks at the moment of the hang shows the split:

```
6 ranks   get_grad_norm_fp32 (core/optimizer/clip_grads.py:133)  <- already in the optimizer step
2 ranks   custom_backward (core/pipeline_parallel/schedules.py:190)  <- still in backward
```

**What was ruled out.** Each of these was tested, not reasoned about:

| Hypothesis | How it was excluded |
|---|---|
| Sequence length | Longest final-step prompt is 6191 tokens (`i4`: 4943). Not long. |
| Packing degraded by `<think>` | Prefix-compatibility is 677/835 = 81.1% for BOTH `i4` and `i4.reasoning`, and `image_indices` sawtooth identically. The exports are structurally the same. |
| Uneven micro-batch counts across ranks | `i1` varies 62->99 per iteration and trains fine; the static path in `dp_schedule.py` raises rather than emitting a ragged count, and no such assert fired. |
| A bad GPU or a bad DP group | Reproduces on `TP=2/DP=4` (GPUs 0-7) and on `TP=2/DP=2` (GPUs 0-3). |
| CPU/IO contention with a concurrent eval | Persists after the eval is killed. |
| Environment drift | `mobile.use.i4` — same launcher, same container, same 8 GPUs — was re-run THREE times, interleaved with the failures: 62/62 at 07:03, to iteration 17 at 16:36 (between failures 3 and 4), and to iteration 15 at 20:23 (after failure 7). Every one at 5.1-6.0 s/micro-batch with zero NCCL events. |
| A poisoned trajectory | Shuffling the row order (seed 20260921, same 1016 rows) moves the failure from iteration 12 to iteration 15, and the two failing steps share ZERO rows. No single trajectory is implicated; the trigger is a property some 32-row steps have. |
| Iteration index or accumulated state | Same shuffle: the failure moves with the data, not with the step number. |
| `torch.compile` shape specialization | `TORCHDYNAMO_DISABLE=1` (via the existing `CUA_LITE_RAY_ENV_VARS` passthrough) hangs at the same iteration. It is also not a speed knob here: eager runs 5.5 s/micro-batch against 5.1-6.0 compiled. |

**What is still open.** Nothing testable is left that this campaign could reach. The DP-imbalance
lead was simulated offline against the real per-segment lengths and does not hold: strided
round-robin over `dp_size=4` gives per-rank token loads within `max/min = 1.030`, and the figure is
the same for `i4` and `i4.reasoning`, so `--balance-data` would not change anything. The failing
steps also look ordinary on every axis measured (segment count, total tokens, longest segment,
length CV) — each sits inside the range spanned by its neighbours.

What survives is narrow but real: the trigger is a property that a 32-row training step can have,
it is not carried by any individual trajectory, and it appears only when the 4-image budget and
`<think>` are combined — `i1` + `<think>` and `i4` without `<think>` each train to 62/62.

**What the cell would have answered.** Both single-axis gains are real and orthogonal on `i1`/`i4`:
`<think>` is worth +6.7pp shaped (0.3444 vs 0.2776) and the 4-image budget +4.2pp (0.3195 vs
0.2776), with the same ordering on the untrained base row (0.2162 vs 0.1896). Whether they compose
is exactly what this cell measures, so it is worth another attempt.

#### Filter ablation

The fifth cell is `(mobile.use.i1.reasoning, gpt5_5)` — the GRPO cell's own profile and teacher —
exported a second time under `episode_return > 0.5` instead of the campaign's `>= 0.30`. The `>= 0.30`
pool keeps trajectories that passed at least 60% of their `check_goals` but did not finish cleanly
or did not terminate; the `> 0.5` pool keeps only true successes.

The question is whether those partial-credit trajectories buy GRPO headroom or just teach the student
to stop short. Two numbers decide it, and neither exists yet: the SFT score of the two cells against
each other, and — if it is worth the second RL run — the GRPO curve started from each. The cheap
half (SFT) is what this campaign runs; a second GRPO run is not budgeted here.

Both cells train on the same teacher and the same profile, so this is the one contrast in the table
where the only moving part is which rows were kept. The row counts of the two pools are **not
measured yet** and belong in the Results front matter: the ablation is only interpretable next to
how much data each filter left.

### RL

GRPO from the local **`gpt5_5` + `<think>` SFT checkpoint** with
[`mobile.use.i1.reasoning.yaml`](/devs/exps/train/mobile/configs/qwen3_5/mobile.use.i1.reasoning.yaml).
Read [docs/grpo.md](/docs/grpo.md) first: env-server prerequisite, sync-vs-async, and the knobs this
block does not repeat.

**The cell is `fam37n`: same-family transfer, fresh instance per rollout, sampled eval.** Three
choices define it, and each one was forced by a measurement rather than picked:

| choice | value | why |
|---|---|---|
| task pool | 37 train / 52 eval, **same families**, `L1-L3` | `train160` and `eval256` share no class name; the only earlier variant that moved was the one where the two sides shared templates |
| train instance | **fresh per rollout** (rows tagged `split="train"`) | the engine then injects `f(rollout_id, group_index)`; pinning one instance instead is the `fam37` control, and it is flat |
| eval decoding | **`temperature=1.0`, 4 samples per task** | greedy scores a policy GRPO never optimised — see [Greedy hides the gain](#greedy-hides-the-gain) |

**Building the pool.** Take `mobilegym`'s EVAL split, keep `L1-L3`, group by family (`app` + the
first CamelCase token, so `wechat.PostSomething` -> `wechat.Post`), keep the 37 families with two or
more members, and put ONE member in train and its siblings in eval: 37 train / 52 eval, zero task
overlap, every eval family covered by a train family. `L4` is excluded — it is 31.2% of the eval
split and 38.6% of the step budget for a 4% solve rate, so it costs denominator, not gradient.

> **The row's `split` tag is what picks the instance policy, and it is not mobilegym's split.**
> `engine.py:315` reads `split` from the PARQUET ROW. A row tagged `"train"` takes the injection
> branch at `engine.py:341` and gets a group-shared seed derived from `(rollout_id, group_index)` —
> a fresh task instance every rollout, shared across the 8 members of one group. A row tagged
> `"eval"` skips that branch and falls back to the registry, which carries `seed=42` for eval-split
> tasks and NOTHING for train-split tasks (`tasks.json`: `('eval',42) x256`, `('train',None) x160`)
> — and a missing seed means `random.Random(None)`, i.e. a fresh instance every EPISODE, which
> breaks the group baseline. So: eval manifests always `"eval"`; a train manifest is `"train"` for
> fresh-per-rollout, or `"eval"` **plus an explicit `seed` in `env_kwargs`** to pin one instance.
> Assembling a manifest by copying rows out of a mixed-provenance source inherits a MIXED policy
> silently — the tag column reads uniform while the behaviour is not. Print the resolved policy per
> row (`env_kwargs["seed"]` -> row tag -> `tasks.json` seed) before launching.

**What "fresh instance per rollout" actually resolves to**, measured on the pod rather than read off
the branch — the three conditions at `engine.py:340-344` all have to hold, and the third one fails
OPEN into the control arm:

    env_supports_kwarg(mobilegym, seed) = True        <- gate (c); False here would silently make
                                                         fam37n identical to fam37, with no error
    injected seed = Random(f"{rollout_id}:{group_index}").randint(0, 2**31-1)
    group 0 over 30 rollouts -> 30/30 DISTINCT instances   (fam37 pins ONE for all 30)

**Seed replicates change the policy, not the tasks.** That derivation reads `rollout_id` and
`group_index` and nothing else — `--rollout-seed` and `--seed` do not enter it. So five pods running
this cell with five seeds see the **same instance sequence** and differ only in initialisation and
sampling. That is a paired design, not a broken one: instance noise is common across arms, so a
between-seed spread is a clean read on optimisation variance — but it does NOT sample the
instance-generator, and no number of seeds will turn it into that.

| pod | `RS` (`--rollout-seed`) | `SD` (`--seed`) |
|---|---|---|
| a | 101 | 5001 |
| b | 202 | 5002 |
| c | 303 | 5003 |
| d | 404 | 5004 |
| e | 505 | 5005 |

#### Five seeds — the record

**Launched 2026-09-22 ~09:25 UTC on five pods; NOTHING below is measured yet.** The dicts are empty
on purpose — an empty dict cannot be misread as a result, a zero can. At the measured 26.5 min per
rollout (`iter_4` -> `iter_9` -> `iter_14` on the first `fam37n` arm) 30 rollouts is ~13.5 h, inside
every pod's remaining lifetime with ~23 h to spare.

Scores are RECORDED AS REWARD, not as solved counts. MobileGym's reward is shaped — `1.0` iff
success, `0.5 x progress` otherwise — so a solved count throws away the partial-progress half of
every episode, which is most of what moves early in a run. `episode_return == 1.0` still recovers
the Success Rate from the same data if it is ever wanted.

```python
EVAL = {  # "rs/sd" -> {rollout: shaped_mean}   fam37ne, 52 tasks x 4 samples at t=1 (n=208)
  "101/5001": {0: 0.3635},
  "202/5002": {0: 0.3134},
  "303/5003": {0: 0.3499},
  "404/5004": {0: 0.3506},
  "505/5005": {0: 0.3537},
}
TRAIN = {}  # "rs/sd" -> [rollout/raw_reward], index = rollout   -- no rollout has completed yet
```

##### The noise floor, measured directly

Rollout 0 is the same checkpoint on all five arms, scored on the same manifest with the same pinned
instances. Nothing differs but the eval sampling, so the spread across those five IS the measurement
noise — not an estimate of it:

    mean 0.3462   sd 1.91pp   range 5.01pp
    per-arm threshold   2*sd        = 3.83pp
    5-arm-mean threshold 2*sd/sqrt(5) = 1.71pp

**Three passes understated this 5.5x.** The same protocol measured by `rollout.py` over three passes
gave `0.3686 / 0.3737 / 0.3671` — sd 0.35pp — and on that basis a 2pp move looks decisive. It is
not. Three draws of a distribution this wide land close together often enough that a tight triple is
evidence of luck, not of precision; use 3.83pp for a single arm and re-read any earlier conclusion
that leaned on the narrower figure. This also supersedes the sd carried over from the browser
campaign's noise floor: that number was measured on a different benchmark and a different decoder.

Two cautions on the number itself. `202/5002` sits 3.28pp below the mean and one outlier moves `sd`
a lot at `n=5`, so the floor is known to about a factor of 1.5 at best. And the five in-training
evals mean `0.3462` while `rollout.py` on the same checkpoint and manifest reads `0.3698` — a 2.36pp
offset between two eval paths that are supposed to agree. Until that is explained, compare a reading
only against a baseline measured the SAME way; do not pool the two.

All five arms run the identical cell — same `fam37n`/`fam37ne` manifests (file sha256 `e239ea71…` /
`4036901…` on every pod, asserted at launch), same `lr=1e-6`, same `EVAL_TEMPERATURE=1.0` with 4
samples, 30 rollouts — and differ ONLY in `--rollout-seed` / `--seed`. Per the paragraph above they
also share the instance sequence, so the spread across these five is optimisation variance and
nothing else.

**One env, two manifests.** Training and eval are both `mobilegym`, so unlike the desktop RL run
`ENV_ID` is unambiguous and the env-server needs `--env-ids mobilegym` only. What differs between the
two sides is the reward, and that lives in the manifests.

<details>
<summary>Data</summary>

```bash
# --- ONE-TIME DATA BUILD; skip generation for any file that already exists ---
# These parquet files are fixed experiment manifests under the repo, so Slime sees them at
# /workspaces/cua-lite/devs/exps/train/mobile/data.
#
# reward_shaping is pinned PER MANIFEST because slime gets ONE CONFIG_PATH for both sides:
# export_tasks.py:81 writes --env-kwargs into every row's metadata.env_kwargs, and
# engine.py:310 deep-merges args (from CONFIG_PATH) with the row, THE ROW WINNING PER LEAF.
#   train -> true : GRPO needs the shaped signal; MobileGym's SR alone is too sparse to
#                   bootstrap from. mobile.use.i1.reasoning.yaml does NOT set the flag, so
#                   without this line training would run on the binary SR.
#   eval  -> true : shaping is a SUPERSET (see #reward-shaping). `episode_return == 1.0` is
#                   still exactly SR success, so the Success Rate is recoverable by counting
#                   them; the mean additionally resolves partial progress, which the binomial
#                   stderr of the bare SR (2.8pp at n=256) cannot. Leaving it to the config
#                   would default to false and silently cost that resolution.
# No --filter on either side: the full 160 / 256 splits, no L1+L2 subset.
DATA=devs/exps/train/mobile/data
TRAIN="$DATA/mobilegym.train160.shaped.parquet"
EVAL="$DATA/mobilegym.eval256.shaped.parquet"
CFG=devs/exps/train/mobile/configs/qwen3_5/mobile.use.i1.reasoning.yaml
mkdir -p "$DATA"

if [ -e "$TRAIN" ]; then
  echo "keep existing fixed train manifest: $TRAIN"
else
  env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
    uv run python -m lite.train.export.export_tasks --env-id mobilegym --split train \
      --env-kwargs '{"reward_shaping": true}' \
      -o "$TRAIN"
fi

if [ -e "$EVAL" ]; then
  echo "keep existing fixed eval manifest: $EVAL"
else
  env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
    uv run python -m lite.train.export.export_tasks --env-id mobilegym --split eval \
      --env-kwargs '{"reward_shaping": true}' \
      -o "$EVAL"
fi
```

Then assert the merge, not the file. The parquet holding the right key is necessary and not
sufficient: what reaches `gym.make` is the deep-merge of the config's `env_kwargs` with the row's,
and this reproduces that merge exactly (same two functions the engine calls) for EVERY row of both
manifests.

```bash
uv run python - "$CFG" "$TRAIN" true "$EVAL" true <<'PY'
import sys

import pandas as pd
import yaml

from lite.data.staging import coerce_meta
from lite.gym import finalize_env_kwargs
from lite.utils.config import deep_merge

# args side: exactly what slime sets on args from CONFIG_PATH (arguments.py, setattr per key)
cfg_kwargs = (yaml.safe_load(open(sys.argv[1])) or {}).get("env_kwargs") or {}
for path, want in zip(sys.argv[2::2], sys.argv[3::2]):
    want = want == "true"
    df = pd.read_parquet(path)
    bad = 0
    for _, row in df.iterrows():
        # engine.py:310 -- args first, row second, row wins per leaf
        merged = finalize_env_kwargs(
            deep_merge(cfg_kwargs, coerce_meta(row["metadata"]).get("env_kwargs") or {})
        )
        # the profile's own env_kwargs must SURVIVE the merge: they are the prompt/tool surface
        bad += int(
            merged.get("reward_shaping") is not want
            or merged.get("loop_detect") != 5
            or list(merged.get("extra_tools") or []) != ["open_app", "response", "terminate"]
        )
    print(f"{path}: rows={len(df)} reward_shaping={want} bad_rows={bad}")
    assert bad == 0, f"{path}: {bad} rows do not merge to reward_shaping={want}"
PY
```

Expected output — 160 train rows and 256 eval rows all merging to `True`, zero bad rows on both:

    devs/exps/train/mobile/data/mobilegym.train160.shaped.parquet:  rows=160 reward_shaping=True  bad_rows=0
    devs/exps/train/mobile/data/mobilegym.eval256.shaped.parquet:   rows=256 reward_shaping=True  bad_rows=0

Re-run this whenever the profile yaml changes. A shaping key added there would be overridden on both
sides by the rows, which is the point — but `loop_detect` or `extra_tools` changing is a prompt/tool
surface change, and this is where it shows up.

**Stage 2 — derive the cell's two manifests from `eval256`.** `fam37n` is not exported; it is a
deterministic FUNCTION of `eval256` plus `tasks.json`, so a new host reproduces it from the repo
alone and every seed replicate trains on the same 37 tasks. Do not assemble it by copying rows out
of whichever manifests happen to be on the box — that is how a MIXED instance policy gets in (see
the warning above), and the tag column still reads uniform afterwards.

```bash
W=${W:-/workspaces/cua-lite}
OUT=${OUT:-$W/devs/exps/train/mobile/data} PINNED=${PINNED:-0} uv run python - <<'PY'
import collections, hashlib, json, os, re

import pandas as pd

W = os.environ.get("W", "/workspaces/cua-lite")
DATA = f"{W}/devs/exps/train/mobile/data"
OUT = os.environ.get("OUT", DATA)
# PINNED=0 -> fam37n/fam37ne (fresh instance per rollout). PINNED=1 -> fam37/fam37e, the control
# whose train rows are tagged "eval" so the registry's seed 42 pins ONE instance for the whole run.
PINNED = os.environ.get("PINNED") == "1"


def family(task: str) -> str:
    app, name = task.split(".", 1)
    m = re.match(r"([A-Z][a-z0-9]*(?:[0-9]+)?)", name)
    return app + "." + (m.group(1) if m else name)


tj = json.load(open(f"{W}/lite/gym/envs/mobilegym/data/tasks.json"))
src = pd.read_parquet(f"{DATA}/mobilegym.eval256.shaped.parquet")
src["_task"] = [r["metadata"]["env_key"].split("@")[-1] for _, r in src.iterrows()]
assert len(src) == 256, f"source is not the 256-row eval export: {len(src)}"

# L1-L3 only: L4 is 31.2% of the eval split and 38.6% of the step budget for a 4% solve rate.
by = collections.defaultdict(list)
for t in src["_task"]:
    if tj[t]["difficulty"] in ("L1", "L2", "L3"):
        by[family(t)].append(t)

# deterministic: families by name, members by name; first member trains, its siblings evaluate
fams = sorted((f, sorted(v)) for f, v in by.items() if len(v) >= 2)
train_ids = [v[0] for _, v in fams]
eval_ids = [t for _, v in fams for t in v[1:]]
assert (len(fams), len(train_ids), len(eval_ids)) == (37, 37, 52), (
    f"pool changed: {len(fams)} families, {len(train_ids)}/{len(eval_ids)} tasks"
)
assert not set(train_ids) & set(eval_ids)

for ids, tag, name in (
    (train_ids, "eval" if PINNED else "train", "fam37" if PINNED else "fam37n"),
    (eval_ids, "eval", "fam37e" if PINNED else "fam37ne"),
):
    rows = src[src["_task"].isin(ids)].copy()
    rows["_o"] = [ids.index(t) for t in rows["_task"]]
    rows = rows.sort_values("_o").drop(columns=["_task", "_o"]).reset_index(drop=True)
    rows["metadata"] = [dict(m, split=tag) for m in rows["metadata"]]
    rows.to_parquet(f"{OUT}/mobilegym.{name}.shaped.parquet", index=False)

    # assert the RESOLVED policy, not the tag: env_kwargs seed > row tag > tasks.json seed
    pol = collections.Counter()
    for m in rows["metadata"]:
        kw = m.get("env_kwargs") or {}
        assert kw.get("reward_shaping") is True, f"{name}: shaping lost in the copy"
        if "seed" in kw:
            pol["PINNED(env_kwargs)"] += 1
        elif m["split"] != "eval":
            pol["ENGINE(f(rollout_id,group_index))"] += 1
        elif tj[m["env_key"].split("@")[-1]].get("seed") is not None:
            pol["PINNED(registry)"] += 1
        else:
            pol["RANDOM(per-episode)"] += 1  # breaks the GRPO group baseline -- never ship this
    sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
    print(f"  {name:8s} rows={len(rows):3d} tag={tag:5s} tasks_sha={sha} policy={dict(pol)}")
    want = "PINNED(registry)" if tag == "eval" else "ENGINE(f(rollout_id,group_index))"
    assert list(pol) == [want], f"{name}: MIXED instance policy {dict(pol)}"
PY
```

Expected output — one policy per manifest, never two:

    fam37n   rows= 37 tag=train tasks_sha=9a9919bc69347310 policy={'ENGINE(f(rollout_id,group_index))': 37}
    fam37ne  rows= 52 tag=eval  tasks_sha=54a4681090218909 policy={'PINNED(registry)': 52}

Those two `tasks_sha` values are the cell's identity: they cover the task list and its order, so a
host that prints different ones is not running this experiment whatever the row counts say.

**This chain was verified end to end, not assumed** (2026-09-22, `tasks.json` sha256 `96acea86…`):

| stage | where | result |
|---|---|---|
| Stage 1 re-export of `eval256` | pod at `18c7382` | **byte-for-byte identical**, file sha256 `7fbd2ee5…` |
| Stage 2 rebuild of `fam37n` / `fam37ne` | pod at `016df2d`, training this cell | **0 differing rows out of 37 and 52** across every column, row order included |

Re-run both stages on a new host and compare the shas before launching; a manifest that arrived by
`scp` has no provenance. One trap on these pods: a script under `/tmp` gets `/tmp` on `sys.path`,
NOT the working directory, so `cd /workspaces/cua-lite && python /tmp/build.py` can import a
DIFFERENT cua-lite checkout that happens to be installed in the environment — same package name,
older code, no error. Run the recipe with `PYTHONPATH=$W` and print `lite.__file__` when it
imports `lite` at all.

</details>

#### Greedy hides the gain

**The in-training eval is greedy and it reads flat; the same checkpoints scored at
`temperature=1.0` read as a rising curve.** Both are the same 52-task `fam37e` manifest and the same
checkpoints; only the decoder differs.

| rollout | greedy (`EVAL_TEMPERATURE=0`, `EVAL_N=4`) | `t=1`, `--group-size 4` | |
|---|---:|---:|---:|
| | `fam37` / `fam37n` | `fam37` | `fam37n` |
| 0 | .41106 / .35817 | **0.3698 ±0.0033** *(3 passes)* | same checkpoint, same manifest |
| 4 | +1.33pp / +6.03pp | 0.3575 −1.23pp *(1)* | 0.4085 ±0.0155 **+3.87pp** *(3)* |
| 9 | +0.25pp / +5.39pp | 0.4198 **+5.00pp** *(1)* | 0.4356 ±0.0132 **+6.58pp** *(3)* |
| 14 | +1.35pp / +4.84pp | 0.4432 **+7.34pp** *(1)* | running |
| 19 | +3.65pp / — | running | — |

`r0` is one number for both arms, not two: they start from the same SFT checkpoint and `fam37e` and
`fam37ne` are byte-identical, so the 3-pass `0.3698` is the shared baseline every `pp` above is
measured against. It also answers the question that prompted the re-runs — the first pass's `0.3686`
was not bad luck: three passes span `0.3671–0.3737`.

This is the gap [`EVAL_TEMPERATURE=0`](#rl) flags and does not size: GRPO optimises the
`rollout_temperature=1.0` distribution, and greedy scores a policy it never trained. Scoring the
sampled objective is what `--group-size 4` (4 rollouts per task, `lite/infer/rollout.py`, "for GRPO
variance analysis") is for.

**Sampled eval also needs the samples.** `--group-size 4` is not decoration: two `t=1` passes of the
same checkpoint on this manifest differ by **0.42–1.39pp**, against **5.29pp** for two greedy
`EVAL_N=4` measurements of the same SFT checkpoint on the same 52 tasks (`fam37` and `fam37n` r0,
identical checkpoint, identical manifest, identical seeds — a pure repeat). Do not read a single
`t=1` pass with `--group-size 1`.

**What is not settled.** `fam37` has one pass per point, so only its `r9`/`r14` clear the noise by a
margin worth quoting; its `r4` dip is inside the repeat noise and is not evidence of early
degradation. And do not adopt `r0`'s ±0.0033 as the noise floor — three passes at `r4` and `r9`
spread ±0.0155 and ±0.0132 on the same protocol, so **±~1.5pp is the working figure** and a tight
triple is luck, not precision. Both arms still rise well past that: `fam37n` +6.58pp by `r9`,
`fam37` +7.34pp by `r14`.

```bash
# --- EVAL HOST --- score a GRPO checkpoint the way GRPO was trained.
# --prompt-data pins the SAME manifest the arm evaluated on; --splits eval would silently
# score all 256 tasks and the numbers would not compare to the in-training curve.
# --sampling-kwargs overrides the profile's pinned temperature: 0.0 (CLI beats yaml).
uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B --model-path "$CKPT" \
  --env-id mobilegym --concurrency 16 \
  --prompt-data "$W/devs/exps/train/mobile/data/mobilegym.fam37e.shaped.parquet" \
  --env-kwargs '{"reward_shaping": true}' \
  --group-size 4 --sampling-kwargs '{"temperature": 1.0}' \
  --config-path "$W/devs/exps/train/mobile/configs/qwen3_5/mobile.use.i1.reasoning.yaml" \
  --log-root "$LOGS/<slug>"
```

- **`num_valid` becomes `tasks x group_size`** — 52 x 4 = 208. `solved` counts trajectories, not
  tasks, and the 4 samples of one task are NOT independent (same task, same registered instance), so
  the effective n sits between 52 and 208. Do not compute a binomial interval on 208.
- **Budget the wall clock at `t=1`, not at greedy.** A greedy pass of these 52 tasks takes ~30 min; the
  same manifest at `t=1 --group-size 4` takes ~60 min on an otherwise-idle pod, and the last 20% of
  the tasks take half of it — unsolved episodes run to the step cap (L3 = 45) instead of stopping
  early, and all 4 samples must finish before the task closes. Task count is a misleading progress
  indicator here; watch step lines.

#### What this cell actually measured

Run as written below, this cell **does not move the eval**: `0.4688 -> 0.4600 -> 0.4569` over 20
rollouts on a 176-task `L1-L3` slice (−1.19pp, 0.22σ). Five variants were run to find out why, all
on Qwen3.5-4B, all with the same optimizer settings. Denominators differ between them, so compare
each line's *delta*, never the absolute across lines.

| variant | start | task pool | delta |
|---|---:|---|---:|
| as written (from SFT, `train160` → `eval256`) | 0.4688 | 176 `L1-L3` | −1.19pp (20 rollouts) |
| same, from **base** | 0.2689 | 176 `L1-L3` | +0.70pp (7) |
| **same templates**, fresh instance per rollout, from **base** | 0.4580 | 109 train-split | **+12.14pp** (11) |
| same templates, **one fixed instance**, from base | 0.2094 | 175 eval-split | −1.16pp (5) |
| same templates, fresh instance, from **SFT**, zero template overlap | 0.3779 | 85 `L1-L3` | +0.66pp (5) |

Three things the spread rules out, and one it does not:

- **Not the optimizer, and not a starved signal.** The same settings produce +12.14pp on line 3, and
  the `rollout/mixed_group_ratio` this repo logs sat at 94% on line 5 — nearly every group carried
  within-group variance, so homogeneous groups were not the constraint.
- **Not difficulty.** `mixed_group_ratio` averages 36% on the 160-task full-difficulty pool vs 39% on
  the `L1+L2` subset — within noise of each other. What L4 costs is the denominator, not the gradient:
  it is 33% of the tasks and 38% of the step budget for a 4% solve rate, so dropping it lifts the same
  checkpoint's start from 0.2983 to 0.3779 without training anything.
- **It is the gap between the train and eval task distributions.** The only variant that moves is the
  one where the two sides share templates (line 3). `train160` and `eval256` share **no class name**,
  and the eval split is 34.4% cross-app against the train split's 11.2%.
- **Unresolved: how much of line 3 survives SFT.** Line 5 is that question asked cleanly — same
  protocol, templates the SFT teacher data never saw — and it returns +0.66pp. But it has one
  post-training point at rollout 5, where line 3 was still only at +7.85pp; it reached +12.14pp at
  rollout 10. Read line 5 as *not yet evidence either way*, not as a null.

A Qwen3-VL-2B control run of the top-level README's GRPO example (same env, same task pool, 35
rollouts) gains **+11.82pp**, and its gain is entirely `L1`: 2/20 → 12/20, McNemar p=0.002, with
`L2` flat at 7→10. The mechanism is visible in the behaviour, not the score — 2B starts with a 25.8%
truncation rate and 18.6 turns per episode; GRPO cuts those to 11.8% and 12.5, which converts tasks
it could already do but could not finish inside the step budget. Qwen3.5-4B starts at 0.8% truncation
and 10.9 turns, so that particular headroom does not exist for it. GRPO's reachable failure modes,
not model scale, are what decide whether this cell pays.

```bash
# --- Slime container ---
# sync, 8 GPUs colocated, TP=4 (-> DP=2). Start from the gpt5_5 + <think> SFT checkpoint for the
# same mobile.use.i1.reasoning surface, so step 0's eval is that cell's Results row.
W=/workspaces/cua-lite
P=mobile.use.i1.reasoning
DS=mobilegym_r30
T=gpt5_5
RLDS=fam37n                              # 37 train / 52 eval, same families, L1-L3
RS=101                                   # --rollout-seed  } one seed is not a result; the
SD=5001                                  # --seed          } a-e table above lists all five
CELL=grpo.$P.$RLDS.from_sft.rs${RS}s${SD}
# Start from the SHIPPED SFT weights, not from whatever iter_* this host happens to carry: a fresh
# machine has none, and `sort -V | tail -1` on a host that has several silently picks another epoch.
CKPT="$W/.ckpts/pulled/sft.$P.$DS.$T/epoch_2"
[ -d "$CKPT" ] || uv run hf download "ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T" \
  --include 'epoch_2/*' --local-dir "$W/.ckpts/pulled/sft.$P.$DS.$T"

: "${CUA_LITE_ENV_SERVER_URL:?paste export line from env-server shell}"
: "${CUA_LITE_ENV_SERVER_TOKEN:?paste export line from env-server shell}"
[ -d "$CKPT" ] || { echo "MISSING CKPT=$CKPT"; exit 1; }
for f in config.json tokenizer_config.json preprocessor_config.json; do
  [ -e "$CKPT/$f" ] || { echo "MISSING $CKPT/$f"; exit 1; }
done

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_TRAIN_GPUS=8 TP_SIZE=4 MBS=1 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  HF_CKPT="$CKPT" \
  CUA_LITE_MULTIMODAL_LAZY_EXPAND=1 \
  ENV_ID=mobilegym \
  PROMPT_DATA="$W/devs/exps/train/mobile/data/mobilegym.fam37n.shaped.parquet" \
  EVAL_PROMPT_DATA="$W/devs/exps/train/mobile/data/mobilegym.fam37ne.shaped.parquet" \
  ENV_CONCURRENCY=32 \
  ROLLOUT_SEED="$RS" SEED="$SD" \
  ROLLOUT_BATCH_SIZE=16 \
  N_SAMPLES_PER_PROMPT=8 \
  NUM_STEPS_PER_ROLLOUT=8 \
  ROLLOUT_MAX_RESPONSE_LEN=2048 \
  ROLLOUT_TEMPERATURE=1.0 \
  LR=1e-6 \
  KL_LOSS_COEF=0.00 REF_LOAD="" \
  DISTRIBUTED_TIMEOUT_MINUTES=60 \
  CONFIG_PATH="$W/devs/exps/train/mobile/configs/qwen3_5/$P.yaml" \
  EVAL_TEMPERATURE=1.0 N_SAMPLES_PER_EVAL_PROMPT=4 \
  SKIP_EVAL_BEFORE_TRAIN=0 \
  SAVE=1 NO_SAVE_OPTIM=1 SAVE_INTERVAL=5 EVAL_INTERVAL=5 NUM_ROLLOUT=30 \
  SAVE_HF_DIR="$W/.ckpts/qwen3_5-4b/$CELL/iter_{rollout_id}" \
  SAVE_DIR="/root/checkpoints/qwen3_5-4b/$CELL/megatron" \
  WANDB_GROUP_SUFFIX=".$CELL" \
  bash "$W/scripts/train/run_grpo.sh" < /dev/null
```

- **`KL_LOSS_COEF=0.00` means there is no KL term at all**, and `REF_LOAD=""` follows from it — no
  reference model is loaded, so the policy is free to move away from the SFT initialisation without
  a penalty. That is a property of this experiment, not an omission: with 37 training tasks and 30
  rollouts the risk being managed is too little movement, not too much. Any run that adds KL is a
  different cell and needs its own row.
- **`DISTRIBUTED_TIMEOUT_MINUTES=60` widens NCCL's watchdog from its 10-minute default.** A rollout
  here takes ~26 minutes and the slowest rank can sit well past 600 s inside one collective; at the
  default that reads as `Watchdog caught collective operation timeout` and kills the run, which is
  how a healthy-but-slow arm gets mistaken for a distributed bug.
- **`EVAL_TEMPERATURE=1.0` is the point of this cell, and `run_grpo.sh` does not default to it.**
  The default is 0 (`run_grpo.sh:264`, `--eval-temperature "${EVAL_TEMPERATURE:-0}"`), which scores a
  greedy policy GRPO never optimised; its own comment at `:256` already says "set
  EVAL_TEMPERATURE=1 to score the optimised objective". Assert it from slime's parsed argument
  table, not from the launcher's echo — the same rule the seed knobs need.
- **`ROLLOUT_SEED`/`SEED` are NOT in `run_grpo.sh` as committed.** It has no such knob and no generic
  env passthrough, so without a pod-local patch every run lands on slime's defaults
  (`--rollout-seed` 42, `--seed` 1234) and a seed sweep silently becomes one run repeated. Patch
  locally, never commit, and gate the launch on both flags being present.
- **`CONFIG_PATH` is mandatory, and its failure mode here is SILENT.** Unset, `run_grpo.sh` derives
  `scripts/configs/qwen3_5/compact/${ENV_ID}.yaml` (run_grpo.sh:120) — and for `mobilegym` that file
  EXISTS, and it differs on THREE surfaces: `history_n: 1`, `reward_shaping: true`, and — by
  OMISSION — `image_max` back to the protocol default 4 and `enable_thinking` back to the adapter
  default `False`. A `<think>`-trained checkpoint would silently run Action-only at four images,
  against a different reward baseline, and the run would start and look normal. Desktop's version of this note
  says the derived path does not exist and the script exits 1; do not carry that reassurance over.
- **The env-server preflight will refuse a dirty or mismatched checkout, and nothing else warns
  you.** `run_grpo.sh:136` sources `scripts/train/utils/preflight.sh`, which requires the env-server's
  `/host_status.cua_lite.commit` to equal the train host's `git rev-parse HEAD` (`preflight.sh:92-99`)
  and **exits 1 when the server reports `dirty: true`** (`:100-108`, the server deriving `dirty` from
  `git status --porcelain`, `lite/gym/remote/server.py:766-768`). So a campaign run wants the
  checkout COMMITTED on both hosts at the same SHA, with `serve_env.py` restarted from it. For a
  deliberate dev run on a dirty tree, set `CUA_LITE_ALLOW_DIRTY_ENV_SERVER=1` — it is an explicit
  opt-out, and it means the run's provenance no longer identifies the code that produced it.
- **`NO_SAVE_OPTIM=1` means this run CANNOT be resumed.** `ckpt_args.sh:25-26` passes
  `--no-save-optim`, so the megatron checkpoints carry no optimizer state; a later `RESUME=1` adds
  `--load "$SAVE_DIR"` without `--no-load-optim` (`ckpt_args.sh:36-38`), and under
  `--megatron-to-hf-mode bridge` slime takes the branch that does not add it either
  (`arguments.py:1732-1737`), so Megatron dies with `KeyError: 'optimizer'`. A desktop campaign hit
  exactly this. The recovery path after a crash is a FRESH run from the last `SAVE_HF_DIR/iter_*`,
  not `RESUME=1`. Set `NO_SAVE_OPTIM=0` instead if you want resumability and can pay the disk.
- **`HF_CKPT` is mandatory.** Omitting it defaults to `/root/models/$MODEL_ID`
  (`run_grpo.sh:146-147`), silently starts from base Qwen3.5-4B, and answers a different question.
  The block pulls `epoch_2` of the shipped SFT repo rather than globbing a local `iter_*`, so every
  replicate starts from the same bytes on any host — the pulled layout is `epoch_2/`, which no
  `iter_*` glob matches, and the guard below turns a failed download into an exit rather than a
  silent fall back to base.
- **Step 0 is the run's own baseline**, paired against the `gpt5_5` + `<think>` / `i1` SFT cell up to
  eval noise. Landing far below that usually means the wrong `P`, `CKPT`, or config was used. The
  step-0 eval runs because `SKIP_EVAL_BEFORE_TRAIN=0`, which is also the shipped default — written
  out so a later change to it cannot silently remove this run's reference point.
- **`EVAL_TEMPERATURE=0` matches the Results table.** The SFT rows are scored through
  `scripts/rollout.py` under a profile yaml that pins `temperature: 0.0`, so a greedy in-training
  curve is the one that can be read against them. It is also `run_grpo.sh`'s default (run_grpo.sh:263)
  — written out so the pairing survives a default change. The cost is that it does not score the
  sampled objective GRPO optimises; pick one per campaign and say which, because mixing them across a
  table makes the rows unreadable. The size of the greedy-vs-sampled gap on MobileGym is **not
  measured**.
- **`mobile.use.i1.reasoning` is heavier than the `compact` profile RL normally uses** (compact pins
  `history_n: 1` to save VRAM). That cost is the price of matching the chosen SFT parent exactly.
  `TP_SIZE=4`, not SFT's 2, is the conservative colocated-RL default because sglang engines hold
  memory for the whole run. Do **not** change `image_max`/`history_n` to save memory — that makes the
  run incomparable to its SFT parent. `ASYNC=1` is the one structural lever that would make lower TP
  plausible. This run has no memory trace yet.
- **`ROLLOUT_MAX_RESPONSE_LEN=2048`** — the 512 default is a real ceiling on a `.reasoning` arm,
  whose tail carries `<think>` before every call. This is a GENERATION budget: it leaves the prompt
  untouched, so comparability with the SFT parent holds. It DOES shift the in-training curve, so keep
  it fixed for the whole run.
- **`16 x 8 = 128` trajectories per rollout, split into 8 optimizer steps** (`128 / nspr 8 = 16` GBS
  — slime derives it, do NOT also pass `--global-batch-size`). All three are `run_grpo.sh`'s shipped
  defaults (`run_grpo.sh:192-194`), written out here for the same reason browser writes them out: so
  a later change to the launcher cannot silently move this run. 160 tasks / `RBS` 16 = one epoch
  every 10 rollouts, and `NUM_ROLLOUT=20` is two epochs.

  Deliberately the shipped defaults, not a tuned guess. A larger global batch averages more
  trajectories into one optimizer step — the `1/sqrt(N)` trade, buying smoothness at the cost of
  per-step signal — and GRPO's signal here is already thin, because a homogeneous group (all-pass or
  all-fail) contributes exactly zero advantage. With only 160 templates a larger `RBS` also takes a
  coarser bite out of the pool per rollout. NOT MEASURED on mobilegym: if you change it, change it as
  a deliberate, recorded arm.

  WARNING: **`nspr` must divide `ROLLOUT_BATCH_SIZE`, not just `RBS x n`**, and `run_grpo.sh` only
  preflights the weaker one. `slime/utils/dp_schedule.py` slices by *trajectory* in task-contiguous
  order, so a task's `n` samples share an optimizer step only when `RBS % nspr == 0` (16 % 8 = 0). A
  split group makes advantages sum non-zero inside each step it lands in. `RBS=12, n=4, nspr=8` passes
  the script's check and still splits.

  **`NUM_ROLLOUT` divisible by the interval is load-bearing.** `should_run_periodic_action` fires on
  `(rollout_id + 1) % interval == 0`, but `train.py` passes `args.num_rollout` for **save** and not
  for eval, so the last rollout always saves whether or not the interval says to — and nothing grants
  eval the same exemption. 20/5 puts the forced final save on a step eval was going to run anyway.
  Re-check this for any other `NUM_ROLLOUT`.
- **`ENV_CONCURRENCY=128`** — 128 trajectories per rollout, so one wave. mobilegym is light: the
  in-container pool is `max_browsers 128 x contexts_per_browser 8` = 1024 contexts, so the binding
  limit is host CPU (96 vCPU here), not the pool. **256 is plausible and UNMEASURED** — measure it on
  a throwaway run before raising it, not mid-campaign, because concurrency changes timing and these
  episodes contain waits. On pool saturation the container returns 503 and the host maps it to
  `CapacityExhausted`; that is the signal you went too far.
- **Train-split tasks randomize per reset, and that is handled.** Only the eval split carries
  registered seeds. For the training side the engine draws one shared env seed per group per
  `rollout_id` (keyed on `sample.group_index`), so a group's 8 samples see the same initial state and
  the GRPO baseline is well-defined; distinct groups still draw independently.
- **`CUA_LITE_MULTIMODAL_LAZY_EXPAND=1` at this batch.** Default 0 expands every screenshot into
  `pixel_values` inside the rollout and ships float tensors through plasma. Desktop measured that
  taking a node to 972 GB of Ray's 1024 at its own `32 x 8`; this run is half that batch, and a
  720x1600 phone frame is not a 1920x1080 desktop
  one, so treat the desktop number as the reason to use the lazy path, not as this run's expected
  footprint. `--multimodal-lazy-expand-fn-path` alone does nothing — it registers the hook, this env
  var gates the payloads.
- **Read `return_mean`, not the bare `eval/{ds}` scalar.** That one averages over the DENSE reward
  list, so every errored rollout dilutes it as a 0.0. The engine logs the valid-only mean beside the
  counts: `Eval <ds>: return_mean=..., N valid / M errored / ...`. They agree only while `M == 0`.
- **Export `WANDB_API_KEY`** or the every-5-steps curve this section is built around silently does
  not exist.
- **The final number must be the full 256 through `scripts/rollout.py`** on the eval host, not the
  in-training curve. The curve selects; it does not report — scoring a checkpoint with the data that
  chose it is winner's curse. The Eval block will not do it unmodified (its `cells()` lists SFT cells
  only) and Ship never uploads `grpo.*`: move the chosen `iter_*` to the eval host yourself, then, in
  the Eval block's shell, add one
  `score mobile.use.i1.reasoning "$GRPO_CKPT" "grpo.mobile.use.i1.reasoning.mobilegym_r30.gpt5_5.mobilegym_full.from_sft@$RUN.$(basename "$GRPO_CKPT")"`.
  Score several saved `iter_*` before copying one number across; the curve's argmax is not
  automatically the checkpoint to report.
- Compare against the `gpt5_5` + `<think>` / `i1` SFT cell, not the base row: this run is
  RL-from-SFT, not RL-from-base.
