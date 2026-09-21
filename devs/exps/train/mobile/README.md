# mobile.use — teacher x screenshot-profile x reasoning campaign (SFT + one filter ablation + one RL run)

Train **eleven checkpoints** — two screenshot profiles x three teachers, plus a `<think>` arm on the
two teachers that emit reasoning, plus one filter ablation — and score them on the full MobileGym
eval split. A single GRPO run from the `gpt5_5` + `<think>` SFT cell closes the file (**RL** at the
end), scored on that same eval.

| | `i1` | `i4` |
|---|---|---|
| | 720x1600, 1 img | 720x1600, 1-4 img |
| **`gpt5_5`** | ✓ | ✓ |
| **`gpt5_5` + `<think>`** | ✓ | ✓ |
| **`qwen3_5_27b`** | ✓ | ✓ |
| **`qwen3_5_27b` + `<think>`** | ✓ | ✓ |
| **`qwen3_8_27b`** | ✓ | ✓ |

Ten cells, plus an eleventh that is not in the grid: the GRPO cell's own
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
filtered pool from the same teacher. Within a COLUMN, a `+ <think>` row differs from the row above by
`enable_thinking` alone — so read a reasoning cell against **its own teacher's** Action-only row,
never the other teacher's. **Across teacher rows is NOT a clean contrast**: each teacher's pool is
its own filtered trajectories, so the pools differ in membership and in size. How much they overlap
on task id is **not measured here** (desktop measured 43% for its two extreme teachers; that number
is desktop's, not mobile's). To make it clean, intersect the pools first and draw every teacher's
rows from that intersection.

`qwen3_8_27b` has no `<think>` arm: thinking was off when it was sampled, so the reasoning config
would train an empty `<think>` on its rows — the same exclusion the desktop campaign documents. The
other two reach the field by different routes: `gpt5_5` is prompted for a `Thought:` line that
[`/devs/data/internalize_cot.py`](/devs/data/internalize_cot.py) canonicalizes before staging;
`qwen3_5_27b` was sampled with `enable_thinking` and writes it natively. The mobile configs say the
same thing in their own headers —
[`mobile.use.i1.reasoning.yaml`](/devs/exps/train/mobile/configs/qwen3_5/mobile.use.i1.reasoning.yaml).

**Naming.** A cell is a `(config stem, teacher, dataset recipe)` triple; the SFT blocks enumerate the
eleven from a `cells()` function that each block redefines (four copies — paste the one in the block
you are running). RL pins one of those cells explicitly. `$P` is the stem taken whole off the
filename and `$DS` the dataset recipe (source + row filter). Both are threaded verbatim into every
artifact — parquet `$P.$DS.$T.parquet`, checkpoint `sft.$P.$DS.$T`, HF repo
`ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T`, W&B group — so nothing downstream re-derives them and two recipes
never collide. To ablate the dataset, change `DS=` **and** `--filter` together.

### Teacher data

Collection, cleaning and publication of the three teachers' MobileGym trajectories are owned by
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
HF_DS=MobileGym                          # CONFIRM against /devs/data/mobilegym/AGENTS.md before
VARIANT=mobile.use                       # the first run; this runbook does not own either name.
                                         # NOT `mobile.use.train`: the HF path already carries the
                                         # split (`mobile/use/train/`), and the config name is
                                         # `mobile.use.<teacher>` (/devs/data/mobilegym/AGENTS.md).
                                         # `--allow-patterns` ERRORS on a 0-file match, so a wrong
                                         # VARIANT aborts the very first command for all teachers.
FILTER_R30="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) >= 0.30"
FILTER_SR="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

# The eleven cells as "<config stem> <teacher> <dataset recipe>". The <think> arm skips
# qwen3_8_27b; the last line is the filter ablation, not a grid cell.
cells() {
  for T in gpt5_5 qwen3_5_27b qwen3_8_27b; do
    for P in mobile.use.i1 mobile.use.i4; do echo "$P $T $DS"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in mobile.use.i1.reasoning mobile.use.i4.reasoning; do echo "$P $T $DS"; done
  done
  echo "mobile.use.i1.reasoning gpt5_5 $DS_SR"
}

# One root per teacher, because export_sft reads whatever --data-paths names and a shared root
# would pool them. --allow-patterns bounds the walk as well as the fetch, so a warm HF cache
# cannot drag in a sibling split. --overwrite makes this re-runnable. KEEP THIS LIST IN SYNC WITH
# cells(): a teacher in cells() but not here exports against a data root that was never
# downloaded -- and cells() is defined FOUR times below, so a teacher edit touches all four.
for T in gpt5_5 qwen3_5_27b qwen3_8_27b; do
  # --org defaults to `cua-lite`; the mobilegym runbook stages to a PRIVATE $HF_ORG first
  # (/devs/data/mobilegym/AGENTS.md), so pass it explicitly until the repo is public.
  env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
    uv run python -m lite.data.hf.download "$HF_DS" --org "${HF_ORG:-cua-lite}" \
      --allow-patterns "mobile/use/train/$VARIANT.$T/*" \
      --out "$DL/$T/cua-lite/$HF_DS" --overwrite
done

# No --sample: every cell takes its teacher's whole filtered pool, so the row counts differ
# between teachers by construction and the campaign has no row-cap axis. The per-teacher pool
# sizes are NOT MEASURED YET -- read them off the `Wrote N trajectory rows` lines of the first
# run and write them into the Results front matter.
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

#### Train

```bash
# --- Slime container, TRAIN HOST ---  (DS / DS_SR must match the Export block)
# SFT at TP=2 (8 GPUs -> DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't
# THD-pack). Serial: each run takes all 8 GPUs.
#
# SAVE_DIR / SAVE_HF_DIR / WANDB_GROUP_SUFFIX are all MANDATORY here. run_sft.sh keys the two
# checkpoint dirs AND the W&B group off PROMPT_DATA's parent dir (DATA_SLUG, run_sft.sh:131),
# which is `mobile.use` for all eleven cells -- so unset, the runs overwrite each other's
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
# The eleven cells as "<config stem> <teacher> <dataset recipe>".
cells() {
  for T in gpt5_5 qwen3_5_27b qwen3_8_27b; do
    for P in mobile.use.i1 mobile.use.i4; do echo "$P $T $DS"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
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
  `if [ -n "${WANDB_API_KEY:-}" ]`, so without it eleven multi-hour runs train with no logging at
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
# for a dirty tree just as happily, so an uncommitted edit tags eleven public repos with a sha
# that does not describe the weights, silently.)
# `uv run hf`, NOT bare `hf`: `hf repos` needs huggingface_hub >= 1.17 and the system hf may be
# older. Needs a WRITE-scoped token (`uv run hf auth login`, or HF_TOKEN). Repos are public,
# so the eval host needs no auth at all.
COMMIT="$(git rev-parse --short HEAD)"   # commit message + tag; eval still pulls main
DS=mobilegym_r30
DS_SR=mobilegym_sr
EPOCHS=2                                 # must match NUM_EPOCH in the Train block
CKPTS=.ckpts/qwen3_5-4b
# The eleven cells as "<config stem> <teacher> <dataset recipe>".
cells() {
  for T in gpt5_5 qwen3_5_27b qwen3_8_27b; do
    for P in mobile.use.i1 mobile.use.i4; do echo "$P $T $DS"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
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

Fifteen runs on the **full 256-task `mobilegym` eval split** — no filter. `mobilegym` registers no
`exclude_reason`, and this campaign deliberately does NOT take the L1+L2 subset that
[docs/grpo.md#mobilegym](/docs/grpo.md#mobilegym) uses for its 2B smoke runs, so the denominator here
is 256 and not 93. Eval tasks carry registered deterministic seeds (`seed=42`), so the task
instances are fixed across runs. Env setup:
[`lite/gym/envs/mobilegym/README.md`](/lite/gym/envs/mobilegym/README.md).

Fifteen, not eleven: **the base model runs once per config stem**, all four of them. A checkpoint
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
# The eleven cells as "<config stem> <teacher> <dataset recipe>".
cells() {
  for T in gpt5_5 qwen3_5_27b qwen3_8_27b; do
    for P in mobile.use.i1 mobile.use.i4; do echo "$P $T $DS"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
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
# The RL run, if it was scored (see the RL section): 16th line, not part of cells().
show "grpo.mobile.use.i1.reasoning.mobilegym_r30.gpt5_5.mobilegym_full.from_sft@$RUN.<iter>"
```

Paste the numbers into a snapshot file, reusing the table LAYOUT from **Results** below (the cell
format — `mean (solved/num_valid)` — is defined there). Front matter:

```markdown
# mobile.use @ <run>

- **Checkpoints**: each cell repo's `main` as of `<date>`, tag `<sha>`. Ship can SKIP a cell and
  re-run it later, so the eleven repos need not share a sha — if they diverge, list the odd ones
  out here (`uv run hf repos tag list <repo>`), or a later `--revision` re-run pulls the wrong
  weights.
- **Dataset**: `mobilegym_r30` (`episode_return >= 0.30`), whole filtered pool per teacher — record
  the per-teacher row counts the Export block printed, they are part of the result
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

**Nothing here is measured yet.** This is the scaffold the first campaign fills in; every cell is
`TBD` on purpose and no desktop or browser number transfers into it — different env, different
teacher pools, different reward definition.

Two numbers per cell, both from the SAME shaped pass: **shaped mean** first, then the **Success
Rate** as `(solved / num_valid)` — the count of `episode_return == 1.0`, which under shaping is still
exactly MobileGym's SR criterion. The shaped mean is the sensitive one and the SR is the comparable
one; report both, and say which you are comparing when you cite a number elsewhere.

**One pass per cell** — eval tasks carry registered seeds and the profiles pin `temperature: 0.0`, so
a pass is reproducible up to env timing. There is still no error bar, and the SR half carries a
binomial stderr of about 2.8pp at `n=256, p~0.3` all by itself; the shaped mean is tighter but its
spread is unmeasured here. Read the table for moves that clear those, and re-run a specific pair
rather than the whole table when one matters.

Cells read `shaped-mean (solved/256)`.

| | `i1` | `i4` |
|---|---:|---:|
| **base** | 0.1823 (34/256) | 0.2162 (43/256) |
| **base + `<think>`** | 0.1896 (38/256) | TBD |
| **`gpt5_5`** | 0.2776 (56/256) | 0.3195 (66/256) |
| **`gpt5_5` + `<think>`** | 0.3444 (69/256) | not run — see below |
| **`qwen3_5_27b`** | TBD | TBD |
| **`qwen3_5_27b` + `<think>`** | TBD | TBD |
| **`qwen3_8_27b`** | TBD | TBD |
| **`gpt5_5` + `<think>`, `> 0.5` rows** | not run | — |
| **GRPO from `gpt5_5` + `<think>`** | see below | — |

Every number in the table is a full 256-task pass. Two earlier cells that read `N/256 only` — a
truncated prefix is not comparable to a full pass, because the eval split is ordered rather than
shuffled — were re-run to completion on 2026-09-21.

Two cells read `not run`. The `> 0.5` filter ablation was dropped as uninformative on this corpus:
the gate removes only 7% of the trajectories (1133 -> 1049) and 3% of the templates (158 -> 153),
because MobileGym scores `1.0` iff success and `0.5 x progress` otherwise, so `> 0.5` IS the success
set and `>= 0.30` adds back only the partial-progress runs above `progress 0.6` — of which this
teacher produces few. The `gpt5_5` + `<think>` / `i4` cell is a training failure, recorded below.

Reading the table, once it has numbers in it:

- **There is no error bar, so read only large moves.** One pass per cell means the campaign's noise
  floor is not measured, and desktop's (±0.0011 .. ±0.0207 over 328 OSWorld tasks) is not a
  substitute — different env, different denominator. The SR half has a floor you can compute without
  re-running: `sqrt(p(1-p)/256)`, about 2.8pp at `p~0.3`. Treat anything inside that as unread on the
  SR, lean on the shaped mean for smaller moves, and re-run a specific pair rather than the whole
  table when one matters.
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

#### The `i4` + `<think>` cell does not train

`mobile.use.i4.reasoning` is the one cell in this campaign that never produced a checkpoint. Six
attempts on 2026-09-21 all died the same way, so the cell is `not run` rather than `TBD`: the work
was done and the result was a failure.

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

The eleventh cell is `(mobile.use.i1.reasoning, gpt5_5)` — the GRPO cell's own profile and teacher —
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

One GRPO run from the local **`gpt5_5` + `<think>` SFT checkpoint** trained with
[`mobile.use.i1.reasoning.yaml`](/devs/exps/train/mobile/configs/qwen3_5/mobile.use.i1.reasoning.yaml).
Trains on the **full 160-task `mobilegym` train split**, evaluates on the **full 256-task eval
split**. Read [docs/grpo.md](/docs/grpo.md) first: env-server prerequisite, sync-vs-async, and the
knobs this block does not repeat.

> **No difficulty filter on either side.** [docs/grpo.md#mobilegym](/docs/grpo.md#mobilegym)
> filters both splits to L1+L2 (80 train / 93 eval) so a 2B has a non-trivial baseline; this campaign
> does not, because its eval number has to be the full-split one the Results table above reports.
> That means the step budget varies per task (the profile yaml omits `max_steps` precisely so it
> does). L1 15 / L2 30 / L3 45 / L4 60 is exact on the TRAIN split; on the eval split a task may
> declare its own `cls.max_steps`, so registered budgets vary WITHIN a tier there. The L3/L4 tail
> is where a 4B is most likely to contribute
> nothing but zero-advantage groups. Whether that tail is worth its rollout budget is not known here;
> desktop's answer was a calibration pass ([`desktop/TASKS.md`](/devs/exps/train/desktop/TASKS.md)),
> and mobile has no equivalent yet.

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

</details>

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
RLDS=mobilegym_full                      # full 160 train / 256 eval, no difficulty filter
CELL=grpo.$P.$DS.$T.$RLDS.from_sft
CKPT=${CKPT:-$(ls -d "$W/.ckpts/qwen3_5-4b/sft.$P.$DS.$T"/iter_* 2>/dev/null | sort -V | tail -1)}

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
  PROMPT_DATA="$W/devs/exps/train/mobile/data/mobilegym.train160.shaped.parquet" \
  EVAL_PROMPT_DATA="$W/devs/exps/train/mobile/data/mobilegym.eval256.shaped.parquet" \
  ENV_CONCURRENCY=128 \
  ROLLOUT_BATCH_SIZE=16 \
  N_SAMPLES_PER_PROMPT=8 \
  NUM_STEPS_PER_ROLLOUT=8 \
  ROLLOUT_MAX_RESPONSE_LEN=2048 \
  ROLLOUT_TEMPERATURE=1.0 \
  LR=2e-6 \
  CONFIG_PATH="$W/devs/exps/train/mobile/configs/qwen3_5/$P.yaml" \
  EVAL_TEMPERATURE=0 N_SAMPLES_PER_EVAL_PROMPT=1 \
  SKIP_EVAL_BEFORE_TRAIN=0 \
  SAVE=1 NO_SAVE_OPTIM=1 SAVE_INTERVAL=5 EVAL_INTERVAL=5 NUM_ROLLOUT=20 \
  SAVE_HF_DIR="$W/.ckpts/qwen3_5-4b/$CELL/iter_{rollout_id}" \
  SAVE_DIR="/root/checkpoints/qwen3_5-4b/$CELL/megatron" \
  WANDB_GROUP_SUFFIX=".$CELL" \
  bash "$W/scripts/train/run_grpo.sh" < /dev/null
```

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
- **`HF_CKPT` is mandatory.** It must point at the local
  `sft.mobile.use.i1.reasoning.mobilegym_r30.gpt5_5/iter_*` HF export; omitting it defaults to
  `/root/models/$MODEL_ID` (run_grpo.sh:146-147), silently starts from base Qwen3.5-4B, and answers a
  different question. If the train host has no local `iter_*`, pull the export and point `CKPT` at
  it: `uv run hf download "ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T" --include "epoch_2/*" --local-dir
  "$W/.ckpts/pulled/sft.$P.$DS.$T"`. The pulled layout does NOT match the `iter_*` glob above it, so
  set the path literally: `CKPT="$W/.ckpts/pulled/sft.$P.$DS.$T/epoch_2"`.
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
