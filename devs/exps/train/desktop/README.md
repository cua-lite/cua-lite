# desktop.use — teacher x screenshot-profile SFT campaign

Train **four checkpoints over a 2x2** — two teachers x two screenshot profiles — and score them
on one eval.

| | `lowr.h4` | `highr.h1` |
|---|---|---|
| | 1280x720, up to 4 screenshots | native 1920x1080, 1 screenshot |
| **`gpt5_5`** | `sft.$DS.gpt5_5.lowr.h4` | `sft.$DS.gpt5_5.highr.h1` |
| **`qwen3_8_27b`** | `sft.$DS.qwen3_8_27b.lowr.h4` | `sft.$DS.qwen3_8_27b.highr.h1` |

Teacher axis: whose trajectories teach better. Profile axis: how to spend a fixed VRAM budget —
four small screenshots, or one big one. Every cell draws 5000 rows from **Lite.ScaleCUA** at the
same seed, so within a column only the teacher moves and within a row only the profile does.

`$DS` names the dataset recipe (`scalecua_5k` = source + row cap) and is threaded through every
artifact — parquet, checkpoint dir, W&B group, HF repo — so two recipes never collide. To ablate
the dataset, change `DS=` **and** `--sample` together; the cap is in the name.

> **Train and eval must use the same profile yaml.** `$P` selects both the checkpoint and
> `--config-path`, in every block below. A `highr.h1` checkpoint scored under `lowr.h4` is
> measuring a prompt surface it never saw. Resolutions, for the record: `lowr.h4` downsamples to
> 1280x720 and Qwen `smart_resize` rounds it to **1280x704**; `highr.h1` leaves the env's native
> **1920x1080**, rounded to **1920x1088**. Both envs (`lite.osworld`, `lite.cuagym`) declare
> `display_resolution: [1920, 1080]`.

### SFT

#### Export

Four exports — the profile changes what the model sees, so each cell needs its own parquet. The
download is profile-independent: one per teacher, shared by both columns.

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
DS=scalecua_5k                           # dataset recipe: source + row cap
OUT=.data/sft/qwen3_5/desktop.use
CFG=devs/exps/train/desktop/configs/qwen3_5
FILTER="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

# --allow-patterns picks ONE teacher: the HF config name IS the shard dir on the Hub, and the
# pattern bounds the walk as well as the fetch. Without it you pull every teacher plus the `rl`
# variant. --overwrite makes this re-runnable (download refuses a non-empty --out).
# --filter runs before --sample, so the 5000 cap lands on kept rows; the fixed --seed means both
# profiles of one teacher draw the same rows.
for T in gpt5_5 qwen3_8_27b; do
  uv run python -m lite.data.hf.download Lite.ScaleCUA \
    --allow-patterns "desktop/use/train/desktop.use.train.$T/*" \
    --out ".data/huggingface-$T/cua-lite/Lite.ScaleCUA" --overwrite

  for P in lowr.h4 highr.h1; do
    uv run python -m lite.train.export.export_sft \
      --config "$CFG/desktop.use.$P.yaml" \
      --model-id Qwen/Qwen3.5-4B \
      --data-paths ".data/huggingface-$T/cua-lite/Lite.ScaleCUA" \
      --image-root ".data/huggingface-$T" \
      --filter "$FILTER" --sample 5000 --seed 42 --no-strict \
      -o "$OUT/$DS.$T.$P.parquet"
  done
done
```

#### Train

```bash
# --- Slime container, TRAIN HOST ---  (DS must match the Export block)
# SFT at TP=2 (8 GPUs -> DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't
# THD-pack). Serial: each run takes all 8 GPUs.
#
# SAVE_DIR / SAVE_HF_DIR / WANDB_GROUP_SUFFIX are all MANDATORY here. run_sft.sh derives every
# one of them from PROMPT_DATA's parent dir (DATA_SLUG), which is `desktop.use` for all four
# cells -- so unset, the four runs overwrite each other's checkpoints and share one W&B group.
#
# One ckpt per epoch needs no flag: slime saves on epoch boundaries and on the final step
# regardless of SAVE_INTERVAL's value (it only has to be set, which ckpt_args.sh always does).
# Leave it alone -- and don't clear it either, an empty value falls back to every 5 rollouts.
#
# CUA_LITE_TRAIN_BROAD_CLEANUP=1 is what makes the LOOP safe: run_sft.sh starts a Ray head and
# never stops it, and its only teardown is utils/cleanup.sh, sourced at startup and opt-in. So
# each cell tears down the previous cell's cluster instead of stacking a second one on top.
# Safe here precisely because the Slime container is dedicated.
DS=scalecua_5k
for T in gpt5_5 qwen3_8_27b; do
  for P in lowr.h4 highr.h1; do
    TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
      CUA_LITE_TRAIN_BROAD_CLEANUP=1 \
      MODEL_ID=Qwen/Qwen3.5-4B \
      SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=2 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
      PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5/desktop.use/$DS.$T.$P.parquet \
      SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-4b/desktop.use/sft.$DS.$T.$P/iter_{rollout_id} \
      SAVE_DIR=/root/checkpoints/qwen3_5-4b/desktop.use/sft.$DS.$T.$P/megatron \
      WANDB_GROUP_SUFFIX=".$DS.$T.$P" \
      bash /workspaces/cua-lite/scripts/train/run_sft.sh
  done
done
```

- `MBS=1` is the safe start, and **tune it on `lowr.h4`** — that column has the longer sequence
  per micro-batch (~4 x 880 vision tokens vs one 2040), so it is what binds. `highr.h1` fits more
  easily but runs ~4x the micro-batches per step, so expect it to be slower, not lighter.
  TP=2 fits both at 4B; `TP_SIZE=4` if it OOMs.
- `NO_SAVE_OPTIM=1` keeps weights only — these checkpoints are for eval, not for resuming.

#### Ship the checkpoints

Train and eval usually run on different machines, so checkpoints travel through the Hub:
**repo** = the cell (`ZHZisZZ/qwen3_5-4b.desktop.use.sft.<dataset>.<teacher>.<profile>`),
**tag** = the producing commit, **subdir** = `epoch_1/`, `epoch_2/`.

Tagging by commit is the dataset runbooks' convention
([`devs/data/lite.scalecua/AGENTS.md`](/devs/data/lite.scalecua/AGENTS.md)) and pins a checkpoint
to the code that made it. A re-run at the same commit *moves* the tag, matching
`lite/data/hf/upload.py:470-501`. Epoch dirs are named by rank at upload time, because training
writes `iter_<N>` with slime's 0-based rollout index — a number you should read off disk, never
predict.

```bash
# --- TRAIN HOST ---  (run from the repo root; commit first, a tag can't carry -dirty)
# `uv run hf`, NOT bare `hf`: `hf repos` needs huggingface_hub >= 1.5 and the system hf may be
# older. Needs a WRITE-scoped token (`uv run hf auth login`, or HF_TOKEN). Repos are public,
# so the eval host needs no auth at all.
COMMIT="$(git rev-parse --short HEAD)"
DS=scalecua_5k
CKPTS=.ckpts/qwen3_5-4b/desktop.use

for T in gpt5_5 qwen3_8_27b; do
  for P in lowr.h4 highr.h1; do
    REPO="ZHZisZZ/qwen3_5-4b.desktop.use.sft.$DS.$T.$P"
    uv run hf repos create "$REPO" --repo-type model --exist-ok

    ep=0
    for D in $(ls -d "$CKPTS/sft.$DS.$T.$P"/iter_* | sort -V); do
      ep=$((ep + 1))
      # eval loads the tokenizer/processor from --model-path, not --model-id, so the epoch dir
      # must carry slime's FULL HF export. Check before spending 8 GB of upload on it:
      # a weights-only dir only fails on the eval host, after upload AND download.
      for f in config.json tokenizer_config.json preprocessor_config.json; do
        [ -e "$D/$f" ] || echo "WARNING $D: no $f -- AutoProcessor will fail at eval"
      done
      uv run hf upload "$REPO" "$D" "epoch_$ep" \
        --repo-type model --commit-message "$COMMIT: epoch $ep (from $(basename "$D"))"
    done
    if [ "$ep" -ne 2 ]; then
      echo "SKIP $REPO: $ep epoch(s), expected 2 -- not tagging an incomplete run"
      continue
    fi

    # create-or-move; the Hub has no move, so it is delete-then-create.
    uv run hf repos tag delete "$REPO" "$COMMIT" --repo-type model --yes \
      || echo "note: no existing '$COMMIT' tag on $REPO (expected on a first upload)"
    uv run hf repos tag create "$REPO" "$COMMIT" --repo-type model -m "$COMMIT" \
      || echo "ERROR $REPO has NO '$COMMIT' tag now; weights are on main -- re-tag by hand"
  done
done
```

`hf upload` is single-commit and not resumable; `upload-large-folder` is, but takes no
path-in-repo and so cannot express `epoch_<k>/`.

#### Eval

Six runs on the `lite.osworld` eval split (the rows left after `--filter` drops `exclude_reason`;
the tracked `catalog.lock.json` gives 328 of 369) — the
[Lite.OSWorld row](/docs/eval.md#osworld--liteosworld) of [docs/eval.md](/docs/eval.md). Env
setup: [`lite/gym/envs/lite/osworld/README.md`](/lite/gym/envs/lite/osworld/README.md).

Six, not four: **the base model runs once per profile.** A checkpoint must be scored against a
baseline that saw the same screenshot surface.

```bash
# --- EVAL HOST ---
# COMMIT must be the TRAIN host's sha, not this host's HEAD. Short-sha width is Git-configured,
# so even at the same commit two clones can abbreviate differently.
# `uv run hf repos tag list <repo>` shows what exists.
COMMIT="${COMMIT:?paste the sha the TRAIN host tagged with}"
EPOCH=epoch_2                            # epoch_1 = after 1 epoch
DS=scalecua_5k
CFG=devs/exps/train/desktop/configs/qwen3_5
PULL=.ckpts/pulled
LOGS=.logs/rollout/Qwen_Qwen3.5-4B/lite.osworld

for T in gpt5_5 qwen3_8_27b; do
  for P in lowr.h4 highr.h1; do
    uv run hf download "ZHZisZZ/qwen3_5-4b.desktop.use.sft.$DS.$T.$P" \
      --revision "$COMMIT" --include "$EPOCH/*" \
      --local-dir "$PULL/sft.$DS.$T.$P@$COMMIT"
    # snapshot_download has no empty-match guard: a wrong $EPOCH yields an empty dir, silently.
    [ -e "$PULL/sft.$DS.$T.$P@$COMMIT/$EPOCH/config.json" ] \
      || echo "ERROR $T.$P: no $EPOCH/config.json at revision $COMMIT"
  done
done

# --config-path uses the SAME $P as the checkpoint -- that pairing is the experiment.
gpu=0
for P in lowr.h4 highr.h1; do
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.5-4B \
    --env-id lite.osworld --splits eval --concurrency 8 \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --config-path "$CFG/desktop.use.$P.yaml" \
    --log-root "$LOGS/base.$P" &
  gpu=$((gpu + 1))

  for T in gpt5_5 qwen3_8_27b; do
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/rollout.py \
      --model-id Qwen/Qwen3.5-4B \
      --model-path "$PULL/sft.$DS.$T.$P@$COMMIT/$EPOCH" \
      --env-id lite.osworld --splits eval --concurrency 8 \
      --filter "lambda m: not m.others.get('exclude_reason')" \
      --config-path "$CFG/desktop.use.$P.yaml" \
      --log-root "$LOGS/sft.$DS.$T.$P@$COMMIT.$EPOCH" &
    gpu=$((gpu + 1))
  done
done

wait
```

`--model-id` picks the adapter and action space; the weights, tokenizer and chat template all
come from `--model-path`.

Needs 6 free GPUs as written. With fewer, run one profile per batch (reset `gpu=0` each `P`) —
dropping `&`/`wait` serializes all six and still names `CUDA_VISIBLE_DEVICES=5`.

Score each run from `<log-root>/summary.json` -> `stats.mean_episode_return` (denominator
`num_valid`). Read the grid down a column for the teacher effect, across a row for the profile
effect, and compare a checkpoint only against the base run of its own column.

<!--
## Ablations

The reasoning ablation: train a checkpoint with the same recipe, only swapping
`desktop.use.lowr.h4.yaml` for `desktop.use.lowr.h4.reasoning.yaml` (reasoning in Qwen3.5's
native `<think>` channel). Everything below mirrors [SFT](#sft) step for step, so it yields one
more checkpoint comparable to its Action-only counterpart.

**This ablation applies to `gpt5_5` only.** It is the one teacher whose rows carry a Thought:
`gpt5_5` is prompted for a `Thought:` line and stores it as an `inline_reasoning` part, while
`qwen3_8_27b` runs with thinking off and stores only `action_description` + `tool_calls`
(see [the dataset runbook](/devs/data/lite.scalecua/AGENTS.md)). Running internalize-CoT on the
`qwen3_8_27b` root is a no-op, and exporting it under a thinking-enabled config would train an
empty `<think>` block.

One flow difference, and it is forced: `desktop.use.lowr.h4.reasoning.yaml` reasons in the
`<think>` channel, so the Thought must first be moved out of `inline_reasoning` and into
`reasoning_content` — an extra internalize-CoT step producing a `.think` copy, which this flow
exports from. Exporting `.think` data with thinking off fails fast in the other direction
(`SFT prompt/target boundary broke (enable_thinking=False)`): the internalized reasoning lives
in the target, so the prompt/target boundary no longer holds.

### SFT (Reasoning)

#### Export (Reasoning)

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
OUT=.data/sft/qwen3_5-reasoning/desktop.use
FILTER="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

# The gpt5_5 root is already downloaded by the SFT flow above. Internalize CoT into a sibling
# `.think` root, then export from that. Images stay path-referenced, so the `.think` copy is
# tiny — but its rows still resolve images against the ORIGINAL root, which is why
# `--image-root` keeps pointing there.
T=gpt5_5

uv run python devs/exps/train/desktop/internalize_cot.py \
  --in  ".data/huggingface-$T/cua-lite/Lite.ScaleCUA" \
  --out ".data/huggingface-$T/cua-lite/Lite.ScaleCUA.think"

uv run python -m lite.train.export.export_sft \
  --config devs/exps/train/desktop/configs/qwen3_5/desktop.use.lowr.h4.reasoning.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths ".data/huggingface-$T/cua-lite/Lite.ScaleCUA.think" \
  --image-root ".data/huggingface-$T" \
  --filter "$FILTER" --sample 5000 --seed 42 \
  -o "$OUT/scalecua_5k.$T.parquet"
```

#### Train (Reasoning)

```bash
# --- Slime container ---
# Same recipe as the Action-only runs, only PROMPT_DATA and SAVE_HF_DIR differ. SFT at TP=2
# (8 GPUs → DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't THD-pack).
# One ckpt per epoch. A single run: this ablation has one teacher.
TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=2 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5-reasoning/desktop.use/scalecua_5k.gpt5_5.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/desktop.use/sft.gpt5_5/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

- `MBS=1` is the safe start (4-image steps are heavy); raise to `MBS=2` only if it fits.
- TP=2 fits the 4-image step at 4B; fall back to `TP_SIZE=4` only if it OOMs.

#### Eval (Reasoning)

One run — `gpt5_5` is the only teacher this ablation has — on the same `lite.osworld` eval
split as [Eval](#eval). The base model is already scored there; it is not re-run.

```bash
# --- host ---

# reasoning SFT on gpt5_5 data (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --model-path .ckpts/qwen3_5-reasoning-4b/desktop.use/sft.gpt5_5/iter_<N> \
  --env-id lite.osworld --splits eval --concurrency 8 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path devs/exps/train/desktop/configs/qwen3_5/desktop.use.lowr.h4.reasoning.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/sft.reasoning.gpt5_5
```

Compare its `stats.mean_episode_return` against the `gpt5_5` Action-only checkpoint from
[Eval](#eval). Both use the same tasks, seeds, and step budget, so the only moving part is the
`<think>` channel. Do not compare runs that use different task sets, sampling
configurations, or task budgets.
-->
