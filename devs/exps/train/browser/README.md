# browser.use — teacher x image-cap x reasoning SFT campaign

Train **eight checkpoints** — two image caps x two teachers, each with a `<think>` arm — from
WebGym's published teacher trajectories. **SFT only: scoring is not part of this campaign yet**
(see the note at the end).

| | `i4` | `i1` |
|---|---|---|
| | 1-4 screenshots | 1 screenshot |
| **`gpt5_5`** | ✓ | ✓ |
| **`gpt5_5` + `<think>`** | ✓ | ✓ |
| **`qwen3_5_27b`** | ✓ | ✓ |
| **`qwen3_5_27b` + `<think>`** | ✓ | ✓ |

**Two teachers only.** `qwen3_8_27b` is a `desktop.use` teacher and is deliberately absent here:
it emits no `reasoning_content`, so it has no `<think>` arm at all, and its Action-only cells
were not judged worth the four extra runs.

**There is no resolution axis.** WebGym renders at its own 1280x720 viewport
(`lite/gym/envs/webgym/configs/default.yaml`), so every profile sees the same pixels —
`smart_resize` rounds to 1280x704, 880 vision tokens. That is what `desktop.use` calls `lowr`,
with no `highr` to pair it against. The only knobs left are how many screenshots survive and
whether `<think>` is on, which is why these configs are named for `image_max` alone.

`iN` sets `image_max: N` with `fold_size` tracking it, and leaves `history_n` at the protocol
default of 100 — so every past turn is still rendered in full: its text, its literal
`<tool_call>`, and on the reasoning arm its `<think>`. Only the old *pixels* become
`"This screenshot has been collapsed."`

**What compares to what.** Within a ROW only the image cap moves. Within a COLUMN, a `+ <think>`
row differs from the row above by `enable_thinking` alone — so read a reasoning cell against
**its own teacher's** Action-only row, never the other teacher's. **Across teacher rows is NOT a
clean contrast**: each teacher's pool is its own successes, so the pools differ in membership and
a shared seed buys nothing — see "Row count is not difficulty" under Export.

**Reading a `<think>` effect, mind the empty-`<think>` rate.** A `.reasoning` cell trains an
empty `<think></think>` on any turn the teacher ended with prose and no `Thought:` line.
Measured on 300 trajectories of each exported `i4.reasoning` parquet, counting the teacher
turns that reach the model as rendered history: **`gpt5_5` 32.9% empty** (750 of 2278),
**`qwen3_5_27b` 0.0%** (0 of 1380 — `enable_thinking` wrote the field natively). That is a
three-times-larger gap than `desktop.use` sees, so the two `+ <think>` rows are not measuring
the same intervention. Compare within a teacher, never across.

The Action-only rows are clean: their rendered history carries **0** filled `<think>` blocks
(1436 of 1436 empty, same sample), so no teacher reasoning leaks into the arm that is supposed
to be without it.

> **Whatever scores these later must pass the same config yaml.** `$P` — the config stem —
> names the checkpoint, and the matching `configs/qwen3_5/$P.yaml` is the prompt surface it was
> trained on. An `i1` checkpoint scored under `i4` is measuring a surface it never saw.

**`i4` costs about half of `i1` to train**, which is not obvious. `fold_size == image_max` keeps
four adjacent steps token- and image-prefix compatible, so `build_segment_samples` packs them
into one sequence; a one-image window is never prefix-compatible with the next step, so `i1`
emits one sequence per step. **Per trajectory**, over 150 exported `webgym_1k` rows with the
real tokenizer and 880 vision tokens per 1280x704 screenshot:

| | seq/traj | text tok | image slots | vision tok | total |
|---|---:|---:|---:|---:|---:|
| `gpt5_5` `i4` | 2.58 | 5 570 | 8.9 | 7 814 | **13 385** |
| `gpt5_5` `i1` | 8.88 | 17 783 | 8.9 | 7 814 | **25 598** |

These are TRAIN tokens, not export size: the parquet is trajectory-level and packing happens at
train time, so an `i4` and an `i1` export of the same cell are within 0.1 MiB of each other.
**0.52x** for `gpt5_5`, **0.57x** for `qwen3_5_27b`; the `seq/traj` column is the packing
itself. (The `fold_size` sweep that fixes `fold_size == image_max` is measured and owned by the
comment in [`configs/qwen3_5/browser.use.i4.yaml`](/devs/exps/train/browser/configs/qwen3_5/browser.use.i4.yaml)
— read the rate there, not here.) The image slots are IDENTICAL — every
screenshot is encoded exactly once either way, so the saving is not in pixels. It is all text:
`i1` re-renders the full turn history once per step, `i4` once per segment. Tune `MBS` on `i4`
anyway: packing makes its sequences the longest.

**Naming.** A cell is a `(config stem, teacher)` pair; each block below redefines the same
`cells()` function (**three copies** — paste the one in the block you are running). `$P` is the
stem taken whole off the filename, `$DS` the dataset recipe (`webgym_1k` = source + row cap).
Both are threaded verbatim into every artifact — parquet `$P.$DS.$T.parquet`, checkpoint
`sft.$P.$DS.$T`, HF repo `ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T`, W&B group — so nothing downstream
re-derives them and two recipes never collide. To ablate the dataset, change `DS=` **and**
`--sample` together.

### SFT

#### Export

`cua-lite/WebGym` publishes both teachers: **`browser.use.gpt5_5`** (3143 rows, 148 shards) and
**`browser.use.qwen3_5_27b`** (967 rows, 32 shards, tag `f50e58f`). All eight cells export today.
Shard counts are the Hub layout; `lite.data.hf.download` consolidates them, so what lands under
`--out` is ONE parquet per teacher plus an `images/` tree (26 101 and 5 634 files).

> **Adding a second teacher is not an append.** `lite.data.hf.upload` is a declarative full sync:
> it plans the whole repo from the local staging dir and deletes everything else, so staging only
> the new config and uploading DELETES the published `gpt5_5` shards. Follow
> [Add A Config To A Published Dataset](/devs/data/AGENTS.md#add-a-config-to-a-published-dataset).
> The two teachers also hand over different roots — `gpt5_5` stages its `.think` siblings
> (`internalize_cot.py` canonicalizes its prompted `Thought:` into `reasoning_content`),
> `qwen3_5_27b` stages the bare `_clean` roots because `enable_thinking` wrote the field
> natively. See [`/devs/data/webgym/AGENTS.md`](/devs/data/webgym/AGENTS.md) step 3.

```bash
# --- TRAIN HOST ---  (the Slime container mounts the repo root, so its .data/ is what Train
# reads; exporting anywhere else leaves Train with no parquet to read)
DS=webgym_1k                             # dataset recipe: source + row cap
DL=.data/huggingface-webgym              # per-teacher roots: $DL/$T/cua-lite/...
OUT=.data/sft/qwen3_5/browser.use
CFG=devs/exps/train/browser/configs/qwen3_5
FILTER="lambda m: (m.others.get('episode_return') or 0) > 0.5"

# The eight cells as "<config stem> <teacher>". Both teachers get a <think> arm here --
# unlike desktop.use, neither of them is thinking-off.
cells() {
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4 browser.use.i1; do echo "$P $T"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4.reasoning browser.use.i1.reasoning; do echo "$P $T"; done
  done
}

# One root per teacher: export_sft reads whatever --data-paths names and a shared root would
# pool them. --allow-patterns bounds the walk as well as the fetch. KEEP THIS LIST IN SYNC
# WITH cells(), which is defined THREE times in this file.
for T in gpt5_5 qwen3_5_27b; do
  uv run python -m lite.data.hf.download WebGym \
    --allow-patterns "browser/use/train/browser.use.$T/*" \
    --out "$DL/$T/cua-lite/WebGym" --overwrite
done

# WebGym rewards are BINARY (0 or 1.0, no partial credit), so `> 0.5` keeps exactly the solved
# episodes. --filter runs before --sample, so the cap lands on kept rows.
# No `exclude_reason` clause, unlike desktop.use's filter: WebGym task metadata has no such
# field (lite/gym/envs/webgym/main.py, _task_metadata), so it could only ever be a no-op.
while read -r P T; do
  uv run python -m lite.train.export.export_sft \
    --config "$CFG/$P.yaml" \
    --model-id Qwen/Qwen3.5-4B \
    --data-paths "$DL/$T/cua-lite/WebGym" \
    --image-root "$DL/$T" \
    --filter "$FILTER" --sample 1000 --seed 42 --no-strict \
    -o "$OUT/$P.$DS.$T.parquet" < /dev/null   # or the child eats the rest of the cell list
done <<< "$(cells)"

# --no-strict makes a conversion failure a SKIP, not an error, so a wrong --image-root writes
# an empty parquet and still exits 0. Count the rows before trusting them.
# Expected rows = min(--sample, pool). qwen3_5_27b's pool IS 967, so its four cells have ZERO
# margin: any future unconvertible row shortens them silently. Check the exact number per
# teacher, not --sample.
while read -r P T; do
  case "$T" in gpt5_5) N=1000;; qwen3_5_27b) N=967;; esac
  uv run python -c "
import sys, pyarrow.parquet as pq
n, want = pq.read_metadata(sys.argv[1]).num_rows, int(sys.argv[2])
print(('OK   ' if n == want else 'SHORT'), n, f'(want {want})', sys.argv[1])
" "$OUT/$P.$DS.$T.parquet" "$N" < /dev/null
done <<< "$(cells)"
```

**Why 1000.** Both configs are success-filtered upstream — staged from `_clean` roots — so
`$FILTER` keeps 100% and the pool IS the published row count; it stays in the command only to
remain correct if a future config ever stages raw roots. So 1000 fills the four `gpt5_5`
cells and takes the ENTIRE qwen pool at 967: `export_sft` prints `Only 967 rows convertible` and
exits 0. That 3.3% row gap is deliberate — an order of magnitude smaller than the task-mix gap
below, and capping every cell at 967 would discard 2176 `gpt5_5` rows to close a difference that
is already noise.

**Row count is not difficulty.** The two teachers solved different tasks, and the pools say so
before any sampling. Measured over the full downloaded pools:

| | distinct tasks | `d1` | `d7` | above `d7` |
|---|---:|---:|---:|---:|
| `gpt5_5` | 3126 | 516 | 514 | 19 (`d8`-`d15`) |
| `qwen3_5_27b` | 964 | 298 | 88 | **0** |

**745** tasks are in both pools — 77% of everything `qwen3_5_27b` solved, but only 24% of
`gpt5_5`'s. At this campaign's caps (1000 rows for `gpt5_5`, the whole 967 for `qwen3_5_27b`)
the two cells end up sharing on the order of 240 tasks. That is not a sampling artifact; it is
what each teacher could solve, and `qwen3_5_27b` simply never solved anything above `d7`. So a
teacher cell compares teacher AND task mix together. Read the teacher rows that way, or
re-sample stratified on `others.difficulty` (present on every row) if you need the mix held
fixed — the intersection is too small to train on.

#### Train

```bash
# --- Slime container, TRAIN HOST ---  (DS must match the Export block)
# SFT at TP=2 (8 GPUs -> DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN cannot
# THD-pack). Serial: each run takes all 8 GPUs.
#
# SAVE_DIR / SAVE_HF_DIR / WANDB_GROUP_SUFFIX are MANDATORY. run_sft.sh keys both checkpoint
# dirs AND the W&B group off PROMPT_DATA's parent dir, which is `browser.use` for every cell --
# so unset, the runs overwrite each other's checkpoints and land in one W&B group.
#
# Export WANDB_API_KEY first: run_sft.sh builds its whole W&B argument list only when that
# variable is set, so without it the runs train fine and log nowhere.
#
# Leave SAVE_INTERVAL unset. What writes the two iter_* dirs is slime's epoch-boundary save, so
# NUM_EPOCH=2 gives exactly 2 -- which is what Ship gates on.
DS=webgym_1k
# The eight cells as "<config stem> <teacher>". Both teachers get a <think> arm.
cells() {
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4 browser.use.i1; do echo "$P $T"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4.reasoning browser.use.i1.reasoning; do echo "$P $T"; done
  done
}

while read -r P T; do
  TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
    MODEL_ID=Qwen/Qwen3.5-4B \
    SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=2 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
    PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5/browser.use/$P.$DS.$T.parquet \
    SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-4b/sft.$P.$DS.$T/iter_{rollout_id} \
    SAVE_DIR=/root/checkpoints/qwen3_5-4b/sft.$P.$DS.$T/megatron \
    WANDB_GROUP_SUFFIX=".$P.$DS.$T" \
    bash /workspaces/cua-lite/scripts/train/run_sft.sh < /dev/null
done <<< "$(cells)"
```

- `MBS=1` is the safe start; tune it on `i4`, which carries the longest sequence.
- `NO_SAVE_OPTIM=1` keeps weights only — these checkpoints are for eval, not for resuming.

#### Ship the checkpoints

The checkpoints are this campaign's deliverable, and whatever consumes them next will run on
another machine — so they travel through the Hub: **repo** = the cell
(`ZHZisZZ/qwen3_5-4b.sft.<config-stem>.<dataset>.<teacher>`), **subdir** = `epoch_1/`,
`epoch_2/`, **tag** = the producing commit.

The tag is provenance, not the entry point: a consumer pulls `main`, so a re-run replaces the
epoch dirs and is picked up with no sha to carry between hosts. `--revision <tag>` is for
re-reading an OLD set; a re-run at the same commit *moves* the tag. Epoch dirs are named by rank
at upload time — training writes `iter_<N>` with slime's 0-based index, a number to read off
disk, never predict.

```bash
# --- TRAIN HOST ---  (run from the repo root; commit FIRST -- `git rev-parse` reports a sha for
# a dirty tree just as happily, so an uncommitted edit tags eight public repos with a sha that
# does not describe the weights, silently.)
# `uv run hf`, NOT bare `hf`: `hf repos` needs huggingface_hub >= 1.17 and the system hf may be
# older. Needs a WRITE-scoped token (`uv run hf auth login`, or HF_TOKEN). Repos are public, so
# whoever reads them later needs no auth at all.
COMMIT="$(git rev-parse --short HEAD)"   # commit message + tag; a consumer still pulls main
DS=webgym_1k
EPOCHS=2                                 # must match NUM_EPOCH in the Train block
CKPTS=.ckpts/qwen3_5-4b
# The eight cells as "<config stem> <teacher>". Both teachers get a <think> arm.
cells() {
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4 browser.use.i1; do echo "$P $T"; done
  done
  for T in gpt5_5 qwen3_5_27b; do
    for P in browser.use.i4.reasoning browser.use.i1.reasoning; do echo "$P $T"; done
  done
}

while read -r P T; do
  REPO="ZHZisZZ/qwen3_5-4b.sft.$P.$DS.$T"
  # `ls` first, and gate on the count: creating the repo up front would leave an empty public one
  # behind for a cell that never trained, and a half-trained cell must not reach whoever scores
  # these, which pulls whatever is on main.
  DIRS=$(ls -d "$CKPTS/sft.$P.$DS.$T"/iter_* 2>/dev/null | sort -V)
  if [ "$(echo "$DIRS" | grep -c .)" -ne "$EPOCHS" ]; then
    echo "SKIP $REPO: $(echo "$DIRS" | grep -c .) iter dir(s), expected $EPOCHS -- not uploading"
    continue
  fi
  uv run hf repos create "$REPO" --repo-type model --exist-ok < /dev/null
  ep=0
  for D in $DIRS; do
    ep=$((ep + 1))
    # Scoring loads the tokenizer/processor from --model-path, not --model-id, so the epoch dir
    # must carry slime's FULL HF export. Check before spending 8 GB of upload on it: a
    # weights-only dir only fails on the consuming host, after upload AND download.
    for f in config.json tokenizer_config.json preprocessor_config.json; do
      [ -e "$D/$f" ] && continue
      echo "SKIP $REPO: $D has no $f -- AutoProcessor would fail later; stopping this cell"
      ep=-1; break
    done
    [ "$ep" = "-1" ] && break
    uv run hf upload "$REPO" "$D" "epoch_$ep" \
      --repo-type model --commit-message "$COMMIT: epoch $ep (from $(basename "$D"))" \
      < /dev/null \
      || { echo "SKIP $REPO: upload of $D failed -- not tagging"; ep=-1; break; }
  done
  # Both breaks above escape the `for D` loop ONLY; without this, a half-uploaded cell falls
  # through and gets tagged as if complete. Pushed epoch dirs stay on main, untagged.
  [ "$ep" = "-1" ] && continue

  # create-or-move; the Hub has no move, so it is delete-then-create.
  uv run hf repos tag delete "$REPO" "$COMMIT" --repo-type model --yes < /dev/null \
    || echo "note: no existing '$COMMIT' tag on $REPO (expected on a first upload)"
  uv run hf repos tag create "$REPO" "$COMMIT" --repo-type model -m "$COMMIT" < /dev/null \
    || echo "ERROR $REPO has NO '$COMMIT' tag now; weights are on main -- re-tag by hand"
done <<< "$(cells)"
```

A `SKIP` line is the signal to re-train or re-export that cell, not to upload it by hand:
`ls -d iter_*` is the only record of how many epochs actually landed, and the per-file check is
the only thing standing between a weights-only export and a repo that fails on the consuming
host after an 8 GB round trip.

#### Eval — deliberately not in this campaign

Nothing here is scored yet. Two things make WebGym eval its own piece of work rather than a
paste of `desktop.use`'s block, and they are the reason it is deferred:

- **Every eval task hits a live website.** Measured over 7.6k training rollouts on this env,
  **18.8%** ended in `EnvBlocked` — a bot wall, not a model failure. Those are void episodes:
  out of the denominator, so two arms can end up averaged over different task counts. Whatever
  results table is eventually added has to carry `num_valid` per cell, not just a mean.
- **The split is 1167 tasks**, and one pass is ten runs — the eight checkpoints plus one base
  run per image cap, since a checkpoint is only comparable to a baseline that saw the same
  prompt surface. That is a lot of live browsing. It will need a seeded `--sample` subset held
  identical across every checkpoint, and more than one pass per cell: `desktop.use` needed three
  on a hermetic env to reach a ±0.009 median noise floor, and this env is strictly noisier.
