# browser.use - gpt5_5 no-goto + WebVoyager-clean WebGym SFT campaign

Train **four SFT checkpoints** from successful `gpt5_5` WebGym teacher trajectories after dropping
trajectories that used `goto` and exact WebVoyager task-instruction matches, then score them on
one fixed WebVoyager read-only eval subset. This campaign expands the validated single-cell
WebGym -> WebVoyager SFT flow to every browser prompt surface from the full filtered `gpt5_5`
pool. SFT checkpoints are uploaded to Hugging Face for eval / RL handoff; GRPO checkpoints stay
local until a winner is selected.

| | `i4` | `i1` |
|---|---|---|
| | 1280x720, 1-4 screenshots | 1280x720, 1 screenshot |
| **base** | eval only | eval only |
| **base + `<think>`** | eval only | eval only |
| **`gpt5_5`** | train + eval | train + eval |
| **`gpt5_5` + `<think>`** | train + eval | train + eval |
| **GRPO from `gpt5_5` + `<think>`** | — | train + eval |

All cells use the no-goto browser surface in
[`configs/qwen3_5/`](/devs/exps/train/browser/configs/qwen3_5): `resolution: [1280, 720]`,
`extra_tools: ["back", "response"]`, and no WebGym `instruction_template`. WebVoyager runs
against an offline WebHarbor mirror, so direct URL navigation is not a transferable action. The
projection lives in the experiment-local
[`utils/filter.py`](/devs/exps/train/browser/utils/filter.py), not in the generic WebGym data
cleaner.

`iN` sets `image_max: N` with `fold_size` tracking it. `history_n` stays at the protocol default,
so text history remains full while older screenshots collapse. There is no resolution axis in
this browser campaign; both profiles use the same 1280x720 viewport.

**What compares to what.** Within a row, only the screenshot cap moves. Within a column, the
`+ <think>` row differs from the Action-only row by `enable_thinking`. The teacher is fixed to
`gpt5_5`, so the SFT rows compare prompt surfaces over the same decontaminated source pool.

**Naming.** A cell is a config stem. `$P` is the config stem and `$DS` is the dataset recipe.
They are threaded verbatim into artifacts: parquet `$P.$DS.parquet`, checkpoint `sft.$P.$DS`,
HF repo `ZHZisZZ/qwen3_5-4b.sft.$P.$DS`, rollout log slug `sft.$P.$DS@$RUN.<epoch>`. For this
campaign:

```bash
DS=webgym_gpt5_5_nogoto_wvclean
```

### SFT

#### Export

One parquet per cell. The config decides how the same filtered trajectories render, so `i4`,
`i1`, Action-only, and `<think>` cells cannot share an exported SFT parquet.

```bash
# --- TRAIN HOST ---  (the Slime container mounts the repo root, so its .data/ is what Train
# reads; exporting on another host leaves Train with no parquet to read)
DS=webgym_gpt5_5_nogoto_wvclean
WANT_ROWS=2381
T=gpt5_5
DL=.data/huggingface-webgym
CLEAN=.data/huggingface-webgym-nogoto-wvclean
OUT=.data/sft/qwen3_5/browser.use
CFG=devs/exps/train/browser/configs/qwen3_5
FILTER="lambda m: (m.others.get('episode_return') or 0) > 0.5"

profiles() {
  printf '%s\n' browser.use.i4 browser.use.i1 browser.use.i4.reasoning browser.use.i1.reasoning
}

uv run python -m lite.data.hf.download WebGym \
  --allow-patterns "browser/use/train/browser.use.$T/*" \
  --out "$DL/$T/cua-lite/WebGym" --overwrite

uv run python devs/exps/train/browser/utils/filter.py \
  --input "$DL/$T/cua-lite/WebGym" \
  --out "$CLEAN/$T/cua-lite/WebGym" \
  --overwrite
```

`filter.py` also excludes rows whose normalized task instruction exactly matches the committed
WebHarbor WebVoyager eval manifest. This is deliberate best-effort decontamination: WebGym's
`pae-webvoyager` source is a large PAE-generated task family, not just the original 643
WebVoyager tasks, so source-level removal would throw away too much useful web-navigation data.

The expected current pool size is measured from the published local WebGym root.

| teacher | source rows | rows with top-level `goto` | exact WebVoyager matches after no-goto | filtered rows |
|---|---:|---:|---:|---:|
| `gpt5_5` | 3143 | 754 | 8 | 2381 |

**Why full `gpt5_5`.** There is no teacher-count fairness axis in this campaign anymore, so keep
all 2381 filtered `gpt5_5` rows. The only rows removed are `goto` trajectories and exact
WebVoyager task matches.

Check every filtered raw root before exporting. This is the boundary that proves `goto`, exact
WebVoyager task matches, and the WebGym reset wrapper are gone, and that the remaining tool
surface is exactly `back,response`.

```bash
uv run python - "$CLEAN/$T/cua-lite/WebGym/browser/use/train/browser.use.$T.parquet" "$WANT_ROWS" <<'PY'
import sys
import json
import re
from pathlib import Path

import pandas as pd

from lite.core.tools.calls import tool_call_name
from lite.data.staging import coerce_messages, coerce_meta

path, want_rows = sys.argv[1], int(sys.argv[2])
df = pd.read_parquet(path)
wv_tasks = json.loads(Path("lite/gym/envs/webharbor/webvoyager/data/tasks.json").read_text())
wv_instructions = {
    re.sub(r"\s+", " ", task["instruction"].strip())
    for task in wv_tasks.values()
}
goto_rows = 0
webvoyager_rows = 0
wrapper_rows = 0
bad_surface_order = 0

for _, row in df.iterrows():
    messages = coerce_messages(row["messages"])
    goto_rows += int(any(
        tool_call_name(tc) == "goto"
        for msg in messages if msg.get("role") == "assistant"
        for tc in msg.get("tool_calls") or []
    ))
    first_text = next(
        (
            part.get("text", "")
            for part in messages[0].get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        ),
        "",
    )
    webvoyager_rows += int(re.sub(r"\s+", " ", first_text.strip()) in wv_instructions)
    wrapper_rows += int("Initial website:" in first_text)
    names = [
        schema["function"]["name"]
        for schema in coerce_meta(row["metadata"]).get("extra_tool_schemas") or []
    ]
    bad_surface_order += int(names != ["back", "response"])

print(
    f"{path}: rows={len(df)} goto_rows={goto_rows} wrapper_rows={wrapper_rows} "
    f"webvoyager_rows={webvoyager_rows} bad_surface_order={bad_surface_order}"
)
assert len(df) == want_rows
assert goto_rows == 0
assert webvoyager_rows == 0
assert wrapper_rows == 0
assert bad_surface_order == 0
PY
```

Export all four SFT parquets from the same filtered full pool.

```bash
while read -r P; do
  uv run python -m lite.train.export.export_sft \
    --config "$CFG/$P.yaml" \
    --model-id Qwen/Qwen3.5-4B \
    --data-paths "$CLEAN/$T/cua-lite/WebGym" \
    --image-root "$CLEAN/$T/cua-lite/WebGym" \
    --filter "$FILTER" --no-strict \
    -o "$OUT/$P.$DS.parquet" < /dev/null
done <<< "$(profiles)"
```

Count rows and check the rendered prompt surface before spending GPU time.

```bash
while read -r P; do
  uv run python - "$OUT/$P.$DS.parquet" "$WANT_ROWS" <<'PY'
import sys

import pandas as pd
import pyarrow.parquet as pq

path, want = sys.argv[1], int(sys.argv[2])
rows = pq.read_metadata(path).num_rows
prompt = pd.read_parquet(path).iloc[0]["steps"][0]["prompt"]
print(f"{path}: rows={rows}")
assert rows == want
assert '"name": "goto"' not in prompt
assert "Initial website:" not in prompt
assert "submit it with the response action" not in prompt
assert '"name": "back"' in prompt
assert '"answer"' in prompt or '"name": "response"' in prompt
PY
done <<< "$(profiles)"
```

#### Train

Run inside the Slime container. Each cell starts from base Qwen3.5-4B weights and writes a local
HF-format checkpoint. The loop is serial: each SFT run takes all eight visible GPUs.

```bash
# --- Slime container, TRAIN HOST ---
# SFT at TP=2 (8 GPUs -> DP=4). BSHD + MBS; do not pass MAX_TOKENS_PER_GPU
# for this qwen3_5/GDN setup.
W=/workspaces/cua-lite
DS=webgym_gpt5_5_nogoto_wvclean

profiles() {
  printf '%s\n' browser.use.i4 browser.use.i1 browser.use.i4.reasoning browser.use.i1.reasoning
}

while read -r P; do
  CELL=sft.$P.$DS
  TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
    MODEL_ID=Qwen/Qwen3.5-4B \
    SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=2 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
    PROMPT_DATA="$W/.data/sft/qwen3_5/browser.use/$P.$DS.parquet" \
    SAVE_HF_DIR="$W/.ckpts/qwen3_5-4b/$CELL/iter_{rollout_id}" \
    SAVE_DIR="/root/checkpoints/qwen3_5-4b/$CELL/megatron" \
    WANDB_GROUP_SUFFIX=".$CELL" \
    bash "$W/scripts/train/run_sft.sh" < /dev/null
done <<< "$(profiles)"
```

- `MBS=1` is the safe start. Tune only after the first cell proves stable.
- `NO_SAVE_OPTIM=1` keeps weights only; these checkpoints are for eval, not optimizer resume.
- `SAVE_HF_DIR` writes local HF-format dirs. Run **Ship the checkpoints** after validation to make
  the SFT checkpoints available to eval and RL hosts.
- These are decontaminated full-pool runs. Do not reuse checkpoints from the older
  no-goto-only or row-capped campaigns for this table.

#### Check checkpoints

Require a full HF export for every cell before eval. A weights-only directory will fail later
when the evaluator loads tokenizer or processor files.

```bash
DS=webgym_gpt5_5_nogoto_wvclean
CKPTS=.ckpts/qwen3_5-4b

profiles() {
  printf '%s\n' browser.use.i4 browser.use.i1 browser.use.i4.reasoning browser.use.i1.reasoning
}

while read -r P; do
  DIR=$(ls -d "$CKPTS/sft.$P.$DS"/iter_* 2>/dev/null | sort -V | tail -1)
  [ -n "$DIR" ] || { echo "MISSING sft.$P.$DS"; continue; }
  for f in config.json tokenizer_config.json preprocessor_config.json; do
    [ -e "$DIR/$f" ] || { echo "MISSING $DIR/$f"; continue 2; }
  done
  echo "OK $P -> $DIR"
done <<< "$(profiles)"
```

#### Ship the checkpoints

Train, eval, and RL often run on different machines, so the SFT checkpoints travel through the
Hub: **repo** = the cell (`ZHZisZZ/qwen3_5-4b.sft.<config-stem>.<dataset>`), **subdir** =
`epoch_1/`, `epoch_2/`, **tag** = the producing commit.

The tag is provenance, not the entry point: eval pulls `main`, so a re-run replaces the epoch dirs
and the next campaign picks them up without a sha to carry between hosts. Use `--revision <tag>`
only to re-run an old campaign; a re-run at the same commit moves the tag. Epoch dirs are named by
rank at upload time - training writes `iter_<N>` with Slime's 0-based index, a number to read off
disk, never predict.

```bash
# --- TRAIN HOST ---  (run from the repo root; commit first so the tag describes the weights)
# `uv run hf`, not bare `hf`: the system hf may be older. Needs a WRITE-scoped token
# (`uv run hf auth login`, or HF_TOKEN). Repos are public, so eval/RL hosts need no auth.
COMMIT="$(git rev-parse --short HEAD)"
DS=webgym_gpt5_5_nogoto_wvclean
EPOCHS=2
CKPTS=.ckpts/qwen3_5-4b

profiles() {
  printf '%s\n' browser.use.i4 browser.use.i1 browser.use.i4.reasoning browser.use.i1.reasoning
}

while read -r P; do
  REPO="ZHZisZZ/qwen3_5-4b.sft.$P.$DS"
  DIRS=$(ls -d "$CKPTS/sft.$P.$DS"/iter_* 2>/dev/null | sort -V)
  if [ "$(echo "$DIRS" | grep -c .)" -ne "$EPOCHS" ]; then
    echo "SKIP $REPO: $(echo "$DIRS" | grep -c .) iter dir(s), expected $EPOCHS -- not uploading"
    continue
  fi
  uv run hf repos create "$REPO" --repo-type model --exist-ok < /dev/null

  ep=0
  for D in $DIRS; do
    ep=$((ep + 1))
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
  [ "$ep" = "-1" ] && continue

  uv run hf repos tag delete "$REPO" "$COMMIT" --repo-type model --yes < /dev/null \
    || echo "note: no existing '$COMMIT' tag on $REPO (expected on a first upload)"
  uv run hf repos tag create "$REPO" "$COMMIT" --repo-type model -m "$COMMIT" < /dev/null \
    || echo "ERROR $REPO has NO '$COMMIT' tag now; weights are on main -- re-tag by hand"
done <<< "$(profiles)"
```

`hf upload` is single-commit and not resumable; `upload-large-folder` is, but takes no
path-in-repo and so cannot express `epoch_<k>/`.

#### Eval

Eight runs on the fixed 128-row WebVoyager read-only eval subset: four base prompt surfaces and
four SFT checkpoints. A checkpoint must be scored with the same config stem it was trained with;
an `i1` checkpoint under `i4` is measuring a prompt surface it never saw.

The base model runs once per full prompt surface, not once per image cap. A reasoning checkpoint's
fair base is the same `.reasoning` config on base weights. Previous browser/WebVoyager numbers are
not copied into this campaign because the dataset contract changed: this run uses full filtered
`gpt5_5` rows plus exact WebVoyager task decontamination.

The read-only WebVoyager eval manifest is a fixed experiment asset:
[`data/webvoyager.eval128.readonly.seed42.parquet`](/devs/exps/train/browser/data/webvoyager.eval128.readonly.seed42.parquet).
Generate it only once. Normal eval and RL runs should reuse the existing file; re-running the
export is an intentional manifest refresh, not part of routine scoring.

```bash
# --- ONE-TIME DATA BUILD; skip this block if $TASKS already exists ---
EVAL=devs/exps/train/browser/data
TASKS="$EVAL/webvoyager.eval128.readonly.seed42.parquet"
mkdir -p "$EVAL"

if [ -e "$TASKS" ]; then
  echo "keep existing fixed eval manifest: $TASKS"
else
  uv run python -m lite.train.export.export_tasks \
    --env-id webharbor.webvoyager --split eval \
    --filter "lambda m: not m.others.get('mutating')" --sample 128 --seed 42 \
    -o "$TASKS"
fi

uv run python - "$TASKS" <<'PY'
import hashlib
import sys

import pandas as pd

import lite.gym as gym
from lite.data.staging import coerce_meta

df = pd.read_parquet(sys.argv[1])
keys = [coerce_meta(row["metadata"])["env_key"] for _, row in df.iterrows()]
key_hash = hashlib.sha256("\n".join(keys).encode()).hexdigest()
assert len(df) == 128
assert all(key.startswith("webharbor.webvoyager@") for key in keys)
assert all(
    not gym.registry.task_metadata("webharbor.webvoyager", key.split("@", 1)[1]).others.get("mutating")
    for key in keys
)
print(f"webvoyager eval subset ok: 128 read-only tasks, env_key_sha256={key_hash}")
PY
```

```bash
# --- ENV-SERVER HOST ---
# WebVoyager eval needs the WebHarbor mirrors and a judge key. Start this once, in its own
# terminal, and paste the two exported CUA_LITE_* values into the eval shell.
uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh status
: "${OPENAI_API_KEY:?export OPENAI_API_KEY before starting WebVoyager env-server}"

PORT=30105
HOST_IP=$(hostname -I | awk '{print $1}')
SESSION_ID=webvoyager-gpt55-wvclean-$(date +%Y%m%d_%H%M%S)

printf 'export CUA_LITE_ENV_SERVER_URL=http://%s:%s\n' "$HOST_IP" "$PORT"
printf 'export CUA_LITE_ENV_SERVER_TOKEN=%s\n' "$SESSION_ID"

WEBHARBOR_WEBVOYAGER_INSTANCES=16 uv run --no-sync python scripts/serve_env.py \
  --port "$PORT" --env-ids webharbor.webvoyager --token "$SESSION_ID"
```

```bash
# --- EVAL HOST ---
# RUN names this campaign's artifacts. Reusing a slug makes rollout resume old samples and
# silently reports the previous campaign's numbers.
: "${RUN:?set RUN to a fresh slug for THIS campaign, e.g. 20260915-browser-gpt55-wvclean}"

# MUST be unset unless you intentionally attach to an already-running model server. With a URL
# in hand, rollout does not start serving.py; --model-path then changes local tokenizer/processor
# plumbing but generation still comes from whatever that server has loaded.
unset SGLANG_SERVER_URL

: "${CUA_LITE_ENV_SERVER_URL:?paste export line from env-server shell}"
: "${CUA_LITE_ENV_SERVER_TOKEN:?paste export line from env-server shell}"

EPOCH=epoch_2
DS=webgym_gpt5_5_nogoto_wvclean
CFG=devs/exps/train/browser/configs/qwen3_5
PULL=.ckpts/pulled
LOGS=.logs/rollout/Qwen_Qwen3.5-4B/webharbor.webvoyager
TASKS=devs/exps/train/browser/data/webvoyager.eval128.readonly.seed42.parquet

profiles() {
  printf '%s\n' browser.use.i4 browser.use.i1 browser.use.i4.reasoning browser.use.i1.reasoning
}

base_profiles() {
  profiles
}

MISSING=
while read -r P; do
  uv run hf download "ZHZisZZ/qwen3_5-4b.sft.$P.$DS" \
    --include "$EPOCH/*" \
    --local-dir "$PULL/sft.$P.$DS@$RUN" < /dev/null
  for f in config.json tokenizer_config.json preprocessor_config.json; do
    [ -e "$PULL/sft.$P.$DS@$RUN/$EPOCH/$f" ] || MISSING="$MISSING $P:$f"
  done
done <<< "$(profiles)"
if [ -n "$MISSING" ]; then
  echo "STOP -- do not run the score loop; missing under $EPOCH:$MISSING"
  return 1 2>/dev/null || exit 1
fi
echo "checkpoints OK"

# One auto-started Qwen3.5-4B server per score job. Pick the host budget first, then divide:
# with WEBHARBOR_WEBVOYAGER_INSTANCES=16, the stable default is one GPU at CONC=16. If GPUS has
# more than one card, raise WEBHARBOR_WEBVOYAGER_INSTANCES or lower CONC so GPUS x CONC does not
# exceed the env-server capacity.
GPUS=(${GPUS:-0})
CONC=${CONC:-16}
GPU_I=0

score() {  # $1 = config stem, $2 = model path (empty means base), $3 = log slug
  if [ "$GPU_I" -ge "${#GPUS[@]}" ]; then wait; GPU_I=0; fi
  local P="$1" MODEL_PATH="$2" SLUG="$3" GPU="${GPUS[$GPU_I]}"
  local MODEL_ARGS=()
  [ -n "$MODEL_PATH" ] && MODEL_ARGS=(--model-path "$MODEL_PATH")
  CUDA_VISIBLE_DEVICES="$GPU" uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3.5-4B "${MODEL_ARGS[@]}" \
    --env-id webharbor.webvoyager \
    --prompt-data "$TASKS" \
    --concurrency "$CONC" \
    --config-path "$CFG/$P.yaml" \
    --log-root "$LOGS/$SLUG" < /dev/null &
  GPU_I=$((GPU_I + 1))
}

while read -r P; do
  score "$P" "" "base.$P@$RUN"
done <<< "$(base_profiles)"

while read -r P; do
  score "$P" "$PULL/sft.$P.$DS@$RUN/$EPOCH" "sft.$P.$DS@$RUN.$EPOCH"
done <<< "$(profiles)"

wait
```

`--model-id` picks the Qwen3.5 adapter and action space; the weights, tokenizer and chat template
come from `--model-path` for SFT cells. The default `GPUS=0` serializes the eight runs, slow but
correct. If `GPUS` names several cards, `score` drains with `wait` before reusing the first card.

#### Record the scores

Collect all eight runs under `devs/exps/train/browser/logs/$RUN.md`: one file per campaign,
edited as runs land, not reconstructed later from memory.

```bash
# --- EVAL HOST, same shell as the Score block ---
# Reuses $RUN, $DS, $EPOCH, $LOGS, profiles(), and base_profiles().
show() {  # $1 = log slug
  uv run python -c "
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / 'summary.json'
if not p.exists():
    print(f'{sys.argv[2]:78s} MISSING')
    raise SystemExit
d = json.loads(p.read_text())
s = d['stats']
returns = [r for t in d['tasks'] for r in t['episode_returns']]
solved = sum(r == 1.0 for r in returns)
pf = (s.get('stop_reasons') or {}).get('parse_failure', 0)
over_all = sum(returns) / s['num_samples']
print(
    f\"{sys.argv[2]:70s} {s['mean_episode_return']:.4f} ({solved}/{s['num_valid']}) \"
    f\"parse_fail={pf} err={s['num_samples'] - s['num_valid']} over128={over_all:.4f}\"
)
" "$LOGS/$1" "$1"
}

while read -r P; do
  show "base.$P@$RUN"
done <<< "$(base_profiles)"

while read -r P; do
  show "sft.$P.$DS@$RUN.$EPOCH"
done <<< "$(profiles)"
```

Paste the numbers into a snapshot file, reusing the two tables' layout from **Results** below.
Front matter:

```markdown
# browser.use @ <run>

- **Checkpoints**: `ZHZisZZ/qwen3_5-4b.sft.<config>.webgym_gpt5_5_nogoto_wvclean`, `epoch_2`
- **Dataset**: `webgym_gpt5_5_nogoto_wvclean`, 2381 filtered `gpt5_5` rows after no-goto and
  exact WebVoyager task decontamination
- **Eval**: `devs/exps/train/browser/data/webvoyager.eval128.readonly.seed42.parquet`, 128
  read-only WebVoyager eval tasks; record the `env_key_sha256` printed by the export check
- **Host / GPUs**: `<host>` / `<gpu-list>`
- **Last updated**: `<date>`

## Results

<the two tables from the Results section of this file>
```

### Results

Mean episode return, (fully-solved / `num_valid`) in parentheses. Fill this from fresh
WebVoyager read-only passes. `over128` is the dense mean over the fixed prompt set and should be
reported beside the valid-only mean whenever `err` is non-zero.

| | `i4` | `i1` |
|---|---:|---:|
| **base** | TBD | TBD |
| **base + `<think>`** | TBD | TBD |
| **`gpt5_5`** | TBD | TBD |
| **`gpt5_5` + `<think>`** | TBD | TBD |
| **GRPO from `gpt5_5` + `<think>`** | — | TBD |

Every individual run - `MER (solved/num_valid) parse_failure err over128`, one column per pass:

| | | `a` | `b` | `c` |
|---|---|---:|---:|---:|
| **base** | `i4` | TBD | TBD | TBD |
|  | `i1` | TBD | TBD | TBD |
| **base+`<think>`** | `i4` | TBD | TBD | TBD |
|  | `i1` | TBD | TBD | TBD |
| **`gpt5_5`** | `i4` | TBD | TBD | TBD |
|  | `i1` | TBD | TBD | TBD |
| **`gpt5_5`+`<think>`** | `i4` | TBD | TBD | TBD |
|  | `i1` | TBD | TBD | TBD |
| **GRPO from `gpt5_5`+`<think>`** | `i4` | — | — | — |
|  | `i1` | TBD | TBD | TBD |

Reading the table:

- **The base row is reusable within this campaign.** It depends only on the eval prompt surface,
  not on the SFT dataset recipe.
- **Do not paste old browser/WebVoyager numbers into this table.** This campaign changes the
  training data contract to `webgym_gpt5_5_nogoto_wvclean`, so every cell needs a fresh score.
- **The denominator is part of the result.** `err` is `num_samples - num_valid`; keep it beside
  every mean instead of silently averaging failures as reward 0.
- **This is a fixed-subset result, not a full WebVoyager benchmark.** Re-run multiple seeds or
  the full split before treating a cell as generally best.
- **The GRPO row is the `i1` reasoning surface only.** It starts from the
  `gpt5_5` + `<think>` SFT checkpoint trained with `browser.use.i1.reasoning.yaml`; do not score
  that checkpoint under `i4`.

### RL

One GRPO run from the local **`gpt5_5` + `<think>` SFT checkpoint** trained with
[`browser.use.i1.reasoning.yaml`](/devs/exps/train/browser/configs/qwen3_5/browser.use.i1.reasoning.yaml).
Train online on the WebGym anchor manifest from
[`TASKS.md`](/devs/exps/train/browser/TASKS.md), and keep the in-training eval curve on the fixed
128-row WebVoyager read-only parquet from the SFT Eval block. This is a transfer run: WebGym
supplies online reward; WebVoyager supplies the held-out browser navigation score.

Read [docs/grpo.md](/docs/grpo.md) first for the Slime lifecycle and shared GRPO knobs. This block
only pins the browser-specific choices.

> **Two envs, one run.** Training rows are `webgym@...`; eval rows are
> `webharbor.webvoyager@...`. `ENV_ID=webgym` is still correct: it controls the run label, default
> preflight env, and cleanup scope. The eval dataset label will therefore be `webgym_eval`, even
> though the eval tasks are WebVoyager. The env-server must serve both envs:
> `--env-ids webgym webharbor.webvoyager`.

<details>
<summary>Data</summary>

```bash
# --- ONE-TIME DATA BUILD; skip generation for any file that already exists ---
# These parquet files are fixed experiment manifests under the repo, so Slime sees them at
# /workspaces/cua-lite/devs/exps/train/browser/data.
DATA=devs/exps/train/browser/data
CAND="$DATA/webgym.train.gpt55_anchor.nogoto.site.candidates.seed42.parquet"
TRAIN="$DATA/webgym.train.gpt55_anchor.nogoto.site.calibrated.head1024.g4.seed42.parquet"
EVAL="$DATA/webvoyager.eval128.readonly.seed42.parquet"
mkdir -p "$DATA"

if [ -e "$CAND" ]; then
  echo "keep existing fixed candidate manifest: $CAND"
else
  uv run python devs/exps/train/browser/utils/tasks.py build-candidates --out "$CAND"
fi
uv run python devs/exps/train/browser/utils/tasks.py verify-candidates --candidate "$CAND"

if [ ! -e "$TRAIN" ]; then
  echo "calibrated train manifest missing: $TRAIN"
  echo "run /devs/exps/train/browser/TASKS.md Step 2, or use the candidate fallback deliberately"
fi

if [ -e "$EVAL" ]; then
  echo "keep existing fixed eval manifest: $EVAL"
else
  uv run python -m lite.train.export.export_tasks \
    --env-id webharbor.webvoyager --split eval \
    --filter "lambda m: not m.others.get('mutating')" --sample 128 --seed 42 \
    -o "$EVAL"
fi

uv run python - "$CAND" "$TRAIN" "$EVAL" <<'PY'
import sys
import hashlib
from pathlib import Path

import pandas as pd

from lite.data.staging import coerce_meta

candidate = pd.read_parquet(sys.argv[1])
train_path = Path(sys.argv[2])
eval_ = pd.read_parquet(sys.argv[3])
train = pd.read_parquet(train_path) if train_path.exists() else candidate
train_label = "calibrated" if train_path.exists() else "candidate-fallback"
train_keys = [coerce_meta(row["metadata"])["env_key"] for _, row in train.iterrows()]
eval_keys = [coerce_meta(row["metadata"])["env_key"] for _, row in eval_.iterrows()]
train_hash = hashlib.sha256("\n".join(train_keys).encode()).hexdigest()
eval_hash = hashlib.sha256("\n".join(eval_keys).encode()).hexdigest()

assert len(candidate) == 2366
assert len(train) >= 500
assert len(eval_) == 128
assert all(key.startswith("webgym@") for key in train_keys)
assert all(key.startswith("webharbor.webvoyager@") for key in eval_keys)
print(
    f"browser RL data ok: {len(train)} WebGym {train_label} train tasks, "
    f"128 WebVoyager read-only eval tasks, train_sha256={train_hash} "
    f"eval_sha256={eval_hash}"
)
PY
```

</details>

```bash
# --- ENV-SERVER HOST ---
# WebGym and WebVoyager both need the judge key; WebVoyager also needs the WebHarbor mirrors.
uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh status
: "${OPENAI_API_KEY:?export OPENAI_API_KEY before starting the env-server}"

PORT=30106
HOST_IP=$(hostname -I | awk '{print $1}')
SESSION_ID=browser-rl-webgym-webvoyager-$(date +%Y%m%d_%H%M%S)

printf 'export CUA_LITE_ENV_SERVER_URL=http://%s:%s\n' "$HOST_IP" "$PORT"
printf 'export CUA_LITE_ENV_SERVER_TOKEN=%s\n' "$SESSION_ID"

WEBGYM_INSTANCES=16 WEBHARBOR_WEBVOYAGER_INSTANCES=16 uv run --no-sync python scripts/serve_env.py \
  --port "$PORT" --env-ids webgym webharbor.webvoyager --token "$SESSION_ID"
```

```bash
# --- Slime container ---
# Start from the gpt5_5 + <think> SFT checkpoint for the same browser.use.i1.reasoning surface.
W=/workspaces/cua-lite
P=browser.use.i1.reasoning
DS=webgym_gpt5_5_nogoto_wvclean
EPOCH=epoch_2
RLDS=webgym_anchor_nogoto_webvoyager128
CELL=grpo.$P.$RLDS.from_sft
PROMPT_DATA="$W/devs/exps/train/browser/data/webgym.train.gpt55_anchor.nogoto.site.calibrated.head1024.g4.seed42.parquet"
if [ ! -e "$PROMPT_DATA" ]; then
  PROMPT_DATA="$W/devs/exps/train/browser/data/webgym.train.gpt55_anchor.nogoto.site.candidates.seed42.parquet"
fi

if [ -z "${CKPT:-}" ]; then
  CKPT=$(ls -d "$W/.ckpts/qwen3_5-4b/sft.$P.$DS"/iter_* 2>/dev/null | sort -V | tail -1)
fi
if [ -z "$CKPT" ]; then
  PULL="$W/.ckpts/pulled"
  uv run hf download "ZHZisZZ/qwen3_5-4b.sft.$P.$DS" \
    --include "$EPOCH/*" \
    --local-dir "$PULL/sft.$P.$DS" < /dev/null
  CKPT="$PULL/sft.$P.$DS/$EPOCH"
fi

: "${CUA_LITE_ENV_SERVER_URL:?paste export line from env-server shell}"
: "${CUA_LITE_ENV_SERVER_TOKEN:?paste export line from env-server shell}"
[ -d "$CKPT" ] || { echo "MISSING CKPT=$CKPT"; exit 1; }
for f in config.json tokenizer_config.json preprocessor_config.json; do
  [ -e "$CKPT/$f" ] || { echo "MISSING $CKPT/$f"; exit 1; }
done

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_TRAIN_GPUS=8 TP_SIZE=4 MBS=1 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  HF_CKPT="$CKPT" \
  ENV_ID=webgym \
  PROMPT_DATA="$PROMPT_DATA" \
  EVAL_PROMPT_DATA="$W/devs/exps/train/browser/data/webvoyager.eval128.readonly.seed42.parquet" \
  ENV_CONCURRENCY=16 \
  ROLLOUT_BATCH_SIZE=16 \
  N_SAMPLES_PER_PROMPT=8 \
  NUM_STEPS_PER_ROLLOUT=8 \
  ROLLOUT_MAX_RESPONSE_LEN=2048 \
  CONFIG_PATH="$W/devs/exps/train/browser/configs/qwen3_5/$P.yaml" \
  SAVE=1 NO_SAVE_OPTIM=1 SAVE_INTERVAL=10 EVAL_INTERVAL=5 NUM_ROLLOUT=100 \
  SAVE_HF_DIR="$W/.ckpts/qwen3_5-4b/$CELL/iter_{rollout_id}" \
  SAVE_DIR="/root/checkpoints/qwen3_5-4b/$CELL/megatron" \
  WANDB_GROUP_SUFFIX=".$CELL" \
  bash "$W/scripts/train/run_grpo.sh" < /dev/null
```

- **`CONFIG_PATH` is mandatory.** The default compact WebGym config includes `goto` and the WebGym
  reset wrapper; this campaign must keep the no-goto WebVoyager-transfer surface.
- **`HF_CKPT` is mandatory for this RL run.** It must point at the
  `browser.use.i1.reasoning` SFT HF export. The block prefers an explicit `CKPT`, then the local
  train-host `iter_*`, then pulls `epoch_2` from
  `ZHZisZZ/qwen3_5-4b.sft.browser.use.i1.reasoning.webgym_gpt5_5_nogoto_wvclean`. Omitting the
  checkpoint entirely would start from base Qwen3.5-4B and answer a different question.
- **The WebGym training tier comes from the anchor manifest.** It is based on successful
  `gpt5_5` no-goto demonstrations, not `difficulty <= 3`. Difficulty is reported in
  [`TASKS.md`](/devs/exps/train/browser/TASKS.md) for curriculum analysis only.
- **Eval stays WebVoyager 128.** Do not replace `EVAL_PROMPT_DATA` with a WebGym eval parquet for
  this run, or the curve stops measuring transfer to the held-out WebVoyager subset. The intended
  subset is the `--sample 128 --seed 42` parquet above; record and compare its `env_key_sha256`
  across SFT eval and RL eval rather than relying on the filename alone.
- **Compare GRPO against its `gpt5_5` + `<think>` SFT parent.** Step 0 should line up with that
  checkpoint's WebVoyager score up to normal 128-task noise; a large gap usually means the wrong
  checkpoint or config was used.
- **Final score uses the same Eval block.** Reuse the Eval section's `score` / `show` helpers with
  `score browser.use.i1.reasoning "$GRPO_CKPT" "grpo.browser.use.i1.reasoning.webgym_anchor_nogoto_webvoyager128.from_sft@$RUN.$(basename "$GRPO_CKPT")"`
  for the saved GRPO `iter_*` being considered.
- **Rollout shape matches `desktop.use` RL.** `ROLLOUT_BATCH_SIZE=16` and
  `N_SAMPLES_PER_PROMPT=8` collect 128 trajectories per rollout; `NUM_STEPS_PER_ROLLOUT=8` splits
  those into 8 learner updates, so the derived GRPO global batch is 16. Keep this shape fixed when
  comparing WebGym RL against the desktop GRPO recipe.
- **Keep env concurrency conservative first.** `ENV_CONCURRENCY=16` matches the browser eval
  section's stable starting point, so the 128 trajectories run in several waves instead of
  pressuring all browser instances at once. Raise it only after the combined WebGym + WebVoyager
  env-server is healthy.
- **Pick checkpoints deliberately.** `SAVE_INTERVAL=10` avoids filling the repo-mounted volume with
  every fifth rollout. Score several saved `iter_*` checkpoints on the same 128-task WebVoyager
  parquet before copying one number into the GRPO row.
