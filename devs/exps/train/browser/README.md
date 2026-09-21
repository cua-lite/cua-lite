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
[`filter.py`](/devs/exps/train/browser/filter.py), not in the generic WebGym data cleaner.

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

uv run python devs/exps/train/browser/filter.py \
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

DS=webgym_gpt5_5_nogoto_wvclean
EPOCH=epoch_2
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

Mean episode return, (fully-solved / `num_valid`) in parentheses, then the pass count — the
cell format of `desktop/README.md`. `±` is **half the range** of the passes; with n=3 a range is
what there is, and an SD over three points pretends to more.

**Three passes per cell**, as in the desktop campaign — same convention, and here is the
browser-side evidence for it. Scoring is at `temperature: 0.0` (from the config yaml, not a
flag), but greedy is not deterministic: the `gpt5_5` + `<think>` / `i1` parent scored **70, 74,
76, 76, 76, 77, 78, 78, 79** of 128 under one judge and manifest — nine measurements, mean 76.0,
**sd 2.7, range 9** — and two passes over that checkpoint differ on 13 of 128 tasks, spread over
eleven sites rather than concentrated in one. An earlier note here put the spread at sd 0.55 over five passes and concluded one pass was
enough; that is superseded. Three passes bring the mean's standard error to ≈ 1.7 tasks. Use a
distinct tag per pass so passes are independent draws, not a resume of one another.

| | `i4` | `i1` |
|---|---:|---:|
| **base** | 0.0833 ±0.0156 (10.7/128) 3/3 | 0.0443 ±0.0039 (5.7/128) 3/3 |
| **base + `<think>`** | 0.1354 ±0.0391 (17.3/128) 3/3 | 0.1484 ±0.0195 (19.0/128) 3/3 |
| **`gpt5_5`** | 0.5052 ±0.0156 (64.7/128) 3/3 | 0.4869 ±0.0060 (62.0/127) 3/3 |
| **`gpt5_5` + `<think>`** | 0.5964 ±0.0079 (76.3/128) 3/3 | **0.6000** ±0.0078 (76.8/128) 5/5 |
| **GRPO from `gpt5_5` + `<think>`** | — | TBD |

Every individual run — `MER (solved) parse_failure`, one column per pass. The pod is named
because a pass here is a (time, host) pair, not just a time. **Every cell above is summarised
over `p1`-`p3`**; a `p4` in *italics* is a spare pass that an idle pod picked up, recorded here
but held out of the mean so that all cells share n=3 and their `±` are comparable.

| | | `p1` | `p2` | `p3` | `p4` |
|---|---|---:|---:|---:|---:|
| **base** | `i4` | 0.0781 (10) 1 *cc9* | 0.1016 (13) 1 *cc9* | 0.0703 (9) 1 *cc9* | — |
|  | `i1` | 0.0391 (5) 1 *cc9* | 0.0469 (6) 1 *cc9* | 0.0469 (6) 1 *cc9* | — |
| **base+`<think>`** | `i4` | 0.1250 (16) 2 *cc9d* | 0.1016 (13) 4 *cc9b* | 0.1797 (23) 2 *cc9b* | — |
|  | `i1` | 0.1328 (17) 2 *cc9d* | 0.1719 (22) 2 *cc9b* | 0.1406 (18) 1 *cc9b* | — |
| **`gpt5_5`** | `i4` | 0.4922 (63) 0 *cc9e* | 0.5000 (64) 0 *cc9c* | 0.5234 (67) 0 *cc9c* | *0.5312 (68) 0 cc9e* |
|  | `i1` | 0.4922 (63/128) 0 *cc9e* | 0.4803 (61/127) 0 *cc9c* | 0.4882 (62/127) 0 *cc9c* | *0.4882 (62/127) 0 cc9e* |
| **`gpt5_5`+`<think>`** | `i4` | 0.6016 (77) 0 *cc9b* | 0.5859 (75) 0 *cc9d* | 0.6016 (77) 0 *cc9d* | — |
|  | `i1` *(standalone)* | 0.5469 (70) 0 *cc9b* | 0.6172 (79) 0 *cc9d* | 0.5781 (74) 0 *cc9d* | *0.6016 (77) 0 cc9b* |
|  | `i1` *(GRPO `eval 0`)* | 78 · 76 · 76 · 78 · 76 — seeds 101/202/303/404/42 | | | |
| **GRPO from `gpt5_5`+`<think>`** | `i1` | TBD | TBD | TBD | — |

The `gpt5_5` + `<think>` / `i1` cell reports the GRPO runs' five `eval 0` points (78, 76, 76, 78,
76 — mean **76.8/128 = 0.600**) rather than its own three standalone passes: they are five draws
of exactly that checkpoint under this judge and manifest. The standalone row above it (70, 79, …,
mean 74.3 over the three tagged passes) agrees to 2.5 tasks, which is also the evidence that
slime's in-training eval and `scripts/rollout.py` carry no systematic offset — so the rest of the
column stays comparable.

Reading the table:

- **The base row is reusable within this campaign.** It depends only on the eval prompt surface,
  not on the SFT dataset recipe.
- **Do not paste old browser/WebVoyager numbers into this table.** This campaign changes the
  training data contract to `webgym_gpt5_5_nogoto_wvclean`, so every cell needs a fresh score.
- **`bbc_news.8` drops out of the `gpt5_5` / `i1` denominator, by design.** That cell reads
  `/127` in three of its four passes. WebVoyager calls the judge once, at the terminating step:
  every earlier step logs `reward=None`, then the final `/step` carries the screenshot to
  gpt-4.1, which returns `400 content_policy_violation` ("your input image may contain content
  that is not allowed by our content safety system") on this BBC News page. `client.py:335`
  raises `RemoteEnvError`, `_run_one` marks the trajectory invalid, and `num_valid` falls to 127.
  That is the right behaviour: a judge refusing to look at a page is not the model failing the
  task, so it must not score 0. `mean_episode_return` already averages over `num_valid`, so
  nothing else is needed — just do not read a `/127` cell as if one task were solved.
  Three details worth keeping: the refusal is frequent but not certain (one pass in four got a
  full 128); it is profile-skewed (`i1` hit it in three passes of four, `i4` in none of four, and
  the one pass that did measure the task under `i1` scored it **0**, so dropping it if anything
  flatters `i1`); and chasing a full 128 costs a ~20-minute pass at roughly one-in-four odds,
  which is why the cell is reported at `/127` instead.
- **`err > 0` otherwise means re-run that pass, not report it** (the desktop rule). A pass that
  came up short for a transient reason is not a measurement of this cell, and averaging it in
  imports whatever took the trajectory down. This supersedes the older `err <= 3` bracket rule:
  with three passes there is no reason to keep a short one.
- **A row on one pod, but passes across pods.** Each row's `i1` and `i4` run together on one pod
  so the two columns share an environment. Passes moved between pods, so a pass-to-pass
  difference mixes time and host; the `base` row is the exception, all three passes on cc9.
- **This is a fixed-subset result, not a full WebVoyager benchmark.** Re-run multiple seeds or
  the full split before treating a cell as generally best.
- **The GRPO row is the `i1` reasoning surface only.** It starts from the
  `gpt5_5` + `<think>` SFT checkpoint trained with `browser.use.i1.reasoning.yaml`; do not score
  that checkpoint under `i4`.

### RL

One GRPO run where **train and eval are the same env, split by task**: the 622 read-only
WebVoyager tasks are partitioned into the fixed 128-row eval manifest and its 494-row complement.
Read [docs/grpo.md](/docs/grpo.md) first for the Slime lifecycle and shared GRPO knobs; this block
pins only the browser-specific choices.

#### Why a task split rather than a site split

An earlier campaign held out four whole sites (165 tasks) and trained on the other eleven. It
measured +0.02 .. +0.07 at t=1, but per site, on every arm and checkpoint of both stages:

    booking  +0.21 .. +0.32        arxiv   ~0
    espn     -0.03 .. -0.15        github  ~0

One site carried all of it: with four held-out sites the effective sample size is nearer **4**
than 165, which is why only two of eight scored checkpoints reached 2σ. Splitting by task over
all 15 sites puts it back on the order of the task count, at the cost of a weaker claim —
transfer to unseen *tasks*, not to unseen *sites*.

**That campaign's `train457` overlaps this eval manifest on 88 of 128 tasks.** Its checkpoints
must never fill the Results table above — two thirds of those prompts were in their training set,
and the score would not show it.

<details>
<summary>Data</summary>

The universe is defined by the registry, not a file: a task is read-only when its metadata
carries no `mutating` flag. The train manifest is **exactly the complement** of the existing
128-row eval sample — no sampling, no seed, a pure function of (registry, eval manifest).

```bash
# --- ONE-TIME DATA BUILD; skip generation for any file that already exists ---
# Both parquets are fixed experiment manifests under the repo, so Slime sees them at
# /workspaces/cua-lite/devs/exps/train/browser/data.
DATA=devs/exps/train/browser/data
EVAL="$DATA/webvoyager.eval128.readonly.seed42.parquet"
TRAIN="$DATA/webvoyager.train494.readonly.parquet"
mkdir -p "$DATA"

if [ -e "$EVAL" ]; then
  echo "keep existing fixed eval manifest: $EVAL"
else
  uv run python -m lite.train.export.export_tasks \
    --env-id webharbor.webvoyager --split eval \
    --filter "lambda m: not m.others.get('mutating')" --sample 128 --seed 42 \
    -o "$EVAL"
fi

if [ -e "$TRAIN" ]; then
  echo "keep existing fixed train manifest: $TRAIN"
else
  # The universe, then everything the eval manifest does not claim. `export_tasks` has no
  # --exclude, so the complement is taken here; it is a set difference, not a sample.
  # A unique temp path and a chained `&&`: a fixed /tmp name can be unwritable or stale
  # (the pods set fs.protected_regular=2 and run as a different user), and without the
  # chain the second step would silently build the complement of whatever was left there.
  UNIV=$(mktemp -t wv_readonly_universe.XXXXXX.parquet) && \
  uv run python -m lite.train.export.export_tasks \
    --env-id webharbor.webvoyager --split eval \
    --filter "lambda m: not m.others.get('mutating')" \
    -o "$UNIV" && \
  uv run python - "$UNIV" "$EVAL" "$TRAIN" <<'PY' && rm -f "$UNIV"
import sys

import pyarrow.parquet as pq

from lite.data.staging import coerce_meta
from lite.utils.parquet import write_records_to_parquet

universe, eval_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
held = {
    coerce_meta(r["metadata"])["env_key"]
    for r in pq.read_table(eval_path).to_pylist()
}
rows = pq.read_table(universe).to_pylist()
records = [r for r in rows if coerce_meta(r["metadata"])["env_key"] not in held]
# the same writer export_tasks uses, so the schema matches the eval manifest
write_records_to_parquet(records, out)
print(f"wrote {len(records)} of {len(rows)} read-only tasks to {out}")
PY
fi
```

The two manifests must partition the universe exactly, and the train side must still be
read-only — a `mutating` task leaking in would train the policy on a site it can damage for
every later task on that mirror.

```bash
uv run python - "$TRAIN" "$EVAL" <<'PY'
import hashlib
import sys

import pandas as pd

import lite.gym as gym
from lite.data.staging import coerce_meta


def keys(path):
    df = pd.read_parquet(path)
    return [coerce_meta(row["metadata"])["env_key"] for _, row in df.iterrows()]


train_keys, eval_keys = keys(sys.argv[1]), keys(sys.argv[2])
assert len(train_keys) == 494 and len(eval_keys) == 128
assert not (set(train_keys) & set(eval_keys)), "train and eval share a task"
assert len(set(train_keys) | set(eval_keys)) == 622, "the two do not cover the universe"
for key in train_keys + eval_keys:
    assert key.startswith("webharbor.webvoyager@")
    task = key.split("@", 1)[1]
    assert not gym.registry.task_metadata("webharbor.webvoyager", task).others.get("mutating")
sites = {key.split("@", 1)[1].split(".")[0] for key in train_keys}
assert len(sites) == 15, sites
print("train env_key_sha256 =", hashlib.sha256("\n".join(train_keys).encode()).hexdigest())
print("eval  env_key_sha256 =", hashlib.sha256("\n".join(eval_keys).encode()).hexdigest())
PY
```

The committed manifests are pinned at

    train  494 tasks, 15 sites, 28-37 per site
           env_key_sha256 8ad1c3415bfc5dc7854ae64964d501787aec01c1c34a6b3da3fe277776125039
    eval   128 tasks, 15 sites
           env_key_sha256 72fe8df6c2056f21548a8f808ff09923986a4b0cb197dd2ac8a3cc7591742c7c

and two independent builds of the train manifest are byte-identical, because `export_tasks` emits
the universe in a stable order and the complement preserves it.

</details>

#### Parameters, and why

Most of this run is `run_grpo.sh`'s defaults. The previous campaign overrode the eval knobs; this
one stops overriding them, so "greedy eval, one rollout per task" is not a new choice here but
the shipped one. Three knobs deviate, plus the hardware and identity settings (`NUM_TRAIN_GPUS`,
`TP_SIZE`, `MBS`, `MODEL_ID`, `HF_CKPT`, the two manifests, `CONFIG_PATH`, the save paths):

| knob | value | default | why deviate |
|---|---|---|---|
| `ENV_CONCURRENCY` | `24` | 32 | what the site-holdout campaign ran at on 8xA100; the pods are known-good there |
| `ROLLOUT_MAX_RESPONSE_LEN` | `2048` | 512 | the `.reasoning` surface emits `<think>` before its calls |
| `CUA_LITE_MULTIMODAL_LAZY_EXPAND` | `1` | 0 | expands multimodal rollout data lazily; what the campaign ran |

The rest are defaults, written out so a later change to one cannot silently change this
experiment: `LR=1e-6`, `ROLLOUT_BATCH_SIZE=16` and `N_SAMPLES_PER_PROMPT=8` (128 trajectories per
rollout), `NUM_STEPS_PER_ROLLOUT=8` (global batch 16), `EVAL_TEMPERATURE=0`,
`N_SAMPLES_PER_EVAL_PROMPT=1`, `EVAL_INTERVAL=SAVE_INTERVAL=5`, `SKIP_EVAL_BEFORE_TRAIN=0`.

**`LR` was `2e-6` here, inherited from the site-holdout campaign; it is now the shipped `1e-6`.**
Matched rollout for rollout on this manifest, one 1e-6 seed beat *both* 2e-6 seeds and by a
widening margin — tasks gained out of 128 on the eval manifest, bracketed where a rollout lost
tasks to environment errors:

| rollout | 2e-6 seed A | 2e-6 seed B | 1e-6 |
|---|---|---|---|
| 5 | +3 .. +6 | +4 .. +8 | **+11 .. +14** |
| 10 | +6 .. +7 | +8 | **+17 .. +18** |
| 15 | — | — | **+21 .. +22** |

Two agreeing control seeds against one treatment seed, so the ordering is better evidenced than
the size of the gap. The failure mode matters more than the margin: both 2e-6 seeds drifted into
degenerate policies — one inflating its responses from 108 to 229 tokens, the other shrinking
them from 122 to 76 — and **both ended below their own step-0 baseline**. 2e-6 is not a slower
version of this run, it is a different regime.

Do not read the in-training `rollout/raw_reward` curve as a convergence curve here. With 494
tasks and 16 prompts per rollout one epoch is 31 rollouts, so no training task repeats for the
whole useful range of this run: that curve is a rolling *held-out* score at temperature 1.0, on
16 tasks, and its per-rollout spread (sd ~0.06) hides the trend for a dozen rollouts. Over the
first 18 rollouts of the 1e-6 run it rose about +0.11 with a slope of +0.006 per rollout (t=2.1),
while greedy eval on the fixed 128 rose +0.17. The fixed eval set is the instrument; the train
curve is not.

`NUM_ROLLOUT=60` is 960 task draws over 494 tasks, just under two epochs. It is an upper bound,
not a target — the campaign's best checkpoints were early (`iter_5` .. `iter_13` of 29) and its
last ones were damaged — so stopping early costs nothing.

**60 is a multiple of the interval, and that is load-bearing.** `should_run_periodic_action` fires
on `(rollout_id + 1) % interval == 0`, but `train.py` passes `args.num_rollout` for **save** and
not for eval, so the last rollout always saves whether or not the interval says to — and nothing
grants eval the same exemption. Equal intervals do not pair the two; a `NUM_ROLLOUT` divisible by
the interval does, because then the forced save lands on a step eval was going to run anyway. At 60/5 the 12 saves and 12 evals
land on the same steps (~104 GB), plus a 13th eval before training. Re-check this for any other
`NUM_ROLLOUT`.

Greedy eval is the default, not a choice made here; the campaign's `EVAL_TEMPERATURE=1` was the
override. It also matches the Results table above, which scores at `temperature: 0.0`. For scale,
the campaign scored the same checkpoints both ways: +0.12 .. +0.15 at t=0, +0.02 .. +0.07 at t=1.

```bash
# --- ENV-SERVER HOST ---
# WebVoyager needs the WebHarbor mirrors and a judge key. Start this once, in its own terminal,
# and paste the two exported CUA_LITE_* values into the training shell.
uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh status
: "${OPENAI_API_KEY:?export OPENAI_API_KEY before starting the env-server}"

PORT=30106
HOST_IP=$(hostname -I | awk '{print $1}')
SESSION_ID=browser-rl-wv-readonly-$(date +%Y%m%d_%H%M%S)

printf 'export CUA_LITE_ENV_SERVER_URL=http://%s:%s\n' "$HOST_IP" "$PORT"
printf 'export CUA_LITE_ENV_SERVER_TOKEN=%s\n' "$SESSION_ID"

# --warm-singleton: SINGLETON backends are lazy, so the container is only created on the first
# instance request -- but run_grpo's preflight wants /envs/<id> available=true before any rollout
# exists. Without it the two conditions wait on each other.
# The host derives the instance pool from the env config (server_kwargs.instances, RAM-based
# when 0) and only passes the resolved number into the container; this variable does not raise
# it. Check the pool the server actually chose before relying on ENV_CONCURRENCY fitting inside it.
uv run --no-sync python scripts/serve_env.py \
  --port "$PORT" --env-ids webharbor.webvoyager --warm-singleton --token "$SESSION_ID"
```

```bash
# --- Slime container ---
# Start from the gpt5_5 + <think> SFT checkpoint for the same browser.use.i1.reasoning surface.
W=/workspaces/cua-lite
P=browser.use.i1.reasoning
DS=webgym_gpt5_5_nogoto_wvclean
EPOCH=epoch_2
CELL=grpo.$P.wv_readonly_split.from_sft

CKPT=${CKPT:-$(ls -d "$W/.ckpts/qwen3_5-4b/sft.$P.$DS"/iter_* 2>/dev/null | sort -V | tail -1)}

: "${CUA_LITE_ENV_SERVER_URL:?paste export line from env-server shell}"
: "${CUA_LITE_ENV_SERVER_TOKEN:?paste export line from env-server shell}"
[ -d "$CKPT" ] || { echo "MISSING CKPT=$CKPT"; exit 1; }
for f in config.json tokenizer_config.json preprocessor_config.json; do
  [ -e "$CKPT/$f" ] || { echo "MISSING $CKPT/$f"; exit 1; }
done

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_TRAIN_GPUS=8 TP_SIZE=4 MBS=1 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  HF_CKPT="$CKPT" \
  ENV_ID=webharbor.webvoyager \
  PROMPT_DATA="$W/devs/exps/train/browser/data/webvoyager.train494.readonly.parquet" \
  EVAL_PROMPT_DATA="$W/devs/exps/train/browser/data/webvoyager.eval128.readonly.seed42.parquet" \
  ENV_CONCURRENCY=24 \
  ROLLOUT_BATCH_SIZE=16 \
  N_SAMPLES_PER_PROMPT=8 \
  NUM_STEPS_PER_ROLLOUT=8 \
  ROLLOUT_MAX_RESPONSE_LEN=2048 \
  ROLLOUT_TEMPERATURE=1.0 \
  LR=1e-6 \
  EVAL_TEMPERATURE=0 \
  N_SAMPLES_PER_EVAL_PROMPT=1 \
  CUA_LITE_MULTIMODAL_LAZY_EXPAND=1 \
  CONFIG_PATH="$W/devs/exps/train/browser/configs/qwen3_5/$P.yaml" \
  SKIP_EVAL_BEFORE_TRAIN=0 \
  SAVE=1 NO_SAVE_OPTIM=1 SAVE_INTERVAL=5 EVAL_INTERVAL=5 NUM_ROLLOUT=60 \
  SAVE_HF_DIR="$W/.ckpts/qwen3_5-4b/$CELL/iter_{rollout_id}" \
  SAVE_DIR="/root/checkpoints/qwen3_5-4b/$CELL/megatron" \
  WANDB_GROUP_SUFFIX=".$CELL" \
  bash "$W/scripts/train/run_grpo.sh" < /dev/null
```

- **`HF_CKPT` is mandatory.** It must point at the local
  `sft.browser.use.i1.reasoning.webgym_gpt5_5_nogoto_wvclean` HF export. Omitting it is silent:
  `run_grpo.sh:146-147` defaults `HF_CKPT` to `/root/models/$MODEL_ID` and downloads base
  Qwen3.5-4B, which answers a different question. If the train host has no local `iter_*`, pull
  the export first and point `CKPT` at it:
  `uv run hf download "ZHZisZZ/qwen3_5-4b.sft.$P.$DS" --include "epoch_2/*" --local-dir "$W/.ckpts/pulled/sft.$P.$DS"`.
  Step 0's eval should land on that parent's WebVoyager score within 128-task noise; a large gap
  usually means the wrong checkpoint or config.
- **`CONFIG_PATH` is mandatory.** Unset, `run_grpo.sh` derives
  `scripts/configs/qwen3_5/compact/webharbor.webvoyager.yaml`, which does not exist, and exits 1
  before Ray starts. Pointing it at another `compact/*.yaml` would be silent and would stop the
  run being comparable to its `gpt5_5` + `<think>` / `i1.reasoning` parent.

#### Scoring

The in-training curve selects; it does not report. Scoring a checkpoint with the data that chose
it is winner's curse — at SE ~0.03 over a dozen candidates, worth +0.04-0.05, the size of the
effect itself. Re-measure the selected checkpoints on fresh trajectories before any number goes
into the Results table, using the Eval block's own `score` / `show` helpers so the GRPO row is
measured exactly like the SFT rows above it:

Run it from inside the Eval block's shell, where `score` and its ambient `GPUS/CONC/TASKS/CFG/
LOGS` are defined, once per checkpoint being considered:

```bash
# relative, like the rest of the Eval block -- $W is an RL-block variable
CELL=grpo.browser.use.i1.reasoning.wv_readonly_split.from_sft
for CKPT_DIR in ".ckpts/qwen3_5-4b/$CELL"/iter_*; do
  score browser.use.i1.reasoning "$CKPT_DIR" "$CELL@$RUN.$(basename "$CKPT_DIR")"
done
```

Score several saved `iter_*` before copying one number across; the curve's argmax is not
automatically the checkpoint to report.

**Score each cell three times and report the mean.** A single pass carries sd ≈ 3 tasks (measured
below), which is the size of most of the differences the table is being asked to show — one pass
per cell cannot separate `i1` from `i4`. Three passes bring the mean's standard error to ≈ 1.7
tasks. Use a distinct tag per pass so the passes are independent draws rather than a resume:

```bash
for TAG in p1 p2 p3; do
  score browser.use.i1 "" "base.browser.use.i1@$TAG"
done
```

The one cell that does not follow this is `gpt5_5` + `<think>` / `i1`, which is the GRPO parent:
its five GRPO `eval 0` measurements (78, 76, 76, 78, 76 — see Seed replication) are five draws of
exactly that cell under the same judge and manifest, so the cell reports their mean, **76.8/128 =
0.600**. Its own standalone passes (70, 79, 74, and a 77 from a repeat of the first) average
74.3, within 2.5 tasks of those five — the evidence that the two scoring paths carry no
systematic offset.

#### Ship the checkpoints

The GRPO checkpoints travel through the Hub the same way the SFT ones do, one level deeper
because this cell has two axes instead of one: **repo** = the cell
(`ZHZisZZ/qwen3_5-4b.grpo.browser.use.i1.reasoning.wv_readonly_split.from_sft`), **subdir** =
`seed_<rollout_seed>/rollout_<N>/`, **tag** = the producing commit.

Two departures from the SFT block, both deliberate:

- **`rollout_<N>`, not `epoch_<k>`.** GRPO has no epoch here — 494 train tasks at
  `ROLLOUT_BATCH_SIZE=16` take 31 rollouts to see the set once — so the subdir counts rollouts.
  Slime writes `iter_<N>` 0-based, so `iter_29` is the checkpoint after **30** rollouts and
  uploads as `rollout_30`; the same off-by-one as `eval 29`. Read the number off disk, never
  predict it.
- **Only the reported rollout goes up.** Each seed saves at `SAVE_INTERVAL=5`, so a full curve is
  six checkpoints per seed and ~53 GB per seed, ~265 GB for five. The intermediate points are
  already recorded numerically in Seed replication below, which is what reproducing the curve
  actually needs; the weights that have to exist are the ones the Results table reports. Upload
  another rollout only when a specific checkpoint is being selected, and say which in the commit
  message.

```bash
# --- ANY POD HOLDING THE RUN ---  (commit first so the tag describes the weights)
# `uv run hf`, not bare `hf`. Needs a WRITE-scoped token (HF_TOKEN, or `uv run hf auth login`).
# The repo is public, so eval hosts need no auth.
COMMIT="$(git rev-parse --short HEAD)"
REPO="ZHZisZZ/qwen3_5-4b.grpo.browser.use.i1.reasoning.wv_readonly_split.from_sft"
CKPTS=.ckpts/qwen3_5-4b
ROLLOUT=30                       # what to publish; ITER is ROLLOUT-1 on disk
ITER=$((ROLLOUT - 1))

# "<run dir under $CKPTS>|<rollout_seed>" -- the seed is in the run name for the 2026-09
# campaign (rs<rollout_seed>s<seed>); older runs left it at run_grpo.sh's default 42/1234.
runs() {
  printf '%s\n' \
    "grpo.browser.use.i1.reasoning.grid494.lr1e6.rs101s5001.from_sft|101" \
    "grpo.browser.use.i1.reasoning.grid494.lr1e6.rs202s5002.from_sft|202" \
    "grpo.browser.use.i1.reasoning.grid494.lr1e6.rs303s5003.from_sft|303" \
    "grpo.browser.use.i1.reasoning.grid494.lr1e6.rs404s5004.from_sft|404" \
    "grpo.browser.use.i1.reasoning.wv_readonly_split.lr1e6.from_sft|42"
}

uv run hf repos create "$REPO" --repo-type model --exist-ok < /dev/null

while IFS='|' read -r RUN SEED; do
  D="$CKPTS/$RUN/iter_$ITER"
  [ -d "$D" ] || { echo "SKIP seed_$SEED: no $D on this pod"; continue; }
  # AutoProcessor needs all three later; a checkpoint missing one is not shippable.
  for f in config.json tokenizer_config.json preprocessor_config.json; do
    [ -e "$D/$f" ] || { echo "SKIP seed_$SEED: $D has no $f"; D=""; break; }
  done
  [ -n "$D" ] || continue
  uv run hf upload "$REPO" "$D" "seed_$SEED/rollout_$ROLLOUT" --repo-type model \
    --commit-message "$COMMIT: seed $SEED, rollout $ROLLOUT (from $RUN/iter_$ITER)" \
    < /dev/null || echo "FAILED seed_$SEED -- not tagging"
done <<< "$(runs)"

uv run hf repos tag delete "$REPO" "$COMMIT" --repo-type model --yes < /dev/null \
  || echo "note: no existing '$COMMIT' tag (expected on a first upload)"
uv run hf repos tag create "$REPO" "$COMMIT" --repo-type model -m "$COMMIT" < /dev/null \
  || echo "ERROR $REPO has NO '$COMMIT' tag; weights are on main -- re-tag by hand"
```

The five runs live on different pods (cc9, cc9b, cc9c, cc9d x2), so this runs once per pod and
the `SKIP ... no <dir> on this pod` lines are expected — each pod uploads the run it holds.

Two things that cost a re-run the first time:

- **Detach it.** `hf upload` outlives the ssh session only under `nohup setsid`; without it the
  session's SIGHUP kills the upload mid-flight. `hf repos create` has already returned by then,
  so the repo exists and looks like progress while nothing is being sent.
- **Verify against the Hub, not the console.** `hf upload` is single-commit and not resumable, so
  a repo shows nothing at all until the commit lands — an in-flight upload and a dead one look
  identical from the Hub side, and a `pgrep`-style check on the pod side can match its own
  pattern and report a phantom process. The pair that actually settles it is the uploader's own
  `UPLOADED`/`FAILED` log line plus `list_repo_files` afterwards.

As of `0d81276` the repo holds all five seeds at `rollout_30` (46 files; 9 per seed), tagged
`0d81276`. Uploading is fast despite ~8.7 GB per checkpoint — Xet dedupes against the SFT parent
already on the Hub, so only 4.0-5.3 GB per seed is new data.

#### Seed replication

Five runs of the block above, identical except for `ROLLOUT_SEED` / `SEED`, all starting from the
same `sft.browser.use.i1.reasoning.webgym_gpt5_5_nogoto_wvclean` export and scored by the same
`gpt-4.1` judge on the same 128-row eval manifest. Four ran to rollout 30 and stopped; seed
42/1234 is an earlier run of the same recipe that went to 35; only its rollouts 0-30 survive
in the eval record.

One seed is not a result here. At rollout 20 the seeds sat at +22/+18/+17/+4; at rollout 25 they
moved +3/+12/−4/−11 in the same hour under the same judge. Final gains are +22, +21, +11, +9
(mean **+15.75**) — a 13-task spread across runs that differ only in a seed.

**A single eval pass is worth about ±3 tasks, so read the mean.** The parent checkpoint was scored
nine times under the identical judge and manifest — five times as `eval 0` of a GRPO run, four
standalone — and returned **70, 74, 76, 76, 76, 77, 78, 78, 79** (mean 76.0, sd 2.7, range 9). Two
standalone passes differ on **13 of 128 tasks** (10 solved only in the second, 3 only in the
first), scattered over eleven sites rather than concentrated in one, so this is per-task
nondeterminism in the rollout — batching-dependent argmax under `EVAL_TEMPERATURE=0`, plus
live-mirror timing — not a bad run or a broken site. Two passes of the same cell therefore differ
by ~3 tasks typically and by 9 in the observed worst case.

**Eval** — tasks solved out of `num_valid`, greedy, `N_SAMPLES_PER_EVAL_PROMPT=1`. Slime logs these
as `eval N` with `N = rollouts - 1`, so `eval 29` is the score after 30 rollouts.

| seed | pod | 0 | 5 | 10 | 15 | 20 | 25 | 30 | gain |
|---|---|---|---|---|---|---|---|---|---|
| 101/5001 | cc9  | 78 | 85 | 88 | 99 | 96 | 85 | **100** | **+22** |
| 404/5004 | cc9d | 78 | 86 | 85 | 88 | 95 | 98 | **99**  | **+21** |
| 303/5003 | cc9c | 76 | 77 | 87 | 84 | 98 | 94 | **87**  | **+11** |
| 202/5002 | cc9b | 76 | 83 | 86 | 87 | 80 | 92 | **85**  | **+9**  |
| 42/1234  | cc9d | 76 | 87 | 93 | 97 | 94 | 99 | **94**  | **+18** |

All points of the four 30-rollout seeds are `num_valid = 128`. Seed 42/1234 is not: its
`num_valid` runs `128, 125, 127, 127, 128, 128, 127` — rollouts 27-29 were caught in a step-timeout
burst — so its rate is solved ÷ `num_valid`, not solved ÷ 128.

**Train** — `rollout/raw_reward`, one value per rollout, temperature 1.0. Not a convergence curve:
494 tasks at `ROLLOUT_BATCH_SIZE=16` means no train task repeats inside 31 rollouts, so this is a
rolling held-out score and single-rollout swings of ±0.28 are ordinary.

```python
# devs/exps/train/browser -- GRPO seed replication, gpt-4.1 judge
EVAL = {  # rollout -> (solved, num_valid)
  "101/5001": {0:(78,128), 5:(85,128), 10:(88,128), 15:(99,128), 20:(96,128), 25:(85,128), 30:(100,128)},
  "202/5002": {0:(76,128), 5:(83,128), 10:(86,128), 15:(87,128), 20:(80,128), 25:(92,128), 30:(85,128)},
  "303/5003": {0:(76,128), 5:(77,128), 10:(87,128), 15:(84,128), 20:(98,128), 25:(94,128), 30:(87,128)},
  "404/5004": {0:(78,128), 5:(86,128), 10:(85,128), 15:(88,128), 20:(95,128), 25:(98,128), 30:(99,128)},
  "42/1234":  {0:(76,128), 5:(87,125), 10:(93,127), 15:(97,127), 20:(94,128), 25:(99,128), 30:(94,127)},
}

TRAIN = {  # rollout/raw_reward, index = rollout
  "101/5001": [0.421875,0.4765625,0.5703125,0.578125,0.4609375,0.6484375,0.6484375,0.8671875,
               0.6796875,0.6796875,0.6015625,0.6875,0.6328125,0.7734375,0.8046875,0.6953125,
               0.8046875,0.7421875,0.671875,0.640625,0.6484375,0.7578125,0.734375,0.828125,
               0.6875,0.8359375,0.71875,0.8671875,0.7421875,0.7265625],
  "202/5002": [0.4609375,0.3203125,0.4375,0.5078125,0.7421875,0.65625,0.59375,0.640625,
               0.546875,0.6640625,0.6328125,0.6328125,0.765625,0.84375,0.75,0.5625,
               0.65625,0.8125,0.6171875,0.8046875,0.7734375,0.640625,0.625,0.6875,
               0.625,0.7578125,0.8046875,0.78125,0.671875,0.71875],
  "303/5003": [0.5,0.3046875,0.4765625,0.453125,0.640625,0.6953125,0.640625,0.7109375,
               0.703125,0.703125,0.6015625,0.6328125,0.6796875,0.6328125,0.6796875,0.671875,
               0.59375,0.8203125,0.7265625,0.8125,0.7421875,0.796875,0.671875,0.8125,
               0.6875,0.8125,0.65625,0.765625,0.875,0.5546875,0.75],
  "404/5004": [0.421875,0.578125,0.46875,0.6171875,0.5625,0.4765625,0.6953125,0.6640625,
               0.65625,0.796875,0.625,0.6015625,0.5859375,0.84375,0.6875,0.7265625,
               0.78125,0.6640625,0.84375,0.75,0.71875,0.6953125,0.6015625,0.6328125,
               0.7421875,0.859375,0.65625,0.890625,0.7109375,0.7421875,0.8359375],
  # NOT on the same denominator as the four above -- see the note below.
  "42/1234":  [0.7063,0.5827,0.6719,0.6457,0.6746,0.5935,0.7295,0.7778,0.6929,0.5781,
               0.6693,0.6719,0.6719,0.6032,0.7874,0.7344,0.7953,0.7869,0.6905,0.7344,
               0.8504,0.8976,0.7540,0.7698,0.6800,0.7795,0.7165,0.6720,0.7500,0.7500,
               0.7315,0.7589,0.7273,0.7647,0.7840],
}
```

**Seed 42/1234's train series was recorded rescaled by each rollout's valid count**, while the four
others are raw over a dummy-padded 128. Read it for shape, not height, and leave it out of any
aggregate over the train reward. Its rollouts 27-29 (0.6720, 0.7500, 0.7500) are the timeout burst.

To reproduce the two aggregate panels:

```python
import numpy as np
xs = [0, 5, 10, 15, 20, 25, 30]
rate = np.array([[100 * EVAL[s][x][0] / EVAL[s][x][1] for x in xs] for s in EVAL])
mean, sem = rate.mean(0), rate.std(0, ddof=1) / np.sqrt(len(rate))   # eval: mean +- SEM

four = [s for s in TRAIN if s != "42/1234"]
n = min(len(TRAIN[s]) for s in four)                                  # 30; two seeds ran a 31st
tr = np.array([TRAIN[s][:n] for s in four])
tmean, tsd = tr.mean(0), tr.std(0, ddof=1)                            # train: mean +- SD
```

SEM for eval and SD for train on purpose: the eval panel asks how precisely the mean is known, the
train panel asks how wide a single rollout's draw is — the spread that buries the trend in any one
run. With n=5, rliable's IQM would average three seeds and is noisier here than the mean, so the
mean is used and every run is drawn alongside it.
