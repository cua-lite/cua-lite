#!/usr/bin/env bash
# browsergym.visualwebarena eval — single-model invocation.
#
# Routes --log-root to <repo-root>/.exps/eval/browsergym.visualwebarena/<commit-ts>_<commit>/<run_id>/<slug>/.
# Repo root is derived from this script's location (worktree-safe — a stale
# CUA_LITE_ROOT inherited from another worktree's shell would otherwise
# silently redirect output).
# Resumes if the same (commit, run_id, model) was run before; completed tasks skipped.
#
# $EVAL_RUN_ID is OPTIONAL — see devs/exps/eval/AGENTS.md "Run id contract".
# If unset, run.sh auto-resolves to the highest-numbered run_<N>[_<label>] under the
# current commit dir (resume-to-latest), or `run_0` if this is the first campaign.
#
# $EVAL_MODE selects the observation/action mode (config filename), default
# "goal_image" (screenshot + coord; the VWA goal image(s) are decoded once and
# re-shown every turn). Other modes: "mixed" (text + AXTree bid-mode + goal
# image(s) — the agent-as-annotators paper repro, recommended for weak open
# models; qwen + claude), "som" (set-of-marks; qwen families only).
#
# VisualWebArena (910 tasks = WebArena stack + Classifieds container) shares ONE
# mutable Docker backend, so for a faithful score this runs the strict
# read/write split (see /lite/gym/envs/browsergym/README.md
# "Strict read/write split"):
#   1. READ  pass — non-mutating tasks, fully parallel ($CONCURRENCY), residue-immune
#      on the clean baseline.
#   2. WRITE pass — mutating tasks, serial (concurrency 1), in TOPOLOGICAL
#      `depends_on` order. Unlike WebArena, VWA's `depends_on` is NOT monotonic in
#      task-id (e.g. task `32` depends on `visualwebarena.223`, registered LATER),
#      so a serial pass in natural task-id order would run a child before its parent
#      and let residue false-satisfy it. This script therefore topo-sorts the
#      mutating set first and drives the write pass from a generated prompt-data
#      parquet whose row order == the topological order.
#   3. RECONCILE — the two passes have disjoint specs, so each rewrites summary.json
#      over its own subset only. A final read-only pass rebuilds one combined
#      summary.json over all tasks that were supposed to run (non-mutating∪mutating,
#      after any EVAL_READ_FILTER/EVAL_WRITE_FILTER).
# All passes write the same $LOG_ROOT and rollout.py skips completed tasks, so
# re-runs resume cleanly. For a rigorous number, start a scoped singleton
# env-server and restart it between models so each run rebuilds the WA/VWA stack
# fresh. GitLab cold boot can take 5-15 min; use `--warm-singleton` before
# launching this runner:
#
#   env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
#     uv run python scripts/serve_env.py \
#       --host 127.0.0.1 --port 30100 \
#       --env-ids browsergym.visualwebarena \
#       --token <token> \
#       --warm-singleton
#   export CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30100
#   export CUA_LITE_ENV_SERVER_TOKEN=<token>
#
# A resume across a server restart needs the whole write pass re-run; the
# read/write split assumes ONE clean baseline for the pass.
#
# Env-server prereq (workflow default): export CUA_LITE_ENV_SERVER_URL +
# CUA_LITE_ENV_SERVER_TOKEN before invoking. Missing vars now fail fast;
# set EVAL_ALLOW_DIRECT=1 for an explicit direct-mode dev run. The
# env-server host must have run `install.sh visualwebarena` once (WebArena ~60 GB
# + the classifieds docker-compose); the Docker stack is auto-started on the first
# task. `VWA_CLASSIFIEDS_RESET_TOKEN` (auto-set by start.sh) is required for the 22
# classifieds `require_reset` tasks. LLM-judge tasks run by default and hard-fail
# at reset if OPENAI_API_KEY is unset on the env-server process. Set OPENAI_BASE_URL
# there too only for a custom endpoint. See /docs/envs.md#env-server.
#
# Usage:
#   # auto-resume to latest run_<N> at this commit (most common):
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/browsergym.visualwebarena/run.sh <model-id>
#   # or open a fresh campaign at this commit:
#   export EVAL_RUN_ID="run_1"        # bump past any existing run_0 / run_1 / ...
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/browsergym.visualwebarena/run.sh <model-id>
#   # text+AXTree+goal-image mode (recommended for weak open models):
#   EVAL_MODE=mixed ./devs/exps/eval/browsergym.visualwebarena/run.sh <model-id>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/browsergym.visualwebarena/run.sh Qwen/Qwen3-VL-8B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/browsergym.visualwebarena/run.sh Qwen/Qwen3-VL-32B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/browsergym.visualwebarena/run.sh Qwen/Qwen3.5-27B
#   ./devs/exps/eval/browsergym.visualwebarena/run.sh microsoft/Fara-7B          # goal_image only
#   ./devs/exps/eval/browsergym.visualwebarena/run.sh claude-opus-4-6            # API model, no GPU
#
# tp_size comes from the model's LOCAL_AGENTS entry (lite/agents/factory.py), NOT from the
# GPU count; serve_sglang.py derives dp_size = visible // tp_size (local HF models only),
# so the GPUs you expose set the REPLICA count. This script never passes --engine-kwargs,
# so tp is unchangeable here.

set -euo pipefail

MODEL="${1:?usage: CUDA_VISIBLE_DEVICES=<gpus> $0 <model-id>}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)"
[[ -n "$ROOT" && -d "$ROOT" ]] || { echo "$0: cannot resolve repo root from ${BASH_SOURCE[0]}" >&2; exit 1; }
cd "$ROOT"
EVAL_ENV_ID="browsergym.visualwebarena"
source "$ROOT/devs/exps/eval/utils/runtime_mode.sh"

ENV_ID="browsergym.visualwebarena"
MODE="${EVAL_MODE:-goal_image}"
ENABLE_THINKING_ENV_SET=0
if [ "${EVAL_ENABLE_THINKING+x}" ]; then
  ENABLE_THINKING_ENV_SET=1
fi
ENABLE_THINKING=0
case "${EVAL_ENABLE_THINKING:-false}" in
  1|true|TRUE|yes|YES|on|ON) ENABLE_THINKING=1 ;;
  0|false|FALSE|no|NO|off|OFF|"") ENABLE_THINKING=0 ;;
  *) echo "[run.sh] ERROR: EVAL_ENABLE_THINKING must be true/false, got: ${EVAL_ENABLE_THINKING}" >&2; exit 1 ;;
esac
# Thinking is a text/annotator concern — the pure-vision goal_image default has no
# thinking path. Mirror browsergym.webarena's text_only/som gate (here: mixed/som).
if [[ "$ENABLE_THINKING" -eq 1 && "$MODE" != "mixed" && "$MODE" != "som" ]]; then
  echo "[run.sh] ERROR: EVAL_ENABLE_THINKING=true is only supported with EVAL_MODE=mixed or som" >&2
  exit 1
fi
SUPPORTS_ENABLE_THINKING=0
case "$MODEL" in
  Qwen/Qwen3-VL-*-Thinking|Qwen/Qwen3.5-*) SUPPORTS_ENABLE_THINKING=1 ;;
esac
if [[ "$ENABLE_THINKING" -eq 1 && "$SUPPORTS_ENABLE_THINKING" -ne 1 ]]; then
  echo "[run.sh] ERROR: EVAL_ENABLE_THINKING=true is only supported for Qwen3-VL Thinking/Qwen3.5 local models, got: $MODEL" >&2
  exit 1
fi
BASE_SLUG="${MODEL//\//_}"
CONFIG_ID="${EVAL_CONFIG_ID:-}"
if [[ -z "$CONFIG_ID" && ( "$MODE" == "mixed" || "$MODE" == "som" ) && "$ENABLE_THINKING_ENV_SET" -eq 1 && "$SUPPORTS_ENABLE_THINKING" -eq 1 ]]; then
  if [ "$ENABLE_THINKING" -eq 1 ]; then
    CONFIG_ID="think_on"
  else
    CONFIG_ID="think_off"
  fi
fi
if [[ -n "$CONFIG_ID" && ! "$CONFIG_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[run.sh] ERROR: EVAL_CONFIG_ID may contain only letters, numbers, '.', '_' and '-': $CONFIG_ID" >&2
  exit 1
fi
SLUG="$BASE_SLUG"
if [ -n "$CONFIG_ID" ]; then
  SLUG="${BASE_SLUG}__${CONFIG_ID}"
fi
ENV_ROOT="$ROOT/.exps/eval/browsergym.visualwebarena"

# Pipeline-relevant paths: changes to these files are what advance the
# commit-id used for path keying. Pure docs (CHANGELOG, README, snapshots)
# and unrelated training/other-env code do NOT — campaigns reuse the
# existing commit dir if pipeline state hasn't moved since.
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/browsergym.visualwebarena/run.sh
  lite/core
  lite/agents
  lite/agents/extensions/browsergym
  lite/gym/envs/browsergym
  lite/gym/utils
  lite/gym/__init__.py lite/gym/types.py lite/gym/registry.py
  lite/gym/factory.py lite/gym/services.py lite/gym/remote
  lite/gym/base.py lite/gym/wrappers.py
  scripts/serve_env.py devs/exps/eval/utils/runtime_mode.sh
  devs/exps/eval/utils/campaign_dir.sh
  lite/agents/factory.py lite/infer/serving.py lite/infer/rollout.py
  scripts/rollout.py
  scripts/configs/*/default/browsergym.visualwebarena/*.yaml
)
shopt -u nullglob

# Pre-flight: pipeline files must be committed (clean working tree + index).
DIRTY=$(git status --porcelain -- "${PIPELINE_PATHS[@]}" 2>/dev/null)
if [ -n "$DIRTY" ]; then
  echo "[run.sh] ERROR: pipeline files have uncommitted changes — commit first (path key would be ambiguous):" >&2
  echo "$DIRTY" | sed 's/^/  /' >&2
  exit 1
fi

# Resolve $COMMIT_DIR. Prefer reusing the latest existing campaign dir whose
# commit's pipeline state matches HEAD's (so doc-only commits between campaigns
# don't fragment paths). Fall through to a fresh HEAD-keyed dir otherwise.
source "$ROOT/devs/exps/eval/utils/campaign_dir.sh"
resolve_eval_commit_dir

# Resolve RUN_ID:
#   - If $EVAL_RUN_ID is set, use it verbatim (user-controlled).
#   - Else auto-resume to highest-numbered run_<N>[_<label>] under $COMMIT_DIR;
#     fall back to "run_0" when the commit dir is empty / missing.
if [ -n "${EVAL_RUN_ID:-}" ]; then
  RUN_ID="$EVAL_RUN_ID"
else
  if [ -d "$COMMIT_DIR" ]; then
    RUN_ID=$(ls -1 "$COMMIT_DIR" 2>/dev/null | grep -E '^run_[0-9]+(_|$)' | sort -t_ -k2,2n | tail -1)
  fi
  RUN_ID="${RUN_ID:-run_0}"
  echo "[run.sh] EVAL_RUN_ID unset → auto-resolved run_id=$RUN_ID (resume-to-latest at this commit)" >&2
fi
LOG_ROOT="$COMMIT_DIR/$RUN_ID/$SLUG"

# Safety net: warn if starting a fresh run_id while other campaigns exist for this commit.
if [ ! -d "$COMMIT_DIR/$RUN_ID" ] && [ -d "$COMMIT_DIR" ]; then
  EXISTING=$(ls -1 "$COMMIT_DIR" 2>/dev/null | grep -vx "$RUN_ID" || true)
  if [ -n "$EXISTING" ]; then
    echo "[run.sh] WARNING: starting fresh at run_id=$RUN_ID, but other campaigns exist at this commit:" >&2
    echo "$EXISTING" | sed 's/^/  - /' >&2
    echo "  if you meant to RESUME, Ctrl-C now and re-export EVAL_RUN_ID to one of the above (or unset to auto-resume)." >&2
    sleep 5
  fi
fi

# model-family → rollout config dir; $MODE picks goal_image.yaml / mixed.yaml / som.yaml.
# NOTE: only fara/qwen3_vl/qwen3_5 have VWA configs. GPT never had one. Claude's
# were removed: VWA goal images reach the model only through the
# `visualwebarena.goal_image` agent, which composes onto the adapter loop, and
# Claude runs its own bespoke `sample()` with no adapter — so a Claude VWA row
# silently drops the goal image and scores a benchmark it cannot see.
case "$MODEL" in
  Qwen/Qwen3-VL-*-Instruct|Qwen/Qwen3-VL-*-Thinking) CFG=scripts/configs/qwen3_vl/default/browsergym.visualwebarena/${MODE}.yaml ;;
  Qwen/Qwen3.5-*)                                      CFG=scripts/configs/qwen3_5/default/browsergym.visualwebarena/${MODE}.yaml ;;
  microsoft/Fara-7B)                                   CFG=scripts/configs/fara/default/browsergym.visualwebarena/${MODE}.yaml ;;
  claude-*)                                            echo "[run.sh] ERROR: Claude is not supported for browsergym.visualwebarena: its bespoke loop reads only obs.image, so the VWA goal image in metadata['goal_images_b64'] never reaches the model. Supporting it needs a Claude-side goal-image agent, not a config (EVAL_MODE=$MODE)" >&2; exit 1 ;;
  gpt-*)                                               echo "[run.sh] ERROR: GPT is not supported for browsergym.visualwebarena: no committed scripts/configs/gpt/default/browsergym.visualwebarena/*.yaml config exists (EVAL_MODE=$MODE)" >&2; exit 1 ;;
  *) echo "unknown model: $MODEL — add a case in $0" >&2; exit 1 ;;
esac
[ -f "$CFG" ] || { echo "[run.sh] ERROR: config not found: $CFG (EVAL_MODE=$MODE)" >&2; exit 1; }

CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EXTRA_ROLLOUT_ARGS=()
if [[ "$ENABLE_THINKING" -eq 1 && "$SUPPORTS_ENABLE_THINKING" -eq 1 ]]; then
  EXTRA_ROLLOUT_ARGS+=(
    --agent-kwargs '{"enable_thinking": true, "sampling_kwargs": {"temperature": 0.6, "top_p": 0.9,"max_new_tokens": 4096}}'
  )
fi

# Optional extra filter clauses ANDed onto the read/write splits (e.g. skip a
# subset on a partially-installed host). Default: no-op (`True`). The read filter
# is passed to rollout.py --filter directly; the write filter is applied while
# generating the topological write order (the write pass uses --prompt-data, which
# rejects --filter).
READ_EXTRA="${EVAL_READ_FILTER:-True}"
WRITE_EXTRA="${EVAL_WRITE_FILTER:-True}"

mkdir -p "$LOG_ROOT"
echo "[run.sh] $MODEL"
echo "         commit_dir=$(basename "$COMMIT_DIR")  run_id=$RUN_ID  mode=$MODE  GPUs=${CUDA_VISIBLE_DEVICES:-?}"
if [ -n "$CONFIG_ID" ]; then
  echo "         config_id=$CONFIG_ID"
fi
echo "         log_root=$LOG_ROOT"
echo "         config=$CFG"
echo "         enable_thinking=$ENABLE_THINKING"
if [ "${#EXTRA_ROLLOUT_ARGS[@]}" -gt 0 ]; then
  echo "         extra_args=${EXTRA_ROLLOUT_ARGS[*]}"
fi

# Common rollout args shared by both passes. max_steps is carried by the config's
# env_kwargs, so nothing is overridden here.
COMMON=(
  --model-id "$MODEL"
  --env-id "$ENV_ID"
  --config-path "$CFG"
  --log-root "$LOG_ROOT"
  "${EXTRA_ROLLOUT_ARGS[@]}"
)

# Pass 1 — READ: non-mutating tasks, fully parallel, residue-immune.
echo "[run.sh] pass 1/3: READ (non-mutating, concurrency=$CONCURRENCY)"
HF_HUB_OFFLINE=1 uv run python scripts/rollout.py \
  "${COMMON[@]}" \
  --splits eval \
  --concurrency "$CONCURRENCY" \
  --filter "lambda m: (not m.others.get('mutating')) and ($READ_EXTRA)"

# Pass 2 — WRITE: mutating tasks, serial, in TOPOLOGICAL depends_on order.
# VWA's depends_on is non-monotonic in task-id, so generate a prompt-data parquet
# whose row order is a topological sort of the mutating set; --prompt-data preserves
# first-seen row order, so concurrency 1 then dispatches writers parent-before-child.
WRITE_ORDER_PARQUET="$LOG_ROOT/.write_order.parquet"
echo "[run.sh] generating topological write order → $WRITE_ORDER_PARQUET"
VWA_ENV_ID="$ENV_ID" VWA_WRITE_FILTER="$WRITE_EXTRA" VWA_OUT_PARQUET="$WRITE_ORDER_PARQUET" \
  uv run --no-sync python - <<'PY'
import os
import pandas as pd
import lite.gym as gym
from lite.gym.registry import ensure_registered
from lite.core.utils.filters import parse_filter

env_id = os.environ["VWA_ENV_ID"]
ensure_registered(env_id)
splits = gym.registry.task_ids(env_id)
eval_ids = list(dict.fromkeys(splits.get("eval", [])))
md = {t: gym.registry.task_metadata(env_id, t) for t in eval_ids}
# VWA_WRITE_FILTER is a boolean CLAUSE (e.g. "True"), matching the read pass's
# ANDed --filter clause — wrap it into a lambda before parsing.
keep_write = parse_filter(f"lambda m: ({os.environ.get('VWA_WRITE_FILTER') or 'True'})")

write_ids = [t for t in eval_ids if md[t].others.get("mutating") and keep_write(md[t])]
wset = set(write_ids)
# depends_on values are in the `visualwebarena.<n>` (bgym_task_id) namespace;
# registry task_ids are the bare `<n>`. Map back before building edges.
bg2tid = {md[t].others["bgym_task_id"]: t for t in eval_ids}


def _key(x):
    # numeric task-id order for deterministic tie-breaks; non-numeric last.
    try:
        return (0, int(x))
    except ValueError:
        return (1, x)


adj = {t: set() for t in write_ids}
indeg = {t: 0 for t in write_ids}
for t in write_ids:
    for d in dict.fromkeys(md[t].others.get("depends_on") or []):
        p = bg2tid.get(d)
        if p in wset and p != t and t not in adj[p]:
            adj[p].add(t)
            indeg[t] += 1

# Kahn's algorithm, ready-queue sorted by task-id for a stable topological order.
ready = sorted([t for t in write_ids if indeg[t] == 0], key=_key)
order = []
while ready:
    t = ready.pop(0)
    order.append(t)
    newly = []
    for c in adj[t]:
        indeg[c] -= 1
        if indeg[c] == 0:
            newly.append(c)
    if newly:
        ready = sorted(ready + newly, key=_key)
# Cycle guard (curated DAG shouldn't cycle) — append any leftover deterministically.
if len(order) < len(write_ids):
    seen = set(order)
    order += sorted([t for t in write_ids if t not in seen], key=_key)

rows = [{"metadata": {"env_key": f"{env_id}@{t}", "split": "eval"}} for t in order]
pd.DataFrame(rows).to_parquet(os.environ["VWA_OUT_PARQUET"])
print(f"[write-order] {len(order)} mutating tasks (topological) → {os.environ['VWA_OUT_PARQUET']}")
PY

echo "[run.sh] pass 2/3: WRITE (mutating, topological, concurrency=1)"
HF_HUB_OFFLINE=1 uv run python scripts/rollout.py \
  "${COMMON[@]}" \
  --concurrency 1 \
  --prompt-data "$WRITE_ORDER_PARQUET"

# Pass 3 — RECONCILE: rebuild ONE combined summary.json over all tasks that were
# supposed to run. The two passes above each rewrite summary.json scoped to their
# own (disjoint) specs, so on its own the file reflects only the write subset.
echo "[run.sh] pass 3/3: RECONCILE combined summary over all tasks"
VWA_ENV_ID="$ENV_ID" VWA_LOG_ROOT="$LOG_ROOT" VWA_MODEL="$MODEL" \
VWA_READ_FILTER="$READ_EXTRA" VWA_WRITE_FILTER="$WRITE_EXTRA" \
  uv run --no-sync python - <<'PY'
import os
from pathlib import Path

import lite.gym as gym
from lite.gym.registry import ensure_registered
from lite.core.utils.filters import parse_filter
from lite.infer.rollout import TaskSpec, print_results, rebuild_results, save_summary

env_id = os.environ["VWA_ENV_ID"]
log_root = Path(os.environ["VWA_LOG_ROOT"])
ensure_registered(env_id)
splits = gym.registry.task_ids(env_id)
eval_ids = list(dict.fromkeys(splits.get("eval", [])))
# VWA_READ_FILTER / VWA_WRITE_FILTER are boolean CLAUSES (e.g. "True") — the same
# clauses ANDed onto the read/write splits — so wrap each into a lambda.
keep_read = parse_filter(f"lambda m: ({os.environ.get('VWA_READ_FILTER') or 'True'})")
keep_write = parse_filter(f"lambda m: ({os.environ.get('VWA_WRITE_FILTER') or 'True'})")


def ran(t):
    m = gym.registry.task_metadata(env_id, t)
    if m.others.get("mutating"):
        return keep_write(m)
    return keep_read(m)


specs = [TaskSpec(t, env_id, "eval") for t in eval_ids if ran(t)]
results = rebuild_results(log_root, specs, 1)
stats = print_results(results, specs, group_size=1)
save_summary(
    log_root / "summary.json",
    results=results, stats=stats, specs=specs,
    model=os.environ["VWA_MODEL"], env_id=env_id, splits=["eval"], group_size=1,
)
print(f"[reconcile] combined summary over {len(specs)} tasks → {log_root / 'summary.json'}")
PY
