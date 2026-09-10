#!/usr/bin/env bash
# webharbor.webvoyager eval — single-model invocation.
#
# Routes --log-root to <repo-root>/.exps/eval/webharbor.webvoyager/<commit-ts>_<commit>/<run_id>/<slug>/.
# Repo root is derived from this script's location (worktree-safe — a stale
# CUA_LITE_ROOT inherited from another worktree's shell would otherwise
# silently redirect output).
# Resumes if the same (commit, run_id, model) was run before; completed tasks skipped.
#
# $EVAL_RUN_ID is OPTIONAL — see devs/exps/eval/AGENTS.md "Campaign id contract".
# If unset, run.sh auto-resolves to the highest-numbered run_<N>[_<label>] under the
# current commit dir (resume-to-latest), or `run_0` if this is the first campaign.
#
# $EVAL_MODE selects the observation/action mode (config filename), default "default".
# Set to "som" to use the Set-of-Marks index-based config instead. The mode is
# appended to the artifact slug by default so default/som variants can share one
# EVAL_RUN_ID without resuming each other's tasks. Override with EVAL_CONFIG_ID.
#
# Two passes (see /docs/eval.md "WebVoyager"): a single shared WebHarbor container
# resets the whole suite once at boot, so for clean scoring this runs
#   1. READ  pass — non-mutating tasks, fully parallel ($CONCURRENCY), residue-immune.
#   2. WRITE pass — mutating tasks, serial (concurrency 1), on the same clean
#      post-read baseline.
# Both passes write the same $LOG_ROOT; their filters are disjoint and rollout.py
# skips completed tasks, so re-runs resume cleanly. For STRICT cross-model eval,
# restart the env-server between models (this single-model script does not).
#
# Env-server prereq (workflow default): export CUA_LITE_ENV_SERVER_URL +
# CUA_LITE_ENV_SERVER_TOKEN before invoking. Missing vars now fail fast;
# set EVAL_ALLOW_DIRECT=1 for an explicit direct-mode dev run. WebVoyager
# is container-only (one shared cua-lite/webharbor.webvoyager container per env-server). See
# /docs/envs.md#env-server and /lite/gym/envs/webharbor/webvoyager/README.md.
#
# Usage:
#   # auto-resume to latest run_<N> at this commit (most common):
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/webharbor.webvoyager/run.sh <model-id>
#   # or open a fresh campaign at this commit:
#   export EVAL_RUN_ID="run_1"        # bump past any existing run_0 / run_1 / ...
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/webharbor.webvoyager/run.sh <model-id>
#   # SoM mode:
#   EVAL_MODE=som ./devs/exps/eval/webharbor.webvoyager/run.sh <model-id>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/webharbor.webvoyager/run.sh Qwen/Qwen3-VL-8B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/webharbor.webvoyager/run.sh Qwen/Qwen3-VL-32B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/webharbor.webvoyager/run.sh Qwen/Qwen3.5-27B
#   ./devs/exps/eval/webharbor.webvoyager/run.sh gpt-5.5         # API model, no GPU
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
EVAL_ENV_ID="webharbor.webvoyager"
source "$ROOT/devs/exps/eval/utils/runtime_mode.sh"

MODE="${EVAL_MODE:-default}"
BASE_SLUG="${MODEL//\//_}"
CONFIG_ID="${EVAL_CONFIG_ID:-$MODE}"
if [[ -n "$CONFIG_ID" && ! "$CONFIG_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[run.sh] ERROR: EVAL_CONFIG_ID may contain only letters, numbers, '.', '_' and '-': $CONFIG_ID" >&2
  exit 1
fi
SLUG="$BASE_SLUG"
if [ -n "$CONFIG_ID" ]; then
  SLUG="${BASE_SLUG}__${CONFIG_ID}"
fi
ENV_ROOT="$ROOT/.exps/eval/webharbor.webvoyager"

# Pipeline-relevant paths: changes to these files are what advance the
# commit-id used for path keying. Pure docs (CHANGELOG, README, snapshots)
# and unrelated training/other-env code do NOT — campaigns reuse the
# existing commit dir if pipeline state hasn't moved since.
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/webharbor.webvoyager/run.sh
  lite/core
  lite/agents
  lite/agents/extensions/webharbor/webvoyager
  lite/gym/envs/webharbor/webvoyager
  lite/gym/utils
  lite/gym/__init__.py lite/gym/types.py lite/gym/registry.py
  lite/gym/factory.py lite/gym/services.py lite/gym/remote
  lite/gym/base.py lite/gym/wrappers.py
  scripts/serve_env.py devs/exps/eval/utils/runtime_mode.sh
  devs/exps/eval/utils/campaign_dir.sh
  lite/agents/factory.py lite/infer/serving.py lite/infer/rollout.py
  scripts/rollout.py
  scripts/configs/*/default/webharbor.webvoyager/*.yaml
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

# model-family → rollout config dir; $MODE picks default.yaml vs som.yaml.
# GPT currently has only default.yaml committed for this env.
case "$MODEL" in
  Qwen/Qwen3-VL-*-Instruct|Qwen/Qwen3-VL-*-Thinking) CFG=scripts/configs/qwen3_vl/default/webharbor.webvoyager/${MODE}.yaml ;;
  Qwen/Qwen3.5-*)           CFG=scripts/configs/qwen3_5/default/webharbor.webvoyager/${MODE}.yaml ;;
  gpt-*)                    CFG=scripts/configs/gpt/default/webharbor.webvoyager/default.yaml ;;
  *) echo "unknown model: $MODEL — add a case in $0" >&2; exit 1 ;;
esac
[ -f "$CFG" ] || { echo "[run.sh] ERROR: config not found: $CFG (EVAL_MODE=$MODE)" >&2; exit 1; }

CONCURRENCY="${EVAL_CONCURRENCY:-32}"

mkdir -p "$LOG_ROOT"
echo "[run.sh] $MODEL"
echo "         commit_dir=$(basename "$COMMIT_DIR")  run_id=$RUN_ID  mode=$MODE  GPUs=${CUDA_VISIBLE_DEVICES:-?}"
echo "         log_root=$LOG_ROOT"
echo "         config=$CFG"

# Common rollout args shared by both passes. step_timeout is carried by the
# config's env_kwargs (see _STEP_TIMEOUT in lite/gym/envs/webharbor/webvoyager/main.py),
# so only max_steps is overridden here.
COMMON=(
  --model-id "$MODEL"
  --env-id webharbor.webvoyager --splits eval
  --config-path "$CFG"
  --env-kwargs '{"max_steps": 30}'
  --log-root "$LOG_ROOT"
)

# Pass 1 — READ: non-mutating tasks, fully parallel, residue-immune.
echo "[run.sh] pass 1/2: READ (non-mutating, concurrency=$CONCURRENCY)"
HF_HUB_OFFLINE=1 uv run python scripts/rollout.py \
  "${COMMON[@]}" \
  --concurrency "$CONCURRENCY" \
  --filter "lambda m: not m.others.get('mutating')"

# Pass 2 — WRITE: mutating tasks, serial, on the clean post-read baseline.
echo "[run.sh] pass 2/2: WRITE (mutating, concurrency=1)"
HF_HUB_OFFLINE=1 exec uv run python scripts/rollout.py \
  "${COMMON[@]}" \
  --concurrency 1 \
  --filter "lambda m: m.others.get('mutating')"
