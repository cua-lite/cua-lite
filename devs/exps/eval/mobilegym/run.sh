#!/usr/bin/env bash
# mobilegym eval — single-model invocation.
#
# Routes --log-root to <repo-root>/.exps/eval/mobilegym/<commit-ts>_<commit>/<run_id>/<slug>/.
# Repo root is derived from this script's location (worktree-safe — a stale
# CUA_LITE_ROOT inherited from another worktree's shell would otherwise
# silently redirect output).
# Resumes if the same (commit, run_id, model) was run before; completed tasks skipped.
#
# $EVAL_RUN_ID is OPTIONAL — see devs/exps/eval/AGENTS.md "Run id contract".
# If unset, run.sh auto-resolves to the highest-numbered run_<N>[_<label>] under the
# current commit dir (resume-to-latest), or `run_0` if this is the first campaign.
#
# Env-server prereq (workflow default): export CUA_LITE_ENV_SERVER_URL +
# CUA_LITE_ENV_SERVER_TOKEN before invoking. Missing vars now fail fast;
# set EVAL_ALLOW_DIRECT=1 for an explicit direct-mode dev run. See
# /docs/envs.md#env-server.
#
# Usage:
#   # auto-resume to latest run_<N> at this commit (most common):
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/mobilegym/run.sh <model-id>
#   # or open a fresh campaign at this commit:
#   export EVAL_RUN_ID="run_1"        # bump past any existing run_0 / run_1 / ...
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/mobilegym/run.sh <model-id>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/mobilegym/run.sh Qwen/Qwen3-VL-8B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/mobilegym/run.sh Qwen/Qwen3-VL-32B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/mobilegym/run.sh Qwen/Qwen3.5-27B
#   ./devs/exps/eval/mobilegym/run.sh gpt-5.5         # API model, no GPU
#
# tp_size comes from the model's LOCAL_AGENTS entry (lite/agents/factory.py), NOT from the
# GPU count; serve_sglang.py derives dp_size = visible // tp_size (local HF models only),
# so the GPUs you expose set the REPLICA count. This script never passes --engine-kwargs,
# so tp is unchangeable here.
# Pre-req: cua-lite/mobilegym docker image built via
#   uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh
#
# Env shape (see lite/gym/envs/mobilegym/README.md):
#   - eval split = 256 parameterized tasks (seed=42, deterministic) across 28
#     simulated apps; reward = progress_rate in [0.0, 1.0] from the state-diff
#     judge at episode end, so mean_episode_return is NOT a plain success rate.
#   - One shared cua-lite/mobilegym container per env-server (SINGLETON backend)
#     holds a Chromium pool (contexts_per_browser=8, max_browsers RAM-derived,
#     floor 4 / ceiling 32). Pool saturation → HTTP 503 → CapacityExhausted on
#     the host, which the rollout retries — safe at the default concurrency.
#   - --env-kwargs step_timeout=180s overrides the framework default 120s
#     (lite/gym/registry.py `make()`; the mobilegym spec doesn't override).
#     Steps are in-container Playwright actions — fast when idle, but under
#     host contention (load ≫ NCPU) they can blow the default; align with
#     androidworld/osworld's 180s.
set -euo pipefail

MODEL="${1:?usage: CUDA_VISIBLE_DEVICES=<gpus> $0 <model-id>}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)"
[[ -n "$ROOT" && -d "$ROOT" ]] || { echo "$0: cannot resolve repo root from ${BASH_SOURCE[0]}" >&2; exit 1; }
cd "$ROOT"
EVAL_ENV_ID="mobilegym"
source "$ROOT/devs/exps/eval/utils/runtime_mode.sh"

SLUG="${MODEL//\//_}"
ENV_ROOT="$ROOT/.exps/eval/mobilegym"

# Pipeline-relevant paths: changes to these files are what advance the
# commit-id used for path keying. Pure docs (CHANGELOG, README, snapshots)
# and unrelated training/other-env code do NOT — campaigns reuse the
# existing commit dir if pipeline state hasn't moved since.
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/mobilegym/run.sh
  lite/core
  lite/agents
  lite/gym/envs/mobilegym
  lite/gym/utils
  lite/gym/__init__.py lite/gym/types.py lite/gym/registry.py
  lite/gym/factory.py lite/gym/services.py lite/gym/remote
  lite/gym/base.py lite/gym/wrappers.py
  scripts/serve_env.py devs/exps/eval/utils/runtime_mode.sh
  devs/exps/eval/utils/campaign_dir.sh
  lite/agents/factory.py lite/infer/serving.py lite/infer/rollout.py
  scripts/rollout.py
  scripts/configs/*/default/mobilegym.yaml
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
resolve_eval_commit_dir --allow-eval-commit-dir

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

# model-family → rollout config
case "$MODEL" in
  Qwen/Qwen3-VL-*-Instruct|Qwen/Qwen3-VL-*-Thinking) CFG=scripts/configs/qwen3_vl/default/mobilegym.yaml ;;
  Qwen/Qwen3.5-*)                  CFG=scripts/configs/qwen3_5/default/mobilegym.yaml ;;
  ByteDance-Seed/UI-TARS-7B-DPO)   CFG=scripts/configs/ui_tars/default/mobilegym.yaml ;;
  ByteDance-Seed/UI-TARS-1.5-7B)   CFG=scripts/configs/ui_tars_15_v1/default/mobilegym.yaml ;;
  Tongyi-MAI/MAI-UI-*)             CFG=scripts/configs/mai_ui/default/mobilegym.yaml ;;
  stepfun-ai/GELab-Zero-*)         CFG=scripts/configs/step_gui/default/mobilegym.yaml ;;
  gpt-*)                           CFG=scripts/configs/gpt/default/mobilegym.yaml ;;
  claude-*)                        CFG=scripts/configs/claude/default/mobilegym.yaml ;;
  *) echo "unknown model: $MODEL — add a case in $0" >&2; exit 1 ;;
esac

CONCURRENCY="${EVAL_CONCURRENCY:-16}"

mkdir -p "$LOG_ROOT"
echo "[run.sh] $MODEL"
echo "         commit_dir=$(basename "$COMMIT_DIR")  run_id=$RUN_ID  GPUs=${CUDA_VISIBLE_DEVICES:-?}"
echo "         log_root=$LOG_ROOT"
echo "         config=$CFG"

HF_HUB_OFFLINE=1 exec uv run python scripts/rollout.py \
  --model-id "$MODEL" \
  --env-id mobilegym --splits eval \
  --concurrency "$CONCURRENCY" \
  --env-kwargs '{"step_timeout": 180}' \
  --config-path "$CFG" \
  --log-root "$LOG_ROOT"
