#!/usr/bin/env bash
# WAA standard 138-task eval; select the model family default YAML.
#
# Usage (from the repo root, after committing pipeline/config changes):
#   CUDA_VISIBLE_DEVICES=0 ./devs/exps/eval/waa/run.sh <model-id> <config-path>
#
# Requires CUA_LITE_ENV_SERVER_URL + CUA_LITE_ENV_SERVER_TOKEN. The env-server
# host must also set CUA_LITE_DOCKER_CREATE_CONCURRENCY=10. EVAL_ALLOW_DIRECT=1
# explicitly selects a local dev run. Cache model weights before launching.
# EVAL_RUN_ID=run_<N>[_<label>] starts a fresh campaign; unset resumes latest.
# EVAL_MODEL_PATH selects local weights; EVAL_ENGINE_KWARGS passes serving JSON
# unchanged. Otherwise serving uses the registered model defaults. API models need
# no GPUs. EVAL_CONCURRENCY defaults to 30; the coordinator must keep the total
# across simultaneous models <= 60. This entrypoint filters to the standard 138 eval tasks.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <model-id> <config-path>" >&2
  exit 1
fi
MODEL="$1"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
CFG="$(realpath -- "$2")"
if [[ "$CFG" != "$ROOT"/scripts/configs/*.yaml ]] || [ ! -f "$CFG" ]; then
  echo "[run.sh] ERROR: config must be an existing YAML under scripts/configs/" >&2
  exit 1
fi
CFG="${CFG#"$ROOT"/}"
CONFIG_ID="${CFG#scripts/configs/}"
CONFIG_ID="${CONFIG_ID%.yaml}"
CONFIG_ID="${CONFIG_ID//\//__}"
CONCURRENCY="${EVAL_CONCURRENCY:-30}"
if [[ ! "$CONCURRENCY" =~ ^[1-9][0-9]?$ ]] || (( CONCURRENCY > 60 )); then
  echo "[run.sh] ERROR: EVAL_CONCURRENCY must be an integer from 1 to 60" >&2
  exit 1
fi
if [ -n "${EVAL_RUN_ID:-}" ] && [[ ! "$EVAL_RUN_ID" =~ ^run_[0-9]+(_[A-Za-z0-9_.-]+)?$ ]]; then
  echo "[run.sh] ERROR: EVAL_RUN_ID must be run_<N>[_<label>]" >&2
  exit 1
fi

SLUG="${MODEL//\//_}__${CONFIG_ID}"
ENV_ROOT="$ROOT/.exps/eval/waa"
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/waa/run.sh
  devs/exps/eval/utils/runtime_mode.sh devs/exps/eval/utils/campaign_dir.sh
  lite/core lite/agents lite/infer
  lite/gym/envs/waa lite/gym/sandbox lite/gym/utils lite/gym/remote
  lite/gym/__init__.py lite/gym/types.py lite/gym/registry.py
  lite/gym/factory.py lite/gym/services.py lite/gym/base.py lite/gym/wrappers.py
  scripts/rollout.py scripts/serve_env.py scripts/serve_sglang.py
  scripts/configs/*/default/waa*.yaml
  "$CFG"
)
shopt -u nullglob
DIRTY=$(git status --porcelain -- "${PIPELINE_PATHS[@]}")
if [ -n "$DIRTY" ]; then
  echo "[run.sh] ERROR: pipeline files have uncommitted changes; commit first:" >&2
  printf '%s\n' "$DIRTY" >&2
  exit 1
fi
source "$ROOT/devs/exps/eval/utils/campaign_dir.sh"
resolve_eval_commit_dir
RUN_ID="${EVAL_RUN_ID:-}"
if [ -z "$RUN_ID" ] && [ -d "$COMMIT_DIR" ]; then
  RUN_ID=$(find "$COMMIT_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
    grep -E '^run_[0-9]+(_[A-Za-z0-9_.-]+)?$' | sort -t_ -k2,2n | tail -1 || true)
fi
RUN_ID="${RUN_ID:-run_0}"
if [ ! -d "$COMMIT_DIR/$RUN_ID" ] && compgen -G "$COMMIT_DIR/run_*" > /dev/null; then
  echo "[run.sh] WARNING: starting fresh $RUN_ID; existing campaigns: $COMMIT_DIR/run_*" >&2
  echo "[run.sh] Unset EVAL_RUN_ID to resume the latest campaign." >&2
  sleep 5
fi
LOG_ROOT="$COMMIT_DIR/$RUN_ID/$SLUG"

MODEL_PATH="${EVAL_MODEL_PATH:-}"
EXTRA_ARGS=()
if [ -n "$MODEL_PATH" ]; then
  EXTRA_ARGS+=(--model-path "$MODEL_PATH")
fi
if [[ "$MODEL" != gpt-* && "$MODEL" != claude-* && "$MODEL" != gemini-* ]]; then
  : "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the allocated GPUs}"
fi
if [ -n "${EVAL_ENGINE_KWARGS:-}" ]; then
  EXTRA_ARGS+=(--engine-kwargs "$EVAL_ENGINE_KWARGS")
fi

EVAL_ENV_ID=waa
source "$ROOT/devs/exps/eval/utils/runtime_mode.sh"
export CUA_LITE_DOCKER_CREATE_CONCURRENCY=10
mkdir -p "$LOG_ROOT"
echo "[run.sh] model=$MODEL config=$CFG concurrency=$CONCURRENCY"
echo "[run.sh] commit_dir=$(basename "$COMMIT_DIR") run_id=$RUN_ID log_root=$LOG_ROOT"
HF_HUB_OFFLINE=1 exec uv run python scripts/rollout.py \
  --model-id "$MODEL" --env-id waa --splits eval \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path "$CFG" --log-root "$LOG_ROOT" \
  --concurrency "$CONCURRENCY" --max-attempts 3 --debug \
  --save-video false --min-valid-frac 1 \
  "${EXTRA_ARGS[@]}"
