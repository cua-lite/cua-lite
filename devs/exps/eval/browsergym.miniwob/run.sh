#!/usr/bin/env bash
# browsergym.miniwob eval — single-model invocation.
#
# Routes --log-root to <repo-root>/.exps/eval/browsergym.miniwob/<commit-ts>_<commit>/<run_id>/<slug>/.
# Repo root is derived from this script's location (worktree-safe — a stale
# CUA_LITE_ROOT inherited from another worktree's shell would otherwise
# silently redirect output).
# Resumes if the same (commit, run_id, model) was run before; completed tasks skipped.
#
# $EVAL_RUN_ID is OPTIONAL — see devs/exps/eval/AGENTS.md "Run id contract".
# If unset, run.sh auto-resolves to the highest-numbered run_<N>[_<label>] under the
# current commit dir (resume-to-latest), or `run_0` if this is the first campaign.
#
# $EVAL_MODE selects the observation/action mode (config filename), default "default"
# (screenshot + coord). Set to "text_only" for the text+AXTree bid-mode config.
#
# MiniWoB is stateless (125 tasks, local HTTP server — no shared mutable backend),
# so this is a single fully-parallel pass, unlike the WA/VWA read/write split.
# Reset timeouts appear at 32+ concurrent envs (see the browsergym README), so
# $EVAL_CONCURRENCY defaults to 16.
#
# Env-server prereq (workflow default): export CUA_LITE_ENV_SERVER_URL +
# CUA_LITE_ENV_SERVER_TOKEN before invoking. Missing vars now fail fast;
# set EVAL_ALLOW_DIRECT=1 for an explicit direct-mode dev run. The
# env-server host must have run `install.sh miniwob` once; the shared MiniWoB
# HTTP singleton is auto-started on the first task. See /docs/envs.md#env-server
# and /lite/gym/envs/browsergym/README.md.
#
# Usage:
#   # auto-resume to latest run_<N> at this commit (most common):
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/browsergym.miniwob/run.sh <model-id>
#   # or open a fresh campaign at this commit:
#   export EVAL_RUN_ID="run_1"        # bump past any existing run_0 / run_1 / ...
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/browsergym.miniwob/run.sh <model-id>
#   # text+bid mode:
#   EVAL_MODE=text_only ./devs/exps/eval/browsergym.miniwob/run.sh <model-id>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/browsergym.miniwob/run.sh Qwen/Qwen3-VL-8B-Instruct
#   CUDA_VISIBLE_DEVICES=0       EVAL_MODE=text_only EVAL_RUN_ID=run_0_textonly EVAL_ENABLE_THINKING=false ./devs/exps/eval/browsergym.miniwob/run.sh Qwen/Qwen3-VL-8B-Thinking
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/browsergym.miniwob/run.sh Qwen/Qwen3-VL-32B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/browsergym.miniwob/run.sh Qwen/Qwen3.5-27B
#   ./devs/exps/eval/browsergym.miniwob/run.sh gpt-5.5              # API model, no GPU
#   ./devs/exps/eval/browsergym.miniwob/run.sh claude-opus-4-6      # API model, no GPU
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
EVAL_ENV_ID="browsergym.miniwob"
source "$ROOT/devs/exps/eval/utils/runtime_mode.sh"

MODE="${EVAL_MODE:-default}"
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
if [[ "$ENABLE_THINKING" -eq 1 && "$MODE" != "text_only" ]]; then
  echo "[run.sh] ERROR: EVAL_ENABLE_THINKING=true is only supported with EVAL_MODE=text_only" >&2
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
if [[ -z "$CONFIG_ID" && "$MODE" == "text_only" && "$ENABLE_THINKING_ENV_SET" -eq 1 && "$SUPPORTS_ENABLE_THINKING" -eq 1 ]]; then
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
ENV_ROOT="$ROOT/.exps/eval/browsergym.miniwob"

# Pipeline-relevant paths: changes to these files are what advance the
# commit-id used for path keying. Pure docs (CHANGELOG, README, snapshots)
# and unrelated training/other-env code do NOT — campaigns reuse the
# existing commit dir if pipeline state hasn't moved since.
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/browsergym.miniwob/run.sh
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
  scripts/configs/*/default/browsergym.miniwob/*.yaml
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

# model-family -> rollout config dir; $MODE picks default.yaml vs text_only.yaml
# for Qwen-family arms that publish both. Other families use default.yaml only.
case "$MODEL" in
  Qwen/Qwen3-VL-*-Instruct|Qwen/Qwen3-VL-*-Thinking) CFG=scripts/configs/qwen3_vl/default/browsergym.miniwob/${MODE}.yaml ;;
  Qwen/Qwen3.5-*)                                      CFG=scripts/configs/qwen3_5/default/browsergym.miniwob/${MODE}.yaml ;;
  Qwen/Qwen3.8-*)                                      CFG=scripts/configs/qwen3_8/default/browsergym.miniwob/${MODE}.yaml ;;
  ByteDance-Seed/UI-TARS-1.5-7B)                       CFG=scripts/configs/ui_tars_15_v1/default/browsergym.miniwob/default.yaml ;;
  meituan/EvoCUA-*)                                    CFG=scripts/configs/evocua/default/browsergym.miniwob/default.yaml ;;
  inclusionAI/UI-Venus-2-*)                            CFG=scripts/configs/ui_venus_2/default/browsergym.miniwob/default.yaml ;;
  gpt-*)                                               CFG=scripts/configs/gpt/default/browsergym.miniwob/default.yaml ;;
  claude-*)                                            CFG=scripts/configs/claude/default/browsergym.miniwob/default.yaml ;;
  *) echo "unknown model: $MODEL — add a case in $0" >&2; exit 1 ;;
esac
[ -f "$CFG" ] || { echo "[run.sh] ERROR: config not found: $CFG (EVAL_MODE=$MODE)" >&2; exit 1; }

CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EXTRA_ROLLOUT_ARGS=()
if [[ "$ENABLE_THINKING" -eq 1 && "$SUPPORTS_ENABLE_THINKING" -eq 1 ]]; then
  EXTRA_ROLLOUT_ARGS+=(
    --agent-kwargs '{"enable_thinking": true, "sampling_kwargs": {"max_new_tokens": 4096}}'
  )
fi

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

# Single stateless pass — max_steps carried by the config's env_kwargs.
ROLLOUT_ARGS=(
  --model-id "$MODEL"
  --env-id browsergym.miniwob --splits eval
  --concurrency "$CONCURRENCY"
  --config-path "$CFG"
  --log-root "$LOG_ROOT"
  "${EXTRA_ROLLOUT_ARGS[@]}"
)
HF_HUB_OFFLINE=1 exec uv run python scripts/rollout.py "${ROLLOUT_ARGS[@]}"
