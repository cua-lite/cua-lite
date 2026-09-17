#!/usr/bin/env bash
# osworld_g (grounding) eval — single-model invocation.
#
# Routes --log-root to <repo-root>/.exps/eval/osworld_g/<commit-ts>_<commit>/<run_id>/<slug>/.
# Repo root is derived from this script's location (worktree-safe — a stale
# CUA_LITE_ROOT inherited from another worktree's shell would otherwise
# silently redirect output).
# Resumes if the same (commit, run_id, model) was run before; completed tasks skipped.
#
# $EVAL_RUN_ID is OPTIONAL — see devs/exps/eval/AGENTS.md "Campaign id contract".
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
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/osworld_g/run.sh <model-id>
#   # or open a fresh campaign at this commit:
#   export EVAL_RUN_ID="run_1"        # bump past any existing run_0 / run_1 / ...
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/osworld_g/run.sh <model-id>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/osworld_g/run.sh Qwen/Qwen3-VL-8B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/osworld_g/run.sh Qwen/Qwen3-VL-32B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/osworld_g/run.sh Qwen/Qwen3.5-27B
#
# tp_size comes from the model's LOCAL_AGENTS entry (lite/agents/factory.py), NOT from the
# GPU count; serve_sglang.py derives dp_size = visible // tp_size, so the GPUs you expose
# set the REPLICA count. This script never passes --engine-kwargs, so tp is unchangeable here.
# Pre-req: `uv run python lite/gym/envs/osworld_g/scripts/utils/download_tasks.py` to
# clone the upstream OSWorld-G repo into ./data/OSWorld-G/ (one-time).

set -euo pipefail

MODEL="${1:?usage: CUDA_VISIBLE_DEVICES=<gpus> $0 <model-id> [config-path]}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)"
[[ -n "$ROOT" && -d "$ROOT" ]] || { echo "$0: cannot resolve repo root from ${BASH_SOURCE[0]}" >&2; exit 1; }
cd "$ROOT"
# An explicit YAML gets its own result slug; omitted YAML keeps the default path.
CONFIG_PATH="${2:-}"
CONFIG_ID=""
if [ -n "$CONFIG_PATH" ]; then
  CONFIG_PATH="$(realpath -- "$CONFIG_PATH")"
  if [[ "$CONFIG_PATH" != "$ROOT"/scripts/configs/*.yaml ]] || [ ! -f "$CONFIG_PATH" ]; then
    echo "[run.sh] ERROR: config must be an existing YAML under scripts/configs/" >&2
    exit 1
  fi
  CONFIG_PATH="${CONFIG_PATH#"$ROOT"/}"
  CONFIG_ID="${CONFIG_PATH#scripts/configs/}"
  CONFIG_ID="${CONFIG_ID%.yaml}"
  CONFIG_ID="${CONFIG_ID//\//__}"
fi
EVAL_ENV_ID="osworld_g"
source "$ROOT/devs/exps/eval/utils/runtime_mode.sh"

# MODEL_PATH optional: point sglang at a local checkpoint instead of
# downloading $MODEL from HF. SLUG_SUFFIX optional: appended to the
# default $MODEL-derived SLUG so a fine-tuned checkpoint eval lands in
# its own log dir (e.g. SLUG_SUFFIX=__sft-iter1999) without overwriting
# the baseline eval at the same $MODEL.
MODEL_PATH="${MODEL_PATH:-}"
case "$MODEL" in
  microsoft/Fara-7B)
    # Upstream config is incompatible with this transformers/SGLang stack.
    # Keep the registered adapter key and serve the config-only HF fork.
    MODEL_PATH="${MODEL_PATH:-cua-lite/Fara-7B}"
    ;;
esac
SLUG="${MODEL//\//_}${CONFIG_ID:+__${CONFIG_ID}}${SLUG_SUFFIX:-}"
ENV_ROOT="$ROOT/.exps/eval/osworld_g"

# Pipeline-relevant paths: changes to these files are what advance the
# commit-id used for path keying. Pure docs (CHANGELOG, README, snapshots)
# and unrelated training/other-env code do NOT — campaigns reuse the
# existing commit dir if pipeline state hasn't moved since.
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/osworld_g/run.sh
  lite/core
  lite/agents
  lite/gym/envs/osworld_g
  lite/gym/utils
  lite/gym/__init__.py lite/gym/types.py lite/gym/registry.py
  lite/gym/factory.py lite/gym/services.py lite/gym/remote
  lite/gym/base.py lite/gym/wrappers.py
  scripts/serve_env.py devs/exps/eval/utils/runtime_mode.sh
  devs/exps/eval/utils/campaign_dir.sh
  lite/agents/factory.py lite/infer/serving.py lite/infer/rollout.py
  scripts/rollout.py
  scripts/configs/*/default/osworld_g.yaml
  scripts/configs/*/default/osworld_g.*.yaml
)
shopt -u nullglob
if [ -n "$CONFIG_PATH" ]; then
  PIPELINE_PATHS+=("$CONFIG_PATH")
fi

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

# model-family → rollout config (grounding subset: qwen3_vl, qwen3_5, evocua, ui_tars, ui_tars_15_v1, mai_ui)
case "$MODEL" in
  gpt-*)                            CFG=scripts/configs/gpt/default/osworld_g.yaml ;;
  Qwen/Qwen3-VL-*-Instruct)        CFG=scripts/configs/qwen3_vl/default/osworld_g.yaml ;;
  Qwen/Qwen2.5-VL-*-Instruct)      CFG=scripts/configs/qwen2_5_vl/default/osworld_g.yaml ;;
  Qwen/Qwen3.5-*)                  CFG=scripts/configs/qwen3_5/default/osworld_g.yaml ;;
  Qwen/Qwen3.8-*)                  CFG=scripts/configs/qwen3_8/default/osworld_g.yaml ;;
  inclusionAI/UI-Venus-2-*)        CFG=scripts/configs/ui_venus_2/default/osworld_g.yaml ;;
  ByteDance-Seed/UI-TARS-7B-DPO)   CFG=scripts/configs/ui_tars/default/osworld_g.yaml ;;
  ByteDance-Seed/UI-TARS-1.5-7B)   CFG=scripts/configs/ui_tars_15_v1/default/osworld_g.yaml ;;
  meituan/EvoCUA-*)                CFG=scripts/configs/evocua/default/osworld_g.yaml ;;
  Tongyi-MAI/MAI-UI-*)             CFG=scripts/configs/mai_ui/default/osworld_g.yaml ;;
  microsoft/Fara-*)                CFG=scripts/configs/fara/default/osworld_g.yaml ;;
  *) echo "unknown model: $MODEL — add a case in $0" >&2; exit 1 ;;
esac
CFG="${CONFIG_PATH:-$CFG}"

mkdir -p "$LOG_ROOT"
echo "[run.sh] $MODEL"
echo "         commit_dir=$(basename "$COMMIT_DIR")  run_id=$RUN_ID  GPUs=${CUDA_VISIBLE_DEVICES:-?}"
echo "         log_root=$LOG_ROOT"
echo "         config=$CFG"

HF_HUB_OFFLINE=1 exec uv run python scripts/rollout.py \
  --model-id "$MODEL" \
  ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
  --env-id osworld_g --splits eval \
  `# Filter drops the 54 refusal tasks. extra_tools defaults to [] now` \
  `# (opt-in), so report_infeasible is already absent from the agent's` \
  `# action list — no need to opt out explicitly. Together they prevent` \
  `# false-positive give-up calls on bbox/polygon tasks.` \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --env-kwargs '{"step_timeout": 180}' \
  --concurrency "${EVAL_CONCURRENCY:-64}" \
  --config-path "$CFG" \
  --log-root "$LOG_ROOT"
