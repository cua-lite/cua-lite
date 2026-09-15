#!/usr/bin/env bash
# mobileworld eval — single-model invocation.
#
# Routes --log-root to <repo-root>/.exps/eval/mobileworld/<commit-ts>_<commit>/<run_id>/<slug>/.
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
# set EVAL_ALLOW_DIRECT=1 for an explicit direct-mode dev run — and for
# mobileworld it is notably worse: direct mode boots a FRESH DinD container per
# task (cold boot 2-5 min each), while env-server mode centralizes admission,
# ownership, idle cleanup, and retry/recovery around those cold instances. See
# /docs/envs.md#env-server and
# /lite/gym/envs/mobileworld/README.md.
#
# Usage:
#   # auto-resume to latest run_<N> at this commit (most common):
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/mobileworld/run.sh <model-id>
#   # or open a fresh campaign at this commit:
#   export EVAL_RUN_ID="run_1"        # bump past any existing run_0 / run_1 / ...
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/mobileworld/run.sh <model-id>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/mobileworld/run.sh Qwen/Qwen3-VL-8B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/mobileworld/run.sh Qwen/Qwen3-VL-32B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/mobileworld/run.sh Qwen/Qwen3.5-27B
#   ./devs/exps/eval/mobileworld/run.sh gpt-5.5       # API model, no GPU
#
# tp_size comes from the model's LOCAL_AGENTS entry (lite/agents/factory.py), NOT from the
# GPU count; serve_sglang.py derives dp_size = visible // tp_size (local HF models only),
# so the GPUs you expose set the REPLICA count. This script never passes --engine-kwargs,
# so tp is unchangeable here.
# Pre-reqs (env-server host):
#   - /dev/kvm rw-accessible (kvm group or setfacl).
#   - cua-lite/mobileworld docker image built via
#       uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh
#   - OPENAI_API_KEY exported before containers spawn; set OPENAI_BASE_URL only
#     for a custom endpoint. The 44 agent-user-interaction tasks drive a
#     simulated-user LLM (server_kwargs.user_agent_model) through it; without
#     it those tasks fail while the 117 GUI-only tasks still run.
#
# Env shape (see lite/gym/envs/mobileworld/README.md):
#   - eval split = 161 deterministic tasks (201 upstream − 40 agent-mcp) across
#     20 apps; reward is the real state-based score from /task/eval (truncated
#     episodes still get their real score — upstream parity, not automatic 0).
#   - Each task runs in a privileged Docker-in-Docker box (nested dockerd +
#     rooted emulator + app backends) — the heaviest per-create env in the
#     matrix. Container spawns gate at server_kwargs.spawn_concurrency=4; keep
#     rollout --concurrency within ~2-4x of that (README: 8-16), so default 8.
#   - --env-kwargs step_timeout=240s overrides the framework default 120s
#     (lite/gym/registry.py `make()`; mobileworld's make_kwargs only set
#     reset_timeout). Chosen > the in-container /step HTTP timeout
#     (server_kwargs.step_timeout=180, which also covers ask_user LLM calls)
#     so the inner timeout surfaces first with its own error.
#   - max_steps: env default 50 (upstream `mw eval --max_round 50`) — no override.
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
EVAL_ENV_ID="mobileworld"
source "$ROOT/devs/exps/eval/utils/runtime_mode.sh"

SLUG="${MODEL//\//_}${CONFIG_ID:+__${CONFIG_ID}}"
ENV_ROOT="$ROOT/.exps/eval/mobileworld"

# Pipeline-relevant paths: changes to these files are what advance the
# commit-id used for path keying. Pure docs (CHANGELOG, README, snapshots)
# and unrelated training/other-env code do NOT — campaigns reuse the
# existing commit dir if pipeline state hasn't moved since.
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/mobileworld/run.sh
  lite/core
  lite/agents
  lite/gym/envs/mobileworld
  lite/gym/utils
  lite/gym/__init__.py lite/gym/types.py lite/gym/registry.py
  lite/gym/factory.py lite/gym/services.py lite/gym/remote
  lite/gym/base.py lite/gym/wrappers.py
  scripts/serve_env.py devs/exps/eval/utils/runtime_mode.sh
  devs/exps/eval/utils/campaign_dir.sh
  lite/agents/factory.py lite/infer/serving.py lite/infer/rollout.py
  scripts/rollout.py
  scripts/configs/*/default/mobileworld.yaml
  scripts/configs/*/default/mobileworld.*.yaml
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

# Pre-flight: the simulated-user LLM key for the 44 agent-user-interaction
# tasks. Non-fatal — GUI-only tasks (117) don't need it — but a full-suite
# campaign without it will leave the ask_user tasks failing until a mop-up
# round with the key exported. This wrapper can only inspect the invoking shell;
# in env-server mode the key that matters is the env-server/container-spawn
# process environment.
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[run.sh] WARNING: OPENAI_API_KEY unset — the 44 agent-user-interaction (ask_user) tasks will fail." >&2
  echo "  export OPENAI_API_KEY on the env-server host before it spawns containers; set OPENAI_BASE_URL only for a custom endpoint." >&2
  sleep 5
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
export SESSION_ID="${SESSION_ID:-mobileworld-${RUN_ID}-${SLUG}}"

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
  Qwen/Qwen3-VL-*-Instruct|Qwen/Qwen3-VL-*-Thinking) CFG=scripts/configs/qwen3_vl/default/mobileworld.yaml ;;
  Qwen/Qwen3.5-*)                  CFG=scripts/configs/qwen3_5/default/mobileworld.yaml ;;
  ByteDance-Seed/UI-TARS-7B-DPO)   CFG=scripts/configs/ui_tars/default/mobileworld.yaml ;;
  ByteDance-Seed/UI-TARS-1.5-7B)   CFG=scripts/configs/ui_tars_15_v1/default/mobileworld.yaml ;;
  Tongyi-MAI/MAI-UI-*)             CFG=scripts/configs/mai_ui/default/mobileworld.yaml ;;
  stepfun-ai/GELab-Zero-*)         CFG=scripts/configs/step_gui/default/mobileworld.yaml ;;
  gpt-*)                           CFG=scripts/configs/gpt/default/mobileworld.yaml ;;
  claude-*)                        CFG=scripts/configs/claude/default/mobileworld.yaml ;;
  *) echo "unknown model: $MODEL — add a case in $0" >&2; exit 1 ;;
esac
CFG="${CONFIG_PATH:-$CFG}"

CONCURRENCY="${EVAL_CONCURRENCY:-8}"
if [ "$CONCURRENCY" -gt 16 ]; then
  echo "[run.sh] WARNING: EVAL_CONCURRENCY=$CONCURRENCY is above the recommended MobileWorld range (8-16)." >&2
  echo "         MobileWorld uses privileged DinD Android containers; high rollout concurrency commonly produces reset/step 404s." >&2
fi

mkdir -p "$LOG_ROOT"
echo "[run.sh] $MODEL"
echo "         commit_dir=$(basename "$COMMIT_DIR")  run_id=$RUN_ID  GPUs=${CUDA_VISIBLE_DEVICES:-?}"
echo "         session_id=$SESSION_ID"
echo "         log_root=$LOG_ROOT"
echo "         config=$CFG"

HF_HUB_OFFLINE=1 exec uv run python scripts/rollout.py \
  --model-id "$MODEL" \
  --env-id mobileworld --splits eval \
  --concurrency "$CONCURRENCY" \
  --env-kwargs '{"step_timeout": 240}' \
  --config-path "$CFG" \
  --log-root "$LOG_ROOT"
