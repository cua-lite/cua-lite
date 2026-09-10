#!/usr/bin/env bash
# osworld_2 (OSWorld-V2) eval — single-model invocation.
#
# Routes --log-root to <repo-root>/.exps/eval/osworld_2/<commit-ts>_<commit>/<run_id>/<slug>/.
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
# set EVAL_ALLOW_DIRECT=1 for an explicit direct-mode dev run. Like osworld v1,
# osworld_2 VMs are local VM-in-Docker containers on the env-server host
# (separate image + task set). See /docs/envs.md#env-server
# and /lite/gym/envs/osworld_2/README.md.
#
# Usage:
#   # auto-resume to latest run_<N> at this commit (most common):
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/osworld_2/run.sh <model-id>
#   # or open a fresh campaign at this commit:
#   export EVAL_RUN_ID="run_1"        # bump past any existing run_0 / run_1 / ...
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/osworld_2/run.sh <model-id>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/osworld_2/run.sh Qwen/Qwen3-VL-8B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/osworld_2/run.sh Qwen/Qwen3-VL-32B-Instruct
#   ./devs/exps/eval/osworld_2/run.sh gpt-5.5         # API model, no GPU
#
# tp_size comes from the model's LOCAL_AGENTS entry (lite/agents/factory.py), NOT from the
# GPU count; serve_sglang.py derives dp_size = visible // tp_size (local HF models only),
# so the GPUs you expose set the REPLICA count. This script never passes --engine-kwargs,
# so tp is unchangeable here.
# Pre-reqs (env-server host):
#   - /dev/kvm AND /dev/net/tun rw-accessible.
#   - cua-lite/osworld_2 image + gated v2 qcow2 + 108 task classes via
#       uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh
#     (needs HF auth with the xlangai/v2-image + xlangai/osworld_v2_tasks gates accepted).
#   - OPENAI_API_KEY exported where the env MODULE IMPORTS (env-server launch,
#     or this shell in direct mode); set OPENAI_BASE_URL only for a custom endpoint.
#     The ~18 llm_judge tasks call an LLM at evaluate() (server_kwargs.eval_model,
#     default gpt-4.1). Without the key they register with
#     exclude_reason="llm_judge" and the filter below silently drops them.
#
# Env shape (see lite/gym/envs/osworld_2/README.md):
#   - eval split = 108 capability-graded tasks (ids 001-108), each on a
#     DEDICATED QEMU/KVM VM-in-Docker container (no snapshot reuse — one
#     container per trajectory, cold boot ~30-90 s).
#   - The SCORED count is service-dependent, not fixed: exclude_reason gates
#     tasks whose service isn't provisioned. With the default website host
#     (web.hku.icu) + OPENAI_API_KEY → 82 scored; without the key → 67
#     (llm_judge drop); gitlab + human_in_the_loop stay excluded unless their
#     server_kwargs knobs are set. Record num_tasks from summary.json — don't
#     assume 82.
#   - No --env-kwargs step_timeout override: osworld_2's make_kwargs already
#     set step_timeout=180 + reset_timeout=600 (configs/default.yaml), unlike
#     androidworld/mobilegym whose specs leave the framework 120s default.
#   - max_steps: env default 200 (OSWorld-V2 official GPT run) — no override.
#     V2 trajectories are far longer than v1's 30-step runs; budget wall-clock
#     accordingly.
set -euo pipefail

MODEL="${1:?usage: CUDA_VISIBLE_DEVICES=<gpus> $0 <model-id>}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)"
[[ -n "$ROOT" && -d "$ROOT" ]] || { echo "$0: cannot resolve repo root from ${BASH_SOURCE[0]}" >&2; exit 1; }
cd "$ROOT"
EVAL_ENV_ID="osworld_2"
source "$ROOT/devs/exps/eval/utils/runtime_mode.sh"

SLUG="${MODEL//\//_}"
ENV_ROOT="$ROOT/.exps/eval/osworld_2"

# Pipeline-relevant paths: changes to these files are what advance the
# commit-id used for path keying. Pure docs (CHANGELOG, README, snapshots)
# and unrelated training/other-env code do NOT — campaigns reuse the
# existing commit dir if pipeline state hasn't moved since.
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/osworld_2/run.sh
  lite/core
  lite/agents
  lite/gym/envs/osworld_2
  lite/gym/utils
  lite/gym/__init__.py lite/gym/types.py lite/gym/registry.py
  lite/gym/factory.py lite/gym/services.py lite/gym/remote
  lite/gym/base.py lite/gym/wrappers.py
  scripts/serve_env.py devs/exps/eval/utils/runtime_mode.sh
  devs/exps/eval/utils/campaign_dir.sh
  lite/agents/factory.py lite/infer/serving.py lite/infer/rollout.py
  scripts/rollout.py
  scripts/configs/*/default/osworld_2.yaml
)
shopt -u nullglob

# Pre-flight: pipeline files must be committed (clean working tree + index).
DIRTY=$(git status --porcelain -- "${PIPELINE_PATHS[@]}" 2>/dev/null)
if [ -n "$DIRTY" ]; then
  echo "[run.sh] ERROR: pipeline files have uncommitted changes — commit first (path key would be ambiguous):" >&2
  echo "$DIRTY" | sed 's/^/  /' >&2
  exit 1
fi

# Pre-flight: the LLM-judge key. Non-fatal, but without it the ~18 llm_judge
# tasks are excluded AT REGISTRATION (on the env-server host), shrinking the
# scored set from 82 to 67 — and a later mop-up with the key would need an
# env-server restart to re-register them.
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[run.sh] WARNING: OPENAI_API_KEY unset in this shell — if it's also unset where the" >&2
  echo "  env-server was launched, the ~18 llm_judge tasks are excluded (82 → 67 scored)." >&2
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
export SESSION_ID="${SESSION_ID:-osworld_2-${RUN_ID}-${SLUG}}"

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

# model-family → rollout config (only these families have osworld_2 configs so far)
case "$MODEL" in
  Qwen/Qwen3-VL-*-Instruct|Qwen/Qwen3-VL-*-Thinking) CFG=scripts/configs/qwen3_vl/default/osworld_2.yaml ;;
  gpt-*)                           CFG=scripts/configs/gpt/default/osworld_2.yaml ;;
  claude-*)                        CFG=scripts/configs/claude/default/osworld_2.yaml ;;
  *) echo "unknown model: $MODEL — add a case (and a scripts/configs/<family>/default/osworld_2.yaml) in $0" >&2; exit 1 ;;
esac

CONCURRENCY="${EVAL_CONCURRENCY:-16}"

mkdir -p "$LOG_ROOT"
echo "[run.sh] $MODEL"
echo "         commit_dir=$(basename "$COMMIT_DIR")  run_id=$RUN_ID  GPUs=${CUDA_VISIBLE_DEVICES:-?}"
echo "         log_root=$LOG_ROOT"
echo "         config=$CFG"

HF_HUB_OFFLINE=1 exec uv run python scripts/rollout.py \
  --model-id "$MODEL" \
  --env-id osworld_2 --splits eval \
  `# Filter drops service-gated tasks (llm_judge without OPENAI_API_KEY,` \
  `# gitlab, human_in_the_loop, volume — whatever this deployment left` \
  `# unprovisioned); the scored count lands at 108 minus those.` \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --concurrency "$CONCURRENCY" \
  --config-path "$CFG" \
  --log-root "$LOG_ROOT"
