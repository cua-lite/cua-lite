#!/usr/bin/env bash
# cua (CUABench) eval — single-model invocation across the cua.bench.local.* suites.
#
# Routes --log-root to <repo-root>/.exps/eval/cua/<commit-ts>_<commit>/<run_id>/<slug>__<suite>/.
# Repo root is derived from this script's location (worktree-safe — a stale
# CUA_LITE_ROOT inherited from another worktree's shell would otherwise
# silently redirect output).
# Resumes if the same (commit, run_id, model, suite) was run before; completed tasks skipped.
#
# $EVAL_RUN_ID is OPTIONAL — see devs/exps/eval/AGENTS.md "Run id contract".
# If unset, run.sh auto-resolves to the highest-numbered run_<N>[_<label>] under the
# current commit dir (resume-to-latest), or `run_0` if this is the first campaign.
#
# $EVAL_SUITE selects the dataset(s): "basic" | "workflows" | "kicad" | "all"
# (default). Each suite is its OWN env_id (cua.bench.local.<suite>) with its own
# rollout config and task budget, so each gets its own artifact dir via the
# `<slug>__<suite>` suffix (the browsergym EVAL_CONFIG_ID slug convention) —
# suites share one (commit, run_id) and resume independently.
#
#   suite     | env_id                    | tasks | max_steps | concurrency
#   basic     | cua.bench.local.basic     |  68   |  30       | 1 — PINNED
#   workflows | cua.bench.local.workflows |  52   | 100       | 1 — PINNED
#   kicad     | cua.bench.local.kicad     |  25   | 200       | 1 — PINNED (see below)
#
# KiCad is pinned to --concurrency 1 regardless of $EVAL_CONCURRENCY: its
# per-task `apt install` streams through cua's run_command, which truncates
# under concurrent load (verified: canary task scored 0.0 at concurrency 4 vs
# 0.667 sequentially — see lite/gym/envs/cua/README.md "Evaluation").
#
# Env-server prereq (workflow default): export CUA_LITE_ENV_SERVER_URL +
# CUA_LITE_ENV_SERVER_TOKEN before invoking. Missing vars now fail fast;
# set EVAL_ALLOW_DIRECT=1 for an explicit direct-mode dev run.
# cua.bench.local.* is a DEDICATED container backend (one trycua/cua-xfce
# container per trajectory) — run behind an env-server so the drift-reaper
# reclaims leaked containers. See /docs/envs.md#env-server and
# /lite/gym/envs/cua/README.md.
#
# Usage:
#   # auto-resume to latest run_<N> at this commit (most common):
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/cua/run.sh <model-id>
#   # or open a fresh campaign at this commit:
#   export EVAL_RUN_ID="run_1"        # bump past any existing run_0 / run_1 / ...
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/cua/run.sh <model-id>
#   # single suite (e.g. mop up kicad only):
#   EVAL_SUITE=kicad ./devs/exps/eval/cua/run.sh <model-id>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/cua/run.sh Qwen/Qwen3-VL-8B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/cua/run.sh Qwen/Qwen3-VL-32B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/cua/run.sh Qwen/Qwen3.5-27B
#   ./devs/exps/eval/cua/run.sh gpt-5.5               # API model, no GPU
#
# tp_size comes from the model's LOCAL_AGENTS entry (lite/agents/factory.py), NOT from the
# GPU count; serve_sglang.py derives dp_size = visible // tp_size (local HF models only),
# so the GPUs you expose set the REPLICA count. This script never passes --engine-kwargs,
# so tp is unchangeable here.
# Pre-reqs (env-server host): Docker +
#   uv run --no-sync bash lite/gym/envs/cua/scripts/install.sh
# (pip-installs cua-sandbox + cua-bench, pulls trycua/cua-xfce, downloads the
# datasets → lite/gym/envs/cua/.cache — no exports needed for local modes).
#
# Env shape (see lite/gym/envs/cua/README.md):
#   - eval split only; 145 tasks total across the 3 suites. Reward is
#     cua-bench's own float score; cua-bench counts solved as reward >= 0.5,
#     so mean_episode_return is NOT a plain success rate.
#   - --env-kwargs step_timeout=180s overrides the framework default 120s
#     (lite/gym/registry.py `make()`); cua.bench also carries
#     make_kwargs.cursor from bench/configs/default.yaml. Steps drive a real
#     xfce desktop container — align with osworld's 180s.
#   - max_steps carried per-suite by the rollout configs (30/100/200).
set -euo pipefail

MODEL="${1:?usage: CUDA_VISIBLE_DEVICES=<gpus> $0 <model-id>}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)"
[[ -n "$ROOT" && -d "$ROOT" ]] || { echo "$0: cannot resolve repo root from ${BASH_SOURCE[0]}" >&2; exit 1; }
cd "$ROOT"

case "${EVAL_SUITE:-all}" in
  all)                  SUITES=(basic workflows kicad) ;;
  basic|workflows|kicad) SUITES=("$EVAL_SUITE") ;;
  *) echo "[run.sh] ERROR: EVAL_SUITE must be basic|workflows|kicad|all, got: ${EVAL_SUITE}" >&2; exit 1 ;;
esac

SLUG="${MODEL//\//_}"
ENV_ROOT="$ROOT/.exps/eval/cua"

# Pipeline-relevant paths: changes to these files are what advance the
# commit-id used for path keying. Pure docs (CHANGELOG, README, snapshots)
# and unrelated training/other-env code do NOT — campaigns reuse the
# existing commit dir if pipeline state hasn't moved since.
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/cua/run.sh
  lite/core
  lite/agents
  lite/gym/envs/cua
  lite/gym/utils
  lite/gym/__init__.py lite/gym/types.py lite/gym/registry.py
  lite/gym/factory.py lite/gym/services.py lite/gym/remote
  lite/gym/base.py lite/gym/wrappers.py
  scripts/serve_env.py devs/exps/eval/utils/runtime_mode.sh
  devs/exps/eval/utils/campaign_dir.sh
  lite/agents/factory.py lite/infer/serving.py lite/infer/rollout.py
  scripts/rollout.py
  scripts/configs/*/default/cua.bench/*.yaml
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
export SESSION_ID="${SESSION_ID:-cua-${RUN_ID}-${SLUG}}"

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

# model-family → rollout config dir; the suite picks <suite>.yaml inside it.
# (only these families have cua.bench config dirs so far)
case "$MODEL" in
  Qwen/Qwen3-VL-*-Instruct|Qwen/Qwen3-VL-*-Thinking) CFG_DIR=scripts/configs/qwen3_vl/default/cua.bench ;;
  Qwen/Qwen3.5-*)                  CFG_DIR=scripts/configs/qwen3_5/default/cua.bench ;;
  gpt-*)                           CFG_DIR=scripts/configs/gpt/default/cua.bench ;;
  *) echo "unknown model: $MODEL — add a case (and a scripts/configs/<family>/default/cua.bench/ dir) in $0" >&2; exit 1 ;;
esac

# One rollout per suite, sequential; each resumes independently from its own
# log-root, so re-running this script mops up all suites in one go.
for SUITE in "${SUITES[@]}"; do
  CFG="$CFG_DIR/$SUITE.yaml"
  [ -f "$CFG" ] || { echo "[run.sh] ERROR: config not found: $CFG" >&2; exit 1; }
  LOG_ROOT="$COMMIT_DIR/$RUN_ID/${SLUG}__${SUITE}"
  EVAL_ENV_ID="cua.bench.local.$SUITE"
  source "$ROOT/devs/exps/eval/utils/runtime_mode.sh"


  SUITE_CONCURRENCY=1        # pinned — see header
  
  mkdir -p "$LOG_ROOT"
  echo "[run.sh] $MODEL — suite=$SUITE (env_id=cua.bench.local.$SUITE, concurrency=$SUITE_CONCURRENCY)"
  echo "         commit_dir=$(basename "$COMMIT_DIR")  run_id=$RUN_ID  GPUs=${CUDA_VISIBLE_DEVICES:-?}"
  echo "         log_root=$LOG_ROOT"
  echo "         config=$CFG"

  HF_HUB_OFFLINE=1 uv run python scripts/rollout.py \
    --model-id "$MODEL" \
    --env-id "cua.bench.local.$SUITE" --splits eval \
    --concurrency "$SUITE_CONCURRENCY" \
    --env-kwargs '{"step_timeout": 180}' \
    --config-path "$CFG" \
    --log-root "$LOG_ROOT"
done
