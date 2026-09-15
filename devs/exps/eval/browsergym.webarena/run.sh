#!/usr/bin/env bash
# browsergym.webarena eval — single-model invocation.
#
# Routes --log-root to <repo-root>/.exps/eval/browsergym.webarena/<commit-ts>_<commit>/<run_id>/<slug>/.
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
# (screenshot + coord). Other modes: "text_only" (text+AXTree bid-mode, the
# AgentLab-aligned path — recommended for weak open models), "som" (set-of-marks;
# qwen families only).
#
# This runner evaluates the paper-style 241-template subset: one instantiated
# intent per exact `(intent_template_id, intent_template)` pair from WebArena's
# 812 raw configs. Generate the committed prompt-data files with:
#
#   WebArena_SRC=~/ref/BrowserGym uv run python \
#     devs/exps/eval/browsergym.webarena/export_template_tasks.py
#
# WebArena shares ONE mutable Docker backend, so for a faithful score this runs
# the strict read/write split (see /lite/gym/envs/browsergym/README.md
# "Strict read/write split"):
#   1. READ  pass — non-mutating template-sampled tasks, fully parallel
#      ($CONCURRENCY), residue-immune on the clean baseline.
#   2. WRITE pass — mutating template-sampled tasks, serial (concurrency 1).
#      WebArena's task-id order already IS the curated `depends_on` order (every
#      parent has a lower id), so the write parquet is task-id ordered.
#   3. RECONCILE — the two passes have disjoint specs, so each rewrites
#      summary.json over its own subset only. A final read-only pass rebuilds one
#      combined summary.json over the full 241-task subset after any optional
#      EVAL_READ_FILTER/EVAL_WRITE_FILTER clauses.
# All passes write the same $LOG_ROOT and rollout.py skips completed tasks, so
# re-runs resume cleanly. For a rigorous number, start a scoped singleton
# env-server and restart it between models so each run rebuilds the WA stack
# fresh. GitLab cold boot can take 5-15 min; use
# `--warm-singleton` before launching this runner:
#
#   env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
#     uv run python scripts/serve_env.py \
#       --host 127.0.0.1 --port 30100 \
#       --env-ids browsergym.webarena \
#       --token <token> \
#       --warm-singleton
#   export CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30100
#   export CUA_LITE_ENV_SERVER_TOKEN=<token>
#
# Env-server prereq (workflow default): export CUA_LITE_ENV_SERVER_URL +
# CUA_LITE_ENV_SERVER_TOKEN before invoking. Missing vars now fail fast;
# set EVAL_ALLOW_DIRECT=1 for an explicit direct-mode dev run. The
# env-server host must have run `install.sh webarena` once (~60 GB); the Docker
# stack is auto-started on the first task. LLM-judge tasks (m.others.llm_as_a_judge)
# RUN by default and hard-fail at reset if OPENAI_API_KEY is unset on the env-server
# process — set OPENAI_BASE_URL there too only for a custom endpoint. Map tasks run by
# default; if the host built with WEBARENA_INSTALL_MAP=0, skip them via
# EVAL_READ_FILTER / EVAL_WRITE_FILTER (see below). See /docs/envs.md#env-server.
#
# Usage:
#   # auto-resume to latest run_<N> at this commit (most common):
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/browsergym.webarena/run.sh <model-id>
#   # or open a fresh campaign at this commit:
#   export EVAL_RUN_ID="run_1"        # bump past any existing run_0 / run_1 / ...
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/browsergym.webarena/run.sh <model-id>
#   # text+bid mode (recommended for weak open models):
#   EVAL_MODE=text_only ./devs/exps/eval/browsergym.webarena/run.sh <model-id>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/browsergym.webarena/run.sh Qwen/Qwen3-VL-8B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/browsergym.webarena/run.sh Qwen/Qwen3-VL-32B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/browsergym.webarena/run.sh Qwen/Qwen3.5-27B
#   ./devs/exps/eval/browsergym.webarena/run.sh gpt-5.5            # API model, no GPU
#   ./devs/exps/eval/browsergym.webarena/run.sh claude-opus-4-6    # API model, no GPU
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
EVAL_ENV_ID="browsergym.webarena"
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
if [[ "$ENABLE_THINKING" -eq 1 && "$MODE" != "text_only" && "$MODE" != "som" ]]; then
  echo "[run.sh] ERROR: EVAL_ENABLE_THINKING=true is only supported with EVAL_MODE=text_only or som" >&2
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
if [[ -z "$CONFIG_ID" && ( "$MODE" == "text_only" || "$MODE" == "som" ) && "$ENABLE_THINKING_ENV_SET" -eq 1 && "$SUPPORTS_ENABLE_THINKING" -eq 1 ]]; then
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
ENV_ROOT="$ROOT/.exps/eval/browsergym.webarena"

# Pipeline-relevant paths: changes to these files are what advance the
# commit-id used for path keying. Pure docs (CHANGELOG, README, snapshots)
# and unrelated training/other-env code do NOT — campaigns reuse the
# existing commit dir if pipeline state hasn't moved since.
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/browsergym.webarena/run.sh
  devs/exps/eval/browsergym.webarena/export_template_tasks.py
  devs/exps/eval/browsergym.webarena/webarena_241_templates.*.prompt_data.parquet
  devs/exps/eval/browsergym.webarena/webarena_241_templates.manifest.*
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
  scripts/configs/*/default/browsergym.webarena/*.yaml
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

# model-family → rollout config dir; $MODE picks default.yaml / text_only.yaml / som.yaml.
# API and non-Qwen local families currently have only default.yaml committed for this env,
# so $MODE applies to the Qwen arms only.
case "$MODEL" in
  Qwen/Qwen3-VL-*-Instruct|Qwen/Qwen3-VL-*-Thinking) CFG=scripts/configs/qwen3_vl/default/browsergym.webarena/${MODE}.yaml ;;
  Qwen/Qwen3.5-*)                                      CFG=scripts/configs/qwen3_5/default/browsergym.webarena/${MODE}.yaml ;;
  Qwen/Qwen3.8-*)                                      CFG=scripts/configs/qwen3_8/default/browsergym.webarena/${MODE}.yaml ;;
  ByteDance-Seed/UI-TARS-1.5-7B)                       CFG=scripts/configs/ui_tars_15_v1/default/browsergym.webarena/default.yaml ;;
  meituan/EvoCUA-*)                                    CFG=scripts/configs/evocua/default/browsergym.webarena/default.yaml ;;
  inclusionAI/UI-Venus-2-*)                            CFG=scripts/configs/ui_venus_2/default/browsergym.webarena/default.yaml ;;
  gpt-*)                                               CFG=scripts/configs/gpt/default/browsergym.webarena/default.yaml ;;
  claude-*)                                            CFG=scripts/configs/claude/default/browsergym.webarena/default.yaml ;;
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

# Optional extra filter clauses ANDed onto the read/write splits (e.g. skip map
# tasks on a host built with WEBARENA_INSTALL_MAP=0). Default: no-op (`True`).
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

PROMPT_DATA_DIR="$ROOT/devs/exps/eval/browsergym.webarena"
ALL_PROMPT_DATA_BASE="$PROMPT_DATA_DIR/webarena_241_templates.all.prompt_data.parquet"
READ_PROMPT_DATA_BASE="$PROMPT_DATA_DIR/webarena_241_templates.read.prompt_data.parquet"
WRITE_PROMPT_DATA_BASE="$PROMPT_DATA_DIR/webarena_241_templates.write.prompt_data.parquet"
for f in "$ALL_PROMPT_DATA_BASE" "$READ_PROMPT_DATA_BASE" "$WRITE_PROMPT_DATA_BASE"; do
  if [ ! -f "$f" ]; then
    echo "[run.sh] ERROR: missing prompt-data file: $f" >&2
    echo "         regenerate with: WebArena_SRC=~/ref/BrowserGym uv run python devs/exps/eval/browsergym.webarena/export_template_tasks.py" >&2
    exit 1
  fi
done

ALL_PROMPT_DATA="$ALL_PROMPT_DATA_BASE"
READ_PROMPT_DATA="$READ_PROMPT_DATA_BASE"
WRITE_PROMPT_DATA="$WRITE_PROMPT_DATA_BASE"
if [[ "$READ_EXTRA" != "True" || "$WRITE_EXTRA" != "True" ]]; then
  ALL_PROMPT_DATA="$LOG_ROOT/.webarena_241_templates.filtered.all.prompt_data.parquet"
  READ_PROMPT_DATA="$LOG_ROOT/.webarena_241_templates.filtered.read.prompt_data.parquet"
  WRITE_PROMPT_DATA="$LOG_ROOT/.webarena_241_templates.filtered.write.prompt_data.parquet"
  echo "[run.sh] deriving filtered prompt-data under $LOG_ROOT"
  WA_ENV_ID="$EVAL_ENV_ID" \
  WA_READ_FILTER="$READ_EXTRA" WA_WRITE_FILTER="$WRITE_EXTRA" \
  WA_ALL_IN="$ALL_PROMPT_DATA_BASE" WA_READ_IN="$READ_PROMPT_DATA_BASE" WA_WRITE_IN="$WRITE_PROMPT_DATA_BASE" \
  WA_ALL_OUT="$ALL_PROMPT_DATA" WA_READ_OUT="$READ_PROMPT_DATA" WA_WRITE_OUT="$WRITE_PROMPT_DATA" \
    uv run --no-sync python - <<'PY'
import os
from pathlib import Path

import pandas as pd

import lite.gym as gym
from lite.core.utils.filters import parse_filter
from lite.utils.parquet import write_records_to_parquet

env_id = os.environ["WA_ENV_ID"]
keep_read = parse_filter(f"lambda m: ({os.environ.get('WA_READ_FILTER') or 'True'})")
keep_write = parse_filter(f"lambda m: ({os.environ.get('WA_WRITE_FILTER') or 'True'})")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    return value


def load_records(path):
    return [clean(row) for row in pd.read_parquet(path).to_dict("records")]


def keep(row, fn):
    env_key = row["metadata"]["env_key"]
    row_env_id, task_id = env_key.split("@", 1)
    if row_env_id != env_id:
        raise RuntimeError(f"unexpected env_key {env_key!r}; expected env_id {env_id!r}")
    return fn(gym.registry.task_metadata(row_env_id, task_id))


read_rows = [row for row in load_records(os.environ["WA_READ_IN"]) if keep(row, keep_read)]
write_rows = [row for row in load_records(os.environ["WA_WRITE_IN"]) if keep(row, keep_write)]
all_rows = read_rows + write_rows

for rows, out in (
    (read_rows, os.environ["WA_READ_OUT"]),
    (write_rows, os.environ["WA_WRITE_OUT"]),
    (all_rows, os.environ["WA_ALL_OUT"]),
):
    if not rows:
        raise RuntimeError(f"filter produced zero rows for {out}")
    write_records_to_parquet(rows, Path(out))

print(
    f"[prompt-data] read={len(read_rows)} write={len(write_rows)} "
    f"all={len(all_rows)}"
)
PY
fi
echo "         prompt_data_read=$READ_PROMPT_DATA"
echo "         prompt_data_write=$WRITE_PROMPT_DATA"

# Common rollout args shared by both passes. max_steps is carried by the config's
# env_kwargs, so nothing is overridden here.
COMMON=(
  --model-id "$MODEL"
  --env-id browsergym.webarena
  --config-path "$CFG"
  --log-root "$LOG_ROOT"
  "${EXTRA_ROLLOUT_ARGS[@]}"
)

# Pass 1 — READ: non-mutating tasks, fully parallel, residue-immune.
echo "[run.sh] pass 1/3: READ (non-mutating, concurrency=$CONCURRENCY)"
HF_HUB_OFFLINE=1 uv run python scripts/rollout.py \
  "${COMMON[@]}" \
  --concurrency "$CONCURRENCY" \
  --prompt-data "$READ_PROMPT_DATA"

# Pass 2 — WRITE: mutating tasks, serial, in task-id (= depends_on) order.
echo "[run.sh] pass 2/3: WRITE (mutating, concurrency=1)"
HF_HUB_OFFLINE=1 uv run python scripts/rollout.py \
  "${COMMON[@]}" \
  --concurrency 1 \
  --prompt-data "$WRITE_PROMPT_DATA"

# Pass 3 — RECONCILE: rebuild ONE combined summary.json over all prompt-data
# rows that were supposed to run. The read/write passes each summarize only
# their own subset.
echo "[run.sh] pass 3/3: RECONCILE combined summary over template subset"
WA_ENV_ID="$EVAL_ENV_ID" WA_LOG_ROOT="$LOG_ROOT" WA_MODEL="$MODEL" WA_ALL_PROMPT_DATA="$ALL_PROMPT_DATA" \
  uv run --no-sync python - <<'PY'
import os
from pathlib import Path

from lite.infer.rollout import (
    print_results,
    rebuild_results,
    resolve_prompt_data_tasks,
    save_summary,
)

env_id = os.environ["WA_ENV_ID"]
log_root = Path(os.environ["WA_LOG_ROOT"])
specs = resolve_prompt_data_tasks(
    os.environ["WA_ALL_PROMPT_DATA"],
    effective_env_id=env_id,
)
results = rebuild_results(log_root, specs, 1)
stats = print_results(results, specs, group_size=1)
save_summary(
    log_root / "summary.json",
    results=results,
    stats=stats,
    specs=specs,
    model=os.environ["WA_MODEL"],
    env_id=env_id,
    splits=["eval"],
    group_size=1,
)
print(f"[reconcile] combined summary over {len(specs)} tasks -> {log_root / 'summary.json'}")
PY
