#!/bin/bash

#
# CUA-Lite GRPO training on Slime.
#
# Purpose:
#   Run online GRPO for GUI agents. The script resolves a local VLM checkpoint,
#   probes the env-server, starts Ray/Slime, launches rollout workers, and
#   submits the Megatron training job.
#
# Required:
#   Run inside the Slime container. Set ENV_ID and PROMPT_DATA. For server-backed
#   envs, start the env-server first and export CUA_LITE_ENV_SERVER_URL +
#   CUA_LITE_ENV_SERVER_TOKEN before launching.
#
# Common overrides:
#   Inputs:
#     MODEL_ID, HF_CKPT, ENV_ID, PROMPT_DATA, EVAL_PROMPT_DATA, CONFIG_PATH
#   Capacity:
#     ASYNC, NUM_TRAIN_GPUS, NUM_ROLLOUT_GPUS, GPUS_PER_ENGINE, ENV_CONCURRENCY
#   Training:
#     ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, NUM_STEPS_PER_ROLLOUT, LR,
#     TP_SIZE, MBS, MAX_TOKENS_PER_GPU
#   Output/logging:
#     SAVE, SAVE_DIR, SAVE_HF_DIR, DUMP, DUMP_DIR, WANDB_API_KEY
#   Advanced:
#     ROLLOUT_MODULE, ADV_ESTIMATOR, EPS_CLIP, EPS_CLIP_HIGH,
#     DROP_ZERO_STD_GROUP, MEM_FRACTION, GRADS_FP32, OPTIM_CPU_OFFLOAD
#   Debug:
#     DEBUG, CUA_LITE_SHELL_TRACE, CUA_LITE_ROLLOUT_PROC_WORKERS,
#     ROLLOUT_STALL_TIMEOUT_S
#
# Related launchers:
#   run_gspo.sh wraps this script with GSPO-specific defaults.
#   run_reinforce.sh and run_dagger.sh share the online rollout shape.
#   run_sft.sh is the offline SFT path.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_TRAIN_GPUS=4 \
#     ENV_ID=lite.osworld \
#     PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/train.parquet \
#     CUA_LITE_ENV_SERVER_URL=http://<env-server-host>:<port> \
#     CUA_LITE_ENV_SERVER_TOKEN=<token> \
#     bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# =============================================================================================== #
#                                            Bootstrap                                            #
# =============================================================================================== #

# Abort on error. Shell tracing is a separate opt-in because xtrace expands
# runtime-env JSON and can expose env-server/W&B credentials in logs.
set -e
if [ "${CUA_LITE_SHELL_TRACE:-0}" = "1" ]; then
   echo "Warning: CUA_LITE_SHELL_TRACE=1 may print credentials in shell logs." >&2
   set -x
elif [ "${DEBUG:-0}" = "1" ]; then
   echo "DEBUG=1 is kept for downstream code; use CUA_LITE_SHELL_TRACE=1 for shell xtrace." >&2
fi
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CUA_LITE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
UTILS_DIR="${CUA_LITE_ROOT}/scripts/train/utils"

source "${UTILS_DIR}/cleanup.sh"

# =============================================================================================== #
#                                 Resolve Inputs And Run Identity                                 #
# =============================================================================================== #

# -- Operator inputs -----------------------------------------------------------

MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-VL-4B-Instruct"}
ENV_ID=${ENV_ID:?"ENV_ID is required"}  # used for wandb group name and eval dataset label
PROMPT_DATA=${PROMPT_DATA:?"PROMPT_DATA is required"}
EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA:-""}

# -- GPU topology --------------------------------------------------------------

NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-4}
NUM_ROLLOUT_GPUS=${NUM_ROLLOUT_GPUS:-${NUM_TRAIN_GPUS}}
ASYNC=${ASYNC:-0}
if [ "${ASYNC}" = "1" ]; then
   SYNC_MODE=async
   NUM_GPUS=$(( NUM_TRAIN_GPUS + NUM_ROLLOUT_GPUS ))
else
   SYNC_MODE=sync
   if [ "${NUM_ROLLOUT_GPUS}" != "${NUM_TRAIN_GPUS}" ]; then
      echo "Error: sync/colocate mode requires NUM_ROLLOUT_GPUS=NUM_TRAIN_GPUS." >&2
      echo "       Set ASYNC=1 for separate rollout GPUs." >&2
      exit 1
   fi
   NUM_GPUS=${NUM_TRAIN_GPUS}  # colocate: train and rollout share GPUs
fi
if [ "${USE_TIS:-0}" != "0" ]; then
   echo "Error: USE_TIS has been retired from this launcher; do not pass it." >&2
   exit 1
fi
if [ -n "${CUA_LITE_ENV_SERVER_URL:-}" ] && [ -z "${CUA_LITE_ENV_SERVER_TOKEN:-}" ]; then
   echo "Error: CUA_LITE_ENV_SERVER_TOKEN is required when CUA_LITE_ENV_SERVER_URL is set." >&2
   exit 1
fi
SESSION_ID=${SESSION_ID:-"manual-$(date +%Y%m%d_%H%M%S)-$$"}

# -- Model family and run identity --------------------------------------------

# Sets MODEL_ARGS_FILE / MODEL_FAMILY and shared launch-shape helpers.
source "${UTILS_DIR}/models.sh"

_require_positive_int() {
   local name=$1 value=$2
   if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: ${name}=${value} must be a positive integer." >&2
      exit 1
   fi
}

ALGO=${ALGO:-grpo}
RUN_REL="${MODEL_SLUG}/${ENV_ID}/${ALGO}/${SYNC_MODE}"
DEFAULT_CONFIG_PATH="${CUA_LITE_ROOT}/scripts/configs/${MODEL_FAMILY}/compact/${ENV_ID}.yaml"
CONFIG_PATH=${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}
if [ ! -f "${CONFIG_PATH}" ]; then
   echo "Error: CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
   echo "       Set CONFIG_PATH explicitly, or choose an ENV_ID with a compact training config." >&2
   exit 1
fi

# =============================================================================================== #
#                                   Preflight And Local Assets                                    #
# =============================================================================================== #

# -- Env-server preflight ------------------------------------------------------

# Probe ENV_ID availability and clean prior-session leftovers before starting
# Ray. Full fail-fast rationale lives in scripts/train/utils/preflight.sh.
source "${UTILS_DIR}/preflight.sh"

# -- Hardware probe ------------------------------------------------------------

# Controls NCCL_NVLS_ENABLE in the Ray worker runtime env.
source "${UTILS_DIR}/nvlink.sh"

# -- Model checkpoint ----------------------------------------------------------

# HF_CKPT override skips download (e.g. feed an SFT'd ckpt back into RL).
HF_CKPT_DEFAULT="/root/models/${MODEL_ID}"
HF_CKPT=${HF_CKPT:-"${HF_CKPT_DEFAULT}"}
if [ "${HF_CKPT}" = "${HF_CKPT_DEFAULT}" ]; then
   mkdir -p /root/models
   if [ ! -d "${HF_CKPT_DEFAULT}" ]; then
      hf download "${MODEL_ID}" --local-dir "${HF_CKPT_DEFAULT}"
   fi
fi

# =============================================================================================== #
#                                 Persistence And Debug Artifacts                                 #
# =============================================================================================== #

# -- Checkpoints ---------------------------------------------------------------

# --hf-checkpoint: tokenizer + sglang init + initial weights (first run)
# --load:          only set when SAVE_DIR has a megatron checkpoint
# --save:          only set when SAVE=1 (default: 1)
# NO_SAVE_OPTIM=1 default: HF weights only (~10× smaller than full Megatron+optim);
# not resumable — flip to 0 to keep the optimizer state if you need resume.
# One canonical run path (model-first, model_slug) → checkpoints + wandb share it.
SAVE_DIR=${SAVE_DIR:-"/root/checkpoints/${RUN_REL}/megatron"}
SAVE_HF_DIR=${SAVE_HF_DIR:-"/root/checkpoints/${RUN_REL}/hf/iter_{rollout_id}"}
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
# Build CKPT_ARGS from SAVE_DIR/SAVE_HF_DIR/HF_CKPT + SAVE/NO_SAVE_OPTIM/RESUME (shared helper).
source "${UTILS_DIR}/ckpt_args.sh"

# -- Rollout dumps -------------------------------------------------------------

# DUMP_DIR defaults under /tmp so accidental traces do not fill shared repo
# mounts. Set DUMP_DIR explicitly when you want persistent rollout traces.
DUMP_DIR=${DUMP_DIR:-"/tmp/cua-lite/${ALGO}/${MODEL_SLUG}/${ENV_ID}/$(date +%Y%m%d_%H%M%S)"}
if [ "${DUMP:-0}" = "1" ]; then
   DUMP_ARGS=(--dump-details "${DUMP_DIR}")
   echo "Dumping rollout details to ${DUMP_DIR}"
else
   DUMP_ARGS=()
   echo "Not dumping rollout details (set DUMP=1 to enable)"
fi

# =============================================================================================== #
#                                     Rollout And Evaluation                                      #
# =============================================================================================== #

# -- Rollout batch shape -------------------------------------------------------

N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT:-8}
_require_positive_int "N_SAMPLES_PER_PROMPT" "${N_SAMPLES_PER_PROMPT}"
_require_positive_int "ROLLOUT_BATCH_SIZE" "${ROLLOUT_BATCH_SIZE}"
_require_positive_int "NUM_STEPS_PER_ROLLOUT" "${NUM_STEPS_PER_ROLLOUT}"

# slime v0.3.0 derives global_batch_size from --num-steps-per-rollout
# (GBS = ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT // NUM_STEPS_PER_ROLLOUT);
# do NOT also pass --global-batch-size (slime asserts they agree).
# Default 8: minibatch-split the single-pass rollout into 8 optimizer steps.
# Data is consumed once either way; k only sets update granularity. Steps 2..k
# are off-policy with respect to the sampled data, bounded by eps-clip and
# dual-clip. Set =1 for strictly on-policy.
# Preflight: k must divide rbs*nsp (else slime floor-drops trailing GROUPS
# silently, or runs a different step count) and must not exceed it (else
# GBS=0 -> bare ZeroDivisionError deep in slime).
_N_TRAJ=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if [ "${NUM_STEPS_PER_ROLLOUT}" -lt 1 ] || [ "${NUM_STEPS_PER_ROLLOUT}" -gt "${_N_TRAJ}" ] \
   || [ $((_N_TRAJ % NUM_STEPS_PER_ROLLOUT)) -ne 0 ]; then
   echo "Error: NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT} must be a divisor of" \
        "ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT=${_N_TRAJ} (and within [1, ${_N_TRAJ}])," \
        "otherwise slime silently drops trailing trajectory groups." \
        "For smoke runs / tiny batches, set NUM_STEPS_PER_ROLLOUT=1 explicitly." >&2
   exit 1
fi

# -- Rollout entrypoints -------------------------------------------------------

# Rollout entry points. slime resolves them by dotted string, so ROLLOUT_MODULE swaps
# all three at once: an examples/ recipe exports it to point at its own shim module
# (whose import registers the recipe's adapter/agent on each Ray actor) instead of
# forking or text-rewriting this script.
ROLLOUT_MODULE=${ROLLOUT_MODULE:-lite.train.rollout.grpo}
ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key problem
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-1000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   # Short agentic turns are usually ~100 tokens; this caps runaway loops that
   # OOM logits. Raise for long-CoT agents.
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-512}"
   # 1.0 follows Slime convention. Lower for precise multi-turn agents where
   # per-step error compounds.
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --custom-generate-function-path "${ROLLOUT_MODULE}.generate"
   --custom-convert-samples-to-train-data-path "${ROLLOUT_MODULE}.convert_samples_to_train_data"
   --rollout-function-path "${ROLLOUT_MODULE}.generate_rollout"
   --custom-config-path "${CONFIG_PATH}"
   --multimodal-lazy-expand-fn-path lite.train.utils.multimodal_expand.expand
)

# -- Evaluation ----------------------------------------------------------------

# RL defaults SKIP_EVAL_BEFORE_TRAIN=0 (don't skip) — baseline is a meaningful
# reference for the reward trajectory. SFT defaults to 1 (skip).
# EVAL_TEMPERATURE defaults to 0: a deterministic score, comparable across runs
# and matching how the agent is deployed. slime instead defaults eval to the
# rollout temperature (slime/utils/eval_config.py), which is what GRPO actually
# maximises -- set EVAL_TEMPERATURE=1 to score the optimised objective, and
# raise N_SAMPLES_PER_EVAL_PROMPT with it, since sampled scores need several
# draws per task to stay as sensitive as the greedy one.
if [ -n "$EVAL_PROMPT_DATA" ]; then
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL:-5}"
      --eval-prompt-data "${ENV_ID}_eval" "${EVAL_PROMPT_DATA}"
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
      --eval-temperature "${EVAL_TEMPERATURE:-0}"
      --eval-top-p 1
   )
   if [ "${SKIP_EVAL_BEFORE_TRAIN:-0}" != "0" ]; then
      EVAL_ARGS+=(--skip-eval-before-train)
   fi
else
   EVAL_ARGS=()
fi

# =============================================================================================== #
#                                       Training Arguments                                        #
# =============================================================================================== #

# -- GRPO objective ------------------------------------------------------------

# eps-clip-c: Dual-clip PPO (arxiv 1912.09729) caps loss in the negative-A +
# large-ratio case (PPO clip's blind spot) at c × |A| instead of ratio × |A|.
# Default 3.0 follows the paper. (Always applied; `EPS_CLIP_C=` empty still
# resolves to 3.0 via `:-`. To disable dual-clip, remove the --eps-clip-c flag —
# slime's compute_policy_loss treats eps_clip_c=None as single-clip PPO.)
GRPO_ARGS=(
   # ADV_ESTIMATOR + EPS_CLIP[_HIGH] are overridable so run_gspo.sh can switch to
   # sequence-level GSPO with its much tighter clip. GRPO defaults are unchanged:
   # grpo + 0.2/0.28 (DAPO clip-higher).
   --advantage-estimator "${ADV_ESTIMATOR:-grpo}"
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip "${EPS_CLIP:-0.2}"
   --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
   --eps-clip-c "${EPS_CLIP_C:-3.0}"
)

# -- Optimizer -----------------------------------------------------------------

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# OPTIM_CPU_OFFLOAD=1 moves the fp32 Adam state to host RAM. Use it when
# distributed optimizer sharding is insufficient on constrained GPUs.
if [ "${OPTIM_CPU_OFFLOAD:-0}" = "1" ]; then
   OPTIMIZER_ARGS+=(--optimizer-cpu-offload --use-precision-aware-optimizer)
fi

# DIST_OPTIM=1 shards fp32 Adam state + master params across DP ranks by default.
if [ "${DIST_OPTIM:-1}" = "1" ]; then
   OPTIMIZER_ARGS+=(--use-distributed-optimizer)
fi

# -- Rollout inference ---------------------------------------------------------

GPUS_PER_ENGINE=${GPUS_PER_ENGINE:-1}
NUM_ENGINES=$(resolve_num_engines "${GPUS_PER_ENGINE}" "${NUM_ROLLOUT_GPUS}")
ENV_CONCURRENCY=${ENV_CONCURRENCY:-32}
# Per-engine server concurrency is ceil(ENV_CONCURRENCY / NUM_ENGINES).
SERVER_CONCURRENCY=$(( (ENV_CONCURRENCY + NUM_ENGINES - 1) / NUM_ENGINES ))
SGLANG_CUDA_GRAPH_BS=(
   1 2 4 8
   16 24 32 40 48 56 64 72
   80 88 96 104 112 120 128
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${MEM_FRACTION:-0.6}"
   --sglang-cuda-graph-bs "${SGLANG_CUDA_GRAPH_BS[@]}"
   --sglang-server-concurrency "${SERVER_CONCURRENCY}"
   --router-policy "${ROUTER_POLICY:-round_robin}"
)

# -- W&B -----------------------------------------------------------------------

if [ -n "${WANDB_API_KEY:-}" ]; then
   WANDB_PROJECT=${WANDB_PROJECT_OVERRIDE:-cua-lite-dev}
   WANDB_GROUP=${WANDB_GROUP_OVERRIDE:-${RUN_REL}${WANDB_GROUP_SUFFIX:-}}
   WANDB_ARGS=(
      --use-wandb
      # dev bucket; WANDB_PROJECT_OVERRIDE=cua-lite for curated/public runs
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP}"
      --wandb-key "${WANDB_API_KEY}"
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

# -- Slime resources -----------------------------------------------------------

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_TRAIN_GPUS}"
   --rollout-num-gpus "${NUM_ROLLOUT_GPUS}"
)

if [ "${ASYNC}" != "1" ]; then
   MISC_ARGS+=(--colocate)
fi

# =============================================================================================== #
#                                        Megatron Backend                                         #
# =============================================================================================== #

# TP: default 1 (pure DP + ZeRO-1); raise TP_SIZE explicitly on OOM. resolve_tp
# defaults + validates divisibility (models.sh).
TP_SIZE=$(resolve_tp "${TP_SIZE:-}" "${NUM_TRAIN_GPUS}")
DP_SIZE=$(( NUM_TRAIN_GPUS / TP_SIZE ))
printf 'Resolved parallelism: TP=%s PP=1 CP=1 EP=1 DP=%s\n' \
   "${TP_SIZE}" "${DP_SIZE}"
printf '  train_gpus=%s model_family=%s\n' \
   "${NUM_TRAIN_GPUS}" "${MODEL_FAMILY}"

BACKEND_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size "${TP_SIZE}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

if [ "${TP_SIZE}" -gt 1 ]; then
   BACKEND_ARGS+=(--sequence-parallel)
fi

if [ "${RECOMPUTE:-1}" = "1" ]; then
   BACKEND_ARGS+=(
      --recompute-granularity full
      --recompute-method "${RECOMPUTE_METHOD:-uniform}"
      --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
   )
fi

# fp32 grad accumulation is on by default. Set GRADS_FP32=0 for bf16 grads to
# save optimizer memory on constrained GPUs.
if [ "${GRADS_FP32:-1}" = "1" ]; then
   BACKEND_ARGS+=(--accumulate-allreduce-grads-in-fp32)
fi

# Default: BSHD + fixed MBS for predictable VLM memory. Set
# MAX_TOKENS_PER_GPU to opt into THD/dynamic packing. Qwen3.5 rejects THD
# because GDN cannot preserve sequence boundaries across packed samples.
if [ -n "${MAX_TOKENS_PER_GPU:-}" ]; then
   if [ "${MODEL_FAMILY}" = "qwen3_5" ]; then
      echo "ERROR: Qwen3.5 (GDN) cannot use THD packing — unset MAX_TOKENS_PER_GPU and use MBS (BSHD)." >&2
      exit 1
   fi
   BACKEND_ARGS+=(
      --use-dynamic-batch-size
      --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   )
else
   MBS=${MBS:-1}
   BACKEND_ARGS+=(--qkv-format bshd --micro-batch-size "${MBS}")
fi

# =============================================================================================== #
#                               Launch: Ray Runtime And Job Submit                                #
# =============================================================================================== #

# -- Model arguments -----------------------------------------------------------

SLIME_DIR="${CUA_LITE_ROOT}/slime"
source "${CUA_LITE_ROOT}/scripts/train/models/${MODEL_ARGS_FILE}.sh"

# -- Ray startup ---------------------------------------------------------------

source "${UTILS_DIR}/ray.sh"
source "${UTILS_DIR}/runtime_env.sh"
start_ray

# -- Worker runtime env --------------------------------------------------------

# CUA_LITE_ENV_SERVER_{URL,TOKEN}: LiteEnvClient reads these to reach the
# env server during rollout (see docs/slime.md + docs/envs.md).
# CUA_LITE_REGISTRATION_MODULES registers out-of-tree direct-mode envs in Ray
# workers; the driver/preflight shell is not enough because workers do not
# inherit arbitrary launcher env.
# Segmenter/rollout knobs must be forwarded explicitly: Ray workers do not
# inherit the driver shell, and GSPO depends on CUA_LITE_DISABLE_RADIX=1.
RUNTIME_ENV_JSON="$(build_runtime_env_json \
  "PYTHONPATH=/root/Megatron-LM/:${CUA_LITE_ROOT}:${SLIME_DIR}" \
  "CUDA_DEVICE_MAX_CONNECTIONS=1" \
  "NCCL_NVLS_ENABLE=${HAS_NVLINK}" \
  "MEGATRON_CONFIG_LOCK_DIR=${MEGATRON_CONFIG_LOCK_DIR:-/tmp}" \
  "SESSION_ID=${SESSION_ID}" \
  "CUA_LITE_ENV_SERVER_URL=${CUA_LITE_ENV_SERVER_URL}" \
  "CUA_LITE_ENV_SERVER_TOKEN=${CUA_LITE_ENV_SERVER_TOKEN}" \
  "CUA_LITE_REGISTRATION_MODULES=${CUA_LITE_REGISTRATION_MODULES:-}" \
  "CUA_LITE_DISABLE_RADIX=${CUA_LITE_DISABLE_RADIX:-0}" \
  "CUA_LITE_MULTIMODAL_LAZY_EXPAND=${CUA_LITE_MULTIMODAL_LAZY_EXPAND:-0}" \
  "CUA_LITE_MULTIMODAL_FP32=${CUA_LITE_MULTIMODAL_FP32:-0}" \
  "CUA_LITE_ROLLOUT_PROC_WORKERS=${CUA_LITE_ROLLOUT_PROC_WORKERS:-8}" \
  "ROLLOUT_STALL_TIMEOUT_S=${ROLLOUT_STALL_TIMEOUT_S:-1800}" \
  "DROP_ZERO_STD_GROUP=${DROP_ZERO_STD_GROUP:-1}" \
)"

# -- Submit --------------------------------------------------------------------

# ASYNC=1 uses train_async.py (separate rollout/train GPUs); else train.py (colocate).
TRAIN_ENTRYPOINT="${SLIME_DIR}/train.py"
if [ "${ASYNC}" = "1" ]; then
   TRAIN_ENTRYPOINT="${SLIME_DIR}/train_async.py"
fi

ALGO_LABEL=${ALGO^^}

printf '\n'
printf 'Launching CUA-Lite %s\n' "${ALGO_LABEL}"
printf '  model=%s env=%s mode=%s\n' "${MODEL_ID}" "${ENV_ID}" "${SYNC_MODE}"
printf '  rollout=batch:%s samples:%s steps:%s module:%s\n' \
   "${ROLLOUT_BATCH_SIZE}" "${N_SAMPLES_PER_PROMPT}" \
   "${NUM_STEPS_PER_ROLLOUT}" "${ROLLOUT_MODULE}"
printf '  objective=%s eps:%s/%s eps_c:%s radix:%s\n' \
   "${ADV_ESTIMATOR:-grpo}" "${EPS_CLIP:-0.2}" "${EPS_CLIP_HIGH:-0.28}" \
   "${EPS_CLIP_C:-3.0}" "${CUA_LITE_DISABLE_RADIX:-0}"
printf '  config=%s\n' "${CONFIG_PATH}"
printf '  checkpoints=%s\n' "${SAVE_DIR}"
printf '  ray_dashboard=http://127.0.0.1:%s\n' "${RAY_PORT}"
printf '\n'

# Everything above builds arguments and Ray runtime env; this is the external
# launch action.
ray job submit --address="http://127.0.0.1:${RAY_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${TRAIN_ENTRYPOINT}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${BACKEND_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DUMP_ARGS[@]}"
