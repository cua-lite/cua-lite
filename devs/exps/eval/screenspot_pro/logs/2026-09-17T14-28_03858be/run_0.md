# screenspot_pro @ 2026-09-17T14-28_03858be - run_0

- **Commit**: `03858be` - reference models.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/screenspot_pro/2026-09-17T14-28_03858be/run_0/` (gitignored).
- **Started**: `2026-09-12 PDT`.
- **Last updated**: `2026-09-25 00:43 PDT`.
- **Notes**: `eval`, all 1,581 tasks. Task concurrency 1-30, at most 60 across the host.
- **Provenance**: [JSON snapshot](/devs/exps/eval/screenspot_pro/logs/2026-09-17T14-28_03858be/run_0.json) records each model's source run, configuration, commits, and artifact checksums. The directory key identifies this report; source runs span multiple commits. Claude source runs include recorded uncommitted patches.

## Results

| Model | Config | Finished | Mean episode return |
|---|---|---:|---:|
| `gpt-6-astra` | medium | 1581/1581 | 0.938646 |
| `claude-opus-5` | default | 1581/1581 | 0.845667 |
| `Qwen/Qwen3.8-27B` | default | 1581/1581 | 0.753953 |
| `gpt-5.6-sol` | default | 1581/1581 | 0.723593 |
| `Qwen/Qwen3.5-27B` | default | 1581/1581 | 0.695130 |
| `inclusionAI/UI-Venus-2-9B` | default | 1581/1581 | 0.649589 |
| `Tongyi-MAI/MAI-UI-8B` | default | 1581/1581 | 0.622391 |
| `Qwen/Qwen3-VL-4B-Instruct` | default | 1581/1581 | 0.582543 |
| `Qwen/Qwen3.5-9B` | default | 1581/1581 | 0.581278 |
| `Qwen/Qwen3.5-4B` | default | 1581/1581 | 0.577483 |
| `Qwen/Qwen3-VL-32B-Instruct` | default | 1581/1581 | 0.576850 |
| `Qwen/Qwen3-VL-8B-Instruct` | default | 1581/1581 | 0.552182 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | default | 1581/1581 | 0.497786 |
| `meituan/EvoCUA-8B-20260105` | default | 1581/1581 | 0.435168 |
| ⚠️ `gpt-5.5` | default | _**1571/1581**_ | _**0.450668**_ |
| ⚠️ `gemini-3.6-flash` | default | _**0/1581**_ | _**—**_ |

Mean episode return excludes API errors and includes valid zero-reward predictions.
Finished counts valid samples; incomplete rows retain the full task count.

## Highlights

- GPT-5.5: 10 fixed-frame validation errors after provider image downsampling; mean return is 708/1571 (45.07%).
- Gemini 3.6 Flash: not started.

## Experiment Specification

| Model | YAML | Sampling |
|---|---|---|
| Qwen3-VL-4B / 8B / 32B (Instruct) | [qwen3_vl/screenspot_pro.yaml](/scripts/configs/qwen3_vl/default/screenspot_pro.yaml) | Greedy; 2048 output tokens |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/screenspot_pro.yaml](/scripts/configs/qwen3_5/default/screenspot_pro.yaml) | Greedy; 2048 output tokens |
| Qwen3.8-27B | [qwen3_8/screenspot_pro.yaml](/scripts/configs/qwen3_8/default/screenspot_pro.yaml) | Greedy; 2048 output tokens |
| UI-TARS-1.5-7B | [ui_tars_15_v1/screenspot_pro.yaml](/scripts/configs/ui_tars_15_v1/default/screenspot_pro.yaml) | Greedy; 2048 output tokens |
| EvoCUA-8B | [evocua/screenspot_pro.yaml](/scripts/configs/evocua/default/screenspot_pro.yaml) | Greedy; 2048 output tokens |
| MAI-UI-8B | [mai_ui/screenspot_pro.yaml](/scripts/configs/mai_ui/default/screenspot_pro.yaml) | Greedy; 2048 output tokens |
| UI-Venus-2-9B | [ui_venus_2/screenspot_pro.yaml](/scripts/configs/ui_venus_2/default/screenspot_pro.yaml) | Greedy; 2048 output tokens |
| GPT-5.5 / GPT-5.6 Sol | [gpt/screenspot_pro.yaml](/scripts/configs/gpt/default/screenspot_pro.yaml) | Reasoning none; 4096 output tokens |
| GPT-6 Astra (Medium) | [gpt/screenspot_pro.medium.yaml](/scripts/configs/gpt/default/screenspot_pro.medium.yaml) | Reasoning medium; 4096 output tokens |
| Claude Opus 5 | [claude/screenspot_pro.yaml](/scripts/configs/claude/default/screenspot_pro.yaml) | Effort low; 1024 output tokens; no explicit thinking budget |

Each task supplies one screenshot and takes one click prediction using the
model's native grounding template and image preprocessing. Multi-turn history
windows do not apply. Qwen thinking remains disabled; UI-Venus uses its native
thinking template. No agent-level resolution override is set. Greedy decoding
does not use top-p.

### Reproduction

Prepare the dataset and a scoped env-server using the
[ScreenSpot-Pro guide](/lite/gym/envs/screenspot_pro/README.md).
Export `CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN`; configure
the API credentials for GPT runs and cache local model weights.

```bash
export EVAL_RUN_ID=run_0
export EVAL_CONCURRENCY=30
CUDA_VISIBLE_DEVICES=0 ./devs/exps/eval/screenspot_pro/run.sh Qwen/Qwen3-VL-8B-Instruct
./devs/exps/eval/screenspot_pro/run.sh gpt-5.6-sol
```

Use the model ID and YAML from the specification for the other rows. Claude
and Astra use the rollout entry directly:

```bash
uv run python scripts/rollout.py --model-id claude-opus-5 \
  --config-path scripts/configs/claude/default/screenspot_pro.yaml \
  --env-id screenspot_pro --splits eval --concurrency 6 \
  --log-root '.exps/eval/screenspot_pro/<commit-dir>/run_0/claude-opus-5'
uv run python scripts/rollout.py --model-id gpt-6-astra \
  --config-path scripts/configs/gpt/default/screenspot_pro.medium.yaml \
  --env-id screenspot_pro --splits eval --concurrency 6 \
  --log-root '.exps/eval/screenspot_pro/<commit-dir>/run_0/gpt-6-astra__medium'
```

Replace `<commit-dir>` with the timestamp and SHA of the code being evaluated.
The JSON source paths identify the retained results. Reusing a source log root
resumes that run; a new log root starts a separate run.

## Breakdown

Values cover valid predictions only.

### Breakdown — by `group`

| Model | `CAD` | `Creative` | `Dev` | `OS` | `Office` | `Scientific` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gpt-6-astra` | 0.9349 | 0.9208 | 0.9298 | 0.9490 | 0.9826 | 0.9291 | 0.9386 |
| `claude-opus-5` | 0.7778 | 0.8358 | 0.8595 | 0.8673 | 0.9435 | 0.8071 | 0.8457 |
| `Qwen/Qwen3.8-27B` | 0.6667 | 0.6833 | 0.7960 | 0.7755 | 0.8913 | 0.7480 | 0.7540 |
| `gpt-5.6-sol` | 0.6015 | 0.7302 | 0.6923 | 0.6939 | 0.9087 | 0.7323 | 0.7236 |
| `Qwen/Qwen3.5-27B` | 0.6667 | 0.6276 | 0.6957 | 0.7143 | 0.8435 | 0.6654 | 0.6951 |
| `inclusionAI/UI-Venus-2-9B` | 0.6513 | 0.5748 | 0.6421 | 0.6480 | 0.7522 | 0.6654 | 0.6496 |
| `Tongyi-MAI/MAI-UI-8B` | 0.6015 | 0.5543 | 0.5987 | 0.5969 | 0.8348 | 0.5906 | 0.6224 |
| `Qwen/Qwen3-VL-4B-Instruct` | 0.4981 | 0.5220 | 0.5619 | 0.6122 | 0.7565 | 0.5945 | 0.5825 |
| `Qwen/Qwen3.5-9B` | 0.5211 | 0.4633 | 0.6455 | 0.5510 | 0.7348 | 0.6102 | 0.5813 |
| `Qwen/Qwen3.5-4B` | 0.5134 | 0.5220 | 0.5619 | 0.5765 | 0.7391 | 0.5906 | 0.5775 |
| `Qwen/Qwen3-VL-32B-Instruct` | 0.5287 | 0.5777 | 0.4883 | 0.4643 | 0.7783 | 0.6339 | 0.5769 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.5019 | 0.4839 | 0.5217 | 0.5204 | 0.7348 | 0.5906 | 0.5522 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 0.4444 | 0.4457 | 0.4515 | 0.4031 | 0.7261 | 0.5433 | 0.4978 |
| `gpt-5.5` | 0.3946 | 0.5044 | 0.2943 | 0.2258 | 0.6522 | 0.6024 | 0.4507 |
| `meituan/EvoCUA-8B-20260105` | 0.2759 | 0.4164 | 0.4214 | 0.3929 | 0.6261 | 0.5000 | 0.4352 |

### Breakdown — by `ui_type`

| Model | `icon` | `text` | Avg |
|---|---:|---:|---:|
| `gpt-6-astra` | 0.9007 | 0.9621 | 0.9386 |
| `claude-opus-5` | 0.7301 | 0.9171 | 0.8457 |
| `Qwen/Qwen3.8-27B` | 0.5944 | 0.8526 | 0.7540 |
| `gpt-5.6-sol` | 0.5613 | 0.8240 | 0.7236 |
| `Qwen/Qwen3.5-27B` | 0.4801 | 0.8280 | 0.6951 |
| `inclusionAI/UI-Venus-2-9B` | 0.4785 | 0.7554 | 0.6496 |
| `Tongyi-MAI/MAI-UI-8B` | 0.3543 | 0.7881 | 0.6224 |
| `Qwen/Qwen3-VL-4B-Instruct` | 0.3013 | 0.7564 | 0.5825 |
| `Qwen/Qwen3.5-9B` | 0.3742 | 0.7093 | 0.5813 |
| `Qwen/Qwen3.5-4B` | 0.3245 | 0.7339 | 0.5775 |
| `Qwen/Qwen3-VL-32B-Instruct` | 0.2930 | 0.7523 | 0.5768 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.2318 | 0.7503 | 0.5522 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 0.2119 | 0.6745 | 0.4978 |
| `gpt-5.5` | 0.3038 | 0.5412 | 0.4507 |
| `meituan/EvoCUA-8B-20260105` | 0.1623 | 0.6039 | 0.4352 |
