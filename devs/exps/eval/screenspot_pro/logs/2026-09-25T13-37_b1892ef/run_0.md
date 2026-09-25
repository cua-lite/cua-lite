# screenspot_pro @ 2026-09-25T13-37_b1892ef - run_0

- **Commit**: `b1892ef` - Medium API grounding defaults.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/screenspot_pro/2026-09-25T13-37_b1892ef/run_0/` (gitignored).
- **Started**: `2026-09-25 PDT`.
- **Last updated**: `2026-09-25 14:02 PDT`.
- **Notes**: `eval`, 1581 tasks; 20 tasks per API model, at most 60 total.
- **Provenance**: [JSON snapshot](/devs/exps/eval/screenspot_pro/logs/2026-09-25T13-37_b1892ef/run_0.json) records per-model source paths, configurations and hashes. Three fresh Medium API runs; other model results retain their recorded sources.

## Results

| Model | Config | Finished | Mean episode return |
|---|---|---:|---:|
| `gpt-6-astra` | medium | 1581/1581 | 0.938646 |
| `claude-opus-5` | default | 1581/1581 | 0.852625 |
| `gpt-5.6-sol` | default | 1581/1581 | 0.824794 |
| `Qwen/Qwen3.8-27B` | default | 1581/1581 | 0.753953 |
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
| ⚠️ `gpt-5.5` | default | _**1570/1581**_ | _**0.744586**_ |
| ⚠️ `gemini-3.6-flash` | default | _**0/1581**_ | _**—**_ |

Mean episode return excludes API errors and includes valid zero-reward predictions.
Finished counts valid samples; incomplete rows retain the full task count.

## Highlights

- gpt-5.5: 11 API errors excluded; default screenshot resolution retained.
- gemini-3.6-flash: not started.

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
| GPT-5.5 / GPT-5.6 Sol | [gpt/screenspot_pro.yaml](/scripts/configs/gpt/default/screenspot_pro.yaml) | Reasoning medium; 4096 output tokens |
| GPT-6 Astra (Medium) | [gpt/screenspot_pro.medium.yaml](/scripts/configs/gpt/default/screenspot_pro.medium.yaml) | Reasoning medium; 4096 output tokens |
| Claude Opus 5 | [claude/screenspot_pro.yaml](/scripts/configs/claude/default/screenspot_pro.yaml) | Effort medium; 1024 output tokens; no explicit thinking budget |

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

Use the model ID from the specification for other rows. The runner selects each family's default YAML; Astra's retained Medium variant is equivalent in reasoning settings.

## Breakdown

Values cover valid predictions only. Overlapping categories count each category membership.

### Breakdown — by `group`

| Model | `CAD` | `Creative` | `Dev` | `OS` | `Office` | `Scientific` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gpt-6-astra` | 0.9349 | 0.9208 | 0.9298 | 0.9490 | 0.9826 | 0.9291 | 0.9386 |
| `claude-opus-5` | 0.8046 | 0.8358 | 0.8495 | 0.8673 | 0.9478 | 0.8307 | 0.8526 |
| `gpt-5.6-sol` | 0.8276 | 0.7947 | 0.7726 | 0.7806 | 0.9826 | 0.8150 | 0.8248 |
| `Qwen/Qwen3.8-27B` | 0.6667 | 0.6833 | 0.7960 | 0.7755 | 0.8913 | 0.7480 | 0.7540 |
| `gpt-5.5` | 0.8123 | 0.7419 | 0.6355 | 0.6162 | 0.9000 | 0.7598 | 0.7446 |
| `Qwen/Qwen3.5-27B` | 0.6667 | 0.6276 | 0.6957 | 0.7143 | 0.8435 | 0.6654 | 0.6951 |
| `inclusionAI/UI-Venus-2-9B` | 0.6513 | 0.5748 | 0.6421 | 0.6480 | 0.7522 | 0.6654 | 0.6496 |
| `Tongyi-MAI/MAI-UI-8B` | 0.6015 | 0.5543 | 0.5987 | 0.5969 | 0.8348 | 0.5906 | 0.6224 |
| `Qwen/Qwen3-VL-4B-Instruct` | 0.4981 | 0.5220 | 0.5619 | 0.6122 | 0.7565 | 0.5945 | 0.5825 |
| `Qwen/Qwen3.5-9B` | 0.5211 | 0.4633 | 0.6455 | 0.5510 | 0.7348 | 0.6102 | 0.5813 |
| `Qwen/Qwen3.5-4B` | 0.5134 | 0.5220 | 0.5619 | 0.5765 | 0.7391 | 0.5906 | 0.5775 |
| `Qwen/Qwen3-VL-32B-Instruct` | 0.5287 | 0.5777 | 0.4883 | 0.4643 | 0.7783 | 0.6339 | 0.5769 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.5019 | 0.4839 | 0.5217 | 0.5204 | 0.7348 | 0.5906 | 0.5522 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 0.4444 | 0.4457 | 0.4515 | 0.4031 | 0.7261 | 0.5433 | 0.4978 |
| `meituan/EvoCUA-8B-20260105` | 0.2759 | 0.4164 | 0.4214 | 0.3929 | 0.6261 | 0.5000 | 0.4352 |

### Breakdown — by `ui_type`

| Model | `icon` | `text` | Avg |
|---|---:|---:|---:|
| `gpt-6-astra` | 0.9007 | 0.9621 | 0.9386 |
| `claude-opus-5` | 0.7334 | 0.9263 | 0.8526 |
| `gpt-5.6-sol` | 0.7219 | 0.8884 | 0.8248 |
| `Qwen/Qwen3.8-27B` | 0.5944 | 0.8526 | 0.7540 |
| `gpt-5.5` | 0.6260 | 0.8177 | 0.7446 |
| `Qwen/Qwen3.5-27B` | 0.4801 | 0.8280 | 0.6951 |
| `inclusionAI/UI-Venus-2-9B` | 0.4785 | 0.7554 | 0.6496 |
| `Tongyi-MAI/MAI-UI-8B` | 0.3543 | 0.7881 | 0.6224 |
| `Qwen/Qwen3-VL-4B-Instruct` | 0.3013 | 0.7564 | 0.5825 |
| `Qwen/Qwen3.5-9B` | 0.3742 | 0.7093 | 0.5813 |
| `Qwen/Qwen3.5-4B` | 0.3245 | 0.7339 | 0.5775 |
| `Qwen/Qwen3-VL-32B-Instruct` | 0.2930 | 0.7523 | 0.5768 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.2318 | 0.7503 | 0.5522 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 0.2119 | 0.6745 | 0.4978 |
| `meituan/EvoCUA-8B-20260105` | 0.1623 | 0.6039 | 0.4352 |
