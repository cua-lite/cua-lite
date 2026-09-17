# screenspot_pro @ 2026-09-17T14-28_03858be - run_0

- **Commit**: `03858be` - reference models and GPT reasoning variants.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/screenspot_pro/2026-09-17T14-28_03858be/run_0/` (gitignored).
- **Started**: `2026-09-12 PDT`.
- **Last updated**: `2026-09-17 14:34 PDT`.
- **Notes**: `eval`, all 1,581 tasks. Task concurrency 1-30, at most 60 across the host.

## Results

| Model | Finished | Mean episode return | Score | Avg steps | Input tokens | Output tokens | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-6 Astra Medium | 1581/1581 | 0.938646 | 93.8646% | 1.00 | 11,237,635 | 47,046 | 0 |
| GPT-5.6 Sol xhigh | 1581/1581 | 0.841872 | 84.1872% | 1.00 | 11,237,635 | 454,821 | 0 |
| GPT-5.6 Sol Medium | 1581/1581 | 0.833017 | 83.3017% | 1.00 | 11,237,635 | 178,125 | 0 |
| Qwen3.8-27B | 1581/1581 | 0.753953 | 75.3953% | 1.00 | 9,691,261 | 75,424 | 0 |
| GPT-5.6 Sol none | 1581/1581 | 0.723593 | 72.3593% | 1.00 | 11,237,635 | 34,281 | 0 |
| Qwen3.5-27B | 1581/1581 | 0.695130 | 69.5130% | 1.00 | 9,691,261 | 75,232 | 0 |
| UI-Venus-2-9B | 1581/1581 | 0.649589 | 64.9589% | 1.00 | 8,985,543 | 822,570 | 0 |
| MAI-UI-8B | 1581/1581 | 0.622391 | 62.2391% | 1.00 | 7,818,609 | 119,900 | 0 |
| Qwen3-VL-4B | 1581/1581 | 0.582543 | 58.2543% | 1.00 | 9,449,397 | 53,116 | 0 |
| Qwen3.5-9B | 1581/1581 | 0.581278 | 58.1278% | 1.00 | 9,691,261 | 75,253 | 0 |
| Qwen3.5-4B | 1581/1581 | 0.577483 | 57.7483% | 1.00 | 9,691,261 | 75,359 | 0 |
| Qwen3-VL-32B | 1581/1581 | 0.576850 | 57.6850% | 1.00 | 9,449,397 | 53,079 | 0 |
| Qwen3-VL-8B | 1581/1581 | 0.552182 | 55.2182% | 1.00 | 9,449,397 | 53,099 | 0 |
| UI-TARS-1.5-7B | 1581/1581 | 0.497786 | 49.7786% | 1.00 | 11,630,446 | 26,202 | 0 |
| EvoCUA-8B | 1581/1581 | 0.435168 | 43.5168% | 1.00 | 9,452,559 | 190,791 | 0 |
| ⚠️ GPT-5.5 Medium | _**1570/1581**_ | _**0.727389**_ | 72.2328% | 1.00 | 10,603,628 | 559,722 | 11 |

Score is the sum of rewards divided by 1581; invalid tasks count as zero.
MER is `summary.json: stats.mean_episode_return`, over valid trajectories.
Avg steps covers valid predictions. API tokens sum recorded provider usage
across logged attempts; unlogged calls are excluded. Local tokens use the native
tokenizer and image geometry for retained trajectories, excluding stripped stop
tokens and failed attempts. `N/A` means provider telemetry is unavailable.

## Highlights

- GPT-6 Astra Medium scores 93.86%; Qwen3.8-27B leads the local models at 75.40%.
- GPT-5.5 has 11 persistent frame-size errors: 6016x3384 input versus 6000x3375 reported by the API. Three attempts failed for each task; they count as zero.

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
| GPT-5.6 Sol none | [gpt/screenspot_pro.yaml](/scripts/configs/gpt/default/screenspot_pro.yaml) | Reasoning none; 4096 output tokens |
| GPT-5.5 / GPT-5.6 Sol / GPT-6 Astra Medium | [gpt/screenspot_pro.medium.yaml](/scripts/configs/gpt/default/screenspot_pro.medium.yaml) | Reasoning Medium; 4096 output tokens |
| GPT-5.6 Sol xhigh | [gpt/screenspot_pro.xhigh.yaml](/scripts/configs/gpt/default/screenspot_pro.xhigh.yaml) | Reasoning xhigh; 4096 output tokens |

Each task supplies one screenshot and takes one click prediction using the
model's native grounding template and image preprocessing. Multi-turn history
windows do not apply. Qwen thinking remains disabled; UI-Venus uses its native
thinking template. No agent-level resolution override is set. Greedy decoding
does not use top-p. The Medium and xhigh YAMLs differ from the GPT grounding
default only in reasoning effort; all other parameters inherit the linked defaults.

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
./devs/exps/eval/screenspot_pro/run.sh gpt-5.5 scripts/configs/gpt/default/screenspot_pro.medium.yaml
```

Use the model ID and YAML from the specification for the other rows. Keep the
same run ID and config selection to resume.

## Breakdown

Values cover valid trajectories; GPT-5.5 has 1,570 valid tasks. Headline scores use all 1,581 tasks.

### By `group`

| Model | `CAD` | `Creative` | `Dev` | `OS` | `Office` | `Scientific` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GPT-6 Astra Medium` | 0.9349 | 0.9208 | 0.9298 | 0.9490 | 0.9826 | 0.9291 | 0.9386 |
| `GPT-5.6 Sol xhigh` | 0.8659 | 0.8387 | 0.7759 | 0.7653 | 0.9739 | 0.8386 | 0.8419 |
| `GPT-5.6 Sol Medium` | 0.8352 | 0.8094 | 0.7860 | 0.7755 | 0.9783 | 0.8307 | 0.8330 |
| `Qwen3.8-27B` | 0.6667 | 0.6833 | 0.7960 | 0.7755 | 0.8913 | 0.7480 | 0.7540 |
| `GPT-5.5 Medium` | 0.8046 | 0.7302 | 0.6020 | 0.5946 | 0.8739 | 0.7559 | 0.7274 |
| `GPT-5.6 Sol none` | 0.6015 | 0.7302 | 0.6923 | 0.6939 | 0.9087 | 0.7323 | 0.7236 |
| `Qwen3.5-27B` | 0.6667 | 0.6276 | 0.6957 | 0.7143 | 0.8435 | 0.6654 | 0.6951 |
| `UI-Venus-2-9B` | 0.6513 | 0.5748 | 0.6421 | 0.6480 | 0.7522 | 0.6654 | 0.6496 |
| `MAI-UI-8B` | 0.6015 | 0.5543 | 0.5987 | 0.5969 | 0.8348 | 0.5906 | 0.6224 |
| `Qwen3-VL-4B` | 0.4981 | 0.5220 | 0.5619 | 0.6122 | 0.7565 | 0.5945 | 0.5825 |
| `Qwen3.5-9B` | 0.5211 | 0.4633 | 0.6455 | 0.5510 | 0.7348 | 0.6102 | 0.5813 |
| `Qwen3.5-4B` | 0.5134 | 0.5220 | 0.5619 | 0.5765 | 0.7391 | 0.5906 | 0.5775 |
| `Qwen3-VL-32B` | 0.5287 | 0.5777 | 0.4883 | 0.4643 | 0.7783 | 0.6339 | 0.5769 |
| `Qwen3-VL-8B` | 0.5019 | 0.4839 | 0.5217 | 0.5204 | 0.7348 | 0.5906 | 0.5522 |
| `UI-TARS-1.5-7B` | 0.4444 | 0.4457 | 0.4515 | 0.4031 | 0.7261 | 0.5433 | 0.4978 |
| `EvoCUA-8B` | 0.2759 | 0.4164 | 0.4214 | 0.3929 | 0.6261 | 0.5000 | 0.4352 |

### By `ui_type`

| Model | `icon` | `text` | Avg |
|---|---:|---:|---:|
| `GPT-6 Astra Medium` | 0.9007 | 0.9621 | 0.9386 |
| `GPT-5.6 Sol xhigh` | 0.7517 | 0.8976 | 0.8419 |
| `GPT-5.6 Sol Medium` | 0.7268 | 0.8987 | 0.8330 |
| `Qwen3.8-27B` | 0.5944 | 0.8526 | 0.7540 |
| `GPT-5.5 Medium` | 0.6144 | 0.7971 | 0.7274 |
| `GPT-5.6 Sol none` | 0.5613 | 0.8240 | 0.7236 |
| `Qwen3.5-27B` | 0.4801 | 0.8280 | 0.6951 |
| `UI-Venus-2-9B` | 0.4785 | 0.7554 | 0.6496 |
| `MAI-UI-8B` | 0.3543 | 0.7881 | 0.6224 |
| `Qwen3-VL-4B` | 0.3013 | 0.7564 | 0.5825 |
| `Qwen3.5-9B` | 0.3742 | 0.7093 | 0.5813 |
| `Qwen3.5-4B` | 0.3245 | 0.7339 | 0.5775 |
| `Qwen3-VL-32B` | 0.2930 | 0.7523 | 0.5768 |
| `Qwen3-VL-8B` | 0.2318 | 0.7503 | 0.5522 |
| `UI-TARS-1.5-7B` | 0.2119 | 0.6745 | 0.4978 |
| `EvoCUA-8B` | 0.1623 | 0.6039 | 0.4352 |
