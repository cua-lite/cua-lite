# waa @ 2026-09-17T14-28_dd36a23 - run_0

- **Commit**: `dd36a23` - WAA reference runner and model configurations.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/waa/2026-09-17T14-28_dd36a23/run_0/` (gitignored).
- **Started**: `2026-09-09 PDT`.
- **Last updated**: `2026-09-17 14:34 PDT`.
- **Notes**: `eval`, 138 tasks after the `exclude_reason` filter. Task concurrency up to 30 per model and 60 total; Docker creation concurrency 10.

## Results

| Model | Finished | Mean episode return | Score | Avg steps | Input tokens | Output tokens | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Sol | 138/138 | 0.700250 | 70.0250% | 11.08 | 18,106,995 | 150,777 | 0 |
| GPT-5.5 | 138/138 | 0.665688 | 66.5688% | 13.28 | 24,837,106 | 257,356 | 0 |
| UI-Venus-2-9B | 138/138 | 0.412795 | 41.2795% | 16.29 | 15,917,306 | 675,193 | 0 |
| Qwen3.8-27B | 138/138 | 0.369995 | 36.9995% | 19.62 | 13,524,820 | 256,512 | 0 |
| Qwen3.5-27B | 138/138 | 0.254053 | 25.4053% | 19.90 | 12,978,850 | 190,282 | 0 |
| Qwen3-VL-32B | 138/138 | 0.239560 | 23.9560% | 18.83 | 13,365,755 | 148,891 | 0 |
| UI-TARS-1.5-7B | 138/138 | 0.167096 | 16.7096% | 21.46 | 21,529,540 | 283,595 | 0 |
| EvoCUA-8B | 138/138 | 0.165989 | 16.5989% | 17.19 | 12,179,274 | 210,648 | 0 |
| Qwen3-VL-8B | 138/138 | 0.145357 | 14.5357% | 15.51 | 10,736,049 | 116,744 | 0 |
| Qwen3.5-9B | 138/138 | 0.130864 | 13.0864% | 19.33 | 12,428,073 | 186,764 | 0 |
| Qwen3.5-4B | 138/138 | 0.116372 | 11.6372% | 17.15 | 10,821,751 | 165,274 | 0 |
| Qwen3-VL-4B | 138/138 | 0.094637 | 9.4637% | 17.39 | 12,187,692 | 130,347 | 0 |
| ⚠️ GPT-6 Astra | _**137/138**_ | _**0.634359**_ | 62.9762% | 8.09 | 9,960,740 | 80,371 | 1 |

Score is the sum of rewards divided by 138; invalid tasks count as zero.
MER is `summary.json: stats.mean_episode_return`, over valid trajectories.
Avg steps covers valid model turns, including model-format failures with a
terminal evaluator reward. GPT tokens use recorded provider usage; Sol/Astra
totals also include logged retries. Unlogged calls are excluded. Local tokens use
the checkpoint tokenizer and image geometry for retained trajectories, excluding
stripped stop tokens and failed attempts.

## Highlights

- GPT-5.6 Sol scores 70.03%; UI-Venus-2-9B leads local models at 41.28%.
- GPT-6 Astra completed 137/138 tasks. One task repeatedly failed on unsupported keyboard key names; it counts as zero.

## Experiment Specification

| Model | YAML | Sampling | History |
|---|---|---|---|
| GPT-5.5 / GPT-5.6 Sol / GPT-6 Astra | [gpt/waa.yaml](/scripts/configs/gpt/default/waa.yaml) | Medium; 4096 output tokens | Chained Responses history |
| Qwen3-VL-4B / 8B / 32B (Instruct) | [qwen3_vl/waa.yaml](/scripts/configs/qwen3_vl/default/waa.yaml) | Temperature 0; 2048 output tokens | 4 full / 100 summary turns |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/waa.yaml](/scripts/configs/qwen3_5/default/waa.yaml) | Thinking off; temperature 0; 2048 output tokens | History 100 / image 4 / fold 4 |
| Qwen3.8-27B | [qwen3_8/waa.yaml](/scripts/configs/qwen3_8/default/waa.yaml) | Thinking off; temperature 0; 2048 output tokens | History 100 / image 4 / fold 4 |
| UI-TARS-1.5-7B | [ui_tars_15_v1/waa.yaml](/scripts/configs/ui_tars_15_v1/default/waa.yaml) | Temperature 0; 2048 output tokens | 5 full turns |
| EvoCUA-8B | [evocua/waa.yaml](/scripts/configs/evocua/default/waa.yaml) | Temperature .01; top-p .9; 2048 output tokens | 4 full / 100 summary turns |
| UI-Venus-2-9B | [ui_venus_2/waa.yaml](/scripts/configs/ui_venus_2/default/waa.yaml) | Temperature 0; top-p .7; 4096 output tokens | 2 past images + current |

All rows use 30 steps and native 1280x800 observations, with no agent-level
resolution override. Local models use `terminate` and loop detection 5; GPT
uses loop detection 0. UI-Venus uses its native thinking template. Greedy
only in reasoning effort; unspecified settings inherit code defaults.

### Reproduction

Prepare the image, assets and ready snapshot using the
[WAA guide](/lite/gym/envs/waa/README.md). Start a WAA env-server with
`CUA_LITE_DOCKER_CREATE_CONCURRENCY=10`; export its
`CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN` on the rollout host.
Configure API credentials for GPT runs and cache local model weights.

```bash
export EVAL_RUN_ID=run_0
export EVAL_CONCURRENCY=30
CUDA_VISIBLE_DEVICES=0 ./devs/exps/eval/waa/run.sh Qwen/Qwen3-VL-8B-Instruct
./devs/exps/eval/waa/run.sh gpt-5.5
```

Use the corresponding model ID and YAML for the other rows. Keep the same run
ID and config selection to resume, with total task concurrency at most 60.
`EVAL_MODEL_PATH` and `EVAL_ENGINE_KWARGS` select local weights and serving options.
