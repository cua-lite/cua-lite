# waa @ 2026-09-12T10-27_77f24c8 · run_0_standard

- **Commit**: `77f24c8` — standard WAA evaluation.
- **Host / GPUs**: `gpublaze` / H100 80 GB; 4B evaluations use GPUs 0 and 1.
- **Artifacts**: `.exps/eval/waa/2026-09-12T10-27_77f24c8/run_0_standard/` (gitignored).
- **Started**: `2026-09-12 02:46 PDT`.
- **Last updated**: `2026-09-12 10:26 PDT`.
- **Notes**: `eval`, 138 scored tasks; task concurrency up to 30 per model
  (60 total), Docker creation concurrency 10.

## Results

| Model | Finished | Mean episode return | Score | Avg steps | Input tokens | Output tokens | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
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
| ⚠️ Gemini 3.6 Flash | _**0/138**_ | _**—**_ | — | — | — | — | 138 |
| ⚠️ Claude Opus 4.8 | _**0/138**_ | _**—**_ | — | — | — | — | 138 |
| ⚠️ GPT-5.6 Sol | _**0/138**_ | _**—**_ | — | — | — | — | 138 |
| ⚠️ GPT-6 Astra | _**0/138**_ | _**—**_ | — | — | — | — | 138 |

Score = sum of evaluator rewards / 138; unresolved and errored tasks count as
zero. MER is `summary.json: stats.mean_episode_return`. A model-format failure
with a terminal evaluator reward is scored normally. Avg steps and tokens
cover final retained attempts: GPT uses provider usage; local counts use the
checkpoint tokenizer including visual tokens. Local output counts tokenize saved
replies and exclude stripped stop/special tokens. Retries are excluded.

## Highlights

- GPT-5.5 leads at 66.57%; UI-Venus-2-9B leads the local models at 41.28%.
- Gemini 3.6 Flash, Claude Opus 4.8, GPT-5.6 Sol and GPT-6 Astra are unmeasured.
  MAI-UI-8B is WAA-inapplicable.

## Experiment specification

| Model | YAML | Sampling | History |
|---|---|---|---|
| GPT-5.5 | [gpt/waa.yaml](/scripts/configs/gpt/default/waa.yaml) | Medium; 4096 output | Chained Responses history |
| Qwen3-VL-4B / 8B / 32B | [qwen3_vl/waa.yaml](/scripts/configs/qwen3_vl/default/waa.yaml) | Temperature 0; 2048 output | 4 full / 100 summary turns |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/waa.yaml](/scripts/configs/qwen3_5/default/waa.yaml) | Thinking off; temperature 0; 2048 output | History 100 / image 4 / fold 4 |
| Qwen3.8-27B | [qwen3_8/waa.yaml](/scripts/configs/qwen3_8/default/waa.yaml) | Thinking off; temperature 0; 2048 output | History 100 / image 4 / fold 4 |
| UI-TARS-1.5-7B | [ui_tars_15_v1/waa.yaml](/scripts/configs/ui_tars_15_v1/default/waa.yaml) | Temperature 0; 2048 output | 5 full turns |
| EvoCUA-8B | [evocua/waa.yaml](/scripts/configs/evocua/default/waa.yaml) | Temperature .01; top-p .9; 2048 output | 4 full / 100 summary turns |
| UI-Venus-2-9B | [ui_venus_2/waa.yaml](/scripts/configs/ui_venus_2/default/waa.yaml) | Temperature 0; top-p .7; 4096 output | 2 past images + current |

All rows use 30 steps and native image resolution. Local models use `terminate`
and loop detection 5. GPT uses loop detection 0. UI-Venus uses its native thinking template.
Unspecified settings inherit code defaults; greedy decoding does not use top-p.

### Reproduction

Prepare the image, assets and ready snapshot using the [WAA guide](/lite/gym/envs/waa/README.md).
Start a WAA env-server with `CUA_LITE_DOCKER_CREATE_CONCURRENCY=10`; export its
`CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN` on the rollout host.

```bash
export EVAL_RUN_ID=run_0_standard
export EVAL_CONCURRENCY=30
CUDA_VISIBLE_DEVICES=0 ./devs/exps/eval/waa/run.sh \
  Qwen/Qwen3-VL-8B-Instruct scripts/configs/qwen3_vl/default/waa.yaml
```

Keep the same run ID to resume. Total task concurrency must remain at most 60.
Local serving uses BF16 and chunked prefill 4096. VL-4B / 8B and Qwen3.5-4B use TP1,
context 65536, max requests 30 and memory fraction .80.
`EVAL_MODEL_PATH` and `EVAL_ENGINE_KWARGS` select local weights and serving options.
