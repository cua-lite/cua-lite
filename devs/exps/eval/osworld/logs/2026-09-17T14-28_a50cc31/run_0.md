# osworld @ 2026-09-17T14-28_a50cc31 - run_0

- **Commit**: `a50cc31` - OSWorld reference models and GPT reasoning variants.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/osworld/2026-09-17T14-28_a50cc31/run_0/` (gitignored).
- **Started**: `2026-09-14 PDT`.
- **Last updated**: `2026-09-17 14:34 PDT`.
- **Notes**: Official OSWorld `eval`, 325 tasks after the `exclude_reason` filter. Up to 30 task workers; Docker creation concurrency 10.

## Results

| Model | Finished | Mean episode return | Score | Avg steps | Input tokens | Output tokens | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Sol xhigh | 325/325 | 0.747666 | 74.7666% | 13.32 | 117,832,226 | 736,487 | 0 |
| GPT-5.6 Sol Medium | 325/325 | 0.732611 | 73.2611% | 11.92 | N/A | N/A | 0 |
| GPT-5.5 Medium | 325/325 | 0.710893 | 71.0893% | 13.03 | 108,955,821 | 695,306 | 0 |
| GPT-5.6 Sol none | 325/325 | 0.602621 | 60.2621% | 9.19 | 53,736,351 | 179,259 | 0 |
| Qwen3.8-27B | 325/325 | 0.591711 | 59.1711% | 16.73 | 41,139,185 | 610,174 | 0 |
| UI-Venus-2-9B | 325/325 | 0.504743 | 50.4743% | 14.46 | 46,888,130 | 1,392,099 | 0 |
| Qwen3.5-9B | 325/325 | 0.385763 | 38.5763% | 17.12 | 39,903,726 | 396,382 | 0 |
| EvoCUA-8B | 325/325 | 0.355749 | 35.5749% | 16.48 | 47,632,263 | 479,703 | 0 |
| Qwen3-VL-32B | 325/325 | 0.342914 | 34.2914% | 16.02 | 46,104,838 | 303,200 | 0 |
| UI-TARS-1.5-7B | 325/325 | 0.274721 | 27.4721% | 21.38 | 89,901,053 | 659,114 | 0 |
| Qwen3.5-4B | 325/325 | 0.270294 | 27.0294% | 19.58 | 46,155,710 | 443,753 | 0 |
| Qwen3-VL-4B | 325/325 | 0.236474 | 23.6474% | 15.14 | 43,235,877 | 270,639 | 0 |
| ⚠️ GPT-6 Astra Medium | _**324/325**_ | _**0.753701**_ | 75.1382% | 9.96 | 68,825,977 | 240,909 | 1 |
| ⚠️ Qwen3.5-27B | _**324/325**_ | _**0.490976**_ | 48.9466% | 17.04 | 39,621,218 | 392,258 | 1 |
| ⚠️ Qwen3-VL-8B | _**324/325**_ | _**0.282190**_ | 28.1322% | 14.13 | 40,021,144 | 262,126 | 1 |

Score is the sum of rewards divided by 325; invalid tasks count as zero.
MER is `summary.json: stats.mean_episode_return`, over valid trajectories.
Avg steps covers valid model turns, including model-format failures with a
terminal evaluator reward. GPT tokens use recorded provider usage; Sol/Astra
totals include logged retries, with unlogged calls excluded. GPT-5.5 uses its
retained usage aggregate; Sol Medium has no retained token telemetry.
Local tokens use the checkpoint tokenizer and image geometry for retained
trajectories, excluding stripped stop tokens and failed attempts.

## Highlights

- GPT-6 Astra Medium scores 75.14%; Qwen3.8-27B leads the local models at 59.17%.
- GPT-6 Astra finished 324/325: one GIMP task repeatedly failed on action-batch timeouts or unsupported key names.
- Qwen3.5-27B finished 324/325: one task repeatedly failed during evaluator PDF cleanup.
- Qwen3-VL-8B finished 324/325: one task repeatedly failed when the evaluator decoded a GIF.

## Experiment Specification

| Model | YAML | Sampling | History |
|---|---|---|---|
| GPT-5.5 / GPT-5.6 Sol / GPT-6 Astra Medium | [gpt/osworld.yaml](/scripts/configs/gpt/default/osworld.yaml) | Medium; 4096 output tokens | Chained Responses history |
| GPT-5.6 Sol none | [gpt/osworld.none.yaml](/scripts/configs/gpt/default/osworld.none.yaml) | None; 4096 output tokens | Chained Responses history |
| GPT-5.6 Sol xhigh | [gpt/osworld.xhigh.yaml](/scripts/configs/gpt/default/osworld.xhigh.yaml) | xhigh; 4096 output tokens | Chained Responses history |
| Qwen3-VL-4B / 8B / 32B (Instruct) | [qwen3_vl/osworld.yaml](/scripts/configs/qwen3_vl/default/osworld.yaml) | Temperature 0; 2048 output tokens | 4 full / 100 summary turns |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/osworld.yaml](/scripts/configs/qwen3_5/default/osworld.yaml) | Thinking off; temperature 0; 2048 output tokens | History 100 / image 4 / fold 4 |
| Qwen3.8-27B | [qwen3_8/osworld.yaml](/scripts/configs/qwen3_8/default/osworld.yaml) | Thinking off; temperature 0; 2048 output tokens | History 100 / image 4 / fold 4 |
| UI-TARS-1.5-7B | [ui_tars_15_v1/osworld.yaml](/scripts/configs/ui_tars_15_v1/default/osworld.yaml) | Temperature 0; 2048 output tokens | 5 full turns |
| EvoCUA-8B | [evocua/osworld.yaml](/scripts/configs/evocua/default/osworld.yaml) | Temperature .01; top-p .9; 2048 output tokens | 4 full / 100 summary turns |
| UI-Venus-2-9B | [ui_venus_2/osworld.yaml](/scripts/configs/ui_venus_2/default/osworld.yaml) | Temperature 0; top-p .7; 4096 output tokens | 2 past images + current |

All rows use 30 steps, native 1920x1080 observations and 2-second post-action
delay, with no agent-level resolution override. The default reset and step
timeouts are 600 and 180 seconds. Local models use `terminate` and loop
detection 5; UI-Venus also enables `response`, and GPT uses loop detection 0.
UI-Venus uses its native thinking template. Greedy decoding does not use top-p.
The none/xhigh YAMLs differ from the GPT default only in reasoning effort;
unspecified settings inherit code defaults.

### Reproduction

Prepare the image and Ubuntu disk using the
[OSWorld guide](/lite/gym/envs/osworld/README.md). The image includes the
process-scoped THP wrapper. Start an OSWorld env-server with the following
environment, then export its `CUA_LITE_ENV_SERVER_URL` and
`CUA_LITE_ENV_SERVER_TOKEN` on the rollout host:

```bash
export CUA_LITE_DOCKER_CREATE_CONCURRENCY=10
export CUA_LITE_DRIFT_SAFETY_MARGIN_S=3600
export CUA_LITE_503_DEADLINE_S=600
```

Configure API credentials for GPT runs and cache local model weights.

```bash
export EVAL_RUN_ID=run_0
export EVAL_CONCURRENCY=30
CUDA_VISIBLE_DEVICES=0,1 ./devs/exps/eval/osworld/run.sh Qwen/Qwen3-VL-32B-Instruct
./devs/exps/eval/osworld/run.sh gpt-5.5
./devs/exps/eval/osworld/run.sh gpt-5.6-sol scripts/configs/gpt/default/osworld.xhigh.yaml
```

Use the corresponding model ID and YAML for the other rows. Keep the same run
ID and config selection to resume. Set concurrency to the host and API quota;
the combined GPT-5.6 Sol concurrency was limited to 15 after rate-limit checks.
