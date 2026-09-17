# androidlab @ 2026-09-17T14-26_5268092 - run_0

- **Commit**: `5268092` - AndroidLab reference models and GPT reasoning variants.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/androidlab/2026-09-17T14-26_5268092/run_0/` (gitignored).
- **Started**: `2026-09-15 PDT`.
- **Last updated**: `2026-09-17 14:34 PDT`.
- **Notes**: `eval`, all 138 tasks. Task concurrency up to 12 for this environment; total task concurrency at most 60.

## Results

| Model | Finished | Mean episode return | Score | Avg steps | Input tokens | Output tokens | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Sol xhigh | 138/138 | 0.637681 | 63.7681% | 8.07 | 12,063,852 | 81,508 | 0 |
| GPT-5.5 Medium | 138/138 | 0.615942 | 61.5942% | 8.15 | 12,325,852 | 78,437 | 0 |
| GPT-5.6 Sol Medium | 138/138 | 0.608696 | 60.8696% | 7.96 | 11,579,574 | 59,231 | 0 |
| GPT-6 Astra Medium | 138/138 | 0.608696 | 60.8696% | 7.10 | 8,831,343 | 27,465 | 0 |
| GPT-5.6 Sol none | 138/138 | 0.565217 | 56.5217% | 7.94 | 11,527,956 | 29,134 | 0 |
| Qwen3.8-27B | 138/138 | 0.376812 | 37.6812% | 7.70 | 4,648,881 | 67,750 | 0 |
| UI-Venus-2-9B | 138/138 | 0.376812 | 37.6812% | 8.57 | 6,629,119 | 316,293 | 0 |
| MAI-UI-8B | 138/138 | 0.347826 | 34.7826% | 9.14 | 5,450,865 | 140,696 | 0 |
| Qwen3-VL-8B | 138/138 | 0.289855 | 28.9855% | 7.25 | 4,950,873 | 52,211 | 0 |
| Qwen3.5-27B | 138/138 | 0.289855 | 28.9855% | 7.77 | 4,664,661 | 68,437 | 0 |
| Qwen3-VL-32B | 138/138 | 0.275362 | 27.5362% | 7.09 | 4,828,296 | 49,151 | 0 |
| UI-TARS-1.5-7B | 138/138 | 0.275362 | 27.5362% | 8.74 | 7,911,826 | 108,539 | 0 |
| Qwen3.5-4B | 138/138 | 0.253623 | 25.3623% | 7.92 | 4,770,658 | 68,553 | 0 |
| Qwen3-VL-4B | 138/138 | 0.246377 | 24.6377% | 7.80 | 5,413,695 | 55,156 | 0 |
| Qwen3.5-9B | 138/138 | 0.246377 | 24.6377% | 9.06 | 5,591,885 | 78,771 | 0 |

Score is the sum of rewards divided by 138; invalid tasks count as zero.
MER is `summary.json: stats.mean_episode_return`, over valid trajectories.
Avg steps covers valid model turns, including model-format failures with a
terminal evaluator reward. GPT tokens use recorded provider usage across
logged attempts, including retries; unlogged calls are excluded. Local tokens
use the checkpoint tokenizer and image geometry for retained trajectories,
excluding stripped stop tokens and failed attempts.

## Highlights

- GPT-5.6 Sol xhigh scores 63.77%; Qwen3.8-27B and UI-Venus-2-9B lead the local models at 37.68%.

## Experiment Specification

| Model | YAML | Sampling | History |
|---|---|---|---|
| GPT-5.5 / GPT-5.6 Sol / GPT-6 Astra Medium | [gpt/androidlab.yaml](/scripts/configs/gpt/default/androidlab.yaml) | Medium; 4096 output tokens | Chained Responses history |
| GPT-5.6 Sol none | [gpt/androidlab.none.yaml](/scripts/configs/gpt/default/androidlab.none.yaml) | None; 4096 output tokens | Chained Responses history |
| GPT-5.6 Sol xhigh | [gpt/androidlab.xhigh.yaml](/scripts/configs/gpt/default/androidlab.xhigh.yaml) | xhigh; 4096 output tokens | Chained Responses history |
| Qwen3-VL-4B / 8B / 32B (Instruct) | [qwen3_vl/androidlab.yaml](/scripts/configs/qwen3_vl/default/androidlab.yaml) | Temperature 0; 2048 output tokens | 4 full / 100 summary turns |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/androidlab.yaml](/scripts/configs/qwen3_5/default/androidlab.yaml) | Thinking off; temperature 0; 2048 output tokens | History 100 / image 4 / fold 4 |
| Qwen3.8-27B | [qwen3_8/androidlab.yaml](/scripts/configs/qwen3_8/default/androidlab.yaml) | Thinking off; temperature 0; 2048 output tokens | History 100 / image 4 / fold 4 |
| UI-TARS-1.5-7B | [ui_tars_15_v1/androidlab.yaml](/scripts/configs/ui_tars_15_v1/default/androidlab.yaml) | Temperature 0; 400 output tokens | 5 full turns |
| MAI-UI-8B | [mai_ui/androidlab.yaml](/scripts/configs/mai_ui/default/androidlab.yaml) | Temperature 0; top-p 1; top-k -1; 2048 output tokens | 3 image turns; unbounded older assistant text |
| UI-Venus-2-9B | [ui_venus_2/androidlab.yaml](/scripts/configs/ui_venus_2/default/androidlab.yaml) | Temperature 0; 16384 output tokens; repetition penalty 1.05 | 2 past images + current |

All rows use 20 steps, screenshot-only observations, a 1600-pixel screenshot
long-edge cap and 3-second post-action delay, with no agent-level resolution
override. Local models use loop detection 5; GPT uses 0. Every config exposes
`response` and `terminate`; all except UI-TARS also expose `open_app`.
UI-Venus uses its native thinking template. Greedy decoding does not use top-p.
The none/xhigh YAMLs differ from the GPT default only in reasoning effort;
unspecified settings inherit code defaults.

### Reproduction

Prepare the image and Quick Boot snapshot using the
[AndroidLab guide](/lite/gym/envs/androidlab/README.md). Start an AndroidLab
env-server and export its `CUA_LITE_ENV_SERVER_URL` and
`CUA_LITE_ENV_SERVER_TOKEN` on the rollout host. Configure API credentials
for GPT runs and cache local model weights.

```bash
export EVAL_RUN_ID=run_0
export EVAL_CONCURRENCY=5
CUDA_VISIBLE_DEVICES=0,1 ./devs/exps/eval/androidlab/run.sh Qwen/Qwen3-VL-32B-Instruct
./devs/exps/eval/androidlab/run.sh gpt-5.5
./devs/exps/eval/androidlab/run.sh gpt-5.6-sol scripts/configs/gpt/default/androidlab.xhigh.yaml
```

Use the corresponding model ID and YAML for the other rows. Keep the same run
ID and config selection to resume. Set concurrency to the host and API quota.
