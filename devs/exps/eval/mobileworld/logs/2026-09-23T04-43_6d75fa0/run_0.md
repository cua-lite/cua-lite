# mobileworld @ 2026-09-23T04-43_6d75fa0 · run_0

- **Commit**: `6d75fa0` — fix(mobileworld): exclude user-interaction tasks from default eval
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB; API models use no local GPU.
- **Artifacts**: `.exps/eval/mobileworld/2026-09-23T04-43_6d75fa0/run_0/`
- **Started**: `2026-09-23 PDT`
- **Last updated**: `2026-09-25 00:43 PDT`
- **Notes**: This snapshot aggregates retained reference runs; it is not a new rollout. The paired JSON records each original artifact path, launch commits, configuration and source hashes. Canonical artifact entries link to those retained runs. Claude code drifted during the campaign; launch-time dirty state and subsequent commits remain recorded in `run_info.txt`.

## Results

| Model | Finished | Mean episode return |
|---|---:|---:|
| `gpt-5.6-sol` | 117/117 | 0.8632 |
| `gpt-6-astra` | 117/117 | 0.8547 |
| `gpt-5.5` | 117/117 | 0.8034 |
| `claude-opus-5` | 117/117 | 0.7863 |
| `inclusionAI/UI-Venus-2-9B` | 117/117 | 0.5897 |
| `Qwen/Qwen3.8-27B` | 117/117 | 0.3248 |
| `Tongyi-MAI/MAI-UI-8B` | 117/117 | 0.2991 |
| `Qwen/Qwen3-VL-32B-Instruct` | 117/117 | 0.1795 |
| `Qwen/Qwen3.5-27B` | 117/117 | 0.1538 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 117/117 | 0.1111 |
| `Qwen/Qwen3.5-4B` | 117/117 | 0.1111 |
| `Qwen/Qwen3.5-9B` | 117/117 | 0.1026 |
| `Qwen/Qwen3-VL-8B-Instruct` | 117/117 | 0.0855 |
| `Qwen/Qwen3-VL-4B-Instruct` | 117/117 | 0.0684 |
| ⚠️ `gemini-3.6-flash` | _**0/117**_ | _**—**_ |

## Highlights

- Gemini 3.6 Flash was not started in this campaign.

## Experiment specification

All rows use the family’s [`mobileworld.yaml`](/scripts/configs/gpt/default/mobileworld.yaml) configuration and the `eval` split. GPT and Claude use Medium effort with a 4096-token output budget. Student models and exploratory reasoning variants are excluded.

| Model family | Configuration |
|---|---|
| GPT | [gpt/mobileworld.yaml](/scripts/configs/gpt/default/mobileworld.yaml) |
| Claude | [claude/mobileworld.yaml](/scripts/configs/claude/default/mobileworld.yaml) |
| Qwen3-VL Instruct | [qwen3_vl/mobileworld.yaml](/scripts/configs/qwen3_vl/default/mobileworld.yaml) |
| Qwen3.5 | [qwen3_5/mobileworld.yaml](/scripts/configs/qwen3_5/default/mobileworld.yaml) |
| Qwen3.8 | [qwen3_8/mobileworld.yaml](/scripts/configs/qwen3_8/default/mobileworld.yaml) |
| UI-TARS-1.5 | [ui_tars_15_v1/mobileworld.yaml](/scripts/configs/ui_tars_15_v1/default/mobileworld.yaml) |
| MAI-UI | [mai_ui/mobileworld.yaml](/scripts/configs/mai_ui/default/mobileworld.yaml) |
| UI-Venus-2 | [ui_venus_2/mobileworld.yaml](/scripts/configs/ui_venus_2/default/mobileworld.yaml) |

Mean episode return excludes API/environment errors; valid zero rewards, including model-format failures with evaluator rewards, remain in the denominator. Finished counts retain the full benchmark task count. Per-sample scores are in `eval/<task>/sample_00/summary.json`; failures retain `error.txt`.

The default filter excludes 44 user-interaction tasks, leaving 117 GUI-only tasks. Local and GPT runs used `step_timeout=240`; Qwen and GPT runs used `cua-lite/mobileworld:think-user-api-20260919`. These deployment overrides are recorded per row in JSON. Claude used the default YAML and lossless WebP screenshots; the transport is preserved in commit `9b0f938`.

## Reproduction

Prepare the environment according to the [mobileworld guide](/lite/gym/envs/mobileworld/README.md), configure API credentials or local model weights, and export `CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN`. Then run:

```bash
export EVAL_RUN_ID=run_0
./devs/exps/eval/mobileworld/run.sh gpt-5.5
```

Use the corresponding model ID for other rows. Saved `run_info.txt` files contain the actual launch commands and concurrency. The paired JSON preserves operational overrides and source provenance.
