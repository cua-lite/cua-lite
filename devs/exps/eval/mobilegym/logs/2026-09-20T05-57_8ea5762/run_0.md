# mobilegym @ 2026-09-20T05-57_8ea5762 · run_0

- **Commit**: `8ea5762` — Merge remote-tracking branch 'origin/dev' into eval/mobilegym
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB; API models use no local GPU.
- **Artifacts**: `.exps/eval/mobilegym/2026-09-20T05-57_8ea5762/run_0/`
- **Started**: `2026-09-12 PDT`
- **Last updated**: `2026-09-25 00:43 PDT`
- **Notes**: This snapshot aggregates retained reference runs; it is not a new rollout. The paired JSON records each original artifact path, launch commits, configuration and source hashes. Canonical artifact entries link to those retained runs. Claude code drifted during the campaign; launch-time dirty state and subsequent commits remain recorded in `run_info.txt`.

## Results

| Model | Finished | Mean episode return |
|---|---:|---:|
| `gpt-6-astra` | 256/256 | 0.8086 |
| `gpt-5.6-sol` | 256/256 | 0.6406 |
| `gpt-5.5` | 256/256 | 0.5391 |
| `Qwen/Qwen3.8-27B` | 256/256 | 0.4023 |
| `inclusionAI/UI-Venus-2-9B` | 256/256 | 0.3438 |
| `Qwen/Qwen3.5-27B` | 256/256 | 0.2656 |
| `Tongyi-MAI/MAI-UI-8B` | 256/256 | 0.2578 |
| `Qwen/Qwen3-VL-32B-Instruct` | 256/256 | 0.2344 |
| `Qwen/Qwen3.5-9B` | 256/256 | 0.1797 |
| `Qwen/Qwen3.5-4B` | 256/256 | 0.1758 |
| `Qwen/Qwen3-VL-8B-Instruct` | 256/256 | 0.1602 |
| `Qwen/Qwen3-VL-4B-Instruct` | 256/256 | 0.1367 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 256/256 | 0.1328 |
| ⚠️ `claude-opus-5` | _**255/256**_ | _**0.7020**_ |
| ⚠️ `gemini-3.6-flash` | _**0/256**_ | _**—**_ |

## Highlights

- Gemini 3.6 Flash was not started in this campaign.
- Claude Opus 5 has 255/256 valid tasks; one API error is excluded from the mean (179/255 = 0.7020). The error and original trajectory logs are retained.

## Experiment specification

All rows use the family’s [`mobilegym.yaml`](/scripts/configs/gpt/default/mobilegym.yaml) configuration and the `eval` split. GPT and Claude use Medium effort with a 4096-token output budget. Student models and exploratory reasoning variants are excluded.

| Model family | Configuration |
|---|---|
| GPT | [gpt/mobilegym.yaml](/scripts/configs/gpt/default/mobilegym.yaml) |
| Claude | [claude/mobilegym.yaml](/scripts/configs/claude/default/mobilegym.yaml) |
| Qwen3-VL Instruct | [qwen3_vl/mobilegym.yaml](/scripts/configs/qwen3_vl/default/mobilegym.yaml) |
| Qwen3.5 | [qwen3_5/mobilegym.yaml](/scripts/configs/qwen3_5/default/mobilegym.yaml) |
| Qwen3.8 | [qwen3_8/mobilegym.yaml](/scripts/configs/qwen3_8/default/mobilegym.yaml) |
| UI-TARS-1.5 | [ui_tars_15_v1/mobilegym.yaml](/scripts/configs/ui_tars_15_v1/default/mobilegym.yaml) |
| MAI-UI | [mai_ui/mobilegym.yaml](/scripts/configs/mai_ui/default/mobilegym.yaml) |
| UI-Venus-2 | [ui_venus_2/mobilegym.yaml](/scripts/configs/ui_venus_2/default/mobilegym.yaml) |

Mean episode return excludes API/environment errors; valid zero rewards, including model-format failures with evaluator rewards, remain in the denominator. Finished counts retain the full benchmark task count. Per-sample scores are in `eval/<task>/sample_00/summary.json`; failures retain `error.txt`.

The 256-task evaluation uses terminal Success Rate, not the former process reward. Several local-model retries used a 240-second step timeout, recorded per row in JSON. Claude retries used lossless WebP screenshots; the transport is preserved in commit `a8e0bc5`. The patch preserves image pixels and history length.

## Reproduction

Prepare the environment according to the [mobilegym guide](/lite/gym/envs/mobilegym/README.md), configure API credentials or local model weights, and export `CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN`. Then run:

```bash
export EVAL_RUN_ID=run_0
./devs/exps/eval/mobilegym/run.sh gpt-5.5
```

Use the corresponding model ID for other rows. Saved `run_info.txt` files contain the actual launch commands and concurrency. The paired JSON preserves operational overrides and source provenance.
