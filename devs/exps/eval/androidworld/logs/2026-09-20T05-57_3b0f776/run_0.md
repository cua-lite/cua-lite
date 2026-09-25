# androidworld @ 2026-09-20T05-57_3b0f776 · run_0

- **Commit**: `3b0f776` — Merge remote-tracking branch 'origin/dev' into eval/androidworld
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB; API models use no local GPU.
- **Artifacts**: `.exps/eval/androidworld/2026-09-20T05-57_3b0f776/run_0/`
- **Started**: `2026-09-12 PDT`
- **Last updated**: `2026-09-25 00:43 PDT`
- **Notes**: This snapshot aggregates retained reference runs; it is not a new rollout. The paired JSON records each original artifact path, launch commits, configuration and source hashes. Canonical artifact entries link to those retained runs. Claude code drifted during the campaign; launch-time dirty state and subsequent commits remain recorded in `run_info.txt`.

## Results

| Model | Finished | Mean episode return |
|---|---:|---:|
| `gpt-6-astra` | 116/116 | 0.8448 |
| `claude-opus-5` | 116/116 | 0.8103 |
| `gpt-5.5` | 116/116 | 0.7845 |
| `gpt-5.6-sol` | 116/116 | 0.7586 |
| `Qwen/Qwen3.5-27B` | 116/116 | 0.6509 |
| `Qwen/Qwen3.8-27B` | 116/116 | 0.6466 |
| `Qwen/Qwen3-VL-32B-Instruct` | 116/116 | 0.6207 |
| `Qwen/Qwen3.5-9B` | 116/116 | 0.6207 |
| `inclusionAI/UI-Venus-2-9B` | 116/116 | 0.6207 |
| `Tongyi-MAI/MAI-UI-8B` | 116/116 | 0.6121 |
| `Qwen/Qwen3.5-4B` | 116/116 | 0.6034 |
| `Qwen/Qwen3-VL-8B-Instruct` | 116/116 | 0.5172 |
| `Qwen/Qwen3-VL-4B-Instruct` | 116/116 | 0.4310 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 116/116 | 0.1983 |
| ⚠️ `gemini-3.6-flash` | _**0/116**_ | _**—**_ |

## Highlights

- Gemini 3.6 Flash was not started in this campaign.

## Experiment specification

All rows use the family’s [`androidworld.yaml`](/scripts/configs/gpt/default/androidworld.yaml) configuration and the `eval` split. GPT and Claude use Medium effort with a 4096-token output budget. Student models and exploratory reasoning variants are excluded.

| Model family | Configuration |
|---|---|
| GPT | [gpt/androidworld.yaml](/scripts/configs/gpt/default/androidworld.yaml) |
| Claude | [claude/androidworld.yaml](/scripts/configs/claude/default/androidworld.yaml) |
| Qwen3-VL Instruct | [qwen3_vl/androidworld.yaml](/scripts/configs/qwen3_vl/default/androidworld.yaml) |
| Qwen3.5 | [qwen3_5/androidworld.yaml](/scripts/configs/qwen3_5/default/androidworld.yaml) |
| Qwen3.8 | [qwen3_8/androidworld.yaml](/scripts/configs/qwen3_8/default/androidworld.yaml) |
| UI-TARS-1.5 | [ui_tars_15_v1/androidworld.yaml](/scripts/configs/ui_tars_15_v1/default/androidworld.yaml) |
| MAI-UI | [mai_ui/androidworld.yaml](/scripts/configs/mai_ui/default/androidworld.yaml) |
| UI-Venus-2 | [ui_venus_2/androidworld.yaml](/scripts/configs/ui_venus_2/default/androidworld.yaml) |

Mean episode return excludes API/environment errors; valid zero rewards, including model-format failures with evaluator rewards, remain in the denominator. Finished counts retain the full benchmark task count. Per-sample scores are in `eval/<task>/sample_00/summary.json`; failures retain `error.txt`.

All 116 tasks are included. Actual concurrency and retry invocations are retained per model in `run_info.txt`.

## Reproduction

Prepare the environment according to the [androidworld guide](/lite/gym/envs/androidworld/README.md), configure API credentials or local model weights, and export `CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN`. Then run:

```bash
export EVAL_RUN_ID=run_0
./devs/exps/eval/androidworld/run.sh gpt-5.5
```

Use the corresponding model ID for other rows. Saved `run_info.txt` files contain the actual launch commands and concurrency. The paired JSON preserves operational overrides and source provenance.
