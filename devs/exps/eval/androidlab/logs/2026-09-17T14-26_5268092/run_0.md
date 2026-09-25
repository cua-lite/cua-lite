# androidlab @ 2026-09-17T14-26_5268092 · run_0

- **Commit**: `5268092` — eval(androidlab): support reference models and GPT reasoning variants
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB; API models use no local GPU.
- **Artifacts**: `.exps/eval/androidlab/2026-09-17T14-26_5268092/run_0/`
- **Started**: `2026-09-15 PDT`
- **Last updated**: `2026-09-25 00:43 PDT`
- **Notes**: This snapshot aggregates retained reference runs; it is not a new rollout. The paired JSON records each original artifact path, launch commits, configuration and source hashes. Canonical artifact entries link to those retained runs. Claude code drifted during the campaign; launch-time dirty state and subsequent commits remain recorded in `run_info.txt`.

## Results

| Model | Finished | Mean episode return |
|---|---:|---:|
| `claude-opus-5` | 138/138 | 0.6232 |
| `gpt-5.5` | 138/138 | 0.6087 |
| `gpt-5.6-sol` | 138/138 | 0.6087 |
| `gpt-6-astra` | 138/138 | 0.6087 |
| `Qwen/Qwen3.8-27B` | 138/138 | 0.5145 |
| `Tongyi-MAI/MAI-UI-8B` | 138/138 | 0.4928 |
| `inclusionAI/UI-Venus-2-9B` | 138/138 | 0.4783 |
| `Qwen/Qwen3.5-27B` | 138/138 | 0.4058 |
| `Qwen/Qwen3-VL-8B-Instruct` | 138/138 | 0.3986 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 138/138 | 0.3623 |
| `Qwen/Qwen3-VL-32B-Instruct` | 138/138 | 0.3623 |
| `Qwen/Qwen3-VL-4B-Instruct` | 138/138 | 0.3623 |
| `Qwen/Qwen3.5-9B` | 138/138 | 0.3623 |
| `Qwen/Qwen3.5-4B` | 138/138 | 0.3478 |
| ⚠️ `gemini-3.6-flash` | _**0/138**_ | _**—**_ |

## Highlights

- Gemini 3.6 Flash was not started in this campaign.

## Experiment specification

All rows use the family’s [`androidlab.yaml`](/scripts/configs/gpt/default/androidlab.yaml) configuration and the `eval` split. GPT and Claude use Medium effort with a 4096-token output budget. Student models and exploratory reasoning variants are excluded.

| Model family | Configuration |
|---|---|
| GPT | [gpt/androidlab.yaml](/scripts/configs/gpt/default/androidlab.yaml) |
| Claude | [claude/androidlab.yaml](/scripts/configs/claude/default/androidlab.yaml) |
| Qwen3-VL Instruct | [qwen3_vl/androidlab.yaml](/scripts/configs/qwen3_vl/default/androidlab.yaml) |
| Qwen3.5 | [qwen3_5/androidlab.yaml](/scripts/configs/qwen3_5/default/androidlab.yaml) |
| Qwen3.8 | [qwen3_8/androidlab.yaml](/scripts/configs/qwen3_8/default/androidlab.yaml) |
| UI-TARS-1.5 | [ui_tars_15_v1/androidlab.yaml](/scripts/configs/ui_tars_15_v1/default/androidlab.yaml) |
| MAI-UI | [mai_ui/androidlab.yaml](/scripts/configs/mai_ui/default/androidlab.yaml) |
| UI-Venus-2 | [ui_venus_2/androidlab.yaml](/scripts/configs/ui_venus_2/default/androidlab.yaml) |

Mean episode return excludes API/environment errors; valid zero rewards, including model-format failures with evaluator rewards, remain in the denominator. Finished counts retain the full benchmark task count. Per-sample scores are in `eval/<task>/sample_00/summary.json`; failures retain `error.txt`.

All 138 tasks use a 20-step limit. Qwen3.5-27B used a 180-second step timeout. LLM-judged answers were regraded from saved final outputs using the updated Azure configuration on 2026-09-21. The JSON records hashes of the regrade manifest and API verdicts; grading failures were not converted to zero. Task concurrency reached 12. Commit `2799848` makes future grader API failures propagate as rollout errors instead of zero rewards.

## Reproduction

Prepare the environment according to the [androidlab guide](/lite/gym/envs/androidlab/README.md), configure API credentials or local model weights, and export `CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN`. Then run:

```bash
export EVAL_RUN_ID=run_0
./devs/exps/eval/androidlab/run.sh gpt-5.5
```

Use the corresponding model ID for other rows. Saved `run_info.txt` files contain the actual launch commands and concurrency. The paired JSON preserves operational overrides and source provenance.
