# browsergym.miniwob @ 2026-09-13T00-13_9685915 · run_0

- **Commit**: `9685915` — `Add MiniWoB eval mappings for new agents`
- **Host / GPUs**: `gpublaze` / `2,3,4,5`
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-09-13T00-13_9685915/run_0/`
- **Started**: `2026-09-13 00:09 PDT`
- **Last updated**: `2026-09-13 01:06 PDT`
- **Notes**: `EVAL_MODE=default`; `EVAL_RUN_ID=run_0`; `EVAL_CONCURRENCY=4` for env-server headroom; env-server `http://169.229.219.180:30100` with token `lhr`; API credentials loaded via `~/env.sh`. Local models are scheduled across GPUs `2,3,4,5`; tensor-parallel models (`Qwen/Qwen3-VL-32B-Instruct`, `Qwen/Qwen3.5-27B`, `Qwen/Qwen3.8-27B`) use two GPUs.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `gpt-6-astra` | 125/125 | 0.8720 |
| `Qwen/Qwen3.5-27B` | 125/125 | 0.6880 |
| `Qwen/Qwen3.8-27B` | 125/125 | 0.6480 |
| `gpt-5.6-sol` | 125/125 | 0.6480 |
| `gpt-5.5` | 125/125 | 0.6240 |
| `Qwen/Qwen3-VL-32B-Instruct` | 125/125 | 0.6080 |
| `inclusionAI/UI-Venus-2-9B` | 125/125 | 0.5920 |
| `meituan/EvoCUA-8B-20260105` | 125/125 | 0.5600 |
| `Qwen/Qwen3-VL-8B-Instruct` | 125/125 | 0.5360 |
| `Qwen/Qwen3.5-9B` | 125/125 | 0.5040 |
| `Qwen/Qwen3-VL-4B-Instruct` | 125/125 | 0.4720 |
| `Qwen/Qwen3-VL-2B-Instruct` | 125/125 | 0.3040 |
| `Qwen/Qwen3.5-4B` | 125/125 | 0.2800 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 125/125 | 0.1840 |

## Highlights

- Best completed model is `gpt-6-astra` at `0.8720` mean episode return.
- Completed all 14/14 requested default-config rows.
