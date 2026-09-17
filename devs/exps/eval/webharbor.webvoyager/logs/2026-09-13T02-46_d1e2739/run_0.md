# webharbor.webvoyager @ 2026-09-13T02-46_d1e2739 · run_0

- **Commit**: `d1e2739` — `Add WebVoyager eval mappings`
- **Host / GPUs**: `gpublaze` / `2-5`
- **Artifacts**: `.exps/eval/webharbor.webvoyager/2026-09-13T02-46_d1e2739/run_0/`
- **Started**: `2026-09-13 04:56 PDT`
- **Last updated**: `2026-09-14 15:13 PDT`
- **Notes**: default cfg only; local owned env-servers at `127.0.0.1:30101-30104` because the provided `30100` server registered `webharbor.webvoyager` but its backend was unavailable; `EVAL_CONCURRENCY=4` for full passes, targeted mop-up rounds used lower concurrency plus `step_timeout` overrides where needed.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `gpt-6-astra` | 643/643 | 0.8429 |
| `gpt-5.6-sol` | 643/643 | 0.7978 |
| `gpt-5.5` | 643/643 | 0.7838 |
| `inclusionAI/UI-Venus-2-9B` | 643/643 | 0.7294 |
| `Qwen/Qwen3.8-27B` | 643/643 | 0.5537 |
| `Qwen/Qwen3.5-9B` | 643/643 | 0.3499 |
| `meituan/EvoCUA-8B-20260105` | 643/643 | 0.0575 |
| ⚠️ `Qwen/Qwen3.5-27B` | _**638/643**_ | _**0.5172**_ |
| ⚠️ `ByteDance-Seed/UI-TARS-1.5-7B` | _**615/643**_ | _**0.4618**_ |
| ⚠️ `Qwen/Qwen3-VL-32B-Instruct` | _**636/643**_ | _**0.4277**_ |
| ⚠️ `Qwen/Qwen3-VL-8B-Instruct` | _**632/643**_ | _**0.2927**_ |
| ⚠️ `Qwen/Qwen3-VL-4B-Instruct` | _**641/643**_ | _**0.2059**_ |
| ⚠️ `Qwen/Qwen3.5-4B` | _**640/643**_ | _**0.1219**_ |

## Highlights

- `gpt-6-astra` leads the completed rows at 0.8429 MER; `gpt-5.6-sol`, `gpt-5.5`, and `inclusionAI/UI-Venus-2-9B` are also complete and above 0.72 MER.
- `Qwen/Qwen3.5-27B` is partial: the 5 residual tasks repeatedly failed in the WebVoyager container with Selenium stale-element / step 500 errors after targeted mop-up.
- `ByteDance-Seed/UI-TARS-1.5-7B` is partial: the 28 residual tasks repeatedly failed with Selenium stale-element or click-intercepted step 500 errors after targeted mop-up.
- `Qwen/Qwen3-VL-32B-Instruct` is partial: the 7 residual tasks repeatedly failed with Selenium stale-element / click-intercepted step 500 errors after targeted mop-up.
- `Qwen/Qwen3-VL-8B-Instruct` is partial: the 11 residual tasks repeatedly failed with Selenium stale-element step 500 errors after targeted mop-up.
- `Qwen/Qwen3-VL-4B-Instruct` is partial: the 2 residual tasks repeatedly failed with Selenium stale-element / click-intercepted step 500 errors after targeted mop-up.
- `Qwen/Qwen3.5-4B` is partial: the 3 residual tasks repeatedly failed with Selenium stale-element / click-intercepted step 500 errors after targeted mop-up.
