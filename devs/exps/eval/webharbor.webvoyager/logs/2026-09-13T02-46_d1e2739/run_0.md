# webharbor.webvoyager @ 2026-09-13T02-46_d1e2739 · run_0

- **Commit**: `d1e2739` — `Add WebVoyager eval mappings`
- **Host / GPUs**: `gpublaze` / `2-5`
- **Artifacts**: `.exps/eval/webharbor.webvoyager/2026-09-13T02-46_d1e2739/run_0/`
- **Started**: `2026-09-13 04:56 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: Default cfg only, 643-task read-only split (`--filter 'lambda m: not m.others.get("mutating")'`). Local owned env-servers at `127.0.0.1:30101-30104`, because the provided `30100` server registered `webharbor.webvoyager` but its backend was unavailable. `EVAL_CONCURRENCY=4` for full passes; targeted mop-up rounds used lower concurrency plus `step_timeout` overrides where needed. `Finished` is against the 643-task eval set; a row's `summary.json` may report a smaller `num_tasks`, which is the task count its final invocation loaded. The thinking / reasoning arm is at [`2026-09-22T01-45_c50ae84/run_0_thinking_xhigh.md`](/devs/exps/eval/webharbor.webvoyager/logs/2026-09-22T01-45_c50ae84/run_0_thinking_xhigh.md).
- **Reproduce**: every row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_0.json` carries the same numbers plus each row's `artifact_path` / `summary_path`.

## Results

| Model | Config | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---|---:|---:|---:|---|
| `gpt-6-astra` | `default` | 643/643 | 0.8429 | 7.84 | `gpt-6-astra__default/` |
| `gpt-5.6-sol` | `default` | 643/643 | 0.7978 | 8.42 | `gpt-5.6-sol__default/` |
| `gpt-5.5` | `default` | 643/643 | 0.7838 | 8.33 | `gpt-5.5__default/` |
| `inclusionAI/UI-Venus-2-9B` | `default` | 643/643 | 0.7294 | 10.54 | `inclusionAI_UI-Venus-2-9B__default/` |
| `Qwen/Qwen3.8-27B` | `default` | 643/643 | 0.5537 | 8.86 | `Qwen_Qwen3.8-27B__default/` |
| `Qwen/Qwen3.5-9B` | `default` | 643/643 | 0.3499 | 7.84 | `Qwen_Qwen3.5-9B__default/` |
| `meituan/EvoCUA-8B-20260105` | `default` | 643/643 | 0.0575 | 10.16 | `meituan_EvoCUA-8B-20260105__default/` |
| ⚠️ `Qwen/Qwen3.5-27B` | `default` | _**638/643**_ | _**0.5172**_ | 7.55 | `Qwen_Qwen3.5-27B__default/` |
| ⚠️ `ByteDance-Seed/UI-TARS-1.5-7B` | `default` | _**615/643**_ | _**0.4618**_ | 7.27 | `ByteDance-Seed_UI-TARS-1.5-7B__default/` |
| ⚠️ `Qwen/Qwen3-VL-32B-Instruct` | `default` | _**636/643**_ | _**0.4277**_ | 7.23 | `Qwen_Qwen3-VL-32B-Instruct__default/` |
| ⚠️ `Qwen/Qwen3-VL-8B-Instruct` | `default` | _**632/643**_ | _**0.2927**_ | 8.23 | `Qwen_Qwen3-VL-8B-Instruct__default/` |
| ⚠️ `Qwen/Qwen3-VL-4B-Instruct` | `default` | _**641/643**_ | _**0.2059**_ | 8.58 | `Qwen_Qwen3-VL-4B-Instruct__default/` |
| ⚠️ `Qwen/Qwen3.5-4B` | `default` | _**640/643**_ | _**0.1219**_ | 7.69 | `Qwen_Qwen3.5-4B__default/` |

## Highlights

- `gpt-6-astra` leads the completed rows at `0.8429`; `gpt-5.6-sol`, `gpt-5.5` and `inclusionAI/UI-Venus-2-9B` are also complete and above `0.72`.
- `meituan/EvoCUA-8B-20260105` collapses to `0.0575` here despite 643/643 coverage and the longest successful episodes (10.16 turns) — it burns steps without finishing.
- Qwen3.5 scales cleanly on WebVoyager (4B `0.1219` → 9B `0.3499` → 27B `0.5172`), and `Qwen/Qwen3.8-27B` is the best local row at `0.5537`.
- `Qwen/Qwen3.5-27B` is partial: the 5 residual tasks repeatedly failed in the WebVoyager container with Selenium stale-element / step 500 errors after targeted mop-up.
- `ByteDance-Seed/UI-TARS-1.5-7B` is partial: the 28 residual tasks repeatedly failed with Selenium stale-element or click-intercepted step 500 errors after targeted mop-up.
- `Qwen/Qwen3-VL-32B-Instruct` (7 residual), `Qwen/Qwen3-VL-8B-Instruct` (11), `Qwen/Qwen3-VL-4B-Instruct` (2) and `Qwen/Qwen3.5-4B` (3) are partial for the same reason; mop-up plateaued on each.
