# browsergym.miniwob @ 2026-09-13T00-13_9685915 · run_0

- **Commit**: `9685915` — `Add MiniWoB eval mappings for new agents`
- **Host / GPUs**: `gpublaze` / `3,5,6,7` for local models; API-only for GPT
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-09-13T00-13_9685915/run_0/`
- **Started**: `2026-09-13 00:09 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: `EVAL_MODE=default` (screenshot + coord), 125-task MiniWoB eval, `--concurrency 4`. This is the branch's default-mode baseline campaign; the thinking arm is at [`2026-09-21T06-20_3f90798/run_0_think_default.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-21T06-20_3f90798/run_0_think_default.md). The `gpt-5.5` and `gpt-5.6-sol` rows here predate the no-op screenshot-scaling fix in `94b0c45` and are superseded by the reruns at `94b0c45` / `1e14e3a`.
- **Reproduce**: every row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_0.json` carries the same numbers plus each row's `artifact_path` / `summary_path`.

## Results

| Model | Config | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---|---:|---:|---:|---|
| `gpt-6-astra` | `default` | 125/125 | 0.8720 | 4.04 | `gpt-6-astra/` |
| `Qwen/Qwen3.5-27B` | `default` | 125/125 | 0.6880 | 2.80 | `Qwen_Qwen3.5-27B/` |
| `Qwen/Qwen3.8-27B` | `default` | 125/125 | 0.6480 | 3.44 | `Qwen_Qwen3.8-27B/` |
| `gpt-5.6-sol` | `default` | 125/125 | 0.6480 | 4.06 | `gpt-5.6-sol/` |
| `gpt-5.5` | `default` | 125/125 | 0.6240 | 4.17 | `gpt-5.5/` |
| `Qwen/Qwen3-VL-32B-Instruct` | `default` | 125/125 | 0.6080 | 2.72 | `Qwen_Qwen3-VL-32B-Instruct/` |
| `inclusionAI/UI-Venus-2-9B` | `default` | 125/125 | 0.5920 | 2.96 | `inclusionAI_UI-Venus-2-9B/` |
| `meituan/EvoCUA-8B-20260105` | `default` | 125/125 | 0.5600 | 2.79 | `meituan_EvoCUA-8B-20260105/` |
| `Qwen/Qwen3-VL-8B-Instruct` | `default` | 125/125 | 0.5360 | 2.73 | `Qwen_Qwen3-VL-8B-Instruct/` |
| `Qwen/Qwen3.5-9B` | `default` | 125/125 | 0.5040 | 2.81 | `Qwen_Qwen3.5-9B/` |
| `Qwen/Qwen3-VL-4B-Instruct` | `default` | 125/125 | 0.4720 | 2.44 | `Qwen_Qwen3-VL-4B-Instruct/` |
| `Qwen/Qwen3-VL-2B-Instruct` | `default` | 125/125 | 0.3040 | 2.45 | `Qwen_Qwen3-VL-2B-Instruct/` |
| `Qwen/Qwen3.5-4B` | `default` | 125/125 | 0.2800 | 2.29 | `Qwen_Qwen3.5-4B/` |
| `ByteDance-Seed/UI-TARS-1.5-7B` | `default` | 125/125 | 0.1840 | 1.52 | `ByteDance-Seed_UI-TARS-1.5-7B/` |

## Highlights

- `gpt-6-astra` leads at `0.8720`; the other two GPT rows are depressed by the screenshot bug `94b0c45` fixed — `gpt-5.5` reruns at `0.8480` once fixed.
- Among local models `Qwen/Qwen3.5-27B` (`0.6880`) edges out `Qwen/Qwen3.8-27B` (`0.6480`) and `Qwen/Qwen3-VL-32B-Instruct` (`0.6080`).
- Qwen3-VL scales monotonically with size (2B `0.3040` → 4B `0.4720` → 8B `0.5360` → 32B `0.6080`); Qwen3.5 does not (4B `0.2800` → 9B `0.5040` → 27B `0.6880`).
- `ByteDance-Seed/UI-TARS-1.5-7B` is last at `0.1840` and also has the shortest successful episodes (1.52 turns), i.e. it only solves the shallow tasks.
- All 14 rows reached 125/125, so no mop-up round was needed at this commit.
