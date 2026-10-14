# browsergym.miniwob @ 2026-09-21T06-20_3f90798 · run_0_think_default

- **Commit**: `3f90798` — `Allow MiniWoB thinking evals in default mode`
- **Host / GPUs**: `gpublaze` / `3,5,6,7`
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-09-21T06-20_3f90798/run_0_think_default/`
- **Started**: `2026-09-21 06:21 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: `EVAL_MODE=default` with `EVAL_ENABLE_THINKING=true`, which `3f90798` first allowed outside `text_only` mode. Every row therefore carries the fixed override `--agent-kwargs '{"enable_thinking": true, "sampling_kwargs": {"max_new_tokens": 4096}}'` and lands in a `__think_on` slug. Pair each row with the same model's `default` row at [`2026-09-13T00-13_9685915/run_0.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-13T00-13_9685915/run_0.md).
- **Reproduce**: every row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_0_think_default.json` carries the same numbers plus each row's `artifact_path` / `summary_path`.

## Results

| Model | Config | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---|---:|---:|---:|---|
| `Qwen/Qwen3-VL-32B-Thinking` | `think_on` | 125/125 | 0.7120 | 2.99 | `Qwen_Qwen3-VL-32B-Thinking__think_on/` |
| `Qwen/Qwen3.5-27B` | `think_on` | 125/125 | 0.6880 | 2.66 | `Qwen_Qwen3.5-27B__think_on/` |
| `Qwen/Qwen3-VL-8B-Thinking` | `think_on` | 125/125 | 0.5920 | 2.72 | `Qwen_Qwen3-VL-8B-Thinking__think_on/` |
| `Qwen/Qwen3-VL-4B-Thinking` | `think_on` | 125/125 | 0.4960 | 2.82 | `Qwen_Qwen3-VL-4B-Thinking__think_on/` |
| `Qwen/Qwen3.5-9B` | `think_on` | 125/125 | 0.4960 | 2.29 | `Qwen_Qwen3.5-9B__think_on/` |
| `Qwen/Qwen3-VL-2B-Thinking` | `think_on` | 125/125 | 0.4000 | 2.74 | `Qwen_Qwen3-VL-2B-Thinking__think_on/` |
| `Qwen/Qwen3.5-4B` | `think_on` | 125/125 | 0.2800 | 1.94 | `Qwen_Qwen3.5-4B__think_on/` |
| `Qwen/Qwen3.5-2B` | `think_on` | 125/125 | 0.1120 | 1.86 | `Qwen_Qwen3.5-2B__think_on/` |

## Highlights

- Thinking is roughly score-neutral on MiniWoB: `Qwen/Qwen3.5-27B` is identical at `0.6880`, `Qwen/Qwen3.5-4B` identical at `0.2800`, and `Qwen/Qwen3.5-9B` moves `0.5040` → `0.4960`.
- What it does change is episode length — `Qwen/Qwen3.5-9B` drops from 2.81 to 2.29 successful turns and `Qwen/Qwen3.5-4B` from 2.29 to 1.94, so thinking buys shorter solutions rather than more of them.
- The Thinking checkpoints beat their Instruct siblings at every size: 2B `0.4000` vs `0.3040`, 4B `0.4960` vs `0.4720`, 8B `0.5920` vs `0.5360`, 32B `0.7120` vs `0.6080`.
- `Qwen/Qwen3.5-2B` is the weakest row in the branch at `0.1120`; `Qwen/Qwen3-VL-32B-Thinking` is the strongest local row at `0.7120`.
- All 8 rows reached 125/125.
