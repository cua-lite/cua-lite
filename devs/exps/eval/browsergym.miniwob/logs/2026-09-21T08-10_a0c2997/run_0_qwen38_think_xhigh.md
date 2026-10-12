# browsergym.miniwob @ 2026-09-21T08-10_a0c2997 · run_0_qwen38_think_xhigh

- **Commit**: `a0c2997` — `Allow MiniWoB Qwen3.8 thinking evals`
- **Host / GPUs**: `gpublaze` / `3,5,6,7`
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-09-21T08-10_a0c2997/run_0_qwen38_think_xhigh/`
- **Started**: `2026-09-21 08:11 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: Single-model run: `a0c2997` extended `EVAL_ENABLE_THINKING` to the Qwen3.8 family, which the `3f90798` runner rejected. Compare against the model's `default` row (`0.6480`) at [`2026-09-13T00-13_9685915/run_0.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-13T00-13_9685915/run_0.md).
- **Reproduce**: every row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_0_qwen38_think_xhigh.json` carries the same numbers plus each row's `artifact_path` / `summary_path`.

## Results

| Model | Config | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---|---:|---:|---:|---|
| `Qwen/Qwen3.8-27B` | `think_xhigh` | 125/125 | 0.6000 | 3.17 | `Qwen_Qwen3.8-27B__think_xhigh/` |

## Highlights

- Thinking costs `Qwen/Qwen3.8-27B` accuracy on MiniWoB: `0.6000` with thinking vs `0.6480` default, while successful episodes shorten from 3.44 to 3.17 turns.
- This is the only Qwen3.8 thinking row on the branch; `EVAL_CONFIG_ID=think_xhigh` keeps its artifact dir separate from the `think_on` slugs at `3f90798`.
