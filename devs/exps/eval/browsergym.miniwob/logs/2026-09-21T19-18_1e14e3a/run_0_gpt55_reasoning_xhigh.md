# browsergym.miniwob @ 2026-09-21T19-18_1e14e3a · run_0_gpt55_reasoning_xhigh

- **Commit**: `1e14e3a` — `Allow MiniWoB GPT reasoning effort override`
- **Host / GPUs**: API-only (no GPU)
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-09-21T19-18_1e14e3a/run_0_gpt55_reasoning_xhigh/`
- **Started**: `2026-09-21 19:19 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: First run at `1e14e3a`, which added the `EVAL_REASONING_EFFORT` knob; this row is `xhigh` via `--api-kwargs '{"reasoning_effort":"xhigh"}'`. The `medium` baseline is the `94b0c45` run and the `none` arm is `run_2_reasoning_none`.
- **Reproduce**: every row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_0_gpt55_reasoning_xhigh.json` carries the same numbers plus each row's `artifact_path` / `summary_path`.

## Results

| Model | Config | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---|---:|---:|---:|---|
| `gpt-5.5` | `reasoning_xhigh` | 125/125 | 0.8480 | 3.08 | `gpt-5.5__reasoning_xhigh/` |

## Highlights

- `xhigh` does not beat `medium` for `gpt-5.5` on MiniWoB — both score `0.8480` — but successful episodes lengthen from 2.71 to 3.08 turns, so the extra reasoning is spent, not banked.
- Read together with `run_2_reasoning_none` (`0.7280`), MiniWoB is sensitive to having *some* reasoning and flat above that.
