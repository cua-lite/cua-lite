# browsergym.miniwob @ 2026-09-21T19-18_1e14e3a · run_2_reasoning_none

- **Commit**: `1e14e3a` — `Allow MiniWoB GPT reasoning effort override`
- **Host / GPUs**: API-only (no GPU)
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-09-21T19-18_1e14e3a/run_2_reasoning_none/`
- **Started**: `2026-09-23 05:10 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: Third campaign at `1e14e3a`: the `reasoning_effort=none` floor for both GPT models, via `--api-kwargs '{"reasoning_effort":"none"}'` at `--concurrency 16`. Compare `gpt-5.5` against `94b0c45` (`medium`) and `run_0_gpt55_reasoning_xhigh` (`xhigh`), and `gpt-5.6-sol` against the two arms in `run_1_gpt56_claude_ablation`.
- **Reproduce**: every row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_2_reasoning_none.json` carries the same numbers plus each row's `artifact_path` / `summary_path`.

## Results

| Model | Config | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---|---:|---:|---:|---|
| `gpt-5.6-sol` | `reasoning_none` | 125/125 | 0.7680 | 2.66 | `gpt-5.6-sol__reasoning_none/` |
| `gpt-5.5` | `reasoning_none` | 125/125 | 0.7280 | 2.82 | `gpt-5.5__reasoning_none/` |

## Highlights

- Turning reasoning off costs both models roughly a sixth of their score: `gpt-5.5` `0.8480` → `0.7280`, `gpt-5.6-sol` `0.8240` → `0.7680`.
- `gpt-5.6-sol` degrades less than `gpt-5.5` and overtakes it with reasoning off, reversing the ranking seen at `medium` and `xhigh`.
- Successful episode length barely moves (`gpt-5.5` 2.82, `gpt-5.6-sol` 2.66 turns), so the loss is failed tasks rather than longer ones.
