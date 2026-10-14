# webharbor.webvoyager @ 2026-09-22T01-45_c50ae84 · run_0_thinking_xhigh

- **Commit**: `c50ae84` — `Support WebVoyager thinking eval options`
- **Host / GPUs**: `gpublaze` / `3,5,6,7` + API agents
- **Artifacts**: `.exps/eval/webharbor.webvoyager/2026-09-22T01-45_c50ae84/run_0_thinking_xhigh/`
- **Started**: `2026-09-22 01:46 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: One run id holding every thinking / reasoning variant, kept apart by `EVAL_CONFIG_ID`: Qwen3-VL-Thinking and Qwen3.5 rows use `EVAL_ENABLE_THINKING=true` (`think_on`), Qwen3.8 uses `think_xhigh`, GPT rows set `EVAL_REASONING_EFFORT` (`reasoning_xhigh` / `reasoning_none`), and `claude-opus-5` ran the default config through the local sub2api `api_base`. Same 643-task read-only split as [`2026-09-13T02-46_d1e2739/run_0.md`](/devs/exps/eval/webharbor.webvoyager/logs/2026-09-13T02-46_d1e2739/run_0.md). `Finished` is against the 643-task eval set; the four partial rows' `summary.json` report a smaller `num_tasks` (639, 634, 628, 639), which is the task count their final invocation loaded rather than the eval set.
- **Reproduce**: every row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_0_thinking_xhigh.json` carries the same numbers plus each row's `artifact_path` / `summary_path`.

## Results

| Model | Config | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---|---:|---:|---:|---|
| `claude-opus-5` | `default` | 643/643 | 0.9036 | 9.44 | `claude-opus-5__default/` |
| `gpt-5.6-sol` | `reasoning_none` | 643/643 | 0.7574 | 6.33 | `gpt-5.6-sol__reasoning_none/` |
| `gpt-5.5` | `reasoning_none` | 643/643 | 0.7092 | 5.59 | `gpt-5.5__reasoning_none/` |
| `Qwen/Qwen3.8-27B` | `think_xhigh` | 643/643 | 0.6221 | 9.10 | `Qwen_Qwen3.8-27B__think_xhigh/` |
| `Qwen/Qwen3-VL-32B-Thinking` | `think_on` | 643/643 | 0.5070 | 8.64 | `Qwen_Qwen3-VL-32B-Thinking__think_on/` |
| `Qwen/Qwen3.5-27B` | `think_on` | 643/643 | 0.4806 | 7.47 | `Qwen_Qwen3.5-27B__think_on/` |
| `Qwen/Qwen3-VL-8B-Thinking` | `think_on` | 643/643 | 0.4028 | 6.03 | `Qwen_Qwen3-VL-8B-Thinking__think_on/` |
| `Qwen/Qwen3-VL-4B-Thinking` | `think_on` | 643/643 | 0.3717 | 6.77 | `Qwen_Qwen3-VL-4B-Thinking__think_on/` |
| `Qwen/Qwen3-VL-2B-Thinking` | `think_on` | 643/643 | 0.3608 | 6.33 | `Qwen_Qwen3-VL-2B-Thinking__think_on/` |
| `Qwen/Qwen3.5-2B` | `think_on` | 643/643 | 0.0575 | 7.41 | `Qwen_Qwen3.5-2B__think_on/` |
| ⚠️ `gpt-5.6-sol` | `reasoning_xhigh` | _**639/643**_ | _**0.7919**_ | 10.07 | `gpt-5.6-sol__reasoning_xhigh/` |
| ⚠️ `gpt-5.5` | `reasoning_xhigh` | _**633/643**_ | _**0.7646**_ | 9.72 | `gpt-5.5__reasoning_xhigh/` |
| ⚠️ `Qwen/Qwen3.5-9B` | `think_on` | _**627/643**_ | _**0.3668**_ | 6.53 | `Qwen_Qwen3.5-9B__think_on/` |
| ⚠️ `Qwen/Qwen3.5-4B` | `think_on` | _**638/643**_ | _**0.1661**_ | 6.05 | `Qwen_Qwen3.5-4B__think_on/` |

## Highlights

- `claude-opus-5` leads at `0.9036` over 643/643, ahead of every GPT arm and `0.28` above the best local row.
- `Qwen/Qwen3-VL-32B-Thinking` (`0.5070`) and `Qwen/Qwen3.8-27B think_xhigh` (`0.6221`) finished after the earlier revision of this snapshot recorded them as not started; both are now 643/643.
- GPT reasoning buys a little accuracy for a lot of turns: `gpt-5.6-sol` goes `0.7574` → `0.7919` from `none` to `xhigh` while successful episodes grow from 6.33 to 10.07 turns, and `gpt-5.5` goes `0.7092` → `0.7646` at 5.59 → 9.72 turns.
- Thinking helps Qwen3-VL at every size (2B `0.3608`, 4B `0.3717`, 8B `0.4028`, 32B `0.5070`) but hurts Qwen3.5-27B against its default row (`0.4806` vs `0.5172`) and lifts Qwen3.8-27B (`0.6221` vs `0.5537`).
- `Qwen/Qwen3.5-2B` is the floor at `0.0575`, matching EvoCUA's default-mode score.
- The four partial rows (`gpt-5.6-sol` xhigh, `gpt-5.5` xhigh, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B`) never loaded the full 643; their residual tasks are absent from the per-task summaries in this artifact directory and mop-up plateaued.
