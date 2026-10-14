# browsergym.miniwob @ 2026-09-21T19-18_1e14e3a · run_1_gpt56_claude_ablation

- **Commit**: `1e14e3a` — `Allow MiniWoB GPT reasoning effort override`
- **Host / GPUs**: API-only (no GPU)
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-09-21T19-18_1e14e3a/run_1_gpt56_claude_ablation/`
- **Started**: `2026-09-21 20:30 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: Second campaign at `1e14e3a`: the `gpt-5.6-sol` effort sweep plus the first Claude MiniWoB row. Claude used the default config with `api_kwargs.prompt_caching=false`, because the sub2api Anthropic endpoint rejects the default prompt-caching request once it carries more than four `cache_control` blocks; the base URL came from mapping `ANTHROPIC_BASE_URL=$ANTHROPIC_API_BASE`. The sibling `claude-opus-5__sub2api` artifact dir is the rejected prompt-caching attempt and has no `summary.json`, so it is deliberately absent from the table.
- **Reproduce**: every row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_1_gpt56_claude_ablation.json` carries the same numbers plus each row's `artifact_path` / `summary_path`.

## Results

| Model | Config | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---|---:|---:|---:|---|
| `claude-opus-5` | `sub2api_no_cache` | 125/125 | 0.9120 | 2.89 | `claude-opus-5__sub2api_no_cache/` |
| `gpt-5.6-sol` | `reasoning_xhigh` | 125/125 | 0.8320 | 2.99 | `gpt-5.6-sol__reasoning_xhigh/` |
| `gpt-5.6-sol` | `reasoning_medium` | 125/125 | 0.8240 | 2.79 | `gpt-5.6-sol__reasoning_medium/` |

## Highlights

- `claude-opus-5` is the strongest MiniWoB row on the branch at `0.9120`, ahead of `gpt-6-astra` (`0.8720`) and every GPT-5.x arm.
- `gpt-5.6-sol` gains little from effort: `xhigh` `0.8320` vs `medium` `0.8240`, at 2.99 vs 2.79 successful turns.
- `gpt-5.6-sol` at either effort trails `gpt-5.5` (`0.8480`) on this benchmark.
