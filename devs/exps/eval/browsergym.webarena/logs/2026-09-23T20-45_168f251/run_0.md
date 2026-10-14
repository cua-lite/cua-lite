# browsergym.webarena @ 2026-09-23T20-45_168f251 · run_0

- **Commit**: `168f251` — `Tune Claude WebArena prompt`
- **Host / GPUs**: `gpublaze` / API-only (no GPU)
- **Artifacts**: `.exps/eval/browsergym.webarena/2026-09-23T20-45_168f251/run_0/`
- **Started**: `2026-09-23 20:45 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: Claude-only campaign closing out the WebArena prompt work on this branch. Same 241-template read subset as [`2026-09-15T05-51_9f4d467/run_0.md`](/devs/exps/eval/browsergym.webarena/logs/2026-09-15T05-51_9f4d467/run_0.md), read from `lite/gym/envs/browsergym/data/webarena_241_templates.read.prompt_data.parquet`. Claude ran through the local sub2api endpoint via `--agent-kwargs '{"api_base":"http://127.0.0.1:8080"}'`, at `--concurrency 16`.
- **Reproduce**: the row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_0.json` carries the same numbers plus the row's `artifact_path` / `summary_path`.

## Results

| Model | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---:|---:|---:|---|
| ⚠️ `claude-opus-5` | _**240/241**_ | _**0.5167**_ | 9.64 | `claude-opus-5/` |

## Highlights

- `claude-opus-5` at `0.5167` MER lands between `gpt-5.5` (`0.4708`) and `gpt-5.6-sol` / `gpt-6-astra` (`0.5333`) from the `9f4d467` campaign, and is the only model re-run at this commit — the preceding rows carry forward from that snapshot rather than being restated here.
- Partial for the same reason as the API rows at `9f4d467`: all 241 task summaries were written, but one sample stayed invalid in the final summary, so MER is over the 240 valid samples.
- Three earlier commit dirs on this branch hold abandoned prompt iterations and deliberately have no snapshot: `2026-09-21T02-09_5cb1752` (125 tasks, 0 valid — ran before the sub2api `api_base` was wired in), `2026-09-23T19-49_b2ea303` (52 tasks) and `2026-09-23T20-14_90db8c8` (56 tasks), both stopped mid-pass to retune the answer prompt. None has a top-level `summary.json`; their `run_info.txt` files record what was launched.
