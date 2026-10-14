# browsergym.miniwob @ 2026-09-21T18-55_94b0c45 · run_0_gpt55_screenshot_fix

- **Commit**: `94b0c45` — `Fix BrowserGym no-op screenshot scaling`
- **Host / GPUs**: API-only (no GPU)
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-09-21T18-55_94b0c45/run_0_gpt55_screenshot_fix/`
- **Started**: `2026-09-21 18:56 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: Single-model verification run for `94b0c45`, which fixed BrowserGym returning an unscaled screenshot after a no-op action. `--concurrency 16`, default `reasoning_effort` (medium).
- **Reproduce**: every row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_0_gpt55_screenshot_fix.json` carries the same numbers plus each row's `artifact_path` / `summary_path`.

## Results

| Model | Config | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---|---:|---:|---:|---|
| `gpt-5.5` | `default` | 125/125 | 0.8480 | 2.71 | `gpt-5.5/` |

## Highlights

- The screenshot fix is worth `+0.2240` MER to `gpt-5.5`: `0.6240` at `9685915` → `0.8480` here, with successful episodes shortening from 4.17 to 2.71 turns.
- This supersedes the `gpt-5.5` and `gpt-5.6-sol` rows at [`2026-09-13T00-13_9685915/run_0.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-13T00-13_9685915/run_0.md); every API number from `94b0c45` onward is on the fixed path.
