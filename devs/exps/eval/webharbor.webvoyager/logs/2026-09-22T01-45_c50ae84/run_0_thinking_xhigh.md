# webharbor.webvoyager @ 2026-09-22T01-45_c50ae84 · run_0_thinking_xhigh

- **Commit**: `c50ae84` — `Support WebVoyager thinking eval options`
- **Host / GPUs**: `gpublaze` / `3,5,6,7` + API agents
- **Artifacts**: `.exps/eval/webharbor.webvoyager/2026-09-22T01-45_c50ae84/run_0_thinking_xhigh/`
- **Started**: `2026-09-22 01:46 PDT`
- **Last updated**: `2026-09-23 00:26 PDT`
- **Notes**: Qwen3-VL-Thinking and Qwen3.5 runs use `EVAL_ENABLE_THINKING=true`; GPT rows use `EVAL_REASONING_EFFORT=xhigh`; Claude Opus 5 ran through the local sub2api `api_base`. Google Doc values use score percentage plus successful-run average steps.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `claude-opus-5` | 643/643 | 0.9036 |
| `Qwen/Qwen3.5-27B` | 643/643 | 0.4806 |
| `Qwen/Qwen3-VL-8B-Thinking` | 643/643 | 0.4028 |
| `Qwen/Qwen3-VL-4B-Thinking` | 643/643 | 0.3717 |
| `Qwen/Qwen3-VL-2B-Thinking` | 643/643 | 0.3608 |
| `Qwen/Qwen3.5-2B` | 643/643 | 0.0575 |
| ⚠️ `gpt-5.6-sol` | _**639/643**_ | _**0.7919**_ |
| ⚠️ `gpt-5.5` | _**633/643**_ | _**0.7646**_ |
| ⚠️ `Qwen/Qwen3.5-9B` | _**627/643**_ | _**0.3668**_ |
| ⚠️ `Qwen/Qwen3.5-4B` | _**638/643**_ | _**0.1661**_ |
| ⚠️ `Qwen/Qwen3-VL-32B-Thinking` | _**0/643**_ | _**—**_ |
| ⚠️ `Qwen/Qwen3.8-27B` | _**0/643**_ | _**—**_ |

## Highlights

- Claude Opus 5 leads the completed rows at 0.9036 MER.
- GPT-5.6 Sol xhigh and GPT-5.5 xhigh are partial; residual tasks are absent from the per-task summaries in this artifact directory.
- Qwen3.5-4B and Qwen3.5-9B thinking are partial; residual tasks are absent from the per-task summaries in this artifact directory.
- No `run_0_thinking_xhigh` artifact was found for `Qwen/Qwen3-VL-32B-Thinking` or `Qwen/Qwen3.8-27B`.
