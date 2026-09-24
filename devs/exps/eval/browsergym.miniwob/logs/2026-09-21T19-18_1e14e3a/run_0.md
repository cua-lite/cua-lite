# browsergym.miniwob @ 2026-09-21T19-18_1e14e3a · run_0

- **Commit**: `1e14e3a` — `Allow MiniWoB GPT reasoning effort override`
- **Host / GPUs**: `gpublaze` / `3,5,6,7` for local-model runs; API-only for GPT/Claude reruns
- **Artifacts**: `.exps/eval/browsergym.miniwob/`
- **Started**: `2026-09-13 00:09 PDT`
- **Last updated**: `2026-09-23 05:59 PDT`
- **Notes**: `EVAL_MODE=default`; all rows are 125-task MiniWoB eval. Scores are mean episode return. Avg turns are over successful samples only. Open-source rows are copied from prior completed MiniWoB snapshots; API rows use the latest screenshot-fix/reasoning reruns. Claude used the default config with `api_kwargs.prompt_caching=false` because the sub2api Anthropic endpoint rejected the default prompt-caching request with more than four `cache_control` blocks. The sub2api base URL was supplied by mapping `ANTHROPIC_BASE_URL=$ANTHROPIC_API_BASE`.

## Results

| Model | Finished | Mean episode return | Successful avg turns | Source |
|---|---:|---:|---:|---|
| `Claude Opus 5` | 125/125 | 0.9120 | 2.89 | `.exps/eval/browsergym.miniwob/2026-09-21T19-18_1e14e3a/run_1_gpt56_claude_ablation/claude-opus-5__sub2api_no_cache/` |
| `GPT-6 Astra` | 125/125 | 0.8720 | 4.04 | prior default snapshot |
| `GPT-5.5 medium` | 125/125 | 0.8480 | 2.71 | `.exps/eval/browsergym.miniwob/2026-09-21T18-55_94b0c45/run_0_gpt55_screenshot_fix/gpt-5.5/` |
| `GPT-5.5 xhigh` | 125/125 | 0.8480 | 3.08 | `.exps/eval/browsergym.miniwob/2026-09-21T19-18_1e14e3a/run_0_gpt55_reasoning_xhigh/gpt-5.5__reasoning_xhigh/` |
| `GPT-5.6 Sol xhigh` | 125/125 | 0.8320 | 2.99 | `.exps/eval/browsergym.miniwob/2026-09-21T19-18_1e14e3a/run_1_gpt56_claude_ablation/gpt-5.6-sol__reasoning_xhigh/` |
| `GPT-5.6 Sol medium` | 125/125 | 0.8240 | 2.79 | `.exps/eval/browsergym.miniwob/2026-09-21T19-18_1e14e3a/run_1_gpt56_claude_ablation/gpt-5.6-sol__reasoning_medium/` |
| `GPT-5.6 Sol none` | 125/125 | 0.7680 | 2.66 | `.exps/eval/browsergym.miniwob/2026-09-21T19-18_1e14e3a/run_2_reasoning_none/gpt-5.6-sol__reasoning_none/` |
| `GPT-5.5 none` | 125/125 | 0.7280 | 2.82 | `.exps/eval/browsergym.miniwob/2026-09-21T19-18_1e14e3a/run_2_reasoning_none/gpt-5.5__reasoning_none/` |
| `Qwen/Qwen3-VL-32B-Thinking` | 125/125 | 0.7120 | 2.99 | prior thinking snapshot |
| `Qwen/Qwen3.5-27B` | 125/125 | 0.6880 | 2.80 | prior default snapshot |
| `Qwen/Qwen3.5-27B Thinking` | 125/125 | 0.6880 | 2.66 | prior thinking snapshot |
| `Qwen/Qwen3.8-27B` | 125/125 | 0.6480 | 3.44 | prior default snapshot |
| `Qwen/Qwen3-VL-32B-Instruct` | 125/125 | 0.6080 | 2.72 | prior default snapshot |
| `Qwen/Qwen3.8-27B Thinking xhigh` | 125/125 | 0.6000 | 3.17 | prior thinking snapshot |
| `Qwen/Qwen3-VL-8B-Thinking` | 125/125 | 0.5920 | 2.72 | prior thinking snapshot |
| `inclusionAI/UI-Venus-2-9B` | 125/125 | 0.5920 | 2.96 | prior default snapshot |
| `meituan/EvoCUA-8B-20260105` | 125/125 | 0.5600 | 2.79 | prior default snapshot |
| `Qwen/Qwen3-VL-8B-Instruct` | 125/125 | 0.5360 | 2.73 | prior default snapshot |
| `Qwen/Qwen3.5-9B` | 125/125 | 0.5040 | 2.81 | prior default snapshot |
| `Qwen/Qwen3.5-9B Thinking` | 125/125 | 0.4960 | 2.29 | prior thinking snapshot |
| `Qwen/Qwen3-VL-4B-Thinking` | 125/125 | 0.4960 | 2.82 | prior thinking snapshot |
| `Qwen/Qwen3-VL-4B-Instruct` | 125/125 | 0.4720 | 2.44 | prior default snapshot |
| `Qwen/Qwen3.5-4B` | 125/125 | 0.2800 | 2.29 | prior default snapshot |
| `Qwen/Qwen3.5-4B Thinking` | 125/125 | 0.2800 | 1.94 | prior thinking snapshot |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 125/125 | 0.1840 | 1.52 | prior default snapshot |

## Highlights

- Updated the Google Doc MiniWoB++ cells to `Claude Opus 5 = 91.2 / 2.89`, `GPT-5.5 none = 72.8 / 2.82`, `GPT-5.5 medium = 84.8 / 2.71`, `GPT-5.5 xhigh = 84.8 / 3.08`, `GPT-5.6 Sol none = 76.8 / 2.66`, `GPT-5.6 Sol medium = 82.4 / 2.79`, and `GPT-5.6 Sol xhigh = 83.2 / 2.99`.
- `Claude Opus 5` is the strongest recorded MiniWoB row in this snapshot at `0.9120`.
- `GPT-5.6 Sol xhigh` slightly improves over medium on score (`0.8320` vs `0.8240`) but uses more turns on successful samples (`2.99` vs `2.79`).
- Disabling GPT reasoning drops MiniWoB relative to the default reasoning rows: `GPT-5.5 none = 0.7280`, `GPT-5.6 Sol none = 0.7680`.
