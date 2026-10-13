# browsergym.miniwob @ 2026-09-21T19-18_1e14e3a · run_0

- **Commit**: `1e14e3a` — `Allow MiniWoB GPT reasoning effort override`
- **Host / GPUs**: `gpublaze` / `3,5,6,7` for local-model runs; API-only for GPT/Claude reruns
- **Artifacts**: `.exps/eval/browsergym.miniwob/`
- **Started**: `2026-09-13 00:09 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: Campaign leaderboard, not a single-run snapshot — it is the whole branch's MiniWoB matrix at this commit's pipeline state, so it has no sibling `run_0.json` and no `run_0/` artifact dir. `EVAL_MODE=default` and the 125-task eval throughout. Per the carry-forward convention in [AGENTS.md](/devs/exps/eval/AGENTS.md#carry-forward-convention), `📌` rows were measured at an earlier commit on this branch and the MER cell names that commit; the pipeline deltas between those commits are GPT/Claude-only (see [CHANGELOG](/devs/exps/eval/browsergym.miniwob/CHANGELOG)), so the local-model rows carry forward unchanged. Each contributing run has its own paired snapshot:
  - [`2026-09-13T00-13_9685915/run_0.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-13T00-13_9685915/run_0.md) — default-mode baseline, 14 models
  - [`2026-09-21T06-20_3f90798/run_0_think_default.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-21T06-20_3f90798/run_0_think_default.md) — `EVAL_ENABLE_THINKING=true`, 8 models
  - [`2026-09-21T08-10_a0c2997/run_0_qwen38_think_xhigh.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-21T08-10_a0c2997/run_0_qwen38_think_xhigh.md) — Qwen3.8 thinking
  - [`2026-09-21T18-55_94b0c45/run_0_gpt55_screenshot_fix.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-21T18-55_94b0c45/run_0_gpt55_screenshot_fix.md) — `gpt-5.5` after the no-op screenshot fix
  - [`2026-09-21T19-18_1e14e3a/run_0_gpt55_reasoning_xhigh.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-21T19-18_1e14e3a/run_0_gpt55_reasoning_xhigh.md), [`run_1_gpt56_claude_ablation.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-21T19-18_1e14e3a/run_1_gpt56_claude_ablation.md), [`run_2_reasoning_none.md`](/devs/exps/eval/browsergym.miniwob/logs/2026-09-21T19-18_1e14e3a/run_2_reasoning_none.md) — the effort sweep and the Claude row
- **Reproduce**: every row's `Artifacts` cell is a path under `.exps/eval/browsergym.miniwob/`, so it names the commit dir, the run id, and the model slug that produced the number; that directory's `run_info.txt` holds the verbatim `scripts/rollout.py` launch command. The `Config` column is the slug's `EVAL_CONFIG_ID` suffix.

## Results

| Model | Config | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---|---:|---:|---:|---|
| `claude-opus-5` | `sub2api_no_cache` | 125/125 | 0.9120 | 2.89 | `2026-09-21T19-18_1e14e3a/run_1_gpt56_claude_ablation/claude-opus-5__sub2api_no_cache/` |
| 📌 `gpt-6-astra` | `default` | 125/125 | 0.8720 (@9685915) | 4.04 | `2026-09-13T00-13_9685915/run_0/gpt-6-astra/` |
| 📌 `gpt-5.5` | `default` | 125/125 | 0.8480 (@94b0c45) | 2.71 | `2026-09-21T18-55_94b0c45/run_0_gpt55_screenshot_fix/gpt-5.5/` |
| `gpt-5.5` | `reasoning_xhigh` | 125/125 | 0.8480 | 3.08 | `2026-09-21T19-18_1e14e3a/run_0_gpt55_reasoning_xhigh/gpt-5.5__reasoning_xhigh/` |
| `gpt-5.6-sol` | `reasoning_xhigh` | 125/125 | 0.8320 | 2.99 | `2026-09-21T19-18_1e14e3a/run_1_gpt56_claude_ablation/gpt-5.6-sol__reasoning_xhigh/` |
| `gpt-5.6-sol` | `reasoning_medium` | 125/125 | 0.8240 | 2.79 | `2026-09-21T19-18_1e14e3a/run_1_gpt56_claude_ablation/gpt-5.6-sol__reasoning_medium/` |
| `gpt-5.6-sol` | `reasoning_none` | 125/125 | 0.7680 | 2.66 | `2026-09-21T19-18_1e14e3a/run_2_reasoning_none/gpt-5.6-sol__reasoning_none/` |
| `gpt-5.5` | `reasoning_none` | 125/125 | 0.7280 | 2.82 | `2026-09-21T19-18_1e14e3a/run_2_reasoning_none/gpt-5.5__reasoning_none/` |
| 📌 `Qwen/Qwen3-VL-32B-Thinking` | `think_on` | 125/125 | 0.7120 (@3f90798) | 2.99 | `2026-09-21T06-20_3f90798/run_0_think_default/Qwen_Qwen3-VL-32B-Thinking__think_on/` |
| 📌 `Qwen/Qwen3.5-27B` | `default` | 125/125 | 0.6880 (@9685915) | 2.80 | `2026-09-13T00-13_9685915/run_0/Qwen_Qwen3.5-27B/` |
| 📌 `Qwen/Qwen3.5-27B` | `think_on` | 125/125 | 0.6880 (@3f90798) | 2.66 | `2026-09-21T06-20_3f90798/run_0_think_default/Qwen_Qwen3.5-27B__think_on/` |
| 📌 `Qwen/Qwen3.8-27B` | `default` | 125/125 | 0.6480 (@9685915) | 3.44 | `2026-09-13T00-13_9685915/run_0/Qwen_Qwen3.8-27B/` |
| 📌 `Qwen/Qwen3-VL-32B-Instruct` | `default` | 125/125 | 0.6080 (@9685915) | 2.72 | `2026-09-13T00-13_9685915/run_0/Qwen_Qwen3-VL-32B-Instruct/` |
| 📌 `Qwen/Qwen3.8-27B` | `think_xhigh` | 125/125 | 0.6000 (@a0c2997) | 3.17 | `2026-09-21T08-10_a0c2997/run_0_qwen38_think_xhigh/Qwen_Qwen3.8-27B__think_xhigh/` |
| 📌 `Qwen/Qwen3-VL-8B-Thinking` | `think_on` | 125/125 | 0.5920 (@3f90798) | 2.72 | `2026-09-21T06-20_3f90798/run_0_think_default/Qwen_Qwen3-VL-8B-Thinking__think_on/` |
| 📌 `inclusionAI/UI-Venus-2-9B` | `default` | 125/125 | 0.5920 (@9685915) | 2.96 | `2026-09-13T00-13_9685915/run_0/inclusionAI_UI-Venus-2-9B/` |
| 📌 `meituan/EvoCUA-8B-20260105` | `default` | 125/125 | 0.5600 (@9685915) | 2.79 | `2026-09-13T00-13_9685915/run_0/meituan_EvoCUA-8B-20260105/` |
| 📌 `Qwen/Qwen3-VL-8B-Instruct` | `default` | 125/125 | 0.5360 (@9685915) | 2.73 | `2026-09-13T00-13_9685915/run_0/Qwen_Qwen3-VL-8B-Instruct/` |
| 📌 `Qwen/Qwen3.5-9B` | `default` | 125/125 | 0.5040 (@9685915) | 2.81 | `2026-09-13T00-13_9685915/run_0/Qwen_Qwen3.5-9B/` |
| 📌 `Qwen/Qwen3-VL-4B-Thinking` | `think_on` | 125/125 | 0.4960 (@3f90798) | 2.82 | `2026-09-21T06-20_3f90798/run_0_think_default/Qwen_Qwen3-VL-4B-Thinking__think_on/` |
| 📌 `Qwen/Qwen3.5-9B` | `think_on` | 125/125 | 0.4960 (@3f90798) | 2.29 | `2026-09-21T06-20_3f90798/run_0_think_default/Qwen_Qwen3.5-9B__think_on/` |
| 📌 `Qwen/Qwen3-VL-4B-Instruct` | `default` | 125/125 | 0.4720 (@9685915) | 2.44 | `2026-09-13T00-13_9685915/run_0/Qwen_Qwen3-VL-4B-Instruct/` |
| 📌 `Qwen/Qwen3-VL-2B-Thinking` | `think_on` | 125/125 | 0.4000 (@3f90798) | 2.74 | `2026-09-21T06-20_3f90798/run_0_think_default/Qwen_Qwen3-VL-2B-Thinking__think_on/` |
| 📌 `Qwen/Qwen3-VL-2B-Instruct` | `default` | 125/125 | 0.3040 (@9685915) | 2.45 | `2026-09-13T00-13_9685915/run_0/Qwen_Qwen3-VL-2B-Instruct/` |
| 📌 `Qwen/Qwen3.5-4B` | `default` | 125/125 | 0.2800 (@9685915) | 2.29 | `2026-09-13T00-13_9685915/run_0/Qwen_Qwen3.5-4B/` |
| 📌 `Qwen/Qwen3.5-4B` | `think_on` | 125/125 | 0.2800 (@3f90798) | 1.94 | `2026-09-21T06-20_3f90798/run_0_think_default/Qwen_Qwen3.5-4B__think_on/` |
| 📌 `ByteDance-Seed/UI-TARS-1.5-7B` | `default` | 125/125 | 0.1840 (@9685915) | 1.52 | `2026-09-13T00-13_9685915/run_0/ByteDance-Seed_UI-TARS-1.5-7B/` |
| 📌 `Qwen/Qwen3.5-2B` | `think_on` | 125/125 | 0.1120 (@3f90798) | 1.86 | `2026-09-21T06-20_3f90798/run_0_think_default/Qwen_Qwen3.5-2B__think_on/` |

## Highlights

- `claude-opus-5` is the strongest MiniWoB row on the branch at `0.9120` / 2.89 successful turns.
- Reasoning effort matters for *presence*, not degree: `gpt-5.5` scores `0.7280` at `none` and `0.8480` at both `medium` and `xhigh`; `gpt-5.6-sol` goes `0.7680` → `0.8240` → `0.8320`. Above `medium` the extra effort only lengthens successful episodes.
- The no-op screenshot-scaling fix in `94b0c45` is the single largest API delta on the branch: `gpt-5.5` moves `0.6240` → `0.8480`. Its pre-fix row, and `gpt-5.6-sol`'s, are therefore left out of this leaderboard and kept only in the `9685915` snapshot.
- Thinking helps Qwen3-VL at every size (2B `0.3040` → `0.4000`, 32B `0.6080` → `0.7120`) but is flat-to-negative for Qwen3.5 (`27B` identical at `0.6880`, `9B` `0.5040` → `0.4960`) and negative for Qwen3.8 (`0.6480` → `0.6000`).
- Best local row is `Qwen/Qwen3-VL-32B-Thinking` at `0.7120`, still `0.20` below `claude-opus-5`.
- Every row reached 125/125, so there are no `⚠️` partials in this matrix.
