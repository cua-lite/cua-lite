# browsergym.webarena @ 2026-09-15T05-51_9f4d467 · run_0

- **Commit**: `9f4d467` — `Add WebArena template eval pipeline`
- **Host / GPUs**: `gpublaze` / `4-7`
- **Artifacts**: `.exps/eval/browsergym.webarena/2026-09-15T05-51_9f4d467/run_0/`
- **Started**: `2026-09-15 05:54 PDT`
- **Last updated**: `2026-10-09 PDT`
- **Notes**: default cfg only; WebArena 241-template subset from parquet prompt data. Per-model fresh local env-servers were used after the original remote env-server became unavailable. EvoCUA and UI-Venus required external SGLang restart with `mem_fraction_static=0.90`.
- **Reproduce**: every row's `Artifacts` cell is a directory under the artifact root above; its `run_info.txt` holds the verbatim `scripts/rollout.py` launch command, the resume command, and the commit the rollout ran at. `run_0.json` carries the same numbers plus each row's `artifact_path` / `summary_path`.

## Results

| Model | Finished | Mean episode return | Successful avg turns | Artifacts |
|---|---:|---:|---:|---|
| `inclusionAI/UI-Venus-2-9B` | 241/241 | 0.3071 | 10.69 | `inclusionAI_UI-Venus-2-9B/` |
| `meituan/EvoCUA-8B-20260105` | 241/241 | 0.1079 | 7.62 | `meituan_EvoCUA-8B-20260105/` |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 241/241 | 0.0913 | 5.86 | `ByteDance-Seed_UI-TARS-1.5-7B/` |
| `Qwen/Qwen3-VL-4B-Instruct` | 241/241 | 0.0871 | 6.38 | `Qwen_Qwen3-VL-4B-Instruct/` |
| `Qwen/Qwen3-VL-8B-Instruct` | 241/241 | 0.0705 | 6.06 | `Qwen_Qwen3-VL-8B-Instruct/` |
| ⚠️ `gpt-5.6-sol` | _**240/241**_ | _**0.5333**_ | 7.04 | `gpt-5.6-sol/` |
| ⚠️ `gpt-6-astra` | _**240/241**_ | _**0.5333**_ | 8.41 | `gpt-6-astra/` |
| ⚠️ `gpt-5.5` | _**240/241**_ | _**0.4708**_ | 7.56 | `gpt-5.5/` |
| ⚠️ `Qwen/Qwen3.8-27B` | _**233/241**_ | _**0.2446**_ | 10.04 | `Qwen_Qwen3.8-27B/` |
| ⚠️ `Qwen/Qwen3.5-27B` | _**233/241**_ | _**0.2189**_ | 5.39 | `Qwen_Qwen3.5-27B/` |
| ⚠️ `Qwen/Qwen3.5-9B` | _**240/241**_ | _**0.1542**_ | 9.46 | `Qwen_Qwen3.5-9B/` |
| ⚠️ `Qwen/Qwen3-VL-32B-Instruct` | _**240/241**_ | _**0.1458**_ | 7.89 | `Qwen_Qwen3-VL-32B-Instruct/` |
| ⚠️ `Qwen/Qwen3.5-4B` | _**240/241**_ | _**0.1125**_ | 8.70 | `Qwen_Qwen3.5-4B/` |

## Highlights

- UI-Venus-2 is the strongest fully valid local run at `0.3071` MER.
- EvoCUA finished after the high-memory SGLang retry, ending at `0.1079` MER.
- The ⚠️ rows all produced 241 task summaries, but `stats.num_valid` remained below 241 after resume/mop-up; the MER shown is over valid samples.
- API models have the highest partial MERs, but each retained one invalid sample in the final summary.
- `Qwen/Qwen3.5-27B` and `Qwen/Qwen3.8-27B` were mopped up against a filtered prompt-data parquet written into their own log root, which their `run_info.txt` records as `--prompt-data …/.webarena_241_templates.filtered.read.prompt_data.parquet`.
- Claude Opus 5 is not in this run; its WebArena row lives in [`2026-09-23T20-45_168f251/run_0.md`](/devs/exps/eval/browsergym.webarena/logs/2026-09-23T20-45_168f251/run_0.md), which is the commit that finished the Claude prompt path.
