# browsergym.webarena @ 2026-09-15T05-51_9f4d467 · run_0

- **Commit**: `9f4d467` — `Add WebArena template eval pipeline`
- **Host / GPUs**: `gpublaze` / `4-7`
- **Artifacts**: `.exps/eval/browsergym.webarena/2026-09-15T05-51_9f4d467/run_0/`
- **Started**: `2026-09-15 05:54 PDT`
- **Last updated**: `2026-09-17 21:22 PDT`
- **Notes**: default cfg only; WebArena 241-template subset from parquet prompt data. Per-model fresh local env-servers were used after the original remote env-server became unavailable. EvoCUA and UI-Venus required external SGLang restart with `mem_fraction_static=0.90`.

## Results

| Model | Finished | Mean episode return |
|---|---:|---:|
| `inclusionAI/UI-Venus-2-9B` | 241/241 | 0.3071 |
| `meituan/EvoCUA-8B-20260105` | 241/241 | 0.1079 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 241/241 | 0.0913 |
| `Qwen/Qwen3-VL-4B-Instruct` | 241/241 | 0.0871 |
| `Qwen/Qwen3-VL-8B-Instruct` | 241/241 | 0.0705 |
| ⚠️ `gpt-5.6-sol` | _**240/241**_ | _**0.5333**_ |
| ⚠️ `gpt-6-astra` | _**240/241**_ | _**0.5333**_ |
| ⚠️ `gpt-5.5` | _**240/241**_ | _**0.4708**_ |
| ⚠️ `Qwen/Qwen3.8-27B` | _**233/241**_ | _**0.2446**_ |
| ⚠️ `Qwen/Qwen3.5-27B` | _**233/241**_ | _**0.2189**_ |
| ⚠️ `Qwen/Qwen3.5-9B` | _**240/241**_ | _**0.1542**_ |
| ⚠️ `Qwen/Qwen3-VL-32B-Instruct` | _**240/241**_ | _**0.1458**_ |
| ⚠️ `Qwen/Qwen3.5-4B` | _**240/241**_ | _**0.1125**_ |

## Highlights

- UI-Venus-2 is the strongest fully valid local run at `0.3071` MER.
- EvoCUA finished after the high-memory SGLang retry, ending at `0.1079` MER.
- The ⚠️ rows all produced 241 task summaries, but `stats.num_valid` remained below 241 after resume/mop-up; the MER shown is over valid samples.
- API models have the highest partial MERs, but each retained one invalid sample in the final summary.
