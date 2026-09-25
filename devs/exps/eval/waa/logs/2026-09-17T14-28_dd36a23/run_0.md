# waa @ 2026-09-17T14-28_dd36a23 - run_0

- **Commit**: `dd36a23` - WAA reference runner and model configurations.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/waa/2026-09-17T14-28_dd36a23/run_0/` (gitignored).
- **Started**: `2026-09-09 PDT`.
- **Last updated**: `2026-09-25 00:43 PDT`.
- **Notes**: `eval`, 138 tasks after the `exclude_reason` filter. Task concurrency up to 30 per model and 60 total; Docker creation concurrency 10. Consolidated sources; Claude code drifted mid-campaign (see Provenance).

## Results

| Model | Finished | Mean episode return |
|---|---:|---:|
| `claude-opus-5` | 138/138 | 0.723965 |
| `gpt-5.6-sol` | 138/138 | 0.700250 |
| `gpt-5.5` | 138/138 | 0.665688 |
| `inclusionAI/UI-Venus-2-9B` | 138/138 | 0.412795 |
| `Qwen/Qwen3.8-27B` | 138/138 | 0.369995 |
| `Qwen/Qwen3.5-27B` | 138/138 | 0.254053 |
| `Qwen/Qwen3-VL-32B-Instruct` | 138/138 | 0.239560 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 138/138 | 0.167096 |
| `meituan/EvoCUA-8B-20260105` | 138/138 | 0.165989 |
| `Qwen/Qwen3-VL-8B-Instruct` | 138/138 | 0.145357 |
| `Qwen/Qwen3.5-9B` | 138/138 | 0.130864 |
| `Qwen/Qwen3.5-4B` | 138/138 | 0.116372 |
| `Qwen/Qwen3-VL-4B-Instruct` | 138/138 | 0.094637 |
| ⚠️ `gpt-6-astra` | _**137/138**_ | _**0.634359**_ |
| ⚠️ `gemini-3.6-flash` | _**0/138**_ | _**—**_ |

Mean episode return uses valid sample summaries only. API/environment errors
are excluded from the mean; normal zero rewards and scored model-format failures
remain included. Finished counts retain the full filtered task total.

## Highlights

- GPT-6 Astra: 137/138; one unsupported-key exception remains excluded.
- Gemini 3.6 Flash: not started; no recorded campaign.

## Experiment Specification

| Model | YAML | Sampling | History |
|---|---|---|---|
| GPT-5.5 / GPT-5.6 Sol / GPT-6 Astra | [gpt/waa.yaml](/scripts/configs/gpt/default/waa.yaml) | Medium; 4096 output tokens | Chained Responses history |
| Claude Opus 5 | [claude/waa.yaml](/scripts/configs/claude/default/waa.yaml) | Medium; 4096 output tokens | Full history |
| Qwen3-VL-4B / 8B / 32B (Instruct) | [qwen3_vl/waa.yaml](/scripts/configs/qwen3_vl/default/waa.yaml) | Temperature 0; 2048 output tokens | 4 full / 100 summary turns |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/waa.yaml](/scripts/configs/qwen3_5/default/waa.yaml) | Thinking off; temperature 0; 2048 output tokens | History 100 / image 4 / fold 4 |
| Qwen3.8-27B | [qwen3_8/waa.yaml](/scripts/configs/qwen3_8/default/waa.yaml) | Thinking off; temperature 0; 2048 output tokens | History 100 / image 4 / fold 4 |
| UI-TARS-1.5-7B | [ui_tars_15_v1/waa.yaml](/scripts/configs/ui_tars_15_v1/default/waa.yaml) | Temperature 0; 2048 output tokens | 5 full turns |
| EvoCUA-8B | [evocua/waa.yaml](/scripts/configs/evocua/default/waa.yaml) | Temperature .01; top-p .9; 2048 output tokens | 4 full / 100 summary turns |
| UI-Venus-2-9B | [ui_venus_2/waa.yaml](/scripts/configs/ui_venus_2/default/waa.yaml) | Temperature 0; top-p .7; 4096 output tokens | 2 past images + current |

Measured rows use 30 steps and native 1280x800 observations, with no agent-level
resolution override. Local models use `terminate` and loop detection 5; GPT and Claude
use loop detection 0. UI-Venus uses its native thinking template. Unspecified
settings inherit the referenced YAML and agent defaults.

### Reproduction

Prepare the image, assets and ready snapshot using the
[WAA guide](/lite/gym/envs/waa/README.md). Start a WAA env-server with
`CUA_LITE_DOCKER_CREATE_CONCURRENCY=10`; export its
`CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN` on the rollout host.
Configure API credentials for API-model runs and cache local model weights.

```bash
export EVAL_RUN_ID=run_0
export EVAL_CONCURRENCY=30
CUDA_VISIBLE_DEVICES=0 ./devs/exps/eval/waa/run.sh Qwen/Qwen3-VL-8B-Instruct scripts/configs/qwen3_vl/default/waa.yaml
./devs/exps/eval/waa/run.sh gpt-5.5 scripts/configs/gpt/default/waa.yaml
./devs/exps/eval/waa/run.sh claude-opus-5 scripts/configs/claude/default/waa.yaml
```

Use the corresponding model ID and YAML for the other rows. Keep the same run
ID and config selection to resume, with total task concurrency at most 60.
`EVAL_MODEL_PATH` and `EVAL_ENGINE_KWARGS` select local weights and serving options.

## Provenance

The paired [JSON snapshot](/devs/exps/eval/waa/logs/2026-09-17T14-28_dd36a23/run_0.json)
records each model ID, YAML, raw summary path, summary SHA-256, and launch-log
path. Raw launch logs retain the actual commands and recorded revisions.

Claude artifacts are in `.exps/eval/waa/probe_20260918/run_1/claude-opus-5/`;
this directory contains the full filtered campaign despite its original name.
The snapshot artifact root links to this retained directory; `source_artifact_path`
in JSON records the original location.
Code drifted mid-campaign: the launch log records initial local Claude patches
and the later merged implementation. `resume_20260920.json` identifies replaced
trajectories; all 8 mouse-modifier repair tasks now have valid summaries.
