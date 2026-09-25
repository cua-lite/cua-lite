# osworld @ 2026-09-17T14-28_a50cc31 - run_0

- **Commit**: `a50cc31` - OSWorld reference models.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/osworld/2026-09-17T14-28_a50cc31/run_0/` (gitignored).
- **Started**: `2026-09-14 PDT`.
- **Last updated**: `2026-09-25 00:43 PDT`.
- **Notes**: Official OSWorld `eval`, 325 tasks after the `exclude_reason` filter. Up to 30 task workers; Docker creation concurrency 10. Consolidated sources; Claude code drifted mid-campaign (see Provenance).

## Results

| Model | Finished | Mean episode return |
|---|---:|---:|
| `claude-opus-5` | 325/325 | 0.787319 |
| `gpt-6-astra` | 325/325 | 0.751382 |
| `gpt-5.6-sol` | 325/325 | 0.732611 |
| `gpt-5.5` | 325/325 | 0.710893 |
| `Qwen/Qwen3.8-27B` | 325/325 | 0.591711 |
| `inclusionAI/UI-Venus-2-9B` | 325/325 | 0.504743 |
| `Qwen/Qwen3.5-27B` | 325/325 | 0.489466 |
| `Qwen/Qwen3.5-9B` | 325/325 | 0.385763 |
| `meituan/EvoCUA-8B-20260105` | 325/325 | 0.355749 |
| `Qwen/Qwen3-VL-32B-Instruct` | 325/325 | 0.342914 |
| `Qwen/Qwen3-VL-8B-Instruct` | 325/325 | 0.281322 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 325/325 | 0.274721 |
| `Qwen/Qwen3.5-4B` | 325/325 | 0.270294 |
| `Qwen/Qwen3-VL-4B-Instruct` | 325/325 | 0.236474 |
| ⚠️ `gemini-3.6-flash` | _**0/325**_ | _**—**_ |

Mean episode return uses valid sample summaries only. API/environment errors
are excluded from the mean; normal zero rewards and scored model-format failures
remain included. Finished counts retain the full filtered task total.

## Highlights

- Gemini 3.6 Flash: not started; no recorded campaign.

## Experiment Specification

| Model | YAML | Sampling | History |
|---|---|---|---|
| GPT-5.5 / GPT-5.6 Sol / GPT-6 Astra | [gpt/osworld.yaml](/scripts/configs/gpt/default/osworld.yaml) | Medium; 4096 output tokens | Chained Responses history |
| Claude Opus 5 | [claude/osworld.yaml](/scripts/configs/claude/default/osworld.yaml) | Medium; 4096 output tokens | Full history |
| Qwen3-VL-4B / 8B / 32B (Instruct) | [qwen3_vl/osworld.yaml](/scripts/configs/qwen3_vl/default/osworld.yaml) | Temperature 0; 2048 output tokens | 4 full / 100 summary turns |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/osworld.yaml](/scripts/configs/qwen3_5/default/osworld.yaml) | Thinking off; temperature 0; 2048 output tokens | History 100 / image 4 / fold 4 |
| Qwen3.8-27B | [qwen3_8/osworld.yaml](/scripts/configs/qwen3_8/default/osworld.yaml) | Thinking off; temperature 0; 2048 output tokens | History 100 / image 4 / fold 4 |
| UI-TARS-1.5-7B | [ui_tars_15_v1/osworld.yaml](/scripts/configs/ui_tars_15_v1/default/osworld.yaml) | Temperature 0; 2048 output tokens | 5 full turns |
| EvoCUA-8B | [evocua/osworld.yaml](/scripts/configs/evocua/default/osworld.yaml) | Temperature .01; top-p .9; 2048 output tokens | 4 full / 100 summary turns |
| UI-Venus-2-9B | [ui_venus_2/osworld.yaml](/scripts/configs/ui_venus_2/default/osworld.yaml) | Temperature 0; top-p .7; 4096 output tokens | 2 past images + current |

Measured rows use 30 steps, native 1920x1080 observations and 2-second post-action
delay, with no agent-level resolution override. The default reset and step
timeouts are 600 and 180 seconds. Local models use `terminate` and loop
detection 5; UI-Venus also enables `response`, and GPT/Claude use loop detection 0.
UI-Venus uses its native thinking template. Greedy decoding does not use top-p.

### Reproduction

Prepare the image and Ubuntu disk using the
[OSWorld guide](/lite/gym/envs/osworld/README.md). The image includes the
process-scoped THP wrapper. Start an OSWorld env-server with the following
environment, then export its `CUA_LITE_ENV_SERVER_URL` and
`CUA_LITE_ENV_SERVER_TOKEN` on the rollout host:

```bash
export CUA_LITE_DOCKER_CREATE_CONCURRENCY=10
export CUA_LITE_DRIFT_SAFETY_MARGIN_S=3600
export CUA_LITE_503_DEADLINE_S=600
```

Configure API credentials for API-model runs and cache local model weights.

```bash
export EVAL_RUN_ID=run_0
export EVAL_CONCURRENCY=30
CUDA_VISIBLE_DEVICES=0,1 ./devs/exps/eval/osworld/run.sh Qwen/Qwen3-VL-32B-Instruct
./devs/exps/eval/osworld/run.sh gpt-5.5
./devs/exps/eval/osworld/run.sh claude-opus-5
```

Use the corresponding model ID and YAML for the other rows. Keep the same run
ID and config selection to resume. Set concurrency to the host and API quota.

## Provenance

The paired [JSON snapshot](/devs/exps/eval/osworld/logs/2026-09-17T14-28_a50cc31/run_0.json)
records each model ID, YAML, raw summary path, summary SHA-256, and launch-log
path. Raw launch logs retain the actual commands and recorded revisions.

Claude artifacts are in `.exps/eval/osworld/probe_20260918/run_3/claude-opus-5/`;
this directory contains the full filtered campaign despite its original name.
The snapshot artifact root links to this retained directory; `source_artifact_path`
in JSON records the original location.
Code drifted mid-campaign: the launch log records initial local Claude patches
and the later merged implementation. `resume_20260920.json` identifies replaced
trajectories; all 18 mouse-modifier repair tasks now have valid summaries.
