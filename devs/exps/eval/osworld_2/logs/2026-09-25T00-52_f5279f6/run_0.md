# osworld_2 @ 2026-09-25T00-52_f5279f6 - run_0

- **Commit**: `f5279f6` — reference snapshot.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/osworld_2/2026-09-25T00-52_f5279f6/run_0/` (gitignored).
- **Started**: `2026-09-22 PDT`.
- **Last updated**: `2026-10-14 PDT`.
- **Notes**: OSWorld 2.1, 99 selected tasks, including task 072 (the earlier Max run selected 98 without 072); simulated-user, multi-phase and unprovisioned-service tasks excluded.
- **Provenance**: [JSON snapshot](/devs/exps/eval/osworld_2/logs/2026-09-25T00-52_f5279f6/run_0.json) records source runs, task replacements, configurations and artifact hashes.

## Results

| Model | Config | Finished | Mean episode return |
|---|---|---:|---:|
| `gpt-6-astra` | default | 99/99 | 0.633955 |
| `gpt-5.6-sol` | default | 99/99 | 0.441434 |
| `gpt-5.5` | default | 99/99 | 0.385766 |
| `Qwen/Qwen3.5-27B` | default | 99/99 | 0.028944 |
| `Qwen/Qwen3.8-27B` | default | 99/99 | 0.023366 |
| `inclusionAI/UI-Venus-2-9B` | default | 99/99 | 0.014015 |
| `Qwen/Qwen3-VL-32B-Instruct` | default | 99/99 | 0.011469 |
| `Qwen/Qwen3-VL-4B-Instruct` | default | 99/99 | 0.008542 |
| `Qwen/Qwen3.5-4B` | default | 99/99 | 0.007295 |
| `meituan/EvoCUA-8B-20260105` | default | 99/99 | 0.006436 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | default | 99/99 | 0.004706 |
| `Qwen/Qwen3.5-9B` | default | 99/99 | 0.004178 |
| `Qwen/Qwen3-VL-8B-Instruct` | default | 99/99 | 0.002854 |
| ⚠️ `claude-opus-5` | max | _**74/98**_ | _**0.630739**_ |
| ⚠️ `claude-opus-5` | default | _**95/99**_ | _**0.599194**_ |

Mean episode return excludes API/environment errors and includes valid zero rewards. Finished counts valid samples out of each run’s selected tasks. Claude’s default configuration uses Medium.

## Highlights

- Claude Opus 5 Max: 74 valid results out of 98 selected tasks; 24 API errors are excluded. This finalized run predates the inclusion of task 072.
- Claude Opus 5 Medium: 95 valid results; tasks 047, 048, 051 and 099 ended with API 502 errors and are excluded from the mean. The campaign is finalized; no tasks remain pending.

## Experiment Specification

| Models | YAML | Key settings |
|---|---|---|
| Qwen3-VL-4B / 8B / 32B Instruct | [qwen3_vl/osworld_2.yaml](/scripts/configs/qwen3_vl/default/osworld_2.yaml) | 200 steps; loop detection 5; thinking off |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/osworld_2.yaml](/scripts/configs/qwen3_5/default/osworld_2.yaml) | 200 steps; loop detection 5; thinking off |
| Qwen3.8-27B | [qwen3_8/osworld_2.yaml](/scripts/configs/qwen3_8/default/osworld_2.yaml) | 200 steps; loop detection 5; thinking off |
| UI-TARS-1.5-7B | [ui_tars_15_v1/osworld_2.yaml](/scripts/configs/ui_tars_15_v1/default/osworld_2.yaml) | 200 steps; loop detection 5 |
| EvoCUA-8B | [evocua/osworld_2.yaml](/scripts/configs/evocua/default/osworld_2.yaml) | 200 steps; loop detection 5 |
| UI-Venus-2-9B | [ui_venus_2/osworld_2.yaml](/scripts/configs/ui_venus_2/default/osworld_2.yaml) | 200 steps; loop detection 5; OSWorld 2.1 guest password |
| GPT-5.5 / GPT-5.6 Sol / GPT-6 Astra | [gpt/osworld_2.yaml](/scripts/configs/gpt/default/osworld_2.yaml) | 200 steps; Medium; 8192 output tokens |
| Claude Opus 5 (default) | [claude/osworld_2.yaml](/scripts/configs/claude/default/osworld_2.yaml) | 200 steps; inherited Medium; 64000 output tokens |
| Claude Opus 5 (Max) | [claude/osworld_2.max.yaml](/scripts/configs/claude/default/osworld_2.max.yaml) | 200 steps; Max; 64000 output tokens |

Other unspecified sampling and history parameters inherit the model defaults. API models have loop detection disabled. LLM judges retain each task's upstream model and scoring rules; failed judge calls are excluded from the mean.

### Reproduction

Install the pinned image and services using the [OSWorld 2.1 guide](/lite/gym/envs/osworld_2/README.md), then export the env-server URL/token and model/judge API credentials. Run from the repository root:

```bash
./devs/exps/eval/osworld_2/run.sh Qwen/Qwen3-VL-8B-Instruct
./devs/exps/eval/osworld_2/run.sh gpt-5.5
./devs/exps/eval/osworld_2/run.sh claude-opus-5
./devs/exps/eval/osworld_2/run.sh claude-opus-5 scripts/configs/claude/default/osworld_2.max.yaml
```

The runner selects each model's default YAML unless a variant is supplied. A fresh reproduction writes its own commit-keyed artifacts.
