# osworld_2 @ 2026-09-25T00-52_f5279f6 - run_0

- **Commit**: `f5279f6` - reference configuration and report anchor.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/osworld_2/2026-09-25T00-52_f5279f6/run_0/` (gitignored).
- **Started**: `2026-09-22 PDT`.
- **Last updated**: `2026-09-25T00:58:52-07:00`.
- **Notes**: OSWorld-2.1, 98 selected tasks; service exclusions and task 072 filtered out.
- **Provenance**: [JSON snapshot](/devs/exps/eval/osworld_2/logs/2026-09-25T00-52_f5279f6/run_0.json) records source runs, task replacements, configuration and artifact hashes. This is a consolidation of retained runs across commits; code drifted mid-campaign, as recorded in the source metadata.

## Results

| Model | Config | Finished | Mean episode return |
|---|---|---:|---:|
| `gpt-6-astra` | medium | 98/98 | 0.630220 |
| `Qwen/Qwen3.5-27B` | default | 98/98 | 0.027505 |
| `Qwen/Qwen3.8-27B` | default | 98/98 | 0.023604 |
| `Qwen/Qwen3-VL-32B-Instruct` | default | 98/98 | 0.011586 |
| `Qwen/Qwen3-VL-4B-Instruct` | default | 98/98 | 0.008630 |
| `Qwen/Qwen3.5-4B` | default | 98/98 | 0.007369 |
| `inclusionAI/UI-Venus-2-9B` | default | 98/98 | 0.004974 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | default | 98/98 | 0.004884 |
| `Qwen/Qwen3.5-9B` | default | 98/98 | 0.004221 |
| `Qwen/Qwen3-VL-8B-Instruct` | default | 98/98 | 0.002883 |
| `meituan/EvoCUA-8B-20260105` | default | 98/98 | 0.002474 |
| ⚠️ `claude-opus-5` | default | _**74/98**_ | _**0.630739**_ |
| ⚠️ `gpt-5.6-sol` | default | _**95/98**_ | _**0.557468**_ |
| ⚠️ `gpt-5.5` | default | _**96/98**_ | _**0.339228**_ |
| ⚠️ `gemini-3.6-flash` | default | _**0/98**_ | _**—**_ |

Mean episode return excludes API/environment errors and includes valid zero rewards.
Finished counts valid samples; the expected task count remains 98.

## Experiment Specification

| Models | YAML | Key settings |
|---|---|---|
| Qwen3-VL-4B / 8B / 32B Instruct | [qwen3_vl/osworld_2.yaml](/scripts/configs/qwen3_vl/default/osworld_2.yaml) | 200 steps; loop detection 5; thinking off |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/osworld_2.yaml](/scripts/configs/qwen3_5/default/osworld_2.yaml) | 200 steps; loop detection 5; thinking off |
| Qwen3.8-27B | [qwen3_8/osworld_2.yaml](/scripts/configs/qwen3_8/default/osworld_2.yaml) | 200 steps; loop detection 5; thinking off |
| UI-TARS-1.5-7B | [ui_tars_15_v1/osworld_2.yaml](/scripts/configs/ui_tars_15_v1/default/osworld_2.yaml) | 30 steps; loop detection 5 |
| EvoCUA-8B | [evocua/osworld_2.yaml](/scripts/configs/evocua/default/osworld_2.yaml) | 30 steps; loop detection 5 |
| UI-Venus-2-9B | [ui_venus_2/osworld_2.yaml](/scripts/configs/ui_venus_2/default/osworld_2.yaml) | 30 steps; loop detection 5; OSWorld-2.1 guest password |
| GPT-5.5 / GPT-5.6 Sol | [gpt/osworld_2.yaml](/scripts/configs/gpt/default/osworld_2.yaml) | 200 steps; XHigh; 8192 output tokens |
| Claude Opus 5 | [claude/osworld_2.yaml](/scripts/configs/claude/default/osworld_2.yaml) | 200 steps; max effort; 64000 output tokens |
| GPT-6 Astra | [gpt/osworld_2.medium.yaml](/scripts/configs/gpt/default/osworld_2.medium.yaml) | 200 steps; Medium; 8192 output tokens |

Specialist runs retain their 30-step desktop budgets; the committed OSWorld-2.1 YAMLs encode those actual settings. Other unspecified sampling and history parameters inherit the model defaults. API models have loop detection disabled.

LLM-judge tasks use the configured Azure endpoint. Listed replacement tasks use their new full rollouts; obsolete judge failures never fall back into the score. Missing or failed API attempts remain excluded. Task-level source mappings are in the JSON.

### Reproduction

Install the pinned image and services using the [OSWorld-2.1 guide](/lite/gym/envs/osworld_2/README.md), then export the env-server URL/token and model/judge API credentials. Run from the repository root:

```bash
./devs/exps/eval/osworld_2/run.sh Qwen/Qwen3-VL-8B-Instruct
./devs/exps/eval/osworld_2/run.sh gpt-5.5
./devs/exps/eval/osworld_2/run.sh gpt-6-astra scripts/configs/gpt/default/osworld_2.medium.yaml
```

Choose the model-specific YAML shown above. A fresh reproduction writes its own commit-keyed artifacts; the retained source runs remain available for audit.
