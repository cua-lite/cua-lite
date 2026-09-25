# osworld_g @ 2026-09-25T15-23_1996121 - run_0

- **Commit**: `1996121` - Official non-thinking UI-Venus grounding defaults.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/osworld_g/2026-09-25T15-23_1996121/run_0/` (gitignored).
- **Started**: `2026-09-25 PDT`.
- **Last updated**: `2026-09-25 15:33 PDT`.
- **Notes**: `eval`, 510 tasks; 12 concurrent Venus grounding tasks.
- **Provenance**: [JSON snapshot](/devs/exps/eval/osworld_g/logs/2026-09-25T15-23_1996121/run_0.json) records per-model source paths, configurations and hashes. Fresh non-thinking UI-Venus run; other model results retain their recorded sources.

## Results

| Model | Config | Finished | Mean episode return |
|---|---|---:|---:|
| `gpt-6-astra` | medium | 510/510 | 0.925490 |
| `gpt-5.6-sol` | default | 510/510 | 0.911765 |
| `gpt-5.5` | default | 510/510 | 0.903922 |
| `claude-opus-5` | default | 510/510 | 0.892157 |
| `Qwen/Qwen3.8-27B` | default | 510/510 | 0.796078 |
| `inclusionAI/UI-Venus-2-9B` | default | 510/510 | 0.782353 |
| `Qwen/Qwen3.5-27B` | default | 510/510 | 0.743137 |
| `Qwen/Qwen3-VL-32B-Instruct` | default | 510/510 | 0.719608 |
| `Tongyi-MAI/MAI-UI-8B` | default | 510/510 | 0.700000 |
| `Qwen/Qwen3.5-9B` | default | 510/510 | 0.682353 |
| `Qwen/Qwen3-VL-4B-Instruct` | default | 510/510 | 0.654902 |
| `Qwen/Qwen3-VL-8B-Instruct` | default | 510/510 | 0.643137 |
| `Qwen/Qwen3.5-4B` | default | 510/510 | 0.629412 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | default | 510/510 | 0.605882 |
| `meituan/EvoCUA-8B-20260105` | default | 510/510 | 0.572549 |
| ⚠️ `gemini-3.6-flash` | default | _**0/510**_ | _**—**_ |

Mean episode return excludes API errors and includes valid zero-reward predictions.
Finished counts valid samples; incomplete rows retain the full task count.

## Highlights

- gemini-3.6-flash: not started.

## Experiment Specification

| Model | YAML | Sampling |
|---|---|---|
| Qwen3-VL-4B / 8B / 32B (Instruct) | [qwen3_vl/osworld_g.yaml](/scripts/configs/qwen3_vl/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/osworld_g.yaml](/scripts/configs/qwen3_5/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| Qwen3.8-27B | [qwen3_8/osworld_g.yaml](/scripts/configs/qwen3_8/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| UI-TARS-1.5-7B | [ui_tars_15_v1/osworld_g.yaml](/scripts/configs/ui_tars_15_v1/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| EvoCUA-8B | [evocua/osworld_g.yaml](/scripts/configs/evocua/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| MAI-UI-8B | [mai_ui/osworld_g.yaml](/scripts/configs/mai_ui/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| UI-Venus-2-9B | [ui_venus_2/osworld_g.yaml](/scripts/configs/ui_venus_2/default/osworld_g.yaml) | Non-thinking; temperature 0; 1024 output tokens |
| GPT-5.5 / GPT-5.6 Sol | [gpt/osworld_g.yaml](/scripts/configs/gpt/default/osworld_g.yaml) | Reasoning medium; 4096 output tokens |
| GPT-6 Astra (Medium) | [gpt/osworld_g.medium.yaml](/scripts/configs/gpt/default/osworld_g.medium.yaml) | Reasoning medium; 4096 output tokens |
| Claude Opus 5 | [claude/osworld_g.yaml](/scripts/configs/claude/default/osworld_g.yaml) | Effort medium; 1024 output tokens; no explicit thinking budget |

Each task supplies one screenshot and takes one click prediction using the
model's native grounding template and image preprocessing. Multi-turn history
windows do not apply. Qwen and UI-Venus grounding use non-thinking prompts. No agent-level resolution override is set. Greedy decoding
does not use top-p.

### Reproduction

Prepare the dataset and a scoped env-server using the
[OSWorld-G guide](/lite/gym/envs/osworld_g/README.md).
Export `CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN`; configure
the API credentials for GPT runs and cache local model weights.

```bash
export EVAL_RUN_ID=run_0
export EVAL_CONCURRENCY=30
CUDA_VISIBLE_DEVICES=0 ./devs/exps/eval/osworld_g/run.sh Qwen/Qwen3-VL-8B-Instruct
./devs/exps/eval/osworld_g/run.sh gpt-5.6-sol
```

Use the model ID from the specification for other rows. The runner selects each family's default YAML; Astra's retained Medium variant is equivalent in reasoning settings.

## Breakdown

Values cover valid predictions only. Overlapping categories count each category membership.

### Breakdown — by `box_type`

| Model | `bbox` | `polygon` | Avg |
|---|---:|---:|---:|
| `gpt-6-astra` | 0.9213 | 0.9750 | 0.9255 |
| `gpt-5.6-sol` | 0.9064 | 0.9750 | 0.9118 |
| `gpt-5.5` | 0.9021 | 0.9250 | 0.9039 |
| `claude-opus-5` | 0.8851 | 0.9750 | 0.8922 |
| `Qwen/Qwen3.8-27B` | 0.7872 | 0.9000 | 0.7961 |
| `inclusionAI/UI-Venus-2-9B` | 0.7745 | 0.8750 | 0.7824 |
| `Qwen/Qwen3.5-27B` | 0.7277 | 0.9250 | 0.7431 |
| `Qwen/Qwen3-VL-32B-Instruct` | 0.7106 | 0.8250 | 0.7196 |
| `Tongyi-MAI/MAI-UI-8B` | 0.6851 | 0.8750 | 0.7000 |
| `Qwen/Qwen3.5-9B` | 0.6745 | 0.7750 | 0.6824 |
| `Qwen/Qwen3-VL-4B-Instruct` | 0.6447 | 0.7750 | 0.6549 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.6255 | 0.8500 | 0.6431 |
| `Qwen/Qwen3.5-4B` | 0.6149 | 0.8000 | 0.6294 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 0.5851 | 0.8500 | 0.6059 |
| `meituan/EvoCUA-8B-20260105` | 0.5511 | 0.8250 | 0.5725 |

### Breakdown — by `paper_category`

| Model | `element_recognition` | `fine_grained_manipulation` | `layout_understanding` | `text_matching` | Avg |
|---|---:|---:|---:|---:|---:|
| `gpt-6-astra` | 0.9477 | 0.8788 | 0.9372 | 0.9250 | 0.9291 |
| `gpt-5.6-sol` | 0.9281 | 0.8788 | 0.9205 | 0.9250 | 0.9182 |
| `gpt-5.5` | 0.9183 | 0.8636 | 0.9331 | 0.9167 | 0.9138 |
| `claude-opus-5` | 0.9314 | 0.8258 | 0.9163 | 0.9083 | 0.9062 |
| `inclusionAI/UI-Venus-2-9B` | 0.8333 | 0.6894 | 0.8410 | 0.8417 | 0.8168 |
| `Qwen/Qwen3.8-27B` | 0.8333 | 0.6894 | 0.8201 | 0.8542 | 0.8146 |
| `Qwen/Qwen3.5-27B` | 0.7876 | 0.6439 | 0.7699 | 0.8125 | 0.7688 |
| `Qwen/Qwen3-VL-32B-Instruct` | 0.7516 | 0.6136 | 0.7615 | 0.8292 | 0.7546 |
| `Tongyi-MAI/MAI-UI-8B` | 0.7353 | 0.5833 | 0.7490 | 0.8167 | 0.7383 |
| `Qwen/Qwen3.5-9B` | 0.7255 | 0.5530 | 0.7197 | 0.7958 | 0.7176 |
| `Qwen/Qwen3-VL-4B-Instruct` | 0.6797 | 0.5455 | 0.6569 | 0.7958 | 0.6848 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.6536 | 0.5758 | 0.6653 | 0.7917 | 0.6816 |
| `Qwen/Qwen3.5-4B` | 0.6601 | 0.5152 | 0.6611 | 0.7583 | 0.6652 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 0.6078 | 0.5530 | 0.6151 | 0.7667 | 0.6434 |
| `meituan/EvoCUA-8B-20260105` | 0.5588 | 0.5227 | 0.5774 | 0.7667 | 0.6129 |
