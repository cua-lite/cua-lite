# osworld_g @ 2026-09-17T14-28_b1d4b6b - run_0

- **Commit**: `b1d4b6b` - reference models.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/osworld_g/2026-09-17T14-28_b1d4b6b/run_0/` (gitignored).
- **Started**: `2026-09-12 23:54 PDT`.
- **Last updated**: `2026-09-17 14:34 PDT`.
- **Notes**: `eval`, 510 tasks after filtering the 54 `exclude_reason` tasks. Task concurrency 1-30, at most 60 across the host.

## Results

| Model | Finished | Mean episode return | Score | Avg steps | Input tokens | Output tokens | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Sol | 510/510 | 0.798039 | 79.8039% | 1.00 | N/A | N/A | 0 |
| Qwen3.8-27B | 510/510 | 0.796078 | 79.6078% | 1.00 | 1,141,189 | 24,468 | 0 |
| GPT-5.5 | 510/510 | 0.768627 | 76.8627% | 1.00 | N/A | N/A | 0 |
| Qwen3.5-27B | 510/510 | 0.743137 | 74.3137% | 1.00 | 1,141,189 | 24,389 | 0 |
| UI-Venus-2-9B | 510/510 | 0.743137 | 74.3137% | 1.00 | 915,649 | 200,002 | 0 |
| Qwen3-VL-32B | 510/510 | 0.719608 | 71.9608% | 1.00 | 1,063,159 | 17,242 | 0 |
| MAI-UI-8B | 510/510 | 0.700000 | 70.0000% | 1.00 | 954,019 | 38,210 | 0 |
| Qwen3.5-9B | 510/510 | 0.682353 | 68.2353% | 1.00 | 1,141,189 | 24,385 | 0 |
| Qwen3-VL-4B | 510/510 | 0.654902 | 65.4902% | 1.00 | 1,063,159 | 17,237 | 0 |
| Qwen3-VL-8B | 510/510 | 0.643137 | 64.3137% | 1.00 | 1,063,159 | 17,238 | 0 |
| Qwen3.5-4B | 510/510 | 0.629412 | 62.9412% | 1.00 | 1,141,189 | 24,435 | 0 |
| UI-TARS-1.5-7B | 510/510 | 0.605882 | 60.5882% | 1.00 | 1,205,274 | 8,304 | 0 |
| EvoCUA-8B | 510/510 | 0.572549 | 57.2549% | 1.00 | 1,064,179 | 58,425 | 0 |

Score is the sum of rewards divided by 510; invalid tasks count as zero.
MER is `summary.json: stats.mean_episode_return`, over valid trajectories.
Avg steps covers valid predictions. API tokens sum recorded provider usage
across logged attempts; unlogged calls are excluded. Local tokens use the native
tokenizer and image geometry for retained trajectories, excluding stripped stop
tokens and failed attempts. `N/A` means provider telemetry is unavailable.

## Highlights

- GPT-5.6 Sol (default) scores 79.80%; Qwen3.8-27B leads local models at 79.61%.

## Experiment Specification

| Model | YAML | Sampling |
|---|---|---|
| Qwen3-VL-4B / 8B / 32B (Instruct) | [qwen3_vl/osworld_g.yaml](/scripts/configs/qwen3_vl/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/osworld_g.yaml](/scripts/configs/qwen3_5/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| Qwen3.8-27B | [qwen3_8/osworld_g.yaml](/scripts/configs/qwen3_8/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| UI-TARS-1.5-7B | [ui_tars_15_v1/osworld_g.yaml](/scripts/configs/ui_tars_15_v1/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| EvoCUA-8B | [evocua/osworld_g.yaml](/scripts/configs/evocua/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| MAI-UI-8B | [mai_ui/osworld_g.yaml](/scripts/configs/mai_ui/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| UI-Venus-2-9B | [ui_venus_2/osworld_g.yaml](/scripts/configs/ui_venus_2/default/osworld_g.yaml) | Greedy; 2048 output tokens |
| GPT-5.5 / GPT-5.6 Sol | [gpt/osworld_g.yaml](/scripts/configs/gpt/default/osworld_g.yaml) | Reasoning none; 4096 output tokens |

Each task supplies one screenshot and takes one click prediction using the
model's native grounding template and image preprocessing. Multi-turn history
windows do not apply. Qwen thinking remains disabled; UI-Venus uses its native
thinking template. No agent-level resolution override is set. Greedy decoding
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

Use the model ID and YAML from the specification for the other rows. Keep the
same run ID and config selection to resume.

## Breakdown

Values are mean rewards within each category. Paper categories overlap; their Avg weights category memberships, not unique tasks.

### By `box_type`

| Model | `bbox` | `polygon` | Avg |
|---|---:|---:|---:|
| `GPT-5.6 Sol` | 0.7851 | 0.9500 | 0.7980 |
| `Qwen3.8-27B` | 0.7872 | 0.9000 | 0.7961 |
| `GPT-5.5` | 0.7574 | 0.9000 | 0.7686 |
| `Qwen3.5-27B` | 0.7277 | 0.9250 | 0.7431 |
| `UI-Venus-2-9B` | 0.7277 | 0.9250 | 0.7431 |
| `Qwen3-VL-32B` | 0.7106 | 0.8250 | 0.7196 |
| `MAI-UI-8B` | 0.6851 | 0.8750 | 0.7000 |
| `Qwen3.5-9B` | 0.6745 | 0.7750 | 0.6824 |
| `Qwen3-VL-4B` | 0.6447 | 0.7750 | 0.6549 |
| `Qwen3-VL-8B` | 0.6255 | 0.8500 | 0.6431 |
| `Qwen3.5-4B` | 0.6149 | 0.8000 | 0.6294 |
| `UI-TARS-1.5-7B` | 0.5851 | 0.8500 | 0.6059 |
| `EvoCUA-8B` | 0.5511 | 0.8250 | 0.5725 |

### By `paper_category`

| Model | `element_recognition` | `fine_grained_manipulation` | `layout_understanding` | `text_matching` | Avg |
|---|---:|---:|---:|---:|---:|
| `Qwen3.8-27B` | 0.8333 | 0.6894 | 0.8201 | 0.8542 | 0.8146 |
| `GPT-5.5` | 0.7909 | 0.6818 | 0.7866 | 0.8250 | 0.7830 |
| `GPT-5.6 Sol` | 0.8170 | 0.7273 | 0.8201 | 0.8500 | 0.8135 |
| `UI-Venus-2-9B` | 0.8007 | 0.6439 | 0.7908 | 0.7875 | 0.7721 |
| `Qwen3.5-27B` | 0.7876 | 0.6439 | 0.7699 | 0.8125 | 0.7688 |
| `Qwen3-VL-32B` | 0.7516 | 0.6136 | 0.7615 | 0.8292 | 0.7546 |
| `MAI-UI-8B` | 0.7353 | 0.5833 | 0.7490 | 0.8167 | 0.7383 |
| `Qwen3.5-9B` | 0.7255 | 0.5530 | 0.7197 | 0.7958 | 0.7176 |
| `Qwen3-VL-4B` | 0.6797 | 0.5455 | 0.6569 | 0.7958 | 0.6848 |
| `Qwen3-VL-8B` | 0.6536 | 0.5758 | 0.6653 | 0.7917 | 0.6816 |
| `Qwen3.5-4B` | 0.6601 | 0.5152 | 0.6611 | 0.7583 | 0.6652 |
| `UI-TARS-1.5-7B` | 0.6078 | 0.5530 | 0.6151 | 0.7667 | 0.6434 |
| `EvoCUA-8B` | 0.5588 | 0.5227 | 0.5774 | 0.7667 | 0.6129 |
