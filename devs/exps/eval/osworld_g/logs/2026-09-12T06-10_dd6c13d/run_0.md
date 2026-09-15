# osworld_g @ 2026-09-12T06-10_dd6c13d · run_0

- **Commit**: `dd6c13d` — default OSWorld-G evaluation.
- **Host / GPUs**: `gpublaze` / H100 80 GB.
- **Artifacts**: `.exps/eval/osworld_g/2026-09-12T06-10_dd6c13d/run_0/` (gitignored).
- **Started**: `2026-09-12 23:54 PDT`.
- **Last updated**: `2026-09-15 08:53 PDT`.
- **Notes**: `eval`, 510 scored tasks after excluding 54 refusal tasks; shared local model services, task concurrency 1–24 (60 total maximum), and step timeout 120 seconds or an explicit 180-second override.

## Results

| Model | Finished | Mean episode return | Score | Avg steps | Input tokens | Output tokens | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 510/510 | 0.796078 | 79.6078% | 1.00 | — | — | 0 |
| GPT-5.5 | 510/510 | 0.770588 | 77.0588% | 1.00 | 1,167,254 | 10,928 | 0 |
| Qwen3.5-27B | 510/510 | 0.743137 | 74.3137% | 1.00 | — | — | 0 |
| UI-Venus-2-9B | 510/510 | 0.743137 | 74.3137% | 1.00 | — | — | 0 |
| Qwen3-VL-32B | 510/510 | 0.719608 | 71.9608% | 1.00 | — | — | 0 |
| MAI-UI-8B | 510/510 | 0.700000 | 70.0000% | 1.00 | — | — | 0 |
| Qwen3.5-9B | 510/510 | 0.682353 | 68.2353% | 1.00 | — | — | 0 |
| Qwen3-VL-4B | 510/510 | 0.654902 | 65.4902% | 1.00 | — | — | 0 |
| Qwen3-VL-8B | 510/510 | 0.643137 | 64.3137% | 1.00 | — | — | 0 |
| Qwen3.5-4B | 510/510 | 0.629412 | 62.9412% | 1.00 | — | — | 0 |
| UI-TARS-1.5-7B | 510/510 | 0.605882 | 60.5882% | 1.00 | — | — | 0 |
| EvoCUA-8B | 510/510 | 0.572549 | 57.2549% | 1.00 | — | — | 0 |

Score is the sum of binary evaluator rewards divided by 510. Avg steps counts
model predictions. GPT token totals sum the provider usage logged for the 510 retained
attempts; local token usage was not recorded and is shown as `—`.

## Highlights

- Qwen3.8-27B leads at 79.61%; all 12 models completed all 510 tasks.

## Experiment specification

| Model | Default YAML |
|---|---|
| Qwen3-VL-4B / 8B / 32B | [qwen3_vl/osworld_g.yaml](/scripts/configs/qwen3_vl/default/osworld_g.yaml) |
| Qwen3.5-4B / 9B / 27B | [qwen3_5/osworld_g.yaml](/scripts/configs/qwen3_5/default/osworld_g.yaml) |
| Qwen3.8-27B | [qwen3_8/osworld_g.yaml](/scripts/configs/qwen3_8/default/osworld_g.yaml) |
| UI-TARS-1.5-7B | [ui_tars_15_v1/osworld_g.yaml](/scripts/configs/ui_tars_15_v1/default/osworld_g.yaml) |
| EvoCUA-8B | [evocua/osworld_g.yaml](/scripts/configs/evocua/default/osworld_g.yaml) |
| MAI-UI-8B | [mai_ui/osworld_g.yaml](/scripts/configs/mai_ui/default/osworld_g.yaml) |
| UI-Venus-2-9B | [ui_venus_2/osworld_g.yaml](/scripts/configs/ui_venus_2/default/osworld_g.yaml) |
| GPT-5.5 | [gpt/osworld_g.yaml](/scripts/configs/gpt/default/osworld_g.yaml) |

Each task uses one screenshot and one click prediction, with original task
instructions and model-native image preprocessing. Local models use greedy
decoding and a 2048-token output limit. GPT uses the grounding defaults:
reasoning effort `none` and a 4096-token output limit. Other settings inherit
the linked defaults; multi-turn history windows do not apply.

### Reproduction

Prepare the dataset using the [OSWorld-G guide](/lite/gym/envs/osworld_g/README.md).
With a local model service available, run from the OSWorld-G worktree:

```bash
uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-27B \
  --model-path /path/to/Qwen3.5-27B \
  --sglang-server-url http://127.0.0.1:31522 \
  --env-id osworld_g --splits eval \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/qwen3_5/default/osworld_g.yaml \
  --concurrency 20 --max-attempts 1 \
  --log-root .exps/eval/osworld_g/2026-09-12T06-10_dd6c13d/run_0/Qwen_Qwen3.5-27B
```

Use the corresponding model ID and YAML for other rows; GPT uses its API
configuration and omits the local model path and SGLang URL. Keep the same log
root to resume.

## Breakdown

`paper_category` labels overlap, so its weighted Avg differs from overall MER.
The wide `GUI_types` axis is retained in each model's `breakdown.json`.

### Breakdown — by `paper_category`

| Model | `element_recognition` | `fine_grained_manipulation` | `layout_understanding` | `text_matching` | Avg |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 0.8333 | 0.6894 | 0.8201 | 0.8542 | 0.8146 |
| GPT-5.5 | 0.7843 | 0.7045 | 0.7950 | 0.8250 | 0.7863 |
| UI-Venus-2-9B | 0.8007 | 0.6439 | 0.7908 | 0.7875 | 0.7721 |
| Qwen3.5-27B | 0.7876 | 0.6439 | 0.7699 | 0.8125 | 0.7688 |
| Qwen3-VL-32B | 0.7516 | 0.6136 | 0.7615 | 0.8292 | 0.7546 |
| MAI-UI-8B | 0.7353 | 0.5833 | 0.7490 | 0.8167 | 0.7383 |
| Qwen3.5-9B | 0.7255 | 0.5530 | 0.7197 | 0.7958 | 0.7176 |
| Qwen3-VL-4B | 0.6797 | 0.5455 | 0.6569 | 0.7958 | 0.6848 |
| Qwen3-VL-8B | 0.6536 | 0.5758 | 0.6653 | 0.7917 | 0.6816 |
| Qwen3.5-4B | 0.6601 | 0.5152 | 0.6611 | 0.7583 | 0.6652 |
| UI-TARS-1.5-7B | 0.6078 | 0.5530 | 0.6151 | 0.7667 | 0.6434 |
| EvoCUA-8B | 0.5588 | 0.5227 | 0.5774 | 0.7667 | 0.6129 |

### Breakdown — by `box_type`

| Model | `bbox` | `polygon` | Avg |
|---|---:|---:|---:|
| Qwen3.8-27B | 0.7872 | 0.9000 | 0.7961 |
| GPT-5.5 | 0.7617 | 0.8750 | 0.7706 |
| Qwen3.5-27B | 0.7277 | 0.9250 | 0.7431 |
| UI-Venus-2-9B | 0.7277 | 0.9250 | 0.7431 |
| Qwen3-VL-32B | 0.7106 | 0.8250 | 0.7196 |
| MAI-UI-8B | 0.6851 | 0.8750 | 0.7000 |
| Qwen3.5-9B | 0.6745 | 0.7750 | 0.6824 |
| Qwen3-VL-4B | 0.6447 | 0.7750 | 0.6549 |
| Qwen3-VL-8B | 0.6255 | 0.8500 | 0.6431 |
| Qwen3.5-4B | 0.6149 | 0.8000 | 0.6294 |
| UI-TARS-1.5-7B | 0.5851 | 0.8500 | 0.6059 |
| EvoCUA-8B | 0.5511 | 0.8250 | 0.5725 |

Regenerate from the retained artifacts:

```bash
uv run python devs/exps/eval/utils/reaggregate_breakdown.py \
  --axes paper_category,box_type,GUI_types \
  .exps/eval/osworld_g/2026-09-12T06-10_dd6c13d/run_0
uv run python devs/exps/eval/utils/render_breakdown.py \
  .exps/eval/osworld_g/2026-09-12T06-10_dd6c13d/run_0
```
