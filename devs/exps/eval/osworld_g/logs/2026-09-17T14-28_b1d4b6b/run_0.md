# osworld_g @ 2026-09-17T14-28_b1d4b6b - run_0

- **Commit**: `b1d4b6b` - reference models.
- **Host / GPUs**: `gpublaze` / NVIDIA H100 80 GB.
- **Artifacts**: `.exps/eval/osworld_g/2026-09-17T14-28_b1d4b6b/run_0/` (gitignored).
- **Started**: `2026-09-12 23:54 PDT`.
- **Last updated**: `2026-09-25 00:43 PDT`.
- **Notes**: `eval`, 510 tasks after filtering the 54 `exclude_reason` tasks. Task concurrency 1-30, at most 60 across the host.
- **Provenance**: [JSON snapshot](/devs/exps/eval/osworld_g/logs/2026-09-17T14-28_b1d4b6b/run_0.json) records each model's source run, configuration, commits, and artifact checksums. The directory key identifies this report; source runs span multiple commits. Claude source runs include recorded uncommitted patches.

## Results

| Model | Config | Finished | Mean episode return |
|---|---|---:|---:|
| `gpt-6-astra` | medium | 510/510 | 0.925490 |
| `claude-opus-5` | default | 510/510 | 0.903922 |
| `gpt-5.6-sol` | default | 510/510 | 0.798039 |
| `Qwen/Qwen3.8-27B` | default | 510/510 | 0.796078 |
| `gpt-5.5` | default | 510/510 | 0.768627 |
| `Qwen/Qwen3.5-27B` | default | 510/510 | 0.743137 |
| `inclusionAI/UI-Venus-2-9B` | default | 510/510 | 0.743137 |
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

- Gemini 3.6 Flash: not started.

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
| GPT-6 Astra (Medium) | [gpt/osworld_g.medium.yaml](/scripts/configs/gpt/default/osworld_g.medium.yaml) | Reasoning medium; 4096 output tokens |
| Claude Opus 5 | [claude/osworld_g.yaml](/scripts/configs/claude/default/osworld_g.yaml) | Effort low; 1024 output tokens; no explicit thinking budget |

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

Use the model ID and YAML from the specification for the other rows. Claude
and Astra use the rollout entry directly:

```bash
uv run python scripts/rollout.py --model-id claude-opus-5 \
  --config-path scripts/configs/claude/default/osworld_g.yaml \
  --env-id osworld_g --splits eval --concurrency 6 \
  --log-root '.exps/eval/osworld_g/<commit-dir>/run_0/claude-opus-5'
uv run python scripts/rollout.py --model-id gpt-6-astra \
  --config-path scripts/configs/gpt/default/osworld_g.medium.yaml \
  --env-id osworld_g --splits eval --concurrency 6 \
  --log-root '.exps/eval/osworld_g/<commit-dir>/run_0/gpt-6-astra__medium'
```

Apply `--filter 'lambda m: not m.others.get("exclude_reason")'` to both commands.

Replace `<commit-dir>` with the timestamp and SHA of the code being evaluated.
The JSON source paths identify the retained results. Reusing a source log root
resumes that run; a new log root starts a separate run.

## Breakdown

Values cover valid predictions only. Paper categories overlap; their Avg weights category memberships.

### Breakdown — by `box_type`

| Model | `bbox` | `polygon` | Avg |
|---|---:|---:|---:|
| `gpt-6-astra` | 0.9213 | 0.9750 | 0.9255 |
| `claude-opus-5` | 0.9000 | 0.9500 | 0.9039 |
| `gpt-5.6-sol` | 0.7851 | 0.9500 | 0.7980 |
| `Qwen/Qwen3.8-27B` | 0.7872 | 0.9000 | 0.7961 |
| `gpt-5.5` | 0.7574 | 0.9000 | 0.7686 |
| `Qwen/Qwen3.5-27B` | 0.7277 | 0.9250 | 0.7431 |
| `inclusionAI/UI-Venus-2-9B` | 0.7277 | 0.9250 | 0.7431 |
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
| `claude-opus-5` | 0.9281 | 0.8712 | 0.9247 | 0.9167 | 0.9160 |
| `Qwen/Qwen3.8-27B` | 0.8333 | 0.6894 | 0.8201 | 0.8542 | 0.8146 |
| `gpt-5.6-sol` | 0.8170 | 0.7273 | 0.8201 | 0.8500 | 0.8135 |
| `gpt-5.5` | 0.7909 | 0.6818 | 0.7866 | 0.8250 | 0.7830 |
| `inclusionAI/UI-Venus-2-9B` | 0.8007 | 0.6439 | 0.7908 | 0.7875 | 0.7721 |
| `Qwen/Qwen3.5-27B` | 0.7876 | 0.6439 | 0.7699 | 0.8125 | 0.7688 |
| `Qwen/Qwen3-VL-32B-Instruct` | 0.7516 | 0.6136 | 0.7615 | 0.8292 | 0.7546 |
| `Tongyi-MAI/MAI-UI-8B` | 0.7353 | 0.5833 | 0.7490 | 0.8167 | 0.7383 |
| `Qwen/Qwen3.5-9B` | 0.7255 | 0.5530 | 0.7197 | 0.7958 | 0.7176 |
| `Qwen/Qwen3-VL-4B-Instruct` | 0.6797 | 0.5455 | 0.6569 | 0.7958 | 0.6848 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.6536 | 0.5758 | 0.6653 | 0.7917 | 0.6816 |
| `Qwen/Qwen3.5-4B` | 0.6601 | 0.5152 | 0.6611 | 0.7583 | 0.6652 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 0.6078 | 0.5530 | 0.6151 | 0.7667 | 0.6434 |
| `meituan/EvoCUA-8B-20260105` | 0.5588 | 0.5227 | 0.5774 | 0.7667 | 0.6129 |
