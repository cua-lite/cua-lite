# PhoneWorld

CUA-Lite integration for the 34-app [PhoneWorld](https://github.com/PhoneBuddyAI/PhoneWorld) benchmark: 120 evaluation tasks and 300 training tasks. The checked-in task manifest is normalized from upstream commit `7d3868adf4163aebb9a3f51babc124a81994d729`; upstream task-level `max_steps` is intentionally ignored in favor of the env-wide default of 60.

## Install

Prerequisites: Docker, writable `/dev/kvm`, and access to the gated [`EthanLeoLYX/PhoneWorld-APKs`](https://huggingface.co/datasets/EthanLeoLYX/PhoneWorld-APKs) dataset. PhoneWorld permits the APKs only for non-commercial academic research. `HF_TOKEN` from your shell or a prior Hugging Face login is used only for the host download and is never copied into a Docker build or image.

```bash
# Build the shared emulator base first.
uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh

# Builds cua-lite/phoneworld:latest locally.
uv run --no-sync bash lite/gym/envs/phoneworld/scripts/install.sh
```

The APK artifact is pinned to HF revision `63b9e86aa6c87ff88dc85eeeddbe62dbeba35ad3` and SHA-256 `aa2516efd0d3907442f56933b69b59f6b1bdcbe1ef9d2c5cc12420d285c9975b`. The installer requires exactly 34 unique PhoneWorld APKs.

The final image contains gated APKs and is therefore **local-only**. `push` and `pull` are deliberately rejected; do not export or redistribute this image or its build cache.

Chinese input uses [senzhk/ADBKeyBoard](https://github.com/senzhk/ADBKeyBoard), GPL-2.0, pinned at `v2.4-dev` / `98dc653e8c5e98b3bf9a69561b723545bc5fd5da`. The APK is built from source. Its corresponding source and license are retained in the image under `/usr/share/src/ADBKeyBoard` and `/usr/share/licenses/ADBKeyBoard`. The build fails unless a focused EditText can receive and expose `中文测试` through `ADB_INPUT_B64`.

## Run

```bash
uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id phoneworld --splits eval \
  --config-path scripts/configs/gpt/default/phoneworld.yaml

uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --env-id phoneworld --head 1 \
  --config-path scripts/configs/qwen3_vl/default/phoneworld.yaml
```

Every episode owns a fresh container. `max_resets_per_container: 0` destroys and recreates the container on reset, restoring the baked AVD snapshot; no upstream `APP_RESET_TABLES`, task `seed_data`, or per-table `DELETE` logic is used.

Verification supports answer keyword checks, SQLite checks, multi-table conjunctions, and the one upstream mixed task. Evaluator infrastructure errors propagate instead of being converted to a failed reward.

## Known upstream issues observed during integration

- The gated archive contains 34 PhoneWorld apps but not the README-promised ADBKeyBoard dependency; this integration supplies it from pinned GPL source.
- Upstream task JSON uses three SQLite path spellings and several task-ID spellings. The checked-in manifest normalizes these once at ingestion.
- Text/keyword verifiers are intentionally retained as published; CUA-Lite does not strengthen benchmark task semantics.
- Some APKs merge immutable seed state into their visible mutable state while SQLite verifiers inspect only runtime tables. For example, `cross_v3_005` shows `活着` as already on the WeRead shelf from `shelf_book_ids`, and `cross_v3_006` can show a QQ Music song as already liked from `liked_song_ids`; completing the visible goal without toggling the item off and on leaves the verifier table empty. This upstream task/environment mismatch is recorded, not patched here.

## Citation and license

Use of the gated APKs is subject to the upstream [PhoneWorld Research License](https://github.com/PhoneBuddyAI/PhoneWorld/blob/main/LICENSE), including its non-commercial academic-use, attribution, and no-redistribution terms. Cite the [PhoneWorld paper](https://arxiv.org/abs/2605.29486) alongside [CUA-Lite](/README.md#citation) when reporting benchmark results.
