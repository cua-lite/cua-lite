# WebGym Anchor Tasks For Browser GRPO

This is the browser-local runbook for building the GRPO train task manifest used
to transfer from online WebGym reward to the fixed offline WebVoyager eval in
[`/devs/exps/train/browser/README.md`](/devs/exps/train/browser/README.md).

Default policy: focus on the **anchor pool**. Candidate tasks come from
successful `gpt5_5` WebGym SFT demonstrations after the browser no-goto
projection:

```text
successful gpt5_5 WebGym demos
+ no top-level goto in the teacher trajectory
+ unique WebGym task id
+ non-search start website
```

Do **not** use `difficulty <= 3` as a hard filter here. That filter was only a
rough proxy for "likely no-goto"; the SFT source already proves the stronger
condition by containing a successful no-goto teacher trajectory. Difficulty is
reported in summaries for curriculum analysis, not used to select tasks.

The flow has two separate stages:

1. Build a deterministic candidate parquet from clean no-goto demonstrations.
2. Optionally run grouped rollout calibration, then write a final GRPO parquet
   from tasks that are both infra-healthy and useful for GRPO signal.

## Ownership

- SFT trajectory filtering is owned by
  [`/devs/exps/train/browser/utils/filter.py`](/devs/exps/train/browser/utils/filter.py).
- RL task selection is owned by
  [`/devs/exps/train/browser/utils/tasks.py`](/devs/exps/train/browser/utils/tasks.py).
- Do not put WebGym/WebVoyager experiment policy into
  `lite.train.export.export_tasks` or the generic WebGym data cleaner.

`export_tasks --filter` cannot express the anchor policy because WebGym registry
metadata has `website/domain/subdomain/difficulty`, but not the teacher success
evidence. The builder reads the clean SFT source parquet for task ids, then
writes the same two-column task parquet schema as `export_tasks`.

## Outputs

Candidate manifest from static no-goto teacher evidence:

```text
/devs/exps/train/browser/data/webgym.train.gpt55_anchor.nogoto.site.candidates.seed42.parquet
```

Candidate sidecar:

```text
/devs/exps/train/browser/data/webgym.train.gpt55_anchor.nogoto.site.candidates.seed42.summary.json
```

Optional calibrated GRPO train manifest:

```text
/devs/exps/train/browser/data/webgym.train.gpt55_anchor.nogoto.site.calibrated.head1024.g4.seed42.parquet
```

Calibration sidecars:

```text
/devs/exps/train/browser/data/webgym.train.gpt55_anchor.nogoto.site.calibrated.head1024.g4.seed42.summary.json
/devs/exps/train/browser/data/webgym.train.gpt55_anchor.nogoto.site.calibrated.head1024.g4.seed42.calibration.parquet
```

Use the existing fixed eval manifest:

```text
/devs/exps/train/browser/data/webvoyager.eval128.readonly.seed42.parquet
```

## Step 0: Check Source Data

The anchor pool is derived from the filtered no-goto SFT source parquet. If this
file is missing, run the SFT export/filter block in
[`/devs/exps/train/browser/README.md`](/devs/exps/train/browser/README.md)
first.

```bash
CLEAN=.data/huggingface-webgym-nogoto
T=gpt5_5
SFT_SRC="$CLEAN/$T/cua-lite/WebGym/browser/use/train/browser.use.$T.parquet"

[ -e "$SFT_SRC" ] || {
  echo "missing no-goto SFT source: $SFT_SRC" >&2
  echo "run the browser README SFT Export block first" >&2
  exit 1
}
```

Expected current source:

| teacher | rows | note |
|---|---:|---|
| `gpt5_5` | 2389 | after dropping 754 top-level `goto` rows |

## Step 1: Build Anchor Candidates

This command is deterministic. If the output already exists, keep it unless you
are intentionally refreshing the experiment manifest.

```bash
DATA=devs/exps/train/browser/data
CAND="$DATA/webgym.train.gpt55_anchor.nogoto.site.candidates.seed42.parquet"

if [ -e "$CAND" ]; then
  echo "keep existing fixed candidate manifest: $CAND"
else
  uv run python devs/exps/train/browser/utils/tasks.py build-candidates --out "$CAND"
fi

uv run python devs/exps/train/browser/utils/tasks.py verify-candidates --candidate "$CAND"
```

Expected current candidate output:

| property | value |
|---|---:|
| rows | 2366 |
| popular / nonpopular eligible | 666 / 1700 |
| exact WebVoyager instruction matches | 8, reported only |
| difficulty 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8+ | 431 / 447 / 246 / 361 / 264 / 257 / 349 / 11 |
| top sources | `insta-v3` 1234, `pae-webvoyager` 1051 |
| max host share | `amazon.com` 434 / 2366 |
| `env_key_sha256` | `c50569db5c0c17dfe38499c4de79ebcb44986f42968998e21bdad8c74e0bdf35` |

The required integrity checks are:

- 2366 unique `webgym@...` train rows.
- No Google/Bing/DuckDuckGo start hosts.
- The `env_key_sha256` matches the expected value above.
- Difficulty and exact WebVoyager matches are reported, not filtered.

## Step 2: Rollout Calibration

Calibration is the only stage that should look at `.logs`. It has two jobs:

- **Infra health:** drop tasks with repeated block, timeout, reset, blank-screen,
  or missing-summary failures.
- **GRPO signal:** prefer tasks where grouped samples have mixed returns. A task
  with all successes or all failures has weak immediate GRPO advantage signal.

Start with a bounded calibration pass. Use `--head 1024` for the default
experiment; remove it only if you want to spend time calibrating all 2366
candidates.

```bash
# --- ENV-SERVER HOST ---
: "${OPENAI_API_KEY:?export OPENAI_API_KEY before starting WebGym env-server}"

PORT=30107
HOST_IP=$(hostname -I | awk '{print $1}')
SESSION_ID=browser-anchor-webgym-$(date +%Y%m%d_%H%M%S)

printf 'export CUA_LITE_ENV_SERVER_URL=http://%s:%s\n' "$HOST_IP" "$PORT"
printf 'export CUA_LITE_ENV_SERVER_TOKEN=%s\n' "$SESSION_ID"

WEBGYM_INSTANCES=16 uv run --no-sync python scripts/serve_env.py \
  --port "$PORT" --env-ids webgym --token "$SESSION_ID"
```

```bash
# --- ROLLOUT HOST ---
: "${CUA_LITE_ENV_SERVER_URL:?paste export line from env-server shell}"
: "${CUA_LITE_ENV_SERVER_TOKEN:?paste export line from env-server shell}"
: "${CKPT:?set CKPT to the browser.use.i1.reasoning gpt5_5 SFT checkpoint}"

DATA=devs/exps/train/browser/data
CAND="$DATA/webgym.train.gpt55_anchor.nogoto.site.candidates.seed42.parquet"
CFG=devs/exps/train/browser/configs/qwen3_5
P=browser.use.i1.reasoning
GROUP_SIZE=4
CALIBRATE_HEAD=1024
LOG=.logs/rollout/Qwen_Qwen3.5-4B/webgym/calibrate.$P.gpt55_anchor_nogoto.head${CALIBRATE_HEAD}.g${GROUP_SIZE}.seed42

# Start this separately if no matching SGLang server is already running.
# CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/serve_sglang.py \
#   --model-id Qwen/Qwen3.5-4B --model-path "$CKPT" \
#   --engine-kwargs '{"tp_size": 4}' --port 30082

uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --model-path "$CKPT" \
  --sglang-server-url http://127.0.0.1:30082 \
  --env-id webgym \
  --prompt-data "$CAND" \
  --head "$CALIBRATE_HEAD" \
  --group-size "$GROUP_SIZE" \
  --concurrency 16 \
  --max-attempts 1 \
  --save-video false \
  --save-gif false \
  --config-path "$CFG/$P.yaml" \
  --log-root "$LOG" < /dev/null
```

After rollout finishes, aggregate sample summaries into a calibrated final
manifest:

```bash
DATA=devs/exps/train/browser/data
CAND="$DATA/webgym.train.gpt55_anchor.nogoto.site.candidates.seed42.parquet"
OUT="$DATA/webgym.train.gpt55_anchor.nogoto.site.calibrated.head1024.g4.seed42.parquet"
LOG=.logs/rollout/Qwen_Qwen3.5-4B/webgym/calibrate.browser.use.i1.reasoning.gpt55_anchor_nogoto.head1024.g4.seed42

uv run python devs/exps/train/browser/utils/tasks.py calibrate \
  --candidate "$CAND" \
  --log-root "$LOG" \
  --out "$OUT" \
  --head 1024 \
  --group-size 4 \
  --target 1024 \
  --min-final 500
```

Selection policy:

- Drop `incomplete` and `infra_bad` tasks.
- Prefer `mixed_success`, because grouped rollouts have direct GRPO advantage
  signal.
- Backfill with `all_fail_with_reward_variance`, where dense/shaped returns
  still differ inside the group.
- Keep at most 128 `all_success` tasks as behavior-retention anchors.
- Cap any single host at 128 tasks.
- Require at least 500 selected tasks; otherwise inspect health before training.

## Smoke Test

Use this to validate the file format and Step 2 aggregator without spending on a
full browser rollout. It builds the real candidate parquet, writes a small
synthetic calibration log for the first eight tasks, aggregates a smoke final
parquet, and checks that the rollout prompt-data resolver can load it.

```bash
DATA=devs/exps/train/browser/data
CAND="$DATA/webgym.train.gpt55_anchor.nogoto.site.candidates.seed42.parquet"
SMOKE_LOG=.logs/rollout/Qwen_Qwen3.5-4B/webgym/smoke.calibrate.browser.use.i1.reasoning.gpt55_anchor_nogoto.head8.g4.seed42
SMOKE_OUT=/tmp/webgym.train.gpt55_anchor.nogoto.site.calibrated.smoke.parquet

rm -rf "$SMOKE_LOG" "$SMOKE_OUT" "${SMOKE_OUT%.parquet}.summary.json" "${SMOKE_OUT%.parquet}.calibration.parquet"

uv run python devs/exps/train/browser/utils/tasks.py build-candidates --out "$CAND"
uv run python devs/exps/train/browser/utils/tasks.py verify-candidates --candidate "$CAND"
uv run python devs/exps/train/browser/utils/tasks.py write-smoke-log \
  --candidate "$CAND" --log-root "$SMOKE_LOG" --head 8 --group-size 4
uv run python devs/exps/train/browser/utils/tasks.py calibrate \
  --candidate "$CAND" \
  --log-root "$SMOKE_LOG" \
  --out "$SMOKE_OUT" \
  --head 8 \
  --group-size 4 \
  --target 8 \
  --min-final 1 \
  --max-per-host 8 \
  --all-success-cap 8
uv run python - "$SMOKE_OUT" <<'PY'
import sys

from lite.infer.rollout import resolve_prompt_data_tasks

specs = resolve_prompt_data_tasks(sys.argv[1], effective_env_id="webgym")
assert specs
assert all(spec.env_id == "webgym" for spec in specs)
assert all(spec.split == "train" for spec in specs)
print(f"rollout prompt-data resolver ok: {len(specs)} tasks")
PY
```

The smoke should produce mixed-success and infra-bad buckets in the calibration
sidecar, plus at least one final parquet row. It does not prove WebVoyager will
improve; that claim requires the full calibration, GRPO training from the
`gpt5_5` + `<think>` SFT checkpoint, and the fixed WebVoyager eval below.

## GRPO Usage

Use the calibrated manifest when Step 2 has been run. If calibration has not
been run, use the candidate manifest as the fallback and treat the first GRPO
evals as the health signal.

```bash
PROMPT_DATA=/workspaces/cua-lite/devs/exps/train/browser/data/webgym.train.gpt55_anchor.nogoto.site.calibrated.head1024.g4.seed42.parquet
EVAL_PROMPT_DATA=/workspaces/cua-lite/devs/exps/train/browser/data/webvoyager.eval128.readonly.seed42.parquet
CONFIG_PATH=/workspaces/cua-lite/devs/exps/train/browser/configs/qwen3_5/browser.use.i1.reasoning.yaml
```

The run still starts from the `gpt5_5` + `<think>` SFT checkpoint. RL is useful
here because the policy is optimized on its own sampled trajectories and
terminal WebGym reward, not because the tasks are unseen. The anchor design asks
for a controlled improvement over SFT on tasks with known no-goto solutions.

The env-server for GRPO must serve both train and eval envs:

```bash
uv run --no-sync python scripts/serve_env.py \
  --env-ids webgym webharbor.webvoyager --token "$SESSION_ID"
```

The final promotion gate remains the fixed WebVoyager eval128 result. A GRPO
checkpoint should not replace the SFT parent unless it improves `return_mean`
without a material increase in parse failures or env errors.
