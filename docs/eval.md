# Evaluation

Run agents on environment tasks to collect trajectories and measure performance.

Evaluation runs on the host (not inside the Slime container) by default. For host uv setup, see [README.md#installation](/README.md#installation).
For per-environment setup, see [docs/envs.md#installation](/docs/envs.md#installation). Running evaluation through a managed env-server is recommended — see [docs/envs.md#env-server](/docs/envs.md#env-server).

> In the commands below, `{a,b,c}` lists a flag's available options — **pick one** (matching `--model-id` ↔ `--config-path`); it is not shell brace-expansion. Blocks show **representative agents** (`gpt-5.5` + `Qwen/Qwen3-VL-8B-Instruct`); the full per-env agent set is whatever exists under `scripts/configs/*/default/`.

## Contents

- **Grounding**
  - [OSWorld-G](#osworld-g)
  - [ScreenSpot-Pro](#screenspot-pro)
- **Desktop**
  - [OSWorld / Lite.OSWorld](#osworld--liteosworld)
  - [WindowsAgentArena](#windowsagentarena)
  - [OSWorld-2](#osworld-2)
  - [CUABench (basic, KiCad, workflows)](#cuabench-basic-kicad-workflows)
- **Browser**
  - [WebGym](#webgym)
  - [WebVoyager](#webvoyager)
  - [Online-Mind2Web](#online-mind2web)
  - [BrowserGym (MiniWoB, WebArena, VisualWebArena)](#browsergym-miniwob-webarena-visualwebarena)
- **Mobile**
  - [AndroidWorld](#androidworld)
  - [AndroidLab](#androidlab)
  - [MobileWorld](#mobileworld)
  - [MobileGym](#mobilegym)
- **General**
  - [Common Options](#common-options)

---

## OSWorld-G

Single-step click grounding (564 tasks: 470 bbox / 40 polygon / 54 refusal). `--filter` drops the 54 `refusal` tasks (tagged `exclude_reason` — the element is absent; scored only if you opt into the `report_infeasible` extra tool).

**Setup:** see [`lite/gym/envs/osworld_g/README.md`](/lite/gym/envs/osworld_g/README.md).

```bash
# OSWorld-G — single-step click grounding.
# --filter drops the refusal tasks (exclude_reason); the agent emits one click.
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id osworld_g \
  --splits eval \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/osworld_g.yaml
```

## ScreenSpot-Pro

Single-step click grounding on professional high-resolution screens (~1581 tasks, pure bounding-box — no refusal split, so no filter).

**Setup:** see [`lite/gym/envs/screenspot_pro/README.md`](/lite/gym/envs/screenspot_pro/README.md).

```bash
# ScreenSpot-Pro — single-step click grounding (high-res professional apps).
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id screenspot_pro \
  --splits eval \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/screenspot_pro.yaml
```

## OSWorld / Lite.OSWorld

369 OSWorld tasks across 10 desktop apps. `--filter` drops tagged `exclude_reason` tasks (infeasible / Google-auth / blocked tasks that are unscored noise) → **`osworld` 325 scored**, **`lite.osworld` 328 scored**.

**Setup:** see [`lite/gym/envs/lite/osworld/README.md`](/lite/gym/envs/lite/osworld/README.md) / [`lite/gym/envs/osworld/README.md`](/lite/gym/envs/osworld/README.md).

```bash
# OSWorld / Lite.OSWorld — desktop GUI navigation.
# --filter drops the exclude_reason tasks (infeasible / broken evaluators — unscored noise).
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id {osworld,lite.osworld} \
  --splits eval \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/{osworld,lite.osworld}.yaml
```

## WindowsAgentArena

WindowsAgentArena (WAA) runs the original 154 Windows 11 tasks in local QEMU
VMs. Thirteen tasks have an upstream `evaluator.func="infeasible"` contract
(`others.exclude_reason="infeasible"`), and three variants per split carry a
hand-curated `block:` exclusion — their evaluator returns reward `1` on the
prepared setup with no agent action (six such variants across the two splits,
split-specific), mirroring `lite.osworld`'s `block:` reasons. The standard
recipe `--filter`s both out — keeping the action space compatible with agents
that cannot expose environment-specific tools — and runs the 138 scored tasks
per split.

**Setup:** see [`lite/gym/envs/waa/README.md`](/lite/gym/envs/waa/README.md)
(KVM, pinned ISO + task assets, prepared qcow2, host requirements, lifecycle).

```bash
# WindowsAgentArena — Windows 11 desktop; --filter drops 13 infeasible + 3 block: → 138 scored.
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id waa \
  --splits {eval,eval_noctxt} \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/waa.yaml
```

Cold-booting a Windows VM per episode dominates wall-clock. `install.sh` builds a
**ready snapshot** so each boot is a ~15-50s restore instead of a ~60-90s cold boot
(used automatically when present — no flag). Run WAA behind the env-server for
admission, ownership cleanup, and retry/recovery during full-suite evals. See
[Snapshot restore](/lite/gym/envs/waa/README.md#snapshot-restore-fast-by-default).

To explicitly reproduce the full upstream 154-task set, use an agent/config
that supports extra tools, omit the filter, and enable `report_infeasible`:

```bash
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id waa \
  --splits eval \
  --env-kwargs '{"extra_tools":["report_infeasible","terminate"]}' \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/waa.yaml
```

A correct `report_infeasible` action scores `1` on an infeasible task and `0`
on a feasible task.

## OSWorld-2

[`osworld_2`](/lite/gym/envs/osworld_2/README.md) is **OSWorld-V2** — 108 capability-graded tasks, a separate benchmark from v1 with **float / partial-credit** scoring. **82 scored** by default (needs a host `OPENAI_API_KEY` for the ~18 LLM-judge tasks); **200-step** budget. Full service/exclusion details are in the [README](/lite/gym/envs/osworld_2/README.md).

**Setup:** see [`lite/gym/envs/osworld_2/README.md`](/lite/gym/envs/osworld_2/README.md).

```bash
# OSWorld-V2 (osworld_2) — harder, capability-graded; 200-step budget set by the config.
# gpt / qwen3_vl / claude osworld_2 configs all exist (each with max_steps=200).
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id osworld_2 \
  --splits eval \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/osworld_2.yaml
```


## CUABench (basic, KiCad, workflows)

145 computer-use tasks across 3 dataset-collection env_ids — `cua.bench.local.basic` (68), `cua.bench.local.kicad` (25), `cua.bench.local.workflows` (52) — on real `cua-xfce` containers (Docker), scored by cua-bench's own evaluator.

**Setup:** see [`lite/gym/envs/cua/README.md`](/lite/gym/envs/cua/README.md).

```bash
# CUABench — computer-use tasks scored by cua-bench's evaluator (env_id = dataset collection).
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id {cua.bench.local.basic,cua.bench.local.kicad,cua.bench.local.workflows} \
  --splits eval \
  --concurrency 8 \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/cua.bench/{basic,kicad,workflows}.yaml
  # --concurrency 1   # KiCad ONLY (per-task apt install truncates under load) — see the env README
# KiCad reproduces the official GPT-5.5 = 6/25 — recipe: /docs/examples/cua.md
```

**Evaluation details:** reward from cua-bench's evaluator (`solved ≥ 0.5`); KiCad must run at
`--concurrency 1`. See [Evaluation](/lite/gym/envs/cua/README.md#evaluation).

## WebGym

292k+ web information-retrieval tasks (Microsoft OmniBoxes), VLM-judged at
episode end. Set `OPENAI_API_KEY` for scored runs, or pass
`--env-kwargs '{"skip_eval": true}'` for unscored smoke tests.

**Setup:** run the WebGym installer first; use env-server mode for concurrent
evals. See
[`lite/gym/envs/webgym/README.md`](/lite/gym/envs/webgym/README.md).

```bash
# WebGym — browser navigation, Qwen3-VL.
uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --env-id webgym \
  --splits eval --head 32 \
  --config-path scripts/configs/qwen3_vl/default/webgym.yaml
  # optional (demo): easy tasks only — add
  #   --filter "lambda m: m.others.get('difficulty', 0) <= 3"
```

## WebVoyager

643 WebVoyager tasks run against WebHarbor self-hosted mirrors inside the
`cua-lite/webharbor.webvoyager:latest` shared container. The container resets the whole
WebHarbor suite once at boot. For strict eval, restart the env-server between
models, run read-only tasks in parallel, then run mutating tasks serially.

Scored runs need `OPENAI_API_KEY`; use `--env-kwargs '{"skip_eval": true}'` for
unscored smoke tests. Build or pull the WebHarbor image before launching the
env-server. See
[`lite/gym/envs/webharbor/webvoyager/README.md`](/lite/gym/envs/webharbor/webvoyager/README.md).

```bash
# Read pass: parallel, residue-immune.
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id webharbor.webvoyager --splits eval --concurrency 32 \
  --filter "lambda m: not m.others.get('mutating')" \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/webharbor.webvoyager/default.yaml

# Write pass: serial, on the same clean baseline after the read pass.
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id webharbor.webvoyager --splits eval --concurrency 1 \
  --filter "lambda m: m.others.get('mutating')" \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/webharbor.webvoyager/default.yaml
```

Swap `default.yaml` → `som.yaml` for Set-of-Marks.

## Online-Mind2Web

300 [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web) tasks (all
under the `eval` split) run inside the `cua-lite/online_mind2web:latest` shared container.
Reward comes from a VLM judge (default `o4-mini`) at episode end — set
`OPENAI_API_KEY` (and `OPENAI_BASE_URL` only for a custom endpoint), or pass
`--env-kwargs '{"skip_eval": true}'` to run without scoring (reward stays `None`).

**Setup:** see [`lite/gym/envs/online_mind2web/README.md`](/lite/gym/envs/online_mind2web/README.md).

```bash
# Online-Mind2Web — online browser navigation (VLM-judged, default o4-mini).
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id online_mind2web \
  --splits eval \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/online_mind2web.yaml
```

## BrowserGym (MiniWoB, WebArena, VisualWebArena)

3 browser benchmarks (MiniWoB, WebArena, VisualWebArena). Run
`lite/gym/envs/browsergym/scripts/install.sh <benchmark>` per benchmark.
MiniWoB runs locally; WebArena and VisualWebArena require Docker services and
`OPENAI_API_KEY` for scored runs.

**Setup:** see [`lite/gym/envs/browsergym/README.md`](/lite/gym/envs/browsergym/README.md).

```bash
# BrowserGym (MiniWoB / WebArena / VisualWebArena). VWA has no default.yaml — use mixed.yaml.
uv run python scripts/rollout.py \
  --model-id {Qwen/Qwen3-VL-8B-Instruct,Qwen/Qwen3.5-9B} \
  --env-id browsergym.{miniwob,webarena,visualwebarena} \
  --splits eval \
  --config-path scripts/configs/{qwen3_vl,qwen3_5}/default/browsergym.{miniwob,webarena}/default.yaml
  # VWA: --config-path scripts/configs/{qwen3_vl,qwen3_5}/default/browsergym.visualwebarena/mixed.yaml
```

**Evaluation details:** runs fully concurrent by default; concurrency corrupts WA/VWA scores. For a
rigorous number, see [Eval rigor (opt-in)](/lite/gym/envs/browsergym/README.md#eval-rigor-opt-in) in the browsergym README.

## AndroidWorld

116 tasks from the official AndroidWorld suite, run on an Android emulator in Docker (needs `/dev/kvm`).

**Setup:** see [`lite/gym/envs/androidworld/README.md`](/lite/gym/envs/androidworld/README.md).

```bash
# AndroidWorld — Android emulator (needs /dev/kvm). Pick a model + its matching config.
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id androidworld \
  --splits eval \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/androidworld.yaml
  # observation_text (env-kwargs): API agents use pixel coords, local VLMs [0,1000] norm. e.g.:
  #   --env-kwargs '{"observation_text": "none"}'        # vision-only (default)
  #   --env-kwargs '{"observation_text": "a11y:pixel"}'  # flat element list (API/pixel)
```

## AndroidLab

138 multi-step tasks across 9 offline apps (bluecoins, calendar, cantook, clock, contacts, map, pimusic, setting, zoom), run on an Android emulator in Docker. Eval-only.

**Setup:** see [`lite/gym/envs/androidlab/README.md`](/lite/gym/envs/androidlab/README.md).

```bash
# AndroidLab — Android emulator, 9 offline apps (eval-only). Pick a model + its matching config.
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id androidlab \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/androidlab.yaml
  # observation_text (env-kwargs): API agents use :pixel coords, local VLMs :norm [0,1000]. e.g.:
  #   --env-kwargs '{"observation_text": "a11y_tree:pixel"}'   # compressed UIAutomator tree (API)
  #   --env-kwargs '{"observation_text": "a11y_tree:norm"}'    # local VLM
```

## MobileWorld

161 tasks (201 upstream − 40 excluded `agent-mcp`) from the [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld) suite, run on a rooted Android emulator in a self-contained **Docker-in-Docker** box (needs `/dev/kvm`; the container runs `--privileged`). 115 GUI-only + 46 agent-user-interaction tasks.

**Setup:** see [`lite/gym/envs/mobileworld/README.md`](/lite/gym/envs/mobileworld/README.md).

```bash
# MobileWorld — DinD Android emulator (needs /dev/kvm, --privileged). Pick a model + its matching config.
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id mobileworld \
  --splits eval \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/mobileworld.yaml
  # agent-user-interaction tasks need a simulated-user LLM: export OPENAI_API_KEY
  # (+ OPENAI_BASE_URL only for a custom endpoint); ask_user is already enabled.
  # Container spawns are heavy (privileged DinD + emulator) — keep --concurrency within ~2-4x of
  # server_kwargs.spawn_concurrency (default 4); for full-suite evals prefer env-server mode.
```

## MobileGym

416 parameterized tasks (256 eval + 160 train) across 24 simulated mobile apps — a lightweight browser-based simulator in Docker (no KVM). Reward is MobileGym's Success Rate (1.0/0.0): goal met, no unexpected state changed, and the episode ended via COMPLETE.

**Setup:** see [`lite/gym/envs/mobilegym/README.md`](/lite/gym/envs/mobilegym/README.md).

```bash
# MobileGym — simulated mobile apps.
uv run python scripts/rollout.py \
  --model-id {gpt-5.5,Qwen/Qwen3-VL-8B-Instruct} \
  --env-id mobilegym --splits eval \
  --config-path scripts/configs/{gpt,qwen3_vl}/default/mobilegym.yaml
  # optional (demo) filters — add any:
  #   --filter "lambda m: not m.others.get('needs_answer_sheet')"  # skip AnswerSheet-query tasks
  #   --filter "lambda m: m.others['difficulty'] in ('L1', 'L2')"  # easy tasks only
  #   --filter "lambda m: m.others['scope'] == 'S1'"               # single-app only
```

### Common Options

```bash
# Subset of tasks (first 32)
--head 32

# Specific tasks via parquet
--prompt-data /path/to/tasks.parquet

# Parallel environments
--concurrency 4

# Resume a previous run
--log-root .logs/rollout/gpt-5.5/osworld/20260329_014332
```

Each run writes its aggregate score to `summary.json` under the selected
`--log-root`. Reuse the same log root to resume an interrupted run.
