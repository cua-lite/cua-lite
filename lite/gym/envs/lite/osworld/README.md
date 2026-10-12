# Lite.OSWorld

`--env-id` `lite.osworld`

CUA-Lite wrapper for [OSWorld](https://github.com/xlang-ai/OSWorld) on a **VM-free**, lightweight Docker desktop — the same tasks + evaluators as [`osworld`](/lite/gym/envs/osworld/README.md), but a GNOME-Shell container instead of the ~6 GB `Ubuntu.qcow2` VM. **No nested virtualization**, so it boots in seconds, runs anywhere Docker does (CI, containers-in-containers), and parallelizes trivially. 369 eval tasks (+ a generated training split), via `gym.make("lite.osworld@<task_id>")` with `LiteDesktopActionSpace`. See [docs/envs.md](/docs/envs.md) for the env contract.

(For the original VM-backed wrapper, see [`osworld`](/lite/gym/envs/osworld/README.md).)

OSWorld is an eval-only benchmark (369 tasks across 10 desktop apps, no training split). We programmatically generate 2429 training tasks with deterministic verifiable rewards for RL; rewards preserve the raw OSWorld aggregate, including partial credit where evaluators return it. Task volume is adjustable via `TARGET` in [`src/gen/train/synth/catalog.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/catalog.py) — a **global** row cap split across domains in proportion to each domain's feasible eval-row share, not a per-domain multiplier. It ships as `math.inf`, i.e. uncapped: every template emits at its natural max.

| Split | Tasks | Description |
|---|---|---|
| **eval** | 369 | All OSWorld tasks (10 domains). Some tagged `exclude_reason` (infeasible / Google auth / live-site drift / trivial pass) — filter out for scoring (`--filter "lambda m: not m.others.get('exclude_reason')"`). |
| **train** | 2429 | 1722 synthetic + 707 perturbation tasks for RL training. |

## Setup

```bash
# Choose one install path:
# Source path: install deps, download pinned assets/catalogs, and build if missing/stale.
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh

# Or, published-image path: pull a matching image, then provision deps/assets/catalogs.
# uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh status      # read-only freshness/assets check
# uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh provision   # deps + assets + catalogs only
# uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh rebuild     # force a fresh rebuild
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
export CUA_LITE_ENV_SERVER_URL=http://$(hostname -I | awk '{print $1}'):30100
```

Use `localhost` only when the env-server and rollout client share the same network namespace.

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("lite.osworld@<task_id>", max_steps=30)
asyncio.run(env.reset())
```

<details>
<summary>Setup notes — two-stage build, freshness</summary>

Two-stage build: stage 1 is the **shared** `cua-lite/sandbox.linux` desktop base (GNOME-Shell + Xvnc/noVNC), built by [`lite/gym/sandbox/scripts/install.sh`](/lite/gym/sandbox/scripts/install.sh) and shared with lite.demo; stage 2 (`Dockerfile`) layers OSWorld apps (LibreOffice, GIMP, VLC, Thunderbird, Chrome, VS Code) + the OSWorld Flask server + the osworld appearance layer (Yaru, single top panel, app-pinned dock) FROM it. Host↔container is exec-stdio — no `:8000` in either image; see [/lite/gym/sandbox/exec_stdio/](/lite/gym/sandbox/exec_stdio/). `install.sh` sets the build context for both stages, stages `exec_stdio/server.py`, and stamps the `lite.src_hash` freshness label the image freshness check reads — a plain `docker build` would omit it and the check would refuse the image as STALE. See [docs/envs.md](/docs/envs.md#image-build-and-freshness). Synth assets come from HuggingFace ([`cua-lite/lite.osworld-assets`](https://huggingface.co/datasets/cua-lite/lite.osworld-assets), pinned) into `.cache/assets/pulled/synth/` via `install.sh provision`.

</details>

## Data Generation

All JSONL data files under `data/` are generated from scratch by code under `src/gen/` and are idempotent. Do **not** hand-edit — fix the generation script and regenerate.

| Split | File | Source | Rows |
|---|---|---|---|
| eval | `data/eval.jsonl` | OSWorld task JSONs | 369 |
| train (synth) | `data/train.synth.jsonl` | Programmatic templates | 1722 |
| train (perturb) | `data/train.perturb.jsonl` | Structural variations of eval | 707 |

### Regenerating from scratch

After each full regen, update the catalog lock in the same commit.

```bash
# Full generated catalog lifecycle
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/utils/tasks.sh generate
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/utils/tasks.sh refresh-lock
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/utils/tasks.sh check

# Train, specific track/domain (no sha256 update needed for partial regen)
uv run python -m lite.gym.envs.lite.osworld.src.gen.train --track synth --domain libreoffice_calc
```

### Key code locations

| What | Path |
|---|---|
| Eval generation (OSWorld JSON → JSONL) | `src/gen/eval/__main__.py` |
| Train generation CLI | `src/gen/train/__main__.py` |
| Track A templates (host-side) | `src/gen/train/synth/{libreoffice_calc,libreoffice_writer,...}.py` |
| Track B perturb functions | `src/gen/train/perturb/{libreoffice_calc,libreoffice_writer,...}.py` |
| Design doc + pitfalls | [`devs/envs/lite.osworld/lite.osworld.md`](/devs/envs/lite.osworld/lite.osworld.md) |

### Validating training data

After regeneration, verify each level: L1 (static checks) → L2 (oracle runs) → L4 (agent runs). See [devs/envs/lite.osworld/lite.osworld.md](/devs/envs/lite.osworld/lite.osworld.md#verification-levels) for the full runbook (validation scripts live in [devs/envs/lite.osworld/validate/](/devs/envs/lite.osworld/validate/)).

## Teacher Data Pipeline

Teacher-data collection uses only `train.synth` and `train.perturb`; see
[`devs/data/lite.osworld/AGENTS.md`](/devs/data/lite.osworld/AGENTS.md) for the
runbook. Current flow: collect through env-server at about 32 concurrent
rollouts → filter/annotate → stage → upload/download → `export_sft`. The
annotation pass strips no-op `screenshot` / `wait` actions, preserves canonical
batched GUI actions when any non-no-op child action remains, normalizes
content-only finals to `Done.`, and leaves final tool-call turns unchanged.
Downstream SFT export selects rows with no `exclude_reason` and
`episode_return > 0.5`.

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("lite.osworld@osworld_os_5ea617a3", max_steps=15)
    result = await env.reset()
    print(result.text)   # task instruction

    result = await env.step([
        make_tool_call(
            "computer",
            {"actions": [
                {"action": "click", "coordinate": [500, 300]},
            ]},
            call_id="call_0000",
        ),
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/lite/osworld/configs/default.yaml) — `env_kwargs` (per-instance) + `server_kwargs` (per-deployment infra), read via `env_config.load`. Swap the whole file with `LITE_OSWORLD_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("lite.osworld"))   # {"eval": [...], "train": [...]}
```

369 eval + a generated training split, across 10 desktop domains. Metadata in `env.metadata.others`: `domain`, `exclude_reason`.

## Task Statistics

| Status | Count | Description |
|--------|-------|-------------|
| `oracle_actions` non-empty, no `exclude_reason` | 324 | Curated oracle solution produces reward=1.0 |
| `exclude_reason` set | 41 | Infeasible (29) / Google auth (8) / live-site drift (2) / generated eval bug (1) / trivial pass (1) |
| Unverified (feasible) | 4 | Feasible tasks without an authored oracle |

Counts are locked in `data/catalog.lock.json` from generated `data/eval.jsonl`; regenerate with
`uv run python -m lite.gym.envs.lite.osworld.src.gen.eval`
and recount if the source JSONs change.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`): the same OSWorld
built-in evaluators as [`osworld`](/lite/gym/envs/osworld/README.md#evaluation) — checking real
system state (files, command output, app state) — return a float `reward` in `[0.0, 1.0]`, just on a
GNOME-Shell container instead of a full KVM VM. `--filter "lambda m: not m.others.get('exclude_reason')"`
drops the 41 excluded tasks (see [Task Statistics](#task-statistics))
→ 328 scored. **Extra tool:** active `report_infeasible(reason)` is an env-local terminal tool evaluated
directly by the OSWorld infeasible checker; it is not rewritten to canonical `terminate`.

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/lite/osworld/
├── main.py                          # LiteOsworldEnv, register "lite.osworld"
├── src/
│   ├── utils/
│   │   ├── setup.py                 # setup_fn → dispatch_actions(config)
│   │   ├── dispatch.py              # shared action dispatch (config/postconfig/oracle)
│   │   └── verify.py                # evaluate_final_fn (raw aggregate reward + infeasible)
│   ├── eval/
│   │   ├── runner.py                # postconfig → getters → OSWorld metrics
│   │   └── metrics.py               # compare_docx_strict, compare_pptx_files, ...
│   └── gen/
│       ├── common.py                # NOISE_CANDIDATES + LO/VS_CODE/GIMP save postconfig (shared across eval+train)
│       ├── eval/                    # OSWorld JSON → eval.jsonl (369 tasks)
│       └── train/
│           ├── __main__.py          # Track A synth + Track B perturb CLI
│           ├── synth/               # Track A: per-domain template files
│           │   ├── _utils.py        # SynthTemplate, make_synth_row, helpers
│           │   └── {libreoffice_calc,libreoffice_writer,...}.py
│           └── perturb/             # Track B: per-domain perturb functions
│               ├── _utils.py        # KnobSpec, make_perturb_row
│               └── {libreoffice_calc,libreoffice_writer,...}.py
├── scripts/
│   ├── cleanup.sh                   # local cleanup helper
│   └── install.sh                   # local install helper
├── docker/
│   ├── Dockerfile                   # additive: 10 apps + OSWorld Flask server + appearance, FROM cua-lite/sandbox.linux (shared base)
│   └── server/main.py               # OSWorld Flask server
└── data/
    ├── catalog.lock.json            # generated catalog row/hash/count lock
    ├── eval.jsonl                   # 369 eval tasks
    ├── train.synth.jsonl            # 1722 synthetic training tasks
    └── train.perturb.jsonl          # 707 perturbation training tasks
```

**References:**
- [OSWorld](https://github.com/xlang-ai/OSWorld) — Original benchmark
- [CUA](https://github.com/trycua/cua) — desktop image lineage only (`cua-xfce` + its noVNC fork); the host↔container transport is [exec-stdio](/lite/gym/sandbox/exec_stdio/), not the `cua` agent / computer-server SDK

</details>

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@inproceedings{xie2024osworld,
  author    = {Tianbao Xie and Danyang Zhang and Jixuan Chen and Xiaochuan Li and Siheng Zhao and Ruisheng Cao and Toh Jing Hua and Zhoujun Cheng and Dongchan Shin and Fangyu Lei and Yitao Liu and Yiheng Xu and Shuyan Zhou and Silvio Savarese and Caiming Xiong and Victor Zhong and Tao Yu},
  title     = {OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments},
  booktitle = {Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024},
  year      = {2024},
  url       = {http://papers.nips.cc/paper\_files/paper/2024/hash/5d413e48f84dc61244b6be550f1cd8f5-Abstract-Datasets\_and\_Benchmarks\_Track.html}
}
```
