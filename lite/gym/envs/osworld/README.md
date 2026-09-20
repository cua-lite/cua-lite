# OSWorld

`--env-id` `osworld`

CUA-Lite wrapper for the official [OSWorld](https://github.com/xlang-ai/OSWorld) benchmark (NeurIPS 2024). 369 real-world desktop tasks across 10 apps (eval-only), via `gym.make("osworld@<task_id>")` with `LiteDesktopActionSpace`. Each trajectory runs on a locally-managed VM-in-Docker container (the derived `cua-lite/osworld` image, `FROM happysixd/osworld-docker`; QEMU/KVM booting `Ubuntu.qcow2`) scored by OSWorld's **native evaluators** (strict upstream reproduction). For the lightweight GNOME-container variant see [`lite.osworld`](/lite/gym/envs/lite/osworld/README.md). See [docs/envs.md](/docs/envs.md) for the env contract.

> **Eval-in-container.** OSWorld's runtime and evaluators are baked into the `cua-lite/osworld` image, so the host only needs the CUA-Lite package plus Docker/KVM. `install.sh` builds the image and provisions the VM disk.

## Setup

> **KVM required** — `/dev/kvm` must be rw-accessible (usually via the `kvm` group); `/dev/net/tun` too.

```bash
# Choose one install path:
# Source path: build the derived cua-lite/osworld image and provision Ubuntu.qcow2 (22.8 GiB).
uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image, then provision Ubuntu.qcow2.
# uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh status     # image / qcow2 / KVM / tun presence
# uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh provision  # qcow2/assets only, assumes image exists
# uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh rebuild    # force-rebuild the image after an image-time source edit
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("osworld@<task_id>", max_steps=30)
asyncio.run(env.reset())
```

<details>
<summary>Setup notes</summary>

Each `gym.make` starts a fresh VM-backed container from the prepared image and disk. For larger runs, prefer env-server mode so the model process does not need local env dependencies.

**KVM setup:**

```bash
sudo usermod -aG kvm $(whoami) && newgrp kvm     # group method
sudo setfacl -m u:$(id -u):rw /dev/kvm           # ACL method (per boot)
ls -l /dev/kvm                                   # verify: your user has rw
```

**Cleanup:** `uv run --no-sync bash lite/gym/envs/osworld/scripts/cleanup.sh` removes lingering `lite-env-*-osworld-*` containers (scope with `SESSION_ID=...`). Full removal: `scripts/uninstall.sh` (qcow2 + image).

</details>

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("osworld@357ef137-7eeb-4c80-a3bb-0951f26a8aff", max_steps=30)
    result = await env.reset()          # cold-boots a VM (~30-90 s), runs OSWorld setup
    print(result.text)
    # "I have calculated the total work hours from the everyday hours. And I have..."

    result = await env.step([
        make_tool_call("computer", {"actions": [
            {"action": "click", "coordinate": [500, 300]},
            {"action": "type", "text": "hello"},
        ]}, call_id="call_0000"),
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/osworld/configs/default.yaml) — `env_kwargs` (per-instance), `server_kwargs` (deployment settings), and `make_kwargs`, read via `env_config.load`. Swap the whole file with `OSWORLD_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("osworld"))
# {"eval": ["357ef137-...", ...]}   # 369 tasks (eval-only)
```

369 tasks across 10 domains (chrome, gimp, libreoffice_calc/impress/writer, multi_apps, os, thunderbird, vlc, vs_code). Metadata identity lives in `env.metadata.others["task_id"]` / `env.metadata.others["env_id"]`; `env.metadata.others` also carries `domain` and `exclude_reason` on the 44 tagged tasks (29 infeasible + 8 google_auth + 7 blocked) — `--filter` them out for scoring → 325 scored. `blocked` (7) + `google_auth` (8) are OSWorld-specific tags; [`lite.osworld`](/lite/gym/envs/lite/osworld/README.md) currently tags 41 excluded tasks (29 infeasible + 8 Google auth + 2 live-site drift + 1 generated eval bug + 1 trivial pass) → 328 scored. Infeasible tasks are scored only with the `report_infeasible` extra tool.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via OSWorld's native evaluators, which check real VM state (files, command output, app / browser state) and return a float `reward` in `[0.0, 1.0]`.

| Action | Description |
|--------|-------------|
| `click` | Mouse click (`clicks` 2→double, 3→triple) |
| `mouse_move` | Move cursor |
| `mouse_down` / `mouse_up` | Press / release button |
| `drag` | Click-drag gesture |
| `scroll` | Scroll wheel, including horizontal scroll |
| `type` | Type text |
| `key` | Hotkey combo |
| `key_down` / `key_up` | Press / release keys |
| `hold_key` | Hold keys for a duration |
| `wait` | Pause |
| `screenshot` / `cursor_position` | Observe without changing state |
| `terminate` / `response` | End episode and trigger evaluation |

**Extra tool:** `report_infeasible(reason)` (opt-in via `env_kwargs.extra_tools`) — the agent gives up; routed to OSWorld's `FAIL` action, so an infeasible task scores 1.0 iff reported.

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/osworld/
├── main.py            # env class, service registration, task registration
├── container.py       # VM-backed container helpers
├── docker/
│   ├── Dockerfile     # derived image with OSWorld runtime
│   └── server.py      # in-container env API
├── data/tasks.json    # vendored 369-task index
├── configs/default.yaml
├── scripts/
│   ├── install.sh     # build / rebuild / pull / status image + provision qcow2
│   ├── uninstall.sh   # rm qcow2 + image
│   └── cleanup.sh     # docker rm -f filter (per-session)
└── .cache/            # (gitignored) Ubuntu.qcow2
```

Each trajectory uses a fresh VM-backed container. Startup usually takes 30-90 seconds, so tune rollout concurrency to host RAM/CPU and prefer env-server mode for full-suite evals.

Key knobs (see [Configuration](#configuration) for the file + override):

| Setting | `default.yaml` key | Default |
|---------|--------------------|---------|
| VM RAM / CPU / disk | `server_kwargs.ram_size` / `cpu_cores` / `disk_size` | 4G / 4 / 32G (OSWorld upstream) |
| Boot timeout (guest `:5000` ready) | `server_kwargs.boot_timeout` | 300 s |
| Reset / step timeout | `make_kwargs.reset_timeout` / `step_timeout` | 600 s / 180 s |
| Screen size (VM-fixed, not resizable) | `server_kwargs.screen_width` / `screen_height` | 1920×1080 |
| Spawn image | `server_kwargs.image` | `cua-lite/osworld` (derived, built by install.sh) |

**Image build:** `install.sh` builds or adopts the derived `cua-lite/osworld` image from [`docker/`](/lite/gym/envs/osworld/docker/). Use `install.sh status` for image and resource diagnostics, and `install.sh rebuild` after image-time source edits.

**References:** [OSWorld paper](https://arxiv.org/abs/2404.07972) · [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)

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
