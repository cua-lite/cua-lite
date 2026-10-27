# OSWorld-V2

`--env-id` `osworld_2`

CUA-Lite wrapper for the official [OSWorld 2.0](https://github.com/xlang-ai/OSWorld) benchmark (release `osworld-v2-2026.06.24`). 108 capability-graded desktop tasks (ids `001`–`108`, eval-only), via `gym.make("osworld_2@<id>")` with `LiteDesktopActionSpace`. Each trajectory runs on a locally-managed VM-in-Docker container (the derived `cua-lite/osworld_2` image, `FROM happysixd/osworld-docker`; QEMU/KVM booting the gated **v2** `osworld-v2-ubuntu-x86.qcow2`) scored by OSWorld's **native evaluators** (strict upstream reproduction).

OSWorld-V2 uses the same VM-in-Docker runtime shape as [`osworld`](/lite/gym/envs/osworld/README.md) v1. Its tasks come from gated Hugging Face task classes and may return structured evaluator scores. See [docs/envs.md](/docs/envs.md) for the env contract.

> **Eval-in-container.** The OSWorld-V2 runtime is baked into the derived Docker image, so the host only needs the CUA-Lite package plus Docker/KVM. `install.sh` builds the image and provisions the gated qcow2 + task classes; the asset snapshot has its own [`scripts/provision_assets.sh`](/lite/gym/envs/osworld_2/scripts/provision_assets.sh).

## Setup

> **KVM required** — `/dev/kvm` must be rw-accessible (usually via the `kvm` group); `/dev/net/tun` too.
>
> **Hugging Face auth required** — accept the gates on [`xlangai/v2-image`](https://huggingface.co/datasets/xlangai/v2-image), [`xlangai/osworld_v2_tasks`](https://huggingface.co/datasets/xlangai/osworld_v2_tasks), and [`xlangai/osworld_v2_assets_gated`](https://huggingface.co/datasets/xlangai/osworld_v2_assets_gated), then `hf auth login`. The qcow2 + task-class + asset downloads 401 otherwise.
>
> **Local OSWorld-V2 source required for image build/pull freshness** — set `OSWORLD_V2_SRC` to a checked-out official V2 source tree, or place it at `lite/gym/envs/osworld_2/_vendor/OSWorld-V2`. The image freshness label includes that source tree's content digest; without the same source identity, `install.sh pull` intentionally rejects published images instead of trusting an unverified binary.

```bash
# Choose one install path:
# Source path: build cua-lite/osworld_2:latest, download the gated ~14 GiB archive,
# verify/extract the ~25.5 GiB qcow2, then stage task classes and a static service scan.
uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image, then provision the gated
# docker-free prerequisites.
# uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh pull

# The gated asset snapshot (~3.4 GiB) is a separate script — install.sh is an
# image-freshness hash input, and this host-only download must not re-stale images.
uv run --no-sync bash lite/gym/envs/osworld_2/scripts/provision_assets.sh

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh status    # image / qcow2 / task-class files / KVM+tun presence
# uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh provision  # docker-free prerequisites (qcow2 + task classes + scan)
# uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh rebuild   # force-rebuild the image after an image-time source edit
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`.

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("osworld_2@001", max_steps=200)
asyncio.run(env.reset())
```

Task **enumeration** (`gym.registry.task_ids("osworld_2")`) works with **none** of the above — it reads the vendored `data/test_v2.json` + `data/capabilities/*.json`. The gated code + assets are needed only to `reset()`.

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("osworld_2@001", max_steps=200)
    result = await env.reset()          # cold-boots a VM (~30-90 s), runs V2 task setup
    print(result.text)       # task instruction

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

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/osworld_2/configs/default.yaml) — `env_kwargs` (per-instance), `server_kwargs` (deployment/service settings), and `make_kwargs`, read via `env_config.load`. Swap the whole file with `OSWORLD_2_CONFIG=<abs-path | name>`.

**Service knobs** (in `server_kwargs`) drive `exclude_reason` at registration. Each set knob is also
threaded into every container as an `-e` var (via `service_env`) so the in-container task setup can
reach the service:
| knob | when unset | when set |
|---|---|---|
| `user_sim_model` | the 7 `human_in_the_loop` tasks are excluded | HITL tasks included (needs a user-sim LLM wired) |
| `website_host_suffix` | website tasks excluded | included (default `"web.hku.icu"`, official OSWorld-web) → ~31 website tasks scored |
| `gitlab_url` + `gitlab_private_token` | gitlab-backed tasks excluded | included |

**LLM-judge evaluators.** ~18 tasks call an LLM at `evaluate()` (desktop_env `model_client`), so they need a host **`OPENAI_API_KEY`** (+ optional `OPENAI_BASE_URL`), auto-threaded into each container. Without the key they'd 500/mis-score, so they're **excluded** (`exclude_reason: "llm_judge"`) unless it's set. The judge model is the `eval_model` knob (**default `"gpt-4.1"`**, like webgym — gpt-4o's deployment 429s at high concurrency).

**Stateful-website tasks** (39 tagged via `others['website']==True`; ~31 net scored when enabled) need the companion website service. With the default hosted suffix, tasks are isolated per session and can be rolled out concurrently.

## Available Tasks

```python
import lite.gym as gym
print(len(gym.registry.task_ids("osworld_2")))   # 108
```

108 tasks across **10 overlapping capabilities** (not app-domains): `conflict_disambiguation`, `cross_source_reasoning`, `dynamic_environment`, `human_in_the_loop`, `implicit_state_inference`, `multi_item_state_tracking`, `multimodal_editing`, `streaming_interaction`, `tutorial_following`, `visual_spatial_precision`. Metadata identity lives in `env.metadata.others["task_id"]` / `env.metadata.others["env_id"]`; `env.metadata.others` also carries `capabilities` (a task is usually in several), the static service-dependency flags (`website`/`gitlab`/`volume`/`multi_phase`/`user_sim`/`llm_judge`, for filtering — e.g. `--filter "lambda m: m.others.get('website')"`), and `exclude_reason` on service-gated tasks. The **scored count is environment-dependent** — `108 − (whichever services are unprovisioned)` — not a fixed number like v1's 325. With the default `website_host_suffix` **and a host `OPENAI_API_KEY`** → **82 scored** (without the key the ~18 `llm_judge` tasks also drop → **67**). `google_auth`/`blocked` from v1 are dropped (V2 needs no Google accounts; the v1 uuid blocklist is unrelated to the v2 `001–108` id space).

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via OSWorld-V2's native evaluators (custom Python `evaluate()` or the JSON fallback), returning a float `reward` in `[0.0, 1.0]` (extracted from `["score"]` when `evaluate()` returns a dict).

The action surface is **identical to [`osworld` v1](/lite/gym/envs/osworld/README.md#evaluation)**: mouse, keyboard, scroll, wait, screenshot, and finish actions. **Extra tool:** `report_infeasible(reason)` (opt-in) gives up on infeasible tasks.

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/osworld_2/
├── main.py            # OSWorldV2Env (thin JSON-RPC client — no desktop_env) + services + Tier-1 registration
├── container.py       # OSWorldV2Container + factory (spawn derived image, map the 1 API port, launch server)
├── docker/            # Dockerfile (FROM happysixd/osworld-docker + V2 desktop_env) + server.py (in-container eval) + _vendor/ (gitignored)
├── configs/default.yaml
├── data/              # (committed) release.json + test_v2.json + capabilities/*.json — registration metadata
├── scripts/
│   ├── install.sh     # build / rebuild / pull / status image + gated qcow2/task classes + service scan
│   ├── provision_assets.sh  # gated asset snapshot → .cache/assets (outside install.sh: freshness hash input)
│   ├── uninstall.sh   # rm v2 qcow2 + task_class/ + assets/ + derived image
│   └── cleanup.sh     # docker rm -f by -osworld_2- name filter
├── README.md
└── .cache/            # (gitignored) osworld-v2-ubuntu-x86.qcow2 + task_class/task_*.py + assets/ (gated snapshot) + evaluator goldens
```

Runtime and evaluator code live inside the Docker image. The host creates one VM-backed container per trajectory, sends actions to it, and receives screenshots/rewards back through the env API. `install.sh status` is the quickest way to check the image, KVM/TUN access, qcow2, task classes, and service scan. Each container bind-mounts the asset snapshot at `/assets` with `OSWORLD_FILE_BASE_URL=/assets`, so `asset(...)` lookups (setup downloads, evaluator gold files) stay local.

**References:** [OSWorld paper](https://arxiv.org/abs/2404.07972) · [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)

</details>

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@misc{yuan26osworld2,
  title         = {OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks},
  author        = {Mengqi Yuan and Zilong Zhou and Xinzhuang Xiong and Weiming Wu and Jiayang Sun and Jiamin Song and Kaiqian Cui and Bowen Wang and Haoyuan Wu and Yitong Li and Dunjie Lu and Haikong Lu and Qi Zhen and Xinyuan Wang and Jiaqi Deng and Yuhao Yang and Cheng Chen and Boyuan Zheng and Alex Su and Xiao Yu and Hao Zou and Saaket Agashe and Xing Han Lu and Manpreet Kaur and Zhengyang Qi and Vincent Sunn Chen and Frederic Sala and Dayiheng Liu and Junyang Lin and Zhou Yu and Yu Su and Siva Reddy and Xin Eric Wang and Peng Qi and Tianbao Xie and Tao Yu},
  year          = {2026},
  eprint        = {2606.29537},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2606.29537}
}
```
