# OSWorld-V2

`--env-id` `osworld_2`

CUA-Lite wrapper for [OSWorld-v2.1](https://github.com/xlang-ai/OSWorld-V2/releases/tag/osworld-v2.1). Its 108 capability-graded desktop tasks (`001`–`108`) run via `gym.make("osworld_2@<id>")` in VM-in-Docker, scored by the unmodified upstream evaluators. The matching code, tasks, gated assets, website, and VM image are pinned in the [official release manifest](https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/benchmark_releases/osworld-v2.1.json).

OSWorld-V2 uses the same VM-in-Docker runtime shape as [`osworld`](/lite/gym/envs/osworld/README.md) v1. Its tasks come from gated Hugging Face task classes and may return structured evaluator scores. See [docs/envs.md](/docs/envs.md) for the env contract.

> **Eval-in-container.** The OSWorld-V2 runtime is baked into the derived Docker image, so the host only needs the CUA-Lite package plus Docker/KVM. `install.sh` builds the image and provisions the required V2 assets.

Tasks 103/104 use FreeCAD 0.19 inside the trusted evaluator container; their guest applications are installed by the unmodified task setup.

## Setup

> **KVM required** — `/dev/kvm` must be rw-accessible (usually via the `kvm` group); `/dev/net/tun` too.
>
> **Hugging Face auth required** — accept the gates on [`xlangai/v2-image`](https://huggingface.co/datasets/xlangai/v2-image), [`xlangai/osworld_v2_tasks`](https://huggingface.co/datasets/xlangai/osworld_v2_tasks), and [`xlangai/osworld_v2_assets_gated`](https://huggingface.co/datasets/xlangai/osworld_v2_assets_gated), then `hf auth login`.
>
The host needs Docker with Compose, `git`, `rsync`, `unzip`, and the CUA-Lite `uv` environment. Runtime/evaluator Python dependencies stay inside the container.

```bash
# Choose one install path:
# Clone pinned upstream code, build the evaluator with Python 3.12 + uv sync --frozen,
# and download/verify the official VM, all 108 task classes, and all 1,084 gated assets.
uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image, then provision gated V2 assets.
# uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh status    # image / source / VM / tasks / verified assets / devices
# uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh provision  # source + gated files; leaves Docker images unchanged
# uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh rebuild   # force-rebuild the image after an image-time source edit
```

The installer clones the official code revision recorded in [`data/release.json`](/lite/gym/envs/osworld_2/data/release.json). That checkout's release manifest selects immutable task, asset, website, and VM revisions. Both the downloaded VM archive and extracted disk are SHA256-checked, including an existing disk cache. Task files are checked against the official task-hash manifest; asset files against Hugging Face LFS SHA256 or Git blob SHA1. The anonymous public asset repository is incomplete and is not used as a substitute. No upstream task files are patched.

Files live under `lite/gym/envs/osworld_2/.cache/`. By default, CUA-Lite mounts the verified `osworld_v2_assets/` directory read-only into every VM container; no extra asset environment variable is needed. `OSWORLD_FILE_BASE_URL` can select another complete absolute asset directory or an upstream-compatible asset URL. `OSWORLD_V2_SRC` can reuse a clean checkout of the pinned official commit; use the same setting for image builds and env-server startup.

**Website tasks also need the matching website stack.** After installation, run the fixed setup script with a DNS suffix that the VM can reach:

```bash
HOST_SUFFIX=<host-IP>.nip.io uv run --no-sync bash lite/gym/envs/osworld_2/scripts/setup_website.sh -p osworld-v21
```

The script selects the official website commit and initializes its pinned submodules over HTTPS, then runs upstream Docker Compose. It publishes ports 80/443 by default. On a shared host, pass native Compose options such as `-f /absolute/path/override.yaml` for your ports/socket; set `server_kwargs.website_host_suffix` to the matching suffix (including an HTTP port when needed) in a copy of [`configs/default.yaml`](/lite/gym/envs/osworld_2/configs/default.yaml), then export `OSWORLD_2_CONFIG=/absolute/path/config.yaml`. Check `http://mailhub.<suffix>/api/state` from the host and the VM. Website data/state is isolated per upstream task session.

**Env-server mode (recommended)** uses [`scripts/serve_env.py`](/scripts/serve_env.py):

```bash
# OPENAI_API_KEY and optionally OPENAI_BASE_URL are required for upstream LLM judges.
uv run python scripts/serve_env.py --env-ids osworld_2 --port 30100 --token <server-token>
```

Clients set `CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30100` and `CUA_LITE_ENV_SERVER_TOKEN=<server-token>`. Run `install.sh status` before rollout; an empty or unverified default asset directory fails before VM creation.

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("osworld_2@001", max_steps=200)
asyncio.run(env.reset())
```

Task **enumeration** (`gym.registry.task_ids("osworld_2")`) works with **none** of the above — it reads the vendored `data/test_v2.json` + `data/capabilities/*.json`. The gated code is needed only to `reset()`.

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
| `user_sim_model` | the 7 simulated-user conversation tasks are excluded | reserved; the conversation bridge is not implemented |
| `website_host_suffix` | website tasks excluded | included when the matching v2.1 site is self-hosted |
| `gitlab_url` + `gitlab_private_token` | gitlab-backed tasks excluded | included |

**LLM-judge evaluators.** ~18 tasks call an LLM at `evaluate()` (desktop_env `model_client`), so they need a host **`OPENAI_API_KEY`** (+ optional `OPENAI_BASE_URL`), auto-threaded into each container. Without the key they'd 500/mis-score, so they're **excluded** (`exclude_reason: "llm_judge"`) unless it's set. The default judge model follows upstream (`gpt-4o`); `eval_model` can override it for a separate experiment.

**Coverage limits.** Matching upstream files do not imply that every interaction protocol is implemented. The current wrapper does not support the seven simulated-user conversation tasks or the multi-phase task. GitLab tasks require the GitLab URL/token; website tasks require the stack above. Check `exclude_reason` and record the selected task IDs/count when reporting a run. Task 029's known setup timeout is also documented in the official v2.1 release; do not patch it silently.

## Available Tasks

```python
import lite.gym as gym
print(len(gym.registry.task_ids("osworld_2")))   # 108
```

108 tasks across **10 overlapping capabilities** (not app-domains): `conflict_disambiguation`, `cross_source_reasoning`, `dynamic_environment`, `human_in_the_loop`, `implicit_state_inference`, `multi_item_state_tracking`, `multimodal_editing`, `streaming_interaction`, `tutorial_following`, `visual_spatial_precision`. Metadata identity lives in `env.metadata.others["task_id"]` / `env.metadata.others["env_id"]`; `env.metadata.others` also carries `capabilities` (a task is usually in several), the static service-dependency flags (`website`/`gitlab`/`multi_phase`/`user_sim`/`llm_judge`), the requested `volume_size` in GiB where present, and `exclude_reason` on service-gated tasks. The **scored count depends on provisioned services, supported interaction protocols, and the run's task filter**; report the actual selected task IDs/count. Provisioning every service does not enable the unsupported conversation or multi-phase tasks. Use the matching v2.1 website and judge credentials when reporting the available scored tasks. `google_auth`/`blocked` from v1 are dropped (V2 needs no Google accounts; the v1 uuid blocklist is unrelated to the v2 `001–108` id space).

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via OSWorld-V2's native evaluators (custom Python `evaluate()` or the JSON fallback), returning a float `reward` in `[0.0, 1.0]` (extracted from `["score"]` when `evaluate()` returns a dict).

The action surface is **identical to [`osworld` v1](/lite/gym/envs/osworld/README.md#evaluation)**: mouse, keyboard, scroll, wait, screenshot, and finish actions. If setup fails after changing the attached VM, rollout retries create a fresh VM; setup is never repeated on a dirty snapshot.

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
│   ├── install.sh     # build / rebuild / pull / status image + gated assets + service scan
│   ├── setup_website.sh # pinned website source + submodules + upstream Compose
│   ├── uninstall.sh   # rm downloaded VM/tasks/assets/source + derived image
│   └── cleanup.sh     # docker rm -f by -osworld_2- name filter
├── README.md
└── .cache/            # (gitignored) OSWorld-V2/ + VM + task_class/ + osworld_v2_assets/ + optional OSWorld-web/
```

Runtime and evaluator code live inside the Docker image. The host creates one VM-backed container per trajectory, sends actions to it, and receives screenshots/rewards back through the env API. `install.sh status` is the quickest way to check the image, KVM/TUN access, qcow2, task classes, and service scan.

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
