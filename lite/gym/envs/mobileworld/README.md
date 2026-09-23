# MobileWorld

`--env-id` `mobileworld`

CUA-Lite wrapper for [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld). 201 upstream tasks (161 registered; 40 `agent-mcp` excluded), with **117 GUI-only tasks selected by the default eval runner**. The 44 registered user-interaction tasks carry `exclude_reason="ask_user"`. Tasks span 20 mobile apps in self-contained Docker-in-Docker boxes (rooted Android emulator + self-hosted Mattermost/Mastodon/Mall backends), via `gym.make("mobileworld@<TaskClassName>")` with the mobile action space. See [docs/envs.md](/docs/envs.md) for the env contract.

## Setup

> **KVM required** — `/dev/kvm` must be rw-accessible (usually via the `kvm` group).
> Build/pull does not need KVM; KVM is required when running tasks.

```bash
# Choose one install path:
# Source path: build cua-lite/mobileworld:latest if missing/stale.
# The build pulls the pinned upstream prebuilt base (~10.5 GB compressed) and overlays source @ 8ae5064.
uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image.
# uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh status    # image freshness/resources
# uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh rebuild   # force a fresh rebuild
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("mobileworld@AcceptMeetingTask", max_steps=30)
asyncio.run(env.reset())
```

<details>
<summary>Setup notes</summary>

Each container runs the Android emulator and the benchmark's app backends. Cold boot can take 2-5 minutes; env-server mode is recommended for full-suite evals because it centralizes admission, ownership cleanup, and retry/recovery.

**KVM setup:**

```bash
sudo usermod -aG kvm $(whoami) && newgrp kvm     # group method
sudo setfacl -m u:$(id -u):rw /dev/kvm           # ACL method (per boot)
ls -l /dev/kvm                                   # verify: your user has rw
```

**Agent-user-interaction tasks are filtered out by the default eval runner.** There are 46 upstream tasks tagged `agent-user-interaction`, of which 44 are registered (the other two also require MCP). To explicitly run them with an agent that supports asking the user, omit the exclusion filter and enable `ask_user` in `env_kwargs.extra_tools`. Before starting the env-server, export `USER_AGENT_API_KEY` and `USER_AGENT_BASE_URL` for its OpenAI-compatible simulator endpoint. These take precedence over `OPENAI_API_KEY` and `OPENAI_BASE_URL`, allowing the simulator and evaluated model to use separate services. The env forwards the credentials into each new container; restart the server and recreate its containers after changing them. The model is `server_kwargs.user_agent_model` in the env config (default `gpt-5.6-luna`). GUI-only tasks need neither the tool nor these credentials.

**MCP tasks are not supported** at this stage: the 40 `agent-mcp`-tagged tasks are excluded from registration (`data/tasks.json` still lists all 201).

**Concurrency:** container startup is heavy. Keep rollout `--concurrency` close to the configured spawn concurrency (`server_kwargs.spawn_concurrency`, default 4). For full-suite evals, prefer env-server mode.

**Cleanup:** `uv run --no-sync bash lite/gym/envs/mobileworld/scripts/cleanup.sh` removes lingering `lite-env-*-mobileworld-*` containers (scope with `SESSION_ID=...`). Image removal: `scripts/uninstall.sh`.

</details>

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("mobileworld@AcceptMeetingTask", max_steps=30)
    result = await env.reset()          # cold boot: several minutes
    print(result.text)
    # "Reply to Daniel's most recent email to tell him: 'I'll be there at 10:00 AM on Thursday.'"

    result = await env.step([
        make_tool_call(
            "mobile",
            {"actions": [{"action": "tap", "coordinate": [500, 500]}]},
            call_id="call_0000",
        )
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/mobileworld/configs/default.yaml) — `env_kwargs` (per-instance) + `server_kwargs` (per-deployment infra), read via `env_config.load`. Swap the whole file with `MOBILE_WORLD_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("mobileworld"))
# {"eval": ["AcceptMeetingTask", ...]}   # 161 tasks (deterministic → eval split only)
```

161 registered tasks (201 upstream − 40 `agent-mcp`): 117 GUI-only + 44 agent-user-interaction. Interaction tasks remain registered with `metadata.others.exclude_reason="ask_user"`, independent of the model or enabled tools. The [eval runner](/devs/exps/eval/mobileworld/run.sh) filters them by default, following the OSWorld convention. When invoking `scripts/rollout.py` directly, pass the same filter (model YAMLs configure the agent and environment, not task selection):

```bash
uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --config-path scripts/configs/gpt/default/mobileworld.yaml \
  --splits eval \
  --filter "lambda m: not m.others.get('exclude_reason')"
```

Apps covered: Mail, Messages, Mastodon, Files, Calendar, Mattermost, Taodian (e-commerce), Maps, Chrome, Settings, Camera, etc. Other metadata in `env.metadata.others`: `tags` (`lang-en`/`lang-cn`, `agent-user-interaction`), `apps` (the full launchable catalog — the `open_app` enum source), and `task_apps` (the apps this task involves).

## Evaluation

Runs **at episode end** — `terminate` / `response` **or** `max_steps` truncation — via the in-container `/task/eval`, which checks real device / backend state (databases, files, app callbacks) and returns the score as `reward` (upstream parity: truncated episodes get their real state-based score, not an automatic 0). Tasks are deterministic; every `reset()` reloads the same `init_state` snapshot.

| Action | Description | MobileWorld Translation |
|--------|-------------|------------------------|
| `tap` | Single tap | `click` at pixel coords (`clicks≥2` → `double_tap`) |
| `long_press` | Touch and hold | `long_press` at pixel coords |
| `swipe` / `drag` | Drag gesture | `drag` (precise start/end coords) |
| `type` | Text input | `input_text` |
| `system_button` | Home/Back/Enter | `navigate_home` / `navigate_back` / `keyboard_enter` |
| `open_app` | Launch app (extra tool, opt-in) | `open_app` |
| `ask_user` | Query the simulated user (extra tool, opt-in) | `ask_user` — the reply lands in the next observation's text |
| `wait` | Pause | `wait` |
| `screenshot` | No-op | Screenshot always taken after each step |
| `terminate` / `response` | Standalone extra tools | `response` first issues `answer` (Q&A evaluators read it), then evaluation |

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/mobileworld/
├── main.py                 # env class, service registration, task registration
├── container.py            # container helpers
├── data/tasks.json         # static task manifest (goal/tags/apps)
├── configs/default.yaml    # env_kwargs + server_kwargs + make_kwargs defaults
├── docker/
│   └── Dockerfile          # FROM upstream prebuilt image + source overlay pinned @ 8ae5064
└── scripts/
    ├── install.sh          # build / rebuild / pull / status
    ├── tasks.sh            # dump data/tasks.json from the image (re-run when the pin changes)
    ├── cleanup.sh          # docker rm -f filter (per-session)
    └── uninstall.sh        # remove the CUA-Lite final image
```

**Why base on the prebuilt image:** the upstream repo's own Dockerfile needs the `Pixel_8_API_34_x86_64.avd` snapshot dir, `docker/skins`, and `docker/images/*.tar` (backend app images) — none of which are in git. Code-level reproducibility is pinned by cloning the upstream source at a fixed SHA over the prebuilt base (the same pattern as upstream's `Dockerfile.update`).

**Runtime model:** env-server mode cold-starts task instances on demand, then owns admission, cleanup, and recovery. Direct mode starts containers in-process and is best kept for smoke tests.

Key `server_kwargs` (see [Configuration](#configuration) for the file + override):

| Setting | `default.yaml` key | Default |
|---------|--------------------|---------|
| Boot deadline (docker run → `/init` 200) | `server_kwargs.boot_timeout` | 900 s |
| Per-container memory cap | `server_kwargs.memory_limit` | 32g |
| `/task/init` HTTP timeout | `server_kwargs.task_init_timeout` | 600 s |
| `/task/eval` HTTP timeout | `server_kwargs.eval_timeout` | 600 s |
| Spawn image | `env_kwargs.image` | `cua-lite/mobileworld:latest` |

**Rebuild semantics:** changes under `docker/` (Dockerfile, the `MOBILEWORLD_SHA` pin) are picked up by re-running `install.sh` (use `rebuild` to force; then re-run `scripts/tasks.sh` if tasks changed). Changes to host-side code (main.py, container.py) take effect on the next env-server restart.

</details>

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@inproceedings{kong2026mobileworld,
  author    = {Quyu Kong and Xu Zhang and Zhenyu Yang and Nolan Gao and Chen Liu and Panrong Tong and Chenglin Cai and Hanzhang Zhou and Jianan Zhang and Liangyu Chen and Zhidan Liu and Steven Hoi and Yue Wang},
  title     = {MobileWorld: Benchmarking Autonomous Mobile Agents in Agent-User Interactive and MCP-Augmented Environments},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2026, San Diego, California, United States, July 2-7, 2026},
  pages     = {6142--6167},
  publisher = {Association for Computational Linguistics},
  year      = {2026},
  url       = {https://doi.org/10.18653/v1/2026.acl-long.278},
  doi       = {10.18653/V1/2026.ACL-LONG.278}
}
```
