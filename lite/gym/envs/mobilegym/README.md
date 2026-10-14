# MobileGym

`--env-id` `mobilegym`

CUA-Lite wrapper for [MobileGym](https://arxiv.org/abs/2605.26114). 416 parameterized task templates across 24 launchable apps (WeChat, Alipay, Bilibili, RedNote, X, Reddit, Spotify, 12306, Map, etc.), via `gym.make("mobilegym@<suite>.<TaskName>")` with `LiteMobileActionSpace`. See [docs/envs.md](/docs/envs.md) for the env contract.

## Setup

```bash
# Choose one install path:
# Source path: build cua-lite/mobilegym:latest if missing/stale (~4.5 GB).
# The build bundles the pinned companion media dataset; no host clone/npm/Playwright/dataset staging.
uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image.
# uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh status    # image freshness/resources
# uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh rebuild   # force a fresh rebuild
```

Requires Docker. `uv run --no-sync bash lite/gym/envs/mobilegym/scripts/uninstall.sh` removes the image.

> **Media is bundled (offline real media).** The build downloads the pinned
> `mobilegym-data-v0.1.0.tar.gz` release (~1.4 GB) into `dist/cdn/`, so the container
> serves `/cdn/...` locally with **no runtime CDN egress** — image-bearing apps
> (Bilibili / eBay / RedNote) render real media offline. (This is most of the image's
> ~4.5 GB.)

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("mobilegym@<suite>.<TaskName>", max_steps=30)
asyncio.run(env.reset())
```

On first `gym.make`, `MobileGymContainerServices.ensure` runs one `cua-lite/mobilegym` container (named `mobilegym-<server_port>`) and points `MOBILEGYM_RPC_URL` at its published RPC port. The container is the shared backend for the whole env-server; it is `docker rm -f`'d only at server shutdown / boot recovery.

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("mobilegym@bilibili.OpenRankingTask", max_steps=10)
    result = await env.reset()
    print(result.text)
    # e.g. "打开B站排行榜。"

    result = await env.step([
        make_tool_call(
            "mobile",
            {"actions": [{"action": "tap", "coordinate": [500, 300]}]},
            call_id="call_0000",
        )
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/mobilegym/configs/default.yaml) — `env_kwargs` (per-instance) + `server_kwargs` (per-deployment infra), read via `env_config.load`. Swap the whole file with `MOBILEGYM_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("mobilegym"))
# {"eval": ["account.Railway12306ChangePassword", ...],  (256 tasks)
#  "train": ["alipay.AnalyzeSpending", ...]}              (160 tasks)
```

416 tasks across 24 launchable apps. Task IDs use `{suite}.{ClassName}` format (e.g. `wechat.ReadMyWxid`, `crossapp_life.RailwayEarliestGTrainToWechat`). Each task is a parameterized template that generates different instances via seed — eval split uses `seed=42` (deterministic), train split randomizes each reset.

**Difficulty distribution (eval, 256 tasks):** L1=20, L2=73, L3=83, L4=80.

Metadata in `env.metadata.others`: `difficulty`, `scope`, `objective`, `composition`, `suite`, `apps`, `capabilities`, `needs_answer_sheet`, `max_steps` (filters below).

### Metadata & Filtering

Rich metadata on every task — use `--filter` to select subsets:

| Field | Values | Example filter |
|---|---|---|
| `difficulty` | L1 / L2 / L3 / L4 | `m.others["difficulty"] in ("L1", "L2")` |
| `scope` | S1 (single-app) / S2 (two-app) / S3 (three+) | `m.others["scope"] == "S1"` |
| `objective` | operate / query / hybrid | `m.others["objective"] == "operate"` |
| `composition` | atomic / sequential / transfer / deep_dive | `m.others["composition"] != "transfer"` |
| `suite` | wechat, alipay, bilibili, crossapp_life, ... | `m.others["suite"] == "wechat"` |
| `apps` | list of app IDs involved | `"alipay" in m.others.get("apps", [])` |
| `capabilities` | nav, search, extract, reasoning, handoff, ... | `"reasoning" in m.others.get("capabilities", [])` |
| `needs_answer_sheet` | True (query tasks requiring AnswerSheet app) | `not m.others.get("needs_answer_sheet")` |
| `max_steps` | 15 / 30 / 45 / 60 | `m.others["max_steps"] <= 30` |

```bash
# Eval: only easy operate tasks (no AnswerSheet)
uv run python scripts/rollout.py --env-id mobilegym --splits eval \
  --filter "lambda m: m.others['difficulty'] in ('L1','L2') and not m.others.get('needs_answer_sheet')"

# Train: single-app tasks only
uv run python scripts/rollout.py --env-id mobilegym --splits train \
  --filter "lambda m: m.others['scope'] == 'S1'"
```

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via a deterministic state-diff judge that reads the full JSON app state. The score is MobileGym's own **Success Rate**: the goal met, no unexpected state changed, and the episode ended via `terminate` with `status="success"` (or a `response` in text mode) — `terminate(status="failure")` steps ABORT and scores 0 even if the goal was met. RL can opt into partial credit with the `reward_shaping` env_kwarg ([`configs/default.yaml`](/lite/gym/envs/mobilegym/configs/default.yaml)); eval must leave it off — see [docs/grpo.md](/docs/grpo.md#mobilegym).

## Action Space

Both MobileGym and CUA-Lite use `[0, 1000]` normalized coordinates — no conversion needed.

| Action | MobileGym Action | Notes |
|---|---|---|
| `tap(coordinate)` | `CLICK(point)` | Direct mapping |
| `long_press(coordinate, duration)` | `LONG_PRESS(point, duration)` | duration: seconds → milliseconds |
| `type(text)` | `TYPE(value)` | Field rename |
| `swipe(start_coordinate, coordinate)` | `SWIPE(point1, point2)` | Field rename |
| `open_app(app_name)` | `AWAKE(value)` | Standalone extra tool; supports Chinese names (微信, 支付宝, ...) |
| `system_button(Back)` | `BACK` | |
| `system_button(Home)` | `HOME` | |
| `system_button(Enter)` | `ENTER` | |
| `system_button(Menu)` | `RECENT` | |
| `wait(duration)` | `WAIT(value)` | |
| `terminate(status="success")` | `COMPLETE` | Standalone extra tool; triggers evaluation |
| `terminate(status="failure")` | `ABORT` | Standalone extra tool; triggers evaluation |
| `response(text)` | `ANSWER(value)` | Standalone extra tool for query tasks |
| `screenshot` | no-op | Screenshot always taken after each step |
| `pinch` | — | MobileGym simulator does not support pinch-to-zoom; the container reports it back as `unsupported action: pinch` on the originating tool call (was a silent no-op) |

<details>
<summary>Architecture</summary>

```
lite/gym/envs/mobilegym/
├── main.py                       # RemoteMobileGymEnv (thin RPC client) + manifest task registration + MobileGymContainerServices
├── data/tasks.json               # static task manifest (split/seed/difficulty/apps/max_steps/...); read on import, ZERO bench_env on the host
├── docker/
│   ├── Dockerfile                # self-contained: clone upstream @ pinned SHA → node stage builds dist/ → python stage + bench_env → bundle media tarball into dist/cdn/ (offline real media)
│   ├── server.py                 # in-container FastAPI RPC server: browser pool + full task lifecycle
│   └── entrypoint.sh             # serve dist/ on :4173 (internal) + uvicorn RPC on :8000
└── scripts/
    ├── install.sh                # docker build (build / rebuild / pull / status)
    ├── uninstall.sh              # docker image rm
    └── utils/tasks.sh  # dump data/tasks.json from the image (dev step; re-run when the pin changes)
```

**How it works:** MobileGym runs an Android-like OS in the browser (React + TypeScript). Inside the container, `entrypoint.sh` serves the prebuilt Vite `dist/` on the internal port 4173 and runs the FastAPI RPC server on 8000 (the only published port). Each `/reset` acquires a Playwright page on a shared browser context; the Python side drives actions through Playwright and reads state as JSON for deterministic evaluation. Only JSON scalars (a base64-encoded screenshot image, reward, flags, instruction) cross the host↔container wire; the host decodes it to raw `bytes` at ingress, so `obs.image` is always bytes.

**Concurrency model:** the container holds a Chromium pool — one browser per `contexts_per_browser` contexts (default 8), up to `max_browsers` (host-RAM-derived, forwarded via `-e`), with in-container idle cleanup. On pool saturation the RPC returns HTTP 503; the host maps it to `CapacityExhausted` (503 + Retry-After) so callers can retry or reduce concurrency.

**Updating the task manifest:** when the mobilegym pin in `docker/Dockerfile` changes, re-run `uv run --no-sync bash lite/gym/envs/mobilegym/scripts/utils/tasks.sh` (requires the built image) and commit the refreshed `data/tasks.json`.

</details>

## Launchable Apps Catalog

| Category | Apps |
|---|---|
| Social & Messaging | WeChat (微信), RedNote (小红书), X (Twitter), Reddit |
| Finance & Commerce | Alipay (支付宝), eBay |
| Media & Reading | Bilibili (哔哩哔哩), Spotify, WeChat Reading (微信读书) |
| Travel & Life | 12306 (铁路12306), Map, Tencent Meeting (腾讯会议), Weather |
| System | Settings, Phone, SMS, Notes, Calendar, Clock, Calculator, File Manager, Gallery, Browser, Compass |

Internal system surfaces such as Launcher and AnswerSheet exist in the simulator,
but are not part of the `open_app` catalog exposed to agents.

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@misc{wu2026mobilegym,
  title         = {MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research},
  author        = {Dingbang Wu and Rui Hao and Haiyang Wang and Shuzhe Wu and Han Xiao and Zhenghong Li and Bojiang Zhou and Zheng Ju and Zichen Liu and Lue Fan and Zhaoxiang Zhang},
  year          = {2026},
  eprint        = {2605.26114},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.26114}
}
```
