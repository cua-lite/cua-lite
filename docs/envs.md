# Environments (Gym)

CUA-Lite gives every Computer-Use task a Gymnasium-style env with an async
`reset()` / `step()` / `close()` loop.

- **[Using an environment](#using-an-environment)** — run / evaluate an agent on an existing env
- **[Adding an environment](#adding-an-environment)** — for contributors writing a new env

(Each section below opens with its own sub-contents.)

---

# Using an environment

**In this section:** [Quick Start](#quick-start) · [Core API](#core-api) · [Available environments](#available-environments) · [Installation](#installation) · [Env-server](#env-server)

## Quick Start

Start with `lite.demo` for a direct local smoke test. It has no env-specific
installer, but it uses the shared `cua-lite/sandbox.linux` desktop image.

**High-level agent loop** (what you'll normally use):

```bash
# one-time: build/pull the shared desktop sandbox image used by lite.demo
uv run --no-sync bash lite/gym/sandbox/scripts/install.sh

# --model-id auto-routes local (sglang/hf) vs API models; --head 1 runs one task.
# GPT API model: needs OPENAI_API_KEY; OPENAI_BASE_URL is optional for a custom endpoint.
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.demo --head 1
# Local model — needs a GPU (serves the model with sglang)
uv run python scripts/rollout.py --model-id Qwen/Qwen3-VL-8B-Instruct --env-id lite.demo --head 1
```

See `agent.sample(env)` in [README.md#quick-start](/README.md#quick-start) for the Python API.

> Everything runs **locally in-process by default**. Most envs need a one-time
> [`install.sh`](#installation) first; `lite.demo` only needs the shared sandbox
> image. You only need the [env-server](#env-server) for training, remote envs,
> or multi-tenant hosts.

**Low-level gym loop** (drive the env yourself — debugging, a custom harness):

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("lite.demo@create_file", max_steps=10, post_action_delay=0.5)
    obs = await env.reset()               # starts the backend, returns first screenshot/instruction
    print(obs.text)                       # task instruction

    result = await env.step([
        make_tool_call(
            "computer",
            {"actions": [
                {"action": "click", "coordinate": [500, 300]},
                {"action": "type", "text": "hello"},
            ]},
            call_id="call_0000",
        ),
    ])
    print(result.reward, result.terminated)
    await env.close()

asyncio.run(main())
```

Full kwargs, the `LiteBaseEnv` interface, and the action format are in [Core API](#core-api).

---

## Core API

### `gym.make(key, **kwargs)`

Creates an env instance. `key` is `"{env_id}@{task_id}"` (e.g. `"lite.demo@create_file"`).
The supported kwargs are env-specific; common ones are:

| kwarg | Meaning |
|---|---|
| `max_steps` | Override the env's default step horizon |
| `post_action_delay` | Seconds to wait after each action before re-taking the screenshot |
| `display_resolution` | Env-side render size for controllable renderers |
| `cursor` | Env-owned cursor compositing, only for cursor-capable envs |

Agent-side screenshot resizing is an agent/model setting, not a generic
`gym.make` kwarg.

### `LiteBaseEnv` interface

```python
from lite.core import LiteBaseMetadata, LiteToolCall      # protocol types — owned by lite.core
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult   # gym envelopes

class LiteBaseEnv:                                      # what every env implements
    async def reset(self) -> LiteEnvObservation: ...     # start the task, return first observation
    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult: ...   # execute actions
    async def close(self) -> None: ...                  # cleanup
    def _runtime_metadata(self) -> LiteBaseMetadata: ... # metadata_kind, dims, extra_tool_schemas, …
```

`LiteToolCall` / `LiteBaseMetadata` are protocol types owned by `lite.core`
(`lite.core.tools.calls` / `lite.core.metadata`) and are **not** importable from
`lite.gym.types`, which owns only the reset/step envelopes.

Callers read `env.metadata`; `gym.make` adds `env_id` / `task_id` under
`env.metadata.others`.

### Result & observation

`reset()` returns the first observation: screenshot/image, task text, and small
metadata. `step()` executes tool calls and returns per-call feedback, plus
`reward`, `terminated` / `truncated`, and `info`.

```python
class LiteEnvObservation:
    image: bytes | None           # screenshot bytes (None on text-only reset)
    text: str | None              # task instruction / reset text
    metadata: dict | None

class LiteEnvStepResult:
    results: list                 # per-call env feedback
    reward: float | None          # from the evaluator
    terminated: bool              # task ended (finish tool / eval says done)
    truncated: bool               # cut off by a limit (e.g. max steps)
    info: dict
```

### Action format

Each top-level action passed to `env.step()` is a canonical nested Lite tool
call. Build it with `make_tool_call(name, arguments, call_id=...)`; the call
`id` is the pairing key for the corresponding
`LiteToolResult.tool_call_id`.
Coordinates are normalized to `[0, 1000]`. Available actions depend on
`env.metadata.dims`, the metadata kind, and the tool schemas advertised by the
selected agent/env config. CUA metadata additionally exposes `.platform` as the
CUA action-surface selector:

| Platform | action-batch tool call | canonical actions |
|---|---|---|
| `desktop` | `computer` | `key`, `type`, `click`, `scroll`, `drag`, `mouse_move`, `screenshot`, `wait`, … |
| `mobile` | `mobile` | `tap`, `long_press`, `type`, `swipe`, `system_button`, `wait`, … |
| `browser` | `computer` | same as desktop; browser nav is exposed through [extra tools](#extra-tools) |

`response` / `terminate` finish calls and `open_app` app launch calls are
standalone extra tools. When an env exposes them, their schemas appear in
`env.metadata.extra_tool_schemas`.

### Extra tools

Some envs need actions beyond the standard space (e.g. WebGym's browser `back` /
`goto`, or finish tools such as `response` / `terminate`).

They're resolved into schemas on `env.metadata.extra_tool_schemas`; agents
created with `agents.make(..., env=env)` receive that metadata automatically.

```python
import asyncio
import lite.gym as gym

async def main():
    env = gym.make(
        "webgym@142359",
        max_steps=10,
        extra_tools=["back", "goto"],
    )
    await env.reset()
    print(env.metadata.extra_tool_schemas)   # schemas, not assistant tool_calls
    await env.close()

asyncio.run(main())
```

Which entries are surfaced is controlled by `env_kwargs.extra_tools`, **off by
default**. Enable names explicitly, for example
`env_kwargs: {extra_tools: ["back", "goto"]}`. The shared resolver
([`resolve_extra_tools`](/lite/gym/utils/feedback/surface.py)) is two-state:
omitted (`null`) and `[]` both mean *none*; a list means that subset.

<!-- **browsergym is the one deliberate exception, and it is tri-state.** Its
catalog is *derived* from the BrowserGym `action_subsets` a benchmark
registered, not a fixed menu, so "everything the action space offers" is the
only default that tracks a re-registered subset — and it is what keeps
`response` (→ `send_msg_to_user`) reachable, without which every WA/VWA
information-seeking task is structurally unanswerable. It therefore ships
`extra_tools: null` and keeps `None` distinct from `[]`
([`configs/default.yaml`](/lite/gym/envs/browsergym/configs/default.yaml),
[`_extra_tool_schemas_for_subsets`](/lite/gym/envs/browsergym/main.py)):

| `env_kwargs.extra_tools` | shared resolver | browsergym |
|---|---|---|
| omitted / `null` | none | the **whole** `action_subsets`-derived catalog |
| `[]` | none | none |
| `["name", ...]` | that subset | that subset |

Do not "normalize" browsergym's `null` to `[]`; the two are not the same value
there. -->

---

## Available environments

The task envs at a glance. **Render resolution** = the screenshot's pixel size,
either **controllable** (we own the renderer) or **fixed** by the backing
VM / viewport / pool / emulator.[^res]

[^res]: Controllable/configurable means set via env-specific `env_kwargs`
such as `display_resolution` or `viewport`; fixed envs generally do not support
those knobs and may reject them.

| Platform | Environment | Description | Tasks | Render resolution |
|---|---|---|---|---|
| **Desktop** | **lite.demo** | Simple demo tasks (file ops, settings, multi-step). No env-specific installer; uses the shared sandbox image. | 5 ([src](/lite/gym/envs/lite/demo/main.py)) | 1920×1080 — controllable |
| | **lite.osworld** | [OSWorld](https://github.com/xlang-ai/OSWorld) on the shared `cua-lite/sandbox.linux` GNOME Shell Docker base. 10 domains, eval + train. | 369 eval + 2429 train; train also exposed as `train.synth` (1722) + `train.perturb` (707) ([readme](/lite/gym/envs/lite/osworld/README.md)) | 1920×1080 — controllable |
| **Desktop / Browser** | **lite.cuagym** | [CUA-Gym](https://github.com/xlang-ai/CUA-Gym) upstream web/cross_app plus desktop-shaped tasks on the shared lite.osworld desktop substrate. | 10,910 pinned upstream tasks; task-level failures are isolated during rollout ([readme](/lite/gym/envs/lite/cuagym/README.md)) | 1920×1080 — upstream task contract |
| **Desktop** | **lite.cuaworld.\*** | 40 [gym-anything](https://github.com/cmu-l3/gym-anything) (CUAWorld) desktop apps on `cua-lite/lite.cuaworld.base`, ~25 domains. Multi-variant like browsergym; content in the `cua-lite/lite.cuaworld-assets` materials repo. | locked per-software train/eval/long-horizon catalogs; query the registry for current counts ([readme](/lite/gym/envs/lite/cuaworld/README.md)) | 1920×1080 — controllable |
| | **lite.scalecua** | [ScaleCUA](https://github.com/THUDM/SCALE-CUA) OSWorld-shaped *training* tasks on the shared lite.osworld desktop substrate (no image of its own). Train/RL splits only — use `lite.osworld` for the canonical `eval` split. | 20,289 train + 2,049 rl ([readme](/lite/gym/envs/lite/scalecua/README.md)) | 1920×1080 — controllable |
| | **osworld** | Official [OSWorld](https://github.com/xlang-ai/OSWorld) on a local VM-in-Docker (QEMU/KVM booting `Ubuntu.qcow2`); native evaluators. 10 domains, eval-only. | 369 ([readme](/lite/gym/envs/osworld/README.md)) | 1920×1080 — **fixed** (VM image) |
| | **osworld_2** | Official [OSWorld-V2](https://github.com/xlang-ai/OSWorld) (2.0) — same VM-in-Docker infra as v1, gated Python `BaseTask` tasks + v2 qcow2; native evaluators. 10 capabilities, eval-only (some tasks need an external website service + an `OPENAI_API_KEY` judge — see readme). | 108 ([readme](/lite/gym/envs/osworld_2/README.md)) | 1920×1080 — **fixed** (VM image) |
| | **waa** | Original [WindowsAgentArena](https://github.com/microsoft/WindowsAgentArena) tasks on a locally prepared Windows 11 QEMU VM; snapshot restore (automatic) for ~5x faster VM boot. | 154 eval + 154 no-context ([readme](/lite/gym/envs/waa/README.md)) | 1440×900 — **fixed** (prepared VM) |
| | **screenspot_pro** | Single-step click grounding ([ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)). No Docker. | 1581 ([readme](/lite/gym/envs/screenspot_pro/README.md)) | varies — **fixed** (static screenshots) |
| | **osworld_g** | Single-step click grounding ([OSWorld-G](https://arxiv.org/abs/2505.13227)), 3 box-type modes. No Docker. | 564 ([readme](/lite/gym/envs/osworld_g/README.md)) | varies — **fixed** (static screenshots) |
| | **cua.bench.\*** | [cua-bench](https://github.com/trycua/cua/tree/main/libs/cua-bench) tasks via the [Cua](https://github.com/trycua/cua) ecosystem. Shipped datasets register as `cua.bench.local.basic`, `.kicad`, and `.workflows`; `webtop` is used only for datasets that declare a webtop/simulated backend. | 68 + 25 + 52 ([readme](/lite/gym/envs/cua/README.md)) | 1920×1080 — **forced** (setup_config) |
| | **cua.sandbox** | Open-ended, instruction-driven Cua sandbox (any OS; local or cloud). Direct `CuaSandboxEnv` class, not a registered `gym.make` env; `reward=None`. | — ([readme](/lite/gym/envs/cua/README.md)) | **fixed** (image) |
| **Browser** | **webgym** | Web info-retrieval over a containerized browser pool. Docker. | 292k+ ([readme](/lite/gym/envs/webgym/README.md)) | 1280×720 — **fixed** (OmniBoxes pool) |
| | **webharbor.webvoyager** | [WebVoyager](https://github.com/MinorJerry/WebVoyager) tasks over WebHarbor self-hosted mirrors. 15 websites, coord + SoM action modes. | 643 ([readme](/lite/gym/envs/webharbor/webvoyager/README.md)) | 1280×720 default — configurable via `env_kwargs.viewport` |
| | **online_mind2web** | [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web) real-world web tasks across diverse domains. VLM judge (`o4-mini`). | 300 ([readme](/lite/gym/envs/online_mind2web/README.md)) | 1280×720 default — configurable via `env_kwargs.viewport` |
| | **captcha** | In-browser CAPTCHA challenges — 8 categories (text OCR, slider, rotation, icon-select, arithmetic, 3×3 image-grid, drag-match, carousel). Playwright/Flask, no Docker. | 603 (297 eval + 10 train + 296 test) ([readme](/lite/gym/envs/captcha/README.md)) | 1920×1080 — controllable |
| | **browsergym.miniwob** | [MiniWoB++](https://github.com/ServiceNow/BrowserGym) web tasks. | 125 ([readme](/lite/gym/envs/browsergym/README.md)) | 498×321 — **fixed** (task screenshot-pixel viewport) |
| | **browsergym.webarena** | [WebArena](https://github.com/ServiceNow/BrowserGym) real-world web tasks. Docker services + `WA_*` env vars. | 812 ([readme](/lite/gym/envs/browsergym/README.md)) | 1280×720 — **fixed** (task viewport) |
| | **browsergym.visualwebarena** | [VisualWebArena](https://github.com/ServiceNow/BrowserGym) multimodal web tasks. Docker services + `VWA_*` env vars. | 910 ([readme](/lite/gym/envs/browsergym/README.md)) | 1280×720 — **fixed** (task viewport) |
| **Mobile** | **androidworld** | [AndroidWorld](https://github.com/google-research/android_world) multi-step, 20 apps. Android SDK + KVM. | 232 (116 eval + 116 train) ([readme](/lite/gym/envs/androidworld/README.md)) | ~1080×2400 — **fixed** (emulator-native) |
| | **phoneworld** | [PhoneWorld](https://github.com/PhoneBuddyAI/PhoneWorld), 34 offline mock apps with answer/SQLite verification. Gated APK access + KVM. | 420 (120 eval + 300 train) ([readme](/lite/gym/envs/phoneworld/README.md)) | ~1080×2400 — **fixed** (emulator-native) |
| | **androidlab** | [AndroidLab](https://github.com/THUDM/Android-Lab) multi-step, 9 offline apps. Docker-per-worker emulator (KVM). | 138 ([readme](/lite/gym/envs/androidlab/README.md)) | ~739×1600 — **fixed** (Pixel-7-Pro, downscaled) |
| | **mobileworld** | [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld) multi-step, 20 apps, on a rooted Android emulator + self-hosted app backends in a self-contained Docker-in-Docker box. Needs `/dev/kvm`; runs `--privileged`. | 161 (201 upstream − 40 excluded `agent-mcp`) ([readme](/lite/gym/envs/mobileworld/README.md)) | ~1080×2400 — **fixed** (Pixel-8 AVD snapshot) |
| | **mobilegym** | [MobileGym](https://arxiv.org/abs/2605.26114) browser-simulated mobile, 24 apps. Node.js + Playwright in Docker; media dataset bundled into the image (~4.5 GB, real media offline). | 416 (256 eval + 160 train) ([readme](/lite/gym/envs/mobilegym/README.md)) | 1080×2400 — controllable |

---

## Installation

Envs install into the unified top-level venv. Create it once, then run the
target env's idempotent `install.sh` — each installer is the single source for
that env's python deps, docker images, and data.

```bash
uv sync --all-extras                                              # one-time: create the unified venv
uv run --no-sync bash <env-dir>/scripts/install.sh              # set up one env
```

The env id is not always the directory path: `lite.osworld` lives under
`lite/gym/envs/lite/osworld/`, `webharbor.webvoyager` under
`lite/gym/envs/webharbor/webvoyager/`, and Cua envs under
`lite/gym/envs/cua/`. Some installers also take a selector, such as
`lite/gym/envs/browsergym/scripts/install.sh <benchmark>` or
`lite/gym/envs/lite/cuaworld/scripts/install.sh build <software>`.

A later bare `uv sync` can evict ad-hoc env deps that `install.sh` adds (e.g.
webgym's host judge). If you re-sync, re-run the env's `install.sh` to re-add
them. Keep `--no-sync` on installer invocations; ordinary rollout/test
commands should use regular `uv run` unless the command specifically depends on
installer-added env deps.

Use the env README linked in the table below for the exact command.

<details>
<summary><b>Per-environment setup requirements</b></summary>

| Env | Needs |
|---|---|
| [androidworld](/lite/gym/envs/androidworld/README.md) | KVM (`/dev/kvm`) + `cua-lite/androidworld:latest` image (JDK + SDK + AVD + apps; built by `install.sh`) |
| [phoneworld](/lite/gym/envs/phoneworld/README.md) | KVM (`/dev/kvm`) + authorized Hugging Face gated APK access; image is built locally and must not be redistributed |
| [androidlab](/lite/gym/envs/androidlab/README.md) | KVM (`/dev/kvm`) + `cua-lite/androidlab:latest` image (needs docker-file.zip) |
| [mobileworld](/lite/gym/envs/mobileworld/README.md) | KVM (`/dev/kvm`) + privileged Docker-in-Docker runtime + `cua-lite/mobileworld:latest` image |
| [mobilegym](/lite/gym/envs/mobilegym/README.md) | `cua-lite/mobilegym:latest` image (built by `install.sh`) |
| [webgym](/lite/gym/envs/webgym/README.md) | `cua-lite/webgym:latest` image (built or pulled by `install.sh`); host judge deps installed by the same script; `OPENAI_API_KEY` for scored runs unless `skip_eval=True` |
| [webharbor.webvoyager](/lite/gym/envs/webharbor/webvoyager/README.md) | `cua-lite/webharbor.webvoyager:latest` image; `OPENAI_API_KEY` for scored runs unless `skip_eval=True` |
| [online_mind2web](/lite/gym/envs/online_mind2web/README.md) | `cua-lite/online_mind2web:latest` image (built by `install.sh`); `OPENAI_API_KEY` for VLM judge unless `skip_eval=True` |
| [captcha](/lite/gym/envs/captcha/README.md) | host Python deps, Playwright Chromium/system deps, and CAPTCHA assets; each episode starts its scoped Flask server + Chromium |
| [browsergym](/lite/gym/envs/browsergym/README.md) | run `install.sh <benchmark>` for each benchmark; installs host BrowserGym/Playwright deps; WebArena / VisualWebArena need Docker services and `OPENAI_API_KEY` |
| [lite.demo](/lite/gym/envs/lite/demo/README.md) | no env-local scripts; uses the shared `cua-lite/sandbox.linux:latest` image from the sandbox installer |
| [lite.osworld](/lite/gym/envs/lite/osworld/README.md) | `cua-lite/lite.osworld:latest` image + host deps + pinned catalogs/assets |
| [lite.cuagym](/lite/gym/envs/lite/cuagym/README.md) | `cua-lite/lite.cuagym:latest` image + pinned HF mirror `cua-lite/lite.cuagym-assets`; judge credentials as described in the env README |
| [lite.cuaworld](/lite/gym/envs/lite/cuaworld/README.md) | per-software `cua-lite/lite.cuaworld.<software>:latest` image (built by `lite/gym/envs/lite/cuaworld/scripts/install.sh build <software>`); public HF materials repo `cua-lite/lite.cuaworld-assets`; VLM judge credentials in the env-server/evaluator process |
| [lite.scalecua](/lite/gym/envs/lite/scalecua/README.md) | no image of its own — runs on `cua-lite/lite.osworld:latest`; `install.sh` delegates to the lite.osworld installer, then imports the pinned ScaleCUA catalogs + judge overlays |
| [osworld_g](/lite/gym/envs/osworld_g/README.md) | `install.sh` clones the OSWorld-G benchmark data |
| [screenspot_pro](/lite/gym/envs/screenspot_pro/README.md) | `install.sh` pre-downloads the dataset from HuggingFace |
| [osworld](/lite/gym/envs/osworld/README.md) | KVM (`/dev/kvm` + `/dev/net/tun`) + the derived `cua-lite/osworld` image (built by install.sh, `FROM happysixd/osworld-docker`) + `Ubuntu.qcow2` (provisioned to `.cache/`) |
| [osworld_2](/lite/gym/envs/osworld_2/README.md) | KVM (`/dev/kvm` + `/dev/net/tun`) + local OSWorld-V2 source (`OSWORLD_V2_SRC` or `_vendor/OSWorld-V2`) + the derived `cua-lite/osworld_2` image + **HF-gated** v2 qcow2 & task classes (`hf auth login` + accept gates); `OPENAI_API_KEY` for the ~18 LLM-judge evaluators, with optional `OPENAI_BASE_URL` only for a custom endpoint |
| [waa](/lite/gym/envs/waa/README.md) | Linux x86_64 + Docker + KVM + enough RAM/disk for the Windows VM; runner `cua-lite/waa:latest` + prepared WAA qcow2 + pinned Windows 11 evaluation ISO + task-asset cache |
| [cua.bench / cua.sandbox](/lite/gym/envs/cua/README.md) | `install.sh` pip-installs `cua-sandbox` + `cua-bench`, installs Playwright Chromium, downloads CUA-bench datasets, and pulls `trycua/cua-xfce` for `cua.bench.local.*` |

</details>

---

## Env-server

Envs run one of two ways — **direct** (in-process, the default) or behind an
**env-server** reached over HTTP. To just point a run at an env-server, follow
[Using an env-server](#using-an-env-server); [How routing works](#how-routing-works)
explains the switch behind it.

### Using an env-server

First, start an env-server (skip if one is already running for you) on a host that
has the env's deps installed (per its README):

```bash
uv run python scripts/serve_env.py --port 30100
```

Then point the agent/training side at an address it can reach. If you just
started the server on this machine, prefer a reachable machine IP or DNS name
over `localhost` so the same URL also works from a Slime/Docker container:

```bash
export CUA_LITE_ENV_SERVER_URL=http://$(hostname -I | awk '{print $1}'):30100
# On multi-interface hosts, replace that IP if it picks a Docker bridge/VPN
# address the client container cannot reach.
# Default server auth is passthrough: any bearer namespaces your instances.
# If the server was started with --token, use the operator-provided token.
export CUA_LITE_ENV_SERVER_TOKEN=$(whoami)
export ENV_ID=lite.osworld

# (optional preflight) check the server + your env are ready (expect "available": true):
# curl -sf -H "Authorization: Bearer $CUA_LITE_ENV_SERVER_TOKEN" "$CUA_LITE_ENV_SERVER_URL/envs/$ENV_ID"

# run the rollout unchanged — it routes to the server because CUA_LITE_ENV_SERVER_URL is set:
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id "$ENV_ID" --head 1
```

For an already-running remote server, replace the URL with
`http://<env-server-host>:<port>`. Avoid `localhost`/`127.0.0.1` unless the
client process is in the same network namespace as `serve_env.py`; inside a
container, `localhost` usually points at the container itself, not the env-server
host. Use an env that is installed on that server; custom deployments may still
restrict the served env ids.

Catalog probes report blocked or missing-deps envs as `available: false`
rather than taking the server down, so other served envs stay up. Task listing
or instance creation for such an env fails terminally for that deployment.

<details>
<summary><b>Warming shared backends to cut startup latency (optional)</b></summary>

One optional knob can pre-pay slow shared-service startup:

- **`--warm-singleton`** is a **`scripts/serve_env.py`** flag (not a
  `scripts/rollout.py` one): pass it when you start the env-server —
  `uv run python scripts/serve_env.py --warm-singleton` — and it warms
  shared-backend envs in the background. Use it for
  BrowserGym/WebArena/VisualWebArena-style stacks before large evals.

DEDICATED VM/emulator/container envs still benefit from env-server admission,
ownership cleanup, retry/recovery, and any env-native snapshot restore, but
each instance is constructed for its own run.

**osworld_2 website service:** OSWorld-2 uses `web.hku.icu` by default for tasks
that need the companion website. `--warm-singleton` does not warm that external
service. To self-host it, run the OSWorld-web stack and set
`website_host_suffix` as described in the env README.

</details>

<details>
<summary><b>Auditing a server's resolved config (optional)</b></summary>

`GET /envs/<env_id>/tasks` includes a `server_config` block:
redacted `server_kwargs`, the resolved config-file provenance, API-key/base-url
availability booleans from the env-server process, build info, and env-server
runtime knobs such as port/auth/allow-list/reset concurrency. This block is
separate from the task `kwargs` / `env_make_kwargs` carried to remote clients and
from per-instance `env_kwargs` shown on `/instances` rows; secret values are not
returned.

</details>

### How routing works

For rollout and registry task discovery, `CUA_LITE_ENV_SERVER_URL` is the
user-facing switch: **unset → direct local env construction; set → task lists
and env instances come from that server.** Low-level Python callers with a known
task key can also pass `env_server_url=...` directly to `gym.make`.

<!-- <details>
<summary><b>Routing table: same env API in both modes (details)</b></summary>

For client processes, the env var routes both registry lookup and `gym.make`.
Inside `serve_env.py`, the server forces direct mode for its own registry and
`gym.make` calls even if the variable is inherited.

| | **Direct** (default) | **Remote** (URL set) |
|---|---|---|
| `gym.make` returns | a real `LiteBaseEnv` in-process | a `LiteEnvClient` HTTP proxy |
| docker / emulator lifecycle | the agent process | [`serve_env.py`](/scripts/serve_env.py) |
| env-specific deps on the model node | installed locally | none; still needs network plus normal model/API deps |

Use remote for **Slime training**, multi-tenant env hosts, heavy-env/light-agent
splits, or crash isolation. Restart the env-server after changing task
registration or metadata used by rollout filters; remote clients see the
server's running copy.

</details> -->

---

# Adding an environment

> For contributors writing a new env. Skip this if you're only *using* one — the
> sections above are all you need.

**In this section:** [Architecture](#architecture) · [Python files](#python-files) · [Lifecycle scripts](#lifecycle-scripts) · [Config](#config) · [Backends](#backends)

## Architecture

Only the files an env author touches (everything else is framework-internal):

```
lite/gym/
├── base.py         # LiteBaseEnv — the caller contract you implement (reset/step/close/_runtime_metadata)
├── services.py     # opt-in server capabilities: EnvServerResource / EnvServerPoolable / EnvServices — and register_services()
├── registry.py     # register() — how your env + tasks get discovered
├── types.py        # gym-owned envelopes only: observation, step result, executed-action trace (LiteToolCall / LiteBaseMetadata live in lite.core)
├── errors.py       # CapacityExhausted — raise from reset() when a bounded resource is full
├── …               # framework internals (you don't touch these)
└── envs/<name>/           # YOUR env goes here — fixed layout:
    ├── main.py            #   REQUIRED — env class(es) + register() / register_services() (auto-discovered)
    ├── configs/
    │   └── default.yaml   #   tunable defaults (env_kwargs + server_kwargs + make_kwargs) — see Config below
    ├── data/              #   small committed manifests / asset locks; large generated caches stay ignored
    ├── README.md          #   setup + task notes (expected for any env with deps)
    ├── scripts/           #   install/uninstall/start/cleanup.sh (optional — see Lifecycle scripts)
    ├── utils/             #   env-side helper modules for simple envs
    ├── src/               #   optional runtime package for larger/multi-backend envs
    └── …
```

Use either `utils/` or `src/` for env-owned runtime helpers; do not put agent
protocols/adapters there. Agent-side integrations belong in
`/lite/agents/extensions/<env>/`.

---

## Python files

Every env is a class in `lite/gym/envs/<name>/main.py` (auto-discovered — no
manual import) plus its task registrations. A dotted name (`lite.osworld`) is a
variant under a shared parent module; underscores otherwise.

**The base case is the whole contract:** subclass [`LiteBaseEnv`](/lite/gym/base.py)
and register the tasks.

```python
from lite.gym.base import LiteBaseEnv
from lite.gym.registry import register
from lite.core.metadata import LiteBaseMetadata, LiteCUAMetadata

class MyEnv(LiteBaseEnv):
    def __init__(self, **kwargs): ...
    async def reset(self):  ...           # -> LiteEnvObservation
    async def step(self, actions): ...    # -> LiteEnvStepResult
    async def close(self): ...

    @staticmethod
    def _task_metadata() -> LiteBaseMetadata:  # the ONE metadata builder — registration
        return LiteCUAMetadata(dims=("desktop", "use"))  # and the live instance both call it

    def _runtime_metadata(self) -> LiteBaseMetadata:   # builder + env_kwargs amendments (none here)
        return self._task_metadata()

register("my_env@task_1", entry_point=lambda **kw: MyEnv(**kw), split="train",
         metadata=MyEnv._task_metadata())
register("my_env@task_2", entry_point=lambda **kw: MyEnv(**kw), split="eval",
         metadata=MyEnv._task_metadata())
```

Registered metadata and live `env.metadata` should come from the same builder.
Live metadata may only differ where `_runtime_metadata` applies explicit
`env_kwargs`. Do not override the `metadata` property or write identity
yourself; the framework injects `env_id` / `task_id` into `others`. Copy mutable fields before
returning them.

`gym.make("my_env@task_1")` now works in-process — and **already works behind the
env-server** if the env is stateless or its backend lives elsewhere (a remote VM, a
shared pool). That is a complete env; a pure-Python or remote-backed env stops here.

An env that owns infrastructure — a Docker simulator, a per-instance resource, a
slow backend needing lifecycle management — adds **opt-in capabilities** from
[`services.py`](/lite/gym/services.py) (`EnvServices` / `EnvServerResource` /
`EnvServerPoolable`) onto this *same* class; the base is never rewritten.
[Backends](#backends) shows each shape concretely, anchored to a real env.

---

## Lifecycle scripts

Each env's `scripts/` dir holds **env-native** shell scripts managing that env's
full lifecycle, independent of how it's used (train/eval/rollout/debug). They run
on the **env node** (where `serve_env.py` lives) or any dev host with the env's
Python deps — **never** in the slime container (no env deps there; RL training
reaches the env-server over HTTP instead — see [docs/slime.md](/docs/slime.md)).

**The four scripts** (all optional; no empty stubs):

| Script | Purpose | Cadence | Example |
|---|---|---|---|
| `install.sh` | One-time install of deps / images / heavy resources | Once per host | Build `cua-lite/androidlab:latest` |
| `uninstall.sh` | One-time teardown of installed resources | Once per host | Delete JDK/SDK/AVD |
| `start.sh` | Lightweight per-rollout startup (background server) | Per rollout | Boot a browsergym WebArena/VWA service |
| `cleanup.sh` | Lightweight per-run cleanup | Per run | pkill stale emulators, `docker rm` session containers |

You normally run `install.sh` once per host (`uninstall.sh` to wipe). `start.sh`
and `cleanup.sh` are invoked automatically where an env needs them. When the
caller provides the env-server/session labels, cleanup is scoped to that owner;
manual cleanup with no session may intentionally sweep all matching resources
for that env on the daemon.

<details>
<summary><b>When to create each script (decision guide) + portability rules</b></summary>

| Script | Create when | Skip when |
|---|---|---|
| `install.sh` | You download/build resources (JDK, images, weights, apps) or install language deps a fresh machine lacks | (almost always present) |
| `start.sh` | A long-running background service must be up before rollouts, or you open host ports / mount FUSE | Per-task lazy Docker (lite.osworld) or a per-worker pool (androidlab) handles startup |
| `cleanup.sh` | You leave processes/lock-files/containers that could disturb future runs, or need session-scoped GC (`lite-env-${SESSION_ID}-<env>-*`) | Stateless envs |
| `uninstall.sh` | `install.sh` pulls multi-GB of irreversible state users may want to wipe | Rare — most envs skip it |

For portability, scripts must work on any host with the env's Python deps: use
bare `python` (the caller's installer command injects the venv) and never reference
training-only concepts (`$RAY_*`, slime paths).

</details>

### Fleet router (multi-node)

[`scripts/serve_fleet.py`](/scripts/serve_fleet.py) puts one client-facing
endpoint in front of multiple env-server nodes. Point
`CUA_LITE_ENV_SERVER_URL` at the router; rollout commands stay the same.

```bash
# nodes.txt: one node per line (host:port or full URL; # comments; the file is
# re-read on mtime change each poll, so appending a line hot-joins a node)
uv run python scripts/serve_fleet.py --port 30300 --nodes-file nodes.txt --node-token <tok>
```

The router sends new instances to healthy nodes and forwards each instance call
to its owner. Each node owns its own containers and shared services. Use
per-node admin URLs for node-specific operations.

### Which scripts each env provides

Empty = file does not exist (no stubs).

<details>
<summary><b>Which scripts each env provides (matrix)</b></summary>

| Env | install | uninstall | start | cleanup | Notes |
|---|:-:|:-:|:-:|:-:|---|
| `androidworld` | ✓ | ✓ | — | ✓ | container-per-worker; install builds the image |
| `phoneworld` | ✓ | ✓ | — | ✓ | fresh container per episode; gated image is local-only |
| `androidlab` | ✓ | ✓ | — | ✓ | emulator container per worker; install builds the image |
| `webgym` | ✓ | ✓ | — | — | container image plus host judge deps; install builds the OmniBoxes image |
| `webharbor.webvoyager` | ✓ | ✓ | — | — | container-only; includes WebHarbor mirrors |
| `online_mind2web` | ✓ | ✓ | — | — | container-only; install builds the Playwright browser image |
| `browsergym` | ✓ | ✓ | ✓ | ✓ | all take a `<benchmark>` arg |
| `mobilegym` | ✓ | ✓ | — | — | container-only; install builds the Node+Playwright image |
| `lite.osworld` | ✓ | — | — | ✓ | installs deps + builds image |
| `lite.cuagym` | ✓ | — | — | ✓ | one `cua-lite/lite.cuagym` image for upstream web/cross_app + desktop-shaped CUA-Gym tasks; install syncs upstream task bundles and bakes referenced mocks |
| `lite.cuaworld` | ✓ | ✓ | — | ✓ | all take a `<software>` arg; install builds `cua-lite/lite.cuaworld.<software>` from materials; no `start.sh` (per-task lazy Docker via Sandbox services) |
| `lite.scalecua` | ✓ | — | — | — | catalog-only env on the `lite.osworld` image; install delegates to lite.osworld's installer, then imports the pinned ScaleCUA task catalogs; utils/tasks.sh regenerates them |
| `mobileworld` | ✓ | ✓ | — | ✓ | DinD benchmark boxes; install pulls the upstream base + overlays pinned source; tasks.sh exports data/tasks.json |
| `osworld` | ✓ | ✓ | — | ✓ | DEDICATED VM-in-Docker; eval-in-container (host has no `desktop_env`); install docker-builds the derived image + provisions qcow2; cleanup removes `-osworld-` containers |
| `osworld_2` | ✓ | ✓ | — | ✓ | DEDICATED VM-in-Docker (OSWorld-V2); eval-in-container; install docker-builds the derived image + 2 HF-gated downloads (v2 qcow2 + task classes); cleanup removes `-osworld_2-` containers |
| `waa` | ✓ | ✓ | — | ✓ | local Windows QEMU container per trajectory; install prepares (or pulls) the qcow2 and asset cache |
| `osworld_g` | ✓ | ✓ | — | — | install clones benchmark data |
| `screenspot_pro` | ✓ | ✓ | — | — | install pre-downloads the dataset |
| `cua` | ✓ | ✓ | — | ✓ | install pip-installs `cua-sandbox`+`cua-bench` + pulls `trycua/cua-xfce` + downloads datasets to `.cache`; cleanup sweeps `cua.bench.local` containers + image-tags; uninstall drops `.cache`. webtop is in-process, `local` is a cleanup-managed cua-xfce container |
| `captcha` | ✓ | ✓ | — | ✓ | per-episode Flask servers; cleanup removes leaked Flask/Chromium procs + `/tmp` files |
| `lite.demo` | — | — | — | — | no env-local scripts; uses the shared sandbox image |

</details>

### Invocation

Run on the env node (or any dev host with the env's deps), from a checkout with `.venv`:

```bash
uv run --no-sync bash <env-dir>/scripts/install.sh
```

`uv run --no-sync bash` is deliberate. The one-time repo bootstrap is
`uv sync --all-extras`; env installers then run inside that already-created
venv without letting `uv run` resync, install, or remove project dependencies as
a side effect. `uv run --no-sync bash` prepends `.venv/bin` to `PATH`, so
`python` inside the script and its subprocesses resolves to the venv. Do **not**
use `uv run python <script.py>` or
`source .venv/bin/activate && bash <script>`; `uv run --no-sync bash <script>`
is the only supported installer form. Each env's README has its concrete
commands.

### Idempotency & reproducibility contract

Two guarantees, two sides of one coin (a script that only works with leftover
state from a prior run violates reproducibility):

- **Idempotent** on the same machine — running twice never breaks anything.
  `install.sh` skips work that is already complete; `cleanup.sh`/`uninstall.sh`
  exit 0 when there's nothing to clean.
- **Reproducible** on a fresh machine — following the env's README from a clean
  checkout reaches the same state, with fail-fast README-pointing errors when a
  genuinely external prerequisite (KVM, docker-file.zip, remote creds) is missing.

---

## Config

Put normal tunable defaults in `configs/default.yaml`, then load them once in
`main.py`. Keep env vars for secrets, runtime-allocated values, or explicit
escape hatches documented by the env.

```python
from pathlib import Path

from lite.gym.utils import config
CFG = config.load(str(Path(__file__).parent))   # -> EnvConfig(env_kwargs, server_kwargs, make_kwargs)
_MAX_STEPS = CFG.env_kwargs["max_steps"]
_MEMORY    = CFG.server_kwargs["memory_limit"]
```

<details>
<summary><b>Config house rules (standardized patterns)</b></summary>

`configs/default.yaml` needs an `env_var_prefix`. Every default lands in
exactly one place — pick it by who sets the value and when:

| Lives in | What | Overridable by |
|---|---|---|
| `env_kwargs` (yaml) | per-instance tunables, fed to the factory / `bind()` | a rollout's / grpo `env_kwargs`, per run |
| `server_kwargs` (yaml) | per-deploy infra (pool sizes, memory, boot timeouts) | deploy time only |
| `make_kwargs` (yaml) | env-wide `gym.make` defaults; wrapper-owned keys are consumed by `make()`, carried env-owned keys such as `cursor` flow to the env | per-task `register(...)`, then `gym.make` / rollout kwargs |
| `register(...)` hardcode | values that vary per task / benchmark / split | — |
| task data | per-task facts (jsonl / HF / `tasks.json`) | — |
| code literal | a structural constant, not a knob | — |
| env-var | secrets, runtime-allocated values, documented escape hatches (`WEBGYM_API_KEY`) | the environment |

Override the whole yaml (direct or server mode) with `<PREFIX>_CONFIG` — an
absolute path or a name under `configs/` (`BROWSERGYM_CONFIG=isolation` →
`configs/isolation.yaml`). It selects a complete replacement file; keys from
`default.yaml` are not carried over. For `env_kwargs` / `server_kwargs`
defaults, precedence is selected config file < `overrides[name]`
(multi-variant envs only, e.g. cua.bench per-dataset — via `for_override`) <
a rollout's `env_kwargs`.
For `make_kwargs`, `gym.make` resolves `make()`'s built-in defaults < yaml
`make_kwargs` < per-task `register(...)` kwargs < per-call `gym.make` / rollout kwargs.
The `overrides[name]` and rollout-`env_kwargs` layers **deep-merge
per-leaf** (a nested override, e.g. `computer.image`, wins only at the named
leaf and keeps its siblings); `<PREFIX>_CONFIG` alone replaces the whole file.
Resolver:
[`defaults.py`](/lite/gym/utils/config/defaults.py).
In server mode, the resolved config source and redacted `server_kwargs` are
reported under `/envs/<env_id>/tasks.server_config`; per-instance `env_kwargs`
remain on live `/instances` rows, and carried `make_kwargs` remain in the
`kwargs` / `env_make_kwargs` task payload.

A few house rules keep config identical across envs:

- An `__init__`/`bind` default should come from `CFG.env_kwargs` /
  `CFG.server_kwargs`, not from a second bare literal.
- `make_kwargs` contains both wrapper-owned keys and env-owned carried keys.
  Use it for env-wide defaults that should be visible to `gym.make`; keep
  per-instance env behavior in `env_kwargs`.
- A sentinel means "derive at runtime": `max_steps: null` → per-task, `seed: null` → unseeded,
  `extra_tools: []` → none, a service `port:` / `url: ""` / `instances: 0` → keep-if-free / auto-size.
  The one sanctioned exception is browsergym, whose `extra_tools` is **tri-state**: `null` means the
  whole `action_subsets`-derived catalog, `[]` means none, `[names]` means that subset. See
  [Extra tools](#extra-tools) — its `null` must not be normalized to `[]`.
- One word per concept: `extra_tools` = tool names (the knob), `extra_tool_schemas` = the
  `LiteBaseMetadata` field; a constant mirrors its yaml key (`display_resolution` → `_DISPLAY_RESOLUTION`).
- Ports: a shared-backend env takes a *preferred* yaml port (auto-reallocated if busy); a
  per-trajectory container always auto-allocates.
- Order by audience everywhere (yaml, constants, signature): identity → user tunables → internal.

</details>

---

## Backends

A pure-Python env is done at [Python files](#python-files). An env that owns a
**backend** — a container, a VM, an emulator, or an external service — declares
its backend family and, when needed, registers env-server services. Find your
row, copy the named env, read its `main.py`:

| Backend | When | Env-server integration | Start from |
|---|---|---|---|
| [Per-trajectory container](#per-trajectory-container-dedicated) (DEDICATED) | a fresh resource per trajectory; stateful sims that can't be shared | `EnvServerResource` on the env when it owns a resource id; `EnvServices` / usually `ContainerServices`; `register_family(..., BackendFamily.DEDICATED)` | `lite.osworld` · `waa` · `androidworld` · `phoneworld` · `cua.bench.local.<dataset>` · `osworld` · `osworld_2` |
| [Shared-backend container](#shared-backend-container-singleton) (SINGLETON) | one long-lived backend shared by every instance; heavy sims | `EnvServices` / often `SingletonContainerServices`; `register_family(..., BackendFamily.SINGLETON)` | `webgym` · `mobilegym` · `webharbor.webvoyager` · `online_mind2web` |
| [Desktop task](#desktop-task-sandbox) (Sandbox) | a managed desktop container; you write only setup + scoring | `register_tasks` plus DEDICATED services/family registration | `lite.demo` · `lite.cuaworld` |
| [Externally managed backend](#externally-managed-backend-no-local-container) | backend runs outside CUA-Lite; no local image | no local lifecycle services; `register_family(..., BackendFamily.REMOTE)` | external VM/service-backed envs |
| [Pure / in-process](/lite/gym/services.py) (PURE) | no backend the env-server manages | `register_family(..., BackendFamily.PURE)`; optional `EnvServices` for lazy catalogs/health | `screenspot_pro` · `osworld_g` · `cua.bench.webtop.<dataset>` |

> **Multi-mode env (cua.bench).** One upstream benchmark can span families, so
> cua.bench splits by env id: `cua.bench.webtop.<dataset>` is in-process,
> while `cua.bench.local.<dataset>` owns a local desktop container. See
> [the Cua env README](/lite/gym/envs/cua/README.md).

### Container

Both container backends (A & B) build their image the same way — a `docker/` dir keeps it
self-contained and fork-free:

```
lite/gym/envs/<name>/docker/
├── Dockerfile   # clones upstream @ a pinned SHA, applies patches/, pip installs — all at build time
└── patches/     # every cua-lite change to the upstream source, one file each (reviewable in-repo)
```

`install.sh build` builds it (or skips when the image is already fresh; use
`rebuild` after editing the Dockerfile or a patch), `uninstall.sh` removes it.
Pass container runtime config explicitly, usually through env vars. If the task
list lives inside the upstream package, commit a `data/tasks.json` dumped by an
env-local script rather than importing it on the host.

They differ only in container lifetime:

#### Per-trajectory container (DEDICATED)

A fresh container per trajectory, torn down at the end. Use this shape for
stateful simulators that cannot safely share a live backend across unrelated
tasks. The env exposes the resource id the env-server should clean up. The
snippet below shows only the env-server additions; keep the normal `LiteBaseEnv`
methods from the base case.

```python
from lite.gym.base import LiteBaseEnv
from lite.gym.remote.reaper import ContainerServices
from lite.gym.services import (
    BackendFamily,
    EnvServerResource,
    register_family,
    register_services,
)

class MyEnv(LiteBaseEnv, EnvServerResource):
    @property
    def external_resource_id(self) -> str | None:
        return self._container_name

class MyEnvServices(ContainerServices):
    def ensure(self, env_id: str) -> None: ...

register_services("my_env", MyEnvServices())
register_family("my_env", BackendFamily.DEDICATED)
```

If acquiring the container can fail (a slot, a port, a quota), raise
`CapacityExhausted(what=..., retry_after_s=...)` from `reset()`; the server returns 503 +
`Retry-After` and the client retries. Add a `cleanup.sh` if a crash could leak containers.

##### Container handles and backend-shape metadata

For container lifecycle code, start from an existing env instead of copying
framework internals. Env classes with immutable backend-shape kwargs and mutable
task kwargs can use `EnvServerPoolable` with explicit construction plus `bind`:

```python
from lite.gym.services import EnvServerPoolable, EnvServerResource


class MyEnv(EnvServerPoolable, EnvServerResource):
    def __init__(self, *, image: str = "cua-lite/my_env:latest",
                 task=None, max_steps: int = 30): ...
    def bind(self, task=None, *, max_steps: int = 30): ...
    async def boot(self): ...
    @property
    def external_resource_id(self) -> str | None: ...
```

Keep resource-shape kwargs in `__init__()` and task-specific state in `bind()`;
constructors should call `bind()` exactly once after constructor-state setup.

#### Shared-backend container (SINGLETON)

One long-lived container per env-server, shared by every instance. Use this
shape when a backend is expensive to boot but safe to share across tasks. Declare
the service, register the family, and provide a health check:

```python
from lite.gym.remote.reaper import SingletonContainerServices
from lite.gym.services import BackendFamily, register_family, register_services
from lite.gym.utils.config.naming import container_name as singleton_container_name

class MyEnvServices(SingletonContainerServices):
    rm_label = "my_env"

    def container_name(self, scope) -> str:
        return singleton_container_name("my_env", scope.server_port)

    def ensure(self, env_id: str) -> None: ...
    def health(self, env_id: str) -> None: ...

register_services("my_env", MyEnvServices())
register_family("my_env", BackendFamily.SINGLETON)
```

Do not use a shared-backend container for tasks that require per-instance
state isolation. Use a per-trajectory or other DEDICATED env instead.

#### Image build and freshness

`install.sh` builds or adopts an image that matches the checked-out sources.
If an image is missing or stale, env startup fails with an install-oriented
error instead of silently using the wrong build. Use `install.sh status` for
diagnostics.

<details>
<summary><b>Distribution</b> — build locally or pull from GHCR</summary>

Local name is registry-agnostic (`cua-lite/<env_id>:latest`); the remote is that + a `ghcr.io/`
prefix. The `lite.src_hash` **label** rides inside the image, so a pulled image is adopted only
if its label matches your local sources:

| `install.sh` | what it does |
|---|---|
| `provision` | set up non-Docker host resources first: pinned assets and catalogs, qcow2/task-class caches, host verifier deps, or other env-specific prerequisites |
| `build` (default) | build → `cua-lite/<env_id>:latest`; skips if a fresh local image exists (from a prior build **or** pull — checked on the label, not provenance) |
| `pull` | read the remote label without downloading layers (`docker buildx imagetools inspect`); if it matches, `docker pull` + retag → local name; else refuse, point to `build` |
| `status` | read-only state check; reports image freshness/missingness plus env-specific provisioned resources, and exits 0 for normal diagnostics |
| `health` | optional env-specific temporary-container probe where supported |

Tags stay `:latest`; the src-hash is the **label**, not the tag (no per-build tag sprawl).
Install helpers live in [`image_build.sh`](/lite/gym/scripts/image_build.sh).
`lite.cuaworld` uses this lifecycle per software: `build <software>` stamps both
the shared base and `cua-lite/lite.cuaworld.<software>:latest`.

**Adding a container env:** (1) add a lightweight env-local `image_spec.py`
with `image_for(env_id) -> ContainerImage` at
`lite/gym/envs/<env path>/image_spec.py`; use
`lite/gym/envs/<env path>/src/image_spec.py` only when one provider owns a
family of prefix variants. Keep it importable before provision/build and do not
import the env runtime from it; (2) `--label "$(src_label <env_id>)"` on the build line; (3) pass
`image_for(env_id)` to `require_image_present`. Config-driven tags (`android_*`) →
`image_for(env_id, tag=…)` (hash is tag-independent). `osworld`/`osworld_2` build their own
derived image (`FROM happysixd/osworld-docker`) and read the tag from YAML (`server_kwargs.image`),
so their preflight passes `image_for(<env_id>, tag=_IMAGE)` — same freshness check as every other
built env (their `install.sh` sources `image_build.sh` and stamps `--label "$(src_label …)"`).

</details>

### Desktop task (Sandbox)

A Docker desktop task that needs only setup + scoring — no custom `reset`/`step`/`close`. The
Sandbox helper wraps a DEDICATED env for you; you write two functions and register:

```python
from lite.gym.sandbox import SandboxTaskConfig, register_tasks

async def setup_fn(task, computer) -> None:               # after the container starts
    await computer.interface.run_command("firefox https://example.com &")

async def evaluate_final_fn(task, computer) -> float:     # score 0.0–1.0
    out = await computer.interface.run_command("cat /tmp/output.txt")
    return 1.0 if "expected" in out.stdout else 0.0

register_tasks("my_env", {"train": [SandboxTaskConfig(
    task_id="my_task", instruction="Do something on the desktop",
    computer={"image": "cua-lite/sandbox.linux:latest", "memory": "4GB"},
    max_steps=20, setup_fn=setup_fn, evaluate_final_fn=evaluate_final_fn,
)]})
```

For env-server use, Sandbox envs still register DEDICATED services/family; use
`lite.demo` or `lite.cuaworld` as the reference.

**Multi-variant Sandbox with external materials (`lite.cuaworld`).** When one
directory hosts many near-identical Sandbox variants, register them from a
factory in one `main.py` and let `install.sh <verb> <variant>` select the
variant. `lite.cuaworld` is the reference for using an external materials repo
instead of vendoring large per-software assets in-tree.

### Externally managed backend (no local container)

The backend runs elsewhere — a VM pool or service CUA-Lite does not manage.
`MyEnv` is a thin client with no local image and no local lifecycle services.
Declare `BackendFamily.REMOTE`; the env can still run direct or behind the
env-server like any other env. A `cleanup.sh` may ask the external service to
release resources.
