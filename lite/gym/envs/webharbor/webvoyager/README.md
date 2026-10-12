# WebVoyager

`--env-id` `webharbor.webvoyager`

CUA-Lite wrapper for [WebVoyager](https://github.com/MinorJerry/WebVoyager). 643 task queries across 15 websites (Allrecipes, Amazon, Apple, ArXiv, BBC News, Booking, GitHub, Google Flights, Google Maps, Google Search, Hugging Face, Wolfram Alpha, Cambridge Dictionary, Coursera, ESPN), each with 40+ queries, via `gym.make("webharbor.webvoyager@<task_id>")` with `LiteBrowserActionSpace`. See [docs/envs.md](/docs/envs.md) for the env contract.

## Setup

```bash
# Choose one install path:
# Source path: build cua-lite/webharbor.webvoyager:latest if missing/stale.
# The build pins WebVoyager/WebHarbor sources and fetches HF-managed assets.
uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image.
# uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh status    # image freshness/resources
# uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh health    # temporary container health probe
# uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh rebuild   # force a fresh rebuild
```

The container resets all site databases to their image-baked baseline at boot, then serves site mirrors internally on `127.0.0.1:40000-40014`; only the RPC port is exposed to the host.

Set a judge API key for scored runs — without one, `reset()` fails loud. For
plumbing checks that do not score, pass `skip_eval=True` / `--env-kwargs
'{"skip_eval": true}'`.

```bash
export OPENAI_API_KEY="..."
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py);
clients set both `CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN`.
Restart the env-server between strict model evals to recreate the shared
WebHarbor container baseline:

```bash
uv run python scripts/serve_env.py --env-ids webharbor.webvoyager --token "$USER"
export CUA_LITE_ENV_SERVER_URL=http://$(hostname -I | awk '{print $1}'):30100
export CUA_LITE_ENV_SERVER_TOKEN="$USER"
```

Use `localhost` only when the env-server and rollout client share the same network namespace.

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("webharbor.webvoyager@<task_id>", max_steps=15)
asyncio.run(env.reset())
```

The task manifest `data/tasks.json` is committed; regenerate it (dev only) with `bash lite/gym/envs/webharbor/webvoyager/scripts/utils/tasks.sh`.

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    splits = gym.registry.task_ids("webharbor.webvoyager")  # {"eval": [...]}
    task_ids = [tid for ids in splits.values() for tid in ids]

    env = gym.make(f"webharbor.webvoyager@{task_ids[0]}", max_steps=15)
    obs = await env.reset()
    print(obs.text)

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

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/webharbor/webvoyager/configs/default.yaml) — `env_kwargs` (per-instance), `server_kwargs` (per-deployment infra), and `make_kwargs` (env-wide `gym.make` defaults), read via `env_config.load`. Swap the whole file with `WEBHARBOR_WEBVOYAGER_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

`make_kwargs.cursor` defaults to `true` for coord mode. The WebHarbor RPC server composites the shared Linux cursor sprite inside screenshot capture; `cursor=False` requests raw screenshots, and SoM mode remains governed by `env_kwargs.use_som`.

| Key | Default | Why |
|---|---|---|
| `max_steps` | `null` (→ per-task, 15) | Step budget per episode; `null` reads the per-task value from `data/tasks.json`. |
| `step_timeout` | `300.0` | Per-step wall-clock cap (seconds). |
| `post_action_delay` | `0.5` | Settle time after each action (seconds). |
| `viewport` | `[1280, 720]` | Browser viewport (fixed render size). |
| `use_som` | `false` | Draw the Set-of-Marks `[N]` boxes on the screenshot. `false` = plain screenshot (coord mode); `som.yaml` sets `true`. |
| `fix_box_color` | `true` | Use a fixed color for the SoM bounding boxes (only applies when `use_som: true`). |
| `valid_actions` | `[click, type, key, scroll, wait]` | Advertised Lite GUI actions; set `[]` for SoM so the model sees only standalone DOM/finish tools. |
| `extra_tools` | `[]` | Advertised standalone tools per rollout YAML: `back`, `goto`, `response` (coord) or the WebHarbor SoM tools. |
| `eval_config.model` | `gpt-4.1` | VLM judge model. |
| `instances` | `0` | In-container Selenium session pool size; `0` = auto-derive from host RAM. |
| `page_load_timeout_s` | `15.0` | Max page load wait per action. |
| `instance_ttl_s` | `600.0` | Idle session cleanup. |

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("webharbor.webvoyager"))   # {"eval": ["allrecipes.0", ...]}
```

643 tasks under the `eval` split (dumped from the pinned WebHarbor/WebVoyager sources). Metadata in `env.metadata.others`: `site_slug`, `sites`, `upstream_url`, `mutating`.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via a VLM judge (`eval_config.model`, default `gpt-4.1`), which receives the last N screenshots (`max_attached_imgs: 3`) and the agent's final response text, assessing task completion against the original instruction. Set `skip_eval=True` to bypass scoring (`reward` is then `None`, not `0.0`); a no-answer episode also yields `reward=None`. See [Setup](#setup) for the key.

### Strict Two-Pass Eval

Restart the env-server between models to recreate the container and reset the whole WebHarbor suite once. Then run read-only tasks in parallel and mutating tasks serially:

```bash
uv run python scripts/rollout.py ... \
  --env-id webharbor.webvoyager --splits eval --concurrency 32 \
  --filter "lambda m: not m.others.get('mutating')" \
  --config-path scripts/configs/qwen3_vl/default/webharbor.webvoyager/default.yaml

uv run python scripts/rollout.py ... \
  --env-id webharbor.webvoyager --splits eval --concurrency 1 \
  --filter "lambda m: m.others.get('mutating')" \
  --config-path scripts/configs/qwen3_vl/default/webharbor.webvoyager/default.yaml
```

No per-task site reset is performed. Correctness comes from one clean container baseline plus the read/write ordering above.

## Action Space

Two action shapes are supported; pick via the rollout config:

**Computer use (coord)** — default; `LiteBrowserActionSpace` with `[0, 1000]` normalized coordinates. The model emits standard `computer_use` tool calls. Use the `webharbor.webvoyager/default.yaml` configs:

| Action | Arguments | Description |
|---|---|---|
| `click(coordinate)` | `coordinate`, `button`, `clicks` | Click at `[0, 1000]` normalized coordinates |
| `type(text)` | `text` | Type text |
| `key(keys)` | `keys` | Press key combo |
| `scroll(coordinate, direction, amount)` | `coordinate`, `direction`, `amount` | Scroll page |
| `wait(duration)` | `duration` | Wait before next action |
| `back()` | — | Browser back (**extra tool**) |
| `response(text)` | `text` | Standalone extra tool; store final answer and terminate |

Coordinates are normalized to `[0, 1000]` → pixels (viewport `1280×720`). `back` is in `env.metadata.extra_tool_schemas` — see [docs/envs.md](/docs/envs.md#extra-tools). Coord mode keeps `use_som: false`, so the observation is a **plain** screenshot (no `[N]` boxes).

**SoM (Set-of-Marks)** — `som.yaml` disables coordinate actions and sets
`use_som: true`. Observation is a screenshot annotated with colored bounding
boxes labeled `[N]` plus accessibility tree text `[N]: <tag> text @ (cx, cy)` in
the conversation. Use `webharbor.webvoyager/som.yaml` configs:

| Action | Arguments | Description |
|---|---|---|
| `click(index)` | `index` (int) | Click element at index `[N]` |
| `input(index, text)` | `index` (int), `text` | Type text into input element at `[N]` |
| `scroll(down, pages)` | `down` (bool), `pages` (float) | Scroll by number of viewport pages |
| `go_back()` | — | Browser back |
| `response(text)` | `text` | Final answer and terminate |

Indices must appear in the accessibility tree; they refer to the same `[N]` labels visible in the annotated screenshot.
The backend still supports navigation tools for custom configs, but the paper-aligned rollout configs intentionally omit `goto`/`navigate`.
Use `response(text)` as the final-answer tool when it is enabled by
`extra_tools`; `done(...)` is not enabled by the provided configs.

## Rollout configs

Paper-aligned rollout configs live under `scripts/configs/`:

```
scripts/configs/gpt/default/webharbor.webvoyager/default.yaml
scripts/configs/qwen3_vl/default/webharbor.webvoyager/{default.yaml,som.yaml}
scripts/configs/qwen3_5/default/webharbor.webvoyager/{default.yaml,som.yaml}

# default.yaml = computer use (coord); extra_tools: [back]
# som.yaml = SoM; coordinate actions disabled
#                 extra_tools: [click, input, scroll, go_back, response]
```

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/webharbor/webvoyager/
├── main.py              # RemoteWebVoyagerEnv (LiteBaseEnv) + RPC client + container services + VLM judge
├── configs/default.yaml # tunable defaults (env_kwargs + server_kwargs + make_kwargs.cursor)
├── data/tasks.json      # static task manifest (643 tasks; regenerate with scripts/utils/tasks.sh)
├── docker/              # Dockerfile (pins WebVoyager + WebHarbor) + server.py + entrypoint.sh
└── scripts/
    ├── install.sh       # build / rebuild / pull / status / health
    └── utils/tasks.sh         # regenerate data/tasks.json from the built image
```

**References:** [paper](https://arxiv.org/abs/2401.13919) · [github](https://github.com/MinorJerry/WebVoyager) · [WebHarbor](https://github.com/aiming-lab/WebHarbor)

</details>

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@inproceedings{he2024webvoyager,
  author    = {Hongliang He and Wenlin Yao and Kaixin Ma and Wenhao Yu and Yong Dai and Hongming Zhang and Zhenzhong Lan and Dong Yu},
  title     = {WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand, August 11-16, 2024},
  pages     = {6864--6890},
  publisher = {Association for Computational Linguistics},
  year      = {2024},
  url       = {https://doi.org/10.18653/v1/2024.acl-long.371},
  doi       = {10.18653/V1/2024.ACL-LONG.371}
}

@misc{webharbor2026,
  title  = {WebHarbor: Docking Real Websites for Evolving GUI Agent Environments},
  author = {{WebHarbor Team and Contributors}},
  year   = {2026},
  url    = {https://aiming-lab.github.io/webharbor.github.io},
  note   = {Project website.}
}
```
