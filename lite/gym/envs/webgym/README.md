# WebGym

`--env-id` `webgym`

CUA-Lite wrapper for [WebGym](https://github.com/microsoft/webgym) OmniBoxes. 292k+ web information-retrieval tasks, via `gym.make("webgym@<task_id>")` with `LiteBrowserActionSpace` (`[0, 1000]` coords) + browser `extra_tools` (`back`, `goto`). See [docs/envs.md](/docs/envs.md) for the env contract.

## Setup

```bash
# Choose one install path:
# Source path: build the self-contained cua-lite/webgym image if missing/stale.
# The build clones/patches OmniBoxes inside the image; no host clone or host Redis.
uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image and install host judge deps.
# uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh status    # image freshness/resources
# uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh rebuild   # force a fresh rebuild
```

Set a judge API key for scored runs — without one, `reset()` fails loud. For
plumbing checks that do not score, pass `skip_eval=True` / `--env-kwargs
'{"skip_eval": true}'`.

```bash
export OPENAI_API_KEY="..."
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py);
clients set both `CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN`:

```bash
uv run --no-sync python scripts/serve_env.py --env-ids webgym --token "$USER"
export CUA_LITE_ENV_SERVER_URL=http://$(hostname -I | awk '{print $1}'):30100
export CUA_LITE_ENV_SERVER_TOKEN="$USER"
```

Use `localhost` only when the env-server and rollout client share the same network namespace.

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("webgym@<task_id>", max_steps=10)
asyncio.run(env.reset())
```

<details>
<summary>Setup notes — ports, verify, local VLM judge</summary>

Ports, pool size, and the master URL **auto-allocate** (conflict-safe, the same allocator as lite.osworld) — you normally set nothing. To change a default, edit [`configs/default.yaml`](/lite/gym/envs/webgym/configs/default.yaml) (see [Configuration](#configuration)). Remove the image with `uv run --no-sync bash lite/gym/envs/webgym/scripts/uninstall.sh`.

**Verify a running container:** `curl -s -H "x-api-key: default_key" http://<host>:<published-port>/info` (or `docker logs webgym-<env-server-port>`).

**Local VLM as judge** (instead of OpenAI) — the evaluator runs host-side; point it at an sglang/vLLM endpoint:

```bash
WEBGYM_EVAL_MODEL=Qwen/Qwen2.5-VL-7B-Instruct WEBGYM_EVAL_BASE_URL=http://localhost:30000/v1 OPENAI_API_KEY=dummy \
    uv run python your_script.py
```

The host-side evaluator needs the `webgym` package importable — `scripts/install.sh` pip-installs it into the venv. If it isn't installed, or `OPENAI_API_KEY` is unset, `reset()` fails loud (pointing at the install step) rather than silently scoring `0.0`. To run without evaluation, pass `skip_eval=True`.

</details>

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    splits = gym.registry.task_ids("webgym")  # {"train": [...], "eval": [...]}
    task_ids = [tid for ids in splits.values() for tid in ids]

    env = gym.make(f"webgym@{task_ids[0]}", max_steps=10)
    result = await env.reset()
    print(result.text)

    result = await env.step([
        make_tool_call(
            "computer",
            {"actions": [
                {"action": "click", "coordinate": [500, 300]},
                {"action": "type", "text": "search query"},
            ]},
            call_id="call_0000",
        ),
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/webgym/configs/default.yaml) — `env_kwargs` (per-instance), `server_kwargs` (per-deployment infra), and `make_kwargs` (env-wide `gym.make` defaults), read via `env_config.load`. Swap the whole file with `WEBGYM_CONFIG=<abs-path | name-under-configs/>`; `env_kwargs.eval_config` is deep-merged over the yaml for the judge. See [the env config contract](/docs/envs.md).

`make_kwargs.cursor` defaults to `true`. WebGym forwards that flag to the in-container screenshot endpoint, which composites the shared Linux cursor sprite at capture time; callers can pass `cursor=False` to inspect raw browser frames.

**Ports / master URL / pool size are auto-allocated — you touch none of them.**
Use `configs/default.yaml` only when you need to override deployment defaults.
The only required runtime env vars are **secrets** (`WEBGYM_API_KEY`, and
`OPENAI_API_KEY` for the judge — see [Setup](#setup)).

Performance knobs (throughput is env-bound, not GPT — the levers cap *wasted* time on the OmniBoxes node's frequent screenshot/page hangs):

| Key | Default | Why |
|---|---|---|
| `step_timeout` | `300.0` | Per-step wall-clock cap, sized to include terminal VLM judge calls. |
| `http_timeout` | `20.0` | Per-request timeout; kept above the patched in-container navigation wait. |
| `max_retries` | `2` | execute/nav retries; 2×`http_timeout` (40s) stays under `step_timeout`. |
| `blank_screenshot_max_retries` / `_wait` | `2` / `1.5` | `/screenshot` often 500s — fall back to the previous frame fast. |
| `instance_lifetime_mins` | `60` | **Upstream-capped at 60** (master 503s above it) — do not raise. |
| `sem_navigate` / `sem_execute` / `sem_screenshot` | `16` / `256` / `256` | Per-process backpressure. |
| `max_steps_train` / `max_steps_eval` | per-difficulty | Step budget by tier; train and eval both use 15/25/35. |
| `viewport` | `[1280, 720]` | Browser viewport (`env_kwarg`); forwarded to the container browsers via `-e WEBGYM_VIEWPORT=WxH`. |
| `goback_skip_threshold` / `goback_terminate_threshold` | `2` / `3` | Consecutive back-from-homepage actions before the step is skipped / the episode is terminated (anti-loop guard). |
| `instances` | `0` | Browser-pool size; `0` = auto-derive (like mobilegym `max_browsers`). |

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("webgym"))   # {"train": ["142359", ...], "eval": ["69059", ...]}
```

292k+ tasks across `train` / `eval`, multiple websites/domains, difficulty 1–8. Metadata in `env.metadata.others`: `website`, `domain`, `subdomain`, `difficulty`, `viewport`.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via WebGym's VLM judge (`eval_config.model`, default `gpt-4.1`; `evaluator.get_verifiable_reward(trajectory)`): image relevance → blocking detection → anti-hallucination → fact verification. Evaluator availability is enforced at `reset()` (fail-loud), so this always runs; with `skip_eval=True` it is skipped (`reward=0.0`).

| Action | WebGym Command | Description |
|--------|---------------|-------------|
| `click(coordinate)` | `click_coords(x, y)` | Click at coordinates |
| `type(text)` | `fill_coords(x, y, value)` | Type text at last click position |
| `key(keys)` | `keypress(keys)` | Key combo; host projects canonical `ctrl`/`esc` to Playwright `Control`/`Escape` before HTTP |
| `scroll(coordinate, direction)` | `hover_and_scroll_coords` / `page_down` | Scroll |
| `mouse_move(coordinate)` | `hover_coords(x, y)` | Move cursor |
| `back()` | `back()` | Browser back (**extra tool**) |
| `goto(url)` | `visit_page(url)` | Navigate to URL (**extra tool**) |
| `response(text)` / `terminate(status)` | — | Standalone extra tools; store answer / end episode |

Coordinates are `[0, 1000]` normalized → pixels (1280×720 viewport). `back` / `goto` are in `env.metadata.extra_tool_schemas` and must be passed to the agent adapter — see [docs/envs.md](/docs/envs.md#extra-tools).

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/webgym/
├── __init__.py            # package marker; runtime main.py is lazy-loaded by the registry
├── main.py                # WebGymEnv (LiteBaseEnv) + WebGymClient (httpx) + WebGymContainerServices
├── configs/default.yaml   # tunable defaults (env_kwargs + server_kwargs + make_kwargs.cursor)
├── pool_sizing.py         # in-container pool auto-sizing
├── docker/                # Dockerfile (self-contained, clones pinned upstream) + entrypoint.sh
│   └── patches/           # 5 vendored OmniBoxes patches cp'd over the clone at build
└── scripts/               # install.sh (build / rebuild / pull / status) / uninstall.sh
```

**Troubleshooting:**

- `503 on /get` — all instances in use; raise yaml `server_kwargs.instances` or reduce rollout concurrency.
- `500 on /execute` — Playwright page crash/timeout; handled gracefully (a dead browser ends the trajectory early).
- `Connection refused` — the `cua-lite/webgym` container isn't up yet; `WebGymContainerServices.ensure` brings it up on demand (env-server / `gym.make`). Check `docker logs webgym-<env-server-port>` and that the image is built (`install.sh status`).

**Testing:**

```bash
uv run pytest tests/gym/envs/webgym/test_webgym.py -k "not live" -q
uv run pytest tests/gym/envs/webgym/test_webgym.py -m live -k cursor_toggle_and_lease_cleanup -q
uv run pytest tests/gym/remote/test_direct_server_parity_matrix.py -q
```

The live cursor/cleanup smoke brings up a real `cua-lite/webgym` container, drives a raw click, captures cursor-off/on/off screenshots, performs a pixel-region shared cursor sprite assertion, then closes the env and polls `/info` until the lease is released. The parity matrix is Docker-free and confirms caller-visible metadata, reset observations, step results, and env kwargs such as `cursor=False` match direct `gym.make` and in-process env-server construction.

**References:** [paper](https://arxiv.org/abs/2601.02439) · [dataset](https://huggingface.co/datasets/microsoft/webgym_tasks) · [github](https://github.com/microsoft/webgym)

</details>

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@misc{bai2026webgym,
  title         = {WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks},
  author        = {Hao Bai and Alexey Taymanov and Tong Zhang and Aviral Kumar and Spencer Whitehead},
  year          = {2026},
  eprint        = {2601.02439},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2601.02439}
}
```
