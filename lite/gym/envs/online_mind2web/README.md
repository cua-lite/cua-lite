# Online-Mind2Web

`--env-id` `online_mind2web`

CUA-Lite wrapper for [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web). 300 diverse tasks across 147 distinct websites (the upstream paper headlines ~136) in domains such as clothing, food, housing, and transportation, via `gym.make("online_mind2web@<task_id>")` with `LiteBrowserActionSpace` (`[0, 1000]` coords). See [docs/envs.md](/docs/envs.md) for the env contract.

## Setup

```bash
# Choose one install path:
# Source path: build the self-contained cua-lite/online_mind2web image if missing/stale.
uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image.
# uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh status    # image freshness/resources
# uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh health    # temporary container health probe
# uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh rebuild   # force a fresh rebuild
```

Set a judge API key — without one, `reset()` fails loud (or pass `skip_eval=True` to run without evaluation):

```bash
export OPENAI_API_KEY="..."
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
env = gym.make("online_mind2web@<task_id>", max_steps=30)
asyncio.run(env.reset())
```

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    splits = gym.registry.task_ids("online_mind2web")  # {"eval": [...]}
    task_ids = [tid for ids in splits.values() for tid in ids]

    env = gym.make(f"online_mind2web@{task_ids[0]}", max_steps=30)
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

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/online_mind2web/configs/default.yaml) — `env_kwargs` (per-instance), `server_kwargs` (per-deployment infra), and `make_kwargs` (env-wide `gym.make` defaults), read via `env_config.load`. Swap the whole file with `ONLINE_MIND2WEB_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

`make_kwargs.cursor` defaults to `true`. The container RPC server composites the shared Linux cursor sprite during screenshot capture; pass `cursor=False` when validating raw frame parity.

| Key | Default | Why |
|---|---|---|
| `max_steps` | `null` (→ 30) | Step budget per episode. |
| `step_timeout` | `600.0` | Terminal steps include WebJudge evaluation — keep high. |
| `post_action_delay` | `1.5` | Settle time after each action (seconds). |
| `viewport` | `[1280, 720]` | Browser viewport. |
| `skip_eval` | `false` | Set `true` to run without the VLM judge (`reward=None`). |
| `instances` | `0` | In-container browser pool size; `0` = auto-derive from host RAM. |
| `page_load_timeout_s` | `20.0` | Max page load wait per action. |
| `instance_ttl_s` | `600.0` | Idle instance expiry. |

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("online_mind2web"))   # {"eval": ["0059adc6b12a3822305deb68929b2de8", ...]}
```

300 tasks, all under the `eval` split (task-ids are upstream Online-Mind2Web hashes — mostly 32-char hex, some with a `_<date>` suffix), across 147 distinct websites and domains including clothing, food, housing, and transportation. The committed manifest is [`data/tasks.json`](/lite/gym/envs/online_mind2web/data/tasks.json), built from HuggingFace ([`osunlp/Online-Mind2Web`](https://huggingface.co/datasets/osunlp/Online-Mind2Web)). Metadata in `env.metadata.others`: `website`, `level` (raw difficulty string `easy`/`medium`/`hard`), `difficulty` (mapped numeric), `reference_length`.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via WebJudge (`"WebJudge_Online_Mind2Web_eval"`, `eval_config.model` default `o4-mini`) using a VLM to assess the final browser state against the task reference. With `skip_eval=True` it is skipped and `reward` is `None` (not `0.0`).

## Action Space

Coordinates are `[0, 1000]` normalized → pixels (`1280×720` viewport by default).

| Action | Arguments | Mind2Web / Playwright Op | Description |
|---|---|---|---|
| `click(coordinate)` | `coordinate`, `button` (default: left), `clicks` (default: 1) | `page.click(x, y)` | Click at coordinates |
| `type(text)` | `text`, `press_enter` (bool, derived host-side from a trailing newline) | `element.fill()` / `keyboard.type()` | Type text at focused element |
| `key(keys)` | `keys` (list or single) | `keyboard.press()` | Key combo (Enter, Tab, ArrowDown, …) |
| `scroll(direction, amount)` | `direction` (down/up/left/right), `amount` (default: 3), `coordinate` (optional) | `mouse.wheel()` / `keyboard.press()` | Scroll page or element (1 amount = 100 px) |
| `wait(duration)` | `duration` (seconds, default: 1.0) | `asyncio.sleep()` | Wait before next action |
| `mouse_move(coordinate)` | `coordinate` | `mouse.move(x, y)` | Move cursor without clicking |
| `drag(start_coordinate, coordinate)` | `start_coordinate`, `coordinate` | `mouse.move` + `down` + `up` | Smooth drag between two points |
| `back()` / `goback()` | — | `page.go_back()` | Browser back (**extra tool**) |
| `forward()` / `goforward()` | — | `page.go_forward()` | Browser forward (**extra tool**) |
| `goto(url)` | `url` | `page.goto(url)` | Navigate to URL (**extra tool**) |
| `response(text)` | `text` | — | Standalone extra tool; store final answer and terminate |
| `terminate(status)` | `status` (optional) | — | Standalone extra tool; end episode without answer |

`back`, `forward`, and `goto` are in `env.metadata.extra_tool_schemas` and must be passed to the agent adapter — see [docs/envs.md](/docs/envs.md#extra-tools).

The container's dispatcher also implements a `refresh` (`page.reload()`), but **no agent can emit
it** and it is deliberately absent from the table above: `refresh` is in neither the `computer`
action enum nor `OnlineMind2WebTools`, so `resolve_valid_actions(["refresh"], env_name="online_mind2web",
platform="browser")` rejects the name and `env_kwargs.extra_tools` cannot select it. Do not add it back
here without first adding it to a surface.

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/online_mind2web/
├── main.py                  # RemoteOnlineMind2WebEnv (LiteBaseEnv) + client + container services
├── configs/default.yaml     # tunable defaults (env_kwargs + server_kwargs + make_kwargs.cursor)
├── data/tasks.json          # static task manifest (300 tasks; regenerate with scripts/utils/tasks.sh)
├── docker/
│   ├── Dockerfile           # self-contained image build
│   ├── server.py            # FastAPI RPC server: action execution over Playwright
│   ├── entrypoint.sh        # container entrypoint
│   └── healthcheck.py       # container health check
└── scripts/
    ├── install.sh           # build / rebuild / pull / status / health
    └── utils/tasks.sh             # regenerate data/tasks.json from the built image
```

**References:** [paper](https://arxiv.org/abs/2410.13232) · [dataset](https://huggingface.co/datasets/osunlp/Online-Mind2Web) · [github](https://github.com/OSU-NLP-Group/Online-Mind2Web)

</details>

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@inproceedings{xue2025an,
  title     = {An Illusion of Progress? Assessing the Current State of Web Agents},
  author    = {Tianci Xue and Weijian Qi and Tianneng Shi and Chan Hee Song and Boyu Gou and Dawn Song and Huan Sun and Yu Su},
  booktitle = {Second Conference on Language Modeling},
  year      = {2025},
  url       = {https://openreview.net/forum?id=6jZi4HSs6o}
}

@inproceedings{deng2023mind2web,
  author    = {Xiang Deng and Yu Gu and Boyuan Zheng and Shijie Chen and Samual Stevens and Boshi Wang and Huan Sun and Yu Su},
  title     = {Mind2Web: Towards a Generalist Agent for the Web},
  booktitle = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023},
  year      = {2023},
  url       = {http://papers.nips.cc/paper\_files/paper/2023/hash/5950bf290a1570ea401bf98882128160-Abstract-Datasets\_and\_Benchmarks.html}
}
```
