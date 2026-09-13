# BrowserGym

`--env-id` `browsergym.miniwob` · `browsergym.visualwebarena` · `browsergym.webarena`

CUA-Lite wrapper for [BrowserGym](https://github.com/ServiceNow/BrowserGym). 1847 tasks across 3 benchmarks (MiniWoB / WebArena / VisualWebArena), via `gym.make("browsergym.<bench>@<task_id>")` with `LiteBrowserActionSpace` + browser `extra_tools` (`back`, `goto`). See [docs/envs.md](/docs/envs.md) for the env contract.

| Benchmark | Task ID prefix | Tasks | Infra | Difficulty |
|---|---|---|---|---|
| MiniWoB | `browsergym.miniwob@` | 125 | Local HTTP server | Low |
| WebArena | `browsergym.webarena@` | 812 | 5 Docker containers, ~32 GB RAM, ~60 GB images + ~200 GB OSM assets by default | Medium |
| VisualWebArena | `browsergym.visualwebarena@` | 910 | WebArena + Classifieds container | Medium |

## Setup

```bash
# One-time: download tars/zip/clone + `docker load` images (required for both modes).
uv run --no-sync bash lite/gym/envs/browsergym/scripts/install.sh <benchmark>
# benchmark = webarena (~60 GB images + ~200 GB OSM assets by default) | visualwebarena (webarena + classifieds) | miniwob (no Docker)
```

WA/VWA also need a judge key — `export OPENAI_API_KEY="..."` (required
benchmark-wide; without it `reset()` hard-fails). MiniWoB needs none.

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py);
clients set both `CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN`.
Use the concrete env id for the benchmark you installed:

```bash
uv run --no-sync python scripts/serve_env.py --env-ids browsergym.webarena --token "$USER"
export CUA_LITE_ENV_SERVER_URL=http://$(hostname -I | awk '{print $1}'):30100
export CUA_LITE_ENV_SERVER_TOKEN="$USER"
```

Use `localhost` only when the env-server and rollout client share the same network namespace.

WA/VWA boot slowly (gitlab ~5-15 min); a casual run just lets first-wave tasks retry through the cold boot (a retriable 503 + a `warming: gitlab …` message). For operator warm/reset notes, see [WA/VWA runtime](#wavwa-runtime).

**Direct mode** — `gym.make`'s `ensure_services` auto-starts the stack; or bring it up yourself first:

```bash
uv run --no-sync bash lite/gym/envs/browsergym/scripts/start.sh <benchmark>   # bring the stack up
```

<details>
<summary>Setup notes</summary>

**Lifecycle verbs** (all take `<bench>` as sole positional arg):

| Script | Purpose |
|---|---|
| `install.sh <bench>` | Download artifacts (tars / zip / git clone) into `.cache/` |
| `start.sh <bench>` | Boot services + `export` URLs (`source` to keep exports in your shell; `bash` keeps services but loses exports) |
| `cleanup.sh <bench>` | Stop services (images kept for fast restart) |
| `uninstall.sh <bench>` | Delete downloaded artifacts (destructive, multi-GB) |

`cleanup.sh miniwob` stops the shared MiniWoB singleton (pkills the HTTP server across the 7560-7659 port range and removes its registry entry) — full host teardown, not a no-op.

**Env-var knobs:**

- `BROWSERGYM_CACHE` — cache root. Defaults to this checkout's
  `lite/gym/envs/browsergym/.cache/`; on shared dev hosts, export the cache
  path for an existing BrowserGym cache before starting services.
- `WEBARENA_HOST` — hostname services claim (default `localhost`)
- `MINIWOB_PORT` — miniwob HTTP server port. Unset → a host-wide **shared singleton**: `_ensure_miniwob_singleton` prefers `7560` (auto-picks the next free port in `7560–7659` on conflict) and records it in a flock'd registry so every env-server reuses the one instance. Set it to pin a private port/corpus. (7560 to avoid colliding with webgym's OmniBoxes node, which uses 8080)

For multi-checkout development hosts, a missing `.cache/` in the current checkout
is not an asset blocker if an existing BrowserGym cache is exported. Point
`BROWSERGYM_CACHE` at that cache before starting services. Then `source` the
start script so `MINIWOB_URL`, `WA_*`, and `VWA_*` exports reach the
rollout/env-server process:

```bash
export BROWSERGYM_CACHE=/path/to/cua-lite/lite/gym/envs/browsergym/.cache
source lite/gym/envs/browsergym/scripts/start.sh miniwob
source lite/gym/envs/browsergym/scripts/start.sh visualwebarena
```

The WebArena / VisualWebArena start path also exports the site URLs it owns, including `WA_MAP` for OpenStreetMap, `WA_HOMEPAGE` / `VWA_HOMEPAGE` for image-goal assets, and `VWA_CLASSIFIEDS_RESET_TOKEN` for classifieds reset tasks.

**Non-localhost / custom ports:** `start.sh` binds services on `localhost` with the ports baked into WebArena's images (matching `main.py` `_DEFAULT_ENV_VARS`, so localhost needs no env vars). For any override, **`source`** start.sh so it propagates `WA_*` / `VWA_*` / `MINIWOB_URL` into your Python process — e.g. `WEBARENA_HOST=remote source lite/gym/envs/browsergym/scripts/start.sh webarena` — or `export` them manually after `bash start.sh`.

**Per-benchmark verification** (after install/start for the given benchmark):

```bash
uv run python -c "
import asyncio, lite.gym as gym
async def main():
    env = gym.make('browsergym.miniwob@click-dialog', max_steps=5)
    # or: 'browsergym.webarena@0' / 'browsergym.visualwebarena@0'
    r = await env.reset(); print(f'Instruction: {r.text}'); await env.close()
asyncio.run(main())"
```

</details>

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("browsergym.miniwob@click-dialog", max_steps=10)
    result = await env.reset()
    print(f"Instruction: {result.text}")

    result = await env.step([
        make_tool_call(
            "computer",
            {"actions": [
                {"action": "click", "coordinate": [500, 500]},
            ]},
            call_id="call_0000",
        ),
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## WA/VWA runtime

For scored WebArena / VisualWebArena runs, start from a clean env-server,
pre-warm the shared stack, and restart the server between model evaluations so
each model sees a fresh backend baseline:

```bash
uv run --no-sync python scripts/serve_env.py \
  --env-ids browsergym.webarena --warm-singleton --token "$USER"
```

Poll `GET /envs/browsergym.webarena` until it reports `"available": true`,
then point rollout clients at the server with `CUA_LITE_ENV_SERVER_URL` and
`CUA_LITE_ENV_SERVER_TOKEN`. For direct-mode service resets, use
`cleanup.sh <benchmark>` followed by `start.sh <benchmark>`.

WA/VWA share mutable services. Read-only tasks can run in parallel on a clean
baseline; mutating tasks should be evaluated from that same baseline in the
benchmark dependency order. Detailed rollout runbooks, backend-isolation
mechanics, and development notes live in
[`devs/envs/browsergym.md`](/devs/envs/browsergym.md).

## Step feedback

`env.step()` returns per-call results for executed or rejected tool calls,
including terminal and max-step calls when a pairable `call_id` exists.
BrowserGym keeps the current page context in metadata/debug logs whenever
possible, while model-visible text stays screenshot-first by default. Text,
SoM, and mixed configs opt in with `include_page_context_text: true` when the
agent needs URL/title/AXTree/HTML feedback.

`valid_actions` constrains only coordinate GUI actions; BrowserGym bid/nav/finish
tools are selected separately by `action_subsets` and `extra_tools`.

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/browsergym/configs/default.yaml) — `env_kwargs` (per-instance), `server_kwargs` (per-deployment infra), and `make_kwargs` (env-wide `gym.make` defaults), read via `env_config.load`. Swap the whole file with `BROWSERGYM_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

`make_kwargs.cursor` defaults to `true` for coord screenshots. BrowserGym paints the shared Linux cursor sprite inside its env-owned screenshot path only when coord actions provide a trustworthy pixel cursor; BID/text and SoM-style native screenshots remain raw until a coord action establishes cursor state.

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("browsergym.miniwob"))        # 125 tasks
print(gym.registry.task_ids("browsergym.webarena"))       # 812 tasks
print(gym.registry.task_ids("browsergym.visualwebarena")) # 910 tasks
```

Metadata in `env.metadata.others`: `sites`, `mutating`, `depends_on`, `llm_as_a_judge` — see [Filtering tasks](#filtering-tasks-static-task-facts) below.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via BrowserGym's per-benchmark evaluator. A VWA subset scores via an LLM judge (`llm_fuzzy_match`, default `gpt-4.1`) / image-caption model through the configured API route, so **WA/VWA require `OPENAI_API_KEY`** benchmark-wide — the upstream evaluator builds its client at import time, so a missing key makes *every* VWA `reset()` hard-fail (not just judged tasks), never a silent `0.0`. See [Setup](#setup).

### Eval rigor (opt-in)

WA/VWA share mutable services, so scored runs should use a clean warmed
baseline and the read/write handling summarized in
[WA/VWA runtime](#wavwa-runtime). Developer runbooks for exact rollout splits
and isolation tuning live in [`devs/envs/browsergym.md`](/devs/envs/browsergym.md).

One env-var matters regardless of approach: `VWA_CLASSIFIEDS_RESET_TOKEN` (auto-set by `start.sh`) — without it the 22 VWA `require_reset` classifieds tasks can't self-reset. Site URLs (`WA_SHOPPING`, …, `MINIWOB_URL`) are also set by `start.sh` and fail *loudly* if wrong, so they aren't a silent-corruption risk.

## Action Space

The env can be driven in two action shapes; pick via `BrowserGymConfig.action_subsets` (yaml `env_kwargs.action_subsets`):

**Coord-mode** — default; vision pipelines (`default.yaml` configs, `goal_image.yaml` for VWA):

| Action | Description |
|---|---|
| `click(coordinate, button, clicks)` | Mouse click at `[0, 1000]` coordinates |
| `type(text)` | Type text |
| `key(keys)` / `key_down(keys)` / `key_up(keys)` | Press / hold / release keys |
| `mouse_move(coordinate)` / `mouse_down(button)` / `mouse_up(button)` | Cursor movement / button state |
| `drag(start_coordinate, coordinate)` | Drag and drop |
| `scroll(coordinate, direction, amount)` | Scroll |
| `back()` / `goto(url)` | Browser navigation (**extra tools**) |
| `response(text)` / `terminate(status, reason)` | Standalone extra tools; end episode |

Coordinates are normalized to `[0, 1000]` and converted to pixel coordinates based on viewport (MiniWoB: 498×321, WebArena/VWA: 1280×720).

**Bid-mode** — text + AXTree pipelines (`text_only.yaml` for MiniWoB / WA; `mixed.yaml` for VWA — agent-as-annotators paper repro). The agent emits `<tool_call>click('a47')</tool_call>` style calls referring to AXTree bids:

| `action_subsets` | Tools surfaced (count) | Used by |
|---|---|---|
| `["webarena"]` | `click, fill, hover, scroll, keyboard_press, select_option, goto, back, forward, new_tab, switch_tab, close_tab, response, terminate` (14) | WA `text_only.yaml` |
| `["visualwebarena"]` | webarena + `upload_file` (15) | VWA `mixed.yaml` |
| `["bid", "chat", "infeas"]` | `click, fill, hover, scroll, select_option, press, dblclick, focus, clear, drag_and_drop, upload_file, response, terminate` (13 — no nav / tab) | MiniWoB `text_only.yaml` |
| `["coord", "chat", "infeas", "nav", "tab"]` (WA `default.yaml` / VWA `goal_image.yaml`) | nav (3) + tab (3) + chat (1) + infeas (1) = 8 — coord actions are routed through cua-lite's `LiteBrowserActionSpace` and not re-exposed as named tools | WA `default.yaml` / VWA `goal_image.yaml` configs |
| `["coord", "chat", "infeas", "nav"]` (MiniWoB `default.yaml`) | nav (3) + chat (1) + infeas (1) = 5 — MiniWoB tasks are single-tab, so the `tab` subset is dropped | MiniWoB `default.yaml` |

Tool schemas are auto-derived per-call from BrowserGym's live `HighLevelActionSet` via `_tools_for_subsets`; no hard-coded tool lists.

`valid_actions` is orthogonal to `extra_tools`: `valid_actions: []` disables
coordinate GUI actions, not BrowserGym bid/nav/upload/finish tools. Bid-mode
configs keep those tools active through `extra_tools`.

## Rollout configs

Paper-aligned cua-lite rollout configs live under `scripts/configs/`:

```
scripts/configs/{qwen3_5,qwen3_vl}/default/browsergym.{miniwob,webarena,visualwebarena}/
├── text_only.yaml | mixed.yaml
└── default.yaml | goal_image.yaml
```

VWA uses `mixed.yaml` (not `text_only.yaml`) because Qwen3.5-9B / Qwen3-VL are multimodal — the env surfaces the task's goal image(s) (single or multi) and they are shown on every turn so the model can ground "find this product" prompts even in text+AXTree mode (see [Goal images](#goal-images) below).

## Filtering tasks (static task facts)

The env exposes **static, deploy-independent task facts** in `metadata.others`
and bakes **no** `exclude_reason` for service reachability. Reachability is a
*mutable* property (whether you've provisioned a service, on what hardware) — so
baking it at registration (and freezing it into exported parquet) would be
wrong. Instead the env stays generic and you compose the filter; a backend that
is merely cold-booting is retried via **503 + Retry-After**, not pre-excluded
(see the cold-boot note below).

| `others` key | Benches | Meaning |
|---|---|---|
| `sites` (`list[str]`) | WA / VWA | WA sites the task touches (`"shopping"`, `"map"`, `"gitlab"`, `"reddit"`, …). Static, from the task json. |
| `llm_as_a_judge` (`bool`) | WA / VWA | `True` iff the evaluator uses `string_match.fuzzy_match` (needs an external LLM judge — cua-lite default `gpt-4.1` via `llm_judge_model`; upstream hard-codes `gpt-4-1106-preview`). Runs by default — reset hard-fails if `OPENAI_API_KEY` is unset (no silent reward-0). Static. |
| `mutating` (`bool`) | WA / VWA | `True` iff the task writes persisted backend state (`isolation.is_mutating`). Use this for the WA/VWA operator split described in [WA/VWA runtime](#wavwa-runtime). Static. |
| `depends_on` (`list[str]`) | WA / VWA | Curated run-order parents (BrowserGym metadata) — orders the write pass so an earlier writer's residue can't false-satisfy a later task. Static. |

VWA goal-image tasks (single or multi) are NOT excluded — see [Goal images](#goal-images). Filter at rollout time on the static facts:

```bash
# map tasks run by default (install.sh installs OSM, start.sh serves it).
# Only if you installed with WEBARENA_INSTALL_MAP=0, skip them:
uv run python scripts/rollout.py ... --filter "lambda m: 'map' not in m.others.get('sites', [])"
```

(LLM-judge tasks RUN by default — set `OPENAI_API_KEY` for the `gpt-4.1` judge; a missing
key hard-fails at reset rather than silently scoring 0. Filter on `llm_as_a_judge` only if you
deliberately want to exclude them.)

> **Cold boot is handled, not excluded.** A service that isn't HTTP-ready yet
> (Magento warming, gitlab's ~5-15 min boot, classifieds) makes the env raise a
> retriable **503** (from `ensure_services` for shopping, or from `reset()` when
> a task navigates to a still-warming service); the client retries until it's up.
> So you don't pre-exclude classifieds/gitlab/map by reachability — you just run,
> and only filter on the *static* facts above (e.g. you genuinely lack OSM).

## Backend isolation

`BROWSERGYM_CONFIG=isolation` is an opt-in env-server backpressure mode for
custom shared-backend runs. It is not required for normal setup, reset, or step
usage. Detailed isolation mechanics are developer/audit material; see
[`devs/envs/browsergym.md`](/devs/envs/browsergym.md) when tuning large WA/VWA
evals.

## Goal images

VWA tasks may carry one or more task-defining goal images ("find/sell THIS item"). The env stores their PNG bytes in BrowserGym's goal-image side channel and puts only small refs in reset metadata; remote reset frames carry the bytes in the binary tail and rewrite those refs client-side. The `visualwebarena.goal_image` agent consumes the refs once at turn 0 and persists the images as ordinary trajectory image parts before the page screenshot.

Configs select that agent (see [goal_image.py](/lite/agents/extensions/browsergym/goal_image.py)) via `agent_id: "visualwebarena.goal_image"`, pointed at the model adapter with `agent_kwargs.adapter_key` (e.g. `"qwen3_vl"` for vision+coord or `"qwen3_vl.base"` for text+bid — a bare slug auto-completes `@{platform}@{task_type}` from the env metadata, like `make` does for `agent_id`; a fully-qualified key with `@` is used verbatim). It's one concrete, model-agnostic agent (serves Qwen3-VL, Qwen3.5, … by changing only `adapter_key`) that re-surfaces the goal image(s) on **every** turn before the current page screenshot, labeled `Task reference image(s) for the instruction below:` and, when needed, `Current screenshot:`. Both `mixed.yaml` (text+goal) and `goal_image.yaml` (page+goal) modes support single- and multi-image goals; they are **not** excluded.

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/browsergym/
├── __init__.py     # package marker; runtime main.py is lazy-loaded by the registry
├── main.py         # BrowserGymEnv + task registration + _wa_vwa_task_facts
├── configs/        # default.yaml (env_kwargs + server_kwargs + make_kwargs.cursor) + isolation.yaml (strict)
├── isolation.py    # shared-backend conflict-key / mutating policy + isolation server_kwargs reader
└── scripts/        # install.sh / start.sh / cleanup.sh / uninstall.sh

# Model-side agent bridges for this env:
lite/agents/extensions/browsergym/
├── protocol.py     # BrowserGymGenericProtocol ("browsergym.generic"; rebuild-per-turn)
└── goal_image.py   # VisualWebArenaGoalImageAgent ("visualwebarena.goal_image"; re-shows VWA goal images each turn)
```

**Testing:**

```bash
uv run pytest tests/gym/envs/browsergym -v -k "not MiniWoBLive"   # no browser needed
uv run pytest tests/gym/envs/browsergym                           # default parallelism from pyproject
MINIWOB_URL=http://localhost:7560/miniwob/ uv run pytest tests/gym/envs/browsergym -v -k MiniWoBLive
```

**Known issues:**

| Issue | Benchmark | Workaround |
|---|---|---|
| GitLab 500 after reconfigure (PostgreSQL connection pool exhausted) | WebArena | `start.sh` skips `gitlab-ctl reconfigure` when `WEBARENA_HOST=localhost`. For non-localhost setups, recreate the container after the configure step. |
| OpenStreetMap (Map) tasks fail if built without the ~200 GB OSM assets | WebArena | Runs by default: `install.sh` installs the OSM assets (opt out with `WEBARENA_INSTALL_MAP=0`) and `start.sh` auto-starts the Map service (`BROWSERGYM_START_MAP=0` to skip). If you built without it, skip map tasks: `--filter "lambda m: 'map' not in m.others.get('sites', [])"`. |
| Classifieds container occasionally fails its DB-init race | VisualWebArena | `start.sh` waits for `classifieds_db` `mysqladmin ping` (30 × 2 s) before running `mysql -e 'source osclass_craigslist.sql'`. On failure (`docker logs classifieds_db`) re-run `start.sh visualwebarena` — `start_classifieds` returns its sentinel correctly so configure runs even when WA is already up. Persistent failures: set `VWA_CLASSIFIEDS=http://...` to a working host. |
| WA `homepage` Flask (`localhost:$HOMEPAGE_PORT`, default `4399`, serves `static/input_images/`) | WebArena / VisualWebArena | `install.sh` sparse-checks the `webarena-homepage/` subtree (~180 MB of input PNGs) into `$BROWSERGYM_CACHE`; `start.sh` launches it via `flask run --port $HOMEPAGE_PORT` (NOT `python app.py`, whose hardcoded `app.run(4399)` ignores the port), so `HOMEPAGE_PORT` is honored. `cleanup.sh` and env-server cleanup stop it by scoped `flask run … --port <HOMEPAGE_PORT>` match. |
| Reset timeouts at 32+ concurrent envs | MiniWoB | Reduce `--concurrency` to 16, or use a faster HTTP server (nginx) |

</details>

## Citation

Please cite BrowserGym and the benchmark(s) you run alongside [CUA-Lite](/README.md#citation).

```bibtex
@article{chezelles2025browsergym,
  title   = {The BrowserGym Ecosystem for Web Agent Research},
  author  = {Thibault Le Sellier de Chezelles and Maxime Gasse and Alexandre Lacoste and Massimo Caccia and Alexandre Drouin and L{\'e}o Boisvert and Megh Thakkar and Tom Marty and Rim Assouel and Sahar Omidi Shayegan and Lawrence Keunho Jang and Xing Han L{\`u} and Ori Yoran and Dehan Kong and Frank F. Xu and Siva Reddy and Graham Neubig and Quentin Cappart and Russ Salakhutdinov and Nicolas Chapados},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2025},
  url     = {https://openreview.net/forum?id=5298fKGmv3},
  note    = {Expert Certification}
}

@inproceedings{liu2018miniwob,
  author    = {Evan Zheran Liu and Kelvin Guu and Panupong Pasupat and Tianlin Shi and Percy Liang},
  title     = {Reinforcement Learning on Web Interfaces using Workflow-Guided Exploration},
  booktitle = {6th International Conference on Learning Representations, {ICLR} 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings},
  publisher = {OpenReview.net},
  year      = {2018},
  url       = {https://openreview.net/forum?id=ryTp3f-0-}
}

@inproceedings{zhou2024webarena,
  author    = {Shuyan Zhou and Frank F. Xu and Hao Zhu and Xuhui Zhou and Robert Lo and Abishek Sridhar and Xianyi Cheng and Tianyue Ou and Yonatan Bisk and Daniel Fried and Uri Alon and Graham Neubig},
  title     = {WebArena: {A} Realistic Web Environment for Building Autonomous Agents},
  booktitle = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher = {OpenReview.net},
  year      = {2024},
  url       = {https://openreview.net/forum?id=oKn9c6ytLx}
}

@inproceedings{koh2024visualwebarena,
  author    = {Jing Yu Koh and Robert Lo and Lawrence Jang and Vikram Duvvur and Ming Chong Lim and Po{-}Yu Huang and Graham Neubig and Shuyan Zhou and Russ Salakhutdinov and Daniel Fried},
  title     = {VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand, August 11-16, 2024},
  pages     = {881--905},
  publisher = {Association for Computational Linguistics},
  year      = {2024},
  url       = {https://doi.org/10.18653/v1/2024.acl-long.50},
  doi       = {10.18653/V1/2024.ACL-LONG.50}
}
```
