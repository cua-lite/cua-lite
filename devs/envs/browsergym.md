See [AGENTS.md](/devs/envs/AGENTS.md) for shared requirements.

**Reference implementations:**
- `${CUA_LITE_REFERENCES_ROOT}/BrowserGym` — `main` branch is the unmodified upstream reference (read-only). If you need to modify BrowserGym (patch action parsing, fix bugs, add features), create a `cua-lite` branch off `main` and work there. Keep `main` clean so we can diff against upstream and pull updates.
- `${CUA_LITE_REFERENCES_ROOT}/webarena` — upstream WebArena (environment setup, Docker configs, evaluation)
- `${CUA_LITE_REFERENCES_ROOT}/visualwebarena` — upstream VisualWebArena

**Status:** implemented design notes. Current source of truth is `lite/gym/envs/browsergym/main.py`; keep this
file aligned with the implementation rather than treating older snippets below as a second spec.

**Goal:**
Wrap [BrowserGym](https://github.com/ServiceNow/BrowserGym) benchmarks as cua-lite gym environments using
the CUA-Lite coordinate action-batch tool and env-owned standalone extras such as navigation/finish tools.
Runtime invalid/unsupported feedback is returned from `step()` as paired current observation plus
`LiteToolResult.error` by BrowserGym's own ingress/result handling.

**Difficulty:** Medium — BrowserGym is well-structured, but WebArena/VisualWebArena need Docker. MiniWoB is trivial (local HTML).

## Infrastructure

| Benchmark | Infrastructure | Deployment |
|---|---|---|
| MiniWoB | Local HTML files (`file://`) | `MINIWOB_URL` env var, zero setup |
| WebArena | 6–7 Docker containers (Shopping, Reddit, GitLab, Wikipedia, Map, Homepage) | `WA_*` env vars, ~32 GB RAM |
| VisualWebArena | 7–8 Docker containers (adds Classifieds) | `VWA_*` env vars |
| WorkArena | ServiceNow SaaS instance | External account required |
| WebLINX | Static traces | Single-step prediction only |
| TimeWarp | 1 container (`cua-lite/browsergym.timewarp`: Wiki / News / Shop Flask sites, one UI version) | Built by `install.sh timewarp`; `TW_*` set by the env |

## Implementation Priority

Work in priority order, but support as many benchmarks as possible — do not skip a benchmark unless you hit a genuine blocker (e.g., requires a SaaS account we don't have). The wrapper code is largely shared; each new benchmark is mostly just registration + config.

| Priority | Benchmark | Difficulty | Rationale |
|---|---|---|---|
| **P0** | MiniWoB | Low | Zero infra, fast reset, 100+ tasks, coordinate actions native. Start here. |
| **P1** | WebArena | Medium | Real-world web tasks, high eval value. Needs Docker services running but BrowserGym wraps them well. |
| **P1** | VisualWebArena | Medium | Same as WebArena + Classifieds. Adds multimodal tasks (images). |
| **P3** | WorkArena | High | Requires ServiceNow SaaS account — integrate if credentials are available, skip if not. |
| **P4** | WebLINX | Low but limited | `max_steps=1`, action prediction only, not interactive. Lowest priority but trivial to add. |
| **Done** | TimeWarp | Medium | Read-only Q&A over 3 self-hosted sites × 6 UI versions; registration + one container. |

## Design Decisions

### Action space — coordinate-based + browser navigation actions

BrowserGym's `HighLevelActionSet` supports composable subsets. We use `subsets=["coord", "chat", "infeas"]` for the BrowserGym side. Deliberately excluded:

- `"nav"` (`goto`, `go_back`, `go_forward`) — handled by CUA-Lite's own `goto()`/`back()` actions (see below)
- `"tab"` (`new_tab`, `tab_close`, `tab_focus`) — agent uses Ctrl+T, Ctrl+W, Ctrl+Tab
- `"bid"` (`click(bid)`, `fill(bid, value)`, etc.) — element-based, requires accessibility tree

**Why `back()` and `goto(url)` are needed:** Playwright screenshots only capture page content, not browser chrome (address bar, back button, tab bar). The agent literally cannot see or click these UI elements via coordinates. This is analogous to mobile's `system_button(button="Back")` and `open_app()` — platform actions for UI that's outside the agent's visual field. Tab management is excluded because Ctrl+T/W/Tab keyboard shortcuts work without seeing the tab bar.

**How `back()`/`goto()` are executed:** These do NOT go through BrowserGym's `HighLevelActionSet` (since we excluded `"nav"`). The wrapper calls Playwright directly: `page.go_back()` and `page.goto(url)`. Then calls `task.validate()` for reward as usual.

### Action space — `LiteBrowserActionSpace` + `extra_tools`

Browser envs use `LiteBrowserActionSpace` (desktop-coordinate actions registered under the browser platform) as their action space. `LiteBrowserNavToolSet` is the schema-only catalog for canonical browser-nav extra tools, and envs resolve selected names through the browser-nav schema resolver. `env_kwargs.extra_tools` selects names such as `back` and `goto`; the env resolves selected names into `env.metadata.extra_tool_schemas`, which adapters append to the rendered tool schema and dispatch in `_translate_action`. `valid_actions` narrows the action-batch tool and env-side unsupported-action feedback for canonical GUI actions, but it never exposes nav/finish extras.

Full action mapping:

| CUA-Lite action | BrowserGym function |
|---|---|
| `click(coordinate, button, clicks)` | `mouse_click(x, y, button)` / `mouse_dblclick(x, y)` |
| `type(text)` | `keyboard_type(text)` |
| `key(keys=["ctrl", "c"])` | `keyboard_press("Control+c")` |
| `key_down(keys)` / `key_up(keys)` | `keyboard_down(key)` / `keyboard_up(key)` |
| `mouse_move(coordinate)` | `mouse_move(x, y)` |
| `mouse_down(button)` / `mouse_up(button)` | `mouse_down(x, y, button)` / `mouse_up(x, y, button)` (use current cursor position) |
| `drag(start_coordinate, coordinate)` | `mouse_drag_and_drop(fx, fy, tx, ty)` |
| `scroll(coordinate, direction, amount)` | `mouse_move(x, y)` then `scroll(dx, dy)` (or bare `scroll(dx, dy)` when no coord) |
| `back()` | `go_back()` |
| `goto(url)` | `goto(url)` |
| `response(text)` | canonical persisted tool; BrowserGym boundary lowers this to `send_msg_to_user(text)` |
| `terminate(status, reason)` | `report_infeasible(reason)` |

Coordinate conversion: CUA-Lite `[0, 1000]` → pixel coordinates based on viewport (MiniWoB 498×321, WebArena 1280×720). The per-benchmark values in `_BENCHMARK_VIEWPORTS` must stay in sync with upstream `browsergym.{miniwob,webarena,visualwebarena}` — check `base.py`/`task.py` after any browsergym bump.

Note: BrowserGym's `fill(bid, value)` is element-based with no coordinate equivalent. In coordinate mode the agent does `click` to focus + `type` to input — this is the natural CUA flow.

### Reset = navigate, not rebuild infrastructure

BrowserGym's `BrowserEnv.reset()` restarts headless Chromium on every call — fine (~200-500ms, gives clean state). What must NOT happen is restarting Docker containers (WebArena's services). Docker services stay running across all episodes; `reset()` only navigates to the task's start URL and re-authenticates if needed. MiniWoB has no infrastructure — `reset()` is just `page.goto(url)` + `core.startEpisodeReal()`.

### Observation — image and/or text

BrowserGym can provide screenshots, AXTree/DOM-derived text, open-tabs/focused-element text, and labelled
error sections. A result may carry image, text, or both; action errors must keep the current observation
payload and set `LiteToolResult.error` separately.

### Sync → async bridging

BrowserGym and Playwright are synchronous APIs; CUA-Lite envs are async. The wrapper must bridge this with `asyncio.to_thread()` (or equivalent) for `reset()`, `step()`, and `close()`. This is critical for parallel rollouts — without it, one env blocks the entire event loop.

### Task IDs

`browsergym.<benchmark>:<task_name>` (e.g., `browsergym.miniwob:click-dialog`, `browsergym.webarena:410`). Each benchmark is a separate env namespace — independent registration, independent optional deps, can be added incrementally per phase.

### Optional dep

`browsergym` package. WebArena tasks additionally need Docker services running and `WA_*` / `VWA_*` env vars set.

## TimeWarp

Upstream: [sparklabutah/timewarp](https://github.com/sparklabutah/timewarp)
(sites + `browsergym-timewarp` task package, pinned at `e3b2b6d` / `0.2.1`).

- **One UI version per process.** Upstream `TimeWarpInstance` reads `TW_WIKI` /
  `TW_NEWS` / `TW_WEBSHOP` from `os.environ` when a task is constructed, so two
  versions cannot share one Python process without changing upstream. The
  version is deployment config (`server_kwargs.timewarp_version`,
  `TIMEWARP_VERSION`) and is recorded as `metadata.others.timewarp_version`.
  Per-task versions would need upstream to accept injected site URLs.
- **Container.** `docker/timewarp/` runs the three upstream Flask apps as
  `run_all_env.sh <version>` does. Two patches: Wiki/News bind `0.0.0.0`, and the
  Shop builds its lazy search engine before opening its port (otherwise
  concurrent first requests — e.g. readiness probes — each start an init).
  `start.sh` reuses `timewarp-<scope>` only when its version label and image ID
  match, so a version switch or rebuild recreates it.
- **Registration** reads `browsergym/timewarp/data/test.raw.json` without
  importing the package (its `__init__` downloads nltk data when missing); the
  import happens on the first reset. Splits follow BrowserGym's `timewarp.csv`,
  which `browsergym-experiments==0.14.3` does not ship yet: tasks with
  `additional_instructions` (TimeTraj plans) are `train`, the rest `eval`.
- **Action set.** `browsergym-core==0.14.3` predates the upstream `timewarp`
  action subset; the text+bid config spells it as
  `["bid", "chat", "infeas", "nav", "tab"]` narrowed by `extra_tools`.
- **Scoring** is upstream's evaluator. The two `llm_judge` tasks (32, 143)
  resolve their judge at reset (`resolve_judge`) so a missing key fails there
  instead of inside `validate()`.

Verification (arm64 macOS host, Docker Desktop): gold answers from each task's
`fuzzy_match` score 1.0 and a non-answer 0.0 through `gym.make` on Wiki (1),
News (33), Shop (54) and a three-site task (84); coordinate `click` / `type` /
`key` / `scroll`, `goto`, `back`, `new_tab`, `switch_tab`, `response` and
`terminate` were checked on the live sites, and a version switch (1 → 5)
recreated the container. Offline, with upstream's verifiers: 214 of the 229
deterministic tasks accept their gold and reject a non-answer; 15 negative-answer
tasks (45, 51, 62, 66, 80, 89, 124, 125, 127, 146, 157, 161, 167, 174, 189)
also accept "I could not find it." because their `must_include` lists any
"not …" phrasing.

## Verification

Must verify every action in `LiteBrowserActionSpace` actually takes effect — not just "no error", but the page state changes as expected (e.g., `type` inserts text, `click` triggers element reaction, `goto` navigates to URL, `scroll` changes scroll position). Write tests that cover all actions: `click`, `type`, `key`, `key_down`/`key_up`, `mouse_move`, `mouse_down`/`mouse_up`, `drag`, `scroll`, `back`, `goto` (the latter two dispatched via `extra_tools`), `response`, `terminate`.

Cursor verification follows the shared gate in
[AGENTS.md#411-cursor-rendering-verification](/devs/envs/AGENTS.md#411-cursor-rendering-verification).
For MiniWoB, compare direct mode and a cold env-server with
`cursor=True`/`cursor=False`. For WebArena and VisualWebArena, also repeat on a
singleton-backed server after the shared stack is available:

```bash
uv run python scripts/serve_env.py \
  --port 30100 \
  --env-ids browsergym.webarena \
  --token browsergym-smoke \
  --warm-singleton
```

Use `--env-ids browsergym.visualwebarena` for the VisualWebArena pass. Coord-mode
screenshots should paint the shared Linux cursor sprite only after a coord action
establishes cursor state; BID/text and SoM-style modes should stay raw rather
than carrying a stale cursor from the previous coord action.

```bash
# MiniWoB (requires MINIWOB_URL set)
uv run python - <<'PY'
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    splits = gym.registry.task_ids('browsergym.miniwob')
    task_ids = [tid for ids in splits.values() for tid in ids]
    print(f'Found {len(task_ids)} MiniWoB tasks')
    env = gym.make('browsergym.miniwob@click-dialog', max_steps=10)
    obs = await env.reset()
    print(f'Instruction: {obs.text}')
    result = await env.step([
        make_tool_call('computer', {'actions': [
            {'action': 'click', 'coordinate': [500, 500]}
        ]}, call_id='call_0000')
    ])
    print(f'Reward: {result.reward}, Terminated: {result.terminated}')
    await env.close()
asyncio.run(main())
PY
```
