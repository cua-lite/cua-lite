See [AGENTS.md](/devs/envs/AGENTS.md) for shared requirements.

**Reference implementation:**
- `${CUA_LITE_REFERENCES_ROOT}/webgym` — `main` branch is the unmodified upstream reference (read-only). If you need to modify WebGym (patch action parsing, fix bugs, add features), create a `cua-lite` branch off `main` and work there. Keep `main` clean so we can diff against upstream and pull updates.
- **Paper:** https://arxiv.org/html/2601.02439v3 — *WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks* (Bai et al., Microsoft/UIUC/CMU)

**Status:** implemented design notes. Current source of truth is `lite/gym/envs/webgym/main.py`; keep this file
aligned with that implementation rather than treating older snippets below as a second spec.

**Goal:**
Wrap [WebGym](https://github.com/microsoft/webgym) as a cua-lite gym environment using CUA-Lite coordinate
actions plus env-owned standalone extras such as `back`, `goto`, `response`, and `terminate`. Runtime
invalid/unsupported feedback is returned from `step()` as paired current observation plus `LiteToolResult.error`,
by the WebGym env's own ingress/result handling.

**Difficulty:** Medium — WebGym is well-structured with a clean HTTP API, but requires the OmniBoxes server stack (Master → Node → Instance + Redis) and wiring up the VLM evaluator. The entire stack runs **inside a single self-contained Docker container** (built by `install.sh build`); the cua-lite wrapper is a thin HTTP client.

## Infrastructure

The OmniBoxes stack runs **inside one shared Docker container** (built once via
`install.sh build`). The env-server starts a single long-lived backend container per
env-server (`webgym-<server_port>`) on first use and publishes its Master URL into the
`WEBGYM_MASTER_URL` env var the host wrapper reads. Container-internal components:

| Component | Role | Container Port |
|---|---|---|
| Redis | Instance pool state (available/in_use sets) | 6379 |
| Instance Servers | One per Playwright browser instance | 9000+ |
| Node Server | Manages instance pool on one machine | 8080 |
| Master Server | Load balancer / API gateway | 7000 |

**Deployment:** build the image once, then let the env-server manage the container:

```bash
uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh build   # build cua-lite/webgym:latest
```

The cua-lite wrapper connects to the Master Server via HTTP. Configure via env vars:

| Env var | Default | Description |
|---|---|---|
| `WEBGYM_MASTER_URL` | published by the env-server's Services | Master server URL (set by `ensure`) |
| `WEBGYM_API_KEY` | `default_key` | API key for `x-api-key` header |

## Design Decisions

### Action space — coordinate-based + browser navigation actions

WebGym supports both coordinate-based and Set-of-Marks (element ID) actions. We use **coordinate mode only** — consistent with CUA-Lite's vision-based approach.

**Why `back()` and `goto(url)` are needed:** Playwright screenshots only capture page content, not browser chrome (address bar, back button, tab bar). The agent literally cannot see or click these UI elements via coordinates. This is analogous to mobile's `system_button(button="Back")` and `open_app()` — platform actions for UI that's outside the agent's visual field.

### Action space — `LiteBrowserActionSpace` + `extra_tools`

Browser envs use `LiteBrowserActionSpace` (desktop-coordinate browser actions) as their action space. `LiteBrowserNavToolSet` is the schema-only catalog for canonical browser-nav extra tools, and envs resolve selected names through the browser-nav schema resolver. `env_kwargs.extra_tools` selects names such as `back` and `goto`; the env resolves selected names into `env.metadata.extra_tool_schemas`, which adapters append to the rendered tool schema and dispatch in `_translate_action`. `valid_actions` narrows the action-batch tool and env-side unsupported-action feedback for canonical GUI actions, but it never exposes nav/finish extras. Full action mapping:

| CUA-Lite action | WebGym HTTP command |
|---|---|
| `click(coordinate, button, clicks)` | `click_coords(x, y)` |
| `type(text)` | `fill_coords(x, y, value, press_enter, delete_existing=True)` — `press_enter` is derived from a trailing newline in `text` and stripped from `value`; uses the last `click` focus point unless `coordinate` is present |
| `key(keys=["ctrl", "c"])` | `keypress(keys=["Control", "c"])` |
| `scroll(coordinate, direction, amount)` | `hover_and_scroll_coords(x, y, direction, amount)` or `page_down`/`page_up` with scaled pixel amount |
| `mouse_move(coordinate)` | `hover_coords(x, y)` |
| `drag(start_coordinate, coordinate)` | `click_coords(end_x, end_y)` (no native drag in WebGym coord mode) |
| `back()` | `back()` |
| `goto(url)` | `visit_page(url)` |
| `response(text)` | Store as agent answer for evaluation |
| `terminate(status, reason)` | End episode, trigger evaluation |

Coordinate conversion: CUA-Lite `[0, 1000]` → pixel coordinates. WebGym viewport is 1280×720 by default (configurable).

**Key differences from BrowserGym mapping:**
- `type(text)` maps to WebGym's `fill_coords` which does click + type in one call. The wrapper uses the last click focus point unless the action carries a coordinate.
- `scroll` uses `hover_and_scroll_coords` for element-aware scrolling (finds nearest scrollable container). Falls back to `page_down`/`page_up` for full-page scroll.
- WebGym host input is Lite canonical key vocabulary; the wrapper projects it to
  Playwright wire names (`ctrl` -> `Control`, `esc` -> `Escape`, etc.) before HTTP.

### Environment lifecycle — HTTP client, not subclass

Unlike BrowserGym (which we wrap by subclassing), WebGym is wrapped via **HTTP API calls**. The env holds an HTTP client and a leased instance ID:

```python
class WebGymEnv(LiteBaseEnv):
    def __init__(self, task_config, master_url, api_key, ...):
        self.client = WebGymClient(master_url, api_key)
        self.instance = None  # Allocated on reset()

    async def reset(self):
        if self.instance:
            await self.client.reset(self.instance)
        self.instance = await self.client.get_instance()
        await self.client.execute(self.instance, {"visit_page": {"url": task.website}})
        screenshot = await self.client.screenshot(self.instance, mode="coordinates")
        return LiteEnvObservation(image=screenshot, text=task.task_name)

    async def step(self, actions):
        results = []
        for action in actions:
            # The env owns support checks and execution feedback. Unsupported or
            # failed calls still get a LiteToolResult paired to the call id,
            # carrying the current observation plus LiteToolResult.error.
            results.append(await self._execute_or_feedback(action))
        reward, terminated = await self._maybe_evaluate(actions)
        return LiteEnvStepResult(
            results=results,
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={},
        )

    async def close(self):
        if self.instance:
            await self.client.reset(self.instance)  # Release back to pool
            self.instance = None
```

**Important:** `reset()` leases an instance from the pool and navigates to the task URL. `close()` releases the instance back to the pool (via `/reset`). Instances are reusable — no browser restart needed between episodes.

### HTTP client — async with httpx

Use `httpx.AsyncClient` for non-blocking HTTP calls. This is critical for parallel rollouts — `requests` would block the event loop.

```python
class WebGymClient:
    def __init__(self, master_url: str, api_key: str):
        self.client = httpx.AsyncClient(base_url=master_url, headers={"x-api-key": api_key})

    async def get_instance(self) -> dict:
        r = await self.client.post("/get", params={"lifetime_mins": 120})
        return r.json()  # {"instance_id": "uuid:port", "node": "hash"}

    async def execute(self, instance: dict, command: dict):
        await self.client.post("/execute", params=instance, json=command)

    async def screenshot(self, instance: dict, mode: str = "coordinates") -> Image:
        r = await self.client.get("/screenshot", params={**instance, "interaction_mode": mode})
        return Image.open(io.BytesIO(r.content))

    async def reset(self, instance: dict):
        await self.client.post("/reset", params=instance)
```

### LiteToolResult — observation plus error

WebGym provides screenshot and page text. Results carry the current observation payload the env can produce
(`image` and/or `text`) plus a separate `error` field when a tool failed. Error text must not replace the
observation payload.

### Evaluation — VLM-based judge

WebGym's evaluation is VLM-based (multi-criterion: blocking detection → anti-hallucination → fact verification). We **reuse WebGym's evaluator directly** — import from the WebGym reference and call at episode end. Requires OpenAI API key. The evaluator needs:
- All screenshots from the trajectory
- The agent's final answer (from `response(text)` action)
- The task's `evaluator_reference` rubrics and `reference_answer`

### Task IDs

`webgym:<task_id>` (e.g., `webgym:0`, `webgym:1`, ...). Task data loaded from `microsoft/webgym_tasks` HuggingFace dataset.

### Max steps by difficulty

| Difficulty | Max steps |
|---|---|
| 1 (easy) | 12 |
| 2 (medium) | 18 |
| 3 (hard) | 25 |

### Optional dep

`httpx` for async HTTP client. The OmniBoxes server runs in the shared backend container (built by `install.sh build`); the env-server's Services starts it and sets `WEBGYM_MASTER_URL` automatically.

## Verification

Must verify the full action → screenshot → evaluate loop works end-to-end:

1. **Action execution**: Every coordinate-based action produces the expected browser effect (click triggers element, type inserts text, scroll moves viewport, back navigates history).
2. **Screenshot fidelity**: Screenshots match viewport size, are valid PNGs, and reflect action effects.
3. **Instance lifecycle**: Lease → use → release cycle works without leaks. Parallel instances don't interfere.
4. **Evaluation**: VLM judge produces correct rewards for known-good and known-bad trajectories.

Cursor verification follows the shared gate in
[AGENTS.md#411-cursor-rendering-verification](/devs/envs/AGENTS.md#411-cursor-rendering-verification).
For WebGym, run the smoke in direct mode, a fresh cold env-server, and a
singleton-backed env-server. Use both `gym.make(..., cursor=True)` and
`gym.make(..., cursor=False)`; after a `mouse_move` or `click`, the true path
must show the shared Linux cursor sprite from the container capture endpoint,
while the false path remains raw. Rebuild the `cua-lite/webgym:latest` image
before this smoke whenever `docker/patches/` or screenshot code changed.

```bash
# Requires the cua-lite/webgym:latest image built (install.sh build); the env-server starts the backend container (see Infrastructure above)
uv run python -c "
import asyncio, lite.gym as gym
async def main():
    splits = gym.registry.task_ids('webgym')
    task_ids = [tid for ids in splits.values() for tid in ids]
    print(f'Found {len(task_ids)} WebGym tasks')
    env = gym.make('webgym@0', max_steps=10)
    obs = await env.reset()
    print(f'Instruction: {obs.text}')
    result = await env.step([
        {
            'id': 'call_0000',
            'type': 'function',
            'function': {
                'name': 'computer',
                'arguments': {
                    'actions': [{'action': 'click', 'coordinate': [500, 500]}],
                },
            },
        },
    ])
    print(f'Reward: {result.reward}, Terminated: {result.terminated}')
    await env.close()
asyncio.run(main())
"
```
