See [AGENTS.md](/devs/agents/local/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# Fara-1.0

**Goal:** Develop the Fara-7B web-browsing workflow (Microsoft's Qwen2.5-VL-7B fine-tune). Function calling via `<tool_call>`; free-form `thoughts` prose precedes each tool call.

**Supported environments:** primary web browsing (`webgym`, `webharbor.webvoyager`, `online_mind2web`, `browsergym.{miniwob,webarena,visualwebarena}`). Also wired for single-step grounding as best-effort off-distribution coverage (`screenspot_pro`, `osworld_g`). Multi-step desktop control (`lite.osworld`, `osworld`) is **not** supported because Fara's web-native actions (`visit_url` / `history_back`) have no desktop equivalent. Configs: [`scripts/configs/fara/default/`](/scripts/configs/fara/default/).

**Reference implementation:** use a local checkout of the Fara reference code; the
path varies by machine.
- `<fara-reference-root>/src/fara/_prompts.py` — `FaraComputerUse` tool schema + `get_computer_use_system_prompt`
- `<fara-reference-root>/src/fara/qwen_helpers/fncall_prompt.py` — `NousFnCallPrompt` + `FN_CALL_TEMPLATE`
- `<fara-reference-root>/src/fara/fara_agent.py` — history (`maybe_remove_old_screenshots`), parsing (`_parse_thoughts_and_action`), coordinate rescale

**Agent slug:** `fara`

**Launch:**
```bash
uv run python scripts/rollout.py \
  --model-id microsoft/Fara-7B \
  --model-path cua-lite/Fara-7B \
  --env-id webgym \
  --config-path scripts/configs/fara/default/webgym.yaml \
  --head 1
```

**Serving (SGLang) — use `cua-lite/Fara-7B`:** the upstream `microsoft/Fara-7B` `config.json` is transformers-4.x-style, keeping `rope_theta` as a top-level field separate from `rope_scaling`. Under transformers v5 + SGLang, `get_rope_config` reads the **top-level** `Qwen2_5_VLConfig.rope_parameters` (not `text_config`'s), which omits `rope_theta`, so the server dies at load with `KeyError: 'rope_theta'`. [`cua-lite/Fara-7B`](https://huggingface.co/cua-lite/Fara-7B) is a config-only fork (weights byte-identical; `rope_theta` folded into `rope_scaling`). Keep `--model-id microsoft/Fara-7B` (the `LOCAL_AGENTS` key that selects the `fara` adapter + processor) and override only the weights/config path:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_sglang.py \
  --model-id microsoft/Fara-7B --model-path cua-lite/Fara-7B --port 30088
```

## What is reused from Qwen2.5-VL

Fara is a Qwen2.5-VL-7B fine-tune, so `FaraBaseAdapter` subclasses `Qwen2_5VLBaseAdapter` and inherits nearly the entire pipeline unchanged:

- **`<tool_call>{json}</tool_call>` text serialization** — Qwen2.5-VL's chat_template drops structured `tool_calls`, so they are serialized into the assistant `content` text. Fara's wire format (`thoughts` prose + one `<tool_call>` block) is exactly this base behavior — there is **no** `Action:` / `Thought:` navigation format, so Fara subclasses the *base*, not the Use adapter.
- **`smart_resize` factor=28** (patch 14 × merge 2), pixel-in-resized coordinate space, and the `{display_width_px}x{display_height_px}` substitution + [0,1000]↔pixel rescaling (`render_step` caches `_current_image_size`).
- **`parse_raw_assistant_response`** — same `<tool_call>` extraction, no `<think>` channel.

## What differs

### 1. Action space — `FaraDesktopActionSpace`

The `computer_use` enum is the 11-action Fara web-browsing form. Versus the Qwen2.5-VL desktop enum:

| Action | Fara | cua-lite Qwen2.5-VL |
|---|:---:|:---:|
| key / type / mouse_move / left_click / scroll | Y | Y |
| wait / terminate | Y | Y |
| left_click_drag / right_click / middle_click / double_click | - | Y |
| **visit_url** | **Y** | - |
| **web_search** | **Y** | - |
| **history_back** | **Y** | - |
| **pause_and_memorize_fact** | **Y** | - |

- `type` carries the `press_enter` / `delete_existing_text` args (the reference runs `include_input_text_key_args=True`).
- **Conversion:** `visit_url` ↔ cua-lite `goto` and `history_back` ↔ `back` map onto the canonical browser-nav verbs; `web_search` → `goto` of the Bing results URL (the env has no `web_search` verb; matches the reference which visits Bing); `pause_and_memorize_fact` stays in assistant thoughts and is not emitted as an env action. `type` decomposes to `click(coordinate)` + `type(text)` (Fara's `type` carries a coordinate to focus the field); its `press_enter` becomes the trailing newline canonical uses to spell Enter. `left_click` is Fara's only click surface; non-left and multi-click behavior belongs in adapter tests, not env-side repair. Fara has no native `response` action; terminate thoughts remain assistant content and are not synthesized into a canonical `response`.

### 2. System prompt — `FN_CALL_TEMPLATE`

`_build_tools_section` is overridden to emit the reference `FN_CALL_TEMPLATE` "web automation agent … Critical Points" preamble instead of the Qwen `# Tools` header, still prefixed with `"You are a helpful assistant."` — reproducing the reference `NousFnCallPrompt` system message byte-for-byte.

### 3. smart_resize cap

`smart_resize_max_pixels = 16384·28·28` (Fara's full `MAX_PIXELS`), vs the Qwen2.5-VL sibling's SFT-trimmed 2M. `min_pixels = 56·56 = 3136` is already the cua-lite default (matches Fara's `MIN_PIXELS`).

### 4. History — `FaraHistoryProtocol`

Keep **all** text (every prior thought / action / observation), but only the newest `max_n_images=3` screenshots; older image parts are stripped (text preserved). Mirrors `FaraAgent.maybe_remove_old_screenshots`. Unlike `Qwen3VLHistoryProtocol`, older turns are **not** collapsed into a rolling text summary.

## Files

```
lite/agents/models/fara/{action_space,adapter,agent,protocol,__init__}.py
tests/agents/models/fara/test_fara_action_space.py   # action space + conversions
tests/agents/models/fara/test_fara_adapter.py         # adapter render/round-trip/history
tests/agents/models/fara/test_fara_agent.py           # agent registry + factory wiring
scripts/configs/fara/default/*.yaml              # per-env rollout configs
```

## Adapters (registered keys)

- `fara.base` — workflow-agnostic base (`FaraBaseAdapter`).
- `fara@(desktop|browser)@use` — multi-turn navigation (`FaraHistoryProtocol`, `max_n_images=3`).
- `fara@(desktop|browser)@grounding.action` — SFT-replay, full schema, single turn.
- `fara@(desktop|browser)@grounding.point` — single-step click for ScreenSpot-Pro. Fara has **no** native grounding mode, so this subclasses `Qwen2_5VLBaseAdapter` directly (standard `# Tools` header + grounding rules block, trimmed `left_click` schema) — NOT the web-automation `FN_CALL_TEMPLATE`.

## Rollout configs — key hyperparameters (from `webeval/scripts`)

- `max_n_images: 3` (`protocol_kwargs`) and `temperature: 0` — the reference `FaraAgent` defaults.
- `max_steps: 100` on web benchmarks — Fara's reference `max_rounds` default. Desktop configs use the cua-lite desktop norm (50; no Fara desktop reference).
- `resolution: null` / unset — do not agent-side stretch screenshots. Fara's
  reference browser viewport was 1440×900, but `agent_kwargs.resolution` is an
  exact image resize, not an env viewport setting; stretching non-1440×900
  screenshots hurts small text/icons. If an env supports a browser viewport and
  you want 1440×900, set that on `env_kwargs.viewport` instead.
- `env_kwargs.extra_tools` is the only source for env extra-tool schemas. Fara browser configs that need nav or finish acceptance must list `goto`/`back`/`response`/`terminate` explicitly; `valid_actions` stays GUI-only and must not persist Fara-native nav/finish names in canonical metadata.

Launch entry: `microsoft/Fara-7B` in `LOCAL_AGENTS` ([`lite/agents/factory.py`](/lite/agents/factory.py)).
