See [AGENTS.md](/devs/agents/local/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# EvoCUA

**Goal:** Develop the EvoCUA-8B workflow. S2 mode only (function calling via `<tool_call>`); S1 (code generation) is out of scope.

**Supported environments:** `lite.demo`, `lite.osworld`, `osworld`, `webgym`

**Reference implementation:**
- `${CUA_LITE_REFERENCES_ROOT}/EvoCUA/mm_agents/evocua/evocua_agent.py`
- `${CUA_LITE_REFERENCES_ROOT}/EvoCUA/mm_agents/evocua/prompts.py`
- `${CUA_LITE_REFERENCES_ROOT}/EvoCUA/run_multienv_evocua.py`

**Branch / Worktree:** `evocua` · `../cua-lite-evocua/`

**Agent slug:** `evocua`

**Launch:**
```bash
uv run python scripts/rollout.py --model-id meituan/EvoCUA-8B-20260105 --head 1
```

## What can be reused from Qwen3VL

EvoCUA S2 shares the same protocol as Qwen3VL: `<tool_call>` XML wrapping `computer_use` JSON, `Action:` prefix, `Qwen3VLHistoryProtocol` with `protocol_kwargs={"full_history_size": 4}`, `smart_resize` with `factor=32`, and the same user instruction template (`"Please generate the next move ..."`). EvoCUA does not produce `<think>` reasoning output (unlike Qwen3VL-Thinking), but the shared parser handles both cases. These components can be reused directly:

- `Qwen3VLBaseAdapter` — system prompt builder, `_build_tools_section`, `parse_raw_assistant_response`, `convert_sample_to_agent`
- `Qwen3VLHistoryProtocol` — window-based history with previous-actions text summary
- `smart_resize` with `factor=32`

## What differs: action space

The action enum differs. EvoCUA needs its own `EvoCUADesktopActionSpace`.

Three-way comparison (enum values actually sent to the model):

| Action | EvoCUA S2 ref | OSWorld Qwen3VL ref | cua-lite Qwen3VL |
|---|:---:|:---:|:---:|
| key | Y | Y | Y |
| type | Y | Y | Y |
| mouse_move | Y | Y | Y |
| left_click | Y | Y | Y |
| left_click_drag | Y | Y | Y |
| right_click | Y | Y | Y |
| middle_click | Y | Y | Y |
| double_click | Y | Y | Y |
| triple_click | Y | - | Y |
| scroll | Y | Y | Y |
| hscroll | - | - | Y |
| wait | Y | Y | Y |
| terminate | Y | Y | schema-gated wire action |
| answer | - | - | schema-gated wire alias for canonical `response` |
| **key_down** | **Y** | - | - |
| **key_up** | **Y** | - | - |

Notes:
- OSWorld Qwen3VL ref mentions `triple_click`/`hscroll`/`answer` in description text but omits them from the enum.
- EvoCUA S2 ref mentions `hscroll`/`answer` in description text but omits them from the enum.
- In the Lite contract, `answer` is qwen wire spelling only. Persisted canonical finish is `response`/`terminate`,
  and those wire actions are rendered only when the saved metadata contains the corresponding
  `extra_tool_schemas`.
- `key_down`/`key_up` are unique to EvoCUA — used for stateful holds (e.g. hold Shift while clicking). Need new CUA-lite conversion logic (map to `key` with `direction="down"`/`"up"`, or pass through).

## What differs: behavior

| Area | EvoCUA S2 ref | cua-lite |
|---|---|---|
| `type` handling | Expands text char-by-char into individual `pyautogui.press()` calls | Passes text as-is (env handles execution) |
| `terminate` status | Distinguishes `status="failure"` → `FAIL` vs `"success"` → `DONE` | Already supported in `Qwen3VLDesktopActionSpace` |
| Description prompt | Omits "You do not have access to a terminal or applications menu" | Includes it (from OSWorld ref) — minor, keep as-is |
| Context overflow | Auto-decrements `max_history_turns` and retries | No agent-level retry |
| API backend | Direct OpenAI API calls (vLLM) | Decoupled `generate_fn` |
| Scroll | Raw wheel-notch counts to `pyautogui.scroll()` | Same counts (`SCROLL_WIRE_UNIT = 1`); only a value >= 100 is read as screen units |
