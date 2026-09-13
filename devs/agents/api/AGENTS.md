# API Agent Guidelines (Claude / GPT / Gemini)

See [shared guidelines](/devs/agents/AGENTS.md) for the full development workflow. This doc covers API-specific setup.

## Prerequisites

- No GPU needed.
- Set env vars for your provider:
  - **Claude:** `ANTHROPIC_API_KEY`, optionally `ANTHROPIC_BASE_URL`
  - **GPT:** `OPENAI_API_KEY`
  - **Gemini:** none — this family does not use liteLLM. Pass `api_base` and
    `api_key` as `agent_kwargs`; see [gemini.md](/devs/agents/api/gemini.md).

## Architecture

API agents inherit directly from `BaseAgent` — no adapter, no processor, no generate_fn. All API logic is self-contained:

- `ClaudeDesktopUseAgent` → `litellm.acompletion` (Chat Completions API)
- `GPTDesktopUseAgent` → `litellm.aresponses` (Responses API)
- `GeminiDesktopUseAgent` → native `generateContent` REST over httpx (**no
  liteLLM**: it has no representation for the `computerUse` tool, so the
  OpenAI-compatible path drops it, taking `environment` with it)

liteLLM routes by model string prefix:

| Model string | Provider |
|---|---|
| `claude-opus-4-6` | Anthropic (direct) |
| `vertex_ai/claude-opus-4-6` | Google Vertex AI |
| `bedrock/claude-sonnet-4-...` | AWS Bedrock |
| `gpt-6-astra` | Azure OpenAI deployment ID |
| `gpt-5.5` | OpenAI |
| `gemini-3.6-flash` | Google (native REST, not routed by liteLLM) |

The prefix is the *model string handed to liteLLM*, which is the `API_AGENTS`
**key**. `--model-id` is `choices=list(AGENTS)` — a closed list — so a prefixed
string is only usable after it is added as an `API_AGENTS` key (see **Launch
Entry**); passing an unregistered `vertex_ai/...` to `scripts/rollout.py` exits 2.

## Files to Create

```
lite/agents/models/<agent>/action_space.py
lite/agents/models/<agent>/agent.py
tests/agents/models/<agent>/test_<agent>_action_space.py
tests/agents/models/<agent>/test_<agent>_agent.py
```

## Launch Entry

Add the model ID to the `API_AGENTS` dict in `lite/agents/factory.py`:

```python
API_AGENTS = {
    "claude-opus-4-6":   {"agent_id": "claude"},
    "gpt-6-astra":       {"agent_id": "gpt"},
    "gpt-5.5":           {"agent_id": "gpt"},
    "gpt-5.6-sol":       {"agent_id": "gpt"},
}
```

Agent creation is automatic via `agents.make` (mirrors `gym.make`):

```python
import lite.agents as agents
agent = agents.make("claude-opus-4-6", env=env)

# Or explicit via registry — register_all() first, or the registry is empty and
# .get() raises KeyError ("Available: none"). agents.make calls it for you.
from lite.agents.bootstrap import register_all
from lite.agents.models import AgentRegistry

register_all()
agent = AgentRegistry.get("claude@desktop@use", model_id="claude-opus-4-6", metadata=env.metadata)
agent = AgentRegistry.get("gpt@desktop@use", model_id="gpt-5.5", metadata=env.metadata)
```

## Verification

```bash
uv run python scripts/rollout.py --model-id <model_id> --env-id <ENV> --head 1 --env-kwargs '{"max_steps": 3}'
```

`--task-id ID` pins a single task; `--head 1` (above) runs the first task as a quick smoke test. Omit both to run all tasks. See **Key flags** below for the full list.

## Stress Test

```bash
uv run python scripts/rollout.py \
    --model-id <model_id> --env-id <ENV> \
    --config-path scripts/configs/<agent>/default/<env>.yaml \
    --sample 32 --concurrency 4 --env-kwargs '{"max_steps": 15}'
```

> **Always pass `--config-path scripts/configs/<agent>/default/<env>.yaml`.** Each agent's rollout config pins API-side sampling kwargs (`max_output_tokens`, `temperature`, `reasoning_effort` for OpenAI; `max_tokens`, `thinking_budget`, `effort` for Claude; `max_output_tokens`, `thinking_level` for Gemini), env `extra_tools` name selectors, agent/system-prompt overrides, and env defaults (`max_steps`, `post_action_delay`, etc.). Without `--config-path` (and absent `--api-kwargs`), `scripts/rollout.py` does not inject any sampling override — the agent runs with whatever the provider's API endpoint defaults to (Claude / GPT differ), which is rarely what the eval matrix expects. Browse `scripts/configs/{claude, gpt, ...}/default/{osworld, webgym, ...}.yaml` for the relevant per-agent / per-env YAMLs.

Key flags (`scripts/rollout.py`):
- `--model-id ID` — model ID (**required**; `choices=list(AGENTS)`, no default). Provider-prefixed strings like `vertex_ai/claude-...` are only accepted once registered as an `API_AGENTS` key
- `--env-id ENV` — environment (required)
- `--task-id ID` — pin a single task (sample mode); omit to run all tasks
- `--sample N` — randomly sample N tasks (omit to run all)
- `--head N` — keep first N tasks only
- `--splits S [S ...]` — splits to evaluate (default: all)
- `--prompt-data FILE` — parquet task list (overrides `--head`/`--sample`)
- `--group-size G` — rollouts per task (default: 1)
- `--concurrency C` — max parallel envs (default: 16; pass a lower value such as `--concurrency 4` for API models to stay under provider rate limits)
- `--config-path FILE` — YAML config providing base env kwargs
- `--api-kwargs JSON` — provider-native API params (default: `None`, falls through to provider's API default). e.g. `'{"max_tokens": 8192, "temperature": 0.5}'`
- `--env-kwargs JSON` — env overrides (default: `{}` — the env's own defaults apply)
