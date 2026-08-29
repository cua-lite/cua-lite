# `lite/agents/models/`

`lite/agents/models/` contains the model-family runtimes. A model family turns a
canonical Lite sample into a provider/model request, then turns the response back
into canonical Lite tool calls or final answers.

This README is a user-facing index. It explains the runtime shapes, registry
keys, family files, and public model ids. Keep detailed deviation notes next to
the affected family code, not in this package front door.

## Quick Start

Most callers create agents through the factory:

```python
from lite.agents.factory import make

# Local/open-weight model: caller supplies processor + generation function.
agent = make(
    "Qwen/Qwen3-VL-8B-Instruct",
    env=env,
    processor=processor,
    generate_fn=generate_fn,
)
sample = await agent.sample(env, max_steps=20)

# API model: provider runtime is self-contained.
agent = make("gpt-5.5", env=env)
sample = await agent.sample(env, max_steps=20)
```

`make()` maps a public model id from [`lite/agents/factory.py`](/lite/agents/factory.py)
to an `AgentRegistry` key:

```text
<family>@<platform>@<task_type>
```

For example, a desktop `use` env with `Qwen/Qwen3-VL-8B-Instruct` resolves to
`qwen3_vl@desktop@use`.

## Runtime Shapes

There are two model runtime shapes.

### Local Adapter Runtime

Open-weight and locally served models use the adapter stack:

- `agent.py`: registers `AutoAdapterAgent` classes and high-level family
  selection.
- `adapter.py`: renders Lite messages into the family prompt format, prepares
  images, and parses raw model text.
- `action_space.py`: owns provider-visible tool schemas and canonical/native
  action conversion.
- `protocol.py`: owns history layout, truncation, and summary formatting when
  the family needs custom history policy.
- `utils/*`: family-local implementation helpers.

These agents need a `processor` and `generate_fn`, normally supplied by rollout,
serving, or training launchers. Use `sample()` for live rollout. Use `predict()`
for focused adapter tests or custom one-step callers.

### API Agent Runtime

GPT, Claude and Gemini are self-contained `BaseAgent` subclasses. Their `agent.py` files
own provider calls, native computer/tool-use surfaces, response parsing,
tool-result feedback, and the rollout loop. They intentionally do not have local
`adapter.py` files because there is no HuggingFace chat-template or local
generation boundary to adapt.

Use `sample()` for live rollout. Their `predict()` methods are not the primary
rollout interface.

## Package Map

- `agent.py`
  Registry classes and high-level runtime selection. Required for
  rollout-facing families.

- `adapter.py`
  Prompt rendering, image handling, raw response parsing, and family message
  conversion. Local adapter families only.

- `action_space.py`
  Provider-visible action/tool schemas and canonical conversion. Present for
  families with a native dialect.

- `protocol.py`
  History windows, summaries, truncation, and prompt-history policy. Present
  only when the family needs custom protocol behavior.

- `utils/*`
  Small family-local helpers that do not deserve public entry status.

The first-screen path should stay predictable. Model-family directories should
normally expose only `action_space.py`, `adapter.py`, `agent.py`, `protocol.py`,
`__init__.py`, and `utils/*`.

## Task Types

| Task type | Meaning |
| --- | --- |
| `use` | Multi-turn GUI operation against an environment. |
| `understanding` | Direct question answering over GUI state or screenshots. |
| `grounding.point` | Single-step point prediction. |
| `grounding.action` | Single-step canonical action prediction. |
| `grounding.bbox` | Single-step box prediction. |

Not every family supports every task type. The registry surfaces below describe
what is wired in this package.

## Model Family Index

Factory ids are the public model ids accepted by
[`lite.agents.factory.make()`](/lite/agents/factory.py). Brace notation is only a
compact way to show groups of exact ids. Specialist rows without factory ids can
still be reached directly through `AgentRegistry` or `AgentAdapterRegistry` when
their registry key is present.

### Local Adapter Families

- `qwen3_vl`
  IDs: `Qwen/Qwen3-VL-{2B,4B,8B,32B}-{Instruct,Thinking}`.
  Surfaces: desktop/browser/mobile `use`, `grounding.action`, `grounding.point`,
  plus adapter rows for `understanding` and `grounding.bbox`.
  Files: `agent.py`, `adapter.py`, `action_space.py`, `protocol.py`.

- `qwen3_5`
  IDs: `Qwen/Qwen3.5-{2B,4B,9B,27B}`.
  Surfaces: desktop/browser/mobile `use`, `grounding.action`, `grounding.point`,
  plus adapter rows for `understanding` and `grounding.bbox`.
  Files: `agent.py`, `adapter.py`, `action_space.py`, `protocol.py`.

- `qwen3_8`
  ID: `Qwen/Qwen3.8-27B`.
  Surfaces: desktop/browser/mobile `use`, `grounding.action`, `grounding.point`,
  plus adapter rows for `understanding` and `grounding.bbox`.
  Files: `agent.py`, `adapter.py`, `action_space.py`.
  Same Qwen3.5 architecture and XML tool_call wire format, so the adapter
  stack is inherited; the family delta is the expanded desktop `computer_use`
  enum (19 values, `call_user` replacing `answer`) that OSWorld's
  `mm_agents/qwen` harness declares. Mobile keeps the Qwen3.5 surface.

- `qwen2_5_vl`
  IDs: `Qwen/Qwen2.5-VL-{3B,7B}-Instruct`.
  Surfaces: desktop/browser/mobile `use`, `grounding.action`, `grounding.point`,
  plus adapter rows for `understanding` and `grounding.bbox`.
  Files: `agent.py`, `adapter.py`, `action_space.py`.

- `fara`
  ID: `microsoft/Fara-7B`.
  Surfaces: desktop/browser `use`, `grounding.action`, `grounding.point`.
  Files: `agent.py`, `adapter.py`, `action_space.py`, `protocol.py`.

- `ui_tars`
  ID: `ByteDance-Seed/UI-TARS-7B-DPO`.
  Surfaces: desktop/browser/mobile `use`, plus adapter rows for `understanding` and
  `grounding.bbox`.
  Files: `agent.py`, `adapter.py`, `action_space.py`, `protocol.py`.

- `ui_tars_15_v1`
  ID: `ByteDance-Seed/UI-TARS-1.5-7B`.
  Surfaces: desktop/browser/mobile `use`, plus adapter rows for `understanding` and
  `grounding.bbox`.
  Files: `agent.py`, `adapter.py`, `action_space.py`.

- `evocua`
  ID: `meituan/EvoCUA-8B-20260105`.
  Surfaces: desktop/browser `use`, `grounding.action`, `grounding.point`, plus
  adapter rows for `understanding` and `grounding.bbox`.
  Files: `agent.py`, `adapter.py`, `action_space.py`.

- `mai_ui`
  IDs: `Tongyi-MAI/MAI-UI-{2B,8B}`.
  Surfaces: mobile `use` and desktop/browser/mobile `grounding.point`.
  Files: `agent.py`, `adapter.py`, `action_space.py`, `protocol.py`.

- `step_gui`
  ID: `stepfun-ai/GELab-Zero-4B-preview`.
  Surface: mobile `use`.
  Files: `agent.py`, `adapter.py`, `action_space.py`, `protocol.py`.

- `ui_venus_2`
  ID: `inclusionAI/UI-Venus-2-9B`.
  Surfaces: desktop `use`, browser `use`, mobile `use`, and
  desktop/browser/mobile `grounding.point`.
  One checkpoint, four upstream harnesses with genuinely different grammars, so
  each `use` platform is its own registry key rather than a shared
  `(desktop|browser)` row: desktop spells points as `box=` and has
  `Sequence(actions=[...])` (this family's spelling of the canonical `computer`
  action batch), browser spells them `point=` with `Scroll(point, direction)`,
  and mobile adds device buttons plus a dedicated `Answer` reply verb. The wire
  is one `<think>...</think>` + `<action>...</action>` block per turn, with the
  task baked into the system prompt; grounding answers with a bare `[x,y]` list
  and `[-1,-1]` for a trained refusal. `ui_venus_2.history` keeps all assistant
  text and evicts only screenshots; its `n_history_images` matches upstream's
  `N_IMG`, which counts PAST screenshots (the current one is always sent on top).
  Built on Qwen3.5, so the agent inherits `Qwen3VLBaseAgent` to forward
  `enable_thinking` to `apply_chat_template`.
  Files: `agent.py`, `adapter.py`, `action_space.py`, `protocol.py`.

### API Agent Families

- `gpt`
  IDs: `gpt-5.5`, `gpt-5.6-sol`.
  Surfaces: desktop/browser/mobile `use` and desktop/browser `grounding.point`.
  Files: `agent.py`, `action_space.py`, `utils/*`.

- `claude`
  IDs: `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`.
  Surfaces: desktop/browser/mobile `use` and desktop/browser `grounding.point`.
  Files: `agent.py`, `action_space.py`, `utils/*`.

- `gemini`
  IDs: `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`.
  Surfaces: desktop/browser/mobile `use`. No `grounding.point`.
  Configs ship for the pure-GUI desktop rows only: rows needing `extra_tools`
  are blocked upstream, and the browser rows fail for a separate reason — see [/devs/agents/api/gemini.md](/devs/agents/api/gemini.md).
  Files: `agent.py`, `action_space.py`, `utils/*` (including `utils/transport.py`
  — this family posts native `generateContent` over httpx instead of litellm,
  because litellm has no representation for the `computerUse` tool and the
  OpenAI-compatible path drops it).

### Utility or Reserved Families

- `lite`
  Canonical pass-through adapter for replay/export. No factory id.
  File: `adapter.py`.

Recommended rollout defaults live under `scripts/configs/<family>/default/`.

## Registry Keys

The public factory id is not the registry key. The factory chooses a family id,
then composes it with env metadata:

```text
factory model id -> family id -> <family>@<platform>@<task_type>
```

Examples:

- `Qwen/Qwen3-VL-8B-Instruct` + desktop `use` env ->
  `qwen3_vl@desktop@use`;
- `gpt-5.5` + mobile `use` env -> `gpt@mobile@use`;
- `ByteDance-Seed/UI-TARS-1.5-7B` + grounding point row ->
  `ui_tars_15_v1@<platform>@grounding.point`.

Some configs override `agent_id` to select a more specific registry family, such
as a `.base` adapter row. Keep that override in rollout config; do not encode
env-specific branching inside the factory.

## Adding or Changing a Model

1. Decide whether the family is local-adapter or API-agent runtime.
2. Add or update the family registry classes in `agent.py`.
3. For local models, wire `adapter.py`, `action_space.py`, and `protocol.py`
   only when the family needs those surfaces.
4. Add a `LOCAL_AGENTS` or `API_AGENTS` row in
   [`lite/agents/factory.py`](/lite/agents/factory.py) only when the model id
   should be accepted by `make()`.
5. Add or update rollout defaults under `scripts/configs/<family>/default/` when
   the model is meant to be run through the standard scripts.
6. Add focused render/parse/action-space tests before changing prompts, tool
   schemas, or response parsing.
7. Update this index when public factory ids, runtime shape, or registered
   surfaces change.

## Documentation Style

Keep this README stable and user-facing:

- describe public ids, registry surfaces, and first-screen files;
- avoid rollout score notes, temporary deltas, or reference-run analysis;
- keep family quirks short and move detailed implementation rationale into the
  family module or tests;
- avoid adding files just to make a family match its neighbors.

If a model family differs because the provider/model wire format differs, keep
that difference visible. If a difference is only helper placement or historical
accident, simplify it in code rather than documenting it as a public feature.
