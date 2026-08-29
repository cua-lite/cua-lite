"""Agent configs and factory for CUA-Lite agents.

Usage::

    from lite.agents.factory import make, AGENTS, LOCAL_AGENTS, API_AGENTS

    # Unified factory — works for local VLM and API agents
    # Local (pass processor + generate_fn):
    agent = make("Qwen/Qwen3-VL-8B-Instruct",
                       env=env, processor=processor, generate_fn=generate_fn)

    # API:
    agent = make("claude-opus-4-6", env=env)
"""

from __future__ import annotations

from typing import Any

from lite.utils.registry import compose_key

# =============================================================================
# Local VLM agent configs
# Key is the HuggingFace model id (used for processor loading + model serving).
# Only model-intrinsic properties live here — per-rollout overrides
# (protocol_kwargs, sampling_kwargs) belong in scripts/configs/<agent>/default/*.yaml,
# and per-model defaults live on the adapter class (default_factory).
#
# Fields:
#   agent_id:      registry slug; combined with env metadata dims to form the agent key
#   engine_kwargs: sglang Engine kwargs (used by scripts/serve_sglang.py and local serving)
#   backends:      allowed inference backends; omit to allow all (["sglang", "hf"])
# =============================================================================

# Reusable per-row fragments (spread into individual entries below).
_NO_THINK = {"agent_kwargs": {"enable_thinking": False}}
def _tp(n: int) -> dict: return {"engine_kwargs": {"tp_size": n}}

LOCAL_AGENTS: dict[str, dict] = {
    # Qwen3-VL ``-Instruct`` checkpoints ship with chat_template's ``<think>``
    # suppressed by default; pair with ``enable_thinking=False`` (the default
    # on :class:`Qwen3VLBaseAdapter`).
    "Qwen/Qwen3-VL-2B-Instruct": {"agent_id": "qwen3_vl"},
    "Qwen/Qwen3-VL-4B-Instruct": {"agent_id": "qwen3_vl"},
    "Qwen/Qwen3-VL-8B-Instruct": {"agent_id": "qwen3_vl"},
    "Qwen/Qwen3-VL-32B-Instruct": {"agent_id": "qwen3_vl", **_tp(2)},
    # Qwen3-VL ``-Thinking`` checkpoints prepend ``<think>\n`` to the
    # generation by default. We pin ``enable_thinking=False`` here to match
    # the cua-lite eval matrix's default of "no native thinking channel"
    # — this lets a Thinking checkpoint be eval'd as a drop-in replacement
    # for its Instruct sibling with identical wire format. Override per-run
    # via ``--agent-kwargs '{"enable_thinking": true}'`` to opt back in.
    "Qwen/Qwen3-VL-2B-Thinking": {"agent_id": "qwen3_vl", **_NO_THINK},
    "Qwen/Qwen3-VL-4B-Thinking": {"agent_id": "qwen3_vl", **_NO_THINK},
    "Qwen/Qwen3-VL-8B-Thinking": {"agent_id": "qwen3_vl", **_NO_THINK},
    "Qwen/Qwen3-VL-32B-Thinking": {"agent_id": "qwen3_vl", **_NO_THINK, **_tp(2)},
    # Qwen2.5-VL uses its own dialect — the wire format is **pixel coords in
    # the smart-resized image space** (factor=28, max_pixels=12.85M) with a
    # dynamic per-image ``{W}x{H}`` system prompt, and the ``computer_use``
    # enum is the 11-action upstream form (no triple_click / hscroll / answer
    # — those are Qwen3-VL additions). See
    # ``lite/agents/models/qwen2_5_vl/adapter.py`` for details.
    "Qwen/Qwen2.5-VL-3B-Instruct": {"agent_id": "qwen2_5_vl"},
    "Qwen/Qwen2.5-VL-7B-Instruct": {"agent_id": "qwen2_5_vl"},
    # Fara-1.0 — Microsoft's Qwen2.5-VL-7B fine-tune for web browsing. Reuses
    # the Qwen2.5-VL dialect machinery (factor=28 smart_resize, pixel-in-resized
    # coords, dynamic ``{W}x{H}`` system prompt) but with the Fara web action
    # enum (adds visit_url / web_search / history_back / pause_and_memorize_fact)
    # and the ``FN_CALL_TEMPLATE`` web-automation system prompt. See
    # ``lite/agents/models/fara/adapter.py``.
    "microsoft/Fara-7B": {"agent_id": "fara"},
    # Qwen3.5 — natively multimodal (no ``-VL`` suffix). XML tool_call wire
    # format; same DesktopActionSpace schema as Qwen3-VL. Qwen3.5's
    # chat_template defaults ``<think>`` to ON, so we explicitly pin
    # ``enable_thinking=False`` for the eval matrix (matches the YAML
    # configs that pair with these entries). Override per-run via
    # ``--agent-kwargs '{"enable_thinking": true}'`` to opt back in.
    "Qwen/Qwen3.5-2B": {"agent_id": "qwen3_5", **_NO_THINK},
    "Qwen/Qwen3.5-4B": {"agent_id": "qwen3_5", **_NO_THINK},
    "Qwen/Qwen3.5-9B": {"agent_id": "qwen3_5", **_NO_THINK},
    "Qwen/Qwen3.5-27B": {"agent_id": "qwen3_5", **_NO_THINK, **_tp(2)},
    # Qwen3.8 — Qwen3.5 architecture (``config.json`` reports
    # ``model_type: "qwen3_5"``) and the same XML tool_call chat_template, but
    # served through OSWorld's expanded ``mm_agents/qwen`` harness: 19
    # ``computer_use`` action values with ``call_user`` replacing ``answer``.
    # See ``lite/agents/models/qwen3_8/action_space.py``. Thinking is ON by
    # default in this checkpoint (``reasoning_effort`` defaults to ``xhigh``),
    # so pin it off for the eval matrix as with Qwen3.5.
    "Qwen/Qwen3.8-27B": {"agent_id": "qwen3_8", **_NO_THINK, **_tp(2)},
    # Other open-weight families.
    "ByteDance-Seed/UI-TARS-7B-DPO": {"agent_id": "ui_tars"},
    "ByteDance-Seed/UI-TARS-1.5-7B": {"agent_id": "ui_tars_15_v1"},
    "meituan/EvoCUA-8B-20260105": {"agent_id": "evocua"},
    "Tongyi-MAI/MAI-UI-2B": {"agent_id": "mai_ui"},
    "Tongyi-MAI/MAI-UI-8B": {"agent_id": "mai_ui"},
    "stepfun-ai/GELab-Zero-4B-preview": {"agent_id": "step_gui"},
    # UI-Venus-2 — inclusionAI's general-purpose GUI agent, built on Qwen3.5.
    # One checkpoint serves four harnesses, and the desktop / browser / mobile
    # ``use`` grammars genuinely differ (``Click(box=)`` + ``Sequence`` vs
    # ``Click(point=)`` + ``Scroll(point, direction)`` vs the Android verbs), so
    # the family registers a separate key per platform instead of sharing one
    # ``(desktop|browser)`` row.
    #
    # Deliberately NOT paired with ``_NO_THINK``, unlike the Qwen rows: on this
    # checkpoint's template ``enable_thinking=False`` emits a CLOSED
    # ``<think>\n\n</think>`` before generation, which leaves the model no room
    # for the ``<think>`` block its own prompt's "### Output Format" section
    # demands. Measured on lite.osworld: 95% of turns came back as a bare
    # ``<action>`` with no reasoning at all. See
    # ``UIVenus2BaseAdapter.enable_thinking``.
    "inclusionAI/UI-Venus-2-9B": {"agent_id": "ui_venus_2"},
}

# =============================================================================
# API agent configs
# Each entry maps a model ID to:
#   agent_id: registry slug; combined with env metadata dims to form the agent key
# =============================================================================

API_AGENTS: dict[str, dict] = {
    # Claude (Anthropic)
    "claude-opus-4-8": {"agent_id": "claude"},
    "claude-opus-4-7": {"agent_id": "claude"},
    "claude-opus-4-6": {"agent_id": "claude"},
    "claude-sonnet-4-6": {"agent_id": "claude"},
    # GPT (OpenAI) — requires native computer tool (GPT-5.4+)
    "gpt-5.5": {"agent_id": "gpt"},
    "gpt-5.6-sol":{"agent_id":"gpt"},
    # Gemini (Google) — native generateContent computer use
    "gemini-3.6-flash": {"agent_id": "gemini"},
    "gemini-3.5-flash": {"agent_id": "gemini"},
    "gemini-3.5-flash-lite": {"agent_id": "gemini"},
}

AGENTS: dict[str, dict] = {**LOCAL_AGENTS, **API_AGENTS}


def make(
    model_id: str,
    *,
    env: Any,
    agent_id: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create an agent from a model ID in :data:`AGENTS`.

    For API agents (Claude, GPT), ``model_id`` is automatically forwarded as
    the agent's ``model_id`` field. Callers do not need to pass it separately.

    Args:
        model_id: Key in AGENTS (e.g. ``"Qwen/Qwen3-VL-8B-Instruct"`` or
            ``"claude-opus-4-6"``).
        env: CUA-Lite environment (provides metadata).
        agent_id: If set, replaces the family-level default from
            :data:`AGENTS` when composing the registry key. Yaml-driven
            rollouts use this to opt into a different adapter family for
            the same model checkpoint — e.g. BrowserGym text+bid configs
            set ``agent_id: "qwen3_vl.base"`` to resolve to the
            workflow-agnostic :class:`Qwen3VLBaseAdapter` instead of the
            env-metadata-composed ``qwen3_vl@browser@use`` default.
        **kwargs: Forwarded to AgentRegistry.get().
            Local agents: ``processor`` and ``generate_fn``.
            API agents: ``api_kwargs``, etc.
    """
    from lite.agents.bootstrap import register_all
    from lite.agents.models import AgentRegistry

    if model_id not in AGENTS:
        raise KeyError(f"Unknown model '{model_id}'. Available: {list(AGENTS)}")
    register_all()
    agent_id = agent_id or AGENTS[model_id]["agent_id"]

    meta = env.metadata
    # This is the canonical site that composes the env metadata ``@`` dims of
    # an agent/adapter key (a few siblings reuse the same
    # rule: /lite/train/rollout/core/engine.py, /lite/train/export/export_sft.py,
    # /lite/agents/extensions/browsergym/goal_image.py). Not every ``@``-dim is composed
    # here, though — the action-space ``@point``/``@bbox`` format dim is fixed by
    # the adapter, not the env. Any ``.`` modifier already on ``agent_id`` (e.g.
    # ``qwen3_vl.base``) binds to the name and rides ahead of the ``@`` suffix.
    # See the key-grammar section in ``lite.utils.registry.BaseRegistry``.
    agent_key = compose_key(agent_id, *meta.dims)

    # Auto-set model_id for API agents (ClaudeDesktopUseAgent, GPTDesktopUseAgent).
    if model_id in API_AGENTS:
        kwargs.setdefault("model_id", model_id)

    # Forward the env's metadata as the SINGLE source of env-specific
    # hints. Adapters that inherit from ``BaseAgentAdapter`` get
    # ``self.metadata`` and read ``extra_tool_schemas`` / ``valid_actions`` /
    # ``others.<env-specific key>`` from it via properties.
    #
    # Env is authoritative — yaml configures the env (env_kwargs), the
    # env then derives its own metadata (including ``valid_actions``
    # and ``extra_tool_schemas``) from that config. E.g. browsergym in
    # text-only mode (use_screenshot=false, action_subsets=["bid"])
    # auto-sets ``metadata.valid_actions=[]`` so the agent's coord-
    # action wrapper is dropped. factory.py stays generic — no
    # env-specific knowledge here.
    kwargs.setdefault("metadata", meta)

    return AgentRegistry.get(agent_key, **kwargs)
