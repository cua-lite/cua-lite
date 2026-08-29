"""UI-Venus-2 agents — one-step creation via AgentRegistry.

The wire-format fold (cua-lite ``LiteMessage`` →
``<think>…</think>\\n<action>…</action>``) lives in
:meth:`UIVenus2BaseAdapter._convert_message_to_agent`, so these classes carry no
body of their own. What they DO carry is
:meth:`Qwen3VLBaseAgent.build_generation_prompt`: ``inclusionAI/UI-Venus-2-9B``
reports ``model_type: "qwen3_5"`` and ships the Qwen3.5 chat template, whose
``<think>`` channel defaults ON. Every upstream UI-Venus-2 harness runs with
thinking OFF and reads the prompted ``<think>`` tag instead, so the flag has to
reach ``apply_chat_template`` — inheriting the Qwen base is how the Qwen3.5
family already does exactly that.

Usage::

    agent = AgentRegistry.get("ui_venus_2@desktop@use", processor=processor, generate_fn=fn)
    agent = AgentRegistry.get("ui_venus_2@browser@use", processor=processor, generate_fn=fn)
    agent = AgentRegistry.get("ui_venus_2@mobile@use", processor=processor, generate_fn=fn)
    # Native <think> instead of the prompted tag:
    agent = AgentRegistry.get(
        "ui_venus_2@desktop@use", processor=processor, generate_fn=fn,
        adapter_kwargs={"enable_thinking": True},
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from lite.agents.models.qwen3_vl.agent import Qwen3VLBaseAgent

# One agent class per platform: unlike UI-Venus-1.5, the three ``use`` grammars
# genuinely differ, so no ``(desktop|browser)`` regex is shared here.


@dataclass
class UIVenus2DesktopUseAgent(Qwen3VLBaseAgent, key="ui_venus_2@desktop@use"):
    """Desktop-OS GUI-use registry entry."""
    pass


@dataclass
class UIVenus2BrowserUseAgent(Qwen3VLBaseAgent, key="ui_venus_2@browser@use"):
    """Browser GUI-use registry entry."""
    pass


@dataclass
class UIVenus2MobileUseAgent(Qwen3VLBaseAgent, key="ui_venus_2@mobile@use"):
    """Mobile GUI-use registry entry."""
    pass


@dataclass
class UIVenus2GroundingPointAgent(
    Qwen3VLBaseAgent, key=r"ui_venus_2@(desktop|browser|mobile)@grounding\.point",
):
    """Point-grounding registry entry, shared by every platform."""
    pass


__all__ = [
    "UIVenus2BrowserUseAgent",
    "UIVenus2DesktopUseAgent",
    "UIVenus2GroundingPointAgent",
    "UIVenus2MobileUseAgent",
]
