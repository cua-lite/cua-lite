"""UI-Venus-2 agents — one-step creation via AgentRegistry.

The wire-format fold (cua-lite ``LiteMessage`` →
``<think>…</think>\\n<action>…</action>``) lives in
:meth:`UIVenus2BaseAdapter._convert_message_to_agent`, so these classes carry no
body of their own. What they DO carry is
:meth:`Qwen3VLBaseAgent.build_generation_prompt`: ``inclusionAI/UI-Venus-2-9B``
reports ``model_type: "qwen3_5"`` and ships the Qwen3.5 chat template, so the
flag has to reach ``apply_chat_template`` — inheriting the Qwen base is how the
Qwen3.5 family already does exactly that.

``enable_thinking`` defaults to ``True`` here and should stay that way. On this
checkpoint it does not select a reasoning channel; it picks a chat-template
branch. ``False`` emits a CLOSED ``<think>\n\n</think>`` before generation,
which leaves the model nowhere to write the ``<think>`` block every UI-Venus-2
prompt's ``### Output Format`` section demands. See
:attr:`UIVenus2BaseAdapter.enable_thinking` for the measured cost.

Usage::

    agent = AgentRegistry.get("ui_venus_2@desktop@use", processor=processor, generate_fn=fn)
    agent = AgentRegistry.get("ui_venus_2@browser@use", processor=processor, generate_fn=fn)
    agent = AgentRegistry.get("ui_venus_2@mobile@use", processor=processor, generate_fn=fn)
    # Adapter fields are passed at the top level, NOT under ``adapter_kwargs``:
    agent = AgentRegistry.get(
        "ui_venus_2@desktop@use", processor=processor, generate_fn=fn,
        enable_thinking=False,   # closed <think>; see the warning above
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
