"""Qwen3.8 agent — one-step creation via AgentRegistry.

Forwards ``adapter.enable_thinking`` and ``reasoning_effort`` to the
processor's chat template, retaining the model's native ``xhigh`` default.

Qwen3.8 ships thinking ON by default with ``reasoning_effort`` defaulting to
``xhigh``, so the default stays OFF here for the same reason as Qwen3.5:
multi-hundred-token reasoning inflates prompt size and can exhaust
``max_new_tokens`` before the ``<tool_call>`` block closes.

Usage::

    agent = AgentRegistry.get(
        "qwen3_8@desktop@use",
        processor=processor,
        generate_fn=generate_fn,
        # Override adapter kwargs if you want native <think>:
        adapter_kwargs={"enable_thinking": True},
        protocol_kwargs={"history_n": 100},
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from lite.agents.models.qwen3_5.agent import Qwen3_5BaseAgent


@dataclass
class Qwen3_8BaseAgent(
    Qwen3_5BaseAgent,
    key=(
        r"qwen3_8\.base"
        r"(@(desktop|browser|mobile)"
        r"@(use|understanding|grounding\.action|grounding\.point|grounding\.bbox))?"
    ),
):
    """Base class for Qwen3.8 agents.

    Registered under a platform/task wildcard key to pair with
    :class:`Qwen3_8BaseAdapter`'s wildcard registration — yaml-driven rollouts
    that set ``agent_id: "qwen3_8.base"`` resolve here (workflow-agnostic, no
    ``Action:`` machinery). Concrete subclasses below carry the
    platform/task-specific keys.
    """

    reasoning_effort: Literal["low", "medium", "xhigh"] = "xhigh"

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Render Qwen3.8's native thinking mode and reasoning effort."""
        if self.processor is None:
            raise RuntimeError("agent.processor is not set")
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.adapter.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        )


# Desktop and browser share one agent class per task type — the
# (desktop|browser) regex makes registry lookups under either platform resolve
# to the same class. Browser nav is an env extra_tool, not a per-platform
# agent. Mobile is its own platform with a different action space.

@dataclass
class Qwen3_8DesktopUseAgent(Qwen3_8BaseAgent, key=r"qwen3_8@(desktop|browser)@use"):
    """Desktop/browser GUI-use registry entry."""


@dataclass
class Qwen3_8DesktopGroundingActionAgent(
    Qwen3_8BaseAgent, key=r"qwen3_8@(desktop|browser)@grounding\.action"
):
    """Desktop/browser action-grounding registry entry."""


@dataclass
class Qwen3_8DesktopGroundingPointAgent(
    Qwen3_8BaseAgent, key=r"qwen3_8@(desktop|browser)@grounding\.point"
):
    """Desktop/browser point-grounding registry entry."""


@dataclass
class Qwen3_8MobileUseAgent(Qwen3_8BaseAgent, key="qwen3_8@mobile@use"):
    """Mobile GUI-use registry entry."""


@dataclass
class Qwen3_8MobileGroundingActionAgent(Qwen3_8BaseAgent, key="qwen3_8@mobile@grounding.action"):
    """Mobile action-grounding registry entry."""


@dataclass
class Qwen3_8MobileGroundingPointAgent(Qwen3_8BaseAgent, key="qwen3_8@mobile@grounding.point"):
    """Mobile point-grounding registry entry."""
