"""Qwen3VL agent — one-step creation via AgentRegistry.

The agent subclass exists to forward the adapter's ``enable_thinking``
flag to ``processor.apply_chat_template(..., enable_thinking=...)``.
Qwen3-VL ships in two checkpoint flavours:

  * ``-Instruct``: chat_template suppresses ``<think>`` by default;
    keep ``enable_thinking=False`` (the default) for shorter outputs.
  * ``-Thinking``: chat_template prepends ``<think>\\n`` to the
    generation; pair with ``enable_thinking=True`` so the multi-hundred-
    token reasoning channel is honoured by the parser and the
    ``<tool_call>`` block survives within ``max_new_tokens``.

Usage::

    agent = AgentRegistry.get(
        "qwen3_vl@desktop@use",
        processor=processor,
        generate_fn=generate_fn,
        protocol_kwargs={"full_history_size": 4},
        # Override adapter kwargs if you want native <think>:
        adapter_kwargs={"enable_thinking": True},
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lite.agents.core.agent import AutoAdapterAgent
from lite.agents.types import AgentMessage
from lite.core import LiteMessage


#: The reply's own closing tag for the block ``build_generation_prompt`` opened.
_THINK_CLOSE = "</think>"


@dataclass
class Qwen3VLBaseAgent(
    AutoAdapterAgent,
    key=(
        r"qwen3_vl\.base"
        r"(@(desktop|browser|mobile)"
        r"@(use|understanding|grounding\.action|grounding\.point|grounding\.bbox))?"
    ),
):
    """Base class for Qwen3-VL agents. Forwards ``adapter.enable_thinking``
    to ``apply_chat_template`` so the ``<think>`` channel toggles correctly.

    Registered under a platform/task wildcard key to
    pair with :class:`Qwen3VLBaseAdapter`'s wildcard registration —
    yaml-driven rollouts that set ``agent_id: "qwen3_vl.base"`` resolve
    here (workflow-agnostic, no Action:/Thought: machinery). Concrete
    subclasses below carry platform/task-specific keys
    (``qwen3_vl@desktop@use`` etc.) for the ``use`` flavor.
    """

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Build prompt with ``enable_thinking`` forwarded from adapter."""
        if self.processor is None:
            raise RuntimeError("agent.processor is not set")
        enable_thinking = getattr(self.adapter, "enable_thinking", False)
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )

    def _parse_generation_response(
        self,
        response: str,
        *,
        call_id_start: int,
    ) -> tuple[AgentMessage, LiteMessage, str | None]:
        """Trim the prompt-supplied ``<think>`` half out of the replay sidecar.

        With ``enable_thinking`` on, ``build_generation_prompt`` above ends the
        prompt at an OPEN ``<think>``, so the model's reply is only the second
        half of that block: ``reasoning </think> answer``. The reasoning is
        already parsed out into ``reasoning_content``, and the chat template
        rebuilds the block from that field when it renders the turn as history.
        Replaying the whole reply as the message CONTENT would therefore emit
        the reasoning twice over -- once inside the template's block, once after
        it -- and leave the reply's own ``</think>`` with no opening tag.

        So the sidecar keeps only what belongs in ``content``. Tool-call markup
        is untouched, which is what the byte-exact replay exists to preserve.
        """
        out_agent_message, out_lite_message, error = super()._parse_generation_response(
            response, call_id_start=call_id_start
        )
        raw = out_lite_message.get("raw_response")
        reasoning = out_lite_message.get("reasoning_content")
        if raw and reasoning and _THINK_CLOSE in raw["text"]:
            raw["text"] = raw["text"].split(_THINK_CLOSE, 1)[-1].lstrip("\n")
            raw["reasoning_content"] = reasoning
        return out_agent_message, out_lite_message, error


# Desktop and browser share one agent class per task type — the
# (desktop|browser) regex makes registry lookups under either platform resolve
# to the same class. Mobile is its own platform with different action space +
# protocol.

@dataclass
class Qwen3VLDesktopUseAgent(
    Qwen3VLBaseAgent, key=r"qwen3_vl@(desktop|browser)@use"
):
    """Desktop/browser GUI-use registry entry."""
    pass


@dataclass
class Qwen3VLDesktopGroundingActionAgent(
    Qwen3VLBaseAgent, key=r"qwen3_vl@(desktop|browser)@grounding\.action"
):
    """Desktop/browser action-grounding registry entry."""
    pass


@dataclass
class Qwen3VLDesktopGroundingPointAgent(
    Qwen3VLBaseAgent, key=r"qwen3_vl@(desktop|browser)@grounding\.point"
):
    """Desktop/browser point-grounding registry entry."""
    pass


@dataclass
class Qwen3VLMobileUseAgent(Qwen3VLBaseAgent, key="qwen3_vl@mobile@use"):
    """Mobile GUI-use registry entry."""
    pass


@dataclass
class Qwen3VLMobileGroundingActionAgent(Qwen3VLBaseAgent, key="qwen3_vl@mobile@grounding.action"):
    """Mobile action-grounding registry entry."""
    pass


@dataclass
class Qwen3VLMobileGroundingPointAgent(Qwen3VLBaseAgent, key="qwen3_vl@mobile@grounding.point"):
    """Mobile point-grounding registry entry."""
    pass
