"""OpenMobile-2 agent registry entries."""

from __future__ import annotations

import dataclasses

from lite.agents.models.qwen3_5.agent import Qwen3_5BaseAgent


@dataclasses.dataclass
class OpenMobile2MobileUseAgent(
    Qwen3_5BaseAgent,
    key="openmobile_2@mobile@use",
):
    """OpenMobile-2 mobile GUI-use registry entry.

    Inherits ``Qwen3VLBaseAgent.build_generation_prompt``: forwards
    ``enable_thinking=False`` to ``processor.apply_chat_template`` and
    never passes ``tools=`` — the official ``qwen35_thought_session``
    shape, where the ``mobile_use`` schema lives in the system prompt.
    """


__all__ = ["OpenMobile2MobileUseAgent"]
