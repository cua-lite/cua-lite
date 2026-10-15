"""Holo4 history compaction policy."""

from __future__ import annotations

import copy
import dataclasses

from lite.agents.core.protocol import BaseProtocol
from lite.core import LiteMessage
from lite.core.messages import IMAGE_PART, TEXT_PART, group_into_turns, peel_system_message
from lite.core.messages.content import require_message_list

# Stands in for each evicted screenshot. The official harness keeps the
# evicted observation turn and swaps its image for this text: per-step
# prompt-token deltas in the official AndroidWorld trajectories grow by a
# constant few tokens per eviction rather than shrinking by a whole
# ``<tool_response>`` turn, and the wording is the one the official OSWorld
# trajectories show in place of evicted images.
EVICTED_SCREENSHOT_TEXT = "[Image omitted by context cleaning]"


@dataclasses.dataclass
class Holo4HistoryProtocol(BaseProtocol, key="holo4.history"):
    """Evict screenshots beyond a sliding window; keep every message.

    Mirrors the official Holo4 harness, which retains the last few
    screenshots — three on AndroidWorld and OSWorld, four on OSWorld 2 —
    while instructions, observations, tool results, and assistant turns
    (tool calls and reasoning included) survive verbatim. Each
    out-of-window image part is replaced by
    :data:`EVICTED_SCREENSHOT_TEXT`, so every observation turn, including
    an image-only tool result, keeps its place in the history.
    """

    max_n_images: int = 3

    def __post_init__(self) -> None:
        if self.max_n_images < 0:
            raise ValueError("max_n_images must be non-negative")

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs,
    ) -> list[LiteMessage]:
        require_message_list(messages, where="process_messages")
        if not messages:
            return []
        messages = copy.deepcopy(messages)

        system_message, content = peel_system_message(messages)
        turns = group_into_turns(content)
        image_locations: list[tuple[int, int, int]] = []
        for turn_index, turn in enumerate(turns):
            for observation_index, observation in enumerate(turn["observations"]):
                parts = observation.get("content")
                if not isinstance(parts, list):
                    continue
                image_locations.extend(
                    (turn_index, observation_index, part_index)
                    for part_index, part in enumerate(parts)
                    if isinstance(part, dict) and part.get("type") == IMAGE_PART
                )

        if self.max_n_images > 0 and len(image_locations) <= self.max_n_images:
            return messages

        kept_images = set(image_locations[-self.max_n_images :]) if self.max_n_images else set()
        compacted: list[LiteMessage] = []
        for turn_index, turn in enumerate(turns):
            for observation_index, observation in enumerate(turn["observations"]):
                parts = observation.get("content")
                if isinstance(parts, list):
                    observation["content"] = [
                        part
                        if not isinstance(part, dict)
                        or part.get("type") != IMAGE_PART
                        or (turn_index, observation_index, part_index) in kept_images
                        else {"type": TEXT_PART, "text": EVICTED_SCREENSHOT_TEXT}
                        for part_index, part in enumerate(parts)
                    ]
                compacted.append(observation)
            if turn["assistant"] is not None:
                compacted.append(turn["assistant"])
        if system_message is not None:
            compacted.insert(0, system_message)
        return compacted


__all__ = ["EVICTED_SCREENSHOT_TEXT", "Holo4HistoryProtocol"]
