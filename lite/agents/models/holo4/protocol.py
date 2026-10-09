"""Holo4 history compaction policy."""

from __future__ import annotations

import copy
import dataclasses

from lite.agents.core.protocol import BaseProtocol
from lite.core import LiteMessage
from lite.core.messages import (
    ASSISTANT_ROLE,
    IMAGE_PART,
    TEXT_PART,
    USER_ROLE,
    group_into_turns,
    peel_system_message,
)


@dataclasses.dataclass
class Holo4HistoryProtocol(BaseProtocol, key="holo4.history"):
    """Keep recent screenshot turns and durable assistant narration.

    Older screenshots, tool output, tool calls, and reasoning are evicted. The
    initial task instruction and ordinary assistant text survive compaction.
    """

    max_n_images: int = 4

    def __post_init__(self) -> None:
        if self.max_n_images < 0:
            raise ValueError("max_n_images must be non-negative")

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs,
    ) -> list[LiteMessage]:
        if not messages:
            return []
        messages = copy.deepcopy(messages)

        for message in messages:
            if message.get("role") == ASSISTANT_ROLE:
                message.pop("raw_response", None)
                message.pop("reasoning_content", None)

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
        compact_before = (
            min((location[0] for location in kept_images), default=len(turns))
            if self.max_n_images
            else 0
        )
        compacted: list[LiteMessage] = []
        instruction_kept = False
        for i, turn in enumerate(turns):
            if i >= compact_before:
                for observation_index, observation in enumerate(turn["observations"]):
                    parts = observation.get("content")
                    if not isinstance(parts, list):
                        compacted.append(observation)
                        continue
                    visible_parts = [
                        part
                        for part_index, part in enumerate(parts)
                        if not isinstance(part, dict)
                        or part.get("type") != IMAGE_PART
                        or (i, observation_index, part_index) in kept_images
                    ]
                    if visible_parts:
                        visible_observation = copy.deepcopy(observation)
                        visible_observation["content"] = visible_parts
                        compacted.append(visible_observation)
                if turn["assistant"] is not None:
                    compacted.append(turn["assistant"])
                continue

            if not instruction_kept:
                for observation in turn["observations"]:
                    if observation.get("role") != USER_ROLE:
                        continue
                    text_parts = [
                        part
                        for part in observation.get("content", [])
                        if isinstance(part, dict)
                        and part.get("type") == TEXT_PART
                        and part.get("text")
                    ]
                    if text_parts:
                        compacted.append({"role": USER_ROLE, "content": text_parts})
                        instruction_kept = True
                        break

            assistant = turn["assistant"]
            if assistant is None or assistant.get("role") != ASSISTANT_ROLE:
                continue
            text_parts = [
                part
                for part in assistant.get("content", [])
                if isinstance(part, dict) and part.get("type") == TEXT_PART and part.get("text")
            ]
            if text_parts:
                compacted.append({"role": ASSISTANT_ROLE, "content": text_parts})

        if system_message is not None:
            compacted.insert(0, system_message)
        return compacted


__all__ = ["Holo4HistoryProtocol"]
