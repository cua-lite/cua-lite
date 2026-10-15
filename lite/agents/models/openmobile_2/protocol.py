"""OpenMobile-2 history protocol (``openmobile_2.history``)."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from lite.agents.core.protocol import BaseProtocol
from lite.core import LiteMessage
from lite.core.messages import IMAGE_PART, TEXT_PART, USER_ROLE, message_has_image
from lite.core.messages.content import require_message_list

#: Official ``QWEN35_SESSION_USER_PROMPT``
#: (OpenMobile2-Code/runtimes/prompts.py:600) — the first user message's
#: text: the instruction given once, trailing newline included.
SESSION_USER_PROMPT = "The user query: {instruction}\n"

#: Official collapse placeholder (COLLAPSED_SCREENSHOT_TEXT in
#: OpenMobile2-Code/AndroidWorld/android_world/agents/session/agent.py).
COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."


@dataclasses.dataclass
class OpenMobile2HistoryProtocol(BaseProtocol, key="openmobile_2.history"):
    """Official session history: every message and all text survive; only
    screenshots beyond a trailing window collapse to placeholder text.

    Mirrors the official ``qwen35_thought_session`` runtime (``--last_n 3``
    in every published eval command): the most recent ``last_n``
    image-bearing messages keep their screenshots; each older screenshot
    is replaced by one ``"This screenshot has been collapsed."`` text part
    — non-image parts first, then the placeholders — so the message slot
    and its text stay in the history. The first user message is
    additionally rewritten to the official shape: query text (``"The user
    query: {instruction}\\n"``) first, screenshot after.
    """

    last_n: int = 3

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs: Any,
    ) -> list[LiteMessage]:
        require_message_list(messages, where="process_messages")
        if not messages:
            return []
        messages = copy.deepcopy(messages)
        self._rewrite_first_user_message(messages)
        self._collapse_old_images(messages)
        return messages

    def _rewrite_first_user_message(self, messages: list[LiteMessage]) -> None:
        """Official first-user shape: query text first, screenshot after.

        cua-lite builds ``[image, text]``; the official session sends the
        prefixed query text before the screenshot. Metadata side-channel
        parts ride along after and are stripped later by
        ``keep_model_visible_content``.
        """
        for message in messages:
            if message.get("role") != USER_ROLE:
                continue
            content = message.get("content")
            if isinstance(content, list):
                first_text = next(
                    (
                        part
                        for part in content
                        if isinstance(part, dict) and part.get("type") == TEXT_PART
                    ),
                    None,
                )
                if first_text is not None:
                    message["content"] = [
                        {
                            "type": TEXT_PART,
                            "text": SESSION_USER_PROMPT.format(
                                instruction=first_text.get("text", "")
                            ),
                        },
                        *[part for part in content if part is not first_text],
                    ]
            return

    def _collapse_old_images(self, messages: list[LiteMessage]) -> None:
        """Collapse screenshots outside the trailing ``last_n`` window in place."""
        # Official semantics: keep the last max(1, last_n) image-bearing
        # MESSAGES (message-level, not per-image); everything older with an
        # image gets collapsed, message slots and text survive untouched.
        image_bearing = [
            i for i, message in enumerate(messages) if message_has_image(message)
        ]
        kept = set(image_bearing[-max(1, int(self.last_n)):])
        for i, message in enumerate(messages):
            if i in kept or not message_has_image(message):
                continue
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = self._collapse_image_parts(content)

    @staticmethod
    def _collapse_image_parts(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Official collapse order: non-image parts first, then one
        placeholder text part per dropped screenshot."""
        parts = [
            part
            for part in content
            if not (isinstance(part, dict) and part.get("type") == IMAGE_PART)
        ]
        n_images = len(content) - len(parts)
        return [
            *parts,
            *(
                {"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT}
                for _ in range(n_images)
            ),
        ]


__all__ = ["OpenMobile2HistoryProtocol"]
