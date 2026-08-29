"""UI-Venus-2 history protocol (``ui_venus_2.history``).

Every UI-Venus-2 harness builds the same history — ``build_messages`` in
``${CUA_LITE_REFERENCES_ROOT}/UI-Venus@UI-Venus-2/models/computer/computer_example.py``
docstrings it as *"Build system + all assistant text + the last n_img history
images"*, and the mobile and browser harnesses do the same thing::

    image_start = max(0, len(history) - n_img)
    for index, turn in enumerate(history):
        content = image_content(turn["image"], "History Screenshot:\\n") \\
            if index >= image_start and n_img > 0 else ""
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": turn["accepted_response"]})
    messages.append({"role": "user", "content": image_content(current_image, ...)})

Two properties of that loop are the whole protocol:

1. **Assistant text is never dropped.** Only screenshots are evicted, so the
   model keeps its full reasoning trail however long the episode runs.
2. **``n_img`` counts HISTORY screenshots, not all of them.** ``history``
   excludes the current observation, which is appended afterwards and always
   carries its image. The shipped ``N_IMG=2`` therefore sends **three** images:
   two historical plus the current one.

That second point is why this family owns a protocol instead of reusing
:class:`~lite.agents.models.fara.protocol.FaraHistoryProtocol`, whose
``max_n_images`` looks equivalent but budgets the total. Sharing the body would
have meant sharing a field whose UNIT differs by one, which is exactly the kind
of split this repo's contract rules exist to prevent.
"""

from __future__ import annotations

import copy
import dataclasses

from lite.agents.core.protocol import BaseProtocol
from lite.core import (
    LiteMessage,
)
from lite.core.messages import message_has_image, peel_system_message
from lite.core.messages.content import require_message_list


@dataclasses.dataclass
class UIVenus2HistoryProtocol(BaseProtocol, key="ui_venus_2.history"):
    """Full text history, newest-``n_history_images`` screenshots kept.

    Attributes:
        n_history_images: How many PAST screenshots survive, matching upstream's
            ``N_IMG`` (default 2 in ``scripts/computer.sh``,
            ``scripts/mobile.sh`` and ``venus_browser.py``'s two-turn deque).
            The current observation's screenshot is always sent on top of these,
            so the rendered prompt holds ``n_history_images + 1`` images.
            ``0`` reproduces upstream's text-only history.
    """

    n_history_images: int = 2

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs,
    ) -> list[LiteMessage]:
        """Keep every message; strip screenshots older than the image window.

        An evicted observation keeps its message (and any ``role:"tool"`` result
        text) and loses only its image parts, so the user/assistant alternation
        upstream sends stays intact — the adapter renders the leftover as an
        empty user bubble.
        """
        require_message_list(messages, where="process_messages")
        if len(messages) == 0:
            return []

        messages = copy.deepcopy(messages)
        system_message, content = peel_system_message(messages)

        # Walk newest-first. The first image-bearing observation is the current
        # screenshot, which upstream appends unconditionally; the next
        # ``n_history_images`` are the history window. Counting one budget of
        # ``n_history_images + 1`` expresses exactly that, and keeps
        # ``n_history_images=0`` meaning "only the current screenshot".
        budget = max(0, self.n_history_images) + 1
        kept = 0
        for message in reversed(content):
            if not message_has_image(message):
                continue
            if kept < budget:
                kept += 1
                continue
            message["content"] = [
                part for part in message["content"]
                if not (isinstance(part, dict) and part.get("type") == "image")
            ]

        return [system_message, *content] if system_message else content


__all__ = ["UIVenus2HistoryProtocol"]
