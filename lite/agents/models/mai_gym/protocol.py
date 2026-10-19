"""MAI-Gym window-wise history protocols.

``mai_gym.history`` (mobile) mirrors ``MAIGymMobileUseWindowsWisePolicy``
(MAI-Gym's training and eval harness, K=10 / O=2 in the paper): every text
turn survives, and screenshots
are kept from a sawtooth window start onward. Within a window the prompt only
grows, so consecutive prompts share a prefix; at a window boundary the older
screenshots collapse into placeholders.

Reference: ``MAI-Gym/src/mai_gym/agents/implementations/mai_gym_mobile_use_windows_wise.py``
(``window_start`` / ``collapse_to_window``) and ``mai_gym_mobile_use.py``
(message layout).

``mai_gym.desktop_history`` mirrors ``MAIGymComputerUsePolicy._build_messages``
with its ``window_size`` option set: the same sawtooth window, but one user
message per step that carries the step's text and screenshot together.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from typing import Any

from lite.agents.core.protocol import BaseProtocol
from lite.core import LiteMessage
from lite.core.messages import (
    ASSISTANT_ROLE,
    IMAGE_PART,
    METADATA_PART,
    TEXT_PART,
    TOOL_ROLE,
    USER_ROLE,
)
from lite.core.messages.content import message_metadata, require_message_list
from lite.core.tools.calls import tool_call_id, tool_call_name

#: Upstream ``HISTORY_IMAGE_PLACEHOLDER``: the whole user turn of a collapsed screenshot.
HISTORY_IMAGE_PLACEHOLDER = "<image_collapsed>"


def _tool_call_names(messages: list[LiteMessage]) -> dict[str | None, str]:
    """Call id -> tool name, so a tool result can tell which tool it answers."""
    return {
        tool_call_id(call): tool_call_name(call)
        for message in messages
        if message.get("role") == ASSISTANT_ROLE
        for call in message.get("tool_calls") or []
    }


@dataclasses.dataclass
class MAIGymHistoryProtocol(BaseProtocol, key="mai_gym.history"):
    """Upstream message layout plus the window-wise screenshot policy.

    Layout, one user turn per part::

        user(instruction text) user(image 0) assistant
        user(image 1) assistant ... user(image n)

    The canonical first user message (``[image, text]``) is split into its
    text turn and its image turn, instruction first. Env feedback
    (``role:"tool"``) becomes plain user turns: its text, if any, first and
    then its screenshot, because the template would otherwise wrap it in
    ``<tool_response>``, which the model never saw. An ``ask_user`` reply is
    the exception: upstream sends the reply text alone, without that step's
    screenshot.

    Screenshot ``i`` (0-based, among the screenshots sent) collapses to a
    ``"<image_collapsed>"`` user turn when ``i < window_start(n)``, where
    ``n`` is the number of assistant turns so far — upstream
    ``len(step_records)`` at the time of the call. With the defaults, turn 11
    (``n=10``) starts a new window: screenshots 0-7 collapse and 8-10 stay.
    """

    window_size: int = 10
    overlap: int = 2

    def __post_init__(self) -> None:
        if self.window_size - self.overlap <= 0:
            raise ValueError(
                f"window_size - overlap must be > 0 "
                f"(window_size={self.window_size}, overlap={self.overlap})"
            )

    def window_start(self, step_idx: int) -> int:
        """First screenshot ordinal kept at step ``step_idx`` (upstream ``window_start``)."""
        stride = self.window_size - self.overlap
        return max(0, stride * ((step_idx - self.overlap) // stride))

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs: Any,
    ) -> list[LiteMessage]:
        require_message_list(messages, where="process_messages")
        call_names = _tool_call_names(messages)
        start = self.window_start(
            sum(1 for message in messages if message.get("role") == ASSISTANT_ROLE)
        )
        image_ordinal = 0
        result: list[LiteMessage] = []
        for message in messages:
            if message.get("role") not in (USER_ROLE, TOOL_ROLE):
                result.append(copy.deepcopy(message))
                continue
            content = copy.deepcopy(message.get("content") or [])
            images = [part for part in content if part.get("type") == IMAGE_PART]
            texts = [part for part in content if part.get("type") == TEXT_PART and part.get("text")]
            if texts and call_names.get(message.get("tool_call_id")) == "ask_user":
                images = []
            # Metadata is a side channel stripped before serialization; it rides
            # on whichever turn this observation emits first.
            side_parts = [part for part in content if part.get("type") == METADATA_PART]
            turns: list[list[dict[str, Any]]] = [texts] if texts else []
            for image in images:
                collapsed = image_ordinal < start
                turns.append(
                    [{"type": TEXT_PART, "text": HISTORY_IMAGE_PLACEHOLDER} if collapsed else image]
                )
                image_ordinal += 1
            if turns:
                turns[0].extend(side_parts)
            result.extend({"role": USER_ROLE, "content": parts} for parts in turns)
        return result


#: Upstream ``_build_messages`` instruction text, sent once at the first step.
DESKTOP_INSTRUCTION_TEMPLATE = (
    "# Task Instruction:\n"
    "{instruction}\n"
    "Complete the task with actions in the system-prompt format."
)

# The sandbox bash tool appends ``[exit N]`` to the output on a nonzero code;
# the return code itself also rides in the result metadata.
_EXIT_SUFFIX_RE = re.compile(r"\n?\[exit -?\d+\]\Z")


def _bash_output_block(output: str, returncode: int) -> str:
    """Upstream ``_format_bash_output``; the env merges stderr into stdout."""
    parts = [f"## Bash Output\nreturncode={returncode}"]
    if output:
        parts.append("stdout:\n" + output.rstrip()[:4000])
    else:
        parts.append("(no output)")
    return "\n\n".join(parts)


@dataclasses.dataclass
class MAIGymDesktopHistoryProtocol(MAIGymHistoryProtocol, key="mai_gym.desktop_history"):
    """Upstream computer-use layout plus the window-wise screenshot policy.

    One user message per step, then that step's assistant turn::

        user[instruction, (bash output), image 0] assistant
        user[(bash output), image 1] assistant ... user[(bash output), image n]

    Step ``i``'s screenshot is the LAST frame its observations returned (one
    per step upstream); a step that returned none reuses the previous one,
    as upstream does. It becomes the text ``"<image_collapsed>"`` when
    ``i < window_start(n)`` (``n`` = assistant turns so far), inside the same
    user message. A ``bash`` result renders as upstream's ``## Bash Output``
    block; other env feedback text follows it verbatim.
    """

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs: Any,
    ) -> list[LiteMessage]:
        require_message_list(messages, where="process_messages")
        call_names = _tool_call_names(messages)
        # Observation groups: groups[i] is what step i's model call observed.
        groups: list[list[LiteMessage]] = [[]]
        assistants: list[LiteMessage] = []
        for message in messages:
            if message.get("role") == ASSISTANT_ROLE:
                assistants.append(copy.deepcopy(message))
                groups.append([])
            else:
                groups[-1].append(message)

        start = self.window_start(len(assistants))
        last_image: dict[str, Any] | None = None
        result: list[LiteMessage] = []
        for step, group in enumerate(groups):
            parts: list[dict[str, Any]] = []
            image: dict[str, Any] | None = None
            for index, message in enumerate(group):
                content = message.get("content") or []
                texts = [p["text"] for p in content if p.get("type") == TEXT_PART and p.get("text")]
                images = [p for p in content if p.get("type") == IMAGE_PART]
                if images:
                    image = copy.deepcopy(images[-1])
                if step == 0 and index == 0 and message.get("role") == USER_ROLE:
                    parts.append(
                        {
                            "type": TEXT_PART,
                            "text": DESKTOP_INSTRUCTION_TEMPLATE.format(
                                instruction="\n".join(texts)
                            ),
                        }
                    )
                    continue
                returncode = message_metadata(message).get("returncode")
                is_bash = call_names.get(message.get("tool_call_id")) == "bash"
                if is_bash and isinstance(returncode, int):
                    output = _EXIT_SUFFIX_RE.sub("", "\n".join(texts))
                    parts.append(
                        {"type": TEXT_PART, "text": _bash_output_block(output, returncode)}
                    )
                else:
                    parts.extend({"type": TEXT_PART, "text": text} for text in texts)
            image = image or last_image
            if image is not None:
                last_image = image
                parts.append(
                    {"type": TEXT_PART, "text": HISTORY_IMAGE_PLACEHOLDER}
                    if step < start
                    else image
                )
            if parts:
                result.append({"role": USER_ROLE, "content": parts})
            if step < len(assistants):
                result.append(assistants[step])
        return result


__all__ = [
    "DESKTOP_INSTRUCTION_TEMPLATE",
    "HISTORY_IMAGE_PLACEHOLDER",
    "MAIGymDesktopHistoryProtocol",
    "MAIGymHistoryProtocol",
]
