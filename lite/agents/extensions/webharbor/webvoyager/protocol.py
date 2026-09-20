"""WebVoyager history protocols (model-side bridge).

WebVoyager's Set-of-Marks (SoM) mode keeps the page's ``[N]``-indexed
accessibility tree in the first observation's ``metadata['model_web_text']``.
The chat-style history protocol replaces the first user message's text item, so
without re-appending it the model would never see the DOM tree on turn 0. That
turn-0 ``model_web_text`` splice is a property of the WebVoyager SoM prompting
convention, NOT of any agent family — so it lives here, as a thin subclass hook
over each family's base history protocol, and the qwen3_vl / qwen3_5 SoM yamls
select it via ``protocol_key`` (the generic ``make`` mechanism, same as
the family history protocols).

This replaces the ``append_initial_metadata_web_text`` flag the migration
grafted onto the shared ``qwen3_vl.history`` protocol (a core boundary
violation that, moreover, silently no-op'd on ``qwen3_5.history`` which never
had the field — so the mirrored SoM pair desynced). Both families now get
identical turn-0 behavior from one place.

Reads the env wire SCHEMA only (the ``metadata`` content item), never the
``lite.gym.envs.webharbor.webvoyager`` runtime. ``metadata['web_text']`` is raw
container output for logs; only ``metadata['model_web_text']`` is prompt-safe
here. Auto-registers on import via the ``key=`` arg; the built-in registration
import is owned by ``lite.agents.bootstrap.register_all()`` before any
``protocol_key`` resolves.
"""

from __future__ import annotations

import dataclasses

from lite.agents.models.qwen3_5.protocol import Qwen3_5HistoryProtocol
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.core import (
    LiteMessage,
)
from lite.core.messages import message_metadata


def _with_initial_web_text(message: LiteMessage, text: str) -> str:
    """Append ``metadata['model_web_text']`` to the injected first prompt."""
    web_text = message_metadata(message).get("model_web_text", "")
    if web_text:
        return f"{text}\n{web_text}"
    return text


@dataclasses.dataclass
class WebVoyagerQwen3VLHistoryProtocol(
    Qwen3VLHistoryProtocol,
    key="webharbor.webvoyager.qwen3_vl.history",
):
    """``qwen3_vl.history`` + the WebVoyager SoM turn-0 model DOM splice."""

    def _inject_text(self, message: LiteMessage, text: str) -> None:
        super()._inject_text(message, _with_initial_web_text(message, text))


@dataclasses.dataclass
class WebVoyagerQwen3_5HistoryProtocol(
    Qwen3_5HistoryProtocol,
    key="webharbor.webvoyager.qwen3_5.history",
):
    """``qwen3_5.history`` + the WebVoyager SoM turn-0 model DOM splice
    (mirror of :class:`WebVoyagerQwen3VLHistoryProtocol` for the qwen3_5 pair)."""

    def _inject_text(self, message: LiteMessage, text: str) -> None:
        super()._inject_text(message, _with_initial_web_text(message, text))
