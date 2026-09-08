"""
Qwen3.5 action spaces (thin subclasses of Qwen3-VL).

Qwen3.5 uses the same single-tool ``computer_use`` / ``mobile_use`` schema
and the same CUA-lite ↔ agent tool_call mapping as Qwen3-VL. The only
dialect difference lives at the *wire format* level (XML tool_calls) and is
handled by :mod:`lite.agents.models.qwen3_5.adapter`. These subclasses exist
only to register under ``qwen3_5@(desktop|browser)`` / ``qwen3_5@mobile`` keys
so the adapter can be looked up via the standard factory path.

Usage:
    from lite.agents.models.qwen3_5.action_space import (
        Qwen3_5DesktopActionSpace,
        Qwen3_5MobileActionSpace,
    )
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.models.qwen3_vl.action_space import (
    Qwen3VLDesktopActionSpace,
    Qwen3VLDesktopGroundingPointActionSpace,
    Qwen3VLMobileActionSpace,
    Qwen3VLMobileGroundingPointActionSpace,
)


@dataclasses.dataclass
class Qwen3_5DesktopActionSpace(Qwen3VLDesktopActionSpace, key=r"qwen3_5@(desktop|browser)"):
    """Desktop+browser action space for Qwen3.5. See parent for semantics. One class
    for both platforms: browser nav is an env extra_tool, not a per-platform
    action space; Qwen3.5's desktop delta is the XML wire format handled by
    :mod:`lite.agents.models.qwen3_5.adapter`."""

    #: Qwen3.5 and Qwen3.8 write raw notch counts, not screen units: 70.0% and
    #: 99.3% of the values they emit are below one click's worth of screen.
    SCROLL_WIRE_UNIT: ClassVar[int] = 1


# Mobile action alias that Qwen3.5 emits on mobile despite the prompt only
# advertising the native ``mobile_use`` enum. Maps the desktop
# ``computer_use`` leakage ``left_click`` → mobile ``click``. Observed in
# the ``2026-04-26T18-38_8b9d4bd`` androidworld snapshot: Qwen3.5-9B emits
# ``left_click`` on 73.6% of mobile turns (1404 / 1907), 4B on 15.5%
# (208 / 1342), 2B 0% — none of Qwen3-VL-{2,4,8,32}B emit it. Without the
# alias these turns fall through to the parent's ``else`` branch and become
# env-side ``noop``, which collapsed 9B's MER from ~0.47 (4B baseline) to
# 0.16. The fix lives here, not on the parent, because the leakage is
# Qwen3.5-family-specific.
_MOBILE_ACTION_ALIASES: dict[str, str] = {
    "left_click": "click",
}


@dataclasses.dataclass
class Qwen3_5MobileActionSpace(Qwen3VLMobileActionSpace, key="qwen3_5@mobile"):
    """Mobile action space for Qwen3.5.

    Inherits all semantics from :class:`Qwen3VLMobileActionSpace`, with one
    delta: tolerant aliasing for ``left_click`` (desktop ``computer_use``
    leakage) → ``click``, which Qwen3.5-9B / 4B emit on mobile despite the
    schema only declaring the native mobile enum. See
    :data:`_MOBILE_ACTION_ALIASES` above.
    """

    def _convert_single_from_agent(
        self,
        agent_tool_call: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if agent_tool_call["name"] != "mobile_use":
            return super()._convert_single_from_agent(agent_tool_call, **kwargs)
        args = agent_tool_call["arguments"]
        action = args.get("action", "")
        canonical = _MOBILE_ACTION_ALIASES.get(action)
        if canonical is None:
            return super()._convert_single_from_agent(agent_tool_call, **kwargs)
        # Rewrite ``action`` to the canonical name, keep all other args as-is,
        # then delegate. The other args (``coordinate``, ``text``, ...) match
        # the parent's expectations for the canonical action.
        rewritten = {
            "name": agent_tool_call["name"],
            "arguments": {**args, "action": canonical},
        }
        return super()._convert_single_from_agent(rewritten, **kwargs)


# =============================================================================
# Grounding (single-step click) action spaces
# =============================================================================
#
# Same trimmed schema as Qwen3-VL, inherited verbatim. Qwen3.5's wire-format
# delta is XML tool_calls (handled by the adapter), plus the same family-prior
# click/left_click tolerance used by the mobile use action space.

_DESKTOP_GROUNDING_ALIASES: dict[str, str] = {
    "click": "left_click",
}
_MOBILE_GROUNDING_ALIASES: dict[str, str] = {
    "left_click": "click",
}


def _alias_grounding_call(
    tc: dict[str, Any],
    aliases: dict[str, str],
    *,
    qwen_native_tool_name: str,
) -> dict[str, Any]:
    if tc.get("name") != qwen_native_tool_name:
        return tc
    args = tc.get("arguments") or {}
    canonical = aliases.get(args.get("action", ""))
    if canonical is None:
        return tc
    return {**tc, "arguments": {**args, "action": canonical}}


@dataclasses.dataclass
class Qwen3_5DesktopGroundingPointActionSpace(Qwen3VLDesktopGroundingPointActionSpace, key=r"qwen3_5@(desktop|browser)@point"):
    """Desktop grounding (single-step click) for Qwen3.5.

    Inherits the Qwen3-VL trimmed schema, with one Qwen3.5-family tolerance:
    ``click`` is accepted as ``left_click`` on the desktop/browser point surface.
    The XML wire format difference lives in
    :mod:`lite.agents.models.qwen3_5.adapter`.
    """

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        rewritten = [
            _alias_grounding_call(
                tc,
                _DESKTOP_GROUNDING_ALIASES,
                qwen_native_tool_name="computer_use",
            )
            for tc in agent_tool_calls
        ]
        return super().convert_tool_calls_from_agent(rewritten, **kwargs)


@dataclasses.dataclass
class Qwen3_5MobileGroundingPointActionSpace(Qwen3VLMobileGroundingPointActionSpace, key="qwen3_5@mobile@point"):
    """Mobile grounding (single-step click) for Qwen3.5.

    Inherits the Qwen3-VL trimmed schema, with one Qwen3.5-family tolerance:
    ``left_click`` is accepted as ``click`` on the mobile point surface. The XML
    wire format difference lives in :mod:`lite.agents.models.qwen3_5.adapter`.
    """

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        rewritten = [
            _alias_grounding_call(
                tc,
                _MOBILE_GROUNDING_ALIASES,
                qwen_native_tool_name="mobile_use",
            )
            for tc in agent_tool_calls
        ]
        return super().convert_tool_calls_from_agent(rewritten, **kwargs)


# Re-export for type hints
__all__ = [
    "Qwen3_5DesktopActionSpace",
    "Qwen3_5MobileActionSpace",
    "Qwen3_5DesktopGroundingPointActionSpace",
    "Qwen3_5MobileGroundingPointActionSpace",
    "BaseActionSpace",
]
