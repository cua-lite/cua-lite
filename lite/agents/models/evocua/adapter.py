"""
EvoCUA Adapters

Reuses Qwen3VLBaseAdapter with EvoCUADesktopActionSpace.

    EvoCUADesktopGroundingPointAdapter   (single-step click; desktop+browser)
    EvoCUADesktopGroundingActionAdapter  (full action vocab; desktop+browser)
    EvoCUADesktopUseAdapter       (summarized history; desktop+browser)

The ``Desktop`` adapters register under the
``r"evocua@(desktop|browser)@..."`` regex, so ``evocua@browser@...`` keys
resolve to the same class.

Usage:
    from lite.agents.models.evocua.adapter import (
        EvoCUADesktopGroundingPointAdapter,
        EvoCUADesktopUseAdapter,
    )
"""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any, ClassVar

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.adapter import (
    AgentAdapterRegistry,
    AsIsAdapter,
)
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.evocua.action_space import (
    EvoCUADesktopActionSpace,
    EvoCUADesktopGroundingPointActionSpace,
)
from lite.agents.models.qwen3_vl.adapter import (
    GROUNDING_POINT_SYSTEM_PROMPT,
    Qwen3VLBaseAdapter,
    Qwen3VLUseAdapter,
)
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.core import (
    LiteCUAMetadata,
)
from lite.core.tools.calls import tool_call_name
from lite.core.tools.schemas import validate_extra_tool_schemas

# =============================================================================
# System Prompt
# =============================================================================
# Same model-facing format as Qwen3-VL's ``USE_SYSTEM_PROMPT``, own copy,
# because the finish rule differs: EvoCUA drops Qwen3-VL's ``answer`` enum
# member (see ``EvoCUADesktopActionSpace``), so ordering ``action=answer`` names
# a call this family's parser drops on the way in and whose canonical
# ``response`` its renderer refuses on the way out. ``terminate`` is the only
# standalone canonical tool EvoCUA declares.

USE_SYSTEM_PROMPT = """# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""

_EVOCUA_NAME_FOR_HUMAN = "computer_use"
_EVOCUA_ARGS_FORMAT = "Format the arguments as a JSON object."

# =============================================================================
# Desktop + Browser Adapters
# =============================================================================
# Desktop and browser share one adapter class per task type via a
# ``(desktop|browser)`` regex key. EvoCUA doesn't ship a mobile variant.

@dataclasses.dataclass
class EvoCUADesktopGroundingActionAdapter(
    Qwen3VLBaseAdapter, key=r"evocua@(desktop|browser)@grounding\.action"
):
    """Desktop+browser grounding/action: full action vocabulary, single turn.

    Currently NOT routed by env eval (env declares ``grounding.point``;
    see :class:`EvoCUADesktopGroundingPointAdapter`). Kept available for
    SFT-data replay via the EvoCUA-specific action_space (adds
    key_down/key_up vs Qwen3-VL). Single-step content is structured
    tool_calls — the assistant text part flows through base's text-flatten
    path, no ``Action:`` / ``Thought:`` text format.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=EvoCUADesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )


@dataclasses.dataclass
class EvoCUADesktopGroundingPointAdapter(
    Qwen3VLBaseAdapter, key=r"evocua@(desktop|browser)@grounding\.point"
):
    """Desktop+browser grounding (single-step click) for EvoCUA.

    Uses the trimmed :class:`EvoCUADesktopGroundingPointActionSpace`
    (``computer_use`` with only ``left_click``) plus the shared
    :data:`GROUNDING_POINT_SYSTEM_PROMPT`. Single turn, full history.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=EvoCUADesktopGroundingPointActionSpace
    )
    metadata: LiteCUAMetadata = dataclasses.field(
        default_factory=lambda: LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.DESKTOP.value,
                LiteCUAMetadata.TaskType.GROUNDING_POINT.value,
            )
        )
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    system_prompt: str | None = GROUNDING_POINT_SYSTEM_PROMPT


@dataclasses.dataclass
class EvoCUADesktopUseAdapter(
    Qwen3VLUseAdapter, key=r"evocua@(desktop|browser)@use"
):
    """Desktop+browser ``use`` (multi-step rollout): with system prompt, summarized history."""
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=EvoCUADesktopActionSpace
    )
    protocol: Qwen3VLHistoryProtocol = dataclasses.field(
        default_factory=lambda: Qwen3VLHistoryProtocol(full_history_size=4)
    )
    system_prompt: str | None = USE_SYSTEM_PROMPT
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = frozenset({
        "terminate",
    })

    def _build_tools_section(self, image_size: tuple[int, int] | None = None) -> str:
        """Render EvoCUA's reference ``computer_use`` metadata in ``<tools>``."""
        tool_schemas = self._tool_schemas_for_tools_section()
        validate_extra_tool_schemas(
            tool_schemas,
            where="EvoCUADesktopUseAdapter._build_tools_section.tool_schemas",
        )
        rendered_schemas = copy.deepcopy(tool_schemas)
        for schema in rendered_schemas:
            function = schema["function"]
            if function["name"] == "computer_use":
                function["name_for_human"] = _EVOCUA_NAME_FOR_HUMAN
                function["args_format"] = _EVOCUA_ARGS_FORMAT
        tools_json = "\n".join(json.dumps(schema) for schema in rendered_schemas)
        if image_size is not None:
            W, H = image_size
            tools_json = (
                tools_json
                .replace("{display_width_px}", str(W))
                .replace("{display_height_px}", str(H))
            )
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_json}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    def _tool_calls_to_agent_ordered(
        self, tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reject canonical ``response`` before the standalone-extra passthrough.

        EvoCUA has no native answer channel (it drops Qwen3-VL's ``answer``
        enum member), so a persisted canonical ``response`` must fail loudly
        rather than be passed through onto the model surface under its canonical
        name.
        Mirrors :meth:`EvoCUADesktopActionSpace._convert_single_to_agent`, which
        the standalone-extra passthrough would otherwise bypass.
        """
        self._require_standalone_tool_schemas(tool_calls)
        for tc in tool_calls:
            if tool_call_name(tc) == "response":
                raise ValueError("EvoCUA cannot render canonical tool 'response'")
        return super()._tool_calls_to_agent_ordered(tool_calls)


# =============================================================================
# Pass-through Registration for Other Tasks
# =============================================================================

AgentAdapterRegistry.register(r"evocua@(desktop|browser)@understanding", AsIsAdapter)
AgentAdapterRegistry.register(r"evocua@(desktop|browser)@grounding\.bbox", AsIsAdapter)
