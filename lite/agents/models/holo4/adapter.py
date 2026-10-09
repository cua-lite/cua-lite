"""Holo4 adapters built on the Qwen3.5 XML tool-call wire format."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from lite.agents.core.action_space import BaseActionSpace, assemble_tool_schemas
from lite.agents.models.holo4.action_space import (
    Holo4DesktopActionSpace,
    Holo4MobileActionSpace,
)
from lite.agents.models.holo4.protocol import Holo4HistoryProtocol
from lite.agents.models.qwen3_5.adapter import Qwen3_5BaseAdapter
from lite.agents.types import AgentMessage, AgentStep
from lite.core import LiteMessage, LiteSample
from lite.core.messages import SYSTEM_ROLE, TEXT_PART
from lite.core.tools.calls import tool_call_name
from lite.core.tools.extra_tools import BASH_TOOL_NAME
from lite.core.tools.schemas import make_tool_schema, tool_schema_name

HOLO4_USE_SYSTEM_PROMPT = """You are Holo, a computer-use agent.

Work in a loop: reason privately, optionally write a brief user-facing message,
then emit one or more tool calls. Tool calls execute sequentially and the next
observation shows the final state.

- Coordinates are normalized to the [0, 1000] range on both axes.
- Follow the advertised tool schemas exactly.
- Chain only actions whose targets are visible in the current screenshot.
- Stop a chain when you need to inspect an intermediate result.
- Keep durable facts in concise message text because old screenshots and tool
  output may be compacted; private reasoning is not durable memory.
- Every non-terminal response must include a tool call.
- Use `answer` when the task asks for an answer. Use `terminate` when it is
  advertised and the task only requires completing actions with no answer text.
"""


@dataclasses.dataclass
class Holo4BaseAdapter(
    Qwen3_5BaseAdapter,
    key=r"holo4\.base(@(desktop|browser|mobile)@use)?",
):
    """Workflow-agnostic Holo4 XML adapter with flat tool projection."""

    enable_thinking: bool = True
    system_prompt: str | None = HOLO4_USE_SYSTEM_PROMPT

    def _assemble_tool_schemas(self) -> list[dict[str, Any]]:
        metadata = self._cua_metadata()
        schemas = assemble_tool_schemas(
            self.action_space,
            self.action_space.get_tool_schemas(),
            valid_actions=metadata.valid_actions,
            extra_tool_schemas=None,
        )
        answer_added = False
        active_extra_names = {tool_schema_name(schema) for schema in metadata.extra_tool_schemas}
        for extra_schema in metadata.extra_tool_schemas:
            name = tool_schema_name(extra_schema)
            if name == BASH_TOOL_NAME:
                projected = copy.deepcopy(extra_schema)
                projected["function"]["name"] = "shell"
                schemas.append(projected)
            elif name == "response" or (
                name == "terminate" and "response" not in active_extra_names
            ):
                if not answer_added:
                    schemas.append(
                        make_tool_schema(
                            "answer",
                            description="Finish the task and provide the final answer.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "The final answer content.",
                                    }
                                },
                                "required": ["content"],
                                "additionalProperties": False,
                            },
                        )
                    )
                    answer_added = True
            elif name == "terminate":
                schemas.append(copy.deepcopy(extra_schema))
            else:
                schemas.append(copy.deepcopy(extra_schema))
        return schemas

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render a step with one model-visible system message.

        The Qwen renderer always creates the Holo/tool system message. A
        trajectory may also carry a task-provided system message, which Holo4
        templates only accept when merged into that first message.
        """
        rendered = super().render_step(sample, k, processed, **kwargs)
        system_messages = [message for message in rendered if message.get("role") == SYSTEM_ROLE]
        if len(system_messages) <= 1:
            return rendered

        content: list[dict[str, Any]] = []
        for message in system_messages:
            message_content = message.get("content")
            if isinstance(message_content, str):
                if message_content:
                    content.append({"type": TEXT_PART, "text": message_content})
            elif isinstance(message_content, list):
                content.extend(copy.deepcopy(message_content))

        first_system_index = next(
            index for index, message in enumerate(rendered) if message.get("role") == SYSTEM_ROLE
        )
        merged_system = copy.deepcopy(rendered[first_system_index])
        merged_system["content"] = content
        return [
            merged_system,
            *(message for message in rendered if message.get("role") != SYSTEM_ROLE),
        ]

    def convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> AgentMessage:
        """Replay canonical assistant fields, never opaque Holo output.

        Holo4's raw completion contains private reasoning and template control
        tokens. Replaying it would bypass canonical tool-call/text rendering
        and produces malformed thinking blocks on Holo4-27B.
        """
        if message.get("role") == "assistant":
            message = copy.deepcopy(message)
            message.pop("raw_response", None)
            message.pop("reasoning_content", None)
        return super().convert_message_to_agent(message, **kwargs)

    def _tool_calls_to_agent_ordered(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self._require_standalone_tool_schemas(tool_calls)
        projected_names = {BASH_TOOL_NAME, "response"}
        if "response" not in self.active_extra_tool_names():
            projected_names.add("terminate")
        result: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            name = tool_call_name(tool_call)
            if name in projected_names:
                result.extend(self.action_space.convert_tool_calls_to_agent([tool_call]))
            elif self._admits_active_extra_tool_call(tool_call):
                result.append(
                    {
                        "name": name,
                        "arguments": copy.deepcopy(tool_call["function"]["arguments"]),
                    }
                )
            else:
                result.extend(self.action_space.convert_tool_calls_to_agent([tool_call]))
        return result


@dataclasses.dataclass
class Holo4DesktopUseAdapter(
    Holo4BaseAdapter,
    key=r"holo4@(desktop|browser)@use",
):
    """Holo4 desktop/browser use adapter."""

    action_space: BaseActionSpace = dataclasses.field(default_factory=Holo4DesktopActionSpace)
    protocol: Holo4HistoryProtocol = dataclasses.field(default_factory=Holo4HistoryProtocol)


@dataclasses.dataclass
class Holo4MobileUseAdapter(Holo4BaseAdapter, key="holo4@mobile@use"):
    """Holo4 mobile use adapter."""

    action_space: BaseActionSpace = dataclasses.field(default_factory=Holo4MobileActionSpace)
    protocol: Holo4HistoryProtocol = dataclasses.field(default_factory=Holo4HistoryProtocol)
    smart_resize_enabled: bool = False


__all__ = [
    "HOLO4_USE_SYSTEM_PROMPT",
    "Holo4BaseAdapter",
    "Holo4DesktopUseAdapter",
    "Holo4MobileUseAdapter",
]
