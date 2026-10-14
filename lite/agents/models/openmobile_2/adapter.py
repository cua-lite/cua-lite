"""OpenMobile-2 adapter (``openmobile_2@mobile@use``)."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.models.openmobile_2.action_space import OpenMobile2MobileActionSpace
from lite.agents.models.openmobile_2.protocol import OpenMobile2HistoryProtocol
from lite.agents.models.qwen3_5.adapter import Qwen3_5BaseAdapter
from lite.agents.types import AgentMessage
from lite.core.messages.final import mark_model_output_error, pop_model_output_error
from lite.core.tools.extra_tools import LiteFinishToolSet

#: Verbatim official ``QWEN35_SESSION_SYSTEM_PROMPT_THOUGHT``
#: (OpenMobile2-Code/runtimes/prompts.py:667-726, the ``qwen35_thought_session``
#: runtime's system prompt): the 8-action ``mobile_use`` JSON schema embedded
#: in a ``<tools>`` block, the Qwen XML format example, and the session /
#: tool-call / visual-context / robustness rules. Byte-exact copy — the JSON
#: descriptions keep their ``\\n`` escapes and the string opens with a
#: newline; the family test pins the bytes with a sha256 golden.
OPENMOBILE2_SYSTEM_PROMPT = """
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x and y coordinates. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The end coordinates. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Required only by `action=system_button`.", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "Required only by `action=terminate`.", "enum": ["success", "failure"], "type": "string"}}, "required": ["action"], "type": "object"}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- Put reasoning after `Thought:` BEFORE the function call, and do not write anything after </tool_call>
- Every assistant turn MUST include exactly one mobile_use function call. Do not answer the user in plain text.
</IMPORTANT>

You are a mobile GUI agent. Use the mobile_use tool to interact with the device touchscreen.

# Session

- The task query is provided only once in the first user message, together with the initial screenshot.
- After each tool call, the environment returns a tool message containing the resulting screenshot.
- Previous assistant reasoning and tool calls remain in the conversation history.
- Each turn: look at the screenshot(s) still provided as images, plus the full history, then make exactly one tool call.
- When the task is complete or cannot be completed, call terminate with status=success or status=failure.

# Tool call

- Every assistant turn must be exactly: a `Thought:` paragraph, then one <tool_call>...</tool_call> block.
- Emit Qwen XML: <tool_call><function=mobile_use><parameter=...>...</parameter></function></tool_call>
- Do not emit a JSON object such as {"action":"click","coordinate":[x,y]} or {"name":"...","arguments":{...}}.
- One <parameter> block per argument. Arrays/objects must be valid JSON inside the block (e.g. [163, 718]).
- Do not output anything after </tool_call>.

# Visual context

- Only the most recent screenshot slots remain as images. Older slots are replaced with the text "This screenshot has been collapsed."
- Act only on images that are actually present. Do not recover pixels from collapsed slots.

# Robustness

- If the history shows repeated similar actions without progress, try a different action.
- Never assume an earlier action succeeded without checking the latest screenshot.
"""


@dataclasses.dataclass
class OpenMobile2MobileUseAdapter(
    Qwen3_5BaseAdapter,
    key="openmobile_2@mobile@use",
):
    """OpenMobile-2 mobile ``use`` — the official ``qwen35_thought_session`` wire.

    The official runtime embeds the ``mobile_use`` schema in the system
    prompt and sends no API ``tools=``; this adapter therefore pins the
    verbatim official prompt with ``render_tools_section=False`` (the
    inherited Qwen3.5 XML tools preamble would duplicate it) and relies on
    the inherited ``build_generation_prompt`` never passing ``tools=``.

    The rest of the official session shape is owned by the pieces this
    class wires together: :class:`OpenMobile2HistoryProtocol` (query-first
    first user message + ``last_n`` screenshot window with collapse
    placeholders) and :class:`OpenMobile2MobileActionSpace` (8-action enum,
    Menu → wait lowering, answer/terminate finish folding). Assistant
    replay and XML parsing inherit from :class:`Qwen3_5BaseAdapter` — its
    inline ``<tool_call>`` rendering reproduces the official ``Thought:``
    + XML format verbatim.
    """

    # ─── Core ────────────────────────────────────────────────────────
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=OpenMobile2MobileActionSpace
    )
    protocol: OpenMobile2HistoryProtocol = dataclasses.field(
        default_factory=OpenMobile2HistoryProtocol
    )
    # ─── Class-specific knobs ────────────────────────────────────────
    system_prompt: str | None = OPENMOBILE2_SYSTEM_PROMPT
    render_tools_section: bool = False
    # Official grid alignment: the official harness sends raw 1080x2400
    # PNGs and the serving processor rounds them to 32-px multiples
    # (1088x2400); the inherited client-side smart_resize applies the
    # identical rounding, so the default stays on.
    smart_resize_enabled: bool = True
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteFinishToolSet.get_tool_names()
    )

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs: Any,
    ) -> AgentMessage:
        """Parse with the official ``qwen35`` dialect order: Qwen XML
        first, JSON-bodied ``<tool_call>`` fallback second
        (``_parse_tool_call_xml(...) or _parse_tool_call_json(...)``)."""
        result = super().parse_raw_assistant_response(response, **kwargs)
        if result.get("tool_calls"):
            return result
        error = pop_model_output_error(result)
        json_calls = self._extract_json_tool_calls(response)
        if json_calls:
            result["tool_calls"] = json_calls
        elif error is not None:
            mark_model_output_error(result, error)
        return result


__all__ = ["OPENMOBILE2_SYSTEM_PROMPT", "OpenMobile2MobileUseAdapter"]
