"""
Qwen3.5 Adapters (XML tool_call wire format).

Qwen3.5 is natively multimodal. It shares the same high-level response
shape as Qwen3-VL (single ``computer_use`` / ``mobile_use`` tool, an
``Action:`` line + a ``<tool_call>`` block) but the tool_call body is
rendered in XML form::

    <tool_call>
    <function=computer_use>
    <parameter=action>
    left_click
    </parameter>
    <parameter=coordinate>
    [491, 91]
    </parameter>
    </function>
    </tool_call>

This module subclasses the qwen3_vl adapters and overrides three things:

* :meth:`Qwen3_5BaseAdapter._build_tools_section` — emit the XML-flavored
  tools block (matches the phrasing baked into Qwen3.5's
  ``chat_template.jinja`` so our constructed system-prompt is consistent
  with what the template itself would add when ``tools=[...]`` is passed).
* :meth:`Qwen3_5BaseAdapter.parse_raw_assistant_response` — parse XML
  tool_calls back into CUA-lite's ``tool_calls`` slot.
* :meth:`Qwen3_5BaseAdapter._convert_message_to_agent` — render assistant
  tool_calls back to XML inline inside the message text (the chat template
  only receives a text blob; we do the rendering here, not via the
  ``tool_calls`` message field).

Round-trip example (doctest-style)::

    >>> text = (
    ...     'Action: Click the three-dots menu.\\n'
    ...     '<tool_call>\\n'
    ...     '<function=computer_use>\\n'
    ...     '<parameter=action>\\nleft_click\\n</parameter>\\n'
    ...     '<parameter=coordinate>\\n[491, 91]\\n</parameter>\\n'
    ...     '</function>\\n'
    ...     '</tool_call>'
    ... )
    >>> calls = _parse_xml_tool_calls(text)
    >>> calls == [{'name': 'computer_use',
    ...     'arguments': {'action': 'left_click', 'coordinate': [491, 91]}}]
    True
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import logging
import math
import re
from typing import Any, ClassVar

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry, AsIsAdapter
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.qwen3_5.action_space import (
    Qwen3_5DesktopActionSpace,
    Qwen3_5DesktopGroundingPointActionSpace,
    Qwen3_5MobileActionSpace,
    Qwen3_5MobileGroundingPointActionSpace,
)
from lite.agents.models.qwen3_5.protocol import Qwen3_5HistoryProtocol
from lite.agents.models.qwen3_vl.adapter import (
    GROUNDING_POINT_SYSTEM_PROMPT,
    USE_SYSTEM_PROMPT,
    Qwen3VLBaseAdapter,
)
from lite.agents.types import AgentMessage, AgentStep
from lite.core import (
    LiteCUAMetadata,
    LiteMessage,
    LiteSample,
)
from lite.core.messages import make_assistant_content
from lite.core.messages.final import mark_model_output_error
from lite.core.messages.turns import truncate_sample_to_turn
from lite.core.tools.extra_tools import LiteAppLaunchToolSet, LiteFinishToolSet
from lite.core.tools.schemas import (
    tool_schema_name,
    tool_schema_parameters,
    validate_extra_tool_schemas,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Wire-format helpers
# =============================================================================

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
# One ``</parameter>`` can be claimed by two openers when the model emits an
# opener it never closes. ``_PARAM_RE`` is non-greedy from the LEFT, so the
# outer one wins -- correct when the inner text is literal (a ``type`` whose
# text quotes XML), wrong when the outer opener is stray. Qwen3.8 has emitted
# ``<parameter=computer_use>`` (the function name repeated as a parameter tag,
# never closed) directly before the real ``<parameter=action>``, which parsed
# as ``{"computer_use": "<parameter=action>\\nscroll"}`` and dropped ``action``
# entirely -- leaving a wrapper call with no action value.
#
# The two cases are lexically identical, so the tie is broken on the SCHEMA the
# model was shown: a stray opener names something the tool does not declare,
# while the swallowed one does. See :func:`_reclaim_swallowed_parameter`.
_NESTED_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*)", re.DOTALL)

# ─── Fallback name allowlists (used ONLY when no schema declares the param) ──
#
# The primary coercion driver is the tool's OWN declared JSON schema (see
# :func:`_param_types_from_tool_schemas` / :func:`_coerce_param_value`). These
# name-keyed sets remain as the tolerant fallback for the two cases where no
# declared type is reachable:
#
#   * the module-level :func:`_parse_xml_tool_calls` called without schemas
#     (doctest / data-preproc replay of a raw response with no adapter);
#   * a parameter the model emits that the tool's schema does not declare
#     (e.g. ``duration``, which no ``computer_use`` schema declares, or a
#     hallucinated key).
#
# Argument keys whose values are lists/tuples/structured JSON in the
# tool schemas. Parse these with ast.literal_eval (JSON fallback). Union of
# Qwen3VLDesktopActionSpace (coordinate/start_coordinate/end_coordinate/keys)
# and Qwen3VLMobileActionSpace (coordinate/coordinate2) plus grounding-style
# schemas used by pass-through adapters (point/bbox).
_STRUCTURED_KEYS: set[str] = {
    "coordinate",
    "coordinate2",
    "start_coordinate",
    "end_coordinate",
    "keys",
    "actions",
    "point",
    "bbox",
}
_INT_KEYS: set[str] = {"pixels"}
_FLOAT_KEYS: set[str] = {"time", "duration"}


def _clean_key_tokens(parsed: Any) -> Any:
    """Recover the intended key list when the model double-wraps it.

    Qwen3.5 sometimes emits ``<parameter=keys>["['ctrl", "a']"]</parameter>`` — a
    *stringified* list nested inside the list literal. ``ast.literal_eval`` parses
    that to ``["['ctrl", "a']"]``: two tokens (``['ctrl`` and ``a']``) that no key
    backend can resolve, so the whole chord (here Ctrl+A) silently no-ops. Strip
    leading/trailing bracket, brace, quote and space junk from each token and drop
    any that become empty, so ``['ctrl`` -> ``ctrl`` and ``a']`` -> ``a``.

    Guard for legitimate single-char punctuation keys: ``['ctrl', '[']`` (vim's
    "go back", a real chord that ``_PUNCT`` resolves as ``ctrl+bracketleft``)
    must NOT be stripped to ``['ctrl']`` — a single-char token IS the key, not
    junk. Only multi-char tokens go through the strip path; single-char tokens
    pass through unchanged. This keeps ``ctrl+[`` / ``ctrl+]`` / ``ctrl+'`` /
    ``ctrl+(`` / ``ctrl+)`` from being silently no-op'd by the cleanup itself —
    same honesty hazard ``server.py``'s ``"No such key name"`` guard avoids on
    the transport side."""
    if not isinstance(parsed, (list, tuple)):
        return parsed
    cleaned: list[Any] = []
    for t in parsed:
        if not isinstance(t, str):
            cleaned.append(t)
            continue
        s = t
        if len(s) > 1:
            s = s.strip().strip("[](){}").strip().strip("'\"").strip()
        if s:
            cleaned.append(s)
    return cleaned


def _param_types_from_tool_schemas(
    tool_schemas: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """``{tool_name: {param_name: json_type}}`` from Lite tool schemas.

    The schemas are exactly the ones rendered into the ``<tools>`` block the
    model was shown (``BaseAgentAdapter._assemble_tool_schemas`` = action-space
    schemas + ``metadata.extra_tool_schemas``), so the declared type of every
    advertised parameter is available at parse time. Properties without a
    string ``type`` (``anyOf``/``$ref``-style entries) are skipped and fall
    back to the name-keyed heuristics.
    """
    out: dict[str, dict[str, str]] = {}
    for schema in tool_schemas:
        properties = tool_schema_parameters(schema).get("properties") or {}
        types = {
            pname: prop["type"]
            for pname, prop in properties.items()
            if isinstance(prop, dict) and isinstance(prop.get("type"), str)
        }
        if types:
            out[tool_schema_name(schema)] = types
    return out


def _fallback_json_type(key: str) -> str | None:
    """Declared-type stand-in for a parameter no schema describes.

    Name-allowlist stand-in for the schema-less entry points; returns ``None``
    (→ string handling) for anything else.
    """
    if key in _STRUCTURED_KEYS:
        return "array"
    if key in _INT_KEYS:
        return "integer"
    if key in _FLOAT_KEYS:
        return "number"
    return None


def _coerce_param_value(
    key: str, raw: str, json_type: str | None = None, *, action: str | None = None
) -> Any:
    """Best-effort conversion of a ``<parameter=KEY>VALUE</parameter>`` body
    string to a Python value, driven by the parameter's DECLARED JSON type.

    *json_type* is the ``type`` the tool's own schema gives this parameter
    (``"integer"`` / ``"number"`` / ``"array"`` / ``"object"`` / ``"boolean"`` /
    ``"string"``). ``None`` means no schema declares it, in which case
    :func:`_fallback_json_type` supplies a name-based guess.
    The XML wire format carries every value as text, so without this a
    standalone extra tool's ``scroll(delta_x: number)`` arrives as ``"135"``
    and blows up inside the env (``unsupported operand type(s) for /``).

    Conversion is TOLERANT: a value that does not parse as its declared type
    is returned as the raw (quote-stripped) string rather than raising —
    callers depend on always getting a value back. *raw* is always wire text
    (the sole caller passes a regex group), so a malformed numeric body is a
    ``ValueError``; a non-str *raw* would be caller breakage and propagates.
    """
    value = raw.strip()
    if json_type is None:
        json_type = _fallback_json_type(key)
    if json_type in ("array", "object"):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
        return _clean_key_tokens(parsed) if key == "keys" else parsed
    if json_type == "integer":
        try:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError("non-finite integer")
            return int(parsed)
        except ValueError:
            return value
    if json_type == "number":
        try:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError("non-finite number")
            return parsed
        except ValueError:
            return value
    if json_type == "boolean":
        lowered = value.strip("'\"").lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        return value
    # Strings (and undeclared params): strip wrapping quotes the model may add
    # verbatim. A numeric-looking value stays a string — browsergym ``bid``s
    # ("12") are strings by schema and must not become ints.
    #
    # ``type``'s text is exempt: it is CONTENT to be typed, so a wrapping quote
    # pair is part of the payload (35 real actions arrived mangled without this).
    # Scoped to the ACTION, not the parameter name -- ``answer``/``call_user``
    # also spell their payload ``text``, and there the quotes are the model's.
    if not (key == "text" and action == "type") and (
        len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'")
    ):
        value = value[1:-1]
    return value


def _reclaim_swallowed_parameter(
    key: str,
    raw: str,
    declared: dict[str, str],
) -> tuple[str, str]:
    """Undo an unclosed opener having swallowed the parameter that follows it.

    ``_PARAM_RE`` gives the leftmost opener the closer, so
    ``<parameter=computer_use>\n<parameter=action>\nscroll\n</parameter>``
    parses as ``("computer_use", "<parameter=action>\nscroll")`` and the real
    ``action`` is gone. The identical byte pattern is also how a legitimate
    ``type`` whose text QUOTES XML looks, so the tie is broken on ``declared``
    (the schema the model was actually shown for this tool):

    * outer key undeclared AND the nested key declared -> the outer opener was
      stray; return the nested pair.
    * anything else (outer declared, nested undeclared, or no schema at all)
      -> leave the match alone, so quoted XML round-trips verbatim.
    """
    if key in declared or "<parameter=" not in raw:
        return key, raw
    nested = _NESTED_PARAM_RE.match(raw.lstrip())
    if nested is None or nested.group(1).strip() not in declared:
        return key, raw
    return nested.group(1).strip(), nested.group(2).strip()


def _parse_xml_tool_calls(
    text: str,
    param_types: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Extract XML-format ``<tool_call>`` blocks from *text*.

    Returns a list of bare Qwen tool-call dicts::

        {"name": str, "arguments": dict}

    *param_types* is ``{tool_name: {param_name: json_type}}`` as produced by
    :func:`_param_types_from_tool_schemas` — the declared types of the tools the
    model was actually shown. Parameters absent from it (unknown tool, unknown
    key, or no schemas passed at all) fall back to the name heuristics in
    :func:`_fallback_json_type`. Unterminated tool_call blocks are skipped with
    a warning.
    """
    types_by_tool = param_types or {}
    calls: list[dict[str, Any]] = []
    for tc_match in _TOOL_CALL_RE.finditer(text):
        body = tc_match.group(1)
        fn_match = _FUNCTION_RE.search(body)
        if not fn_match:
            logger.warning("qwen3_5: <tool_call> with no <function=...> block: %r", body[:120])
            continue
        name = fn_match.group(1).strip()
        fn_body = fn_match.group(2)
        raw_params = [(pm.group(1).strip(), pm.group(2)) for pm in _PARAM_RE.finditer(fn_body)]
        declared = types_by_tool.get(name, {})
        # The model may keep the wrapper but name another advertised tool in the
        # ``action`` slot (``computer_use(action=switch_tab, index=0)``). The
        # remaining parameters are that tool's, so its declared types drive their
        # coercion — otherwise ``index`` stays the string ``"0"`` and fails
        # ``switch_tab``'s integer schema once the adapter lifts the call back to
        # the top level. Keys the wrapper declares itself still win.
        acted_on = next((v.strip() for k, v in raw_params if k == "action"), None)
        if acted_on in types_by_tool:
            declared = {**types_by_tool[acted_on], **declared}
        arguments: dict[str, Any] = {}
        for pkey, praw in raw_params:
            pkey, praw = _reclaim_swallowed_parameter(pkey, praw, declared)
            arguments[pkey] = _coerce_param_value(
                pkey, praw, declared.get(pkey), action=acted_on
            )
        calls.append({"name": name, "arguments": arguments})
    return calls


def _render_param_value(value: Any) -> str:
    """Inverse of :func:`_coerce_param_value`. Single-line repr preferred."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(list(value) if isinstance(value, tuple) else value)
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _render_xml_tool_call(tc: dict[str, Any]) -> str:
    name = tc["name"]
    args = tc["arguments"]
    lines = ["<tool_call>", f"<function={name}>"]
    for k, v in args.items():
        lines.append(f"<parameter={k}>")
        lines.append(_render_param_value(v))
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


# =============================================================================
# Qwen3.5 XML tool preamble
# =============================================================================

# Model-family wire-format scaffolding for XML tool calls. This belongs to the
# Qwen3.5 adapter because the model expects <tools> plus <tool_call>/<function>
# examples in the system text; env/task prompts are appended separately.
# Runtime-only reference lines are intentionally omitted for reproducibility:
#
#   * ``- The current date is {datetime.today().strftime(...)}``
#     — non-stable across runs; would break SFT replay / caching.
#   * ``- Collapsed screenshots appear as text: {collapse_text}``
#     — we don't fold images; no runtime state to render here.
#
# Everything else in this block is stable model-side tool grammar.

QWEN35_XML_TOOLS_PREAMBLE = """You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.

# Tools

You have access to the following functions:

<tools>
{tools_json}
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
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""

# USE_SYSTEM_PROMPT is shared verbatim with Qwen3-VL — imported at the top
# of this module (single source of truth in qwen3_vl/adapter.py) rather than redefined.


# =============================================================================
# Base adapter
# =============================================================================


@dataclasses.dataclass
class Qwen3_5BaseAdapter(
    Qwen3VLBaseAdapter,
    key=(
        r"qwen3_5\.base"
        r"(@(desktop|browser|mobile)"
        r"@(use|understanding|grounding\.action|grounding\.point|grounding\.bbox))?"
    ),
):
    """Qwen3.5 adapter family. Self-contained: chat-template-level XML
    tool_call rendering on top of :class:`Qwen3VLBaseAdapter`.

    Decoupled from any specific workflow / system prompt. The base
    handles only the platform-invariant Qwen3.5 specifics:

      * XML ``<tool_call><function=...>...</function></tool_call>``
        rendering inline in the assistant text content (NOT in
        ``tool_calls`` field — chat_template-independent);
      * Qwen3.5 message order (tools-section first, then
        ``system_prompt``);
      * canonical ``role:"tool"`` observations pass through so the Qwen3.5
        chat template owns ``<tool_response>`` grouping.

    Workflow-specific structured content (``action_description`` /
    ``inline_reasoning``, ``Thought:`` / ``Action:`` line conventions,
    multi-part Memory/Progress blocks) is the responsibility of
    :class:`Qwen3_5UseAdapter` and below — base does NOT preset
    any of those.

    BrowserGym agents resolve directly to base via the platform/task wildcard
    key; yaml ``valid_actions: []`` + ``extra_tools`` fully override the
    action surface.

    The ``enable_thinking`` toggle is inherited from
    :class:`Qwen3VLBaseAdapter` (default ``False``). Qwen3.5-specific note:
    its chat_template.jinja unconditionally prepends ``<think>\\n`` to the
    assistant turn's generation unless ``enable_thinking=False`` is
    passed, in which case the template emits an empty
    ``<think>\\n\\n</think>\\n\\n`` block that suppresses thinking. With
    thinking on, the 9B burns a few hundred tokens reasoning before
    ``Action:`` + ``<tool_call>``, which inflates latency and makes
    ``max_new_tokens=1024`` insufficient. Flip to ``True`` when you
    explicitly want the reasoning channel. The flag is consumed by
    :class:`Qwen3_5BaseAgent.build_generation_prompt` which forwards it
    as ``processor.apply_chat_template(..., enable_thinking=...)``.

    ``system_prompt`` defaults are set per workflow subclass:
    :class:`Qwen3_5UseAdapter` pins :data:`USE_SYSTEM_PROMPT`
    (XML-flavored, no ``enable_inline_reasoning`` toggle; reasoning happens
    in the native ``<think>`` channel gated by ``enable_thinking``).
    Grounding/action adapters pin :data:`GROUNDING_POINT_SYSTEM_PROMPT`,
    and the base leaves it at None so the system message is just the
    tools schema.
    """

    # ``action_space`` + ``protocol`` inherit the workflow-agnostic
    # defaults from :class:`Qwen3VLBaseAdapter` (``BaseActionSpace`` +
    # ``FullHistoryProtocol``). Concrete use/grounding leaves
    # below override these per platform.

    def _build_tools_section(self) -> str:
        """Emit the intro + ``<tools>`` block + format example + ``<IMPORTANT>``.

        This is the Qwen3.5 XML tool-call grammar block. It is model-family
        wire-format scaffolding, not an env/task prompt.
        """
        tool_schemas = self._tool_schemas_for_tools_section()
        # Byte policy (this family owns it; there is no shared Qwen prompt
        # helper): one JSON object per line, ``json.dumps`` defaults, so
        # non-ASCII escapes as ``\uXXXX``. The reference uses a single
        # ``json.dumps(tools_def)`` holding one ``computer_use`` function;
        # cua-lite supports filtering + multi-tool (e.g. webgym ``goto`` /
        # ``back`` extras), so schemas are newline-joined instead — the model
        # still parses correctly.
        validate_extra_tool_schemas(
            tool_schemas,
            where="Qwen3_5BaseAdapter._build_tools_section.tool_schemas",
        )
        tools_json = "\n".join(json.dumps(schema) for schema in tool_schemas)
        return QWEN35_XML_TOOLS_PREAMBLE.format(tools_json=tools_json)

    def _xml_param_types(self) -> dict[str, dict[str, str]]:
        """Declared parameter types of every tool this sample advertises.

        Read straight off the assembled schemas (action space +
        ``metadata.extra_tool_schemas``) — i.e. the same objects
        :meth:`_build_tools_section` serialises into the prompt's ``<tools>``
        block — so the parse side coerces by the contract the model was
        actually given. Uses ``_assemble_tool_schemas`` rather than
        ``_tool_schemas_for_tools_section`` so native finish extras hidden
        behind the provider-native wrapper still coerce correctly if emitted.
        """
        return _param_types_from_tool_schemas(self._assemble_tool_schemas())

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render turn ``k``: protocol on truncated history + Qwen3.5-order
        system prompt.

        Qwen3.5 message order:

            QWEN35_XML_TOOLS_PREAMBLE  (tools + IMPORTANT)
            \\n\\n
            USE_SYSTEM_PROMPT  (# Response format)

        The parent (``Qwen3VLBaseAdapter.render_step``) joins
        ``[self.system_prompt, self._build_tools_section()]`` which would
        put ``# Response format`` *before* ``# Tools`` — off-distribution
        for Qwen3.5; this override flips the order.

        Canonical ``role:"tool"`` observations pass through unchanged so
        Qwen3.5's chat template groups and wraps them.
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        result_messages: list[AgentMessage] = []

        # Qwen3.5 order: tools first, then response-format.
        parts: list[str] = []
        if self.render_tools_section:
            parts.append(self._build_tools_section())
        system_prompt = self._system_prompt_for_active_surface()
        if system_prompt:
            parts.append(system_prompt)
        if parts:
            system_text = "\n\n".join(parts)
            result_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            })

        for msg in messages:
            converted = self.convert_message_to_agent(msg)
            result_messages.append(converted)

        return result_messages

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """LiteMessage → AgentMessage. Renders tool_calls inline as XML
        ``<tool_call><function=...>...</function></tool_call>`` blocks
        inside the text content (NOT in the message's ``tool_calls``
        field). Keeps the wire format independent of whatever XML/JSON
        rendering the chat_template applies for ``tool_calls``.

        For assistant content, picks ONLY the ``type: "text"`` parts
        and concatenates their ``text`` fields. Other content types
        (``action_description``, ``inline_reasoning``, etc.) are
        workflow-specific and are NOT consumed by base —
        :class:`Qwen3_5UseAdapter` overrides this method to
        render them as ``Thought:`` / ``Action:`` lines.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        had_tool_calls = bool(result.get("tool_calls"))
        tc_xml_blocks: list[str] = []
        if had_tool_calls:
            agent_calls = self._tool_calls_to_agent_ordered(result["tool_calls"])
            tc_xml_blocks = [_render_xml_tool_call(tc) for tc in agent_calls]
            result.pop("tool_calls", None)

        content = message.get("content") or []
        text_lines = [
            p["text"] for p in content
            if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
        ]
        lines: list[str] = list(text_lines)
        if tc_xml_blocks:
            lines.extend(tc_xml_blocks)

        if lines:
            result["content"] = [{"type": "text", "text": "\n".join(lines)}]
        elif not had_tool_calls:
            pass
        else:
            result["content"] = []
        return result

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """Parse raw model output into an ``AgentMessage``.

        Extracts (chat-template-level tokens only):
          * ``<think>...</think>`` → top-level ``reasoning_content``.
          * ``<tool_call><function=...>...</function></tool_call>`` → top-level
            ``tool_calls`` in raw agent format (``computer_use`` / etc.).

        The prose remainder (minus those tokens) goes into a single text
        content part. The ``Action:`` / ``Thought:`` line extraction happens
        later in :meth:`convert_message_from_agent`.
        """
        result: AgentMessage = {"role": "assistant"}

        m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if m:
            result["reasoning_content"] = m.group(1).strip()
        elif "</think>" in response:
            result["reasoning_content"] = response.split("</think>", 1)[0].strip()

        tool_calls = _parse_xml_tool_calls(response, self._xml_param_types())
        if tool_calls:
            result["tool_calls"] = tool_calls
        elif "<tool_call" in response or "</tool_call>" in response:
            mark_model_output_error(result, "malformed <tool_call> XML")

        if "</think>" in response:
            clean = response.split("</think>", 1)[-1]
        elif "<think>" in _TOOL_CALL_RE.sub("", response):
            # An unclosed opener: everything after it is reasoning, and must
            # not fall through into published content. Both the test and the
            # split run on the tool-call-scrubbed text, because a ``type``
            # payload may contain the literal characters ``<think>``.
            head, _, tail = _TOOL_CALL_RE.sub("", response).partition("<think>")
            trailing = tail.strip()
            if trailing:
                result["reasoning_content"] = trailing
            clean = head
        else:
            clean = response
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
        clean = _TOOL_CALL_RE.sub("", clean)
        clean = clean.strip()
        if clean:
            result["content"] = [{"type": "text", "text": clean}]

        return result


# =============================================================================
# ``use`` adapter (intermediate)
# =============================================================================


@dataclasses.dataclass
class Qwen3_5UseAdapter(Qwen3_5BaseAdapter):
    """Intermediate adapter that adds the ``use`` wire format on top of
    :class:`Qwen3_5BaseAdapter`. Concrete ``use`` adapters (desktop,
    browser, mobile) inherit from this class.

    Wire format on the to-agent side:
      * 2-part: ``Action: <text>`` + XML ``<tool_call>`` block
        (default — ``enable_inline_reasoning=False``).
      * 3-part: ``Thought: <text>`` + ``Action: <text>`` + XML
        ``<tool_call>`` block (``enable_inline_reasoning=True``).

    Structured ``action_description`` / ``inline_reasoning`` content parts
    render as ``Action:`` / ``Thought:`` body lines (the latter only when
    :attr:`enable_inline_reasoning`); the XML tool_call blocks are appended
    after. On the from-agent side, the symmetric regexes parse those lines
    back out into structured parts.

    Decomposed only — same contract as :class:`Qwen3VLUseAdapter`
    (opaque/verbatim turns use a passthrough adapter, never a branch here).
    The only Qwen3.5-specific difference is the chat-template-level XML inline
    tool_call rendering, which lives in the :class:`Qwen3_5BaseAdapter` parent.

    Note on ``system_prompt``: pinned here (default
    :data:`USE_SYSTEM_PROMPT`) because the ``Thought:`` / ``Action:``
    wire format is pinned here. Concrete platform subclasses inherit it.
    """

    enable_inline_reasoning: bool = False
    system_prompt: str | None = USE_SYSTEM_PROMPT

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Render structured ``action_description`` (+ optional
        ``inline_reasoning``) parts as ``Action:`` / ``Thought:`` lines, then
        append the XML ``<tool_call>`` block(s)."""
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        had_tool_calls = bool(result.get("tool_calls"))
        tc_xml_blocks: list[str] = []
        if had_tool_calls:
            agent_calls = self._tool_calls_to_agent_ordered(result["tool_calls"])
            tc_xml_blocks = [_render_xml_tool_call(tc) for tc in agent_calls]
            result.pop("tool_calls", None)

        content = message.get("content") or []
        lines: list[str] = []
        if self.enable_inline_reasoning:
            lines += [
                f"Thought: {p['text']}" for p in content
                if p.get("type") == "inline_reasoning" and p.get("text")
            ]
        lines += [
            f"Action: {p['text']}" for p in content
            if p.get("type") == "action_description" and p.get("text")
        ]
        lines += tc_xml_blocks

        if lines:
            result["content"] = [{"type": "text", "text": "\n".join(lines)}]
        elif not had_tool_calls:
            pass
        else:
            result["content"] = []
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Parse the ``Action:`` line into ``action_description`` and, when
        :attr:`enable_inline_reasoning`, the ``Thought:`` line into
        ``inline_reasoning``. Mirrors
        :meth:`Qwen3VLUseAdapter.convert_message_from_agent`.

        The retag applies only to a turn that actually carries ``tool_calls``.
        A no-tool-call turn is the text-final path, so its prose must remain a
        plain ``text`` part for ``no_tool_call_final_text`` and DAgger round-trip
        replay.

        Tool-call routing + deepcopy are handled by
        :meth:`Qwen3VLBaseAdapter.convert_message_from_agent` (super).
        """
        result = super().convert_message_from_agent(message, **kwargs)
        if result.get("role") != "assistant":
            return result
        if not result.get("tool_calls"):
            return result

        raw_text = ""
        for part in result.get("content") or []:
            if part.get("type") == "text" and part.get("text"):
                raw_text = part["text"]
                break

        parts: list[dict[str, Any]] = []
        if raw_text:
            inline_reasoning = ""
            if self.enable_inline_reasoning:
                # Stop at the ``\nAction:`` line, not the first ``\n`` — the
                # inline reasoning is multi-line (matches the to-agent render
                # at ``_convert_message_to_agent``, which emits the full body).
                # ``\Z`` (not ``$``) so a multi-line body is not clipped.
                m = re.search(r"Thought:\s*(.*?)(?:\n(?=Action:)|\Z)", raw_text, re.DOTALL)
                if m:
                    inline_reasoning = m.group(1).strip()
            # Capture the action body from the LAST ``Action:`` line to end-of-
            # string (symmetric with Thought, so a multi-line action_description
            # round-trips intact). The greedy ``.*\n`` skips any ``Action:``-prefixed
            # line nested inside the thought body and anchors on the FINAL one.
            m = re.search(r"(?:.*\n)?Action:\s*(.*)\Z", raw_text, re.DOTALL)
            if m:
                action_text = m.group(1).strip()
            else:
                # No ``Action:`` marker: the WHOLE reply is the action
                # description. Keeping only its first line used to drop the
                # model's reasoning -- 17% of tool-call turns, a median of 232
                # characters each -- and this teacher runs with thinking OFF, so
                # that prose IS its reasoning. Nothing at runtime wanted the
                # truncation either: same-family replay re-renders
                # ``raw_response.text`` verbatim, so it only ever cost the
                # durable record, which is exactly what a cross-family student
                # is trained on.
                action_text = raw_text.strip()
            parts = make_assistant_content(
                inline_reasoning=inline_reasoning, action_description=action_text,
            )
        result["content"] = parts
        return result


# =============================================================================
# Desktop + Browser Adapters
# =============================================================================


# Desktop and browser share one adapter class per task type — same action
# space, protocol, and system prompt. The (desktop|browser) regex on each
# class's key registers the same body under both ``qwen3_5@desktop@...``
# and ``qwen3_5@browser@...`` so WebGym-style envs route to the desktop
# harness (no dedicated Qwen3.5 browser reference exists).

@dataclasses.dataclass
class Qwen3_5DesktopGroundingActionAdapter(
    Qwen3_5BaseAdapter, key=r"qwen3_5@(desktop|browser)@grounding\.action"
):
    """Desktop+browser grounding/action: full action vocabulary, single turn.

    Currently NOT routed by env eval (env declares ``grounding.point``;
    see :class:`Qwen3_5DesktopGroundingPointAdapter`). Kept available for
    SFT-data replay: family-native XML ``<function=computer_use>`` wire
    format with the full Qwen3-VL desktop action enum.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_5DesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = LiteFinishToolSet.get_tool_names()


@dataclasses.dataclass
class Qwen3_5DesktopGroundingPointAdapter(
    Qwen3_5BaseAdapter, key=r"qwen3_5@(desktop|browser)@grounding\.point"
):
    """Desktop+browser grounding (single-step click) for Qwen3.5.

    Uses the trimmed :class:`Qwen3_5DesktopGroundingPointActionSpace`
    (``computer_use`` with only ``left_click`` + ``terminate``) plus the
    minimal :data:`GROUNDING_POINT_SYSTEM_PROMPT` (shared with Qwen3-VL —
    schema is identical, only the wire format differs). Single turn, full
    history. The agent emits exactly one tool_call.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_5DesktopGroundingPointActionSpace
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
class Qwen3_5DesktopUseAdapter(
    Qwen3_5UseAdapter, key=r"qwen3_5@(desktop|browser)@use"
):
    """Desktop+browser ``use`` (multi-step rollout): Qwen3.5 rolling-window + image-fold history +
    XML tool_call wire format.
    One class for both platforms: browser nav verbs arrive as env
    extra_tools through the shared action space, not a per-platform subclass.

    Default ``enable_inline_reasoning=False`` → 2-part Action + ``<tool_call>`` (the
    reference has no Thought: line).
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_5DesktopActionSpace
    )
    protocol: Qwen3_5HistoryProtocol = dataclasses.field(
        default_factory=Qwen3_5HistoryProtocol
    )
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = LiteFinishToolSet.get_tool_names()


# =============================================================================
# Mobile Adapters
# =============================================================================


@dataclasses.dataclass
class Qwen3_5MobileGroundingActionAdapter(Qwen3_5BaseAdapter, key="qwen3_5@mobile@grounding.action"):
    """Mobile grounding/action: full mobile action vocabulary. SFT-replay only."""
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_5MobileActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteFinishToolSet.get_tool_names() | LiteAppLaunchToolSet.get_tool_names()
    )


@dataclasses.dataclass
class Qwen3_5MobileGroundingPointAdapter(Qwen3_5BaseAdapter, key="qwen3_5@mobile@grounding.point"):
    """Mobile grounding (single-step click) — same trimmed harness as
    desktop. See :class:`Qwen3_5DesktopGroundingPointAdapter`.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_5MobileGroundingPointActionSpace
    )
    metadata: LiteCUAMetadata = dataclasses.field(
        default_factory=lambda: LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.MOBILE.value,
                LiteCUAMetadata.TaskType.GROUNDING_POINT.value,
            )
        )
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    system_prompt: str | None = GROUNDING_POINT_SYSTEM_PROMPT


@dataclasses.dataclass
class Qwen3_5MobileUseAdapter(Qwen3_5UseAdapter, key="qwen3_5@mobile@use"):
    """Mobile ``use`` (multi-step rollout; SPECULATIVE — no upstream reference).

    The Qwen3.5 OSWorld reference (``mm_agents/qwen35vl_agent.py``) is
    desktop-only; there is no published mobile agent for Qwen3.5 yet. We
    extrapolate here by reusing the ``qwen3_vl`` mobile action space:

    * :class:`Qwen3_5MobileActionSpace` inherits ``Qwen3VLMobileActionSpace``
      (the mobile ``mobile_use`` tool: ``click / long_press / swipe /
      type / open / answer / system_button / wait / terminate``).
    * ``smart_resize_enabled=False`` — emulator screenshots are already
      small enough that 32-px rounding would shift coordinates.

    Use with care on androidworld; the wire format is
    believed correct by construction (the mobile chat_template of
    Qwen3.5 is the same XML-tools format as desktop) but behavior has
    not been validated against a reference implementation.

    Same as qwen3_vl mobile, ``enable_inline_reasoning=False`` and the canonical
    desktop history protocol (:class:`Qwen3_5HistoryProtocol` here,
    :class:`Qwen3VLHistoryProtocol` there) — Qwen3.5 has no dedicated
    mobile flow, so we just reuse the desktop summary
    (``Please generate the next move ... Previous actions: ...``).
    Reasoning, if desired, rides the native ``<think>`` channel via
    :attr:`enable_thinking`.
    """
    # ─── Core ────────────────────────────────────────────────────────
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_5MobileActionSpace
    )
    protocol: Qwen3_5HistoryProtocol = dataclasses.field(
        default_factory=Qwen3_5HistoryProtocol
    )
    # ─── Class-specific knobs ────────────────────────────────────────
    # (system_prompt + enable_inline_reasoning inherited from Qwen3_5UseAdapter)
    smart_resize_enabled: bool = False
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteFinishToolSet.get_tool_names() | LiteAppLaunchToolSet.get_tool_names()
    )


# =============================================================================
# Pass-through Adapters
# =============================================================================

AgentAdapterRegistry.register(r"qwen3_5@(desktop|browser|mobile)@understanding", AsIsAdapter)
AgentAdapterRegistry.register(r"qwen3_5@(desktop|browser|mobile)@grounding\.bbox", AsIsAdapter)
# ``grounding.point`` and ``grounding.action`` are both served by concrete
# per-platform classes above (exact-key match in the registry).
