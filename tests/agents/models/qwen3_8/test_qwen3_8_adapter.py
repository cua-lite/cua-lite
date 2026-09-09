"""Qwen3.8 adapter tests — wiring and the expanded response format.

Run:
    uv run pytest tests/agents/models/qwen3_8/test_qwen3_8_adapter.py -v

The XML wire format itself is Qwen3.5's and is covered by
``tests/agents/models/qwen3_5/test_qwen3_5_adapter.py``. These tests pin the Qwen3.8
delta and the wiring that carries it:

  * every registry key resolves, to the right class, with the right
    action space / protocol / system prompt bound.
  * the expanded ``# Response format`` matches the upstream
    ``build_internal_system_prompt`` tail, and licenses a terminal turn with
    no tool call — the behavior a real rollout exercised on turn 1.
  * mobile keeps the Qwen3.5 response format, because the mobile enum has
    ``answer`` and no ``call_user``.
  * the rendered system message carries the expanded enum, in Qwen3.5's
    tools-before-response-format order.
  * ``enable_thinking`` stays off by default even though the Qwen3.8
    checkpoint ships it on.
"""

from __future__ import annotations

import json
import re

import pytest

from lite.agents.core.adapter import AgentAdapterRegistry, AsIsAdapter
from lite.agents.models.qwen3_5.adapter import (
    USE_SYSTEM_PROMPT,
    Qwen3_5UseAdapter,
    _parse_xml_tool_calls,
    _render_xml_tool_call,
)
from lite.agents.models.qwen3_5.protocol import Qwen3_5HistoryProtocol
from lite.agents.models.qwen3_8 import agent as _qwen3_8_agent  # noqa: F401 — register agents
from lite.agents.models.qwen3_8.action_space import (
    Qwen3_8DesktopActionSpace,
    Qwen3_8DesktopGroundingPointActionSpace,
    Qwen3_8MobileActionSpace,
    Qwen3_8MobileGroundingPointActionSpace,
)
from lite.agents.models.qwen3_8.adapter import (
    QWEN38_USE_SYSTEM_PROMPT,
    Qwen3_8BaseAdapter,
    Qwen3_8DesktopGroundingActionAdapter,
    Qwen3_8DesktopGroundingPointAdapter,
    Qwen3_8DesktopUseAdapter,
    Qwen3_8MobileGroundingActionAdapter,
    Qwen3_8MobileGroundingPointAdapter,
    Qwen3_8MobileUseAdapter,
)
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages import no_tool_call_final_text
from lite.core.tools.schemas import tool_schema_parameters


async def _dummy_gen(**_):
    return {"response": ""}


class _DummyProcessor:
    def apply_chat_template(self, messages, **kwargs):
        return ""


NO_TOOL_FINAL_TEXTS = (
    "Done.",
    "Done! I've restored the last closed tab. It was a Tripadvisor page.",
)


def _md(platform=LiteCUAMetadata.Platform.DESKTOP, extra_tools=None, valid_actions=None):
    return LiteCUAMetadata(
        dims=(platform, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tools or [],
        valid_actions=valid_actions,
    )


# =============================================================================
# Registry wiring
# =============================================================================


@pytest.mark.parametrize(
    "key,cls",
    [
        ("qwen3_8@desktop@use", Qwen3_8DesktopUseAdapter),
        ("qwen3_8@browser@use", Qwen3_8DesktopUseAdapter),
        ("qwen3_8@desktop@grounding.action", Qwen3_8DesktopGroundingActionAdapter),
        ("qwen3_8@browser@grounding.action", Qwen3_8DesktopGroundingActionAdapter),
        ("qwen3_8@desktop@grounding.point", Qwen3_8DesktopGroundingPointAdapter),
        ("qwen3_8@browser@grounding.point", Qwen3_8DesktopGroundingPointAdapter),
        ("qwen3_8@mobile@use", Qwen3_8MobileUseAdapter),
        ("qwen3_8@mobile@grounding.action", Qwen3_8MobileGroundingActionAdapter),
        ("qwen3_8@mobile@grounding.point", Qwen3_8MobileGroundingPointAdapter),
    ],
)
def test_adapter_registry_lookup(key, cls):
    assert isinstance(AgentAdapterRegistry.get(key), cls)


@pytest.mark.parametrize(
    "key",
    [
        "qwen3_8@desktop@understanding",
        "qwen3_8@mobile@understanding",
        "qwen3_8@desktop@grounding.bbox",
        "qwen3_8@mobile@grounding.bbox",
    ],
)
def test_passthrough_pattern_keys_resolve(key):
    assert isinstance(AgentAdapterRegistry.get(key), AsIsAdapter)


@pytest.mark.parametrize(
    "key",
    [
        "qwen3_8.base",
        "qwen3_8.base@desktop@use",
        "qwen3_8.base@browser@use",
        "qwen3_8.base@mobile@use",
    ],
)
def test_base_wildcard_resolves_to_base_adapter(key):
    adapter = AgentAdapterRegistry.get(key)
    assert type(adapter) is Qwen3_8BaseAdapter
    assert not isinstance(adapter, Qwen3_5UseAdapter)


@pytest.mark.parametrize(
    "key",
    [
        "qwen3_8@desktop@use",
        "qwen3_8@browser@use",
        "qwen3_8@desktop@grounding.action",
        "qwen3_8@desktop@grounding.point",
        "qwen3_8@mobile@use",
        "qwen3_8@mobile@grounding.action",
        "qwen3_8@mobile@grounding.point",
    ],
)
def test_agent_registry_lookup(key):
    """The agent registry must mirror the adapter registry, or ``make`` cannot
    build the family — the audit-surfaced gap where one side was registered
    and the other was not."""
    from lite.agents.models import AgentRegistry
    from lite.agents.models.qwen3_8.agent import Qwen3_8BaseAgent

    agent = AgentRegistry.get(key, generate_fn=_dummy_gen, processor=_DummyProcessor())
    assert isinstance(agent, Qwen3_8BaseAgent), key


# =============================================================================
# Bound components
# =============================================================================


@pytest.mark.parametrize(
    "adapter_cls,space_cls",
    [
        (Qwen3_8DesktopUseAdapter, Qwen3_8DesktopActionSpace),
        (Qwen3_8DesktopGroundingActionAdapter, Qwen3_8DesktopActionSpace),
        (Qwen3_8DesktopGroundingPointAdapter, Qwen3_8DesktopGroundingPointActionSpace),
        (Qwen3_8MobileUseAdapter, Qwen3_8MobileActionSpace),
        (Qwen3_8MobileGroundingActionAdapter, Qwen3_8MobileActionSpace),
        (Qwen3_8MobileGroundingPointAdapter, Qwen3_8MobileGroundingPointActionSpace),
    ],
    ids=lambda x: getattr(x, "__name__", x),
)
def test_adapters_bind_the_qwen3_8_action_spaces(adapter_cls, space_cls):
    assert isinstance(adapter_cls().action_space, space_cls)


@pytest.mark.parametrize(
    "adapter_cls", [Qwen3_8DesktopUseAdapter, Qwen3_8MobileUseAdapter], ids=lambda c: c.__name__
)
def test_use_adapters_share_the_qwen3_5_history_protocol(adapter_cls):
    """Upstream's ``mm_agents/qwen/history.py`` is the same rolling-window +
    image-fold mechanism Qwen3.5 uses, so the protocol is reused, not aliased."""
    assert isinstance(adapter_cls().protocol, Qwen3_5HistoryProtocol)


def test_mobile_use_adapter_disables_smart_resize():
    """Emulator screenshots are small enough that 32-px rounding would shift
    coordinates."""
    assert Qwen3_8MobileUseAdapter().smart_resize_enabled is False
    assert Qwen3_8DesktopUseAdapter().smart_resize_enabled is True


# =============================================================================
# Expanded response format
# =============================================================================


# The tail of ``mm_agents/qwen/prompts.py::build_internal_system_prompt``,
# from ``# Response format`` on. Frozen so prompt drift is a test failure.
UPSTREAM_INTERNAL_RESPONSE_FORMAT = """# Response format

For normal UI interaction steps:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block.

For terminal steps, you may either:
- output a final natural-language response with no tool call, or
- use a terminal tool call such as call_user or terminate.

Rules:
- For non-terminal UI steps, output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything after a tool call.
- Use call_user when you need user information or confirmation.
- Use terminate when you want to explicitly end the task with a success or failure status.
- If the task is infeasible, say so explicitly in the response."""


def test_expanded_response_format_matches_upstream():
    assert QWEN38_USE_SYSTEM_PROMPT == UPSTREAM_INTERNAL_RESPONSE_FORMAT


def test_desktop_use_adapter_pins_the_expanded_response_format():
    assert Qwen3_8DesktopUseAdapter().system_prompt == QWEN38_USE_SYSTEM_PROMPT
    assert Qwen3_8DesktopUseAdapter().system_prompt != USE_SYSTEM_PROMPT


def test_mobile_use_adapter_keeps_the_qwen3_5_response_format():
    """The expanded format names ``call_user``, which the mobile ``mobile_use``
    enum does not have — advertising it there would be a lie."""
    assert Qwen3_8MobileUseAdapter().system_prompt == USE_SYSTEM_PROMPT


def test_expanded_response_format_licenses_a_tool_call_free_terminal_turn():
    """The load-bearing difference from Qwen3.5, which demands a
    ``<tool_call>`` on every turn."""
    assert "with no tool call" in QWEN38_USE_SYSTEM_PROMPT
    assert "with no tool call" not in USE_SYSTEM_PROMPT


@pytest.mark.parametrize("final_text", NO_TOOL_FINAL_TEXTS, ids=["short", "prose"])
def test_no_tool_call_terminal_turn_stays_plain_text(final_text):
    """A real lite.osworld rollout ended this way on turn 1; the parse path must
    keep it as a text final rather than flagging a missing tool call."""
    adapter = Qwen3_8DesktopUseAdapter()
    out = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(final_text))
    assert not out.get("tool_calls")
    assert no_tool_call_final_text(out) == final_text


# =============================================================================
# Rendered system message
# =============================================================================


def _tools_json(section: str) -> list[dict]:
    match = re.search(r"<tools>\n(.*?)\n</tools>", section, re.DOTALL)
    assert match is not None
    return [json.loads(line) for line in match.group(1).splitlines()]


def test_tools_section_carries_the_expanded_enum():
    section = Qwen3_8DesktopUseAdapter()._build_tools_section()
    enum = tool_schema_parameters(_tools_json(section)[0])["properties"]["action"]["enum"]
    assert {"key_down", "key_up", "left_mouse_down", "left_mouse_up", "screenshot"} <= set(enum)
    # Terminal spellings stay gated by extra_tool_schemas, none active here.
    assert "call_user" not in enum
    assert "terminate" not in enum
    assert "answer" not in enum


def test_system_message_orders_tools_before_the_response_format():
    """Qwen3.5's message order, inherited: ``# Tools`` then ``# Response
    format``. The reverse is off-distribution for this template."""
    adapter = Qwen3_8DesktopUseAdapter(metadata=_md())
    sample = LiteSample(
        messages=[{"role": "user", "content": [{"type": "text", "text": "do a thing"}]}],
        metadata=_md(),
    )
    rendered = adapter.render_step(sample, 1, None)
    system_text = rendered[0]["content"][0]["text"]
    assert system_text.index("# Tools") < system_text.index("# Response format")
    assert system_text.endswith(QWEN38_USE_SYSTEM_PROMPT)


# =============================================================================
# Inherited wire format still works end to end
# =============================================================================


@pytest.mark.parametrize(
    "args",
    [
        {"action": "key_down", "keys": ["ctrl"]},
        {"action": "left_mouse_down", "coordinate": [12, 34]},
        {"action": "screenshot"},
        {"action": "call_user", "text": "need confirmation"},
        {"action": "type", "text": "line one\nline two"},
    ],
    ids=["key_down", "left_mouse_down", "screenshot", "call_user", "multiline-type"],
)
def test_expanded_actions_round_trip_through_the_xml_wire(args):
    tc = {"name": "computer_use", "arguments": args}
    parsed = _parse_xml_tool_calls(_render_xml_tool_call(tc))
    assert parsed == [tc]


def test_wrapper_call_with_no_action_is_a_named_parse_error():
    """A hallucinated parameter with no ``action`` must not reach history.

    Verbatim from a Qwen3.8 lite.osworld rollout
    (``osworld_libreoffice_calc_3a7c8185`` turn 12). Building the unnamed
    action-batch child instead killed the episode with an uncaught ``ValueError``
    on the NEXT turn's history render. The owner-side fix and its reasoning live
    in ``tests/agents/core/action_space/test_unknown_wrapper_action.py``.
    """
    from lite.agents.core.action_space.errors import ModelToolCallParseError

    raw = (
        "<tool_call>\n<function=computer_use>\n"
        "<parameter=command>\ntrue\n</parameter>\n"
        "<parameter=coordinate>\n[263, 798]\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    adapter = Qwen3_8DesktopUseAdapter()
    parsed = adapter.parse_raw_assistant_response(raw)
    with pytest.raises(ModelToolCallParseError):
        adapter.convert_message_from_agent(parsed)


def test_stray_parameter_opener_is_inherited_from_the_qwen3_5_parser():
    """The XML parser is Qwen3.5's, so its malformation handling must reach here.

    Observed on a Qwen3.8 lite.osworld rollout: the model repeated the function
    name as an unclosed ``<parameter=`` tag before the real ``action``, and the
    leftmost-opener match handed it the closer -- producing a wrapper call with
    no action value, which lowers to an unnamed action-batch child and is fatal
    when the turn is replayed. The parser breaks the tie on the schema the model
    was shown; the owner-side fix, and the quoted-XML case it must NOT disturb,
    live in ``tests/agents/models/qwen3_5/test_qwen3_5_adapter.py``.
    """
    raw = (
        "<tool_call>\n<function=computer_use>\n"
        "<parameter=computer_use>\n"
        "<parameter=action>\nscroll\n</parameter>\n"
        "<parameter=pixels>\n-3\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    adapter = Qwen3_8DesktopUseAdapter()
    assert adapter.parse_raw_assistant_response(raw)["tool_calls"] == [
        {"name": "computer_use", "arguments": {"action": "scroll", "pixels": -3}}
    ]


def test_parse_action_line_plus_expanded_tool_call():
    adapter = Qwen3_8DesktopUseAdapter()
    raw = "Action: Hold the ctrl key.\n" + _render_xml_tool_call(
        {"name": "computer_use", "arguments": {"action": "key_down", "keys": ["ctrl"]}}
    )
    msg = adapter.parse_raw_assistant_response(raw)
    assert msg["tool_calls"] == [
        {"name": "computer_use", "arguments": {"action": "key_down", "keys": ["ctrl"]}}
    ]


# =============================================================================
# Thinking
# =============================================================================


@pytest.mark.parametrize(
    "adapter_cls", [Qwen3_8DesktopUseAdapter, Qwen3_8MobileUseAdapter], ids=lambda c: c.__name__
)
def test_enable_thinking_defaults_off(adapter_cls):
    """Qwen3.8 ships thinking ON with ``reasoning_effort`` defaulting to
    ``xhigh``; the eval matrix pins it off so prompts stay bounded."""
    assert adapter_cls().enable_thinking is False


def test_enable_thinking_is_still_opt_in():
    assert Qwen3_8DesktopUseAdapter(enable_thinking=True).enable_thinking is True


@pytest.mark.parametrize("effort", [None, "low", "medium", "xhigh"])
def test_agent_forwards_reasoning_effort(effort):
    from unittest.mock import Mock

    from lite.agents.models import AgentRegistry

    processor = Mock()
    kwargs = {} if effort is None else {"reasoning_effort": effort}
    agent = AgentRegistry.get(
        "qwen3_8@desktop@use",
        processor=processor,
        generate_fn=_dummy_gen,
        enable_thinking=True,
        **kwargs,
    )
    messages = [{"role": "user", "content": []}]
    agent.build_generation_prompt(messages)
    processor.apply_chat_template.assert_called_once_with(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=effort or "xhigh",
    )
