"""Qwen3.5 adapter tests.

Run:
    uv run pytest tests/agents/models/qwen3_5/test_qwen3_5_adapter.py -v

Coverage:
  * Wire-format helpers — round-trip every action in the computer_use /
    mobile_use schemas.
  * Edge cases: multi-line ``text``, JSON-vs-ast parsing, malformed XML,
    tools-section filtering (``valid_actions`` + ``extra_tools``),
    coordinate_2 / modifier ``text`` on clicks.
  * Registry lookup: every six adapter keys + the three pass-through
    ``understanding/grounding.bbox/grounding.point`` patterns resolve.
  * End-to-end: ``parse_raw_assistant_response`` →
    ``convert_message_from_agent`` recovers CUA-lite tool_calls.
"""

from __future__ import annotations

import pytest

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.qwen3_5.action_space import (
    Qwen3_5DesktopGroundingPointActionSpace,
    Qwen3_5MobileGroundingPointActionSpace,
)
from lite.agents.models.qwen3_5.adapter import (
    Qwen3_5BaseAdapter,
    Qwen3_5DesktopGroundingPointAdapter,
    Qwen3_5DesktopUseAdapter,
    Qwen3_5MobileUseAdapter,
    Qwen3_5UseAdapter,
    _parse_xml_tool_calls,
    _render_xml_tool_call,
)
from lite.core import LiteCUAMetadata, LiteGenericMetadata, LiteSample
from lite.core.messages import no_tool_call_final_text
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters


def _md(extra_tools=None, valid_actions=None):
    """Test helper — build a desktop-navigation LiteCUAMetadata."""
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tools or [],
        valid_actions=valid_actions,
    )


def _md_mobile(extra_tools=None, valid_actions=None):
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tools or [],
        valid_actions=valid_actions,
    )


QWEN3_5_USE_ADAPTERS = (
    Qwen3_5UseAdapter,
    Qwen3_5DesktopUseAdapter,
    Qwen3_5MobileUseAdapter,
)

NO_TOOL_FINAL_TEXTS = (
    "Done.",
    "The answer is 42.",
)


# =============================================================================
# _parse_xml_tool_calls — structural
# =============================================================================


def test_parse_single_tool_call_basic():
    text = (
        "Action: Click the three-dots menu.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[491, 91]\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    assert _parse_xml_tool_calls(text) == [
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [491, 91]}}
    ]


def test_parse_multiple_tool_calls():
    text = (
        "<tool_call>\n<function=computer_use>\n"
        "<parameter=action>\ntype\n</parameter>\n"
        "<parameter=text>\nhello world\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=computer_use>\n"
        "<parameter=action>\nterminate\n</parameter>\n"
        "<parameter=status>\nsuccess\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = _parse_xml_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["arguments"] == {"action": "type", "text": "hello world"}
    assert calls[1]["arguments"] == {"action": "terminate", "status": "success"}


def test_parse_empty_text_returns_empty_list():
    assert _parse_xml_tool_calls("") == []


def test_parse_malformed_no_function_block_is_skipped():
    """<tool_call> without <function=...> must be skipped (logged warning)."""
    text = "<tool_call>\njust some garbage\n</tool_call>"
    assert _parse_xml_tool_calls(text) == []


def test_parse_orphan_text_outside_tool_call_is_ignored():
    """Text outside <tool_call> is prose — parser ignores it."""
    text = "I think I should click the menu first."
    assert _parse_xml_tool_calls(text) == []


def test_parse_unterminated_tool_call_is_skipped():
    """An unterminated <tool_call>...EOF is skipped (regex is non-greedy)."""
    text = "<tool_call>\n<function=computer_use>\n<parameter=action>\nterminate"
    assert _parse_xml_tool_calls(text) == []


# An UNCLOSED ``<parameter=...>`` opener, verbatim from a Qwen3.8 lite.osworld
# rollout (``osworld_chrome_030eeff7`` turn 16): the model repeated the function
# name as a parameter tag and never closed it, directly before the real
# ``action`` parameter.
_STRAY_OPENER_TOOL_CALL = (
    "<tool_call>\n"
    "<function=computer_use>\n"
    "<parameter=computer_use>\n"
    "<parameter=action>\n"
    "scroll\n"
    "</parameter>\n"
    "<parameter=pixels>\n"
    "-3\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


# A ``type`` whose text QUOTES XML is byte-identical to the stray-opener shape
# above. It must keep parsing as one ``text`` value.
_LITERAL_XML_TOOL_CALL = (
    "<tool_call>\n"
    "<function=computer_use>\n"
    "<parameter=action>\n"
    "type\n"
    "</parameter>\n"
    "<parameter=text>\n"
    "example: <parameter=foo> literal\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

_DESKTOP_PARAM_TYPES = Qwen3_5DesktopUseAdapter()._xml_param_types()


def test_parse_unclosed_parameter_opener_does_not_swallow_the_next_parameter():
    """``_PARAM_RE`` gives the leftmost opener the closer, so the stray one ate
    ``<parameter=action>`` and produced ``{"computer_use": ...}`` with no
    ``action`` at all. The model's real intent must survive instead."""
    assert _parse_xml_tool_calls(_STRAY_OPENER_TOOL_CALL, _DESKTOP_PARAM_TYPES) == [
        {"name": "computer_use", "arguments": {"action": "scroll", "pixels": -3}}
    ]


def test_quoted_xml_in_a_declared_parameter_is_not_resplit():
    """The other side of the same ambiguity. The reclaim is gated on the schema
    the model was shown: ``text`` IS declared, so its value stays whole even
    though it contains a ``<parameter=`` opener."""
    assert _parse_xml_tool_calls(_LITERAL_XML_TOOL_CALL, _DESKTOP_PARAM_TYPES) == [
        {
            "name": "computer_use",
            "arguments": {"action": "type", "text": "example: <parameter=foo> literal"},
        }
    ]


def test_reclaim_needs_the_schema_and_is_inert_without_it():
    """No schema (the workflow-agnostic base adapter, or an unknown tool) means
    no evidence for the tie-break, so the pre-existing leftmost-opener reading
    stands rather than guessing."""
    assert _parse_xml_tool_calls(_STRAY_OPENER_TOOL_CALL) == [
        {
            "name": "computer_use",
            "arguments": {
                "computer_use": "<parameter=action>\nscroll",
                "pixels": -3,
            },
        }
    ]


def test_stray_opener_turn_still_renders_back_into_history():
    """The end-to-end consequence, which is why this is not a cosmetic parse bug.

    A wrapper call with no ``action`` lowers to an action-batch whose child is
    UNNAMED, and an unnamed child is fatal in ``_action_batch_feedback`` — but
    only when the turn is replayed, so the crash landed on a LATER turn and
    killed the episode with a traceback.
    """
    # The concrete desktop adapter, because the wrapper tool this malformation
    # names (``computer_use``) only exists on a bound desktop action space.
    adapter = Qwen3_5DesktopUseAdapter()
    lite = adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response("Action: Scroll down.\n" + _STRAY_OPENER_TOOL_CALL)
    )
    child_actions = [
        child["action"]
        for call in lite["tool_calls"]
        for child in tool_call_arguments(call).get("actions", [])
    ]
    assert child_actions and all(child_actions), child_actions
    adapter.convert_message_to_agent(lite)  # must not raise


def test_parse_keys_double_wrapped_list_is_recovered():
    """Qwen3.5 sometimes double-wraps the keys list, emitting the literal
    ``["['ctrl", "a']"]`` (a stringified list nested in a list). literal_eval
    parses that to two junk tokens (``['ctrl``, ``a']``) that no key backend can
    resolve, so Ctrl+A silently no-ops. The parser must recover ['ctrl', 'a'].
    (Real payload seen across 96 turns of the 2026-04-26 qwen3_5 eval run.)"""
    text = (
        "<tool_call>\n<function=computer_use>\n"
        "<parameter=action>\nkey\n</parameter>\n"
        '<parameter=keys>\n["[\'ctrl", "a\']"]\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    [call] = _parse_xml_tool_calls(text)
    assert call["arguments"]["keys"] == ["ctrl", "a"]


def test_parse_keys_wellformed_list_is_unchanged():
    """A correctly-formed keys list must pass through the recovery untouched."""
    text = (
        "<tool_call>\n<function=computer_use>\n"
        "<parameter=action>\nkey\n</parameter>\n"
        "<parameter=keys>\n['ctrl', ',']\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = _parse_xml_tool_calls(text)
    assert call["arguments"]["keys"] == ["ctrl", ","]


@pytest.mark.parametrize(
    "tc",
    [
        {
            "name": "computer",
            "arguments": {
                "actions": [
                    {"action": "click", "coordinate": [100, 200]},
                    {"action": "type", "text": "hi"},
                ],
            },
        },
        {
            "name": "mobile",
            "arguments": {
                "actions": [
                    {"action": "tap", "coordinate": [111, 222]},
                    {"action": "type", "text": "hi"},
                ],
            },
        },
    ],
)
def test_parse_actions_parameter_is_structured_list(tc):
    assert _parse_xml_tool_calls(_render_xml_tool_call(tc)) == [tc]


@pytest.mark.parametrize(
    "raw_keys,expected",
    [
        # Bare bracket / paren / quote single-chars are LEGITIMATE keys —
        # _PUNCT (in exec_stdio/server.py) maps each to a real X keysym
        # (``[`` -> ``bracketleft``, ``'`` -> ``apostrophe``, ``(`` ->
        # ``parenleft``), so the recovery must NOT strip them away. Pre-fix
        # the strip("[](){}").strip("'\"") chain ate every one of these,
        # silently turning ``ctrl+[`` (vim "go back") into a lone ``ctrl``
        # — the exact silent no-op the rest of the cb stack carefully
        # avoids (server.py raises on unmapped keys for the same reason).
        ("['ctrl', '[']", ["ctrl", "["]),
        ("['ctrl', ']']", ["ctrl", "]"]),
        ("['ctrl', '(']", ["ctrl", "("]),
        ("['ctrl', ')']", ["ctrl", ")"]),
        ('["ctrl", "\\\'"]', ["ctrl", "'"]),
        ("['ctrl', '+']", ["ctrl", "+"]),
        ("['ctrl', '-']", ["ctrl", "-"]),
        ("['ctrl', '=']", ["ctrl", "="]),
    ],
)
def test_parse_keys_single_char_punctuation_survives_cleanup(raw_keys, expected):
    """Single-char punctuation keys must not be stripped by _clean_key_tokens.
    The cleanup targets multi-char junk like ``['ctrl`` / ``a']`` — a bare
    single char IS the key, not bracket-junk to strip."""
    text = (
        "<tool_call>\n<function=computer_use>\n"
        "<parameter=action>\nkey\n</parameter>\n"
        f"<parameter=keys>\n{raw_keys}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = _parse_xml_tool_calls(text)
    assert call["arguments"]["keys"] == expected


def test_parse_keys_non_string_list_element_is_not_stringified():
    text = (
        "<tool_call>\n<function=computer_use>\n"
        "<parameter=action>\nkey\n</parameter>\n"
        "<parameter=keys>\n[1]\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    [call] = _parse_xml_tool_calls(text)
    assert call["arguments"]["keys"] == [1]


# =============================================================================
# Value coercion: every param key expected by the Qwen3-VL desktop + mobile
# schemas must round-trip through render+parse.
# =============================================================================


def _roundtrip(args: dict) -> dict:
    tc = {"name": "computer_use", "arguments": args}
    rendered = _render_xml_tool_call(tc)
    parsed = _parse_xml_tool_calls(rendered)
    assert len(parsed) == 1
    return parsed[0]["arguments"]


# --- Desktop: every action enum ---


@pytest.mark.parametrize(
    "args",
    [
        # key — list of modifier strings via `keys` param
        {"action": "key", "keys": ["ctrl", "s"]},
        # type — free-form text
        {"action": "type", "text": "Hello, world!"},
        # mouse_move
        {"action": "mouse_move", "coordinate": [500, 300]},
        # clicks — all button variants
        {"action": "left_click", "coordinate": [100, 200]},
        {"action": "right_click", "coordinate": [100, 200]},
        {"action": "middle_click", "coordinate": [100, 200]},
        {"action": "double_click", "coordinate": [100, 200]},
        {"action": "triple_click", "coordinate": [100, 200]},
        # left_click with modifier keys (ref agent accepts `text` as modifier list)
        {"action": "left_click", "coordinate": [100, 200], "text": "ctrl+shift"},
        # left_click_drag — single end coord (qwen3_vl style)
        {"action": "left_click_drag", "coordinate": [400, 500]},
        # scrolls
        {"action": "scroll", "pixels": 3, "coordinate": [500, 500]},
        {"action": "hscroll", "pixels": -2, "coordinate": [500, 500]},
        # wait with time (float)
        {"action": "wait", "time": 1.5},
        # terminate
        {"action": "terminate", "status": "success"},
        {"action": "terminate", "status": "failure"},
        # answer with free text
        {"action": "answer", "text": "The answer is 42."},
    ],
    ids=lambda a: a["action"] + ("_" + "_".join(k for k in a if k != "action")),
)
def test_desktop_action_roundtrip(args):
    assert _roundtrip(args) == args


def test_desktop_type_preserves_newlines():
    """The `text` param may span multiple lines — must round-trip verbatim.

    The parser's ``(.*?)`` between ``<parameter=KEY>\\n`` and
    ``\\n</parameter>`` is DOTALL; leading/trailing whitespace around the
    value is stripped. Internal newlines are preserved.
    """
    args = {"action": "type", "text": "line one\nline two\nline three"}
    assert _roundtrip(args) == args


# --- Mobile: every action enum ---


def _mobile_roundtrip(args: dict) -> dict:
    tc = {"name": "mobile_use", "arguments": args}
    rendered = _render_xml_tool_call(tc)
    parsed = _parse_xml_tool_calls(rendered)
    assert len(parsed) == 1
    return parsed[0]["arguments"]


@pytest.mark.parametrize(
    "args",
    [
        {"action": "click", "coordinate": [200, 400]},
        {"action": "long_press", "coordinate": [200, 400], "time": 1.0},
        # swipe with coordinate2 (start + end)
        {"action": "swipe", "coordinate": [100, 500], "coordinate2": [100, 100]},
        {"action": "type", "text": "hello"},
        {"action": "answer", "text": "42"},
        {"action": "system_button", "button": "Back"},
        {"action": "system_button", "button": "Home"},
        {"action": "wait", "time": 2.0},
        {"action": "terminate", "status": "success"},
    ],
    ids=lambda a: a["action"],
)
def test_mobile_action_roundtrip(args):
    assert _mobile_roundtrip(args) == args


# =============================================================================
# Renderer value encoding
# =============================================================================


def test_render_keys_list_json_serialized():
    tc = {"name": "computer_use", "arguments": {"action": "key", "keys": ["ctrl", "+"]}}
    rendered = _render_xml_tool_call(tc)
    # keys renders as a JSON array on a single line so parsing is trivial
    assert '["ctrl", "+"]' in rendered
    assert _parse_xml_tool_calls(rendered) == [tc]


def test_render_coordinate_tuple_becomes_list_on_parse():
    """Renderer accepts tuple for coordinate; parser recovers it as list."""
    tc = {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": (491, 91)}}
    rendered = _render_xml_tool_call(tc)
    parsed = _parse_xml_tool_calls(rendered)
    assert parsed[0]["arguments"]["coordinate"] == [491, 91]


def test_render_bool_lowercase():
    tc = {"name": "custom_tool", "arguments": {"flag": True, "other": False}}
    rendered = _render_xml_tool_call(tc)
    # Both "true" and "false" present (lowercase like JSON/YAML)
    assert ">\ntrue\n<" in rendered
    assert ">\nfalse\n<" in rendered


# =============================================================================
# _build_tools_section filtering
# =============================================================================


def test_build_tools_section_default_includes_full_enum():
    adapter = Qwen3_5DesktopUseAdapter()
    section = adapter._build_tools_section()
    assert "<tools>" in section
    assert "</tools>" in section
    assert "<function=example_function_name>" in section
    # Default GUI actions are present. The qwen native finish spellings are
    # ORTHOGONAL: they are gated by ``extra_tool_schemas``, and this default
    # adapter has none, so they must not be advertised.
    for action in ["left_click", "type"]:
        assert action in section
    assert '"terminate"' not in section
    assert '"answer"' not in section


def test_build_tools_section_valid_actions_does_not_enable_finish_without_extra_schema():
    """``valid_actions=["click"]`` restricts GUI actions only, and never opens a
    native finish spelling — that needs a matching ``extra_tool_schemas`` entry.
    """
    import json as _json
    import re as _re

    adapter = Qwen3_5DesktopUseAdapter(metadata=_md(valid_actions=["click"]))
    section = adapter._build_tools_section()
    # Extract the raw JSON schema line from <tools>...</tools>
    m = _re.search(r"<tools>\n(.*?)\n</tools>", section, _re.DOTALL)
    assert m is not None
    # Qwen renders the nested Hermes envelope; read parameters through the helper.
    schema = _json.loads(m.group(1).splitlines()[0])  # first tool only
    properties = tool_schema_parameters(schema)["properties"]
    action_enum = properties["action"]["enum"]
    # Kept
    assert "left_click" in action_enum
    # Native finish spellings stay CLOSED: no finish extra_tool_schema is active,
    # and ``valid_actions`` must never add or remove a semantic entry.
    assert "terminate" not in action_enum
    assert "answer" not in action_enum
    assert properties["text"]["description"] == ""
    assert "action=answer" not in properties["text"]["description"]
    # Dropped (not mapped by click)
    assert "hscroll" not in action_enum
    assert "scroll" not in action_enum


def test_build_tools_section_finish_extra_schemas_do_not_rewrite_native_enum():
    """response/terminate extras gate env acceptance; qwen native schema stays stable."""
    import json as _json
    import re as _re

    finish_tools = [
        make_tool_schema(
            "response",
            description="Return the final answer.",
            parameters={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        ),
        make_tool_schema(
            "terminate",
            description="Finish the task.",
            parameters={
                "type": "object",
                "required": ["status"],
                "properties": {"status": {"type": "string"}},
            },
        ),
    ]
    adapter = Qwen3_5DesktopUseAdapter(
        metadata=_md(extra_tools=finish_tools, valid_actions=["click"])
    )
    section = adapter._build_tools_section()
    m = _re.search(r"<tools>\n(.*?)\n</tools>", section, _re.DOTALL)
    assert m is not None
    # Qwen renders the nested Hermes envelope; read parameters through the helper.
    schema = _json.loads(m.group(1).splitlines()[0])
    properties = tool_schema_parameters(schema)["properties"]
    action_enum = properties["action"]["enum"]
    assert "answer" in action_enum
    assert "terminate" in action_enum
    assert properties["text"]["description"] == "Required only by `action=answer`."
    # Native finish schemas should not also be rendered as standalone tools.
    assert '"name": "response"' not in "\n".join(m.group(1).splitlines()[1:])
    assert '"name": "terminate"' not in "\n".join(m.group(1).splitlines()[1:])


def test_mobile_tools_section_descriptions_follow_active_native_semantics():
    """Wrapper parameter descriptions must not mention filtered enum entries."""
    import json as _json
    import re as _re

    adapter = Qwen3_5MobileUseAdapter(metadata=_md_mobile(valid_actions=["tap"]))
    section = adapter._build_tools_section()
    m = _re.search(r"<tools>\n(.*?)\n</tools>", section, _re.DOTALL)
    assert m is not None
    schema = _json.loads(m.group(1).splitlines()[0])
    properties = tool_schema_parameters(schema)["properties"]

    assert properties["action"]["enum"] == ["click"]
    assert properties["text"]["description"] == ""
    assert properties["status"]["description"] == "The status of the task."
    rendered_schema = _json.dumps(schema)
    assert "action=open" not in rendered_schema
    assert "action=answer" not in rendered_schema
    assert "action=terminate" not in rendered_schema


def test_build_tools_section_extra_tools_appended():
    """extra_tools entries must appear as additional functions."""
    extra = [
        make_tool_schema(
            "goto",
            description="Navigate the browser to an URL.",
            parameters={
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string"}},
            },
        )
    ]
    adapter = Qwen3_5DesktopUseAdapter(metadata=_md(extra_tools=extra))
    section = adapter._build_tools_section()
    assert "computer_use" in section  # standard schema still present
    assert '"goto"' in section  # extra tool rendered
    assert '"Navigate the browser to an URL."' in section


# =============================================================================
# Registry lookup
# =============================================================================


@pytest.mark.parametrize(
    "key,cls",
    [
        ("qwen3_5@desktop@use", Qwen3_5DesktopUseAdapter),
        ("qwen3_5@desktop@grounding.point", Qwen3_5DesktopGroundingPointAdapter),
        # @browser collapses to the desktop adapter (tools refactor: browser == desktop;
        # nav is an env extra_tool, no dedicated browser adapter).
        ("qwen3_5@browser@use", Qwen3_5DesktopUseAdapter),
        ("qwen3_5@browser@grounding.point", Qwen3_5DesktopGroundingPointAdapter),
        ("qwen3_5@mobile@use", Qwen3_5MobileUseAdapter),
    ],
)
def test_registry_lookup(key, cls):
    adapter = AgentAdapterRegistry.get(key)
    assert isinstance(adapter, cls)


def test_passthrough_pattern_keys_resolve():
    """understanding / grounding.bbox patterns resolve to AsIsAdapter.

    ``grounding.point`` and ``grounding.action`` are both served by
    concrete per-platform classes (env eval → point, SFT replay → action).
    """
    from lite.agents.core.adapter import AsIsAdapter

    for key in [
        "qwen3_5@desktop@understanding",
        "qwen3_5@mobile@understanding",
        "qwen3_5@desktop@grounding.bbox",
    ]:
        adapter = AgentAdapterRegistry.get(key)
        assert isinstance(adapter, AsIsAdapter), f"{key} should be AsIsAdapter"


@pytest.mark.parametrize(
    "space_cls,agent_call",
    [
        (
            Qwen3_5DesktopGroundingPointActionSpace,
            {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [10, 20]}},
        ),
        (
            Qwen3_5MobileGroundingPointActionSpace,
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [10, 20]}},
        ),
    ],
    ids=["desktop-left-click", "mobile-click"],
)
def test_grounding_point_native_action_converts_to_point(space_cls, agent_call):
    assert space_cls().convert_tool_calls_from_agent([agent_call]) == [
        make_tool_call("point", {"coordinate": [10, 20]})
    ]


@pytest.mark.parametrize(
    "space_cls,agent_call",
    [
        (
            Qwen3_5DesktopGroundingPointActionSpace,
            {"name": "computer_use", "arguments": {"action": "click", "coordinate": [10, 20]}},
        ),
        (
            Qwen3_5MobileGroundingPointActionSpace,
            {"name": "mobile_use", "arguments": {"action": "left_click", "coordinate": [10, 20]}},
        ),
    ],
    ids=[
        "desktop-click-alias",
        "mobile-left-click-alias",
    ],
)
def test_grounding_point_repairs_qwen3_5_family_click_aliases(space_cls, agent_call):
    assert space_cls().convert_tool_calls_from_agent([agent_call]) == [
        make_tool_call("point", {"coordinate": [10, 20]})
    ]


@pytest.mark.parametrize(
    "space_cls,agent_call",
    [
        (
            Qwen3_5DesktopGroundingPointActionSpace,
            {
                "name": "computer_use",
                "arguments": {"action": "mouse_click", "coordinate": [10, 20]},
            },
        ),
        (
            Qwen3_5MobileGroundingPointActionSpace,
            {"name": "mobile_use", "arguments": {"action": "tap", "coordinate": [10, 20]}},
        ),
    ],
    ids=[
        "desktop-no-use-alias",
        "mobile-no-tap-alias",
    ],
)
def test_grounding_point_does_not_repair_off_schema_actions(space_cls, agent_call):
    assert space_cls().convert_tool_calls_from_agent([agent_call]) == []


# =============================================================================
# End-to-end: parse_raw_assistant_response → convert_message_from_agent
# =============================================================================


def test_parser_matrix_single_tool_call():
    adapter = Qwen3_5DesktopUseAdapter()
    raw = "Action: Click the search bar.\n" + _render_xml_tool_call(
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [12, 34]}}
    )

    msg = adapter.parse_raw_assistant_response(raw)

    assert pop_model_output_error(msg.copy()) is None
    assert msg["tool_calls"] == [
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [12, 34]}}
    ]


def test_parser_matrix_consecutive_multiple_tool_calls():
    adapter = Qwen3_5DesktopUseAdapter()
    raw = (
        "Action: Click, then type.\n"
        + _render_xml_tool_call(
            {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [12, 34]}}
        )
        + _render_xml_tool_call(
            {"name": "computer_use", "arguments": {"action": "type", "text": "hello"}}
        )
    )

    msg = adapter.parse_raw_assistant_response(raw)

    assert pop_model_output_error(msg.copy()) is None
    assert [tc["arguments"]["action"] for tc in msg["tool_calls"]] == [
        "left_click",
        "type",
    ]


def test_parser_matrix_duplicated_opener():
    adapter = Qwen3_5DesktopUseAdapter()
    raw = "Action: Click the search bar.\n<tool_call>\n" + _render_xml_tool_call(
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [12, 34]}}
    )

    msg = adapter.parse_raw_assistant_response(raw)

    assert pop_model_output_error(msg.copy()) is None
    assert msg["tool_calls"] == [
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [12, 34]}}
    ]


def test_parser_matrix_malformed_xml_sets_model_output_error():
    adapter = Qwen3_5DesktopUseAdapter()
    raw = "Action: Click.\n<tool_call>\n{not valid json}\n</tool_call>"

    msg = adapter.parse_raw_assistant_response(raw)

    assert not msg.get("tool_calls")
    assert pop_model_output_error(msg) == "malformed <tool_call> XML"


def test_parser_matrix_plain_final_text_roundtrip_stays_text():
    adapter = Qwen3_5DesktopUseAdapter()
    final_text = "Done."

    parsed = adapter.parse_raw_assistant_response(final_text)
    lite = adapter.convert_message_from_agent(parsed)
    rendered = adapter.convert_message_to_agent(lite)

    assert not parsed.get("tool_calls")
    assert lite == {
        "role": "assistant",
        "content": [{"type": "text", "text": final_text}],
    }
    assert no_tool_call_final_text(lite) == final_text
    assert rendered == {
        "role": "assistant",
        "content": [{"type": "text", "text": final_text}],
    }


def test_parse_and_remap_to_cua_lite_click():
    """Full pipeline: raw text → AgentMessage → LiteMessage with CUA-lite action."""
    adapter = Qwen3_5DesktopUseAdapter()
    response = (
        "Action: Click the button.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[100, 200]\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    agent_msg = adapter.parse_raw_assistant_response(response)
    assert agent_msg["role"] == "assistant"
    assert agent_msg["tool_calls"][0]["name"] == "computer_use"
    assert agent_msg["tool_calls"][0]["arguments"]["coordinate"] == [100, 200]
    # Prose retained, tool_call tokens stripped
    assert "Action: Click the button." in agent_msg["content"][0]["text"]
    assert "<tool_call>" not in agent_msg["content"][0]["text"]

    lite_msg = adapter.convert_message_from_agent(agent_msg)
    assert lite_msg["tool_calls"][0] == make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": [100, 200]}]},
    )


def test_parse_answer_wire_alias_persists_as_response():
    """Qwen wire ``answer`` is not a canonical Lite tool name."""
    response_schema = make_tool_schema(
        "response",
        description="Submit a final answer.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    adapter = Qwen3_5DesktopUseAdapter(metadata=_md(extra_tools=[response_schema]))
    response = (
        "Action: Answer the task.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nanswer\n</parameter>\n"
        "<parameter=text>\n42\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    lite_msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(response))
    names = [tool_call_name(tc) for tc in lite_msg["tool_calls"]]
    assert names == ["response"]
    assert "answer" not in names
    assert tool_call_arguments(lite_msg["tool_calls"][0])["text"] == "42"


def test_parse_adjacent_wrappers_batch_and_render_back_in_order():
    bash_schema = make_tool_schema(
        "bash",
        description="Run a shell command.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )
    adapter = Qwen3_5DesktopUseAdapter(metadata=_md(extra_tools=[bash_schema]))
    raw = (
        "Action: Click, run bash, then type.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[100, 200]\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\ntype\n</parameter>\n"
        "<parameter=text>\na\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\npwd\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[300, 400]\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )

    lite_msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
    assert lite_msg["tool_calls"] == [
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [100, 200]},
                    {"action": "type", "text": "a"},
                ],
            },
        ),
        make_tool_call("bash", {"command": "pwd"}),
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [300, 400]},
                ],
            },
        ),
    ]

    rendered = adapter.convert_message_to_agent(lite_msg)["content"][0]["text"]
    reparsed = _parse_xml_tool_calls(rendered)
    assert [tc["name"] for tc in reparsed] == [
        "computer_use",
        "computer_use",
        "bash",
        "computer_use",
    ]
    assert [tc["arguments"].get("action") for tc in reparsed] == [
        "left_click",
        "type",
        None,
        "left_click",
    ]


def test_parse_with_think_channel():
    """`<think>...</think>` goes into reasoning_content; prose remainder cleaned."""
    adapter = Qwen3_5DesktopUseAdapter()
    response = (
        "<think>\nI should open the menu first.\n</think>\n"
        "Action: Click the three-dots menu.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[980, 100]\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    agent_msg = adapter.parse_raw_assistant_response(response)
    assert agent_msg.get("reasoning_content", "").strip() == "I should open the menu first."
    assert agent_msg["tool_calls"][0]["arguments"]["coordinate"] == [980, 100]
    # Prose: Action line retained, <think> and <tool_call> stripped
    clean = agent_msg["content"][0]["text"]
    assert "<think>" not in clean
    assert "<tool_call>" not in clean


def test_parse_thinking_on_no_open_tag():
    """When ``enable_thinking=True``, the chat_template emits ``<think>\\n``
    BEFORE generation, so the model response begins mid-thought with only
    a ``</think>`` close tag. Parser must still funnel the pre-``</think>``
    text into ``reasoning_content`` and keep the post-``</think>`` prose
    clean.
    """
    adapter = Qwen3_5DesktopUseAdapter(enable_thinking=True)
    response = (
        "I need to dismiss the Welcome dialog first.\n"
        "The Skip button is at the bottom-left.\n"
        "</think>\n\n"
        "Action: Click the Skip button.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[120, 780]\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    agent_msg = adapter.parse_raw_assistant_response(response)
    # Entire pre-</think> text becomes reasoning_content
    assert "dismiss the Welcome dialog" in agent_msg.get("reasoning_content", "")
    assert "Skip button is at the bottom-left" in agent_msg["reasoning_content"]
    # tool_call still parsed
    assert agent_msg["tool_calls"][0]["arguments"]["coordinate"] == [120, 780]
    # Clean content: only the Action: prose remains (no thinking leak)
    clean = agent_msg["content"][0]["text"]
    assert "dismiss the Welcome dialog" not in clean
    assert "</think>" not in clean
    assert "<tool_call>" not in clean
    assert "Action: Click the Skip button." in clean


def test_enable_thinking_default_false():
    """Default is thinking OFF (no native <think> channel at generation time)."""
    assert Qwen3_5DesktopUseAdapter().enable_thinking is False


def test_enable_thinking_toggle_respects_kwarg():
    """User passes ``adapter_kwargs={'enable_thinking': True}`` via AgentRegistry."""
    adapter = AgentAdapterRegistry.get(
        "qwen3_5@desktop@use",
        enable_thinking=True,
    )
    assert isinstance(adapter, Qwen3_5DesktopUseAdapter)
    assert adapter.enable_thinking is True


def test_agent_forwards_enable_thinking_to_chat_template():
    """Qwen3_5BaseAgent.build_generation_prompt must pass ``enable_thinking`` to
    the processor. Use a dummy processor that records the kwarg."""
    from lite.agents.models.qwen3_5.agent import Qwen3_5DesktopUseAgent

    calls: dict = {}

    class _DummyProcessor:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["kwargs"] = kwargs
            return "dummy-prompt"

    async def _dummy_gen(**_):
        return {"response": ""}

    # AutoAdapterAgent rebuilds its adapter from the registry in __post_init__,
    # consuming ``kwargs`` as adapter-init overrides. Test via that path.
    agent = Qwen3_5DesktopUseAgent(
        generate_fn=_dummy_gen,
        processor=_DummyProcessor(),
        kwargs={"enable_thinking": True},
    )
    assert agent.adapter.enable_thinking is True
    out = agent.build_generation_prompt([{"role": "user", "content": []}])
    assert out == "dummy-prompt"
    assert calls["kwargs"].get("enable_thinking") is True
    assert calls["kwargs"].get("add_generation_prompt") is True
    assert calls["kwargs"].get("tokenize") is False

    # Default (no override) → enable_thinking=False
    calls.clear()
    agent2 = Qwen3_5DesktopUseAgent(
        generate_fn=_dummy_gen,
        processor=_DummyProcessor(),
    )
    assert agent2.adapter.enable_thinking is False
    agent2.build_generation_prompt([{"role": "user", "content": []}])
    assert calls["kwargs"].get("enable_thinking") is False


def test_parse_enable_inline_reasoning_line():
    """enable_inline_reasoning=True extracts `Thought:` line into inline_reasoning part."""
    adapter = Qwen3_5DesktopUseAdapter(enable_inline_reasoning=True)
    response = (
        "Thought: The settings menu icon is at top-right.\n"
        "Action: Click the settings menu.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[980, 100]\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    agent_msg = adapter.parse_raw_assistant_response(response)
    lite_msg = adapter.convert_message_from_agent(agent_msg)
    # reasoning part surfaces as inline_reasoning
    kinds = {p["type"] for p in lite_msg["content"]}
    assert "inline_reasoning" in kinds
    assert "action_description" in kinds
    # Tool call remapped into canonical action-batch call.
    assert lite_msg["tool_calls"][0] == make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": [980, 100]}]},
    )


def test_mobile_left_click_alias_surfaces_as_tap_without_backend_wording(caplog):
    """Qwen3.5 mobile left_click leak aliases to Lite tap, not a backend term."""
    adapter = Qwen3_5MobileUseAdapter()
    raw = "Action: Tap Settings.\n" + _render_xml_tool_call(
        {"name": "mobile_use", "arguments": {"action": "left_click", "coordinate": [123, 456]}}
    )

    lite_msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))

    assert lite_msg["tool_calls"] == [
        make_tool_call(
            "mobile",
            {
                "actions": [
                    {"action": "tap", "coordinate": [123, 456], "clicks": 1},
                ]
            },
        )
    ]
    assert "backend" not in caplog.text.lower()
    assert "canonical" not in caplog.text.lower()


def test_parse_multiline_thought_body_round_trips():
    """A MULTI-LINE ``Thought:`` body must be captured in full — the case the
    non-greedy ``Thought:\\s*(.*?)(?:\\n(?=Action:)|\\Z)`` regex fix was created
    for. Pre-fix the ``Action:`` capture clipped at the first ``\\n``; feed
    ``Thought: line1\\nline2\\nAction: ...`` and assert inline_reasoning keeps
    BOTH lines while the action stays intact."""
    adapter = Qwen3_5DesktopUseAdapter(enable_inline_reasoning=True)
    response = (
        "Thought: line1\n"
        "line2\n"
        "Action: Click the settings menu.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[980, 100]\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    agent_msg = adapter.parse_raw_assistant_response(response)
    lite_msg = adapter.convert_message_from_agent(agent_msg)
    reasoning = next(p["text"] for p in lite_msg["content"] if p["type"] == "inline_reasoning")
    action = next(p["text"] for p in lite_msg["content"] if p["type"] == "action_description")
    assert reasoning == "line1\nline2", (
        "multi-line Thought body must keep both lines (regex must not clip at the first newline)"
    )
    assert action == "Click the settings menu."

    # The greedy last-``Action:`` anchor (``(?:.*\n)?Action:\s*(.*)\Z``) was
    # added so an ``Action:`` substring NESTED inside the Thought body does NOT
    # short-circuit the action capture: the REAL trailing ``Action:`` line wins.
    response_nested = (
        "Thought: I considered Action: foo\n"
        "Action: real_action.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[980, 100]\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    agent_msg_nested = adapter.parse_raw_assistant_response(response_nested)
    lite_msg_nested = adapter.convert_message_from_agent(agent_msg_nested)
    action_nested = next(
        p["text"] for p in lite_msg_nested["content"] if p["type"] == "action_description"
    )
    assert action_nested == "real_action.", (
        "the LAST Action: line must win, not an Action: nested in Thought"
    )


# =============================================================================
# Lock-in tests for the navigation wire-format behavior currently inherited
# from Qwen3VLBaseAdapter. Pin the contract before the refactor that splits
# Qwen3_5BaseAdapter (XML inline tool_call render only) and
# Qwen3_5UseAdapter (XML + Action: / Thought: line rendering).
# =============================================================================


def test_to_agent_inline_reasoning_dropped_when_thought_disabled():
    """``enable_inline_reasoning=False`` (desktop default): inline_reasoning parts MUST
    be dropped — qwen3_5 desktop wire format has no ``Thought:`` line."""
    adapter = Qwen3_5DesktopUseAdapter()
    msg = {
        "role": "assistant",
        "content": [
            {"type": "inline_reasoning", "text": "I should click."},
            {"type": "action_description", "text": "Click button."},
        ],
    }
    out = adapter.convert_message_to_agent(msg)
    assert adapter.enable_inline_reasoning is False
    text = out["content"][0]["text"]
    # The Action line is rendered with the prefix; reasoning is dropped.
    assert "Action: Click button." in text
    assert "I should click." not in text


# NOTE: the opaque/verbatim wire format (formerly ``extract_action_only=False``)
# now lives on the WebGym ``qwen3_5.passthrough`` adapter — see
# tests/agents/models/qwen3_5/test_qwen3_5_webgym_wireformat_chars.py.
# The decomposed nav adapter is Action-only by construction (no such knob).


@pytest.mark.parametrize("final_text", NO_TOOL_FINAL_TEXTS, ids=["done", "prose"])
@pytest.mark.parametrize("adapter_cls", QWEN3_5_USE_ADAPTERS, ids=lambda c: c.__name__)
def test_no_tool_call_raw_text_stays_plain_text(adapter_cls, final_text):
    """No-tool-call qwen3_5 output is a final text turn, not action narration."""
    adapter = adapter_cls()
    msg = adapter.parse_raw_assistant_response(final_text)
    out = adapter.convert_message_from_agent(msg)

    assert not out.get("tool_calls")
    assert out.get("content") == [{"type": "text", "text": final_text}]
    assert no_tool_call_final_text(out) == final_text


@pytest.mark.parametrize("final_text", NO_TOOL_FINAL_TEXTS, ids=["done", "prose"])
@pytest.mark.parametrize("adapter_cls", QWEN3_5_USE_ADAPTERS, ids=lambda c: c.__name__)
def test_no_tool_call_text_round_trips_through_agent_wire(adapter_cls, final_text):
    """``from_agent(to_agent(text final))`` must preserve qwen3_5 text finals."""
    source = {
        "role": "assistant",
        "content": [{"type": "text", "text": final_text}],
    }
    adapter = adapter_cls()

    rendered = adapter.convert_message_to_agent(source)
    out = adapter.convert_message_from_agent(rendered)

    assert not out.get("tool_calls")
    assert out.get("content") == [{"type": "text", "text": final_text}]
    assert no_tool_call_final_text(out) == final_text


def test_from_agent_keeps_the_whole_prose_when_no_action_prefix():
    """Without an ``Action: `` line the WHOLE reply is the action description.

    Keeping only its first line dropped the model's reasoning on 17% of
    tool-call turns, and these teachers run with thinking OFF, so that prose IS
    the reasoning a cross-family student is trained on. No-tool-call turns are
    covered separately by the shared content-only-final tests and stay plain
    text."""
    adapter = Qwen3_5DesktopUseAdapter()
    msg = adapter.parse_raw_assistant_response(
        "Click the search bar.\nAdditional reasoning.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[491, 91]\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    out = adapter.convert_message_from_agent(msg)
    action_part = next((c for c in out["content"] if c["type"] == "action_description"), None)
    assert action_part is not None
    assert action_part["text"] == "Click the search bar.\nAdditional reasoning."


def test_to_agent_renders_xml_tool_call_inline_in_text():
    """Qwen3.5 wire format: tool_calls are rendered as XML
    ``<tool_call><function=...>...</function></tool_call>`` inline in the
    text content (NOT in the message's ``tool_calls`` field) — this is
    chat-template-level for qwen3_5 and stays in the base adapter.
    """
    from lite.agents.core.action_space.base import LiteDesktopActionSpace

    adapter = Qwen3_5DesktopUseAdapter()
    msg = {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "Click button."}],
        "tool_calls": [LiteDesktopActionSpace.click(coordinate=[100, 200])],
    }
    out = adapter.convert_message_to_agent(msg)
    # tool_calls field is removed; XML block lives inside the text content.
    assert "tool_calls" not in out
    text = out["content"][0]["text"]
    assert "Action: Click button." in text
    assert "<tool_call>" in text
    assert "<function=computer_use>" in text
    assert "left_click" in text


def test_role_tool_results_reach_chat_template_unwrapped_and_grouped():
    """Qwen3.5's chat template owns ``role:"tool"`` -> ``<tool_response>``.

    The adapter must not pre-project canonical tool results to role:user or
    split consecutive results by truncating the second one away.
    """
    bash_schema = make_tool_schema(
        "bash",
        description="Run a shell command.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )
    metadata = LiteCUAMetadata(dims=("desktop", "use"), extra_tool_schemas=[bash_schema])
    sample = LiteSample(
        metadata=metadata,
        images=[],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Do the task."}]},
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Click and inspect."}],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [100, 200]}]},
                        call_id="call_0000",
                    ),
                    make_tool_call("bash", {"command": "pwd"}, call_id="call_0001"),
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "text", "text": "screen changed"}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0001",
                "content": [{"type": "text", "text": "stdout"}],
            },
        ],
    )

    step = Qwen3_5DesktopUseAdapter(metadata=metadata).unroll(sample).steps[-1]
    tool_messages = [message for message in step if message.get("role") == "tool"]

    assert [message["tool_call_id"] for message in tool_messages] == [
        "call_0000",
        "call_0001",
    ]
    text = "\n".join(
        part["text"]
        for message in tool_messages
        for part in message.get("content") or []
        if part.get("type") == "text"
    )
    assert "screen changed" in text
    assert "stdout" in text
    assert "<tool_response>" not in text


def test_to_agent_unwraps_canonical_mobile_batch_before_xml_render():
    adapter = Qwen3_5MobileUseAdapter()
    msg = {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "Do it."}],
        "tool_calls": [
            make_tool_call(
                "mobile",
                {
                    "actions": [
                        {"action": "tap", "coordinate": [111, 222]},
                        {"action": "type", "text": "hi"},
                    ],
                },
            )
        ],
    }

    out = adapter.convert_message_to_agent(msg)
    text = out["content"][0]["text"]
    assert text.count("<function=mobile_use>") == 2
    assert "<function=mobile>" not in text
    assert "<parameter=actions>" not in text
    parsed = _parse_xml_tool_calls(text)
    assert [tc["name"] for tc in parsed] == ["mobile_use", "mobile_use"]
    assert [tc["arguments"]["action"] for tc in parsed] == ["click", "type"]


def test_mobile_parse_adjacent_wrappers_batch_and_render_back_in_order():
    open_app_schema = make_tool_schema(
        "open_app",
        description="Launch an app.",
        parameters={
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
        },
    )
    response_schema = make_tool_schema(
        "response",
        description="Submit the final answer.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    adapter = Qwen3_5MobileUseAdapter(
        metadata=_md_mobile(extra_tools=[open_app_schema, response_schema])
    )
    raw = (
        "Action: Tap, type, open settings, answer, then wait.\n"
        "<tool_call>\n"
        "<function=mobile_use>\n"
        "<parameter=action>\nclick\n</parameter>\n"
        "<parameter=coordinate>\n[100, 200]\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=mobile_use>\n"
        "<parameter=action>\ntype\n</parameter>\n"
        "<parameter=text>\na\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=mobile_use>\n"
        "<parameter=action>\nopen\n</parameter>\n"
        "<parameter=text>\nSettings\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=mobile_use>\n"
        "<parameter=action>\nanswer\n</parameter>\n"
        "<parameter=text>\ndone\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=mobile_use>\n"
        "<parameter=action>\nwait\n</parameter>\n"
        "<parameter=time>\n1\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )

    lite_msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
    assert lite_msg["tool_calls"] == [
        make_tool_call(
            "mobile",
            {
                "actions": [
                    {"action": "tap", "coordinate": [100, 200], "clicks": 1},
                    {"action": "type", "text": "a"},
                ],
            },
        ),
        make_tool_call("open_app", {"app_name": "Settings"}),
        make_tool_call("response", {"text": "done"}),
        make_tool_call(
            "mobile",
            {
                "actions": [
                    {"action": "wait", "duration": 1.0},
                ],
            },
        ),
    ]

    rendered = adapter.convert_message_to_agent(lite_msg)["content"][0]["text"]
    reparsed = _parse_xml_tool_calls(rendered)
    assert reparsed == [
        {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [100, 200]}},
        {"name": "mobile_use", "arguments": {"action": "type", "text": "a"}},
        {"name": "mobile_use", "arguments": {"action": "open", "text": "Settings"}},
        {"name": "mobile_use", "arguments": {"action": "answer", "text": "done"}},
        {"name": "mobile_use", "arguments": {"action": "wait", "time": 1}},
    ]


def _open_app_schema():
    return make_tool_schema(
        "open_app",
        description="Launch an app.",
        parameters={
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
        },
    )


def _open_app_schema_with_apps(apps: list[str]):
    return make_tool_schema(
        "open_app",
        description="Launch an app.",
        parameters={
            "type": "object",
            "properties": {"app_name": {"type": "string", "enum": apps}},
            "required": ["app_name"],
        },
    )


def _wrapped_open_app_raw(action: str, param: str) -> str:
    return (
        "Action: Open Settings.\n"
        "<tool_call>\n"
        "<function=mobile_use>\n"
        f"<parameter=action>\n{action}\n</parameter>\n"
        f"<parameter={param}>\nSettings\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )


def test_mobile_native_open_is_canonicalized_when_offered(caplog):
    """Qwen mobile app launch is native ``action=open`` with app name in text."""
    adapter = Qwen3_5MobileUseAdapter(metadata=_md_mobile(extra_tools=[_open_app_schema()]))

    lite_msg = adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response(_wrapped_open_app_raw("open", "text"))
    )

    assert pop_model_output_error(lite_msg.copy()) is None
    assert lite_msg.get("tool_calls") == [make_tool_call("open_app", {"app_name": "Settings"})]
    assert "backend" not in caplog.text.lower()


def test_mobile_open_app_wrapper_spelling_uses_active_open_app_surface():
    """Observed rollout spelling: ``action=open_app`` maps only when active."""
    adapter = Qwen3_5MobileUseAdapter(metadata=_md_mobile(extra_tools=[_open_app_schema()]))

    lite_msg = adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response(_wrapped_open_app_raw("open_app", "app_name"))
    )

    assert pop_model_output_error(lite_msg.copy()) is None
    assert lite_msg.get("tool_calls") == [make_tool_call("open_app", {"app_name": "Settings"})]


def test_mobile_open_app_lowercase_name_normalizes_to_active_app_enum():
    adapter = Qwen3_5MobileUseAdapter(
        metadata=_md_mobile(
            extra_tools=[
                _open_app_schema_with_apps(
                    [
                        "Markor",
                        "Simple Calendar Pro",
                    ]
                )
            ]
        )
    )

    raw = (
        "Action: Open the calendar.\n"
        "<tool_call>\n"
        "<function=mobile_use>\n"
        "<parameter=action>\nopen\n</parameter>\n"
        "<parameter=text>\nsimple calendar pro\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )

    lite_msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))

    assert pop_model_output_error(lite_msg.copy()) is None
    assert lite_msg.get("tool_calls") == [
        make_tool_call("open_app", {"app_name": "Simple Calendar Pro"})
    ]


def test_mobile_native_open_reaches_env_without_open_app_schema(caplog):
    """Native open is canonicalized; env owns open_app availability errors."""
    adapter = Qwen3_5MobileUseAdapter(metadata=_md_mobile(extra_tools=[]))

    lite_msg = adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response(_wrapped_open_app_raw("open", "text"))
    )

    assert pop_model_output_error(lite_msg.copy()) is None
    assert lite_msg.get("tool_calls") == [make_tool_call("open_app", {"app_name": "Settings"})]
    assert "backend" not in caplog.text.lower()


def test_qwen3_5_base_wildcard_resolves_to_base_adapter():
    """Canonical ``qwen3_5.base`` keys resolve directly to
    :class:`Qwen3_5BaseAdapter` -- workflow-agnostic. Browser rows use the
    platform/task suffix to opt out of the navigation wire format."""
    from lite.agents.models.qwen3_5.adapter import (
        Qwen3_5BaseAdapter,
        Qwen3_5UseAdapter,
    )

    for key in [
        "qwen3_5.base",
        "qwen3_5.base@desktop@use",
        "qwen3_5.base@browser@use",
        "qwen3_5.base@mobile@use",
    ]:
        adapter = AgentAdapterRegistry.get(key)
        assert type(adapter) is Qwen3_5BaseAdapter, key
        assert not isinstance(adapter, Qwen3_5UseAdapter), key
        assert not hasattr(adapter, "enable_inline_reasoning"), key


def test_qwen3_5_base_renders_generic_metadata_response_tool():
    adapter = Qwen3_5BaseAdapter(
        metadata=LiteGenericMetadata(
            dims=(),
            extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")],
        )
    )

    assert [tool_schema_name(s) for s in adapter._tool_schemas_for_tools_section()] == ["response"]
    assert "response" in adapter._build_tools_section()


def test_qwen3_5_base_can_render_text_answer_prompt_without_tools():
    sample = LiteSample(
        metadata=LiteGenericMetadata(dims=()),
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 2+2?"}],
            }
        ],
    )
    adapter = AgentAdapterRegistry.get(
        "qwen3_5.base",
        metadata=LiteGenericMetadata(dims=()),
        render_tools_section=False,
        system_prompt="Answer in text.",
    )

    step = adapter.render_step(sample, 1, [])
    system_text = step[0]["content"][0]["text"]

    assert step[0]["role"] == "system"
    assert "Answer in text." in system_text
    assert "# Tools" not in system_text
    assert "<tool_call>" not in system_text


def test_qwen3_5_cua_use_adapter_still_rejects_generic_metadata():
    adapter = Qwen3_5DesktopUseAdapter(metadata=LiteGenericMetadata(dims=()))

    with pytest.raises(TypeError, match="requires LiteCUAMetadata"):
        adapter._assemble_tool_schemas()


def test_base_adapter_drops_navigation_specific_content_types():
    """``Qwen3_5BaseAdapter._convert_message_to_agent`` MUST consume only
    ``type: "text"`` content parts. ``action_description`` /
    ``inline_reasoning`` are workflow-specific to
    :class:`Qwen3_5UseAdapter`; the base adapter (used by
    BrowserGym via ``qwen3_5.base@...``) drops them. Pins the
    strict-base contract from the cross-branch adapter-dump diff."""
    from lite.agents.models.qwen3_5.adapter import Qwen3_5BaseAdapter

    adapter = AgentAdapterRegistry.get("qwen3_5.base@desktop@use")
    assert type(adapter) is Qwen3_5BaseAdapter
    msg = {
        "role": "assistant",
        "content": [
            {"type": "inline_reasoning", "text": "I should click."},
            {"type": "action_description", "text": "Click button."},
            {"type": "text", "text": "explicit text part"},
        ],
    }
    out = adapter.convert_message_to_agent(msg)
    # Only the ``type: text`` part survives (no XML tool_call blocks
    # since we passed no ``tool_calls``).
    assert out["content"] == [{"type": "text", "text": "explicit text part"}]


def test_qwen3_5_base_agent_registry_resolves():
    """End-to-end: canonical ``qwen3_5.base`` keys must
    resolve to :class:`Qwen3_5BaseAgent` (NOT the navigation agent), so
    yaml-driven rollouts with ``agent_id: "qwen3_5.base"`` plumbed
    through ``make`` actually find a registered agent class. Pins
    the audit-surfaced gap where the adapter side was registered but the
    agent side wasn't."""
    from lite.agents.models import AgentRegistry
    from lite.agents.models.qwen3_5.agent import (
        Qwen3_5BaseAgent,
        Qwen3_5DesktopUseAgent,
    )

    async def _dummy_gen(**_):
        return {"response": ""}

    class _DummyProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return ""

    for key in [
        "qwen3_5.base",
        "qwen3_5.base@desktop@use",
        "qwen3_5.base@browser@use",
    ]:
        agent = AgentRegistry.get(
            key,
            generate_fn=_dummy_gen,
            processor=_DummyProcessor(),
        )
        # Must be the base agent, not the desktop-navigation flavor.
        assert type(agent) is Qwen3_5BaseAgent, key
        assert not isinstance(agent, Qwen3_5DesktopUseAgent), key


def test_tools_section_byte_policy_is_one_ascii_escaped_json_object_per_line():
    """Qwen3.5 owns its ``<tools>`` byte policy inline in its own adapter.

    There is no shared root Qwen prompt helper: each Qwen family serialises at
    its own ``_build_tools_section``. The policy is one JSON object per line
    with ``json.dumps`` defaults, so non-ASCII escapes as ``\\uXXXX``. Fara
    reuses the same nested schema shape but deliberately dumps its own bytes
    with ``ensure_ascii=False``.
    """
    import json
    import re

    adapter = Qwen3_5DesktopUseAdapter(
        metadata=_md(
            extra_tools=[
                make_tool_schema(
                    "goto",
                    description="Résumé — ✓",
                    parameters={
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                )
            ],
        )
    )

    section = adapter._build_tools_section()
    match = re.search(r"<tools>\n(.*?)\n</tools>", section, re.DOTALL)
    assert match is not None
    lines = match.group(1).splitlines()
    # One JSON object per line — one line per advertised tool.
    assert [json.loads(line)["function"]["name"] for line in lines] == [
        "computer_use",
        "goto",
    ]
    # Default ASCII escaping: the rendered prompt carries no raw non-ASCII.
    assert "\\u2713" in lines[1]
    assert "✓" not in section


def test_a_literal_think_tag_in_typed_text_does_not_split_the_tool_call():
    """A ``<think>`` inside a tool-call payload is CONTENT, not a control token.

    The unclosed-``<think>`` branch partitions the reply so the body cannot fall
    through into ``action_description``. Partitioning the RAW reply cut a tool
    call in half when the model typed the literal characters -- publishing the
    block's first half as narration and its tail as reasoning, which is exactly
    the leak the branch exists to prevent. The decision runs on the
    tool-call-scrubbed text instead.
    """
    from lite.agents.models.qwen3_5.adapter import Qwen3_5DesktopUseAdapter

    adapter = Qwen3_5DesktopUseAdapter()
    raw = (
        "Action: type the tag\n"
        "<tool_call>\n<function=computer_use>\n"
        "<parameter=action>\ntype\n</parameter>\n"
        "<parameter=text>\n<think>\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    agent_message = adapter.parse_raw_assistant_response(raw)
    message = adapter.convert_message_from_agent(agent_message)

    described = [
        part["text"] for part in message.get("content") or []
        if isinstance(part, dict) and part.get("type") == "action_description"
    ]
    assert described == ["type the tag"]
    assert "reasoning_content" not in agent_message
    actions = [
        action
        for call in message["tool_calls"]
        for action in call["function"]["arguments"]["actions"]
    ]
    assert actions == [{"action": "type", "text": "<think>"}]
