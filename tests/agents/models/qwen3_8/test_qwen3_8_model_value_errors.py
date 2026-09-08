"""Malformed model VALUES become feedback, and content survives the parse.

Run:
    uv run pytest tests/agents/models/qwen3_8/test_qwen3_8_model_value_errors.py -v

Three separate ways a reply used to lose something on the way to canonical:

  * a bad ``keys`` or ``time`` raised a BARE ``ValueError``. Only
    ``ModelToolCallParseError`` is caught at the agent's parse boundary
    (``lite/agents/core/agent/base.py``), so the whole trajectory was destroyed
    instead of the model being told.
  * a ``+`` chord inside a LIST element was one unknown token, where upstream's
    ``parse_keys`` splits it.
  * a ``type`` whose text is legitimately quote-wrapped had both glyphs eaten by
    the generic string coercion, so the keyboard got broken content.
"""
from __future__ import annotations

import pytest

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.qwen3_5.adapter import _coerce_param_value
from lite.core.metadata import LiteCUAMetadata


@pytest.fixture(scope="module")
def adapter():
    from lite.agents.bootstrap import register_all

    register_all()
    return AgentAdapterRegistry.get("qwen3_8@desktop@use", metadata=LiteCUAMetadata())


def _xml(action: str, key: str, value: str) -> str:
    return (
        "<tool_call>\n<function=computer_use>\n"
        f"<parameter=action>\n{action}\n</parameter>\n"
        f"<parameter={key}>\n{value}\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def _actions(adapter, xml: str) -> list[dict]:
    lite = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(xml))
    return [
        action
        for call in (lite.get("tool_calls") or [])
        for action in (call["function"]["arguments"] or {}).get("actions", [])
    ]


@pytest.mark.parametrize(
    "wire,keys",
    [
        ('["ctrl+a"]', ["ctrl", "a"]),
        ('["ctrl", "a"]', ["ctrl", "a"]),
        ('["ctrl", "+"]', ["ctrl", "+"]),
        ('["ctrl+shift+p"]', ["ctrl", "shift", "p"]),
    ],
    ids=["chord-in-element", "already-split", "literal-plus", "three-way-chord"],
)
def test_a_plus_chord_inside_a_list_element_is_split(adapter, wire, keys):
    """Upstream's ``parse_keys`` splits it; ours used to see one unknown token
    and raise past the parse boundary, losing the trajectory. The literal ``+``
    glyph must survive that split."""
    assert _actions(adapter, _xml("key", "keys", wire)) == [{"action": "key", "keys": keys}]


@pytest.mark.parametrize(
    "action,key,value",
    [
        ("key", "keys", '["down down"]'),
        ("key", "keys", '["nosuchkey"]'),
        ("wait", "time", '"1.0"'),
        ("wait", "time", '"a few"'),
    ],
    ids=["unknown-token", "unknown-name", "quoted-number", "prose-duration"],
)
def test_a_bad_value_is_model_visible_feedback_not_a_lost_trajectory(adapter, action, key, value):
    with pytest.raises(ModelToolCallParseError):
        _actions(adapter, _xml(action, key, value))


@pytest.mark.parametrize(
    "raw,typed",
    [
        ('"when": "terminalFocus"', '"when": "terminalFocus"'),
        ("'Invoice # GES-1.pdf'", "'Invoice # GES-1.pdf'"),
        ('"0.0")&"%"', '"0.0")&"%"'),
        ("plain text", "plain text"),
    ],
    ids=["json-fragment", "quoted-filename", "calc-formula", "unquoted"],
)
def test_type_text_keeps_its_quotes(raw, typed):
    """A ``type``'s text is CONTENT, so a wrapping quote pair belongs to the
    model, not to the wire. Unwrapping it typed broken formulas into Calc.

    The exemption is keyed on the ACTION -- ``answer`` and friends also spell
    their payload ``text``, and there the quotes are not content.
    """
    assert _coerce_param_value("text", raw, "string", action="type") == typed


def test_other_params_and_other_actions_still_lose_wrapping_quotes():
    """Everywhere else a wrapping quote really is something the model added on
    top of the value -- including a ``text`` that is not being typed."""
    assert _coerce_param_value("status", "'success'", "string", action="terminate") == "success"
    assert _coerce_param_value("text", '"42"', "string", action="answer") == "42"
    assert _coerce_param_value("text", '"42"', "string") == "42"


def test_action_description_keeps_the_whole_prose(adapter):
    """Keeping only the first line dropped the model's reasoning. This teacher
    runs with thinking OFF, so the prose IS the reasoning, and it is what a
    cross-family student is trained on."""
    prose = "Now I can see the structure:\n- Column A: Symbol\n\nI need to count the empties."
    lite = adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response(
            prose + "\n" + _xml("left_click", "coordinate", "[10, 20]")
        )
    )
    described = [
        part["text"]
        for part in lite["content"]
        if part.get("type") == "action_description"
    ]
    assert described == [prose]


@pytest.mark.parametrize(
    "raw,described",
    [
        ("<think>\nreasoning that never closes\nNow I click.", None),
        ("<think>\nreasoning\n</think>\nNow I click.", "Now I click."),
        ("prompt-supplied opener\n</think>\nNow I click.", "Now I click."),
        ("Now I click.", "Now I click."),
    ],
    ids=["unclosed", "closed", "closer-only", "no-think"],
)
def test_an_unclosed_think_block_is_never_published(adapter, raw, described):
    """An opener the model never closed used to fall through into
    ``action_description`` -- control token, reasoning body and all -- and be
    replayed back into the model's own context. Everything after the opener is
    reasoning exactly as it would be with a closer."""
    body = raw + "\n" + _xml("left_click", "coordinate", "[1, 2]")
    lite = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(body))
    got = next(
        (p["text"] for p in lite["content"] if p.get("type") == "action_description"), None
    )
    assert got == described
    assert _actions(adapter, body) == [{"action": "click", "coordinate": [1, 2]}], (
        "the action must survive whatever happens to the prose"
    )


@pytest.mark.parametrize(
    "action,wire,typed",
    [
        ("type", '"a": "b"', '"a": "b"'),
        ("answer", '"42"', "42"),
        ("call_user", "'help me'", "help me"),
    ],
    ids=["type-keeps", "answer-strips", "call_user-strips"],
)
def test_the_quote_exemption_is_scoped_to_the_typing_action(adapter, action, wire, typed):
    """``type``'s text is CONTENT, so a wrapping quote pair belongs to the model.
    ``answer`` / ``call_user`` / ``response`` spell their payload ``text`` too,
    but there the quotes are the model quoting its own answer -- keying the
    exemption on the parameter NAME leaked them into the final-answer channel."""
    lite = adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response(_xml(action, "text", wire))
    )
    values = [
        a.get("text")
        for call in lite["tool_calls"]
        for a in ((call["function"]["arguments"] or {}).get("actions") or [call["function"]["arguments"]])
    ]
    assert values == [typed]


def test_a_type_missing_its_text_is_refused_not_invented(adapter):
    """A dropped ``<parameter=text>`` opener used to yield ``type(text="")`` --
    recorded as a successfully executed action that moved no state, with no
    feedback to the model. An EXPLICIT empty string is the model's own choice
    and still passes; the annotate pass strips that as a no-op."""
    malformed = (
        "<tool_call>\n<function=computer_use>\n"
        "<parameter=action>\ntype\n</parameter>\n</function>\n</tool_call>"
    )
    with pytest.raises(ModelToolCallParseError, match="requires the 'text' argument"):
        _actions(adapter, malformed)
    assert _actions(adapter, _xml("type", "text", "")) == [{"action": "type", "text": ""}]
