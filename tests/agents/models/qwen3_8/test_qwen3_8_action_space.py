"""Qwen3.8 action-space tests — the expanded ``computer_use`` enum.

Run:
    uv run pytest tests/agents/models/qwen3_8/test_qwen3_8_action_space.py -v

Qwen3.8 reuses Qwen3.5's wire format wholesale, so these tests cover only the
family delta: the schema Qwen3.8 is served with
(``${CUA_LITE_REFERENCES_ROOT}/OSWorld/mm_agents/qwen/prompts.build_internal_tools_def``)
and the projections it needs.

Coverage:
  * enum identity — exactly the upstream expanded set, ``answer`` gone.
  * each added action, both directions, nested under the provider-native
    wrapper AND flat (the model dropping the wrapper).
  * ``call_user`` <-> ``response``; env extras still outrank a name collision.
  * ``type`` carries its text verbatim in both directions.
  * gate wiring — ``valid_actions`` reaches the added actions; ``call_user``
    stays closed until ``response`` is an active extra.
  * mobile deliberately does NOT inherit the Qwen3.5 ``left_click`` alias.
"""

from __future__ import annotations

import pytest

from lite.agents.models.qwen3_5.action_space import (
    Qwen3_5DesktopActionSpace,
    Qwen3_5MobileActionSpace,
)
from lite.agents.models.qwen3_8.action_space import (
    Qwen3_8DesktopActionSpace,
    Qwen3_8DesktopGroundingPointActionSpace,
    Qwen3_8MobileActionSpace,
    Qwen3_8MobileGroundingPointActionSpace,
)
from lite.core.tools.action_space.base import LiteDesktopActionSet
from lite.core.tools.calls import make_tool_call
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import make_tool_schema, tool_schema_parameters

# The enum `mm_agents/qwen/prompts.py::build_internal_tools_def` declares, in
# upstream order. Frozen here so a drift in either direction is a test failure,
# not a silent prompt change.
EXPANDED_ENUM = [
    "key",
    "key_down",
    "key_up",
    "left_mouse_down",
    "left_mouse_up",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "scroll",
    "hscroll",
    "screenshot",
    "wait",
    "terminate",
    "call_user",
]

RESPONSE_SCHEMA = make_tool_schema(
    "response",
    description="Submit a final answer.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

ACTIVE_FINISH = {
    "active_extra_tool_names": {"response", "terminate"},
    "active_extra_tool_schemas": [RESPONSE_SCHEMA],
}


def _action_enum(space) -> list[str]:
    schema = space.get_tool_schema("computer_use")
    return tool_schema_parameters(schema)["properties"]["action"]["enum"]


def _from_agent(space, args: dict, *, name: str = "computer_use", **kwargs):
    return space.convert_tool_calls_from_agent([{"name": name, "arguments": args}], **kwargs)


def _batch(*actions: dict) -> dict:
    return make_tool_call("computer", {"actions": list(actions)})


# =============================================================================
# Schema identity
# =============================================================================


def test_desktop_enum_is_the_upstream_expanded_set():
    assert _action_enum(Qwen3_8DesktopActionSpace()) == EXPANDED_ENUM


def test_answer_is_gone_and_call_user_took_its_place():
    """The expanded harness drops ``answer``; ``call_user`` is the only
    text-bearing terminal action left."""
    enum = _action_enum(Qwen3_8DesktopActionSpace())
    assert "answer" not in enum
    assert "call_user" in enum
    # And the Qwen3.5 schema is the mirror image, so the two are really distinct.
    base_enum = _action_enum(Qwen3_5DesktopActionSpace())
    assert "answer" in base_enum
    assert "call_user" not in base_enum


def test_enum_delta_against_qwen3_5_is_exactly_the_documented_one():
    expanded = set(_action_enum(Qwen3_8DesktopActionSpace()))
    base = set(_action_enum(Qwen3_5DesktopActionSpace()))
    assert expanded - base == {
        "key_down",
        "key_up",
        "left_mouse_down",
        "left_mouse_up",
        "screenshot",
        "call_user",
    }
    assert base - expanded == {"answer"}


def test_param_descriptions_name_the_expanded_actions():
    """The ``keys`` / ``text`` prose must name the new consumers, or the model
    is told a parameter is unavailable to an action it can emit."""
    props = tool_schema_parameters(
        Qwen3_8DesktopActionSpace().get_tool_schema("computer_use")
    )["properties"]
    assert "action=key_down" in props["keys"]["description"]
    assert "action=key_up" in props["keys"]["description"]
    assert "action=call_user" in props["text"]["description"]
    assert "action=answer" not in props["text"]["description"]


# =============================================================================
# Added actions — from agent (nested wrapper form)
# =============================================================================


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            {"action": "key_down", "keys": ["ctrl"]},
            [_batch({"action": "key_down", "keys": ["ctrl"]})],
        ),
        (
            {"action": "key_up", "keys": ["ctrl"]},
            [_batch({"action": "key_up", "keys": ["ctrl"]})],
        ),
        (
            {"action": "left_mouse_down"},
            [_batch({"action": "mouse_down"})],
        ),
        (
            {"action": "left_mouse_up"},
            [_batch({"action": "mouse_up"})],
        ),
        (
            {"action": "screenshot"},
            [_batch({"action": "screenshot"})],
        ),
    ],
    ids=["key_down", "key_up", "left_mouse_down", "left_mouse_up", "screenshot"],
)
def test_added_actions_lower_to_canonical(args, expected):
    assert _from_agent(Qwen3_8DesktopActionSpace(), args) == expected


def test_mouse_down_with_coordinate_lowers_to_move_then_press():
    """``coordinate`` is optional on the press/release pair; when present the
    press happens after a move, which is two canonical actions."""
    assert _from_agent(
        Qwen3_8DesktopActionSpace(),
        {"action": "left_mouse_down", "coordinate": [100, 200]},
    ) == [
        _batch(
            {"action": "mouse_move", "coordinate": [100, 200]},
            {"action": "mouse_down"},
        )
    ]


def test_mouse_up_with_coordinate_lowers_to_move_then_release():
    assert _from_agent(
        Qwen3_8DesktopActionSpace(),
        {"action": "left_mouse_up", "coordinate": [7, 8]},
    ) == [
        _batch(
            {"action": "mouse_move", "coordinate": [7, 8]},
            {"action": "mouse_up"},
        )
    ]


@pytest.mark.parametrize(
    "action,raw_keys,expected",
    [
        ("key_down", "ctrl++", ["ctrl", "+"]),
        ("key_down", "ctrl+-", ["ctrl", "-"]),
        ("key_down", "ctrl+=", ["ctrl", "="]),
        ("key_up", "ctrl++", ["ctrl", "+"]),
    ],
)
def test_key_actions_accept_bare_string_chords(action, raw_keys, expected):
    """The model may emit ``keys`` as a string instead of a list."""
    assert _from_agent(Qwen3_8DesktopActionSpace(), {"action": action, "keys": raw_keys}) == [
        _batch({"action": action, "keys": expected})
    ]


@pytest.mark.parametrize("action", ["key", "key_down", "key_up"])
@pytest.mark.parametrize("raw_keys", ["ctrl left", "ctrl -"])
def test_key_actions_reject_phrase_like_strings(action, raw_keys):
    with pytest.raises(ValueError, match="unknown key token"):
        _from_agent(Qwen3_8DesktopActionSpace(), {"action": action, "keys": raw_keys})


# =============================================================================
# call_user <-> response
# =============================================================================


def test_call_user_lowers_to_response():
    assert _from_agent(
        Qwen3_8DesktopActionSpace(), {"action": "call_user", "text": "need the password"},
        **ACTIVE_FINISH,
    ) == [LiteFinishToolSet.response(text="need the password")]


def test_call_user_does_not_sniff_prose_for_infeasibility():
    """Upstream flips ``call_user`` to FAIL when the prose looks infeasible.
    cua-lite must not: refusal is the env-gated ``report_infeasible`` tool, and
    a text heuristic would silently relabel ordinary answers."""
    text = "This task is not possible without an extension."
    assert _from_agent(
        Qwen3_8DesktopActionSpace(), {"action": "call_user", "text": text}, **ACTIVE_FINISH
    ) == [LiteFinishToolSet.response(text=text)]


def test_terminate_status_still_rides_through():
    assert _from_agent(
        Qwen3_8DesktopActionSpace(), {"action": "terminate", "status": "failure"},
        **ACTIVE_FINISH,
    ) == [LiteFinishToolSet.terminate(status="failure")]


# =============================================================================
# Flat emission — the model drops the computer_use wrapper
# =============================================================================


@pytest.mark.parametrize(
    "name,args,expected",
    [
        ("key_down", {"keys": ["alt"]}, [_batch({"action": "key_down", "keys": ["alt"]})]),
        ("key_up", {"keys": ["alt"]}, [_batch({"action": "key_up", "keys": ["alt"]})]),
        ("left_mouse_down", {}, [_batch({"action": "mouse_down"})]),
        ("left_mouse_up", {}, [_batch({"action": "mouse_up"})]),
        ("screenshot", {}, [_batch({"action": "screenshot"})]),
    ],
    ids=["key_down", "key_up", "left_mouse_down", "left_mouse_up", "screenshot"],
)
def test_added_actions_parse_from_the_flat_shape(name, args, expected):
    """The parent promotes tool-name -> action off THIS class's enum, so without
    the same promotion here an expanded value would be promoted into a branch
    the parent does not have and fall through to ``unknown``."""
    assert _from_agent(Qwen3_8DesktopActionSpace(), args, name=name) == expected


def test_flat_call_user_lowers_to_response():
    assert _from_agent(
        Qwen3_8DesktopActionSpace(), {"text": "hi"}, name="call_user", **ACTIVE_FINISH
    ) == [LiteFinishToolSet.response(text="hi")]


def test_active_env_extra_outranks_a_colliding_flat_native_value():
    """An env that exposes its own ``screenshot`` tool owns the name — the
    native action value must not claim it."""
    env_screenshot = make_tool_schema(
        "screenshot",
        description="Env-owned screenshot tool.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    assert _from_agent(
        Qwen3_8DesktopActionSpace(),
        {},
        name="screenshot",
        active_extra_tool_names={"screenshot"},
        active_extra_tool_schemas=[env_screenshot],
    ) == [make_tool_call("screenshot", {})]


# =============================================================================
# type: text passes through verbatim
# =============================================================================


@pytest.mark.parametrize(
    "text",
    ["hi", "hi\n", "a\nb", "a\r\nb", ""],
    ids=["plain", "trailing-newline", "internal-newline", "crlf", "empty"],
)
def test_type_text_is_not_split_on_newlines(text):
    """The sandbox already executes an embedded newline as a real Enter press
    (``exec_stdio`` splits on ``\n`` and sends Return), so one Qwen ``type``
    lowers to exactly one canonical ``type`` carrying the same text."""
    assert _from_agent(Qwen3_8DesktopActionSpace(), {"action": "type", "text": text}) == [
        _batch({"action": "type", "text": text})
    ]


def test_type_renders_back_verbatim():
    """Qwen3.8 inherits Qwen3.5's ``type`` projection: canonical and wire use the
    same spelling for Enter, so nothing is rewritten in either direction."""
    rendered = Qwen3_8DesktopActionSpace().convert_tool_calls_to_agent(
        [LiteDesktopActionSet.type(text="hi\n")]
    )
    assert rendered == [{"name": "computer_use", "arguments": {"action": "type", "text": "hi\n"}}]


@pytest.mark.parametrize("text", ["hi", "hi\n", "a\nb", "a\nb\n"])
def test_type_round_trips_through_both_directions(text):
    space = Qwen3_8DesktopActionSpace()
    lowered = _from_agent(space, {"action": "type", "text": text})
    back = space.convert_tool_calls_to_agent(lowered)
    assert "".join(tc["arguments"]["text"] for tc in back) == text


# =============================================================================
# to-agent projections for the added actions
# =============================================================================


@pytest.mark.parametrize(
    "lite_call,expected",
    [
        (LiteDesktopActionSet.key_down(keys=["ctrl"]), {"action": "key_down", "keys": ["ctrl"]}),
        (
            LiteDesktopActionSet.key_down(keys=["ctrl", "+"]),
            {"action": "key_down", "keys": ["ctrl", "+"]},
        ),
        (LiteDesktopActionSet.key_up(keys=["ctrl"]), {"action": "key_up", "keys": ["ctrl"]}),
        (LiteDesktopActionSet.mouse_down(button="left"), {"action": "left_mouse_down"}),
        (LiteDesktopActionSet.mouse_up(button="left"), {"action": "left_mouse_up"}),
        (LiteDesktopActionSet.screenshot(), {"action": "screenshot"}),
        (LiteFinishToolSet.response(text="done"), {"action": "call_user", "text": "done"}),
        (
            LiteFinishToolSet.terminate(status="success"),
            {"action": "terminate", "status": "success"},
        ),
    ],
    ids=[
        "key_down",
        "key_down_ctrl_plus",
        "key_up",
        "mouse_down",
        "mouse_up",
        "screenshot",
        "response",
        "terminate",
    ],
)
def test_canonical_calls_render_to_the_expanded_enum(lite_call, expected):
    assert Qwen3_8DesktopActionSpace().convert_tool_calls_to_agent([lite_call]) == [
        {"name": "computer_use", "arguments": expected}
    ]


@pytest.mark.parametrize(
    "lite_call",
    [
        LiteDesktopActionSet.key_down(keys=["ctrl"]),
        LiteDesktopActionSet.key_down(keys=["ctrl", "+"]),
        LiteDesktopActionSet.key_up(keys=["ctrl"]),
        LiteDesktopActionSet.mouse_down(button="left"),
        LiteDesktopActionSet.mouse_up(button="left"),
        LiteDesktopActionSet.screenshot(),
    ],
    ids=["key_down", "key_down_ctrl_plus", "key_up", "mouse_down", "mouse_up", "screenshot"],
)
def test_added_actions_survive_a_full_round_trip(lite_call):
    space = Qwen3_8DesktopActionSpace()
    agent_calls = space.convert_tool_calls_to_agent([lite_call])
    assert space.convert_tool_calls_from_agent(agent_calls) == [lite_call]


# =============================================================================
# Gate wiring
# =============================================================================


def test_valid_actions_can_reach_the_added_actions():
    """``LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES`` is what the ``valid_actions``
    gate reads; without the new rows the added values would be pruned away."""
    space = Qwen3_8DesktopActionSpace()
    schemas = space.filter_tool_schemas_for_valid_actions(
        space.get_tool_schemas(), ["key_down", "key_up", "mouse_down", "mouse_up", "screenshot"]
    )
    enum = tool_schema_parameters(schemas[0])["properties"]["action"]["enum"]
    assert set(enum) >= {
        "key_down",
        "key_up",
        "left_mouse_down",
        "left_mouse_up",
        "screenshot",
    }
    # A GUI-only gate never opens a terminal channel.
    assert "left_click" not in enum


def test_call_user_stays_closed_without_an_active_response_extra():
    """``call_user`` spells the canonical ``response`` tool, so it is gated by
    ``extra_tool_schemas`` — not advertised when the row does not offer one."""
    space = Qwen3_8DesktopActionSpace()
    schemas = space.filter_qwen_action_values_for_active_extra_tools(
        space.get_tool_schemas(), set()
    )
    enum = tool_schema_parameters(schemas[0])["properties"]["action"]["enum"]
    assert "call_user" not in enum
    assert "terminate" not in enum


def test_call_user_opens_when_response_is_active():
    space = Qwen3_8DesktopActionSpace()
    schemas = space.filter_qwen_action_values_for_active_extra_tools(
        space.get_tool_schemas(), {"response"}
    )
    enum = tool_schema_parameters(schemas[0])["properties"]["action"]["enum"]
    assert "call_user" in enum
    assert "terminate" not in enum


# =============================================================================
# Mobile — deliberately NOT the Qwen3.5 subclass
# =============================================================================


def test_mobile_keeps_the_qwen3_vl_mobile_enum():
    space = Qwen3_8MobileActionSpace()
    schema = space.get_tool_schema("mobile_use")
    enum = tool_schema_parameters(schema)["properties"]["action"]["enum"]
    assert enum == [
        "click", "long_press", "swipe", "type", "open",
        "answer", "system_button", "wait", "terminate",
    ]


def test_mobile_does_not_inherit_the_qwen3_5_left_click_alias():
    """Qwen3.5's mobile ``left_click`` -> ``click`` repair was justified by
    measured 3.5-family leakage. A Qwen3.8 rollout on mobilegym emitted zero
    ``left_click`` turns, so inheriting the repair would be an unproven branch.
    """
    assert not isinstance(Qwen3_8MobileActionSpace(), Qwen3_5MobileActionSpace)
    leaked = [{"name": "mobile_use", "arguments": {"action": "left_click", "coordinate": [1, 2]}}]

    def actions(space):
        converted = space.convert_tool_calls_from_agent(leaked)
        return [
            child["action"]
            for call in converted
            for child in call["function"]["arguments"].get("actions", [])
        ]

    assert actions(Qwen3_5MobileActionSpace()) == ["tap"]
    assert actions(Qwen3_8MobileActionSpace()) != ["tap"]


# =============================================================================
# Grounding surfaces are unaffected by the expanded enum
# =============================================================================


@pytest.mark.parametrize(
    "space_cls,agent_call",
    [
        (
            Qwen3_8DesktopGroundingPointActionSpace,
            {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [10, 20]}},
        ),
        (
            Qwen3_8MobileGroundingPointActionSpace,
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [10, 20]}},
        ),
    ],
    ids=["desktop", "mobile"],
)
def test_grounding_point_converts_the_single_click(space_cls, agent_call):
    assert space_cls().convert_tool_calls_from_agent([agent_call]) == [
        make_tool_call("point", {"coordinate": [10, 20]})
    ]
