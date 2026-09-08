"""
Intent pins for the Lite action-space tool catalog.

``lite_builtin_tool_names_for_metadata`` is the runtime action surface. These tests
pin which action-batch tools metadata admits and keep that question separate
from the child GUI action names each batch carries.

Run:
    uv run pytest tests/core/tools/action_space/test_tool_catalog.py -q
"""

from __future__ import annotations

import copy

import pytest

from lite.agents.core.action_space.base import (
    LiteBBoxActionSpace,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.core.errors import LiteContractError
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    LITE_ACTION_SET_TOOL_NAMES,
    LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER,
    LITE_MOBILE_ACTION_NAMES_IN_SCHEMA_ORDER,
    LITE_VALID_ACTION_NAMES,
    LiteBBoxActionSet,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
    is_lite_action_name_or_action_batch_tool_name,
    lite_action_names_for_action_batch_tool,
    lite_action_set_tool_names_for_metadata,
    lite_builtin_tool_names_for_metadata,
)
from lite.core.tools.action_space import batches as batches_module
from lite.core.tools.action_space.batches import (
    LiteActionBatchValidationKind,
    filter_action_batch_schema,
    lite_action_batch_child_name_errors,
    make_lite_action_batch_schema,
    validate_lite_action_batch_child_arguments,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import tool_call_arguments
from lite.core.tools.extra_tools import (
    FINISH_TOOL_ORDER,
    LiteAppLaunchToolSet,
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
    make_open_app_tool,
)
from lite.core.tools.schemas import (
    make_tool_schema,
    tool_schema_name,
    tool_schema_parameters,
)


def _meta(platform: str, task_type: str) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(
            LiteCUAMetadata.Platform(platform).value,
            LiteCUAMetadata.TaskType(task_type).value,
        ),
        extra_tool_schemas=[],
    )


def test_core_name_catalogs_are_authority_for_agent_lite_classes() -> None:
    assert LiteDesktopActionSet.get_tool_names() == LiteDesktopActionSpace.get_tool_names()
    assert LiteMobileActionSet.get_tool_names() == LiteMobileActionSpace.get_tool_names()
    assert LitePointActionSet.get_tool_names() == LitePointActionSpace.get_tool_names()
    assert LiteBBoxActionSet.get_tool_names() == LiteBBoxActionSpace.get_tool_names()
    # The GUI action catalogs are the ACTION question: an action-batch space
    # emits one top-level tool (``computer`` / ``mobile``) but CARRIES every action,
    # so these compare against ``get_action_names()``, never ``get_tool_names()``.
    assert LiteDesktopActionSet.get_action_names() == LiteDesktopActionSpace.get_action_names()
    assert LiteMobileActionSet.get_action_names() == LiteMobileActionSpace.get_action_names()
    # ONE-DIRECTIONAL order pin: the mirrors recompile ``_SCHEMAS`` through their
    # own MRO walk, so the child-action enum they emit must still be the core
    # catalog order the ORDER constants name.
    for space, order in (
        (LiteDesktopActionSpace, LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER),
        (LiteMobileActionSpace, LITE_MOBILE_ACTION_NAMES_IN_SCHEMA_ORDER),
    ):
        (schema,) = space.get_tool_schemas()
        actions = tool_schema_parameters(schema)["properties"]["actions"]
        assert actions["items"]["properties"]["action"]["enum"] == list(order)


def test_action_batch_tool_names_match_the_action_sets() -> None:
    """The action-batch tool set is the desktop/mobile tool-name union."""
    assert LITE_ACTION_BATCH_TOOL_NAMES == (
        LiteDesktopActionSet.get_tool_names() | LiteMobileActionSet.get_tool_names()
    )
    # The action-batch half of the tool layer, and strictly a subset of it: the two
    # grounding sets contribute ``point``/``bbox``, which are not action-batch tools.
    assert LITE_ACTION_BATCH_TOOL_NAMES < LITE_ACTION_SET_TOOL_NAMES


def test_action_layer_union_excludes_the_action_batch_tool_layer() -> None:
    """``LITE_VALID_ACTION_NAMES`` answers ONE question, for all four sets.

    Action names and action-batch tool names are separate layers. Their union is only
    accepted by the one deliberately compound predicate.
    """
    actions = LITE_VALID_ACTION_NAMES
    assert actions == (
        LiteDesktopActionSpace.get_action_names()
        | LiteMobileActionSpace.get_action_names()
        | LitePointActionSpace.get_action_names()
        | LiteBBoxActionSpace.get_action_names()
    )
    # Action-batch tools are disjoint from action names.
    assert not (actions & LITE_ACTION_BATCH_TOOL_NAMES)
    assert "computer" not in actions
    assert "mobile" not in actions
    assert {"click", "tap", "point", "bbox"} <= actions

    for name in sorted(actions | LITE_ACTION_BATCH_TOOL_NAMES):
        assert is_lite_action_name_or_action_batch_tool_name(name), name
    for name in ("terminate", "response", "goto", "open_app", "bash", "nope"):
        assert not is_lite_action_name_or_action_batch_tool_name(name), name


def test_schema_resolvers_come_from_action_space_authority() -> None:
    """``make_open_app_tool`` is the last schema builder outside the classes.

    It survives because it stamps the ENV's app enum onto a copy — a runtime
    fact the class cannot carry — and with no arguments it must be exactly what
    the class declares. Every other body now comes off the class directly.
    """
    assert make_open_app_tool() == LiteAppLaunchToolSet.get_tool_schema("open_app")
    assert LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "back"]) == [
        LiteBrowserNavToolSet.get_tool_schema("goto"),
        LiteBrowserNavToolSet.get_tool_schema("back"),
    ]


def test_emitted_names_all_come_from_a_compiled_schema() -> None:
    """Every emitted tool name resolves to exactly one public schema."""
    for tools in (LiteFinishToolSet, LiteBrowserNavToolSet, LiteAppLaunchToolSet):
        schema_names = [tool_schema_name(schema) for schema in tools.get_tool_schemas()]
        assert frozenset(schema_names) == tools.get_tool_names()
        assert len(schema_names) == len(tools.get_tool_names())
        for name in schema_names:
            assert tools.get_tool_schema(name) is not None


def test_finish_tool_descriptions_route_answers() -> None:
    """The two finish descriptions must keep naming EACH OTHER.

    Reward exposure, not prose. On the answer-scored web benchmarks the grader
    reads ``response.text`` and nothing else, while
    ``gym/envs/browsergym/main.py`` renders ``terminate(status="success")`` as
    the literal message ``"Task completed"`` on that same channel — so an agent
    that reports its answer through ``terminate`` scores 0 on a task it solved
    ``[335/812 WebArena + 280/910 VisualWebArena tasks are answer-scored]``.
    This catalog is the ONLY place that steering survives: it is shared by all
    11 families, and the claude rows carry no ``system_prompt`` of their own.

    Pinned as a PROPERTY, never as a sentence. A golden of the exact wording is
    a rewrite-and-re-bless target — it goes green on any replacement string,
    including one that drops the routing. What actually cannot be lost is the
    CROSS-REFERENCE, so that is what is asserted, off the real accessors and
    with both roles DERIVED (the answering tool is the one whose schema
    requires answer text) rather than spelled here.
    """
    schemas = {tool_schema_name(schema): schema for schema in LiteFinishToolSet.get_tool_schemas()}
    assert set(schemas) == set(FINISH_TOOL_ORDER)

    answering = [
        name
        for name, schema in schemas.items()
        if "text" in tool_schema_parameters(schema)["required"]
    ]
    # The unpacks are the cardinality checks: a third finish tool, or a second
    # answer-bearing one, must not slip in without this routing being redecided.
    (answering,) = answering
    (ending,) = [name for name in schemas if name != answering]

    # Each half of the pair points at the other, so the model is told both which
    # tool submits an answer and which one does not.
    assert ending in schemas[answering]["function"]["description"], schemas[answering]
    assert answering in schemas[ending]["function"]["description"], schemas[ending]

    # ...and through the SINGULAR accessor too, which is what the envs call
    # (``browsergym/main.py`` decorates ``get_tool_schema`` by canonical finish name).
    for name, other in ((answering, ending), (ending, answering)):
        assert other in LiteFinishToolSet.get_tool_schema(name)["function"]["description"]


def test_one_lookup_failure_policy() -> None:
    """One rule per QUESTION, and the two questions differ.

    A SINGULAR getter asks "do you have this?" and answers ``None``. The plural
    ``get_tool_schemas(include=)`` asks "give me exactly these" and on an
    unbatched set is a SELECTOR: an undeclared name RAISES.

    This was briefly a filter that dropped unknown names, and the drop was
    invisible: ``LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "nope"])``
    returned ``["goto"]``. Three envs (``webgym``, ``online_mind2web``,
    ``webharbor.webvoyager``) build their nav schema catalogs from hard-coded
    class-body tuples, so a typo there shrank the env's advertised tool set
    at import time with no signal — the silent-empty-schema class
    ``tests/gym/utils/feedback/test_extra_tools_surface.py`` has matching
    silent-empty-schema
    regression. Do not "restore" drop semantics on the unbatched sets.
    """
    assert LiteFinishToolSet.get_tool_schema("nope") is None
    assert LiteBrowserNavToolSet.get_tool_schema("nope") is None
    assert LiteAppLaunchToolSet.get_tool_schema("nope") is None

    # Unbatched sets: SELECTOR. The message names the class and lists what IS
    # there, because the caller is a hard-coded tuple that needs correcting.
    for tools in (LiteFinishToolSet, LiteBrowserNavToolSet, LiteAppLaunchToolSet):
        with pytest.raises(LiteContractError, match=r"unknown \w+ tool names in include="):
            tools.get_tool_schemas(include=["nope"])
    with pytest.raises(
        LiteContractError,
        match=r"unknown LiteBrowserNavToolSet tool names in include=: \['nope'\]",
    ) as excinfo:
        LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "nope"])
    # One bad name poisons the whole selection — no partial result is returned.
    assert "goto" in str(excinfo.value)  # listed as available, not as a result

    # Action-batch sets keep FILTER semantics, and must: ``include=`` ranges over
    # the ACTION layer there, so naming the action-batch tool is a layer crossing
    # rather than a typo, and filtering to nothing drops that tool.
    assert LiteDesktopActionSet.get_tool_schemas(include=["computer"]) == []
    assert LiteDesktopActionSet.get_tool_schemas(include=["nope"]) == []
    assert [
        tool_schema_name(s) for s in LiteDesktopActionSet.get_tool_schemas(include=["click"])
    ] == ["computer"]


def test_browser_nav_selection_emits_declaration_order_not_caller_order() -> None:
    """Browser-nav selection emits declaration order, not caller order.

    Tool order in ``<tools>`` is model-visible input, so declaration order is a
    catalog contract rather than a caller-selected formatting detail.
    """
    declared = [tool_schema_name(schema) for schema in LiteBrowserNavToolSet.get_tool_schemas()]
    assert declared == [
        "goto",
        "back",
        "forward",
        "new_tab",
        "switch_tab",
        "close_tab",
    ]

    # The caller's order is IGNORED — both spellings give declaration order.
    assert [
        tool_schema_name(s)
        for s in LiteBrowserNavToolSet.get_tool_schemas(include=["back", "goto"])
    ] == ["goto", "back"]
    assert [
        tool_schema_name(s)
        for s in LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "back"])
    ] == ["goto", "back"]

    # ...and duplicates collapse under the selector semantics.
    assert [
        tool_schema_name(s)
        for s in LiteBrowserNavToolSet.get_tool_schemas(include=["back", "back", "goto"])
    ] == ["goto", "back"]


def test_open_app_enum_is_stamped_on_copy_without_mutating_catalog() -> None:
    base_before = LiteAppLaunchToolSet.get_tool_schema("open_app")

    schema = make_open_app_tool(["Settings"])

    assert tool_schema_parameters(schema)["properties"]["app_name"]["enum"] == ["Settings"]
    assert LiteAppLaunchToolSet.get_tool_schema("open_app") == base_before
    base_schema = LiteAppLaunchToolSet.get_tool_schema("open_app")
    assert "enum" not in tool_schema_parameters(base_schema)["properties"]["app_name"]

    schema_again = make_open_app_tool(["Chrome"])
    assert tool_schema_parameters(schema_again)["properties"]["app_name"]["enum"] == ["Chrome"]
    assert tool_schema_parameters(schema)["properties"]["app_name"]["enum"] == ["Settings"]


@pytest.mark.parametrize("platform", ["desktop", "browser"])
def test_grounding_action_desktop_accepts_action_batch_only(platform: str):
    names = lite_builtin_tool_names_for_metadata(_meta(platform, "grounding.action"))
    assert names == {"computer"}
    assert "tap" not in names and "mobile" not in names


def test_grounding_action_mobile_accepts_action_batch_only():
    names = lite_builtin_tool_names_for_metadata(_meta("mobile", "grounding.action"))
    assert names == {"mobile"}
    assert "click" not in names and "computer" not in names


@pytest.mark.parametrize(
    "platform,expected",
    [("desktop", {"computer"}), ("browser", {"computer"}), ("mobile", {"mobile"})],
)
def test_use_accepts_the_action_batch_only(platform: str, expected: set[str]):
    """``use`` is the rollout surface: action-batch tools, no bare actions."""
    names = lite_builtin_tool_names_for_metadata(_meta(platform, "use"))
    assert names == expected
    assert names == lite_action_set_tool_names_for_metadata(_meta(platform, "use"))


def test_filter_action_batch_schema_drops_malformed_same_name_schema():
    """A same-name schema without the action enum must not pass through unfiltered."""
    child_schemas = {
        name: make_tool_schema(name)
        for name in ("click", "key")
    }
    schema = make_lite_action_batch_schema(
        action_batch_tool_name="computer",
        description="Perform desktop actions.",
        child_schemas=child_schemas,
    )
    assert schema is not None
    unrelated = make_tool_schema("response")

    filtered = filter_action_batch_schema(
        [unrelated, schema],
        action_batch_tool_name="computer",
        description="Perform desktop actions.",
        child_schemas=child_schemas,
        valid_actions=["click"],
    )

    assert [tool_schema_name(item) for item in filtered] == ["response", "computer"]
    action_enum = (
        tool_schema_parameters(filtered[1])["properties"]["actions"]["items"][
            "properties"
        ]["action"]["enum"]
    )
    assert action_enum == ["click"]

    malformed = copy.deepcopy(schema)
    del malformed["function"]["parameters"]["properties"]["actions"]["items"][
        "properties"
    ]["action"]["enum"]

    assert filter_action_batch_schema(
        [unrelated, malformed],
        action_batch_tool_name="computer",
        description="Perform desktop actions.",
        child_schemas=child_schemas,
        valid_actions=["click"],
    ) == [unrelated]


# --- ported from the retired source-text tool import ownership check ---
#
# That file asserted, by parsing source text, that registered Lite action spaces
# did not redefine core actions. The runtime form below checks the public
# behavior that matters instead: emitted calls, schema access, and action names.

_SURFACE_PAIRS = (
    ("desktop", LiteDesktopActionSpace, LiteDesktopActionSet),
    ("mobile", LiteMobileActionSpace, LiteMobileActionSet),
    ("point", LitePointActionSpace, LitePointActionSet),
    ("bbox", LiteBBoxActionSpace, LiteBBoxActionSet),
)


_ACTION_EXAMPLES = {
    "key": {"keys": ["enter"]},
    "key_down": {"keys": ["shift"]},
    "key_up": {"keys": ["shift"]},
    "type": {"text": "hello"},
    "hold_key": {"keys": ["shift"], "duration": 0.1},
    "mouse_move": {"coordinate": [10, 20]},
    "click": {"coordinate": [10, 20]},
    "drag": {"start_coordinate": [10, 20], "coordinate": [30, 40]},
    "mouse_down": {},
    "mouse_up": {},
    "scroll": {"direction": "down", "amount": 2, "coordinate": [10, 20]},
    "wait": {"duration": 0.1},
    "screenshot": {},
    "cursor_position": {},
    "tap": {"coordinate": [10, 20]},
    "long_press": {"coordinate": [10, 20], "duration": 0.1},
    "swipe": {"start_coordinate": [10, 20], "coordinate": [30, 40]},
    "pinch": {"coordinate": [10, 20], "direction": "in"},
    "system_button": {"button": "Home"},
    "point": {"coordinate": [10, 20]},
    "bbox": {"coordinate": [10, 20, 30, 40]},
}


@pytest.mark.parametrize("label,space,tools", _SURFACE_PAIRS)
def test_lite_action_spaces_emit_core_action_behavior(label, space, tools):
    """Lite action spaces preserve the core action catalog without pinning storage."""
    names = sorted(tools.get_action_names())
    assert names, f"{label}: tool set declares no actions"
    assert space.get_action_names() == tools.get_action_names()

    mismatched = []
    for name in names:
        kwargs = _ACTION_EXAMPLES[name]
        if getattr(space, name)(**kwargs) != getattr(tools, name)(**kwargs):
            mismatched.append(name)
    assert not mismatched, f"{label}: emitted calls drifted for {mismatched}"


@pytest.mark.parametrize("label,space,tools", _SURFACE_PAIRS)
def test_lite_action_spaces_publish_core_schema_behavior(label, space, tools):
    """Public schema access stays value-equal to the core action set."""
    assert space.get_tool_schemas() == tools.get_tool_schemas(), (
        f"{label}: emitted schemas drifted"
    )
    assert space.get_tool_names() == tools.get_tool_names(), (
        f"{label}: emitted tool names drifted"
    )
    first_action = min(space.get_declared_action_schema_names())
    assert space.get_tool_schemas(include=[first_action]) == tools.get_tool_schemas(
        include=[first_action]
    )


def test_action_batch_child_lookup_keeps_a_miss_distinct_from_an_empty_set() -> None:
    """The miss sentinel of ``lite_action_names_for_action_batch_tool`` is ``None``, and
    that is load-bearing rather than stylistic.

    Two accessors were collapsed into this one, and the survivor returns
    ``frozenset[str] | None`` while every other ``*_names`` accessor in the
    module returns a bare ``frozenset``. Being the odd one out is exactly why it
    needs a pin: the obvious "tidy-up" is to restore the uniform default with
    ``.get(name, frozenset())``, which type-checks, reads better, and silently
    merges two production branches into one.

    ``validate_lite_action_batch_structure`` branches on the ``None`` to report
    ``UNKNOWN_LITE_ACTION_BATCH_TOOL``, and ``lite/gym/utils/feedback/ingress.py`` routes on
    that kind to tell "you called an action-batch tool that does not exist"
    apart from ``CHILD_ACTION_UNKNOWN``, "you called a real action-batch tool
    with an illegal child" — two different things to say back to a model. An
    empty frozenset collapses them, because an action-batch tool admitting no
    children would report the illegal-child kind instead.
    """
    action_batch_tool = sorted(LITE_ACTION_BATCH_TOOL_NAMES)[0]

    # HIT: a real action-batch tool answers with a non-empty child set.
    children = lite_action_names_for_action_batch_tool(action_batch_tool)
    assert isinstance(children, frozenset)
    assert children

    # MISS: ``None`` — NOT ``frozenset()``.
    assert lite_action_names_for_action_batch_tool("not_an_action_batch_tool") is None

    # ...and the distinction survives into the two errors the validator emits.
    # These are the pair that a ``frozenset()`` default collapses into one.
    legal_child = sorted(children)[0]
    _, miss = validate_lite_action_batch_structure(
        "not_an_action_batch_tool", {"actions": [{"action": legal_child}]}
    )
    assert miss is not None
    assert miss.kind is LiteActionBatchValidationKind.UNKNOWN_LITE_ACTION_BATCH_TOOL
    assert miss.reason == "not_an_action_batch_tool is not an action-batch tool"

    illegal_child = "definitely_not_an_action"
    assert illegal_child not in children
    # A missing BATCH TOOL is an envelope error; a missing CHILD ACTION is not.
    # The child survives so env ingress can forward it with its reason and the
    # env can answer it per action -- the name check owns its own function.
    kids, envelope = validate_lite_action_batch_structure(
        action_batch_tool, {"actions": [{"action": illegal_child}]}
    )
    assert envelope is None
    assert [k["name"] for k in kids] == [illegal_child]
    bad_child = lite_action_batch_child_name_errors(action_batch_tool, kids)[0]
    assert bad_child is not None
    assert bad_child.kind is LiteActionBatchValidationKind.CHILD_ACTION_UNKNOWN
    assert bad_child.child_action_name == illegal_child
    assert bad_child.child_index == 0
    assert bad_child.reason == f"{action_batch_tool}.actions cannot contain {illegal_child}"


@pytest.mark.parametrize(
    "arguments,kind,reason",
    [
        (
            "not-an-object",
            LiteActionBatchValidationKind.ARGUMENTS_NOT_OBJECT,
            "computer.arguments must be an object, got str",
        ),
        (
            {},
            LiteActionBatchValidationKind.ACTIONS_NOT_LIST,
            "computer.arguments.actions must be a list",
        ),
        (
            {"actions": []},
            LiteActionBatchValidationKind.ACTIONS_EMPTY,
            "computer.arguments.actions must be a non-empty list",
        ),
        (
            {"actions": ["bad"]},
            LiteActionBatchValidationKind.CHILD_NOT_OBJECT,
            "computer.arguments.actions[0] must be a dict",
        ),
        (
            {"actions": [{"keys": ["enter"]}]},
            LiteActionBatchValidationKind.CHILD_ACTION_MISSING,
            "computer.arguments.actions[0].action must be a non-empty string",
        ),
        (
            {"actions": [{"action": "definitely_not_an_action"}]},
            LiteActionBatchValidationKind.CHILD_ACTION_UNKNOWN,
            "computer.actions cannot contain definitely_not_an_action",
        ),
    ],
)
def test_action_batch_validation_errors_explain_public_rejections(
    arguments,
    kind: LiteActionBatchValidationKind,
    reason: str,
) -> None:
    """``kind`` is the routing contract; ``reason`` is display text only.

    Env feedback (``lite/gym/utils/feedback/ingress.py``) picks its message off
    ``kind``, so every rejection this validator can emit must carry a distinct
    machine-readable class and never force a downstream string match.
    """
    children, error = validate_lite_action_batch_structure("computer", arguments)

    if kind is LiteActionBatchValidationKind.CHILD_ACTION_UNKNOWN:
        # A child naming an action this batch tool does not carry is NOT an
        # envelope error: env ingress forwards it so the env can answer it per
        # action and still frame its slot. The name check has its own function,
        # and the child survives with it.
        assert error is None
        assert len(children) == 1
        name_errors = lite_action_batch_child_name_errors("computer", children)
        assert list(name_errors) == [0]
        error = name_errors[0]
    else:
        assert children == []
        assert error is not None
    assert error.kind is kind
    assert error.action_batch_tool_name == "computer"
    assert error.reason == reason


@pytest.mark.parametrize(
    "arguments,reason",
    [
        (
            {"actions": [{"action": "key"}]},
            "computer.arguments.actions[0].keys is required for key",
        ),
        (
            {"actions": [{"action": "key", "keys": "enter"}]},
            "computer.arguments.actions[0].keys must be an array",
        ),
        (
            {"actions": [{"action": "key", "keys": ["enter"], "unknown": 1}]},
            "computer.arguments.actions[0] has unknown argument 'unknown' for key",
        ),
    ],
)
def test_action_batch_child_argument_errors_explain_public_rejections(
    arguments,
    reason: str,
) -> None:
    """Bad child arguments are one kind, located by child name and index."""
    children, structure_error = validate_lite_action_batch_structure("computer", arguments)
    assert structure_error is None

    error = validate_lite_action_batch_child_arguments("computer", children)

    assert error is not None
    assert error.kind is LiteActionBatchValidationKind.CHILD_ARGUMENTS_INVALID
    assert error.child_action_name == "key"
    assert error.child_index == 0
    assert error.reason == reason


def _action_batch_child_properties(tools: type, action: str) -> dict:
    """Child-item properties an action-batch set PUBLISHES for one action.

    This is the model-visible surface, and the reason the two tests below do
    not read the class declaration table: a batched set emits ONE tool
    (``computer`` / ``mobile``) whose ``actions[]`` items carry the child
    arguments, so what a model may actually send is decided after the batch
    projection, not by the per-action schema the class declares.
    """
    (schema,) = tools.get_tool_schemas(include=[action])
    return tool_schema_parameters(schema)["properties"]["actions"]["items"]["properties"]


def test_desktop_type_carries_only_text() -> None:
    """Enter has ONE canonical spelling: a trailing newline inside ``text``.

    Every transport executes a real newline as the Return key -- the desktop
    sandbox splits on it (``exec_stdio/server.py``), and the browser envs
    project it onto their container's own submit flag at the host boundary. A
    second ``press_enter`` argument would be a second representation of the
    same fact.
    """
    properties = _action_batch_child_properties(LiteDesktopActionSet, "type")
    assert set(properties) == {"action", "text"}

    children, structure_error = validate_lite_action_batch_structure(
        "computer", tool_call_arguments(LiteDesktopActionSpace.type(text="query\n"))
    )
    assert structure_error is None
    assert children == [{"name": "type", "arguments": {"text": "query\n"}}]
    assert validate_lite_action_batch_child_arguments("computer", children) is None


def test_mobile_type_submits_through_system_button() -> None:
    """Mobile has no Enter spelling in ``type`` at all; it submits via ``system_button``."""
    properties = _action_batch_child_properties(LiteMobileActionSet, "type")
    assert set(properties) == {"action", "text"}
    button = _action_batch_child_properties(LiteMobileActionSet, "system_button")["button"]
    assert "Enter" in button["enum"]


def test_action_batch_child_argument_validation_reports_a_non_batch_tool() -> None:
    """A non-action-batch tool is the same kind here as in structural validation."""
    error = validate_lite_action_batch_child_arguments(
        "not_an_action_batch_tool",
        [{"name": "key", "arguments": {"keys": ["enter"]}}],
    )

    assert error is not None
    assert error.kind is LiteActionBatchValidationKind.UNKNOWN_LITE_ACTION_BATCH_TOOL
    assert error.reason == "not_an_action_batch_tool is not an action-batch tool"

    assert validate_lite_action_batch_child_arguments(
        "computer",
        [{"name": "key", "arguments": {"keys": ["enter"]}}],
    ) is None


def test_action_batch_module_exports_only_the_intended_owner_surface() -> None:
    """The leaf ``__all__`` is the action-batch owner API, nothing more.

    ``filter_action_batch_schema`` is importable from this module because
    ``base.py`` calls it, but it is package-internal: it stays off both the leaf
    ``__all__`` and the ``lite.core.tools.action_space`` facade
    (``tests/static/test_public_facades.py`` pins the facade half).
    """
    assert batches_module.__all__ == [
        "LITE_COMPUTER_ACTION_BATCH_TOOL_NAME",
        "LITE_MOBILE_ACTION_BATCH_TOOL_NAME",
        "make_lite_action_batch_call",
        "make_lite_action_batch_schema",
        "merge_adjacent_lite_action_batches",
        "unpack_action_batch_call",
        "validate_lite_action_batch_structure",
        "LiteActionBatchMergeResult",
        "LiteActionBatchValidationError",
        "LiteActionBatchValidationKind",
        "merge_adjacent_lite_action_batches_with_provenance",
        "validate_lite_action_batch_child_arguments",
    ]
    assert "filter_action_batch_schema" not in batches_module.__all__
    assert hasattr(batches_module, "filter_action_batch_schema")
