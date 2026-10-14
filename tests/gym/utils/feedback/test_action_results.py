"""Env-side tool-result / unpack contract.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/gym/utils/feedback/test_action_results.py -p no:cacheprovider -q
"""

from __future__ import annotations

import asyncio
from typing import Any, get_type_hints

import pytest

from lite.agents.core.agent.utils.annotations import action_inspection_records
from lite.agents.core.agent.utils.final import (
    NoToolCallFinal,
    mark_no_tool_call_final_result,
)
from lite.core.messages.final import ENV_INTERNAL_TERMINATE_REASON
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    unpack_action_batch_call,
    validate_lite_action_batch_structure,
)
from lite.core.tools.action_space import unpack_action_batch_call as core_unpack_action_batch_call
from lite.core.tools.calls import (
    RUNTIME_INTERNAL_STOP_REASON_KEY,
    RUNTIME_RESULT_CALL_ID_KEY,
    LiteToolCall,
    make_tool_call,
)
from lite.core.tools.extra_tools import (
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
)
from lite.core.tools.results import (
    LiteToolResult,
    make_tool_result,
    project_tool_result_text,
)
from lite.core.tools.results import (
    LiteToolResult as CoreLiteToolResult,
)
from lite.core.tools.schemas import (
    make_tool_schema,
    tool_call_satisfies_schema,
    tool_schema_parameters,
)
from lite.gym.base import LiteBaseEnv
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
    coerce_model_numeric,
    project_model_keys,
)
from lite.gym.utils.feedback.errors import (
    BATCH_ABORT_PREFIX,
    BATCH_ABORT_SIBLING_MESSAGE,
    ToolErrorFeedback,
    batch_abort_message,
    current_feedback,
    error_only_feedback,
    invalid_action_arguments_message,
    record_model_action_error,
    record_tool_execution_error,
    tool_execution_error_message,
    unavailable_action_message,
    unknown_tool_message,
    unsupported_action_message,
)
from lite.gym.utils.feedback.ingress import (
    classify_standalone_tool_call,
    is_active_extra_tool_call,
    is_inactive_tool_call,
    is_loop_detect_terminate,
    is_unknown_tool_call,
    make_internal_terminate_action,
    prepare_env_tool_calls,
    standalone_tool_call_feedback,
    unsupported_env_action_message,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)


def _simple_tool_schema(
    name: str,
    properties: dict,
    required: list[str],
) -> dict:
    return make_tool_schema(
        name,
        description=f"{name} tool",
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


def _call(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str | None = "call_0001",
) -> dict[str, Any]:
    return make_tool_call(name, arguments or {}, call_id=call_id)


def _env_action(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "arguments": arguments or {}}


def _batch(
    wrapper: str,
    actions: list[dict[str, Any]],
    *,
    call_id: str | None = "call_0001",
) -> dict[str, Any]:
    return _call(wrapper, {"actions": actions}, call_id=call_id)


def _call_with_raw_arguments(
    name: str,
    arguments: Any,
    *,
    call_id: str | None = "call_0001",
) -> dict[str, Any]:
    call: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    if call_id is not None:
        call = {"id": call_id, **call}
    return call


def _assemble(
    result: LiteEnvStepResult,
    tool_calls: list[LiteToolCall],
    *,
    images: list[bytes] | None = None,
    text: str | None = None,
    metadata: dict | None = None,
    feedback: dict[str, ToolErrorFeedback] | None = None,
    continue_call_ids: set[str | None] | None = None,
) -> LiteEnvStepResult:
    ordered = ordered_tool_call_ids(tool_calls)
    return build_tool_results_from_decisions(
        result,
        ordered_call_ids=ordered,
        continue_call_ids=set(ordered) if continue_call_ids is None else continue_call_ids,
        images=images,
        text=text,
        metadata=metadata,
        feedback=feedback,
    )


def test_lite_tool_result_is_core_owned_gym_compat_export():
    assert LiteToolResult is CoreLiteToolResult
    hints = get_type_hints(LiteEnvStepResult)
    assert hints["results"].__origin__ is list
    assert hints["results"].__args__ == (CoreLiteToolResult,)


def test_core_tool_result_helper_marks_error_metadata_without_carrier_policy():
    metadata = {"url": "https://example.test"}

    ok = make_tool_result(
        tool_call_id="call_ok",
        images=[b"png"],
        text="current observation",
        metadata=metadata,
    )
    err = make_tool_result(
        tool_call_id="call_err",
        images=[b"png"],
        text="current observation",
        metadata=metadata,
        error="invalid action",
    )
    error_only = make_tool_result(tool_call_id="call_error_only", error="unknown tool")

    assert ok == LiteToolResult(
        tool_call_id="call_ok",
        images=[b"png"],
        text="current observation",
        metadata=metadata,
    )
    assert err == LiteToolResult(
        tool_call_id="call_err",
        images=[b"png"],
        text="current observation",
        metadata={"url": "https://example.test", "is_error": True},
        error="invalid action",
    )
    assert error_only == LiteToolResult(
        tool_call_id="call_error_only",
        metadata={"is_error": True},
        error="unknown tool",
    )
    assert metadata == {"url": "https://example.test"}


def test_action_batch_abort_feedback_preserves_model_visible_text():
    assert BATCH_ABORT_PREFIX == "batch aborted: "
    assert BATCH_ABORT_SIBLING_MESSAGE == (
        "batch aborted: not executed, because an earlier action in this step was rejected"
    )
    assert batch_abort_message(1) == (
        "batch aborted: the 1 later action was not executed; "
        "a batch stops at the first rejected action"
    )
    assert batch_abort_message(2) == (
        "batch aborted: the 2 later actions were not executed; "
        "a batch stops at the first rejected action"
    )


def test_unpack_action_batch_keeps_child_calls_unstamped():
    actions = action_inspection_records(
        [
            _batch(
                "computer",
                [
                    {"action": "click", "coordinate": [100, 100]},
                    {"action": "key", "keys": ["ctrl c"]},
                ],
                call_id="call_0000",
            )
        ]
    )

    assert all("call_id" not in action for action in actions)
    assert [action["name"] for action in actions] == ["click", "key"]
    assert [action["result_call_id"] for action in actions] == ["call_0000", "call_0000"]


def test_action_inspection_records_return_uniform_expanded_records():
    expanded = action_inspection_records(
        [
            _batch(
                "computer",
                [
                    {"action": "click", "coordinate": [100, 100]},
                    {"action": "key", "keys": ["ctrl c"]},
                ],
                call_id="call_0000",
            ),
            _call("bash", {"command": "pwd"}, call_id="call_0001"),
        ]
    )

    assert expanded == [
        {
            "name": "click",
            "arguments": {"coordinate": [100, 100]},
            "result_call_id": "call_0000",
        },
        {
            "name": "key",
            "arguments": {"keys": ["ctrl c"]},
            "result_call_id": "call_0000",
        },
        {
            "name": "bash",
            "arguments": {"command": "pwd"},
            "result_call_id": "call_0001",
        },
    ]


def test_action_error_result_preserves_post_step_observation_and_image():
    result = LiteEnvStepResult()
    obs_text = "## AXTree:\nbutton 'Search'\n\n## HTML:\n<button>Search</button>"

    out = _assemble(
        result,
        [
            _batch(
                "computer",
                [{"action": "key", "keys": ["ctrl c"]}],
                call_id="call_0000",
            )
        ],
        images=[b"png"],
        text=obs_text,
        metadata={"url": "https://example.test"},
        feedback={"call_0000": current_feedback("invalid action arguments: bad key")},
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.tool_call_id == "call_0000"
    assert tool_result.images[-1] == b"png"
    assert tool_result.text == obs_text
    assert tool_result.error == "invalid action arguments: bad key"
    assert tool_result.metadata == {"url": "https://example.test", "is_error": True}


def test_terminal_action_error_preserves_error_separately():
    out = _assemble(
        LiteEnvStepResult(terminated=True),
        [
            _batch(
                "computer",
                [{"action": "click", "coordinate": [1, 2]}],
                call_id="call_0000",
            )
        ],
        images=[b"png"],
        text="final observation",
        metadata={"url": "https://example.test"},
        feedback={"call_0000": current_feedback("invalid action arguments: bad key")},
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.tool_call_id == "call_0000"
    assert tool_result.images[-1] == b"png"
    assert tool_result.text == "final observation"
    assert tool_result.error == "invalid action arguments: bad key"
    assert tool_result.metadata == {"url": "https://example.test", "is_error": True}


def test_terminal_active_response_error_preserves_observation():
    errors: dict[str, ToolErrorFeedback] = {}
    record_tool_execution_error(
        errors,
        "call_response",
        "response side effect failed: cache write failed",
        action_name="response",
    )
    out = _assemble(
        LiteEnvStepResult(terminated=True),
        [make_tool_call("response", {"text": "done"}, call_id="call_response")],
        images=[b"png"],
        text="final observation",
        metadata={"url": "https://example.test"},
        feedback=errors,
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.tool_call_id == "call_response"
    assert tool_result.images[-1] == b"png"
    assert tool_result.text == "final observation"
    assert tool_result.error == "response failed: execution failed"
    assert tool_result.metadata == {"url": "https://example.test", "is_error": True}


def test_literal_unknown_result_is_error_only():
    result = LiteEnvStepResult()
    error = unknown_tool_message("frobnicate")

    out = _assemble(
        result,
        [_call("frobnicate")],
        images=[b"png"],
        text="page text",
        feedback={"call_0001": error_only_feedback(error)},
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.tool_call_id == "call_0001"
    assert tool_result.images == []
    assert tool_result.text is None
    assert tool_result.error == error
    assert tool_result.metadata == {"is_error": True}


def test_unsupported_gui_result_carries_post_step_image():
    """An ``unsupported`` GUI rejection must not blind the model for the turn.

    After turn 0 the ONLY source of the current screen is
    ``pending_tool_results[*].images`` (``lite/agents/core/agent/base.py``, the
    ``elif pending_tool_results is not None`` observe branch). A turn whose
    single result carries no image therefore leaves the model with no current
    frame. The sibling ``tool_call_errors`` branch already attaches the
    end-of-turn capture; envs route an unknown GUI action name through
    ``unsupported`` instead (osworld, osworld_2, browsergym, webgym,
    webvoyager, mobilegym, waa, captcha, lite.osworld, sandbox …), so for a GUI
    call the two branches must agree.
    """
    result = LiteEnvStepResult()

    out = _assemble(
        result,
        [_batch("computer", [{"action": "scroll_left"}])],
        images=[b"png"],
        text="page text",
        feedback={"call_0001": current_feedback("unsupported action: scroll_left")},
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.tool_call_id == "call_0001"
    assert tool_result.images[-1] == b"png"
    assert tool_result.text == "page text"
    assert tool_result.error == "unsupported action: scroll_left"
    assert tool_result.metadata == {"is_error": True}


def test_terminal_unsupported_gui_result_carries_post_step_image():
    out = _assemble(
        LiteEnvStepResult(terminated=True),
        [_batch("computer", [{"action": "scroll_left"}])],
        images=[b"png"],
        text="final page text",
        feedback={"call_0001": current_feedback("unsupported action: scroll_left")},
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.tool_call_id == "call_0001"
    assert tool_result.images[-1] == b"png"
    assert tool_result.text == "final page text"
    assert tool_result.error == "unsupported action: scroll_left"
    assert tool_result.metadata == {"is_error": True}


@pytest.mark.parametrize("name", ["bash", "goto", "terminate"])
def test_unsupported_text_tool_result_stays_error_only(name):
    """Scope guard: the image is keyed on the call being GUI, not unconditional.

    An inactive standalone text tool executed nothing and moved no pixels, so
    its rejection stays text-only even though the env captured a frame this
    turn (pinned by e.g. ``tests/gym/envs/webgym/test_webgym.py``'s
    ``test_unknown_extra_returns_text_result``).
    """
    out = _assemble(
        LiteEnvStepResult(),
        [_call(name)],
        images=[b"png"],
        text=None,
        feedback={"call_0001": error_only_feedback(f"unsupported action: {name}")},
    )

    assert len(out.results) == 1
    assert out.results[0].images == []
    assert out.results[0].text is None
    assert out.results[0].error == f"unsupported action: {name}"
    assert out.results[0].metadata == {"is_error": True}


@pytest.mark.parametrize("name", ["bash", "goto", "terminate"])
def test_terminal_unsupported_text_tool_result_stays_error_only(name):
    out = _assemble(
        LiteEnvStepResult(terminated=True),
        [_call(name)],
        images=[b"png"],
        text="final page text",
        feedback={"call_0001": error_only_feedback(f"unsupported action: {name}")},
    )

    assert len(out.results) == 1
    assert out.results[0].images == []
    assert out.results[0].text is None
    assert out.results[0].error == f"unsupported action: {name}"
    assert out.results[0].metadata == {"is_error": True}


def test_unsupported_result_has_no_image_when_the_env_captured_none():
    """A no-capture env passes ``images=None``; nothing invents a frame."""
    out = _assemble(
        LiteEnvStepResult(),
        [_batch("computer", [{"action": "scroll_left"}])],
        images=None,
        text=None,
        feedback={"call_0001": current_feedback("unsupported action: scroll_left")},
    )

    assert len(out.results) == 1
    assert out.results[0].images == []


def test_unsupported_gui_action_message_uses_shared_catalog():
    supported = frozenset({"tap", "swipe"})

    assert unsupported_env_action_message("tap", supported) is None
    assert unsupported_env_action_message("pinch", supported) == "unsupported action: pinch"
    assert unsupported_env_action_message("open_app", supported) is None


@pytest.mark.parametrize(
    ("raw_keys", "message"),
    [
        ("enter", "key.keys must be a list of strings, not a string"),
        (["ctrl", 3], r"key.keys\[1\] must be a string"),
        ([], "key.keys must not be empty"),
        (["Ctrl"], "keys must be lowercase tokens"),
        (["ctrl+a"], "split chords into separate keys"),
        (["plus"], "unknown key token"),
        (["minus"], "unknown key token"),
        (["equal"], "unknown key token"),
        (["comma"], "unknown key token"),
        ([" "], "unknown key token"),
        (["\n"], "unknown key token"),
        (["\t"], "unknown key token"),
        (["\r"], "unknown key token"),
        (["\x1b"], "unknown key token"),
        (["\x00"], "unknown key token"),
        (["definitely_not_a_key"], "unknown key token"),
    ],
)
def test_project_model_keys_rejects_model_controlled_bad_shapes(
    raw_keys,
    message,
):
    with pytest.raises(ValueError, match=message):
        project_model_keys(raw_keys, action_name="key")


def test_project_model_keys_accepts_canonical_printable_glyphs():
    assert project_model_keys(["ctrl", "+", "-", "=", ","], action_name="key") == [
        "ctrl",
        "+",
        "-",
        "=",
        ",",
    ]


def test_action_batch_child_key_validation_uses_canonical_glyphs_for_feedback():
    actions, errors = prepare_env_tool_calls(
        [
            _batch(
                "computer",
                [
                    {"action": "key", "keys": ["ctrl", "+"]},
                    {"action": "key", "keys": ["ctrl", "plus"]},
                ],
                call_id="call_gui",
            )
        ],
        LiteCUAMetadata(),
    )

    assert errors == {}
    assert project_model_keys(actions[0][0]["arguments"]["keys"], action_name="key") == [
        "ctrl",
        "+",
    ]

    feedback: dict[str, ToolErrorFeedback] = {}
    with pytest.raises(ValueError, match="unknown key token 'plus'") as exc_info:
        project_model_keys(actions[1][0]["arguments"]["keys"], action_name="key")
    record_model_action_error(feedback, "call_gui", exc_info.value, action_name="key")

    assert feedback["call_gui"].message == "invalid arguments for key: unknown key token 'plus'"


def test_project_model_keys_checks_backend_specific_vocabulary():
    assert project_model_keys(["ctrl", "a"], action_name="key", backend="xdotool") == [
        "ctrl",
        "a",
    ]

    with pytest.raises(ValueError, match="no pynput key for 'f24'"):
        project_model_keys(["f24"], action_name="key", backend="pynput")


def test_pairable_model_key_validation_error_becomes_tool_feedback():
    errors: dict[str, ToolErrorFeedback] = {}

    try:
        project_model_keys("enter", action_name="key")
    except ValueError as e:
        record_model_action_error(errors, "call_key", e, action_name="key")

    out = build_tool_results_from_decisions(
        LiteEnvStepResult(),
        ordered_call_ids=["call_key"],
        feedback=errors,
        images=[b"png"],
        text="current observation",
    )

    assert out.results == [
        LiteToolResult(
            tool_call_id="call_key",
            images=[b"png"],
            text="current observation",
            metadata={"is_error": True},
            error=("invalid arguments for key: key.keys must be a list of strings, not a string"),
        )
    ]


@pytest.mark.parametrize(
    "bad_duration",
    [float("nan"), float("inf"), -1, "slow", "", 31],
)
def test_coerce_model_duration_rejects_bad_or_huge_values(bad_duration):
    with pytest.raises(ValueError, match="wait.duration"):
        coerce_model_duration(bad_duration, action_name="wait")


def test_coerce_model_duration_accepts_finite_numeric_strings_with_cap():
    assert coerce_model_duration("1.25", action_name="wait") == 1.25
    assert coerce_model_duration(5, action_name="wait", maximum=5.0) == 5.0


@pytest.mark.parametrize("bad_value", [True, "many", 0, 21, 2.5, float("nan")])
def test_coerce_model_numeric_rejects_bad_bounded_integer_fields(bad_value):
    with pytest.raises(ValueError, match="click.clicks"):
        coerce_model_numeric(
            bad_value,
            field="clicks",
            action_name="click",
            min_value=1,
            max_value=20,
            integer=True,
        )


def test_coerce_model_numeric_accepts_bounded_integer_string():
    assert (
        coerce_model_numeric(
            "3",
            field="clicks",
            action_name="click",
            min_value=1,
            max_value=20,
            integer=True,
        )
        == 3
    )


def test_shared_tool_classification_predicates_split_active_inactive_unknown_and_gui():
    bash_schema = _simple_tool_schema(
        "bash",
        {"command": {"type": "string"}},
        ["command"],
    )
    active_schemas = [bash_schema]
    known_standalone = {"bash", "goto", "response", "terminate"}
    active = _env_action("bash", {"command": "pwd"})
    inactive = _env_action("response", {"text": "done"})
    unknown = _env_action("frobnicate")
    gui = _env_action("click", {"coordinate": [1, 2]})

    assert is_active_extra_tool_call(active, active_schemas)
    assert classify_standalone_tool_call(active, known_standalone, active_schemas) == "active"
    assert not is_inactive_tool_call(active, known_standalone, active_schemas)
    assert not is_unknown_tool_call(active, known_standalone, active_schemas)

    assert not is_active_extra_tool_call(inactive, active_schemas)
    assert classify_standalone_tool_call(inactive, known_standalone, active_schemas) == "inactive"
    assert is_inactive_tool_call(inactive, known_standalone, active_schemas)
    assert not is_unknown_tool_call(inactive, known_standalone, active_schemas)

    assert classify_standalone_tool_call(unknown, known_standalone, active_schemas) == "unknown"
    assert not is_inactive_tool_call(unknown, known_standalone, active_schemas)
    assert is_unknown_tool_call(unknown, known_standalone, active_schemas)

    assert classify_standalone_tool_call(gui, known_standalone, active_schemas) == "not_standalone"
    assert not is_inactive_tool_call(gui, known_standalone, active_schemas)
    assert not is_unknown_tool_call(gui, known_standalone, active_schemas)


def test_shared_tool_classification_keeps_same_name_gui_and_extra_shapes_separate():
    click_schema = _simple_tool_schema(
        "click",
        {"index": {"type": "integer"}},
        ["index"],
    )
    known_standalone = {"click"}
    dom_click = _env_action("click", {"index": 1})
    gui_click = _env_action("click", {"coordinate": [1, 2]})

    def is_dom_shape(action: dict[str, Any]) -> bool:
        return "index" in action["arguments"]

    assert (
        classify_standalone_tool_call(
            dom_click,
            known_standalone,
            [],
            is_standalone_action_tool=is_dom_shape,
        )
        == "inactive"
    )
    assert (
        classify_standalone_tool_call(
            dom_click,
            known_standalone,
            [click_schema],
            is_standalone_action_tool=is_dom_shape,
        )
        == "active"
    )
    assert (
        classify_standalone_tool_call(
            gui_click,
            known_standalone,
            [click_schema],
            is_standalone_action_tool=is_dom_shape,
        )
        == "not_standalone"
    )


def test_shared_tool_classification_takes_one_kind_of_known_name_answer():
    """A NAME SET, never a predicate.

    Every env used to hand this a hand-rolled ``(name) -> bool`` closure, which
    is the only reason the parameter's type was a union. Envs now declare a
    ``BaseTools`` subclass and pass ``get_tool_names() | LiteFinishToolSet.get_tool_names()``,
    so there is one kind of answer and the union is gone.
    """
    active_schemas = [_simple_tool_schema("response", {"text": {"type": "string"}}, ["text"])]
    known = frozenset({"response", "terminate"})

    inactive = _env_action("terminate", {"status": "success"})
    unknown = _env_action("report_bug")

    assert classify_standalone_tool_call(inactive, known, active_schemas) == "inactive"
    assert classify_standalone_tool_call(unknown, known, active_schemas) == "unknown"


def test_standalone_tool_call_feedback_owns_default_carrier_mapping():
    active_schemas = [_simple_tool_schema("response", {"text": {"type": "string"}}, ["text"])]
    known = frozenset({"response", "terminate"})

    inactive = _env_action("terminate", {"status": "success"})
    unknown = _env_action("report_bug")
    active = _env_action("response", {"text": "done"})
    gui = _env_action("click", {"coordinate": [1, 2]})

    assert standalone_tool_call_feedback(inactive, known, active_schemas) == error_only_feedback(
        unavailable_action_message("terminate")
    )
    assert standalone_tool_call_feedback(unknown, known, active_schemas) == error_only_feedback(
        unknown_tool_message("report_bug")
    )
    assert standalone_tool_call_feedback(active, known, active_schemas) is None
    assert standalone_tool_call_feedback(gui, known, active_schemas) is None


def test_standalone_tool_call_feedback_respects_same_name_shape_override():
    click_schema = _simple_tool_schema(
        "click",
        {"index": {"type": "integer"}},
        ["index"],
    )
    known = {"click"}
    dom_click = _env_action("click", {"index": 1})
    gui_click = _env_action("click", {"coordinate": [1, 2]})

    def is_dom_shape(action: dict[str, Any]) -> bool:
        return "index" in action["arguments"]

    assert standalone_tool_call_feedback(
        dom_click,
        known,
        [],
        is_standalone_action_tool=is_dom_shape,
    ) == error_only_feedback(unavailable_action_message("click"))
    assert (
        standalone_tool_call_feedback(
            dom_click,
            known,
            [click_schema],
            is_standalone_action_tool=is_dom_shape,
        )
        is None
    )
    assert (
        standalone_tool_call_feedback(
            gui_click,
            known,
            [click_schema],
            is_standalone_action_tool=is_dom_shape,
        )
        is None
    )


def test_env_owned_native_tool_can_keep_current_feedback_explicitly():
    out = _assemble(
        LiteEnvStepResult(),
        [_call("goto", {"url": "https://example.test"})],
        images=[b"png"],
        text="current page",
        metadata={"url": "https://example.test"},
        feedback={"call_0001": current_feedback("goto is not available in this task.")},
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.images[-1] == b"png"
    assert tool_result.text == "current page"
    assert tool_result.error == "goto is not available in this task."
    assert tool_result.metadata == {"url": "https://example.test", "is_error": True}


def test_env_owned_native_tool_with_bad_arguments_keeps_current_feedback_explicitly():
    out = _assemble(
        LiteEnvStepResult(),
        [_call("goto")],
        images=[b"png"],
        text="current page",
        feedback={
            "call_0001": current_feedback("invalid arguments for goto: url must be a string")
        },
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.images[-1] == b"png"
    assert tool_result.text == "current page"
    assert tool_result.error == "invalid arguments for goto: url must be a string"
    assert tool_result.metadata == {"is_error": True}


def test_unsupported_known_native_name_can_keep_env_feedback():
    out = _assemble(
        LiteEnvStepResult(),
        [_call("done", {"text": "ok"})],
        images=[b"png"],
        text="current page",
        metadata={"url": "https://example.test"},
        feedback={"call_0001": current_feedback("unsupported action: done")},
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.images[-1] == b"png"
    assert tool_result.text == "current page"
    assert tool_result.error == "unsupported action: done"
    assert tool_result.metadata == {"url": "https://example.test", "is_error": True}


def test_invalid_active_extra_arguments_do_not_inherit_feedback_carrier():
    out = _assemble(
        LiteEnvStepResult(),
        [_call("bash", {"command": ["pwd"]})],
        images=[b"png"],
        text="current screen text",
        metadata={"screen": "current"},
        feedback={
            "call_0001": error_only_feedback(
                "invalid arguments for bash: bash.arguments.command must be a string"
            )
        },
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.images == []
    assert tool_result.text is None
    assert tool_result.error == (
        "invalid arguments for bash: bash.arguments.command must be a string"
    )
    assert tool_result.metadata == {"is_error": True}


def test_invalid_native_arguments_keep_env_feedback_by_key_contract():
    out = _assemble(
        LiteEnvStepResult(),
        [_call("goto", {"url": 123})],
        images=[b"png"],
        text="current page",
        metadata={"url": "https://example.test"},
        feedback={
            "call_0001": current_feedback(
                "invalid action arguments: goto.arguments.url must be a string"
            )
        },
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.images[-1] == b"png"
    assert tool_result.text == "current page"
    assert tool_result.error == ("invalid action arguments: goto.arguments.url must be a string")
    assert tool_result.metadata == {"url": "https://example.test", "is_error": True}


def test_prepare_env_tool_calls_rejects_active_extra_with_bad_arguments():
    metadata = LiteCUAMetadata(
        extra_tool_schemas=LiteBrowserNavToolSet.get_tool_schemas(include=["goto"])
    )

    actions, errors = prepare_env_tool_calls(
        [_call("goto")],
        metadata,
    )

    assert actions == []
    assert errors == {
        "call_0001": current_feedback("invalid arguments for goto: goto.arguments.url is required")
    }


def test_prepare_env_tool_calls_bad_extra_schema_fails_loudly():
    from jsonschema.exceptions import SchemaError

    bad_schema = make_tool_schema(
        "goto",
        parameters={
            "type": "object",
            "properties": {"url": {"type": 123}},
            "required": ["url"],
        },
    )
    metadata = LiteCUAMetadata(extra_tool_schemas=[bad_schema])

    with pytest.raises(SchemaError):
        prepare_env_tool_calls(
            [_call("goto", {"url": 123})],
            metadata,
        )


def test_tool_call_satisfies_schema_bad_schema_fails_loudly():
    from jsonschema.exceptions import SchemaError

    bad_schema = make_tool_schema(
        "goto",
        parameters={
            "type": "object",
            "properties": {"url": {"type": 123}},
            "required": ["url"],
        },
    )

    with pytest.raises(SchemaError):
        tool_call_satisfies_schema(
            _call("goto", {"url": 123}),
            bad_schema,
        )


def test_prepare_env_tool_calls_rejects_colliding_extra_with_bad_type():
    click_schema = _simple_tool_schema(
        "click",
        {"index": {"type": "integer"}},
        ["index"],
    )
    metadata = LiteCUAMetadata(extra_tool_schemas=[click_schema])

    actions, errors = prepare_env_tool_calls(
        [_call("click", {"index": "1"})],
        metadata,
    )

    assert actions == []
    assert errors == {
        "call_0001": current_feedback(
            "invalid arguments for click: click.arguments.index must be an integer"
        )
    }


def test_prepare_env_tool_calls_leaves_colliding_gui_shape_to_env():
    click_schema = _simple_tool_schema(
        "click",
        {"index": {"type": "integer"}},
        ["index"],
    )
    metadata = LiteCUAMetadata(extra_tool_schemas=[click_schema])

    actions, errors = prepare_env_tool_calls(
        [_call("click", {"coordinate": [1, 2]})],
        metadata,
    )

    assert actions == [
        (
            {"call_id": "call_0001", "name": "click", "arguments": {"coordinate": [1, 2]}},
            "call_0001",
        )
    ]
    assert errors == {}


def test_prepare_env_tool_calls_rejects_nested_standalone_extra_tool_shape():
    click_schema = _simple_tool_schema(
        "click",
        {"index": {"type": "integer"}},
        ["index"],
    )
    metadata = LiteCUAMetadata(extra_tool_schemas=[click_schema])

    actions, errors = prepare_env_tool_calls(
        [_batch("computer", [{"action": "click", "index": 7}])],
        metadata,
    )

    # R4: the child keeps its slot so the env can frame it; the reason travels
    # on the action instead of killing the call at ingress.
    assert [a["name"] for a, _ in actions] == ["click"]
    assert actions[0][0]["_rejected_reason"] == (
        "invalid action: click; computer.actions cannot contain standalone extra tool click"
    )
    assert errors == {}


@pytest.mark.parametrize(
    ("wrapper", "first", "second", "valid_actions", "message"),
    [
        (
            "computer",
            {"action": "click", "coordinate": [10, 20]},
            {"action": "key", "keys": ["ctrl", "c"]},
            ["click"],
            "invalid action: key; choose an available action for this task",
        ),
        (
            "mobile",
            {"action": "tap", "coordinate": [10, 20]},
            {"action": "swipe", "start": [10, 20], "end": [30, 40]},
            ["tap"],
            "invalid action: swipe; choose an available action for this task",
        ),
    ],
)
def test_prepare_env_tool_calls_rejects_only_the_bad_child_for_valid_actions(
    wrapper,
    first,
    second,
    valid_actions,
    message,
):
    metadata = LiteCUAMetadata(valid_actions=valid_actions)

    actions, errors = prepare_env_tool_calls(
        [_batch(wrapper, [first, second], call_id="call_gui")],
        metadata,
    )

    # R4: the bad child costs itself, not the batch. Both slots survive, the
    # valid sibling is untouched, and the reason rides on the child that earned
    # it so the env can answer it per action and still frame its slot.
    assert [a["name"] for a, _ in actions] == [first["action"], second["action"]]
    assert actions[0][0].get("_rejected_reason") is None
    assert actions[1][0]["_rejected_reason"] == message
    assert errors == {}


@pytest.mark.parametrize(
    ("wrapper", "first", "second", "message"),
    [
        (
            "computer",
            {"action": "click", "coordinate": [10, 20]},
            {"action": "open_app", "app_name": "Clock"},
            "invalid action: open_app; computer.actions cannot contain open_app",
        ),
        (
            "mobile",
            {"action": "tap", "coordinate": [10, 20]},
            {"action": "click", "coordinate": [30, 40]},
            "invalid action: click; mobile.actions cannot contain click",
        ),
    ],
)
def test_prepare_env_tool_calls_rejects_only_the_bad_child_for_structure(
    wrapper,
    first,
    second,
    message,
):
    actions, errors = prepare_env_tool_calls(
        [_batch(wrapper, [first, second], call_id="call_gui")],
        LiteCUAMetadata(),
    )

    # R4: the bad child costs itself, not the batch. Both slots survive, the
    # valid sibling is untouched, and the reason rides on the child that earned
    # it so the env can answer it per action and still frame its slot.
    assert [a["name"] for a, _ in actions] == [first["action"], second["action"]]
    assert actions[0][0].get("_rejected_reason") is None
    assert actions[1][0]["_rejected_reason"] == message
    assert errors == {}


@pytest.mark.parametrize(
    ("wrapper", "schema_name", "first", "second", "message"),
    [
        (
            "computer",
            "click",
            {"action": "key", "keys": ["ctrl", "c"]},
            {"action": "click", "index": 7},
            "invalid action: click; computer.actions cannot contain standalone extra tool click",
        ),
        (
            "mobile",
            "tap",
            {"action": "swipe", "start": [10, 20], "end": [30, 40]},
            {"action": "tap", "index": 7},
            "invalid action: tap; mobile.actions cannot contain standalone extra tool tap",
        ),
    ],
)
def test_prepare_env_tool_calls_rejects_only_the_bad_child_for_nested_extra(
    wrapper,
    schema_name,
    first,
    second,
    message,
):
    extra_schema = _simple_tool_schema(
        schema_name,
        {"index": {"type": "integer"}},
        ["index"],
    )
    metadata = LiteCUAMetadata(extra_tool_schemas=[extra_schema])

    actions, errors = prepare_env_tool_calls(
        [_batch(wrapper, [first, second], call_id="call_gui")],
        metadata,
    )

    # R4: the bad child costs itself, not the batch. Both slots survive, the
    # valid sibling is untouched, and the reason rides on the child that earned
    # it so the env can answer it per action and still frame its slot.
    assert [a["name"] for a, _ in actions] == [first["action"], second["action"]]
    assert actions[0][0].get("_rejected_reason") is None
    assert actions[1][0]["_rejected_reason"] == message
    assert errors == {}


def test_prepare_env_tool_calls_wraps_shared_action_batch_structure_error():
    arguments = {"actions": [{"coordinate": [1, 2]}]}
    _children, error = validate_lite_action_batch_structure("computer", arguments)
    assert error is not None
    metadata = LiteCUAMetadata()

    actions, errors = prepare_env_tool_calls(
        [_call("computer", arguments)],
        metadata,
    )

    assert actions == []
    assert errors == {
        "call_0001": current_feedback(
            "invalid arguments for computer: "
            "computer.arguments.actions[0].action must be a non-empty string"
        )
    }


@pytest.mark.parametrize(
    "child",
    [
        {"action": "key"},
        {"action": "key", "keys": "enter"},
        {"action": "key", "keys": ["enter"], "extra": 1},
    ],
)
def test_prepare_env_tool_calls_leaves_child_arguments_to_the_env(child):
    """Ingress checks child MEMBERSHIP, never child ARGUMENTS.

    Only the concrete env knows whether it can run a canonical action at all
    (mobilegym drops ``pinch``) and what its backend does with the arguments it
    is handed, so it owns both the wording --
    ``invalid arguments for key: ...``, keyed on the CHILD action -- and the
    carrier. Rejecting here answers a capability question with an argument
    question, and does it with the current-observation carrier, which resends an
    unchanged screen for an action that never ran.
    """
    actions, errors = prepare_env_tool_calls(
        [_batch("computer", [child], call_id="call_gui")],
        LiteCUAMetadata(),
    )

    assert errors == {}
    assert [action["name"] for action, _parent in actions] == ["key"]


def test_prepare_env_tool_calls_routes_valid_action_batch_children():
    actions, errors = prepare_env_tool_calls(
        [
            _batch(
                "computer",
                [{"action": "click", "coordinate": [1, 2]}, {"action": "key", "keys": ["a"]}],
                call_id="call_gui",
            )
        ],
        LiteCUAMetadata(),
    )

    assert actions == [
        (_env_action("click", {"coordinate": [1, 2]}), "call_gui"),
        (_env_action("key", {"keys": ["a"]}), "call_gui"),
    ]
    assert errors == {}


def test_prepare_env_tool_calls_checks_valid_actions_before_child_arguments():
    actions, errors = prepare_env_tool_calls(
        [_batch("computer", [{"action": "key"}], call_id="call_gui")],
        LiteCUAMetadata(valid_actions=["click"]),
    )

    # The withheld name is caught BEFORE child arguments are looked at -- the
    # reason names availability, not the missing ``keys``. R4 keeps the slot.
    assert [a["name"] for a, _ in actions] == ["key"]
    assert actions[0][0]["_rejected_reason"] == (
        "invalid action: key; choose an available action for this task"
    )
    assert errors == {}


def test_action_unpack_helpers_are_core_implementations():
    assert unpack_action_batch_call is core_unpack_action_batch_call


def test_prepare_env_tool_calls_rejects_response_without_text():
    metadata = LiteCUAMetadata(extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")])

    actions, errors = prepare_env_tool_calls(
        [_call("response")],
        metadata,
    )

    assert actions == []
    assert errors == {
        "call_0001": current_feedback(
            "invalid arguments for response: response.arguments.text is required"
        )
    }


def test_prepare_env_tool_calls_rejects_terminate_without_status():
    terminate_schema = LiteFinishToolSet.get_tool_schema("terminate")
    assert tool_schema_parameters(terminate_schema)["required"] == ["status"]
    metadata = LiteCUAMetadata(extra_tool_schemas=[terminate_schema])

    actions, errors = prepare_env_tool_calls(
        [_call("terminate")],
        metadata,
    )

    assert actions == []
    assert errors == {
        "call_0001": current_feedback(
            "invalid arguments for terminate: terminate.arguments.status is required"
        )
    }


def test_prepare_env_tool_calls_allows_internal_terminate_marker_with_schema_status():
    metadata = LiteCUAMetadata(extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("terminate")])
    internal = make_internal_terminate_action()

    actions, errors = prepare_env_tool_calls([internal], metadata)

    assert len(actions) == 1
    assert actions[0][0] == {
        "name": "terminate",
        "arguments": {"status": "success", "reason": "REPETITIVE_LOOP"},
        RUNTIME_INTERNAL_STOP_REASON_KEY: "REPETITIVE_LOOP",
    }
    assert actions[0][1] is None
    assert errors == {}


def test_internal_terminate_loop_detect_predicate_only_matches_loop_reason():
    loop = make_internal_terminate_action()
    env_internal = make_internal_terminate_action(
        reason="infeasible",
        internal_reason=ENV_INTERNAL_TERMINATE_REASON,
    )

    assert is_loop_detect_terminate(loop)
    assert not is_loop_detect_terminate(env_internal)


def test_malformed_envelope_without_call_id_fails_loudly():
    with pytest.raises(TypeError, match="canonical Lite tool calls"):
        ordered_tool_call_ids([{"function": {"name": "click", "arguments": {}}}])


def test_prepare_env_tool_calls_reports_malformed_lite_call_before_unknown_tool():
    expected_error = "invalid tool call: tool_call.function.arguments must be an object, got list"

    actions, errors = prepare_env_tool_calls(
        [
            {
                "id": "call_unknown",
                "type": "function",
                "function": {"name": "frobnicate", "arguments": ["bad"]},
            }
        ],
        LiteCUAMetadata(),
    )
    out = build_tool_results_from_decisions(
        LiteEnvStepResult(),
        ordered_call_ids=["call_unknown"],
        images=[b"png"],
        text="current observation",
        feedback=errors,
    )

    assert actions == []
    assert errors == {"call_unknown": current_feedback(expected_error)}
    assert out.results == [
        LiteToolResult(
            tool_call_id="call_unknown",
            images=[b"png"],
            text="current observation",
            metadata={"is_error": True},
            error=expected_error,
        )
    ]
    assert out.results[0].error != unknown_tool_message("frobnicate")


@pytest.mark.parametrize("bad_call_id", [None, "", 123])
def test_ordered_tool_call_ids_rejects_present_bad_call_id(bad_call_id):
    with pytest.raises(ValueError, match="tool call id must be a non-empty string"):
        action = _call("click", {"coordinate": [1, 2]}, call_id=None)
        action["id"] = bad_call_id
        ordered_tool_call_ids([action])


def test_ordered_tool_call_ids_allows_absent_call_id():
    assert ordered_tool_call_ids(
        [
            _call("wait", call_id=None),
        ]
    ) == [None]


def test_public_result_call_id_remap_is_not_accepted_as_call_id():
    with pytest.raises(TypeError, match="_result_call_id is reserved"):
        action = _call("click", {"coordinate": [1, 2]}, call_id=None)
        action[RUNTIME_RESULT_CALL_ID_KEY] = "call_private"
        ordered_tool_call_ids([action])


def test_malformed_result_call_id_envelope_without_name_fails_loudly():
    action = {RUNTIME_RESULT_CALL_ID_KEY: "call_private", "arguments": {}}

    with pytest.raises(TypeError, match="_result_call_id is reserved"):
        ordered_tool_call_ids([action])

    with pytest.raises(TypeError, match="_result_call_id is reserved"):
        prepare_env_tool_calls([action], LiteCUAMetadata())


def test_internal_finish_result_call_id_remap_is_private_but_pairable():
    action = make_internal_terminate_action(result_call_id="call_internal")

    assert ordered_tool_call_ids([action]) == ["call_internal"]
    prepared, feedback = prepare_env_tool_calls([action], LiteCUAMetadata())
    assert feedback == {}
    assert len(prepared) == 1
    prepared_action, result_call_id = prepared[0]
    assert prepared_action == {
        "name": "terminate",
        "arguments": {"status": "success", "reason": "REPETITIVE_LOOP"},
        RUNTIME_INTERNAL_STOP_REASON_KEY: "REPETITIVE_LOOP",
        RUNTIME_RESULT_CALL_ID_KEY: "call_internal",
    }
    assert result_call_id == "call_internal"


@pytest.mark.parametrize("bad_call_id", [None, "", 123])
def test_internal_finish_result_call_id_remap_rejects_bad_ids(bad_call_id):
    action = make_internal_terminate_action()
    action[RUNTIME_RESULT_CALL_ID_KEY] = bad_call_id

    with pytest.raises(ValueError, match="tool call id must be a non-empty string"):
        ordered_tool_call_ids([action])
    with pytest.raises(ValueError, match="tool call id must be a non-empty string"):
        prepare_env_tool_calls([action], LiteCUAMetadata())


@pytest.mark.parametrize("bad_call_id", [None, "", 123])
def test_prepare_env_tool_calls_rejects_present_bad_call_id(bad_call_id):
    with pytest.raises(ValueError, match="tool call id must be a non-empty string"):
        action = _call("click", {"coordinate": [1, 2]}, call_id=None)
        action["id"] = bad_call_id
        prepare_env_tool_calls(
            [action],
            LiteCUAMetadata(),
        )


def test_prepare_env_tool_calls_invalid_valid_action_hides_action_list():
    metadata = LiteCUAMetadata(
        dims=("mobile", "use"),
        valid_actions=["swipe"],
    )

    actions, errors = prepare_env_tool_calls(
        [
            _batch(
                "mobile",
                [{"action": "tap", "coordinate": [10, 20]}],
                call_id="call_mobile",
            )
        ],
        metadata,
    )
    # R4 keeps the slot, so the reason now rides on the action rather than
    # arriving as an ingress error. What must NOT change is the wording: it may
    # never leak the allowed-action list back to the model.
    assert [a["name"] for a, _ in actions] == ["tap"]
    visible_error = actions[0][0]["_rejected_reason"]
    out = build_tool_results_from_decisions(
        LiteEnvStepResult(),
        ordered_call_ids=["call_mobile"],
        images=[b"png"],
        text="current observation",
        feedback={"call_mobile": current_feedback(visible_error)},
    )

    assert out.results[0].images[-1] == b"png"
    assert out.results[0].text == "current observation"
    assert visible_error == ("invalid action: tap; choose an available action for this task")
    assert "valid_actions" not in visible_error
    assert "['swipe']" not in visible_error
    projected = project_tool_result_text(out.results[0].text, visible_error)
    assert projected is not None
    assert "valid_actions" not in projected
    assert "['swipe']" not in projected


def test_prepare_env_tool_calls_invalid_top_level_gui_error_hides_allowed_action_list():
    metadata = LiteCUAMetadata(
        dims=("browser", "use"),
        valid_actions=["click"],
    )

    actions, errors = prepare_env_tool_calls(
        [_call("tap", {"coordinate": [10, 20]}, call_id="call_tap")],
        metadata,
        validate_top_level_action=True,
    )
    out = build_tool_results_from_decisions(
        LiteEnvStepResult(),
        ordered_call_ids=["call_tap"],
        feedback=errors,
    )

    assert actions == []
    visible_error = out.results[0].error
    assert visible_error == ("invalid action: tap; choose an available action for this task")
    assert "expected one of" not in visible_error
    assert "[" not in visible_error
    assert "click" not in visible_error


def test_ambiguous_argument_provenance_uses_general_error_not_backend_payload():
    """No visible action provenance means no guessed canonical/backend wording."""
    backend_payload = {
        "canonical_action": "click",
        "normalized_arguments": {"coordinate": [960, 540]},
        "backend_action": "left_click",
        "backend_arguments": {"x": 960, "y": 540},
    }
    errors: dict[str, ToolErrorFeedback] = {}
    record_model_action_error(
        errors,
        "call_ambiguous",
        ValueError("the previous action had invalid arguments"),
    )

    out = build_tool_results_from_decisions(
        LiteEnvStepResult(info={"debug": {"backend_payload": backend_payload}}),
        ordered_call_ids=["call_ambiguous"],
        feedback=errors,
        images=[b"png"],
        text="current observation",
        metadata={"debug": {"backend_payload": backend_payload}},
    )

    assert len(out.results) == 1
    tool_result = out.results[0]
    assert tool_result.error == (
        "invalid action arguments: the previous action had invalid arguments"
    )
    for backend_detail in ("click", "left_click", "coordinate", "x", "960", "540"):
        assert backend_detail not in tool_result.error
    assert tool_result.metadata == {
        "debug": {"backend_payload": backend_payload},
        "is_error": True,
    }
    assert out.info == {"debug": {"backend_payload": backend_payload}}


def test_build_tool_results_from_decisions_rejects_bad_decision_keys():
    result = LiteEnvStepResult()

    for bad_call_id in (None, "", 123):
        if bad_call_id is None:
            continue
        with pytest.raises(ValueError, match="tool call id must be a non-empty string"):
            build_tool_results_from_decisions(
                LiteEnvStepResult(),
                ordered_call_ids=[bad_call_id],  # type: ignore[list-item]
            )
    with pytest.raises(ValueError, match="duplicate tool call ids"):
        build_tool_results_from_decisions(result, ordered_call_ids=["call_0", "call_0"])
    with pytest.raises(ValueError, match="feedback for unknown"):
        build_tool_results_from_decisions(
            LiteEnvStepResult(),
            ordered_call_ids=["call_0"],
            feedback={"call_1": error_only_feedback("bad")},
        )
    with pytest.raises(ValueError, match="explicit results for unknown"):
        build_tool_results_from_decisions(
            LiteEnvStepResult(),
            ordered_call_ids=["call_0"],
            explicit_results={"call_1": LiteToolResult(tool_call_id="call_1", text="x")},
        )
    with pytest.raises(ValueError, match="conflicting feedback and explicit results"):
        build_tool_results_from_decisions(
            LiteEnvStepResult(),
            ordered_call_ids=["call_0"],
            feedback={"call_0": error_only_feedback("bad")},
            explicit_results={"call_0": LiteToolResult(tool_call_id="call_0", text="x")},
        )
    with pytest.raises(ValueError, match="continue results for unknown"):
        build_tool_results_from_decisions(
            LiteEnvStepResult(),
            ordered_call_ids=["call_0"],
            continue_call_ids={"call_1"},
        )
    with pytest.raises(ValueError, match="tool call id must be a non-empty string"):
        build_tool_results_from_decisions(
            LiteEnvStepResult(),
            ordered_call_ids=["call_0"],
            continue_call_ids={123},  # type: ignore[arg-type]
        )


def test_build_tool_results_from_decisions_rejects_explicit_call_id_mismatch():
    with pytest.raises(ValueError, match="explicit result tool_call_id mismatch"):
        build_tool_results_from_decisions(
            LiteEnvStepResult(),
            ordered_call_ids=["call_0"],
            explicit_results={"call_0": LiteToolResult(tool_call_id="call_1", text="x")},
        )


def test_build_tool_results_from_decisions_validates_decisions_before_prefilled_return():
    with pytest.raises(ValueError, match="feedback for unknown"):
        build_tool_results_from_decisions(
            LiteEnvStepResult(results=[LiteToolResult(tool_call_id="call_0", text="already")]),
            ordered_call_ids=["call_0"],
            feedback={"call_1": error_only_feedback("bad")},
        )


def test_build_tool_results_from_decisions_rejects_mixed_prefilled_decisions():
    prefilled = LiteEnvStepResult(results=[LiteToolResult(tool_call_id="call_0", text="already")])

    with pytest.raises(ValueError, match="prebuilt tool results cannot be combined"):
        build_tool_results_from_decisions(
            prefilled,
            ordered_call_ids=["call_0"],
            feedback={"call_0": error_only_feedback("bad")},
        )
    with pytest.raises(ValueError, match="prebuilt tool results cannot be combined"):
        build_tool_results_from_decisions(
            prefilled,
            ordered_call_ids=["call_0"],
            continue_call_ids={"call_0"},
        )
    with pytest.raises(ValueError, match="prebuilt tool results cannot be combined"):
        build_tool_results_from_decisions(
            prefilled,
            ordered_call_ids=["call_0"],
            explicit_results={"call_0": LiteToolResult(tool_call_id="call_0", text="new")},
        )


def test_build_tool_results_from_decisions_supports_unpaired_current_result():
    out = build_tool_results_from_decisions(
        LiteEnvStepResult(),
        ordered_call_ids=[None],
        continue_call_ids={None},
        images=[b"png"],
        text="obs",
        metadata={"source": "env"},
    )

    assert out.results == [
        LiteToolResult(
            tool_call_id=None,
            images=[b"png"],
            text="obs",
            metadata={"source": "env"},
        )
    ]


@pytest.mark.parametrize(
    "result",
    [LiteEnvStepResult(terminated=True), LiteEnvStepResult(truncated=True)],
    ids=["terminated", "truncated"],
)
def test_build_tool_results_from_decisions_preserves_terminal_current_by_call_id(result):
    out = build_tool_results_from_decisions(
        result,
        ordered_call_ids=["call_current"],
        continue_call_ids={"call_current"},
        images=[b"png"],
        text="final observation",
        metadata={"source": "env"},
    )

    assert out.results == [
        LiteToolResult(
            tool_call_id="call_current",
            images=[b"png"],
            text="final observation",
            metadata={"source": "env"},
        )
    ]


def test_build_tool_results_from_decisions_current_payload_is_per_call():
    out = build_tool_results_from_decisions(
        LiteEnvStepResult(),
        ordered_call_ids=["call_screen", "call_text"],
        continue_call_ids={"call_screen"},
        explicit_results={
            "call_text": LiteToolResult(tool_call_id="call_text", text="bash output")
        },
        images=[b"png"],
        text="current observation",
        metadata={"source": "env"},
    )

    assert out.results == [
        LiteToolResult(
            tool_call_id="call_screen",
            images=[b"png"],
            text="current observation",
            metadata={"source": "env"},
        ),
        LiteToolResult(tool_call_id="call_text", text="bash output"),
    ]


def test_build_tool_results_from_decisions_keeps_full_slate_after_later_error():
    out = build_tool_results_from_decisions(
        LiteEnvStepResult(),
        ordered_call_ids=["call_screen", "call_error"],
        continue_call_ids={"call_screen"},
        images=[b"png"],
        text="current observation",
        feedback={"call_error": error_only_feedback("tool call failed")},
    )

    assert out.results == [
        LiteToolResult(
            tool_call_id="call_screen",
            images=[b"png"],
            text="current observation",
        ),
        LiteToolResult(
            tool_call_id="call_error",
            metadata={"is_error": True},
            error="tool call failed",
        ),
    ]


def test_build_tool_results_from_decisions_accounts_mixed_decisions_exactly_once():
    out = build_tool_results_from_decisions(
        LiteEnvStepResult(truncated=True),
        ordered_call_ids=[
            "call_active",
            "call_inactive",
            "call_unsupported",
            "call_malformed",
            "call_execution",
            "call_unknown",
        ],
        continue_call_ids={"call_active"},
        images=[b"png"],
        text="current observation",
        feedback={
            "call_inactive": current_feedback(unavailable_action_message("open_app")),
            "call_unsupported": current_feedback(unsupported_action_message("pinch")),
            "call_malformed": current_feedback(
                invalid_action_arguments_message(
                    "click",
                    "coordinate must be [x, y]",
                )
            ),
            "call_execution": current_feedback(
                tool_execution_error_message(
                    "click",
                    "target coordinate is outside the screen",
                )
            ),
            "call_unknown": error_only_feedback(unknown_tool_message("frobnicate")),
        },
    )

    assert [result.tool_call_id for result in out.results] == [
        "call_active",
        "call_inactive",
        "call_unsupported",
        "call_malformed",
        "call_execution",
        "call_unknown",
    ]
    assert len(out.results) == len({result.tool_call_id for result in out.results})
    assert out.results[0].error is None
    assert out.results[0].images[-1] == b"png"
    assert out.results[1].error == "open_app is not available in this task."
    assert out.results[1].images[-1] == b"png"
    assert out.results[5].error == "unknown tool: frobnicate"
    assert out.results[5].images == []


def test_build_tool_results_from_decisions_suppresses_unpaired_terminal_current_result():
    out = build_tool_results_from_decisions(
        LiteEnvStepResult(terminated=True),
        ordered_call_ids=[None],
        continue_call_ids={None},
        images=[b"png"],
        text="obs",
    )

    assert out.results == []


# ---------------------------------------------------------------------------
# unpack_action_batch_call totality over model-emittable envelopes.
#
# An adapter passes the model's tool-call JSON through, so `arguments.actions`
# and its children are raw model output. `unpack_action_batch_call` runs in places that
# are NOT inside any per-env `try`: above the env (`LoopDetectWrapper`) and
# outside `env.step` entirely (`TrajectoryLogger` unpacks the RAW model tool
# calls in `on_step`). A raise there kills the
# rollout, so the function must be total: malformed -> no actions.
# ---------------------------------------------------------------------------

MALFORMED_ACTION_BATCHES: list[tuple[str, dict]] = [
    ("actions-missing", {}),
    ("actions-none", {"actions": None}),
    ("actions-str", {"actions": "click"}),
    ("actions-dict", {"actions": {"action": "click"}}),
    ("actions-int", {"actions": 3}),
    ("child-missing-action", {"actions": [{"coordinate": [1, 2]}]}),
    ("child-not-a-dict", {"actions": ["click"]}),
    ("child-action-not-a-str", {"actions": [{"action": 5}]}),
]

MALFORMED_RAW_ACTIONS: list[tuple[str, Any]] = [
    (
        "noncanonical-tool-call",
        {
            "id": "provider_1",
            "type": "function",
            "function": {"name": "computer", "arguments": "{}"},
        },
    ),
    (
        "missing-arguments",
        {"id": "call_0000", "type": "function", "function": {"name": "click"}},
    ),
    (
        "empty-name",
        {
            "id": "call_0000",
            "type": "function",
            "function": {"name": "", "arguments": {}},
        },
    ),
    ("non-object", "raw click"),
]


@pytest.mark.parametrize(
    "arguments",
    [pytest.param(args, id=label) for label, args in MALFORMED_ACTION_BATCHES],
)
def test_unpack_action_batch_call_is_total_over_malformed_action_batches(arguments):
    action = _call_with_raw_arguments("computer", arguments, call_id="call_0000")

    assert unpack_action_batch_call(action) == []
    assert action_inspection_records([action]) == []


@pytest.mark.parametrize(
    "action",
    [pytest.param(action, id=label) for label, action in MALFORMED_RAW_ACTIONS],
)
def test_trace_unpack_helpers_skip_noncanonical_raw_actions(action):
    """Trace/log helpers are best-effort even when raw actions are not Lite calls."""
    assert action_inspection_records([action]) == []


def test_trace_unpack_helpers_skip_public_reserved_result_call_id():
    """Runtime-private result routing keys are not public canonical shape."""
    action = _call("click", {"coordinate": [1, 2]}, call_id="call_0000")
    action[RUNTIME_RESULT_CALL_ID_KEY] = "call_private"

    assert action_inspection_records([action]) == []


def test_unpack_action_batch_call_still_expands_a_canonical_action_batch():
    """Totality must not swallow well-formed work."""
    action = _batch(
        "computer",
        [{"action": "click", "coordinate": [10, 20]}],
        call_id="call_0000",
    )

    assert unpack_action_batch_call(action) == [
        {"name": "click", "arguments": {"coordinate": [10, 20]}}
    ]


def test_unpack_action_batch_call_still_rejects_provider_envelopes():
    """Infra faults stay fail-loud: only MODEL shapes are total."""
    with pytest.raises(TypeError):
        unpack_action_batch_call({"name": "click"})
    with pytest.raises(TypeError):
        unpack_action_batch_call({"function": {"name": "click", "arguments": {}}})


def test_trace_unpack_helpers_reject_a_mixed_envelope_structurally():
    """A mixed raw payload must be rejected before action-batch unpacking.

    The trace helpers must SKIP it on the envelope check -- never reach
    ``unpack_action_batch_call`` and swallow its ``TypeError``, which would make
    the action vanish from the trace under this repo's own "not provider
    payloads" error.
    """
    action = {
        "id": "call_0000",
        "function": {"name": "computer", "arguments": "{}"},
        "name": "computer",
        "arguments": {"actions": [{"action": "click", "coordinate": [1, 2]}]},
    }

    with pytest.raises(TypeError, match="canonical Lite tool calls"):
        unpack_action_batch_call(action)
    assert action_inspection_records([action]) == []


class _StubEnv(LiteBaseEnv):
    """Records the actions it received (post-wrapper). Idiom from test_wrappers.py."""

    def __init__(self):
        self.stepped: list[list[LiteToolCall]] = []

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(dims=("desktop", "use"))

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=None, text="reset")

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.stepped.append(list(actions))
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


@pytest.mark.parametrize(
    "arguments",
    [pytest.param(args, id=label) for label, args in MALFORMED_ACTION_BATCHES],
)
def test_wrappers_above_the_env_survive_a_malformed_action_batch(arguments):
    """``LoopDetectWrapper`` unpacks raw model calls.

    It sits above the env, so no per-env ``try`` can reach a raise from its
    ``unpack_action_batch_call``. Env-owned malformed-envelope feedback is tested in the
    concrete env/helper suites; this test only pins wrapper tolerance.
    """
    from lite.gym.wrappers import LoopDetectWrapper

    action = _call_with_raw_arguments("computer", arguments, call_id="call_0000")

    inner = _StubEnv()
    wrapped = LoopDetectWrapper(inner, loop_threshold=3)
    asyncio.run(wrapped.reset())
    asyncio.run(wrapped.step([action]))
    assert inner.stepped == [[action]]


@pytest.mark.parametrize(
    "action",
    [pytest.param(action, id=label) for label, action in MALFORMED_RAW_ACTIONS],
)
def test_loopdetect_forwards_noncanonical_raw_actions_to_env_ingress(action):
    """Loop detection is observational; env ingress owns malformed-call feedback."""
    from lite.gym.wrappers import LoopDetectWrapper

    inner = _StubEnv()
    wrapped = LoopDetectWrapper(inner, loop_threshold=3)
    asyncio.run(wrapped.reset())
    asyncio.run(wrapped.step([action]))
    assert inner.stepped == [[action]]


@pytest.mark.parametrize(
    "arguments",
    [pytest.param(args, id=label) for label, args in MALFORMED_ACTION_BATCHES],
)
def test_trajectory_logger_survives_a_malformed_action_batch(arguments, tmp_path):
    """``TrajectoryLogger.on_step`` unpacks the RAW model tool calls.

    It runs after ``env.step`` and still receives the original model-emitted
    envelope from ``predict_result.lite_message["tool_calls"]``. A bad envelope
    must not turn log serialization into the rollout failure.
    """
    from lite.agents.core.agent.hooks import SampleStepData
    from lite.agents.core.agent.logger import TrajectoryLogger

    hook = TrajectoryLogger(tmp_path, save_data=False)
    hook.on_step(
        SampleStepData(
            step_idx=0,
            image=None,
            predict_result=None,
            step_result=LiteEnvStepResult(),
            actions=[_call_with_raw_arguments("computer", arguments, call_id="call_0000")],
        )
    )


@pytest.mark.parametrize(
    "action",
    [pytest.param(action, id=label) for label, action in MALFORMED_RAW_ACTIONS],
)
def test_trajectory_logger_survives_noncanonical_raw_actions(action, tmp_path):
    """Logger action inspection is best-effort after env classification."""
    from lite.agents.core.agent.hooks import SampleStepData
    from lite.agents.core.agent.logger import TrajectoryLogger

    hook = TrajectoryLogger(tmp_path, save_data=False)
    hook.on_step(
        SampleStepData(
            step_idx=0,
            image=None,
            predict_result=None,
            step_result=LiteEnvStepResult(),
            actions=[action],
        )
    )


def test_content_only_final_normalizes_executed_actions_to_lite_action():
    final = NoToolCallFinal(
        message={"role": "assistant", "content": []},
        actions=[make_tool_call("response", {"text": "Done."})],
        stop_reason="content_only_final",
    )
    result = LiteEnvStepResult(
        info={"executed_actions": [{"call": "ANSWER", "args": {"text": "Done."}}]},
    )

    out = mark_no_tool_call_final_result(result, final)

    assert out.terminated is False
    assert out.truncated is False
    assert "stop_reason" not in out.info
    assert out.info["executed_actions"] == [
        {"call": "response", "args": {"text": "Done."}},
    ]


def test_parse_failure_final_marks_terminal_stop_reason():
    final = NoToolCallFinal(
        message={"role": "assistant", "content": []},
        actions=[make_tool_call("response", {"text": "bad json"})],
        stop_reason="parse_failure",
    )
    result = LiteEnvStepResult(
        info={"stop_reason": "env-owned reason"},
        terminated=False,
        truncated=True,
    )

    out = mark_no_tool_call_final_result(result, final)

    assert out.terminated is True
    assert out.truncated is False
    assert out.info["stop_reason"] == "parse_failure"
    assert out.info["executed_actions"] == [
        {"call": "response", "args": {"text": "bad json"}},
    ]
