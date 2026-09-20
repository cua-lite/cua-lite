"""
Tests for STEPGUIMobileActionSpace.

Covers:
  1. Registry + single mobile_use schema + 9-action enum
  2. Tool call creation (mobile_use)
  3. Forward conversion: cua-lite -> Step-GUI
  4. Reverse conversion: Step-GUI -> cua-lite (incl. `summary` / `_step_gui_point` preservation)
  5. Coordinate pass-through ([0, 1000] on both sides, no rescale)
  6. Round-trip
  7. The prompt action rows and the mobile_use action enum read the same gates

Run:
    uv run pytest tests/agents/models/step_gui/test_step_gui_action_space.py -v
"""

from __future__ import annotations

import dataclasses
import itertools
import re

import pytest
from agents._support.valid_actions_gating import (
    OPEN_APP_SCHEMA,
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    agent_adapter_for,
)
from lite_samples import sample_trajectory_two_turns

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import ActionSpaceRegistry, LiteMobileActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.step_gui.action_space import STEPGUIMobileActionSpace
from lite.core import LiteCUAMetadata
from lite.core.tools.action_space import (
    lite_action_batch_child_name_errors,
    validate_lite_action_batch_structure,
)
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet, make_open_app_tool
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

# ``AgentAdapterRegistry.get("step_gui@mobile@use")`` needs the step_gui ADAPTER
# imported, which importing the action space alone does not do. Without this the
# module only passes when some other test module in the same xdist worker
# happens to have registered first.
register_all()

#: The three extra tools every shipped Step-GUI config advertises
#: (``scripts/configs/step_gui/default/*.yaml``), and the ones whose action
#: values used to disappear from the prompt when they were absent.
_EXTRA_TOOL_SCHEMAS = (
    make_open_app_tool(["Settings", "微信"]),
    LiteFinishToolSet.get_tool_schema("response"),
    LiteFinishToolSet.get_tool_schema("terminate"),
)


@pytest.mark.parametrize(
    "call,schema",
    [
        (make_tool_call("response", {"text": "42"}), RESPONSE_SCHEMA),
        (make_tool_call("terminate", {"status": "success"}), TERMINATE_SCHEMA),
    ],
)
def test_step_gui_finish_render_delegates_to_family_converter(call, schema) -> None:
    message = {"role": "assistant", "content": [], "tool_calls": [call]}

    adapter = agent_adapter_for("step_gui@mobile@use", "mobile")
    adapter.convert_message_to_agent(message)

    adapter = agent_adapter_for(
        "step_gui@mobile@use",
        "mobile",
        extra_tool_schemas=[schema],
    )
    adapter.convert_message_to_agent(message)


def test_step_gui_open_app_render_does_not_require_matching_extra_schema() -> None:
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [make_tool_call("open_app", {"app_name": "Settings"})],
    }

    adapter = agent_adapter_for("step_gui@mobile@use", "mobile")
    adapter.convert_message_to_agent(message)

    adapter = agent_adapter_for(
        "step_gui@mobile@use",
        "mobile",
        extra_tool_schemas=[OPEN_APP_SCHEMA],
    )
    adapter.convert_message_to_agent(message)


def _prompt_only_sample():
    """First user turn only — enough to render the Step-GUI ``use`` prompt."""
    sample = sample_trajectory_two_turns()
    return dataclasses.replace(
        sample,
        messages=[sample.messages[0]],
        metadata=dataclasses.replace(
            sample.metadata,
            dims=("mobile", "use"),
        ),
    )


def _only_mobile_action(tool_calls):
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert tool_call_name(call) == "mobile"
    actions = tool_call_arguments(call)["actions"]
    assert len(actions) == 1
    return actions[0]


# =============================================================================
# 1. Registry + schema
# =============================================================================


class TestRegistryAndSchema:
    def test_registry_key(self):
        space = ActionSpaceRegistry.get("step_gui@mobile")
        assert isinstance(space, STEPGUIMobileActionSpace)

    def test_single_tool(self):
        schemas = STEPGUIMobileActionSpace.get_tool_schemas()
        assert len(schemas) == 1
        assert tool_schema_name(schemas[0]) == "mobile_use"

    def test_action_enum_has_exact_9_actions(self):
        schema = STEPGUIMobileActionSpace.get_tool_schemas()[0]
        actions = set(tool_schema_parameters(schema)["properties"]["action"]["enum"])
        expected = {
            "CLICK",
            "LONGPRESS",
            "TYPE",
            "SLIDE",
            "AWAKE",
            "WAIT",
            "INFO",
            "COMPLETE",
            "ABORT",
        }
        assert actions == expected

    def test_action_enum_is_uppercase(self):
        """Step-GUI SFT distribution uses uppercase action names."""
        schema = STEPGUIMobileActionSpace.get_tool_schemas()[0]
        for name in tool_schema_parameters(schema)["properties"]["action"]["enum"]:
            assert name == name.upper()

    def test_info_schema_describes_final_answer_not_ask_user(self):
        schema = STEPGUIMobileActionSpace.get_tool_schemas()[0]
        params = tool_schema_parameters(schema)
        action_description = params["properties"]["action"]["description"]
        value_description = params["properties"]["value"]["description"]
        assert "最终回答" in action_description
        assert "final answer for INFO" in value_description


# =============================================================================
# 2. Tool call creation
# =============================================================================


class TestToolCalls:
    def test_click(self):
        tc = STEPGUIMobileActionSpace.mobile_use(action="CLICK", point=[500, 300])
        assert tool_call_name(tc) == "mobile_use"
        assert tool_call_arguments(tc) == {"action": "CLICK", "point": [500, 300]}

    def test_long_press(self):
        tc = STEPGUIMobileActionSpace.mobile_use(action="LONGPRESS", point=[100, 200])
        assert tool_call_arguments(tc) == {"action": "LONGPRESS", "point": [100, 200]}

    def test_type_with_point(self):
        tc = STEPGUIMobileActionSpace.mobile_use(
            action="TYPE",
            value="hello",
            point=[500, 300],
        )
        args = tool_call_arguments(tc)
        assert args["action"] == "TYPE"
        assert args["value"] == "hello"
        assert args["point"] == [500, 300]

    def test_slide_uses_point1_and_point2(self):
        tc = STEPGUIMobileActionSpace.mobile_use(
            action="SLIDE",
            point1=[500, 700],
            point2=[500, 200],
        )
        args = tool_call_arguments(tc)
        assert args["point1"] == [500, 700]
        assert args["point2"] == [500, 200]

    def test_complete_uses_return_value(self):
        tc = STEPGUIMobileActionSpace.mobile_use(action="COMPLETE", return_value="done")
        args = tool_call_arguments(tc)
        assert args == {"action": "COMPLETE", "return_value": "done"}

    def test_none_args_dropped(self):
        tc = STEPGUIMobileActionSpace.mobile_use(action="WAIT", value="5")
        assert tool_call_arguments(tc) == {"action": "WAIT", "value": "5"}


# =============================================================================
# 3. Forward conversion: cua-lite -> Step-GUI
# =============================================================================


class TestForwardConversion:
    def setup_method(self):
        self.space = STEPGUIMobileActionSpace()

    def test_tap_to_click(self):
        tc = [LiteMobileActionSpace.tap(coordinate=[500, 300])]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert set(result[0]) == {"name", "arguments"}
        assert result[0]["name"] == "mobile_use"
        args = result[0]["arguments"]
        assert args["action"] == "CLICK"
        assert args["point"] == [500, 300]  # coordinate pass-through, no rescale

    def test_canonical_nested_to_agent_projects_bare_mobile_use(self):
        tc = [make_tool_call("tap", {"coordinate": [500, 300]})]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result == [
            {
                "name": "mobile_use",
                "arguments": {"action": "CLICK", "point": [500, 300]},
            }
        ]

    def test_mobile_batch_to_agent_unwraps(self):
        tc = make_tool_call(
            "mobile",
            {
                "actions": [
                    {"action": "tap", "coordinate": [500, 300]},
                    {"action": "type", "text": "hello"},
                ]
            },
        )
        result = self.space.convert_tool_calls_to_agent([tc])
        assert [r["arguments"]["action"] for r in result] == ["CLICK", "TYPE"]
        assert result[1]["arguments"]["value"] == "hello"

    def test_long_press_to_longpress(self):
        tc = [LiteMobileActionSpace.long_press(coordinate=[100, 200], duration=2.0)]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "LONGPRESS"
        assert args["point"] == [100, 200]

    def test_type_to_type(self):
        tc = [LiteMobileActionSpace.type(text="hello")]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "TYPE"
        assert args["value"] == "hello"

    def test_swipe_to_slide(self):
        tc = [
            LiteMobileActionSpace.swipe(
                start_coordinate=[500, 800],
                coordinate=[500, 200],
            )
        ]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "SLIDE"
        assert args["point1"] == [500, 800]
        assert args["point2"] == [500, 200]

    def test_open_app_to_awake(self):
        tc = [make_tool_call("open_app", {"app_name": "微信"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "AWAKE"
        assert args["value"] == "微信"

    def test_response_to_info(self):
        """`response` is terminal answer text. Mapped to INFO on the Step-GUI side."""
        tc = [make_tool_call("response", {"text": "结果是42"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "INFO"
        assert args["value"] == "结果是42"

    def test_wait_to_wait(self):
        tc = [LiteMobileActionSpace.wait(duration=3.5)]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "WAIT"
        assert args["value"] == "3.5"

    def test_terminate_success_to_complete(self):
        tc = [make_tool_call("terminate", {"status": "success", "reason": "done"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "COMPLETE"
        assert args["return_value"] == "done"

    def test_terminate_failure_to_abort_drops_value(self):
        """Reference `action2action` emits nothing beyond base for ABORT
        (parser_0920_summary.py:188 `pass`). `value` is discarded — we match."""
        tc = [make_tool_call("terminate", {"status": "failure", "reason": "blocked"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "ABORT"
        assert "value" not in args

    def test_type_drops_point(self):
        """Reference `action2action` comments out `point` for TYPE
        (parser_0920_summary.py:142). The SFT distribution has only
        `action:TYPE\\tvalue:...`, never a trailing `point:`."""
        tc = [LiteMobileActionSpace.type(text="hello")]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "TYPE"
        assert args["value"] == "hello"
        assert "point" not in args

    def test_unsupported_extra_to_agent_is_dropped(self):
        tc = make_tool_call("search_web", {"query": "weather"})
        assert self.space.convert_tool_calls_to_agent([tc]) == []


# =============================================================================
# 4. Reverse conversion: Step-GUI -> cua-lite
# =============================================================================


class TestReverseConversion:
    def setup_method(self):
        self.space = STEPGUIMobileActionSpace()

    def test_click_from_agent(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "CLICK", "point": [500, 300]},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_mobile_action(result) == {
            "action": "tap",
            "coordinate": [500, 300],
            "clicks": 1,
        }

    def test_longpress_from_agent(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "LONGPRESS", "point": [100, 200]},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_mobile_action(result) == {
            "action": "long_press",
            "coordinate": [100, 200],
        }

    def test_slide_from_agent(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "SLIDE",
                    "point1": [500, 700],
                    "point2": [500, 200],
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        args = _only_mobile_action(result)
        assert args["action"] == "swipe"
        assert args["start_coordinate"] == [500, 700]
        assert args["coordinate"] == [500, 200]

    def test_awake_from_agent(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "AWAKE", "value": "微信"},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "open_app"
        assert tool_call_arguments(result[0])["app_name"] == "微信"

    def test_info_from_agent_becomes_final_response(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "INFO", "value": "结果是42"},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert result == [LiteFinishToolSet.response(text="结果是42")]

    def test_wait_from_agent_parses_numeric_value(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "WAIT", "value": "3.5"},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_mobile_action(result) == {"action": "wait", "duration": 3.5}

    def test_wait_from_agent_non_numeric_falls_back_to_1(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "WAIT", "value": "一会儿"},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_mobile_action(result)["duration"] == 1.0

    def test_complete_from_agent_becomes_success_terminate(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "COMPLETE", "return_value": "done"},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert result == [LiteFinishToolSet.terminate(status="success", reason="done")]

    def test_abort_from_agent_becomes_failure_terminate(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "ABORT", "value": "blocked"},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert result == [LiteFinishToolSet.terminate(status="failure", reason="blocked")]

    def test_type_from_agent_drops_point(self):
        """Reference `action2action` discards TYPE's `point`
        (parser_0920_summary.py:142). Matching the reference, cua-lite does not
        preserve it on the reverse edge either — point is lost on this boundary."""
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "TYPE", "value": "hello", "point": [500, 300]},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        args = _only_mobile_action(result)
        assert args["action"] == "type"
        assert args["text"] == "hello"
        assert "point" not in args
        assert "_step_gui_point" not in args

    def test_summary_in_args_is_dropped_at_action_space_boundary(self):
        """``summary`` is trajectory metadata, not an action parameter — the
        action space no longer propagates it through ``arguments``. The
        canonical home is :class:`HistorySummaryContent` on the assistant
        message; ``STEPGUIMobileBaseAdapter.parse_raw_assistant_response``
        owns that mapping."""
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "CLICK", "point": [500, 300], "summary": "已点击"},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert "summary" not in _only_mobile_action(result)

    def test_unknown_wrapped_action_becomes_invalid_action_batch(self):
        """Unknown wrapper actions must reach env ingress as model-visible errors."""
        tc = [
            {
                "name": "mobile_use",
                "arguments": {"action": "UNSUPPORTED", "value": "x"},
            }
        ]
        out = self.space.convert_tool_calls_from_agent(tc)

        assert len(out) == 1
        assert tool_call_name(out[0]) == "mobile"
        assert tool_call_arguments(out[0]) == {
            "actions": [{"action": "UNSUPPORTED", "value": "x"}],
        }
        children, error = validate_lite_action_batch_structure(
            "mobile",
            tool_call_arguments(out[0]),
        )
        assert error is None
        assert len(children) == 1
        error = lite_action_batch_child_name_errors("mobile", children).get(0)
        assert error is not None
        assert error.child_action_name == "UNSUPPORTED"


# =============================================================================
# 5. Coordinate pass-through ([0, 1000] identical to cua-lite)
# =============================================================================


class TestCoordinatePassThrough:
    @pytest.mark.parametrize("xy", [[0, 0], [500, 500], [1000, 1000], [123, 456]])
    def test_no_rescale_forward(self, xy):
        space = STEPGUIMobileActionSpace()
        tc = [LiteMobileActionSpace.tap(coordinate=xy)]
        result = space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["point"] == xy

    @pytest.mark.parametrize("xy", [[0, 0], [500, 500], [1000, 1000], [123, 456]])
    def test_no_rescale_reverse(self, xy):
        space = STEPGUIMobileActionSpace()
        tc = [{"name": "mobile_use", "arguments": {"action": "CLICK", "point": xy}}]
        result = space.convert_tool_calls_from_agent(tc)
        assert _only_mobile_action(result)["coordinate"] == xy


# =============================================================================
# 6. Round-trip
# =============================================================================


class TestRoundTrip:
    @pytest.mark.parametrize(
        "cua_tc",
        [
            LiteMobileActionSpace.tap(coordinate=[100, 100]),
            LiteMobileActionSpace.tap(coordinate=[1000, 1000]),
            LiteMobileActionSpace.long_press(coordinate=[500, 500], duration=2.0),
            LiteMobileActionSpace.swipe(start_coordinate=[200, 800], coordinate=[200, 200]),
            make_tool_call("open_app", {"app_name": "Settings"}),
            make_tool_call("response", {"text": "answer"}),
            LiteMobileActionSpace.wait(duration=1.5),
            make_tool_call("terminate", {"status": "success"}),
            make_tool_call("terminate", {"status": "failure"}),
        ],
    )
    def test_round_trip_preserves_action_name(self, cua_tc):
        """cua-lite → step_gui → cua-lite should preserve the action name.

        Lossy fields (coordinate drift, etc.) are checked case-by-case elsewhere;
        here we only verify the function name survives the round-trip.
        """
        space = STEPGUIMobileActionSpace()
        agent_tcs = space.convert_tool_calls_to_agent([cua_tc])
        restored = space.convert_tool_calls_from_agent(agent_tcs)
        assert tool_call_name(restored[0]) == tool_call_name(cua_tc)

    def test_type_round_trip_drops_point(self):
        """TYPE's `point` is intentionally lost on round-trip to match reference
        `action2action` behavior — the reference serializer strips it, so the
        SFT distribution never carried it."""
        space = STEPGUIMobileActionSpace()
        agent_tc = {
            "name": "mobile_use",
            "arguments": {"action": "TYPE", "value": "hi", "point": [400, 600]},
        }
        lite_tc = space.convert_tool_calls_from_agent([agent_tc])[0]
        back = space.convert_tool_calls_to_agent([lite_tc])[0]
        assert "point" not in back["arguments"]
        assert back["arguments"]["value"] == "hi"

    def test_summary_in_lite_args_is_not_re_emitted_by_action_space(self):
        """``summary`` lives on the assistant message (``HistorySummaryContent``),
        not in the tool_call's ``arguments``. The action space treats any
        leftover args-side ``summary`` as an unknown kwarg and drops it.
        ``STEPGUIMobileBaseAdapter.convert_message_to_agent`` owns the canonical
        re-injection via
        ``format_agent_tool_call_as_wire_text(summary=...)``."""
        lite_tc = make_tool_call(
            "tap",
            {"coordinate": [500, 300], "summary": "已点击确认"},
        )
        space = STEPGUIMobileActionSpace()
        agent_tc = space.convert_tool_calls_to_agent([lite_tc])[0]
        assert "summary" not in agent_tc["arguments"]


# =============================================================================
# 7. The prompt action rows are trained text; the mobile_use enum is composed
# =============================================================================


class TestActionValueGates:
    """Step-GUI bakes its action space into the SFT-trained Chinese system
    prompt and renders no ``<tools>`` section, so those numbered rows ARE its
    model-facing tool surface — and neither gate may edit them. Both gates,
    ``metadata.valid_actions`` (GUI actions) and the active extra tools derived
    from ``metadata.extra_tool_schemas`` (``AWAKE``/``INFO``/``COMPLETE``/
    ``ABORT``), narrow the ``mobile_use`` SCHEMA only, which is composed per
    request and reaches no model on this family's wire."""

    KEY = "step_gui@mobile@use"

    @staticmethod
    def _system_prompt_text(adapter, sample) -> str:
        msgs = adapter.unroll(sample).steps[-1]
        assert msgs[0]["role"] == "system"
        return msgs[0]["content"][0]["text"]

    @staticmethod
    def _action_enum(schemas) -> list[str]:
        assert len(schemas) == 1 and tool_schema_name(schemas[0]) == "mobile_use"
        return tool_schema_parameters(schemas[0])["properties"]["action"]["enum"]

    @classmethod
    def _prompt_action_values(cls, valid_actions) -> set[str]:
        text = cls._system_prompt_text(
            AgentAdapterRegistry.get(
                cls.KEY,
                metadata=LiteCUAMetadata(dims=("mobile", "use"), valid_actions=valid_actions),
            ),
            _prompt_only_sample(),
        )
        return set(re.findall(r"^\d+\. ([A-Z]+)：", text, flags=re.M))

    #: Every numbered row of the trained prompt, GUI verbs and the four values
    #: that spell a standalone extra tool alike. No gate may delete one.
    ALL_PROMPT_ROWS = {
        "CLICK",
        "TYPE",
        "COMPLETE",
        "WAIT",
        "AWAKE",
        "INFO",
        "ABORT",
        "SLIDE",
        "LONGPRESS",
    }

    def test_valid_actions_does_not_reach_the_prompt_rows(self):
        """The GUI gate narrows the ``mobile_use`` SCHEMA, never the prompt.

        The behaviour this replaces rendered only ``AWAKE``/``INFO``/``COMPLETE``/
        ``ABORT`` under ``valid_actions=[]`` — every GUI verb the model can emit
        deleted from its own trained grammar.
        """
        for valid_actions in (None, [], ["tap"], ["tap", "swipe"], ["terminate"]):
            assert self._prompt_action_values(valid_actions) == self.ALL_PROMPT_ROWS, valid_actions

    def test_prompt_rows_ignore_active_extra_tools(self):
        """IDENTITY across every active-extras combination.

        Step-GUI renders no ``<tools>`` block, so these rows ARE the trained
        action grammar; deleting the ``COMPLETE``/``ABORT``/``INFO``/``AWAKE``
        rows when their extra tool is inactive left the model with no finish
        form it had ever been trained to emit. Containment (``"CLICK" in rows``)
        matched both the full and the truncated prompt, which is why the
        previous pinning missed it.
        """
        renders = {}
        for size in range(len(_EXTRA_TOOL_SCHEMAS) + 1):
            for combination in itertools.combinations(_EXTRA_TOOL_SCHEMAS, size):
                adapter = AgentAdapterRegistry.get(
                    self.KEY,
                    metadata=LiteCUAMetadata(
                        dims=("mobile", "use"), extra_tool_schemas=list(combination)
                    ),
                )
                names = tuple(sorted(tool_schema_name(s) for s in combination))
                renders[names] = self._system_prompt_text(adapter, _prompt_only_sample())

        assert len(set(renders.values())) == 1, sorted(renders)
        rows = set(re.findall(r"^\d+\. ([A-Z]+)：", renders[()], flags=re.M))
        assert rows == self.ALL_PROMPT_ROWS

    def test_valid_actions_trims_the_wrapper_action_enum(self):
        """The declaration makes Step-GUI gate like any other wrapper family.

        This is the surface ``valid_actions`` DOES narrow: the ``mobile_use``
        schema is composed per request. The prompt rows above are not.
        """
        schemas = STEPGUIMobileActionSpace.get_tool_schemas()
        # Extra-tool action values (AWAKE/INFO/COMPLETE/ABORT) are gated by
        # ``extra_tool_schemas``, never by ``valid_actions``, so they survive.
        extra_tool_values = ["AWAKE", "INFO", "COMPLETE", "ABORT"]
        assert (
            self._action_enum(
                STEPGUIMobileActionSpace.filter_tool_schemas_for_valid_actions(schemas, [])
            )
            == extra_tool_values
        )
        assert self._action_enum(
            STEPGUIMobileActionSpace.filter_tool_schemas_for_valid_actions(
                schemas, ["tap", "swipe"]
            )
        ) == ["CLICK", "SLIDE", "AWAKE", "INFO", "COMPLETE", "ABORT"]
        # ``terminate`` is an EXTRA tool, not a valid_action: it adds nothing.
        assert (
            self._action_enum(
                STEPGUIMobileActionSpace.filter_tool_schemas_for_valid_actions(
                    schemas, ["terminate"]
                )
            )
            == extra_tool_values
        )


# =============================================================================
# 8. Malformed-but-recoverable provider output
# =============================================================================


class TestMalformedProviderOutput:
    def setup_method(self):
        self.space = STEPGUIMobileActionSpace()

    @pytest.mark.parametrize(
        "name,args",
        [
            ("CLICK", {"point": [500, 300]}),
            ("LONGPRESS", {"point": [500, 300]}),
            ("TYPE", {"value": "hello"}),
            ("WAIT", {}),
        ],
    )
    def test_flat_native_action_value_used_as_tool_name_converts(self, name, args):
        """STEP-GUI packs its whole surface into one ``mobile_use`` wrapper. When
        the model drops the wrapper and uses the action VALUE as the tool name,
        the call must still run the same dispatch branch."""
        flat = self.space.convert_tool_calls_from_agent([{"name": name, "arguments": args}])
        nested = self.space.convert_tool_calls_from_agent(
            [{"name": "mobile_use", "arguments": {"action": name, **args}}]
        )
        assert flat == nested

    def test_wrong_wrapper_name_with_nested_action_converts(self):
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "mobile", "arguments": {"action": "CLICK", "point": [5, 6]}}]
        )
        assert _only_mobile_action(out) == {
            "action": "tap",
            "coordinate": [5, 6],
            "clicks": 1,
        }

    def test_flat_info_becomes_response(self):
        """``INFO`` is STEP-GUI's answer channel; flat output must reach it."""
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "INFO", "arguments": {"value": "42"}}]
        )
        assert out == [make_tool_call("response", {"text": "42"})]

    def test_non_native_name_stays_a_standalone_tool(self):
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "summarize", "arguments": {"text": "hi"}}]
        )
        assert out == [make_tool_call("summarize", {"text": "hi"})]

    def test_active_extra_tool_outranks_a_colliding_native_action_value(self):
        schema = {
            "type": "function",
            "function": {
                "name": "CLICK",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {"bid": {"type": "string"}},
                    "required": ["bid"],
                },
            },
        }
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "CLICK", "arguments": {"bid": "a51"}}],
            active_extra_tool_names={"CLICK"},
            active_extra_tool_schemas=[schema],
        )
        assert out == [make_tool_call("CLICK", {"bid": "a51"})]
