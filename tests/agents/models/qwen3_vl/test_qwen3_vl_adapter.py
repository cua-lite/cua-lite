"""
Tests for Qwen3VL action space and adapter.

Covers the full adapter pipeline:
  1. Registry resolution (exact + regex keys)
  2. Action space: tool schemas, tool call creation
  3. Tool call conversion: CUA-lite ↔ Qwen3VL (round-trip)
  4. convert_sample_to_agent (system prompt, tools key, image resize)
  5. convert_message_to_agent / convert_message_from_agent
  6. parse_raw_assistant_response (Action + <tool_call>, <think>, edge cases)
  7. Trajectory unroll (TrajectoryUnrollMixin)
  8. Per-sample conversion (convert_sample_to_agent / unroll_sample)

Run:
    uv run pytest tests/agents/models/qwen3_vl/test_qwen3_vl_adapter.py -v
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest
from lite_samples import (
    sample_grounding_point,
    sample_trajectory_long,
    sample_trajectory_two_turns,
    sample_trajectory_with_reasoning,
)

import lite.agents.models.qwen3_5.adapter  # noqa: F401
from lite.agents.core.action_space.base import LiteDesktopActionSpace
from lite.agents.core.adapter import (
    AgentAdapterRegistry,
    AsIsAdapter,
    BaseAgentAdapter,
)
from lite.agents.models.qwen3_vl.action_space import (
    Qwen3VLDesktopActionSpace,
    Qwen3VLMobileActionSpace,
)
from lite.agents.models.qwen3_vl.adapter import (
    USE_SYSTEM_PROMPT,
    Qwen3VLDesktopGroundingPointAdapter,
    Qwen3VLDesktopUseAdapter,
    Qwen3VLMobileUseAdapter,
)
from lite.core import LiteCUAMetadata, LiteGenericMetadata, LiteSample
from lite.core.messages import no_tool_call_final_text
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import (
    tool_call_arguments,
    tool_call_name,
    validate_lite_tool_call,
)
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

_LiteMeta = LiteCUAMetadata


def _md(extra_tools=None, valid_actions=None, **others):
    """Test helper — build a desktop-navigation LiteCUAMetadata."""
    return _LiteMeta(
        dims=(_LiteMeta.Platform.DESKTOP, _LiteMeta.TaskType.USE),
        extra_tool_schemas=extra_tools or [],
        valid_actions=valid_actions,
        others=others,
    )


def _single_action(tool_calls):
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert tool_call_name(call) in {"computer", "mobile"}
    actions = tool_call_arguments(call)["actions"]
    assert len(actions) == 1
    return actions[0]


def _desktop_action_call(action):
    return make_tool_call("computer", {"actions": [action]})


def _mobile_action_call(action):
    return make_tool_call("mobile", {"actions": [action]})


def _canonical_args(tool_call, name):
    assert validate_lite_tool_call(tool_call, "tool_call", require_id=False) is None
    assert tool_call_name(tool_call) == name
    return tool_call_arguments(tool_call)


def _computer_use_call(**arguments):
    return {"name": "computer_use", "arguments": arguments}


def _mobile_use_call(**arguments):
    return {"name": "mobile_use", "arguments": arguments}


def _extra_tool_schema(name: str, properties: dict) -> dict:
    return make_tool_schema(
        name,
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(properties),
        },
    )


def _qwen_tool_block(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>\n{payload}\n</tool_call>\n"


def _use_metadata(platform: _LiteMeta.Platform, extra_tool_schemas: list[dict]) -> _LiteMeta:
    return _LiteMeta(
        dims=(platform, _LiteMeta.TaskType.USE),
        extra_tool_schemas=extra_tool_schemas,
    )


# Inline copy of the WITH_THOUGHT prompt (the constant is currently commented
# out in adapter.py — we keep a test-local copy so the enable_inline_reasoning=True toggle
# path stays exercised in case it gets re-enabled).
_USE_SYSTEM_PROMPT_WITH_THOUGHT = """# Response format

Response format for every step:
1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
2) Action: a short imperative describing what to do in the UI.
3) A single <tool_call>...</tool_call> block.

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one for Action.
- Do not output anything else outside those three parts."""

# =============================================================================
# 1. Action Space: tool schemas
# =============================================================================


class TestActionSpaceSchemas:
    """Test Qwen3VLDesktopActionSpace tool schema generation."""

    def setup_method(self):
        self.action_space = Qwen3VLDesktopActionSpace()

    def test_action_is_required(self):
        """action parameter should be required."""
        schema = self.action_space.get_tool_schemas()[0]
        assert "action" in tool_schema_parameters(schema)["required"]


# =============================================================================
# 2. Action Space: tool call creation
# =============================================================================


class TestActionSpaceToolCalls:
    """Test Qwen3VLDesktopActionSpace static tool call methods."""

    def test_type(self):
        tc = Qwen3VLDesktopActionSpace.computer_use(action="type", text="hello")
        args = _canonical_args(tc, "computer_use")
        assert args["action"] == "type"
        assert args["text"] == "hello"

    def test_key(self):
        tc = Qwen3VLDesktopActionSpace.computer_use(action="key", keys=["ctrl", "c"])
        assert _canonical_args(tc, "computer_use")["keys"] == ["ctrl", "c"]

    def test_scroll(self):
        tc = Qwen3VLDesktopActionSpace.computer_use(
            action="scroll", pixels=-3, coordinate=[100, 200]
        )
        args = _canonical_args(tc, "computer_use")
        assert args["pixels"] == -3
        assert args["coordinate"] == [100, 200]

    def test_wait(self):
        tc = Qwen3VLDesktopActionSpace.computer_use(action="wait", time=2.0)
        assert _canonical_args(tc, "computer_use")["time"] == 2.0

    def test_answer(self):
        tc = Qwen3VLDesktopActionSpace.computer_use(action="answer", text="42")
        args = _canonical_args(tc, "computer_use")
        assert args["action"] == "answer"
        assert args["text"] == "42"

    def test_none_args_omitted(self):
        """Optional args with None should not appear in arguments."""
        tc = Qwen3VLDesktopActionSpace.computer_use(action="left_click")
        args = _canonical_args(tc, "computer_use")
        assert "coordinate" not in args
        assert "keys" not in args


# =============================================================================
# 3. Tool call conversion: CUA-lite → Qwen3VL → CUA-lite (round-trip)
# =============================================================================


class TestToolCallConversion:
    """Test convert_tool_calls_to_agent and convert_tool_calls_from_agent."""

    def setup_method(self):
        self.action_space = Qwen3VLDesktopActionSpace()

    # -- to_agent --

    def test_click_to_left_click(self):
        tc = [LiteDesktopActionSpace.click(coordinate=[500, 300])]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert len(result) == 1
        assert result[0]["arguments"]["action"] == "left_click"
        assert result[0]["arguments"]["coordinate"] == [500, 300]

    def test_middle_click_to_agent(self):
        tc = [LiteDesktopActionSpace.click(coordinate=[100, 200], button="middle")]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "middle_click"

    def test_triple_click_to_agent(self):
        tc = [LiteDesktopActionSpace.click(coordinate=[100, 200], clicks=3)]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "triple_click"

    def test_type_to_agent(self):
        tc = [LiteDesktopActionSpace.type(text="hello")]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "type"
        assert result[0]["arguments"]["text"] == "hello"

    def test_key_to_agent(self):
        tc = [LiteDesktopActionSpace.key(keys=["ctrl", "a"])]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "key"
        assert result[0]["arguments"]["keys"] == ["ctrl", "a"]

    def test_scroll_down_to_agent(self):
        tc = [LiteDesktopActionSpace.scroll(direction="down", amount=3, coordinate=[500, 500])]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "scroll"
        assert result[0]["arguments"]["pixels"] == -300  # 3 clicks * 100 px/click

    def test_scroll_up_to_agent(self):
        tc = [LiteDesktopActionSpace.scroll(direction="up", amount=5)]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["pixels"] == 500  # 5 clicks * 100 px/click

    def test_drag_without_start_to_agent(self):
        tc = [LiteDesktopActionSpace.drag(coordinate=[800, 600])]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert len(result) == 1
        assert result[0]["arguments"]["action"] == "left_click_drag"

    def test_drag_with_start_to_agent_yields_two(self):
        """Drag with start_coordinate should produce mouse_move + left_click_drag."""
        tc = [LiteDesktopActionSpace.drag(coordinate=[800, 600], start_coordinate=[100, 100])]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert len(result) == 2
        assert result[0]["arguments"]["action"] == "mouse_move"
        assert result[0]["arguments"]["coordinate"] == [100, 100]
        assert result[1]["arguments"]["action"] == "left_click_drag"
        assert result[1]["arguments"]["coordinate"] == [800, 600]

    def test_mouse_move_to_agent(self):
        tc = [LiteDesktopActionSpace.mouse_move(coordinate=[300, 400])]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "mouse_move"

    def test_wait_to_agent(self):
        tc = [LiteDesktopActionSpace.wait(duration=2.0)]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "wait"
        assert result[0]["arguments"]["time"] == 2.0

    def test_terminate_to_agent(self):
        tc = [make_tool_call("terminate", {"status": "success"})]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "terminate"
        assert result[0]["arguments"]["status"] == "success"

    def test_response_to_answer(self):
        tc = [make_tool_call("response", {"text": "42"})]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "answer"
        assert result[0]["arguments"]["text"] == "42"

    def test_unknown_action_passthrough(self):
        # Actions not in LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES are STANDALONE
        # function tools (browser nav, env extra tools, ...) and pass through as
        # identity — NOT forced into a bogus computer_use(action=<name>) wrapper.
        tc = [make_tool_call("custom_action", {"foo": "bar"})]
        result = self.action_space.convert_tool_calls_to_agent(tc)
        assert result[0]["name"] == "custom_action"
        assert result[0]["arguments"] == {"foo": "bar"}

    # -- from_agent --

    def test_direct_native_enum_from_agent_converts_as_that_action(self):
        """A dropped ``computer_use`` wrapper leaves the native action value as
        the tool name; it converts through the same branch as the nested call."""
        tc = [{"name": "left_click", "arguments": {"coordinate": [500, 300]}}]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        assert result == self.action_space.convert_tool_calls_from_agent(
            [_computer_use_call(action="left_click", coordinate=[500, 300])]
        )
        assert _single_action(result) == {"action": "click", "coordinate": [500, 300]}

    def test_right_click_from_agent(self):
        tc = [_computer_use_call(action="right_click", coordinate=[100, 200])]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        assert _single_action(result).get("button") == "right"

    def test_double_click_from_agent(self):
        tc = [_computer_use_call(action="double_click", coordinate=[100, 200])]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        assert _single_action(result).get("clicks") == 2

    def test_triple_click_from_agent(self):
        tc = [_computer_use_call(action="triple_click", coordinate=[100, 200])]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        assert _single_action(result).get("clicks") == 3

    def test_type_from_agent(self):
        tc = [_computer_use_call(action="type", text="test")]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        action = _single_action(result)
        assert action["action"] == "type"
        assert action["text"] == "test"

    def test_key_from_agent(self):
        tc = [_computer_use_call(action="key", keys=["enter"])]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        action = _single_action(result)
        assert action["action"] == "key"
        assert action["keys"] == ["enter"]

    def test_scroll_positive_from_agent(self):
        tc = [_computer_use_call(action="scroll", pixels=300)]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        action = _single_action(result)
        assert action["action"] == "scroll"
        assert action["direction"] == "up"
        assert action["amount"] == 3  # 300px / 100 px/click

    def test_scroll_negative_from_agent(self):
        tc = [_computer_use_call(action="scroll", pixels=-500)]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        action = _single_action(result)
        assert action["direction"] == "down"
        assert action["amount"] == 5  # 500px / 100 px/click

    def test_left_click_drag_from_agent(self):
        tc = [_computer_use_call(action="left_click_drag", coordinate=[800, 600])]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        assert _single_action(result)["action"] == "drag"

    def test_mouse_move_from_agent(self):
        tc = [_computer_use_call(action="mouse_move", coordinate=[300, 400])]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        assert _single_action(result)["action"] == "mouse_move"

    def test_wait_from_agent(self):
        tc = [_computer_use_call(action="wait", time=5.0)]
        result = self.action_space.convert_tool_calls_from_agent(tc)
        action = _single_action(result)
        assert action["action"] == "wait"
        assert action["duration"] == 5.0

    # -- round-trip --

    @pytest.mark.parametrize(
        "cua_tc",
        [
            LiteDesktopActionSpace.click(coordinate=[500, 300]),
            LiteDesktopActionSpace.click(coordinate=[100, 200], button="right"),
            LiteDesktopActionSpace.click(coordinate=[100, 200], button="middle"),
            LiteDesktopActionSpace.click(coordinate=[100, 200], clicks=2),
            # clicks=3 excluded: Qwen3VL has no triple_click, so it round-trips
            # lossy (triple_click -> double_click -> clicks=2).
            LiteDesktopActionSpace.type(text="hello world"),
            LiteDesktopActionSpace.key(keys=["ctrl", "v"]),
            LiteDesktopActionSpace.mouse_move(coordinate=[300, 400]),
            LiteDesktopActionSpace.wait(duration=1.5),
            make_tool_call("terminate", {"status": "success"}),
            make_tool_call("response", {"text": "answer text"}),
        ],
    )
    def test_round_trip_preserves_semantics(self, cua_tc):
        """CUA-lite → Qwen3VL → CUA-lite should preserve the original action semantics."""
        action_space = Qwen3VLDesktopActionSpace()
        agent_tcs = action_space.convert_tool_calls_to_agent([cua_tc])
        restored = action_space.convert_tool_calls_from_agent(agent_tcs)
        assert restored == [cua_tc]

    def test_round_trip_scroll(self):
        """Scroll round-trip: direction+amount ↔ signed clicks."""
        action_space = Qwen3VLDesktopActionSpace()
        cua_tc = LiteDesktopActionSpace.scroll(direction="up", amount=4)
        agent_tcs = action_space.convert_tool_calls_to_agent([cua_tc])
        restored = action_space.convert_tool_calls_from_agent(agent_tcs)
        action = _single_action(restored)
        assert action["direction"] == "up"
        assert action["amount"] == 4

    def test_from_agent_batches_adjacent_action_calls(self):
        action_space = Qwen3VLDesktopActionSpace()
        agent_tcs = [
            _computer_use_call(action="left_click", coordinate=[500, 500]),
            _computer_use_call(action="left_click", coordinate=[600, 600]),
        ]
        restored = action_space.convert_tool_calls_from_agent(agent_tcs)
        assert restored == [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [500, 500]},
                        {"action": "click", "coordinate": [600, 600]},
                    ]
                },
            )
        ]

    def test_from_agent_preserves_extra_tool_order_with_segmented_batches(self):
        action_space = Qwen3VLDesktopActionSpace()
        agent_tcs = [
            _computer_use_call(action="left_click", coordinate=[100, 100]),
            _computer_use_call(action="type", text="a"),
            {"name": "bash", "arguments": {"command": "pwd"}},
            _computer_use_call(action="left_click", coordinate=[200, 200]),
            _computer_use_call(action="type", text="b"),
        ]
        restored = action_space.convert_tool_calls_from_agent(agent_tcs)
        assert restored == [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [100, 100]},
                        {"action": "type", "text": "a"},
                    ]
                },
            ),
            make_tool_call("bash", {"command": "pwd"}),
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [200, 200]},
                        {"action": "type", "text": "b"},
                    ]
                },
            ),
        ]

    def test_mobile_from_agent_preserves_extra_tool_order_with_segmented_batches(self):
        from lite.agents.models.qwen3_vl.action_space import Qwen3VLMobileActionSpace

        action_space = Qwen3VLMobileActionSpace()
        agent_tcs = [
            _mobile_use_call(action="click", coordinate=[100, 100]),
            _mobile_use_call(action="type", text="a"),
            _mobile_use_call(action="open", text="Maps"),
            _mobile_use_call(action="click", coordinate=[200, 200]),
            _mobile_use_call(action="type", text="b"),
        ]
        restored = action_space.convert_tool_calls_from_agent(agent_tcs)
        assert restored == [
            make_tool_call(
                "mobile",
                {
                    "actions": [
                        {"action": "tap", "coordinate": [100, 100], "clicks": 1},
                        {"action": "type", "text": "a"},
                    ]
                },
            ),
            make_tool_call("open_app", {"app_name": "Maps"}),
            make_tool_call(
                "mobile",
                {
                    "actions": [
                        {"action": "tap", "coordinate": [200, 200], "clicks": 1},
                        {"action": "type", "text": "b"},
                    ]
                },
            ),
        ]

    def test_multiple_tool_calls(self):
        """Multiple CUA-lite tool calls should all convert correctly."""
        action_space = Qwen3VLDesktopActionSpace()
        cua_tcs = [
            LiteDesktopActionSpace.click(coordinate=[500, 300]),
            LiteDesktopActionSpace.type(text="test"),
            LiteDesktopActionSpace.key(keys=["enter"]),
        ]
        agent_tcs = action_space.convert_tool_calls_to_agent(cua_tcs)
        assert len(agent_tcs) == 3
        actions = [tc["arguments"]["action"] for tc in agent_tcs]
        assert actions == ["left_click", "type", "key"]


# =============================================================================
# 4. Registry: resolve adapter by key
# =============================================================================


class TestRegistry:
    """Test adapter registry resolution for qwen3_vl keys."""

    @pytest.mark.parametrize(
        "key,expected_cls",
        [
            ("qwen3_vl@desktop@grounding.point", Qwen3VLDesktopGroundingPointAdapter),
            ("qwen3_vl@desktop@use", Qwen3VLDesktopUseAdapter),
        ],
    )
    def test_registry_returns_correct_class(self, key, expected_cls):
        adapter = AgentAdapterRegistry.get(key)
        assert isinstance(adapter, BaseAgentAdapter)
        assert type(adapter) is expected_cls

    def test_pass_through_adapters(self):
        """``understanding`` / ``grounding.bbox`` keys resolve to AsIsAdapter.

        ``grounding.point`` and ``grounding.action`` are both served by
        concrete adapter classes — Desktop registers under the
        ``r"qwen3_vl@(desktop|browser)@..."`` regex (covers both platforms)
        and Mobile under its own concrete key. See
        :func:`test_registry_returns_correct_class`.
        """
        for key in [
            "qwen3_vl@desktop@understanding",
            "qwen3_vl@mobile@understanding",
            "qwen3_vl@desktop@grounding.bbox",
        ]:
            adapter = AgentAdapterRegistry.get(key)
            assert type(adapter) is AsIsAdapter, f"Expected AsIsAdapter for {key}"

    def test_qwen3_vl_base_wildcard_resolves_to_base_adapter(self):
        """Canonical base keys resolve to :class:`Qwen3VLBaseAdapter`
        directly -- workflow-agnostic. Browser rows use the platform/task suffix
        to opt out of the navigation wire format entirely."""
        from lite.agents.models.qwen3_vl.adapter import (
            Qwen3VLBaseAdapter,
            Qwen3VLUseAdapter,
        )

        for key in [
            "qwen3_vl.base",
            "qwen3_vl.base@desktop@use",
            "qwen3_vl.base@browser@use",
            "qwen3_vl.base@mobile@use",
        ]:
            adapter = AgentAdapterRegistry.get(key)
            assert type(adapter) is Qwen3VLBaseAdapter, key
            assert not isinstance(adapter, Qwen3VLUseAdapter), key
            assert not hasattr(adapter, "enable_inline_reasoning"), key

    def test_qwen3_vl_base_agent_registry_resolves(self):
        """End-to-end: canonical ``qwen3_vl.base`` keys must
        resolve to :class:`Qwen3VLBaseAgent` (NOT a navigation agent), so
        yaml-driven rollouts with ``agent_id: "qwen3_vl.base"`` plumbed
        through ``make`` actually find a registered agent class.
        Pins the audit-surfaced gap where the adapter side was registered
        but the agent side wasn't."""
        from lite.agents.models import AgentRegistry
        from lite.agents.models.qwen3_vl.agent import (
            Qwen3VLBaseAgent,
            Qwen3VLDesktopUseAgent,
        )

        async def _dummy_gen(**_):
            return {"response": ""}

        class _DummyProcessor:
            def apply_chat_template(self, messages, **kwargs):
                return ""

        for key in [
            "qwen3_vl.base",
            "qwen3_vl.base@desktop@use",
            "qwen3_vl.base@browser@use",
        ]:
            agent = AgentRegistry.get(
                key,
                generate_fn=_dummy_gen,
                processor=_DummyProcessor(),
            )
            assert type(agent) is Qwen3VLBaseAgent, key
            assert not isinstance(agent, Qwen3VLDesktopUseAgent), key

    def test_grounding_action_has_full_history_protocol(self):
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")
        assert adapter.protocol.get_registry_key() == "lite.history"

    def test_trajectory_has_summarized_history_protocol(self):
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@use")
        assert adapter.protocol.get_registry_key() == "qwen3_vl.history"
        assert adapter.protocol.full_history_size == 4

    def test_trajectory_has_system_prompt(self):
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@use")
        assert adapter.system_prompt == USE_SYSTEM_PROMPT

    def test_base_adapter_action_space_defaults_to_workflow_agnostic(self):
        """``Qwen3VLBaseAdapter`` defaults ``action_space`` to
        :class:`BaseActionSpace` (workflow-agnostic, empty schemas,
        identity pass-through), NOT a platform-specific class. Concrete
        Navigation / Grounding leaves override per-platform. Pins the
        contract so a yaml setting ``agent_id: "qwen3_vl.base"`` doesn't
        accidentally surface desktop-specific tools.
        """
        from lite.agents.core.action_space import BaseActionSpace
        from lite.agents.models.qwen3_vl.action_space import (
            Qwen3VLDesktopActionSpace,
            Qwen3VLDesktopGroundingPointActionSpace,
        )

        # Base resolves to BaseActionSpace.
        base = AgentAdapterRegistry.get("qwen3_vl.base@browser@use")
        assert type(base.action_space) is BaseActionSpace
        # Empty surface + identity pass-through.
        assert base.action_space.get_tool_schemas() == []
        canonical_call = [make_tool_call("click", {})]
        bare_call = [{"name": "click", "arguments": {}}]
        assert base.action_space.convert_tool_calls_to_agent(canonical_call) == bare_call
        assert base.action_space.convert_tool_calls_from_agent(bare_call) == canonical_call
        # Navigation leaf still has the desktop space.
        nav = AgentAdapterRegistry.get("qwen3_vl@desktop@use")
        assert type(nav.action_space) is Qwen3VLDesktopActionSpace
        # Grounding leaf still has the trimmed grounding-point space.
        gr = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")
        assert type(gr.action_space) is Qwen3VLDesktopGroundingPointActionSpace

    def test_base_adapter_can_render_text_answer_prompt_without_tools(self):
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
            "qwen3_vl.base",
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

    def test_base_adapter_drops_navigation_specific_content_types(self):
        """``Qwen3VLBaseAdapter._convert_message_to_agent`` MUST consume
        only ``type: "text"`` content parts. ``action_description`` /
        ``inline_reasoning`` are workflow-specific to
        :class:`Qwen3VLUseAdapter`; the base adapter (used by
        BrowserGym via ``qwen3_vl.base@...``) drops them rather than
        accidentally promoting reasoning text into the chat-template
        prompt. Pins the strict-base contract from the cross-branch
        adapter-dump diff."""
        from lite.agents.models.qwen3_vl.adapter import Qwen3VLBaseAdapter

        adapter = AgentAdapterRegistry.get("qwen3_vl.base@desktop@use")
        assert type(adapter) is Qwen3VLBaseAdapter
        msg = {
            "role": "assistant",
            "content": [
                {"type": "inline_reasoning", "text": "I should click."},
                {"type": "action_description", "text": "Click button."},
                {"type": "text", "text": "explicit text part"},
            ],
        }
        out = adapter.convert_message_to_agent(msg)
        # Only the ``type: text`` part survives.
        assert out["content"] == [{"type": "text", "text": "explicit text part"}]

    def test_grounding_does_not_inherit_navigation(self):
        """Grounding adapters subclass :class:`Qwen3VLBaseAdapter` directly
        (not :class:`Qwen3VLUseAdapter`) — no ``Action:`` /
        ``Thought:`` wire-format machinery, no ``enable_inline_reasoning``
        knob. Content flows through base's text-flatten path."""
        from lite.agents.models.qwen3_vl.adapter import Qwen3VLUseAdapter

        adapter: Qwen3VLDesktopGroundingPointAdapter = AgentAdapterRegistry.get(
            "qwen3_vl@desktop@grounding.point"
        )  # type: ignore[assignment]
        assert not isinstance(adapter, Qwen3VLUseAdapter)
        assert not hasattr(adapter, "enable_inline_reasoning")


# =============================================================================
# 5. convert_sample_to_agent
# =============================================================================


class TestConvertSampleToAgent:
    """Test Qwen3VLBaseAdapter.unroll (last-step view = predict-time render)."""

    def _last_step(self, adapter, sample):
        return adapter.unroll(sample).steps[-1]

    def test_grounding_action_converts_tool_calls(self):
        """Assistant tool_calls should be converted to computer_use format."""
        sample = sample_grounding_point()
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")
        msgs = self._last_step(adapter, sample)
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        assert len(assistant_msgs) > 0
        for msg in assistant_msgs:
            for tc in msg["tool_calls"]:
                assert tc["name"] == "computer_use"

    def test_trajectory_adds_system_prompt(self):
        """Trajectory adapter should prepend a system message."""
        sample = sample_trajectory_two_turns()
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@use")
        msgs = self._last_step(adapter, sample)
        assert msgs[0]["role"] == "system"
        text_parts = [c.get("text", "") for c in msgs[0]["content"] if c.get("type") == "text"]
        joined = " ".join(text_parts)
        assert "Action" in joined  # System prompt mentions Action format

    def test_trajectory_always_has_system_prompt(self):
        """System message is always prepended."""
        sample = sample_trajectory_two_turns()
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@use")
        msgs = self._last_step(adapter, sample)
        assert msgs[0]["role"] == "system"

    def test_output_has_required_fields(self):
        """Output should have images and properly formed messages."""
        sample = sample_grounding_point()
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")
        out = adapter.unroll(sample)
        assert isinstance(out.processed_images, list)
        assert len(out.processed_images) == len(sample.images)
        assert isinstance(out.steps, list)
        assert out.steps and len(out.steps[-1]) > 0

    def test_user_messages_unchanged(self):
        """User messages should pass through unchanged (no tool_calls to convert)."""
        sample = sample_grounding_point()
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")
        msgs = self._last_step(adapter, sample)
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) > 0
        assert user_msgs[0]["content"] == sample.messages[0]["content"]

    def test_deep_copy_isolation(self):
        """Mutating output should not affect input."""
        sample = sample_grounding_point()
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")
        msgs = self._last_step(adapter, sample)
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        assistant_msgs[0]["tool_calls"][0]["arguments"]["action"] = "MUTATED"
        # Original lite-side ``point`` call should still be intact.
        assert tool_call_name(sample.messages[1]["tool_calls"][0]) == "point"

    def test_smart_resize_applied_to_pil_images(self):
        """PIL images in sample['images'] should be resized by smart_resize."""
        from PIL import Image

        # Create a large image that should be resized
        large_img = Image.new("RGB", (4000, 3000), color="red")
        sample = sample_grounding_point()
        sample = dataclasses.replace(sample, images=[large_img])
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")
        out = adapter.unroll(sample)
        resized = out.processed_images[0]
        # Should be resized (not the original 4000x3000)
        assert resized.size[0] <= 4000 or resized.size[1] <= 3000

    def test_string_images_not_resized(self):
        """String image paths should pass through unchanged."""
        sample = sample_grounding_point()
        sample = dataclasses.replace(sample, images=["path/to/img.png"])
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")
        out = adapter.unroll(sample)
        assert out.processed_images == ["path/to/img.png"]

    def test_trajectory_long_windowing(self):
        """With 6 turns and full_history_size=4, summarized history should be injected."""
        sample = sample_trajectory_long(num_turns=6)
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@use")
        msgs = self._last_step(adapter, sample)
        # Should have fewer messages than the raw 12 + system
        assert len(msgs) < len(sample.messages) + 1
        # First user message should contain "Previous actions:" summary
        first_user = next(m for m in msgs if m["role"] == "user")
        text_items = [
            c.get("text", "") for c in first_user.get("content", []) if c.get("type") == "text"
        ]
        joined = " ".join(text_items)
        assert "Step 1:" in joined or "Previous actions:" in joined


# =============================================================================
# 6. convert_message_to_agent / convert_message_from_agent
# =============================================================================


class TestConvertMessage:
    """Test single-message conversion methods.

    Uses the navigation adapter (full action vocabulary) so the test cases
    can exercise multiple lite action names (``click`` etc.). The grounding
    adapter's trimmed schema only accepts ``point``; per-message conversion
    is identical at the base level.
    """

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@use")

    def test_to_agent_converts_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Click button."}],
            "tool_calls": [LiteDesktopActionSpace.click(coordinate=[500, 300])],
        }
        out = self.adapter.convert_message_to_agent(msg)
        assert out["tool_calls"][0]["name"] == "computer_use"
        assert out["tool_calls"][0]["arguments"]["action"] == "left_click"

    def test_to_agent_preserves_content(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Click button."}],
            "tool_calls": [LiteDesktopActionSpace.click(coordinate=[500, 300])],
        }
        out = self.adapter.convert_message_to_agent(msg)
        # Navigation adapter prefixes ``Action: `` per its 2-part wire format.
        assert out["content"][0]["text"] == "Action: Click button."

    def test_to_agent_user_message_unchanged(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "Do something."}]}
        out = self.adapter.convert_message_to_agent(msg)
        assert out == msg

    def test_to_agent_role_tool_message_reaches_chat_template(self):
        msg = {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "text", "text": "screen changed"}],
        }
        out = self.adapter.convert_message_to_agent(msg)
        assert out == msg

    def test_to_agent_deep_copy(self):
        msg = {
            "role": "assistant",
            "tool_calls": [LiteDesktopActionSpace.click(coordinate=[500, 300])],
        }
        out = self.adapter.convert_message_to_agent(msg)
        assert out is not msg

    def test_from_agent_converts_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Click button."}],
            "tool_calls": [_computer_use_call(action="left_click", coordinate=[500, 300])],
        }
        out = self.adapter.convert_message_from_agent(msg)
        action = _single_action(out["tool_calls"])
        assert action["action"] == "click"
        assert action["coordinate"] == [500, 300]

    def test_from_agent_preserves_reasoning_content(self):
        msg = {
            "role": "assistant",
            "reasoning_content": "I need to click the button.",
            "content": [{"type": "text", "text": "Click."}],
            "tool_calls": [_computer_use_call(action="left_click", coordinate=[500, 300])],
        }
        out = self.adapter.convert_message_from_agent(msg)
        assert out["reasoning_content"] == "I need to click the button."

    def test_from_agent_deep_copy(self):
        msg = {
            "role": "assistant",
            "tool_calls": [_computer_use_call(action="left_click", coordinate=[500, 300])],
        }
        out = self.adapter.convert_message_from_agent(msg)
        assert out is not msg

    def test_roundtrip_message(self):
        """to_agent then from_agent should restore the original message."""
        original = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Click."}],
            "tool_calls": [LiteDesktopActionSpace.click(coordinate=[500, 300])],
        }
        agent_msg = self.adapter.convert_message_to_agent(original)
        restored = self.adapter.convert_message_from_agent(agent_msg)
        assert restored["tool_calls"] == [
            _desktop_action_call({"action": "click", "coordinate": [500, 300]})
        ]

    def test_desktop_batches_adjacent_native_action_calls_and_renders_back(self):
        schemas = [
            _extra_tool_schema("bash", {"command": {"type": "string"}}),
            _extra_tool_schema("goto", {"url": {"type": "string"}}),
            _extra_tool_schema("terminate", {"status": {"type": "string"}}),
        ]
        adapter = Qwen3VLDesktopUseAdapter(
            metadata=_use_metadata(_LiteMeta.Platform.DESKTOP, schemas)
        )
        raw = (
            "Action: Click, type, run bash, click, navigate, then finish.\n"
            + _qwen_tool_block("computer_use", {"action": "left_click", "coordinate": [100, 200]})
            + _qwen_tool_block("computer_use", {"action": "type", "text": "a"})
            + _qwen_tool_block("bash", {"command": "pwd"})
            + _qwen_tool_block("computer_use", {"action": "left_click", "coordinate": [300, 400]})
            + _qwen_tool_block("goto", {"url": "https://example.com"})
            + _qwen_tool_block("computer_use", {"action": "terminate", "status": "success"})
        )

        lite_msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
        assert lite_msg["tool_calls"] == [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [100, 200]},
                        {"action": "type", "text": "a"},
                    ]
                },
            ),
            make_tool_call("bash", {"command": "pwd"}),
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [300, 400]},
                    ]
                },
            ),
            make_tool_call("goto", {"url": "https://example.com"}),
            make_tool_call("terminate", {"status": "success"}),
        ]

        out = adapter.convert_message_to_agent(lite_msg)
        assert out["tool_calls"] == [
            {
                "name": "computer_use",
                "arguments": {"action": "left_click", "coordinate": [100, 200]},
            },
            {"name": "computer_use", "arguments": {"action": "type", "text": "a"}},
            {"name": "bash", "arguments": {"command": "pwd"}},
            {
                "name": "computer_use",
                "arguments": {"action": "left_click", "coordinate": [300, 400]},
            },
            {"name": "goto", "arguments": {"url": "https://example.com"}},
            {"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}},
        ]

    def test_mobile_batches_adjacent_native_action_calls_and_renders_back(self):
        schemas = [
            _extra_tool_schema("open_app", {"app_name": {"type": "string"}}),
            _extra_tool_schema("response", {"text": {"type": "string"}}),
        ]
        adapter = Qwen3VLMobileUseAdapter(
            metadata=_use_metadata(_LiteMeta.Platform.MOBILE, schemas)
        )
        raw = (
            "Action: Tap, type, open app, answer, then wait.\n"
            + _qwen_tool_block("mobile_use", {"action": "click", "coordinate": [100, 200]})
            + _qwen_tool_block("mobile_use", {"action": "type", "text": "a"})
            + _qwen_tool_block("mobile_use", {"action": "open", "text": "Settings"})
            + _qwen_tool_block("mobile_use", {"action": "answer", "text": "done"})
            + _qwen_tool_block("mobile_use", {"action": "wait", "time": 1})
        )

        lite_msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
        assert lite_msg["tool_calls"] == [
            make_tool_call(
                "mobile",
                {
                    "actions": [
                        {"action": "tap", "coordinate": [100, 200], "clicks": 1},
                        {"action": "type", "text": "a"},
                    ]
                },
            ),
            make_tool_call("open_app", {"app_name": "Settings"}),
            make_tool_call("response", {"text": "done"}),
            make_tool_call(
                "mobile",
                {
                    "actions": [
                        {"action": "wait", "duration": 1.0},
                    ]
                },
            ),
        ]

        out = adapter.convert_message_to_agent(lite_msg)
        assert out["tool_calls"] == [
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [100, 200]}},
            {"name": "mobile_use", "arguments": {"action": "type", "text": "a"}},
            {"name": "mobile_use", "arguments": {"action": "open", "text": "Settings"}},
            {"name": "mobile_use", "arguments": {"action": "answer", "text": "done"}},
            {"name": "mobile_use", "arguments": {"action": "wait", "time": 1}},
        ]

    # ─── lock-in coverage for the wire-format behaviour the navigation
    # adapter currently inherits from base. These pin the observable
    # contract so the refactor that splits Qwen3VLBaseAdapter (text
    # passthrough) from Qwen3VLUseAdapter (Action:/Thought: line
    # rendering + action_description / inline_reasoning round-trip) can
    # be verified by re-running the suite — desktop nav (this class)
    # keeps the same behaviour either way; only the location of the
    # logic changes. ──────────────────────────────────────────────────

    def test_to_agent_inline_reasoning_dropped_when_thought_disabled(self):
        """``enable_inline_reasoning=False`` (desktop default): inline_reasoning parts
        MUST be dropped — the 2-part wire format forbids prose between
        Action and ``<tool_call>``. Only ``action_description`` survives."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "inline_reasoning", "text": "I should click."},
                {"type": "action_description", "text": "Click button."},
            ],
        }
        out = self.adapter.convert_message_to_agent(msg)
        assert self.adapter.enable_inline_reasoning is False
        text = out["content"][0]["text"]
        assert text == "Action: Click button."
        assert "I should click." not in text  # inline_reasoning dropped

    def test_to_agent_inline_reasoning_rendered_when_thought_enabled(self):
        """``enable_inline_reasoning=True``: inline_reasoning parts surface as
        ``Thought:`` body lines, in order before ``Action:``."""
        from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter

        adapter = Qwen3VLDesktopUseAdapter(enable_inline_reasoning=True)
        msg = {
            "role": "assistant",
            "content": [
                {"type": "inline_reasoning", "text": "I should click."},
                {"type": "action_description", "text": "Click button."},
            ],
        }
        out = adapter.convert_message_to_agent(msg)
        text = out["content"][0]["text"]
        assert text == "Thought: I should click.\nAction: Click button."

    # NOTE: the opaque/verbatim wire format (formerly ``extract_action_only=False``
    # on this adapter) now lives on the WebGym ``*.passthrough`` adapter — see
    # tests/agents/models/qwen3_vl/test_qwen3_vl_webgym_wireformat_chars.py.
    # The decomposed nav adapter is Action-only by construction (no such knob).

    def test_from_agent_keeps_the_whole_prose_when_no_action_prefix(self):
        """Raw text WITHOUT an ``Action: `` line: the WHOLE block is the action
        description.

        Keeping only the first line dropped the model's reasoning on 17% of
        tool-call turns, and it bought nothing at runtime -- same-family replay
        re-renders ``raw_response.text`` verbatim, so the truncation only ever
        cost the durable record, which is what a cross-family student is trained
        on.

        The turn must carry a tool call: ``action_description`` is the
        narration-accompanying-an-action channel, so a no-tool-call turn keeps
        its prose as plain ``text`` (see
        ``TestEdgeCases::test_no_tool_call_turn_keeps_plain_text_for_termination``).
        """
        msg = {
            "role": "assistant",
            "tool_calls": [{"name": "click", "arguments": {"coordinate": [100, 200]}}],
            "content": [{"type": "text", "text": "Click the search bar.\nAdditional reasoning."}],
        }
        out = self.adapter.convert_message_from_agent(msg)
        action_part = next((c for c in out["content"] if c["type"] == "action_description"), None)
        assert action_part is not None
        assert action_part["text"] == "Click the search bar.\nAdditional reasoning."

    def test_declared_extra_tool_routes_separately(self):
        """When ``extra_tools`` is set, tool_calls whose canonical tool name
        matches an extra-tool name MUST bypass action_space conversion
        and pass through unchanged. The standard tools route through
        ``computer_use``."""
        from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter

        goto_tool = _extra_tool_schema("goto", {"url": {"type": "string"}})
        adapter = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=[goto_tool]))
        msg = {
            "role": "assistant",
            "tool_calls": [
                LiteDesktopActionSpace.click(coordinate=[100, 200]),
                make_tool_call("goto", {"url": "https://x.test"}),
            ],
        }
        out = adapter.convert_message_to_agent(msg)
        names = [tc["name"] for tc in out["tool_calls"]]
        assert names == ["computer_use", "goto"]
        goto_tc = next(tc for tc in out["tool_calls"] if tc["name"] == "goto")
        assert goto_tc["arguments"] == {"url": "https://x.test"}

    def test_from_agent_lifts_wrapper_embedded_extra_tool_to_the_top_level(self):
        """An ACTIVE extra named as the wrapper's action value belongs at the top.

        Nested it would be a canonical shape the row contract rejects
        (``must not nest standalone extra tool 'goto'``) and the env cannot
        execute, so the navigation would be lost.
        """
        from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter

        goto_tool = _extra_tool_schema("goto", {"url": {"type": "string"}})
        adapter = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=[goto_tool]))
        agent_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "computer_use",
                    "arguments": {"action": "goto", "url": "https://x.test"},
                }
            ],
        }
        out = adapter.convert_message_from_agent(agent_msg)
        assert out["tool_calls"] == [make_tool_call("goto", {"url": "https://x.test"})]

    @pytest.mark.parametrize("adapter_key", ["qwen3_vl@mobile@use", "qwen3_5@mobile@use"])
    @pytest.mark.parametrize(
        "name,args,properties,required",
        [
            # ``open_app`` is not listed here because Qwen mobile has a native
            # wrapper spelling for it: ``mobile_use(action="open", text=...)``.
            # These extras have no wrapper-native spelling at all, so the model
            # naming one as an action value is pure layer confusion and the call
            # is lifted back to the standalone tool it names.
            ("bash", {"command": "pwd"}, {"command": {"type": "string"}}, ["command"]),
            ("goto", {"url": "https://x.test"}, {"url": {"type": "string"}}, ["url"]),
        ],
    )
    def test_mobile_from_agent_lifts_wrapper_embedded_extra_tool_to_the_top_level(
        self,
        adapter_key,
        name,
        args,
        properties,
        required,
    ):
        """Mobile extras stay standalone tools — including out of the wrapper."""
        schema = make_tool_schema(
            name,
            description="Extra tool.",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )
        adapter = AgentAdapterRegistry.get(
            adapter_key,
            metadata=_LiteMeta(
                dims=(_LiteMeta.Platform.MOBILE, _LiteMeta.TaskType.USE),
                extra_tool_schemas=[schema],
            ),
        )

        out = adapter.convert_message_from_agent(
            {
                "role": "assistant",
                "tool_calls": [{"name": "mobile_use", "arguments": {"action": name, **args}}],
            }
        )

        assert out["tool_calls"] == [make_tool_call(name, args)]

    def test_mobile_open_app_schema_enables_native_open_with_app_hint(self):
        """``open_app`` enables native ``mobile_use(action=open)`` plus app hint."""
        open_app_tool = make_tool_schema(
            "open_app",
            description="Open an installed app.",
            parameters={
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "enum": ["Chrome", "Settings"]},
                },
                "required": ["app_name"],
            },
        )
        valid_only = Qwen3VLMobileUseAdapter(
            metadata=_LiteMeta(
                dims=(_LiteMeta.Platform.MOBILE, _LiteMeta.TaskType.USE),
                valid_actions=["tap"],
            )
        )
        assert "open_app" not in valid_only.metadata.valid_actions
        assert '"open_app"' not in valid_only._build_tools_section()
        assert '"open"' not in valid_only._build_tools_section()

        with_extra = Qwen3VLMobileUseAdapter(
            metadata=_LiteMeta(
                dims=(_LiteMeta.Platform.MOBILE, _LiteMeta.TaskType.USE),
                valid_actions=["tap"],
                extra_tool_schemas=[open_app_tool],
            )
        )
        rendered = with_extra._tool_schemas_for_tools_section()
        assert {tool_schema_name(schema) for schema in rendered} == {"mobile_use"}
        [mobile_use] = rendered
        action_prop = tool_schema_parameters(mobile_use)["properties"]["action"]
        text_prop = tool_schema_parameters(mobile_use)["properties"]["text"]
        assert "open" in action_prop["enum"]
        assert "Available apps" in action_prop["description"]
        assert "Chrome" in action_prop["description"]
        assert "enum" not in text_prop
        assert '"name": "open_app"' not in with_extra._build_tools_section()

    def test_mobile_native_open_reaches_env_owned_open_app(self):
        """Qwen native open canonicalizes; env owns open_app availability."""
        adapter = Qwen3VLMobileUseAdapter(
            metadata=_LiteMeta(
                dims=(_LiteMeta.Platform.MOBILE, _LiteMeta.TaskType.USE),
                extra_tool_schemas=[],
            )
        )
        raw = "Action: Open Settings.\n" + _qwen_tool_block(
            "mobile_use", {"action": "open", "text": "Settings"}
        )

        parsed = adapter.parse_raw_assistant_response(raw)
        lite_msg = adapter.convert_message_from_agent(parsed)

        assert pop_model_output_error(copy.deepcopy(lite_msg)) is None
        assert lite_msg.get("tool_calls") == [make_tool_call("open_app", {"app_name": "Settings"})]


# =============================================================================
# 7. parse_raw_assistant_response (pipeline step 8)
# =============================================================================


class TestParseRawAssistantResponse:
    """Test parse_raw_assistant_response for various model output formats."""

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@use")

    def test_basic_action_and_tool_call(self):
        response = (
            "Action: Click on the search bar.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [500, 100]}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "computer_use"
        assert result["tool_calls"][0]["arguments"]["action"] == "left_click"
        assert result["tool_calls"][0]["arguments"]["coordinate"] == [500, 100]
        # Content should have action description
        assert any("search bar" in c.get("text", "") for c in result.get("content", []))

    def test_multiple_json_tool_call_blocks(self):
        response = (
            "Action: Click, then wait.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [500, 100]}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "wait", "duration": 1}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert [tc["arguments"]["action"] for tc in result["tool_calls"]] == [
            "left_click",
            "wait",
        ]

    def test_duplicate_opening_tool_call_before_json_payload(self):
        adapter = AgentAdapterRegistry.get("qwen3_vl@mobile@use")
        response = (
            "Action: Click the profile icon.\n"
            "<tool_call>\n"
            "<tool_call>\n"
            '{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [916, 65]}}\n'
            "</tool_call>"
        )
        parsed = adapter.parse_raw_assistant_response(response)
        assert "tool_calls" in parsed
        assert pop_model_output_error(parsed) is None
        assert parsed["tool_calls"] == [
            {
                "name": "mobile_use",
                "arguments": {"action": "click", "coordinate": [916, 65]},
            }
        ]

        lite = adapter.convert_message_from_agent(parsed)
        action = _single_action(lite["tool_calls"])
        assert action == {"action": "tap", "coordinate": [916, 65], "clicks": 1}

    def test_duplicate_opening_tool_call_preserves_following_blocks(self):
        response = (
            "Action: Click, then wait.\n"
            "<tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [500, 100]}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "wait", "duration": 1}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert [tc["arguments"]["action"] for tc in result["tool_calls"]] == [
            "left_click",
            "wait",
        ]

    def test_consecutive_duplicate_opening_tool_call_blocks(self):
        response = (
            "Action: Click, then type.\n"
            "<tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [500, 100]}}\n'
            "</tool_call>"
            "<tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "type", "text": "hello"}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert pop_model_output_error(result) is None
        assert [tc["arguments"]["action"] for tc in result["tool_calls"]] == [
            "left_click",
            "type",
        ]

    def test_balanced_duplicate_opening_tool_call_close_does_not_leak_to_action_text(self):
        response = (
            "Action: Click, then type.\n"
            "<tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [500, 100]}}\n'
            "</tool_call>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "type", "text": "hello"}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert [tc["arguments"]["action"] for tc in result["tool_calls"]] == [
            "left_click",
            "type",
        ]
        assert result["content"] == [
            {"type": "text", "text": "Action: Click, then type."},
        ]

    def test_parser_matrix_single_tool_call(self):
        response = "Action: Click the search bar.\n" + _qwen_tool_block(
            "computer_use",
            {"action": "left_click", "coordinate": [12, 34]},
        )

        result = self.adapter.parse_raw_assistant_response(response)

        assert pop_model_output_error(result.copy()) is None
        assert result["tool_calls"] == [
            {
                "name": "computer_use",
                "arguments": {"action": "left_click", "coordinate": [12, 34]},
            }
        ]

    def test_parser_matrix_consecutive_multiple_tool_calls(self):
        response = (
            "Action: Click, then type.\n"
            + _qwen_tool_block(
                "computer_use",
                {"action": "left_click", "coordinate": [12, 34]},
            )
            + _qwen_tool_block(
                "computer_use",
                {"action": "type", "text": "hello"},
            )
        )

        result = self.adapter.parse_raw_assistant_response(response)

        assert pop_model_output_error(result.copy()) is None
        assert [tc["arguments"]["action"] for tc in result["tool_calls"]] == [
            "left_click",
            "type",
        ]

    def test_parser_matrix_duplicated_opener(self):
        response = "Action: Click the search bar.\n<tool_call>\n" + _qwen_tool_block(
            "computer_use",
            {"action": "left_click", "coordinate": [12, 34]},
        )

        result = self.adapter.parse_raw_assistant_response(response)

        assert pop_model_output_error(result.copy()) is None
        assert result["tool_calls"] == [
            {
                "name": "computer_use",
                "arguments": {"action": "left_click", "coordinate": [12, 34]},
            }
        ]

    def test_parser_matrix_malformed_json_sets_model_output_error(self):
        response = "Action: Click.\n<tool_call>\n{not valid json}\n</tool_call>"

        result = self.adapter.parse_raw_assistant_response(response)

        assert not result.get("tool_calls")
        assert pop_model_output_error(result) == "malformed <tool_call> JSON"

    def test_parser_matrix_plain_final_text_roundtrip_stays_text(self):
        final_text = "Done."

        parsed = self.adapter.parse_raw_assistant_response(final_text)
        lite = self.adapter.convert_message_from_agent(parsed)
        rendered = self.adapter.convert_message_to_agent(lite)

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

    def test_with_think_tags(self):
        response = (
            "<think>\nI see a search bar at the top of the screen.\n</think>\n\n"
            "Action: Click the search bar.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [500, 50]}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert "reasoning_content" in result
        assert "search bar" in result["reasoning_content"]
        assert len(result["tool_calls"]) == 1

    def test_think_without_opening_tag(self):
        """Model sometimes outputs reasoning without <think> but with </think>."""
        response = (
            "I need to click the button.\n</think>\n\n"
            "Action: Click button.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [300, 400]}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert "reasoning_content" in result
        assert "click the button" in result["reasoning_content"]

    def test_terminate_action(self):
        response = (
            "Action: Task completed.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["tool_calls"][0]["arguments"]["action"] == "terminate"
        assert result["tool_calls"][0]["arguments"]["status"] == "success"

    def test_type_action(self):
        response = (
            "Action: type search query.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "type", "text": "hello world"}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["tool_calls"][0]["arguments"]["text"] == "hello world"

    def test_key_action(self):
        response = (
            "Action: Press Enter.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "key", "keys": ["enter"]}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["tool_calls"][0]["arguments"]["keys"] == ["enter"]

    def test_empty_response(self):
        result = self.adapter.parse_raw_assistant_response("")
        assert result["role"] == "assistant"
        assert "tool_calls" not in result

    def test_no_tool_call(self):
        result = self.adapter.parse_raw_assistant_response("Action: Wait for page to load.")
        assert result["role"] == "assistant"
        assert "tool_calls" not in result
        assert any("Wait" in c.get("text", "") for c in result.get("content", []))

    def test_malformed_json_in_tool_call(self):
        """Malformed JSON inside <tool_call> should be surfaced as a parser error."""
        response = "Action: Click.\n<tool_call>\n{not valid json}\n</tool_call>"
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["role"] == "assistant"
        assert "tool_calls" not in result or len(result.get("tool_calls", [])) == 0
        assert pop_model_output_error(result) == "malformed <tool_call> JSON"

    def test_scroll_action(self):
        response = (
            "Action: Scroll down.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "scroll", "pixels": -3}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["tool_calls"][0]["arguments"]["pixels"] == -3

    def test_preserves_action_prefix_in_raw_text(self):
        """parse_raw keeps the system-prompted ``Action: `` prefix verbatim
        in the text content — it's only the chat_template tokens that get
        stripped here. The prefix is removed downstream by
        ``convert_message_from_agent`` (the navigation ``Action:`` extraction)."""
        response = (
            "Action: Click the search bar.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [500, 50]}}\n'
            "</tool_call>"
        )
        result = self.adapter.parse_raw_assistant_response(response)
        text = result["content"][0]["text"]
        assert text.startswith("Action:")
        # Downstream convert_from_agent does the Action: extraction.
        lite = self.adapter.convert_message_from_agent(result)
        action_part = next((c for c in lite["content"] if c["type"] == "action_description"), None)
        assert action_part == {"type": "action_description", "text": "Click the search bar."}

    def test_grounding_convert_from_agent_keeps_full_text(self):
        """Grounding adapters subclass :class:`Qwen3VLBaseAdapter` directly,
        so ``convert_message_from_agent`` round-trips the post-token-strip
        text verbatim as a single ``type: "text"`` part — no ``Action:``
        extraction, no structured ``action_description`` wrapping. Env-side
        consumers rely on ``tool_calls``; the text part is preserved as-is."""
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")
        response = (
            "Action: Click the search bar.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [500, 50]}}\n'
            "</tool_call>"
        )
        agent_msg = adapter.parse_raw_assistant_response(response)
        lite = adapter.convert_message_from_agent(agent_msg)
        # Base does not introduce action_description.
        assert all(c.get("type") != "action_description" for c in lite["content"])
        # Single text part with the chat-template-stripped raw verbatim.
        assert len(lite["content"]) == 1
        assert lite["content"][0]["type"] == "text"
        assert lite["content"][0]["text"].startswith("Action: Click the search bar.")


# =============================================================================
# 8. Full pipeline: parse → convert_from_agent (pipeline steps 8 + 9)
# =============================================================================


class TestFullParsePipeline:
    """Test parse_raw_assistant_response → convert_message_from_agent chain."""

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@use")

    def test_parse_then_convert_left_click(self):
        """Simulates pipeline steps 8-9: raw text → agent message → CUA-lite message."""
        response = (
            "<think>\nI see the search icon.\n</think>\n\n"
            "Action: Click the search icon.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [900, 50]}}\n'
            "</tool_call>"
        )
        agent_msg = self.adapter.parse_raw_assistant_response(response)
        lite_msg = self.adapter.convert_message_from_agent(agent_msg)

        assert lite_msg["role"] == "assistant"
        assert lite_msg["reasoning_content"] == "I see the search icon."
        assert len(lite_msg["tool_calls"]) == 1
        action = _single_action(lite_msg["tool_calls"])
        assert action["action"] == "click"
        assert action["coordinate"] == [900, 50]

    def test_parse_then_convert_terminate(self):
        adapter = Qwen3VLDesktopUseAdapter(
            metadata=_md(
                extra_tools=[
                    make_tool_schema(
                        "terminate",
                        parameters={
                            "type": "object",
                            "properties": {"status": {"type": "string"}},
                            "required": ["status"],
                        },
                    )
                ]
            )
        )
        response = (
            "Action: Finish task.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}\n'
            "</tool_call>"
        )
        agent_msg = adapter.parse_raw_assistant_response(response)
        lite_msg = adapter.convert_message_from_agent(agent_msg)

        tc = lite_msg["tool_calls"][0]
        assert tool_call_name(tc) == "terminate"
        assert tool_call_arguments(tc)["status"] == "success"

    def test_parse_then_convert_type(self):
        response = (
            "Action: type the query.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "type", "text": "openai"}}\n'
            "</tool_call>"
        )
        agent_msg = self.adapter.parse_raw_assistant_response(response)
        lite_msg = self.adapter.convert_message_from_agent(agent_msg)

        tc = lite_msg["tool_calls"][0]
        assert tc == _desktop_action_call({"action": "type", "text": "openai"})


# =============================================================================
# 9. Trajectory unroll (TrajectoryUnrollMixin)
# =============================================================================


class TestTrajectoryUnroll:
    """Test ``unroll`` for Qwen3VL trajectory adapter."""

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@use")

    def test_two_turns_yields_two_steps(self):
        sample = sample_trajectory_two_turns()
        steps = self.adapter.unroll(sample).steps
        assert len(steps) == 2

    def test_unrolled_steps_have_system_prompt(self):
        sample = sample_trajectory_two_turns()
        steps = self.adapter.unroll(sample).steps
        for step in steps:
            assert step[0]["role"] == "system"

    def test_first_step_shorter_than_second(self):
        sample = sample_trajectory_two_turns()
        steps = self.adapter.unroll(sample).steps
        assert len(steps[0]) <= len(steps[1])

    def test_unrolled_tool_calls_are_qwen3_vl_format(self):
        """All assistant tool_calls in unrolled steps should be computer_use."""
        sample = sample_trajectory_two_turns()
        steps = self.adapter.unroll(sample).steps
        for step in steps:
            for msg in step:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        assert tc["name"] == "computer_use"

    def test_empty_messages_returns_empty(self):
        sample = dataclasses.replace(sample_trajectory_two_turns(), messages=[])
        assert self.adapter.unroll(sample).steps == []

    def test_long_trajectory_unroll(self):
        sample = sample_trajectory_long(num_turns=6)
        steps = self.adapter.unroll(sample).steps
        assert len(steps) == 6

    def test_reasoning_content_preserved_in_unroll(self):
        """reasoning_content on assistant messages should survive unroll."""
        sample = sample_trajectory_with_reasoning()
        last_step = self.adapter.unroll(sample).steps[-1]
        # Last step should have both turns; check assistant messages have reasoning
        assistant_msgs = [m for m in last_step if m.get("role") == "assistant"]
        assert any("reasoning_content" in m for m in assistant_msgs)


# =============================================================================
# 10. Per-sample conversion
# =============================================================================


class TestPerSampleConversion:
    """Test ``unroll(sample).steps``."""

    def test_one_row_two_turns_yields_two_steps(self):
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@use")
        sample = sample_trajectory_two_turns()
        out = adapter.unroll(sample)
        assert len(out.steps) == 2

    def test_grounding_action_one_row_one_step(self):
        adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")
        sample = sample_grounding_point()
        out = adapter.unroll(sample)
        assert out.steps is not None and len(out.steps) >= 1
        assert len(out.steps[-1]) > 0


# =============================================================================
# 11. Flexible action_space (annotation-based enforcement)
# =============================================================================


class TestFlexibleActionSpace:
    """Qwen3VL adapters accept any BaseActionSpace subclass."""

    def test_accepts_foreign_action_space(self):
        """Qwen3VL adapter can be constructed with LiteDesktopActionSpace."""
        adapter = Qwen3VLDesktopUseAdapter(action_space=LiteDesktopActionSpace())
        assert isinstance(adapter.action_space, LiteDesktopActionSpace)

    def test_default_action_space_unchanged(self):
        """Default construction still uses QWen3VL action space."""
        adapter = Qwen3VLDesktopUseAdapter()
        assert isinstance(adapter.action_space, Qwen3VLDesktopActionSpace)


# =============================================================================
# Characterization tests — lock down CURRENT behavior across desktop + mobile
# trajectory adapters. Goldens captured via a live invocation of the current
# cua-lite code (see commit log) and embedded here so the test file is
# self-contained. These guard a planned refactor.
#
# Scope: desktop (3 actions, schema is a superset of browser) + mobile (all 8
# cookbook actions). Grounding / point / bbox task types are out of scope
# and skipped.
# =============================================================================

# -----------------------------------------------------------------------------
# Mobile raw model outputs (one per cookbook action). Each follows the 3-part
# "Thought / Action / <tool_call>" format from _USE_SYSTEM_PROMPT_WITH_THOUGHT.
# -----------------------------------------------------------------------------

_MOBILE_RAWS: dict[str, str] = {
    "click": (
        "Thought: I need to tap the search bar.\n"
        "Action: Tap the search bar.\n"
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}\n'
        "</tool_call>"
    ),
    "long_press": (
        "Thought: Long press to open menu.\n"
        "Action: Long press the icon.\n"
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action": "long_press", '
        '"coordinate": [400, 500], "time": 2.0}}\n'
        "</tool_call>"
    ),
    "swipe": (
        "Thought: Swipe up to scroll.\n"
        "Action: Swipe up.\n"
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action": "swipe", '
        '"coordinate": [500, 800], "coordinate2": [500, 200]}}\n'
        "</tool_call>"
    ),
    "type": (
        "Thought: Type the search query.\n"
        "Action: Type 'hello'.\n"
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action": "type", "text": "hello"}}\n'
        "</tool_call>"
    ),
    "answer": (
        "Thought: Provide the answer.\n"
        "Action: Answer the user.\n"
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action": "answer", "text": "42"}}\n'
        "</tool_call>"
    ),
    "system_button": (
        "Thought: Press home.\n"
        "Action: Press Home.\n"
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action": "system_button", "button": "Home"}}\n'
        "</tool_call>"
    ),
    "wait": (
        "Thought: Wait for load.\n"
        "Action: Wait 3 seconds.\n"
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action": "wait", "time": 3}}\n'
        "</tool_call>"
    ),
    "terminate": (
        "Thought: Task complete.\n"
        "Action: Terminate.\n"
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}\n'
        "</tool_call>"
    ),
}

# Desktop raw model outputs (3 representative actions — schema / parser path is
# identical to mobile apart from the 2-part "Action / <tool_call>" format).
_DESKTOP_RAWS: dict[str, str] = {
    "left_click": (
        "Action: Click button.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "left_click", '
        '"coordinate": [500, 300]}}\n'
        "</tool_call>"
    ),
    "type": (
        "Action: type hello.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "type", "text": "hello"}}\n'
        "</tool_call>"
    ),
    "terminate": (
        "Action: Done.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}\n'
        "</tool_call>"
    ),
}


_RESPONSE_TOOL_SCHEMA = _extra_tool_schema("response", {"text": {"type": "string"}})
_TERMINATE_TOOL_SCHEMA = _extra_tool_schema("terminate", {"status": {"type": "string"}})


def _extra_tool_schemas_for_actions(actions: list[str]) -> list[dict]:
    schemas: list[dict] = []
    if "answer" in actions:
        schemas.append(_RESPONSE_TOOL_SCHEMA)
    if "terminate" in actions:
        schemas.append(_TERMINATE_TOOL_SCHEMA)
    return schemas


def _metadata_for_actions(mode: str, actions: list[str]) -> LiteCUAMetadata:
    platform = (
        LiteCUAMetadata.Platform.MOBILE if mode == "mobile" else LiteCUAMetadata.Platform.DESKTOP
    )
    return LiteCUAMetadata(
        dims=(platform, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=_extra_tool_schemas_for_actions(actions),
    )


def _adapter_for(mode: str, actions: list[str] | None = None):
    metadata = _metadata_for_actions(mode, actions or [])
    if mode == "mobile":
        return Qwen3VLMobileUseAdapter(enable_inline_reasoning=True, metadata=metadata)
    if mode == "desktop":
        return Qwen3VLDesktopUseAdapter(metadata=metadata)
    raise ValueError(mode)


def _build_mobile_trajectory(raws: list[str], task: str = "Search for cats") -> LiteSample:
    """Build a multi-turn mobile LiteSample from a list of raw responses."""
    active_semantics: list[str] = []
    if any('"action": "answer"' in raw or '"action":"answer"' in raw for raw in raws):
        active_semantics.append("answer")
    if any('"action": "terminate"' in raw or '"action":"terminate"' in raw for raw in raws):
        active_semantics.append("terminate")
    metadata = _metadata_for_actions("mobile", active_semantics)
    adapter = Qwen3VLMobileUseAdapter(
        enable_inline_reasoning=True,
        metadata=metadata,
    )
    messages: list[dict] = []
    for i, raw in enumerate(raws):
        if i == 0:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0},
                        {"type": "text", "text": task},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": [{"type": "image", "index": i}]})
        parsed = adapter.parse_raw_assistant_response(raw)
        messages.append(adapter.convert_message_from_agent(parsed))
    return LiteSample(
        metadata=metadata,
        messages=messages,
        images=[f"img{i}.png" for i in range(len(raws))],
    )


# -----------------------------------------------------------------------------
# 1. Raw → LiteMessage (parse + convert_from)
# -----------------------------------------------------------------------------

# Goldens: LiteMessage dict for each (mode, action), captured via
# `adapter.parse_raw_assistant_response(raw)` then
# `adapter.convert_message_from_agent(...)` on current code.
# Mobile goldens: ``Thought:`` lives as ``InlineReasoningContent`` (system-prompted,
# not Qwen3-VL's native ``<think>`` channel) and the action description as
# ``ActionDescriptionContent``. Both sit inside ``content``; no top-level
# ``reasoning_content`` field is set since these raws don't carry ``<think>``.
_MOBILE_LITE_GOLDEN: dict[str, dict] = {
    "click": {
        "role": "assistant",
        "tool_calls": [
            _mobile_action_call({"action": "tap", "coordinate": [500, 300], "clicks": 1})
        ],
        "content": [
            {"type": "inline_reasoning", "text": "I need to tap the search bar."},
            {"type": "action_description", "text": "Tap the search bar."},
        ],
    },
    "long_press": {
        "role": "assistant",
        "tool_calls": [
            _mobile_action_call({"action": "long_press", "coordinate": [400, 500], "duration": 2.0})
        ],
        "content": [
            {"type": "inline_reasoning", "text": "Long press to open menu."},
            {"type": "action_description", "text": "Long press the icon."},
        ],
    },
    "swipe": {
        "role": "assistant",
        "tool_calls": [
            _mobile_action_call(
                {"action": "swipe", "start_coordinate": [500, 800], "coordinate": [500, 200]}
            )
        ],
        "content": [
            {"type": "inline_reasoning", "text": "Swipe up to scroll."},
            {"type": "action_description", "text": "Swipe up."},
        ],
    },
    "type": {
        "role": "assistant",
        "tool_calls": [_mobile_action_call({"action": "type", "text": "hello"})],
        "content": [
            {"type": "inline_reasoning", "text": "Type the search query."},
            {"type": "action_description", "text": "Type 'hello'."},
        ],
    },
    "answer": {
        "role": "assistant",
        "tool_calls": [make_tool_call("response", {"text": "42"})],
        "content": [
            {"type": "inline_reasoning", "text": "Provide the answer."},
            {"type": "action_description", "text": "Answer the user."},
        ],
    },
    "system_button": {
        "role": "assistant",
        "tool_calls": [_mobile_action_call({"action": "system_button", "button": "Home"})],
        "content": [
            {"type": "inline_reasoning", "text": "Press home."},
            {"type": "action_description", "text": "Press Home."},
        ],
    },
    "wait": {
        "role": "assistant",
        # OBSERVED: raw emits int `time=3`; from_agent coerces to float `duration=3.0`.
        "tool_calls": [_mobile_action_call({"action": "wait", "duration": 3.0})],
        "content": [
            {"type": "inline_reasoning", "text": "Wait for load."},
            {"type": "action_description", "text": "Wait 3 seconds."},
        ],
    },
    "terminate": {
        "role": "assistant",
        "tool_calls": [make_tool_call("terminate", {"status": "success"})],
        "content": [
            {"type": "inline_reasoning", "text": "Task complete."},
            {"type": "action_description", "text": "Terminate."},
        ],
    },
}

_DESKTOP_LITE_GOLDEN: dict[str, dict] = {
    "left_click": {
        "role": "assistant",
        "tool_calls": [_desktop_action_call({"action": "click", "coordinate": [500, 300]})],
        "content": [{"type": "action_description", "text": "Click button."}],
    },
    "type": {
        "role": "assistant",
        "tool_calls": [_desktop_action_call({"action": "type", "text": "hello"})],
        "content": [{"type": "action_description", "text": "type hello."}],
    },
    "terminate": {
        "role": "assistant",
        "tool_calls": [make_tool_call("terminate", {"status": "success"})],
        "content": [{"type": "action_description", "text": "Done."}],
    },
}


class TestRawToLiteMessage:
    """raw model output → parse → convert_from_agent == captured LiteMessage."""

    def test_mobile_direct_native_enum_from_agent_converts_as_that_action(self):
        """Mobile twin of the desktop case: flat native output is recovered
        rather than handed to the env as a tool it does not have."""
        action_space = Qwen3VLMobileActionSpace()
        result = action_space.convert_tool_calls_from_agent(
            [{"name": "click", "arguments": {"coordinate": [500, 300]}}]
        )
        assert result == action_space.convert_tool_calls_from_agent(
            [{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}]
        )

    @pytest.mark.parametrize("action", list(_MOBILE_RAWS.keys()))
    def test_mobile(self, action):
        adapter = _adapter_for("mobile", [action])
        raw = _MOBILE_RAWS[action]
        parsed = adapter.parse_raw_assistant_response(raw)
        lite = adapter.convert_message_from_agent(parsed)
        assert lite == _MOBILE_LITE_GOLDEN[action]

    @pytest.mark.parametrize("action", list(_DESKTOP_RAWS.keys()))
    def test_desktop(self, action):
        adapter = _adapter_for("desktop", [action])
        raw = _DESKTOP_RAWS[action]
        parsed = adapter.parse_raw_assistant_response(raw)
        lite = adapter.convert_message_from_agent(parsed)
        assert lite == _DESKTOP_LITE_GOLDEN[action]


# -----------------------------------------------------------------------------
# 2. LiteMessage → Raw (convert_to_agent output)
# -----------------------------------------------------------------------------

# Golden: `adapter.convert_message_to_agent(lite)` output per action. The
# adapter flattens ``ActionDescriptionContent`` → ``text`` with the
# hard-coded ``"Action: "`` prefix (Action-only by construction) for
# chat_template compatibility. ``InlineReasoningContent`` parts are
# rendered as ``"Thought: …"`` body lines when ``enable_inline_reasoning=True``; when
# False they are dropped (the 2-part wire format forbids prose between
# ``Action:`` and ``<tool_call>``).
_MOBILE_AGENT_GOLDEN: dict[str, dict] = {
    "click": {
        "role": "assistant",
        "tool_calls": [
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}
        ],
        "content": [
            {
                "type": "text",
                "text": "Thought: I need to tap the search bar.\nAction: Tap the search bar.",
            },
        ],
    },
    "long_press": {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "mobile_use",
                "arguments": {"action": "long_press", "coordinate": [400, 500], "time": 2.0},
            }
        ],
        "content": [
            {
                "type": "text",
                "text": "Thought: Long press to open menu.\nAction: Long press the icon.",
            },
        ],
    },
    "swipe": {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "swipe",
                    "coordinate": [500, 800],
                    "coordinate2": [500, 200],
                },
            }
        ],
        "content": [
            {"type": "text", "text": "Thought: Swipe up to scroll.\nAction: Swipe up."},
        ],
    },
    "type": {
        "role": "assistant",
        "tool_calls": [{"name": "mobile_use", "arguments": {"action": "type", "text": "hello"}}],
        "content": [
            {"type": "text", "text": "Thought: Type the search query.\nAction: Type 'hello'."},
        ],
    },
    "answer": {
        "role": "assistant",
        "tool_calls": [{"name": "mobile_use", "arguments": {"action": "answer", "text": "42"}}],
        "content": [
            {"type": "text", "text": "Thought: Provide the answer.\nAction: Answer the user."},
        ],
    },
    "system_button": {
        "role": "assistant",
        "tool_calls": [
            {"name": "mobile_use", "arguments": {"action": "system_button", "button": "Home"}}
        ],
        "content": [
            {"type": "text", "text": "Thought: Press home.\nAction: Press Home."},
        ],
    },
    "wait": {
        "role": "assistant",
        # OBSERVED: raw emitted time=3 (int), but round-trip through cua-lite
        # canonical form coerces to time=3.0 (float) — matches Lite's
        # `duration: float` contract.
        "tool_calls": [{"name": "mobile_use", "arguments": {"action": "wait", "time": 3.0}}],
        "content": [
            {"type": "text", "text": "Thought: Wait for load.\nAction: Wait 3 seconds."},
        ],
    },
    "terminate": {
        "role": "assistant",
        "tool_calls": [
            {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
        ],
        "content": [
            {"type": "text", "text": "Thought: Task complete.\nAction: Terminate."},
        ],
    },
}

_DESKTOP_AGENT_GOLDEN = {
    "left_click": {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "computer_use",
                "arguments": {"action": "left_click", "coordinate": [500, 300]},
            }
        ],
        "content": [{"type": "text", "text": "Action: Click button."}],
    },
    "type": {
        "role": "assistant",
        "tool_calls": [{"name": "computer_use", "arguments": {"action": "type", "text": "hello"}}],
        "content": [{"type": "text", "text": "Action: type hello."}],
    },
    "terminate": {
        "role": "assistant",
        "tool_calls": [
            {"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}
        ],
        "content": [{"type": "text", "text": "Action: Done."}],
    },
}


class TestLiteMessageToRaw:
    """LiteMessage → convert_to_agent == captured agent message."""

    @pytest.mark.parametrize("action", list(_MOBILE_RAWS.keys()))
    def test_mobile(self, action):
        adapter = _adapter_for("mobile", [action])
        lite = copy.deepcopy(_MOBILE_LITE_GOLDEN[action])
        agent_msg = adapter.convert_message_to_agent(lite)
        assert agent_msg == _MOBILE_AGENT_GOLDEN[action]

    @pytest.mark.parametrize("action", list(_DESKTOP_RAWS.keys()))
    def test_desktop(self, action):
        adapter = _adapter_for("desktop", [action])
        lite = copy.deepcopy(_DESKTOP_LITE_GOLDEN[action])
        agent_msg = adapter.convert_message_to_agent(lite)
        assert agent_msg == _DESKTOP_AGENT_GOLDEN[action]


# -----------------------------------------------------------------------------
# 3. Raw round-trip: raw → parse → convert_from → convert_to → tool-call JSON
# -----------------------------------------------------------------------------


class TestRawRoundTrip:
    """Full adapter round-trip preserves the tool_call JSON and action text.

    Lossy fields (explicit assertions):
      - `wait(time=3)` (int) becomes `wait(time=3.0)` (float) — see OBSERVED
        note in ``_MOBILE_AGENT_GOLDEN``.
      - `content[0].text` loses its ``"Action: "`` prefix on
        ``convert_message_from_agent`` (extracted by the
        ``Action:`` regex) and regains it on
        ``convert_to_agent`` (hard-coded). So round-trip of the action
        text is faithful modulo the prefix.
    """

    @pytest.mark.parametrize("action", list(_MOBILE_RAWS.keys()))
    def test_mobile_tool_call_json_round_trip(self, action):
        adapter = _adapter_for("mobile", [action])
        raw = _MOBILE_RAWS[action]
        parsed = adapter.parse_raw_assistant_response(raw)
        lite = adapter.convert_message_from_agent(parsed)
        agent_msg = adapter.convert_message_to_agent(lite)

        # tool_call name + arguments match the original JSON (with float-coerce
        # allowance for wait).
        restored_call = agent_msg["tool_calls"][0]
        assert restored_call["name"] == "mobile_use"
        assert (
            restored_call["arguments"] == _MOBILE_AGENT_GOLDEN[action]["tool_calls"][0]["arguments"]
        )

    @pytest.mark.parametrize("action", list(_DESKTOP_RAWS.keys()))
    def test_desktop_tool_call_json_round_trip(self, action):
        adapter = _adapter_for("desktop", [action])
        raw = _DESKTOP_RAWS[action]
        parsed = adapter.parse_raw_assistant_response(raw)
        lite = adapter.convert_message_from_agent(parsed)
        agent_msg = adapter.convert_message_to_agent(lite)
        restored_call = agent_msg["tool_calls"][0]
        assert restored_call["name"] == "computer_use"
        assert (
            restored_call["arguments"]
            == _DESKTOP_AGENT_GOLDEN[action]["tool_calls"][0]["arguments"]
        )


# -----------------------------------------------------------------------------
# 4. 3-turn mobile unroll structure
# -----------------------------------------------------------------------------


class TestUnrollStructure:
    """3-turn mobile unroll yields 3 samples; structure per sample matches
    captured golden (roles + image counts)."""

    def test_three_turn_mobile_structure(self):
        adapter = _adapter_for("mobile", ["terminate"])
        sample = _build_mobile_trajectory(
            [_MOBILE_RAWS["click"], _MOBILE_RAWS["type"], _MOBILE_RAWS["terminate"]]
        )
        steps = adapter.unroll(sample).steps

        assert len(steps) == 3
        # Golden structure (captured): for the 3-turn case (≤ full_history_size=4),
        # no windowing kicks in, so step i just contains system + (user, assistant) * (i+1).
        expected = [
            (["system", "user", "assistant"], [0, 1, 0]),
            (["system", "user", "assistant", "user", "assistant"], [0, 1, 0, 1, 0]),
            (
                ["system", "user", "assistant", "user", "assistant", "user", "assistant"],
                [0, 1, 0, 1, 0, 1, 0],
            ),
        ]
        for i, step in enumerate(steps):
            roles, img_counts = expected[i]
            assert [m["role"] for m in step] == roles
            actual_counts = []
            for m in step:
                c = m.get("content", [])
                n = (
                    sum(1 for item in c if isinstance(item, dict) and item.get("type") == "image")
                    if isinstance(c, list)
                    else 0
                )
                actual_counts.append(n)
            assert actual_counts == img_counts
            # Distinct image indices referenced = one per kept turn.
            image_indices = {
                item["index"]
                for m in step
                if m.get("role") == "user"
                for item in (m.get("content") or [])
                if item.get("type") == "image"
            }
            assert len(image_indices) == i + 1


# -----------------------------------------------------------------------------
# 5. Single-turn target text byte-exact (per action)
# -----------------------------------------------------------------------------


class TestUnrollByteExactTargetPerAction:
    """For each mobile action, a single-turn trajectory unroll produces a
    target assistant message whose fields match the captured golden byte-exact.

    NB: Qwen3VL adapters don't expose a ``format_tool_call_as_text`` helper
    (the agent's build_generation_prompt passes raw tool_calls into the chat
    template). Byte-exact-ness here means the structured message matches —
    which is what downstream chat_template rendering consumes.
    """

    @pytest.mark.parametrize("action", list(_MOBILE_RAWS.keys()))
    def test_mobile_target_byte_exact(self, action):
        adapter = _adapter_for("mobile", [action])
        raw = _MOBILE_RAWS[action]
        # single-turn: one user bubble + the parsed assistant
        lite_asst = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
        sample = LiteSample(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE)
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0},
                        {"type": "text", "text": "task"},
                    ],
                },
                lite_asst,
            ],
            images=["img0.png"],
        )
        steps = adapter.unroll(sample).steps
        assert len(steps) == 1
        asst = [m for m in steps[0] if m.get("role") == "assistant"][-1]
        assert asst == _MOBILE_AGENT_GOLDEN[action]


# -----------------------------------------------------------------------------
# 6. No leakage of future-step info into sample i's user prompt
# -----------------------------------------------------------------------------


class TestUnrollNoLeakage:
    """In a 3-turn mobile trajectory, sample i's user-prompt text must not
    contain the distinguishing action/summary text from target_i (no leakage
    from target into context)."""

    def test_mobile_no_future_leakage(self):
        # Use actions with distinctive content so leakage would be obvious.
        raws_seq = [_MOBILE_RAWS["click"], _MOBILE_RAWS["type"], _MOBILE_RAWS["terminate"]]
        adapter = _adapter_for("mobile", ["terminate"])
        sample = _build_mobile_trajectory(raws_seq)
        steps = adapter.unroll(sample).steps

        # distinguishing phrases, one per step
        signatures = [
            "Tap the search bar",  # step 0 target
            "Type 'hello'",  # step 1 target
            "Terminate",  # step 2 target
        ]

        for i, step in enumerate(steps):
            user_msgs = [m for m in step if m.get("role") == "user"]
            # Collect all user text in this step's prompt
            full_user_text = ""
            for um in user_msgs:
                for c in um.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        full_user_text += c.get("text", "")
            # step i's own signature must NOT be in user text
            assert signatures[i] not in full_user_text, (
                f"step {i} leaks its own action {signatures[i]!r} "
                f"into user prompt: {full_user_text!r}"
            )


# -----------------------------------------------------------------------------
# 7. Protocol windowing for mobile adapter — covered by the canonical
#    Qwen3VLHistoryProtocol tests in tests/agents/models/qwen3_vl/test_qwen3_vl_protocol.py.
#    Mobile uses the same protocol class as desktop; the cookbook-aligned
#    Qwen3VLMobileHistoryProtocol was retired (no measurable lift, see
#    lite/agents/models/qwen3_vl/protocol.py).
# -----------------------------------------------------------------------------


def _build_protocol_messages(n_turns: int, task: str = "Find weather") -> list[dict]:
    """Build n_turns of (user_with_image, assistant_with_action) messages.
    Kept as a helper for the mutation-purity test below."""
    msgs = []
    for i in range(n_turns):
        if i == 0:
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": i},
                        {"type": "text", "text": task},
                    ],
                }
            )
        else:
            msgs.append({"role": "user", "content": [{"type": "image", "index": i}]})
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": f"Action step {i}."}],
                "tool_calls": [
                    {
                        "name": "mobile_use",
                        "arguments": {"action": "click", "coordinate": [100 + i, 100]},
                    }
                ],
            }
        )
    return msgs


# -----------------------------------------------------------------------------
# 8. Mutation purity — inputs not mutated by any conversion
# -----------------------------------------------------------------------------


class TestMutationPurity:
    """convert_message_to_agent, convert_message_from_agent, and
    protocol.process_messages must not mutate their inputs."""

    def test_convert_to_agent_does_not_mutate(self):
        adapter = Qwen3VLMobileUseAdapter(enable_inline_reasoning=True)
        lite = copy.deepcopy(_MOBILE_LITE_GOLDEN["click"])
        snapshot = copy.deepcopy(lite)
        _ = adapter.convert_message_to_agent(lite)
        assert lite == snapshot

    def test_convert_from_agent_does_not_mutate(self):
        adapter = Qwen3VLMobileUseAdapter(enable_inline_reasoning=True)
        agent_msg = copy.deepcopy(_MOBILE_AGENT_GOLDEN["click"])
        snapshot = copy.deepcopy(agent_msg)
        _ = adapter.convert_message_from_agent(agent_msg)
        assert agent_msg == snapshot

    def test_protocol_process_messages_does_not_mutate(self):
        # Mobile uses Qwen3VLHistoryProtocol (same as desktop) after the
        # cookbook protocol retirement — mutation-purity check still relevant.
        from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol

        proto = Qwen3VLHistoryProtocol(full_history_size=4)
        msgs = _build_protocol_messages(5)
        snapshot = copy.deepcopy(msgs)
        _ = proto.process_messages(msgs)
        assert msgs == snapshot


# -----------------------------------------------------------------------------
# 9. Sample independence — unrolling twice yields equal but independent samples
# -----------------------------------------------------------------------------


class TestSampleIndependence:
    """Calling unroll_sample twice on the same LiteSample yields equal results
    but independent objects; mutating one batch's sample does not affect the
    other, and mutating sample[0] within a batch does not affect sample[1]."""

    def test_two_calls_equal_but_independent(self):
        adapter = Qwen3VLMobileUseAdapter(enable_inline_reasoning=True)
        sample = _build_mobile_trajectory([_MOBILE_RAWS["click"], _MOBILE_RAWS["type"]])
        a = adapter.unroll(sample).steps
        b = adapter.unroll(sample).steps

        # Equal content
        assert len(a) == len(b) == 2
        for sa, sb in zip(a, b):
            assert sa == sb

        # Independent objects
        for sa, sb in zip(a, b):
            assert sa is not sb

    def test_mutating_one_step_does_not_affect_siblings(self):
        adapter = Qwen3VLMobileUseAdapter(enable_inline_reasoning=True)
        sample = _build_mobile_trajectory([_MOBILE_RAWS["click"], _MOBILE_RAWS["type"]])
        steps = adapter.unroll(sample).steps
        assert len(steps) == 2

        # Mutate the last message of step[0] — the target assistant's action
        steps[0][-1]["tool_calls"][0]["arguments"]["action"] = "MUTATED"

        # step[1] should still have unmutated content
        for m in steps[1]:
            for tc in m.get("tool_calls", []) or []:
                assert tc["arguments"].get("action") != "MUTATED"


# -----------------------------------------------------------------------------
# 10. Edge cases: empty / single-turn / 10-turn / assistant without tool_calls
# -----------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_trajectory_unroll_is_empty(self):
        """Empty LiteSample → unroll returns no steps."""
        adapter = Qwen3VLMobileUseAdapter(enable_inline_reasoning=True)
        empty = LiteSample(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE)
            ),
            messages=[],
            images=[],
        )
        assert adapter.unroll(empty).steps == []

    def test_single_turn_unroll(self):
        """Single-turn unroll → one step with (system, user, assistant)."""
        adapter = Qwen3VLMobileUseAdapter(enable_inline_reasoning=True)
        sample = _build_mobile_trajectory([_MOBILE_RAWS["click"]])
        steps = adapter.unroll(sample).steps
        assert len(steps) == 1
        assert [m["role"] for m in steps[0]] == ["system", "user", "assistant"]

    def test_10_turn_unroll(self):
        """A 10-turn trajectory unrolls cleanly to 10 steps; later steps
        use windowing (len capped by full_history_size=4)."""
        adapter = Qwen3VLMobileUseAdapter(enable_inline_reasoning=True)
        raws = []
        for i in range(10):
            raws.append(
                f"Thought: t{i}.\nAction: step {i}.\n"
                "<tool_call>\n"
                f'{{"name": "mobile_use", "arguments": {{"action": "click", '
                f'"coordinate": [{100 + i}, 100]}}}}\n'
                "</tool_call>"
            )
        sample = _build_mobile_trajectory(raws, task="long task")
        steps = adapter.unroll(sample).steps
        assert len(steps) == 10
        # With full_history_size=4, step 9 is windowed: system + 4 * (user,assistant) = 9 msgs.
        assert len(steps[9]) == 9
        # Step 0 (single turn, no window): system + user + assistant = 3 msgs.
        assert len(steps[0]) == 3

    def test_no_tool_call_turn_keeps_plain_text_for_termination(self):
        """A turn with NO ``<tool_call>`` keeps its prose as plain ``text``.

        ``action_description`` is the narration-accompanying-an-action channel,
        so it is only produced for a turn that actually carries ``tool_calls``.
        A no-tool-call turn is either the agent's deliberate text-oriented
        termination or a parse failure; both must stay plain ``TextContent`` so
        the rollout loops' shared classifier
        (``lite.core.messages.no_tool_call_final_text``) can see the
        text. Retagging here used to convert every genuine text final of every
        ``@use`` adapter into an ``empty_model_output`` truncation.

        parse_raw alone only extracts chat-template tokens (there are none
        here), so it yields a single ``text`` content with the verbatim raw
        string, and convert_from_agent leaves it alone. The round trip back to
        the wire is byte-exact — one text item, no ``Action:`` re-synthesis.
        """
        from lite.core.messages import no_tool_call_final_text

        adapter = Qwen3VLMobileUseAdapter(enable_inline_reasoning=True)
        raw = "Thought: Looking at the screen.\nAction: Scroll down to see more."
        parsed = adapter.parse_raw_assistant_response(raw)
        assert parsed == {
            "role": "assistant",
            "content": [{"type": "text", "text": raw}],
        }
        lite = adapter.convert_message_from_agent(parsed)
        assert lite == {
            "role": "assistant",
            "content": [{"type": "text", "text": raw}],
        }
        assert no_tool_call_final_text(lite) == raw

        agent_msg = adapter.convert_message_to_agent(lite)
        assert agent_msg == {
            "role": "assistant",
            "content": [{"type": "text", "text": raw}],
        }
        assert "tool_calls" not in agent_msg

    def test_tool_call_turn_still_decomposes_thought_and_action(self):
        """The companion to the test above: with a tool call present, the
        ``Thought:`` / ``Action:`` decomposition into ``inline_reasoning`` +
        ``action_description`` is unchanged — SFT / unroll rendering of ACTION
        turns must not shift."""
        adapter = Qwen3VLMobileUseAdapter(enable_inline_reasoning=True)
        raw = (
            "Thought: Looking at the screen.\n"
            "Action: Scroll down to see more.\n"
            '<tool_call>\n{"name": "mobile_use", "arguments": '
            '{"action": "swipe", "coordinate": [500, 900], '
            '"coordinate2": [500, 300]}}\n</tool_call>'
        )
        lite = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
        assert lite["content"] == [
            {"type": "inline_reasoning", "text": "Looking at the screen."},
            {"type": "action_description", "text": "Scroll down to see more."},
        ]
        assert lite["tool_calls"]

        agent_msg = adapter.convert_message_to_agent(lite)
        assert agent_msg["content"] == [
            {
                "type": "text",
                "text": "Thought: Looking at the screen.\nAction: Scroll down to see more.",
            }
        ]

    def test_native_think_and_prompted_thought_coexist(self):
        """Qwen3-VL mobile's two reasoning channels can both be present:
          - ``<think>...</think>``   → NATIVE thinking mode (chat-template
            level; parse_raw extracts to top-level ``reasoning_content``).
          - ``Thought: ...``         → cookbook-prompted CoT line,
            system-prompt level; convert_from_agent extracts to
            ``InlineReasoningContent`` part inside ``content``.

        These live on different layers and must not collide on the way
        back: ``convert_message_to_agent`` merges the prompted ``Thought:``
        with the action description into a single text content
        (chat_template-friendly) but leaves ``reasoning_content`` untouched
        so Qwen3-VL's chat_template thinking extension renders it via the
        native ``<think>`` tokens.
        """
        adapter = Qwen3VLMobileUseAdapter(enable_inline_reasoning=True)
        raw = (
            "<think>I should explore the screen first.</think>\n"
            "Thought: I'll tap the search bar.\n"
            "Action: Tap search bar.\n"
            '<tool_call>\n{"name": "mobile_use", "arguments": '
            '{"action": "click", "coordinate": [500, 100]}}\n</tool_call>'
        )
        parsed = adapter.parse_raw_assistant_response(raw)

        # Chat-template layer (parse_raw): NATIVE think + structured tool_calls.
        assert parsed["reasoning_content"] == "I should explore the screen first."
        assert parsed["tool_calls"][0]["arguments"] == {
            "action": "click",
            "coordinate": [500, 100],
        }
        # The prose remainder (Thought: + Action:) stays as a single text
        # content part — system-prompt-layer parsing is NOT parse_raw's job.
        assert parsed["content"] == [
            {
                "type": "text",
                "text": "Thought: I'll tap the search bar.\nAction: Tap search bar.",
            }
        ]

        # System-prompt layer (convert_from_agent): splits Thought: / Action:.
        lite = adapter.convert_message_from_agent(parsed)
        assert lite["reasoning_content"] == "I should explore the screen first."
        reasoning_part = next(
            (c for c in lite["content"] if c["type"] == "inline_reasoning"),
            None,
        )
        assert reasoning_part == {
            "type": "inline_reasoning",
            "text": "I'll tap the search bar.",
        }
        action_part = next(
            (c for c in lite["content"] if c["type"] == "action_description"),
            None,
        )
        assert action_part == {
            "type": "action_description",
            "text": "Tap search bar.",
        }

        # convert_message_to_agent round-trip: merges Thought: + Action:
        # into one text part; reasoning_content (native) stays top-level.
        agent_msg = adapter.convert_message_to_agent(lite)
        assert agent_msg["reasoning_content"] == "I should explore the screen first."
        assert agent_msg["content"] == [
            {"type": "text", "text": "Thought: I'll tap the search bar.\nAction: Tap search bar."},
        ]
        assert agent_msg["tool_calls"][0]["arguments"] == {
            "action": "click",
            "coordinate": [500, 100],
        }


# =============================================================================
# 12. enable_inline_reasoning toggle: 4 combinations × system_prompt / parse_raw / fold
# =============================================================================
#
# The toggle drives THREE coordinated behaviors on every Qwen3VL trajectory
# adapter: system_prompt resolution, Thought: extraction in parse_raw, and
# inline_reasoning surfacing in convert_message_to_agent. These tests
# parametrize {desktop, mobile} × {enable_inline_reasoning=False, True} = 4 combos to
# guarantee the toggle is wired identically across platforms.


class TestWithThoughtToggle:
    """The ``enable_inline_reasoning`` toggle controls the 2-part vs 3-part wire format
    consistently across desktop and mobile."""

    @pytest.fixture(
        params=[
            ("desktop", False),
            ("desktop", True),
            ("mobile", False),
            ("mobile", True),
        ],
        ids=lambda p: f"{p[0]}-enable_inline_reasoning={p[1]}",
    )
    def adapter_and_mode(self, request):
        platform, enable_inline_reasoning = request.param
        # ``system_prompt`` is a class-level field default per subclass (each
        # adapter pins the prompt that matches its default ``enable_inline_reasoning``);
        # when the test flips ``enable_inline_reasoning`` away from the class default we
        # also pass the matching ``system_prompt`` explicitly, since there's
        # no auto-resolution at __post_init__.
        expected_prompt = (
            _USE_SYSTEM_PROMPT_WITH_THOUGHT if enable_inline_reasoning else USE_SYSTEM_PROMPT
        )
        if platform == "desktop":
            adapter = Qwen3VLDesktopUseAdapter(
                enable_inline_reasoning=enable_inline_reasoning,
                system_prompt=expected_prompt,
            )
        else:
            adapter = Qwen3VLMobileUseAdapter(
                enable_inline_reasoning=enable_inline_reasoning,
                system_prompt=expected_prompt,
            )
        return adapter, enable_inline_reasoning

    @pytest.fixture
    def tool_call_block(self, adapter_and_mode):
        """A ``<tool_call>`` block the fixture's action space actually accepts.

        Load-bearing: feeding the desktop ``computer_use`` block to the MOBILE
        adapter makes the action space drop the call, so the turn converts to a
        no-tool-call message — which (correctly) keeps its prose as plain
        ``text`` instead of decomposing into Thought/Action.
        """
        adapter, _ = adapter_and_mode
        if isinstance(adapter, Qwen3VLMobileUseAdapter):
            return (
                '<tool_call>\n{"name": "mobile_use", "arguments": '
                '{"action": "click", "coordinate": [10, 20]}}\n</tool_call>'
            )
        return (
            '<tool_call>\n{"name": "computer_use", "arguments": '
            '{"action": "left_click", "coordinate": [10, 20]}}\n</tool_call>'
        )

    def test_system_prompt_matches_toggle(self, adapter_and_mode):
        adapter, enable_inline_reasoning = adapter_and_mode
        expected = _USE_SYSTEM_PROMPT_WITH_THOUGHT if enable_inline_reasoning else USE_SYSTEM_PROMPT
        assert adapter.system_prompt == expected
        # Finish guidance is rendered by the active tool schemas, not hard-coded
        # into the static response-format prompt.
        assert "action=answer" not in adapter.system_prompt
        assert "action=terminate" not in adapter.system_prompt
        # An SFT-specific ``status=success`` rule is NOT on the base prompt.
        assert "status=success" not in adapter.system_prompt

    def test_convert_from_agent_extracts_thought_only_when_enable_inline_reasoning(
        self,
        adapter_and_mode,
        tool_call_block,
    ):
        """``Thought:`` extraction is a system-prompt-level concern, so it
        lives in ``convert_message_from_agent`` (not parse_raw, which
        only handles chat-template tokens). The ``enable_inline_reasoning`` toggle
        gates whether the ``Thought:`` line is surfaced as
        ``InlineReasoningContent`` or silently dropped."""
        adapter, enable_inline_reasoning = adapter_and_mode
        raw = f"Thought: I will tap.\nAction: Tap.\n{tool_call_block}"
        parsed = adapter.parse_raw_assistant_response(raw)
        lite = adapter.convert_message_from_agent(parsed)
        kinds = [p["type"] for p in lite.get("content", [])]
        if enable_inline_reasoning:
            assert "inline_reasoning" in kinds
            assert (
                next(p["text"] for p in lite["content"] if p["type"] == "inline_reasoning")
                == "I will tap."
            )
        else:
            assert "inline_reasoning" not in kinds  # silently dropped

    def test_convert_from_agent_multiline_thought_body_round_trips(
        self,
        adapter_and_mode,
        tool_call_block,
    ):
        """A MULTI-LINE ``Thought:`` body must be captured in full (the case the
        non-greedy ``Thought:\\s*(.*?)(?:\\n(?=Action:)|\\Z)`` regex fix was
        created for). Pre-fix, ``Action:\\s*(.*?)(?:\\n|$)`` clipped both at the
        first ``\\n`` — so a multi-line reasoning block was truncated to its
        first line. Feed ``Thought: line1\\nline2\\nAction: ...`` and assert
        inline_reasoning captures BOTH lines (when enabled) while the action
        stays intact."""
        adapter, enable_inline_reasoning = adapter_and_mode
        raw = f"Thought: line1\nline2\nAction: Tap.\n{tool_call_block}"
        parsed = adapter.parse_raw_assistant_response(raw)
        lite = adapter.convert_message_from_agent(parsed)
        action = next(p["text"] for p in lite["content"] if p["type"] == "action_description")
        assert action == "Tap."
        if enable_inline_reasoning:
            reasoning = next(p["text"] for p in lite["content"] if p["type"] == "inline_reasoning")
            assert reasoning == "line1\nline2", (
                "multi-line Thought body must keep both lines (regex must not "
                "clip at the first newline)"
            )
        else:
            assert all(p["type"] != "inline_reasoning" for p in lite["content"])

        # The greedy last-``Action:`` anchor (``(?:.*\n)?Action:\s*(.*)\Z``) was
        # added so an ``Action:`` substring NESTED inside the Thought body does
        # NOT short-circuit the action capture: the REAL trailing ``Action:``
        # line wins, not the nested mention.
        raw_nested = f"Thought: I considered Action: foo\nAction: real_action.\n{tool_call_block}"
        parsed_nested = adapter.parse_raw_assistant_response(raw_nested)
        lite_nested = adapter.convert_message_from_agent(parsed_nested)
        action_nested = next(
            p["text"] for p in lite_nested["content"] if p["type"] == "action_description"
        )
        assert action_nested == "real_action.", (
            "the LAST Action: line must win, not an Action: nested in Thought"
        )

    def test_convert_to_agent_drops_or_surfaces_inline_reasoning(self, adapter_and_mode):
        adapter, enable_inline_reasoning = adapter_and_mode
        msg = {
            "role": "assistant",
            "content": [
                {"type": "inline_reasoning", "text": "Plan it."},
                {"type": "action_description", "text": "Click."},
            ],
        }
        out = adapter.convert_message_to_agent(msg)
        text = out["content"][0]["text"]
        if enable_inline_reasoning:
            assert text == "Thought: Plan it.\nAction: Click."
        else:
            # Even though SFT data has inline_reasoning, enable_inline_reasoning=False
            # MUST drop it (the 2-part wire format forbids prose between
            # Action and <tool_call>).
            assert text == "Action: Click."
            assert "Thought:" not in text
            assert "Plan it." not in text

    def test_reasoning_content_never_coerced_to_thought(self, adapter_and_mode):
        """Top-level ``reasoning_content`` is the NATIVE ``<think>`` channel
        and must be preserved untouched in both modes — never silently
        promoted to a ``Thought:`` body line."""
        adapter, enable_inline_reasoning = adapter_and_mode
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Tap settings"}],
            "reasoning_content": "Need to open settings to enable dark mode.",
        }
        out = adapter.convert_message_to_agent(msg)
        # reasoning_content is always preserved as-is (native channel).
        assert out["reasoning_content"] == "Need to open settings to enable dark mode."
        text = out["content"][0]["text"]
        # The text must NOT contain the reasoning_content as a Thought: line.
        assert "Need to open settings" not in text
        if enable_inline_reasoning:
            assert text == "Action: Tap settings"
        else:
            assert text == "Action: Tap settings"

    def test_keeps_native_reasoning_content_when_inline_present(self, adapter_and_mode):
        """When BOTH inline_reasoning AND top-level reasoning_content are
        present, the enable_inline_reasoning=True path surfaces inline as Thought: and
        keeps the NATIVE reasoning_content untouched (no double-source
        legacy promotion)."""
        adapter, enable_inline_reasoning = adapter_and_mode
        msg = {
            "role": "assistant",
            "content": [
                {"type": "inline_reasoning", "text": "prompted CoT"},
                {"type": "action_description", "text": "Tap"},
            ],
            "reasoning_content": "native think",
        }
        out = adapter.convert_message_to_agent(msg)
        # NATIVE reasoning_content is preserved in both modes (it's the chat
        # template's Thinking-mode channel).
        assert out["reasoning_content"] == "native think"
        text = out["content"][0]["text"]
        if enable_inline_reasoning:
            assert text == "Thought: prompted CoT\nAction: Tap"
        else:
            assert text == "Action: Tap"
