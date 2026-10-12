"""
Tests for Step-GUI adapter + agent integration.

Covers:
  1. Registry resolution
  2. System prompt keeps Step-GUI key markers and cua-lite INFO semantics
  3. parse_raw_assistant_response intermediate shape + convert_message_from_agent:
     <THINK> tolerances, kv parsing, missing THINK
  4. _parse_kv_to_action / _action_dict_to_tool_call field order
  5. format_agent_tool_call_as_wire_text (renames return_value -> return,
     joins with \t)
  6. Direct parse -> convert_from -> convert_to -> format round-trip preserves
     `summary:` (SFT-format fidelity)
  7. Lossy round-trip through cua-lite canonical form (summary dropped)
  8. STEPGUIMobileAgent.build_generation_prompt — <THINK> + tab-separated fold
  9. No tools= kwarg passed to apply_chat_template

Run:
    uv run pytest tests/agents/models/step_gui/test_step_gui_adapter.py -v
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from lite.agents.core.action_space.base import BaseActionSpace, LiteMobileActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent import AgentRegistry
from lite.agents.models.step_gui.action_space import STEPGUIMobileActionSpace
from lite.agents.models.step_gui.adapter import (
    STEP_GUI_MOBILE_SYS_PROMPT,
    STEPGUIMobileUseAdapter,
)
from lite.agents.models.step_gui.agent import STEPGUIMobileAgent
from lite.agents.models.step_gui.protocol import StepGUIHistoryProtocol
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call
from lite.core.tools.action_space import (
    lite_action_batch_child_name_errors,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet, make_open_app_tool
from lite.core.tools.schemas import tool_schema_name

_STEP_EXTRA_TOOL_SCHEMAS = [
    make_open_app_tool(["微信", "Settings"]),
    LiteFinishToolSet.get_tool_schema("response"),
    LiteFinishToolSet.get_tool_schema("terminate"),
]


def _step_metadata() -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=list(_STEP_EXTRA_TOOL_SCHEMAS),
    )


def _step_adapter() -> STEPGUIMobileUseAdapter:
    return STEPGUIMobileUseAdapter(metadata=_step_metadata())


def _single_mobile_action(tool_call: dict[str, Any]) -> dict[str, Any]:
    assert tool_call_name(tool_call) == "mobile"
    actions = tool_call_arguments(tool_call)["actions"]
    assert len(actions) == 1
    return actions[0]


def _mobile_call(action: str, **arguments: Any) -> dict[str, Any]:
    return make_tool_call("mobile", {"actions": [{"action": action, **arguments}]})


# =============================================================================
# 1. Registry resolution
# =============================================================================

class TestRegistry:
    def test_adapter_registry(self):
        adapter = AgentAdapterRegistry.get("step_gui@mobile@use")
        assert isinstance(adapter, STEPGUIMobileUseAdapter)

    def test_agent_registry(self):
        agent = AgentRegistry.get(
            "step_gui@mobile@use", generate_fn=lambda **kw: {"response": ""},
        )
        assert isinstance(agent, STEPGUIMobileAgent)
        assert isinstance(agent.adapter, STEPGUIMobileUseAdapter)
        assert isinstance(agent.adapter.action_space, STEPGUIMobileActionSpace)
        assert isinstance(agent.adapter.protocol, StepGUIHistoryProtocol)

    def test_default_system_prompt(self):
        adapter = _step_adapter()
        assert adapter.system_prompt == STEP_GUI_MOBILE_SYS_PROMPT


# =============================================================================
# 2. System prompt key markers
# =============================================================================

class TestSystemPrompt:
    def test_contains_chinese_role_header(self):
        """First sentence of gelab-zero's task_define_prompt."""
        assert "你是一个手机 GUI-Agent 操作专家" in STEP_GUI_MOBILE_SYS_PROMPT

    def test_contains_all_9_action_examples(self):
        """Every action's `例如：action:XXX` example line must be present."""
        for verb in ["CLICK", "TYPE", "COMPLETE", "WAIT", "AWAKE",
                    "INFO", "ABORT", "SLIDE", "LONGPRESS"]:
            assert f"action:{verb}" in STEP_GUI_MOBILE_SYS_PROMPT, verb

    def test_coordinate_range_declared(self):
        """Coordinate system ([0, 1000]) must be stated — models learned this."""
        assert "0-1000" in STEP_GUI_MOBILE_SYS_PROMPT

    def test_info_prompt_describes_final_answer_not_question(self):
        assert (
            "INFO：向用户提交最终回答，需包含回答内容 value。"
            in STEP_GUI_MOBILE_SYS_PROMPT
        )
        assert "action:INFO\tvalue:最终回答内容" in STEP_GUI_MOBILE_SYS_PROMPT
        assert "询问用户问题" not in STEP_GUI_MOBILE_SYS_PROMPT
        assert "提问内容" not in STEP_GUI_MOBILE_SYS_PROMPT

    @staticmethod
    def _rendered_system_prompt(adapter: STEPGUIMobileUseAdapter) -> str:
        sample = _build_trajectory_sample()
        step = adapter.unroll(sample).steps[0]
        assert step[0]["role"] == "system"
        return step[0]["content"][0]["text"]

    def test_full_finish_metadata_renders_full_prompt(self):
        text = self._rendered_system_prompt(_step_adapter())
        assert text == STEP_GUI_MOBILE_SYS_PROMPT

    def test_rendered_prompt_is_identical_across_active_extra_tools(self):
        """The action rows are SFT-trained prompt text, so no combination of
        active extra tools may change a single byte of them.

        IDENTITY, not existence: the defect this replaces advertised
        ``以下5类操作`` with ``COMPLETE``/``ABORT``/``INFO``/``AWAKE`` deleted
        whenever the sample carried no matching extra-tool schema — which is the
        state every one of these envs defaults to (``extra_tools: []``). A
        containment assertion (``"action:CLICK" in text``) passes on both the
        full and the truncated prompt, so it cannot see that.
        """
        renders = {}
        for size in range(len(_STEP_EXTRA_TOOL_SCHEMAS) + 1):
            for combination in itertools.combinations(_STEP_EXTRA_TOOL_SCHEMAS, size):
                adapter = STEPGUIMobileUseAdapter(
                    metadata=LiteCUAMetadata(
                        dims=("mobile", "use"),
                        extra_tool_schemas=list(combination),
                    ),
                )
                names = tuple(sorted(tool_schema_name(s) for s in combination))
                renders[names] = self._rendered_system_prompt(adapter)

        assert set(renders.values()) == {STEP_GUI_MOBILE_SYS_PROMPT}, {
            names: text for names, text in renders.items()
            if text != STEP_GUI_MOBILE_SYS_PROMPT
        }
        # Negative half, stated explicitly: every finish/app row the extras gate
        # used to delete is present with NO extra tool active.
        no_extras = renders[()]
        assert "以下9类操作" in no_extras
        for verb in ["CLICK", "TYPE", "COMPLETE", "WAIT", "AWAKE",
                     "INFO", "ABORT", "SLIDE", "LONGPRESS"]:
            assert f"action:{verb}" in no_extras, verb

    @pytest.mark.parametrize(
        "valid_actions", [None, [], ["tap"], ["tap", "type"]],
    )
    def test_rendered_prompt_is_identical_across_valid_actions(self, valid_actions):
        """``valid_actions`` does not reach the prompt either.

        These rows are the tool surface itself, not a rendered schema, so the
        GUI gate narrows what the env ADMITS, never what the model is shown.
        The defect this replaces rendered ``以下4类操作`` under ``valid_actions=[]``
        — every GUI row deleted — and ``以下6类操作`` under ``["tap", "type"]``.
        """
        adapter = STEPGUIMobileUseAdapter(
            metadata=LiteCUAMetadata(
                dims=("mobile", "use"),
                valid_actions=valid_actions,
                extra_tool_schemas=list(_STEP_EXTRA_TOOL_SCHEMAS),
            ),
        )
        assert self._rendered_system_prompt(adapter) == STEP_GUI_MOBILE_SYS_PROMPT


# =============================================================================
# 3. parse_raw_assistant_response + convert_message_from_agent
# =============================================================================

class TestParseRawIntermediate:
    """Sanity check: parse_raw_assistant_response returns a trivial wrapper
    ``{role: "assistant", content: [{type: "text", text: raw}]}`` — all
    parsing (THINK, kv) lives in convert_message_from_agent."""

    def test_parse_raw_returns_trivial_text_wrapper(self):
        adapter = _step_adapter()
        raw = "<THINK> x </THINK>\nexplain:e\taction:CLICK\tpoint:1,1\tsummary:s"
        msg = adapter.parse_raw_assistant_response(raw)
        assert msg == {
            "role": "assistant",
            "content": [{"type": "text", "text": raw}],
        }


class TestParseResponse:
    """Tests that assert on the FINAL structured LiteMessage — chaining
    ``parse_raw_assistant_response`` through ``convert_message_from_agent``."""

    def setup_method(self):
        self.adapter = _step_adapter()

    def _parse(self, raw: str) -> dict:
        return self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(raw)
        )

    def test_parses_think_into_inline_reasoning(self):
        """StepGUI's <THINK> tag is part of its SFT-trained protocol (plain
        BPE text, uppercase) — captured as ``InlineReasoningContent``, NOT
        Qwen3-VL's native ``<think>`` channel."""
        resp = (
            "<THINK> 需要点击搜索框 </THINK>\n"
            "explain:点击搜索框\taction:CLICK\tpoint:500,300\tsummary:已点击搜索框"
        )
        msg = self._parse(resp)
        assert msg["role"] == "assistant"
        reasoning = next(
            (c["text"] for c in msg["content"] if c["type"] == "inline_reasoning"),
            None,
        )
        assert reasoning == "需要点击搜索框"
        assert len(msg["tool_calls"]) == 1
        action = _single_mobile_action(msg["tool_calls"][0])
        assert action["action"] == "tap"
        assert action["coordinate"] == [500, 300]
        # ``summary:`` is stored as HistorySummaryContent, not in tool_call args
        # (it's trajectory metadata, not an action parameter).
        summary = next(
            (c["text"] for c in msg["content"] if c["type"] == "history_summary"),
            None,
        )
        assert summary == "已点击搜索框"

    def test_explain_stored_as_action_description(self):
        """``explain`` text lives in ``content`` as ``ActionDescriptionContent``
        so ``build_generation_prompt`` can re-prepend it when rendering the
        assistant message."""
        resp = "<THINK> x </THINK>\nexplain:点击确认\taction:CLICK\tpoint:1,1\tsummary:s"
        msg = self._parse(resp)
        assert any(
            c["type"] == "action_description" and c["text"] == "点击确认"
            for c in msg["content"]
        )

    def test_missing_think_tags_still_parses(self):
        resp = "explain:点击\taction:CLICK\tpoint:500,300\tsummary:s"
        msg = self._parse(resp)
        # No <THINK> → no reasoning of any kind.
        assert "reasoning_content" not in msg
        assert not any(c["type"] == "inline_reasoning" for c in msg.get("content", []))
        assert _single_mobile_action(msg["tool_calls"][0])["action"] == "tap"

    @pytest.mark.parametrize("variant", [
        "<TINK> x </TINK>",                # typo
        "<think> x </think>",              # lowercase
        "< THINK > x < /THINK >",          # extra whitespace
        "<THINK> x </THINK>",              # canonical
    ])
    def test_think_tag_tolerances(self, variant):
        resp = f"{variant}\naction:WAIT\tvalue:1"
        msg = self._parse(resp)
        reasoning = next(
            (c["text"] for c in msg["content"] if c["type"] == "inline_reasoning"),
            None,
        )
        assert reasoning == "x"

    def test_point_comma_and_space_both_parse(self):
        resp_comma = "action:CLICK\tpoint:100,200"
        resp_space = "action:CLICK\tpoint:100 200"
        m_c = self._parse(resp_comma)
        m_s = self._parse(resp_space)
        assert _single_mobile_action(m_c["tool_calls"][0])["coordinate"] == [100, 200]
        assert _single_mobile_action(m_s["tool_calls"][0])["coordinate"] == [100, 200]

    def test_action_normalized_to_uppercase(self):
        """`_action_dict_to_tool_call` normalizes the action enum to uppercase.
        After convert_from_agent, CLICK becomes cua-lite ``tap``."""
        resp = "action:click\tpoint:500,300"
        msg = self._parse(resp)
        assert _single_mobile_action(msg["tool_calls"][0])["action"] == "tap"

    def test_complete_uses_terminate(self):
        """On parse + convert, ``COMPLETE`` maps to cua-lite ``terminate``
        with ``status="success"``."""
        resp = "action:COMPLETE\treturn:任务完成\tsummary:s"
        msg = self._parse(resp)
        fn = msg["tool_calls"][0]
        assert tool_call_name(fn) == "terminate"
        assert tool_call_arguments(fn)["reason"] == "任务完成"

    def test_slide_parses_point1_point2(self):
        resp = "action:SLIDE\tpoint1:500,700\tpoint2:500,200\tsummary:s"
        msg = self._parse(resp)
        action = _single_mobile_action(msg["tool_calls"][0])
        assert action["action"] == "swipe"
        assert action["start_coordinate"] == [500, 700]
        assert action["coordinate"] == [500, 200]

    def test_empty_response_no_tool_calls_no_reasoning(self):
        """Empty input: parse_raw wraps as text, convert_from_agent finds
        no kv and no <THINK> — result has empty content, no tool_calls."""
        msg = self._parse("")
        assert msg["role"] == "assistant"
        assert "tool_calls" not in msg


# =============================================================================
# 4. Action dict -> tool_call field order
# =============================================================================

class TestActionDictOrdering:
    def test_fields_emitted_in_sft_order(self):
        """The parser must emit tool_call args in the correct order so that
        ``format_agent_tool_call_as_wire_text`` serializes the same sequence the
        model was SFT'd to produce. ``summary`` lives in ``HistorySummaryContent``
        (not args) and is re-injected by ``convert_message_to_agent`` at the
        canonical ``action:X\\tsummary:S\\t<fields>`` slot."""
        resp = (
            # Intentionally scrambled on the wire — parser should still re-sort.
            "summary:s\tpoint:1,2\taction:CLICK"
        )
        adapter = _step_adapter()
        msg = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(resp)
        )
        # After convert_from_agent, CLICK becomes cua-lite ``tap`` with
        # ``coordinate`` instead of ``point``.
        action = _single_mobile_action(msg["tool_calls"][0])
        assert action["action"] == "tap"
        assert action["coordinate"] == [1, 2]
        # Summary still survives — it just lives in content now.
        assert any(
            c["type"] == "history_summary" and c["text"] == "s"
            for c in msg["content"]
        )


# =============================================================================
# 5. format_agent_tool_call_as_wire_text
# =============================================================================

class TestFormatAgentToolCallAsWireText:
    def setup_method(self):
        self.adapter = _step_adapter()

    def test_return_value_is_renamed_to_return(self):
        tc = self.adapter.action_space.convert_tool_calls_to_agent(
            [LiteFinishToolSet.terminate(status="success", reason="done")]
        )[0]
        text = self.adapter.format_agent_tool_call_as_wire_text(tc)
        assert "return:done" in text
        assert "return_value" not in text

    def test_list_point_is_comma_joined(self):
        tc = self.adapter.action_space.convert_tool_calls_to_agent(
            [LiteMobileActionSpace.tap(coordinate=[500, 300])]
        )[0]
        text = self.adapter.format_agent_tool_call_as_wire_text(tc)
        assert "point:500,300" in text
        # Fields are tab-separated
        assert text == "action:CLICK\tpoint:500,300"

    def test_renderer_takes_the_bare_projection_not_a_canonical_call(self):
        """Input shape is this renderer's contract, not an implementation detail.

        Step-GUI's renderer lives on the ADAPTER (the SFT wire format is the
        adapter's prompt boundary) and speaks the ``mobile_use`` agent
        projection, so a canonical Lite call must fail here rather than render
        something plausible.
        """
        with pytest.raises(KeyError):
            self.adapter.format_agent_tool_call_as_wire_text(
                LiteMobileActionSpace.tap(coordinate=[500, 300])
            )

    def test_action_space_keeps_the_canonical_base_renderer(self):
        """The two renderers sit on different layers, and now say so by name.

        The action space did not override the base ``format_tool_call_as_text``,
        so it still speaks canonical Lite calls; only the adapter's
        ``format_agent_tool_call_as_wire_text`` speaks the agent projection.
        """
        space = self.adapter.action_space
        assert type(space).format_tool_call_as_text is BaseActionSpace.format_tool_call_as_text
        assert "tap" in space.format_tool_call_as_text(
            LiteMobileActionSpace.tap(coordinate=[500, 300])
        )
        # Closure guard: the adapter must not answer to the action space's
        # canonical renderer name again.
        assert not hasattr(self.adapter, "format_tool_call_as_text")

    def test_newlines_and_tabs_stripped_from_string_values(self):
        """`\\n` and `\\t` must be stripped from value text to avoid
        breaking the tab-separated format."""
        tc = self.adapter.action_space.convert_tool_calls_to_agent(
            [LiteMobileActionSpace.type(text="hel\nlo\tworld")]
        )[0]
        text = self.adapter.format_agent_tool_call_as_wire_text(tc)
        assert "value:helloworld" in text

    def test_full_field_order_matches_reference_action2str(self):
        """Render order follows reference `action2str` output:
        `action:XXX\\tsummary:...\\t<per-action fields>`. Summary lives in
        ``HistorySummaryContent`` and is injected explicitly via the
        ``summary=`` kwarg (matches what ``convert_message_to_agent`` does).

        Chain through convert_from_agent → convert_to_agent to get the
        mobile_use tool_call back for ``format_agent_tool_call_as_wire_text``."""
        adapter = _step_adapter()
        resp = "action:CLICK\tsummary:已点击\tpoint:500,300"
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(resp)
        )
        agent = adapter.convert_message_to_agent(lite)
        # After convert_to_agent, tool_calls are dropped (folded into text).
        # But we can also verify via the folded text directly:
        assert agent["content"][0]["text"] == "action:CLICK\tsummary:已点击\tpoint:500,300"


# =============================================================================
# 6. Direct parse -> format round-trip (fresh-rollout fidelity)
# =============================================================================

class TestDirectRoundTrip:
    """`parse_raw → convert_from_agent → convert_to_agent → format` round-trip.

    This is the fresh-rollout path (model output -> history text for next turn)
    and must be byte-exact for SFT distribution fidelity.
    """

    @staticmethod
    def _summary_of(msg: dict[str, Any]) -> str:
        """Pull ``summary`` out of ``HistorySummaryContent`` for byte-exact
        re-rendering (mirrors what ``build_generation_prompt`` does)."""
        for c in msg.get("content") or []:
            if c["type"] == "history_summary":
                return c["text"]
        return ""

    @pytest.mark.parametrize("resp", [
        # Shape matches reference action2str output (verified by live invocation):
        # `action:XXX\tsummary:...\t<per-action fields per action2action>`.
        "action:CLICK\tsummary:已点击搜索框\tpoint:500,300",
        "action:LONGPRESS\tsummary:已长按\tpoint:100,200",
        "action:TYPE\tsummary:已输入天气\tvalue:天气",
        "action:SLIDE\tsummary:已向下滑动\tpoint1:500,700\tpoint2:500,200",
        "action:AWAKE\tsummary:已唤醒\tvalue:微信",
        "action:INFO\tsummary:需要确认\tvalue:你要哪个？",
        "action:WAIT\tsummary:已等待\tvalue:5",
        "action:COMPLETE\tsummary:任务完成\treturn:已找到天气信息",
        "action:ABORT\tsummary:任务终止",
    ])
    def test_parse_format_byte_exact(self, resp):
        adapter = _step_adapter()
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(resp)
        )
        agent = adapter.convert_message_to_agent(lite)
        # convert_to_agent folds everything into a single text block
        assert agent["content"][0]["text"] == resp

    def test_abort_value_stripped_on_render(self):
        """Reference `action2action` drops ABORT's `value` — even if the model
        emits `value:...` on the wire (per task_define_prompt line 47 which
        tells it to), the rendered history text strips it."""
        resp_with_value = "action:ABORT\tsummary:任务终止\tvalue:页面无法加载"
        adapter = _step_adapter()
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(resp_with_value)
        )
        agent = adapter.convert_message_to_agent(lite)
        assert agent["content"][0]["text"] == "action:ABORT\tsummary:任务终止"

    def test_type_point_stripped_on_render(self):
        """Reference `action2action` drops TYPE's `point` (commented out at
        parser_0920_summary.py:142). The model's inline prompt tells it to
        emit point, but the SFT-distribution serializer strips it."""
        resp_with_point = "action:TYPE\tsummary:已输入天气\tvalue:天气\tpoint:500,300"
        adapter = _step_adapter()
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(resp_with_point)
        )
        agent = adapter.convert_message_to_agent(lite)
        assert agent["content"][0]["text"] == "action:TYPE\tsummary:已输入天气\tvalue:天气"


# =============================================================================
# 7. Lossy round-trip via cua-lite canonical form
# =============================================================================

class TestLossyRoundTripViaCuaLite:
    """Round-trip through `convert_message_from_agent` -> `convert_message_to_agent`.

    This path is used by `unroll_sample` for SFT export. Any field lost here
    means the rendered SFT target differs from the original distribution.
    """

    def test_summary_survives_lite_round_trip(self):
        """``summary`` survives raw → LiteMessage (HistorySummaryContent) →
        AgentMessage (folded into the wire-format text)."""
        adapter = _step_adapter()
        parsed = adapter.parse_raw_assistant_response(
            "<THINK> x </THINK>\n"
            "explain:e\taction:CLICK\tpoint:500,300\tsummary:已点击确认"
        )
        lite = adapter.convert_message_from_agent(parsed)
        agent = adapter.convert_message_to_agent(lite)
        # convert_message_to_agent folds everything into a single text block;
        # the summary appears in the canonical ``...\tsummary:S\t...`` slot.
        assert "summary:已点击确认" in agent["content"][0]["text"]


# =============================================================================
# 8. Unroll fixtures
# =============================================================================

def _build_trajectory_sample() -> LiteSample:
    adapter = _step_adapter()
    raws = [
        "<THINK> t0 </THINK>\nexplain:e0\taction:CLICK\tsummary:summary_0\tpoint:100,200",
        "<THINK> t1 </THINK>\nexplain:e1\taction:TYPE\tsummary:summary_1\tvalue:hi",
        "<THINK> t2 </THINK>\nexplain:e2\taction:COMPLETE\tsummary:summary_2\treturn:done",
    ]
    messages: list[dict[str, Any]] = []
    for i, raw in enumerate(raws):
        if i == 0:
            messages.append({"role": "user", "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "task"},
            ]})
        else:
            messages.append({"role": "user", "content": [
                {"type": "image", "index": i}]})
        parsed = adapter.parse_raw_assistant_response(raw)
        messages.append(adapter.convert_message_from_agent(parsed))
    meta = _step_metadata()
    return LiteSample(
        metadata=meta, messages=messages,
        images=[f"img{i}.png" for i in range(3)],
    )


class TestUnrollStructure:
    """End-to-end: storing a trajectory and unrolling it for SFT must produce
    one step per turn with (system, user, assistant) structure where the
    assistant IS the training target."""

    def test_one_step_per_turn(self):
        sample = _build_trajectory_sample()
        adapter = _step_adapter()
        steps = adapter.unroll(sample).steps
        assert len(steps) == 3

    def test_each_step_has_target_assistant(self):
        sample = _build_trajectory_sample()
        adapter = _step_adapter()
        steps = adapter.unroll(sample).steps
        for i, step in enumerate(steps):
            roles = [m["role"] for m in step]
            assert roles == ["system", "user", "assistant"], f"turn {i}: {roles}"

    def test_target_assistant_carries_folded_sft_text(self):
        """Target assistant has been folded by ``convert_message_to_agent``
        into a single text block matching the SFT distribution: a ``<THINK>``
        block followed by ``explain:E\\taction:X\\t...``. The structured
        ``tool_calls`` and ``reasoning_content`` fields are dropped on
        purpose so the chat_template never re-renders them in JSON form."""
        sample = _build_trajectory_sample()
        adapter = _step_adapter()
        steps = adapter.unroll(sample).steps
        for step in steps:
            asst = [m for m in step if m.get("role") == "assistant"][-1]
            assert "tool_calls" not in asst
            assert "reasoning_content" not in asst
            text = asst["content"][0]["text"]
            assert "<THINK>" in text and "</THINK>" in text
            assert "explain:" in text
            assert "action:" in text

    def test_each_step_has_exactly_one_image_at_target_step(self):
        """Each rendered step's rolled-up user message must contain EXACTLY
        ONE image."""
        from lite.core import LiteSample
        adapter = _step_adapter()
        raws = [
            _RAW_PER_ACTION["CLICK"],
            _RAW_PER_ACTION["TYPE"],
            _RAW_PER_ACTION["COMPLETE"],
        ]
        messages = []
        for i, raw in enumerate(raws):
            if i == 0:
                messages.append({"role": "user", "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "task"}]})
            else:
                messages.append({"role": "user", "content": [
                    {"type": "image", "index": i}]})
            parsed = adapter.parse_raw_assistant_response(raw)
            messages.append(adapter.convert_message_from_agent(parsed))
        meta = _step_metadata()
        sample = LiteSample(metadata=meta, messages=messages,
                            images=[f"img{i}.png" for i in range(3)])
        steps = adapter.unroll(sample).steps
        for i, step in enumerate(steps):
            user_msg = [m for m in step if m.get("role") == "user"][-1]
            # The user message's image content refers to the target step's
            # index in the ORIGINAL trajectory. After image reordering, the
            # concrete image file should correspond to step i.
            image_items = [c for c in user_msg["content"] if c.get("type") == "image"]
            assert len(image_items) == 1, f"turn {i}: {len(image_items)} images"


class TestUnrollNoLeakage:
    """Rolling summary must lag by one step — target i's own summary must
    not appear in its own prompt."""

    def test_rolling_summary_is_lagged_by_one_step(self):
        """Summary from step i-1 should appear in step i's prompt; step i's
        own summary must NOT be visible to itself (leakage check)."""
        sample = _build_trajectory_sample()
        adapter = _step_adapter()
        steps = adapter.unroll(sample).steps

        for i, step in enumerate(steps):
            user_msg = [m for m in step if m.get("role") == "user"][-1]
            text = "".join(
                c.get("text", "") for c in user_msg["content"]
                if c.get("type") == "text"
            )
            if i == 0:
                assert "暂无历史操作" in text
                assert "summary_0" not in text
                assert "summary_1" not in text
            elif i == 1:
                assert "summary_0" in text
                assert "summary_1" not in text
            elif i == 2:
                assert "summary_1" in text
                assert "summary_2" not in text


class TestSampleIndependence:
    """unroll twice returns independent outputs; mutating one must
    not affect the other."""

    def test_two_unroll_calls_are_independent(self):
        sample = _build_trajectory_sample()
        adapter = _step_adapter()
        first = adapter.unroll(sample).steps
        second = adapter.unroll(sample).steps
        # Mutate first's target-assistant text. After convert_message_to_agent
        # the target is a single text block (tool_calls dropped post-fold).
        first[-1][-1]["content"][0]["text"] = "MUTATED"
        assert second[-1][-1]["content"][0]["text"] != "MUTATED"

    def test_siblings_are_independent(self):
        sample = _build_trajectory_sample()
        adapter = _step_adapter()
        steps = adapter.unroll(sample).steps
        # Mutate step 0's target text; step 1 must be untouched.
        steps[0][-1]["content"][0]["text"] = "MUTATED"
        assert steps[1][-1]["content"][0]["text"] != "MUTATED"


# =============================================================================
# 9. STEPGUIMobileAgent.build_generation_prompt (THINK + tab-separated fold)
# =============================================================================

class _FakeProcessor:
    """Captures apply_chat_template args."""
    def __init__(self):
        self.captured_messages = None
        self.captured_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.captured_messages = messages
        self.captured_kwargs = kwargs
        return "RENDERED"


class TestBuildGenerationPrompt:
    def _make_agent(self, processor):
        return AgentRegistry.get(
            "step_gui@mobile@use",
            generate_fn=lambda **kw: {"response": ""},
            processor=processor,
        )

    def test_reasoning_and_tool_calls_folded_into_text(self):
        """``adapter.convert_message_to_agent`` folds reasoning + content +
        tool_calls into the SFT-trained ``<THINK>...</THINK>\\nexplain:...\\t
        action:...`` plain-text wire format, dropping the structured
        ``tool_calls`` / ``reasoning_content`` fields so the chat_template
        never re-renders them in JSON form."""
        adapter = _step_adapter()
        out = adapter.convert_message_to_agent({
            "role": "assistant",
            "content": [
                {"type": "inline_reasoning", "text": "需要点击确认按钮"},
                {"type": "action_description", "text": "点击确认"},
            ],
            "tool_calls": [
                _mobile_call("tap", coordinate=[500, 300]),
            ],
        })
        assert "reasoning_content" not in out
        assert "tool_calls" not in out
        assert len(out["content"]) == 1
        text = out["content"][0]["text"]
        assert text.startswith("<THINK> 需要点击确认按钮 </THINK>")
        assert "explain:点击确认" in text
        assert "action:CLICK\tpoint:500,300" in text

    def test_no_reasoning_no_tool_calls_passes_through(self):
        """Plain assistant turns (no reasoning, no tool_calls) collapse to
        whatever ``action_description`` text was already there."""
        adapter = _step_adapter()
        out = adapter.convert_message_to_agent({
            "role": "assistant",
            "content": [{"type": "action_description", "text": "y"}],
        })
        assert out["content"] == [{"type": "text", "text": "y"}]

    def test_does_not_pass_tools_kwarg(self):
        """Step-GUI was SFT'd without the chat_template's `<tools>` schema
        block — we must not pass `tools=`."""
        proc = _FakeProcessor()
        agent = self._make_agent(proc)
        agent.build_generation_prompt(
            [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        )
        assert "tools" not in proc.captured_kwargs

    def test_tool_calls_without_reasoning_still_folds(self):
        """``reasoning_content=None`` but tool_calls present: fold tool_calls only."""
        adapter = _step_adapter()
        out = adapter.convert_message_to_agent({
            "role": "assistant",
            "content": [],
            "tool_calls": [
                _mobile_call("wait", duration=1),
            ],
        })
        assert out["content"][0]["text"] == "action:WAIT\tvalue:1"

    def test_canonical_nested_input_renders(self):
        adapter = _step_adapter()
        msg = {
            "role": "assistant",
            "tool_calls": [make_tool_call("tap", {"coordinate": [1, 2]})],
        }
        out = adapter.convert_message_to_agent(msg)
        assert out["content"][0]["text"] == "action:CLICK\tpoint:1,2"


# =============================================================================
# 10. Golden comparisons vs. reference gelab-zero action2str
# =============================================================================

# Reference output for each action, captured by a live invocation of
# `Parser0920Summary.action2str` (see commit log). These are the EXACT bytes
# the SFT distribution contains and what cua-lite must reproduce end-to-end.
_RAW_PER_ACTION = {
    "CLICK":     "<THINK> t </THINK>\nexplain:e\taction:CLICK\tsummary:s\tpoint:500,300",
    "LONGPRESS": "<THINK> t </THINK>\nexplain:e\taction:LONGPRESS\tsummary:s\tpoint:100,200",
    "TYPE":      "<THINK> t </THINK>\nexplain:e\taction:TYPE\tsummary:s\tvalue:天气",
    "SLIDE": (
        "<THINK> t </THINK>\nexplain:e\taction:SLIDE\tsummary:s\tpoint1:500,700\t"
        "point2:500,200"
    ),
    "AWAKE":     "<THINK> t </THINK>\nexplain:e\taction:AWAKE\tsummary:s\tvalue:微信",
    "INFO":      "<THINK> t </THINK>\nexplain:e\taction:INFO\tsummary:s\tvalue:q?",
    "WAIT":      "<THINK> t </THINK>\nexplain:e\taction:WAIT\tsummary:s\tvalue:5",
    "COMPLETE":  "<THINK> t </THINK>\nexplain:e\taction:COMPLETE\tsummary:s\treturn:done",
    "ABORT":     "<THINK> t </THINK>\nexplain:e\taction:ABORT\tsummary:s",
}


def _render_assistant_as_raw_text(adapter, asst: dict[str, Any]) -> str:
    """Reproduce the `<THINK>...\\nexplain:...\\t<tc_text>` concat that
    ``STEPGUIMobileAgent.build_generation_prompt`` produces.

    Mirrors the production helper: pulls reasoning via ``get_inline_reasoning``
    (handles both ``reasoning_content`` field and ``InlineReasoningContent``
    parts) and pulls the rolling history summary from
    ``HistorySummaryContent`` parts before passing it to
    ``format_agent_tool_call_as_wire_text``.
    """
    from lite.core.messages import get_inline_reasoning as _get_inline_reasoning
    thinking = _get_inline_reasoning(asst)
    content = asst.get("content") or []
    explain = "".join(
        c["text"] for c in content
        if c["type"] in ("text", "action_description")
    )
    summary = "".join(c["text"] for c in content if c["type"] == "history_summary")
    blocks: list[str] = []
    if thinking:
        blocks.append(f"<THINK> {thinking} </THINK>")
    for tc in asst.get("tool_calls", []):
        tc_text = adapter.format_agent_tool_call_as_wire_text(tc, summary=summary)
        summary = ""  # only emit on the first call
        if explain:
            tc_text = f"explain:{explain}\t{tc_text}"
            explain = ""
        blocks.append(tc_text)
    if explain:
        blocks.append(explain)
    return "\n".join(blocks)


# Expected LiteMessage tool_call shape per action, after
# `parse_raw_assistant_response` + `convert_message_from_agent`. These match
# cua-lite canonical action names / argument schemas (observed from the
# current pipeline); any silent drift surfaces as a test diff.
# ``summary`` is no longer in tool_call.arguments — it lives as
# ``HistorySummaryContent`` in the assistant message. The expected text "s"
# (from ``_RAW_PER_ACTION``) is asserted separately below.
_EXPECTED_LITE_TOOL_CALL: dict[str, dict[str, Any]] = {
    "CLICK":     {"name": "tap",        "arguments": {"coordinate": [500, 300]}},
    "LONGPRESS": {"name": "long_press", "arguments": {"coordinate": [100, 200]}},
    "TYPE":      {"name": "type",       "arguments": {"text": "天气"}},
    "SLIDE": {
        "name": "swipe",
        "arguments": {"start_coordinate": [500, 700], "coordinate": [500, 200]},
    },
    "AWAKE":     {"name": "open_app",   "arguments": {"app_name": "微信"}},
    "INFO":      {"name": "response",   "arguments": {"text": "q?"}},
    "WAIT":      {"name": "wait",       "arguments": {"duration": 5.0}},
    "COMPLETE":  {"name": "terminate",  "arguments": {"status": "success", "reason": "done"}},
    "ABORT":     {"name": "terminate",  "arguments": {"status": "failure", "reason": ""}},
}


class TestRawToLiteMessage:
    """Per-action: raw → parse → convert_from_agent → assert LiteMessage
    fields. Asserts the cua-lite canonical tool_call shape, reasoning, and
    explain content."""

    @pytest.mark.parametrize("action", list(_RAW_PER_ACTION.keys()))
    def test_parse_then_convert_from_agent(self, action):
        raw = _RAW_PER_ACTION[action]
        adapter = _step_adapter()
        parsed = adapter.parse_raw_assistant_response(raw)
        lite = adapter.convert_message_from_agent(parsed)

        # <THINK> reasoning lives as InlineReasoningContent (StepGUI is not
        # a native-thinking model); explain text is ActionDescriptionContent;
        # rolling summary is HistorySummaryContent (no longer in tool_call args).
        content = lite.get("content") or []
        reasoning = next(
            (c["text"] for c in content if c["type"] == "inline_reasoning"),
            None,
        )
        assert reasoning == "t"
        action_descs = [c["text"] for c in content if c["type"] == "action_description"]
        assert any(t == "e" for t in action_descs), f"explain text missing for {action}"
        summaries = [c["text"] for c in content if c["type"] == "history_summary"]
        assert any(s == "s" for s in summaries), f"summary missing for {action}"
        # exactly one tool_call per raw
        assert len(lite["tool_calls"]) == 1
        fn = lite["tool_calls"][0]
        expected = _EXPECTED_LITE_TOOL_CALL[action]
        if expected["name"] in {"tap", "long_press", "type", "swipe", "wait"}:
            actual = _single_mobile_action(fn)
            assert actual["action"] == expected["name"], (
                f"{action}: action={actual['action']!r} != {expected['name']!r}"
            )
        else:
            actual = tool_call_arguments(fn)
            assert tool_call_name(fn) == expected["name"], (
                f"{action}: name={tool_call_name(fn)!r} != {expected['name']!r}"
            )
        for k, v in expected["arguments"].items():
            assert actual.get(k) == v, (
                f"{action}: args[{k}]={actual.get(k)!r} != {v!r}"
            )


class TestLiteMessageToRaw:
    """Per-action: LiteMessage → convert_to_agent + render → assert golden
    raw string byte-exact."""

    @pytest.mark.parametrize("action", list(_RAW_PER_ACTION.keys()))
    def test_convert_to_agent_and_render(self, action):
        raw = _RAW_PER_ACTION[action]
        adapter = _step_adapter()
        # Build the LiteMessage by parsing the golden raw, then convert
        # back out via convert_to_agent and render the text form.
        parsed = adapter.parse_raw_assistant_response(raw)
        lite = adapter.convert_message_from_agent(parsed)
        agent_msg = adapter.convert_message_to_agent(lite)
        rendered = _render_assistant_as_raw_text(adapter, agent_msg)
        assert rendered == raw


class TestRawRoundTrip:
    """For every action, the full adapter pipeline
    `raw → parse → convert_from → convert_to → format` must reproduce the
    reference `action2str` bytes.

    This is the contract SFT unroll relies on. Tests are independent of
    reference imports (goldens embedded)."""

    @pytest.mark.parametrize("action", list(_RAW_PER_ACTION.keys()))
    def test_raw_round_trip_byte_exact(self, action):
        raw = _RAW_PER_ACTION[action]
        adapter = _step_adapter()

        parsed = adapter.parse_raw_assistant_response(raw)
        lite = adapter.convert_message_from_agent(parsed)
        agent_msg = adapter.convert_message_to_agent(lite)
        # Fold back to raw text the same way build_generation_prompt would
        assert _render_assistant_as_raw_text(adapter, agent_msg) == raw


# =============================================================================
# 11. Unroll-for-SFT for every action (single-turn trajectory)
# =============================================================================

def _make_single_turn_sample(adapter, raw: str):
    from lite.core import LiteSample
    parsed = adapter.parse_raw_assistant_response(raw)
    lite_asst = adapter.convert_message_from_agent(parsed)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "task"},
        ]},
        lite_asst,
    ]
    meta = _step_metadata()
    return LiteSample(metadata=meta, messages=messages, images=["img0.png"])


class TestUnrollByteExactTargetPerAction:
    """For every action, a single-turn trajectory unrolled for SFT must emit
    a target assistant whose serialized text matches reference action2str."""

    @pytest.mark.parametrize("action", list(_RAW_PER_ACTION.keys()))
    def test_unrolled_target_matches_reference(self, action):
        raw = _RAW_PER_ACTION[action]
        adapter = _step_adapter()
        sample = _make_single_turn_sample(adapter, raw)

        steps = adapter.unroll(sample).steps
        assert len(steps) == 1
        asst = [m for m in steps[0] if m.get("role") == "assistant"][-1]
        rendered = _render_assistant_as_raw_text(adapter, asst)
        assert rendered == raw


# =============================================================================
# 12. Conversion purity — no input mutation
# =============================================================================

class TestMutationPurity:
    def test_convert_from_agent_does_not_mutate(self):
        adapter = _step_adapter()
        input_msg = {
            "role": "assistant",
            "tool_calls": [{
                "name": "mobile_use",
                "arguments": {"action": "CLICK", "point": [1, 2], "summary": "s"},
            }],
        }
        import copy as _copy
        snapshot = _copy.deepcopy(input_msg)
        _ = adapter.convert_message_from_agent(input_msg)
        assert input_msg == snapshot

    def test_convert_to_agent_does_not_mutate(self):
        adapter = _step_adapter()
        input_msg = {
            "role": "assistant",
            "tool_calls": [
                make_tool_call(
                    "mobile",
                    {
                        "actions": [{
                            "action": "tap",
                            "coordinate": [1, 2],
                            "summary": "s",
                        }],
                    },
                )
            ],
        }
        import copy as _copy
        snapshot = _copy.deepcopy(input_msg)
        _ = adapter.convert_message_to_agent(input_msg)
        assert input_msg == snapshot

    def test_unroll_samples_are_independent(self):
        """Mutating one rendered step's messages must not affect siblings."""
        from lite.core import LiteSample
        adapter = _step_adapter()
        raws = [
            _RAW_PER_ACTION["CLICK"],
            _RAW_PER_ACTION["TYPE"],
        ]
        messages = []
        for i, raw in enumerate(raws):
            if i == 0:
                messages.append({"role": "user", "content": [
                    {"type": "image", "index": 0}, {"type": "text", "text": "task"}]})
            else:
                messages.append({"role": "user", "content": [{"type": "image", "index": i}]})
            parsed = adapter.parse_raw_assistant_response(raw)
            messages.append(adapter.convert_message_from_agent(parsed))
        meta = _step_metadata()
        sample = LiteSample(metadata=meta, messages=messages,
                            images=["img0.png", "img1.png"])
        steps = adapter.unroll(sample).steps
        # Mutate step[0]'s target text. After convert_message_to_agent the
        # target is a single text block (tool_calls dropped post-fold).
        steps[0][-1]["content"][0]["text"] = "MUTATED"
        assert steps[1][-1]["content"][0]["text"] != "MUTATED"


# =============================================================================
# 13. Parse edge cases
# =============================================================================

class TestParseEdgeCases:
    """Edge cases chained through ``convert_message_from_agent``."""

    def setup_method(self):
        self.adapter = _step_adapter()

    def _parse(self, raw: str) -> dict:
        return self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(raw)
        )

    def test_empty_think_block(self):
        """`<THINK></THINK>` must not error; reasoning_content stays absent."""
        resp = "<THINK></THINK>\naction:WAIT\tvalue:1"
        msg = self._parse(resp)
        assert "reasoning_content" not in msg
        assert _single_mobile_action(msg["tool_calls"][0])["action"] == "wait"

    def test_whitespace_only_think_block(self):
        """Whitespace-only thinking content is stripped and treated as absent."""
        resp = "<THINK>   \n  </THINK>\naction:WAIT\tvalue:1"
        msg = self._parse(resp)
        assert "reasoning_content" not in msg

    def test_only_action_no_think_no_explain_no_summary(self):
        """Minimal valid response: just `action:X\\t<params>`."""
        resp = "action:CLICK\tpoint:100,200"
        msg = self._parse(resp)
        action = _single_mobile_action(msg["tool_calls"][0])
        assert action["action"] == "tap"
        assert action["coordinate"] == [100, 200]
        # No summary in content
        assert not any(
            c["type"] == "history_summary" and c.get("text")
            for c in msg.get("content") or []
        )

    def test_extra_unknown_keys_preserved_in_args(self):
        """Unknown kv keys must not crash the parser (forward-compat)."""
        resp = "action:CLICK\tpoint:1,1\tfoo:bar\tsummary:s"
        msg = self._parse(resp)
        action = _single_mobile_action(msg["tool_calls"][0])
        assert action["action"] == "tap"
        assert "coordinate" in action

    def test_cjk_and_punctuation_in_summary_preserved(self):
        resp = "action:INFO\tsummary:你确认吗？（点击 OK 继续）\tvalue:q"
        msg = self._parse(resp)
        summary = next(
            (c["text"] for c in msg["content"] if c["type"] == "history_summary"),
            None,
        )
        assert summary == "你确认吗？（点击 OK 继续）"

    def test_colon_inside_value_preserved(self):
        """The split is on the FIRST colon, so values containing colons survive."""
        resp = "action:INFO\tsummary:time 12:30\tvalue:q"
        msg = self._parse(resp)
        summary = next(
            (c["text"] for c in msg["content"] if c["type"] == "history_summary"),
            None,
        )
        assert summary == "time 12:30"

    def test_think_only_no_action(self):
        resp = "<THINK> just thinking </THINK>"
        msg = self._parse(resp)
        assert msg["content"] == [{"type": "text", "text": resp}]
        assert "tool_calls" not in msg


# =============================================================================
# 14. Edge cases (multi-tool_call, long trajectories)
# =============================================================================

class TestEdgeCases:
    """Edge cases: multiple tool_calls per assistant, long trajectories."""

    def test_multiple_tool_calls_each_rendered(self):
        """If parse/upstream ever yields multiple tool_calls in one assistant,
        each must render as its own tab-separated block. Upstream only ever
        emits one, but the renderer must not crash or silently drop extras."""
        adapter = _step_adapter()
        out = adapter.convert_message_to_agent({
            "role": "assistant",
            "content": [],
            "tool_calls": [
                _mobile_call("tap", coordinate=[1, 1]),
                _mobile_call("wait", duration=1),
            ],
        })
        text = out["content"][0]["text"]
        assert "action:CLICK" in text
        assert "action:WAIT" in text

    def test_multi_action_render_parses_back_to_every_action(self):
        """Render emits one ``action:...`` record per line; the parser must
        treat the newline as a record boundary and give back N tool calls
        (regression: it used to fold the whole blob into ONE record)."""
        adapter = _step_adapter()
        out = adapter.convert_message_to_agent({
            "role": "assistant",
            "content": [],
            "tool_calls": [
                _mobile_call("tap", coordinate=[100, 100]),
                _mobile_call("type", text="hi"),
            ],
        })
        text = out["content"][0]["text"]
        assert text.split("\n") == ["action:CLICK\tpoint:100,100", "action:TYPE\tvalue:hi"]
        back = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(text)
        )
        actions = tool_call_arguments(back["tool_calls"][0])["actions"]
        assert [a["action"] for a in actions] == ["tap", "type"]

    def test_10_turn_trajectory_unrolls_to_10_steps(self):
        from lite.core import LiteSample
        adapter = _step_adapter()
        n = 10
        messages = []
        expected_summaries: list[str] = []
        for i in range(n):
            raw = (
                f"<THINK> t{i} </THINK>\nexplain:e{i}\taction:CLICK\t"
                f"summary:summary_{i}\tpoint:{100+i},{200+i}"
            )
            expected_summaries.append(f"summary_{i}")
            if i == 0:
                messages.append({"role": "user", "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "long task"}]})
            else:
                messages.append({"role": "user", "content": [{"type": "image", "index": i}]})
            parsed = adapter.parse_raw_assistant_response(raw)
            messages.append(adapter.convert_message_from_agent(parsed))
        meta = _step_metadata()
        sample = LiteSample(metadata=meta, messages=messages,
                            images=[f"img{i}.png" for i in range(n)])
        steps = adapter.unroll(sample).steps
        assert len(steps) == n
        # Verify the prompt for step i has summary from step (i-1), not i.
        for i, step in enumerate(steps):
            user_msg = [m for m in step if m.get("role") == "user"][-1]
            text = "".join(
                c.get("text", "") for c in user_msg["content"]
                if c.get("type") == "text"
            )
            if i == 0:
                assert "暂无历史操作" in text
            else:
                assert expected_summaries[i - 1] in text
            # Target's own summary must not leak in
            assert expected_summaries[i] not in text.split("当前手机屏幕截图如下")[0]


# =============================================================================
# 15. Multi-record wire format (newline = record boundary)
# =============================================================================

_MOBILE_BATCH = [
    {"action": "tap", "coordinate": [100, 100], "clicks": 1},
    {"action": "type", "text": "hi"},
    {"action": "tap", "coordinate": [200, 200], "clicks": 1},
]


def _render_and_reparse(adapter, tool_calls: list[dict[str, Any]]) -> tuple[str, dict]:
    """Canonical tool_calls → StepGUI wire text → canonical tool_calls."""
    agent = adapter.convert_message_to_agent({
        "role": "assistant",
        "content": [{"type": "action_description", "text": "Do it."}],
        "tool_calls": tool_calls,
    })
    text = agent["content"][0]["text"]
    return text, adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response(text)
    )


class TestMultiRecordRoundTrip:
    """Records are newline-delimited and fields tab-delimited (reference
    ``Parser0920Summary.action2str`` emits ``<THINK>...</THINK>\\n`` +
    ``"\\t".join(kvs)`` + ``"\\n"`` per action). N rendered records must parse
    back to N tool calls, in order."""

    def test_canonical_batch_round_trips_identically(self):
        adapter = _step_adapter()
        calls = [make_tool_call("mobile", {"actions": _MOBILE_BATCH})]
        text, back = _render_and_reparse(adapter, calls)
        assert len(text.split("\n")) == 3
        assert back["tool_calls"] == calls

    def test_action_terminate_action_stays_three_calls(self):
        """A standalone extra splits the GUI runs around it — never merged."""
        adapter = _step_adapter()
        calls = [
            make_tool_call(
                "mobile",
                {"actions": [{"action": "tap", "coordinate": [100, 100], "clicks": 1}]},
            ),
            make_tool_call("terminate", {"status": "success"}),
            make_tool_call("mobile", {"actions": [{"action": "type", "text": "after"}]}),
        ]
        text, back = _render_and_reparse(adapter, calls)
        lines = text.split("\n")
        assert len(lines) == 3
        assert [
            next(p.split(":", 1)[1] for p in line.split("\t") if p.startswith("action:"))
            for line in lines
        ] == ["CLICK", "COMPLETE", "TYPE"]
        assert back["tool_calls"] == calls

    def test_type_value_does_not_leak_into_preceding_click(self):
        """Regression: the newline used to be swallowed by the point branch's
        ``value.replace(",", " ").split()``, so ``TYPE``'s ``value`` landed on
        the ``CLICK`` and the TYPE itself vanished."""
        adapter = _step_adapter()
        raw = "action:CLICK\tpoint:100,100\naction:TYPE\tvalue:hi"
        back = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(raw)
        )
        actions = tool_call_arguments(back["tool_calls"][0])["actions"]
        assert len(actions) == 2
        click, type_ = actions
        assert click["action"] == "tap" and click["coordinate"] == [100, 100]
        assert "text" not in click and "value" not in click
        assert type_ == {"action": "type", "text": "hi"}

    def test_blank_lines_and_trailing_whitespace_ignored(self):
        adapter = _step_adapter()
        raw = (
            "<THINK> t </THINK>\n"
            "explain:e\taction:CLICK\tsummary:s\tpoint:1,2  \n"
            "\n"
            "   \n"
            "action:TYPE\tvalue:hi\n"
        )
        back = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(raw)
        )
        actions = tool_call_arguments(back["tool_calls"][0])["actions"]
        assert [a["action"] for a in actions] == ["tap", "type"]
        assert any(
            c["type"] == "history_summary" and c["text"] == "s" for c in back["content"]
        )
        assert any(
            c["type"] == "action_description" and c["text"] == "e" for c in back["content"]
        )

    def test_single_record_parse_is_unchanged(self):
        """A working single-action parse must be byte-identical to before."""
        adapter = _step_adapter()
        raw = _RAW_PER_ACTION["CLICK"]
        back = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(raw)
        )
        assert back["content"] == [
            {"type": "inline_reasoning", "text": "t"},
            {"type": "action_description", "text": "e"},
            {"type": "history_summary", "text": "s"},
        ]
        assert back["tool_calls"] == [
            make_tool_call(
                "mobile",
                {"actions": [{"action": "tap", "coordinate": [500, 300], "clicks": 1}]},
            )
        ]


# =============================================================================
# 16. Parse-failure signalling
# =============================================================================

class TestParseFailureSignalling:
    """A grammar failure must be reported via ``mark_model_output_error`` so
    the shared agent loop records a terminal parse-failure final instead of a
    clean content-only final."""

    def setup_method(self):
        self.adapter = _step_adapter()

    def _error_of(self, raw: str) -> str | None:
        msg = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(raw)
        )
        return pop_model_output_error(msg)

    @pytest.mark.parametrize("raw", [
        "<THINK> t </THINK>\naction:",                  # empty action value
        "<THINK> t </THINK>\naction:\tpoint:1,2",       # empty action value + args
        "explain:e\taction:\tsummary:s",                # explain/summary but no verb
    ])
    def test_malformed_action_grammar_marks_error(self, raw):
        assert self._error_of(raw)

    def test_unknown_verb_routes_to_model_visible_invalid_action(self):
        msg = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response("action:NOSUCHVERB\tpoint:1,2")
        )

        assert pop_model_output_error(msg) is None
        actions = tool_call_arguments(msg["tool_calls"][0])["actions"]
        assert actions == [{"action": "NOSUCHVERB", "point": [1, 2]}]
        children, error = validate_lite_action_batch_structure("mobile", {"actions": actions})
        assert error is None
        error = lite_action_batch_child_name_errors("mobile", children).get(0)
        assert error is not None
        assert error.child_action_name == "NOSUCHVERB"

    @pytest.mark.parametrize("raw", [
        "<THINK> t </THINK>\n任务已经完成，无需继续操作。",   # prose final w/ THINK
        "I am done, the answer is 42.",                  # bare prose final
        "<THINK> just thinking </THINK>",                # thinking only
    ])
    def test_plain_prose_does_not_mark_error(self, raw):
        """No grammar marker → deliberate content-only final, stays terminal."""
        assert self._error_of(raw) is None

    def test_valid_action_does_not_mark_error(self):
        assert self._error_of(_RAW_PER_ACTION["CLICK"]) is None

    def test_partial_batch_parse_keeps_the_good_records(self):
        """If at least one record parses, the turn is actionable — no error."""
        raw = "action:CLICK\tpoint:100,100\naction:\naction:TYPE\tvalue:hi"
        msg = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(raw)
        )
        assert pop_model_output_error(msg) is None
        actions = tool_call_arguments(msg["tool_calls"][0])["actions"]
        assert [a["action"] for a in actions] == ["tap", "type"]
