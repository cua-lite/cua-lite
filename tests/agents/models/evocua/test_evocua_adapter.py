"""
Tests for EvoCUA adapters.

Covers:
  1. Adapter registry resolution
  2. Action space type
  3. System prompt contains key_down/key_up in tools section
  4. Response parsing (inherited from Qwen3VLBaseAdapter)
  5. Characterization tests (desktop trajectory, per-action goldens,
     unroll / protocol windowing / mutation purity / edge cases).

Run:
    uv run pytest tests/agents/models/evocua/test_evocua_adapter.py -v
"""

from __future__ import annotations

import copy
import json

import pytest

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.evocua.action_space import EvoCUADesktopActionSpace
from lite.agents.models.evocua.adapter import EvoCUADesktopUseAdapter
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_name
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

_TERMINATE_SCHEMA = make_tool_schema(
    "terminate",
    description="Finish the task.",
    parameters={
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["success", "failure"]}},
        "required": ["status"],
    },
)


def _metadata(*, include_terminate: bool = False) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=[copy.deepcopy(_TERMINATE_SCHEMA)] if include_terminate else [],
    )


def _adapter_for_action(action: str) -> EvoCUADesktopUseAdapter:
    return EvoCUADesktopUseAdapter(
        metadata=_metadata(include_terminate=(action == "terminate")),
    )


def _computer_action(action: str, **arguments):
    return make_tool_call(
        "computer",
        {"actions": [{"action": action, **arguments}]},
    )


# =============================================================================
# 1. Adapter registry resolution
# =============================================================================


class TestAdapterRegistry:
    def test_desktop_trajectory(self):
        adapter = AgentAdapterRegistry.get("evocua@desktop@use")
        assert isinstance(adapter, EvoCUADesktopUseAdapter)

    def test_browser_trajectory(self):
        adapter = AgentAdapterRegistry.get("evocua@browser@use")
        assert isinstance(adapter, EvoCUADesktopUseAdapter)


# =============================================================================
# 2. Action space type
# =============================================================================


class TestActionSpaceType:
    def test_desktop_uses_evocua_action_space(self):
        adapter = AgentAdapterRegistry.get("evocua@desktop@use")
        assert isinstance(adapter.action_space, EvoCUADesktopActionSpace)

    def test_browser_uses_evocua_action_space(self):
        adapter = AgentAdapterRegistry.get("evocua@browser@use")
        assert isinstance(adapter.action_space, EvoCUADesktopActionSpace)


# =============================================================================
# 2b. The finish sentence the ``use`` prompt orders
# =============================================================================


class TestUseSystemPromptFinishRule:
    """EvoCUA's prompt must order the finish name its own surface accepts."""

    @pytest.mark.parametrize("key", ["evocua@desktop@use", "evocua@browser@use"])
    def test_prompt_never_orders_answer(self, key):
        prompt = AgentAdapterRegistry.get(key).system_prompt
        assert "action=answer" not in prompt

    @pytest.mark.parametrize("key", ["evocua@desktop@use", "evocua@browser@use"])
    def test_prompt_orders_the_declared_terminate_spelling(self, key):
        adapter = AgentAdapterRegistry.get(key)
        # The literal, not a resolver call: N19(b) deleted native_name_for_canonical
        # because its fallback returned "response" for this very space while the renderer
        # raises on "response". The declared spelling is "terminate"; see
        # QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES in evocua/action_space.py.
        native = "terminate"
        assert f"action={native} in the tool call." in adapter.system_prompt
        # Not papered over by deleting the sentence: the model is still told
        # how to finish.
        assert "If finishing" in adapter.system_prompt

    def test_the_ordered_finish_name_round_trips_through_the_surface(self):
        """The name in the prompt parses IN and renders OUT — unlike ``answer``."""
        adapter = EvoCUADesktopUseAdapter(metadata=_metadata(include_terminate=True))
        native = "terminate"
        assert f"action={native}" in adapter.system_prompt
        parsed = adapter.action_space.convert_tool_calls_from_agent(
            [
                {"name": "computer_use", "arguments": {"action": native, "status": "success"}},
            ]
        )
        assert [tool_call_name(c) for c in parsed] == ["terminate"]
        assert adapter._tool_calls_to_agent_ordered(parsed) == [
            {
                "name": "computer_use",
                "arguments": {"action": native, "status": "success"},
            }
        ]

    def test_prompt_is_no_longer_qwen3_vls_object(self):
        """The two families' prompts diverge; qwen3_5 keeps sharing qwen3_vl's.

        qwen3_5 may share the OBJECT only while it spells the finish tools
        identically — and it does by construction, subclassing without
        overriding the table. If qwen3_5 ever declares its own spellings, this
        fails and it needs its own prompt copy too.
        """
        from lite.agents.models.evocua import adapter as evocua_adapter
        from lite.agents.models.qwen3_5 import adapter as qwen3_5_adapter
        from lite.agents.models.qwen3_5.action_space import Qwen3_5DesktopActionSpace
        from lite.agents.models.qwen3_vl import adapter as qwen3_vl_adapter
        from lite.agents.models.qwen3_vl.action_space import Qwen3VLDesktopActionSpace

        assert evocua_adapter.USE_SYSTEM_PROMPT != qwen3_vl_adapter.USE_SYSTEM_PROMPT
        assert qwen3_5_adapter.USE_SYSTEM_PROMPT is qwen3_vl_adapter.USE_SYSTEM_PROMPT
        assert (
            Qwen3_5DesktopActionSpace.QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES
            is Qwen3VLDesktopActionSpace.QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES
        )


# =============================================================================
# 3. Tools section contains key_down/key_up
# =============================================================================


class TestToolsSection:
    @staticmethod
    def _rendered_schemas(adapter):
        lines = adapter._build_tools_section().splitlines()
        start = lines.index("<tools>") + 1
        end = lines.index("</tools>")
        schemas = [json.loads(line) for line in lines[start:end] if line.strip()]
        # The rendered tools block carries canonical nested Lite function-tool
        # schemas; require and unwrap that shape so the assertions below read the
        # schema body. Direct action-space assertions use the canonical Lite schema
        # readers instead.
        assert all(set(s) == {"type", "function"} for s in schemas)
        return [s["function"] for s in schemas]

    def test_tools_section_has_key_down_key_up(self):
        adapter = AgentAdapterRegistry.get("evocua@desktop@use")
        tools_text = adapter._build_tools_section()
        assert "key_down" in tools_text
        assert "key_up" in tools_text

    def test_tools_section_enum_no_hscroll_or_answer(self):
        """hscroll/answer appear in description text (matching reference) but NOT in the enum."""
        adapter = AgentAdapterRegistry.get("evocua@desktop@use")
        schemas = adapter.action_space.get_tool_schemas()
        action_enum = set(tool_schema_parameters(schemas[0])["properties"]["action"]["enum"])
        assert "hscroll" not in action_enum
        assert "answer" not in action_enum

    def test_tools_section_filters_native_terminate_without_extra_schema(self):
        adapter = EvoCUADesktopUseAdapter(metadata=_metadata())
        schemas = self._rendered_schemas(adapter)
        assert [s["name"] for s in schemas] == ["computer_use"]
        enum = schemas[0]["parameters"]["properties"]["action"]["enum"]
        assert "terminate" not in enum
        assert "answer" not in enum
        assert "open" not in enum

    def test_action_space_schema_has_computer_use_name(self):
        adapter = AgentAdapterRegistry.get("evocua@desktop@use")
        schemas = adapter.action_space.get_tool_schemas()
        assert tool_schema_name(schemas[0]) == "computer_use"

    def test_action_space_schema_has_no_provider_specific_fields(self):
        adapter = AgentAdapterRegistry.get("evocua@desktop@use")
        schemas = adapter.action_space.get_tool_schemas()
        assert set(schemas[0]) == {"type", "function"}
        assert "name_for_human" not in schemas[0]
        assert "args_format" not in schemas[0]
        assert "name_for_human" not in schemas[0]["function"]
        assert "args_format" not in schemas[0]["function"]

    def test_terminate_extra_schema_does_not_duplicate_native_finish_tool(self):
        adapter = EvoCUADesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=("desktop", "use"), extra_tool_schemas=[copy.deepcopy(_TERMINATE_SCHEMA)]
            )
        )
        schemas = self._rendered_schemas(adapter)
        assert [s["name"] for s in schemas] == ["computer_use"]
        enum = schemas[0]["parameters"]["properties"]["action"]["enum"]
        assert "terminate" in enum
        rendered = adapter._tool_calls_to_agent_ordered(
            [
                make_tool_call("terminate", {"status": "success"}),
            ]
        )
        assert rendered == [
            {
                "name": "computer_use",
                "arguments": {"action": "terminate", "status": "success"},
            }
        ]

    def test_native_terminate_render_does_not_require_extra_schema(self):
        adapter = EvoCUADesktopUseAdapter()
        rendered = adapter._tool_calls_to_agent_ordered(
            [
                make_tool_call("terminate", {"status": "success"}),
            ]
        )
        assert rendered == [
            {
                "name": "computer_use",
                "arguments": {"action": "terminate", "status": "success"},
            }
        ]

# =============================================================================
# 4. Response parsing (inherited)
# =============================================================================


class TestResponseParsing:
    def test_parse_action_with_tool_call(self):
        """parse_raw only strips chat-template tokens; ``Action:`` prefix
        stays in text (system-prompt layer, removed downstream by
        convert_from_agent)."""
        adapter = AgentAdapterRegistry.get("evocua@desktop@use")
        response = (
            "Action: Hold shift key\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "key_down", "keys": ["shift"]}}\n'
            "</tool_call>"
        )
        result = adapter.parse_raw_assistant_response(response)
        assert result["role"] == "assistant"
        assert result["content"][0]["text"] == "Action: Hold shift key"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["arguments"]["action"] == "key_down"
        # Downstream convert_from_agent extracts the action text.
        lite = adapter.convert_message_from_agent(result)
        action_part = next((c for c in lite["content"] if c["type"] == "action_description"), None)
        assert action_part == {"type": "action_description", "text": "Hold shift key"}

    def test_parse_with_think_tags(self):
        adapter = AgentAdapterRegistry.get("evocua@desktop@use")
        response = (
            "<think>I need to click the button</think>\n"
            "Action: Click the submit button\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [500, 300]}}\n'
            "</tool_call>"
        )
        result = adapter.parse_raw_assistant_response(response)
        assert "reasoning_content" in result
        # parse_raw's cleaned text has ``Action:`` prefix preserved.
        assert result["content"][0]["text"] == "Action: Click the submit button"
        # Full pipeline: convert_from_agent strips the prefix.
        lite = adapter.convert_message_from_agent(result)
        action_part = next((c for c in lite["content"] if c["type"] == "action_description"), None)
        assert action_part == {"type": "action_description", "text": "Click the submit button"}


# =============================================================================
# Characterization tests — lock down CURRENT behavior of the EvoCUA desktop
# trajectory adapter. Goldens captured via a throwaway capture script running
# against this commit of cua-lite; embedded below so this file is
# self-contained. Browser adapter is a thin alias over desktop — covered
# transitively via the existing TestAdapterRegistry/TestActionSpaceType tests.
#
# Grounding task type is out of scope (evocua only ships trajectory adapters).
# =============================================================================

# -----------------------------------------------------------------------------
# Raw model outputs — one per supported computer_use action enum value.
# (Format matches USE_SYSTEM_PROMPT: "Action: ...\n<tool_call>".)
# -----------------------------------------------------------------------------

_DESKTOP_RAWS: dict[str, str] = {
    "left_click": (
        "Action: Click button.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "left_click", '
        '"coordinate": [500, 300]}}\n'
        "</tool_call>"
    ),
    "right_click": (
        "Action: Right click.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "right_click", '
        '"coordinate": [100, 200]}}\n'
        "</tool_call>"
    ),
    "double_click": (
        "Action: Double click.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "double_click", '
        '"coordinate": [100, 200]}}\n'
        "</tool_call>"
    ),
    "triple_click": (
        "Action: Triple click.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "triple_click", '
        '"coordinate": [100, 200]}}\n'
        "</tool_call>"
    ),
    "middle_click": (
        "Action: Middle click.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "middle_click", '
        '"coordinate": [100, 200]}}\n'
        "</tool_call>"
    ),
    "type": (
        "Action: type hello.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "type", "text": "hello"}}\n'
        "</tool_call>"
    ),
    "key": (
        "Action: Press ctrl+v.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "key", "keys": ["ctrl", "v"]}}\n'
        "</tool_call>"
    ),
    "key_down": (
        "Action: Hold shift.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "key_down", "keys": ["shift"]}}\n'
        "</tool_call>"
    ),
    "key_up": (
        "Action: Release shift.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "key_up", "keys": ["shift"]}}\n'
        "</tool_call>"
    ),
    "mouse_move": (
        "Action: Move mouse.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "mouse_move", '
        '"coordinate": [300, 400]}}\n'
        "</tool_call>"
    ),
    "left_click_drag": (
        "Action: Drag to target.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "left_click_drag", '
        '"coordinate": [800, 600]}}\n'
        "</tool_call>"
    ),
    "scroll": (
        "Action: Scroll down.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "scroll", '
        '"pixels": -3, "coordinate": [500, 500]}}\n'
        "</tool_call>"
    ),
    "wait": (
        "Action: Wait.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "wait", "time": 3}}\n'
        "</tool_call>"
    ),
    "terminate": (
        "Action: Done.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}\n'
        "</tool_call>"
    ),
}


def _raw_computer_use(arguments: dict) -> str:
    return (
        "Action: Done.\n"
        "<tool_call>\n"
        f"{json.dumps({'name': 'computer_use', 'arguments': arguments})}\n"
        "</tool_call>"
    )


# -----------------------------------------------------------------------------
# LiteMessage goldens — output of parse_raw_assistant_response + convert_from.
# -----------------------------------------------------------------------------

# LITE goldens: assistant action descriptions are tagged as
# ``ActionDescriptionContent`` (paired with tool_calls) — distinct from the
# raw ``text`` kind used for plain replies.
_DESKTOP_LITE_GOLDEN: dict[str, dict] = {
    "left_click": {
        "role": "assistant",
        "tool_calls": [_computer_action("click", coordinate=[500, 300])],
        "content": [{"type": "action_description", "text": "Click button."}],
    },
    "right_click": {
        "role": "assistant",
        "tool_calls": [_computer_action("click", coordinate=[100, 200], button="right")],
        "content": [{"type": "action_description", "text": "Right click."}],
    },
    "double_click": {
        "role": "assistant",
        "tool_calls": [_computer_action("click", coordinate=[100, 200], clicks=2)],
        "content": [{"type": "action_description", "text": "Double click."}],
    },
    "triple_click": {
        "role": "assistant",
        "tool_calls": [_computer_action("click", coordinate=[100, 200], clicks=3)],
        "content": [{"type": "action_description", "text": "Triple click."}],
    },
    "middle_click": {
        "role": "assistant",
        "tool_calls": [_computer_action("click", coordinate=[100, 200], button="middle")],
        "content": [{"type": "action_description", "text": "Middle click."}],
    },
    "type": {
        "role": "assistant",
        "tool_calls": [_computer_action("type", text="hello")],
        "content": [{"type": "action_description", "text": "type hello."}],
    },
    "key": {
        "role": "assistant",
        "tool_calls": [_computer_action("key", keys=["ctrl", "v"])],
        "content": [{"type": "action_description", "text": "Press ctrl+v."}],
    },
    "key_down": {
        "role": "assistant",
        "tool_calls": [_computer_action("key_down", keys=["shift"])],
        "content": [{"type": "action_description", "text": "Hold shift."}],
    },
    "key_up": {
        "role": "assistant",
        "tool_calls": [_computer_action("key_up", keys=["shift"])],
        "content": [{"type": "action_description", "text": "Release shift."}],
    },
    "mouse_move": {
        "role": "assistant",
        "tool_calls": [_computer_action("mouse_move", coordinate=[300, 400])],
        "content": [{"type": "action_description", "text": "Move mouse."}],
    },
    "left_click_drag": {
        "role": "assistant",
        # OBSERVED: EvoCUA `left_click_drag` (with only end coordinate) maps to
        # cua-lite `drag(coordinate=...)` — no start_coordinate in the Lite form.
        "tool_calls": [_computer_action("drag", coordinate=[800, 600])],
        "content": [{"type": "action_description", "text": "Drag to target."}],
    },
    "scroll": {
        "role": "assistant",
        # EvoCUA writes scrolls as raw notch counts -- every value it emitted
        # across its OSWorld evals is in 1..20 -- so ``pixels=-3`` IS three
        # wheel clicks down, not three hundredths of a screen.
        "tool_calls": [
            _computer_action("scroll", coordinate=[500, 500], direction="down", amount=3)
        ],
        "content": [{"type": "action_description", "text": "Scroll down."}],
    },
    "wait": {
        "role": "assistant",
        # OBSERVED: raw emits int `time=3`; from_agent coerces to float `duration=3.0`.
        "tool_calls": [_computer_action("wait", duration=3.0)],
        "content": [{"type": "action_description", "text": "Wait."}],
    },
    "terminate": {
        "role": "assistant",
        "tool_calls": [make_tool_call("terminate", {"status": "success"})],
        "content": [{"type": "action_description", "text": "Done."}],
    },
}


# -----------------------------------------------------------------------------
# Agent-message goldens — output of convert_message_to_agent(LiteMessage).
# The "Action: " prefix is prepended to content[0].text by convert_to_agent
# (the navigation adapter is Action-only by construction).
# -----------------------------------------------------------------------------

_DESKTOP_AGENT_GOLDEN: dict[str, dict] = {
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
    "right_click": {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "computer_use",
                "arguments": {"action": "right_click", "coordinate": [100, 200]},
            }
        ],
        "content": [{"type": "text", "text": "Action: Right click."}],
    },
    "double_click": {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "computer_use",
                "arguments": {"action": "double_click", "coordinate": [100, 200]},
            }
        ],
        "content": [{"type": "text", "text": "Action: Double click."}],
    },
    "triple_click": {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "computer_use",
                "arguments": {"action": "triple_click", "coordinate": [100, 200]},
            }
        ],
        "content": [{"type": "text", "text": "Action: Triple click."}],
    },
    "middle_click": {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "computer_use",
                "arguments": {"action": "middle_click", "coordinate": [100, 200]},
            }
        ],
        "content": [{"type": "text", "text": "Action: Middle click."}],
    },
    "type": {
        "role": "assistant",
        "tool_calls": [{"name": "computer_use", "arguments": {"action": "type", "text": "hello"}}],
        "content": [{"type": "text", "text": "Action: type hello."}],
    },
    "key": {
        "role": "assistant",
        "tool_calls": [
            {"name": "computer_use", "arguments": {"action": "key", "keys": ["ctrl", "v"]}}
        ],
        "content": [{"type": "text", "text": "Action: Press ctrl+v."}],
    },
    "key_down": {
        "role": "assistant",
        "tool_calls": [
            {"name": "computer_use", "arguments": {"action": "key_down", "keys": ["shift"]}}
        ],
        "content": [{"type": "text", "text": "Action: Hold shift."}],
    },
    "key_up": {
        "role": "assistant",
        "tool_calls": [
            {"name": "computer_use", "arguments": {"action": "key_up", "keys": ["shift"]}}
        ],
        "content": [{"type": "text", "text": "Action: Release shift."}],
    },
    "mouse_move": {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "computer_use",
                "arguments": {"action": "mouse_move", "coordinate": [300, 400]},
            }
        ],
        "content": [{"type": "text", "text": "Action: Move mouse."}],
    },
    "left_click_drag": {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "computer_use",
                "arguments": {"action": "left_click_drag", "coordinate": [800, 600]},
            }
        ],
        "content": [{"type": "text", "text": "Action: Drag to target."}],
    },
    "scroll": {
        "role": "assistant",
        # Rendered in EvoCUA's own dialect (``SCROLL_WIRE_UNIT = 1``), so the
        # wire value equals the click count -- the round trip is byte-exact.
        # OBSERVED: on convert_to_agent, `coordinate` is emitted BEFORE `pixels`
        # (argument order in `_convert_single_to_agent` injects coordinate via
        # the `computer_use(...)` kwargs, and computer_use's own signature puts
        # `coordinate` before `pixels`).
        "tool_calls": [
            {
                "name": "computer_use",
                "arguments": {"action": "scroll", "coordinate": [500, 500], "pixels": -3},
            }
        ],
        "content": [{"type": "text", "text": "Action: Scroll down."}],
    },
    "wait": {
        "role": "assistant",
        # OBSERVED: raw int time=3 → float time=3.0 after round-trip through
        # the Lite canonical `duration: float` contract.
        "tool_calls": [{"name": "computer_use", "arguments": {"action": "wait", "time": 3.0}}],
        "content": [{"type": "text", "text": "Action: Wait."}],
    },
    "terminate": {
        "role": "assistant",
        "tool_calls": [
            {"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}
        ],
        "content": [{"type": "text", "text": "Action: Done."}],
    },
}


# -----------------------------------------------------------------------------
# Helpers: build a multi-turn desktop LiteSample from a list of raw responses.
# -----------------------------------------------------------------------------


def _build_desktop_trajectory(raws: list[str], task: str = "Do tasks") -> LiteSample:
    adapter = EvoCUADesktopUseAdapter(
        metadata=_metadata(include_terminate=any("terminate" in raw for raw in raws))
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
        metadata=_metadata(include_terminate=any("terminate" in raw for raw in raws)),
        messages=messages,
        images=[f"img{i}.png" for i in range(len(raws))],
    )


# -----------------------------------------------------------------------------
# 1. Raw → LiteMessage (parse + convert_from)
# -----------------------------------------------------------------------------


class TestRawToLiteMessage:
    """raw model output → parse → convert_from_agent == captured LiteMessage."""

    @pytest.mark.parametrize("action", list(_DESKTOP_RAWS.keys()))
    def test_desktop(self, action):
        adapter = _adapter_for_action(action)
        raw = _DESKTOP_RAWS[action]
        parsed = adapter.parse_raw_assistant_response(raw)
        lite = adapter.convert_message_from_agent(parsed)
        assert lite == _DESKTOP_LITE_GOLDEN[action]

    def test_inactive_native_terminate_raw_reaches_env_owned_feedback(self):
        adapter = EvoCUADesktopUseAdapter(metadata=_metadata())
        parsed = adapter.parse_raw_assistant_response(_DESKTOP_RAWS["terminate"])
        lite = adapter.convert_message_from_agent(parsed)

        assert lite.get("tool_calls") == [make_tool_call("terminate", {"status": "success"})]
        assert pop_model_output_error(copy.deepcopy(lite)) is None

    @pytest.mark.parametrize(
        "arguments",
        [
            {"action": "terminate", "status": "bogus"},
            {"action": "terminate"},
        ],
    )
    def test_active_native_terminate_raw_keeps_arguments_for_env_feedback(self, arguments):
        adapter = EvoCUADesktopUseAdapter(metadata=_metadata(include_terminate=True))
        parsed = adapter.parse_raw_assistant_response(_raw_computer_use(arguments))
        lite = adapter.convert_message_from_agent(parsed)

        expected = {k: v for k, v in arguments.items() if k != "action"}
        assert lite.get("tool_calls") == [make_tool_call("terminate", expected)]
        assert pop_model_output_error(copy.deepcopy(lite)) is None


# -----------------------------------------------------------------------------
# 2. LiteMessage → Raw (convert_to_agent output)
# -----------------------------------------------------------------------------


class TestLiteMessageToRaw:
    """LiteMessage → convert_message_to_agent == captured agent message."""

    @pytest.mark.parametrize("action", list(_DESKTOP_RAWS.keys()))
    def test_desktop(self, action):
        adapter = _adapter_for_action(action)
        lite = copy.deepcopy(_DESKTOP_LITE_GOLDEN[action])
        agent_msg = adapter.convert_message_to_agent(lite)
        assert agent_msg == _DESKTOP_AGENT_GOLDEN[action]


# -----------------------------------------------------------------------------
# 3. Raw round-trip: raw → parse → from → to → tool-call JSON
# -----------------------------------------------------------------------------


class TestRawRoundTrip:
    """Full adapter round-trip preserves the tool_call JSON and action text.

    Lossy fields (documented inline via OBSERVED notes in the goldens):
      - ``wait(time=3)`` (int) becomes ``wait(time=3.0)`` (float).
      - ``scroll`` argument-dict ordering: ``coordinate`` precedes ``pixels``
        after convert_to_agent even if the raw JSON had them in the opposite
        order.
      - ``content[0].text`` loses its ``"Action: "`` prefix on
        convert_from_agent (extracted by the navigation adapter's ``Action:``
        regex) and regains it on convert_to_agent (hard-coded prefix).
    """

    @pytest.mark.parametrize("action", list(_DESKTOP_RAWS.keys()))
    def test_desktop_tool_call_json_round_trip(self, action):
        adapter = _adapter_for_action(action)
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
# 4. 3-turn desktop unroll structure
# -----------------------------------------------------------------------------


class TestUnrollStructure:
    """3-turn desktop unroll yields 3 steps; structure per step matches
    the captured golden (roles + image counts). With full_history_size=4 and
    only 3 turns, no windowing engages — step i contains
    system + (user, assistant) * (i+1)."""

    def test_three_turn_desktop_structure(self):
        adapter = EvoCUADesktopUseAdapter(metadata=_metadata(include_terminate=True))
        sample = _build_desktop_trajectory(
            [
                _DESKTOP_RAWS["left_click"],
                _DESKTOP_RAWS["type"],
                _DESKTOP_RAWS["terminate"],
            ]
        )
        agent_sample = adapter.unroll(sample)
        steps = agent_sample.steps

        assert len(steps) == 3
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
            # Per-step image references = i + 1 distinct indices.
            image_indices = {
                item["index"]
                for m in step
                if m.get("role") == "user"
                for item in (m.get("content") or [])
                if item.get("type") == "image"
            }
            assert len(image_indices) == i + 1


# -----------------------------------------------------------------------------
# 5. Single-turn target byte-exact (per action)
# -----------------------------------------------------------------------------


class TestUnrollByteExactTargetPerAction:
    """For each action, a single-turn trajectory unroll produces a target
    assistant message that matches the captured golden byte-exact.

    NB: EvoCUA/Qwen3VL adapters don't expose a ``format_tool_call_as_text``
    helper; byte-exactness here means the structured message itself matches
    — which is what downstream chat-template rendering consumes.
    """

    @pytest.mark.parametrize("action", list(_DESKTOP_RAWS.keys()))
    def test_desktop_target_byte_exact(self, action):
        adapter = _adapter_for_action(action)
        raw = _DESKTOP_RAWS[action]
        lite_asst = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
        sample = LiteSample(
            metadata=_metadata(include_terminate=(action == "terminate")),
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
        assert asst == _DESKTOP_AGENT_GOLDEN[action]


# -----------------------------------------------------------------------------
# 6. No leakage of future-step info into sample i's user prompt
# -----------------------------------------------------------------------------


class TestUnrollNoLeakage:
    """In a 3-turn desktop trajectory, sample i's user-prompt text must not
    contain the distinguishing action/summary text from target_i (no leakage
    from target into context)."""

    def test_desktop_no_future_leakage(self):
        raws_seq = [_DESKTOP_RAWS["left_click"], _DESKTOP_RAWS["type"], _DESKTOP_RAWS["terminate"]]
        adapter = EvoCUADesktopUseAdapter(metadata=_metadata(include_terminate=True))
        sample = _build_desktop_trajectory(raws_seq)
        steps = adapter.unroll(sample).steps

        # Distinguishing phrases; each should appear only in its own step's
        # target assistant message (not in prompt messages of its step).
        signatures = [
            "Click button",  # step 0 target
            "type hello",  # step 1 target
            "Done",  # step 2 target
        ]

        for i, step in enumerate(steps):
            prompt_msgs = step[:-1]  # everything except the target assistant
            prompt_blob = ""
            for m in prompt_msgs:
                c = m.get("content", [])
                if isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt_blob += item.get("text", "")
            assert signatures[i] not in prompt_blob, (
                f"step {i} leaks its own action {signatures[i]!r}: {prompt_blob!r}"
            )


# -----------------------------------------------------------------------------
# 7. Qwen3VLHistoryProtocol windowing at the full_history_size=4 boundary.
# -----------------------------------------------------------------------------


def _build_protocol_msgs(num_turns: int, task: str = "Task A") -> list[dict]:
    """Build n_turns of (user_with_image, assistant_with_action) messages."""
    adapter = EvoCUADesktopUseAdapter()
    msgs: list[dict] = []
    for i in range(num_turns):
        if i == 0:
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0},
                        {"type": "text", "text": task},
                    ],
                }
            )
        else:
            msgs.append({"role": "user", "content": [{"type": "image", "index": i}]})
        parsed = adapter.parse_raw_assistant_response(_DESKTOP_RAWS["left_click"])
        msgs.append(adapter.convert_message_from_agent(parsed))
    return msgs


class TestProtocolWindowing:
    """Qwen3VLHistoryProtocol(full_history_size=4) behavior at
    boundaries: below / at / above window."""

    def _first_user_text(self, out: list[dict]) -> str:
        c = out[0]["content"]
        return "".join(x["text"] for x in c if x.get("type") == "text")

    def test_at_window_boundary_no_summary(self):
        """n==full_history_size → no drop, summary reads 'None'."""
        proto = Qwen3VLHistoryProtocol(full_history_size=4)
        msgs = _build_protocol_msgs(4)
        out = proto.process_messages(msgs)

        assert len(out) == 8
        assert [m["role"] for m in out] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert self._first_user_text(out) == (
            "Please generate the next move according to the UI screenshot, "
            "instruction and previous actions.\n\n"
            "Instruction: Task A\n\n"
            "Previous actions:\nNone"
        )

    def test_5turn_above_window_summary_one_step(self):
        """5 turns, window=4 → turn 0 collapsed into summary 'Step 1:'."""
        proto = Qwen3VLHistoryProtocol(full_history_size=4)
        msgs = _build_protocol_msgs(5)
        out = proto.process_messages(msgs)

        assert len(out) == 8
        img_counts = []
        for m in out:
            c = m.get("content", [])
            n = (
                sum(1 for item in c if isinstance(item, dict) and item.get("type") == "image")
                if isinstance(c, list)
                else 0
            )
            img_counts.append(n)
        assert img_counts == [1, 0, 1, 0, 1, 0, 1, 0]

        assert self._first_user_text(out) == (
            "Please generate the next move according to the UI screenshot, "
            "instruction and previous actions.\n\n"
            "Instruction: Task A\n\n"
            "Previous actions:\nStep 1: Click button."
        )

    def test_6turn_above_window_summary_two_steps(self):
        """6 turns, window=4 → turns 0,1 collapse into 2-line summary."""
        proto = Qwen3VLHistoryProtocol(full_history_size=4)
        msgs = _build_protocol_msgs(6)
        out = proto.process_messages(msgs)
        assert self._first_user_text(out) == (
            "Please generate the next move according to the UI screenshot, "
            "instruction and previous actions.\n\n"
            "Instruction: Task A\n\n"
            "Previous actions:\nStep 1: Click button.\nStep 2: Click button."
        )


# -----------------------------------------------------------------------------
# 8. Mutation purity — inputs not mutated by any conversion
# -----------------------------------------------------------------------------


class TestMutationPurity:
    """convert_message_to_agent, convert_message_from_agent,
    protocol.process_messages, and unroll_sample must not mutate inputs."""

    def test_convert_to_agent_does_not_mutate(self):
        adapter = EvoCUADesktopUseAdapter()
        lite = copy.deepcopy(_DESKTOP_LITE_GOLDEN["key_down"])
        snapshot = copy.deepcopy(lite)
        _ = adapter.convert_message_to_agent(lite)
        assert lite == snapshot

    def test_convert_from_agent_does_not_mutate(self):
        adapter = EvoCUADesktopUseAdapter()
        agent_msg = copy.deepcopy(_DESKTOP_AGENT_GOLDEN["key_down"])
        snapshot = copy.deepcopy(agent_msg)
        _ = adapter.convert_message_from_agent(agent_msg)
        assert agent_msg == snapshot

    def test_protocol_process_messages_does_not_mutate(self):
        proto = Qwen3VLHistoryProtocol(full_history_size=4)
        msgs = _build_protocol_msgs(5)
        snapshot = copy.deepcopy(msgs)
        _ = proto.process_messages(msgs)
        assert msgs == snapshot

    def test_unroll_sample_does_not_mutate(self):
        adapter = EvoCUADesktopUseAdapter()
        sample = _build_desktop_trajectory(
            [
                _DESKTOP_RAWS["left_click"],
                _DESKTOP_RAWS["type"],
            ]
        )
        snap_messages = copy.deepcopy(sample.messages)
        snap_images = copy.deepcopy(sample.images)
        _ = adapter.unroll(sample)
        assert sample.messages == snap_messages
        assert sample.images == snap_images


# -----------------------------------------------------------------------------
# 9. Sample independence — unrolling twice yields equal but independent samples
# -----------------------------------------------------------------------------


class TestSampleIndependence:
    """Calling unroll_sample twice on the same LiteSample yields equal results
    but independent objects; mutating one does not affect the other."""

    def _sample(self) -> LiteSample:
        return _build_desktop_trajectory(
            [
                _DESKTOP_RAWS["left_click"],
                _DESKTOP_RAWS["type"],
            ]
        )

    def test_two_calls_equal_but_independent(self):
        adapter = EvoCUADesktopUseAdapter()
        sample = self._sample()
        a = adapter.unroll(sample).steps
        b = adapter.unroll(sample).steps

        assert len(a) == len(b) == 2
        for sa, sb in zip(a, b):
            assert sa == sb
            assert sa is not sb

    def test_mutating_one_sample_does_not_affect_siblings(self):
        adapter = EvoCUADesktopUseAdapter()
        sample = self._sample()
        steps = adapter.unroll(sample).steps
        assert len(steps) == 2

        # Mutate target assistant in step[0].
        steps[0][-1]["tool_calls"][0]["arguments"]["action"] = "MUTATED"

        # step[1]'s earlier-turn click (echo of step 0's action) must be unmutated.
        for m in steps[1]:
            for tc in m.get("tool_calls", []) or []:
                assert tc["arguments"].get("action") != "MUTATED"


# -----------------------------------------------------------------------------
# 10. Edge cases: empty / single / 10-turn / no-tool_calls
# -----------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_trajectory_unroll_is_empty(self):
        """Empty LiteSample → unroll returns [] for steps."""
        adapter = EvoCUADesktopUseAdapter()
        empty = LiteSample(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE)
            ),
            messages=[],
            images=[],
        )
        assert adapter.unroll(empty).steps == []

    def test_single_turn_unroll(self):
        """1-turn sample: system + user + target assistant (no windowing)."""
        adapter = EvoCUADesktopUseAdapter()
        sample = _build_desktop_trajectory([_DESKTOP_RAWS["left_click"]])
        steps = adapter.unroll(sample).steps
        assert len(steps) == 1
        assert [m["role"] for m in steps[0]] == ["system", "user", "assistant"]

    def test_10_turn_unroll_windowing_engages(self):
        """10-turn unroll yields 10 steps. With full_history_size=4,
        steps 0-3 are unwindowed; steps 4..9 are windowed and each has
        the same structural shape (system + 4 user/assistant pairs = 9 msgs).

        OBSERVED (captured live): step 9 user #0 carries the summary
        'Step 1..Step 6: Click button.' (the 6 dropped turns)."""
        adapter = EvoCUADesktopUseAdapter()
        sample = _build_desktop_trajectory([_DESKTOP_RAWS["left_click"]] * 10)
        steps = adapter.unroll(sample).steps
        assert len(steps) == 10

        # Expected lengths: step i (0-indexed) has len = min(1 + 2*(i+1), 9).
        expected_lens = [3, 5, 7, 9, 9, 9, 9, 9, 9, 9]
        assert [len(s) for s in steps] == expected_lens

        # Step 9's first user message (post-system) carries a 6-line summary.
        first_user = steps[9][1]
        text = "".join(c["text"] for c in first_user["content"] if c.get("type") == "text")
        for step_n in range(1, 7):
            assert f"Step {step_n}: Click button." in text
        # Step 7+ must NOT be summarized (those are still in the image window).
        assert "Step 7:" not in text

    def test_assistant_without_tool_calls_keeps_plain_text(self):
        """Model outputs prose with no <tool_call>. parse_raw returns a
        text-only AgentMessage (no tool_calls), and convert_from_agent LEAVES it
        as plain ``text``: ``action_description`` narrates an action, and a
        no-tool-call turn is instead the loop's text-oriented termination (or a
        parse failure). Retagging it would hide the text from
        ``lite.core.messages.no_tool_call_final_text`` and truncate
        the episode as ``empty_model_output``. The round trip is byte-exact."""
        from lite.core.messages import no_tool_call_final_text

        adapter = EvoCUADesktopUseAdapter()
        raw = "Action: Just observe the screen."
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
