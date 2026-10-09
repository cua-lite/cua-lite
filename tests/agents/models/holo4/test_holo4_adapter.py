"""CPU-only adapter and registry tests for Holo4."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent import AgentRegistry
from lite.agents.core.protocol import ProtocolRegistry
from lite.agents.factory import LOCAL_AGENTS
from lite.agents.models.holo4.adapter import Holo4DesktopUseAdapter
from lite.agents.models.holo4.agent import Holo4BaseAgent
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.final import pop_model_output_error
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet, LiteShellToolSet
from lite.core.tools.schemas import make_tool_schema, tool_schema_name


def _metadata() -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[
            LiteShellToolSet.get_tool_schema("bash"),
            LiteFinishToolSet.get_tool_schema("response"),
            LiteFinishToolSet.get_tool_schema("terminate"),
            make_tool_schema(
                "custom_lookup",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ],
    )


def test_catalog_and_bootstrap_registration() -> None:
    assert LOCAL_AGENTS["Hcompany/Holo4-27B"] == {
        "agent_id": "holo4",
        "engine_kwargs": {"tp_size": 2},
    }
    assert LOCAL_AGENTS["Hcompany/Holo4-35B-A3B"] == {
        "agent_id": "holo4",
        "engine_kwargs": {"tp_size": 4},
    }
    register_all()
    assert ActionSpaceRegistry.contains("holo4@desktop")
    assert ActionSpaceRegistry.contains("holo4@mobile")
    assert ProtocolRegistry.contains("holo4.history")
    assert AgentAdapterRegistry.contains("holo4@desktop@use")
    assert AgentRegistry.contains("holo4@mobile@use")


def test_projected_tools_replace_only_owned_extra_spellings() -> None:
    adapter = Holo4DesktopUseAdapter(metadata=_metadata())
    names = [tool_schema_name(schema) for schema in adapter._assemble_tool_schemas()]
    assert "bash" not in names
    assert "response" not in names
    assert names.count("terminate") == 1
    assert names.count("shell") == 1
    assert names.count("answer") == 1
    assert names.count("custom_lookup") == 1
    assert adapter._tool_calls_to_agent_ordered(
        [LiteFinishToolSet.terminate(status="success")]
    ) == [{"name": "terminate", "arguments": {"status": "success"}}]


def test_terminate_only_surface_projects_answer_to_successful_terminate() -> None:
    adapter = Holo4DesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("terminate")],
        )
    )
    names = [tool_schema_name(schema) for schema in adapter._assemble_tool_schemas()]
    assert "answer" in names
    assert "terminate" not in names
    message = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "tool_calls": [{"name": "answer", "arguments": {"content": "done"}}],
        }
    )
    assert message["tool_calls"] == [LiteFinishToolSet.terminate(status="success")]


def test_xml_parse_handles_thinking_prose_and_multiple_flat_calls() -> None:
    adapter = Holo4DesktopUseAdapter(metadata=_metadata())
    raw = """<think>Need inspect first.</think>
I am checking the visible page.
<tool_call>
<function=click_desktop>
<parameter=element>Menu</parameter>
<parameter=x>250</parameter>
<parameter=y>120</parameter>
</function>
</tool_call>
<tool_call>
<function=shell>
<parameter=command>pwd</parameter>
</function>
</tool_call>"""
    agent_message = adapter.parse_raw_assistant_response(raw)
    lite_message = adapter.convert_message_from_agent(agent_message)
    assert agent_message["reasoning_content"] == "Need inspect first."
    assert [tool_call_name(call) for call in lite_message["tool_calls"]] == [
        "computer",
        "bash",
    ]
    assert tool_call_arguments(lite_message["tool_calls"][0])["actions"][0]["coordinate"] == [
        250,
        120,
    ]
    assert tool_call_arguments(lite_message["tool_calls"][1]) == {"command": "pwd"}


def test_arbitrary_active_extra_passes_through_unchanged() -> None:
    adapter = Holo4DesktopUseAdapter(metadata=_metadata())
    message = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "tool_calls": [{"name": "custom_lookup", "arguments": {"query": "license"}}],
        }
    )
    assert tool_call_name(message["tool_calls"][0]) == "custom_lookup"
    assert tool_call_arguments(message["tool_calls"][0]) == {"query": "license"}


def test_malformed_xml_is_marked_by_the_inherited_parser() -> None:
    adapter = Holo4DesktopUseAdapter(metadata=_metadata())
    message = adapter.parse_raw_assistant_response(
        "<tool_call><function=click_desktop><parameter=x>100"
    )
    assert "tool_calls" not in message
    assert pop_model_output_error(message) == "malformed <tool_call> XML"


def test_agent_forwards_holo_template_controls() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    agent = SimpleNamespace(
        processor=processor,
        adapter=SimpleNamespace(enable_thinking=True),
        reasoning_effort=None,
    )
    assert Holo4BaseAgent.build_generation_prompt(agent, [{"role": "user"}]) == "prompt"
    processor.apply_chat_template.assert_called_once_with(
        [{"role": "user"}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
        preserve_thinking=False,
    )


def test_agent_forwards_explicit_reasoning_effort() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    agent = SimpleNamespace(
        processor=processor,
        adapter=SimpleNamespace(enable_thinking=True),
        reasoning_effort="medium",
    )
    assert Holo4BaseAgent.build_generation_prompt(agent, [{"role": "user"}]) == "prompt"
    processor.apply_chat_template.assert_called_once_with(
        [{"role": "user"}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
        preserve_thinking=False,
        reasoning_effort="medium",
    )


def test_render_step_merges_task_system_with_holo_system() -> None:
    adapter = Holo4DesktopUseAdapter(metadata=_metadata())
    sample = LiteSample(
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
        messages=[
            {"role": "system", "content": "Task-specific policy."},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Inspect the page."}],
            },
        ],
    )

    rendered = adapter.render_step(sample, 1, [])

    assert [message["role"] for message in rendered].count("system") == 1
    assert rendered[0]["role"] == "system"
    system_text = "\n".join(part["text"] for part in rendered[0]["content"])
    assert "Task-specific policy." in system_text
    assert "You are Holo, a computer-use agent." in system_text


def test_holo_replay_uses_canonical_text_and_tool_calls() -> None:
    adapter = Holo4DesktopUseAdapter(metadata=_metadata())
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "I will inspect the page."}],
        "reasoning_content": "private reasoning",
        "raw_response": {
            "adapter_key": adapter._registry_key,
            "text": "<think>private reasoning</think>\n<孤立模板标记>",
        },
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "custom_lookup",
                    "arguments": {"query": "visible page"},
                },
            }
        ],
    }

    rendered = adapter.convert_message_to_agent(message)

    assert "private reasoning" not in rendered["content"][0]["text"]
    assert "<孤立模板标记>" not in rendered["content"][0]["text"]
    assert "I will inspect the page." in rendered["content"][0]["text"]
    assert "<tool_call>" in rendered["content"][0]["text"]
    assert "raw_response" not in rendered
    assert "reasoning_content" not in rendered
