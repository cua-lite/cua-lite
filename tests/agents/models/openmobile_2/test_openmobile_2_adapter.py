"""OpenMobile-2 adapter tests: official qwen35_thought_session alignment."""

from __future__ import annotations

import hashlib

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent import AgentRegistry
from lite.agents.core.protocol import ProtocolRegistry
from lite.agents.factory import LOCAL_AGENTS
from lite.agents.models.openmobile_2.adapter import (
    OPENMOBILE2_SYSTEM_PROMPT,
    OpenMobile2MobileUseAdapter,
)
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.final import pop_model_output_error

_XML_RESPONSE = (
    "Thought: Tap the search icon.\n"
    "<tool_call>\n"
    "<function=mobile_use>\n"
    "<parameter=action>click</parameter>\n"
    "<parameter=coordinate>[163, 718]</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

_JSON_RESPONSE = (
    "Thought: Type the query.\n"
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "type", "text": "hi"}}\n</tool_call>'
)


def _adapter() -> OpenMobile2MobileUseAdapter:
    return OpenMobile2MobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"))
    )


def _sample(messages: list[dict]) -> LiteSample:
    return LiteSample(
        metadata=LiteCUAMetadata(dims=("mobile", "use")), messages=messages
    )


def test_system_prompt_is_pinned_byte_exact_to_the_official_constant() -> None:
    """The prompt must stay byte-identical to the official
    ``QWEN35_SESSION_SYSTEM_PROMPT_THOUGHT``
    (OpenMobile2-Code/runtimes/prompts.py:667-726). A digest mismatch means
    the copy drifted; re-diff against the official file before updating."""
    assert (
        hashlib.sha256(OPENMOBILE2_SYSTEM_PROMPT.encode()).hexdigest()
        == "340c48f72b0d9a97fbff87cc6b5af52ec75e6d788b8773bb53ec9578d67748c1"
    )
    # Structural anchors: embedded schema, no separate tools section needed.
    assert OPENMOBILE2_SYSTEM_PROMPT.startswith("\n# Tools\n")
    assert '"name": "mobile_use"' in OPENMOBILE2_SYSTEM_PROMPT
    assert '"enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"]' in OPENMOBILE2_SYSTEM_PROMPT


def test_adapter_pins_the_official_wire_knobs() -> None:
    adapter = _adapter()
    # The official prompt embeds the mobile_use schema, so no Qwen3.5 XML
    # tools preamble is layered on top, and thinking stays off.
    assert adapter.system_prompt == OPENMOBILE2_SYSTEM_PROMPT
    assert adapter.render_tools_section is False
    assert adapter.enable_thinking is False
    # Client-side smart_resize reproduces the official serving grid
    # (1080x2400 -> 1088x2400).
    assert adapter.smart_resize_enabled is True
    # response/terminate fold into the mobile_use answer/terminate action
    # values; open_app does not (no `open` in the official enum).
    assert set(type(adapter).natively_rendered_extra_tool_names) == {
        "response",
        "terminate",
    }


def test_render_step_serves_the_official_session_shape() -> None:
    adapter = _adapter()
    sample = _sample(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Add milk to the cart"},
                ],
            }
        ]
    )

    rendered = adapter.render_step(sample, 1, [])

    assert [message["role"] for message in rendered] == ["system", "user"]
    # System message: the official prompt alone — no tools section, no
    # finish guidance appended.
    assert rendered[0]["content"] == [
        {"type": "text", "text": OPENMOBILE2_SYSTEM_PROMPT}
    ]
    # First user message: query text first, screenshot after (official
    # QWEN35_SESSION_USER_PROMPT order).
    assert rendered[1]["content"] == [
        {"type": "text", "text": "The user query: Add milk to the cart\n"},
        {"type": "image", "index": 0},
    ]


def test_render_step_replays_assistant_turns_as_thought_plus_inline_xml() -> None:
    adapter = _adapter()
    sample = _sample(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "task"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Thought: Open the app."}],
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "mobile_use",
                            "arguments": {"action": "click", "coordinate": [500, 500]},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": [{"type": "image", "index": 1}],
            },
        ]
    )

    rendered = adapter.render_step(sample, 99, [])

    assert [message["role"] for message in rendered] == ["system", "user", "assistant", "tool"]
    assistant_text = "".join(
        part["text"] for part in rendered[2]["content"] if part.get("type") == "text"
    )
    assert assistant_text.startswith("Thought: Open the app.")
    assert "<tool_call>" in assistant_text
    assert "<function=mobile_use>" in assistant_text
    # The qwen3_5 renderer re-serializes parsed calls in its canonical XML
    # wire format — parameter values on their own lines (the official
    # harness replays the model's own output verbatim; same XML, different
    # whitespace).
    assert "<parameter=action>\nclick\n</parameter>" in assistant_text
    assert "<parameter=coordinate>\n[500, 500]\n</parameter>" in assistant_text
    # Tool observation passes through as the pure screenshot message.
    assert rendered[3]["content"] == [{"type": "image", "index": 1}]


def test_render_step_collapses_old_screenshots_beyond_last_n() -> None:
    # End-to-end protocol composition: 5 image-bearing messages, last_n=3 —
    # the initial screenshot collapses while its query text survives.
    adapter = _adapter()
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "task"},
            ],
        }
    ]
    for i in range(1, 5):
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Thought: step {i}."}],
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": "mobile_use",
                            "arguments": {"action": "wait", "time": 1},
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": [{"type": "image", "index": i}],
            }
        )

    rendered = adapter.render_step(_sample(messages), 99, [])

    assert rendered[1]["content"] == [
        {"type": "text", "text": "The user query: task\n"},
        {"type": "text", "text": "This screenshot has been collapsed."},
    ]
    assert rendered[3]["content"] == [
        {"type": "text", "text": "This screenshot has been collapsed."}
    ]
    assert rendered[5]["content"] == [{"type": "image", "index": 2}]
    assert rendered[7]["content"] == [{"type": "image", "index": 3}]
    assert rendered[9]["content"] == [{"type": "image", "index": 4}]


def test_parse_prefers_xml_and_falls_back_to_json_tool_calls() -> None:
    # Official qwen35 dialect order: _parse_tool_call_xml(...) or
    # _parse_tool_call_json(...).
    adapter = _adapter()

    xml = adapter.parse_raw_assistant_response(_XML_RESPONSE)
    assert xml["tool_calls"] == [
        {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [163, 718]}}
    ]
    assert xml["content"] == [{"type": "text", "text": "Thought: Tap the search icon."}]
    assert pop_model_output_error(xml) is None

    json_bodied = adapter.parse_raw_assistant_response(_JSON_RESPONSE)
    assert json_bodied["tool_calls"] == [
        {"name": "mobile_use", "arguments": {"action": "type", "text": "hi"}}
    ]
    assert pop_model_output_error(json_bodied) is None


def test_parse_marks_malformed_tool_calls_and_passes_plain_text_through() -> None:
    adapter = _adapter()

    malformed = adapter.parse_raw_assistant_response(
        "Thought: broken.\n<tool_call>not xml or json</tool_call>"
    )
    assert "tool_calls" not in malformed
    assert pop_model_output_error(malformed) is not None

    plain = adapter.parse_raw_assistant_response("Thought: done looking around.")
    assert "tool_calls" not in plain
    assert pop_model_output_error(plain) is None
    assert plain["content"] == [
        {"type": "text", "text": "Thought: done looking around."}
    ]


def test_catalog_and_bootstrap_registration() -> None:
    assert LOCAL_AGENTS["OpenMobile-2/OpenMobile-2-9B"] == {
        "agent_id": "openmobile_2",
        "agent_kwargs": {"enable_thinking": False},
    }
    assert LOCAL_AGENTS["OpenMobile-2/OpenMobile-2-27B"] == {
        "agent_id": "openmobile_2",
        "agent_kwargs": {"enable_thinking": False},
        "engine_kwargs": {"tp_size": 2},
    }
    register_all()
    assert ActionSpaceRegistry.contains("openmobile_2@mobile")
    assert ProtocolRegistry.contains("openmobile_2.history")
    assert AgentAdapterRegistry.contains("openmobile_2@mobile@use")
    assert AgentRegistry.contains("openmobile_2@mobile@use")
