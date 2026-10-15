"""CPU-only adapter and registry tests for Holo4."""

from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent import AgentRegistry
from lite.agents.core.protocol import ProtocolRegistry
from lite.agents.factory import LOCAL_AGENTS
from lite.agents.models.holo4.adapter import (
    HOLO4_ANDROIDWORLD_APP_NOTES,
    HOLO4_DESKTOP_GENERAL_SYSTEM_PROMPT,
    HOLO4_MOBILE_SYSTEM_PROMPT,
    HOLO4_OSWORLD_SYSTEM_PROMPTS,
    Holo4DesktopUseAdapter,
    Holo4MobileUseAdapter,
)
from lite.agents.models.holo4.agent import Holo4BaseAgent
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.final import pop_model_output_error
from lite.core.tools.action_space import LiteMobileActionSet
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet, LiteShellToolSet, make_open_app_tool
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


def test_mobile_prompt_adds_androidworld_app_notes_only_for_androidworld() -> None:
    androidworld = Holo4MobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"), others={"env_id": "androidworld"})
    )
    other_mobile = Holo4MobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"), others={"env_id": "androidlab"})
    )

    androidworld_prompt = androidworld._system_prompt_for_active_surface()
    other_mobile_prompt = other_mobile._system_prompt_for_active_surface()
    assert HOLO4_ANDROIDWORLD_APP_NOTES in androidworld_prompt
    assert other_mobile_prompt.startswith("You are Holo, a mobile-use agent")
    assert "# Workflow" in other_mobile_prompt
    assert "# Termination" in other_mobile_prompt
    assert "# App notes" not in other_mobile_prompt
    assert "# Planning" not in other_mobile_prompt
    assert "# Notes" not in other_mobile_prompt


def test_mobile_prompt_keeps_plan_and_note_sections_only_with_the_tools() -> None:
    note = make_tool_schema(
        "note",
        parameters={
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    )
    update_plan = make_tool_schema(
        "update_plan",
        parameters={
            "type": "object",
            "properties": {"goals": {"type": "array"}},
            "required": ["goals"],
        },
    )

    without_tools = Holo4MobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"), others={"env_id": "androidworld"})
    )._system_prompt_for_active_surface()
    assert "# Planning" not in without_tools
    assert "# Notes" not in without_tools
    assert "# App notes" in without_tools
    assert "# Termination" in without_tools

    with_tools = Holo4MobileUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("mobile", "use"),
            others={"env_id": "androidworld"},
            extra_tool_schemas=[note, update_plan],
        )
    )._system_prompt_for_active_surface()
    assert with_tools.startswith(HOLO4_MOBILE_SYSTEM_PROMPT)
    assert "# Planning" in with_tools
    assert "# Notes" in with_tools


def test_mobile_prompt_names_open_app_and_ask_user_only_when_advertised() -> None:
    # The official body rules both tools out ("go through the drawer",
    # "nobody is standing by"); an env that advertises them gets the bullet
    # naming the tool instead of a prompt that contradicts the tool list.
    open_app = make_open_app_tool(["Settings"])
    ask_user = make_tool_schema(
        "ask_user",
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    )

    without_tools = Holo4MobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"), others={"env_id": "mobileworld"})
    )._system_prompt_for_active_surface()
    assert "- To open an app, go through the drawer: " in without_tools
    assert "open_app" not in without_tools
    assert "ask_user" not in without_tools

    with_tools = Holo4MobileUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("mobile", "use"),
            others={"env_id": "mobileworld"},
            extra_tool_schemas=[open_app, ask_user],
        )
    )._system_prompt_for_active_surface()
    assert (
        "- To open an app, call `open_app` with its name from the tool's list. "
        "For an app the list does not carry, go through the drawer: "
    ) in with_tools
    assert "The one exception is `ask_user`" in with_tools
    # Only those two bullets change.
    assert len(with_tools.splitlines()) == len(without_tools.splitlines())
    assert len(set(with_tools.splitlines()) - set(without_tools.splitlines())) == 2


def test_mobile_custom_prompt_on_androidworld_appends_env_app_notes() -> None:
    custom = "Custom mobile policy."
    prompt = Holo4MobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"), others={"env_id": "androidworld"}),
        system_prompt=custom,
    )._system_prompt_for_active_surface()
    assert prompt == f"{custom}\n\n{HOLO4_ANDROIDWORLD_APP_NOTES}"


def test_mobile_promptless_androidworld_serves_app_notes_alone() -> None:
    prompt = Holo4MobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"), others={"env_id": "androidworld"}),
        system_prompt=None,
    )._system_prompt_for_active_surface()
    assert prompt == HOLO4_ANDROIDWORLD_APP_NOTES


def test_mobile_official_body_pinned_on_androidworld_keeps_one_app_notes() -> None:
    # Pinning the assembled official body must not append the app notes a
    # second time; with the plan/note tools active it is served unchanged.
    note = make_tool_schema(
        "note",
        parameters={
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    )
    update_plan = make_tool_schema(
        "update_plan",
        parameters={
            "type": "object",
            "properties": {"goals": {"type": "array"}},
            "required": ["goals"],
        },
    )
    prompt = Holo4MobileUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("mobile", "use"),
            others={"env_id": "androidworld"},
            extra_tool_schemas=[note, update_plan],
        ),
        system_prompt=HOLO4_MOBILE_SYSTEM_PROMPT,
    )._system_prompt_for_active_surface()
    assert prompt == HOLO4_MOBILE_SYSTEM_PROMPT


def test_desktop_prompt_omits_uncarried_official_tools() -> None:
    for token in (
        "zoom_desktop",
        "poll_execution",
        "document_ocr",
        "transcribe_audio",
        "# Session",
        "budget alert",
        "<PII removed>",
    ):
        for prompt in (HOLO4_DESKTOP_GENERAL_SYSTEM_PROMPT, *HOLO4_OSWORLD_SYSTEM_PROMPTS.values()):
            assert token not in prompt


def test_desktop_general_prompt_is_a_subtraction_of_the_official_body() -> None:
    # Every general line is an official line, except the one "Working on
    # files" bullet whose /home/user paths were dropped.
    official_lines = set(HOLO4_OSWORLD_SYSTEM_PROMPTS["osworld"].splitlines())
    extra = set(HOLO4_DESKTOP_GENERAL_SYSTEM_PROMPT.splitlines()) - official_lines
    assert extra == {
        "- The file to edit is the one the application has open. Take its path "
        "from the window title or the task text. Never take the first `find` "
        "hit as the target."
    }


def test_osworld_prompt_states_the_cua_lite_desktop_facts() -> None:
    # The official system information describes H Company's own VM; the
    # served body must describe the stock OSWorld desktop instead.
    for env_id, password in (("osworld", "password"), ("lite.osworld", "user")):
        prompt = HOLO4_OSWORLD_SYSTEM_PROMPTS[env_id]
        assert "OS: Ubuntu 22.04, GNOME" in prompt
        assert f"with password '{password}'" in prompt
        assert f"`echo '{password}' | sudo -S <cmd>`" in prompt
        assert "The dock has no Terminal entry." in prompt
        for token in (
            "Ubuntu 24.04",
            "osworld-public-evaluation",
            "uv pip install",
            "10. Terminal",
            "`curl` is absent",
            "xdotool",
        ):
            assert token not in prompt
    # Only the password differs between the two env bodies.
    assert HOLO4_OSWORLD_SYSTEM_PROMPTS["lite.osworld"] == HOLO4_OSWORLD_SYSTEM_PROMPTS[
        "osworld"
    ].replace("'password'", "'user'")


def test_osworld_prompt_digest() -> None:
    """Pins the assembled OSWorld body: the official OSWorld text minus the
    cua-lite-uncarried tools documented on the general body, with the system
    information rewritten to the cua-lite desktop's facts. A mismatch means
    the general body, an OSWorld overlay, or a splice anchor drifted."""
    assert (
        hashlib.sha256(HOLO4_OSWORLD_SYSTEM_PROMPTS["osworld"].encode()).hexdigest()
        == "bcaee35ed525720900d9ae4a2ddb9600ede050cd084d22843c158b1df7ec2e47"
    )


def test_desktop_prompt_serves_official_prompt_only_on_osworld_family_envs() -> None:
    bash = LiteShellToolSet.get_tool_schema("bash")
    response = LiteFinishToolSet.get_tool_schema("response")
    terminate = LiteFinishToolSet.get_tool_schema("terminate")
    update_plan = make_tool_schema(
        "update_plan",
        parameters={
            "type": "object",
            "properties": {"goals": {"type": "array"}},
            "required": ["goals"],
        },
    )

    for env_id in ("osworld", "lite.osworld"):
        prompt = Holo4DesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=("desktop", "use"),
                others={"env_id": env_id},
                extra_tool_schemas=[bash, response, terminate, update_plan],
            )
        )._system_prompt_for_active_surface()
        assert prompt == HOLO4_OSWORLD_SYSTEM_PROMPTS[env_id]

    # The shipped osworld surface has no bash and no update_plan; the official
    # body must keep its judged section through both trims.
    osworld_no_bash = Holo4DesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("desktop", "use"),
            others={"env_id": "osworld"},
            extra_tool_schemas=[response, terminate],
        )
    )._system_prompt_for_active_surface()
    assert "## System information" in osworld_no_bash
    assert "## Shell" not in osworld_no_bash
    assert "## How your work is judged" in osworld_no_bash

    for env_id in ("waa", "osworld_2", "webgym", "browsergym"):
        prompt = Holo4DesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=("desktop", "use"),
                others={"env_id": env_id},
                extra_tool_schemas=[response, terminate],
            )
        )._system_prompt_for_active_surface()
        assert prompt.startswith("You are Holo, a computer-use agent designed by H Company.")
        for token in (
            "Ubuntu 22.04",
            "Dock order",
            "remote-debugging-port",
            "## How your work is judged",
            "sudo -S",
            "## System information",
        ):
            assert token not in prompt


def test_desktop_prompt_tracks_the_active_tool_surface() -> None:
    bash = LiteShellToolSet.get_tool_schema("bash")
    response = LiteFinishToolSet.get_tool_schema("response")
    terminate = LiteFinishToolSet.get_tool_schema("terminate")
    update_plan = make_tool_schema(
        "update_plan",
        parameters={
            "type": "object",
            "properties": {"goals": {"type": "array"}},
            "required": ["goals"],
        },
    )

    with_shell = Holo4DesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[bash, response, terminate],
        )
    )._system_prompt_for_active_surface()
    assert with_shell.startswith("You are Holo, a computer-use agent designed by H Company.")
    assert "## Shell" in with_shell
    assert "## GUI vs Code" in with_shell
    assert "## Working on files" in with_shell
    assert "# Planning" not in with_shell
    assert "## System information" not in with_shell

    without_shell = Holo4DesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[response, terminate],
        )
    )._system_prompt_for_active_surface()
    assert "two surfaces" not in without_shell
    assert "## Shell" not in without_shell
    assert "## GUI vs Code" not in without_shell
    assert "## Working on files" not in without_shell
    assert "# Principles" in without_shell
    assert "## System information" not in without_shell

    with_plan = Holo4DesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("desktop", "use"),
            others={"env_id": "osworld"},
            extra_tool_schemas=[bash, response, terminate, update_plan],
        )
    )._system_prompt_for_active_surface()
    assert "# Planning" in with_plan
    assert "update_plan" in with_plan
    assert "## System information" in with_plan
    assert with_plan == HOLO4_OSWORLD_SYSTEM_PROMPTS["osworld"]


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
    assert "terminate" not in names
    assert names.count("shell") == 1
    assert names.count("answer") == 1
    assert names.count("custom_lookup") == 1
    assert adapter._tool_calls_to_agent_ordered(
        [LiteFinishToolSet.terminate(status="success")]
    ) == [{"name": "answer", "arguments": {"content": "Task completed."}}]


def test_canonical_bash_and_response_render_as_family_spellings() -> None:
    # ``_tool_calls_to_agent_ordered`` must route canonical bash/response
    # through the action-space projection: passing them through as-is would
    # show the model tool names it was never offered.
    adapter = Holo4DesktopUseAdapter(metadata=_metadata())
    assert adapter._tool_calls_to_agent_ordered(
        [
            make_tool_call("bash", {"command": "ls"}),
            LiteFinishToolSet.response(text="found it"),
        ]
    ) == [
        {"name": "shell", "arguments": {"command": "ls"}},
        {"name": "answer", "arguments": {"content": "found it"}},
    ]


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


def test_write_payloads_keep_wrapping_quotes_but_answer_loses_them() -> None:
    # The write tools' payloads are typed content: a wrapping quote pair is
    # part of what gets typed. ``answer``'s payload is the model's own
    # formatting, so its quotes are stripped — matching qwen3_5's
    # ``answer``/``response``.
    adapter = Holo4DesktopUseAdapter(metadata=_metadata())
    raw = """<tool_call>
<function=write_desktop>
<parameter=content>"Invoice # GES-1.pdf"</parameter>
<parameter=press_enter>true</parameter>
</function>
</tool_call>
<tool_call>
<function=answer>
<parameter=content>"42"</parameter>
</function>
</tool_call>"""
    lite_message = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
    assert tool_call_arguments(lite_message["tool_calls"][0])["actions"] == [
        {"action": "type", "text": '"Invoice # GES-1.pdf"\n'}
    ]
    assert tool_call_arguments(lite_message["tool_calls"][1]) == {"text": "42"}


def test_mobile_write_text_keeps_wrapping_quotes() -> None:
    adapter = Holo4MobileUseAdapter(metadata=LiteCUAMetadata(dims=("mobile", "use")))
    raw = """<tool_call>
<function=mobile_write>
<parameter=element>Name</parameter>
<parameter=text>"42"</parameter>
<parameter=x>500</parameter>
<parameter=y>300</parameter>
</function>
</tool_call>"""
    lite_message = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
    assert tool_call_arguments(lite_message["tool_calls"][0])["actions"] == [
        {"action": "tap", "coordinate": [500, 300], "clicks": 1},
        {"action": "type", "text": '"42"'},
    ]


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
        adapter=SimpleNamespace(enable_thinking=True, _assemble_tool_schemas=lambda: []),
        reasoning_effort=None,
    )
    assert Holo4BaseAgent.build_generation_prompt(agent, [{"role": "user"}]) == "prompt"
    processor.apply_chat_template.assert_called_once_with(
        [{"role": "user"}],
        tools=[],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
        preserve_thinking=True,
    )


def test_agent_forwards_explicit_reasoning_effort() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    agent = SimpleNamespace(
        processor=processor,
        adapter=SimpleNamespace(enable_thinking=True, _assemble_tool_schemas=lambda: []),
        reasoning_effort="medium",
    )
    assert Holo4BaseAgent.build_generation_prompt(agent, [{"role": "user"}]) == "prompt"
    processor.apply_chat_template.assert_called_once_with(
        [{"role": "user"}],
        tools=[],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
        preserve_thinking=True,
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
    # The Holo body leads the merged message: Holo4 templates only accept
    # the task text when it follows the family system body.
    assert system_text.startswith("You are Holo, a computer-use agent designed by H Company.")
    assert "Task-specific policy." in system_text


def test_holo_replay_falls_back_to_canonical_when_sidecar_key_mismatches(
    caplog,
) -> None:
    adapter = Holo4DesktopUseAdapter(metadata=_metadata())
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "I will inspect the page."}],
        "reasoning_content": "private reasoning",
        "raw_response": {
            "adapter_key": "other-family@desktop@use",
            "text": "<think>another adapter's output</think>",
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

    with caplog.at_level(logging.WARNING, logger="lite.agents.core.adapter"):
        rendered = adapter.convert_message_to_agent(message)

    assert rendered["content"] == [{"type": "text", "text": "I will inspect the page."}]
    assert rendered["tool_calls"] == [
        {"name": "custom_lookup", "arguments": {"query": "visible page"}}
    ]
    assert rendered["reasoning_content"] == "private reasoning"
    assert any("adapter_key mismatch" in record.getMessage() for record in caplog.records)


def test_holo_replay_short_circuits_to_verbatim_raw_response() -> None:
    adapter = Holo4MobileUseAdapter(metadata=LiteCUAMetadata(dims=("mobile", "use")))
    raw_response = """<think>
I need to open Retro Music.
</think>

Opening Retro Music.

<tool_call>
<function=mobile_click>
<parameter=element>
Retro Music app icon
</parameter>
<parameter=x>
[601, 925]
</parameter>
</function>
</tool_call>"""
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Opening Retro Music."}],
        "reasoning_content": "I need to open Retro Music.",
        "raw_response": {
            "adapter_key": adapter._registry_key,
            "text": raw_response,
        },
        "tool_calls": [LiteMobileActionSet.tap(coordinate=[601, 925])],
    }

    rendered = adapter.convert_message_to_agent(message)

    assert rendered == {
        "role": "assistant",
        "content": [{"type": "text", "text": raw_response}],
    }


def test_compacted_history_renders_verbatim_minus_windowed_images() -> None:
    # End-to-end contract for the aligned history policy: the protocol swaps
    # only out-of-window screenshots for a placeholder, so a compacted turn keeps its
    # raw_response sidecar and replays through the same verbatim path as
    # recent turns — the model re-sees its own past output (reasoning and
    # tool calls included) byte-exact, like the official harness.
    adapter = Holo4MobileUseAdapter(metadata=LiteCUAMetadata(dims=("mobile", "use")))
    raw_first = """<think>
Opening Retro Music.
</think>

<tool_call>
<function=mobile_click>
<parameter=element>Retro Music app icon</parameter>
<parameter=x>[601, 925]</parameter>
</function>
</tool_call>"""
    raw_third = """<think>
Tapping the first album.
</think>

<tool_call>
<function=mobile_click>
<parameter=element>first album row</parameter>
<parameter=x>[312, 700]</parameter>
</function>
</tool_call>"""
    sample = LiteSample(
        metadata=LiteCUAMetadata(dims=("mobile", "use")),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Sort the albums."},
                    {"type": "image", "index": 0},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Opening Retro Music."}],
                "reasoning_content": "I need to open Retro Music.",
                "raw_response": {"adapter_key": adapter._registry_key, "text": raw_first},
                "tool_calls": [LiteMobileActionSet.tap(coordinate=[601, 925])],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "text", "text": "played the first track"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "second screen"},
                    {"type": "image", "index": 1},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Listing albums."}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0001",
                "content": [{"type": "text", "text": "sorted A to Z"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "third screen"},
                    {"type": "image", "index": 2},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Opening the first album."}],
                "reasoning_content": "I need to open the first album.",
                "raw_response": {"adapter_key": adapter._registry_key, "text": raw_third},
                "tool_calls": [LiteMobileActionSet.tap(coordinate=[312, 700])],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "fourth screen"},
                    {"type": "image", "index": 3},
                ],
            },
        ],
    )

    rendered = adapter.render_step(sample, 4, [None, None, None, None])

    replayed = [
        message["content"][0]["text"]
        for message in rendered
        if message.get("role") == "assistant"
        and message.get("content")
        and message["content"][0].get("text", "").startswith("<think>")
    ]
    assert replayed == [raw_first, raw_third]
    joined_text = "\n".join(
        part["text"]
        for message in rendered
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "text"
    )
    assert "Sort the albums." in joined_text
    assert "played the first track" in joined_text
    assert "sorted A to Z" in joined_text
    image_indices = [
        part["index"]
        for message in rendered
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image"
    ]
    assert image_indices == [1, 2, 3]


def test_disabled_reasoning_effort_disables_template_thinking() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    agent = SimpleNamespace(
        processor=processor,
        adapter=SimpleNamespace(enable_thinking=True, _assemble_tool_schemas=lambda: []),
        reasoning_effort="disabled",
    )
    Holo4BaseAgent.build_generation_prompt(agent, [{"role": "user"}])
    processor.apply_chat_template.assert_called_once_with(
        [{"role": "user"}],
        tools=[],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
        preserve_thinking=True,
    )


def test_native_frames_pass_through_to_the_processor() -> None:
    # The official harness sends native-resolution captures and lets the
    # vision processor own any rounding (per-step prompt-token deltas in the
    # official trajectories match native 1080x2400 / 1920x1080 inputs, not
    # the downscaled 576x1280 / 1280x720 webp the published dataset stores).
    # So the adapter applies no resize of its own on either surface.
    desktop = Holo4DesktopUseAdapter()
    mobile = Holo4MobileUseAdapter()
    assert desktop.resolution is None
    assert mobile.resolution is None
    assert not desktop.smart_resize_enabled
    assert not mobile.smart_resize_enabled
    vm_frame = Image.new("RGB", (1920, 1080))
    phone_frame = Image.new("RGB", (1080, 2400))
    assert desktop.process_image(vm_frame) is vm_frame
    assert mobile.process_image(phone_frame) is phone_frame
