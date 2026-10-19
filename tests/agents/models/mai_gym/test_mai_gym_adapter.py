"""MAI-Gym adapter + agent: upstream window-wise harness alignment."""

from __future__ import annotations

import hashlib
from typing import Any

from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent import AgentRegistry
from lite.agents.core.protocol import ProtocolRegistry
from lite.agents.factory import LOCAL_AGENTS
from lite.agents.models.mai_gym.adapter import (
    MAIGymMobileUseAdapter,
    render_mai_gym_mobile_sys_prompt,
)
from lite.agents.models.mai_gym.protocol import HISTORY_IMAGE_PLACEHOLDER
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.final import pop_model_output_error
from lite.core.tools.action_space import LiteMobileActionSet
from lite.core.tools.calls import make_tool_call
from lite.core.tools.extra_tools import LiteFinishToolSet, LiteShellToolSet, make_open_app_tool

register_all()

#: Upstream ``ROLLOUT_APPS`` (mai_gym_mobile_use_windows_wise.py).
_UPSTREAM_ROLLOUT_APPS = [
    "Contacts",
    "Settings",
    "Clock",
    "Maps",
    "Chrome",
    "Calendar",
    "files",
    "Gallery",
    "Taodian",
    "Mattermost",
    "Mastodon",
    "Mail",
    "SMS",
    "Camera",
]

_RESPONSE = (
    "Need to open Files.\n</think>\n\n<tool_call>\n"
    '{"name": "mobile_use", "arguments": {"action_type": "open", "text": "Files"}}\n'
    "</tool_call>"
)


def _metadata(*extra_schemas: dict[str, Any]) -> LiteCUAMetadata:
    return LiteCUAMetadata(dims=("mobile", "use"), extra_tool_schemas=list(extra_schemas))


def _finish_schemas() -> list[dict[str, Any]]:
    return [LiteFinishToolSet.get_tool_schema(name) for name in ("response", "terminate")]


def _adapter(*extra_schemas: dict[str, Any]) -> MAIGymMobileUseAdapter:
    return MAIGymMobileUseAdapter(metadata=_metadata(*extra_schemas))


def test_system_prompt_is_byte_exact_to_the_upstream_window_wise_render() -> None:
    """Digest of upstream ``mai_gym_mobile_use_windows_wise.SYSTEM_PROMPT``
    (``render_system_prompt(ROLLOUT_APPS)``). A mismatch means the copy drifted
    from ``mai_gym_mobile_use_windows_wise_system.j2``; re-diff before updating."""
    rendered = render_mai_gym_mobile_sys_prompt(_UPSTREAM_ROLLOUT_APPS)
    assert (
        hashlib.sha256(rendered.encode()).hexdigest()
        == "ac15eaf95b86e1b70783ba0e3514b20c856bf2e3903e4297395e03d9307099cc"
    )


def test_prompt_renders_the_env_open_app_enum_and_drops_app_lines_without_one() -> None:
    with_apps = _adapter(make_open_app_tool(["Files", "设置"]), *_finish_schemas()).system_prompt
    assert '- Available Apps: `["Files", "设置"]`.\n' in with_apps
    # open_app / response / terminate are native rows, never MCP tools.
    assert "## MCP Tools" not in with_apps

    without_apps = _adapter(*_finish_schemas()).system_prompt
    assert "Available Apps" not in without_apps
    assert without_apps.endswith(
        '"status": "success or failed"}\n\n\n## Note\n- You must follow the Action Space strictly'
        ", and return the correct json object within <think> </think> and"
        " <tool_call></tool_call> XML tags."
    )


def test_non_native_extras_render_through_the_mcp_block() -> None:
    prompt = _adapter(LiteShellToolSet.get_tool_schema("bash"), *_finish_schemas()).system_prompt
    assert "## MCP Tools\nYou are also provided with MCP tools" in prompt
    assert '"name": "bash"' in prompt


def test_adapter_wire_knobs() -> None:
    adapter = _adapter()
    assert adapter.enable_thinking is True
    assert adapter.protocol.window_size == 10
    assert adapter.protocol.overlap == 2
    # Native resolution: no client-side resize.
    img = Image.new("RGB", (1080, 2400))
    assert adapter.process_image(img).size == (1080, 2400)


def test_parse_splits_reasoning_and_the_first_tool_call() -> None:
    adapter = _adapter(make_open_app_tool(["Files"]), *_finish_schemas())
    response = _RESPONSE + (
        '\n<tool_call>\n{"name": "mobile_use", "arguments": {"action_type": "wait"}}\n</tool_call>'
    )
    agent_message = adapter.parse_raw_assistant_response(response)
    assert agent_message["reasoning_content"] == "Need to open Files."
    assert agent_message["tool_calls"] == [
        {"name": "mobile_use", "arguments": {"action_type": "open", "text": "Files"}}
    ]
    lite_message = adapter.convert_message_from_agent(agent_message)
    assert lite_message["tool_calls"] == [make_tool_call("open_app", {"app_name": "Files"})]


def test_missing_or_malformed_tool_call_is_model_output_error() -> None:
    adapter = _adapter()
    no_call = adapter.parse_raw_assistant_response("thinking only, truncated")
    assert pop_model_output_error(no_call) == "No <tool_call> block found"
    bad = adapter.parse_raw_assistant_response("r</think><tool_call>{not json}</tool_call>")
    assert pop_model_output_error(bad) == "malformed <tool_call> JSON"


def test_render_step_serves_the_upstream_layout_with_inline_json_history() -> None:
    adapter = _adapter(make_open_app_tool(["Files"]), *_finish_schemas())
    sample = LiteSample(
        metadata=adapter.metadata,
        messages=[
            {
                "role": "user",
                "content": [{"type": "image", "index": 0}, {"type": "text", "text": "Open Files"}],
            },
            {
                "role": "assistant",
                "content": [],
                "reasoning_content": "Need to open Files.",
                "tool_calls": [
                    make_tool_call("open_app", {"app_name": "Files"}, call_id="call_0000")
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "image", "index": 1}],
            },
        ],
    )
    rendered = adapter.render_step(sample, 2, [None, None])
    assert [m["role"] for m in rendered] == ["system", "user", "user", "assistant", "user"]
    assert rendered[1]["content"] == [{"type": "text", "text": "Open Files"}]
    assert rendered[2]["content"] == [{"type": "image", "index": 0}]
    assert rendered[3] == {
        "role": "assistant",
        "reasoning_content": "Need to open Files.",
        "content": [
            {
                "type": "text",
                "text": '<tool_call>\n{"name": "mobile_use", "arguments": {"action_type": "open", '
                '"text": "Files"}}\n</tool_call>',
            }
        ],
    }
    assert rendered[4]["content"] == [{"type": "image", "index": 1}]


def test_long_trajectories_collapse_old_screenshots() -> None:
    adapter = _adapter()
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}, {"type": "text", "text": "t"}]}
    ]
    for i in range(10):
        messages.append(
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [{**LiteMobileActionSet.wait(duration=1.0), "id": f"call_{i:04d}"}],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i:04d}",
                "content": [{"type": "image", "index": i + 1}],
            }
        )
    rendered = adapter.render_step(
        LiteSample(metadata=adapter.metadata, messages=messages), 11, [None] * 11
    )
    placeholders = [
        m for m in rendered if m["content"] == [{"type": "text", "text": HISTORY_IMAGE_PLACEHOLDER}]
    ]
    assert len(placeholders) == 8


class _RecordingProcessor:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.kwargs = kwargs
        return "PROMPT"


async def _generate(**_: Any) -> dict[str, Any]:
    return {"response": _RESPONSE}


async def test_agent_keeps_history_reasoning_and_trims_the_replay_sidecar() -> None:
    processor = _RecordingProcessor()
    metadata = _metadata(make_open_app_tool(["Files"]), *_finish_schemas())
    agent = AgentRegistry.get(
        "mai_gym@mobile@use", processor=processor, generate_fn=_generate, metadata=metadata
    )
    sample = LiteSample(
        metadata=metadata,
        messages=[
            {
                "role": "user",
                "content": [{"type": "image", "index": 0}, {"type": "text", "text": "t"}],
            }
        ],
    )
    result = await agent._predict_with_details(sample, processed_images=[Image.new("RGB", (8, 8))])

    assert processor.kwargs == {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    # The sidecar keeps only the post-</think> half; the template rebuilds the
    # <think> block from reasoning_content when the turn replays as history.
    assert result.lite_message["raw_response"] == {
        "text": '<tool_call>\n{"name": "mobile_use", "arguments": {"action_type": "open", '
        '"text": "Files"}}\n</tool_call>',
        "adapter_key": "mai_gym@mobile@use",
        "reasoning_content": "Need to open Files.",
    }


async def test_rumination_is_regenerated_until_a_tool_call_parses() -> None:
    """Upstream mobile retry: an unclosed ``<think>`` regenerates the identical prompt."""
    replies = iter(["wait, let me reconsider the tap target...", _RESPONSE])
    calls: list[tuple[str, list[dict[str, Any]]]] = []

    async def generate(*, prompt: str, messages: list[dict], **_: Any) -> dict[str, Any]:
        calls.append((prompt, messages))
        return {"response": next(replies), "finish_reason": "stop"}

    metadata = _metadata(make_open_app_tool(["Files"]), *_finish_schemas())
    agent = AgentRegistry.get(
        "mai_gym@mobile@use",
        processor=_RecordingProcessor(),
        generate_fn=generate,
        metadata=metadata,
    )
    sample = LiteSample(
        metadata=metadata,
        messages=[
            {
                "role": "user",
                "content": [{"type": "image", "index": 0}, {"type": "text", "text": "t"}],
            }
        ],
    )
    result = await agent._predict_with_details(sample, processed_images=[Image.new("RGB", (8, 8))])

    assert len(calls) == 2
    assert calls[1] == calls[0]  # no correction turn on mobile
    assert result.model_output_error is None
    assert result.lite_message["tool_calls"] == [
        make_tool_call("open_app", {"app_name": "Files"}, call_id="call_0000")
    ]


def test_catalog_row_and_registries() -> None:
    assert LOCAL_AGENTS["Tongyi-MAI/MAI-Gym-9B"] == {"agent_id": "mai_gym"}
    assert "mai_gym@mobile@use" in AgentRegistry.list()
    assert "mai_gym@mobile@use" in AgentAdapterRegistry.list()
    assert "mai_gym@mobile" in ActionSpaceRegistry.list()
    assert "mai_gym.history" in ProtocolRegistry.list()
