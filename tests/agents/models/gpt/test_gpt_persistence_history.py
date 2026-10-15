"""Split GPT agent characterization tests.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_*.py -v
"""

from __future__ import annotations

import base64
import copy
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agents.models.gpt._support import (
    _colored_png_bytes,
    _ContentOnlyContinuationEnv,
    _DropsOneResultEnv,
    _fake_response,
    _FakeEnv,
    _image_rgb,
    _MixedComputerAndBashResultsEnv,
    _MixedComputerAndVisualExtraResultsEnv,
    _MobileContentOnlyContinuationEnv,
    _MultiImageToolResultsEnv,
    _RecordingFakeEnv,
    _RejectEmptyActionsEnv,
    _ReversedResultOrderEnv,
    _TerminalNoResultsEnv,
    _ToolResultsEnv,
)

from lite.agents.models.gpt.action_space import GPTDesktopGroundingPointActionSpace
from lite.agents.models.gpt.agent import GPTDesktopUseAgent, GPTMobileUseAgent
from lite.core import LiteCUAMetadata
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.results import LiteToolResult


class TestImageLimitRecovery:
    @pytest.mark.parametrize("chained", [True, False])
    @pytest.mark.parametrize("system_in_input", [True, False])
    async def test_image_limit_restarts_context_without_restarting_task(
        self, monkeypatch, chained, system_in_input
    ):
        import litellm

        requests = []
        screenshot = _fake_response(
            [{"type": "computer_call", "call_id": "c1", "actions": [{"type": "screenshot"}]}]
        )

        async def respond(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            if len(requests) == 1:
                env._shot = _colored_png_bytes(color=(20, 180, 20))
            if len(requests) in {2, 4}:
                raise litellm.BadRequestError(
                    message="Exceeded maximum number of images (50) allowed in the request.",
                    model="gpt-6-astra",
                    llm_provider="openai",
                )
            return screenshot if len(requests) < 5 else _fake_response()

        monkeypatch.setattr("litellm.aresponses", respond)
        monkeypatch.setattr(
            "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
            AsyncMock(return_value=[(800, 600)]),
        )
        sleep = AsyncMock()
        monkeypatch.setattr("lite.agents.core.agent.utils.retry.asyncio.sleep", sleep)
        env = _RecordingFakeEnv(terminate_after=3)
        reset = AsyncMock(wraps=env.reset)
        monkeypatch.setattr(env, "reset", reset)
        agent = GPTDesktopUseAgent(
            model_id="gpt-6-astra",
            system_prompt="Keep task rules.",
            metadata=LiteCUAMetadata(
                extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("terminate")]
            ),
            api_kwargs={
                "chain_previous_response": chained,
                "system_prompt_in_input": system_in_input,
            },
        )
        result = await agent.sample(env, max_steps=3)

        assert len(requests) == 5
        assert reset.await_count == 1
        assert len(env.actions_seen) == len(result.steps) == 3
        assert result.terminated
        sleep.assert_not_awaited()
        assert [step.image_indices for step in result.steps] == [(0,), (1,), (2,)]
        assert len(result.lite_sample.images) == 3
        assert [m["role"] for m in result.lite_sample.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
        ]
        for failed, recovered in [(requests[1], requests[2]), (requests[3], requests[4])]:
            assert recovered.get("previous_response_id") is None
            assert recovered["tools"] == failed["tools"] == requests[0]["tools"]
            assert recovered["tools"][0] == {"type": "computer"}
            assert recovered["tools"][1]["name"] == "terminate"
            assert recovered["reasoning"] == failed["reasoning"]
            assert recovered["extra_body"] == failed["extra_body"]
            assert [item.get("role") for item in recovered["input"]] == (
                ["developer", "user"] if system_in_input else ["user"]
            )
            if system_in_input:
                assert recovered["input"][0]["content"] == "Keep task rules."
            else:
                assert recovered["instructions"] == "Keep task rules."
            content = recovered["input"][-1]["content"]
            assert content[0] == {"type": "input_text", "text": "instr"}
            assert content[1]["type"] == "input_image"
            current_screenshot = next(
                item["output"]["image_url"]
                for item in reversed(failed["input"])
                if item.get("type") == "computer_call_output"
            )
            assert content[1]["image_url"] == current_screenshot
            assert content[1]["image_url"] != requests[0]["input"][-1]["content"][1]["image_url"]

    @pytest.mark.parametrize("image_limit", [True, False])
    async def test_persistent_error_is_not_hidden_by_context_reset(self, monkeypatch, image_limit):
        import litellm

        error = litellm.BadRequestError(
            message=(
                "Exceeded maximum number of images (50) allowed in the request."
                if image_limit
                else "Unsupported tool type: computer"
            ),
            model="gpt-6-astra",
            llm_provider="openai",
        )
        screenshot = _fake_response(
            [{"type": "computer_call", "call_id": "c1", "actions": [{"type": "screenshot"}]}]
        )
        mock = AsyncMock(side_effect=[screenshot, error, error])
        monkeypatch.setattr("litellm.aresponses", mock)
        monkeypatch.setattr(
            "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
            AsyncMock(return_value=[(800, 600)]),
        )
        env = _RecordingFakeEnv(terminate_after=3)
        agent = GPTDesktopUseAgent(model_id="gpt-6-astra", api_retry_max=0)
        with pytest.raises(litellm.BadRequestError) as caught:
            await agent.sample(env, max_steps=3)
        assert caught.value is error
        assert mock.await_count == (3 if image_limit else 2)
        assert len(env.actions_seen) == 1
        assert env.closed


class TestCanonicalPersistence:
    async def test_max_steps_exhaustion_marks_truncated_with_paired_feedback(self, monkeypatch):
        resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        result = await agent.sample(_RecordingFakeEnv(terminate_after=99), max_steps=1)

        assert result.terminated is False
        assert result.truncated is True
        assert result.steps[-1].status == "truncated"
        assert mock.call_count == 1
        assert [m["role"] for m in result.lite_sample.messages] == ["user", "assistant", "tool"]
        tool_msg = result.lite_sample.messages[-1]
        assert tool_msg["tool_call_id"] == "call_0000"
        assert tool_msg["content"] == [
            {"type": "image", "index": 1},
            {"type": "text", "text": "instr"},
        ]

    async def test_terminal_step_with_no_results_ends_on_assistant_tool_call(self, monkeypatch):
        resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "provider_computer_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _TerminalNoResultsEnv()
        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        result = await agent.sample(env, max_steps=2)

        assert result.terminated is True
        assert result.truncated is False
        assert mock.call_count == 1
        assert len(env.actions_seen) == 1
        assert env.actions_seen[0] == [
            make_tool_call(
                "computer",
                {"actions": [{"action": "screenshot"}]},
                call_id="call_0000",
            )
        ]

        assert [m["role"] for m in result.lite_sample.messages] == ["user", "assistant"]
        assistant = result.lite_sample.messages[-1]
        assert assistant["tool_calls"] == [
            make_tool_call(
                "computer",
                {"actions": [{"action": "screenshot"}]},
                call_id="call_0000",
            )
        ]

    async def test_provider_call_persists_canonical_id_no_raw_response(self, monkeypatch):
        resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        result = await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert "raw_response" not in assistant
        assert assistant["tool_calls"] == [
            make_tool_call(
                "computer",
                {"actions": [{"action": "screenshot"}]},
                call_id="call_0000",
            ),
        ]
        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_0000"
        assert tool_msg["content"] == [
            {"type": "image", "index": 1},
            {"type": "text", "text": "instr"},
        ]

    async def test_multi_image_tool_result_preserves_all_but_sends_last_to_provider(
        self,
        monkeypatch,
    ):
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "provider_computer_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[first_resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _MultiImageToolResultsEnv(terminate_after=99)

        class _Hook:
            def __init__(self):
                self.current_image_indices = []

            def on_step(self, data):
                self.current_image_indices.append(data.current_image_index)

            def on_complete(self, result):
                del result

        hook = _Hook()
        result = await GPTDesktopUseAgent(model_id="gpt-5.5").sample(
            env,
            max_steps=3,
            hooks=[hook],
        )

        assert [_image_rgb(image) for image in result.lite_sample.images] == [
            (255, 255, 255),
            (180, 20, 20),
            (20, 180, 20),
            (20, 20, 180),
        ]
        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 3)]
        assert hook.current_image_indices == [0, 3]
        assert len(result.processed_images) == len(result.lite_sample.images)
        assert result.processed_images[0] is result.lite_sample.images[0]
        assert result.processed_images[1:3] == [None, None]
        assert result.processed_images[3] is result.lite_sample.images[3]

        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_0000"
        assert tool_msg["content"] == [
            {"type": "image", "index": 3},
            {"type": "text", "text": "visual obs"},
        ]

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "computer_call_output"]
        assert len(outputs) == 1
        payload = outputs[0]["output"]["image_url"].split("base64,", 1)[1]
        assert base64.b64decode(payload) == env._result_shots[-1]
        sent_payloads = [
            json.dumps(call.kwargs["input"], ensure_ascii=False, default=str)
            for call in mock.call_args_list
        ]
        assert all("_cua_lite_image_index" not in payload for payload in sent_payloads)
        assert all("_cua_lite_image_index" not in step.prompt for step in result.steps)

    @pytest.mark.parametrize(
        "terminate_after,terminal",
        [
            pytest.param(99, False, id="mid-episode-history-feedback"),
            pytest.param(1, True, id="terminal-feedback"),
        ],
    )
    async def test_out_of_order_env_results_persist_in_tool_call_order(
        self,
        terminate_after,
        terminal,
        monkeypatch,
    ):
        """Multi-result ``role:"tool"`` ordering plus text/error/metadata projection.

        The env answers the two canonical calls in the opposite order. The
        env-result boundary re-pairs them once, so the persisted messages follow
        the assistant's own call order on both the mid-episode feedback path and
        the terminal-feedback path.
        """
        extra = make_tool_schema("bash", description="Run a command.")
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "provider_computer_1",
                    "actions": [{"type": "screenshot"}],
                },
                {
                    "type": "function_call",
                    "id": "provider_bash_1",
                    "name": "bash",
                    "arguments": "{}",
                },
            ]
        )
        responses = [first_resp] if terminal else [first_resp, _fake_response()]
        mock = AsyncMock(side_effect=responses)
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        result = await agent.sample(
            _ReversedResultOrderEnv(terminate_after=terminate_after),
            max_steps=3,
        )

        # A terminal first step persists feedback through the terminal appender
        # and never asks the provider again; the mid-episode step goes through
        # the next-turn history appender instead.
        assert mock.call_count == (1 if terminal else 2)
        assert result.terminated is True
        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert [tool_call_name(call) for call in assistant["tool_calls"]] == ["computer", "bash"]

        tool_messages = [m for m in result.lite_sample.messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_messages] == [
            tool_call_id(call) for call in assistant["tool_calls"]
        ]
        # Images stay indexed against the aligned order, and metadata/error are
        # projected into the same canonical message.
        assert tool_messages[0]["content"] == [
            {"type": "image", "index": 1},
            {"type": "text", "text": "computer screen text"},
            {"type": "metadata", "data": {"source": "screen"}},
        ]
        assert tool_messages[1]["content"] == [
            {
                "type": "text",
                "text": "bash stdout\n\n## Error from previous action:\nexit status 1",
            },
        ]

    async def test_non_terminal_step_missing_a_result_fails_loudly(self, monkeypatch):
        """A non-terminal step owes one result per canonical call.

        The env-result boundary is the single place this is checked, so a
        dropped result must raise there instead of silently persisting a turn
        with fewer ``role:"tool"`` messages than tool calls.
        """
        extra = make_tool_schema("bash", description="Run a command.")
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "provider_computer_1",
                    "actions": [{"type": "screenshot"}],
                },
                {
                    "type": "function_call",
                    "id": "provider_bash_1",
                    "name": "bash",
                    "arguments": "{}",
                },
            ]
        )
        monkeypatch.setattr("litellm.aresponses", AsyncMock(return_value=first_resp))

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        env = _DropsOneResultEnv(terminate_after=99)
        with pytest.raises(RuntimeError, match="do not match tool_calls"):
            await agent.sample(env, max_steps=3)
        assert env.closed is True

    async def test_chained_computer_feedback_acks_valid_and_malformed_provider_calls(
        self,
        monkeypatch,
    ):
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "provider_bad",
                    "actions": [{"type": "click"}],
                },
                {
                    "type": "computer_call",
                    "call_id": "provider_good",
                    "actions": [{"type": "screenshot"}],
                },
            ]
        )
        mock = AsyncMock(side_effect=[first_resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _MultiImageToolResultsEnv(terminate_after=99)
        await GPTDesktopUseAgent(model_id="gpt-5.5").sample(env, max_steps=3)

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "computer_call_output"]
        assert [item["call_id"] for item in outputs] == ["provider_bad", "provider_good"]
        payloads = [
            base64.b64decode(item["output"]["image_url"].split("base64,", 1)[1]) for item in outputs
        ]
        assert payloads == [env._shot, env._result_shots[-1]]

    async def test_merged_computer_calls_ack_all_but_only_final_gets_result_image(
        self,
        monkeypatch,
    ):
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "provider_click",
                    "actions": [{"type": "click", "x": 100, "y": 200}],
                },
                {
                    "type": "computer_call",
                    "call_id": "provider_type",
                    "actions": [{"type": "type", "text": "hello"}],
                },
            ]
        )
        mock = AsyncMock(side_effect=[first_resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _MultiImageToolResultsEnv(terminate_after=99)
        result = await GPTDesktopUseAgent(model_id="gpt-5.5").sample(env, max_steps=3)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert [tool_call_name(call) for call in assistant["tool_calls"]] == ["computer"]
        assert tool_call_arguments(assistant["tool_calls"][0])["actions"] == [
            {"action": "click", "coordinate": [125, 333]},
            {"action": "type", "text": "hello"},
        ]

        assert [_image_rgb(image) for image in result.lite_sample.images] == [
            (255, 255, 255),
            (180, 20, 20),
            (20, 180, 20),
            (20, 20, 180),
        ]
        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 3)]

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "computer_call_output"]
        assert [item["call_id"] for item in outputs] == ["provider_click", "provider_type"]
        payloads = [
            base64.b64decode(item["output"]["image_url"].split("base64,", 1)[1]) for item in outputs
        ]
        assert payloads == [env._shot, env._result_shots[-1]]
        final_output_index = next(
            idx
            for idx, item in enumerate(second_input)
            if item.get("type") == "computer_call_output" and item["call_id"] == "provider_type"
        )
        text_indices = [
            idx
            for idx, item in enumerate(second_input)
            if item.get("role") == "user"
            and any(
                block.get("type") == "input_text" and block.get("text") == "visual obs"
                for block in item.get("content", [])
                if isinstance(block, dict)
            )
        ]
        assert text_indices
        assert all(idx > final_output_index for idx in text_indices)

    async def test_second_turn_computer_feedback_uses_provider_id_but_canonical_result_lookup(
        self, monkeypatch
    ):
        class _CanonicalTextEnv(_FakeEnv):
            async def step(self, actions):
                result = await super().step(actions)
                result.results = [
                    LiteToolResult(
                        tool_call_id=tool_call_id(action),
                        images=[self._shot],
                        text=f"result {tool_call_id(action)}",
                    )
                    for action in actions
                ]
                return result

        provider_id = "provider_computer_1"
        resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": provider_id,
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        result = await agent.sample(_CanonicalTextEnv(terminate_after=2), max_steps=3)

        canonical_id = "call_0000"
        assert provider_id != canonical_id
        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert assistant["tool_calls"] == [
            make_tool_call(
                "computer",
                {"actions": [{"action": "screenshot"}]},
                call_id=canonical_id,
            )
        ]

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "computer_call_output"]
        assert [item["call_id"] for item in outputs] == [provider_id]
        second_turn_texts = [
            block.get("text", "")
            for item in second_input
            if item.get("role") == "user"
            for block in item.get("content", [])
            if isinstance(block, dict) and block.get("type") == "input_text"
        ]
        assert f"result {canonical_id}" in second_turn_texts

        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == canonical_id
        assert {"type": "text", "text": f"result {canonical_id}"} in tool_msg["content"]

    def test_stamp_rejects_legacy_tool_call_id(self):
        from lite.core.errors import ToolCallValidationError
        from lite.core.tools.calls import stamp_tool_call_list_ids

        calls = [
            {
                "tool_call_id": "legacy_1",
                "type": "function",
                "function": {"name": "computer", "arguments": {}},
            }
        ]

        with pytest.raises(ToolCallValidationError, match="tool_call_id"):
            stamp_tool_call_list_ids(calls, preserve=False)

    def test_desktop_parser_restamps_provider_call_ids_to_canonical_ids(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "computer_call",
                    "call_id": "computer_1",
                    "actions": [{"type": "screenshot"}],
                },
                {
                    "type": "function_call",
                    "id": "extra_1",
                    "name": "report_infeasible",
                    "arguments": '{"reason": "blocked"}',
                },
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset({"report_infeasible"}),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_id(call) for call in calls] == ["call_0000", "call_0001"]
        assert [tool_call_name(call) for call in calls] == ["computer", "report_infeasible"]
        assert all("tool_call_id" not in call for call in calls)

    def test_desktop_parser_batches_adjacent_computer_call_items(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "computer_call",
                    "call_id": "computer_1",
                    "actions": [{"type": "click", "x": 100, "y": 200}],
                },
                {
                    "type": "computer_call",
                    "call_id": "computer_2",
                    "actions": [{"type": "type", "text": "hello"}],
                },
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["computer"]
        assert [tool_call_id(call) for call in calls] == ["call_0000"]
        assert tool_call_arguments(calls[0])["actions"] == [
            {"action": "click", "coordinate": [98, 260]},
            {"action": "type", "text": "hello"},
        ]

    def test_desktop_parser_rejects_replayed_computer_call_when_request_hid_native(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "computer_call",
                    "call_id": "computer_1",
                    "actions": [{"type": "screenshot"}],
                }
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
            active_provider_tool_names=frozenset(),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared computer_call"

    @pytest.mark.parametrize(
        "action",
        [
            {"type": "click"},
            {"type": "click", "x": [], "y": 2},
            {"type": "click", "x": False, "y": 2},
            {"type": "move"},
            {"type": "drag", "start_x": 1, "start_y": 2},
            {"type": "drag", "start_x": "x", "start_y": 2, "end_x": 3, "end_y": 4},
        ],
    )
    def test_desktop_parser_marks_malformed_native_coordinates_as_model_output_error(self, action):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "computer_call",
                    "call_id": "computer_1",
                    "actions": [action],
                }
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg)

    def test_desktop_lite_scroll_click_amount_converts_to_native_pixels(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace

        [action] = GPTDesktopActionSpace().convert_tool_calls_to_agent(
            [
                make_tool_call(
                    "scroll",
                    {"direction": "down", "amount": 3, "coordinate": [250, 750]},
                )
            ],
            resolution=(1000, 1000),
        )

        assert action["x"] == 250
        assert action["y"] == 750
        assert action["scroll_y"] == 300
        assert action["scroll_x"] == 0

    def test_desktop_lite_scroll_without_coordinate_fails_loudly(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace

        with pytest.raises(ValueError, match="scroll requires coordinate"):
            GPTDesktopActionSpace().convert_tool_calls_to_agent(
                [make_tool_call("scroll", {"direction": "down", "amount": 3})],
                resolution=(1000, 1000),
            )

    def test_desktop_dropped_middle_provider_call_does_not_get_screenshot(self):
        """R22 red: GPT provider drop mask must compose with action-batch provenance.

        Provider emits ``computer_call, undeclared_tool, computer_call``. The
        parser drops the middle provider call, then the two surviving GUI calls
        become adjacent and merge into one canonical ``computer`` action-batch call. The
        third provider call did execute inside ``call_0000``, so only it should
        carry the screenshot.
        """
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        parsed = parse_output_items_with_provenance(
            [
                {
                    "type": "computer_call",
                    "call_id": "computer_1",
                    "actions": [{"type": "click", "x": 100, "y": 200}],
                },
                {
                    "type": "function_call",
                    "id": "unknown_1",
                    "name": "mystery",
                    "arguments": "{}",
                },
                {
                    "type": "computer_call",
                    "call_id": "computer_2",
                    "actions": [{"type": "type", "text": "hello"}],
                },
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
            active_provider_tool_names=frozenset({"computer"}),
        )

        assert [tool_call_name(call) for call in parsed.message["tool_calls"]] == ["computer"]
        assert [call.canonical_call_id for call in parsed.provider_calls] == [
            "call_0000",
            None,
            "call_0000",
        ]
        assert [call.is_final_for_canonical for call in parsed.provider_calls] == [
            False,
            False,
            True,
        ]

    def test_desktop_merge_maps_ids_and_provider_errors_directly(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        output_items = [
            {
                "type": "computer_call",
                "call_id": "computer_1",
                "actions": [{"type": "click", "x": 100, "y": 200}],
            },
            {
                "type": "function_call",
                "id": "unknown_1",
                "name": "mystery",
                "arguments": "{}",
            },
            {
                "type": "computer_call",
                "call_id": "computer_2",
                "actions": [{"type": "type", "text": "hello"}],
            },
            {
                "type": "function_call",
                "id": "bash_1",
                "name": "bash",
                "arguments": '{"command": "pwd"}',
            },
            {
                "type": "function_call",
                "id": "bad_args",
                "name": "bash",
                "arguments": "{",
            },
        ]

        parsed = parse_output_items_with_provenance(
            output_items,
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset({"bash"}),
            declared_agent_tool_names=frozenset(),
            active_provider_tool_names=frozenset({"computer"}),
        )

        calls_by_provider_id = {call.provider_call_id: call for call in parsed.provider_calls}
        assert [tool_call_name(call) for call in parsed.message["tool_calls"]] == [
            "computer",
            "bash",
        ]
        assert {
            call_id: call.canonical_call_id for call_id, call in calls_by_provider_id.items()
        } == {
            "computer_1": "call_0000",
            "unknown_1": None,
            "computer_2": "call_0000",
            "bash_1": "call_0001",
            "bad_args": None,
        }
        unknown_error = calls_by_provider_id["unknown_1"].error
        assert "undeclared function_call: mystery" in unknown_error
        assert "This tool call was not executed" in unknown_error
        assert "No action was executed" not in unknown_error
        assert (
            "malformed function_call arguments for bash" in calls_by_provider_id["bad_args"].error
        )
        assert calls_by_provider_id["computer_1"].is_final_for_canonical is False
        assert calls_by_provider_id["computer_2"].is_final_for_canonical is True

    def test_desktop_merge_reports_undeclared_computer_call_directly(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        item = {
            "type": "computer_call",
            "call_id": "computer_hidden",
            "actions": [{"type": "screenshot"}],
        }

        parsed = parse_output_items_with_provenance(
            [item],
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset(),
            declared_agent_tool_names=frozenset(),
            active_provider_tool_names=frozenset(),
        )

        [provider_call] = parsed.provider_calls
        assert provider_call.canonical_call_id is None
        assert "undeclared computer_call" in provider_call.error

    def test_desktop_merge_reports_malformed_native_directly(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        item = {
            "type": "computer_call",
            "call_id": "computer_bad",
            "actions": [{"type": "click"}],
        }

        parsed = parse_output_items_with_provenance(
            [item],
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset(),
            declared_agent_tool_names=frozenset(),
            active_provider_tool_names=frozenset({"computer"}),
        )

        [provider_call] = parsed.provider_calls
        assert provider_call.canonical_call_id is None
        assert provider_call.error.startswith("model output error:")

    def test_desktop_call_without_provider_id_is_a_parse_error(self):
        """An empty ``call_id`` cannot be echoed back, so it is a parse error."""
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        parsed = parse_output_items_with_provenance(
            [
                {"type": "computer_call", "actions": [{"type": "screenshot"}]},
                {"type": "function_call", "name": "bash", "arguments": "{}"},
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset({"bash"}),
            declared_agent_tool_names=frozenset(),
            active_provider_tool_names=frozenset({"computer"}),
        )

        computer_call, function_call = parsed.provider_calls
        assert computer_call.provider_call_id == ""
        assert "missing provider id for computer" in computer_call.error
        assert function_call.provider_call_id == ""
        assert "missing provider id for bash" in function_call.error
        assert parsed.message["tool_calls"] == []
        assert "missing provider id for computer" in pop_model_output_error(parsed.message)

    def test_desktop_parser_preserves_extra_between_computer_calls(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "computer_call",
                    "call_id": "computer_1",
                    "actions": [{"type": "click", "x": 100, "y": 200}],
                },
                {
                    "type": "function_call",
                    "id": "bash_1",
                    "name": "bash",
                    "arguments": '{"command": "pwd"}',
                },
                {
                    "type": "computer_call",
                    "call_id": "computer_2",
                    "actions": [{"type": "type", "text": "hello"}],
                },
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset({"bash"}),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["computer", "bash", "computer"]
        assert tool_call_arguments(calls[1]) == {"command": "pwd"}
        assert [tool_call_id(call) for call in calls] == [
            "call_0000",
            "call_0001",
            "call_0002",
        ]

    def test_desktop_parser_marks_undeclared_function_call_as_model_output_error(self, caplog):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "id": "native_1",
                    "name": "click",
                    "arguments": '{"x": 512, "y": 384}',
                }
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared function_call click"
        assert "Ignoring undeclared GPT function_call: click" in caplog.text

    def test_desktop_parser_does_not_promote_unknown_native_computer_action_to_extra(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "computer_call",
                    "call_id": "computer_1",
                    "actions": [{"type": "goto", "url": "https://example.com"}],
                }
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset({"goto"}),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "unknown GPT native computer action: goto"

    def test_desktop_parser_rejects_entire_computer_call_batch_on_unknown_action(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "computer_call",
                    "call_id": "computer_1",
                    "actions": [
                        {"type": "click", "x": 100, "y": 200},
                        {"type": "goto", "url": "https://example.com"},
                    ],
                }
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "unknown GPT native computer action: goto"

    def test_desktop_action_space_unknown_action_raises_by_default(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace

        with pytest.raises(ValueError, match="unknown GPT native computer action: goto"):
            GPTDesktopActionSpace().convert_tool_calls_from_agent(
                [{"type": "goto", "url": "https://example.com"}],
                resolution=(1024, 768),
            )

    def test_grounding_parser_rejects_bool_click_coordinates(self):
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "id": "native_1",
                    "name": "click",
                    "arguments": '{"x": false, "y": 10}',
                }
            ],
            GPTDesktopGroundingPointActionSpace(),
            (1024, 768),
            declared_agent_tool_names=frozenset({"click"}),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "click requires numeric x/y coordinates"

    def test_grounding_action_space_unknown_action_raises_by_default(self):
        with pytest.raises(ValueError, match="unknown GPT desktop grounding action: move"):
            GPTDesktopGroundingPointActionSpace().convert_tool_calls_from_agent(
                [{"type": "move", "x": 10, "y": 20}],
                resolution=(1024, 768),
            )

    async def test_function_call_output_uses_per_call_text_and_provider_id(self, monkeypatch):
        extra = make_tool_schema("bash", description="Run a command.")
        r1 = _fake_response(
            [
                {
                    "type": "function_call",
                    "id": "provider_bash_1",
                    "name": "bash",
                    "arguments": "{}",
                }
            ]
        )
        r2 = _fake_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        result = await agent.sample(_ToolResultsEnv(terminate_after=2), max_steps=3)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert assistant["tool_calls"] == [
            make_tool_call("bash", {}, call_id="call_0000"),
        ]

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "function_call_output"]
        assert outputs == [
            {
                "type": "function_call_output",
                "call_id": "provider_bash_1",
                "output": "per-call stdout",
            }
        ]

        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_0000"
        assert {"type": "text", "text": "per-call stdout"} in tool_msg["content"]

    async def test_mixed_computer_and_function_outputs_keep_per_call_text(
        self,
        monkeypatch,
    ):
        extra = make_tool_schema("bash", description="Run a command.")
        r1 = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "provider_computer_1",
                    "actions": [{"type": "screenshot"}],
                },
                {
                    "type": "function_call",
                    "id": "provider_bash_1",
                    "name": "bash",
                    "arguments": "{}",
                },
            ]
        )
        r2 = _fake_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        result = await agent.sample(
            _MixedComputerAndBashResultsEnv(terminate_after=2),
            max_steps=3,
        )

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert [tool_call_name(call) for call in assistant["tool_calls"]] == ["computer", "bash"]

        second_input = mock.call_args_list[1].kwargs["input"]
        function_outputs = [
            item for item in second_input if item.get("type") == "function_call_output"
        ]
        assert function_outputs == [
            {
                "type": "function_call_output",
                "call_id": "provider_bash_1",
                "output": "bash stdout",
            }
        ]

        user_texts = [
            block.get("text", "")
            for item in second_input
            if item.get("role") == "user"
            for block in item.get("content", [])
            if isinstance(block, dict) and block.get("type") == "input_text"
        ]
        assert "computer screen text" in user_texts
        assert "bash stdout" not in user_texts

        tool_messages = {
            message["tool_call_id"]: message
            for message in result.lite_sample.messages
            if message.get("role") == "tool"
        }
        assert {"type": "text", "text": "computer screen text"} in tool_messages["call_0000"][
            "content"
        ]
        assert {"type": "text", "text": "bash stdout"} in tool_messages["call_0001"]["content"]

    async def test_mixed_computer_and_image_extra_do_not_share_provider_images(
        self,
        monkeypatch,
    ):
        extra = make_tool_schema("visual_extra", description="Return visual feedback.")
        r1 = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "provider_computer_1",
                    "actions": [{"type": "screenshot"}],
                },
                {
                    "type": "function_call",
                    "id": "provider_visual_1",
                    "name": "visual_extra",
                    "arguments": "{}",
                },
            ]
        )
        mock = AsyncMock(side_effect=[r1, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        env = _MixedComputerAndVisualExtraResultsEnv(terminate_after=2)
        result = await agent.sample(env, max_steps=3)

        assert [_image_rgb(image) for image in result.lite_sample.images] == [
            (255, 255, 255),
            (110, 30, 30),
            (30, 110, 30),
        ]
        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 1, 2)]

        second_input = mock.call_args_list[1].kwargs["input"]
        computer_outputs = [
            item for item in second_input if item.get("type") == "computer_call_output"
        ]
        assert len(computer_outputs) == 1
        computer_payload = computer_outputs[0]["output"]["image_url"].split("base64,", 1)[1]
        assert base64.b64decode(computer_payload) == env._computer_shot

        function_outputs = [
            item for item in second_input if item.get("type") == "function_call_output"
        ]
        assert len(function_outputs) == 1
        output = function_outputs[0]["output"]
        assert isinstance(output, list)
        extra_payloads = [
            base64.b64decode(block["image_url"].split("base64,", 1)[1])
            for block in output
            if block.get("type") == "input_image"
        ]
        assert extra_payloads == [env._extra_shot]

    async def test_function_call_output_errors_for_malformed_and_undeclared_siblings(
        self,
        monkeypatch,
    ):
        extra = make_tool_schema(
            "bash",
            description="Run a command.",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        )
        r1 = _fake_response(
            [
                {
                    "type": "function_call",
                    "id": "provider_bash_1",
                    "name": "bash",
                    "arguments": '{"command": "pwd"}',
                },
                {
                    "type": "function_call",
                    "id": "provider_bad_json_1",
                    "name": "bash",
                    "arguments": "{not json",
                },
                {
                    "type": "function_call",
                    "id": "provider_unknown_1",
                    "name": "unknown_tool",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "id": "provider_inactive_report_1",
                    "name": "report_infeasible",
                    "arguments": '{"reason": "blocked"}',
                },
            ]
        )
        r2 = _fake_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        result = await agent.sample(_ToolResultsEnv(terminate_after=2), max_steps=3)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert assistant["tool_calls"] == [
            make_tool_call("bash", {"command": "pwd"}, call_id="call_0000"),
        ]

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = {
            item["call_id"]: item["output"]
            for item in second_input
            if item.get("type") == "function_call_output"
        }
        assert outputs["provider_bash_1"] == "per-call stdout"
        assert "malformed function_call arguments for bash" in outputs["provider_bad_json_1"]
        assert "undeclared function_call: unknown_tool" in outputs["provider_unknown_1"]
        assert (
            "undeclared function_call: report_infeasible" in outputs["provider_inactive_report_1"]
        )
        assert outputs["provider_bad_json_1"] != "ok"
        assert outputs["provider_unknown_1"] != "ok"
        assert outputs["provider_inactive_report_1"] != "ok"
        assert "per-call stdout" not in outputs["provider_bad_json_1"]
        assert "per-call stdout" not in outputs["provider_unknown_1"]
        assert "per-call stdout" not in outputs["provider_inactive_report_1"]

    async def test_chained_function_call_image_result_does_not_append_user_image(
        self,
        monkeypatch,
    ):
        extra = make_tool_schema("visual_extra", description="Return visual feedback.")
        r1 = _fake_response(
            [
                {
                    "type": "function_call",
                    "id": "provider_visual_1",
                    "name": "visual_extra",
                    "arguments": "{}",
                }
            ]
        )
        r2 = _fake_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        )
        env = _MultiImageToolResultsEnv(terminate_after=2)
        result = await agent.sample(env, max_steps=3)
        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 1, 2, 3)]
        assert result.processed_images == result.lite_sample.images

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "function_call_output"]
        assert len(outputs) == 1
        output = outputs[0]["output"]
        assert isinstance(output, list)
        assert [block["text"] for block in output if block.get("type") == "input_text"] == [
            "visual obs"
        ]
        output_images = [block for block in output if block.get("type") == "input_image"]
        assert [
            base64.b64decode(block["image_url"].split("base64,", 1)[1]) for block in output_images
        ] == env._result_shots
        function_output_index = next(
            i for i, item in enumerate(second_input) if item.get("type") == "function_call_output"
        )
        user_images = [
            block
            for item in second_input[function_output_index + 1 :]
            if item.get("role") == "user"
            for block in item.get("content", [])
            if isinstance(block, dict) and block.get("type") == "input_image"
        ]
        assert not user_images, (
            "Responses API chaining rejects standalone user images after "
            f"function_call_output: {second_input}"
        )

    async def test_client_managed_function_call_image_result_is_model_visible(
        self,
        monkeypatch,
    ):
        extra = make_tool_schema("visual_extra", description="Return visual feedback.")
        r1 = _fake_response(
            [
                {
                    "type": "function_call",
                    "id": "provider_visual_1",
                    "name": "visual_extra",
                    "arguments": "{}",
                }
            ]
        )
        r2 = _fake_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
            api_kwargs={"max_output_tokens": 4096, "chain_previous_response": False},
        )
        env = _MultiImageToolResultsEnv(terminate_after=2)
        result = await agent.sample(env, max_steps=3)
        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 1, 2, 3)]
        assert result.processed_images == result.lite_sample.images

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "function_call_output"]
        assert len(outputs) == 1
        output = outputs[0]["output"]
        assert isinstance(output, list)
        assert [block["text"] for block in output if block.get("type") == "input_text"] == [
            "visual obs"
        ]
        output_images = [block for block in output if block.get("type") == "input_image"]
        assert [
            base64.b64decode(block["image_url"].split("base64,", 1)[1]) for block in output_images
        ] == env._result_shots
        user_image_payloads = [
            base64.b64decode(block["image_url"].split("base64,", 1)[1])
            for item in second_input
            if item.get("role") == "user"
            for block in item.get("content", [])
            if isinstance(block, dict) and block.get("type") == "input_image"
        ]
        for shot in env._result_shots:
            assert shot not in user_image_payloads

    async def test_desktop_feedback_never_echoes_an_empty_provider_call_id(self, monkeypatch):
        """A call the provider left unidentified is not replayed or answered.

        Paired with an identified call in the SAME turn: a response attempt
        without a provider call id cannot be answered on the provider wire. The
        provider feedback step this test targets is reachable because another
        call in the same turn is valid and receives feedback.
        """
        resp1 = _fake_response(
            [
                # provider omitted call_id entirely
                {"type": "computer_call", "actions": [{"type": "screenshot"}]},
                {"type": "computer_call", "call_id": "good1", "actions": [{"type": "screenshot"}]},
            ]
        )
        mock = AsyncMock(side_effect=[resp1, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _FakeEnv(terminate_after=99)
        await GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"chain_previous_response": False},
        ).sample(env, max_steps=3)

        second_input = mock.call_args_list[1].kwargs["input"]
        # The unidentified call is never echoed as a call or an output: only
        # the identified sibling call gets a request item / ack pair.
        assert [
            item["call_id"] for item in second_input if item.get("type") == "computer_call"
        ] == ["good1"]
        assert [
            item["call_id"] for item in second_input if item.get("type") == "computer_call_output"
        ] == ["good1"]
        error_texts = [
            block["text"]
            for item in second_input
            if item.get("role") == "user"
            for block in item.get("content", [])
            if isinstance(block, dict) and "text" in block
        ]
        assert any("missing provider id for computer" in text for text in error_texts)

    async def test_desktop_feedback_sends_the_parse_error_beside_the_screenshot_ack(
        self,
        monkeypatch,
    ):
        """A malformed but identified computer_call still needs its output item,
        so the parse error rides along as its own model-visible user item.

        Paired with a valid sibling call in the same turn for the same
        unconditional-termination reason as the empty-call-id case above.
        """
        resp1 = _fake_response(
            [
                {"type": "computer_call", "call_id": "c1", "actions": [{"type": "click"}]},
                {"type": "computer_call", "call_id": "good1", "actions": [{"type": "screenshot"}]},
            ]
        )
        mock = AsyncMock(side_effect=[resp1, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _FakeEnv(terminate_after=99)
        await GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"chain_previous_response": False},
        ).sample(env, max_steps=3)

        second_input = mock.call_args_list[1].kwargs["input"]
        # The malformed call is still replayed and acknowledged, alongside its
        # valid sibling.
        assert [
            item["call_id"] for item in second_input if item.get("type") == "computer_call"
        ] == ["c1", "good1"]
        acks = [item for item in second_input if item.get("type") == "computer_call_output"]
        assert [item["call_id"] for item in acks] == ["c1", "good1"]
        error_texts = [
            block["text"]
            for item in second_input
            if item.get("role") == "user"
            for block in item.get("content", [])
            if isinstance(block, dict) and "text" in block
        ]
        assert any(text.startswith("model output error:") for text in error_texts)

    async def test_reasoning_only_output_terminates_through_the_env(self, monkeypatch):
        """N3: reasoning-only output is a final turn, not a RuntimeError.

        This exact shape exists in published Lite.ScaleCUA gpt data (a final
        turn whose only content part is ``inline_reasoning``); it used to
        discard the whole episode.
        """
        resp = _fake_response(
            [
                {
                    "type": "reasoning",
                    "summary": [{"text": "checking the screen"}],
                }
            ]
        )
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _RecordingFakeEnv(terminate_after=1)
        agent = GPTDesktopUseAgent(model_id="gpt-5.5")

        rl = await agent.sample(env, max_steps=2)
        assert len(env.actions_seen) == 1
        assert tool_call_name(env.actions_seen[0][0]) == "response"
        assert tool_call_arguments(env.actions_seen[0][0]) == {"text": ""}
        assert rl.terminated is True
        assert not rl.lite_sample.messages[-1].get("tool_calls")

    async def test_content_only_final_text_is_not_saved_as_response_tool(self, monkeypatch):
        resp = _fake_response(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "  18 x 24  "}],
                }
            ]
        )
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _RejectEmptyActionsEnv(terminate_after=99)
        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        result = await agent.sample(env, max_steps=2)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert "raw_response" not in assistant
        assert assistant["content"] == [{"type": "text", "text": "  18 x 24  "}]
        assert not assistant.get("tool_calls")
        assert len(env.actions_seen) == 1
        assert tool_call_name(env.actions_seen[0][0]) == "response"
        assert tool_call_id(env.actions_seen[0][0]) is None
        assert tool_call_arguments(env.actions_seen[0][0]) == {"text": "18 x 24"}
        assert result.terminated is True
        assert result.truncated is False
        assert result.episode_return == 1.0
        assert result.steps[0].reward == 1.0

    async def test_content_only_response_attempt_can_continue_with_provider_feedback(
        self,
        monkeypatch,
    ):
        first = _fake_response(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "First answer"}],
                }
            ]
        )
        second = _fake_response(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Second answer"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[first, second])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _ContentOnlyContinuationEnv(terminate_after=2)
        result = await GPTDesktopUseAgent(model_id="gpt-5.5").sample(env, max_steps=3)

        assert result.terminated is True
        assert result.truncated is False
        assert len(env.actions_seen) == 2
        assert [tool_call_arguments(turn[0]) for turn in env.actions_seen] == [
            {"text": "First answer"},
            {"text": "Second answer"},
        ]
        assert [message["role"] for message in result.lite_sample.messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        second_input = mock.call_args_list[1].kwargs["input"]
        user_blocks = [
            block
            for item in second_input
            if item.get("role") == "user"
            for block in item.get("content", [])
            if isinstance(block, dict)
        ]
        assert {"type": "input_text", "text": "attempt 1: First answer"} in user_blocks
        assert any(block.get("type") == "input_image" for block in user_blocks)

    async def test_mobile_content_only_response_attempt_can_continue_with_image_feedback(
        self,
        monkeypatch,
    ):
        first = _fake_response(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "First mobile answer"}],
                }
            ]
        )
        second = _fake_response(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Second mobile answer"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[first, second])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _MobileContentOnlyContinuationEnv(terminate_after=2, feedback_image=True)
        result = await GPTMobileUseAgent(model_id="gpt-5.5").sample(env, max_steps=3)

        assert result.terminated is True
        assert result.truncated is False
        assert len(env.actions_seen) == 2
        assert [tool_call_arguments(turn[0]) for turn in env.actions_seen] == [
            {"text": "First mobile answer"},
            {"text": "Second mobile answer"},
        ]
        assert [message["role"] for message in result.lite_sample.messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        second_input = mock.call_args_list[1].kwargs["input"]
        user_blocks = [
            block
            for item in second_input
            if item.get("role") == "user"
            for block in item.get("content", [])
            if isinstance(block, dict)
        ]
        assert {"type": "input_text", "text": "attempt 1: First mobile answer"} in user_blocks
        assert any(block.get("type") == "input_image" for block in user_blocks)

    async def test_mobile_content_only_text_only_feedback_failure_is_intentional(
        self,
        monkeypatch,
    ):
        first = _fake_response(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "First mobile answer"}],
                }
            ]
        )
        mock = AsyncMock(return_value=first)
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _MobileContentOnlyContinuationEnv(terminate_after=2, feedback_image=False)

        with pytest.raises(RuntimeError, match="No image returned after env.step"):
            await GPTMobileUseAgent(model_id="gpt-5.5").sample(env, max_steps=3)

        assert len(env.actions_seen) == 1
        assert tool_call_arguments(env.actions_seen[0][0]) == {
            "text": "First mobile answer"
        }

    async def test_content_only_final_uses_runtime_response_when_enabled(self, monkeypatch):
        resp = _fake_response(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Final answer"}],
                }
            ]
        )
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _RejectEmptyActionsEnv(terminate_after=99)
        env.metadata.extra_tool_schemas = [LiteFinishToolSet.get_tool_schema("response")]
        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        result = await agent.sample(env, max_steps=2)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert "raw_response" not in assistant
        assert assistant["content"] == [{"type": "text", "text": "Final answer"}]
        assert not assistant.get("tool_calls")
        assert len(env.actions_seen) == 1
        assert tool_call_name(env.actions_seen[0][0]) == "response"
        assert tool_call_id(env.actions_seen[0][0]) is None
        assert tool_call_arguments(env.actions_seen[0][0]) == {"text": "Final answer"}
        assert result.terminated is True
        assert result.truncated is False


# -----------------------------------------------------------------------------
# chain_previous_response
# -----------------------------------------------------------------------------


class TestUsePreviousResponseId:
    """chaining via previous_response_id; kwarg-gated."""

    async def test_default_sends_previous_response_id(self, monkeypatch):
        # Two steps: first returns a computer_call, second returns final text.
        r1 = {
            "output": [
                {"type": "computer_call", "call_id": "c1", "actions": [{"type": "screenshot"}]}
            ],
            "id": "resp1",
            "usage": {},
        }
        r2 = _fake_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(_FakeEnv(terminate_after=2), max_steps=5)

        # First call has no previous_response_id; second does, and equals "resp1"
        assert mock.call_args_list[0].kwargs.get("previous_response_id") is None
        assert mock.call_args_list[1].kwargs.get("previous_response_id") == "resp1"

    async def test_kwarg_false_omits_previous_response_id(self, monkeypatch):
        # Capture input snapshots by deep-copying at call time (input_items is
        # passed by reference and mutates after each API call).
        import copy

        snapshots: list[list[dict[str, Any]]] = []

        async def capturing_aresponses(**kwargs):
            snapshots.append(copy.deepcopy(kwargs.get("input", [])))
            if len(snapshots) < 2:
                return {
                    "output": [
                        {
                            "type": "computer_call",
                            "call_id": f"c{len(snapshots)}",
                            "actions": [{"type": "screenshot"}],
                        }
                    ],
                    "id": f"resp{len(snapshots)}",
                    "usage": {},
                }
            return _fake_response()

        monkeypatch.setattr("litellm.aresponses", capturing_aresponses)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096, "chain_previous_response": False},
        )
        await agent.sample(_FakeEnv(terminate_after=3), max_steps=5)

        # call 1 has only initial user items (instruction + screenshot); call 2
        # has those PLUS a computer_call_output echoing step 1's computer_call.
        assert len(snapshots) >= 2
        call1_has_cco = any(it.get("type") == "computer_call_output" for it in snapshots[0])
        call2_has_cco = any(it.get("type") == "computer_call_output" for it in snapshots[1])
        assert not call1_has_cco, "call 1 should have no computer_call_output yet"
        assert call2_has_cco, (
            "call 2 should have computer_call_output from step 1 (no chaining → accumulated)"
        )

    async def test_prompt_cache_key_is_stable_across_turns(self, monkeypatch):
        r1 = {
            "output": [
                {"type": "computer_call", "call_id": "c1", "actions": [{"type": "screenshot"}]}
            ],
            "id": "resp1",
            "usage": {},
        }
        mock = AsyncMock(side_effect=[r1, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(_FakeEnv(terminate_after=2), max_steps=3)

        cache_keys = [
            call.kwargs.get("extra_body", {}).get("prompt_cache_key")
            for call in mock.call_args_list
        ]
        assert len(cache_keys) == 2
        assert cache_keys[0]
        assert cache_keys[0] == cache_keys[1]
        assert cache_keys[0].startswith("cua-lite-")


# -----------------------------------------------------------------------------
# history compaction (full_history_size)
# -----------------------------------------------------------------------------


class TestHistoryCompaction:
    """``full_history_size`` (native computer-use, chain=False): keep the last N
    steps in full, fold older steps into a text summary, and preserve
    computer_call<->output pairing.

    Every assertion reads the provider ``input`` payload ``agent.sample()``
    actually sent, so the window contract is pinned on the request surface
    rather than on the compaction helper's placement.
    """

    @staticmethod
    def _click_response(rid: str, cid: str, x: int) -> dict[str, Any]:
        return {
            "output": [
                {
                    "type": "computer_call",
                    "call_id": cid,
                    "actions": [{"type": "click", "x": x, "y": x}],
                }
            ],
            "id": rid,
            "usage": {},
        }

    @staticmethod
    def _agent(full_history_size: int | None, **kwargs: Any) -> GPTDesktopUseAgent:
        # chain=False is what puts client-managed history under the window.
        return GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"chain_previous_response": False},
            full_history_size=full_history_size,
            **kwargs,
        )

    @staticmethod
    def _head_blocks(request_input: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Content blocks of the request's role-bearing (developer/user) items."""
        return [
            block
            for item in request_input
            if item.get("role") and isinstance(item.get("content"), list)
            for block in item["content"]
        ]

    @classmethod
    def _head_text(cls, request_input: list[dict[str, Any]]) -> str:
        return "\n".join(
            block["text"]
            for block in cls._head_blocks(request_input)
            if block.get("type") == "input_text"
        )

    @classmethod
    def _head_image_blocks(cls, request_input: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            block
            for block in cls._head_blocks(request_input)
            if block.get("type") in ("input_image", "computer_screenshot")
        ]

    @staticmethod
    def _call_ids(request_input: list[dict[str, Any]], item_type: str) -> list[str]:
        return [it["call_id"] for it in request_input if it.get("type") == item_type]

    async def test_no_window_keeps_every_step(self, monkeypatch):
        mock = AsyncMock(
            side_effect=[
                self._click_response("r1", "c1", 1),
                self._click_response("r2", "c2", 2),
                self._click_response("r3", "c3", 3),
            ]
        )
        monkeypatch.setattr("litellm.aresponses", mock)

        await self._agent(None).sample(_FakeEnv(terminate_after=3), max_steps=5)

        last_input = mock.call_args_list[-1].kwargs["input"]
        assert self._call_ids(last_input, "computer_call") == ["c1", "c2"]
        assert self._call_ids(last_input, "computer_call_output") == ["c1", "c2"]
        assert "Previous actions" not in self._head_text(last_input)
        # The task screenshot is still attached to the first user message.
        assert len(self._head_image_blocks(last_input)) == 1

    async def test_window_at_or_above_step_count_is_noop(self, monkeypatch):
        mock = AsyncMock(
            side_effect=[
                self._click_response("r1", "c1", 1),
                self._click_response("r2", "c2", 2),
                self._click_response("r3", "c3", 3),
            ]
        )
        monkeypatch.setattr("litellm.aresponses", mock)

        await self._agent(2).sample(_FakeEnv(terminate_after=3), max_steps=5)

        # 2 accumulated steps <= window 2: nothing is folded.
        last_input = mock.call_args_list[-1].kwargs["input"]
        assert self._call_ids(last_input, "computer_call") == ["c1", "c2"]
        assert "Previous actions" not in self._head_text(last_input)
        assert len(self._head_image_blocks(last_input)) == 1

    async def test_zero_window_is_rejected(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        # At least one full call/output pair must remain, so ``0`` is rejected
        # on the first request instead of silently degrading the window.
        with pytest.raises(ValueError, match="positive integer"):
            await self._agent(0).sample(_FakeEnv(terminate_after=3), max_steps=5)
        assert mock.call_args_list == []

    async def test_folds_old_steps_preserves_pairing_and_injects_summary(self, monkeypatch):
        mock = AsyncMock(
            side_effect=[
                self._click_response("r1", "c1", 1),
                self._click_response("r2", "c2", 2),
                self._click_response("r3", "c3", 3),
                self._click_response("r4", "c4", 4),
            ]
        )
        monkeypatch.setattr("litellm.aresponses", mock)

        await self._agent(1).sample(_FakeEnv(terminate_after=4), max_steps=6)

        last_input = mock.call_args_list[-1].kwargs["input"]
        # Only the last step survives in full; pairing intact, no orphans.
        assert self._call_ids(last_input, "computer_call") == ["c3"]
        assert self._call_ids(last_input, "computer_call_output") == ["c3"]

        # Older steps folded into a text summary in the head user message; the
        # head's stale screenshot is dropped.
        head_text = self._head_text(last_input)
        assert "Previous actions" in head_text
        assert "Step 1: click(1,1)" in head_text
        assert "Step 2: click(2,2)" in head_text
        assert "Step 3" not in head_text
        assert self._head_image_blocks(last_input) == []

    async def test_folds_old_steps_preserves_developer_prompt(self, monkeypatch):
        mock = AsyncMock(
            side_effect=[
                self._click_response("r1", "c1", 1),
                self._click_response("r2", "c2", 2),
                self._click_response("r3", "c3", 3),
            ]
        )
        monkeypatch.setattr("litellm.aresponses", mock)

        await self._agent(1, system_prompt="CUSTOM_ANCHOR").sample(
            _FakeEnv(terminate_after=3), max_steps=5
        )

        last_input = mock.call_args_list[-1].kwargs["input"]
        assert last_input[0] == {"role": "developer", "content": "CUSTOM_ANCHOR"}
        assert last_input[1]["role"] == "user"
        head_text = self._head_text(last_input)
        # The task instruction survives alongside the injected summary.
        assert "instr" in head_text
        assert "Previous actions" in head_text
        assert self._call_ids(last_input, "computer_call") == ["c2"]

    async def test_sample_loop_window_and_pairing(self, monkeypatch):
        # chain=False + full_history_size=1: history compacts to the last step;
        # the surviving computer_call_output stays paired to its computer_call.
        r1 = {
            "output": [
                {"type": "computer_call", "call_id": "c1", "actions": [{"type": "screenshot"}]}
            ],
            "id": "r1",
            "usage": {},
        }
        r2 = {
            "output": [
                {"type": "computer_call", "call_id": "c2", "actions": [{"type": "screenshot"}]}
            ],
            "id": "r2",
            "usage": {},
        }
        r3 = {
            "output": [
                {"type": "computer_call", "call_id": "c3", "actions": [{"type": "screenshot"}]}
            ],
            "id": "r3",
            "usage": {},
        }
        r4 = {
            "output": [
                {"type": "computer_call", "call_id": "c4", "actions": [{"type": "screenshot"}]}
            ],
            "id": "r4",
            "usage": {},
        }
        mock = AsyncMock(side_effect=[r1, r2, r3, r4])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096, "chain_previous_response": False},
            full_history_size=1,
        )
        # env stops after exactly 4 steps; mock supplies exactly 4 responses
        # (no StopIteration → no retry-backoff hang).
        result = await agent.sample(_FakeEnv(terminate_after=4), max_steps=6)

        # Saved training metadata follows the compacted request, not the
        # cumulative set of images that were visible before pruning.
        assert [tuple(step.image_indices) for step in result.steps] == [
            (0,),
            (0, 1),
            (2,),
            (3,),
        ]
        assert "_cua_lite_image_index" not in result.steps[-1].prompt

        # Final request: compacted to <=1 screenshot-bearing step, pairing intact.
        last_input = mock.call_args_list[-1].kwargs["input"]
        assert "_cua_lite_image_index" not in repr(last_input)
        call_ids = {
            it["call_id"]
            for it in last_input
            if isinstance(it, dict) and it.get("type") == "computer_call"
        }
        output_ids = {
            it["call_id"]
            for it in last_input
            if isinstance(it, dict) and it.get("type") == "computer_call_output"
        }
        screenshots = sum(
            1
            for it in last_input
            if isinstance(it, dict) and it.get("type") == "computer_call_output"
        )
        assert screenshots <= 1, f"window=1 should keep <=1 screenshot, got {screenshots}"
        assert output_ids, "expected a computer_call_output in compacted history"
        assert output_ids <= call_ids, f"unpaired computer_call_outputs: {output_ids - call_ids}"
