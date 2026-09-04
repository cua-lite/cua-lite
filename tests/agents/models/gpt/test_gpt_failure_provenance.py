"""Split GPT agent characterization tests.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_*.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from agents.models.gpt._support import _FakeEnv, _RecordingFakeEnv

from lite.agents.models.gpt.agent import GPTDesktopUseAgent
from lite.agents.models.gpt.utils.responses import ResponseAPIError
from lite.core.tools.calls import tool_call_id


class TestFailedResponseIsNeverAFinal:
    """A failed Responses result must raise, never be parsed as model output."""

    _FAILED_WITH_TEXT = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "not a final answer"}],
            }
        ],
        "id": "resp_failed",
        "status": "failed",
        "error": {"message": "oops"},
        "usage": {},
    }

    async def test_failed_status_with_text_output_is_not_parsed_as_a_final(self, monkeypatch):
        mock = AsyncMock(return_value=dict(self._FAILED_WITH_TEXT))
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _RecordingFakeEnv(terminate_after=1)
        with pytest.raises(ResponseAPIError):
            await GPTDesktopUseAgent(model_id="gpt-5.5").sample(env, max_steps=3)
        # The turn raised before parse, so the text never reached the env as a
        # no-tool-call final.
        assert env.actions_seen == []

    async def test_fail_fast_on_api_error_in_api_kwargs_cannot_re_enable_the_old_path(
        self,
        monkeypatch,
    ):
        """``fail_fast_on_api_error`` is gone. Smuggling it through the free-form
        ``api_kwargs`` dict must not restore "failed == finished with no output"."""
        mock = AsyncMock(return_value=dict(self._FAILED_WITH_TEXT))
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"fail_fast_on_api_error": False},
        )
        with pytest.raises(ResponseAPIError):
            await agent.sample(_FakeEnv(terminate_after=1), max_steps=3)


class TestParseProvenanceIsReturnedNotStashed:
    """Provider->canonical provenance travels in the parser's return value.

    A mutable side channel on the agent (the retired ``_last_parsed_output``
    style) would let a second turn read stale provenance; this pins the return
    value as the only carrier.
    """

    def test_parse_returns_provenance_without_mutating_agent_state(self):
        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        state_before = dict(vars(agent))

        parsed = agent._parse_output_items(
            [
                {
                    "type": "computer_call",
                    "call_id": "provider_1",
                    "actions": [{"type": "click", "x": 10, "y": 20}],
                }
            ],
            resolution=(800, 600),
        )

        assert vars(agent) == state_before
        [provider_call] = parsed.provider_calls
        assert provider_call.provider_call_id == "provider_1"
        assert provider_call.canonical_call_id == "call_0000"
        assert provider_call.error is None
        assert provider_call.is_final_for_canonical is True
        assert tool_call_id(parsed.message["tool_calls"][0]) == "call_0000"


class TestTruncatedCallIsNeverAFinal:
    """A call the provider cut off is malformed output, not a deliberate answer.

    The Responses API emits ``{"type": "computer_call", "status": "incomplete"}``
    with no action when the item hits ``max_output_tokens`` mid-emission. Parsed as
    zero tool calls with no error, the loop reads that as the model choosing to stop
    and ends the episode -- so a truncated turn is scored as an agent that gave up on
    an env it never touched. It has to surface as a parse error instead.
    """

    def test_incomplete_computer_call_is_a_parse_error(self):
        from lite.agents.models.gpt.utils.parse import _normalized_gpt_output_records

        (record,) = _normalized_gpt_output_records([
            {
                "id": "cu_1",
                "type": "computer_call",
                "status": "incomplete",
                "call_id": "call_1",
            }
        ])

        assert record.parse_error is not None
        assert "truncated" in record.parse_error

    def test_completed_computer_call_with_actions_is_not_an_error(self):
        from lite.agents.models.gpt.utils.parse import _normalized_gpt_output_records

        (record,) = _normalized_gpt_output_records([
            {
                "id": "cu_2",
                "type": "computer_call",
                "status": "completed",
                "call_id": "call_2",
                "actions": [{"type": "click", "x": 10, "y": 20, "button": "left"}],
            }
        ])

        assert record.parse_error is None
        assert record.actions
