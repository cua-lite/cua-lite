"""Native-provider browser prompt guards.

Browser envs may keep page context in observation metadata for logs, but
provider-native agents should only render the model-visible text channel.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from lite.agents.models.gemini import agent as gemini_agent_module
from lite.agents.models.gemini.agent import GeminiDesktopUseAgent
from lite.agents.models.gpt.agent import GPTDesktopUseAgent
from lite.core import LiteCUAMetadata
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

_SENTINELS = {
    "url": "https://metadata-leak.test",
    "title": "METADATA PAGE TITLE",
    "web_text": "CURRENT URL: https://metadata-leak.test\nDOM: <html>secret</html>",
    "body_text": "SECRET BODY TEXT",
}


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), color="white").save(buf, format="PNG")
    return buf.getvalue()


class _BrowserMetadataEnv:
    def __init__(self) -> None:
        self.metadata = LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
            others={"resolution": [800, 600]},
        )
        self._shot = _png_bytes()
        self.actions_seen: list[list[dict[str, Any]]] = []

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(
            image=self._shot,
            text="Instruction: use the screenshot only.",
            metadata=dict(_SENTINELS),
        )

    async def step(self, actions: list[dict[str, Any]]) -> LiteEnvStepResult:
        self.actions_seen.append(list(actions))
        return LiteEnvStepResult(reward=1.0, terminated=True)

    async def close(self) -> None:
        pass


def _assert_no_metadata_sentinel(payload: Any) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    for sentinel in _SENTINELS.values():
        assert sentinel not in serialized
    assert "Instruction: use the screenshot only." in serialized


def _gpt_response() -> dict[str, Any]:
    return {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
        "id": "resp_test",
        "usage": {},
    }


async def test_gpt_browser_request_does_not_render_observation_metadata(monkeypatch):
    pytest.importorskip("litellm")
    monkeypatch.setattr(
        "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
        AsyncMock(return_value=[(800, 600)]),
    )
    mock = AsyncMock(return_value=_gpt_response())
    monkeypatch.setattr("litellm.aresponses", mock)

    await GPTDesktopUseAgent(model_id="gpt-5.5").sample(_BrowserMetadataEnv(), max_steps=2)

    _assert_no_metadata_sentinel(mock.call_args.kwargs["input"])


def _claude_response(content: Any = "done") -> Any:
    msg = SimpleNamespace(content=content, tool_calls=[], role="assistant")
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model_dump=lambda: {"choices": []},
    )


async def test_claude_browser_request_does_not_render_observation_metadata(monkeypatch):
    pytest.importorskip("litellm")
    from lite.agents.models.claude.agent import ClaudeDesktopUseAgent

    mock = AsyncMock(return_value=_claude_response())
    monkeypatch.setattr("litellm.acompletion", mock)

    await ClaudeDesktopUseAgent(model_id="claude-opus-4-6").sample(
        _BrowserMetadataEnv(),
        max_steps=2,
    )

    _assert_no_metadata_sentinel(mock.call_args.kwargs["messages"])


def _gemini_response() -> dict[str, Any]:
    return {
        "candidates": [
            {"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": "done"}]}}
        ]
    }


async def test_gemini_browser_request_does_not_render_observation_metadata(monkeypatch):
    seen: list[dict[str, Any]] = []

    async def _fake_generate(*, model, api_base, api_key, payload, timeout):
        seen.append(payload)
        return _gemini_response()

    monkeypatch.setattr(gemini_agent_module, "agenerate_content", _fake_generate)

    await GeminiDesktopUseAgent(
        model_id="gemini-3.5-flash",
        api_base="https://proxy",
        api_key="k",
        metadata=_BrowserMetadataEnv().metadata,
    ).sample(_BrowserMetadataEnv(), max_steps=2)

    _assert_no_metadata_sentinel(seen[0])
