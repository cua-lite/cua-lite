"""Browser page-context prompt guards.

These tests pin the prompt boundary for screenshot-only browser rollouts: envs
may keep URL/title/DOM/body details in metadata for logs, but the model-visible
prompt only gets that context when the env emits it through the text channel.
"""

from __future__ import annotations

import pytest
from PIL import Image

from lite.agents.models.fara.adapter import FaraDesktopUseAdapter
from lite.agents.models.qwen3_5.adapter import Qwen3_5DesktopUseAdapter
from lite.agents.models.qwen3_8.adapter import Qwen3_8DesktopUseAdapter
from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter
from lite.agents.models.ui_venus_2.adapter import UIVenus2BrowserUseAdapter
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call
from lite.core.tools.results import project_tool_result_text


def _browser_md() -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=[],
        valid_actions=None,
    )


def _visible_text(step: list[dict]) -> str:
    return "\n".join(
        part["text"]
        for message in step
        for part in message.get("content") or []
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _browser_post_action_sample(
    *,
    tool_text: str | None,
    tool_metadata: dict | None = None,
) -> LiteSample:
    img = Image.new("RGB", (32, 32), color=(90, 120, 150))
    tool_content: list[dict] = [{"type": "image", "index": 1}]
    if tool_text is not None:
        tool_content.append({"type": "text", "text": tool_text})
    if tool_metadata is not None:
        tool_content.append({"type": "metadata", "data": tool_metadata})
    return LiteSample(
        metadata=_browser_md(),
        images=[img, img],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Instruction: click the target."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Click the target."}],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [100, 200]}]},
                        call_id="call_0000",
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": tool_content,
            },
        ],
    )


@pytest.mark.parametrize(
    "adapter_cls",
    [
        Qwen3VLDesktopUseAdapter,
        Qwen3_5DesktopUseAdapter,
        Qwen3_8DesktopUseAdapter,
        FaraDesktopUseAdapter,
        UIVenus2BrowserUseAdapter,
    ],
)
def test_browser_page_context_metadata_is_not_rendered_as_prompt_text(adapter_cls) -> None:
    adapter = adapter_cls(metadata=_browser_md())
    sample = _browser_post_action_sample(
        tool_text=None,
        tool_metadata={
            "web_text": "CURRENT URL: https://leak.test\nPAGE TITLE: Leak\nDOM: <html>",
            "model_web_text": "CURRENT URL: https://model-leak.test\nDOM: secret model text",
            "url": "https://leak.test",
            "page_title": "Leak",
            "body_text": "secret body text",
        },
    )

    text = _visible_text(adapter.render_step(sample, 2, [None, None]))

    assert "Instruction: click the target." in text
    assert "CURRENT URL" not in text
    assert "PAGE TITLE" not in text
    assert "DOM:" not in text
    assert "https://leak.test" not in text
    assert "https://model-leak.test" not in text
    assert "secret model text" not in text
    assert "secret body text" not in text


@pytest.mark.parametrize(
    "adapter_cls",
    [
        Qwen3VLDesktopUseAdapter,
        Qwen3_5DesktopUseAdapter,
        Qwen3_8DesktopUseAdapter,
        FaraDesktopUseAdapter,
        UIVenus2BrowserUseAdapter,
    ],
)
def test_browser_action_error_stays_prompt_text_without_page_context(adapter_cls) -> None:
    adapter = adapter_cls(metadata=_browser_md())
    sample = _browser_post_action_sample(
        tool_text=project_tool_result_text(None, "invalid action: click outside viewport"),
        tool_metadata={"is_error": True, "url": "https://leak.test"},
    )

    text = _visible_text(adapter.render_step(sample, 2, [None, None]))

    assert "## Error from previous action:" in text
    assert "invalid action: click outside viewport" in text
    assert "https://leak.test" not in text
