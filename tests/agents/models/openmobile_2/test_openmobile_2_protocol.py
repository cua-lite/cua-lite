"""OpenMobile-2 history protocol tests."""

from __future__ import annotations

import pytest

from lite.agents.models.openmobile_2.protocol import (
    COLLAPSED_SCREENSHOT_TEXT,
    OpenMobile2HistoryProtocol,
    SESSION_USER_PROMPT,
)


def _initial_user(instruction: str, image: int = 0) -> dict:
    # cua-lite's build order: screenshot first, instruction text after.
    return {
        "role": "user",
        "content": [
            {"type": "image", "index": image},
            {"type": "text", "text": instruction},
        ],
    }


def _assistant(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": "Thought: act now."}],
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "mobile_use", "arguments": {"action": "wait", "time": 1}},
            }
        ],
    }


def _observation(image: int, text: str | None = None) -> dict:
    content = [{"type": "image", "index": image}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return {"role": "tool", "tool_call_id": f"call_{image}", "content": content}


def test_first_user_message_is_rewritten_to_the_official_shape() -> None:
    # Official QWEN35_SESSION_USER_PROMPT: "The user query: {instruction}\n"
    # as TEXT FIRST, screenshot after — cua-lite builds [image, text].
    messages = [_initial_user("Add milk to the cart"), _assistant("call_0")]
    processed = OpenMobile2HistoryProtocol().process_messages(messages)

    assert processed[0]["content"] == [
        {"type": "text", "text": "The user query: Add milk to the cart\n"},
        {"type": "image", "index": 0},
    ]
    # Assistant turns survive verbatim.
    assert processed[1] == messages[1]
    # The input list is not mutated.
    assert messages[0]["content"][0]["type"] == "image"


def test_short_history_keeps_every_screenshot() -> None:
    # last_n=3: with at most 3 image-bearing messages nothing collapses.
    messages = [
        _initial_user("task"),
        _assistant("call_0"),
        _observation(1),
        _assistant("call_1"),
        _observation(2),
    ]
    processed = OpenMobile2HistoryProtocol().process_messages(messages)
    assert [m["content"] for m in processed] == [
        [
            {"type": "text", "text": "The user query: task\n"},
            {"type": "image", "index": 0},
        ],
        messages[1]["content"],
        messages[2]["content"],
        messages[3]["content"],
        messages[4]["content"],
    ]


def test_long_history_collapses_only_out_of_window_screenshots() -> None:
    # 5 image-bearing messages, last_n=3: the two oldest screenshot slots
    # collapse to placeholder text; every message slot and every text part
    # (query, Thoughts, tool_calls) survives.
    messages = [
        _initial_user("task"),
        _assistant("call_0"),
        _observation(1),
        _assistant("call_1"),
        _observation(2),
        _assistant("call_2"),
        _observation(3),
        _assistant("call_3"),
        _observation(4),
    ]
    processed = OpenMobile2HistoryProtocol().process_messages(messages)

    assert len(processed) == len(messages)
    # Window: the initial screenshot and observation 1 collapse; the
    # query text of the first user message survives in place.
    assert processed[0]["content"] == [
        {"type": "text", "text": "The user query: task\n"},
        {"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT},
    ]
    assert processed[2]["content"] == [{"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT}]
    # The last 3 image-bearing messages keep their screenshots.
    assert processed[4]["content"] == messages[4]["content"]
    assert processed[6]["content"] == messages[6]["content"]
    assert processed[8]["content"] == messages[8]["content"]
    # Text history survives untouched.
    assert processed[1] == messages[1]
    assert processed[3] == messages[3]


def test_collapse_keeps_non_image_parts_before_the_placeholders() -> None:
    # Official _collapse_image_parts order: non-image parts first, then one
    # placeholder per dropped screenshot (here: the error text of a failed
    # action, which must stay model-visible).
    messages = [
        _initial_user("task"),
        _assistant("call_0"),
        _observation(1, text="## Error from previous action: tap missed"),
        _assistant("call_1"),
        _observation(2),
        _assistant("call_2"),
        _observation(3),
        _assistant("call_3"),
        _observation(4),
    ]
    processed = OpenMobile2HistoryProtocol().process_messages(messages)
    assert processed[2]["content"] == [
        {"type": "text", "text": "## Error from previous action: tap missed"},
        {"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT},
    ]


def test_last_n_keeps_at_least_one_screenshot() -> None:
    # Official clamping: keep_n = max(1, last_n) — the most recent
    # screenshot survives even with last_n=0; the older initial one
    # collapses.
    messages = [_initial_user("task"), _assistant("call_0"), _observation(1)]
    processed = OpenMobile2HistoryProtocol(last_n=0).process_messages(messages)
    assert processed[0]["content"] == [
        {"type": "text", "text": "The user query: task\n"},
        {"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT},
    ]
    assert processed[2]["content"] == messages[2]["content"]


def test_metadata_part_rides_along_after_the_rewrite() -> None:
    # The metadata side-channel part stays in the message (stripped later
    # by keep_model_visible_content); the model-visible prefix is still
    # query-text-first.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "task"},
                {"type": "metadata", "payload": {"env_id": "androidworld"}},
            ],
        }
    ]
    processed = OpenMobile2HistoryProtocol().process_messages(messages)
    assert processed[0]["content"] == [
        {"type": "text", "text": "The user query: task\n"},
        {"type": "image", "index": 0},
        {"type": "metadata", "payload": {"env_id": "androidworld"}},
    ]


def test_session_user_prompt_matches_the_official_constant() -> None:
    assert SESSION_USER_PROMPT == "The user query: {instruction}\n"
    assert COLLAPSED_SCREENSHOT_TEXT == "This screenshot has been collapsed."


def test_process_messages_rejects_none_history() -> None:
    with pytest.raises(TypeError):
        OpenMobile2HistoryProtocol().process_messages(None)  # type: ignore[arg-type]
