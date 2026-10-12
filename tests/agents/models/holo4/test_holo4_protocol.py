"""Holo4 history compaction tests."""

from __future__ import annotations

import pytest

from lite.agents.models.holo4.protocol import Holo4HistoryProtocol


def _user(text: str, image: int) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image", "index": image},
        ],
    }


def _assistant(note: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": note}],
        "reasoning_content": f"private {note}",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "computer", "arguments": {"actions": []}},
            }
        ],
    }


def test_compaction_keeps_all_text_but_drops_out_of_window_images() -> None:
    # The official harness keeps every text message (instructions, tool
    # outputs, observation text, assistant turns with their tool calls and
    # reasoning) and evicts only out-of-window screenshots.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        _user("task instruction", 0),
        _assistant("remember alpha", "call_0"),
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [{"type": "text", "text": "old tool output"}],
        },
        _user("old screenshot", 1),
        _assistant("remember beta", "call_1"),
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "recent tool output"}],
        },
        _user("latest screenshot", 2),
    ]
    compacted = Holo4HistoryProtocol(max_n_images=1).process_messages(messages)
    assert compacted == [
        messages[0],
        {"role": "user", "content": [{"type": "text", "text": "task instruction"}]},
        messages[2],
        messages[3],
        {"role": "user", "content": [{"type": "text", "text": "old screenshot"}]},
        messages[5],
        messages[6],
        messages[7],
    ]


def test_short_history_keeps_sidecars_and_native_reasoning() -> None:
    messages = [_user("task", 0), _assistant("note", "call_0")]
    messages[1]["raw_response"] = {
        "adapter_key": "holo4@mobile@use",
        "text": "native response",
    }

    processed = Holo4HistoryProtocol(max_n_images=4).process_messages(messages)

    assert processed[0] == messages[0]
    assert processed[1]["content"] == messages[1]["content"]
    assert processed[1]["tool_calls"] == messages[1]["tool_calls"]
    assert processed[1]["reasoning_content"] == "private note"
    assert processed[1]["raw_response"] == messages[1]["raw_response"]


def test_image_only_first_user_message_drops_but_later_text_survives() -> None:
    # An image-only first user message carries no text to keep, so it drops
    # entirely; later observation text survives as an observation.
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        _assistant("saw the screen", "call_0"),
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [{"type": "text", "text": "old tool output"}],
        },
        _user("SYSTEM_ALERT: battery low", 1),
        _assistant("noted", "call_1"),
        _user("latest", 2),
    ]

    compacted = Holo4HistoryProtocol(max_n_images=1).process_messages(messages)

    assert compacted == [
        messages[1],
        messages[2],
        {"role": "user", "content": [{"type": "text", "text": "SYSTEM_ALERT: battery low"}]},
        messages[4],
        messages[5],
    ]


def test_default_window_matches_official_harness() -> None:
    # The official Holo4 harness keeps three screenshots (current + two
    # past) on AndroidWorld and OSWorld; OSWorld 2 keeps four.
    assert Holo4HistoryProtocol().max_n_images == 3


def test_none_messages_is_rejected_as_caller_error() -> None:
    with pytest.raises(TypeError, match="process_messages expects a list"):
        Holo4HistoryProtocol().process_messages(None)


def test_image_limit_counts_parts_not_observation_turns() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "task"},
                {"type": "image", "index": 0},
                {"type": "image", "index": 1},
            ],
        },
        _assistant("remember", "call_0"),
        _user("latest", 2),
    ]

    processed = Holo4HistoryProtocol(max_n_images=1).process_messages(messages)

    image_parts = [
        part
        for message in processed
        for part in message.get("content", [])
        if part.get("type") == "image"
    ]
    assert [part["index"] for part in image_parts] == [2]
    assert {"type": "text", "text": "remember"} in processed[1]["content"]


def test_zero_image_limit_removes_images_but_keeps_narration() -> None:
    messages = [_user("task", 0), _assistant("remember", "call_0")]

    processed = Holo4HistoryProtocol(max_n_images=0).process_messages(messages)

    assert processed == [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "remember"}],
            "reasoning_content": "private remember",
            "tool_calls": messages[1]["tool_calls"],
        },
    ]


def test_negative_image_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Holo4HistoryProtocol(max_n_images=-1).process_messages([])
