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


def test_compaction_keeps_instruction_and_old_notes_but_drops_old_tool_state() -> None:
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
    assert compacted[:4] == [
        messages[0],
        {"role": "user", "content": [{"type": "text", "text": "task instruction"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "remember alpha"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "remember beta"}]},
    ]
    assert compacted[-2:] == [messages[-2], messages[-1]]
    assert "old tool output" not in str(compacted)
    assert "private" not in str(compacted)
    assert "call_0" not in str(compacted)


def test_short_history_drops_private_assistant_sidecars() -> None:
    messages = [_user("task", 0), _assistant("note", "call_0")]

    processed = Holo4HistoryProtocol(max_n_images=4).process_messages(messages)

    assert processed[0] == messages[0]
    assert processed[1]["content"] == messages[1]["content"]
    assert processed[1]["tool_calls"] == messages[1]["tool_calls"]
    assert "reasoning_content" not in processed[1]
    assert "raw_response" not in processed[1]


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
            "tool_calls": messages[1]["tool_calls"],
        },
    ]


def test_negative_image_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Holo4HistoryProtocol(max_n_images=-1).process_messages([])
