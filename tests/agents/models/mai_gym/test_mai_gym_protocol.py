"""MAI-Gym window-wise history protocol."""

from __future__ import annotations

import pytest

from lite.agents.models.mai_gym.protocol import HISTORY_IMAGE_PLACEHOLDER, MAIGymHistoryProtocol
from lite.core.tools.calls import make_tool_call


def _trajectory(n_steps: int) -> list[dict]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "index": 0}, {"type": "text", "text": "task"}],
        }
    ]
    for i in range(n_steps):
        messages.append({"role": "assistant", "content": [], "tool_calls": []})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i:04d}",
                "content": [{"type": "image", "index": i + 1}],
            }
        )
    return messages


def _signature(messages: list[dict]) -> list[str]:
    out = []
    for message in messages:
        if message["role"] == "assistant":
            out.append("A")
            continue
        assert message["role"] == "user"
        (part,) = message["content"]
        out.append(f"img{part['index']}" if part["type"] == "image" else part["text"])
    return out


def test_window_start_matches_upstream_sawtooth() -> None:
    # Upstream ``window_start`` with W=10, O=2.
    assert [MAIGymHistoryProtocol().window_start(i) for i in range(28)] == (
        [0] * 10 + [8] * 8 + [16] * 8 + [24] * 2
    )


def test_first_user_message_splits_into_instruction_then_image_turns() -> None:
    assert _signature(MAIGymHistoryProtocol().process_messages(_trajectory(0))) == ["task", "img0"]


def test_screenshots_before_the_window_collapse_and_text_survives() -> None:
    rendered = _signature(MAIGymHistoryProtocol().process_messages(_trajectory(10)))
    # Turn 11 (10 prior steps) starts a new window: screenshots 0-7 collapse.
    assert rendered[:2] == ["task", HISTORY_IMAGE_PLACEHOLDER]
    assert rendered.count(HISTORY_IMAGE_PLACEHOLDER) == 8
    assert rendered[-5:] == ["img8", "A", "img9", "A", "img10"]
    assert rendered.count("A") == 10


def test_inside_a_window_the_prompt_only_grows() -> None:
    proto = MAIGymHistoryProtocol()
    for n in range(10, 17):
        shorter = _signature(proto.process_messages(_trajectory(n)))
        longer = _signature(proto.process_messages(_trajectory(n + 1)))
        assert longer[: len(shorter)] == shorter


def test_tool_text_becomes_its_own_user_turn_before_the_screenshot() -> None:
    messages = _trajectory(1)
    messages[-1]["content"].append({"type": "text", "text": 'The user replied: "blue"'})
    messages[-1]["content"].append({"type": "metadata", "data": {"k": 1}})
    rendered = MAIGymHistoryProtocol().process_messages(messages)
    assert rendered[-2] == {
        "role": "user",
        "content": [
            {"type": "text", "text": 'The user replied: "blue"'},
            {"type": "metadata", "data": {"k": 1}},
        ],
    }
    assert rendered[-1] == {"role": "user", "content": [{"type": "image", "index": 1}]}


def test_ask_user_reply_is_sent_without_its_screenshot() -> None:
    # Upstream appends ``ask_user_response`` as a text turn and skips that
    # step's screenshot, so the screenshot ordinals skip it too.
    messages = _trajectory(2)
    messages[1]["tool_calls"] = [
        make_tool_call("ask_user", {"text": "Color?"}, call_id="call_0000")
    ]
    messages[2]["content"].append({"type": "text", "text": 'The user replied: "blue"'})
    assert _signature(MAIGymHistoryProtocol().process_messages(messages)) == [
        "task", "img0", "A", 'The user replied: "blue"', "A", "img2",
    ]


def test_metadata_only_feedback_adds_no_empty_turn() -> None:
    messages = _trajectory(1)
    messages[-1]["content"].append({"type": "metadata", "data": {"k": 1}})
    rendered = MAIGymHistoryProtocol().process_messages(messages)
    assert rendered[-1] == {
        "role": "user",
        "content": [{"type": "image", "index": 1}, {"type": "metadata", "data": {"k": 1}}],
    }


def test_input_is_not_mutated() -> None:
    messages = _trajectory(12)
    before = repr(messages)
    MAIGymHistoryProtocol().process_messages(messages)
    assert repr(messages) == before


def test_window_must_advance() -> None:
    with pytest.raises(ValueError, match="window_size - overlap"):
        MAIGymHistoryProtocol(window_size=2, overlap=2)
