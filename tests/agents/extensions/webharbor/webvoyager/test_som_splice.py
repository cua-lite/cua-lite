"""WebVoyager SoM turn-0 model_web_text splice + qwen3_vl/qwen3_5 parity.

The SoM model_web_text splice was re-homed from a shared ``qwen3_vl.history`` flag to
``lite/agents/extensions/webharbor/webvoyager/protocol.py`` precisely because the flag
silently no-op'd on ``qwen3_5.history``, desyncing the mirrored SoM pair. These
tests lock the guarantee that BOTH family protocols splice the turn-0
``metadata['model_web_text']`` identically, so a future
change to either family's ``_inject_text`` / ``set_or_append_text`` can't silently
re-introduce the desync. Raw ``metadata['web_text']`` remains a log/debug field
and must not be spliced into model-visible prompt text.

Run:
    uv run pytest tests/agents/extensions/webharbor/webvoyager/test_som_splice.py -v
"""

from __future__ import annotations

import pytest

from lite.agents.extensions.webharbor.webvoyager.protocol import (
    WebVoyagerQwen3_5HistoryProtocol,
    WebVoyagerQwen3VLHistoryProtocol,
    _with_initial_web_text,
)
from lite.core.tools import make_tool_call

_WEB_TEXT = "[1] <button> Search @ (10, 20);\t[2] <input> Query @ (30, 40)"
_MODEL_WEB_TEXT = 'DOM:\n[1] <button> Search;\t[2] <input> Query'
_NEXT_WEB_TEXT = "[3] <a> Product @ (50, 60)"
_NEXT_MODEL_WEB_TEXT = "DOM:\n[3] <a> Product"

_BOTH = (WebVoyagerQwen3VLHistoryProtocol, WebVoyagerQwen3_5HistoryProtocol)


def _user_with_metadata(
    web_text: str | None,
    *,
    model_web_text: str | None = None,
) -> dict:
    data = {}
    if web_text is not None:
        data["web_text"] = web_text
    if model_web_text is not None:
        data["model_web_text"] = model_web_text
    return {
        "role": "user",
        "content": [
            {"type": "metadata", "data": data},
            {"type": "text", "text": "Find the search box and search for jackets."},
            {"type": "image", "image": "data:image/png;base64,Zm9v"},
        ],
    }


def _assistant(call_id: str = "call_0") -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "click search"}],
        "tool_calls": [
            make_tool_call("click", {"coordinate": [10, 20]}, call_id=call_id),
        ],
    }


def _tool_with_metadata(
    web_text: str | None,
    *,
    model_web_text: str | None = None,
) -> dict:
    data = {"url": "https://example.test/results"}
    if web_text is not None:
        data["web_text"] = web_text
    if model_web_text is not None:
        data["model_web_text"] = model_web_text
    return {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": [
            {"type": "image", "index": 1},
            {"type": "text", "text": "Current page."},
            {"type": "metadata", "data": data},
        ],
    }


def _windowed_instance(cls):
    if cls is WebVoyagerQwen3VLHistoryProtocol:
        return cls(full_history_size=1)
    return cls(history_n=1)


def _first_user_text(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            texts = [c["text"] for c in m["content"] if c.get("type") == "text"]
            return texts[-1] if texts else ""
    return ""


def _first_user_texts(messages: list[dict]) -> list[str]:
    for m in messages:
        if m.get("role") == "user":
            return [c["text"] for c in m["content"] if c.get("type") == "text"]
    return []


# ---------------------------------------------------------------------------
# _with_initial_web_text helper
# ---------------------------------------------------------------------------

def test_helper_appends_model_web_text():
    msg = _user_with_metadata(_WEB_TEXT, model_web_text=_MODEL_WEB_TEXT)
    assert _with_initial_web_text(msg, "PROMPT") == f"PROMPT\n{_MODEL_WEB_TEXT}"


def test_helper_noop_when_model_web_text_empty():
    assert _with_initial_web_text(_user_with_metadata("", model_web_text=""), "PROMPT") == "PROMPT"


def test_helper_ignores_raw_web_text_without_model_web_text():
    msg = _user_with_metadata(_WEB_TEXT)
    assert _with_initial_web_text(msg, "PROMPT") == "PROMPT"


def test_helper_noop_when_no_metadata():
    msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    assert _with_initial_web_text(msg, "PROMPT") == "PROMPT"


def test_helper_scans_later_metadata_items():
    msg = {
        "role": "user",
        "content": [
            {"type": "metadata", "data": {"url": "https://example.test"}},
            {"type": "metadata", "data": {"model_web_text": _MODEL_WEB_TEXT}},
            {"type": "text", "text": "hi"},
        ],
    }
    assert _with_initial_web_text(msg, "PROMPT") == f"PROMPT\n{_MODEL_WEB_TEXT}"


# ---------------------------------------------------------------------------
# _inject_text override — parity across the qwen3_vl / qwen3_5 pair (D10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", _BOTH)
def test_inject_text_splices_model_web_text(cls):
    proto = cls()
    msg = _user_with_metadata(_WEB_TEXT, model_web_text=_MODEL_WEB_TEXT)
    proto._inject_text(msg, "PROMPT")
    assert _first_user_text([msg]) == f"PROMPT\n{_MODEL_WEB_TEXT}"


def test_inject_text_identical_across_pair():
    """The mirrored SoM pair must produce byte-identical turn-0 injected text."""
    outs = []
    for cls in _BOTH:
        msg = _user_with_metadata(_WEB_TEXT, model_web_text=_MODEL_WEB_TEXT)
        cls()._inject_text(msg, "PROMPT")
        outs.append(_first_user_text([msg]))
    assert outs[0] == outs[1]


@pytest.mark.parametrize("cls", _BOTH)
def test_inject_text_noop_without_model_web_text(cls):
    msg = _user_with_metadata(None)
    cls()._inject_text(msg, "PROMPT")
    assert _first_user_text([msg]) == "PROMPT"


# ---------------------------------------------------------------------------
# Full process_messages path (turn-0 in-window) — the real injection site
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", _BOTH)
def test_process_messages_injects_model_web_text_on_turn0(cls):
    proto = cls()
    out = proto.process_messages([
        _user_with_metadata(_WEB_TEXT, model_web_text=_MODEL_WEB_TEXT)
    ])
    text = _first_user_text(out)
    assert _MODEL_WEB_TEXT in text, f"{cls.__name__} did not splice model_web_text: {text!r}"
    assert _WEB_TEXT not in text


def test_process_messages_parity_across_pair():
    msgs = [_user_with_metadata(_WEB_TEXT, model_web_text=_MODEL_WEB_TEXT)]
    a = _first_user_text(WebVoyagerQwen3VLHistoryProtocol().process_messages(msgs))
    b = _first_user_text(WebVoyagerQwen3_5HistoryProtocol().process_messages(msgs))
    assert _MODEL_WEB_TEXT in a and _MODEL_WEB_TEXT in b
    assert _WEB_TEXT not in a and _WEB_TEXT not in b


@pytest.mark.parametrize("cls", _BOTH)
def test_process_messages_splices_model_web_text_from_boundary_role_tool(cls):
    messages = [
        _user_with_metadata(None),
        _assistant(),
        _tool_with_metadata(_NEXT_WEB_TEXT, model_web_text=_NEXT_MODEL_WEB_TEXT),
    ]

    out = _windowed_instance(cls).process_messages(messages)
    first_user = next(message for message in out if message["role"] == "user")
    texts = _first_user_texts(out)

    assert first_user.get("tool_call_id") is None
    assert any(_NEXT_MODEL_WEB_TEXT in text for text in texts)
    assert not any(_NEXT_WEB_TEXT in text for text in texts)
    assert "Current page." in texts
    assert any(
        item.get("type") == "metadata"
        and item.get("data", {}).get("web_text") == _NEXT_WEB_TEXT
        and item.get("data", {}).get("model_web_text") == _NEXT_MODEL_WEB_TEXT
        for item in first_user["content"]
    )


@pytest.mark.parametrize("cls", _BOTH)
def test_windowed_role_tool_observation_preserves_metadata_by_default(cls):
    messages = [
        _user_with_metadata(_WEB_TEXT, model_web_text=_MODEL_WEB_TEXT),
        _assistant(),
        _tool_with_metadata(_NEXT_WEB_TEXT, model_web_text=_NEXT_MODEL_WEB_TEXT),
    ]

    out = cls().process_messages(messages)
    tool_message = next(message for message in out if message["role"] == "tool")

    assert [item["type"] for item in tool_message["content"]] == [
        "image",
        "text",
        "metadata",
    ]
    assert {"type": "text", "text": "Current page."} in tool_message["content"]
    assert tool_message["content"][2]["data"]["web_text"] == _NEXT_WEB_TEXT
    assert tool_message["content"][2]["data"]["model_web_text"] == _NEXT_MODEL_WEB_TEXT


def test_qwen3_5_collapsed_role_tool_error_preserves_projected_text():
    error_text = "## Error from previous action:\ninvalid action: screenshot"
    messages = [
        _user_with_metadata(_WEB_TEXT, model_web_text=_MODEL_WEB_TEXT),
        _assistant("call_0"),
        _tool_with_metadata(_NEXT_WEB_TEXT, model_web_text=_NEXT_MODEL_WEB_TEXT),
        _assistant("call_1"),
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "image", "index": 2},
                {"type": "text", "text": error_text},
                {"type": "metadata", "data": {"is_error": True}},
            ],
        },
        _assistant("call_2"),
        _tool_with_metadata(None),
    ]

    out = WebVoyagerQwen3_5HistoryProtocol(
        history_n=10,
        image_max=1,
        fold_size=10,
        keep_text_with_images=True,
    ).process_messages(messages)

    tool_message = next(
        message for message in out
        if message.get("role") == "tool" and message.get("tool_call_id") == "call_1"
    )
    texts = [item["text"] for item in tool_message["content"] if item.get("type") == "text"]
    assert texts == [
        "This screenshot has been collapsed.",
        error_text,
    ]
    assert {"type": "metadata", "data": {"is_error": True}} in tool_message["content"]


def test_qwen3_5_collapsed_regular_user_observation_keeps_text_when_configured():
    messages = [
        _user_with_metadata(_WEB_TEXT, model_web_text=_MODEL_WEB_TEXT),
        _assistant("call_0"),
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 1},
                {"type": "text", "text": "Second page text."},
                {
                    "type": "metadata",
                    "data": {
                        "web_text": _NEXT_WEB_TEXT,
                        "model_web_text": _NEXT_MODEL_WEB_TEXT,
                    },
                },
            ],
        },
        _assistant("call_1"),
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 2},
                {"type": "text", "text": "Latest page text."},
            ],
        },
    ]

    out = WebVoyagerQwen3_5HistoryProtocol(
        history_n=10,
        image_max=1,
        fold_size=10,
        keep_text_with_images=True,
    ).process_messages(messages)

    collapsed_user = next(
        message for message in out
        if any(
            item.get("type") == "text" and item.get("text") == "Second page text."
            for item in message.get("content", [])
        )
    )
    texts = [item["text"] for item in collapsed_user["content"] if item.get("type") == "text"]
    assert texts == [
        "This screenshot has been collapsed.",
        "Second page text.",
    ]
    assert {
        "type": "metadata",
        "data": {
            "web_text": _NEXT_WEB_TEXT,
            "model_web_text": _NEXT_MODEL_WEB_TEXT,
        },
    } in collapsed_user["content"]
