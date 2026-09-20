"""WebVoyager protocol render goldens."""

from __future__ import annotations

import os
from pathlib import Path
from pprint import pformat

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.extensions.webharbor.webvoyager.protocol import (
    WebVoyagerQwen3_5HistoryProtocol,
    WebVoyagerQwen3VLHistoryProtocol,
)

register_all()

_GOLDEN_DIR = Path(__file__).parent / "_protocol_goldens"
_UPDATE = os.environ.get("UPDATE_BROWSER_GOLDENS") == "1"


def _sys() -> dict:
    return {"role": "system", "content": [{"type": "text", "text": "You are a web agent."}]}


def _webvoyager_som_traj() -> list[dict]:
    return [
        _sys(),
        {
            "role": "user",
            "content": [
                {
                    "type": "metadata",
                    "data": {
                        "web_text": (
                            "[1] <button> Search @ (10, 20);\t"
                            "[2] <input> Query @ (30, 40)"
                        ),
                        "model_web_text": "DOM:\n[1] <button> Search;\t[2] <input> Query",
                    },
                },
                {"type": "text", "text": "Find and search for jackets."},
                {"type": "image", "index": 0},
            ],
        },
    ]


def _render_protocol(proto, messages: list[dict]) -> str:
    return pformat(proto.process_messages(messages), sort_dicts=False, width=100)


_CASES = {
    "webvoyager_qwen3_vl__som": lambda: _render_protocol(
        WebVoyagerQwen3VLHistoryProtocol(),
        _webvoyager_som_traj(),
    ),
    "webvoyager_qwen3_5__som": lambda: _render_protocol(
        WebVoyagerQwen3_5HistoryProtocol(),
        _webvoyager_som_traj(),
    ),
}


@pytest.mark.parametrize("case_id", list(_CASES), ids=lambda v: str(v))
def test_webvoyager_protocol_render_golden(case_id: str) -> None:
    rendered = _CASES[case_id]()
    assert not ("0x" in rendered and "Image" in rendered)

    path = _GOLDEN_DIR / f"{case_id}.txt"
    if _UPDATE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        pytest.skip(f"regenerated golden {path.name}")

    assert path.exists(), f"missing golden {path} - regenerate with UPDATE_BROWSER_GOLDENS=1"
    assert rendered == path.read_text()
