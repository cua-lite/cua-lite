"""The replay sidecar must not double the reasoning when ``<think>`` is open.

``Qwen3VLBaseAgent.build_generation_prompt`` ends the prompt at an OPEN
``<think>`` when thinking is on, so the model's reply is the SECOND HALF of that
block: ``reasoning </think> answer``. The chat template rebuilds the block from
``reasoning_content`` when it renders the turn as history, so the sidecar must
carry only the ``content`` half -- otherwise history shows an empty block, the
reasoning outside it, and a ``</think>`` with no opening tag.

Only families that park the reasoning in ``reasoning_content`` need the trim.
``ui_venus_2`` keeps it as an ``inline_reasoning`` CONTENT part and ships a
chat template that splits ``</think>`` out of ``content`` itself, so its verbatim
replay is already correct -- the last test pins that it stays untouched.
"""

from __future__ import annotations

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.models.qwen3_5.agent import Qwen3_5DesktopUseAgent
from lite.agents.models.qwen3_8.agent import Qwen3_8DesktopUseAgent
from lite.agents.models.ui_venus_2.agent import UIVenus2DesktopUseAgent

register_all()

REASONING = "The terminal closed. I should reopen it from the sidebar."
ANSWER = "Reopening the terminal from the sidebar."

#: What each family's model emits after the prompt's open ``<think>``.
REPLIES = {
    Qwen3_5DesktopUseAgent: (
        f"{REASONING}\n</think>\n\n{ANSWER}\n"
        "<tool_call>\n<function=computer_use>\n<parameter=action>\nleft_click\n"
        "</parameter>\n<parameter=coordinate>\n[18, 559]\n</parameter>\n"
        "</function>\n</tool_call>"
    ),
    Qwen3_8DesktopUseAgent: (
        f"{REASONING}\n</think>\n\n{ANSWER}\n"
        "<tool_call>\n<function=computer_use>\n<parameter=action>\nleft_click\n"
        "</parameter>\n<parameter=coordinate>\n[18, 559]\n</parameter>\n"
        "</function>\n</tool_call>"
    ),
}

#: ``ui_venus_2`` carries reasoning as a content part, not ``reasoning_content``.
UI_VENUS_REPLY = (
    f"{REASONING}\n</think>\n\n{ANSWER}\n<action>Click(box=(18, 559))</action>"
)


async def _gen(**_):
    return {"response": ""}


def _agent(cls):
    return cls(generate_fn=_gen, processor=None, kwargs={"enable_thinking": True})


@pytest.mark.parametrize("cls", list(REPLIES))
def test_sidecar_drops_the_prompt_supplied_think_half(cls) -> None:
    """The sidecar keeps the answer, not the reasoning the template re-emits."""
    agent = _agent(cls)
    _, lite, err = agent._parse_generation_response(REPLIES[cls], call_id_start=0)

    assert err is None
    assert lite["reasoning_content"] == REASONING
    raw = lite["raw_response"]["text"]
    assert REASONING not in raw, "reasoning would be rendered twice"
    assert "</think>" not in raw, "an orphan closing tag would reach history"
    assert raw.startswith(ANSWER)


@pytest.mark.parametrize("cls", list(REPLIES))
def test_sidecar_still_carries_the_action_markup_verbatim(cls) -> None:
    """Trimming the think half must not disturb byte-exact tool-call replay."""
    reply = REPLIES[cls]
    agent = _agent(cls)
    _, lite, _ = agent._parse_generation_response(reply, call_id_start=0)
    assert lite["raw_response"]["text"] == reply.split("</think>", 1)[-1].lstrip("\n")


@pytest.mark.parametrize("cls", list(REPLIES))
def test_a_reply_with_no_open_think_is_untouched(cls) -> None:
    """Thinking off: the reply carries no tag, so the sidecar is the whole reply."""
    reply = REPLIES[cls].split("</think>", 1)[-1].lstrip("\n")
    agent = cls(generate_fn=_gen, processor=None, kwargs={"enable_thinking": False})
    _, lite, _ = agent._parse_generation_response(reply, call_id_start=0)
    assert not lite.get("reasoning_content")
    assert lite["raw_response"]["text"] == reply


def test_ui_venus_2_replay_stays_verbatim() -> None:
    """Its template extracts ``</think>`` from ``content``, so the whole reply
    must survive -- trimming it would strip the reasoning the template needs."""
    agent = _agent(UIVenus2DesktopUseAgent)
    _, lite, _ = agent._parse_generation_response(UI_VENUS_REPLY, call_id_start=0)

    assert not lite.get("reasoning_content"), "reasoning rides a content part here"
    assert [p["type"] for p in lite["content"]] == ["inline_reasoning"]
    assert lite["raw_response"]["text"] == UI_VENUS_REPLY
