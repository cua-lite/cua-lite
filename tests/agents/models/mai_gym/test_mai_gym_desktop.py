"""MAI-Gym desktop (computer use): upstream ``MAIGymComputerUsePolicy`` alignment."""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import pytest
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent import AdapterBasedAgent, AgentRegistry
from lite.agents.core.protocol import ProtocolRegistry
from lite.agents.models.mai_gym.action_space import (
    MAIGymDesktopActionSpace,
    pyautogui_code_to_lite,
    render_desktop_block,
)
from lite.agents.models.mai_gym.adapter import (
    MAIGymDesktopUseAdapter,
    parse_mai_gym_desktop_reply,
    render_mai_gym_desktop_retry_feedback,
    render_mai_gym_desktop_sys_prompt,
)
from lite.agents.models.mai_gym.protocol import (
    DESKTOP_INSTRUCTION_TEMPLATE,
    HISTORY_IMAGE_PLACEHOLDER,
)
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools.action_space import LiteDesktopActionSet
from lite.core.tools.calls import make_tool_call
from lite.core.tools.extra_tools import LiteFinishToolSet, LiteShellToolSet

register_all()


def _metadata(bash: bool) -> LiteCUAMetadata:
    schemas = [LiteFinishToolSet.get_tool_schema(n) for n in ("response", "terminate")]
    if bash:
        schemas.append(LiteShellToolSet.get_tool_schema("bash"))
    return LiteCUAMetadata(dims=("desktop", "use"), extra_tool_schemas=schemas)


def _adapter(bash: bool = False) -> MAIGymDesktopUseAdapter:
    return MAIGymDesktopUseAdapter(metadata=_metadata(bash), password="password")


def _children(calls: list[dict]) -> list[dict]:
    assert [c["function"]["name"] for c in calls] == ["computer"]
    return calls[0]["function"]["arguments"]["actions"]


# -- prompt -------------------------------------------------------------------


def test_bash_prompt_is_byte_exact_to_the_upstream_render() -> None:
    """Digest of upstream ``_render_system_prompt(password="password",
    screen_size=(1920, 1080), range_scale=1000, enable_ask_user=False)``."""
    assert (
        hashlib.sha256(
            render_mai_gym_desktop_sys_prompt("password", bash=True).encode()
        ).hexdigest()
        == "dec75b3a158024c412f0591840de22e7106e6e3a082a3a56620c44a1cddcbaa4"
    )


def test_prompt_cuts_every_bash_passage_without_a_bash_tool() -> None:
    gui_only = _adapter(bash=False).system_prompt
    assert "bash" not in gui_only.lower()
    assert "The computer password is password." in gui_only
    assert "## GUI actions" in gui_only and "## Flow-control actions" in gui_only
    assert _adapter(bash=True).system_prompt == render_mai_gym_desktop_sys_prompt(
        "password", bash=True
    )


# -- parsing (port of parse_action_response) ----------------------------------


def test_parse_keeps_reasoning_and_accepted_blocks_in_order() -> None:
    reasoning, calls, history, error = parse_mai_gym_desktop_reply(
        "Open it.\n</think>\nprose <tool_call>pyautogui.click(100, 200)</tool_call>\n"
        "<tool_call>os.system('x')</tool_call>\n<bash>\nls\n</bash>"
    )
    assert reasoning == "Open it."
    assert error is None
    assert [c["name"] for c in calls] == ["pyautogui", "bash"]
    # The unsafe block is skipped and absent from the replayed history.
    assert history == "<tool_call>pyautogui.click(100, 200)</tool_call>\n<bash>\nls\n</bash>"


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ("r</think>no action", "no_executable_action"),
        ("r</think><tool_call>computer.bash('ls')</tool_call>", "legacy_computer_bash_is_disabled"),
        ("r</think><tool_call>computer.ask_user(question='q')</tool_call>", "ask_user_is_disabled"),
    ],
)
def test_unparseable_replies_name_upstream_errors(reply: str, error: str) -> None:
    assert parse_mai_gym_desktop_reply(reply)[3] == error


def test_bash_wrapped_in_tool_call_is_unwrapped_like_upstream() -> None:
    _, calls, history, _ = parse_mai_gym_desktop_reply(
        "r</think><tool_call><bash>echo 1</bash></tool_call>"
    )
    assert calls == [{"name": "bash", "arguments": {"command": "echo 1"}}]
    assert history == "<bash>echo 1</bash>"


def test_unparseable_reply_keeps_its_visible_prose_as_content() -> None:
    parsed = _adapter().parse_raw_assistant_response("r\n</think>\nThe answer is 42.")
    assert parsed["content"] == [{"type": "text", "text": "The answer is 42."}]
    assert parsed["reasoning_content"] == "r"
    assert "tool_calls" not in parsed


def test_env_extra_tool_renders_as_a_json_tool_call_block() -> None:
    call = make_tool_call("goto", {"url": "http://example.com"})
    (agent_call,) = MAIGymDesktopActionSpace().convert_tool_calls_to_agent([call])
    assert render_desktop_block(agent_call) == (
        '<tool_call>{"name": "goto", "arguments": {"url": "http://example.com"}}</tool_call>'
    )


# -- action conversion ---------------------------------------------------------


def test_pyautogui_calls_convert_to_canonical_computer_actions() -> None:
    children = _children(
        MAIGymDesktopActionSpace().convert_tool_calls_from_agent(
            [
                {"name": "pyautogui", "arguments": {"code": code}}
                for code in (
                    "pyautogui.click(500, 300)",
                    "pyautogui.rightClick(10, 20)",
                    "pyautogui.tripleClick(1, 2)",
                    "pyautogui.dragTo(800, 600, duration=0.5)",
                    "pyautogui.scroll(-5, 500, 500)",
                    "pyautogui.hotkey('ctrl', 'c')",
                    "pyautogui.press('enter', presses=2)",
                    "pyautogui.typewrite('hi')",
                    "pyautogui.keyDown('shift')",
                )
            ]
        )
    )
    # Canonical constructors omit default arguments (left button, one click).
    assert children == [
        {"action": "click", "coordinate": [500, 300]},
        {"action": "click", "coordinate": [10, 20], "button": "right"},
        {"action": "click", "coordinate": [1, 2], "clicks": 3},
        {"action": "drag", "coordinate": [800, 600]},
        {"action": "scroll", "coordinate": [500, 500], "direction": "down", "amount": 5},
        {"action": "key", "keys": ["ctrl", "c"]},
        {"action": "key", "keys": ["enter"]},
        {"action": "key", "keys": ["enter"]},
        {"action": "type", "text": "hi"},
        {"action": "key_down", "keys": ["shift"]},
    ]


def test_unsafe_pyautogui_code_is_a_parse_error() -> None:
    with pytest.raises(ModelToolCallParseError):
        pyautogui_code_to_lite("import os; os.system('x')")


def test_flow_control_maps_to_finish_tools() -> None:
    space = MAIGymDesktopActionSpace()
    convert = space.convert_tool_calls_from_agent
    assert convert(
        [{"name": "computer.terminate", "arguments": {"status": "success", "answer": "42"}}]
    ) == [LiteFinishToolSet.response(text="42")]
    # A failure finish is osworld's FAIL verdict even when it carries an answer.
    assert convert(
        [{"name": "computer.terminate", "arguments": {"status": "failure", "answer": "no"}}]
    ) == [LiteFinishToolSet.terminate(status="failure")]
    assert _children(convert([{"name": "computer.wait", "arguments": {}}])) == [
        {"action": "wait", "duration": 1.0}
    ]


def test_canonical_calls_render_back_to_blocks() -> None:
    adapter = _adapter(bash=True)
    message = {
        "role": "assistant",
        "reasoning_content": "r",
        "content": [],
        "tool_calls": [
            {**LiteDesktopActionSet.click(coordinate=[1, 2], clicks=2), "id": "call_0000"},
            {**LiteFinishToolSet.terminate(status="failure"), "id": "call_0001"},
        ],
    }
    rendered = adapter._convert_message_to_agent(message)
    assert rendered["content"] == [
        {
            "type": "text",
            "text": "<tool_call>pyautogui.doubleClick(1, 2)</tool_call>\n"
            '<tool_call>computer.terminate(status="failure")</tool_call>',
        }
    ]
    assert rendered["reasoning_content"] == "r"


# -- history layout -------------------------------------------------------------


def _step(i: int, *, bash: bool = False) -> list[dict[str, Any]]:
    call_id = f"call_{i:04d}"
    call = (
        {**LiteShellToolSet.bash(command="ls"), "id": call_id}
        if bash
        else {**LiteDesktopActionSet.click(coordinate=[i, i]), "id": call_id}
    )
    observation: list[dict[str, Any]] = [{"type": "image", "index": i + 1}]
    if bash:
        observation += [
            {"type": "text", "text": "a.txt\n[exit 2]"},
            {"type": "metadata", "data": {"returncode": 2}},
        ]
    return [
        {"role": "assistant", "content": [], "reasoning_content": f"r{i}", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call_id, "content": observation},
    ]


def _sample(n_steps: int, bash_at: int | None = None) -> LiteSample:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "index": 0}, {"type": "text", "text": "Do X"}],
        }
    ]
    for i in range(n_steps):
        messages += _step(i, bash=(i == bash_at))
    return LiteSample(metadata=_metadata(bash=True), messages=messages)


def test_one_user_message_per_step_with_instruction_and_bash_output() -> None:
    rendered = _adapter(bash=True).render_step(_sample(2, bash_at=1), 3, [None] * 3)
    assert [m["role"] for m in rendered] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert rendered[1]["content"] == [
        {"type": "text", "text": DESKTOP_INSTRUCTION_TEMPLATE.format(instruction="Do X")},
        {"type": "image", "index": 0},
    ]
    assert rendered[5]["content"] == [
        {"type": "text", "text": "## Bash Output\nreturncode=2\n\nstdout:\na.txt"},
        {"type": "image", "index": 2},
    ]


def test_window_collapses_screenshots_inside_their_step_message() -> None:
    rendered = _adapter().render_step(_sample(10), 11, [None] * 11)
    users = [m for m in rendered if m["role"] == "user"]
    # Turn 11 (10 prior steps): screenshots 0-7 collapse, 8-10 stay.
    assert users[0]["content"][-1] == {"type": "text", "text": HISTORY_IMAGE_PLACEHOLDER}
    assert users[0]["content"][0]["text"].startswith("# Task Instruction:")
    assert sum(u["content"][-1].get("type") == "image" for u in users) == 3


# -- agent ------------------------------------------------------------------------


class _Processor:
    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return f"PROMPT[{len(messages)}]"


def _agent(replies: list[str], seen: list[tuple[str, list[dict]]], **agent_kwargs: Any):
    replies_iter = iter(replies)

    async def generate(prompt: str, images: list, messages: list[dict]) -> dict[str, Any]:
        seen.append((prompt, messages))
        return {"response": next(replies_iter), "finish_reason": "stop"}

    return AgentRegistry.get(
        "mai_gym@desktop@use",
        processor=_Processor(),
        generate_fn=generate,
        metadata=_metadata(bash=False),
        password="password",
        **agent_kwargs,
    )


def _first_turn() -> LiteSample:
    return LiteSample(
        metadata=_metadata(bash=False),
        messages=[
            {
                "role": "user",
                "content": [{"type": "image", "index": 0}, {"type": "text", "text": "t"}],
            }
        ],
    )


async def test_history_replays_the_reply_verbatim() -> None:
    """``token_native_history``: prose around the blocks replays too."""
    seen: list[tuple[str, list[dict]]] = []
    agent = _agent(["ok\n</think>\nClick it.\n<tool_call>pyautogui.click(1, 2)</tool_call>"], seen)
    result = await agent._predict_with_details(
        _first_turn(), processed_images=[Image.new("RGB", (8, 8))]
    )
    assert len(seen) == 1
    assert result.model_output_error is None
    assert result.lite_message["raw_response"] == {
        "text": "Click it.\n<tool_call>pyautogui.click(1, 2)</tool_call>",
        "adapter_key": "mai_gym@desktop@use",
        "reasoning_content": "ok",
    }


async def test_unparseable_reply_is_retried_with_upstream_correction_turn() -> None:
    """Upstream computer-use retry: rendered turn + failed reply + correction.

    Only the latest failure rides along (never accumulated), and the rescued
    turn's RL step records the prompt it was generated from.
    """
    seen: list[tuple[str, list[dict]]] = []
    replies = [
        "still planning the click",  # rumination: think never closed
        "bad\n</think>\nno action",
        "ok\n</think>\n<tool_call>pyautogui.click(1, 2)</tool_call>",
    ]
    agent = _agent(replies, seen)
    result = await agent._predict_with_details(
        _first_turn(), processed_images=[Image.new("RGB", (8, 8))]
    )
    assert len(seen) == 3  # two failed attempts, then the rescued one
    base = seen[0][1]
    for attempt, failed in ((1, replies[0]), (2, replies[1])):
        feedback = render_mai_gym_desktop_retry_feedback(failed, bash=False)
        assert seen[attempt][1] == [
            *base,
            {"role": "assistant", "content": [{"type": "text", "text": failed}]},
            {"role": "user", "content": [{"type": "text", "text": feedback}]},
        ]
        assert seen[attempt][0] == f"PROMPT[{len(base) + 2}]"
    assert result.step.prompt == seen[2][0]
    assert result.model_output_error is None
    assert result.lite_message["reasoning_content"] == "ok"
    # Failed attempts consumed no call ids: the rescued turn stamps from the start.
    assert [c["id"] for c in result.lite_message["tool_calls"]] == ["call_0000"]


def test_retry_feedback_is_upstream_text_with_bash_cut_when_bash_is_off() -> None:
    reply = "  bad\n</think>\nno action  "
    assert render_mai_gym_desktop_retry_feedback(reply, bash=True) == (
        "Your previous response could not be parsed into an executable action.\n"
        "Use raw <bash>...</bash>, or one supported pyautogui/wait/terminate "
        "call inside <tool_call>...</tool_call>. Never call computer.bash.\n\n"
        "Unparseable previous response excerpt:\nbad\n</think>\nno action"
    )
    no_bash = render_mai_gym_desktop_retry_feedback(reply, bash=False)
    assert "<bash>" not in no_bash
    assert "Use one supported pyautogui/wait/terminate call" in no_bash
    long_reply = "x" * 2001
    assert render_mai_gym_desktop_retry_feedback(long_reply, bash=True).endswith(
        "x" * 2000 + "\n... [truncated]"
    )


async def test_desktop_retry_result_matches_the_base_predict_path() -> None:
    """``_regenerate`` copies base generate/parse/step; only the prompt may differ."""
    reply = "ok\n</think>\n<tool_call>pyautogui.click(1, 2)</tool_call>"
    gen = {
        "response": reply,
        "finish_reason": "length",
        "response_tokens": [1, 2],
        "response_log_probs": [-0.5, -0.25],
    }

    async def generate(**_: Any) -> dict[str, Any]:
        return dict(gen)

    agent = AgentRegistry.get(
        "mai_gym@desktop@use",
        processor=_Processor(),
        generate_fn=generate,
        metadata=_metadata(bash=False),
        password="password",
    )
    images = [Image.new("RGB", (8, 8))]
    base = await AdapterBasedAgent._predict_with_details(
        agent, _first_turn(), processed_images=images, call_id_start=5
    )
    failed = dataclasses.replace(base, step=dataclasses.replace(base.step, response="bad"))
    retried = await agent._regenerate(
        _first_turn(), processed_images=images, call_id_start=5, failed=failed
    )
    assert retried.step.prompt != base.step.prompt
    assert dataclasses.replace(retried.step, prompt=base.step.prompt) == base.step
    assert retried.lite_message == base.lite_message
    assert retried.agent_message == base.agent_message
    assert retried.model_output_error == base.model_output_error


async def test_unparseable_reply_takes_the_shared_parse_failure_final() -> None:
    """Retries exhausted: the last failed attempt still takes the shared final."""
    seen: list[tuple[str, list[dict]]] = []
    agent = _agent(["bad\n</think>\nno action"] * 4, seen)
    result = await agent._predict_with_details(
        _first_turn(), processed_images=[Image.new("RGB", (8, 8))]
    )
    assert len(seen) == 4  # 1 + max_parse_retries (upstream default 3)
    assert result.model_output_error == "no_executable_action"
    assert "tool_calls" not in result.lite_message


async def test_max_parse_retries_zero_disables_regeneration() -> None:
    seen: list[tuple[str, list[dict]]] = []
    agent = _agent(["bad\n</think>\nno action"], seen, max_parse_retries=0)
    result = await agent._predict_with_details(
        _first_turn(), processed_images=[Image.new("RGB", (8, 8))]
    )
    assert len(seen) == 1
    assert result.model_output_error == "no_executable_action"


def test_registries() -> None:
    assert "mai_gym@desktop@use" in AgentRegistry.list()
    assert "mai_gym@desktop@use" in AgentAdapterRegistry.list()
    assert "mai_gym.desktop_history" in ProtocolRegistry.list()
