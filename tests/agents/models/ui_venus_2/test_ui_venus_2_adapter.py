"""UI-Venus-2 adapters: prompt assembly, history shape, and the tag codec.

Run:
    uv run pytest tests/agents/models/ui_venus_2 -p no:cacheprovider -q
"""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter.base import AgentAdapterRegistry
from lite.agents.models.ui_venus_2.adapter import (
    CURRENT_SCREENSHOT_LABEL,
    GROUNDING_INFEASIBLE_REASON,
    HISTORY_SCREENSHOT_LABEL,
    UIVenus2BrowserUseAdapter,
    UIVenus2DesktopUseAdapter,
    UIVenus2GroundingPointAdapter,
    UIVenus2MobileUseAdapter,
)
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.final import MODEL_OUTPUT_ERROR_KEY
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name

register_all()

TASK = "Open Settings and turn on Wi-Fi."


def _adapter(key: str, platform: str, **kwargs):
    return AgentAdapterRegistry.get(
        key, metadata=LiteCUAMetadata(dims=(platform, "use")), **kwargs
    )


def _sample(platform: str, n_turns: int, *, task: str = TASK) -> LiteSample:
    """A GUI trajectory of ``n_turns`` completed turns plus a fresh observation."""
    batch = "mobile" if platform == "mobile" else "computer"
    action = "tap" if platform == "mobile" else "click"
    images = [Image.new("RGB", (1280, 720), (index * 10, 0, 0)) for index in range(n_turns)]
    messages: list[dict[str, Any]] = []
    for index in range(n_turns):
        content: list[dict[str, Any]] = [{"type": "image", "index": index}]
        if index == 0:
            content.append({"type": "text", "text": task})
        messages.append({"role": "user", "content": content})
        if index < n_turns - 1:
            messages.append({
                "role": "assistant",
                "content": [{"type": "inline_reasoning", "text": f"thought {index}"}],
                "tool_calls": [make_tool_call(
                    batch,
                    {"actions": [{"action": action, "coordinate": [100 + index, 200]}]},
                    call_id=f"call_{index}",
                )],
            })
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, "use")), images=images, messages=messages,
    )


def _texts(message: dict[str, Any]) -> list[str]:
    return [
        part["text"] for part in (message.get("content") or [])
        if part.get("type") == "text"
    ]


def _reasonings(message: dict) -> list[str]:
    return [
        part["text"] for part in message["content"]
        if part["type"] == "inline_reasoning"
    ]


def _kinds(message: dict[str, Any]) -> list[str]:
    return [part.get("type") for part in (message.get("content") or [])]


# =============================================================================
# Prompt assembly
# =============================================================================

class TestSystemPrompt:
    @pytest.mark.parametrize(
        ("key", "platform", "marker"),
        [
            ("ui_venus_2@desktop@use", "desktop", "Sequence(actions="),
            ("ui_venus_2@browser@use", "browser", "GetUrl()"),
            ("ui_venus_2@mobile@use", "mobile", "LaunchApp(app='')"),
        ],
    )
    def test_task_goes_into_the_system_prompt_with_the_surface_rows(
        self, key, platform, marker,
    ) -> None:
        step = _adapter(key, platform).unroll(_sample(platform, 1)).steps[0]
        assert step[0]["role"] == "system"
        system_text = _texts(step[0])[0]
        assert system_text.rstrip().endswith(TASK)
        assert marker in system_text

    def test_action_rows_survive_a_narrowed_tool_surface(self) -> None:
        """The prompt is SFT text: neither ``valid_actions`` nor the active
        extra tools may delete a trained row."""
        adapter = AgentAdapterRegistry.get(
            "ui_venus_2@desktop@use",
            metadata=LiteCUAMetadata(dims=("desktop", "use"), valid_actions=["click"]),
        )
        system_text = _texts(adapter.unroll(_sample("desktop", 1)).steps[0][0])[0]
        for row in ("TripleClick(box=", "KeyDown(keys=", "Sequence(actions=", "Wait()"):
            assert row in system_text

    def test_sudo_password_is_substituted_and_overridable(self) -> None:
        """It is a per-environment secret the Computer prompt states verbatim."""
        default = _adapter("ui_venus_2@desktop@use", "desktop")
        assert "The password of the computer is password." in _texts(
            default.unroll(_sample("desktop", 1)).steps[0][0]
        )[0]
        custom = _adapter("ui_venus_2@desktop@use", "desktop", sudo_password="hunter2")
        assert "The password of the computer is hunter2." in _texts(
            custom.unroll(_sample("desktop", 1)).steps[0][0]
        )[0]

    def test_browser_date_defaults_to_today_and_can_be_pinned(self) -> None:
        """Upstream fills the wall clock, which makes a run unreproducible; a
        pinned ISO date freezes the prompt."""
        import datetime

        today = datetime.date.today().isoformat()
        assert f"Today is {today}." in _texts(
            _adapter("ui_venus_2@browser@use", "browser")
            .unroll(_sample("browser", 1)).steps[0][0]
        )[0]
        pinned = _adapter("ui_venus_2@browser@use", "browser", current_date="2026-01-02")
        assert "Today is 2026-01-02." in _texts(
            pinned.unroll(_sample("browser", 1)).steps[0][0]
        )[0]


# =============================================================================
# History shape
# =============================================================================

class TestHistoryShape:
    def test_alternation_and_labels(self) -> None:
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        step = adapter.unroll(_sample("desktop", 3)).steps[-1]

        assert [message["role"] for message in step] == [
            "system", "user", "assistant", "user", "assistant", "user",
        ]
        assert _texts(step[-1]) == [CURRENT_SCREENSHOT_LABEL]
        assert _texts(step[3]) == [HISTORY_SCREENSHOT_LABEL]

    def test_the_goal_is_not_printed_twice(self) -> None:
        """It is already in the system prompt, so the first observation keeps
        only its screenshot."""
        step = _adapter("ui_venus_2@desktop@use", "desktop").unroll(
            _sample("desktop", 2)
        ).steps[-1]
        observations = [message for message in step[1:] if message["role"] == "user"]
        assert all(TASK not in text for message in observations for text in _texts(message))

    def test_evicted_screenshot_leaves_an_empty_user_bubble(self) -> None:
        """Only screenshots are evicted; upstream still emits the user turn so
        the alternation the model was trained on is unbroken."""
        adapter = _adapter(
            "ui_venus_2@desktop@use", "desktop", protocol_kwargs={"n_history_images": 0},
        )
        step = adapter.unroll(_sample("desktop", 3)).steps[-1]
        observations = [message for message in step if message["role"] == "user"]
        assert [_kinds(message) for message in observations] == [
            [], [], ["text", "image"],
        ]

    @pytest.mark.parametrize(
        ("n_history_images", "expected"),
        [
            (0, [4]),
            (1, [3, 4]),
            (2, [2, 3, 4]),        # the shipped N_IMG=2 -> THREE images
            (10, [0, 1, 2, 3, 4]),
        ],
    )
    def test_n_history_images_counts_past_screenshots_only(
        self, n_history_images, expected,
    ) -> None:
        """Upstream's ``N_IMG`` budgets the ``history`` list, and the current
        screenshot is appended afterwards unconditionally — so ``N_IMG=2`` sends
        three images, not two."""
        adapter = _adapter(
            "ui_venus_2@desktop@use", "desktop",
            protocol_kwargs={"n_history_images": n_history_images},
        )
        step = adapter.unroll(_sample("desktop", 5)).steps[-1]
        indices = [
            part["index"] for message in step
            for part in (message.get("content") or [])
            if part.get("type") == "image"
        ]
        assert indices == expected

    def test_assistant_text_is_never_evicted_with_its_screenshot(self) -> None:
        """"Build system + all assistant text + the last n_img history images"."""
        adapter = _adapter(
            "ui_venus_2@desktop@use", "desktop", protocol_kwargs={"n_history_images": 0},
        )
        step = adapter.unroll(_sample("desktop", 5)).steps[-1]
        assistants = [message for message in step if message["role"] == "assistant"]
        assert len(assistants) == 4
        assert all(_texts(message) for message in assistants)

    def test_browser_sends_the_screenshot_without_a_label(self) -> None:
        """``make_user`` in the reference sends the image bare."""
        step = _adapter("ui_venus_2@browser@use", "browser").unroll(
            _sample("browser", 2)
        ).steps[-1]
        assert _kinds(step[-1]) == ["image"]

    def test_history_assistants_are_replayed_as_wire_text(self) -> None:
        step = _adapter("ui_venus_2@desktop@use", "desktop").unroll(
            _sample("desktop", 3)
        ).steps[-1]
        assistants = [message for message in step if message["role"] == "assistant"]
        assert _texts(assistants[0]) == [
            "<think>thought 0</think>\n<action>Click(box=(100, 200))</action>"
        ]

    def _sample_with_trailing_tool_result(self) -> LiteSample:
        """A final turn whose observation block is a screenshot AND a tool result."""
        sample = _sample("desktop", 1)
        sample.messages.append({
            "role": "assistant",
            "content": [{"type": "inline_reasoning", "text": "t"}],
            "tool_calls": [make_tool_call("computer", {
                "actions": [{"action": "click", "coordinate": [1, 2]}]
            }, call_id="call_0")],
        })
        sample.images.append(Image.new("RGB", (1280, 720)))
        sample.messages.append({"role": "user", "content": [{"type": "image", "index": 1}]})
        sample.messages.append({
            "role": "tool",
            "call_id": "call_0",
            "content": [{"type": "text", "text": "error: element not clickable"}],
        })
        return sample

    def test_a_turns_observations_render_as_one_bubble_feedback_first(self) -> None:
        """Upstream sends ONE user message per turn, and the browser harness
        builds it as ``[feedback, image]`` — ``GetUrl``'s result is documented as
        arriving "at the beginning of the next user message"."""
        step = _adapter("ui_venus_2@desktop@use", "desktop").unroll(
            self._sample_with_trailing_tool_result()
        ).steps[-1]
        assert [message["role"] for message in step] == [
            "system", "user", "assistant", "user",
        ]
        assert _texts(step[-1]) == [
            "error: element not clickable", CURRENT_SCREENSHOT_LABEL,
        ]
        assert _kinds(step[-1]) == ["text", "text", "image"]

    def test_a_trailing_tool_result_does_not_steal_the_current_label(self) -> None:
        """The label describes a SCREENSHOT, so it belongs to the last bubble
        that carries one — not merely the last observation message."""
        step = _adapter("ui_venus_2@desktop@use", "desktop").unroll(
            self._sample_with_trailing_tool_result()
        ).steps[-1]
        labels = [
            text for message in step if message["role"] == "user"
            for text in _texts(message)
            if text in (HISTORY_SCREENSHOT_LABEL, CURRENT_SCREENSHOT_LABEL)
        ]
        assert labels == [HISTORY_SCREENSHOT_LABEL, CURRENT_SCREENSHOT_LABEL]


# =============================================================================
# Tag codec
# =============================================================================

class TestWireCodec:
    def test_think_and_action_round_trip(self) -> None:
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        raw = "<think>press save</think>\n<action>Hotkey(keys=['ctrl', 's'])</action>"
        lite = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
        (child,) = tool_call_arguments(lite["tool_calls"][0])["actions"]
        assert child == {"action": "key", "keys": ["ctrl", "s"]}
        assert _texts(adapter.convert_message_to_agent(lite)) == [raw]

    @pytest.mark.parametrize(
        "key", ["ui_venus_2@desktop@use", "ui_venus_2@browser@use", "ui_venus_2@mobile@use"],
    )
    def test_thinking_stays_enabled_so_the_prompt_leaves_think_open(self, key) -> None:
        """The checkpoint's template renders ``enable_thinking=False`` as a
        CLOSED ``<think>\n\n</think>`` before generation, leaving no room for the
        ``<think>`` block every UI-Venus-2 prompt's Output Format demands.
        Measured cost of getting this wrong on lite.osworld: 95% of turns came
        back as a bare ``<action>`` with no reasoning, and 0.337 vs 0.805."""
        platform = key.split("@")[1]
        assert _adapter(key, platform).enable_thinking is True

    def test_reasoning_arrives_with_only_the_closing_tag(self) -> None:
        """The generation prompt ends with an OPEN ``<think>\n``, so the model's
        reasoning is everything before the first ``</think>``. Requiring a
        matched ``<think>…</think>`` pair would drop the whole reasoning trail
        from the replayed history."""
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        raw = "planning the click\n</think>\n\n<action>Click(box=(79, 403))</action>"
        lite = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
        assert lite["content"] == [
            {"type": "inline_reasoning", "text": "planning the click"}
        ]
        (child,) = tool_call_arguments(lite["tool_calls"][0])["actions"]
        assert child == {"action": "click", "coordinate": [79, 403]}

    def test_reasoning_replays_into_history_as_a_closed_block(self) -> None:
        """The template re-splits an assistant turn on ``</think>``, so the
        rendered history has to carry both tags even though the live response
        only had one."""
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        raw = "why I click\n</think>\n<action>Wait()</action>"
        lite = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
        assert _texts(adapter.convert_message_to_agent(lite)) == [
            "<think>why I click</think>\n<action>Wait()</action>"
        ]

    def test_reasoning_is_inline_not_the_native_channel(self) -> None:
        """``<think>`` here is prompted CoT text, so it belongs in
        ``InlineReasoningContent`` — the top-level ``reasoning_content`` slot is
        for a chat-template thinking channel."""
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(
                "<think>because</think><action>Wait()</action>"
            )
        )
        assert "reasoning_content" not in lite
        assert lite["content"] == [{"type": "inline_reasoning", "text": "because"}]

    def test_prose_with_no_action_tag_stays_a_content_only_final(self) -> None:
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response("The task is already done.")
        )
        assert "tool_calls" not in lite
        assert MODEL_OUTPUT_ERROR_KEY not in lite
        assert _texts(lite) == ["The task is already done."]

    def test_prose_after_a_closed_think_stays_a_content_only_final(self) -> None:
        """The same contract on the thinking-ON default, where the model has to
        close the block before it can answer."""
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(
                "checked every panel</think>\nThe task is already done."
            )
        )
        assert "tool_calls" not in lite
        assert MODEL_OUTPUT_ERROR_KEY not in lite
        assert _texts(lite) == ["The task is already done."]

    def test_a_content_only_final_ships_the_answer_without_the_reasoning(self) -> None:
        """This text becomes the model's ANSWER through
        ``summarize_no_tool_call_final``, and answer-graded envs string-match it,
        so the CoT and the literal ``</think>`` must not ride along. The
        reasoning is kept as its own part rather than dropped."""
        adapter = _adapter("ui_venus_2@mobile@use", "mobile")
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(
                "I can see the number on screen.\n</think>\nThe number is 555-0134."
            )
        )
        assert "tool_calls" not in lite
        assert _texts(lite) == ["The number is 555-0134."]
        assert [
            part["text"] for part in lite["content"]
            if part["type"] == "inline_reasoning"
        ] == ["I can see the number on screen."]

    def test_a_content_only_final_keeps_its_answer_through_render(self) -> None:
        """The turn's answer is its whole point. Once the parser splits
        reasoning from answer, the renderer has to emit BOTH -- emitting only
        the think block trains an SFT/DAgger target to reason and never answer."""
        adapter = _adapter("ui_venus_2@mobile@use", "mobile")
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(
                "I can see it.\n</think>\nThe number is 555-0134."
            )
        )
        rendered = adapter.convert_message_to_agent(dict(lite))
        assert rendered["content"][0]["text"] == (
            "<think>I can see it.</think>\nThe number is 555-0134."
        )

    def test_a_final_closed_then_cut_keeps_its_text(self) -> None:
        """Closed, then EOS: the split leaves nothing after the tag, so the
        answer is what came BEFORE it. Emitting the split would submit an empty
        string; keeping the raw text would ship the literal tag to a grader."""
        adapter = _adapter("ui_venus_2@mobile@use", "mobile")
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response("The task is done.</think>")
        )
        assert _texts(lite) == ["The task is done."]
        # The reasoning is reset, not duplicated: the text before the tag is the
        # ANSWER here, so shipping it as inline_reasoning too would say it twice.
        assert not _reasonings(lite)

    def test_a_final_answer_comes_from_the_LAST_think_close(self) -> None:
        """A ``</think>`` the model quotes mid-reasoning is not the boundary.

        Splitting on the FIRST one would ship the rest of the CoT -- and a
        literal ``</think>`` -- to a grader that string-matches the answer.
        """
        adapter = _adapter("ui_venus_2@mobile@use", "mobile")
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(
                "I should not write </think> yet. I can read the number.\n"
                "</think>\nThe number is 555-0134."
            )
        )
        assert _texts(lite) == ["The number is 555-0134."]

    def test_a_malformed_action_block_is_a_terminal_parse_failure(self) -> None:
        """``<action>`` is the trained grammar marker: reaching for it and
        failing must not be read as clean prose."""
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response("<think>t</think><action>Click(box=(1</action>")
        )
        assert "tool_calls" not in lite
        assert "malformed <action> block" in lite[MODEL_OUTPUT_ERROR_KEY]

    def test_two_action_blocks_are_a_parse_failure_not_the_first_one(self) -> None:
        """Upstream ``parse_response`` raises on anything but exactly one block.
        Silently running the first would execute an action the model never
        singled out."""
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(
                "<action>Click(box=(1, 2))</action><action>Wait()</action>"
            )
        )
        assert "tool_calls" not in lite
        assert "exactly one block" in lite[MODEL_OUTPUT_ERROR_KEY]

    def test_an_action_that_converts_to_nothing_is_a_parse_failure(self) -> None:
        """An empty ``Sequence`` parses as a call but yields no Lite call. It
        must not collapse into a clean content-only final."""
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response("<action>Sequence(actions=[])</action>")
        )
        assert not lite.get("tool_calls")
        assert MODEL_OUTPUT_ERROR_KEY in lite

    def test_content_only_final_keeps_its_text_as_an_sft_target(self) -> None:
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        rendered = adapter.convert_message_to_agent(
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
        )
        assert _texts(rendered) == ["Done."]

    def test_structured_tool_calls_survive_when_there_is_no_text(self) -> None:
        """The API path: nothing to parse out of ``content``, so only the calls
        are normalized."""
        adapter = _adapter("ui_venus_2@mobile@use", "mobile")
        lite = adapter.convert_message_from_agent({
            "role": "assistant",
            "content": [],
            "tool_calls": [{"name": "Click", "arguments": {"point": [5, 6]}}],
        })
        (child,) = tool_call_arguments(lite["tool_calls"][0])["actions"]
        assert child == {"action": "tap", "coordinate": [5, 6], "clicks": 1}


# =============================================================================
# Vision
# =============================================================================

class TestImageProcessing:
    def test_screenshots_are_smart_resized_onto_the_32px_grid(self) -> None:
        adapter = _adapter("ui_venus_2@desktop@use", "desktop")
        processed = adapter.unroll(_sample("desktop", 1)).processed_images[0]
        width, height = processed.size
        assert width % 32 == 0 and height % 32 == 0

    def test_the_pixel_cap_is_overridable(self) -> None:
        adapter = _adapter(
            "ui_venus_2@desktop@use", "desktop", smart_resize_max_pixels=64 * 32 * 32,
        )
        processed = adapter.unroll(_sample("desktop", 1)).processed_images[0]
        assert processed.size[0] * processed.size[1] <= 64 * 32 * 32


# =============================================================================
# Grounding
# =============================================================================

def _grounding_adapter(platform: str = "desktop") -> UIVenus2GroundingPointAdapter:
    return AgentAdapterRegistry.get(
        f"ui_venus_2@{platform}@grounding.point",
        metadata=LiteCUAMetadata(dims=(platform, "grounding.point")),
    )


def _grounding_sample(instruction: str) -> LiteSample:
    return LiteSample(
        metadata=LiteCUAMetadata(dims=("desktop", "grounding.point")),
        images=[Image.new("RGB", (1920, 1080))],
        messages=[{
            "role": "user",
            "content": [{"type": "image", "index": 0}, {"type": "text", "text": instruction}],
        }],
    )


class TestGroundingAdapter:
    def test_one_user_message_image_first_and_no_system(self) -> None:
        """``Qwen35GroundModel._build_messages`` sends no system message and puts
        the image before the text — the opposite of the ``use`` surfaces."""
        step = _grounding_adapter().unroll(_grounding_sample("the Save button.")).steps[0]
        assert len(step) == 1
        assert step[0]["role"] == "user"
        assert _kinds(step[0]) == ["image", "text"]

    def test_trailing_period_is_stripped(self) -> None:
        """The template supplies its own ``.`` after the slot."""
        text = _texts(
            _grounding_adapter().unroll(_grounding_sample("the Save button.")).steps[0][0]
        )[0]
        assert "instruction: \nthe Save button. \n\n" in text
        assert "the Save button.." not in text

    @pytest.mark.parametrize("platform", ["desktop", "browser", "mobile"])
    def test_one_harness_serves_every_platform(self, platform) -> None:
        assert isinstance(_grounding_adapter(platform), UIVenus2GroundingPointAdapter)

    @pytest.mark.parametrize(
        ("raw", "coordinate"),
        [
            ("[512,300]", [512, 300]),
            ("[512, 300]", [512, 300]),
            ("The point is [123,456].", [123, 456]),
            ("[10, 20, 30, 40]", [20, 30]),          # bbox -> centre
            ("[10, 20], [30, 40]", [20, 30]),        # two points -> centre
        ],
    )
    def test_point_answers(self, raw, coordinate) -> None:
        adapter = _grounding_adapter()
        lite = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
        (call,) = lite["tool_calls"]
        assert tool_call_name(call) == "point"
        assert tool_call_arguments(call)["coordinate"] == coordinate

    def test_a_box_whose_first_pair_is_the_marker_is_a_refusal(self) -> None:
        """``extract_coordinates_qwen35`` tests the first pair BEFORE averaging,
        so this is a refusal, not a point at the midpoint of a nonsense box."""
        adapter = _grounding_adapter()
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response("[-1,-1,5,5]")
        )
        assert tool_call_name(lite["tool_calls"][0]) == "report_infeasible"

    def test_the_refusal_marker_becomes_report_infeasible(self) -> None:
        """``[-1,-1]`` is a TRAINED output of this checkpoint, not a heuristic
        read of prose."""
        adapter = _grounding_adapter()
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response("[-1,-1]")
        )
        (call,) = lite["tool_calls"]
        assert tool_call_name(call) == "report_infeasible"
        assert tool_call_arguments(call)["reason"] == GROUNDING_INFEASIBLE_REASON

    @pytest.mark.parametrize("raw", ["[512,300]", "[-1,-1]"])
    def test_answers_render_back_to_the_bare_list(self, raw) -> None:
        adapter = _grounding_adapter()
        lite = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
        assert _texts(adapter.convert_message_to_agent(lite)) == [raw.replace(" ", "")]

    def test_prose_with_no_list_stays_content_only(self) -> None:
        adapter = _grounding_adapter()
        lite = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response("I cannot tell.")
        )
        assert "tool_calls" not in lite
        assert _texts(lite) == ["I cannot tell."]


# =============================================================================
# Registry wiring
# =============================================================================

@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("ui_venus_2@desktop@use", UIVenus2DesktopUseAdapter),
        ("ui_venus_2@browser@use", UIVenus2BrowserUseAdapter),
        ("ui_venus_2@mobile@use", UIVenus2MobileUseAdapter),
        ("ui_venus_2@desktop@grounding.point", UIVenus2GroundingPointAdapter),
        ("ui_venus_2@mobile@grounding.point", UIVenus2GroundingPointAdapter),
    ],
)
def test_adapter_keys_resolve(key, expected) -> None:
    assert AgentAdapterRegistry.get_class(key) is expected
