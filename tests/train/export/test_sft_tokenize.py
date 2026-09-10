"""Unit tests for the SFT export-time tokenizer (lite/train/export/sft_tokenize.py).

Covers the producer-side contract directly (no slime needed): serialize/deserialize
round-trip, the malformed-step guard, image-index ordering, and the load-bearing
``assert full_text.startswith(prompt_text)`` boundary guard that fires on
enable_thinking drift.

Run: uv run pytest tests/train/export/test_sft_tokenize.py -q
"""
from __future__ import annotations

import pytest

pytest.importorskip("transformers", reason="transformers not installed")

from lite.core.messages.image_refs import IMAGE_REFERENCE_ROLES
from lite.core.samples import STATUS_COMPLETED, STATUS_TRUNCATED, LiteRLStep
from lite.train.export.sft_tokenize import (
    agent_step_to_rl_step,
    deserialize_rl_step,
    serialize_rl_step,
)

# Qwen3.5 is the one family whose chat template HONORS enable_thinking (empty
# <think></think> prefix when off vs <think>\n when on), so it exercises the
# boundary guard. Cached on the dev host.
_THINK_MODEL = "Qwen/Qwen3.5-2B"


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class _FakeProcessorOwnedSlotProcessor:
    tokenizer = _FakeTokenizer()
    image_token = "<processor_image>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ):
        rendered = ""
        for message in messages:
            rendered += f"<{message['role']}>"
            content = message.get("content") or []
            if isinstance(content, str):
                rendered += content
            else:
                for part in content:
                    if part.get("type") == "image":
                        rendered += self.image_token
                    elif part.get("type") == "text":
                        rendered += part["text"]
            rendered += "</turn>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def test_serialize_deserialize_roundtrip():
    step = LiteRLStep(
        prompt="<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
        image_indices=(0, 2, 5),
        response="ok<|im_end|>\n",
        response_tokens=[1, 2, 3, 4],
        response_log_probs=[0.0] * 4,  # dropped on serialize; must round-trip to []
        reward=0.0,
        status=STATUS_COMPLETED,
    )
    d = serialize_rl_step(step)
    assert set(d) == {
        "prompt", "image_indices", "response", "response_tokens",
        "reward", "status", "prompt_tokens",
    }
    assert "response_log_probs" not in d  # intentionally not persisted
    assert d["prompt_tokens"] is None  # image step → not precomputed
    back = deserialize_rl_step(d)
    assert back.prompt == step.prompt
    assert back.image_indices == (0, 2, 5)  # list -> tuple
    assert back.response == step.response
    assert back.response_tokens == [1, 2, 3, 4]
    assert back.response_log_probs == []  # defaulted, not stored
    assert back.status == STATUS_COMPLETED
    assert back.prompt_tokens is None


def test_prompt_tokens_roundtrip_for_text_only_step():
    """Text-only steps persist prompt_tokens; image steps store None."""
    text_step = LiteRLStep(
        prompt="<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
        image_indices=(),
        response="ok<|im_end|>\n",
        response_tokens=[1, 2],
        prompt_tokens=[10, 20, 30],
    )
    d = serialize_rl_step(text_step)
    assert d["prompt_tokens"] == [10, 20, 30]
    assert deserialize_rl_step(d).prompt_tokens == [10, 20, 30]


def test_deserialize_missing_prompt_tokens_fails_loud():
    old = {"prompt": "p", "image_indices": [], "response": "r",
           "response_tokens": [1], "reward": 0.0, "status": STATUS_COMPLETED}
    with pytest.raises(ValueError, match="missing prompt_tokens") as exc:
        deserialize_rl_step(old)
    msg = str(exc.value)
    assert "current exporter" in msg
    assert "devs/migration" not in msg


def test_deserialize_normalizes_null_list_columns():
    """parquet can hand back None for an empty list column — must not crash."""
    back = deserialize_rl_step(
        {"prompt": "p", "image_indices": None, "response": "r",
         "response_tokens": None, "reward": 0.0, "status": STATUS_COMPLETED,
         "prompt_tokens": None}
    )
    assert back.image_indices == ()
    assert back.response_tokens == []


@pytest.mark.parametrize("status", [None, "", "COMPLETED", "done"])
def test_deserialize_rejects_out_of_vocabulary_status(status):
    """A status the segmenter cannot score is rejected at parquet ingestion.

    The segmenter also raises on it, but only this boundary can tell the
    operator which published parquet to re-export.
    """
    with pytest.raises(ValueError) as exc:
        deserialize_rl_step(
            {"prompt": "p", "image_indices": [], "response": "r",
             "response_tokens": [1], "reward": 0.0, "status": status,
             "prompt_tokens": None}
        )
    assert "current exporter" in str(exc.value)


@pytest.mark.parametrize("field", ["prompt", "response", "reward", "status"])
def test_deserialize_rejects_null_producer_owned_field(field):
    """The producer writes every field unconditionally; a null scalar means a
    stale parquet, not a value to substitute a default for."""
    step = {"prompt": "p", "image_indices": [], "response": "r",
            "response_tokens": [1], "reward": 0.0, "status": STATUS_COMPLETED,
            "prompt_tokens": None}
    step[field] = None
    with pytest.raises(ValueError, match=f"missing {field}"):
        deserialize_rl_step(step)


def test_deserialize_preserves_truncated_status():
    back = deserialize_rl_step(
        {"prompt": "p", "image_indices": [], "response": "r",
         "response_tokens": [1], "reward": 0.0, "status": STATUS_TRUNCATED,
         "prompt_tokens": None}
    )
    assert back.status == STATUS_TRUNCATED


@pytest.fixture(scope="module")
def think_processor():
    # Production (export_sft / dagger.teacher) always passes an AutoProcessor;
    # use the same here so the tests exercise the real code path (no shim that
    # could silently divert to a text-only fallback and mask a regression).
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(_THINK_MODEL, trust_remote_code=True)


def _step(reasoning: str):
    return [
        {"role": "user", "content": [{"type": "text", "text": "enable do-not-track"}]},
        {"role": "assistant", "content": "Action: click", "reasoning_content": reasoning},
    ]


def test_none_on_non_assistant_terminated(think_processor):
    # last message is a user turn → not a valid SFT target
    bad = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert agent_step_to_rl_step(bad, think_processor) is None
    assert agent_step_to_rl_step([], think_processor) is None


def test_thinking_match_ok_then_response_roundtrips(think_processor):
    """enable_thinking=True on reasoning data: boundary holds, response_tokens
    decode back to response."""
    rl = agent_step_to_rl_step(_step("I should open settings"), think_processor,
                               enable_thinking=True)
    assert rl is not None
    assert rl.prompt.rstrip().endswith("<think>")  # thinking-on generation prefix
    assert think_processor.tokenizer.decode(rl.response_tokens) == rl.response


def test_thinking_off_strips_reasoning_from_the_target(think_processor):
    """Thinking off renders reasoning data WITHOUT reasoning, instead of refusing it.

    One internalized dataset has to serve both recipes: the same ``.think`` rows are
    exported with reasoning under a thinking-on config and without it under a
    thinking-off one. Before the strip this raised on the prompt/target prefix check,
    which left no path from internalized rows to Action-only training data.
    """
    rl = agent_step_to_rl_step(_step("some real reasoning"), think_processor,
                               enable_thinking=False)
    assert rl is not None
    assert "some real reasoning" not in rl.prompt
    assert "some real reasoning" not in rl.response
    assert "Action: click" in rl.response
    assert think_processor.tokenizer.decode(rl.response_tokens) == rl.response


def test_thinking_off_strips_reasoning_from_history_too(think_processor):
    """The leak the prefix check cannot see.

    A ``<tool_response>`` turn is not a "real user query", so Qwen's template keeps
    the ``<think>`` body of every assistant turn after the opening instruction. A
    target turn WITHOUT reasoning therefore passes the prefix check while history
    reasoning is still rendered into the thinking-off prompt — silently.
    """
    step = [
        {"role": "user", "content": [{"type": "text", "text": "do it"}]},
        {"role": "assistant", "content": "Action: a", "reasoning_content": "HISTORY_THOUGHT"},
        {"role": "tool", "content": [{"type": "text", "text": "obs"}]},
        {"role": "assistant", "content": "Action: b"},
    ]
    off = agent_step_to_rl_step(step, think_processor, enable_thinking=False)
    assert off is not None
    assert "HISTORY_THOUGHT" not in off.prompt

    # Thinking on is untouched: the template's own handling stays the contract.
    on = agent_step_to_rl_step(step, think_processor, enable_thinking=True)
    assert on is not None
    assert "HISTORY_THOUGHT" in on.prompt


def test_thinking_off_does_not_mutate_the_caller_s_messages(think_processor):
    """The strip runs on ``keep_model_visible_content``'s deepcopy, never the input."""
    step = _step("some real reasoning")
    agent_step_to_rl_step(step, think_processor, enable_thinking=False)
    assert step[-1]["reasoning_content"] == "some real reasoning"


def test_thinking_off_no_reasoning_ok(think_processor):
    """Happy path for the default (instruct-style) mode: enable_thinking=False
    with NO reasoning_content must NOT trip the boundary guard — the empty
    <think></think> prefix is a clean prefix of the reasoning-free target."""
    step = [
        {"role": "user", "content": [{"type": "text", "text": "do it"}]},
        {"role": "assistant", "content": "Action: click", "reasoning_content": ""},
    ]
    rl = agent_step_to_rl_step(step, think_processor, enable_thinking=False)
    assert rl is not None
    assert "Action: click" in rl.response
    assert think_processor.tokenizer.decode(rl.response_tokens) == rl.response


def test_no_image_step_has_empty_indices(think_processor):
    """A text-only step (no image parts) yields image_indices=() — exercises the
    downstream tokenize-only / multimodal_train_inputs=None branch."""
    step = [
        {"role": "user", "content": [{"type": "text", "text": "no images here"}]},
        {"role": "assistant", "content": "Action: done", "reasoning_content": ""},
    ]
    rl = agent_step_to_rl_step(step, think_processor, enable_thinking=True)
    assert rl.image_indices == ()
    # response_log_probs is populated 1:1 with response_tokens by the producer
    assert len(rl.response_log_probs) == len(rl.response_tokens)


def test_text_only_step_precomputes_prompt_tokens(think_processor):
    """FIX B: a text-only step stores prompt_tokens equal to
    tokenizer.encode(prompt, add_special_tokens=False); an image step leaves it
    None (its prompt_ids depend on image_grid_thw expansion at train time)."""
    text_step = [
        {"role": "user", "content": [{"type": "text", "text": "no images here"}]},
        {"role": "assistant", "content": "Action: done", "reasoning_content": ""},
    ]
    rl = agent_step_to_rl_step(text_step, think_processor, enable_thinking=True)
    assert rl.image_indices == ()
    assert rl.prompt_tokens is not None
    assert rl.prompt_tokens == think_processor.tokenizer.encode(
        rl.prompt, add_special_tokens=False
    )

    img_processor = _FakeProcessorOwnedSlotProcessor()
    img_step = [
        {"role": "user", "content": [
            {"type": "image", "index": 0}, {"type": "text", "text": "obs"}]},
        {"role": "assistant", "content": "Action: a"},
    ]
    rl_img = agent_step_to_rl_step(img_step, img_processor)
    assert rl_img.image_indices == (0,)
    assert rl_img.prompt.count(img_processor.image_token) == 1
    assert rl_img.prompt_tokens is None  # image step → not precomputed


def test_image_indices_user_and_tool_in_render_order():
    """image_indices follow prompt image-slot order across user/tool observations."""
    assert IMAGE_REFERENCE_ROLES == frozenset({"user", "tool"})
    step = [
        {"role": "user", "content": [
            {"type": "image", "index": 0}, {"type": "image", "index": 1},
            {"type": "text", "text": "obs"}]},
        {"role": "assistant", "content": "Action: a", "reasoning_content": ""},
        {"role": "tool", "tool_call_id": "call_0000", "content": [
            {"type": "image", "index": 2},
            {"type": "image", "index": 2},
            {"type": "text", "text": "obs2"}]},
        {"role": "assistant", "content": "Action: b", "reasoning_content": ""},
    ]
    proc = _FakeProcessorOwnedSlotProcessor()
    rl = agent_step_to_rl_step(step, proc)
    assert rl.image_indices == (0, 1, 2, 2)
    assert rl.prompt.count(proc.image_token) == 4
