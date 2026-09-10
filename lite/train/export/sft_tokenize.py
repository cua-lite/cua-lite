"""Tokenize a rendered AgentStep into a :class:`LiteRLStep`.

The training contract is a generic RL step, independent of any env-specific
state shape: rendered ``prompt`` text, ordered ``image_indices`` into the
trajectory image list, assistant ``response`` text, and ``response_tokens``.

Shared by ``export_sft`` (offline, on the host) and ``dagger.teacher`` (online
relabel). Kept slime-free so the export path can run outside the training
container — it needs only ``transformers`` (for ``apply_chat_template`` /
``tokenizer.encode``) and the cua-lite ``LiteRLStep`` type.

Run: imported as a library (no ``__main__``).
"""

from __future__ import annotations

from lite.core.messages import keep_model_visible_content
from lite.core.messages.image_refs import referenced_image_indices_in_message_order
from lite.core.samples import (
    STATUS_COMPLETED,
    STEP_STATUSES_BY_SEVERITY,
    LiteRLStep,
)


def agent_step_to_rl_step(
    agent_step: list[dict], processor, enable_thinking: bool = False
) -> LiteRLStep | None:
    """Convert one rendered AgentStep (messages list) into a :class:`LiteRLStep`.

    The step's last message is the assistant target; everything before is the
    prompt. The boundary is derived by rendering the prompt-only / full views
    with ``apply_chat_template`` and string-slicing (two-pass technique).
    ``image_indices`` are collected in the same order images appear in the
    prompt-side messages; that order must match the rendered processor-owned
    image slots because the trainer supplies images positionally to the
    processor.

    ``enable_thinking`` is forwarded to BOTH renders so the boundary matches how
    the data was generated — the template's generation prefix differs by mode
    (e.g. Qwen3.5 emits ``<think>\\n`` thinking-on vs an empty
    ``<think>\\n\\n</think>\\n\\n`` off). Sourced from the adapter
    (``adapter.enable_thinking``); defaults False for families without a
    reasoning channel.

    Thinking OFF additionally strips ``reasoning_content`` from every message.
    That makes one internalized dataset serve both recipes: the same ``.think``
    rows export with reasoning under a thinking-on config and without it under a
    thinking-off one. Thinking ON changes nothing here — the template's own
    handling of history reasoning is the contract.

    The emitted ``status`` is ``STATUS_COMPLETED``: a rendered step is a
    finished assistant turn and a saved row carries no per-turn
    ``finish_reason``. Episode outcome is a TRAJECTORY-level fact that only the
    caller holds — ``export_sft._convert_sample`` re-stamps the last step from
    the row's own env feedback, exactly as the online rollout does.

    Returns ``None`` when the step does not end in an assistant message: a
    partial turn carries no target, so there is nothing to supervise. Callers
    drop that step, never the trajectory.
    """
    if not agent_step or agent_step[-1].get("role") != "assistant":
        return None
    # Model boundary (offline tokenize path): drop internal side-channel parts
    # (MetadataContent etc.) the protocol windowing preserved for webgym/browsergym
    # — they crash the strict qwen3.5 template. The assistant target is
    # already rendered to a ``text`` part by the adapter, so it survives untouched;
    # only prompt-side metadata is stripped. Same shared boundary as base.py.
    agent_step = keep_model_visible_content(agent_step)
    if not enable_thinking:
        # Thinking off means the rendered sequence carries NO ``<think>`` body,
        # anywhere. Qwen's template renders ``reasoning_content`` on every turn
        # past the last real user query — and a ``<tool_response>`` turn is not
        # one, so in an agentic chain that is every assistant turn. Leaving it in
        # would render internalized reasoning into a thinking-off prompt, and on
        # a target turn that also carries it, break the prefix check below. With
        # thinking ON the template's own behavior is the contract; nothing here.
        for message in agent_step:
            message.pop("reasoning_content", None)
    prompt_messages = agent_step[:-1]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_text = processor.apply_chat_template(
        agent_step, tokenize=False, add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    # The generation prefix MUST be a byte-prefix of the rendered target, else
    # the slice below masks the whole sequence as the SFT target (silent,
    # catastrophic mis-supervision). Fail loud on drift.
    if not full_text.startswith(prompt_text):
        raise ValueError(
            f"SFT prompt/target boundary broke (enable_thinking={enable_thinking}): "
            f"prompt_text is not a prefix of full_text (generation-prefix drift). "
            f"The reasoning-vs-thinking mismatch that used to cause this is handled "
            f"above (thinking off strips reasoning_content), so a break here means "
            f"the template renders the target turn differently from the generation "
            f"prefix for some OTHER reason — a family whose chat template varies by "
            f"message shape, or a content part that survives "
            f"keep_model_visible_content and only appears in the full render. "
            f"prompt_tail={prompt_text[-48:]!r}"
        )
    response_text = full_text[len(prompt_text):]
    response_tokens = processor.tokenizer.encode(response_text, add_special_tokens=False)
    image_indices = referenced_image_indices_in_message_order(prompt_messages)
    # Text-only steps (no images): precompute prompt_tokens here so the train
    # path skips re-tokenizing the prompt string. Image steps leave it None —
    # their prompt_ids depend on per-image image_grid_thw vision-token expansion
    # and must be built with the processor at train time.
    prompt_tokens = None
    if not image_indices:
        prompt_tokens = list(
            processor.tokenizer.encode(prompt_text, add_special_tokens=False)
        )
    return LiteRLStep(
        prompt=prompt_text,
        image_indices=image_indices,
        response=response_text,
        response_tokens=list(response_tokens),
        response_log_probs=[0.0] * len(response_tokens),
        reward=0.0,
        status=STATUS_COMPLETED,
        prompt_tokens=prompt_tokens,
    )


# --- SFT parquet ``steps`` contract --------------------------------------
# ``export_sft`` stores each tokenized step as one struct; ``rollout_sft``
# reads it straight back into a ``LiteRLStep`` (no re-render at train time).
# ``response_log_probs`` is omitted — it is all-zeros for SFT and the train
# path recomputes rollout_log_probs anyway, so it round-trips as the default.

#: Every field ``serialize_rl_step`` writes. All are producer-owned and written
#: unconditionally, so an absent one means the parquet predates this contract.
_SERIALIZED_STEP_FIELDS = (
    "prompt", "image_indices", "response", "response_tokens",
    "reward", "status", "prompt_tokens",
)

#: Fields whose stored ``null`` is a legitimate value: parquet round-trips an
#: empty list column as null, and ``prompt_tokens`` is null for every image
#: step by contract. Null in any OTHER field is a stale-parquet signal.
_NULLABLE_STEP_FIELDS = frozenset({"image_indices", "response_tokens", "prompt_tokens"})


def serialize_rl_step(step: LiteRLStep) -> dict:
    """LiteRLStep → parquet-friendly struct (homogeneous fields, no nested
    message dicts → pyarrow infers a clean struct schema)."""
    return {
        "prompt": step.prompt,
        "image_indices": list(step.image_indices),
        "response": step.response,
        "response_tokens": list(step.response_tokens),
        "reward": float(step.reward),
        "status": step.status,
        # Only set for text-only steps (None otherwise) — see LiteRLStep. Always
        # present (homogeneous struct field) so pyarrow infers a clean
        # ``list<int>`` column; image steps store null. Lets the train path skip
        # re-tokenizing text-only prompts.
        "prompt_tokens": list(step.prompt_tokens) if step.prompt_tokens is not None else None,
    }


def deserialize_rl_step(d: dict) -> LiteRLStep:
    """Parquet struct → LiteRLStep (inverse of :func:`serialize_rl_step`).

    The producer writes every field unconditionally, so this reader takes them
    directly instead of substituting defaults. A stale/absent ``status`` in
    particular must not become ``completed``: the segmenter scores sample status
    as the max severity over a segment, so a defaulted status would train a
    truncated or failed turn as a complete one with nothing to notice it.
    """
    stale = [
        key for key in _SERIALIZED_STEP_FIELDS
        if key not in d or (d[key] is None and key not in _NULLABLE_STEP_FIELDS)
    ]
    if stale:
        raise ValueError(
            f"SFT step missing {', '.join(stale)}; re-export from the raw rollout "
            "or canonical dataset with the current exporter"
        )
    status = d["status"]
    if status not in STEP_STATUSES_BY_SEVERITY:
        raise ValueError(
            f"SFT step has out-of-vocabulary status {status!r}; expected one of "
            f"{sorted(STEP_STATUSES_BY_SEVERITY)}. Re-export from the raw rollout "
            "or canonical dataset with the current exporter"
        )
    prompt_tokens = d["prompt_tokens"]
    return LiteRLStep(
        prompt=d["prompt"],
        image_indices=tuple(d["image_indices"] or ()),
        response=d["response"],
        response_tokens=list(d["response_tokens"] or []),
        reward=float(d["reward"]),
        status=status,
        prompt_tokens=list(prompt_tokens) if prompt_tokens is not None else None,
    )
