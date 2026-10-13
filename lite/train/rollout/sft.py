"""VLM SFT rollout for CUA-Lite datasets.

Custom Slime rollout function. Reads the SFT parquet schema:

  processed_images : list[bytes | None]
                                    — PNG bytes for model-visible images and
                                      None placeholders for stored screenshots
                                      no step references
  steps            : list[struct]   — serialized LiteRLStep records (already
                                      tokenized at export; see
                                      lite.train.export.sft_tokenize). Each
                                      step carries prompt + ordered image_indices
                                      + response/response_tokens.
  metadata         : string         — Lite metadata provenance kept in the
                                      parquet, but not routed to this harness

Steps are tokenized **at export time** (``export_sft`` loads the processor via
``--model-id``, derives each step's prompt/response boundary with the model's
chat template, and stores a :class:`LiteRLStep`). So this rollout does NO
re-render: it just deserializes the steps into a :class:`LiteRLSample` and runs
it through the **same** :func:`build_segment_samples` radix path the GRPO /
REINFORCE rollouts use. Both training surfaces produce identical multi-span
``slime.Sample`` shapes under prefix-extensional protocols (qwen3_5 with
``history_n=∞``, qwen3_vl with ``full_history_size=∞``), so a length-K segment
trains on K assistant spans in one forward.

For non-extensional protocols (Markov, sliding-window, fold-mid-trajectory),
the segmenter falls back to length-1 segments.

The processor is still loaded here — ``build_segment_samples`` re-tokenizes
each step's stored ``prompt`` WITH its images to produce ``pixel_values`` and
the vision-expanded ``input_ids`` — but it never touches ``apply_chat_template``
(and thus never needs ``enable_thinking``): the rendered prompt string is frozen
in the parquet. The Nth ``image_indices`` entry supplies the Nth
processor-owned image slot in that frozen prompt; producer/export code owns
validation and ordering before this rollout consumes the indices verbatim.

Slime config (in scripts/train/run_sft*.sh):
  --rollout-function-path lite.train.rollout.sft.generate_rollout
  --input-key steps                 -> sample.prompt = list[LiteRLStep struct]
  --metadata-key processed_images   -> sample.metadata = list[bytes | None]
  (the parquet ``metadata`` column stays provenance-only)
  (NO --apply-chat-template, --multimodal-keys — we handle everything here.)
"""

from __future__ import annotations

import copy
import logging

from slime.utils.processing_utils import load_processor

from lite.core.samples import LiteRLSample
from lite.train.export.sft_tokenize import deserialize_rl_step
from lite.train.rollout.core import build_segment_samples, dummy_sample
from lite.utils.image import decode_hf_images

# Slime API: scripts/train/run_sft.sh resolves this module's public symbol by
# dotted string. Keep ``__all__`` and the launcher flags in sync when renaming.
# SFT exposes only ``generate_rollout``; the RL shims also expose ``generate``
# and ``convert_samples_to_train_data``.
__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

PROCESSOR = None
SAMPLE_PRINTED = False


class _SFTState:
    """Stub of the slime ``GenerateState``. ``build_segment_samples`` reads
    ``aborted`` (always False here — SFT doesn't abort mid-rollout) and
    ``args`` (only when ``CUA_LITE_MULTIMODAL_LAZY_EXPAND=1`` triggers the
    fail-fast assert; otherwise unused)."""
    aborted = False

    def __init__(self, args=None):
        self.args = args


# Slime entrypoint: selected by ``--rollout-function-path``.
def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """SFT rollout: decode a trajectory row → deserialize its pre-tokenized
    steps into a ``LiteRLSample`` → fan out via ``build_segment_samples`` (same
    radix path RL uses) → emit one ``slime.Sample`` per segment."""
    if evaluation:
        raise ValueError("SFT rollout does not support evaluation mode")
    if not args.rollout_global_dataset:
        raise ValueError("SFT rollout requires rollout_global_dataset=True")

    global PROCESSOR, SAMPLE_PRINTED
    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    rows = data_buffer.get_samples(args.rollout_batch_size)
    out_samples: list = []
    state = _SFTState(args=args)

    for i, sample in enumerate(rows):
        (sample,) = sample

        # Empty processed_images is fine for text-only trajectories (e.g.
        # browsergym text_only.yaml WA/MiniWoB/AB) — the downstream tokenize-only
        # branch in `build_segment_samples` handles `image_indices=()` and emits
        # `multimodal_train_inputs=None`.
        processed_images = decode_hf_images(sample.metadata or []) or []

        # `steps` are serialized LiteRLStep structs (tokenized at export). A
        # parquet from BEFORE this refactor stores `steps` as message-dict lists
        # (each step is a list, not a struct); fail loud with a fixit rather than
        # a cryptic TypeError deep in deserialize_rl_step.
        raw_steps = sample.prompt or []
        if raw_steps and not isinstance(raw_steps[0], dict):
            raise ValueError(
                f"row {i}: SFT parquet 'steps' is the legacy message-list schema; "
                f"re-export with the current export_sft (--model-id required) — steps are "
                f"now pre-tokenized LiteRLStep structs."
            )
        steps = [deserialize_rl_step(s) for s in raw_steps]

        # Episode outcome rides on the per-step ``status`` the exporter stamped
        # (``build_segment_samples`` takes the max severity over a segment), so
        # the sample-level terminated/truncated flags stay at their defaults —
        # the SFT parquet has no separate trajectory-outcome column to fill them
        # from, and nothing in the train path reads them.
        rl_sample = LiteRLSample(
            processed_images=processed_images,
            steps=steps,
            episode_return=0.0,
        )
        if not rl_sample.steps:
            # Degenerate row → emit one zero-gradient dummy so SFT's native
            # converter keeps the expected group count. Clear metadata first:
            # here it carries PNG bytes, while ``dummy_sample`` expects a dict.
            cleared = copy.copy(sample)
            cleared.metadata = {}
            out_samples.append(dummy_sample(cleared))
            continue

        # Build a clean ``original_sample`` — slime's SFT config repurposes
        # ``Sample.metadata`` to carry the PNG-bytes list (via
        # ``--metadata-key processed_images``), but ``_emit_rl_sample``
        # spreads ``original_sample.metadata`` as kwargs into the emitted
        # ``Sample.metadata`` dict. So we hand it a copy with metadata
        # cleared to a (dict) so the spread is well-defined; the actual
        # PNG bytes are held in ``rl_sample.processed_images`` already.
        original_sample = copy.copy(sample)
        original_sample.metadata = {}

        # Same radix path as GRPO / REINFORCE — packs prefix-extensional
        # segments into a single Sample with multi-span loss_mask.
        segment_samples = build_segment_samples(
            rl_sample=rl_sample,
            original_sample=original_sample,
            processor=PROCESSOR,
            tokenizer=PROCESSOR.tokenizer,
            state=state,
        )

        out_samples.extend(segment_samples)

        if i == 0 and not SAMPLE_PRINTED and segment_samples:
            first = segment_samples[0]
            n_steps = len(rl_sample.steps)
            n_segs = len(segment_samples)
            logger.info(
                "rollout_sft (radix): row 0: T=%d steps → %d segments; "
                "first-segment prompt=%d response=%d total=%d images=%d",
                n_steps, n_segs,
                len(first.tokens) - first.response_length,
                first.response_length,
                len(first.tokens),
                # Non-None only: the list is index-aligned with the source row, so
                # orphan frames (a batch's non-final slots) occupy NULL slots and
                # its length is a slot count, not an image count.
                sum(1 for img in processed_images if img is not None),
            )
            SAMPLE_PRINTED = True

    return out_samples
