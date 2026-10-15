"""Response-record, failure, and telemetry helpers for GPT Responses-API agents.

Owns the normalized response record every GPT caller reads, the failed-response
error surface, and prompt-cache/usage telemetry. Output-item parsing and
provider-call provenance live in ``lite.agents.models.gpt.utils.parse``; image
resizing and processed-dimension fetches in
``lite.agents.models.gpt.utils.image_io``; provider-history assembly,
compaction, and log redaction in ``lite.agents.models.gpt.utils.history``.
Agent classes import this module directly; package init should not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GPTResponseRecord:
    """Normalized Responses API response fields used by the GPT agent loop."""

    response_id: str | None
    status: str | None
    error: Any
    output_items: list[dict[str, Any]]
    finish_reason: str | None
    usage: dict[str, Any]


def _normalized_gpt_response(response: Any) -> _GPTResponseRecord:
    """Normalize the Responses API wrapper into one local record.

    This is the only place that handles the provider boundary shape: liteLLM
    returns a Responses model, tests and replay fixtures pass the equivalent
    plain dict. Everything downstream reads the record fields, so no other GPT
    code re-sniffs the wrapper.
    """
    data = response if isinstance(response, dict) else response.model_dump()
    incomplete_details = data.get("incomplete_details") or {}
    return _GPTResponseRecord(
        response_id=data.get("id"),
        status=data.get("status"),
        error=data.get("error"),
        output_items=data.get("output") or [],
        finish_reason=incomplete_details.get("reason"),
        usage=data.get("usage") or {},
    )


#: Responses-API ``incomplete_details.reason`` values that mean the model hit a
#: token limit rather than naturally finishing.
_TRUNCATION_INCOMPLETE_REASONS = ("max_output_tokens", "max_tokens", "length")


class ResponseAPIError(RuntimeError):
    """Raised when the Responses API returns a failed / errored result.

    counterpart to openai-cua-sample-app's ``ensureResponseSucceeded``
    (``responses-loop.ts``). Lets the sample loop fail fast rather than
    silently treating ``status == "failed"`` as "model finished with no
    output" — which would otherwise cause premature termination.
    """


def _is_image_limit_error(error: Exception) -> bool:
    """Whether the provider rejected the accumulated screenshot count."""
    return (
        getattr(error, "status_code", None) == 400
        and "maximum number of images" in str(error).lower()
    )


def _raise_if_response_failed(response: _GPTResponseRecord) -> None:
    """Raise ``ResponseAPIError`` if the Responses API response is failed."""
    if response.status == "failed" or response.error:
        raise ResponseAPIError(
            f"Responses API failed: status={response.status!r} error={response.error!r}"
        )


# Minimum input_tokens for which Azure / OpenAI's prefix cache activates —
# below this the cache returns 0 by design (per OpenAI docs: ~1024-token
# minimum prefix). Below this size, cache=0 is not a problem to warn about.
_CACHE_MIN_TOKENS = 1024


@dataclass
class _CacheStats:
    """Per-trajectory accumulator for prompt-cache + token-usage telemetry."""

    n_turns: int = 0
    total_input: int = 0
    total_cached: int = 0
    total_output: int = 0
    # Chained, large-enough requests that returned no cached tokens.
    unexpected_misses: int = 0


def _log_usage_and_check_cache(
    response: _GPTResponseRecord,
    turn_idx: int,
    *,
    chained: bool,
    stats: _CacheStats | None = None,
) -> None:
    """Per-turn cache + usage observability.

    Behavior:
        * DEBUG: per-turn `turn=N input=... cached=...` line (off by default).
        * WARNING: ``cached==0`` mid-chain with input > _CACHE_MIN_TOKENS —
          typically a load-balancer reroute to a cold-cache server.
        * Accumulates into ``stats`` so the sample loop can log a single
          INFO-level summary at trajectory end.

    Azure / OpenAI Responses API returns ``usage.input_tokens_details.cached_tokens``
    indicating how many prefix tokens were reused from the prompt cache.
    """
    usage = response.usage
    in_tok = usage.get("input_tokens") or 0
    out_tok = usage.get("output_tokens") or 0
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    hit_rate = (cached / in_tok * 100.0) if in_tok else 0.0
    logger.debug(
        "turn=%d input=%d cached=%d (%.1f%%) output=%d",
        turn_idx,
        in_tok,
        cached,
        hit_rate,
        out_tok,
    )
    unexpected_miss = chained and turn_idx >= 1 and cached == 0 and in_tok > _CACHE_MIN_TOKENS
    if unexpected_miss:
        logger.warning(
            "turn=%d Azure prompt-cache MISS despite chain_previous_response=True "
            "(input=%d, cached=0). Likely a transient load-balancer reroute to a "
            "cold-cache server. If sustained over many turns, check region / "
            "prompt_cache_key config.",
            turn_idx,
            in_tok,
        )
    if stats is not None:
        stats.n_turns += 1
        stats.total_input += in_tok
        stats.total_cached += cached
        stats.total_output += out_tok
        if unexpected_miss:
            stats.unexpected_misses += 1


def _log_trajectory_usage_summary(stats: _CacheStats) -> None:
    """One-line INFO summary of token + cache usage over the whole trajectory."""
    if stats.n_turns == 0:
        return
    hit_rate = (stats.total_cached / stats.total_input * 100.0) if stats.total_input else 0.0
    logger.info(
        "trajectory: turns=%d input=%d cached=%d (%.1f%%) output=%d unexpected_cache_misses=%d",
        stats.n_turns,
        stats.total_input,
        stats.total_cached,
        hit_rate,
        stats.total_output,
        stats.unexpected_misses,
    )
