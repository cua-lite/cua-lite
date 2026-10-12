"""Provider-call retry policy for the agent model families.

Owns the one capped exponential backoff formula the GPT and Claude agents wrap
around their paid provider entrypoints (``litellm.aresponses`` /
``litellm.acompletion``). Retry policy is runtime behavior — latency, cost, and
failure visibility — so it lives with the provider-independent agent runtime,
not in the low-semantic ``lite.utils`` namespace.

This module is agent-side only. ``lite/gym`` never imports ``lite/agents``, so
env-side judges keep their own local retry policy.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

#: Backoff ceiling for callers that do not name their own.
DEFAULT_MAX_DELAY = 60.0


def retry_delay(
    attempt: int,
    *,
    base_delay: float = 1.0,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> float:
    """Seconds to sleep before retrying attempt ``attempt`` (0-based).

    ``min(max_delay, base_delay * 2**attempt)`` scaled by U(0.5, 1.5).
    """
    return min(max_delay, base_delay * (2**attempt)) * (0.5 + random.random())


async def acompletion_with_retry(
    call: Callable[..., Awaitable[Any]],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    log_name: str = "llm-call",
    retry_if: Callable[[Exception], bool] | None = None,
    **api_kwargs: Any,
) -> Any:
    """Capped exp backoff + jitter wrapper around an async LLM ``call``.

    Retries ``call(**api_kwargs)`` on any Exception up to ``max_retries``
    attempts (total ``max_retries + 1`` calls). The per-attempt sleep is
    ``retry_delay(attempt, base_delay=..., max_delay=...)``.

    ``call`` is the provider entrypoint (e.g. ``litellm.acompletion`` /
    ``litellm.aresponses``), injected so families share one retry loop.
    ``retry_if`` lets the caller handle errors that require changing the request.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await call(**api_kwargs)
        except Exception as e:  # noqa: BLE001 — retry is broad by design
            if retry_if is not None and not retry_if(e):
                raise
            last_exc = e
            if attempt >= max_retries:
                break
            delay = retry_delay(attempt, base_delay=base_delay, max_delay=max_delay)
            logger.warning(
                "%s failed (attempt %d/%d): %s; sleeping %.2fs",
                log_name,
                attempt + 1,
                max_retries + 1,
                e,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


__all__ = [
    "acompletion_with_retry",
]
