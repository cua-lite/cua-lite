"""Claude image sizing helpers used before provider API calls.

Anthropic's vision encoder works in 28px patches with model-specific long-edge
and patch-count limits. CUA-Lite resizes before the request so coordinates use
the sent image size; legacy computer tools also declare that size explicitly.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ClaudeImageLimits:
    max_edge_px: int
    max_tokens: int
    px_per_token: int = 28


CLAUDE_IMAGE_LIMITS = {
    "high_res": _ClaudeImageLimits(max_edge_px=2576, max_tokens=4784),
    "default": _ClaudeImageLimits(max_edge_px=1568, max_tokens=1568),
}

#: A request carrying SEVERAL images is held to a stricter per-image ceiling than
#: a single-image one: Anthropic rejects the whole request with
#: ``"At least one of the image dimensions exceed max allowed size for
#: many-image requests: 2000 pixels"``. Only the high-res profile can exceed it
#: (2576), and only for a long edge in (2000, 2576] — a 1080x2400 phone
#: screenshot lands exactly there, while 1920x1080 desktop never does.
MANY_IMAGE_MAX_EDGE_PX = 2000


def get_claude_image_limits(model_id: str) -> _ClaudeImageLimits:
    if re.search(r"(?:^|/)claude-opus-(?:4-(?:7|8)|5)$", model_id, re.IGNORECASE):
        return CLAUDE_IMAGE_LIMITS["high_res"]
    return CLAUDE_IMAGE_LIMITS["default"]


def effective_max_edge_px(model_id: str, *, many_image: bool) -> int:
    """Long-edge ceiling for one image, given how many the request will carry.

    ``many_image`` is a per-TRAJECTORY fact, not a per-request one, and callers
    must resolve it once: the computer-use coordinate frame IS the sent size
    (see the module docstring), so a ceiling that changed as history grew would
    silently move the frame mid-episode. An agentic ``use`` trajectory
    accumulates history and is therefore many-image from its first turn; a
    single-step grounding/understanding trajectory never is and keeps the full
    single-image ceiling.
    """
    max_edge = get_claude_image_limits(model_id).max_edge_px
    return min(max_edge, MANY_IMAGE_MAX_EDGE_PX) if many_image else max_edge


def n_tokens_for_px(px: int, px_per_token: int = 28) -> int:
    """Ceil-divide a pixel span by the provider patch size."""
    if px <= 0:
        raise ValueError(f"image dimension must be positive, got {px}")
    return (px - 1) // px_per_token + 1


def n_tokens_for_image(w: int, h: int, *, px_per_token: int = 28) -> int:
    return n_tokens_for_px(w, px_per_token) * n_tokens_for_px(h, px_per_token)


def would_trigger_claude_auto_downsample(
    w: int, h: int, model_id: str, *, many_image: bool
) -> bool:
    limits = get_claude_image_limits(model_id)
    return (
        max(w, h) > effective_max_edge_px(model_id, many_image=many_image)
        or n_tokens_for_image(w, h, px_per_token=limits.px_per_token) > limits.max_tokens
    )


def target_image_size(
    width: int, height: int, model_id: str, *, many_image: bool
) -> tuple[int, int]:
    """Largest aspect-preserving size the provider accepts as-is.

    ``many_image`` is REQUIRED, with no default in either direction: guessing
    ``True`` silently costs a single-image caller 22% of its coordinate frame
    (2576 -> 2000) with no error, and guessing ``False`` emits a request the
    provider rejects. Declare the regime; :func:`effective_max_edge_px` explains
    why it is a per-trajectory fact.
    """
    limits = get_claude_image_limits(model_id)
    max_edge = effective_max_edge_px(model_id, many_image=many_image)
    if not would_trigger_claude_auto_downsample(width, height, model_id, many_image=many_image):
        return width, height

    if height > width:
        w, h = target_image_size(height, width, model_id, many_image=many_image)
        return h, w

    aspect = width / height
    lo, hi = 1, width
    while lo + 1 < hi:
        mid_w = (lo + hi) // 2
        mid_h = max(round(mid_w / aspect), 1)
        # ``mid_w`` alone bounds the long edge: the caller swapped so width >= height
        # above, so ``mid_h = round(mid_w / aspect) <= mid_w`` for aspect >= 1.
        valid = (
            mid_w <= max_edge
            and n_tokens_for_image(mid_w, mid_h, px_per_token=limits.px_per_token)
            <= limits.max_tokens
        )
        if valid:
            lo = mid_w
        else:
            hi = mid_w
    return lo, max(round(lo / aspect), 1)


def downscale_to_target(png: bytes, target: tuple[int, int]) -> tuple[str, int, int]:
    """Resize to exactly ``target``.

    Takes raw encoded screenshot bytes (``obs.image``) and decodes once. Returns
    ``(b64, w, h)`` because the output feeds straight into the Claude API
    data-URI payload, which every call site labels ``image/png``.

    ``obs.image`` is not guaranteed PNG — mobilegym emits JPEG — so the
    already-at-target path re-encodes anything else rather than passing the
    source bytes through under a PNG label. Anthropic validates the declared
    media type against the magic bytes and rejects the mismatch with a 400.
    """
    img = Image.open(io.BytesIO(png))
    w, h = img.size
    if (w, h) == target:
        if img.format == "PNG":
            return base64.b64encode(png).decode("utf-8"), w, h
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8"), w, h
    img = img.resize(target, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    logger.info("Claude resize: %dx%d -> %dx%d", w, h, target[0], target[1])
    return base64.b64encode(buf.getvalue()).decode("utf-8"), target[0], target[1]


def resize_for_claude_api(
    png: bytes,
    model_id: str,
    target: tuple[int, int] | None = None,
    *,
    many_image: bool,
) -> tuple[str, int, int]:
    """Encode the screenshot at the size Claude will actually see.

    With ``target=None`` (the default), choose the largest aspect-preserving
    size allowed by the Anthropic vision limits. A non-``None`` target is an
    explicit operator override and is encoded exactly.

    ``many_image`` selects the ceiling (see :func:`effective_max_edge_px`) and
    must be constant for a trajectory, since the sent size is the coordinate
    frame.
    """
    img = Image.open(io.BytesIO(png))
    src_w, src_h = img.size
    out_target = target or target_image_size(src_w, src_h, model_id, many_image=many_image)
    out_b64, out_w, out_h = downscale_to_target(png, out_target)
    if would_trigger_claude_auto_downsample(out_w, out_h, model_id, many_image=many_image):
        raise ValueError(
            f"Claude image target {out_w}x{out_h} still triggers provider "
            f"auto-downsample for model={model_id} (many_image={many_image})"
        )
    return out_b64, out_w, out_h


__all__ = [
    "CLAUDE_IMAGE_LIMITS",
    "MANY_IMAGE_MAX_EDGE_PX",
    "downscale_to_target",
    "effective_max_edge_px",
    "get_claude_image_limits",
    "n_tokens_for_image",
    "n_tokens_for_px",
    "resize_for_claude_api",
    "target_image_size",
    "would_trigger_claude_auto_downsample",
]
