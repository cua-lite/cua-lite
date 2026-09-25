"""Claude image sizing follows Anthropic vision limits."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from lite.agents.models.claude.utils.image_io import (
    MANY_IMAGE_MAX_EDGE_PX,
    effective_max_edge_px,
    n_tokens_for_image,
    resize_for_claude_api,
    target_image_size,
    would_trigger_claude_auto_downsample,
)


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(32, 64, 96)).save(buf, format="PNG")
    return buf.getvalue()


def _valid_default(width: int, height: int) -> bool:
    return width <= 1568 and height <= 1568 and n_tokens_for_image(width, height) <= 1568


def test_small_image_stays_native_size() -> None:
    assert target_image_size(800, 600, "claude-opus-4-6", many_image=True) == (800, 600)
    assert not would_trigger_claude_auto_downsample(800, 600, "claude-opus-4-6", many_image=True)


def test_desktop_hd_uses_largest_valid_aspect_preserving_size() -> None:
    assert target_image_size(1920, 1080, "claude-opus-4-6", many_image=True) == (1456, 819)

    width, height = target_image_size(1568, 1014, "claude-opus-4-6", many_image=True)
    assert (width, height) != (1568, 1014)
    assert _valid_default(width, height)
    assert not _valid_default(width + 1, round((width + 1) * 1014 / 1568))


def test_portrait_transposes_the_same_sizing_rule() -> None:
    landscape = target_image_size(3000, 2000, "claude-opus-4-6", many_image=True)
    portrait = target_image_size(2000, 3000, "claude-opus-4-6", many_image=True)
    assert portrait == (landscape[1], landscape[0])


def test_opus_4_7_tier_keeps_larger_frames_when_within_budget() -> None:
    # 1920x1080 sits under BOTH ceilings, so it is kept in either regime.
    assert target_image_size(1920, 1080, "claude-opus-4-7", many_image=False) == (1920, 1080)
    assert target_image_size(1920, 1080, "claude-opus-4-7", many_image=True) == (1920, 1080)
    # 2560x1440 exceeds the many-image cap: kept whole only for a single-image
    # request, clamped once the request carries several images.
    assert target_image_size(2560, 1440, "claude-opus-4-7", many_image=False) == (2560, 1440)
    assert max(target_image_size(2560, 1440, "claude-opus-4-7", many_image=True)) <= 2000


@pytest.mark.parametrize(
    "model_id", ["claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "anthropic/claude-sonnet-5"]
)
def test_high_res_models_keep_larger_frames_when_within_budget(model_id: str) -> None:
    # 1920x1080 sits under BOTH ceilings, so it is kept in either regime.
    assert target_image_size(1920, 1080, model_id, many_image=False) == (1920, 1080)
    assert target_image_size(1920, 1080, model_id, many_image=True) == (1920, 1080)
    # 2560x1440 exceeds the many-image cap: kept whole only for a single-image
    # request, clamped once the request carries several images.
    assert target_image_size(2560, 1440, model_id, many_image=False) == (2560, 1440)
    assert max(target_image_size(2560, 1440, model_id, many_image=True)) <= 2000


def test_opus_4_8_tier_downsizes_frames_over_high_res_budget() -> None:
    target = target_image_size(4000, 2500, "claude-opus-4-8", many_image=True)

    assert target != (4000, 2500)
    assert not would_trigger_claude_auto_downsample(*target, "claude-opus-4-8", many_image=True)


@pytest.mark.parametrize("model_id", ["claude-opus-5", "claude-sonnet-5"])
def test_high_res_phone_keeps_more_detail(model_id: str) -> None:
    assert effective_max_edge_px(model_id, many_image=True) == 2000
    assert target_image_size(1080, 2400, model_id, many_image=True)[1] == 2000


def test_resize_returns_b64_and_sent_dimensions() -> None:
    encoded, width, height = resize_for_claude_api(
        _png(1920, 1080),
        "claude-opus-4-6",
        many_image=True,
    )

    assert (width, height) == (1456, 819)
    assert Image.open(io.BytesIO(base64.b64decode(encoded))).size == (1456, 819)


def test_explicit_target_is_still_honored() -> None:
    encoded, width, height = resize_for_claude_api(
        _png(800, 600),
        "claude-opus-4-6",
        target=(1024, 768),
        many_image=True,
    )

    assert (width, height) == (1024, 768)
    assert Image.open(io.BytesIO(base64.b64decode(encoded))).size == (1024, 768)


def test_explicit_target_that_would_auto_downsample_is_rejected() -> None:
    with pytest.raises(ValueError, match="auto-downsample"):
        resize_for_claude_api(
            _png(800, 600),
            "claude-opus-4-6",
            target=(1920, 1080),
            many_image=True,
        )


# ---------------------------------------------------------------------------
# The many-image ceiling: a request carrying several images is held to a
# stricter per-image limit than a single-image one.
# ---------------------------------------------------------------------------

#: A 1080x2400 phone screenshot is the shape that exposed this: its long edge
#: falls in (2000, 2576], so the high-res profile leaves it untouched while
#: Anthropic rejects it as soon as history makes the request many-image.
_PHONE = (1080, 2400)
_DESKTOP = (1920, 1080)


def test_many_image_ceiling_only_narrows_the_high_res_profile() -> None:
    # 2576 is above the many-image cap, so it is clamped; 1568 is already below.
    assert effective_max_edge_px("claude-opus-4-8", many_image=False) == 2576
    assert effective_max_edge_px("claude-opus-4-8", many_image=True) == MANY_IMAGE_MAX_EDGE_PX
    assert effective_max_edge_px("claude-opus-4-6", many_image=False) == 1568
    assert effective_max_edge_px("claude-opus-4-6", many_image=True) == 1568


def test_tall_phone_screenshot_is_capped_only_in_the_many_image_regime() -> None:
    single = target_image_size(*_PHONE, "claude-opus-4-8", many_image=False)
    many = target_image_size(*_PHONE, "claude-opus-4-8", many_image=True)

    assert single == _PHONE, "single-image requests keep the full high-res ceiling"
    assert max(many) <= MANY_IMAGE_MAX_EDGE_PX, f"{many} still exceeds the many-image cap"
    assert many[0] / many[1] == pytest.approx(_PHONE[0] / _PHONE[1], rel=1e-2)


def test_desktop_screenshot_is_unaffected_by_the_many_image_ceiling() -> None:
    """1920x1080 sits below both ceilings, so nothing about desktop changes."""
    for many_image in (False, True):
        assert target_image_size(*_DESKTOP, "claude-opus-4-8", many_image=many_image) == _DESKTOP


def test_many_image_must_be_declared() -> None:
    """No default in either direction: guessing True silently shrinks a
    single-image caller's frame, guessing False emits a rejected request."""
    with pytest.raises(TypeError, match="many_image"):
        target_image_size(*_PHONE, "claude-opus-4-8")
    with pytest.raises(TypeError, match="many_image"):
        resize_for_claude_api(_png(*_DESKTOP), "claude-opus-4-8")
    assert would_trigger_claude_auto_downsample(*_PHONE, "claude-opus-4-8", many_image=True) is True


@pytest.mark.parametrize("source_format", ["JPEG", "PNG"])
@pytest.mark.parametrize("target", [None, (960, 540)])
def test_webp_encoding_preserves_sent_pixels(source_format, target) -> None:
    """The transport encoding changes neither the coordinate frame nor pixels."""
    image = Image.new("RGB", _DESKTOP, (32, 64, 96))
    image.paste((201, 39, 82), (300, 250, 1200, 900))
    source = io.BytesIO()
    image.save(source, format=source_format)
    expected = Image.open(io.BytesIO(source.getvalue()))
    if target is not None:
        expected = expected.resize(target, Image.Resampling.LANCZOS)
    encoded, w, h = resize_for_claude_api(
        source.getvalue(), "claude-opus-4-8", target=target, many_image=True,
    )
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
        assert decoded.format == "WEBP"
        assert decoded.size == (w, h) == expected.size
        assert decoded.tobytes() == expected.tobytes()
