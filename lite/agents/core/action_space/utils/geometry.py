"""Adapter-local action-space helpers.

Provider-free geometry and batch helpers are imported from
``lite.core.tools.action_space`` directly by callers.
"""

from __future__ import annotations

import math
from typing import Any

from lite.agents.core.action_space.errors import ModelToolCallParseError

# Screen units <-> wheel clicks, and the boundary between the two spellings a
# qwen-style ``pixels`` field can carry. One number for both: 100 screen units
# is one click (on a 1000-tall screen), and a value below that cannot have been
# meant as screen units. Tune it here, for everyone, or not at all.
PIXELS_PER_CLICK = 100


def _finite_float(val: Any) -> float:
    value = float(val)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def _parse_coord(val: Any) -> list[int]:
    if isinstance(val, str):
        val = val.strip().strip("[]()").split(",")
    try:
        return [int(_finite_float(str(c).strip())) for c in val]
    except (ValueError, TypeError, OverflowError) as exc:
        raise ModelToolCallParseError(
            "coordinate must contain only finite numeric values"
        ) from exc


def _check_coord_dimensions(
    parsed: list[int],
    *,
    dimensions: int | None,
    name: str,
) -> list[int]:
    if dimensions is not None and len(parsed) != dimensions:
        raise ModelToolCallParseError(
            f"{name} must contain exactly {dimensions} numeric values; "
            f"got {len(parsed)}"
        )
    return parsed


def required_coord(
    val: Any,
    *,
    dimensions: int | None = None,
    name: str = "coordinate",
) -> list[int]:
    """Parse a required coordinate or raise a model-output parse error.

    ``dimensions`` lets callers distinguish two-point coordinates from bboxes.
    """
    if val is None:
        raise ModelToolCallParseError(f"{name} is required")
    return _check_coord_dimensions(
        _parse_coord(val),
        dimensions=dimensions,
        name=name,
    )


def optional_coord(val: Any, *, dimensions: int | None = None) -> list[int] | None:
    """Parse an optional coordinate; missing or malformed values return ``None``."""
    if val is None:
        return None
    try:
        return required_coord(val, dimensions=dimensions)
    except ModelToolCallParseError:
        return None


def model_keys(raw: Any, *, action: str) -> list[str]:
    """``args["keys"]`` as canonical key tokens, or a MODEL-VISIBLE parse error.

    Two things the canonical constructor cannot do on its own:

    * A ``+``-joined chord inside a LIST element (``["ctrl+a"]``) is split, the
      way upstream's ``parse_keys`` does. ``normalize_keys`` only splits a bare
      string, so the list spelling reached it as one unknown token.
    * The failure is raised as :class:`ModelToolCallParseError`. ``keys.py``
      lives in ``lite.core`` and can only raise a bare ``ValueError``, which
      escapes the agent's parse boundary and destroys the whole trajectory
      instead of being handed back to the model as feedback.
    """
    from lite.core.tools.action_space.keys import normalize_keys

    if isinstance(raw, list):
        tokens: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise ModelToolCallParseError(
                    f"{action}: 'keys' must be strings; "
                    f"got {type(item).__name__}"
                )
            tokens.append(item)
        raw = "+".join(tokens) if tokens else raw
    try:
        return normalize_keys(raw)
    except (TypeError, ValueError) as exc:
        raise ModelToolCallParseError(
            f"{action}: invalid 'keys': {exc}"
        ) from exc


def model_duration(args: dict[str, Any], *, action: str, default: float) -> float:
    """``args["time"]`` as a float, or a MODEL-VISIBLE parse error.

    The XML coercion hands a non-numeric ``time`` back as a raw string (a quoted
    ``"1.0"`` included), so a bare ``float()`` here raised ``ValueError`` past
    the agent's parse boundary and lost the trajectory.
    """
    raw = args.get("time", default)
    try:
        return _finite_float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelToolCallParseError(
            f"{action}: needs a numeric 'time'; got {raw!r}"
        ) from exc


def required_model_text(args: dict[str, Any], *, action: str) -> str:
    """``args["text"]`` for a typing action, or raise instead of inventing one.

    Absent is MALFORMED, not empty: every wire that carries a ``type`` declares
    ``text`` required for it, so a missing key means the reply lost the
    parameter (a dropped ``<parameter=text>`` opener does exactly that). Passing
    ``""`` through instead recorded a successfully executed action that typed
    nothing -- the env moved no state and the model was told nothing, so it had
    to infer the failure from the next screenshot.

    An explicitly empty ``text`` is the model's own choice and passes; the
    annotate pass strips it as a no-op the way it strips ``screenshot``.
    """
    if "text" not in args:
        raise ModelToolCallParseError(f"{action}: requires the 'text' argument")
    return str(args["text"] or "")


def required_scroll_pixels(args: dict[str, Any], action: str) -> int:
    """``args["pixels"]`` as a nonzero int, or raise instead of inventing one.

    The sign of this value is the ONLY carrier of the scroll direction, so a
    missing, non-numeric or ZERO value all fail the same way. Zero used to fall
    through to the magnitude conversion, where ``max(1, ...)`` turned "scroll
    nothing" into a real one-click scroll UP -- an action the model never asked
    for, executed and recorded as if it had.
    """
    if "pixels" not in args:
        raise ModelToolCallParseError(
            f"{action}: requires the 'pixels' argument "
            "(negative = down/left, positive = up/right)"
        )
    raw = args["pixels"]
    try:
        pixels = int(_finite_float(raw))
    except (TypeError, ValueError, OverflowError):
        raise ModelToolCallParseError(
            f"{action}: needs a numeric 'pixels'; "
            "got a non-numeric or non-finite value"
        ) from None
    if pixels == 0:
        raise ModelToolCallParseError(
            f"{action}: 'pixels' must be nonzero "
            "(negative = down/left, positive = up/right)"
        )
    return pixels


def scroll_clicks(scroll_pixels: int) -> int:
    """Wheel clicks meant by a family's wire magnitude.

    The same prompt yields two units -- a raw notch count or screen units -- and
    only the magnitude discriminates them, so a value below one click's worth of
    screen is read as notches. Inverse of :func:`scroll_wire_magnitude`.
    """
    magnitude = abs(scroll_pixels)
    if magnitude < PIXELS_PER_CLICK:
        return round(magnitude)
    return round(magnitude / PIXELS_PER_CLICK)


def scroll_wire_magnitude(clicks: int, *, wire_unit: int) -> int:
    """The wire magnitude :func:`scroll_clicks` reads back as ``clicks``.

    ``wire_unit`` is the spelling the family itself writes: 1 for the notch
    writers, :data:`PIXELS_PER_CLICK` for the screen-unit ones. A notch spelling
    stops being readable at the boundary -- ``100`` would come back as ONE -- so
    from there on this switches to screen units, which is also what a family
    writing that many clicks actually emits.

    Magnitude in, magnitude out: ``direction`` carries the heading and every
    caller applies the sign itself. Taking ``abs`` here rather than trusting the
    sign is what keeps the two functions inverse over the whole domain.

    Zero is refused rather than rendered. ``required_scroll_pixels`` -- the read
    side in this same module -- rejects a zero wire value because the sign is the
    only carrier of direction, so emitting ``0`` here would write a target that
    the family's own parser is defined to refuse.
    """
    if clicks == 0:
        raise ValueError(
            "scroll amount must be nonzero: a zero-click scroll has no direction "
            "to render and 'pixels' == 0 is refused on the read side"
        )
    magnitude = abs(clicks) * wire_unit
    if magnitude >= PIXELS_PER_CLICK:
        magnitude = abs(clicks) * PIXELS_PER_CLICK
    return magnitude


def compact_number(val: Any) -> Any:
    """Format a numeric value without the trailing ``.0`` when integer-valued."""
    if val is None:
        return None
    try:
        f = _finite_float(val)
        i = int(f)
    except (TypeError, ValueError, OverflowError):
        return val
    return i if f == i else f


__all__ = [
    "PIXELS_PER_CLICK",
    "model_duration",
    "required_model_text",
    "model_keys",
    "compact_number",
    "optional_coord",
    "required_coord",
    "required_scroll_pixels",
]
