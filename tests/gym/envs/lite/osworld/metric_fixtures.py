"""Metric stand-ins resolved by NAME inside the isolation child process.

``runner`` runs metrics in a subprocess, so a test cannot inject one by patching the production
metrics module in its own process — the child re-imports and would not see it. Tests point
``runner._METRIC_MODULES`` at this module instead, which is the same seam production uses to say
where metric names resolve.
"""


def unit_score(_result, _expected, *, value):
    """Return the score handed to it — lets a test drive aggregation with exact values."""
    return value


def spin(seconds, *_args, **_kwargs) -> float:
    """A pure-Python CPU burn — the shape of fastdtw inside ``compare_audios``."""
    import time

    end = time.monotonic() + float(seconds)
    while time.monotonic() < end:
        pass
    return 1.0


def explode(*_args, **_kwargs):
    raise ValueError("boom")
