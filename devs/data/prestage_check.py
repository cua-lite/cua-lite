"""Run ``stage``'s canonical row gate over rollout log-roots, before staging.

``lite.data.hf.stage`` validates every row it emits and raises on the first bad
one -- but only after copying that root's screenshots into the image store, so a
single unpublishable trajectory ends the run tens of minutes and tens of GB in,
with nothing staged. This walks the same roots, builds the same ``out_row``, and
calls the same :func:`validate_canonical_rows`, so a root that passes here stages
to completion.

This is NOT a second row validator: it owns no rules and repairs nothing. It
calls the gate's own entrypoint earlier and in parallel, and prints what stage
would have died on. ``stage`` remains the gate -- if the two disagree, ``stage``
is right and this file is stale.

That is also why :func:`check_one` imports ``stage``'s private row-preparation
helpers rather than restating them: coupling to the owner is the point, because a
copy would drift into a second, wrong contract. If those names move, follow them
here or delete this file.

Usage mirrors the stage command it precedes::

    uv run python devs/data/prestage_check.py --log-roots <root> [<root> ...]

Exits 0 when every row is publishable, 1 otherwise, naming each offending
trajectory so it can be set aside before the real stage run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from lite.core.metadata import metadata_from_dict
from lite.data.hf.stage import (
    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY,
    _apply_stop_reason_publication_policy,
)
from lite.data.staging import coerce_messages
from lite.data.utils.rows import validate_canonical_rows


def check_one(trajectory_path: str) -> tuple[str, str] | None:
    """``(path, reason)`` when *trajectory_path* would fail stage, else ``None``.

    Mirrors ``stage``'s per-row preparation: canonicalize the metadata, coerce the
    messages, apply the stop-reason publication policy, drop the private runtime
    sidecars. The image list is passed through as stored -- the gate's failures are
    message- and metadata-shaped, and the ImageStore rewrite stage does here is
    exactly the expensive part this check exists to skip.

    The catch is deliberately broad: the gate raises whatever the violated rule
    raises, and reporting every offender in one pass is this file's whole purpose.
    """
    try:
        row = pd.read_parquet(trajectory_path).iloc[0]
        metadata = row["metadata"]
        metadata = metadata_from_dict(
            json.loads(metadata) if isinstance(metadata, str) else metadata
        ).to_dict()
        messages = coerce_messages(row["messages"])
        _apply_stop_reason_publication_policy(metadata)
        for message in messages:
            if isinstance(message, dict):
                message.pop("raw_response", None)
                message.pop(CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY, None)
        out_row = {
            "images": list(row["images"]),
            "messages": messages,
            "metadata": metadata,
        }
        validate_canonical_rows([out_row], trajectory_path)
    except Exception as exc:
        reason = str(exc).split(": row 0 is not canonical: ")[-1]
        return (trajectory_path, reason.strip()[:200])
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-flight lite.data.hf.stage's canonical row gate over log-roots."
    )
    parser.add_argument(
        "--log-roots",
        nargs="+",
        required=True,
        help="Rollout log-roots, the same ones the following stage call takes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(64, os.cpu_count() or 8),
        help="Process pool size (default: min(64, cpu_count))",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    files: list[str] = []
    for root in args.log_roots:
        files.extend(sorted(glob.glob(f"{root}/*/sample_*/trajectory.parquet")))
    if not files:
        print(f"no trajectories under {args.log_roots}", file=sys.stderr)
        return 1

    print(f"checking {len(files)} trajectories with {args.workers} workers", flush=True)
    bad: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(check_one, files, chunksize=32):
            if result is not None:
                bad.append(result)

    if not bad:
        print(f"all {len(files)} rows are publishable")
        return 0

    print(f"\n{len(bad)} of {len(files)} rows would fail stage:")
    for reason, count in Counter(reason for _, reason in bad).most_common():
        print(f"  {count:5d}  {reason}")
    print("\noffending trajectories:")
    for path, _ in bad:
        print(f"  {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
