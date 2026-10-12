"""Private reference-asset import; no original game artwork is distributed.

Import allowlisted bytes from a reviewed user-supplied ZIP, or derive the rhythm
chart from its pinned analysis module, with
``python -m examples.not_a_robot.reference_assets /path/to/reference-input``.
The module is parsed as data, never executed or served. Pinned sources are
implementation evidence, not redistribution licenses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

REFERENCE_ROOT = Path(__file__).with_name("local") / "reference_assets"
REFERENCE_CATALOG = json.loads(Path(__file__).with_name("reference_manifest.json").read_text())
SUPPLEMENT_SHA256 = REFERENCE_CATALOG["archives"]["first10"]
REFERENCE_HASHES = {name: item["sha256"] for name, item in REFERENCE_CATALOG["files"].items()}
REFERENCE_HASHES.update(
    {name: item["derived_sha256"] for name, item in REFERENCE_CATALOG["derived_files"].items()}
)


def import_reference(archive: Path, destination: Path = REFERENCE_ROOT) -> int:
    """Verify the complete input before importing, and never overwrite other bytes."""
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    derived = next(
        (
            (name, item)
            for name, item in REFERENCE_CATALOG["derived_files"].items()
            if item["source_sha256"] == digest
        ),
        None,
    )
    archive_id = next(
        (name for name, expected in REFERENCE_CATALOG["archives"].items() if expected == digest),
        None,
    )
    if archive_id is None and derived is None:
        raise ValueError("Reference input SHA256 does not match a reviewed ZIP or analysis module")
    if destination.is_symlink():
        raise ValueError("Reference destination must not be a symlink")
    contents = {}
    if derived is not None:
        filename, entry = derived
        # This exact pinned module has one literal chart. No JavaScript evaluator
        # or source bytes are made available to the page or the solver.
        match = re.search(r"\bc = (\[.*?\]),\s*l = n\(293\);", archive.read_text("utf-8"), re.S)
        if match is None:
            raise ValueError("Reviewed rhythm module has no expected literal chart")
        chart = json.loads(re.sub(r"\b(key|time):", r'"\1":', match[1]))
        data = (
            json.dumps(chart, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("ascii")
        if hashlib.sha256(data).hexdigest() != entry["derived_sha256"]:
            raise ValueError("Derived rhythm chart failed integrity verification")
        contents[filename] = data
    else:
        with zipfile.ZipFile(archive) as source:
            for filename, entry in REFERENCE_CATALOG["files"].items():
                if entry["archive"] != archive_id:
                    continue
                data = source.read(entry["member"])
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ValueError(
                        f"Reference member failed integrity verification: {entry['member']}"
                    )
                contents[filename] = data
    for filename, data in contents.items():
        target = destination / filename
        if target.is_symlink() or (target.exists() and target.read_bytes() != data):
            raise FileExistsError(f"Refusing to overwrite existing reference asset: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for filename, data in contents.items():
        target = destination / filename
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(data)
    return len(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    count = import_reference(args.archive)
    print(f"Verified {count} private assets in {REFERENCE_ROOT}. Redistribution rights unknown.")


if __name__ == "__main__":
    main()
