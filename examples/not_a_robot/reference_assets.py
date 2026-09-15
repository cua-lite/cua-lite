"""Private reference-asset import; no original game artwork is distributed.

Import allowlisted bytes from a reviewed user-supplied ZIP with
``python -m examples.not_a_robot.reference_assets /path/to/reference.zip``.
The pinned archives are implementation evidence, not upstream source licenses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

REFERENCE_ROOT = Path(__file__).with_name("local") / "reference_assets"
REFERENCE_CATALOG = json.loads(Path(__file__).with_name("reference_manifest.json").read_text())
SUPPLEMENT_SHA256 = REFERENCE_CATALOG["archives"]["first10"]
REFERENCE_HASHES = {name: item["sha256"] for name, item in REFERENCE_CATALOG["files"].items()}


def import_reference(archive: Path, destination: Path = REFERENCE_ROOT) -> int:
    """Verify the complete input before importing, and never overwrite other bytes."""
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive_id = next(
        (name for name, expected in REFERENCE_CATALOG["archives"].items() if expected == digest),
        None,
    )
    if archive_id is None:
        raise ValueError("Reference ZIP SHA256 does not match a reviewed reference archive")
    if destination.is_symlink():
        raise ValueError("Reference destination must not be a symlink")
    contents = {}
    with zipfile.ZipFile(archive) as source:
        for filename, entry in REFERENCE_CATALOG["files"].items():
            if entry["archive"] != archive_id:
                continue
            data = source.read(entry["member"])
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(
                    f"Reference member failed integrity verification: {entry['member']}"
                )
            target = destination / filename
            if target.is_symlink() or (target.exists() and target.read_bytes() != data):
                raise FileExistsError(f"Refusing to overwrite existing reference asset: {target}")
            contents[filename] = data
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
