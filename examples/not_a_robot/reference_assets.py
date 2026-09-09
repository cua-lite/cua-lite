"""Private reference-asset import; no original game artwork is distributed.

Import only the allowlisted bytes from the user-supplied supplement ZIP with
``python -m examples.not_a_robot.reference_assets /path/to/supplement.zip``.
The pinned archive is implementation evidence, not an upstream source license.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

REFERENCE_ROOT = Path(__file__).with_name("local") / "reference_assets"
SUPPLEMENT_SHA256 = "a2900366ef7ef7ea18312b607d62b65e003290ff97a20d4318b00e5f740323d2"
REFERENCE_HASHES = {
    "level02_background.webp": "a73362e169465bdd5e3246cf94c39d4930389f1658120cb0755f408d553484bd",
    "level04_image_01.webp": "3470ab900bf688e932df9b3c05b17b4018bdd69aeee8758289230e734f27bef8",
    "level04_image_02.webp": "43d09975835ee196f61fdaf829e1e355c4c34c62ec5726a5d54a42323567d247",
    "level04_image_03.webp": "4b1652e23d28da209192f8d884710f5418a0e11887fcf9886f4da4e5be9b04b4",
    "level04_image_04.webp": "a64453949ab3d0aab0af705982720eb2e1ca818e3fc43a893ae796bcdf7cb2ad",
    "level04_image_05.webp": "2fbd1458e55671bb23ee650ff0906323e28613f16ac939d93a846bc0c2f39afa",
    "level04_image_06.webp": "1a380ceb88d8fe9099fa8e94a4d9521250094fdc4206c51a2ad3042015120683",
    "level04_image_07.webp": "749581ffd77456272a502c25735b971189c47d012265c0f65303dce94a931341",
    "level04_image_08.webp": "6f9057b8c998c208b8fc6ac5ed594d34d41ae003c58a94ee3084be3eb2c9b4cc",
    "level04_image_09.webp": "0aeaad75dc4ccd6d7a8f48c7e48f2944d96f7720be94b1cf26a0811581654051",
    "level05_intersection.webp": "9c49039c0005dcb73ecafb78ed9645d80ed6b3ce960774719e9854d3df4dc49a",
    "level09_background.webp": "3597867bb8ed5a245e279fe053918be829fea731cd3db518172e40d633fb4d25",
    "level10_grass.webp": "1f567515d0fd62aaf824e45dc3d1325812570d69c49104b8caa425e540cf2049",
    "level10_mole.png": "bd03b3c5f5b3880b7d2bd457c224567b19e7a0f52b67492b01da2e3a2b8a9191",
    "level08_reference.jpg": "6d87c48450c6293ad1cf3d436ce317c93d4fc61d51c17b328176e7b15775b3fb",
}


def import_reference(archive: Path, destination: Path = REFERENCE_ROOT) -> int:
    """Verify the complete input before importing, and never overwrite other bytes."""
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SUPPLEMENT_SHA256:
        raise ValueError("Supplement ZIP SHA256 does not match the reviewed reference archive")
    if destination.is_symlink():
        raise ValueError("Reference destination must not be a symlink")
    contents = {}
    with zipfile.ZipFile(archive) as source:
        for filename, expected in REFERENCE_HASHES.items():
            member = (
                "tasks/level_08/attempts/attempt_101/initial.jpg"
                if filename == "level08_reference.jpg"
                else f"implementation_reference/assets/{filename}"
            )
            data = source.read(member)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Reference member failed integrity verification: {member}")
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
