"""Image freshness spec for osworld_2."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from lite.gym.utils.backend.freshness import ContainerImage, tree_content_hash

ENV_DIR = Path(__file__).resolve().parent
_OSWORLD_V2_SOURCE_SKIP_SEGMENTS = frozenset({".git", "__pycache__"})
_OSWORLD_V2_SOURCE_SKIP_NAMES = frozenset(
    {"assets", "docs", "mm_agents", "monitor"}
)
_OSWORLD_V2_SOURCE_SKIP_PATTERNS = ("*.qcow2", "*.zip", "*.pdf", "*.pyc", "*.pyo")
_OSWORLD_V2_SOURCE_SKIP_REL_PATTERNS = (
    "evaluation_examples/task_class/task_*.py",
)


def osworld_v2_rsync_excludes() -> tuple[str, ...]:
    """Rsync exclude args for the source tree staged into the Docker context."""
    return (
        *tuple(f"--exclude={name}" for name in sorted(_OSWORLD_V2_SOURCE_SKIP_SEGMENTS)),
        *tuple(f"--exclude={name}" for name in sorted(_OSWORLD_V2_SOURCE_SKIP_NAMES)),
        *tuple(f"--exclude={pattern}" for pattern in _OSWORLD_V2_SOURCE_SKIP_PATTERNS),
        *tuple(f"--exclude={pattern}" for pattern in _OSWORLD_V2_SOURCE_SKIP_REL_PATTERNS),
    )


def _osworld_v2_source_path() -> Path:
    return Path(
        os.environ.get("OSWORLD_V2_SRC") or ENV_DIR / "docker" / "_vendor" / "OSWorld-V2"
    ).expanduser().resolve()


def _skip_osworld_v2_source(root: Path, path: Path) -> bool:
    rel = path.relative_to(root)
    parts = rel.parts
    rel_posix = rel.as_posix()
    return (
        bool(_OSWORLD_V2_SOURCE_SKIP_SEGMENTS.intersection(parts))
        or any(part in _OSWORLD_V2_SOURCE_SKIP_NAMES for part in parts)
        or any(fnmatch.fnmatch(path.name, pat) for pat in _OSWORLD_V2_SOURCE_SKIP_PATTERNS)
        or any(
            fnmatch.fnmatch(rel_posix, pat)
            for pat in _OSWORLD_V2_SOURCE_SKIP_REL_PATTERNS
        )
    )


def osworld_v2_source_identity() -> str:
    src = _osworld_v2_source_path()
    if not (src / "pyproject.toml").is_file():
        return "osworld-v2-source=absent"
    digest = tree_content_hash(src, skip=_skip_osworld_v2_source)
    return f"osworld-v2-source={digest}"


def image_for(env_id: str) -> ContainerImage:
    if env_id != "osworld_2":
        raise KeyError(env_id)
    # install.sh stages OSWORLD_V2_SRC into docker/_vendor; hash the staging
    # logic plus the selected source-tree content identity so a local V2 source
    # edit cannot leave the derived image falsely fresh.
    return ContainerImage(
        "cua-lite/osworld_2:osworld-v2.1-volume",
        (
            "lite/gym/envs/osworld_2/docker/Dockerfile",
            "lite/gym/envs/osworld_2/docker/grow_root_disk.sh",
            "lite/gym/envs/osworld_2/docker/freecad_python.py",
            "lite/gym/envs/osworld_2/image_spec.py",
            "lite/gym/envs/osworld_2/scripts/install.sh",
        ),
        "uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh",
        "lite/gym/envs/osworld_2/README.md",
        extra_hash_inputs=(osworld_v2_source_identity(),),
    )
