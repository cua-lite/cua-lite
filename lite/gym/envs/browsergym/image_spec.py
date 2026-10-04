"""Image freshness spec for browsergym benchmarks that cua-lite builds itself.

Only TimeWarp has a cua-lite-built image (its Wiki/News/Shop sites); the
WebArena/VisualWebArena images are upstream tars loaded by install.sh.
"""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def supported_env_ids() -> tuple[str, ...]:
    return ("browsergym.timewarp",)


def image_for(env_id: str) -> ContainerImage:
    if env_id != "browsergym.timewarp":
        raise KeyError(env_id)
    return ContainerImage(
        "cua-lite/browsergym.timewarp:latest",
        ("lite/gym/envs/browsergym/docker/timewarp",),
        "uv run --no-sync bash lite/gym/envs/browsergym/scripts/install.sh timewarp",
        "lite/gym/envs/browsergym/README.md",
    )
