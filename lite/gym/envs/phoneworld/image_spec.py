"""Image freshness spec for PhoneWorld's local-only image."""

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "phoneworld":
        raise KeyError(env_id)
    return ContainerImage(
        "cua-lite/phoneworld:latest",
        (
            "lite/gym/envs/androidworld/docker",
            "lite/gym/envs/phoneworld/docker",
            "lite/gym/envs/phoneworld/scripts/install.sh",
        ),
        "uv run --no-sync bash lite/gym/envs/phoneworld/scripts/install.sh",
        "lite/gym/envs/phoneworld/README.md",
        exclude=(
            "lite/gym/envs/androidworld/docker/apps.sh",
            "lite/gym/envs/androidworld/docker/server.py",
        ),
    )
