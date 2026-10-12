"""Image freshness spec for lite.cuagym."""

from __future__ import annotations

import os
import subprocess

from lite.gym.utils.backend.freshness import ContainerImage
from lite.gym.utils.config.manifest import baked_asset_lock_sources


def _base_identity(base_image: str) -> str:
    """The base's image ID when it is built, else its tag.

    Hashing the TAG alone only catches a base whose Dockerfile changed -- and the
    parent's Dockerfile is already a source above. It misses the other way a base
    moves: a rebuild that changes CONTENT while the tag stays put, which is what an
    apt dependency bump does (an openjdk point release re-registered java on
    /usr/bin here without a single line of parent Dockerfile changing). That leaves
    this image silently running on a stale base with no test able to see it.

    Falls back to the tag when docker cannot answer -- a host with no daemon, or a
    base that was never built. Both already fail later with a clearer error than a
    freshness probe could give, and neither should make `image_for` raise.
    """
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", base_image],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return base_image
    image_id = out.stdout.strip()
    return image_id if out.returncode == 0 and image_id else base_image


def image_for(env_id: str) -> ContainerImage:
    if env_id != "lite.cuagym":
        raise KeyError(env_id)
    # CUA-Gym builds FROM lite.osworld and bakes web mocks plus desktop reward
    # deps. install.sh is a source because stage_context() selects mock files and
    # writes mocks-package.json into the Docker context.
    base_image = os.environ.get(
        "LITE_CUAGYM_BASE_IMAGE", "cua-lite/lite.osworld:latest"
    )
    return ContainerImage(
        "cua-lite/lite.cuagym:latest",
        (
            "lite/gym/envs/lite/cuagym/docker",
            "lite/gym/envs/lite/cuagym/scripts/install.sh",
            "lite/gym/envs/lite/cuagym/scripts/utils",
            "lite/gym/envs/lite/cuagym/src/utils/dataset.py",
            "lite/gym/envs/lite/osworld/docker",
            "lite/gym/sandbox/exec_stdio/server.py",
            "lite/gym/sandbox/docker",
            *baked_asset_lock_sources("lite/gym/envs/lite/cuagym"),
        ),
        "uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh",
        "lite/gym/envs/lite/cuagym/README.md",
        # ``scripts/utils`` stays a WHOLE-DIRECTORY source, so a future importer is
        # hashed by default (over-hash = the safe direction). Only the CUAGym
        # exception freshness.py's RULE names earns its place there:
        # ``import_web_tasks.py`` writes ``apps.txt``, which install.sh's
        # ``_apps``/``stage_context`` read to pick which mocks get COPY'd — so it
        # and its ``import_tasks.py`` dispatcher stay hashed. The two below are
        # host-only, i.e. exactly what the RULE forbids hashing:
        #   * ``import_desktop_tasks.py`` writes only
        #     ``.cache/desktop/lite.cuagym_desktop_tasks/`` (train.jsonl +
        #     bundles), read by setup_fn/evaluate_final_fn at RUN time. The build
        #     context is ``docker/`` plus ``.cache/web/cua-gym-hub/websites`` — a
        #     disjoint tree, so it cannot move apps.txt or a baked byte.
        #   * ``validation_sweep.py`` writes ``data/validation_excludes.json`` and
        #     sweep reports; nothing on the web import path reads either.
        exclude=(
            "lite/gym/envs/lite/cuagym/scripts/utils/import_desktop_tasks.py",
            "lite/gym/envs/lite/cuagym/scripts/utils/validation_sweep.py",
        ),
        extra_hash_inputs=(f"base={_base_identity(base_image)}",),
    )
