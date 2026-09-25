# cua-lite/osworld_2 — OSWorld-V2 VM-in-Docker image with the eval stack baked IN.
#
# Why a derived image (vs osworld v1's PULL-only host-attach): V2's `desktop_env` and v1's
# share the one import name `desktop_env`, so they can't coexist in the host's single uv.
# Baking V2's desktop_env into THIS image (the container's Debian userspace, outside the
# guest VM) means the cua-lite HOST needs ZERO desktop_env — osworld_2's eval runs here,
# driven by the host over a thin HTTP-RPC (docker/server.py). v1 osworld + lite.osworld
# keep the v1 dist on the host, untouched. (androidworld's docker/server.py pattern.)
#
# The base `happysixd/osworld-docker` (Debian trixie) boots the QEMU/KVM guest VM via its
# entrypoint `tini → /run/entry.sh`; we DON'T change that. We only add Python + V2
# desktop_env + server.py; the host `docker exec -d`'s server.py once the VM's :5000 answers.
#
# Build context = this dir. install.sh stages the local OSWorld-V2 checkout into
# `_vendor/OSWorld-V2/` (gitignored) first, at the official release commit.
#
# Build: uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh
# The released guest installs FreeCAD 0.19 for tasks 103/104. The Debian runtime's
# FreeCAD 1.0 exhausts those tasks' 180s CPU limit on a known-good reference STEP.
FROM condaforge/miniforge3@sha256:95195a50e20f7929a31a68ea407cefe117423258639efb6b66f0518e4b86f8ee AS cad-score
RUN conda create -y --prefix /opt/cad-score --override-channels -c conda-forge \
      --repodata-fn repodata.json python=3.9 freecad=0.19.3=py39h0f650a1_3 \
      numpy=2.0.2 scipy=1.13.1 \
 && conda clean -afy

FROM happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9

# ── 1. Python + build tools (base image has none) ────────────────────────────
# build-essential/ninja/dev headers: a few eval libs (pyacoustid/formulas/…) may build
# from source; without them pip fails on the meson/C builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip python3-dev unzip curl sudo patch \
      build-essential ninja-build pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Keep the evaluator separate from the base image's system Python.
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# ── 2. OSWorld-V2 — the official frozen dependency environment ──────────────
# uv.lock includes fastapi, uvicorn, and the bundled osworld-ffmpeg source.
COPY _vendor/OSWorld-V2 /opt/osworld-v2
WORKDIR /opt/osworld-v2
COPY patches/llm-judge-errors.patch /tmp/llm-judge-errors.patch
RUN patch -p1 < /tmp/llm-judge-errors.patch && rm /tmp/llm-judge-errors.patch
RUN uv sync --frozen --no-dev --python 3.12 --project /opt/osworld-v2 --no-cache

# ── 2b. Runtime system libs the eval stack needs (opencv → libGL etc.) ───────
# Separate layer so adjusting system libraries does not reinstall Python dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      file imagemagick libreoffice-calc libreoffice-impress libreoffice-writer \
      netpbm tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

# Keep the trusted CAD scorer outside the agent's VM, on the guest's FreeCAD 0.19
# major version. The official task classes already provide these Python overrides.
COPY --from=cad-score /opt/cad-score /opt/cad-score
COPY --chmod=755 freecad_python.py /usr/local/bin/freecad-python
ENV TASK_103_FREECAD_PYTHON=/usr/local/bin/freecad-python \
    TASK_104_FREECAD_PYTHON=/usr/local/bin/freecad-python \
    TASK_103_FREECAD_LIB=/opt/cad-score/lib \
    TASK_104_FREECAD_LIB=/opt/cad-score/lib

# ── 2c. Smoke-test the full eval stack (fail the build early if an import breaks) ─
RUN python -c "from desktop_env.desktop_env import DesktopEnv; \
               from desktop_env.controllers.python import PythonController; \
               from desktop_env.controllers.setup import SetupController; \
               from desktop_env.task_base import BaseTask; \
               import cv2, fastapi, uvicorn; print('V2 desktop_env eval stack OK')"
RUN PYTHONPATH=/opt/cad-score/lib /usr/local/bin/freecad-python -c "import FreeCAD, Part, numpy, scipy"

# ── 3. In-container eval server (bind-mounted over at run time for hot edits) ─
COPY server.py /usr/local/bin/server.py
RUN chmod +x /usr/local/bin/server.py

# Grow only each VM's writable boot overlay after the base entrypoint creates it.
COPY --chmod=755 grow_root_disk.sh /run/grow_root_disk.sh
RUN sed -i '/^\. install.sh /a . /run/grow_root_disk.sh' /run/entry.sh

# The eval server listens here (fixed in-container); host maps it to a dynamic port and
# `docker exec -d`'s it after the VM boots. The base image's VM ports (5000/9222/8080/8006)
# are unchanged.
EXPOSE 6000
