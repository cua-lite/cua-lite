# cua-lite/osworld — OSWorld v1 VM-in-Docker image with the eval stack baked IN.
#
# Eval-in-container (androidworld pattern): v1's `desktop_env` + `osworld_evaluation_examples`
# live INSIDE this image, driven by docker/server.py. The host also installs this same
# packaging-only fork through lite/gym/envs/lite/osworld/scripts/install.sh for
# lite.osworld/lite.scalecua evaluator imports, so keep the fork pin aligned there.
#
# The base `happysixd/osworld-docker` (Debian trixie) boots the QEMU/KVM guest VM via its
# entrypoint; we only add Python + v1 desktop_env + server.py. The host `docker exec -d`'s
# server.py once the VM's :5000 answers.
#
# Build: uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh
FROM happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9

# ── 1. Python + build tools ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip python3-dev unzip git \
      build-essential ninja-build pkg-config \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# ── 2. v1 OSWorld dist (same pin + deps as scripts/install.sh) ───────────────
# KEEP this `osworld @ git+...@<pin>` byte-identical to lite/gym/envs/lite/osworld's
# install. The cua-lite/OSWorld fork carries only packaging metadata on top of the
# upstream desktop_env tree. oauth2client<4: pydrive's unpinned transitive break
# (removed symbol in 4.x).
RUN pip install --no-cache-dir --prefer-binary "oauth2client<4" \
      "osworld @ git+https://github.com/cua-lite/OSWorld.git@3b33421192da4de61abe52c8da5f7a8432633837" \
      pandas "requests>=2.28" python-dotenv playwright pydrive requests_toolbelt \
      fastapi "uvicorn[standard]"

# ── 2b. Runtime system libs (opencv → libGL etc.) — separate layer ───────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ── 2c. Smoke-test the eval stack ────────────────────────────────────────────
RUN python -c "from desktop_env.desktop_env import DesktopEnv; \
               from desktop_env.controllers.setup import SetupController; \
               from desktop_env.controllers.python import PythonController; \
               import osworld_evaluation_examples, cv2, fastapi, uvicorn, numpy; from PIL import Image; \
               print('v1 desktop_env eval stack OK')"

# ── 3. In-container eval server ──────────────────────────────────────────────
COPY server.py /usr/local/bin/server.py
RUN chmod +x /usr/local/bin/server.py
EXPOSE 6000

# QEMU inherits this process-local setting through the original VM entrypoint.
COPY disable_thp.py /usr/local/bin/disable_thp.py
ENTRYPOINT ["/usr/bin/tini", "-s", "/opt/venv/bin/python", "/usr/local/bin/disable_thp.py", "/run/entry.sh"]
