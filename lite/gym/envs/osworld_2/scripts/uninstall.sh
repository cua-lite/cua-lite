#!/bin/bash
# Reverse install.sh. Removes the provisioned v2 qcow2, the gated task classes, the staged
# V2 build source, and (by default) the derived image. There is NO host `desktop_env` to
# worry about — it lives only inside the image (the whole point of eval-in-container), so
# uninstalling osworld_2 can't affect osworld v1 or lite.osworld.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/osworld_2/scripts/uninstall.sh          # qcow2 + task_class + image + _vendor
#   uv run --no-sync bash lite/gym/envs/osworld_2/scripts/uninstall.sh keep-image   # everything except the image

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
QCOW2="$ENV_DIR/.cache/osworld-v2-ubuntu-x86.qcow2"
TASK_CLASS_DIR="$ENV_DIR/.cache/task_class"
VENDOR_DIR="$ENV_DIR/docker/_vendor"
IMAGE="cua-lite/osworld_2:osworld-v2.1-volume"

if [ -e "$QCOW2" ]; then rm -f "$QCOW2" && echo "[uninstall] removed $QCOW2" >&2; else echo "[uninstall] qcow2 already absent." >&2; fi
if [ -d "$TASK_CLASS_DIR" ]; then rm -rf "$TASK_CLASS_DIR" && echo "[uninstall] removed $TASK_CLASS_DIR" >&2; else echo "[uninstall] task_class/ already absent." >&2; fi
if [ -d "$VENDOR_DIR" ]; then rm -rf "$VENDOR_DIR" && echo "[uninstall] removed staged V2 source ($VENDOR_DIR)" >&2; fi

if [ "${1:-}" != "keep-image" ]; then
    docker rmi "$IMAGE" >/dev/null 2>&1 && echo "[uninstall] removed image $IMAGE" >&2 || echo "[uninstall] image $IMAGE absent/in-use." >&2
fi

echo "[uninstall] done ✓ (no host desktop_env involved — osworld v1 + lite.osworld unaffected)" >&2
