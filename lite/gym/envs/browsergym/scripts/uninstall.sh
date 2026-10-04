#!/bin/bash
set -euo pipefail
# Remove browsergym benchmark Docker images and downloaded tar/zim/zip files.
# Destructive: deletes multi-GB resources. Safe to re-run — missing items are
# silently skipped.
#
# Usage:
#   bash lite/gym/envs/browsergym/scripts/uninstall.sh <benchmark>
#
# Benchmarks:
#   webarena         — remove webarena Docker images + tar/zim files
#   visualwebarena   — all webarena artifacts + Classifieds zip/unpack dir
#   miniwob          — delete the cloned miniwob-plusplus directory
#   timewarp         — remove the cua-lite/browsergym.timewarp image

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BROWSERGYM_CACHE="${BROWSERGYM_CACHE:-$ENV_DIR/.cache}"
IMAGES_DIR="$BROWSERGYM_CACHE"
MINIWOB_CLONE_DIR="$BROWSERGYM_CACHE/miniwob-plusplus"

uninstall_webarena_images() {
    echo "Removing WebArena Docker images..."
    for image in shopping_final_0712 shopping_admin_final_0719 \
                 postmill-populated-exposed-withimg gitlab-populated-final-port8023 \
                 ghcr.io/kiwix/kiwix-serve:3.3.0; do
        docker image rm "$image" 2>/dev/null \
            || echo "  kept $image (absent, or a container still references it)"
    done

    echo "Removing WebArena tar / zim files from $IMAGES_DIR ..."
    rm -f "$IMAGES_DIR/shopping_final_0712.tar" \
          "$IMAGES_DIR/shopping_admin_final_0719.tar" \
          "$IMAGES_DIR/postmill-populated-exposed-withimg.tar" \
          "$IMAGES_DIR/gitlab-populated-final-port8023.tar" \
          "$IMAGES_DIR/wikipedia_en_all_maxi_2022-05.zim" 2>/dev/null || true
}

uninstall_classifieds() {
    if [ -d "$IMAGES_DIR/classifieds_docker_compose" ]; then
        echo "Tearing down Classifieds compose stack (removes images)..."
        (cd "$IMAGES_DIR/classifieds_docker_compose" && docker compose down --rmi all 2>/dev/null) || true
        rm -rf "$IMAGES_DIR/classifieds_docker_compose"
    fi
    rm -f "$IMAGES_DIR/classifieds_docker_compose.zip" 2>/dev/null || true
}

usage() {
    cat <<USAGE
Usage: $0 <benchmark>

Benchmarks:
  webarena         remove webarena images + tar files
  visualwebarena   webarena artifacts + classifieds
  miniwob          delete cloned miniwob-plusplus at \$MINIWOB_CLONE_DIR
  timewarp         remove the cua-lite/browsergym.timewarp image

This is destructive — images and tar files (multi-GB) are deleted.
USAGE
}

case "${1:-}" in
    webarena)
        uninstall_webarena_images
        echo "WebArena uninstalled."
        ;;
    visualwebarena)
        uninstall_webarena_images
        uninstall_classifieds
        echo "VisualWebArena uninstalled."
        ;;
    miniwob)
        if [ -d "$MINIWOB_CLONE_DIR" ]; then
            echo "Removing $MINIWOB_CLONE_DIR ..."
            rm -rf "$MINIWOB_CLONE_DIR"
        else
            echo "$MINIWOB_CLONE_DIR does not exist — nothing to remove."
        fi
        ;;
    timewarp)
        docker image rm cua-lite/browsergym.timewarp:latest 2>/dev/null \
            || echo "  kept cua-lite/browsergym.timewarp:latest (absent, or a container still references it)"
        echo "TimeWarp uninstalled."
        ;;
    ""|-h|--help)
        usage
        [ -z "${1:-}" ] && exit 1 || exit 0
        ;;
    *)
        echo "error: unknown benchmark '$1'" >&2
        usage
        exit 1
        ;;
esac
