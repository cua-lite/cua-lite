#!/bin/bash
# Build the cua-lite/androidlab Docker image.
# Safe to run repeatedly — re-runs exit early only when the image is FRESH;
# auto-rebuilds on stale (image sources changed)
# (Docker's layer cache doesn't help us much here since the ~32 GB SDK
# layers already cache the one expensive path). Force a fresh build with
# ``install.sh rebuild``.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh           # build if missing/stale
#   uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh rebuild   # force rebuild
#   uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh status
#   uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh pull|push # GHCR distribution (src_hash-gated)
#
# Required one-time download: docker-file.zip (8.65 GB) from AndroidLab's
# Google Drive, placed at lite/gym/envs/androidlab/.cache/docker-file.zip.
# Contains the AVD bundle, JDK deb, SDK tools, skins, and x86_64 system image.
# Plus a pinned emulator 34.2.15 zip (build 11906825), which this script pulls
# from Google's redirector automatically.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$ENV_DIR/docker"
IMAGE="cua-lite/androidlab:latest"

# Build-source hashing + stale-skip live in the shared helper (one impl for all
# container envs); it stamps the same ``lite.src_hash`` the runtime freshness gate
# reads, via the same python hash, so build and launch can't drift. The hash
# covers ``docker/`` plus this staging script; the multi-GB emulator/AVD assets
# are selected by the pinned URLs and extraction logic here, so changing that
# build-context assembly changes the source hash.
source "$ENV_DIR/../../scripts/image_build.sh"

# Per-env cache (git-ignored). Colocated with the code that reads it so you
# don't have to hunt for the zip at the repo root.
CACHE_DIR="${ANDROID_LAB_CACHE:-$ENV_DIR/.cache}"
DOCKER_FILE_ZIP="$CACHE_DIR/docker-file.zip"
EXTRACTED_DIR="$CACHE_DIR/extracted/docker-file"

# Exact emulator build the reference snapshot was saved with.
EMULATOR_ZIP_URL="https://dl.google.com/android/repository/emulator-linux_x64-11906825.zip"

# Google Drive ID for the AndroidLab-published docker-file.zip (8.65 GB).
# Override via ``ANDROID_LAB_DOCKER_FILE_URL`` for an internal mirror —
# any plain HTTPS URL works with curl. Falls back to gdown if unset.
# Rationale: Google Drive is a single point of failure (upstream THUDM
# could revoke share access, or the file could be deleted), and gdown
# is fragile against rate-limited / "confirm large download" interstitial
# pages. Internal mirrors let an org cache the zip once and rebuild from
# scratch without depending on third-party availability.
DOCKER_FILE_GDRIVE_ID="${ANDROID_LAB_DOCKER_FILE_GDRIVE_ID:-1SJ79gdO7whgUod3HnuS87aOKihRk1i-U}"
DOCKER_FILE_URL="${ANDROID_LAB_DOCKER_FILE_URL:-}"

log() { echo "[install] $*" >&2; }

image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

_extracted_assets_ready() {
    for f in Pixel_7_Pro_API_33.avd.zip Pixel_7_Pro_API_33.ini x86_64.zip skins.zip \
             sdk-tools-linux-4333796.zip openlogic-openjdk-8u412-b08-linux-x64-deb.deb; do
        [ -f "$EXTRACTED_DIR/$f" ] || return 1
    done
}

_fetch_docker_file_zip() {
    # Three-path acquisition strategy:
    #   1. ``ANDROID_LAB_DOCKER_FILE_URL`` set → curl from that URL (internal
    #      mirror, hermetic, no third-party dep).
    #   2. Default → gdown the upstream Google Drive ID.
    #   3. On failure of both → a clear message tells the operator to place the
    #      zip manually at ``$DOCKER_FILE_ZIP`` (recovery without re-reading this code).
    mkdir -p "$CACHE_DIR"
    if [ -n "$DOCKER_FILE_URL" ]; then
        log "downloading docker-file.zip (~8.65 GB) from \$ANDROID_LAB_DOCKER_FILE_URL ..."
        if ! curl -fL --retry 5 --retry-delay 5 -o "$DOCKER_FILE_ZIP" "$DOCKER_FILE_URL"; then
            echo "ERROR: download failed from configured \$ANDROID_LAB_DOCKER_FILE_URL" >&2
            echo "       Check URL/credentials, or unset \$ANDROID_LAB_DOCKER_FILE_URL to fall back to gdown." >&2
            rm -f "$DOCKER_FILE_ZIP"
            exit 1
        fi
        return 0
    fi
    log "downloading docker-file.zip (~8.65 GB) from AndroidLab's Google Drive (id=$DOCKER_FILE_GDRIVE_ID) ..."
    log "    (set \$ANDROID_LAB_DOCKER_FILE_URL to a mirror URL to bypass Google Drive)"
    if ! command -v gdown >/dev/null 2>&1; then
        log "installing gdown ..."
        # `uv pip`, never a bare-pip fallback: this venv ships NO pip, so a bare
        # `pip` resolves to /usr/local/bin/pip — a python3.10 interpreter — and
        # installs into system dist-packages while this script's `python` is
        # .venv's 3.12; the gdown console script never appears on PATH and the
        # `gdown` call below then fails with a misleading "command not found".
        # This script is ALWAYS invoked as `uv run --no-sync bash <script>`
        # (CLAUDE.md / docs/envs.md#invocation), so uv is present by construction.
        command -v uv >/dev/null 2>&1 || {
            echo "FATAL: uv not found. Run this as: uv run --no-sync bash $0" >&2
            exit 1
        }
        uv pip install gdown >&2
    fi
    if ! gdown "$DOCKER_FILE_GDRIVE_ID" -O "$DOCKER_FILE_ZIP"; then
        echo "" >&2
        echo "ERROR: gdown failed for Google Drive id=$DOCKER_FILE_GDRIVE_ID" >&2
        echo "       Common causes:" >&2
        echo "         * upstream THUDM revoked / removed the share" >&2
        echo "         * Google Drive rate-limited this host" >&2
        echo "         * gdown's 'confirm large download' parser broke" >&2
        echo "       Recovery: stage docker-file.zip on an internal mirror, then re-run with" >&2
        echo "         ANDROID_LAB_DOCKER_FILE_URL=<mirror-url> $0 $*" >&2
        echo "       Or place the file directly at: $DOCKER_FILE_ZIP" >&2
        rm -f "$DOCKER_FILE_ZIP"
        exit 1
    fi
}

_ensure_unzip() {
    # Both require_assets and build() invoke `unzip` to expand docker-file.zip
    # and emulator-11906825.zip; fresh Ubuntu Server images don't ship unzip,
    # so install it via apt before the first call. Idempotent.
    if command -v unzip >/dev/null 2>&1; then return; fi
    log "installing unzip (required to extract zip assets)..."
    if [ "$(id -u)" = "0" ]; then
        apt-get update -qq && apt-get install -y -qq unzip
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq unzip
    else
        echo "ERROR: unzip missing and passwordless sudo unavailable." >&2
        echo "  Run manually:  sudo apt-get install -y unzip" >&2
        exit 1
    fi
}

require_assets() {
    if [ ! -f "$EXTRACTED_DIR/Dockerfile" ]; then
        if [ ! -f "$DOCKER_FILE_ZIP" ]; then
            _fetch_docker_file_zip
        else
            log "using cached $DOCKER_FILE_ZIP"
        fi
        log "extracting $DOCKER_FILE_ZIP -> $EXTRACTED_DIR ..."
        mkdir -p "$(dirname "$EXTRACTED_DIR")"
        unzip -q -o "$DOCKER_FILE_ZIP" -x "__MACOSX/*" -d "$(dirname "$EXTRACTED_DIR")"
    fi
    for f in Pixel_7_Pro_API_33.avd.zip Pixel_7_Pro_API_33.ini x86_64.zip skins.zip \
             sdk-tools-linux-4333796.zip openlogic-openjdk-8u412-b08-linux-x64-deb.deb; do
        if [ ! -f "$EXTRACTED_DIR/$f" ]; then
            echo "ERROR: $f missing from $EXTRACTED_DIR" >&2
            echo "       The docker-file.zip you got may be corrupt or stale." >&2
            echo "       Re-download by removing $DOCKER_FILE_ZIP and $EXTRACTED_DIR." >&2
            exit 1
        fi
    done
}

require_emulator() {
    local emu_zip="$CACHE_DIR/emulator-11906825.zip"
    if [ ! -f "$emu_zip" ]; then
        log "downloading pinned emulator 34.2.15 (build 11906825) ..."
        mkdir -p "$CACHE_DIR"
        curl -fL --retry 3 -o "$emu_zip" "$EMULATOR_ZIP_URL"
    fi
}

require_adb_keys() {
    if [ -f "$CACHE_DIR/adbkey" ] && [ -f "$CACHE_DIR/adbkey.pub" ]; then
        return
    fi
    log "generating adb keys for pre-bake ..."
    # ``adb keygen FILE`` writes both FILE (private) and FILE.pub in the
    # same call — must run once so the pair matches. Two separate keygens
    # would produce mismatched keys.
    local host_adb
    host_adb="$(command -v adb || true)"
    if [ -n "$host_adb" ]; then
        "$host_adb" keygen "$CACHE_DIR/adbkey"
    else
        # No host adb — build a throwaway helper image and mount the
        # cache dir so both keys land on the host from a single run.
        # Use ubuntu:24.04 instead of debian:bullseye-slim because Debian
        # bullseye's adb (v10, 2018) only writes the private key; modern
        # adb (Ubuntu noble has v34) writes both private + .pub in one
        # call, which is what the [-f adbkey] && [-f adbkey.pub] check
        # downstream requires.
        local tmp_img="androidlab_keygen_helper:tmp"
        docker build -q -t "$tmp_img" - <<'EOF' >/dev/null
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y -q --no-install-recommends adb && rm -rf /var/lib/apt/lists/*
EOF
        # The generated adbkey is mode 0600 and must end up owned by the host
        # user, or install.sh later fails at `cp adbkey* $ctx/` with EACCES.
        # How to achieve that depends on the Docker mode:
        #   * rootful  — container root maps to host root, so pass --user
        #     uid:gid to write the key as the host user directly.
        #   * rootless — container root ALREADY maps to the host user, while
        #     --user uid:gid maps to an unprivileged subuid that owns the key
        #     0600 (unreadable/undeletable by the host). So OMIT --user and
        #     let the container run as root == host user.
        local user_flag=(--user "$(id -u):$(id -g)")
        if docker info -f '{{println .SecurityOptions}}' 2>/dev/null | grep -q rootless; then
            user_flag=()
        fi
        docker run --rm \
            "${user_flag[@]}" \
            -v "$CACHE_DIR":/host "$tmp_img" \
            adb keygen /host/adbkey
        image_rm "$tmp_img"
    fi
    [ -f "$CACHE_DIR/adbkey" ] && [ -f "$CACHE_DIR/adbkey.pub" ] || {
        echo "ERROR: adb keygen did not produce both adbkey + adbkey.pub" >&2
        exit 1
    }
}

build() {
    # Auto-rebuild only when docker/ sources changed since the existing image was
    # built; a fresh image skips the entire (multi-GB) asset-staging + build path.
    if image_is_fresh "$IMAGE" androidlab; then
        log "$IMAGE up to date (build sources unchanged); skipping build. Run 'install.sh rebuild' to force."
        return
    fi
    image_rm "$IMAGE"
    _ensure_unzip       # used by both require_assets and the emulator-zip expansion below
    require_assets
    require_emulator
    require_adb_keys

    local ctx
    ctx="$(mktemp -d)"
    trap 'rm -rf "$ctx"' RETURN

    log "assembling build context at $ctx ..."
    cp "$DOCKER_DIR/Dockerfile" "$DOCKER_DIR/server.py" "$ctx/"
    cp -r "$DOCKER_DIR/patches" "$ctx/"
    cp "$EXTRACTED_DIR"/*.deb "$EXTRACTED_DIR"/*.zip "$EXTRACTED_DIR"/*.ini "$ctx/"
    unzip -q -o "$CACHE_DIR/emulator-11906825.zip" -d "$ctx/"  # → ctx/emulator/
    cp "$CACHE_DIR/adbkey" "$CACHE_DIR/adbkey.pub" "$ctx/"

    log "docker build -t $IMAGE  (~45 min first time, layer cache afterwards) ..."
    docker build -t "$IMAGE" --label "$(src_label androidlab)" "$ctx"
    log "built $IMAGE"
}

status() {
    echo "cache dir        : $CACHE_DIR"
    echo "docker-file.zip  : $([ -f "$DOCKER_FILE_ZIP" ] && echo PRESENT || echo MISSING)"
    echo "extracted assets : $(_extracted_assets_ready && echo PRESENT || echo MISSING)"
    echo "emulator zip     : $([ -f "$CACHE_DIR/emulator-11906825.zip" ] && echo PRESENT || echo MISSING)"
    echo "adb keys         : $([ -f "$CACHE_DIR/adbkey" ] && [ -f "$CACHE_DIR/adbkey.pub" ] && echo PRESENT || echo MISSING)"
    if image_exists "$IMAGE"; then
        local sz
        sz="$(docker image inspect --format='{{.Size}}' "$IMAGE" | awk '{printf "%.1fGB", $1/1024/1024/1024}')"
        if image_is_fresh "$IMAGE" androidlab; then
            echo "image $IMAGE : PRESENT ($sz, up to date)"
        else
            echo "image $IMAGE : PRESENT ($sz, STALE — sources changed; run install.sh to rebuild)"
        fi
    else
        echo "image $IMAGE : MISSING"
    fi
}

rebuild() {
    image_rm "$IMAGE"
    build
}

case "${1:-build}" in
    build)    build ;;
    rebuild)  rebuild ;;
    pull)     do_pull "$IMAGE" androidlab ;;
    push)     do_push "$IMAGE" androidlab ;;
    status)   status ;;
    *)        echo "Usage: $0 [build|rebuild|pull|push|status]" >&2; exit 1 ;;
esac
