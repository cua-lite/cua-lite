#!/usr/bin/env bash
# Build the local-only PhoneWorld image from the user's gated HF access.
set -euo pipefail

export DOCKER_BUILDKIT=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DOCKER_DIR=$(cd "$SCRIPT_DIR/../docker" && pwd)
source "$SCRIPT_DIR/../../../scripts/image_build.sh"

IMG=cua-lite/phoneworld:latest
BASE_IMG=cua-lite/phoneworld:base
ANDROID_BASE=cua-lite/androidworld:base
BUILDER=lite-phoneworld-apps-installer
HF_REPO=EthanLeoLYX/PhoneWorld-APKs
HF_REVISION=63b9e86aa6c87ff88dc85eeeddbe62dbeba35ad3
HF_FILE=phoneworld_external_safe.zip
HF_SHA256=aa2516efd0d3907442f56933b69b59f6b1bdcbe1ef9d2c5cc12420d285c9975b

log() { echo "[phoneworld install] $*" >&2; }
cleanup_builder() { docker rm -fv "$BUILDER" >/dev/null 2>&1 || true; }

ensure_kvm() {
  if [ ! -w /dev/kvm ]; then
    echo "ERROR: PhoneWorld image build requires writable /dev/kvm." >&2
    exit 1
  fi
}

ensure_android_base() {
  if ! image_is_fresh "$ANDROID_BASE" androidworld; then
    echo "ERROR: $ANDROID_BASE is missing or stale." >&2
    echo "Build it first: uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh" >&2
    exit 1
  fi
}

download_zip() {
  uv run --with huggingface-hub python - "$HF_REPO" "$HF_REVISION" "$HF_FILE" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id=sys.argv[1], repo_type="dataset", revision=sys.argv[2],
    filename=sys.argv[3],
))
PY
}

build() {
  if image_is_fresh "$IMG" phoneworld; then
    log "$IMG is up to date"
    return
  fi
  ensure_kvm
  ensure_android_base
  zip_path=$(download_zip | tail -1)
  test -r "$zip_path"
  echo "$HF_SHA256  $zip_path" | sha256sum -c -

  log "building ADBKeyBoard from pinned source"
  docker build -t "$BASE_IMG" --label "$(src_label phoneworld)" "$DOCKER_DIR"
  cleanup_builder
  docker run -d --name "$BUILDER" --device /dev/kvm --privileged \
    --mount "type=bind,src=$zip_path,dst=/mnt/phoneworld_external_safe.zip,readonly" \
    --tmpfs /tmp/phoneworld-apks:rw,size=1g \
    --entrypoint sleep "$BASE_IMG" infinity >/dev/null
  log "installing 34 gated APKs and validating Chinese input"
  # Emulator 37 can terminate the invoking shell after a successful snapshot
  # save. The artifacts are authoritative in both the zero and non-zero cases.
  docker exec "$BUILDER" bash /usr/local/bin/phoneworld-apps.sh || true
  if ! docker exec "$BUILDER" test -f /tmp/phoneworld-smoke-ok \
    || ! docker exec "$BUILDER" test -s \
      /root/.android/avd/lite_avd_androidworld.avd/snapshots/default_boot/snapshot.pb; then
    log "build failed; builder kept for debugging: $BUILDER"
    exit 1
  fi
  docker commit "$BUILDER" "${IMG}-stage2-raw" >/dev/null
  cleanup_builder

  wrap_dir=$(mktemp -d)
  printf 'FROM %s\nENTRYPOINT []\nCMD ["/bin/bash"]\n' "${IMG}-stage2-raw" >"$wrap_dir/Dockerfile"
  docker build -t "$IMG" --label "$(src_label phoneworld)" "$wrap_dir" >/dev/null
  find "$wrap_dir" -type f -delete
  rmdir "$wrap_dir"
  image_rm "${IMG}-stage2-raw"
  log "built local-only image $IMG"
  log "Do not push/export it: the installed PhoneWorld APKs prohibit redistribution."
}

rebuild() {
  cleanup_builder
  image_rm "$IMG"
  image_rm "$BASE_IMG"
  build
}

status() {
  image_status_line "android base" "$ANDROID_BASE" androidworld || true
  image_status_line "build base  " "$BASE_IMG" phoneworld || true
  image_status_line "local image " "$IMG" phoneworld || true
  echo "HF revision   : $HF_REVISION"
  echo "distribution  : local-only (push/pull intentionally unsupported)"
}

case "${1:-build}" in
  build) build ;;
  rebuild) rebuild ;;
  status) status ;;
  push|pull) echo "ERROR: PhoneWorld images contain gated APKs and cannot be redistributed." >&2; exit 2 ;;
  *) echo "Usage: $0 [build|rebuild|status]" >&2; exit 1 ;;
esac
