#!/bin/bash
# Provision the osworld_2 gated asset snapshot — self-contained, idempotent, resumable.
#
# Downloads the complete xlangai/osworld_v2_assets_gated set pinned by data/release.json
# into <env>/.cache/assets/ and stamps .assets_revision (required by the _check_assets gate
# in main.py; the container mounts the dir :ro at /assets with OSWORLD_FILE_BASE_URL=/assets).
# An interrupted download of the same pin resumes on re-run instead of starting over.
# The public xlangai/osworld_v2_assets repo is only a minimal subset — this release's assets
# 404 there and the nested retries blow the 300 s reset RPC timeout.
#
# SEPARATE from install.sh on purpose: install.sh is an image-freshness hash input
# (image_spec.py), and this host-only download must not re-stale built images.
#
# ⚠ Accept the HF gate on xlangai/osworld_v2_assets_gated, then `hf auth login`
# (HF_ENDPOINT honored for mirrors).
#
# Usage: uv run --no-sync bash lite/gym/envs/osworld_2/scripts/provision_assets.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../../.." && pwd)"
ASSETS_DIR="$ENV_DIR/.cache/assets"
STAMP="$ASSETS_DIR/.assets_revision"    # written after a complete download; read by main.py
PARTIAL="$ASSETS_DIR/.assets_partial"   # the pin an in-progress download belongs to

# Shared invocation contract (python = the venv interpreter; uv pip, never bare pip).
source "$REPO_ROOT/lite/gym/scripts/image_build.sh"

read -r ASSETS_REPO HF_REVISION ASSETS_IDENTITY < <(
    python - "$ENV_DIR/data/release.json" <<'PY'
import json
import sys

release = json.load(open(sys.argv[1], encoding="utf-8"))
assets_repo = release["assets"]["repo"]
hf_revision = release["hf_revision"]
print(assets_repo, hf_revision, f"{assets_repo}@{hf_revision}")
PY
)

if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$ASSETS_IDENTITY" ]; then
    echo "[provision_assets] asset snapshot present ($ASSETS_IDENTITY); skipping." >&2
    exit 0
fi

require_uv
uv pip install -q huggingface_hub
echo "[provision_assets] HF-gated download $ASSETS_REPO@$HF_REVISION (complete asset snapshot, ~3.4 GiB) ..." >&2
# snapshot_download RESUMES into a pre-existing local_dir, keeping files the current pin no
# longer lists. Resume only a partial download of this same pin; start clean otherwise
# (another pin's snapshot or partial, or an unmarked dir) so the dir mirrors the pin exactly.
if [ -f "$PARTIAL" ] && [ "$(cat "$PARTIAL")" = "$ASSETS_IDENTITY" ]; then
    echo "[provision_assets] resuming interrupted download." >&2
else
    rm -rf "$ASSETS_DIR"
    mkdir -p "$ASSETS_DIR"
    echo "$ASSETS_IDENTITY" > "$PARTIAL"
fi
python - "$ASSETS_REPO" "$HF_REVISION" "$ASSETS_DIR" "$ASSETS_IDENTITY" <<'PY'
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError
repo, rev, dest, identity = sys.argv[1:5]
try:
    snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, local_dir=dest)
except GatedRepoError:
    sys.exit(f"[provision_assets] gated repo {repo}: accept the gate on huggingface.co then `hf auth login`.")
open(f"{dest}/.assets_revision", "w").write(identity + "\n")
print(f"[provision_assets] asset snapshot ready ({identity}).", file=sys.stderr)
PY
rm -f "$PARTIAL"
