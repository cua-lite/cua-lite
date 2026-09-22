#!/bin/bash
# Install osworld_2 (OSWorld-V2) — self-contained + idempotent. EVAL-IN-CONTAINER: the V2
# `desktop_env` is baked into a DERIVED image (docker build), NOT installed on the host — so
# osworld_2 needs ZERO host `desktop_env` and coexists with osworld v1 + lite.osworld (v1) in
# one uv with no conflict. The installer pins official code, dependencies, VM, tasks, and assets:
#   image      → docker build docker/Dockerfile → cua-lite/osworld_2:osworld-v2.1-volume
#                (FROM happysixd/osworld-docker + Python venv + V2 desktop_env + docker/server.py)
#   qcow2      → HF-GATED snapshot_download xlangai/v2-image → <env>/.cache/osworld-v2-ubuntu-x86.qcow2 (sha256-verified)
#   task_class → HF-GATED snapshot_download xlangai/osworld_v2_tasks → <env>/.cache/task_class/task_*.py (108)
#   assets     → complete HF-GATED xlangai/osworld_v2_assets_gated snapshot, hash-verified
#   scan       → static source scan → <env>/.cache/task_class/_service_deps.json (website/gitlab/user_sim/volume/multi_phase/llm_judge)
# The qcow2, task classes, and assets are downloaded on the host and mounted into containers.
#
# HF auth: accept all three dataset gates in README.md, then `hf auth login`.
# Official source is cloned automatically. OSWORLD_V2_SRC may select a clean checkout
# of the pinned revision instead; upstream task/evaluator code is never patched.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh          # full install (idempotent; rebuilds if image sources changed)
#   uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh status   # image freshness / qcow2 / task-class files
#   uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh rebuild  # force-rebuild the image
#   uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh pull|push # GHCR distribution (src_hash-gated)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../../.." && pwd)"
CACHE_DIR="$ENV_DIR/.cache"
TASK_CLASS_DIR="$CACHE_DIR/task_class"
ASSET_DIR="$CACHE_DIR/osworld_v2_assets"
IMAGE="cua-lite/osworld_2:osworld-v2.1-volume"
DOCKER_DIR="$ENV_DIR/docker"
OSWORLD_V2_SRC="${OSWORLD_V2_SRC:-$CACHE_DIR/OSWorld-V2}"
export OSWORLD_V2_SRC

# Shared image-build helpers (image_is_fresh / src_label / do_pull / do_push). Freshness is gated
# on the Dockerfile, this staging script, and the selected OSWORLD_V2_SRC content identity.
# server.py is bind-mounted at runtime, so it hot-reloads without an image rebuild.
source "$REPO_ROOT/lite/gym/scripts/image_build.sh"

TASK_IDS_JSON="$ENV_DIR/data/test_v2.json"
read -r QCOW2_FILENAME QCOW2_REPO QCOW2_ZIP QCOW2_SHA256 QCOW2_SIZE TASKS_REPO HF_REVISION TASK_COUNT TASK_CLASS_IDENTITY CODE_REPO CODE_REVISION < <(
    python - "$ENV_DIR/data/release.json" "$TASK_IDS_JSON" <<'PY'
import json
import sys

release = json.load(open(sys.argv[1], encoding="utf-8"))
task_ids = json.load(open(sys.argv[2], encoding="utf-8"))["tasks"]
qcow2 = release["qcow2"]
tasks_repo = release["tasks"]["repo"]
hf_revision = release["hf_revision"]
task_count = len(task_ids)
print(
    qcow2["filename"],
    qcow2["repo"],
    qcow2["zip"],
    qcow2["sha256"],
    qcow2["size"],
    tasks_repo,
    hf_revision,
    task_count,
    f"{tasks_repo}@{hf_revision}:{task_count}",
    release["code"]["repository"],
    release["code"]["commit"],
)
PY
)
QCOW2="$CACHE_DIR/$QCOW2_FILENAME"
QCOW2_RAW_SHA256="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["qcow2"]["raw_sha256"])' "$ENV_DIR/data/release.json")"

prepare_source() {
    if [ ! -e "$OSWORLD_V2_SRC" ]; then
        mkdir -p "$(dirname "$OSWORLD_V2_SRC")"
        git clone --depth 1 --branch "$HF_REVISION" "$CODE_REPO" "$OSWORLD_V2_SRC"
    fi
    [ -e "$OSWORLD_V2_SRC/.git" ] &&
      [ "$(git -C "$OSWORLD_V2_SRC" rev-parse HEAD)" = "$CODE_REVISION" ] &&
      [ -z "$(git -C "$OSWORLD_V2_SRC" status --porcelain --untracked-files=no)" ] || {
        echo "[install] source must be a clean official $HF_REVISION checkout ($CODE_REVISION): $OSWORLD_V2_SRC" >&2
        exit 1
    }
}

build_image() {
    prepare_source
    if [ "${1:-}" != "force" ] && image_is_fresh "$IMAGE" osworld_2; then
        echo "[install] image $IMAGE up-to-date (src_hash match); skipping build (pass 'rebuild' to force)." >&2
        return 0
    fi
    [ -f "$OSWORLD_V2_SRC/pyproject.toml" ] || {
        echo "[install] ✗ OSWorld-V2 source not at $OSWORLD_V2_SRC — set OSWORLD_V2_SRC=<checkout>." >&2; exit 1; }
    echo "[install] staging V2 source into build context (docker/_vendor/OSWorld-V2) ..." >&2
    mkdir -p "$DOCKER_DIR/_vendor"
    mapfile -t rsync_excludes < <(
        python - <<'PY'
from lite.gym.envs.osworld_2.image_spec import osworld_v2_rsync_excludes
print("\n".join(osworld_v2_rsync_excludes()))
PY
    )
    rsync -a --delete "${rsync_excludes[@]}" \
      "$OSWORLD_V2_SRC/" "$DOCKER_DIR/_vendor/OSWorld-V2/"
    image_rm "$IMAGE"   # drop the stale tag so the rebuild can't ride the old label
    echo "[install] docker build $IMAGE ..." >&2
    docker build -t "$IMAGE" --label "$(src_label osworld_2)" "$DOCKER_DIR"
}

provision_qcow2() {
    mkdir -p "$CACHE_DIR"
    if [ -e "$QCOW2" ]; then
        local sz; sz="$(stat -Lc%s "$QCOW2" 2>/dev/null || echo 0)"
        if [ "$sz" = "$QCOW2_SIZE" ]; then
            echo "$QCOW2_RAW_SHA256  $QCOW2" | sha256sum --check || {
                echo "[install] qcow2 checksum mismatch; remove the corrupt cache and re-run provision" >&2
                exit 1
            }
            echo "[install] official qcow2 verified ($sz bytes); skipping download." >&2
            return 0
        fi
        echo "[install] qcow2 present but wrong size ($sz != $QCOW2_SIZE); re-downloading." >&2
        rm -f "$QCOW2"
    fi
    echo "[install] HF-gated download $QCOW2_REPO/$QCOW2_ZIP (~14 GiB zip) ..." >&2
    python - "$QCOW2_REPO" "$QCOW2_ZIP" "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["provider_images"]["docker"]["ubuntu"]["artifact_revision"])' "$OSWORLD_V2_SRC/benchmark_releases/$HF_REVISION.json")" "$CACHE_DIR" "$QCOW2_SHA256" "$QCOW2" <<'PY'
import hashlib, subprocess, sys, os
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError
repo, zipf, rev, cache, want_sha, qcow2 = sys.argv[1:7]
try:
    p = hf_hub_download(repo_id=repo, repo_type="dataset", revision=rev, filename=zipf, local_dir=cache)
except GatedRepoError:
    sys.exit(f"[install] ✗ gated repo {repo}: accept the gate on huggingface.co then `hf auth login`.")
h = hashlib.sha256()
with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
if h.hexdigest() != want_sha:
    sys.exit(f"[install] ✗ sha256 mismatch on {zipf}: {h.hexdigest()} != {want_sha}")
print(f"[install] sha256 OK; unzipping {zipf} ...", file=sys.stderr)
subprocess.run(["unzip", "-o", p, "-d", cache], check=True)
os.remove(p)
if not os.path.exists(qcow2):
    sys.exit(f"[install] ✗ expected {qcow2} after unzip, not found.")
PY
    echo "$QCOW2_RAW_SHA256  $QCOW2" | sha256sum --check
    echo "[install] qcow2 ready ($(stat -Lc%s "$QCOW2") bytes)." >&2
}

download_task_classes() {
    local revision
    revision="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["tasks"]["commit"])' "$OSWORLD_V2_SRC/benchmark_releases/$HF_REVISION.json")"
    python "$OSWORLD_V2_SRC/scripts/tools/download_osworld_v2_tasks.py" \
      --benchmark-release "$HF_REVISION" --revision "$revision" --target-dir "$TASK_CLASS_DIR"
    python - "$TASK_CLASS_DIR" "$TASK_CLASS_IDENTITY" "$OSWORLD_V2_SRC/benchmark_releases/$HF_REVISION.task_hashes.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
dest, identity, manifest = sys.argv[1:]
for name, expected in json.loads(Path(manifest).read_text())["files"].items():
    data = (Path(dest) / name).read_bytes()
    if len(data) != expected["size"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        sys.exit(f"[install] official task hash mismatch: {name}")
(Path(dest) / ".task_class_revision").write_text(identity + "\n")
print("[install] all official task hashes verified", file=sys.stderr)
PY
}

provision_assets() {
    local revision
    rm -f "$ASSET_DIR/.asset_revision"
    revision="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["assets"]["commit"])' "$OSWORLD_V2_SRC/benchmark_releases/$HF_REVISION.json")"
    python "$OSWORLD_V2_SRC/scripts/tools/download_osworld_v2_assets.py" \
      --benchmark-release "$HF_REVISION" --revision "$revision" --target-dir "$ASSET_DIR"
    python - "$ASSET_DIR" "$revision" "$HF_REVISION" <<'PY'
import hashlib, sys
from pathlib import Path
from huggingface_hub import HfApi
dest, revision, release = sys.argv[1:]
repo = "xlangai/osworld_v2_assets_gated"
files = HfApi().repo_info(repo, repo_type="dataset", revision=revision, files_metadata=True).siblings
for item in files:
    path = Path(dest) / item.rfilename
    if not path.is_file() or path.stat().st_size != item.size:
        sys.exit(f"[install] missing or incomplete official asset: {item.rfilename}")
    digest = hashlib.sha256() if item.lfs else hashlib.sha1(f"blob {item.size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    expected = item.lfs.sha256 if item.lfs else item.blob_id
    if digest.hexdigest() != expected:
        sys.exit(f"[install] official asset hash mismatch: {item.rfilename}")
(Path(dest) / ".asset_revision").write_text(f"{release} {repo}@{revision} {len(files)}\n")
print(f"[install] all {len(files)} official asset hashes verified", file=sys.stderr)
PY
}

service_scan() {
    [ -d "$TASK_CLASS_DIR" ] || { echo "[install] no task_class/ to scan; skipping." >&2; return 0; }
    echo "[install] static service scan → _service_deps.json ..." >&2
    python - "$TASK_CLASS_DIR" <<'PY'
import ast, json, os, re, glob, sys
d = sys.argv[1]; out = {}
for f in glob.glob(os.path.join(d, "**", "task_*.py"), recursive=True):
    tid = re.search(r"task_(\w+)\.py$", os.path.basename(f)).group(1)
    src = open(f, encoding="utf-8", errors="replace").read()
    dep = {}
    if re.search(r"controllers\.website|website_host_suffix|get_stateful_website|WEBSITE_HOST_SUFFIX", src): dep["website"] = True
    if re.search(r"controllers\.gitlab|GITLAB_URL|python-gitlab|import gitlab\b", src): dep["gitlab"] = True
    if "user_simulator" in src: dep["user_sim"] = True
    if "volume_size" in src:
        task_class = next((node for node in ast.parse(src).body
                           if isinstance(node, ast.ClassDef) and node.name == f"Task{tid}"), None)
        sizes = [node.value.value for node in task_class.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "volume_size"
                         for target in node.targets)
                 and isinstance(node.value, ast.Constant)] if task_class else []
        if len(sizes) != 1 or type(sizes[0]) is not int or not 1 <= sizes[0] <= 100:
            sys.exit(f"[install] unsupported volume_size in {f}; cannot provision this task safely")
        dep["volume_size"] = sizes[0]
    if re.search(r"MultiPhaseTask|def get_phases\b", src): dep["multi_phase"] = True
    # LLM-judge evaluators (~18 tasks) call desktop_env's model_client at evaluate() → need OPENAI_API_KEY.
    if re.search(r"model_client|generate_text|OSWORLD_EVAL_MODEL", src): dep["llm_judge"] = True
    if dep: out[tid] = dep
json.dump(out, open(os.path.join(d, "_service_deps.json"), "w"), indent=0, sort_keys=True)
print(f"[install] scanned; {len(out)} tasks need a service.", file=sys.stderr)
PY
}

# The docker-free prerequisites: pinned source + gated VM/tasks/assets +
# the static service scan the runtime bind-mounts into each container. NO Docker
# image is touched. Split out so a fresh git worktree (shares the host docker
# daemon but has its own gitignored .cache/) can be provisioned on its own —
# without the image stage that rebuilds the versioned cua-lite/osworld_2 image a
# co-tenant may be using. install calls this after the image; pull gates the
# published image first, then provisions. The standalone `provision` verb runs
# just this. (HF auth prerequisite applies — see header.)
provision() {
    prepare_source
    require_uv
    python -c "import huggingface_hub" || uv pip install -q huggingface_hub
    provision_qcow2
    download_task_classes
    provision_assets
    service_scan
}

status() {
    echo "image:      $(image_is_fresh "$IMAGE" osworld_2 && echo fresh || { docker image inspect "$IMAGE" >/dev/null 2>&1 && echo 'STALE (run rebuild)' || echo MISSING; })  ($IMAGE)"
    if [ -e "$QCOW2" ]; then echo "qcow2:      $(stat -Lc%s "$QCOW2" 2>/dev/null) bytes  ($QCOW2)"; else echo "qcow2:      MISSING  ($QCOW2)"; fi
    echo "task_class: $(python - "$TASK_CLASS_DIR" "$TASK_CLASS_IDENTITY" "$TASK_IDS_JSON" <<'PY'
import json
import os
import sys

dest, identity, ids_json = sys.argv[1:4]
ids = json.load(open(ids_json, encoding="utf-8"))["tasks"]
try:
    revision = open(
        os.path.join(dest, ".task_class_revision"), encoding="utf-8"
    ).read().strip()
except OSError:
    revision = ""
missing = [
    tid for tid in ids if not os.path.isfile(os.path.join(dest, f"task_{tid}.py"))
]
present = len(ids) - len(missing)
state = "FRESH" if not missing and revision == identity else "STALE"
detail = f"{present}/{len(ids)} required; revision {revision or 'MISSING'}"
if missing:
    detail += f"; {len(missing)} missing"
print(f"{state}; {detail}; expected {identity}")
PY
    )   ($TASK_CLASS_DIR)"
    echo "assets:     $(cat "$ASSET_DIR/.asset_revision" 2>/dev/null || echo MISSING)  ($ASSET_DIR)"
    echo "source:     $(git -C "$OSWORLD_V2_SRC" rev-parse HEAD 2>/dev/null || echo MISSING)  (expected $CODE_REVISION)"
    echo "kvm:        $([ -e /dev/kvm ] && echo present || echo MISSING)  /dev/kvm"
    echo "tun:        $([ -e /dev/net/tun ] && echo present || echo MISSING)  /dev/net/tun"
    echo "(this env's desktop_env is in its image; note: lite.osworld still host-installs a v1 desktop_env until it migrates)"
}

case "${1:-install}" in
    status)    status ;;
    build)     build_image; provision; echo "[install] done ✓" >&2 ;;
    rebuild)   build_image force; provision ;;
    provision) provision ;;   # docker-free source + gated files + scan; safe on a shared host
    pull)      prepare_source; do_pull "$IMAGE" osworld_2; provision ;;
    push)      do_push "$IMAGE" osworld_2 ;;
    install)   build_image; provision; echo "[install] done ✓" >&2 ;;
    *) echo "usage: install.sh [install|build|status|rebuild|provision|pull|push]   (no arg = full idempotent install)" >&2; exit 1 ;;
esac
