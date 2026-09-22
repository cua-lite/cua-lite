#!/bin/bash
# Install osworld_2 (OSWorld-V2) — self-contained + idempotent. EVAL-IN-CONTAINER: the V2
# `desktop_env` is baked into a DERIVED image (docker build), NOT installed on the host — so
# osworld_2 needs ZERO host `desktop_env` and coexists with osworld v1 + lite.osworld (v1) in
# one uv with no conflict. Four things:
#   image      → docker build docker/Dockerfile → cua-lite/osworld_2:osworld-v2.1-volume
#                (FROM happysixd/osworld-docker + Python venv + V2 desktop_env + docker/server.py)
#   qcow2      → HF-GATED snapshot_download xlangai/v2-image → <env>/.cache/osworld-v2-ubuntu-x86.qcow2 (sha256-verified)
#   task_class → HF-GATED snapshot_download xlangai/osworld_v2_tasks → <env>/.cache/task_class/task_*.py (108)
#   scan       → static source scan → <env>/.cache/task_class/_service_deps.json (website/gitlab/user_sim/volume/multi_phase/llm_judge)
# The qcow2 + task_class are downloaded on the HOST and bind-mounted into each container at run time.
#
# ⚠ HF AUTH PREREQUISITE: accept the gates on https://huggingface.co/datasets/xlangai/v2-image
#   and .../xlangai/osworld_v2_tasks, then `hf auth login` (a token). The two downloads 401 otherwise.
#
# The image build needs the OSWorld-V2 source (staged into docker/_vendor/, gitignored). Set
# OSWORLD_V2_SRC to the pinned source checkout for the first build. Later builds
# can reuse the staged copy under docker/_vendor/OSWorld-V2.
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
IMAGE="cua-lite/osworld_2:osworld-v2.1-volume"
DOCKER_DIR="$ENV_DIR/docker"
OSWORLD_V2_SRC="${OSWORLD_V2_SRC:-$DOCKER_DIR/_vendor/OSWorld-V2}"
export OSWORLD_V2_SRC

# Shared image-build helpers (image_is_fresh / src_label / do_pull / do_push). Freshness is gated
# on the Dockerfile, this staging script, and the selected OSWORLD_V2_SRC content identity.
# server.py is bind-mounted at runtime, so it hot-reloads without an image rebuild.
source "$REPO_ROOT/lite/gym/scripts/image_build.sh"

TASK_IDS_JSON="$ENV_DIR/data/test_v2.json"
read -r QCOW2_FILENAME QCOW2_REPO QCOW2_ZIP QCOW2_SHA256 QCOW2_SIZE TASKS_REPO HF_REVISION TASK_COUNT TASK_CLASS_IDENTITY < <(
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
)
PY
)
QCOW2="$CACHE_DIR/$QCOW2_FILENAME"

build_image() {
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
        if [ "$sz" = "$QCOW2_SIZE" ]; then echo "[install] qcow2 present ($sz bytes); skipping." >&2; return 0; fi
        echo "[install] qcow2 present but wrong size ($sz != $QCOW2_SIZE); re-downloading." >&2
        rm -f "$QCOW2"
    fi
    require_uv
    uv pip install -q huggingface_hub
    echo "[install] HF-gated download $QCOW2_REPO/$QCOW2_ZIP (~14 GiB zip) ..." >&2
    python - "$QCOW2_REPO" "$QCOW2_ZIP" "$HF_REVISION" "$CACHE_DIR" "$QCOW2_SHA256" "$QCOW2" <<'PY'
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
    echo "[install] qcow2 ready ($(stat -Lc%s "$QCOW2") bytes)." >&2
}

download_task_classes() {
    if python - "$TASK_CLASS_DIR" "$TASK_CLASS_IDENTITY" "$TASK_IDS_JSON" <<'PY'
import json, os, sys
dest, identity, ids_json = sys.argv[1:4]
try:
    if open(os.path.join(dest, ".task_class_revision")).read().strip() != identity:
        sys.exit(1)
    ids = json.load(open(ids_json))["tasks"]
    missing = [tid for tid in ids if not os.path.isfile(os.path.join(dest, f"task_{tid}.py"))]
except OSError:
    sys.exit(1)
sys.exit(1 if missing else 0)
PY
    then
        echo "[install] task classes present ($TASK_COUNT); skipping." >&2
        return 0
    fi
    require_uv
    uv pip install -q huggingface_hub
    echo "[install] HF-gated download $TASKS_REPO (task_*.py) ..." >&2
    rm -rf "$TASK_CLASS_DIR"
    python - "$TASKS_REPO" "$HF_REVISION" "$TASK_CLASS_DIR" "$TASK_CLASS_IDENTITY" "$TASK_IDS_JSON" <<'PY'
import sys, os, json
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError
repo, rev, dest, identity, ids_json = sys.argv[1:6]
os.makedirs(dest, exist_ok=True)
try:
    snapshot_download(repo_id=repo, repo_type="dataset", revision=rev,
                      allow_patterns="task_*.py", local_dir=dest)
except GatedRepoError:
    sys.exit(f"[install] ✗ gated repo {repo}: accept the gate on huggingface.co then `hf auth login`.")
ids = json.load(open(ids_json))["tasks"]
missing = [tid for tid in ids if not os.path.isfile(os.path.join(dest, f"task_{tid}.py"))]
if missing:
    sys.exit(f"[install] ✗ missing task files after download: {', '.join(missing[:10])}")
open(os.path.join(dest, ".task_class_revision"), "w").write(identity + "\n")
print(f"[install] {len(ids)} task classes downloaded.", file=sys.stderr)
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

# The docker-free prerequisites: host qcow2 (HF-gated) + task classes (HF-gated) +
# the static service scan the runtime bind-mounts into each container. NO Docker
# image is touched. Split out so a fresh git worktree (shares the host docker
# daemon but has its own gitignored .cache/) can be provisioned on its own —
# without the image stage that rebuilds the versioned cua-lite/osworld_2 image a
# co-tenant may be using. install calls this after the image; pull gates the
# published image first, then provisions. The standalone `provision` verb runs
# just this. (HF auth prerequisite applies — see header.)
provision() {
    provision_qcow2
    download_task_classes
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
    echo "kvm:        $([ -e /dev/kvm ] && echo present || echo MISSING)  /dev/kvm"
    echo "tun:        $([ -e /dev/net/tun ] && echo present || echo MISSING)  /dev/net/tun"
    echo "(this env's desktop_env is in its image; note: lite.osworld still host-installs a v1 desktop_env until it migrates)"
}

case "${1:-install}" in
    status)    status ;;
    build)     build_image; provision; echo "[install] done ✓" >&2 ;;
    rebuild)   build_image force; provision ;;
    provision) provision ;;   # docker-free prerequisites only (host qcow2 + task classes + scan); safe on a shared host
    pull)      do_pull "$IMAGE" osworld_2; provision ;;
    push)      do_push "$IMAGE" osworld_2 ;;
    install)   build_image; provision; echo "[install] done ✓" >&2 ;;
    *) echo "usage: install.sh [install|build|status|rebuild|provision|pull|push]   (no arg = full idempotent install)" >&2; exit 1 ;;
esac
