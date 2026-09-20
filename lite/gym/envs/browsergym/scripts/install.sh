#!/bin/bash
# Install script for browsergym benchmarks. Run from the cua-lite repo root
# under `uv run --no-sync bash` so the project venv is active for any nested git /
# python invocations.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/browsergym/scripts/install.sh <benchmark>
#
# Benchmarks:
#   webarena         — Shopping, Shopping Admin, Forum, GitLab, Wikipedia images
#                      + OpenStreetMap assets + webarena-homepage Flask source
#                      (for tasks that reference __HOMEPAGE__)
#   visualwebarena   — all webarena images + OpenStreetMap assets +
#                      Classifieds zip + homepage Flask source (image-goal
#                      tasks resolve __HOMEPAGE__/static/...)
#   miniwob          — clone miniwob-plusplus so it can be self-served locally
#
# Prerequisites: Docker running (webarena/vwa); git (miniwob). This script also
# installs the HOST-SIDE Playwright Chromium (browsergym drives the WA/VWA/MiniWoB
# pages via Playwright on this host, not inside a container) + its system libs.
#
# Env vars (optional):
#   BROWSERGYM_CACHE      base dir for all downloads (default:
#                         lite/gym/envs/browsergym/.cache). Webarena tars live
#                         at its root; miniwob clone at
#                         $BROWSERGYM_CACHE/miniwob-plusplus.
#   WEBARENA_IMAGES_MIRROR_DIR
#                         local directory containing official WebArena resource
#                         files. Used before network mirrors when set.
#   WEBARENA_IMAGES_MIRROR_BASE_URL
#                         HTTP(S) mirror base whose paths are
#                         $WEBARENA_IMAGES_MIRROR_BASE_URL/<resource>.
#   WEBARENA_INSTALL_MAP  set to 0 to skip the ~200GB OpenStreetMap assets.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BROWSERGYM_CACHE="${BROWSERGYM_CACHE:-$ENV_DIR/.cache}"
IMAGES_DIR="$BROWSERGYM_CACHE"
MINIWOB_CLONE_DIR="$BROWSERGYM_CACHE/miniwob-plusplus"
OPENSTREETMAP_DIR="$BROWSERGYM_CACHE/openstreetmap-website"
OPENSTREETMAP_TEMPLATES_DIR="$BROWSERGYM_CACHE/openstreetmap-templates"

# No cua-lite-owned image here, but the `uv run --no-sync bash` invocation contract
# (bare `python` + `uv pip`, never bare pip) is the same one every install.sh signs;
# it and ``require_uv`` live in the shared helper.
source "$SCRIPT_DIR/../../../scripts/image_build.sh"

# ─── WebArena image set (shared by webarena + visualwebarena) ────────────────
resource_is_valid() {
    local resource="$1"
    local path="$2"

    [ -s "$path" ] || return 1
    case "$resource" in
        shopping_final_0712.tar|shopping_admin_final_0719.tar|postmill-populated-exposed-withimg.tar|gitlab-populated-final-port8023.tar|openstreetmap-website-db.tar.gz|openstreetmap-website-web.tar.gz)
            python - "$resource" "$path" <<'PY'
import json
import tarfile
import sys

resource, path = sys.argv[1:3]
expected_tags = {
    "shopping_final_0712.tar": "shopping_final_0712:latest",
    "shopping_admin_final_0719.tar": "shopping_admin_final_0719:latest",
    "postmill-populated-exposed-withimg.tar": "postmill-populated-exposed-withimg:latest",
    "gitlab-populated-final-port8023.tar": "gitlab-populated-final-port8023:latest",
    "openstreetmap-website-db.tar.gz": "openstreetmap-website-db:latest",
    "openstreetmap-website-web.tar.gz": "openstreetmap-website-web:latest",
}
try:
    with tarfile.open(path, "r:*") as archive:
        manifest_file = archive.extractfile("manifest.json")
        if manifest_file is None:
            raise SystemExit(1)
        manifest = json.load(manifest_file)
        tags = {
            tag
            for image in manifest
            for tag in (image.get("RepoTags") or [])
        }
        if expected_tags[resource] not in tags:
            raise SystemExit(1)
        required = ["manifest.json"]
        layers = []
        for image in manifest:
            config = image.get("Config")
            if config:
                required.append(config)
            image_layers = image.get("Layers") or []
            layers.extend(image_layers)
            required.extend(image_layers)
        if len(required) == 1:
            raise SystemExit(1)

        layer_names = set(layers)
        for member_name in dict.fromkeys(required):
            member_file = archive.extractfile(member_name)
            if member_file is None:
                raise SystemExit(1)
            if member_name in layer_names:
                with tarfile.open(fileobj=member_file, mode="r|*") as layer_archive:
                    for _ in layer_archive:
                        pass
            else:
                while member_file.read(1024 * 1024):
                    pass
except (OSError, EOFError, json.JSONDecodeError, tarfile.TarError, KeyError):
    raise SystemExit(1)
PY
            ;;
        openstreetmap-website.tar.gz)
            python - "$path" <<'PY'
import tarfile
import sys

path = sys.argv[1]
required = {
    "openstreetmap-website/docker-compose.yml",
    "openstreetmap-website/config/settings.yml",
    "openstreetmap-website/vendor/assets/leaflet/leaflet.osm.js",
    "openstreetmap-website/app/assets/javascripts/index/directions/fossgis_osrm.js",
}
try:
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
    if not required <= names:
        raise SystemExit(1)
except (OSError, EOFError, tarfile.TarError):
    raise SystemExit(1)
PY
            ;;
        classifieds_docker_compose.zip)
            python - "$path" <<'PY'
import sys
import zipfile

path = sys.argv[1]
required = {
    "classifieds_docker_compose/docker-compose.yml",
    "classifieds_docker_compose/mysql/init_db.sh",
    "classifieds_docker_compose/mysql/osclass_craigslist.sql",
}
try:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise SystemExit(1)
        names = set(archive.namelist())
    if not required <= names:
        raise SystemExit(1)
except (OSError, zipfile.BadZipFile):
    raise SystemExit(1)
PY
            ;;
        wikipedia_en_all_maxi_2022-05.zim)
            python - "$path" <<'PY'
import os
import sys

path = sys.argv[1]
expected_size = 95199730590
try:
    if os.stat(path).st_size != expected_size:
        raise SystemExit(1)
    with open(path, "rb") as fh:
        raise SystemExit(0 if fh.read(4) == b"ZIM\x04" else 1)
except OSError:
    raise SystemExit(1)
PY
            ;;
        *)
            return 0
            ;;
    esac
}

stage_local_resource() {
    local resource="$1"
    local output="$2"
    local partial="$output.part"
    local mirror_dir
    local mirror_file
    local mirror_real

    [ -n "${WEBARENA_IMAGES_MIRROR_DIR:-}" ] || return 1
    if ! mirror_dir="$(cd -- "$WEBARENA_IMAGES_MIRROR_DIR" 2>/dev/null && pwd -P)"; then
        echo "  WARN WEBARENA_IMAGES_MIRROR_DIR is not readable: $WEBARENA_IMAGES_MIRROR_DIR" >&2
        return 1
    fi

    mirror_file="$mirror_dir/$resource"
    if ! resource_is_valid "$resource" "$mirror_file"; then
        echo "  WARN WEBARENA_IMAGES_MIRROR_DIR lacks valid $resource" >&2
        return 1
    fi

    if ! mirror_real="$(readlink -f -- "$mirror_file" 2>/dev/null)"; then
        echo "  WARN failed to resolve $mirror_file" >&2
        return 1
    fi
    if ! resource_is_valid "$resource" "$mirror_real"; then
        echo "  WARN WEBARENA_IMAGES_MIRROR_DIR resolved $resource to an invalid file" >&2
        return 1
    fi

    echo "  Staging $resource from WEBARENA_IMAGES_MIRROR_DIR ..."
    rm -f "$output" "$partial"
    if ln -f "$mirror_real" "$output" 2>/dev/null && resource_is_valid "$resource" "$output"; then
        return 0
    fi
    rm -f "$output"
    if ln -sf "$mirror_real" "$output" 2>/dev/null && resource_is_valid "$resource" "$output"; then
        return 0
    fi
    rm -f "$output"
    if cp -f --reflink=auto "$mirror_real" "$partial" && resource_is_valid "$resource" "$partial"; then
        mv -f "$partial" "$output"
        return 0
    fi
    rm -f "$partial"
    echo "  WARN failed to stage $resource from $mirror_file" >&2
    return 1
}

template_is_valid() {
    local resource="$1"
    local path="$2"
    [ -s "$path" ] || return 1
    case "$resource" in
        docker-compose.yml)
            grep -q "MAP_PORT:3000" "$path" && grep -q "POSTGRES_DB: openstreetmap" "$path"
            ;;
        leaflet.osm.js)
            grep -q "L.OSM" "$path"
            ;;
        fossgis_osrm.js)
            grep -q "__OSMCarSuffix__" "$path"
            ;;
        *)
            return 1
            ;;
    esac
}

classifieds_dir_is_valid() {
    local dir="$1"
    [ -f "$dir/docker-compose.yml" ] &&
        [ -f "$dir/mysql/init_db.sh" ] &&
        [ -f "$dir/mysql/osclass_craigslist.sql" ]
}

pull_docker_image_if_missing() {
    local image="$1"
    if docker image inspect "$image" >/dev/null 2>&1; then
        echo "  SKIP $image (image already loaded)"
        return 0
    fi
    echo "  Pulling $image ..."
    docker pull "$image"
}

install_classifieds_images() {
    # These are the exact images referenced by the upstream Classifieds compose.
    # Pulling them during install keeps start.sh from doing a surprise network
    # fetch on the first VWA rollout.
    pull_docker_image_if_missing "jykoh/classifieds:latest"
    pull_docker_image_if_missing "mysql:8.1"
}

openstreetmap_dir_is_valid() {
    local dir="$1"
    [ -f "$dir/docker-compose.yml" ] &&
        [ -f "$dir/config/settings.yml" ] &&
        [ -f "$dir/vendor/assets/leaflet/leaflet.osm.js" ] &&
        [ -f "$dir/app/assets/javascripts/index/directions/fossgis_osrm.js" ]
}

download_template_file() {
    local resource="$1"
    local url="$2"
    local output="$3"
    local partial="$output.part"

    if [ -e "$output" ] && ! template_is_valid "$resource" "$output"; then
        echo "  WARN removing invalid OpenStreetMap template $resource" >&2
        rm -f "$output"
    fi
    if template_is_valid "$resource" "$output"; then
        return 0
    fi
    rm -f "$partial"
    if wget -q "$url" -O "$partial" && template_is_valid "$resource" "$partial"; then
        mv -f "$partial" "$output"
        return 0
    fi
    rm -f "$partial"
    echo "  ERROR failed to download valid OpenStreetMap template $resource" >&2
    return 1
}

partial_is_resumable() {
    local path="$1"
    [ -s "$path" ] || return 1
    python - "$path" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
try:
    with path.open("rb") as fh:
        head = fh.read(2048).lstrip().lower()
except OSError:
    raise SystemExit(1)

error_prefixes = (b"<!doctype html", b"<html", b"<?xml")
error_needles = (
    b"403 forbidden",
    b"404 not found",
    b"access denied",
    b"not found",
)
if head.startswith(error_prefixes) or any(needle in head for needle in error_needles):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

download_resource_from_urls() {
    local resource="$1"
    local output="$2"
    shift 2
    local partial="$output.part"
    local url

    if [ -e "$output" ] && ! resource_is_valid "$resource" "$output"; then
        echo "  WARN removing invalid $resource at $output" >&2
        rm -f "$output"
    fi
    if resource_is_valid "$resource" "$output"; then
        return 0
    fi
    if resource_is_valid "$resource" "$partial"; then
        mv -f "$partial" "$output"
        return 0
    fi
    if [ -e "$partial" ] && ! partial_is_resumable "$partial"; then
        echo "  WARN removing non-resumable partial $partial" >&2
        rm -f "$partial"
    fi

    if stage_local_resource "$resource" "$output"; then
        return 0
    fi

    for url in "$@"; do
        if [[ "$url" == gdrive:* ]]; then
            echo "  Downloading $resource from Google Drive (${url#gdrive:}) ..."
            if python -m gdown --continue "${url#gdrive:}" -O "$partial" && resource_is_valid "$resource" "$partial"; then
                mv -f "$partial" "$output"
                return 0
            fi
        else
            echo "  Downloading $resource from $url ..."
            if wget -q -c "$url" -O "$partial" && resource_is_valid "$resource" "$partial"; then
                mv -f "$partial" "$output"
                return 0
            fi
        fi
        if [ -e "$partial" ]; then
            if partial_is_resumable "$partial"; then
                echo "  WARN keeping partial $partial for resume" >&2
            else
                echo "  WARN removing non-resumable partial $partial" >&2
                rm -f "$partial"
            fi
        fi
        echo "  WARN failed to download valid $resource from $url" >&2
    done

    return 1
}

webarena_resource_urls() {
    local resource="$1"
    local metis="http://metis.lti.cs.cmu.edu/webarena-images/$resource"
    if [ -n "${WEBARENA_IMAGES_MIRROR_BASE_URL:-}" ]; then
        printf '%s\n' "${WEBARENA_IMAGES_MIRROR_BASE_URL%/}/$resource"
    fi
    case "$resource" in
        shopping_final_0712.tar)
            printf '%s\n' \
                "gdrive:1gxXalk9O0p9eu1YkIJcmZta1nvvyAJpA" \
                "https://archive.org/download/webarena-env-shopping-image/$resource" \
                "$metis"
            ;;
        shopping_admin_final_0719.tar)
            printf '%s\n' \
                "gdrive:1See0ZhJRw0WTTL9y8hFlgaduwPZ_nGfd" \
                "https://archive.org/download/webarena-env-shopping-admin-image/$resource" \
                "$metis"
            ;;
        postmill-populated-exposed-withimg.tar)
            printf '%s\n' \
                "gdrive:17Qpp1iu_mPqzgO_73Z9BnFjHrzmX9DGf" \
                "https://archive.org/download/postmill-populated-exposed-withimg/$resource" \
                "https://archive.org/download/webarena-env-forum-image/$resource" \
                "$metis"
            ;;
        gitlab-populated-final-port8023.tar)
            printf '%s\n' \
                "gdrive:19W8qM0DPyRvWCLyQe0qtnCWAHGruolMR" \
                "https://archive.org/download/webarena-env-gitlab-image/$resource" \
                "$metis"
            ;;
        wikipedia_en_all_maxi_2022-05.zim)
            printf '%s\n' \
                "gdrive:1Um4QLxi_bGv5bP6kt83Ke0lNjuV9Tm0P" \
                "https://archive.org/download/webarena-env-wiki-image/$resource" \
                "$metis"
            ;;
        *)
            printf '%s\n' "$metis"
            ;;
    esac
}

download_webarena_resource() {
    local resource="$1"
    local output="$IMAGES_DIR/$resource"
    local -a urls

    mapfile -t urls < <(webarena_resource_urls "$resource")
    download_resource_from_urls "$resource" "$output" "${urls[@]}" && return 0
    echo "  ERROR failed to download $resource from all configured WebArena sources" >&2
    return 1
}

install_webarena_images() {
    mkdir -p "$IMAGES_DIR"
    local tars=(shopping_final_0712.tar shopping_admin_final_0719.tar \
                postmill-populated-exposed-withimg.tar gitlab-populated-final-port8023.tar)
    local zim="wikipedia_en_all_maxi_2022-05.zim"

    # Idempotent: an image already loaded into docker needs NEITHER download NOR
    # load. The multi-GB tar is usually deleted after a successful `docker load`,
    # so a naive `wget -c` (which the old version ran unconditionally, BEFORE the
    # load-skip check) would re-fetch 50-80 GB per image for nothing. Only touch
    # what's actually missing.
    local missing=()
    for tar in "${tars[@]}"; do
        if docker image inspect "${tar%.tar}" >/dev/null 2>&1; then
            echo "  SKIP ${tar%.tar} (image already loaded)"
        else
            missing+=("$tar")
        fi
    done

    # Stage/download only the tars for missing images (+ the wikipedia .zim data
    # file if absent).
    local dl=()
    for tar in "${missing[@]}"; do
        resource_is_valid "$tar" "$IMAGES_DIR/$tar" || dl+=("$tar")
    done
    resource_is_valid "$zim" "$IMAGES_DIR/$zim" || dl+=("$zim")
    if [ ${#dl[@]} -gt 0 ]; then
        echo "Downloading ${#dl[@]} missing WebArena resource(s) to $IMAGES_DIR ..."
        local failed=0
        for f in "${dl[@]}"; do
            # Download to a ``.part`` file and atomic-rename only after the
            # resource-specific validator passes, so the final path exists iff
            # the artifact is usable. Failed or invalid partials are removed
            # before trying the next source.
            download_webarena_resource "$f" || failed=1
        done
        if [ "$failed" = "1" ]; then
            return 1
        fi
    else
        echo "All WebArena images already loaded and wikipedia .zim present — nothing to download."
    fi

    # Load only the missing images.
    for tar in "${missing[@]}"; do
        if resource_is_valid "$tar" "$IMAGES_DIR/$tar"; then
            echo "  Loading $tar ..."
            docker load --input "$IMAGES_DIR/$tar"
            if ! docker image inspect "${tar%.tar}" >/dev/null 2>&1; then
                echo "  ERROR docker load did not create expected image ${tar%.tar}" >&2
                return 1
            fi
        else
            echo "  ERROR ${tar%.tar} not loaded and no valid $tar present" >&2
            return 1
        fi
    done
}

# ─── OpenStreetMap / Map (WebArena + VisualWebArena) ─────────────────────────
install_map_assets() {
    if [ "${WEBARENA_INSTALL_MAP:-1}" = "0" ]; then
        echo "Skipping OpenStreetMap assets (WEBARENA_INSTALL_MAP=0)."
        return
    fi

    mkdir -p "$IMAGES_DIR" "$OPENSTREETMAP_TEMPLATES_DIR"
    local base="https://zenodo.org/records/12636845/files"
    local db_archive="openstreetmap-website-db.tar.gz"
    local web_archive="openstreetmap-website-web.tar.gz"
    local source_archive="openstreetmap-website.tar.gz"
    local archives=("$db_archive" "$web_archive" "$source_archive")

    local dl=()
    for f in "${archives[@]}"; do
        case "$f" in
            "$db_archive")
                docker image inspect openstreetmap-website-db >/dev/null 2>&1 || resource_is_valid "$f" "$IMAGES_DIR/$f" || dl+=("$f")
                ;;
            "$web_archive")
                docker image inspect openstreetmap-website-web >/dev/null 2>&1 || resource_is_valid "$f" "$IMAGES_DIR/$f" || dl+=("$f")
                ;;
            "$source_archive")
                openstreetmap_dir_is_valid "$OPENSTREETMAP_DIR" || resource_is_valid "$f" "$IMAGES_DIR/$f" || dl+=("$f")
                ;;
        esac
    done

    if [ ${#dl[@]} -gt 0 ]; then
        echo "Downloading ${#dl[@]} OpenStreetMap resource(s) to $IMAGES_DIR ..."
        local failed=0
        for f in "${dl[@]}"; do
            download_resource_from_urls "$f" "$IMAGES_DIR/$f" "$base/$f" || failed=1
        done
        if [ "$failed" = "1" ]; then
            return 1
        fi
    else
        echo "OpenStreetMap archives/images already present — nothing to download."
    fi

    if ! docker image inspect openstreetmap-website-db >/dev/null 2>&1; then
        if ! resource_is_valid "$db_archive" "$IMAGES_DIR/$db_archive"; then
            echo "  ERROR openstreetmap-website-db image missing and no valid $db_archive present" >&2
            return 1
        else
            echo "  Loading $db_archive ..."
            docker load --input "$IMAGES_DIR/$db_archive"
            if ! docker image inspect openstreetmap-website-db >/dev/null 2>&1; then
                echo "  ERROR docker load did not create expected image openstreetmap-website-db" >&2
                return 1
            fi
        fi
    else
        echo "  SKIP openstreetmap-website-db (image already loaded)"
    fi

    if ! docker image inspect openstreetmap-website-web >/dev/null 2>&1; then
        if ! resource_is_valid "$web_archive" "$IMAGES_DIR/$web_archive"; then
            echo "  ERROR openstreetmap-website-web image missing and no valid $web_archive present" >&2
            return 1
        else
            echo "  Loading $web_archive ..."
            docker load --input "$IMAGES_DIR/$web_archive"
            if ! docker image inspect openstreetmap-website-web >/dev/null 2>&1; then
                echo "  ERROR docker load did not create expected image openstreetmap-website-web" >&2
                return 1
            fi
        fi
    else
        echo "  SKIP openstreetmap-website-web (image already loaded)"
    fi

    if openstreetmap_dir_is_valid "$OPENSTREETMAP_DIR"; then
        echo "  SKIP openstreetmap-website source (already extracted)"
    else
        if [ -d "$OPENSTREETMAP_DIR" ]; then
            echo "  WARN removing invalid OpenStreetMap source extract at $OPENSTREETMAP_DIR" >&2
            rm -rf "$OPENSTREETMAP_DIR"
        fi
        if ! resource_is_valid "$source_archive" "$IMAGES_DIR/$source_archive"; then
            echo "  ERROR OpenStreetMap source missing and no valid $source_archive present" >&2
            return 1
        else
            echo "  Extracting $source_archive ..."
            tar -xzf "$IMAGES_DIR/$source_archive" -C "$IMAGES_DIR"
            if ! openstreetmap_dir_is_valid "$OPENSTREETMAP_DIR"; then
                echo "  ERROR OpenStreetMap source extract is incomplete at $OPENSTREETMAP_DIR" >&2
                return 1
            fi
        fi
    fi

    local template_base="https://raw.githubusercontent.com/gasse/webarena-setup/main/webarena/openstreetmap-templates"
    local templates=(docker-compose.yml leaflet.osm.js fossgis_osrm.js)
    for f in "${templates[@]}"; do
        if ! download_template_file "$f" "$template_base/$f" "$OPENSTREETMAP_TEMPLATES_DIR/$f"; then
            return 1
        fi
    done
    echo "OpenStreetMap templates installed at $OPENSTREETMAP_TEMPLATES_DIR"
}

# ─── Classifieds (VisualWebArena only) ───────────────────────────────────────
install_classifieds() {
    mkdir -p "$IMAGES_DIR"
    local resource="classifieds_docker_compose.zip"
    local zip="$IMAGES_DIR/classifieds_docker_compose.zip"
    local -a urls=(
        "gdrive:1m79lp84yXfqdTBHr6IS7_1KkL4sDSemR"
        "https://archive.org/download/classifieds_docker_compose/classifieds_docker_compose.zip"
    )
    # Idempotent: skip if already extracted (start.sh unpacks it) or the zip is here.
    if classifieds_dir_is_valid "$IMAGES_DIR/classifieds_docker_compose"; then
        echo "Classifieds already extracted at $IMAGES_DIR/classifieds_docker_compose; skipping download."
        return
    fi
    if [ -d "$IMAGES_DIR/classifieds_docker_compose" ]; then
        echo "  WARN removing invalid Classifieds extract at $IMAGES_DIR/classifieds_docker_compose" >&2
        rm -rf "$IMAGES_DIR/classifieds_docker_compose"
    fi
    if resource_is_valid "$resource" "$zip"; then
        echo "Classifieds zip already present at $zip; skipping download."
        return
    fi
    echo "Downloading VisualWebArena Classifieds zip to $IMAGES_DIR ..."
    if ! download_resource_from_urls "$resource" "$zip" "${urls[@]}"; then
        echo "  ERROR failed to download VisualWebArena Classifieds zip" >&2
        return 1
    fi
    echo "Classifieds zip downloaded. (docker-compose build happens at start.sh time.)"
}

# ─── Homepage Flask (WA/VWA shared) ──────────────────────────────────────────
# VisualWebArena's image-goal tasks template `image` URLs as
# `__HOMEPAGE__/static/input_images/...`; `task.setup` runs an unconditional
# `requests.get` against that URL during `env.reset`, so a missing Flask hard-
# fails the reset with no fallback. WA tasks also reference `__HOMEPAGE__` in
# some configs (calculator, scratchpad, password). We sparse-checkout just the
# `environment_docker/webarena-homepage/` subtree of the upstream VWA repo —
# it ships the `app.py` + `templates/` + `static/input_images/` (~180 MB of
# PNGs) without dragging in the rest of the harness.
install_homepage() {
    local target="$IMAGES_DIR/webarena-homepage"
    if [ -f "$target/app.py" ] && [ -d "$target/static/input_images" ]; then
        echo "Homepage Flask already installed at $target"
        return
    fi
    mkdir -p "$IMAGES_DIR"
    local tmpdir="$IMAGES_DIR/.vwa-homepage-clone"
    rm -rf "$tmpdir"
    echo "Sparse-checkout webarena-homepage from web-arena-x/visualwebarena ..."
    git clone --depth 1 --filter=blob:none --no-checkout \
        https://github.com/web-arena-x/visualwebarena.git "$tmpdir"
    (
        cd "$tmpdir"
        git sparse-checkout init --cone
        git sparse-checkout set environment_docker/webarena-homepage
        git checkout
    )
    rm -rf "$target"
    mv "$tmpdir/environment_docker/webarena-homepage" "$target"
    rm -rf "$tmpdir"
    # Install our UNIFIED homepage template as templates/index.html.tmpl. This
    # REPLACES the upstream one-shot host substitution (VWA
    # environment_docker/README.md runs a perl `<your-server-hostname>` ->
    # hostname rewrite once at install). cua-lite instead RENDERS the served
    # index.html from this template at homepage-START time
    # (start.sh::render_homepage_index), substituting each site's live host:port
    # from the WA_*/VWA_* env vars. This is the correct home for the rewrite
    # because `_auto_pick_webarena_ports` (main.py) may move a service off its
    # default port — but only at `_ensure_services` time, long after install.sh.
    # Rendering at start time makes the portal links port-accurate + scope-aware
    # (correct even on a 2nd env-server whose ports were reallocated), and merges
    # in the sites the upstream VWA index.html omits (gitlab / shopping_admin /
    # map). The vendored upstream `templates/index.html` is left in place; the
    # render overwrites it from the .tmpl on every start.
    cp "$ENV_DIR/assets/homepage_index.html.tmpl" "$target/templates/index.html.tmpl"
    echo "Installed unified homepage template at $target/templates/index.html.tmpl"
    # The upstream checkout ships app.py's `/password.html` ROUTE but no
    # matching template, so the credential page the homepage hint points at
    # 500s. Drop in our reference-credential password.html so the route
    # resolves (creds verbatim from webarena.browser_env.env_config.ACCOUNTS).
    cp "$ENV_DIR/assets/homepage_password.html" "$target/templates/password.html"
    echo "Installed homepage password page at $target/templates/password.html"
    echo "Homepage Flask installed at $target"
}

# ─── MiniWoB (clone so it can be self-served by start.sh) ────────────────────
install_miniwob() {
    if [ -d "$MINIWOB_CLONE_DIR/miniwob/html" ]; then
        echo "miniwob-plusplus already cloned at $MINIWOB_CLONE_DIR"
        return
    fi
    echo "Cloning miniwob-plusplus to $MINIWOB_CLONE_DIR ..."
    git clone --depth 1 https://github.com/Farama-Foundation/miniwob-plusplus.git "$MINIWOB_CLONE_DIR"
}

# ─── Python deps ─────────────────────────────────────────────────────────────
install_python_deps() {
    echo "Installing Python dependencies..."
    # Pin browsergym to an EXACT version (not a `>=` floor) for reproducibility —
    # matches the convention of the other container envs (osworld pins an exact
    # git commit; android_* pin base-image tags). cua-lite's env wrapper depends
    # on browsergym INTERNALS that can shift between releases: the unconditional
    # ``_get_obs`` extraction, action ``timeout=500`` + ``retry_with_force``,
    # ``GenericWebArenaTask.validate`` (unauthorized-url judging) + the
    # ``flatten_axtree_to_str`` signature. 0.14.3 is the version this code +
    # the reference-parity audit were validated against (installed task.py /
    # instance.py are byte-identical to references/BrowserGym). The ``browsergym``
    # meta-package cascades ``==0.14.3`` to core/webarena/visualwebarena/miniwob.
    #
    # webarena/vwa transitively pull text-generation, whose stale huggingface-hub<1.0
    # cap (for HF TGI baseline agents we never run) is unsatisfiable with the model
    # stack's transformers>=5 once browsergym lands in the unified top-level venv.
    # Override the unused cap, scoped to THIS env (overrides.txt) so the repo-root
    # pyproject stays env-agnostic.
    #
    # ``nltk<3.10.1`` — nltk is an unpinned transitive of browsergym-webarena
    # (``browsergym/webarena/__init__.py`` line 1 is ``import nltk``), so uv picks
    # the newest. 3.10.1 added ``nltk/inisec.py``, a CWE-427 meta-path finder that
    # blocks any nltk-initiated import whose resolved origin lives UNDER
    # ``Path.cwd()``. It tests containment only — not whether the module was
    # actually found via the cwd ``sys.path`` entry — so a venv that sits INSIDE
    # the project dir (ours: ``<repo>/.venv``) makes every site-packages module
    # look like a cwd hijack. From the repo root ``import nltk`` therefore dies on
    # its own ``import regex`` with ``ImportError: Blocked import of regex from
    # current working directory``, which ``_load_tasks``' ``except ImportError``
    # swallows as "webarena not installed" — silently dropping all 812 WA tasks
    # from the registry for anyone running from the repo root (i.e. everyone:
    # pytest, rollout.py, serve_env.py). 3.10.0 has no such hook and is otherwise
    # identical for our use (we only need the package importable — browsergym's
    # webarena evaluator uses it for fuzzy string matching). Lift this cap once
    # upstream scopes the check to the cwd sys.path entry.
    require_uv
    uv pip install -q --override "$ENV_DIR/overrides.txt" \
        "browsergym==0.14.3" "flask>=2.0" "gdown>=5,<6" "nltk<3.10.1"
}

# BrowserGym drives WA/VWA/MiniWoB pages via Playwright on THIS host (the browser
# runs host-side — unlike the container-based envs whose Playwright lives in their
# Dockerfile), so the host needs the Chromium binary + its system libs. Same idiom
# as captcha's install.sh.
install_playwright_browser() {
    # The `browsergym` pip install pulls the playwright PACKAGE but not the browser
    # BINARY, so without this every `reset()` 500s with "Executable doesn't exist at
    # .../chromium-<rev>/...". `playwright install` is idempotent (checksums the
    # installed revision and skips the download when present). No root needed — the
    # binary lands in the user-writable ~/.cache/ms-playwright.
    echo "Installing Playwright Chromium (host-side browser) ..."
    python -m playwright install chromium
    # Chromium needs system shared libs (libatk / libnss / libgbm / …) that
    # `playwright install chromium` does NOT install; fresh Ubuntu Server images
    # lack them. This DOES need root (apt-get) — the only step here that does — so
    # degrade gracefully when neither root nor passwordless sudo is available:
    # warn + print the manual command rather than failing the install. Idempotent
    # (apt-get of already-present packages is a no-op).
    if [ "$(id -u)" = "0" ]; then
        python -m playwright install-deps chromium
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo env PATH="$PATH" python -m playwright install-deps chromium
    else
        echo "WARNING: passwordless sudo unavailable; skipping Chromium system libs." >&2
        echo "  If the browser fails to launch, run:  sudo python -m playwright install-deps chromium" >&2
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<USAGE
Usage: $0 <benchmark>

Benchmarks:
  webarena         install webarena Docker images + OpenStreetMap assets
  visualwebarena   install webarena images + OpenStreetMap assets + classifieds zip
  miniwob          clone miniwob-plusplus for local self-serving
USAGE
}

case "${1:-}" in
    webarena)
        install_python_deps
        install_playwright_browser
        install_webarena_images
        install_map_assets
        install_homepage
        ;;
    visualwebarena)
        install_python_deps
        install_playwright_browser
        install_webarena_images
        install_map_assets
        install_classifieds
        install_classifieds_images
        install_homepage
        ;;
    miniwob)
        install_python_deps
        install_playwright_browser
        install_miniwob
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
