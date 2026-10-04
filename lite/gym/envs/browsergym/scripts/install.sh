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
#   timewarp         — browsergym-timewarp + nltk data, and build the
#                      cua-lite/browsergym.timewarp image (Wiki, News, Shop sites;
#                      pinned upstream source + site data, ~4 GB uncompressed)
#
# Prerequisites: Docker running (webarena/vwa/timewarp); git (miniwob). This script also
# installs the HOST-SIDE Playwright Chromium (browsergym drives the WA/VWA/MiniWoB
# pages via Playwright on this host, not inside a container) + its system libs.
#
# Env vars (optional):
#   BROWSERGYM_CACHE      base dir for all downloads (default:
#                         lite/gym/envs/browsergym/.cache). Webarena tars live
#                         at its root; miniwob clone at
#                         $BROWSERGYM_CACHE/miniwob-plusplus.
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
install_webarena_images() {
    mkdir -p "$IMAGES_DIR"
    local base="http://metis.lti.cs.cmu.edu/webarena-images"
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

    # Download only the tars for missing images (+ the wikipedia .zim data file if absent).
    local dl=()
    for tar in "${missing[@]}"; do
        [ -f "$IMAGES_DIR/$tar" ] || dl+=("$tar")
    done
    [ -f "$IMAGES_DIR/$zim" ] || dl+=("$zim")
    if [ ${#dl[@]} -gt 0 ]; then
        echo "Downloading ${#dl[@]} missing WebArena resource(s) to $IMAGES_DIR ..."
        for f in "${dl[@]}"; do
            # Download to a ``.part`` file and atomic-rename only on success, so
            # the FINAL path exists iff the download completed. A killed /
            # interrupted wget (e.g. a 90GB .zim aborted midway) then leaves the
            # ``.part`` for ``-c`` to resume next run, instead of leaving a
            # truncated file at the final path that the ``[ -f ]`` existence
            # check above would wrongly treat as "present, nothing to download".
            ( wget -q -c "$base/$f" -O "$IMAGES_DIR/$f.part" \
                && mv -f "$IMAGES_DIR/$f.part" "$IMAGES_DIR/$f" ) &
        done
        wait
    else
        echo "All WebArena images already loaded and wikipedia .zim present — nothing to download."
    fi

    # Load only the missing images.
    for tar in "${missing[@]}"; do
        if [ -f "$IMAGES_DIR/$tar" ]; then
            echo "  Loading $tar ..."
            docker load --input "$IMAGES_DIR/$tar"
        else
            echo "  WARN ${tar%.tar} not loaded and $tar absent (download may have failed)" >&2
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
                docker image inspect openstreetmap-website-db >/dev/null 2>&1 || [ -f "$IMAGES_DIR/$f" ] || dl+=("$f")
                ;;
            "$web_archive")
                docker image inspect openstreetmap-website-web >/dev/null 2>&1 || [ -f "$IMAGES_DIR/$f" ] || dl+=("$f")
                ;;
            "$source_archive")
                [ -d "$OPENSTREETMAP_DIR" ] || [ -f "$IMAGES_DIR/$f" ] || dl+=("$f")
                ;;
        esac
    done

    if [ ${#dl[@]} -gt 0 ]; then
        echo "Downloading ${#dl[@]} OpenStreetMap resource(s) to $IMAGES_DIR ..."
        for f in "${dl[@]}"; do
            ( wget -q -c "$base/$f" -O "$IMAGES_DIR/$f.part" \
                && mv -f "$IMAGES_DIR/$f.part" "$IMAGES_DIR/$f" ) &
        done
        wait
    else
        echo "OpenStreetMap archives/images already present — nothing to download."
    fi

    if ! docker image inspect openstreetmap-website-db >/dev/null 2>&1; then
        if [ ! -f "$IMAGES_DIR/$db_archive" ]; then
            echo "  WARN openstreetmap-website-db image missing and $db_archive absent" >&2
        else
            echo "  Loading $db_archive ..."
            docker load --input "$IMAGES_DIR/$db_archive"
        fi
    else
        echo "  SKIP openstreetmap-website-db (image already loaded)"
    fi

    if ! docker image inspect openstreetmap-website-web >/dev/null 2>&1; then
        if [ ! -f "$IMAGES_DIR/$web_archive" ]; then
            echo "  WARN openstreetmap-website-web image missing and $web_archive absent" >&2
        else
            echo "  Loading $web_archive ..."
            docker load --input "$IMAGES_DIR/$web_archive"
        fi
    else
        echo "  SKIP openstreetmap-website-web (image already loaded)"
    fi

    if [ ! -d "$OPENSTREETMAP_DIR" ]; then
        if [ ! -f "$IMAGES_DIR/$source_archive" ]; then
            echo "  WARN OpenStreetMap source missing and $source_archive absent" >&2
        else
            echo "  Extracting $source_archive ..."
            tar -xzf "$IMAGES_DIR/$source_archive" -C "$IMAGES_DIR"
        fi
    else
        echo "  SKIP openstreetmap-website source (already extracted)"
    fi

    local template_base="https://raw.githubusercontent.com/gasse/webarena-setup/main/webarena/openstreetmap-templates"
    local templates=(docker-compose.yml leaflet.osm.js fossgis_osrm.js)
    for f in "${templates[@]}"; do
        if [ ! -f "$OPENSTREETMAP_TEMPLATES_DIR/$f" ]; then
            wget -q -c "$template_base/$f" -O "$OPENSTREETMAP_TEMPLATES_DIR/$f"
        fi
    done
    echo "OpenStreetMap templates installed at $OPENSTREETMAP_TEMPLATES_DIR"
}

# ─── Classifieds (VisualWebArena only) ───────────────────────────────────────
install_classifieds() {
    mkdir -p "$IMAGES_DIR"
    local zip="$IMAGES_DIR/classifieds_docker_compose.zip"
    # Idempotent: skip if already extracted (start.sh unpacks it) or the zip is here.
    if [ -d "$IMAGES_DIR/classifieds_docker_compose" ]; then
        echo "Classifieds already extracted at $IMAGES_DIR/classifieds_docker_compose; skipping download."
        return
    fi
    if [ -f "$zip" ]; then
        echo "Classifieds zip already present at $zip; skipping download."
        return
    fi
    echo "Downloading VisualWebArena Classifieds zip to $IMAGES_DIR ..."
    wget -q -c "https://archive.org/download/classifieds_docker_compose/classifieds_docker_compose.zip" -O "$zip"
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

# ─── TimeWarp (sites image + host package) ───────────────────────────────────
# The three TimeWarp sites run in cua-lite/browsergym.timewarp (built from
# ../docker/timewarp; pyserini + Java 21 stay out of the host venv). The host side
# is the BrowserGym task package, whose __init__ fetches nltk's punkt_tab on first
# import when it is missing — fetch it here so env resets never hit the network.
TIMEWARP_IMAGE="cua-lite/browsergym.timewarp:latest"

install_timewarp_nltk_data() {
    python -c "import nltk; nltk.download('punkt_tab', quiet=True, raise_on_error=True)"
}

build_timewarp_image() {
    if image_is_fresh "$TIMEWARP_IMAGE" browsergym.timewarp; then
        echo "$TIMEWARP_IMAGE up to date (build sources unchanged); skipping build."
        return
    fi
    image_rm "$TIMEWARP_IMAGE"
    echo "docker build -t $TIMEWARP_IMAGE (pinned TimeWarp source + site data) ..."
    DOCKER_BUILDKIT=1 docker build -t "$TIMEWARP_IMAGE" \
        --label "$(src_label browsergym.timewarp)" \
        -f "$ENV_DIR/docker/timewarp/Dockerfile" "$ENV_DIR/docker/timewarp"
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
    #
    # Extra args are benchmark packages resolved in the SAME call, under the same
    # overrides (``timewarp`` passes browsergym-timewarp, which also pulls
    # browsergym-webarena).
    require_uv
    uv pip install -q --override "$ENV_DIR/overrides.txt" \
        "browsergym==0.14.3" "flask>=2.0" "nltk<3.10.1" "$@"
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
  timewarp         install browsergym-timewarp + build the TimeWarp sites image
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
        install_homepage
        ;;
    miniwob)
        install_python_deps
        install_playwright_browser
        install_miniwob
        ;;
    timewarp)
        install_python_deps "browsergym-timewarp==0.2.1"
        install_timewarp_nltk_data
        install_playwright_browser
        build_timewarp_image
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
