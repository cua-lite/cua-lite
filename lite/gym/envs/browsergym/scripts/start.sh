#!/bin/bash
# Start long-lived browsergym benchmark services (WebArena / VisualWebArena /
# MiniWoB / TimeWarp). Idempotent: re-running against already-up services is a no-op.
#
# Usage:
#   source lite/gym/envs/browsergym/scripts/start.sh <benchmark>   # URLs persist in caller's shell
#   bash   lite/gym/envs/browsergym/scripts/start.sh <benchmark>   # services start but exports die with subshell
#
# The tier-1 contract (see docs/envs.md): this script starts services and
# exports `MINIWOB_URL` / `WA_*` / `VWA_*` / `TW_*` via real `export` statements. For
# those to reach your shell, `source` this script rather than `bash`-running
# it. The env server can `source` this internally on its own startup;
# interactive users activate the venv first and then source, or set the URL
# env vars manually.
#
# Benchmarks:
#   webarena         — start Shopping, Shopping Admin, Forum, GitLab, Wikipedia,
#                      Map, Homepage
#   visualwebarena   — all webarena services + Map + Classifieds
#   miniwob          — serve cloned miniwob-plusplus HTML via `python3 -m http.server`
#   timewarp         — run the cua-lite/browsergym.timewarp container (Wiki, News,
#                      Shop) for one UI version (TIMEWARP_VERSION, 1-6)
#
# Map startup requires the OpenStreetMap assets installed by install.sh. To keep
# non-map tasks usable on lighter setups, missing map assets are reported and the
# rest of the WebArena stack still starts. Set BROWSERGYM_START_MAP=0 to skip map
# startup explicitly.
#
# Prerequisites: install.sh for the same benchmark already ran (images loaded).
#
# Env vars (optional):
#   WEBARENA_HOST         hostname services should claim (default: localhost)
#   BROWSERGYM_CACHE  tar/zim/zip location (default: lite/gym/envs/browsergym/.cache)
#   MINIWOB_PORT          port for the miniwob http.server (default: 7560).
#                         Normally set by main.py _ensure_miniwob_singleton (the
#                         host-wide shared-singleton resolver) before this runs;
#                         the 7560 default is for a bare standalone invocation.
#   TIMEWARP_VERSION      TimeWarp UI version to serve, 1-6 (default: 1).
#   TIMEWARP_{WIKI,NEWS,WEBSHOP}_PORT
#                         TimeWarp host ports (default: 6200/6201/6202). Normally
#                         set by main.py _auto_pick_timewarp_ports before this runs.
#   TIMEWARP_START_TIMEOUT_S
#                         how long to wait for the TimeWarp sites (default: 150).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BROWSERGYM_CACHE="${BROWSERGYM_CACHE:-$ENV_DIR/.cache}"
IMAGES_DIR="$BROWSERGYM_CACHE"
MINIWOB_CLONE_DIR="$BROWSERGYM_CACHE/miniwob-plusplus"
OPENSTREETMAP_DIR="$BROWSERGYM_CACHE/openstreetmap-website"
OPENSTREETMAP_TEMPLATES_DIR="$BROWSERGYM_CACHE/openstreetmap-templates"
HOST="${WEBARENA_HOST:-localhost}"

# Per-WA-service host ports. Honour any *_PORT env var the caller exports
# (browsergym/main.py's _auto_pick_webarena_ports auto-allocates a free port
# in 8500-8999 when the default is busy, so two env-servers' WA stacks don't
# collide), else fall back to the canonical published-image default.
SHOPPING_PORT="${SHOPPING_PORT:-7770}"
SHOPPING_ADMIN_PORT="${SHOPPING_ADMIN_PORT:-7780}"
REDDIT_PORT="${REDDIT_PORT:-9999}"
GITLAB_PORT="${GITLAB_PORT:-8023}"
WIKIPEDIA_PORT="${WIKIPEDIA_PORT:-8888}"
CLASSIFIEDS_PORT="${CLASSIFIEDS_PORT:-9980}"
HOMEPAGE_PORT="${HOMEPAGE_PORT:-4399}"
MAP_PORT="${MAP_PORT:-3000}"
MAP_DB_PORT="${MAP_DB_PORT:-54321}"
MINIWOB_PORT="${MINIWOB_PORT:-7560}"

# Per-env-server scope id: every container this run creates is suffixed with it
# (e.g. ``shopping-30200``) so a co-resident env-server's WA stack is a disjoint
# set of containers — our ``docker run`` never collides on a name, and our
# pre-run ``docker rm -fv`` (restart cleanup) only ever touches OUR own leftovers,
# never a co-tenant's. Passed in by _ensure_services (= CUA_LITE_ENV_SERVER_PORT
# or "default" in direct mode).
SERVER_PORT="${CUA_LITE_ENV_SERVER_PORT:-default}"

# ─── WebArena services ───────────────────────────────────────────────────────
container_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]
}

webarena_all_up() {
    # Each WA service: container running OR port answering HTTP.
    # GitLab serves 302 on /, which curl -f rejects; treat any reachable
    # response (2xx/3xx) as alive via --output /dev/null --write-out '%{http_code}'.
    local svc
    for svc in "shopping:${SHOPPING_PORT}" "shopping_admin:${SHOPPING_ADMIN_PORT}" \
               "forum:${REDDIT_PORT}" "gitlab:${GITLAB_PORT}" "wikipedia:${WIKIPEDIA_PORT}"; do
        local name="${svc%:*}-${SERVER_PORT}" port="${svc#*:}"
        if container_running "$name"; then continue; fi
        local code
        code=$(curl -s --max-time 2 -o /dev/null -w '%{http_code}' "http://localhost:${port}/" 2>/dev/null) || return 1
        case "$code" in 2??|3??) ;; *) return 1 ;; esac
    done
}

start_webarena_services() {
    if webarena_all_up; then
        echo "All WebArena containers already running — skipping docker run / configure."
        return 1   # sentinel: caller skips the post-start sleep + configure
    fi

    echo "Starting WebArena containers..."

    # Per-service idempotent (re)create: only touch a container that is NOT
    # already running. CRITICAL for the env-server's 503 cold-boot retry loop —
    # ensure_services re-enters start.sh on every client retry while the stack
    # warms, and ``webarena_all_up`` can stay false for minutes (e.g. a slow
    # Magento, or a wikipedia whose .zim is absent/partial). The OLD
    # unconditional ``docker rm -fv X; docker run X`` then tore down the
    # still-warming shopping/gitlab on EVERY retry and reset their boot → the
    # stack never converged. ``_run_wa_service`` leaves a running container be,
    # and only (re)creates a missing/exited one — so a slow service finishes
    # warming across retries instead of restarting from scratch.
    _run_wa_service() {
        local name="$1"; shift
        if container_running "$name"; then
            echo "  $name already running — skip (warming)"
            return 0
        fi
        docker rm -fv "$name" 2>/dev/null || true
        docker run "$@"
    }

    _run_wa_service shopping-${SERVER_PORT} \
        --name shopping-${SERVER_PORT} -p ${SHOPPING_PORT}:80 -d shopping_final_0712
    echo "  Shopping: http://${HOST}:${SHOPPING_PORT}"

    _run_wa_service shopping_admin-${SERVER_PORT} \
        --name shopping_admin-${SERVER_PORT} -p ${SHOPPING_ADMIN_PORT}:80 -d shopping_admin_final_0719
    echo "  Shopping Admin: http://${HOST}:${SHOPPING_ADMIN_PORT}"

    _run_wa_service forum-${SERVER_PORT} \
        --name forum-${SERVER_PORT} -p ${REDDIT_PORT}:80 -d postmill-populated-exposed-withimg
    echo "  Reddit: http://${HOST}:${REDDIT_PORT}"

    # GitLab is the slow long pole of the WA stack (the whole reason readiness gates on
    # it). The populated image bundles a full monitoring stack + auto-sizes Puma/Sidekiq
    # to the HOST cpu count — on a many-vCPU box (e.g. 384) it spawns a huge worker fleet
    # and runs Prometheus/Grafana/Alertmanager/exporters → minutes of extra CPU/IO + an
    # unstable Sidekiq, with no benefit to WA browse tasks. We trim ONLY things with no
    # functional surface for those tasks (kept deliberately conservative — no GitLab
    # *feature* like registry/KAS/repos/issues is disabled, so no eval regression):
    #   - prometheus_monitoring off  → drops ~7 monitoring procs (WA never navigates them)
    #   - puma 2 workers, sidekiq concurrency 5 → resource sizing only (still fully serves)
    #   - --shm-size=256m            → Postgres headroom (default 64m throttles it)
    # Validated live: trims gitlab-ctl from 15→10 services (Prometheus/Grafana/
    # Alertmanager/exporters gone), docker-healthy in ~2 min. Only MONITORING is
    # disabled — no GitLab feature (repos/issues/MRs/registry/web) — so WA tasks see no
    # functional change by construction. Revert this block if a WA eval ever regresses.
    # Publish container:${GITLAB_PORT} (NOT :8023). gitlab's nginx listens on the port
    # in its external_url: 8023 for the baked image, but the reconfigure below rewrites
    # external_url to localhost:${GITLAB_PORT} when 8023 was busy → nginx then listens on
    # ${GITLAB_PORT} *inside* the container. Mapping to a fixed :8023 left host→:8023
    # pointing at nothing (gitlab unreachable, endless 000) whenever 8023 was taken by a
    # co-tenant and a port was auto-picked. Container port == GITLAB_PORT holds in both
    # cases (==8023: baked nginx on 8023; !=8023: reconfigured nginx on GITLAB_PORT).
    _run_wa_service gitlab-${SERVER_PORT} \
        --name gitlab-${SERVER_PORT} -d -p ${GITLAB_PORT}:${GITLAB_PORT} \
        --shm-size=256m \
        -e GITLAB_OMNIBUS_CONFIG="prometheus_monitoring['enable'] = false; puma['worker_processes'] = 2; sidekiq['concurrency'] = 5" \
        gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
    echo "  GitLab: http://${HOST}:${GITLAB_PORT} (takes several min to boot; trimmed config)"

    if container_running "wikipedia-${SERVER_PORT}"; then
        echo "  wikipedia-${SERVER_PORT} already running — skip (warming)"
        echo "  Wikipedia: http://${HOST}:${WIKIPEDIA_PORT}"
        return 0
    fi
    docker rm -fv wikipedia-${SERVER_PORT} 2>/dev/null || true
    # The .zim in IMAGES_DIR is often a SYMLINK into a sibling repo (to share the
    # ~90GB dump across worktrees). A symlink target outside the mounted IMAGES_DIR
    # is a dangling symlink inside the container → kiwix can't open it. So overlay
    # the RESOLVED file at the path kiwix opens (readlink -f is a no-op for a real
    # file, so this is safe whether or not it's a symlink).
    _zim_real="$(readlink -f "${IMAGES_DIR}/wikipedia_en_all_maxi_2022-05.zim")"
    docker run -d --name=wikipedia-${SERVER_PORT} \
        --volume="${IMAGES_DIR}:/data" \
        --volume="${_zim_real}:/data/wikipedia_en_all_maxi_2022-05.zim:ro" \
        -p ${WIKIPEDIA_PORT}:80 \
        ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim
    echo "  Wikipedia: http://${HOST}:${WIKIPEDIA_PORT}"
    return 0
}

# ─── OpenStreetMap / Map (WA + VWA) ──────────────────────────────────────────
# This is a host-wide singleton: the map dump is static/read-only from the task
# perspective and the published port is fixed (MAP_PORT, default 3000; override
# host-wide if it collides), so there is no per-env-server port reallocation or
# container suffixing.
OPENSTREETMAP_PROJECT="openstreetmap-website"

_sed_repl() {
    printf '%s' "$1" | sed -e 's/[&|]/\\&/g'
}

_map_alive() {
    local body
    body=$(curl -sf --max-time 3 "http://localhost:${MAP_PORT}/" 2>/dev/null) || return 1
    [[ "$body" == *"OpenStreetMap"* ]]
}

prepare_map_workspace() {
    if [ "${BROWSERGYM_START_MAP:-1}" = "0" ]; then
        echo "  SKIP Map (BROWSERGYM_START_MAP=0)"
        return 1
    fi

    if [ ! -d "$OPENSTREETMAP_DIR" ]; then
        if [ ! -f "$IMAGES_DIR/openstreetmap-website.tar.gz" ]; then
            echo "  SKIP Map (OpenStreetMap source archive missing — run install.sh webarena)" >&2
            return 1
        fi
        echo "  Extracting OpenStreetMap source archive..."
        tar -xzf "$IMAGES_DIR/openstreetmap-website.tar.gz" -C "$IMAGES_DIR"
    fi

    for f in docker-compose.yml leaflet.osm.js fossgis_osrm.js; do
        if [ ! -f "$OPENSTREETMAP_TEMPLATES_DIR/$f" ]; then
            echo "  SKIP Map (template $f missing — run install.sh webarena)" >&2
            return 1
        fi
    done

    cp "$OPENSTREETMAP_TEMPLATES_DIR/docker-compose.yml" "$OPENSTREETMAP_DIR/docker-compose.yml"
    cp "$OPENSTREETMAP_TEMPLATES_DIR/leaflet.osm.js" "$OPENSTREETMAP_DIR/vendor/assets/leaflet/leaflet.osm.js"
    cp "$OPENSTREETMAP_TEMPLATES_DIR/fossgis_osrm.js" \
        "$OPENSTREETMAP_DIR/app/assets/javascripts/index/directions/fossgis_osrm.js"

    local tile_url="${OSM_TILE_SERVER_URL:-https://tile.openstreetmap.org/{z}/{x}/{y}.png}"
    local geocoding_url="${OSM_GEOCODING_SERVER_URL:-https://nominatim.openstreetmap.org/}"
    local routing_url="${OSM_ROUTING_SERVER_URL:-https://routing.openstreetmap.de}"
    local car_suffix="${OSM_CAR_SUFFIX:-/routed-car}"
    local bike_suffix="${OSM_BIKE_SUFFIX:-/routed-bike}"
    local foot_suffix="${OSM_FOOT_SUFFIX:-/routed-foot}"

    sed -i \
        -e "s|MAP_PORT|${MAP_PORT}|g" \
        -e "s|54321:5432|${MAP_DB_PORT}:5432|g" \
        "$OPENSTREETMAP_DIR/docker-compose.yml"
    sed -i "s|url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'|url: '$(_sed_repl "$tile_url")'|g" \
        "$OPENSTREETMAP_DIR/vendor/assets/leaflet/leaflet.osm.js"
    sed -i "s|nominatim_url:.*|nominatim_url: \"$(_sed_repl "$geocoding_url")\"|g" \
        "$OPENSTREETMAP_DIR/config/settings.yml"
    sed -i "s|fossgis_osrm_url:.*|fossgis_osrm_url: \"$(_sed_repl "$routing_url")\"|g" \
        "$OPENSTREETMAP_DIR/config/settings.yml"
    sed -i \
        -e "s|__OSMCarSuffix__|$(_sed_repl "$car_suffix")|g" \
        -e "s|__OSMBikeSuffix__|$(_sed_repl "$bike_suffix")|g" \
        -e "s|__OSMFootSuffix__|$(_sed_repl "$foot_suffix")|g" \
        "$OPENSTREETMAP_DIR/app/assets/javascripts/index/directions/fossgis_osrm.js"
}

start_map_services() {
    if _map_alive; then
        echo "  Map already up on port ${MAP_PORT}"
        return 1
    fi
    if ! prepare_map_workspace; then
        return 1
    fi

    echo "  Starting OpenStreetMap containers..."
    (
        cd "$OPENSTREETMAP_DIR"
        docker compose -p "$OPENSTREETMAP_PROJECT" up -d
    ) || {
        echo "  ERROR: OpenStreetMap compose up failed — check 'docker compose -p ${OPENSTREETMAP_PROJECT} logs'" >&2
        return 1
    }
    echo "  Map: http://${HOST}:${MAP_PORT}"
    return 0
}

configure_map() {
    if [ ! -f "$OPENSTREETMAP_DIR/docker-compose.yml" ]; then
        return 0
    fi

    echo "Configuring OpenStreetMap..."
    local ok=0
    for _ in $(seq 1 30); do
        if (
            cd "$OPENSTREETMAP_DIR"
            docker compose -p "$OPENSTREETMAP_PROJECT" exec -T web \
                bin/rails db:migrate RAILS_ENV=development
        ) >/dev/null 2>&1; then
            ok=1
            break
        fi
        sleep 5
    done
    if [ "$ok" = "1" ]; then
        echo "  OpenStreetMap DB migrated"
    else
        echo "  OpenStreetMap DB migrate did not complete — map may still be warming; check compose logs" >&2
    fi

    for _ in $(seq 1 60); do
        if _map_alive; then
            echo "  OpenStreetMap ready on port ${MAP_PORT}"
            return 0
        fi
        sleep 2
    done
    echo "  OpenStreetMap did not become reachable on port ${MAP_PORT}; map tasks will fail until it is ready" >&2
}

# ─── Classifieds (VisualWebArena only) ───────────────────────────────────────
# Per-env-server compose project so a co-resident env-server's classifieds
# stack is a disjoint set of containers (compose names them
# ``<project>-<service>-N``). ``docker compose -p`` scopes every up/down/ps to
# THIS env-server's project.
# Per-env-server compose project + suffixed container names so a co-resident
# env-server's classifieds stack is a disjoint set of containers. The upstream
# compose file PINS ``container_name: classifieds`` / ``classifieds_db`` (which
# defeats compose's project prefix), so we rewrite those names — and the
# hard-coded ``9980:9980`` port map — to THIS env-server's scope at start time.
CLASSIFIEDS_PROJECT="classifieds-${SERVER_PORT}"
CLASSIFIEDS_WEB="classifieds-${SERVER_PORT}"
CLASSIFIEDS_DB="classifieds_db-${SERVER_PORT}"

classifieds_all_up() {
    # Same dual-mode liveness check as webarena_all_up: both this scope's
    # containers running OR the public port answers HTTP.
    if container_running "$CLASSIFIEDS_DB" && container_running "$CLASSIFIEDS_WEB"; then
        return 0
    fi
    local code
    code=$(curl -s --max-time 2 -o /dev/null -w '%{http_code}' "http://localhost:${CLASSIFIEDS_PORT}/" 2>/dev/null) || return 1
    case "$code" in 2??|3??) return 0 ;; *) return 1 ;; esac
}

start_classifieds() {
    if [ ! -f "$IMAGES_DIR/classifieds_docker_compose.zip" ]; then
        echo "  SKIP Classifieds (zip not found — run install.sh visualwebarena first)"
        return 1   # sentinel: caller skips configure
    fi
    if classifieds_all_up; then
        echo "  Classifieds containers already running — skipping compose up / configure."
        return 1   # sentinel: caller skips configure
    fi
    echo "  Setting up Classifieds..."
    cd "$IMAGES_DIR"
    # Extract the upstream compose into a PER-ENV-SERVER directory so two env
    # servers never mutate (and race on) the same docker-compose.yml. Always
    # re-derive this scope's copy from the pristine extraction (overwriting any
    # leftover from a prior abnormal exit) so the rewrite below always applies to
    # the bare upstream tokens.
    if [ ! -d classifieds_docker_compose ]; then
        unzip -q classifieds_docker_compose.zip
    fi
    local scope_dir="classifieds_docker_compose-${SERVER_PORT}"
    rm -rf "$scope_dir"
    cp -r classifieds_docker_compose "$scope_dir"
    cd "$scope_dir"
    # Per-env-server scope: point the in-app CLASSIFIEDS URL at our host:port,
    # suffix the pinned container names, and remap the published port to
    # CLASSIFIEDS_PORT. ``container_name: classifieds`` must be matched without
    # clobbering ``classifieds_db`` — anchor the line end.
    sed -i \
        -e "s|http://[^/]*:9980/|http://${HOST}:${CLASSIFIEDS_PORT}/|g" \
        -e "s|^\(\s*container_name:\s*\)classifieds_db\s*\$|\1${CLASSIFIEDS_DB}|" \
        -e "s|^\(\s*container_name:\s*\)classifieds\s*\$|\1${CLASSIFIEDS_WEB}|" \
        -e "s|\"9980:9980\"|\"${CLASSIFIEDS_PORT}:9980\"|" \
        docker-compose.yml
    # Defensive: clear any leftover container that holds one of THIS scope's
    # names from a prior abnormal exit (a half-created compose stack, or a
    # crash between ``create`` and ``start``). ``docker compose up`` aborts with
    # "container name already in use" if such a leftover exists, so a single
    # failed boot would otherwise wedge every subsequent attempt. Scoped names
    # only — never touches a co-resident env-server's stack.
    docker rm -fv "$CLASSIFIEDS_WEB" "$CLASSIFIEDS_DB" 2>/dev/null || true
    if ! docker compose -p "$CLASSIFIEDS_PROJECT" up --build -d; then
        echo "  ERROR: classifieds compose up failed (scope ${SERVER_PORT}) —" >&2
        echo "         check 'docker compose -p ${CLASSIFIEDS_PROJECT} logs'" >&2
        return 1
    fi
    echo "  Classifieds: http://${HOST}:${CLASSIFIEDS_PORT}"
    return 0
}

# ─── Post-start configuration ────────────────────────────────────────────────
configure_webarena() {
    echo "Configuring WebArena services for hostname: ${HOST}"

    docker exec shopping-${SERVER_PORT} /var/www/magento2/bin/magento setup:store-config:set \
        --base-url="http://${HOST}:${SHOPPING_PORT}" 2>/dev/null || true
    docker exec shopping-${SERVER_PORT} mysql -u magentouser -pMyPassword magentodb -e \
        "UPDATE core_config_data SET value='http://${HOST}:${SHOPPING_PORT}/' WHERE path = 'web/secure/base_url';" 2>/dev/null || true
    docker exec shopping-${SERVER_PORT} /var/www/magento2/bin/magento cache:flush 2>/dev/null || true
    echo "  Shopping configured"

    docker exec shopping_admin-${SERVER_PORT} /var/www/magento2/bin/magento setup:store-config:set \
        --base-url="http://${HOST}:${SHOPPING_ADMIN_PORT}" 2>/dev/null || true
    docker exec shopping_admin-${SERVER_PORT} mysql -u magentouser -pMyPassword magentodb -e \
        "UPDATE core_config_data SET value='http://${HOST}:${SHOPPING_ADMIN_PORT}/' WHERE path = 'web/secure/base_url';" 2>/dev/null || true
    docker exec shopping_admin-${SERVER_PORT} php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0 2>/dev/null || true
    docker exec shopping_admin-${SERVER_PORT} php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0 2>/dev/null || true
    docker exec shopping_admin-${SERVER_PORT} /var/www/magento2/bin/magento cache:flush 2>/dev/null || true
    echo "  Shopping Admin configured"

    # GitLab nginx listens on the port in its external_url. The baked image uses :8023;
    # when 8023 was free we keep it (skip the costly reconfigure). When 8023 was taken by
    # a co-tenant and a port was auto-picked, we rewrite external_url → ${GITLAB_PORT} and
    # reconfigure, so nginx then listens on ${GITLAB_PORT} *inside* the container — which
    # is why the `docker run` above publishes container:${GITLAB_PORT} (== nginx's port in
    # both cases), NOT a fixed :8023. `gitlab-ctl reconfigure` is known to exhaust the
    # postgres pool and serve 500s, so we skip it when the baked :8023 already matches.
    if [ "${HOST}" = "localhost" ] && [ "${GITLAB_PORT}" = "8023" ]; then
        echo "  GitLab: http://${HOST}:${GITLAB_PORT} (image pre-configured — skipping reconfigure)"
    else
        docker exec gitlab-${SERVER_PORT} sed -i "s|^external_url.*|external_url 'http://${HOST}:${GITLAB_PORT}'|" \
            /etc/gitlab/gitlab.rb 2>/dev/null || true
        docker exec gitlab-${SERVER_PORT} gitlab-ctl reconfigure 2>/dev/null || true
        docker restart gitlab-${SERVER_PORT} >/dev/null 2>&1 || true
        echo "  GitLab reconfigured + restarted (workaround for postgres-pool bug)"
    fi
}

configure_classifieds() {
    # Wait for mysqld to accept queries before populating — without this the
    # populate runs before the daemon is ready, silently no-ops under
    # `2>/dev/null`, and every VWA classifieds task hits an empty catalog.
    for _ in $(seq 1 30); do
        if docker exec "$CLASSIFIEDS_DB" mysqladmin -u root -ppassword ping 2>/dev/null | grep -q 'alive'; then
            break
        fi
        sleep 2
    done
    if docker exec "$CLASSIFIEDS_DB" mysql -u root -ppassword osclass -e \
            'source docker-entrypoint-initdb.d/osclass_craigslist.sql' 2>/dev/null; then
        echo "  Classifieds DB populated"
    else
        echo "  Classifieds DB populate FAILED — check 'docker logs classifieds_db' and re-run start.sh"
    fi
}

# ─── Env-var export helpers ──────────────────────────────────────────────────
# When sourced, these actually set env vars in the caller's shell.
export_webarena_env() {
    export WA_SHOPPING="http://${HOST}:${SHOPPING_PORT}/"
    export WA_SHOPPING_ADMIN="http://${HOST}:${SHOPPING_ADMIN_PORT}/admin"
    export WA_REDDIT="http://${HOST}:${REDDIT_PORT}"
    export WA_GITLAB="http://${HOST}:${GITLAB_PORT}"
    export WA_WIKIPEDIA="http://${HOST}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
    export WA_MAP="http://${HOST}:${MAP_PORT}"
    export WA_HOMEPAGE="http://${HOST}:${HOMEPAGE_PORT}"
    # WA/VWA full-reset endpoint (OPT-IN). When isolation is engaged
    # (BROWSERGYM_CONFIG=isolation), isolation.restore_backend hits
    # "$WA_FULL_RESET/reset" + polls "$WA_FULL_RESET/status" after a *writer*
    # task closes, restoring the shared Magento/GitLab/Postmill backend to its
    # baseline before the next writer is admitted (mirrors upstream
    # browsergym.webarena.instance.full_reset). This is the ONLY mechanism that
    # clears cross-task DB residue on the writable benches — upstream itself has
    # no cheaper per-task reseed (even VWA's reset_shopping.sh is a full
    # container restart, ~1-2 min/site). It is DISABLED by default (empty) for
    # the same reason isolation is off by default: a full reset costs minutes and
    # is a throughput killer, so an operator must explicitly stand up a reset
    # service and point these at it. An operator-set value is respected verbatim;
    # otherwise both stay empty → restore_backend logs + no-ops (interleaving is
    # still prevented by the conflict gate; only RESIDUE cleanup is skipped).
    export WA_FULL_RESET="${WA_FULL_RESET:-}"
    export VWA_SHOPPING="http://${HOST}:${SHOPPING_PORT}/"
    export VWA_REDDIT="http://${HOST}:${REDDIT_PORT}"
    export VWA_WIKIPEDIA="http://${HOST}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
    export VWA_HOMEPAGE="http://${HOST}:${HOMEPAGE_PORT}"
    export VWA_FULL_RESET="${VWA_FULL_RESET:-}"
}

export_classifieds_env() {
    export VWA_CLASSIFIEDS="http://${HOST}:${CLASSIFIEDS_PORT}"
    export VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
}

# ─── MiniWoB local HTTP server ───────────────────────────────────────────────
# Probe path picked to identify miniwob specifically: ``ascending-numbers.html``
# is one of the simplest task pages in miniwob-plusplus/miniwob/html/miniwob/
# and exists in every release of the corpus. Generic ``/miniwob/`` would
# return 200 from any HTTP server (Apache, nginx, dev proxies) squatting
# the port, so we verify the response BODY contains a miniwob-framework
# JS reference instead of just trusting the status code. ``core/core.js``
# is referenced by every miniwob task page; Apache directory listings or
# substitute index.html won't carry it.
_MINIWOB_PROBE_PATH="miniwob/ascending-numbers.html"
_MINIWOB_PROBE_NEEDLE="core/core.js"

_miniwob_alive() {
    # Returns 0 (alive) only if curl gets 200 AND body contains the
    # task-name needle — catches port-squatters that return 200 for /.
    local port="$1"
    local body
    body=$(curl -sf --max-time 3 "http://localhost:${port}/${_MINIWOB_PROBE_PATH}" 2>/dev/null) || return 1
    [[ "$body" == *"$_MINIWOB_PROBE_NEEDLE"* ]]
}

start_miniwob_server() {
    if [ ! -d "$MINIWOB_CLONE_DIR/miniwob/html" ]; then
        echo "error: miniwob-plusplus not cloned at $MINIWOB_CLONE_DIR" >&2
        echo "       run: uv run --no-sync bash lite/gym/envs/browsergym/scripts/install.sh miniwob" >&2
        return 1 2>/dev/null || exit 1
    fi
    if _miniwob_alive "$MINIWOB_PORT"; then
        echo "MiniWoB HTTP server already up on port ${MINIWOB_PORT}"
        return 0
    fi
    # If something OTHER than miniwob is listening on the configured port,
    # bail with a clear message — silently picking a different port would
    # leave MINIWOB_URL pointing at the wrong place. Operator can re-run
    # with ``MINIWOB_PORT=N`` to pick a free one.
    if curl -sf --max-time 2 "http://localhost:${MINIWOB_PORT}/" -o /dev/null 2>&1; then
        echo "error: port ${MINIWOB_PORT} is in use by another service (not miniwob)." >&2
        echo "       Set MINIWOB_PORT=<free-port> and re-run." >&2
        return 1 2>/dev/null || exit 1
    fi
    # Detach from controlling terminal so the server survives if the user
    # closes the shell that ran start.sh. ``nohup`` ignores SIGHUP;
    # ``< /dev/null`` detaches stdin; ``disown`` removes it from bash's
    # job table. (setsid avoided: a new session detaches process-group
    # signalling and breaks deploy.py-style multiprocessing child spawn.)
    #
    # The ``( ... ) & disown`` shape is deliberate — backgrounding
    # ``nohup ...`` INSIDE the subshell so the subshell itself exits
    # promptly after spawning. The previous ``( cd && nohup ...) &``
    # left the subshell waiting on the long-lived http.server, holding
    # any inherited pipes open; when called via ``subprocess.run`` with
    # ``capture_output=True`` that hung the parent for the full timeout.
    (
        cd "$MINIWOB_CLONE_DIR/miniwob/html"
        nohup python3 -m http.server "$MINIWOB_PORT" \
            < /dev/null > /tmp/miniwob_serve_${USER}.log 2>&1 &
        disown 2>/dev/null || true
    )
    for _ in $(seq 1 15); do
        if _miniwob_alive "$MINIWOB_PORT"; then
            echo "MiniWoB HTTP server ready on port ${MINIWOB_PORT}"
            return 0
        fi
        sleep 1
    done
    echo "error: MiniWoB HTTP server failed to start (see /tmp/miniwob_server_${USER}.log)" >&2
    return 1 2>/dev/null || exit 1
}

export_miniwob_env() {
    export MINIWOB_URL="http://${HOST}:${MINIWOB_PORT}/miniwob/"
}

# ─── Homepage Flask (WA/VWA shared) ──────────────────────────────────────────
# Tiny Flask app from the upstream VWA repo's `environment_docker/
# webarena-homepage/` subtree. Serves the static `input_images/` referenced
# by VWA image-goal tasks (`__HOMEPAGE__/static/input_images/...`) and the
# helper pages (`calculator.html`, `scratchpad.html`, `password.html`) some
# WA tasks navigate to. install.sh drops the source into
# `$BROWSERGYM_CACHE/webarena-homepage/`; we launch it via `flask run --port
# $HOMEPAGE_PORT` (NOT `python app.py`, whose hardcoded app.run(4399) ignores
# HOMEPAGE_PORT) with the same nohup+disown shape miniwob uses. Teardown
# (cleanup.sh + the env-server reap) matches `flask run … --port <HOMEPAGE_PORT>`
# scoped by port so co-resident env-servers' homepages are never killed.
HOMEPAGE_DIR="$BROWSERGYM_CACHE/webarena-homepage"

# ─── Homepage index render (closed-world "homepage hint") ────────────────────
# The original WebArena prompt ALWAYS includes a "Homepage:" hint pointing the
# agent at $WA_HOMEPAGE, whose index.html lists every provisioned site so the
# agent can navigate the closed world (reference:
# references/webarena/agent/prompts/raw/p_cot_id_actree_2s.py L33). For that to
# work, the homepage's per-site `<a href>` links must point at the REAL running
# URLs. Upstream ships an index.html with `<your-server-hostname>:<DEFAULT_PORT>`
# placeholders and REQUIRES the operator to perl-substitute the hostname before
# serving (VWA environment_docker/README.md). cua-lite auto-picks WA ports
# (_auto_pick_webarena_ports, 8500-8999), so both the placeholder host AND the
# hardcoded default ports (7770/7780/9999/8023/8888/3000/9980) are wrong on most
# runs → every homepage link was dead.
#
# This render step REPLACES that upstream perl one-liner. It copies the unified
# `templates/index.html.tmpl` (WA cards + VWA cards merged) to the served
# `templates/index.html`, substituting each site's `__SITE_<NAME>__` host:port
# placeholder with the host:port parsed from the matching WA_*/VWA_* env var
# (already exported by export_webarena_env / export_classifieds_env). The
# trailing PATH on each link is preserved by the template. A site whose env var
# is empty/unset (e.g. Map when OSM isn't provisioned, or Classifieds under
# plain webarena) has its whole `<div class="card">…</div>` DROPPED so no broken
# link is emitted. Idempotent: always re-rendered from the pristine .tmpl.
#
# Reads the same env vars the Python client resolves, so the links are
# port-accurate and scope-aware (correct even on a 2nd env-server whose ports
# were reallocated). POSIX sed only; no perl/python dependency.
# Prefer the .tmpl install.sh dropped into the cache; fall back to the
# checked-in copy in the env's assets/ (older install, or .cache wiped).
if [ -f "$HOMEPAGE_DIR/templates/index.html.tmpl" ]; then
    HOMEPAGE_TMPL="$HOMEPAGE_DIR/templates/index.html.tmpl"
else
    HOMEPAGE_TMPL="$ENV_DIR/assets/homepage_index.html.tmpl"
fi
HOMEPAGE_INDEX="$HOMEPAGE_DIR/templates/index.html"

# Strip scheme + trailing path from a URL → bare `host:port`.
#   http://localhost:8615/            -> localhost:8615
#   http://localhost:7780/admin       -> localhost:7780
#   http://host:8888/wikipedia_.../X  -> host:8888
_url_host_port() {
    local u="$1"
    u="${u#http://}"; u="${u#https://}"   # drop scheme
    printf '%s' "${u%%/*}"                 # drop first '/' and everything after
}

# Delete the whole `<div class="card" ...>…</div>` block that contains a given
# placeholder. Used to drop a card whose site env var is unset. The template
# keeps each card as a contiguous block opened by `<div class="card"` and closed
# by the FIRST following `</div>` at card indentation — match from the card open
# preceding the placeholder through that close.
_drop_card_with() {
    local file="$1" placeholder="$2"
    # Range-delete: from the last `<div class="card"` line at or above the
    # placeholder, through the next `</div>`. Implemented with awk (portable,
    # no GNU-sed range tricks): buffer each card block, emit it only if it
    # doesn't contain the placeholder.
    awk -v ph="$placeholder" '
        /<div class="card"/ { inblock=1; buf=$0 "\n"; hit=0; next }
        inblock {
            buf = buf $0 "\n"
            if (index($0, ph)) hit=1
            if ($0 ~ /<\/div>/) { if (!hit) printf "%s", buf; inblock=0; buf="" }
            next
        }
        { print }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

# Render templates/index.html from templates/index.html.tmpl, substituting the
# live host:port for each provisioned site and dropping cards for unprovisioned
# ones. Called by start_homepage_flask BEFORE `flask run`.
render_homepage_index() {
    if [ ! -f "$HOMEPAGE_TMPL" ]; then
        # Older install without the unified template: leave whatever index.html
        # ships (back-compat). Nothing to render.
        echo "  Homepage: no index.html.tmpl — serving as-installed (no render)"
        return 0
    fi
    # Copy the template, stripping the leading documentation comment block
    # (the `<!-- … -->` that documents the placeholder convention) so the SERVED
    # page carries no `__SITE_*` / `<your-server-hostname>` strings that would
    # otherwise read as un-rendered. Deletes the first `<!--`…`-->` block only.
    awk '
        skip { if ($0 ~ /-->/) skip=0; next }
        /<!--/ && !done { skip=1; done=1; if ($0 ~ /-->/) skip=0; next }
        { print }
    ' "$HOMEPAGE_TMPL" > "$HOMEPAGE_INDEX"
    # placeholder | env-var-value pairs. Empty value => card dropped.
    local pairs="\
__SITE_SHOPPING__|${WA_SHOPPING:-${VWA_SHOPPING:-}}
__SITE_SHOPPING_ADMIN__|${WA_SHOPPING_ADMIN:-}
__SITE_REDDIT__|${WA_REDDIT:-${VWA_REDDIT:-}}
__SITE_GITLAB__|${WA_GITLAB:-}
__SITE_WIKIPEDIA__|${WA_WIKIPEDIA:-${VWA_WIKIPEDIA:-}}
__SITE_MAP__|${WA_MAP:-}
__SITE_CLASSIFIEDS__|${VWA_CLASSIFIEDS:-}"
    local ph val hp
    while IFS='|' read -r ph val; do
        [ -z "$ph" ] && continue
        if [ -z "$val" ]; then
            _drop_card_with "$HOMEPAGE_INDEX" "$ph"
            continue
        fi
        hp="$(_url_host_port "$val")"
        # Substitute the bare host:port placeholder; template keeps the path.
        sed -i "s|${ph}|${hp}|g" "$HOMEPAGE_INDEX"
    done <<EOF
$pairs
EOF
    echo "  Homepage index rendered (live site URLs substituted)"
}

# Body-needle probe: the Flask `/` route renders `index.html` (a static
# WA/VWA homepage page); look for a stable marker so port-squatters can't
# pass as "alive". `WebArena` shows up in the page title and headings.
_HOMEPAGE_PROBE_PATH=""
_HOMEPAGE_PROBE_NEEDLE="WebArena"

_homepage_alive() {
    local port="$1"
    local body
    body=$(curl -sf --max-time 3 "http://localhost:${port}/${_HOMEPAGE_PROBE_PATH}" 2>/dev/null) || return 1
    [[ "$body" == *"$_HOMEPAGE_PROBE_NEEDLE"* ]]
}

start_homepage_flask() {
    if [ ! -f "$HOMEPAGE_DIR/app.py" ] || [ ! -d "$HOMEPAGE_DIR/static/input_images" ]; then
        echo "error: webarena-homepage not installed at $HOMEPAGE_DIR" >&2
        echo "       run: uv run --no-sync bash lite/gym/envs/browsergym/scripts/install.sh visualwebarena" >&2
        return 1 2>/dev/null || exit 1
    fi
    # Render the served index.html from the template with the live site URLs
    # BEFORE (re)starting flask. Done unconditionally — even when flask is
    # already up — because Flask renders `index.html` per-request, so a fresh
    # render is picked up by an already-running homepage too (idempotent).
    render_homepage_index
    if _homepage_alive "$HOMEPAGE_PORT"; then
        echo "Homepage Flask already up on port ${HOMEPAGE_PORT}"
        return 0
    fi
    # Distinct service squatting :4399 → bail loudly so the operator picks
    # a different port rather than silently mis-routing __HOMEPAGE__.
    if curl -sf --max-time 2 "http://localhost:${HOMEPAGE_PORT}/" -o /dev/null 2>&1; then
        echo "error: port ${HOMEPAGE_PORT} is in use by another service (not webarena-homepage)." >&2
        echo "       Free the port and re-run." >&2
        return 1 2>/dev/null || exit 1
    fi
    # Find a Python interpreter that has Flask importable. The project venv
    # (active under `uv run --no-sync bash`) is the intended source; fall back to
    # `python3` for shells that source this script directly without uv.
    local PYBIN
    PYBIN=$(command -v python || command -v python3 || true)
    if [ -z "$PYBIN" ]; then
        echo "error: no python interpreter on PATH for webarena-homepage" >&2
        return 1 2>/dev/null || exit 1
    fi
    if ! "$PYBIN" -c 'import flask' 2>/dev/null; then
        echo "error: ${PYBIN} cannot import flask — run \`uv run --no-sync bash lite/gym/envs/browsergym/scripts/install.sh <benchmark>\` first" >&2
        return 1 2>/dev/null || exit 1
    fi
    # Detach: backgrounding INSIDE the subshell + disown so the subshell
    # exits immediately and any inherited pipes from a `subprocess.run`
    # caller see EOF (same shape as start_miniwob_server).
    (
        cd "$HOMEPAGE_DIR"
        # Launch via `flask run`, NOT `python app.py`: the vendored app.py hardcodes
        # `app.run(port=4399)` under `__main__`, ignoring HOMEPAGE_PORT — so a second
        # env-server (whose port was reallocated off 4399) would collide on :4399 and
        # fail to start its homepage, breaking VWA `__HOMEPAGE__` image-goal tasks.
        # `flask run` imports the module-level `app` (the `__main__` block never fires)
        # and binds the port we pass, so HOMEPAGE_PORT is honored. Log is per-port so
        # co-resident env-servers don't clobber each other's homepage log.
        FLASK_APP=app.py nohup "$PYBIN" -m flask run --host 0.0.0.0 --port "$HOMEPAGE_PORT" \
            < /dev/null > "/tmp/webarena_homepage_${USER:-unknown}_${HOMEPAGE_PORT}.log" 2>&1 &
        disown 2>/dev/null || true
    )
    for _ in $(seq 1 15); do
        if _homepage_alive "$HOMEPAGE_PORT"; then
            echo "Homepage Flask ready on port ${HOMEPAGE_PORT}"
            return 0
        fi
        sleep 1
    done
    echo "error: Homepage Flask failed to start (see /tmp/webarena_homepage_${USER:-unknown}_${HOMEPAGE_PORT}.log)" >&2
    return 1 2>/dev/null || exit 1
}

# ─── TimeWarp ────────────────────────────────────────────────────────────────
# One cua-lite/browsergym.timewarp container per env-server scope
# (``timewarp-<scope>``) serves the Wiki / News / Shop sites for ONE UI version.
# The version rides on a container label, so asking for another version (or
# rebuilding the image) recreates the container — on the host ports main.py
# passes — instead of reusing the old UI. Ports are published on loopback only:
# the browser runs on this host.
TIMEWARP_IMAGE="cua-lite/browsergym.timewarp:latest"
TIMEWARP_VERSION="${TIMEWARP_VERSION:-1}"
TIMEWARP_WIKI_PORT="${TIMEWARP_WIKI_PORT:-6200}"
TIMEWARP_NEWS_PORT="${TIMEWARP_NEWS_PORT:-6201}"
TIMEWARP_WEBSHOP_PORT="${TIMEWARP_WEBSHOP_PORT:-6202}"

_timewarp_alive() {
    # Each site opens its port only after its index has loaded, so 200 = ready.
    local url
    for url in "http://localhost:${TIMEWARP_WIKI_PORT}/" \
               "http://localhost:${TIMEWARP_NEWS_PORT}/" \
               "http://localhost:${TIMEWARP_WEBSHOP_PORT}/abc"; do
        curl -sf --max-time 3 -o /dev/null "$url" || return 1
    done
}

start_timewarp() {
    local name="timewarp-${SERVER_PORT}"
    if ! [[ "$TIMEWARP_VERSION" =~ ^[1-6]$ ]]; then
        echo "error: TIMEWARP_VERSION must be 1-6, got '${TIMEWARP_VERSION}'" >&2
        return 1
    fi
    if ! docker image inspect "$TIMEWARP_IMAGE" >/dev/null 2>&1; then
        echo "error: $TIMEWARP_IMAGE not built" >&2
        echo "       run: uv run --no-sync bash lite/gym/envs/browsergym/scripts/install.sh timewarp" >&2
        return 1
    fi
    # Reuse a running container only if it serves this version from the CURRENT
    # image build; a version switch or an image rebuild recreates it.
    local running="" want
    if container_running "$name"; then
        running="$(docker inspect -f '{{index .Config.Labels "lite.timewarp.version"}} {{.Image}}' "$name")"
    fi
    want="${TIMEWARP_VERSION} $(docker image inspect -f '{{.Id}}' "$TIMEWARP_IMAGE")"
    if [ "$running" = "$want" ]; then
        echo "  $name already serving UI version ${TIMEWARP_VERSION} — skip"
    else
        docker rm -fv "$name" >/dev/null 2>&1 || true
        docker run -d --init --name "$name" \
            --label "lite.timewarp.version=${TIMEWARP_VERSION}" \
            -e "TIMEWARP_VERSION=${TIMEWARP_VERSION}" \
            -p "127.0.0.1:${TIMEWARP_WIKI_PORT}:5000" \
            -p "127.0.0.1:${TIMEWARP_NEWS_PORT}:5001" \
            -p "127.0.0.1:${TIMEWARP_WEBSHOP_PORT}:5002" \
            "$TIMEWARP_IMAGE" >/dev/null || {
            echo "error: docker run $name failed" >&2
            return 1
        }
        echo "  started $name (UI version ${TIMEWARP_VERSION})"
    fi
    # A reused container keeps the ports it was started with.
    TIMEWARP_WIKI_PORT="$(docker port "$name" 5000/tcp | head -1 | sed 's/.*://')"
    TIMEWARP_NEWS_PORT="$(docker port "$name" 5001/tcp | head -1 | sed 's/.*://')"
    TIMEWARP_WEBSHOP_PORT="$(docker port "$name" 5002/tcp | head -1 | sed 's/.*://')"

    local deadline=$((SECONDS + ${TIMEWARP_START_TIMEOUT_S:-150}))
    until _timewarp_alive; do
        if ! container_running "$name"; then
            echo "error: $name exited while starting; last logs:" >&2
            docker logs --tail 40 "$name" >&2 || true
            return 1
        fi
        if (( SECONDS >= deadline )); then
            echo "  $name still loading its site indexes; callers retry until it serves"
            return 0
        fi
        sleep 3
    done
    echo "  TimeWarp UI version ${TIMEWARP_VERSION}: wiki :${TIMEWARP_WIKI_PORT}, news :${TIMEWARP_NEWS_PORT}, shop :${TIMEWARP_WEBSHOP_PORT}"
}

export_timewarp_env() {
    export TW_WIKI="http://localhost:${TIMEWARP_WIKI_PORT}"
    export TW_NEWS="http://localhost:${TIMEWARP_NEWS_PORT}"
    export TW_WEBSHOP="http://localhost:${TIMEWARP_WEBSHOP_PORT}/abc"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<USAGE
Usage: [source|bash] $0 <benchmark>

Benchmarks:
  webarena         start Shopping, Shopping Admin, Forum, GitLab, Wikipedia, Map
  visualwebarena   all webarena services + Map + Classifieds
  miniwob          serve miniwob-plusplus HTML via python3 -m http.server
  timewarp         run the TimeWarp sites for TIMEWARP_VERSION (1-6, default 1)

Exports MINIWOB_URL / WA_* / VWA_* / TW_* on success. Use \`source\` to keep the
exports in your shell; \`bash\` starts services only (exports die with the
subshell).
USAGE
}

case "${1:-}" in
    webarena)
        wa_need_wait=0
        map_need_wait=0
        start_webarena_services && wa_need_wait=1
        start_map_services && map_need_wait=1
        if [ "$wa_need_wait" = "1" ]; then
            echo ""
            echo "Waiting 60s for WebArena services to start..."
            sleep 60
            configure_webarena
        fi
        if [ "$map_need_wait" = "1" ]; then
            configure_map
        fi
        export_webarena_env
        start_homepage_flask
        ;;
    visualwebarena)
        wa_need_wait=0
        map_need_wait=0
        classifieds_need_wait=0
        start_webarena_services && wa_need_wait=1
        start_map_services && map_need_wait=1
        start_classifieds && classifieds_need_wait=1
        if [ "$wa_need_wait" = "1" ]; then
            echo ""
            echo "Waiting 60s for WebArena services to start..."
            sleep 60
            configure_webarena
        fi
        if [ "$map_need_wait" = "1" ]; then
            configure_map
        fi
        # configure_classifieds has its own internal mysqladmin-ping wait so
        # it can be called without an extra sleep here.
        if [ "$classifieds_need_wait" = "1" ]; then
            configure_classifieds
        fi
        export_webarena_env
        export_classifieds_env
        start_homepage_flask
        ;;
    miniwob)
        start_miniwob_server
        export_miniwob_env
        ;;
    timewarp)
        start_timewarp || { return 1 2>/dev/null || exit 1; }
        export_timewarp_env
        ;;
    ""|-h|--help)
        usage
        [ -z "${1:-}" ] && { return 1 2>/dev/null || exit 1; }
        return 0 2>/dev/null || exit 0
        ;;
    *)
        echo "error: unknown benchmark '$1'" >&2
        usage
        return 1 2>/dev/null || exit 1
        ;;
esac

return 0 2>/dev/null || exit 0
