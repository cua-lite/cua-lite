#!/bin/bash
# Stop and remove browsergym benchmark service containers.
# Does NOT delete Docker images or downloaded tar files (use uninstall.sh for that).
#
# Usage:
#   bash lite/gym/envs/browsergym/scripts/cleanup.sh <benchmark>
#
# Benchmarks:
#   webarena         — stop/remove shopping, shopping_admin, forum, gitlab,
#                      wikipedia, and the webarena-homepage Flask
#   visualwebarena   — all webarena containers + classifieds compose stack
#                      + webarena-homepage Flask
#   miniwob          — kill the local http.server serving miniwob-plusplus
#   timewarp         — stop/remove this scope's timewarp-<scope> container

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BROWSERGYM_CACHE="${BROWSERGYM_CACHE:-$ENV_DIR/.cache}"
IMAGES_DIR="$BROWSERGYM_CACHE"
# Repo root holds the shared .tmp/ dir (port reservations + miniwob registry).
REPO_ROOT="$(cd "$ENV_DIR/../../../.." && pwd)"
MINIWOB_REGISTRY="$REPO_ROOT/.tmp/miniwob-singleton.json"

# Per-env-server scope id — every WA/VWA container start.sh creates is suffixed
# with it (e.g. ``shopping-30200``). Cleanup stops/removes ONLY this scope's
# containers, never the bare global names, so one env-server's cleanup never
# takes down a co-resident env-server's stack. Matches start.sh's
# SERVER_PORT="${CUA_LITE_ENV_SERVER_PORT:-default}".
SERVER_PORT="${CUA_LITE_ENV_SERVER_PORT:-default}"
# Homepage Flask is scoped by its auto-picked HOMEPAGE_PORT (NOT by SERVER_PORT).
# Only reap it when HOMEPAGE_PORT is explicitly provided, so we never guess and
# kill a co-resident env-server's homepage (e.g. the shared default :4399).
HOMEPAGE_PORT="${HOMEPAGE_PORT:-}"

stop_webarena_services() {
    echo "Stopping WebArena containers (scope ${SERVER_PORT})..."
    for name in shopping shopping_admin forum gitlab wikipedia; do
        docker stop "${name}-${SERVER_PORT}" 2>/dev/null && docker rm -v "${name}-${SERVER_PORT}" 2>/dev/null || true
    done
}

stop_classifieds() {
    # start.sh brings classifieds up under compose project ``classifieds-<scope>``
    # in a per-scope extraction dir; tear down the SAME project + dir only.
    local project="classifieds-${SERVER_PORT}"
    local scope_dir="$IMAGES_DIR/classifieds_docker_compose-${SERVER_PORT}"
    # ``-v --remove-orphans``: a plain ``down`` leaves the named MySQL data
    # volume (``*_db_data``, 100s of MB) AND the project network behind, so a
    # crash-restart cycle would leak one volume+network per scope. A host teardown
    # must reclaim them. (The env-server's boot-reap now does this too —
    # ``_classifieds_compose_down_scoped`` in main.py runs the SAME scoped
    # ``compose down -v`` before serving — so a server restart is already a clean
    # baseline; this manual path is the host-side / direct-mode equivalent.)
    if [ -d "$scope_dir" ]; then
        echo "Stopping Classifieds compose stack (scope ${SERVER_PORT})..."
        (cd "$scope_dir" && docker compose -p "$project" down -v --remove-orphans 2>/dev/null) || true
    else
        # Fall back to project-only teardown (dir may have been cleaned) so the
        # containers still go away.
        docker compose -p "$project" down -v --remove-orphans 2>/dev/null || true
    fi
}

stop_miniwob() {
    # miniwob is a HOST-WIDE shared singleton (see main.py
    # _ensure_miniwob_singleton): this is the ONLY place that stops it (the
    # env-server never does — that would cross-kill a co-resident server). As a
    # host teardown we kill the registered port (if any) + the whole reserved
    # range, then drop the registry so the next start re-picks cleanly.
    local ports=""
    if [ -n "${MINIWOB_PORT:-}" ]; then
        ports="$MINIWOB_PORT"                      # explicit operator override
    else
        if [ -f "$MINIWOB_REGISTRY" ]; then
            ports="$(grep -oE '"port"[: ]*[0-9]+' "$MINIWOB_REGISTRY" | grep -oE '[0-9]+' || true)"
        fi
        # Plus the whole reserved range, in case the registry was lost.
        ports="$ports $(seq 7560 7659)"
    fi
    for p in $ports; do
        pkill -9 -f "python3 -m http\\.server ${p}" 2>/dev/null || true
    done
    rm -f "$MINIWOB_REGISTRY" 2>/dev/null || true
}

stop_homepage_flask() {
    # The homepage launches via `python -m flask run --port <HOMEPAGE_PORT>`
    # (start.sh), NOT `python app.py` — FLASK_APP is an env var invisible to
    # `pkill -f`, so the old `…/app.py` match reaped NOTHING (homepage leaked).
    # Match the scoped flask-run launcher BY PORT, and only when HOMEPAGE_PORT is
    # set, so a co-resident env-server's homepage is never killed.
    [ -n "$HOMEPAGE_PORT" ] || return 0
    pkill -9 -f "flask run .*--port ${HOMEPAGE_PORT}([^0-9]|$)" 2>/dev/null || true
}

stop_timewarp() {
    echo "Stopping TimeWarp container (scope ${SERVER_PORT})..."
    docker rm -fv "timewarp-${SERVER_PORT}" >/dev/null 2>&1 || true
}

usage() {
    cat <<USAGE
Usage: $0 <benchmark>

Benchmarks:
  webarena         stop shopping, shopping_admin, forum, gitlab, wikipedia
  visualwebarena   stop all webarena services + classifieds
  miniwob          kill local http.server on MINIWOB_PORT (default 7560)
  timewarp         stop/remove timewarp-<scope>
USAGE
}

case "${1:-}" in
    webarena)
        stop_webarena_services
        stop_homepage_flask
        echo "WebArena stopped."
        ;;
    visualwebarena)
        stop_webarena_services
        stop_classifieds
        stop_homepage_flask
        echo "VisualWebArena stopped."
        ;;
    miniwob)
        stop_miniwob
        echo "MiniWoB HTTP server killed (if running)."
        ;;
    timewarp)
        stop_timewarp
        echo "TimeWarp stopped."
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
