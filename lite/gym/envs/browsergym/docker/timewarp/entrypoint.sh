#!/bin/bash
# Entrypoint for cua-lite/browsergym.timewarp: serve TimeWarp's Wiki, News and
# Shop for ONE UI version, mirroring upstream scripts/environment/run_all_env.sh.
#
#   docker run --init -e TIMEWARP_VERSION=3 -p ...:5000 -p ...:5001 -p ...:5002 <image>
#
# Container ports: Wiki 5000, News 5001, Shop 5002. The container exits as soon
# as any site exits, so the host never sees a half-served UI version.
set -euo pipefail

VERSION="${TIMEWARP_VERSION:-}"
if ! [[ "$VERSION" =~ ^[1-6]$ ]]; then
    echo "[timewarp] TIMEWARP_VERSION must be 1-6, got '${VERSION}'" >&2
    exit 2
fi

ENV_ROOT="${TIMEWARP_HOME}/env"
echo "[timewarp] serving UI version ${VERSION}: wiki :5000, news :5001, shop :5002" >&2

(cd "$ENV_ROOT/wiki" && exec python wiki_app.py "-${VERSION}" --port=5000) &
(cd "$ENV_ROOT/news" && exec python news_app.py "-${VERSION}" --port=5001) &
(cd "$ENV_ROOT/webshop" && exec python -m web_agent_site.app "${VERSION}" --port=5002 --log --attrs) &

wait -n
echo "[timewarp] a site exited; stopping the container" >&2
exit 1
