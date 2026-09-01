#!/usr/bin/env bash
set -euo pipefail
if [ -n "${SESSION_ID:-}" ]; then
  safe_session="${SESSION_ID//[^[:alnum:]_]/_}"
  name_filter="${safe_session}-phoneworld-"
else
  name_filter="-phoneworld-"
fi
docker ps -aq --filter "name=$name_filter" | xargs -r docker rm -fv 2>/dev/null || true
docker rm -fv lite-phoneworld-apps-installer >/dev/null 2>&1 || true
