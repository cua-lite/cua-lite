#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname "$0")/cleanup.sh"
docker image rm cua-lite/phoneworld:latest cua-lite/phoneworld:base >/dev/null 2>&1 || true
