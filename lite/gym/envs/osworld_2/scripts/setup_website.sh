#!/usr/bin/env bash
# Provision the matching upstream websites. Extra arguments are native Compose options.
# Usage: HOST_SUFFIX=<VM-reachable DNS suffix> uv run --no-sync bash <script> [-p <project>] [-f <override.yaml>]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OSWORLD_V2_SRC="${OSWORLD_V2_SRC:-$ENV_DIR/.cache/OSWorld-V2}"
WEBSITE_SRC="${OSWORLD_WEBSITE_SRC:-$ENV_DIR/.cache/OSWorld-web}"
read -r repository revision < <(python - "$ENV_DIR/data/release.json" "$OSWORLD_V2_SRC" <<'PY'
import json, sys
from pathlib import Path
release = json.loads(Path(sys.argv[1]).read_text())["hf_revision"]
website = json.loads((Path(sys.argv[2]) / "benchmark_releases" / f"{release}.json").read_text())["website_code"]
print(website["repository"], website["commit"])
PY
)
: "${HOST_SUFFIX:?Set HOST_SUFFIX to a DNS suffix reachable from the VM (see README)}"
if [ ! -e "$WEBSITE_SRC" ]; then
    git clone "https://github.com/$repository.git" "$WEBSITE_SRC"
    git -C "$WEBSITE_SRC" checkout --detach "$revision"
fi
[ -e "$WEBSITE_SRC/.git" ] && [ "$(git -C "$WEBSITE_SRC" rev-parse HEAD)" = "$revision" ] || {
    echo "Website source must match official revision $revision: $WEBSITE_SRC" >&2
    exit 1
}
git -C "$WEBSITE_SRC" -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive
[ -z "$(git -C "$WEBSITE_SRC" status --porcelain --untracked-files=no)" ] || {
    echo "Website source and submodules must be unmodified: $WEBSITE_SRC" >&2
    exit 1
}
docker compose --project-directory "$WEBSITE_SRC" -f "$WEBSITE_SRC/docker-compose.yml" "$@" up -d --build
