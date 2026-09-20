#!/usr/bin/env bash
# Generate and verify lite.osworld runtime task catalogs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OSWORLD_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$OSWORLD_DIR/data"
LOCK="$DATA_DIR/catalog.lock.json"
ASSET_LOCK_HELPER="$SCRIPT_DIR/asset_lock.py"

log() { echo "[lite.osworld tasks] $*" >&2; }

generate() {
    log "generating eval catalog"
    python -m lite.gym.envs.lite.osworld.src.gen.eval
    log "generating train catalogs"
    python -m lite.gym.envs.lite.osworld.src.gen.train
}

check() {
    python - "$OSWORLD_DIR" "$LOCK" "$ASSET_LOCK_HELPER" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

env_dir = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
asset_lock_helper = Path(sys.argv[3])
lock = json.loads(lock_path.read_text())
spec = importlib.util.spec_from_file_location("lite_osworld_asset_lock", asset_lock_helper)
asset_lock = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(asset_lock)
from lite.gym.envs.lite.osworld import exclude_reasons


def catalog_summary(path):
    rows = 0
    excluded_rows = 0
    reason_counts = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}")
            reason = (
                row.get("metadata", {})
                .get("others", {})
                .get("exclude_reason")
            )
            if reason:
                try:
                    exclude_reasons.validate(reason)
                except ValueError as exc:
                    raise SystemExit(
                        f"{path}:{line_number}: invalid exclude_reason {reason!r}: {exc}"
                    )
                excluded_rows += 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "rows": rows,
        "excluded_rows": excluded_rows,
        "scored_rows": rows - excluded_rows,
        "exclude_reasons": dict(sorted(reason_counts.items())),
    }


expected_asset = asset_lock.component_identity(env_dir, "synth")
actual_asset = lock.get("sources", {}).get("asset_identity")
if actual_asset != expected_asset:
    raise SystemExit(
        f"{lock_path}: asset_identity stale\n"
        f"  expected: {expected_asset}\n"
        f"  actual:   {actual_asset}"
    )
for split, entry in sorted(lock["splits"].items()):
    path = env_dir / "data" / entry["path"]
    if not path.is_file():
        raise SystemExit(f"missing catalog for {split}: {path}")
    data = path.read_bytes()
    summary = catalog_summary(path)
    digest = hashlib.sha256(data).hexdigest()
    mismatches = []
    for field in ("rows", "excluded_rows", "scored_rows", "exclude_reasons"):
        if summary[field] != entry.get(field):
            mismatches.append(f"{field}: {summary[field]!r} != {entry.get(field)!r}")
    if digest != entry["sha256"]:
        mismatches.append(f"sha256: {digest} != {entry['sha256']}")
    if mismatches:
        raise SystemExit(
            f"{path} does not match catalog.lock.json for {split}\n"
            + "\n".join(f"  {line}" for line in mismatches)
        )
print("catalogs fresh")
PY
}

refresh_lock() {
    python - "$OSWORLD_DIR" "$LOCK" "$ASSET_LOCK_HELPER" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

env_dir = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
asset_lock_helper = Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("lite_osworld_asset_lock", asset_lock_helper)
asset_lock = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(asset_lock)
from lite.gym.envs.lite.osworld import exclude_reasons


def catalog_summary(path):
    rows = 0
    excluded_rows = 0
    reason_counts = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}")
            reason = (
                row.get("metadata", {})
                .get("others", {})
                .get("exclude_reason")
            )
            if reason:
                try:
                    exclude_reasons.validate(reason)
                except ValueError as exc:
                    raise SystemExit(
                        f"{path}:{line_number}: invalid exclude_reason {reason!r}: {exc}"
                    )
                excluded_rows += 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "rows": rows,
        "excluded_rows": excluded_rows,
        "scored_rows": rows - excluded_rows,
        "exclude_reasons": dict(sorted(reason_counts.items())),
    }


splits = {
    "eval": "eval.jsonl",
    "train.perturb": "train.perturb.jsonl",
    "train.synth": "train.synth.jsonl",
}
lock = {
    "version": 1,
    "generated": True,
    "sources": {
        "generator": "scripts/utils/tasks.sh",
        "asset_identity": asset_lock.component_identity(env_dir, "synth"),
    },
    "splits": {},
}
for split, rel in splits.items():
    path = env_dir / "data" / rel
    data = path.read_bytes()
    lock["splits"][split] = catalog_summary(path) | {
        "path": rel,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
PY
}

ensure() {
    if ! check >/dev/null 2>&1; then
        generate
        check
    else
        log "catalogs up to date"
    fi
}

case "${1:-check}" in
    generate) generate ;;
    check) check ;;
    refresh-lock) refresh_lock ;;
    ensure) ensure ;;
    status) check ;;
    *)
        echo "Usage: $0 [generate|check|refresh-lock|ensure|status]" >&2
        exit 1
        ;;
esac
