#!/bin/bash
# Dump WebHarbor-backed WebVoyager task metadata from the docker image into a
# checked-in JSON file the host reads at module-import time. This mirrors
# mobilegym/scripts/utils/tasks.sh: the host never needs a local WebHarbor/WebVoyager
# checkout, and the task manifest is generated from the same pinned upstream
# repos baked into cua-lite/webharbor.webvoyager:latest.
#
# Re-run this whenever the WebVoyager/WebHarbor pins in docker/Dockerfile move.
# Output is deterministic — no output-changing env knobs, sorted keys — so a
# clean re-run produces a byte-identical data/tasks.json (the manifest
# drift-check, D11).
#
# Usage:
#   bash lite/gym/envs/webharbor/webvoyager/scripts/utils/tasks.sh
#
# Pre-req: cua-lite/webharbor.webvoyager:latest image must exist (run scripts/install.sh first).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OUT="$SCRIPT_DIR/../../data/tasks.json"
IMAGE="cua-lite/webharbor.webvoyager:latest"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: $IMAGE not found. Run install.sh first." >&2
    exit 1
fi

DUMP_PY='
import json
import re
import sys
from pathlib import Path

sites_dir = Path("/opt/WebSyn")
webvoyager_data_dir = Path("/opt/webharbor.webvoyager/data")
out_path = Path("/tmp/tasks.json")
# Fixed, knob-free for reproducibility: emit ALL 643 tasks (each carries its own
# others.mutating flag for the strict two-pass rollout recipe), URLs normalized
# to the loopback host the in-container WebSyn mirror serves.
local_host = "127.0.0.1"

if not sites_dir.is_dir():
    raise SystemExit(f"missing WebHarbor sites dir in image: {sites_dir}")

SITE_ORDER = [
    "allrecipes",
    "amazon",
    "apple",
    "arxiv",
    "bbc_news",
    "booking",
    "github",
    "google_flights",
    "google_map",
    "google_search",
    "huggingface",
    "wolfram_alpha",
    "cambridge_dictionary",
    "coursera",
    "espn",
]

WRITE_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"buy|purchase|order|book|reserve|rent|checkout|leave|"
    r"post|submit|create|add|delete|remove|change|edit|update|set|save|"
    r"schedule|register|login|log in|sign in|sign up|send|reply|fork|"
    r"upload|enroll|join|apply|subscribe|unsubscribe|follow|unfollow"
    r")\b"
)

WRITE_PHRASES = (
    " add to ",
    " add it to ",
    " add them to ",
    " save ",
    " saved ",
    " recipe box",
    " shopping list",
    " wishlist",
    " wish list",
    " to cart",
    " my cart",
    " your cart",
    " add to bag",
    " checkout",
    " purchase ",
    " place an order",
    " order confirmation",
    " and book ",
    " make a reservation",
    " reserve a ",
    " reserve an ",
    " schedule an ",
    " schedule a ",
    " appointment",
    " register",
    " sign up",
    " sign in",
    " log in",
    " login",
    " change password",
    " create account",
    " create a ",
    " create an ",
    " open an issue",
    " close an issue",
    " submit ",
    " post ",
    " reply ",
    " send ",
    " upload ",
    " fork ",
    " star this",
    " star the",
    " subscribe",
    " enroll",
    " join ",
    " leave a rating",
    " leave a review",
    " leave a ",
    " generate a ",
    " use the huggingface inference api",
)


def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "task"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def load_reference_answers(path: Path) -> dict[str, dict]:
    """Map upstream WebVoyager ids like Allrecipes--3 to answer metadata."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    refs: dict[str, dict] = {}
    for web_name, group in raw.items():
        notice = group.get("notice", "") if isinstance(group, dict) else ""
        answers = group.get("answers", []) if isinstance(group, dict) else []
        for answer in answers:
            if not isinstance(answer, dict) or "id" not in answer:
                continue
            answer_id = answer["id"]
            upstream_id = f"{web_name}--{answer_id}"
            refs[upstream_id] = {
                "reference_answer": answer.get("ans"),
                "reference_type": answer.get("type"),
                "reference_notice": notice,
            }
    return refs


def rewrite_local_url(url: str) -> str:
    return re.sub(r"^http://(?:localhost|127\.0\.0\.1)(:\d+)", rf"http://{local_host}\1", url)


def is_mutating(instruction: str) -> bool:
    """Conservative WebHarbor write classifier.

    This is deliberately task-text based and independent of deployment state.
    It avoids BrowserGym generic star/set/make false positives on WebVoyager
    read intents such as star ratings or mathematical sets, while still marking
    account/session/form/cart/order/review/API-generation tasks as writers for
    the strict two-pass rollout recipe.
    """
    text = " " + re.sub(r"\s+", " ", instruction.strip().lower()) + " "
    bare = text.strip()
    if WRITE_PREFIX_RE.search(bare):
        return True
    return any(phrase in text for phrase in WRITE_PHRASES)


tasks: dict[str, dict] = {}
refs = load_reference_answers(webvoyager_data_dir / "reference_answer.json")

site_paths = [sites_dir / site / "tasks.jsonl" for site in SITE_ORDER]
site_paths.extend(sorted(p for p in sites_dir.glob("*/tasks.jsonl") if p not in site_paths))

for path in site_paths:
    if not path.is_file():
        raise SystemExit(f"missing WebHarbor task file in image: {path}")
    site_slug = path.parent.name
    for row in read_jsonl(path):
        upstream_id = str(row["id"])
        web_name = str(row.get("web_name") or upstream_id.split("--", 1)[0])
        suffix = upstream_id.split("--", 1)[1] if "--" in upstream_id else upstream_id
        mutating = is_mutating(str(row["ques"]))
        # Env id already carries the WebVoyager namespace:
        # webharbor.webvoyager@allrecipes.0.
        task_id = f"{site_slug}.{slug(suffix)}"
        start_url = rewrite_local_url(str(row["web"]))
        upstream_url = row.get("upstream_url")
        others = {
            "source": "webharbor",
            "upstream_id": upstream_id,
            "web_name": web_name,
            "site_slug": site_slug,
            "sites": [site_slug],
            "start_url": start_url,
            "mutating": mutating,
        }
        if upstream_url:
            others["upstream_url"] = upstream_url
        item = {
            "source": "webharbor",
            "split": "eval",
            "upstream_id": upstream_id,
            "web_name": web_name,
            "site_slug": site_slug,
            "start_url": start_url,
            "upstream_url": upstream_url,
            "instruction": row["ques"],
            "max_steps": 15,
            "others": others,
        }
        item.update({k: v for k, v in refs.get(upstream_id, {}).items() if v is not None})
        tasks[task_id] = item

if not tasks:
    raise SystemExit("no tasks found in the WebHarbor sites dir")

counts = {
    "total": len(tasks),
    "read": sum(1 for t in tasks.values() if not t["others"]["mutating"]),
    "write": sum(1 for t in tasks.values() if t["others"]["mutating"]),
}
count_total = counts["total"]
count_read = counts["read"]
count_write = counts["write"]

with out_path.open("w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2, sort_keys=True, ensure_ascii=False)
    f.write("\n")

print(
    f"wrote {count_total} WebHarbor-backed tasks to {out_path} "
    f"(read={count_read}, write={count_write})",
    file=sys.stderr,
)
'

CNAME="webharbor.webvoyager-tasks-dump-$$"
cleanup() {
    docker rm -fv "$CNAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run \
    --name "$CNAME" \
    --entrypoint python \
    "$IMAGE" \
    -c "$DUMP_PY"

mkdir -p "$(dirname "$OUT")"
docker cp "$CNAME:/tmp/tasks.json" "$OUT"
echo "wrote $OUT"
