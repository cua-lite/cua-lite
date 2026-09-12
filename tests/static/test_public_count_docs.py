"""Static guards for public count prose."""

from __future__ import annotations

import json
import re
from pathlib import Path

from lite.gym.envs.mobilegym.main import _MOBILEGYM_APPS

REPO = Path(__file__).resolve().parents[2]
MOBILEGYM_README = REPO / "lite" / "gym" / "envs" / "mobilegym" / "README.md"
MOBILEGYM_DEFAULT_CONFIGS = tuple(
    sorted((REPO / "scripts" / "configs").glob("*/default/mobilegym.yaml"))
)
LITE_OSWORLD_LOCK = (
    REPO / "lite" / "gym" / "envs" / "lite" / "osworld" / "data" / "catalog.lock.json"
)
LITE_OSWORLD_PUBLIC_DOCS = {
    REPO / "README.md": "lite.osworld` eval split ({scored} valid tasks)",
    REPO / "docs" / "eval.md": "**`lite.osworld` {scored} scored**",
    REPO / "docs" / "grpo.md": "{scored} non-excluded eval tasks",
    REPO / "lite" / "gym" / "envs" / "osworld" / "README.md": (
        "{excluded} excluded tasks"
    ),
    REPO / "lite" / "gym" / "envs" / "lite" / "osworld" / "README.md": (
        "drops the {excluded} excluded tasks"
    ),
    REPO / "devs" / "exps" / "train" / "desktop" / "README.md": (
        "**{scored} of the {rows}**"
    ),
}


def _squash(text: str) -> str:
    return " ".join(text.split())


def _lite_osworld_eval_count_lock() -> dict[str, object]:
    entry = json.loads(LITE_OSWORLD_LOCK.read_text(encoding="utf-8"))["splits"]["eval"]
    assert entry["rows"] == entry["excluded_rows"] + entry["scored_rows"]
    assert sum(entry["exclude_reasons"].values()) == entry["excluded_rows"]
    return entry


def test_mobilegym_public_app_count_matches_owner_surface() -> None:
    expected = len(_MOBILEGYM_APPS)
    assert expected == 24
    assert MOBILEGYM_DEFAULT_CONFIGS

    stale: list[str] = []
    missing_current_count: list[str] = []
    for path in (MOBILEGYM_README, *MOBILEGYM_DEFAULT_CONFIGS):
        text = _squash(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO)
        if "28 simulated apps" in text:
            stale.append(str(rel))
        if not re.search(rf"\b416\b.*\bacross {expected} launchable apps\b", text):
            missing_current_count.append(str(rel))

    assert not stale, "stale MobileGym 28-app prose:\n  " + "\n  ".join(stale)
    assert not missing_current_count, (
        "MobileGym public docs/configs must cite the open_app catalog count "
        f"({expected}) rather than deriving an app count from task metadata:\n  "
        + "\n  ".join(missing_current_count)
    )


def test_mobilegym_readme_catalog_matches_open_app_surface() -> None:
    text = MOBILEGYM_README.read_text(encoding="utf-8")
    catalog = text.split("## Launchable Apps Catalog", 1)[1].split("## Citation", 1)[0]
    catalog_table = catalog.split("Internal system surfaces", 1)[0]

    missing = [name for name in _MOBILEGYM_APPS if name not in catalog_table]
    assert not missing, "MobileGym README launchable catalog missed:\n  " + "\n  ".join(missing)

    assert "Launcher" not in catalog_table
    assert "Contacts" not in catalog_table
    assert "AnswerSheet" not in catalog_table
    assert "ThemeStore" not in catalog_table


def test_lite_osworld_public_scored_count_matches_catalog_lock() -> None:
    counts = _lite_osworld_eval_count_lock()
    rows = counts["rows"]
    excluded = counts["excluded_rows"]
    scored = counts["scored_rows"]

    assert rows == 369
    assert excluded == 41
    assert scored == 328

    stale_patterns = (
        r"lite\.osworld` 332 scored",
        r"330 non-excluded eval tasks",
        r"332 valid tasks",
        r"37 excluded tasks",
        r"39 excluded tasks",
        r"330 scored",
    )
    for path, snippet_template in LITE_OSWORLD_PUBLIC_DOCS.items():
        text = _squash(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO)
        assert snippet_template.format(
            rows=rows, excluded=excluded, scored=scored,
        ) in text, f"{rel} missing Lite.OSWorld count from catalog lock"
        stale = [pattern for pattern in stale_patterns if re.search(pattern, text)]
        assert not stale, f"{rel} has stale Lite.OSWorld count prose: {stale}"
