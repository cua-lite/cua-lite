"""Static guards for public count prose."""

from __future__ import annotations

import re
from pathlib import Path

from lite.gym.envs.mobilegym.main import _MOBILEGYM_APPS

REPO = Path(__file__).resolve().parents[2]
MOBILEGYM_README = REPO / "lite" / "gym" / "envs" / "mobilegym" / "README.md"
MOBILEGYM_DEFAULT_CONFIGS = tuple(
    sorted((REPO / "scripts" / "configs").glob("*/default/mobilegym.yaml"))
)


def _squash(text: str) -> str:
    return " ".join(text.split())


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
