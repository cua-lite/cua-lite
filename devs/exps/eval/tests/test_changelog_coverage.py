"""Static guards for eval CHANGELOG coverage and comparability notes."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EVAL_ROOT = REPO / "devs" / "exps" / "eval"


def _squash(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _eval_env_dirs() -> list[Path]:
    return sorted(path for path in EVAL_ROOT.iterdir() if (path / "run.sh").is_file())


def test_all_eval_envs_have_nonempty_changelog() -> None:
    missing: list[str] = []
    empty: list[str] = []
    for env_dir in _eval_env_dirs():
        changelog = env_dir / "CHANGELOG"
        rel = changelog.relative_to(REPO)
        if not changelog.is_file():
            missing.append(str(rel))
        elif not changelog.read_text(encoding="utf-8").strip():
            empty.append(str(rel))

    assert not missing, "eval envs missing CHANGELOG:\n  " + "\n  ".join(missing)
    assert not empty, "eval envs with empty CHANGELOG:\n  " + "\n  ".join(empty)


def test_known_semantic_eval_changes_are_recorded() -> None:
    mobilegym = _squash(EVAL_ROOT / "mobilegym" / "CHANGELOG")
    assert "official Success Rate" in mobilegym
    assert "progress_rate" in mobilegym
    assert "NOT comparable" in mobilegym

    for env in ("screenspot_pro", "osworld_g"):
        text = _squash(EVAL_ROOT / env / "CHANGELOG")
        assert "first well-formed" in text
        assert "original first-valid grounding metric" in text
        assert "exactly-one anti-hedging metric" in text


def test_pending_browser_coord_policy_is_not_marked_landed_yet() -> None:
    text = _squash(EVAL_ROOT / "webharbor.webvoyager" / "CHANGELOG")

    assert "pending Issue 1b" in text
    assert "is not recorded here as landed yet" in text
