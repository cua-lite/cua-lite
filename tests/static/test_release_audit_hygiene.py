"""Static guards for reviewable release commits."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BASE_REF = "origin/dev"
BAD_COMMIT_SUBJECT_RE = re.compile(
    r"^(?:wip(?:[:\s]|$)|tmp(?:\s|$)|save(?:\s|$)|checkpoint(?:\s|$)|smell\.md$)",
    re.IGNORECASE,
)
REVIEW_RULE_SNIPPETS = (
    "Keep semantic default changes in their own commits.",
    "Keep mechanical migrations separate from behavior changes.",
    "When a refactor makes a config key, flag, prompt knob, or compatibility path",
    "Do not commit placeholder subjects such as `wip`, `tmp save`, `save smell.md`,",
)
SEMANTIC_SURFACE_ROOTS = (
    Path("lite/agents"),
    Path("lite/core/tools"),
    Path("lite/gym/envs"),
    Path("lite/gym/remote"),
    Path("lite/gym/sandbox"),
    Path("lite/infer"),
    Path("scripts/configs"),
)
SEMANTIC_SURFACE_FILES = {
    Path("scripts/rollout.py"),
    Path("scripts/serve_env.py"),
    Path("scripts/serve_fleet.py"),
}
SEMANTIC_SURFACE_SUFFIXES = {".py", ".sh", ".yaml", ".yml"}
AUDIT_MARKER_TEXT_RE = re.compile(
    r"(?im)^(?:Audit|Issue|Regression|Owner decision|Comparability):"
)


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=check,
    )


def _ref_exists(ref: str) -> bool:
    return _git("rev-parse", "--verify", "--quiet", ref, check=False).returncode == 0


def _release_range() -> str:
    explicit = os.environ.get("CUA_LITE_RELEASE_AUDIT_RANGE")
    if explicit:
        return explicit
    if _ref_exists(DEFAULT_BASE_REF):
        return f"{DEFAULT_BASE_REF}..HEAD"
    pytest.skip(
        "set CUA_LITE_RELEASE_AUDIT_RANGE or fetch origin/dev to audit commit subjects"
    )


def _commit_subjects(range_spec: str) -> list[str]:
    result = _git("log", "--format=%h%x00%s", range_spec)
    subjects: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        subjects.append(line.replace("\x00", " ", 1))
    return subjects


def _commit_messages(range_spec: str) -> list[str]:
    result = _git("log", "--format=%B%x1e", range_spec)
    return [message.strip() for message in result.stdout.split("\x1e") if message.strip()]


def _changed_paths(range_spec: str) -> list[Path]:
    result = _git("diff", "--name-only", range_spec)
    return [Path(line) for line in result.stdout.splitlines() if line]


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_unreviewable_subject(subject: str) -> bool:
    parts = subject.split(" ", 1)
    text = parts[1] if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{7,40}", parts[0]) else subject
    return BAD_COMMIT_SUBJECT_RE.search(text.strip()) is not None


def _is_audit_marker_path(path: Path) -> bool:
    return path == Path("ERROR.md") or (
        len(path.parts) == 5
        and path.parts[:3] == ("devs", "exps", "eval")
        and path.name == "CHANGELOG"
    )


def _semantic_surface_paths(paths: list[Path]) -> list[Path]:
    return sorted(
        path
        for path in paths
        if path.suffix in SEMANTIC_SURFACE_SUFFIXES
        and not _is_audit_marker_path(path)
        and (
            path in SEMANTIC_SURFACE_FILES
            or any(_is_under(path, root) for root in SEMANTIC_SURFACE_ROOTS)
        )
    )


def _has_audit_marker(paths: list[Path], messages: list[str]) -> bool:
    return any(_is_audit_marker_path(path) for path in paths) or any(
        AUDIT_MARKER_TEXT_RE.search(message) for message in messages
    )


def test_bad_commit_subject_detector_catches_migration_examples() -> None:
    bad_subjects = [
        "tmp save",
        "wip: checkpoint tool-io refactor",
        "save smell.md",
        "checkpoint",
        "smell.md",
    ]
    good_subjects = [
        "fix(env-server): reject abbreviated warm flags",
        "docs(lite.osworld): lock scored eval count",
        "refactor(gym): rename reward shaping field",
    ]

    assert all(_is_unreviewable_subject(subject) for subject in bad_subjects)
    assert not any(_is_unreviewable_subject(subject) for subject in good_subjects)


def test_semantic_audit_marker_detector_catches_unmarked_surface_changes() -> None:
    semantic_paths = [
        Path("lite/agents/models/qwen3_vl/adapter.py"),
        Path("scripts/configs/qwen3_vl/default/webgym.yaml"),
    ]
    audit_paths = [Path("ERROR.md")]
    changelog_paths = [Path("devs/exps/eval/osworld_g/CHANGELOG")]
    harmless_paths = [Path("tests/static/test_release_audit_hygiene.py")]

    assert _semantic_surface_paths(semantic_paths) == semantic_paths
    assert not _semantic_surface_paths(harmless_paths)
    assert not _has_audit_marker(semantic_paths, ["fix(qwen): adjust prompt"])
    assert _has_audit_marker([*semantic_paths, *audit_paths], ["fix(qwen): adjust prompt"])
    assert _has_audit_marker([*semantic_paths, *changelog_paths], ["fix(score): update"])
    assert _has_audit_marker(
        semantic_paths,
        ["fix(qwen): adjust prompt\n\nAudit: prompt surface reviewed."],
    )


def test_release_range_has_no_unreviewable_commit_subjects() -> None:
    offenders = [
        subject
        for subject in _commit_subjects(_release_range())
        if _is_unreviewable_subject(subject)
    ]

    assert not offenders, "unreviewable commit subjects:\n  " + "\n  ".join(offenders)


def test_semantic_prompt_tool_config_changes_have_audit_marker() -> None:
    range_spec = _release_range()
    paths = _changed_paths(range_spec)
    semantic_paths = _semantic_surface_paths(paths)
    if not semantic_paths:
        return

    assert _has_audit_marker(paths, _commit_messages(range_spec)), (
        "semantic prompt/tool/config/env changes need an audit marker in the "
        f"release range {range_spec}; add an ERROR.md note, an eval CHANGELOG "
        "entry, or a commit body line such as 'Audit: ...'. Changed semantic "
        "paths:\n  "
        + "\n  ".join(str(path) for path in semantic_paths[:50])
    )


def test_agents_records_reviewable_commit_rules() -> None:
    text = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    missing = [snippet for snippet in REVIEW_RULE_SNIPPETS if snippet not in text]

    assert not missing, "AGENTS.md lost reviewable-commit rules:\n  " + "\n  ".join(missing)
