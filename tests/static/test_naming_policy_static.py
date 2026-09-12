"""Static guards for generic protocol/message naming boundaries.

Run:
    uv run pytest tests/static/test_naming_policy_static.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SKIPPED_PARTS = {
    "__pycache__",
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "_vendor",
    "build",
    "node_modules",
    "slime",
}

TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}

GENERIC_PROTOCOL_ROOTS = (
    REPO / "lite" / "agents" / "core" / "protocol",
)

CONCRETE_SURFACE_TERMS = (
    "screenshot",
    "gui",
    "cli",
    "text_tool",
    "image_tool",
)

BROAD_TEXT_POLICY_NAMES = (
    "keep_regular_text",
    "preserve_regular_text",
    "keep_observation_text",
    "preserve_observation_text",
    "keep_screenshot_text",
    "keep_gui_text",
    "keep_text_tool_output",
)

# Retired spellings for a canonical Lite action-batch call (``computer`` /
# ``mobile``). The canonical wording is "action-batch tool/call" for the
# envelope, "child action" for entries under ``arguments.actions[]``, "step
# action list" for ``env.step([...])``, and "provider wire format" only for an
# actual provider payload.
RETIRED_ACTION_BATCH_TERMS = (
    "gui wrapper",
    "screen-surface wrapper",
    "wrapper-batch",
    "batched computer",
    "wire shape",
)

# Every root that carries prose about Lite calls. The retired terms are retired
# in all of them, including the two layers where the underlying concepts are
# real: ``lite/agents/models`` says "provider-native wrapper" for
# ``computer_use`` / ``mobile_use``, and ``lite/gym`` says "env transport" for
# env-server payloads and "provider wire format" only for an actual provider
# payload. Excluding those two would make this gate unable to reach zero, which
# is not a gate. This file is skipped by the test itself, since it must spell
# the retired terms to detect them.
CANONICAL_VOCABULARY_ROOTS = (
    REPO / "devs",
    REPO / "examples",
    REPO / "lite",
    REPO / "scripts",
    REPO / "tests",
)

KEEP_TEXT_WITH_IMAGES_ALLOWLIST = {
    "lite/agents/extensions/webharbor/webvoyager/protocol.py",
    "lite/agents/models/qwen3_5/protocol.py",
    "lite/agents/models/qwen3_vl/protocol.py",
    "tests/agents/core/protocol/test_observation_text.py",
    "tests/agents/core/protocol/test_role_tool_grouping.py",
    "tests/agents/extensions/webharbor/webvoyager/test_som_splice.py",
    "tests/agents/models/qwen3_vl/test_qwen3_vl_observation_text.py",
    "tests/agents/models/qwen3_vl/test_qwen3_vl_role_tool_grouping.py",
    "tests/scripts/test_rollout_configs.py",
}


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _is_source_path(path: Path) -> bool:
    if path.name == "rule.md":
        return False
    return not (set(path.relative_to(REPO).parts) & SKIPPED_PARTS)


def _iter_text_files(roots: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.suffix in TEXT_SUFFIXES and _is_source_path(root):
                files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES and _is_source_path(path):
                files.append(path)
    # Discovery canary — see ``test_zero_variance_subclass._iter_source_files``:
    # an emptied walk would make every gate below pass vacuously.
    assert files, f"discovery found no text files under {[str(r) for r in roots]}"
    return sorted(files)


def _iter_python_files(roots: tuple[Path, ...]) -> list[Path]:
    return [
        path
        for path in _iter_text_files(roots)
        if path.suffix == ".py" and path.name != Path(__file__).name
    ]


def _public_identifier_locations(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    locations: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            locations.append((node.name, node.lineno))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                locations.extend((arg.arg, arg.lineno) for arg in args)
                if node.args.vararg is not None:
                    locations.append((node.args.vararg.arg, node.args.vararg.lineno))
                if node.args.kwarg is not None:
                    locations.append((node.args.kwarg.arg, node.args.kwarg.lineno))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            locations.append((node.target.id, node.lineno))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    locations.append((target.id, node.lineno))

    return [
        (name, lineno)
        for name, lineno in locations
        if not name.startswith("_")
    ]


def test_generic_protocol_public_names_avoid_concrete_backend_terms() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(GENERIC_PROTOCOL_ROOTS):
        for name, lineno in _public_identifier_locations(path):
            lowered = name.lower()
            if any(term in lowered for term in CONCRETE_SURFACE_TERMS):
                offenders.append(f"{_rel(path)}:{lineno}: {name}")

    assert not offenders, (
        "generic protocol/message APIs should use image/text/message wording; "
        "concrete GUI/screenshot/backend terms belong in env/logger/annotation/"
        "action-space/model-specific layers:\n  "
        + "\n  ".join(offenders)
    )


def test_broad_text_retention_flags_do_not_spread() -> None:
    roots = (
        REPO / "README.md",
        REPO / "docs",
        REPO / "devs",
        REPO / "examples",
        REPO / "lite",
        REPO / "tests",
    )
    offenders: list[str] = []

    for path in _iter_text_files(roots):
        rel = _rel(path)
        if rel == _rel(Path(__file__)):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in BROAD_TEXT_POLICY_NAMES:
            if name in text:
                offenders.append(f"{rel}: contains {name}")
        if (
            "keep_text_with_images" in text
            and rel not in KEEP_TEXT_WITH_IMAGES_ALLOWLIST
        ):
            offenders.append(f"{rel}: contains keep_text_with_images outside allowlist")

    assert not offenders, (
        "broad history text booleans must stay quarantined; prefer precise "
        "image/text/message policy names for new surfaces:\n  "
        + "\n  ".join(offenders)
    )


def _retired_action_batch_offenders(rel: str, text: str) -> list[str]:
    lowered_lines = text.lower().splitlines()
    return [
        f"{rel}:{index + 1}: {term!r}"
        for index, line in enumerate(lowered_lines)
        for term in RETIRED_ACTION_BATCH_TERMS
        if term in line
    ]


def test_retired_action_batch_vocabulary_stays_out_of_canonical_layers() -> None:
    offenders: list[str] = []

    for path in _iter_text_files(CANONICAL_VOCABULARY_ROOTS):
        rel = _rel(path)
        if rel == _rel(Path(__file__)):
            continue
        offenders.extend(
            _retired_action_batch_offenders(
                rel,
                path.read_text(encoding="utf-8", errors="ignore"),
            )
        )

    assert not offenders, (
        "canonical Lite layers must say action-batch tool/call, child action, "
        "step action list, or bare model-function projection; provider-native "
        "wrapper and provider wire wording belongs to lite/agents/models and "
        "lite/gym:\n  "
        + "\n  ".join(offenders)
    )


def test_retired_action_batch_detector_catches_every_term() -> None:
    planted = "\n".join(
        f"# a {term} sentence" for term in RETIRED_ACTION_BATCH_TERMS
    )

    offenders = _retired_action_batch_offenders("lite/core/tools/_planted.py", planted)

    assert offenders == [
        f"lite/core/tools/_planted.py:{index + 1}: {term!r}"
        for index, term in enumerate(RETIRED_ACTION_BATCH_TERMS)
    ]


def test_retired_action_batch_detector_allows_provider_native_wrapper_wording() -> None:
    offenders = _retired_action_batch_offenders(
        "lite/agents/models/qwen3_vl/_planted.py",
        "# The provider-native wrapper tool ``computer_use`` carries one action.",
    )

    assert offenders == []
