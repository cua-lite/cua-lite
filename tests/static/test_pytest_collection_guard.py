"""Static guard for pytest collection ownership."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

EXPLICIT_TEST_LANE_ROOTS = (
    Path("devs"),
)

TEMPORARY_HIDDEN_TEST_ALLOWLIST: set[Path] = set()

ZERO_COLLECTION_EDGE_CASES = {
    Path("tests/train/rollout/core/test_engine_integration.py"): "module-level "
    "pytest.importorskip for Slime-only rollout integration coverage",
    Path("tests/train/rollout/core/test_env_failure.py"): "module-level "
    "pytest.importorskip for Slime-only rollout failure coverage",
    Path("tests/train/rollout/core/test_eval_rollout_parse_failure.py"): "module-level "
    "pytest.importorskip for Slime-only eval keep/retry coverage",
    Path("tests/train/rollout/core/test_lazy_dp_split.py"): "module-level "
    "pytest.importorskip for Slime-only lazy-DP coverage",
    Path("tests/train/rollout/core/test_lazy_multimodal_format.py"): "module-level "
    "pytest.importorskip for Slime-only lazy multimodal coverage",
    Path("tests/train/rollout/core/test_length1_equivalence.py"): "module-level "
    "pytest.importorskip for Slime-only rollout equivalence coverage",
    Path("tests/train/rollout/core/test_packing_arithmetic.py"): "module-level "
    "pytest.importorskip for Slime-only packing coverage",
    Path("tests/train/rollout/core/test_radix_segmenter.py"): "module-level "
    "pytest.importorskip for Slime-only radix coverage",
    Path("tests/train/rollout/core/test_rollout_engine.py"): "module-level "
    "pytest.importorskip for Slime-only rollout engine coverage",
    Path("tests/train/rollout/core/test_static_segment_padding.py"): "module-level "
    "pytest.importorskip for Slime-only segment-padding coverage",
    Path("tests/train/rollout/sft/test_radix_packing.py"): "module-level "
    "pytest.importorskip for Slime-only SFT radix coverage",
    Path("tests/train/rollout/test_grpo_e2e_math.py"): "module-level "
    "pytest.importorskip for Slime-only GRPO coverage",
    Path("tests/train/rollout/test_grpo_lazy_e2e.py"): "module-level "
    "pytest.importorskip for ray/training-container lazy multimodal coverage",
    Path("tests/train/rollout/test_reinforce.py"): "module-level "
    "pytest.importorskip for Slime-only REINFORCE coverage",
    Path("tests/train/rollout/test_rollout.py"): "module-level "
    "pytest.importorskip for Slime-only rollout coverage",
    Path("tests/train/rollout/test_smoke.py"): "module-level pytest.importorskip "
    "for Slime-Docker-only smoke coverage",
    Path("tests/train/utils/test_dp_schedule_bshd_divisibility.py"): "module-level "
    "pytest.importorskip for Slime-only DP schedule coverage",
    Path("tests/train/utils/test_dual_clip.py"): "module-level pytest.importorskip "
    "for Slime-only dual-clip coverage",
    Path("tests/train/export/test_qwen3_5_mtp_alias.py"): "module-level pytest.importorskip "
    "for training-image megatron-bridge coverage",
}

ZERO_COLLECTION_EDGE_MARKERS = ("pytest.importorskip(", "allow_module_level=True")
ZERO_COLLECTION_FORBIDDEN_ALLOWLIST_TEXT = (
    "helper-only",
    "not a test suite itself",
    "split pointer",
)
PYTEST_FILE_COUNT_RE = re.compile(r"^(?P<path>.+\.py): (?P<count>\d+)$")


def _working_tree_test_files() -> set[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*test_*.py",
            "test_*.py",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    paths: set[Path] = set()
    for line in result.stdout.splitlines():
        if not line:
            continue
        path = Path(line)
        if (REPO / path).is_file():
            paths.add(path)
    return paths


def _pytest_testpaths() -> tuple[Path, ...]:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    values = pyproject["tool"]["pytest"]["ini_options"]["testpaths"]
    return tuple(Path(value) for value in values)


def _default_collected_test_files() -> set[Path]:
    default_roots = _pytest_testpaths()
    return {
        path
        for path in _working_tree_test_files()
        if any(_is_under(path, root) for root in default_roots)
    }


def _pytest_import_name(path: Path) -> str:
    package_dir = path.parent
    if not (REPO / package_dir / "__init__.py").is_file():
        return path.stem

    while (REPO / package_dir.parent / "__init__.py").is_file():
        package_dir = package_dir.parent
    return ".".join(path.with_suffix("").relative_to(package_dir.parent).parts)


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_under_dev_tests_dir(path: Path) -> bool:
    return (
        len(path.parts) > 3
        and path.parts[0] == "devs"
        and "tests" in path.parts[2:-1]
    )


def _is_explicitly_owned(path: Path) -> bool:
    return (
        any(_is_under(path, root) for root in EXPLICIT_TEST_LANE_ROOTS)
        or path in TEMPORARY_HIDDEN_TEST_ALLOWLIST
    )


def _invalid_dev_lane_test_paths(paths: set[Path]) -> list[Path]:
    return sorted(
        path
        for path in paths
        if path.parts
        and path.parts[0] == "devs"
        and not _is_under_dev_tests_dir(path)
    )


def _invalid_top_level_owner_test_paths(paths: set[Path]) -> list[Path]:
    return sorted(
        path
        for path in paths
        if (
            len(path.parts) >= 2
            and path.parts[0] == "tests"
            and path.parts[1] in {"devs", "examples"}
        )
    )


def _hidden_test_paths(paths: set[Path]) -> list[Path]:
    return sorted(
        path
        for path in paths
        if any(part.startswith(".") for part in path.parts)
    )


def _duplicate_test_basenames(paths: set[Path]) -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(paths):
        by_name[path.name].append(path)
    return {
        name: grouped
        for name, grouped in sorted(by_name.items())
        if len(grouped) > 1
    }


def _pytest_collected_test_counts(
    paths: tuple[Path, ...],
    *,
    cwd: Path = REPO,
    marker_expression: str | None = "",
) -> dict[Path, int]:
    if not paths:
        return {}

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-qq",
        "-p",
        "no:cacheprovider",
        # Collection is what is being measured, so it must be serial: xdist
        # distributes collection across workers and reports it in a shape this
        # parser cannot read, which would score every file as zero. `-n 0`
        # rather than `-p no:xdist` -- unloading the plugin would make the
        # `addopts` default an unrecognized argument.
        "-n",
        "0",
    ]
    if marker_expression is not None:
        command.extend(["-m", marker_expression])
    command.extend(str(path) for path in paths)

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("CUA_LITE_ENV_SERVER_URL", None)
    env.pop("CUA_LITE_ENV_SERVER_TOKEN", None)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in {0, 5}:
        raise AssertionError(
            "pytest collection failed while checking test-file item counts:\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    counts = {path: 0 for path in paths}
    for line in result.stdout.splitlines():
        if "::" in line:
            node_path = Path(line.split("::", 1)[0])
            count = 1
        elif match := PYTEST_FILE_COUNT_RE.match(line):
            node_path = Path(match.group("path"))
            count = int(match.group("count"))
        else:
            continue
        if node_path in counts:
            counts[node_path] += count
            continue
        try:
            relative_node_path = node_path.relative_to(cwd)
        except ValueError:
            continue
        if relative_node_path in counts:
            counts[relative_node_path] += count

    return counts


@cache
def _worktree_marker_neutral_collected_counts() -> dict[Path, int]:
    return _pytest_collected_test_counts(tuple(sorted(_working_tree_test_files())))


def _zero_collection_test_files(
    paths: set[Path],
    counts: dict[Path, int],
) -> list[Path]:
    return sorted(
        path
        for path in paths
        if counts.get(path, 0) == 0 and path not in ZERO_COLLECTION_EDGE_CASES
    )


def test_tracked_first_party_tests_are_collected_or_explicitly_owned() -> None:
    default_roots = _pytest_testpaths()

    hidden = sorted(
        path
        for path in _working_tree_test_files()
        if not any(_is_under(path, root) for root in default_roots)
        and not _is_explicitly_owned(path)
    )

    assert not hidden, "tracked tests outside default collection or allowlist:\n" + "\n".join(
        str(path) for path in hidden
    )


def test_explicit_dev_lane_tests_live_under_owner_tests_directories() -> None:
    invalid = _invalid_dev_lane_test_paths(_working_tree_test_files())

    assert not invalid, "dev-lane tests must live under devs/**/tests/:\n" + "\n".join(
        str(path) for path in invalid
    )


def test_tests_tree_does_not_host_dev_or_example_owner_tests() -> None:
    invalid = _invalid_top_level_owner_test_paths(_working_tree_test_files())

    assert not invalid, "dev/example owner tests must live with their owner:\n" + "\n".join(
        str(path) for path in invalid
    )


def test_tracked_first_party_tests_do_not_live_under_hidden_dirs() -> None:
    hidden = _hidden_test_paths(_working_tree_test_files())

    assert not hidden, (
        "tracked test files in hidden directories are not default-collected:\n"
        + "\n".join(str(path) for path in hidden)
    )


def test_first_party_test_basenames_are_unique() -> None:
    duplicates = _duplicate_test_basenames(_working_tree_test_files())

    assert not duplicates, "first-party test files share basenames:\n" + "\n".join(
        f"{name}: {', '.join(str(path) for path in paths)}"
        for name, paths in duplicates.items()
    )


def test_tracked_first_party_tests_collect_marker_neutral_items() -> None:
    paths = _working_tree_test_files()
    zero = _zero_collection_test_files(paths, _worktree_marker_neutral_collected_counts())

    assert not zero, (
        "tracked test_*.py files collected zero marker-neutral pytest items; "
        "rename helpers/split pointers out of test_*.py or use a real "
        "environment-skip edge case:\n"
        + "\n".join(str(path) for path in zero)
    )


def test_zero_collection_edge_allowlist_is_narrow() -> None:
    tracked = _working_tree_test_files()

    missing = sorted(path for path in ZERO_COLLECTION_EDGE_CASES if path not in tracked)
    assert not missing, "zero-collection edge cases are no longer tracked:\n" + "\n".join(
        str(path) for path in missing
    )

    invalid: list[str] = []
    for path, reason in sorted(ZERO_COLLECTION_EDGE_CASES.items()):
        text = (REPO / path).read_text(encoding="utf-8")
        normalized_text = text.replace(" ", "")
        lower_text = text.lower()
        if not reason.strip():
            invalid.append(f"{path}: missing reason")
        if "__test__=False" in normalized_text:
            invalid.append(f"{path}: __test__ = False is not a collection edge case")
        if any(marker in lower_text for marker in ZERO_COLLECTION_FORBIDDEN_ALLOWLIST_TEXT):
            invalid.append(f"{path}: helper/split files may not be allowlisted")
        if not any(marker in text for marker in ZERO_COLLECTION_EDGE_MARKERS):
            invalid.append(f"{path}: no module-level skip/importorskip marker")

    assert not invalid, "invalid zero-collection edge allowlist entries:\n" + "\n".join(invalid)


def test_temporary_hidden_test_allowlist_matches_tracked_files() -> None:
    tracked = _working_tree_test_files()
    default_roots = _pytest_testpaths()

    missing = sorted(path for path in TEMPORARY_HIDDEN_TEST_ALLOWLIST if path not in tracked)
    assert not missing, "allowlisted tests are no longer tracked:\n" + "\n".join(
        str(path) for path in missing
    )

    collected = sorted(
        path
        for path in TEMPORARY_HIDDEN_TEST_ALLOWLIST
        if any(_is_under(path, root) for root in default_roots)
    )
    assert not collected, "allowlisted tests are now default-collected:\n" + "\n".join(
        str(path) for path in collected
    )


def test_explicit_test_lane_roots_are_not_default_collected() -> None:
    default_roots = _pytest_testpaths()

    collected = sorted(
        root
        for root in EXPLICIT_TEST_LANE_ROOTS
        if any(_is_under(root, default_root) for default_root in default_roots)
    )

    assert not collected, "explicit test lane roots are now default-collected:\n" + "\n".join(
        str(path) for path in collected
    )


def test_default_collected_test_import_names_are_unique() -> None:
    by_import_name: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(_default_collected_test_files()):
        by_import_name[_pytest_import_name(path)].append(path)

    collisions = {
        name: paths
        for name, paths in sorted(by_import_name.items())
        if len(paths) > 1
    }

    assert not collisions, "default-collected tests share pytest import names:\n" + "\n".join(
        f"{name}: {', '.join(str(path) for path in paths)}"
        for name, paths in collisions.items()
    )


def test_dev_lane_topology_rejects_flat_dev_tests() -> None:
    paths = {
        Path("devs/test_root.py"),
        Path("devs/data/test_flat.py"),
        Path("devs/tests/test_root_owner.py"),
        Path("devs/data/tests/test_owner.py"),
        Path("devs/data/tests/unit/test_nested_owner.py"),
    }

    assert _invalid_dev_lane_test_paths(paths) == [
        Path("devs/data/test_flat.py"),
        Path("devs/test_root.py"),
        Path("devs/tests/test_root_owner.py"),
    ]


def test_top_level_owner_guard_rejects_tests_devs_and_tests_examples() -> None:
    paths = {
        Path("tests/devs/test_bad.py"),
        Path("tests/examples/test_bad.py"),
        Path("devs/data/tests/test_owner.py"),
        Path("examples/geo3k/tests/test_owner.py"),
        Path("tests/static/test_owner.py"),
    }

    assert _invalid_top_level_owner_test_paths(paths) == [
        Path("tests/devs/test_bad.py"),
        Path("tests/examples/test_bad.py"),
    ]


def test_hidden_directory_guard_rejects_dot_paths() -> None:
    paths = {
        Path("examples/.hidden/tests/test_bad.py"),
        Path("devs/data/.scratch/tests/test_bad.py"),
        Path("tests/static/test_good.py"),
    }

    assert _hidden_test_paths(paths) == [
        Path("devs/data/.scratch/tests/test_bad.py"),
        Path("examples/.hidden/tests/test_bad.py"),
    ]


def test_duplicate_basename_guard_covers_all_first_party_lanes() -> None:
    paths = {
        Path("tests/static/test_guard.py"),
        Path("examples/geo3k/tests/test_guard.py"),
        Path("devs/data/webgym/tests/test_webgym_filter.py"),
    }

    assert _duplicate_test_basenames(paths) == {
        "test_guard.py": [
            Path("examples/geo3k/tests/test_guard.py"),
            Path("tests/static/test_guard.py"),
        ]
    }


def test_zero_collection_guard_rejects_dunder_test_false_file(tmp_path: Path) -> None:
    test_file = tmp_path / "test_helper_only.py"
    test_file.write_text(
        "from __future__ import annotations\n"
        "__test__ = False\n"
        "def test_hidden() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )

    counts = _pytest_collected_test_counts((Path(test_file.name),), cwd=tmp_path)

    assert counts == {Path("test_helper_only.py"): 0}
    assert _zero_collection_test_files({Path("test_helper_only.py")}, counts) == [
        Path("test_helper_only.py")
    ]


def test_zero_collection_guard_counts_live_only_tests_marker_neutral(tmp_path: Path) -> None:
    test_file = tmp_path / "test_live_only.py"
    test_file.write_text(
        "from __future__ import annotations\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live\n"
        "def test_live_only() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )

    default_counts = _pytest_collected_test_counts(
        (Path(test_file.name),),
        cwd=tmp_path,
        marker_expression="not live and not stress",
    )
    marker_neutral_counts = _pytest_collected_test_counts((Path(test_file.name),), cwd=tmp_path)

    assert default_counts == {Path("test_live_only.py"): 0}
    assert marker_neutral_counts == {Path("test_live_only.py"): 1}
    assert _zero_collection_test_files({Path("test_live_only.py")}, marker_neutral_counts) == []
