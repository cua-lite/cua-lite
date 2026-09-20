"""Validate committed eval Markdown/JSON snapshot pairs."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
EVAL_ROOT = REPO_ROOT / "devs" / "exps" / "eval"

HEADER_RE = re.compile(
    r"^#\s+(?P<env>\S+)\s+@\s+(?P<commit_dir>\S+)\s+(?:\u00b7|-)\s+(?P<run_id>\S+)\s*$"
)
FINISHED_RE = re.compile(r"^(?P<valid>\d+)\s*/\s*(?P<tasks>\d+)$")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True, order=True)
class SnapshotViolation:
    path: str
    kind: str
    detail: str

    def id(self) -> str:
        return f"{self.path}|{self.kind}|{self.detail}"


@dataclass(frozen=True)
class SnapshotPathMeta:
    env: str
    commit_dir: str
    run_id: str


@dataclass(frozen=True, order=True)
class ResultKey:
    config_id: str
    model: str


@dataclass(frozen=True)
class MarkdownResult:
    key: ResultKey
    num_valid: int | None
    num_tasks: int | None
    mean_episode_return: float | None
    malformed: bool = False

    @property
    def is_placeholder(self) -> bool:
        return self.num_valid == 0 and self.mean_episode_return is None


@dataclass(frozen=True)
class JsonResult:
    key: ResultKey
    num_valid: int | None
    num_tasks: int | None
    mean_episode_return: float | None


@dataclass(frozen=True)
class MarkdownSnapshot:
    path: Path
    env: str | None
    commit_dir: str | None
    run_id: str | None
    commit_short_sha: str | None
    artifact_root: str | None
    results: dict[ResultKey, MarkdownResult]
    violations: tuple[SnapshotViolation, ...]


@dataclass(frozen=True)
class JsonSnapshot:
    path: Path
    env: str | None
    commit_dir: str | None
    run_id: str | None
    commit_short_sha: str | None
    artifact_root: str | None
    results: dict[ResultKey, JsonResult]
    violations: tuple[SnapshotViolation, ...]


def repo_rel(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def snapshot_path_meta(path: Path, *, eval_root: Path = EVAL_ROOT) -> SnapshotPathMeta:
    rel = path.resolve().relative_to(eval_root.resolve())
    parts = rel.parts
    if len(parts) != 4 or parts[1] != "logs":
        raise ValueError(f"not an eval snapshot path: {path}")
    return SnapshotPathMeta(env=parts[0], commit_dir=parts[2], run_id=path.stem)


def collect_snapshot_stems(eval_root: Path = EVAL_ROOT) -> tuple[set[Path], set[Path]]:
    markdown = {path.with_suffix("") for path in eval_root.glob("*/logs/*/run*.md")}
    payloads = {path.with_suffix("") for path in eval_root.glob("*/logs/*/run*.json")}
    return markdown, payloads


def orphan_snapshot_paths(
    eval_root: Path = EVAL_ROOT, *, repo_root: Path = REPO_ROOT
) -> tuple[list[str], list[str]]:
    markdown, payloads = collect_snapshot_stems(eval_root)
    md_only = [
        repo_rel(stem.with_suffix(".md"), repo_root=repo_root) for stem in markdown - payloads
    ]
    json_only = [
        repo_rel(stem.with_suffix(".json"), repo_root=repo_root) for stem in payloads - markdown
    ]
    return sorted(md_only), sorted(json_only)


def _clean_cell(cell: str) -> str:
    text = cell.strip()
    for prefix in ("\U0001f4cc", "\u26a0\ufe0f", "\u26a0"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = text.replace("**", "").replace("__", "").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"`", "_"}:
        text = text[1:-1].strip()
    return text.strip()


def _parse_finished(cell: str) -> tuple[int, int] | None:
    match = FINISHED_RE.match(_clean_cell(cell))
    if not match:
        return None
    return int(match.group("valid")), int(match.group("tasks"))


def _parse_metric(cell: str) -> float | None:
    text = _clean_cell(cell)
    if text in {"", "-", "--", "---", "\u2014"}:
        return None
    match = NUMBER_RE.search(text)
    if not match:
        return None
    return float(match.group(0))


def _markdown_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return all(cell.strip("-: ") == "" for cell in cells)


def _violation(path: Path, kind: str, detail: str, *, repo_root: Path) -> SnapshotViolation:
    return SnapshotViolation(repo_rel(path, repo_root=repo_root), kind, detail)


def parse_markdown_snapshot(
    path: Path, *, repo_root: Path = REPO_ROOT
) -> MarkdownSnapshot:
    env: str | None = None
    commit_dir: str | None = None
    run_id: str | None = None
    commit_short_sha: str | None = None
    artifact_root: str | None = None
    results: dict[ResultKey, MarkdownResult] = {}
    violations: list[SnapshotViolation] = []

    lines = path.read_text(encoding="utf-8").splitlines()
    in_results = False
    saw_results = False
    header: list[str] | None = None
    column_index: dict[str, int] = {}

    for line_no, line in enumerate(lines, start=1):
        if env is None:
            header_match = HEADER_RE.match(line)
            if header_match:
                env = header_match.group("env")
                commit_dir = header_match.group("commit_dir")
                run_id = header_match.group("run_id")

        if line.startswith("- **Commit**:") and commit_short_sha is None:
            match = re.search(r"`([^`]+)`", line)
            if match:
                commit_short_sha = match.group(1)

        if line.startswith("- **Artifacts**:") and artifact_root is None:
            match = re.search(r"`([^`]+)`", line)
            if match:
                artifact_root = match.group(1).rstrip("/")

        if line.strip() == "## Results":
            in_results = True
            saw_results = True
            header = None
            column_index = {}
            continue

        if not in_results:
            continue

        if line.startswith("## "):
            in_results = False
            continue

        if not line.lstrip().startswith("|"):
            continue

        cells = _markdown_cells(line)
        if _is_separator(cells):
            continue

        if header is None:
            header = [_clean_cell(cell).lower() for cell in cells]
            column_index = {name: index for index, name in enumerate(header)}
            for required in ("model", "finished", "mean episode return"):
                if required not in column_index:
                    violations.append(
                        _violation(
                            path,
                            "md.results.missing_column",
                            f"column={required}",
                            repo_root=repo_root,
                        )
                    )
            continue

        model_index = column_index.get("model")
        if model_index is None or model_index >= len(cells):
            violations.append(
                _violation(
                    path,
                    "md.results.malformed_row",
                    f"line={line_no} expected_columns={len(header)} actual_columns={len(cells)}",
                    repo_root=repo_root,
                )
            )
            continue

        model = _clean_cell(cells[model_index])
        if not model:
            continue

        config_index = column_index.get("config")
        config_id = "default"
        if config_index is not None and config_index < len(cells):
            config_id = _clean_cell(cells[config_index])

        malformed = len(cells) != len(header)
        if malformed:
            violations.append(
                _violation(
                    path,
                    "md.results.malformed_row",
                    f"line={line_no} expected_columns={len(header)} actual_columns={len(cells)}",
                    repo_root=repo_root,
                )
            )

        num_valid: int | None = None
        num_tasks: int | None = None
        finished_index = column_index.get("finished")
        if finished_index is not None and finished_index < len(cells):
            finished = _parse_finished(cells[finished_index])
            if finished is not None:
                num_valid, num_tasks = finished

        mean_episode_return: float | None = None
        metric_index = column_index.get("mean episode return")
        if metric_index is not None and metric_index < len(cells):
            mean_episode_return = _parse_metric(cells[metric_index])

        key = ResultKey(config_id=config_id, model=model)
        if key in results:
            violations.append(
                _violation(
                    path,
                    "md.results.duplicate_row",
                    f"config={key.config_id} model={key.model}",
                    repo_root=repo_root,
                )
            )
        results[key] = MarkdownResult(
            key=key,
            num_valid=num_valid,
            num_tasks=num_tasks,
            mean_episode_return=mean_episode_return,
            malformed=malformed,
        )

    if not saw_results:
        violations.append(
            _violation(path, "md.results.missing", "section=Results", repo_root=repo_root)
        )

    return MarkdownSnapshot(
        path=path,
        env=env,
        commit_dir=commit_dir,
        run_id=run_id,
        commit_short_sha=commit_short_sha,
        artifact_root=artifact_root,
        results=results,
        violations=tuple(violations),
    )


def _json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def parse_json_snapshot(path: Path, *, repo_root: Path = REPO_ROOT) -> JsonSnapshot:
    data = _json_object(path)
    results_raw = data.get("results")
    if not isinstance(results_raw, list):
        results_raw = []

    results: dict[ResultKey, JsonResult] = {}
    violations: list[SnapshotViolation] = []
    for index, row in enumerate(results_raw):
        if not isinstance(row, dict):
            violations.append(
                _violation(
                    path,
                    "json.results.malformed_row",
                    f"index={index}",
                    repo_root=repo_root,
                )
            )
            continue

        model = str(row.get("model") or "")
        config_id = str(row.get("config_id") or "default")
        key = ResultKey(config_id=config_id, model=model)
        if key in results:
            violations.append(
                _violation(
                    path,
                    "json.results.duplicate_row",
                    f"config={key.config_id} model={key.model}",
                    repo_root=repo_root,
                )
            )

        results[key] = JsonResult(
            key=key,
            num_valid=_int_or_none(row.get("num_valid")),
            num_tasks=_int_or_none(row.get("num_tasks")),
            mean_episode_return=_float_or_none(row.get("mean_episode_return")),
        )

        artifact_path = _string_or_none(row.get("artifact_path"))
        summary_path = _string_or_none(row.get("summary_path"))
        artifact_root = _string_or_none(data.get("artifact_root"))
        artifact_root_prefix = artifact_root.rstrip("/") + "/" if artifact_root else None
        if (
            artifact_path
            and artifact_root_prefix
            and not artifact_path.startswith(artifact_root_prefix)
        ):
            violations.append(
                _violation(
                    path,
                    "json.result.artifact_path",
                    (
                        f"model={model} artifact_path={artifact_path} "
                        f"artifact_root={artifact_root.rstrip('/')}"
                    ),
                    repo_root=repo_root,
                )
            )
        expected_summary_path = (
            artifact_path.rstrip("/") + "/summary.json" if artifact_path else None
        )
        if summary_path and expected_summary_path and summary_path != expected_summary_path:
            violations.append(
                _violation(
                    path,
                    "json.result.summary_path",
                    f"model={model} summary_path={summary_path} artifact_path={artifact_path}",
                    repo_root=repo_root,
                )
            )

    if data.get("result_count") != len(results_raw):
        violations.append(
            _violation(
                path,
                "json.result_count",
                f"expected={len(results_raw)} actual={data.get('result_count')}",
                repo_root=repo_root,
            )
        )

    status_counts = data.get("status_counts")
    if isinstance(status_counts, dict):
        expected = Counter(
            str(row.get("status") or "") for row in results_raw if isinstance(row, dict)
        )
        actual = {str(key): int(value) for key, value in status_counts.items()}
        actual_nonzero = {key: value for key, value in actual.items() if value}
        if actual_nonzero != dict(expected) or sum(actual.values()) != len(results_raw):
            violations.append(
                _violation(
                    path,
                    "json.status_counts",
                    f"expected={dict(expected)} actual={actual_nonzero}",
                    repo_root=repo_root,
                )
            )

    commit = data.get("commit")
    commit_short_sha = None
    if isinstance(commit, dict):
        commit_short_sha = _string_or_none(commit.get("short_sha"))

    return JsonSnapshot(
        path=path,
        env=_string_or_none(data.get("env")),
        commit_dir=_string_or_none(data.get("commit_dir")),
        run_id=_string_or_none(data.get("run_id")),
        commit_short_sha=commit_short_sha,
        artifact_root=_string_or_none(data.get("artifact_root")),
        results=results,
        violations=tuple(violations),
    )


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def validate_same_stem_snapshot_pair(
    markdown_path: Path,
    json_path: Path,
    *,
    eval_root: Path = EVAL_ROOT,
    repo_root: Path = REPO_ROOT,
) -> list[SnapshotViolation]:
    meta = snapshot_path_meta(markdown_path, eval_root=eval_root)
    markdown = parse_markdown_snapshot(markdown_path, repo_root=repo_root)
    payload = parse_json_snapshot(json_path, repo_root=repo_root)

    violations: list[SnapshotViolation] = []
    violations.extend(markdown.violations)
    violations.extend(payload.violations)
    violations.extend(
        _validate_snapshot_identity(markdown_path, "md", markdown, meta, repo_root=repo_root)
    )
    violations.extend(
        _validate_snapshot_identity(json_path, "json", payload, meta, repo_root=repo_root)
    )
    violations.extend(
        _compare_rows(markdown_path, markdown.results, payload.results, repo_root=repo_root)
    )
    return sorted(violations)


def validate_snapshot_pairs(
    eval_root: Path = EVAL_ROOT, *, repo_root: Path = REPO_ROOT
) -> list[SnapshotViolation]:
    markdown, payloads = collect_snapshot_stems(eval_root)
    violations: list[SnapshotViolation] = []
    for stem in sorted(markdown & payloads):
        violations.extend(
            validate_same_stem_snapshot_pair(
                stem.with_suffix(".md"),
                stem.with_suffix(".json"),
                eval_root=eval_root,
                repo_root=repo_root,
            )
        )
    return sorted(violations)


def validate_snapshot_files(
    eval_root: Path = EVAL_ROOT, *, repo_root: Path = REPO_ROOT
) -> list[SnapshotViolation]:
    violations: list[SnapshotViolation] = []
    for path in sorted(eval_root.glob("*/logs/*/run*.md")):
        violations.extend(parse_markdown_snapshot(path, repo_root=repo_root).violations)
    for path in sorted(eval_root.glob("*/logs/*/run*.json")):
        meta = snapshot_path_meta(path, eval_root=eval_root)
        snapshot = parse_json_snapshot(path, repo_root=repo_root)
        violations.extend(snapshot.violations)
        violations.extend(
            _validate_snapshot_identity(path, "json", snapshot, meta, repo_root=repo_root)
        )
    return sorted(violations)


def _validate_snapshot_identity(
    path: Path,
    source_name: str,
    source: MarkdownSnapshot | JsonSnapshot,
    meta: SnapshotPathMeta,
    *,
    repo_root: Path,
) -> list[SnapshotViolation]:
    expected_artifact_root = f".exps/eval/{meta.env}/{meta.commit_dir}/{meta.run_id}"
    expected_short_sha = meta.commit_dir.split("_")[-1]
    violations: list[SnapshotViolation] = []

    for field, expected, actual in (
        ("env", meta.env, source.env),
        ("commit_dir", meta.commit_dir, source.commit_dir),
        ("run_id", meta.run_id, source.run_id),
    ):
        if actual != expected:
            violations.append(
                _violation(
                    path,
                    f"{source_name}.{field}",
                    f"expected={expected} actual={actual}",
                    repo_root=repo_root,
                )
            )

    if not _same_commit_prefix(expected_short_sha, source.commit_short_sha):
        violations.append(
            _violation(
                path,
                f"{source_name}.commit_short_sha",
                f"expected={expected_short_sha} actual={source.commit_short_sha}",
                repo_root=repo_root,
            )
        )

    if source.artifact_root != expected_artifact_root:
        violations.append(
            _violation(
                path,
                f"{source_name}.artifact_root",
                f"expected={expected_artifact_root} actual={source.artifact_root}",
                repo_root=repo_root,
            )
        )

    return violations


def _compare_rows(
    markdown_path: Path,
    markdown_rows: dict[ResultKey, MarkdownResult],
    json_rows: dict[ResultKey, JsonResult],
    *,
    repo_root: Path,
) -> list[SnapshotViolation]:
    violations: list[SnapshotViolation] = []
    markdown_models = _keys_by_model(markdown_rows)
    json_models = _keys_by_model(json_rows)

    config_mismatch_models = sorted(
        model
        for model in markdown_models.keys() & json_models.keys()
        if markdown_models[model] != json_models[model]
    )
    for model in config_mismatch_models:
        violations.append(
            _violation(
                markdown_path,
                "row.config_id",
                (
                    f"model={model} md_configs={','.join(markdown_models[model])} "
                    f"json_configs={','.join(json_models[model])}"
                ),
                repo_root=repo_root,
            )
        )

    mismatched_models = set(config_mismatch_models)
    for key in sorted(json_rows.keys() - markdown_rows.keys()):
        if key.model in mismatched_models:
            continue
        violations.append(
            _violation(
                markdown_path,
                "row.json_only",
                f"config={key.config_id} model={key.model}",
                repo_root=repo_root,
            )
        )

    for key in sorted(markdown_rows.keys() - json_rows.keys()):
        if key.model in mismatched_models or markdown_rows[key].is_placeholder:
            continue
        violations.append(
            _violation(
                markdown_path,
                "row.md_only",
                f"config={key.config_id} model={key.model}",
                repo_root=repo_root,
            )
        )

    for key in sorted(markdown_rows.keys() & json_rows.keys()):
        md_row = markdown_rows[key]
        json_row = json_rows[key]
        if md_row.malformed:
            continue

        if (
            md_row.num_valid is not None
            and md_row.num_tasks is not None
            and (md_row.num_valid, md_row.num_tasks) != (json_row.num_valid, json_row.num_tasks)
        ):
            violations.append(
                _violation(
                    markdown_path,
                    "row.finished",
                    (
                        f"config={key.config_id} model={key.model} "
                        f"md={md_row.num_valid}/{md_row.num_tasks} "
                        f"json={json_row.num_valid}/{json_row.num_tasks}"
                    ),
                    repo_root=repo_root,
                )
            )

        if (
            md_row.mean_episode_return is not None
            and json_row.mean_episode_return is not None
            and abs(md_row.mean_episode_return - json_row.mean_episode_return) > 0.00005
        ):
            violations.append(
                _violation(
                    markdown_path,
                    "row.mean_episode_return",
                    (
                        f"config={key.config_id} model={key.model} "
                        f"md={md_row.mean_episode_return:.4f} "
                        f"json={json_row.mean_episode_return:.4f}"
                    ),
                    repo_root=repo_root,
                )
            )

    return violations


def _keys_by_model(rows: dict[ResultKey, object]) -> dict[str, tuple[str, ...]]:
    by_model: dict[str, list[str]] = {}
    for key in rows:
        by_model.setdefault(key.model, []).append(key.config_id)
    return {model: tuple(sorted(configs)) for model, configs in by_model.items()}


def _same_commit_prefix(expected_short_sha: str, actual_sha: str | None) -> bool:
    return actual_sha is not None and actual_sha.startswith(expected_short_sha)
