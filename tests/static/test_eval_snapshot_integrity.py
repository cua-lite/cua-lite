"""Static guards for committed eval snapshot Markdown/JSON consistency."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_INTEGRITY = ROOT / "devs" / "exps" / "eval" / "utils" / "snapshot_integrity.py"


def _load_snapshot_integrity():
    spec = importlib.util.spec_from_file_location(
        "snapshot_integrity_under_test", SNAPSHOT_INTEGRITY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


snapshot_integrity = _load_snapshot_integrity()


KNOWN_MD_ONLY_SNAPSHOTS = {
    "devs/exps/eval/androidlab/logs/2026-04-26T18-38_8b9d4bd/run_0.md",
    "devs/exps/eval/androidlab/logs/2026-04-28T21-31_785df232/run_0.md",
    "devs/exps/eval/androidworld/logs/2026-04-26T18-38_8b9d4bd/run_0.md",
    "devs/exps/eval/androidworld/logs/2026-04-28T21-31_785df232/run_0.md",
    "devs/exps/eval/browsergym.miniwob/logs/2026-07-01T20-57_f57d3cbb92/run_1_default.md",
    "devs/exps/eval/browsergym.miniwob/logs/2026-07-01T20-57_f57d3cbb92/run_1_textonly.md",
    "devs/exps/eval/cua/logs/2026-07-09T22-41_d982edcfb3/run_0.md",
    "devs/exps/eval/lite.osworld/logs/2026-04-26T18-38_8b9d4bd/run_0.md",
    "devs/exps/eval/osworld/logs/2026-04-26T18-38_8b9d4bd/run_0.md",
    "devs/exps/eval/osworld_g/logs/2026-04-27T23-21_828ea14/run_0.md",
    "devs/exps/eval/osworld_g/logs/2026-04-28T05-32_ff51fcf/run_0.md",
    "devs/exps/eval/osworld_g/logs/2026-04-28T21-31_785df232/run_0.md",
    "devs/exps/eval/screenspot_pro/logs/2026-04-27T23-21_828ea14/run_0.md",
    "devs/exps/eval/screenspot_pro/logs/2026-04-28T05-32_ff51fcf/run_0.md",
    "devs/exps/eval/screenspot_pro/logs/2026-04-28T21-31_785df232/run_0.md",
    "devs/exps/eval/webharbor.webvoyager/logs/2026-07-08T01-38_c1001a613c/run_0.md",
}

KNOWN_JSON_ONLY_SNAPSHOTS = {
    "devs/exps/eval/browsergym.miniwob/logs/2026-07-01T20-57_f57d3cbb92/run_0_default.json",
    "devs/exps/eval/lite.osworld/logs/2026-07-12T22-59_4a6cb49b0d/run_0.json",
    "devs/exps/eval/lite.osworld/logs/2026-07-18T14-23_0cde57702f/run_0.json",
    "devs/exps/eval/webharbor.webvoyager/logs/2026-07-08T01-38_c1001a613c/run_0_default.json",
}

def _snap(env: str, commit_dir: str, run_id: str = "run_0") -> str:
    return f"devs/exps/eval/{env}/logs/{commit_dir}/{run_id}.md"


def _json_snap(env: str, commit_dir: str, run_id: str = "run_0") -> str:
    return f"devs/exps/eval/{env}/logs/{commit_dir}/{run_id}.json"


def _vid(path: str, kind: str, detail: str) -> str:
    return f"{path}|{kind}|{detail}"


ANDROIDLAB_JUL09 = _snap("androidlab", "2026-07-09T05-10_d6f368e4ff")
ANDROIDWORLD_JUL09 = _snap("androidworld", "2026-07-09T03-27_03fc490a07")
BROWSERGYM_JUL17 = _snap("browsergym.miniwob", "2026-07-17T15-42_c4719f1ca1")
BROWSERGYM_JUL01_JSON = _json_snap(
    "browsergym.miniwob",
    "2026-07-01T20-57_f57d3cbb92",
    "run_0_default",
)
LITE_OSWORLD_JUL09 = _snap("lite.osworld", "2026-07-09T06-01_8ad4aaa03c")
MOBILEGYM_JUL08 = _snap("mobilegym", "2026-07-08T20-40_2a40f5258e")
OSWORLD_JUL18 = _snap("osworld", "2026-07-18T17-00_65ea6496ef")
OSWORLD_2_JUL18 = _snap("osworld_2", "2026-07-18T14-23_0cde57702f")
OSWORLD_G_JUL09 = _snap("osworld_g", "2026-07-09T03-09_d30ef912ca")
SCREENSPOT_JUL09 = _snap("screenspot_pro", "2026-07-09T04-31_5b887a39c1")
WEBVOYAGER_SOM = _snap(
    "webharbor.webvoyager",
    "2026-07-10T15-40_651af22040",
    "run_0_som",
)


KNOWN_PAIR_VIOLATIONS = {
    _vid(
        ANDROIDLAB_JUL09,
        "md.artifact_root",
        "expected=.exps/eval/androidlab/2026-07-09T05-10_d6f368e4ff/run_0 "
        "actual=.exps/eval/androidlab/2026-07-04T20-37_82b70d153d/run_0",
    ),
    _vid(
        ANDROIDLAB_JUL09,
        "md.commit_dir",
        "expected=2026-07-09T05-10_d6f368e4ff "
        "actual=2026-07-04T20-37_82b70d153d",
    ),
    _vid(ANDROIDLAB_JUL09, "md.commit_short_sha", "expected=d6f368e4ff actual=82b70d153d"),
    _vid(ANDROIDLAB_JUL09, "row.json_only", "config=default model=Tongyi-MAI/MAI-UI-8B"),
    _vid(
        ANDROIDWORLD_JUL09,
        "md.artifact_root",
        "expected=.exps/eval/androidworld/2026-07-09T03-27_03fc490a07/run_0 "
        "actual=.exps/eval/androidworld/2026-07-07T03-53_4fa96a6e41/run_0",
    ),
    _vid(
        ANDROIDWORLD_JUL09,
        "md.commit_dir",
        "expected=2026-07-09T03-27_03fc490a07 "
        "actual=2026-07-07T03-53_4fa96a6e41",
    ),
    _vid(
        ANDROIDWORLD_JUL09,
        "md.commit_short_sha",
        "expected=03fc490a07 actual=4fa96a6e41",
    ),
    _vid(
        ANDROIDWORLD_JUL09,
        "row.json_only",
        "config=default model=Tongyi-MAI/MAI-UI-8B",
    ),
    _vid(BROWSERGYM_JUL17, "md.commit_short_sha", "expected=c4719f1ca1 actual=f28f81402b"),
    _vid(
        LITE_OSWORLD_JUL09,
        "md.artifact_root",
        "expected=.exps/eval/lite.osworld/2026-07-09T06-01_8ad4aaa03c/run_0 "
        "actual=.exps/eval/lite.osworld/2026-07-04T23-38_bc2a8585ec/run_0",
    ),
    _vid(
        LITE_OSWORLD_JUL09,
        "md.commit_dir",
        "expected=2026-07-09T06-01_8ad4aaa03c "
        "actual=2026-07-04T23-38_bc2a8585ec",
    ),
    _vid(
        LITE_OSWORLD_JUL09,
        "md.commit_short_sha",
        "expected=8ad4aaa03c actual=bc2a8585ec",
    ),
    _vid(LITE_OSWORLD_JUL09, "row.md_only", "config=default model=xlangai/OpenCUA-7B"),
    _vid(
        LITE_OSWORLD_JUL09,
        "row.mean_episode_return",
        "config=default model=Qwen/Qwen3.5-4B md=0.2112 json=0.2212",
    ),
    _vid(
        MOBILEGYM_JUL08,
        "md.commit_dir",
        "expected=2026-07-08T20-40_2a40f5258e "
        "actual=2026-07-08T16-14_2a40f5258e",
    ),
    _vid(MOBILEGYM_JUL08, "row.json_only", "config=default model=Tongyi-MAI/MAI-UI-8B"),
    _vid(OSWORLD_JUL18, "md.commit_short_sha", "expected=65ea6496ef actual=4694f38a40"),
    _vid(OSWORLD_JUL18, "row.json_only", "config=default model=gpt-5.5"),
    _vid(
        OSWORLD_2_JUL18,
        "md.artifact_root",
        "expected=.exps/eval/osworld_2/2026-07-18T14-23_0cde57702f/run_0 "
        "actual=.exps/eval/osworld_2/2026-07-17T19-06_551131f803/run_0",
    ),
    _vid(
        OSWORLD_2_JUL18,
        "md.commit_dir",
        "expected=2026-07-18T14-23_0cde57702f "
        "actual=2026-07-17T19-06_551131f803",
    ),
    _vid(
        OSWORLD_2_JUL18,
        "md.commit_short_sha",
        "expected=0cde57702f actual=551131f803",
    ),
    _vid(
        OSWORLD_2_JUL18,
        "md.results.malformed_row",
        "line=16 expected_columns=3 actual_columns=2",
    ),
    _vid(
        OSWORLD_G_JUL09,
        "md.artifact_root",
        "expected=.exps/eval/osworld_g/2026-07-09T03-09_d30ef912ca/run_0 "
        "actual=.exps/eval/osworld_g/2026-07-06T20-34_0180ceefd8/run_0",
    ),
    _vid(
        OSWORLD_G_JUL09,
        "md.commit_dir",
        "expected=2026-07-09T03-09_d30ef912ca "
        "actual=2026-07-06T20-34_0180ceefd8",
    ),
    _vid(OSWORLD_G_JUL09, "md.commit_short_sha", "expected=d30ef912ca actual=0180ceefd8"),
    _vid(SCREENSPOT_JUL09, "md.commit_short_sha", "expected=5b887a39c1 actual=1936546595"),
    _vid(SCREENSPOT_JUL09, "row.json_only", "config=default model=Tongyi-MAI/MAI-UI-8B"),
    _vid(
        SCREENSPOT_JUL09,
        "row.mean_episode_return",
        "config=default model=Qwen/Qwen3.5-27B md=0.0702 json=0.6806",
    ),
    _vid(
        WEBVOYAGER_SOM,
        "row.config_id",
        "model=Qwen/Qwen3-VL-2B-Instruct md_configs=som json_configs=default",
    ),
    _vid(
        WEBVOYAGER_SOM,
        "row.config_id",
        "model=Qwen/Qwen3-VL-32B-Instruct md_configs=som json_configs=default",
    ),
    _vid(
        WEBVOYAGER_SOM,
        "row.config_id",
        "model=Qwen/Qwen3-VL-4B-Instruct md_configs=som json_configs=default",
    ),
    _vid(
        WEBVOYAGER_SOM,
        "row.config_id",
        "model=Qwen/Qwen3-VL-8B-Instruct md_configs=som json_configs=default",
    ),
    _vid(
        WEBVOYAGER_SOM,
        "row.config_id",
        "model=Qwen/Qwen3.5-27B md_configs=som json_configs=default",
    ),
    _vid(
        WEBVOYAGER_SOM,
        "row.config_id",
        "model=Qwen/Qwen3.5-2B md_configs=som json_configs=default",
    ),
    _vid(
        WEBVOYAGER_SOM,
        "row.config_id",
        "model=Qwen/Qwen3.5-4B md_configs=som json_configs=default",
    ),
    _vid(
        WEBVOYAGER_SOM,
        "row.config_id",
        "model=Qwen/Qwen3.5-9B md_configs=som json_configs=default",
    ),
}


KNOWN_FILE_VIOLATIONS = {
    _vid(
        BROWSERGYM_JUL01_JSON,
        "json.artifact_root",
        "expected=.exps/eval/browsergym.miniwob/2026-07-01T20-57_f57d3cbb92/"
        "run_0_default "
        "actual=.exps/eval/browsergym.miniwob/2026-07-01T20-57_f57d3cbb92/run_0",
    ),
    _vid(
        BROWSERGYM_JUL01_JSON,
        "json.result.artifact_path",
        "model=gpt-5.5 "
        "artifact_path=.exps/eval/browsergym.miniwob/2026-07-09T02-57_6b023d0d7a/"
        "run_0/gpt-5.5 "
        "artifact_root=.exps/eval/browsergym.miniwob/2026-07-01T20-57_f57d3cbb92/"
        "run_0",
    ),
    _vid(
        BROWSERGYM_JUL01_JSON,
        "json.run_id",
        "expected=run_0_default actual=run_0",
    ),
    _vid(
        OSWORLD_2_JUL18,
        "md.results.malformed_row",
        "line=16 expected_columns=3 actual_columns=2",
    ),
}


def _snapshot_pair(tmp_path: Path, *, md_config: str = "default", md_mer: str = "0.5000"):
    eval_root = tmp_path / "eval"
    commit_dir = "2026-01-02T03-04_abcdef123"
    run_id = "run_0"
    snapshot_dir = eval_root / "toy" / "logs" / commit_dir
    snapshot_dir.mkdir(parents=True)
    artifact_root = f".exps/eval/toy/{commit_dir}/{run_id}"
    em_dash = chr(0x2014)

    md_path = snapshot_dir / f"{run_id}.md"
    md_path.write_text(
        "\n".join(
            [
                f"# toy @ {commit_dir} - {run_id}",
                "",
                "- **Commit**: `abcdef1234567890`",
                f"- **Artifacts**: `{artifact_root}/`",
                "",
                "## Results",
                "",
                "| Config | Model | Finished | Mean episode return |",
                "|---|---|---|---|",
                f"| `{md_config}` | `model/a` | 2/2 | {md_mer} |",
                f"| `default` | `not-started/model` | _**0/2**_ | _**{em_dash}**_ |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    json_path = snapshot_dir / f"{run_id}.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "env": "toy",
                "commit_dir": commit_dir,
                "run_id": run_id,
                "commit": {"short_sha": "abcdef123"},
                "artifact_root": artifact_root,
                "result_count": 1,
                "status_counts": {"complete": 1, "partial": 0, "empty": 0},
                "results": [
                    {
                        "model": "model/a",
                        "slug": "model_a",
                        "base_slug": "model_a",
                        "config_id": "default",
                        "status": "complete",
                        "num_valid": 2,
                        "num_tasks": 2,
                        "num_samples": 2,
                        "mean_episode_return": 0.5,
                        "summary_path": f"{artifact_root}/model_a/summary.json",
                        "artifact_path": f"{artifact_root}/model_a",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return eval_root, md_path, json_path


def _json_only_snapshot_with_bad_artifact_path(tmp_path: Path):
    eval_root = tmp_path / "eval"
    commit_dir = "2026-01-02T03-04_abcdef123"
    run_id = "run_0"
    snapshot_dir = eval_root / "toy" / "logs" / commit_dir
    snapshot_dir.mkdir(parents=True)
    artifact_root = f".exps/eval/toy/{commit_dir}/{run_id}"
    wrong_artifact_path = ".exps/eval/other/run_0/model_a"
    json_path = snapshot_dir / f"{run_id}.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "env": "toy",
                "commit_dir": commit_dir,
                "run_id": run_id,
                "commit": {"short_sha": "abcdef123"},
                "artifact_root": artifact_root,
                "result_count": 1,
                "status_counts": {"complete": 1, "partial": 0, "empty": 0},
                "results": [
                    {
                        "model": "model/a",
                        "config_id": "default",
                        "status": "complete",
                        "num_valid": 2,
                        "num_tasks": 2,
                        "mean_episode_return": 0.5,
                        "summary_path": f"{wrong_artifact_path}/summary.json",
                        "artifact_path": wrong_artifact_path,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return eval_root


def test_eval_snapshot_orphan_debt_is_explicit() -> None:
    md_only, json_only = snapshot_integrity.orphan_snapshot_paths()

    assert set(md_only) == KNOWN_MD_ONLY_SNAPSHOTS
    assert set(json_only) == KNOWN_JSON_ONLY_SNAPSHOTS


def test_eval_snapshot_file_violations_are_explicit() -> None:
    violations = {violation.id() for violation in snapshot_integrity.validate_snapshot_files()}

    assert violations == KNOWN_FILE_VIOLATIONS


def test_eval_snapshot_pair_violations_are_explicit() -> None:
    violations = {violation.id() for violation in snapshot_integrity.validate_snapshot_pairs()}

    assert violations == KNOWN_PAIR_VIOLATIONS


def test_snapshot_validator_accepts_matching_pair(tmp_path: Path) -> None:
    eval_root, md_path, json_path = _snapshot_pair(tmp_path)

    violations = snapshot_integrity.validate_same_stem_snapshot_pair(
        md_path,
        json_path,
        eval_root=eval_root,
        repo_root=tmp_path,
    )

    assert violations == []


def test_snapshot_validator_catches_config_id_drift(tmp_path: Path) -> None:
    eval_root, md_path, json_path = _snapshot_pair(tmp_path, md_config="som")

    violations = snapshot_integrity.validate_same_stem_snapshot_pair(
        md_path,
        json_path,
        eval_root=eval_root,
        repo_root=tmp_path,
    )

    assert {violation.kind for violation in violations} == {"row.config_id"}


def test_snapshot_validator_catches_metric_drift(tmp_path: Path) -> None:
    eval_root, md_path, json_path = _snapshot_pair(tmp_path, md_mer="0.1000")

    violations = snapshot_integrity.validate_same_stem_snapshot_pair(
        md_path,
        json_path,
        eval_root=eval_root,
        repo_root=tmp_path,
    )

    assert {violation.kind for violation in violations} == {"row.mean_episode_return"}


def test_snapshot_file_validator_catches_json_identity_drift(tmp_path: Path) -> None:
    eval_root, _md_path, json_path = _snapshot_pair(tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["run_id"] = "run_1"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    violations = snapshot_integrity.validate_snapshot_files(eval_root=eval_root, repo_root=tmp_path)

    assert {violation.kind for violation in violations} == {"json.run_id"}


def test_snapshot_file_validator_catches_json_only_internal_drift(tmp_path: Path) -> None:
    eval_root = _json_only_snapshot_with_bad_artifact_path(tmp_path)

    violations = snapshot_integrity.validate_snapshot_files(eval_root=eval_root, repo_root=tmp_path)

    assert {violation.kind for violation in violations} == {"json.result.artifact_path"}
