"""ScaleCUA catalog and dataset contract tests."""

from __future__ import annotations

import ast
import copy
import json
import re
import textwrap
from pathlib import Path

import pytest

from lite.gym.envs.lite.scalecua import main as M
from lite.gym.envs.lite.scalecua.src.utils import assets, dataset
from lite.gym.errors import EnvDepsMissingError


def _cache_ready() -> bool:
    return all(dataset.catalog_path(split).is_file() for split in dataset.RUNTIME_SPLITS)


def test_asset_import_is_download_free_and_identity_has_manifest():
    snap = assets.asset_snapshot()
    assert snap["repo"] == "extreme1228/ScaleCUA"
    assert "local_eval_manifest" not in snap
    assert set(snap["components"]) == {
        "generated_judges",
        "generated_tasks",
        "rl_judges",
        "rl_tasks",
    }


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_catalog_lock_matches_generated_jsonl():
    """Reproducibility gate — the scalecua analogue of lite.osworld's.

    Its absence let 23611d76 commit a lock whose `splits.rl.sha256` predated its
    own oracle-fixture edit. Because `import_all` used to WRITE the lock before
    validating it, install.sh silently rewrote the tracked file instead of
    failing, and on a host with an equally stale rl.jsonl the mismatched pair
    agreed — certifying a stale catalog as fresh.

    The `splits` loop alone CANNOT see that accident: the oracle fixtures are a
    generator *input*, not a split, so editing `data/oracle/*.jsonl` leaves every
    pinned split hash intact. `sources.oracle_fixture_identity` is the field that
    records them, so it is checked here too.
    """
    import hashlib

    data_dir = Path(dataset.__file__).resolve().parents[2] / "data"
    lock = json.loads((data_dir / "catalog.lock.json").read_text())
    for split, entry in lock["splits"].items():
        path = data_dir / entry["path"]
        raw = path.read_bytes()
        rows = sum(1 for line in raw.splitlines() if line.strip())
        actual = hashlib.sha256(raw).hexdigest()
        assert rows == entry["rows"], f"{split} row count changed"
        assert actual == entry["sha256"], (
            f"{entry['path']} bytes changed!\n"
            f"  pinned : {entry['sha256']}\n"
            f"  actual : {actual}\n"
            "  If intentional: regenerate and run scripts/utils/tasks.sh refresh-lock."
        )

    actual_oracle = dataset.oracle_fixture_identity()
    assert lock["sources"]["oracle_fixture_identity"] == actual_oracle, (
        "oracle fixtures changed since the lock was written!\n"
        f"  pinned : {lock['sources']['oracle_fixture_identity']}\n"
        f"  actual : {actual_oracle}\n"
        "  The generated splits above may predate this edit. If intentional: "
        "regenerate and run scripts/utils/tasks.sh refresh-lock."
    )


_EVAL_DOMAIN_ROWS = [
    {
        "task_id": "osworld_chrome_1",
        "metadata": {"osworld_id": "id-1", "others": {"domain": "chrome"}},
    },
    {
        "task_id": "osworld_gimp_2",
        "metadata": {"osworld_id": "id-2", "others": {"domain": "gimp"}},
    },
    {
        "task_id": "osworld_vlc_3",
        "metadata": {"osworld_id": "id-3", "others": {"domain": "vlc"}},
    },
]


def _domain_map_identity(tmp_path, monkeypatch, name: str, rows: list[dict]) -> dict:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(assets, "OSWORLD_EVAL_JSONL", path)
    return dataset.osworld_eval_domain_map_identity()


def test_osworld_domain_lock_ignores_eval_catalog_edits_outside_the_mapping(
    tmp_path, monkeypatch
):
    """Pinning eval.jsonl bytes here makes any unrelated catalog edit read as stale.

    A stale read registers zero lite.scalecua tasks.
    """
    base = _domain_map_identity(tmp_path, monkeypatch, "base", _EVAL_DOMAIN_ROWS)
    assert base["entries"] == 3

    reordered = _domain_map_identity(
        tmp_path, monkeypatch, "reordered", list(reversed(_EVAL_DOMAIN_ROWS))
    )
    enriched_rows = copy.deepcopy(_EVAL_DOMAIN_ROWS)
    enriched_rows[0]["instruction"] = "Enable Do Not Track."
    enriched_rows[0]["metadata"]["others"]["oracle_actions"] = [{"type": "execute"}]
    enriched_rows[1]["metadata"]["config"] = [{"type": "launch"}]
    enriched = _domain_map_identity(tmp_path, monkeypatch, "enriched", enriched_rows)

    assert reordered == base
    assert enriched == base


def _flip_domain(rows: list[dict]) -> list[dict]:
    rows[1]["metadata"]["others"]["domain"] = "os"
    return rows


def _drop_row(rows: list[dict]) -> list[dict]:
    return rows[:-1]


def _rename_osworld_id(rows: list[dict]) -> list[dict]:
    rows[0]["metadata"]["osworld_id"] = "id-renamed"
    return rows


@pytest.mark.parametrize("mutate", [_flip_domain, _drop_row, _rename_osworld_id])
def test_osworld_domain_lock_changes_when_the_mapping_changes(tmp_path, monkeypatch, mutate):
    """A real mapping change must invalidate the lock, or a wrong domain map ships silently."""
    base = _domain_map_identity(tmp_path, monkeypatch, "base", _EVAL_DOMAIN_ROWS)
    mutated = _domain_map_identity(
        tmp_path, monkeypatch, mutate.__name__, mutate(copy.deepcopy(_EVAL_DOMAIN_ROWS))
    )
    assert mutated["sha256"] != base["sha256"]


def test_catalog_lock_records_the_domain_mapping_not_the_eval_catalog_bytes():
    """The retired byte-pin key must not come back — it is what registered zero tasks."""
    sources = json.loads(dataset.catalog_lock_path().read_text(encoding="utf-8"))["sources"]
    assert "osworld_eval_domain_catalog_identity" not in sources
    assert set(sources["osworld_eval_domain_map_identity"]) == {
        "source",
        "entries",
        "sha256",
    }


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_catalog_counts_and_static_contract():
    report = dataset.validate_all(strict_counts=True)
    assert report["splits"]["train"]["rows"] == 20289
    assert report["splits"]["rl"]["rows"] == 2049
    assert set(report["splits"]) == {"train", "rl"}


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_catalog_key_lists_are_token_lists_not_chord_strings():
    offenders: list[str] = []

    def scan(value: object, path: str) -> None:
        if isinstance(value, dict):
            params = value.get("parameters")
            if value.get("type") == "key" and isinstance(params, dict):
                keys = params.get("keys")
                if isinstance(keys, list):
                    bad = [
                        token
                        for token in keys
                        if isinstance(token, str) and "+" in token and token != "+"
                    ]
                    if bad:
                        offenders.append(f"{path}: {bad!r}")
            for key, child in value.items():
                scan(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan(child, f"{path}[{index}]")

    for split in dataset.RUNTIME_SPLITS:
        for line_number, row in dataset.iter_jsonl(dataset.catalog_path(split)):
            scan(row, f"{split}:{line_number}")

    assert offenders == []


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_registration_catalog_injects_rl_oracles_like_osworld():
    registered_path = dataset.registration_catalog_path("rl")
    rows = [row for _, row in dataset.iter_jsonl(registered_path)]
    excluded = [
        row
        for row in rows
        if row["metadata"]["others"].get("exclude_reason")
    ]
    runnable = [
        row
        for row in rows
        if not row["metadata"]["others"].get("exclude_reason")
    ]

    assert len(rows) == 2049
    assert len(excluded) == 240
    assert len(runnable) == 1809
    assert all(row["metadata"]["others"].get("oracle_actions") for row in runnable)
    assert all("oracle_actions" not in row["metadata"] for row in rows)
    assert all("oracle_after_postconfig" not in row["metadata"] for row in rows)
    assert all("oracle_expected_reward" not in row["metadata"] for row in rows)
    assert all("oracle_fixture_id" not in row["metadata"] for row in rows)


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_generated_oracle_domains_are_generated_from_python_source():
    from lite.gym.envs.lite.scalecua.src.gen.oracle import __main__ as gen_oracle

    # Per-domain counts are build_rows() output = {curated ORACLES} intersect
    # {runnable tasks}. Excluded task rows are filtered out at generation, so
    # these track the exclusion set, not the raw frozen ORACLES blob length.
    expected_counts = {
        "chrome": 168,
        "gimp": 174,
        "libreoffice_calc": 251,
        "libreoffice_impress": 255,
        "libreoffice_writer": 180,
        "multi_apps": 335,
        "os": 186,
        "thunderbird": 143,
        "vlc": 135,
        "vs_code": 182,
    }
    assert set(gen_oracle.SHARDS) == set(expected_counts)

    oracle_src = assets.ENV_DIR / "src" / "gen" / "oracle"
    assert not sorted(path.name for path in oracle_src.glob("rl_auto*.py"))
    assert not sorted(path.name for path in oracle_src.glob("rl_*.py"))
    assert sorted(
        path.stem
        for path in (oracle_src / "domains").glob("*.py")
        if path.stem != "__init__"
    ) == sorted(expected_counts)

    oracle_dir = assets.ENV_DIR / "data" / "oracle"
    assert sorted(path.name for path in oracle_dir.glob("*.jsonl")) == [
        "rl.jsonl",
        "train.jsonl",
    ]
    committed_by_fixture = {}
    for path in sorted(oracle_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            committed_by_fixture[row["fixture_id"]] = row

    for domain, expected_count in expected_counts.items():
        rows = gen_oracle.build_rows(domain)

        assert len(rows) == expected_count
        assert {row["domain"] for row in rows} == {domain}
        for row in rows:
            assert committed_by_fixture[row["fixture_id"]] == row
            if "source" not in row:
                continue
            actions = row.get("oracle_actions") or (
                row.get("oracle_trajectory") or {}
            ).get("actions") or []
            for action in actions:
                command = action.get("parameters", {}).get("command", "")
                if "python3 - <<'PY'\n" not in command:
                    continue
                body = command.split("python3 - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
                ast.parse(textwrap.dedent(body))


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_oracle_fixtures_do_not_target_excluded_or_missing_tasks():
    for split in dataset.RUNTIME_SPLITS:
        rows = [row for _, row in dataset.iter_jsonl(dataset.catalog_path(split))]
        assert dataset._apply_oracle_fixtures(split, rows)


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_oracle_heredoc_terminators_are_standalone_lines():
    from lite.gym.envs.lite.scalecua.src.gen.oracle import __main__ as gen_oracle

    bad: list[str] = []
    for domain in gen_oracle.SHARDS:
        for row in gen_oracle.build_rows(domain):
            actions = row.get("oracle_actions") or (
                row.get("oracle_trajectory") or {}
            ).get("actions") or []
            for action in actions:
                command = action.get("parameters", {}).get("command", "")
                if not isinstance(command, str):
                    continue
                tokens = set(re.findall(r"<<'([A-Za-z_][A-Za-z0-9_]*)'", command))
                if not tokens:
                    continue
                for line in command.splitlines():
                    for token in tokens:
                        suffix = line[len(token) : len(token) + 1]
                        if line.startswith(token) and line != token and suffix in {
                            ";",
                            " ",
                            "\t",
                            "|",
                            "&",
                        }:
                            bad.append(
                                f"{row['fixture_id']}:{row['task_id']}: {line!r}"
                            )

    assert bad == []


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_domain_oracle_registry_preserves_stable_legacy_fixture_gap():
    from lite.gym.envs.lite.scalecua.src.gen.oracle import __main__ as gen_oracle

    rows = gen_oracle.build_rows("chrome")
    task_ids = {row["task_id"] for row in rows}
    fixture_ids = {row["fixture_id"] for row in rows}

    assert (
        "scalecua_osworld_rl_chrome_47543840_672a_467d_80df_8f7c3b9788c9_traj_verify_0"
        not in task_ids
    )
    assert "oracle_rl_auto_b1_0013" not in fixture_ids
    assert "oracle_rl_auto_b1_0014" in fixture_ids


def _minimal_scalecua_catalog_row(task_id: str, *, runtime_split: str = "train") -> dict:
    return {
        "task_id": task_id,
        "instruction": "Do a task.",
        "max_steps": 30,
        "metadata": {
            "others": {
                "source_split": dataset.RUNTIME_TO_SOURCE[runtime_split],
                "domain": "os",
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_validate_catalog_rejects_noncanonical_exclude_reason(tmp_path):
    path = tmp_path / "train.jsonl"
    row = _minimal_scalecua_catalog_row("bad_reason")
    row["metadata"]["others"]["exclude_reason"] = "block: prose"
    _write_jsonl(path, [row])

    with pytest.raises(RuntimeError, match="invalid exclude_reason"):
        dataset.validate_catalog(path, expected_split="train")


@pytest.mark.parametrize("reason", ["proxy_required:any", "missing_dependency:any"])
def test_validate_catalog_rejects_forbidden_exclude_reason_detail(tmp_path, reason):
    path = tmp_path / "train.jsonl"
    row = _minimal_scalecua_catalog_row("bad_detail")
    row["metadata"]["others"]["exclude_reason"] = reason
    _write_jsonl(path, [row])

    with pytest.raises(RuntimeError, match="invalid exclude_reason"):
        dataset.validate_catalog(path, expected_split="train")


@pytest.mark.parametrize("keys", [["ctrl+s"], ["ctrl+s", "enter"]])
def test_validate_catalog_rejects_chord_strings_inside_key_lists(tmp_path, keys):
    path = tmp_path / "train.jsonl"
    row = _minimal_scalecua_catalog_row("bad_chord_string_key_list")
    row["metadata"]["evaluator"] = {
        "postconfig": [{"type": "key", "parameters": {"keys": keys}}],
    }
    _write_jsonl(path, [row])

    with pytest.raises(RuntimeError, match="key list contains chord string"):
        dataset.validate_catalog(path, expected_split="train")


def test_validate_all_validates_exclude_reason_in_each_runtime_split(tmp_path):
    train = _minimal_scalecua_catalog_row("train_ok", runtime_split="train")
    train["metadata"]["others"]["exclude_reason"] = "infeasible"
    rl = _minimal_scalecua_catalog_row("rl_bad", runtime_split="rl")
    rl["metadata"]["others"]["exclude_reason"] = "trivial_pass: prose"

    _write_jsonl(tmp_path / "train.jsonl", [train])
    _write_jsonl(tmp_path / "rl.jsonl", [rl])

    with pytest.raises(RuntimeError, match="rl.jsonl:1: invalid exclude_reason"):
        dataset.validate_all(tmp_path)


def test_scalecua_excludes_exact_imagemagick_missing_dependency_rows():
    exact_ids = {
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_0",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_1",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_2",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_3",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_4",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_5",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_6",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_7",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_8",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_9",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_10",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_11",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_12",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_23",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_34",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_45",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_56",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_57",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_58",
        "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_d303c73d4bf6_task_verify_59",
    }
    assert dataset.MISSING_DEPENDENCY_IMAGEMAGICK_TASK_IDS == exact_ids
    payload = {"instruction": "Use ImageMagick on the command line.", "evaluator": {}}

    for task_id in exact_ids:
        assert (
            dataset._exclude_reason(
                payload,
                task_id=task_id,
                inherited_exclusion=None,
                unsupported=[],
                runtime_split="train",
                broken_metrics=frozenset(),
            )
            == "missing_dependency:imagemagick"
        )

    for index in (13, 22, 24, 33, 35, 44, 46, 55, 60):
        assert (
            dataset._exclude_reason(
                payload,
                task_id=(
                    "scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_"
                    f"d303c73d4bf6_task_verify_{index}"
                ),
                inherited_exclusion=None,
                unsupported=[],
                runtime_split="train",
                broken_metrics=frozenset(),
            )
            is None
        )


def test_scalecua_excludes_exact_java_missing_dependency_row_only():
    exact_id = (
        "scalecua_osworld_train_libreoffice_writer_e1fc0df3_c8b9_4ee7_864c_"
        "d0b590d3aa56_task_verify_24"
    )
    assert dataset.MISSING_DEPENDENCY_JAVA_TASK_IDS == {exact_id}
    payload = {"instruction": "Install the NLPSOLVER extension.", "evaluator": {}}

    assert (
        dataset._exclude_reason(
            payload,
            task_id=exact_id,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "missing_dependency:java"
    )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=exact_id,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )
    for neighbor in (23, 25):
        assert (
            dataset._exclude_reason(
                payload,
                task_id=(
                    "scalecua_osworld_train_libreoffice_writer_e1fc0df3_c8b9_"
                    f"4ee7_864c_d0b590d3aa56_task_verify_{neighbor}"
                ),
                inherited_exclusion=None,
                unsupported=[],
                runtime_split="train",
                broken_metrics=frozenset(),
            )
            is None
        )


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_no_legacy_catalog_names_or_resolve_main_urls():
    assert sorted(p.name for p in assets.CATALOG_DIR.glob("*.jsonl")) == [
        "rl.jsonl",
        "train.jsonl",
    ]
    assert not list(assets.CACHE_DIR.glob("*.jsonl"))
    for split in dataset.RUNTIME_SPLITS:
        for _, row in dataset.iter_jsonl(dataset.catalog_path(split)):
            if not row["metadata"]["others"].get("exclude_reason"):
                assert "resolve/main" not in json.dumps(row, ensure_ascii=False)


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_catalog_excludes_missing_author_results_reference_assets():
    leaked: list[str] = []
    for split in ("train", "rl"):
        for _, row in dataset.iter_jsonl(dataset.catalog_path(split)):
            evaluator_text = json.dumps(
                row["metadata"].get("evaluator", {}),
                ensure_ascii=False,
            )
            if (
                "/home/lvbowen/project/AutoGen/results/" in evaluator_text
                and '"reference_path"' in evaluator_text
                and not row["metadata"]["others"].get("exclude_reason")
            ):
                leaked.append(row["task_id"])

    assert leaked == []


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_registry_counts_direct_mode(monkeypatch):
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)
    import lite.gym as gym

    try:
        ids = gym.registry.task_ids("lite.scalecua")
    except EnvDepsMissingError as e:
        pytest.skip(f"lite.scalecua catalogs not generated (run install.sh): {e}")
    assert {k: len(v) for k, v in ids.items()} == {
        "train": 20289,
        "rl": 2049,
    }


def test_missing_catalog_error_message(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset.assets, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(M.dataset.assets, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(M.dataset, "catalog_root", lambda: tmp_path)
    monkeypatch.setattr(
        M.dataset,
        "catalog_path",
        lambda split, root=None: tmp_path / f"{split}.jsonl",
    )
    monkeypatch.setattr(M, "_catalog_registered", False)
    with pytest.raises(EnvDepsMissingError) as raised:
        M._register_tasks()
    assert "lite/gym/envs/lite/scalecua/scripts/install.sh" in raised.value.install
