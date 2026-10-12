"""Private asset integrity and implemented reference-task registration contracts."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from examples.not_a_robot import reference_assets, registration  # noqa: F401
from examples.not_a_robot.local_tasks import ASSETS, LOCAL_TASKS
from lite import gym


def test_first_ten_have_explicit_reference_evidence_and_limits():
    tasks = [LOCAL_TASKS[f"neal_{level:02d}"] for level in range(1, 11)]
    assert len(LOCAL_TASKS) == 54
    for level, task in enumerate(tasks, 1):
        env = gym.make(f"visual_tasks@{task['id']}", seed=7)
        metadata = env.metadata.others
        reference = metadata["reference"]
        assert metadata["source"] == "neal_reference_reconstruction"
        assert metadata["seed_scope"] == "local_dynamics_not_original_challenge_seed"
        assert reference["level"] == level
        assert reference["fidelity"] == "captured_instance"
        assert reference["completion_recorded"] is True
        assert reference["limitations"]
        assert (
            reference["supplement_sha256"]
            == reference_assets.REFERENCE_CATALOG["archives"][
                "incremental" if level == 6 else "first10"
            ]
        )
        assert not {"answer", "targets", "selected_regions", "solution"} & reference.keys()
        assert env.unwrapped._page is None


def test_reference_asset_http_allowlist_does_not_include_capture_logs():
    reference_paths = {path for path in ASSETS if path.startswith("/reference_assets/")}
    assert reference_paths == {
        f"/reference_assets/{name}" for name in reference_assets.REFERENCE_HASHES
    }
    assert len(reference_paths) == 66
    assert ASSETS["/reference_assets/level47_chart.json"][1] == "application/json"
    assert "/reference_assets/level47_cover.webp" in reference_paths
    assert "/reference_assets/module_1070.js" not in ASSETS
    assert "/reference_assets/level20_image_01.webp" in reference_paths
    assert all(
        f"/reference_assets/level37_image_{index:02d}.webp" in reference_paths
        for index in range(1, 10)
    )
    assert "/reference_assets/outcome.json" not in ASSETS
    assert "/reference_assets/manifest.json" not in ASSETS


@pytest.fixture
def small_archive(tmp_path, monkeypatch):
    archive = tmp_path / "reference.zip"
    data = b"private-test-reference"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("implementation_reference/assets/test.webp", data)
        source.writestr("../../not-imported.txt", b"never extracted")
    monkeypatch.setattr(
        reference_assets,
        "REFERENCE_CATALOG",
        {
            "archives": {"fixture": hashlib.sha256(archive.read_bytes()).hexdigest()},
            "derived_files": {},
            "files": {
                "test.webp": {
                    "archive": "fixture",
                    "member": "implementation_reference/assets/test.webp",
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            },
        },
    )
    return archive, data


def test_import_is_allowlisted_and_idempotent(small_archive, tmp_path):
    archive, data = small_archive
    destination = tmp_path / "assets"
    assert reference_assets.import_reference(archive, destination) == 1
    assert reference_assets.import_reference(archive, destination) == 1
    assert sorted(path.name for path in destination.iterdir()) == ["test.webp"]
    assert (destination / "test.webp").read_bytes() == data
    assert not (tmp_path.parent / "not-imported.txt").exists()


def test_import_preserves_different_user_bytes(small_archive, tmp_path):
    archive, _ = small_archive
    destination = tmp_path / "assets"
    destination.mkdir()
    asset = destination / "test.webp"
    asset.write_bytes(b"user-change")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        reference_assets.import_reference(archive, destination)
    assert asset.read_bytes() == b"user-change"


def test_import_rejects_wrong_archive_before_creating_destination(tmp_path):
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"not the supplied archive")
    destination = tmp_path / "assets"
    with pytest.raises(ValueError, match="input SHA256"):
        reference_assets.import_reference(archive, destination)
    assert not destination.exists()


def test_import_does_not_follow_destination_symlink(small_archive, tmp_path):
    archive, _ = small_archive
    actual = tmp_path / "user-folder"
    actual.mkdir()
    destination = tmp_path / "assets"
    try:
        destination.symlink_to(actual, target_is_directory=True)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows host does not grant symlink creation privileges")
        raise
    with pytest.raises(ValueError, match="symlink"):
        reference_assets.import_reference(archive, destination)
    assert not list(actual.iterdir())


def test_import_does_not_follow_existing_asset_symlink(small_archive, tmp_path):
    archive, data = small_archive
    destination = tmp_path / "assets"
    destination.mkdir()
    actual = tmp_path / "user.webp"
    actual.write_bytes(data)
    try:
        (destination / "test.webp").symlink_to(actual)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows host does not grant symlink creation privileges")
        raise
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        reference_assets.import_reference(archive, destination)
    assert actual.read_bytes() == data
    assert (destination / "test.webp").is_symlink()


def test_import_integrity_failure_writes_nothing(small_archive, tmp_path):
    archive, _ = small_archive
    reference_assets.REFERENCE_CATALOG["files"]["bad.webp"] = {
        "archive": "fixture",
        "member": "implementation_reference/assets/test.webp",
        "sha256": "0" * 64,
    }
    destination = tmp_path / "assets"
    with pytest.raises(ValueError, match="integrity verification"):
        reference_assets.import_reference(archive, destination)
    assert not destination.exists()


def test_import_selects_only_matching_reviewed_archive(small_archive, tmp_path):
    first, first_data = small_archive
    second = tmp_path / "expansion.zip"
    second_data = b"second-reviewed-private-asset"
    with zipfile.ZipFile(second, "w") as source:
        source.writestr("expansion/image.webp", second_data)
    reference_assets.REFERENCE_CATALOG["archives"]["second"] = hashlib.sha256(
        second.read_bytes()
    ).hexdigest()
    reference_assets.REFERENCE_CATALOG["files"]["second.webp"] = {
        "archive": "second",
        "member": "expansion/image.webp",
        "sha256": hashlib.sha256(second_data).hexdigest(),
    }
    destination = tmp_path / "assets"
    assert reference_assets.import_reference(first, destination) == 1
    assert list(path.name for path in destination.iterdir()) == ["test.webp"]
    assert reference_assets.import_reference(second, destination) == 1
    assert (destination / "test.webp").read_bytes() == first_data
    assert (destination / "second.webp").read_bytes() == second_data
    assert reference_assets.import_reference(first, destination) == 1


def test_expansion_metadata_tracks_instance_limits_without_answers():
    levels = (11, 12, 13, 14, 18, 21, 22, 24, 29, 30, 31, 33, 34, 39, 46)
    for level in levels:
        env = gym.make(f"visual_tasks@neal_{level:02d}", seed=7)
        reference = env.metadata.others["reference"]
        assert reference["level"] == level
        assert reference["fidelity"] == "captured_instance"
        assert reference["completion_recorded"] is True
        assert reference["evidence_attempt"].startswith("attempt_")
        assert (
            reference["supplement_sha256"]
            == reference_assets.REFERENCE_CATALOG["archives"]["expansion"]
        )
        if level in (29, 46):
            # These pinned source modules use fixed images, not a missing random generator.
            assert (
                reference["rules_status"] == "observed_instance_with_source_derived_selection_rule"
            )
            assert reference["selection_rule"]["basis"] == "source_derived"
            assert reference["selection_rule"]["source_modules"] == {29: [1117], 46: [1077]}[level]
            assert "separate_asset_load_not_same_attempt_identity" in reference["limitations"]
            assert (
                "original_site_selection_boundary_replay_not_performed" in reference["limitations"]
            )
        elif level == 14:
            assert (
                reference["rules_status"]
                == "observed_instance_with_source_derived_checkbox_feedback"
            )
            assert reference["checkbox_rule"]["source_modules"] == [478, 508, 1096]
            assert reference["checkbox_rule"]["correct_feedback_ms"] == 700
            assert reference["checkbox_rule"]["completion_ms"] == 1600
            assert reference["checkbox_rule"]["wrong_feedback_ms"] == 800
            assert "original_site_feedback_replay_not_performed" in reference["limitations"]
        else:
            assert "refresh_same_instance_only" in reference["limitations"]
            assert "original_random_generator_not_recovered" in reference["limitations"]
        assert not {"answer", "targets", "selected_regions", "solution"} & reference.keys()
        assert env.unwrapped._page is None


def test_camera_reference_does_not_claim_facial_recognition():
    reference = LOCAL_TASKS["neal_39"]["reference"]
    assert reference["rules_status"] == "captured_unavailable_device_path"
    assert reference["completion_recorded"] is True
    assert reference["camera_flow_tested"] is False
    assert reference["facial_expression_detection_verified"] is False
    assert reference["human_identity_verified"] is False
    assert "only_observed_camera_unavailable_path" in reference["limitations"]


@pytest.fixture
def small_rhythm_module(tmp_path, monkeypatch):
    module = tmp_path / "module_1070.js"
    # Executing this would throw. The importer may only parse the literal array.
    source = (
        b'throw new Error("must not execute"); var c = [{key: "down", time: 1.55},'
        b'{key: "left", time: 2}], l = n(293);'
    )
    module.write_bytes(source)
    data = b'[{"key":"down","time":1.55},{"key":"left","time":2}]\n'
    monkeypatch.setattr(
        reference_assets,
        "REFERENCE_CATALOG",
        {
            "archives": {},
            "files": {},
            "derived_files": {
                "level47_chart.json": {
                    "source_format": "reviewed_analysis_module_1070",
                    "source_sha256": hashlib.sha256(source).hexdigest(),
                    "derivation": "literal_c_array_json_ascii_compact_with_final_lf",
                    "derived_sha256": hashlib.sha256(data).hexdigest(),
                }
            },
        },
    )
    return module, data


def test_raw_module_derives_only_chart_without_executing_or_copying_source(
    small_rhythm_module, tmp_path
):
    module, expected = small_rhythm_module
    destination = tmp_path / "assets"
    assert reference_assets.import_reference(module, destination) == 1
    assert reference_assets.import_reference(module, destination) == 1
    assert [path.name for path in destination.iterdir()] == ["level47_chart.json"]
    assert (destination / "level47_chart.json").read_bytes() == expected
    assert b"throw" not in expected and b"n(293)" not in expected


def test_raw_module_hash_is_checked_before_derivation(small_rhythm_module, tmp_path):
    module, _ = small_rhythm_module
    module.write_bytes(module.read_bytes() + b"\nchanged source")
    destination = tmp_path / "assets"
    with pytest.raises(ValueError, match="input SHA256"):
        reference_assets.import_reference(module, destination)
    assert not destination.exists()


def test_raw_module_derivation_hash_mismatch_writes_nothing(small_rhythm_module, tmp_path):
    module, _ = small_rhythm_module
    reference_assets.REFERENCE_CATALOG["derived_files"]["level47_chart.json"]["derived_sha256"] = (
        "0" * 64
    )
    destination = tmp_path / "assets"
    with pytest.raises(ValueError, match="Derived rhythm chart failed integrity"):
        reference_assets.import_reference(module, destination)
    assert not destination.exists()


def test_raw_module_refuses_to_overwrite_modified_chart(small_rhythm_module, tmp_path):
    module, _ = small_rhythm_module
    destination = tmp_path / "assets"
    destination.mkdir()
    target = destination / "level47_chart.json"
    target.write_bytes(b"user-edited chart")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        reference_assets.import_reference(module, destination)
    assert target.read_bytes() == b"user-edited chart"


def test_reviewed_original_chart_has_all_notes_and_chords(tmp_path):
    source = os.environ.get("NEAL_RHYTHM_MODULE")
    if source is None:
        pytest.skip("Private reviewed module path must be supplied via NEAL_RHYTHM_MODULE")
    destination = tmp_path / "assets"
    assert reference_assets.import_reference(Path(source), destination) == 1
    data = (destination / "level47_chart.json").read_bytes()
    chart = json.loads(data)
    assert len(chart) == 336
    assert sum(count > 1 for count in Counter(note["time"] for note in chart).values()) == 49
    assert chart[0] == {"key": "down", "time": 1.55}
    assert chart[-4:] == [
        {"key": "up", "time": 101.05},
        {"key": "down", "time": 101.05},
        {"key": "left", "time": 101.05},
        {"key": "right", "time": 101.05},
    ]
    assert len(data) == 9335
    assert hashlib.sha256(data).hexdigest() == (
        "e3ee00ae0e7c3c1ec2250cf41ea8634894416b6d082642470e602d5273b34316"
    )
    assert (
        reference_assets.REFERENCE_HASHES["level47_chart.json"] == hashlib.sha256(data).hexdigest()
    )
    assert all(set(note) == {"key", "time"} for note in chart)
    assert {note["key"] for note in chart} == {"up", "down", "left", "right"}
    # Preserve the source's non-monotonic segment instead of silently sorting it.
    assert chart[251:254] == [
        {"key": "down", "time": 77.425},
        {"key": "right", "time": 77.85},
        {"key": "left", "time": 77.075},
    ]
    assert not list(destination.glob("*.js"))
