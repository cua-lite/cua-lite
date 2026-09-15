"""Private asset integrity and implemented reference-task registration contracts."""

from __future__ import annotations

import hashlib
import zipfile

import pytest

from examples.not_a_robot import reference_assets, registration  # noqa: F401
from examples.not_a_robot.local_tasks import ASSETS, LOCAL_TASKS
from lite import gym


def test_first_ten_have_explicit_reference_evidence_and_limits():
    tasks = [LOCAL_TASKS[f"neal_{level:02d}"] for level in range(1, 11)]
    assert len(LOCAL_TASKS) == 33
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
    assert len(reference_paths) == 54
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
    with pytest.raises(ValueError, match="ZIP SHA256"):
        reference_assets.import_reference(archive, destination)
    assert not destination.exists()


def test_import_does_not_follow_destination_symlink(small_archive, tmp_path):
    archive, _ = small_archive
    actual = tmp_path / "user-folder"
    actual.mkdir()
    destination = tmp_path / "assets"
    destination.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        reference_assets.import_reference(archive, destination)
    assert not list(actual.iterdir())


def test_import_does_not_follow_existing_asset_symlink(small_archive, tmp_path):
    archive, data = small_archive
    destination = tmp_path / "assets"
    destination.mkdir()
    actual = tmp_path / "user.webp"
    actual.write_bytes(data)
    (destination / "test.webp").symlink_to(actual)
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
