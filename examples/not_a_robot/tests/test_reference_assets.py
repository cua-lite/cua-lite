"""Private asset integrity and first-ten task registration contracts."""

from __future__ import annotations

import hashlib
import zipfile

import pytest

from examples.not_a_robot import reference_assets, registration  # noqa: F401
from examples.not_a_robot.local_tasks import ASSETS, LOCAL_TASKS
from lite import gym


def test_first_ten_have_explicit_reference_evidence_and_limits():
    tasks = [LOCAL_TASKS[f"neal_{level:02d}"] for level in range(1, 11)]
    assert len(LOCAL_TASKS) == 18
    for level, task in enumerate(tasks, 1):
        env = gym.make(f"visual_tasks@{task['id']}", seed=7)
        metadata = env.metadata.others
        reference = metadata["reference"]
        assert metadata["source"] == "neal_reference_reconstruction"
        assert metadata["seed_scope"] == "local_dynamics_not_original_challenge_seed"
        assert reference["level"] == level
        assert reference["fidelity"] == "captured_instance"
        assert reference["completion_recorded"] is (level != 6)
        assert reference["limitations"]
        assert reference["supplement_sha256"] == reference_assets.SUPPLEMENT_SHA256
        assert not {"answer", "targets", "selected_regions", "solution"} & reference.keys()
        assert env.unwrapped._page is None


def test_reference_asset_http_allowlist_does_not_include_capture_logs():
    reference_paths = {path for path in ASSETS if path.startswith("/reference_assets/")}
    assert reference_paths == {
        f"/reference_assets/{name}" for name in reference_assets.REFERENCE_HASHES
    }
    assert len(reference_paths) == 15
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
        reference_assets, "SUPPLEMENT_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        reference_assets, "REFERENCE_HASHES", {"test.webp": hashlib.sha256(data).hexdigest()}
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
