from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import lite.gym.utils.backend.freshness as freshness
from lite.gym.errors import CapacityExhausted, EnvDepsMissingError
from lite.gym.utils.backend.freshness import (
    ContainerImage,
    UnknownImageFreshnessProvider,
    _hash_sources,
    image_for,
    supported_env_ids,
)
from lite.gym.utils.config import manifest
from lite.gym.utils.config.manifest import baked_asset_lock_sources, load_asset_lock
from lite.utils.path import project_root


def _write_lock(env_dir, revision: str) -> None:
    data = env_dir / "data"
    data.mkdir(parents=True)
    (data / "assets.lock.yaml").write_text(textwrap.dedent(f"""
        assets:
          repo: cua-lite/sample-assets
          revision: {revision}
          components:
            runtime_only:
              path: data/tasks.json
              baked: false
            baked_bundle:
              path: artifacts/bundle.tar.zst
              baked: true
    """))


def test_asset_lock_requires_immutable_revision(tmp_path):
    _write_lock(tmp_path, "main")
    with pytest.raises(ValueError, match="immutable 40-hex"):
        load_asset_lock(tmp_path)


def test_manifest_module_is_config_owner() -> None:
    repo = Path(__file__).resolve().parents[4]
    assert Path(manifest.__file__).resolve() == (
        repo / "lite" / "gym" / "utils" / "config" / "manifest.py"
    )
    assert manifest._REPO_ROOT == project_root()


def test_asset_lock_exposes_baked_lock_as_image_source(tmp_path, monkeypatch):
    env_dir = tmp_path / "env"
    monkeypatch.setattr(manifest, "_REPO_ROOT", tmp_path)
    rev = "0123456789abcdef0123456789abcdef01234567"
    _write_lock(env_dir, rev)

    pin = load_asset_lock(env_dir)
    assert pin.repo == "cua-lite/sample-assets"
    assert pin.component_path("baked_bundle") == "artifacts/bundle.tar.zst"
    assert json.loads(pin.component_identity("runtime_only")) == {
        "component": "runtime_only",
        "path": "data/tasks.json",
        "repo": "cua-lite/sample-assets",
        "revision": rev,
    }
    assert baked_asset_lock_sources(env_dir) == ("env/data/assets.lock.yaml",)

    lock = env_dir / "data" / "assets.lock.yaml"
    lock.write_text(lock.read_text().replace("baked: true", "baked: false"))
    assert baked_asset_lock_sources(env_dir) == ()


def test_cuaworld_image_hashes_asset_lock_once(tmp_path, monkeypatch):
    import lite.gym.envs.lite.cuaworld.src.image_spec as image_spec
    import lite.gym.utils.backend.freshness as freshness

    env_dir = tmp_path / "lite/gym/envs/lite/cuaworld"
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    (recipe / "Dockerfile").write_text("FROM scratch\n")
    monkeypatch.setattr(manifest, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(freshness, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(image_spec, "CUAWORLD_DIR", env_dir)
    monkeypatch.setattr(image_spec, "_CUAWORLD_IMAGE_SOURCES", ("recipe",))
    monkeypatch.setitem(image_spec._REGISTERED_UPSTREAM_ENVS, "test", "test_env")

    first_revision = "0123456789abcdef0123456789abcdef01234567"
    _write_lock(env_dir, first_revision)
    first = freshness.image_for("lite.cuaworld.test")
    assert first.extra_hash_inputs == ("software=test", "upstream_env=test_env")
    assert first.sources.count(
        "lite/gym/envs/lite/cuaworld/data/assets.lock.yaml"
    ) == 1
    first_hash = first.src_hash()

    second_revision = "1123456789abcdef0123456789abcdef01234567"
    lock = env_dir / "data" / "assets.lock.yaml"
    lock.write_text(lock.read_text().replace(first_revision, second_revision))
    assert freshness.image_for("lite.cuaworld.test").src_hash() != first_hash


@pytest.mark.parametrize("env_id", supported_env_ids())
def test_image_freshness_dispatches_env_local_specs(env_id, monkeypatch, tmp_path):
    if env_id == "osworld_2":
        monkeypatch.setenv("OSWORLD_V2_SRC", str(tmp_path / "missing"))
    image = image_for(env_id)
    assert isinstance(image, ContainerImage)
    assert image.tag
    assert image.sources
    assert image.install.startswith("uv run --no-sync bash ")
    assert image.see.endswith("README.md")
    assert image.src_hash()


def test_all_env_local_image_specs_are_discoverable():
    discovered = freshness._discover_provider_paths()
    assert set(discovered) == {
        "androidlab",
        "androidworld",
        "lite.cuagym",
        "lite.cuaworld",
        "lite.demo",
        "lite.osworld",
        "mobilegym",
        "mobileworld",
        "online_mind2web",
        "osworld",
        "osworld_2",
        "waa",
        "webgym",
        "webharbor.webvoyager",
    }
    for env_id, provider in discovered.items():
        assert freshness._provider_path(env_id) == provider
    assert freshness._provider_path("lite.cuaworld.pymol").endswith(
        "lite.cuaworld.src.image_spec:image_for"
    )


def test_provider_discovery_prunes_skipped_segments(tmp_path, monkeypatch):
    """``.cache``/``_vendor``/``__pycache__`` are pruned during the descent, not
    rejected after it: ``lite/gym/envs`` holds a very large ``.cache`` tree, and
    visiting it makes every discovery slow."""
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample" / "image_spec.py").write_text("")
    for skipped in freshness._SKIP_PROVIDER_SEGMENTS:
        buried = tmp_path / "sample" / skipped / "nested"
        buried.mkdir(parents=True)
        (buried / "image_spec.py").write_text("")
    monkeypatch.setattr(freshness, "_ENVS_ROOT", tmp_path)

    assert set(freshness._discover_provider_paths()) == {"sample"}
    assert [p.parent.name for p in freshness._provider_files()] == ["sample"]


def test_unknown_image_freshness_provider_is_specific():
    with pytest.raises(UnknownImageFreshnessProvider):
        freshness.image_for("not.a.real.env")


def test_unknown_image_freshness_provider_does_not_import_unrelated_providers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(freshness, "_ENVS_ROOT", tmp_path)
    monkeypatch.setattr(
        freshness,
        "supported_env_ids",
        lambda: pytest.fail("unknown-provider path loaded unrelated providers"),
    )

    with pytest.raises(UnknownImageFreshnessProvider):
        freshness.image_for("not.a.real.env")


def test_conflicting_direct_and_src_providers_fail_closed(tmp_path, monkeypatch):
    env_dir = tmp_path / "sample"
    (env_dir / "src").mkdir(parents=True)
    (env_dir / "image_spec.py").write_text("")
    (env_dir / "src" / "image_spec.py").write_text("")
    monkeypatch.setattr(freshness, "_ENVS_ROOT", tmp_path)

    with pytest.raises(ValueError, match="conflicting image freshness providers"):
        freshness._provider_path("sample.variant")


def test_provider_errors_fail_closed(monkeypatch):
    def rejecting_provider(_env_id):
        raise KeyError("broken provider")

    monkeypatch.setattr(freshness, "_provider_path", lambda _env_id: "provider")
    monkeypatch.setattr(freshness, "_load_provider", lambda _path: rejecting_provider)

    with pytest.raises(ValueError, match="rejected env_id"):
        freshness.image_for("sample.env")


def test_provider_return_type_is_validated(monkeypatch):
    monkeypatch.setattr(freshness, "_provider_path", lambda _env_id: "provider")
    monkeypatch.setattr(
        freshness,
        "_load_provider",
        lambda _path: lambda _env_id: object(),
    )

    with pytest.raises(TypeError, match="expected ContainerImage"):
        freshness.image_for("sample.env")


def test_supported_env_ids_rejects_string_return(monkeypatch):
    class BadProviderModule:
        @staticmethod
        def supported_env_ids():
            return "bad.env"

    monkeypatch.setattr(
        freshness,
        "_discover_provider_paths",
        lambda: {"bad": "bad.module:image_for"},
    )
    monkeypatch.setattr(
        freshness.importlib,
        "import_module",
        lambda _module_name: BadProviderModule,
    )

    with pytest.raises(TypeError, match="iterable of non-empty strings"):
        freshness.supported_env_ids()


def test_image_freshness_cli_hash_path():
    result = subprocess.run(
        [sys.executable, "-m", "lite.gym.utils.backend.freshness", "hash", "lite.demo"],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = result.stdout.strip()
    assert digest == image_for("lite.demo").src_hash()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_image_freshness_cli_hash_path_with_dynamic_source(tmp_path, monkeypatch):
    src = tmp_path / "OSWorld-V2"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\nname='osworld-v2-test'\n")
    (src / "package.py").write_text("VALUE = 1\n")
    monkeypatch.setenv("OSWORLD_V2_SRC", str(src))
    result = subprocess.run(
        [sys.executable, "-m", "lite.gym.utils.backend.freshness", "hash", "osworld_2"],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = result.stdout.strip()
    assert digest == image_for("osworld_2").src_hash()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_image_freshness_cli_hash_path_with_asset_manifest():
    result = subprocess.run(
        [sys.executable, "-m", "lite.gym.utils.backend.freshness", "hash", "lite.cuagym"],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = result.stdout.strip()
    assert digest == image_for("lite.cuagym").src_hash()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_osworld2_image_hash_tracks_selected_source_tree(tmp_path, monkeypatch):
    import lite.gym.envs.osworld_2.image_spec as image_spec

    src = tmp_path / "OSWorld-V2"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\nname='osworld-v2-test'\n")
    (src / "package.py").write_text("VALUE = 1\n")
    ignored = src / "docs"
    ignored.mkdir()
    (ignored / "guide.md").write_text("ignored\n")
    monkeypatch.setenv("OSWORLD_V2_SRC", str(src))

    image = image_spec.image_for("osworld_2")
    assert "lite/gym/envs/osworld_2/image_spec.py" in image.sources
    assert "lite/gym/envs/osworld_2/scripts/install.sh" in image.sources
    assert image.extra_hash_inputs[0].startswith("osworld-v2-source=")
    first_hash = image.src_hash()

    (ignored / "guide.md").write_text("ignored change\n")
    assert image_spec.image_for("osworld_2").src_hash() == first_hash
    (src / "package.py").write_text("VALUE = 2\n")
    assert image_spec.image_for("osworld_2").src_hash() != first_hash


def test_osworld2_image_identity_is_explicit_when_source_absent(tmp_path, monkeypatch):
    import lite.gym.envs.osworld_2.image_spec as image_spec

    monkeypatch.setenv("OSWORLD_V2_SRC", str(tmp_path / "missing"))
    image = image_spec.image_for("osworld_2")
    assert image.extra_hash_inputs == ("osworld-v2-source=absent",)


def test_cuaworld_image_hash_excludes_host_only_runtime_modules(monkeypatch):
    import lite.gym.envs.lite.cuaworld.src.image_spec as image_spec

    monkeypatch.delenv("LITE_CUAWORLD_MATERIALS_REPO", raising=False)

    image = image_spec.image_for("lite.cuaworld.gcompris")
    assert "lite/gym/envs/lite/cuaworld/docker" in image.sources
    assert "lite/gym/envs/lite/cuaworld/scripts/install.sh" in image.sources
    assert "lite/gym/envs/lite/cuaworld/src/image_spec.py" in image.sources
    assert "lite/gym/envs/lite/cuaworld/src/adapter.py" not in image.sources


def test_component_identity_is_unambiguous(tmp_path):
    rev = "0123456789abcdef0123456789abcdef01234567"
    _write_lock(tmp_path, rev)
    pin = load_asset_lock(tmp_path)
    assert pin.component_identity("a:b", path="c") != (
        pin.component_identity("a", path="b:c")
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", 123, "missing path"),
        ("baked", "false", "baked must be boolean"),
    ],
)
def test_asset_lock_rejects_ambiguous_component_types(tmp_path, field, value, message):
    rev = "0123456789abcdef0123456789abcdef01234567"
    _write_lock(tmp_path, rev)
    path = tmp_path / "data" / "assets.lock.yaml"
    raw = path.read_text()
    if field == "path":
        raw = raw.replace("path: artifacts/bundle.tar.zst", f"path: {value}")
    else:
        raw = raw.replace("baked: true", f'baked: "{value}"')
    path.write_text(raw)
    with pytest.raises(ValueError, match=message):
        load_asset_lock(tmp_path)


def test_container_image_extra_hash_inputs_preserve_plain_hash():
    sources = ("lite/gym/utils/config/manifest.py",)
    plain = ContainerImage(
        "cua-lite/test:latest",
        sources,
        "install",
        "README.md",
    )
    assert plain.src_hash() == _hash_sources(sources, ())

    with_extra = ContainerImage(
        "cua-lite/test:latest",
        sources,
        "install",
        "README.md",
        extra_hash_inputs=("cua-lite/sample-assets@0123456789abcdef0123456789abcdef01234567",),
    )
    assert with_extra.src_hash() != plain.src_hash()


def test_container_image_extra_hash_inputs_are_unambiguous():
    sources = ("lite/gym/utils/config/manifest.py",)
    left = ContainerImage(
        "cua-lite/test:latest",
        sources,
        "install",
        "README.md",
        extra_hash_inputs=("a|b", "c"),
    )
    right = ContainerImage(
        "cua-lite/test:latest",
        sources,
        "install",
        "README.md",
        extra_hash_inputs=("a", "b|c"),
    )
    assert left.src_hash() != right.src_hash()


def test_container_image_extra_hash_inputs_must_be_strings():
    with pytest.raises(TypeError, match="extra_hash_inputs"):
        ContainerImage(
            "cua-lite/test:latest",
            ("lite/gym/utils/config/manifest.py",),
            "install",
            "README.md",
            extra_hash_inputs=(object(),),  # type: ignore[arg-type]
        )


def test_container_image_missing_image_stays_env_deps_missing(monkeypatch):
    image = ContainerImage(
        "cua-lite/test:latest",
        ("lite/gym/utils/config/manifest.py",),
        "install",
        "README.md",
    )

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            args[0],
            stderr="Error response from daemon: No such image: cua-lite/test:latest",
        )

    monkeypatch.setattr(freshness.subprocess, "run", fake_run)
    with pytest.raises(EnvDepsMissingError, match="STALE"):
        image.ensure_runnable()


def test_container_image_transient_inspect_failure_is_retryable(monkeypatch):
    image = ContainerImage(
        "cua-lite/test:latest",
        ("lite/gym/utils/config/manifest.py",),
        "install",
        "README.md",
    )

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            args[0],
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
        )

    monkeypatch.setattr(freshness.subprocess, "run", fake_run)
    with pytest.raises(CapacityExhausted, match="docker image inspect failed") as raised:
        image.ensure_runnable()
    assert raised.value.retryable is True


def test_container_image_inspect_timeout_is_retryable(monkeypatch):
    image = ContainerImage(
        "cua-lite/test:latest",
        ("lite/gym/utils/config/manifest.py",),
        "install",
        "README.md",
    )
    seen: dict[str, float | None] = {}

    def fake_run(*args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(freshness.subprocess, "run", fake_run)
    with pytest.raises(CapacityExhausted, match="docker image inspect failed") as raised:
        image.ensure_runnable()
    assert seen["timeout"] == freshness._INSPECT_TIMEOUT_S
    assert raised.value.retry_after_s == 2.0


def test_container_image_invalid_tag_is_terminal_env_deps(monkeypatch):
    image = ContainerImage(
        "bad tag",
        ("lite/gym/utils/config/manifest.py",),
        "install",
        "README.md",
    )

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            args[0],
            stderr="invalid reference format",
        )

    monkeypatch.setattr(freshness.subprocess, "run", fake_run)
    with pytest.raises(EnvDepsMissingError, match="cannot be inspected"):
        image.ensure_runnable()


def test_repo_asset_locks_cover_hf_external_assets():
    locks = {
        "lite/gym/envs/lite/cuagym": "cua-lite/lite.cuagym-assets",
        "lite/gym/envs/lite/cuaworld": "cua-lite/lite.cuaworld-assets",
    }
    for env_dir, repo in locks.items():
        pin = load_asset_lock(env_dir)
        assert pin.repo == repo
        assert baked_asset_lock_sources(env_dir) == (
            f"{env_dir}/data/assets.lock.yaml",
        )
