"""Static contract tests for the BrowserGym benchmark installer."""

from __future__ import annotations

import gzip
import io
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[4] / "lite/gym/envs/browsergym/scripts/install.sh"


def _install_script() -> str:
    return INSTALL_SH.read_text()


def _shell_function(name: str) -> str:
    script = _install_script()
    start = script.index(f"{name}() {{")
    rest = script[start:]
    depth = 0
    lines = []
    for line in rest.splitlines():
        lines.append(line)
        depth += line.count("{")
        depth -= line.count("}")
        if depth == 0:
            return "\n".join(lines)
    raise AssertionError(f"could not extract shell function {name}")


def _resource_is_valid(resource: str, path: Path) -> bool:
    command = f'{_shell_function("resource_is_valid")}\nresource_is_valid "$1" "$2"'
    return (
        subprocess.run(
            ["bash", "-c", command, "bash", resource, str(path)],
            check=False,
        ).returncode
        == 0
    )


def _openstreetmap_dir_is_valid(path: Path) -> bool:
    command = f'{_shell_function("openstreetmap_dir_is_valid")}\nopenstreetmap_dir_is_valid "$1"'
    return (
        subprocess.run(
            ["bash", "-c", command, "bash", str(path)],
            check=False,
        ).returncode
        == 0
    )


def _write_fake_docker(path: Path, *, existing_images: set[str] | None = None) -> Path:
    existing_images = existing_images or set()
    log = path / "docker.log"
    docker = path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {log}\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n"
        f"  case \"$3\" in {'|'.join(existing_images) or '__none__'}) exit 0 ;; esac\n"
        "  exit 1\n"
        "fi\n"
        "if [ \"$1\" = pull ]; then exit 0; fi\n"
        "exit 1\n"
    )
    docker.chmod(0o755)
    return log


def _docker_layer_tar_bytes() -> bytes:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        content = b"layer file"
        info = tarfile.TarInfo("layer-file")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return data.getvalue()


def _write_docker_save_tar(
    path: Path,
    *,
    repo_tag: str = "shopping_final_0712:latest",
    layer_bytes: bytes | None = None,
    mode: str = "w",
) -> None:
    if layer_bytes is None:
        layer_bytes = _docker_layer_tar_bytes()
    manifest = (
        f'[{{"Config":"config.json","RepoTags":["{repo_tag}"],"Layers":["layer.tar"]}}]'
    ).encode()
    with tarfile.open(path, mode) as archive:
        for name, content in (
            ("manifest.json", manifest),
            ("config.json", b"{}"),
            ("layer.tar", layer_bytes),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _write_tar_gz(path: Path, entries: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _write_zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def _write_gzip_bytes(path: Path, data: bytes) -> None:
    with gzip.open(path, "wb") as fh:
        fh.write(data)


def _run_shell(
    command: str,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    if env is None:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"WEBARENA_IMAGES_MIRROR_DIR", "WEBARENA_IMAGES_MIRROR_BASE_URL"}
        }
    return subprocess.run(
        ["bash", "-c", command, "bash", *args],
        check=False,
        env=env,
    )


def test_webarena_resources_have_official_multi_mirror_sources():
    script = _install_script()

    assert "WEBARENA_IMAGES_MIRROR_DIR" in script
    assert "WEBARENA_IMAGES_MIRROR_BASE_URL" in script
    assert "pwd -P" in script
    assert 'readlink -f -- "$mirror_file"' in script
    assert 'resource_is_valid "$resource" "$output"' in script
    assert 'resource_is_valid "$resource" "$partial"' in script
    assert 'partial_is_resumable "$partial"' in script
    assert "path.read_bytes()" not in script
    assert "fh.read(2048)" in script
    assert 'ln -f "$mirror_real" "$output"' in script
    assert 'ln -sf "$mirror_real" "$output"' in script
    assert "gdrive:1gxXalk9O0p9eu1YkIJcmZta1nvvyAJpA" in script
    assert "gdrive:1See0ZhJRw0WTTL9y8hFlgaduwPZ_nGfd" in script
    assert "gdrive:17Qpp1iu_mPqzgO_73Z9BnFjHrzmX9DGf" in script
    assert "gdrive:19W8qM0DPyRvWCLyQe0qtnCWAHGruolMR" in script
    assert "gdrive:1Um4QLxi_bGv5bP6kt83Ke0lNjuV9Tm0P" in script
    assert "gdrive:1m79lp84yXfqdTBHr6IS7_1KkL4sDSemR" in script
    assert "https://archive.org/download/postmill-populated-exposed-withimg/$resource" in script
    assert "http://metis.lti.cs.cmu.edu/webarena-images/$resource" in script
    assert "pull_docker_image_if_missing \"jykoh/classifieds:latest\"" in script
    assert "pull_docker_image_if_missing \"mysql:8.1\"" in script


def test_webarena_installer_does_not_accept_empty_or_missing_resources():
    script = _install_script()

    assert 'resource_is_valid "$tar" "$IMAGES_DIR/$tar" || dl+=("$tar")' in script
    assert 'resource_is_valid "$zim" "$IMAGES_DIR/$zim" || dl+=("$zim")' in script
    assert 'resource_is_valid "$f" "$IMAGES_DIR/$f" || dl+=("$f")' in script
    assert 'resource_is_valid "$resource" "$zip"' in script
    assert 'tarfile.open(path, "r:*")' in script
    assert "expected_tags[resource] not in tags" in script
    assert "archive.extractfile(member_name)" in script
    assert 'tarfile.open(path, "r:gz")' in script
    assert "openstreetmap-website/docker-compose.yml" in script
    assert "classifieds_docker_compose/docker-compose.yml" in script
    assert "expected_size = 95199730590" in script
    assert 'docker image inspect "${tar%.tar}"' in script
    assert "docker load did not create expected image openstreetmap-website-db" in script
    assert 'if [ -e "$partial" ]; then' in script
    assert "ERROR failed to download $resource from all configured WebArena sources" in script
    assert "ERROR failed to download VisualWebArena Classifieds zip" in script
    assert "ERROR openstreetmap-website-db image missing and no valid $db_archive present" in script
    assert "classifieds_dir_is_valid" in script
    assert "openstreetmap_dir_is_valid" in script
    assert "WARN removing invalid OpenStreetMap source extract" in script
    assert "download_template_file" in script
    assert 'template_is_valid "$resource" "$partial"' in script
    assert "ERROR failed to download valid OpenStreetMap template" in script
    assert 'if [ "$failed" = "1" ]; then' in script


def test_osm_template_validation_rejects_html_error_pages(tmp_path: Path):
    command = f'{_shell_function("template_is_valid")}\ntemplate_is_valid "$1" "$2"'

    docker_compose = tmp_path / "docker-compose.yml"
    docker_compose.write_text(
        "services:\n"
        "  web:\n"
        "    ports:\n"
        '      - "MAP_PORT:3000"\n'
        "  db:\n"
        "    environment:\n"
        "      POSTGRES_DB: openstreetmap\n"
    )
    html = tmp_path / "docker-compose-html.yml"
    html.write_text("<html>403 Forbidden</html>")

    assert (
        subprocess.run(
            ["bash", "-c", command, "bash", "docker-compose.yml", str(docker_compose)],
            check=False,
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            ["bash", "-c", command, "bash", "docker-compose.yml", str(html)],
            check=False,
        ).returncode
        != 0
    )


def test_docker_tar_validation_reads_manifest_referenced_payloads(tmp_path: Path):
    valid = tmp_path / "shopping_final_0712.tar"
    _write_docker_save_tar(
        valid,
        repo_tag="shopping_final_0712:latest",
    )

    html = tmp_path / "html.tar"
    html.write_text("<html>403 Forbidden</html>")

    wrong_tag = tmp_path / "wrong-tag.tar"
    _write_docker_save_tar(wrong_tag, repo_tag="wrong:latest")

    bad_layer = tmp_path / "bad-layer.tar"
    _write_docker_save_tar(bad_layer, layer_bytes=b"not a tar layer")

    missing_layer = tmp_path / "missing_layer.tar"
    with tarfile.open(missing_layer, "w") as archive:
        content = b'[{"Config":"missing.json","Layers":["missing.tar"]}]'
        info = tarfile.TarInfo("manifest.json")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    truncated = tmp_path / "truncated.tar"
    data = valid.read_bytes()
    truncated.write_bytes(data[:4096])

    assert _resource_is_valid("shopping_final_0712.tar", valid)
    assert not _resource_is_valid("shopping_final_0712.tar", html)
    assert not _resource_is_valid("shopping_final_0712.tar", wrong_tag)
    assert not _resource_is_valid("shopping_final_0712.tar", bad_layer)
    assert not _resource_is_valid("shopping_final_0712.tar", missing_layer)
    assert not _resource_is_valid("shopping_final_0712.tar", truncated)


def test_osm_archive_validation_checks_docker_tags_and_source_layout(tmp_path: Path):
    valid_db = tmp_path / "openstreetmap-website-db.tar.gz"
    _write_docker_save_tar(
        valid_db,
        repo_tag="openstreetmap-website-db:latest",
        mode="w:gz",
    )
    wrong_db = tmp_path / "wrong-openstreetmap-website-db.tar.gz"
    _write_docker_save_tar(wrong_db, repo_tag="wrong:latest", mode="w:gz")
    html_gzip = tmp_path / "html.tar.gz"
    _write_gzip_bytes(html_gzip, b"<html>403 Forbidden</html>")

    valid_source = tmp_path / "openstreetmap-website.tar.gz"
    _write_tar_gz(
        valid_source,
        {
            "openstreetmap-website/docker-compose.yml": b"services: {}\n",
            "openstreetmap-website/config/settings.yml": b"settings\n",
            "openstreetmap-website/vendor/assets/leaflet/leaflet.osm.js": b"L.OSM\n",
            "openstreetmap-website/app/assets/javascripts/index/directions/fossgis_osrm.js": (
                b"route\n"
            ),
        },
    )
    wrong_source = tmp_path / "wrong-source.tar.gz"
    _write_tar_gz(wrong_source, {"index.html": b"<html></html>"})

    assert _resource_is_valid("openstreetmap-website-db.tar.gz", valid_db)
    assert not _resource_is_valid("openstreetmap-website-db.tar.gz", wrong_db)
    assert not _resource_is_valid("openstreetmap-website-db.tar.gz", html_gzip)
    assert _resource_is_valid("openstreetmap-website.tar.gz", valid_source)
    assert not _resource_is_valid("openstreetmap-website.tar.gz", wrong_source)
    assert not _resource_is_valid("openstreetmap-website.tar.gz", html_gzip)


def test_osm_extracted_dir_validation_rejects_interrupted_extracts(tmp_path: Path):
    source_dir = tmp_path / "openstreetmap-website"
    required = [
        "docker-compose.yml",
        "config/settings.yml",
        "vendor/assets/leaflet/leaflet.osm.js",
        "app/assets/javascripts/index/directions/fossgis_osrm.js",
    ]
    for rel in required:
        path = source_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n")

    assert _openstreetmap_dir_is_valid(source_dir)

    (source_dir / "config/settings.yml").unlink()

    assert not _openstreetmap_dir_is_valid(source_dir)


def test_classifieds_zip_validation_checks_required_entries(tmp_path: Path):
    valid = tmp_path / "classifieds_docker_compose.zip"
    _write_zip(
        valid,
        {
            "classifieds_docker_compose/docker-compose.yml": b"services: {}\n",
            "classifieds_docker_compose/mysql/init_db.sh": b"#!/bin/sh\n",
            "classifieds_docker_compose/mysql/osclass_craigslist.sql": b"select 1;\n",
        },
    )
    missing_compose = tmp_path / "missing-compose.zip"
    _write_zip(
        missing_compose,
        {"classifieds_docker_compose/mysql/init_db.sh": b"#!/bin/sh\n"},
    )
    html = tmp_path / "html.zip"
    html.write_text("<html>403 Forbidden</html>")

    assert _resource_is_valid("classifieds_docker_compose.zip", valid)
    assert not _resource_is_valid("classifieds_docker_compose.zip", missing_compose)
    assert not _resource_is_valid("classifieds_docker_compose.zip", html)


def test_classifieds_image_preflight_pulls_only_missing_images(tmp_path: Path):
    log = _write_fake_docker(tmp_path, existing_images={"mysql:8.1"})
    command = (
        f"{_shell_function('pull_docker_image_if_missing')}\n"
        f"{_shell_function('install_classifieds_images')}\n"
        "install_classifieds_images"
    )
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}

    result = _run_shell(command, env=env)

    assert result.returncode == 0
    assert log.read_text().splitlines() == [
        "image inspect jykoh/classifieds:latest",
        "pull jykoh/classifieds:latest",
        "image inspect mysql:8.1",
    ]


def test_invalid_partial_is_removed_before_download_attempt(tmp_path: Path):
    command = (
        f"{_shell_function('resource_is_valid')}\n"
        f"{_shell_function('stage_local_resource')}\n"
        f"{_shell_function('partial_is_resumable')}\n"
        f"{_shell_function('download_resource_from_urls')}\n"
        'download_resource_from_urls "$1" "$2" "$3"'
    )
    output = tmp_path / "shopping_final_0712.tar"
    partial = tmp_path / "shopping_final_0712.tar.part"
    partial.write_text("<html>403 Forbidden</html>")

    result = _run_shell(
        command,
        "shopping_final_0712.tar",
        str(output),
        "file:///definitely/missing.tar",
    )

    assert result.returncode != 0
    assert not partial.exists()


def test_interrupted_binary_partial_is_kept_for_resume(tmp_path: Path):
    command = (
        f"{_shell_function('resource_is_valid')}\n"
        f"{_shell_function('stage_local_resource')}\n"
        f"{_shell_function('partial_is_resumable')}\n"
        f"{_shell_function('download_resource_from_urls')}\n"
        'download_resource_from_urls "$1" "$2" "$3"'
    )
    output = tmp_path / "shopping_final_0712.tar"
    partial = tmp_path / "shopping_final_0712.tar.part"
    partial.write_bytes(b"\x00\x01partial docker layer bytes")

    result = _run_shell(
        command,
        "shopping_final_0712.tar",
        str(output),
        "file:///definitely/missing.tar",
    )

    assert result.returncode != 0
    assert partial.read_bytes() == b"\x00\x01partial docker layer bytes"


def test_local_mirror_staging_resolves_relative_symlink_sources(tmp_path: Path):
    real_dir = tmp_path / "real"
    mirror_dir = tmp_path / "mirror"
    cache_dir = tmp_path / "cache"
    real_dir.mkdir()
    mirror_dir.mkdir()
    cache_dir.mkdir()

    resource = "shopping_final_0712.tar"
    real_resource = real_dir / resource
    _write_docker_save_tar(real_resource, repo_tag="shopping_final_0712:latest")
    (mirror_dir / resource).symlink_to(Path("..") / "real" / resource)

    output = cache_dir / resource
    command = (
        f"{_shell_function('resource_is_valid')}\n"
        f"{_shell_function('stage_local_resource')}\n"
        'stage_local_resource "$1" "$2"'
    )
    env = {
        **os.environ,
        "WEBARENA_IMAGES_MIRROR_DIR": str(mirror_dir),
    }

    result = _run_shell(command, resource, str(output), env=env)

    assert result.returncode == 0
    assert _resource_is_valid(resource, output)
    if output.is_symlink():
        assert Path(os.readlink(output)).is_absolute()
