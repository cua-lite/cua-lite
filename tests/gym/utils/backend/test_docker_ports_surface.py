"""Backend docker/ports helper ownership guards.

Run: uv run pytest tests/gym/utils/backend/test_docker_ports_surface.py
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import pytest
from active_sources import scan_active_sources


def test_backend_docker_and_ports_owner_modules() -> None:
    repo = Path(__file__).resolve().parents[4]
    import lite.gym.utils as gym_utils

    for old_name in ("docker", "port"):
        old_module = f"lite.gym.utils.{old_name}"
        assert not (repo / "lite" / "gym" / "utils" / f"{old_name}.py").exists()
        assert importlib.util.find_spec(old_module) is None
        assert not hasattr(gym_utils, old_name)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old_module)

    expected = {
        "docker": repo / "lite" / "gym" / "utils" / "backend" / "docker.py",
        "ports": repo / "lite" / "gym" / "utils" / "backend" / "ports.py",
    }
    for new_name, path in expected.items():
        spec = importlib.util.find_spec(f"lite.gym.utils.backend.{new_name}")
        assert spec is not None
        assert Path(spec.origin).resolve() == path

    assert importlib.util.find_spec("lite.gym.utils.backend.port") is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lite.gym.utils.backend.port")


def test_retired_flat_docker_port_paths_stay_out_of_active_sources() -> None:
    offenders = scan_active_sources(
        (
            r"\blite\.gym\.utils\.(docker|port)\b",
            r"from\s+lite\.gym\.utils\s+import\s+[^#\n]*(docker|port)\b",
            r"python\s+-m\s+lite\.gym\.utils\.(docker|port)\b",
            r"(^|/)lite/gym/utils/(docker|port)\.py\b",
            r"\blite\.gym\.utils\.backend\.port\b",
            r"(^|/)lite/gym/utils/backend/port\.py\b",
        ),
        exclude=("tests/gym/utils/backend/test_docker_ports_surface.py",),
    )

    assert not offenders, (
        "retired flat gym docker/port helpers and singular backend.port should "
        "not appear in active source text; use "
        "lite.gym.utils.backend.docker / lite.gym.utils.backend.ports:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("other_owner", [False, True])
def test_unbound_reservation_survives_slow_start_until_release(
    monkeypatch, tmp_path, other_owner,
):
    from lite.gym.errors import CapacityExhausted
    from lite.gym.utils.backend import ports

    monkeypatch.setattr(ports, "_LOCK_FILE", tmp_path / "ports.lock")
    monkeypatch.setattr(ports, "_RESERVATION_FILE", tmp_path / "ports.json")
    monkeypatch.setattr(ports, "_is_port_free", lambda _port: True)
    clock = [1000.0]
    monkeypatch.setattr(ports.time, "time", lambda: clock[0])
    monkeypatch.setattr(ports.time, "monotonic", lambda: clock[0])
    kwargs = dict(n=1, range_start=22000, range_end=22001)

    assert ports.allocate_ports(**kwargs) == [22000]
    if other_owner:
        reservations = ports._read_reservations()
        reservations[22000]["pid"] = os.getppid()
        ports._write_reservations(reservations)
    clock[0] += 3600
    with pytest.raises(CapacityExhausted):
        ports.allocate_ports(**kwargs)

    ports.release_ports(22000)
    assert ports.allocate_ports(**kwargs) == [22000]


def test_dead_owner_reservation_is_reclaimed(monkeypatch, tmp_path):
    from lite.gym.utils.backend import ports

    monkeypatch.setattr(ports, "_LOCK_FILE", tmp_path / "ports.lock")
    monkeypatch.setattr(ports, "_RESERVATION_FILE", tmp_path / "ports.json")
    monkeypatch.setattr(ports, "_is_port_free", lambda _port: True)
    kwargs = dict(n=1, range_start=22000, range_end=22001)

    assert ports.allocate_ports(**kwargs) == [22000]
    monkeypatch.setattr(ports, "_pid_alive", lambda _pid: False)
    assert ports.allocate_ports(**kwargs) == [22000]
