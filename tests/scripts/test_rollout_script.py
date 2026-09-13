from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rollout.py"
THREAD_CAP_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def test_rollout_help_exits_zero(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["scripts/rollout.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert exc_info.value.code == 0


def test_rollout_caps_native_thread_pools_before_numpy_import() -> None:
    script = f"""
import os
import sys
from pathlib import Path

source = Path({str(SCRIPT)!r}).read_text()
exec(source.split("setup_logging()")[0])
assert "numpy" not in sys.modules
for var in {THREAD_CAP_VARS!r}:
    assert os.environ[var] == "1", var
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")},
        text=True,
        capture_output=True,
        check=True,
    )

    assert proc.stdout == ""
