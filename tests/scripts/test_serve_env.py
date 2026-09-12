from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from lite.gym.remote.alive import SERVER_KEEP_ALIVE_TIMEOUT_SEC, resolve_keep_alive_timeout

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "serve_env.py"


def _load_serve_env():
    module_name = "_cua_lite_serve_env_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_serve_env_help_documents_operator_flags() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--max-live-envs" in proc.stdout
    assert "--warm-singleton" in proc.stdout
    assert "--timeout-keep-alive" in proc.stdout
    assert "--reset-concurrency" not in proc.stdout


def test_warm_abbreviation_is_rejected(monkeypatch) -> None:
    mod = _load_serve_env()

    for flag in ("--warm", "--war"):
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), flag])
        with pytest.raises(SystemExit) as exc:
            mod._parse_args()
        assert exc.value.code == 2

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--warm-singleton"])
    args = mod._parse_args()
    assert args.warm_singleton is True


def test_serve_env_passes_the_derived_timeout_to_uvicorn() -> None:
    """The launcher must apply the derived keep-alive timeout and expose the flag."""
    text = SCRIPT.read_text()

    assert "timeout_keep_alive=keep_alive" in text
    assert "resolve_keep_alive_timeout(args.timeout_keep_alive)" in text
    assert "--timeout-keep-alive" in text, "operators need a flag, not a redeploy"
    assert resolve_keep_alive_timeout(None) == SERVER_KEEP_ALIVE_TIMEOUT_SEC


def test_serve_env_builds_real_state_without_default_blocks(monkeypatch) -> None:
    mod = _load_serve_env()
    captured: dict = {}

    monkeypatch.setattr(mod, "cached_host_capacity", lambda: object())
    monkeypatch.setattr(mod, "derive_admission_config", lambda _host, **_kwargs: object())
    monkeypatch.setattr(mod, "AdmissionGate", lambda cfg: ("gate", cfg))
    monkeypatch.setattr(mod, "log_admission_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "resolve_keep_alive_timeout", lambda _explicit: 15.0)
    monkeypatch.setattr(mod, "log_keep_alive_timeout", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("CUA_LITE_RESET_CONCURRENCY", "4")

    def fake_make_app(state, *_args, **_kwargs):
        captured["state"] = state
        return object()

    monkeypatch.setattr(mod, "make_app", fake_make_app)
    monkeypatch.setattr(mod.uvicorn, "run", lambda *_args, **_kwargs: None)

    mod.main(
        argparse.Namespace(
            host="127.0.0.1",
            port=30100,
            max_live_envs=None,
            idle_ttl_sec=600.0,
            token="token",
            admin_token=None,
            env_ids=None,
            warm_singleton=False,
            timeout_keep_alive=None,
        )
    )

    state = captured["state"]
    assert isinstance(state, mod.State)
    assert state.blocked_env_ids == frozenset()
    assert state.allowed_env_ids is None
    assert state.reset_concurrency == 4
