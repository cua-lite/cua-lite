#!/usr/bin/env python3
"""osworld_2 in-container eval server — runs INSIDE `cua-lite/osworld_2` (the container's
Linux userspace, OUTSIDE the guest VM).

Hosts OSWorld-V2's `desktop_env` here so the cua-lite HOST needs ZERO `desktop_env` — this
is what resolves the single-uv v1↔v2 dist conflict (v1 osworld + lite.osworld keep the v1
dist on the host; osworld_2 carries V2 only in this image). Attaches to the guest VM's
Flask server at the qemu-docker guest IP (20.20.20.21):5000 and exposes a thin JSON HTTP-RPC the host drives:

  GET  /healthz                 -> {ok, vm_ready}
  POST /reset  {task_id}        -> load BaseTask task_<id>.py + dte.reset(task)
                                   -> {instruction, screenshot_b64}
  POST /step   {cmd, pause}     -> dte.step(cmd, pause)  (cmd already translated host-side) -> {ok}
  POST /screenshot              -> {screenshot_b64}
  POST /evaluate                -> {reward, payload}     (dict-score extracted)
  POST /close                   -> {ok}                  (drop the DesktopEnv; cua-lite reaps the container)

Baked into the image at /usr/local/bin/server.py; the host bind-mounts the checked-in copy
over it (edits take effect without a rebuild) and launches it via `docker exec -d` once the
VM's :5000 answers. Listens on a fixed in-container port (mapped to a dynamic host port).

Config via env (set at `docker run`): SCREEN_W/SCREEN_H (1920/1080), CLIENT_PASSWORD
(osworld-public-evaluation), TASK_CLASS_DIR (/opt/osworld-v2/evaluation_examples/task_class), CACHE_DIR (/root/.cache),
SERVER_PORT (6000). No cua-lite imports — pure desktop_env + FastAPI.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import shlex
import sys
import threading
import traceback
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_SCREEN_W = int(os.environ.get("SCREEN_W", "1920"))
_SCREEN_H = int(os.environ.get("SCREEN_H", "1080"))
_CLIENT_PASSWORD = os.environ.get("CLIENT_PASSWORD", "osworld-public-evaluation")
_TASK_CLASS_DIR = os.environ.get("TASK_CLASS_DIR", "/opt/osworld-v2/evaluation_examples/task_class")
_CACHE_DIR = os.environ.get("CACHE_DIR", "/root/.cache")
_SERVER_PORT = int(os.environ.get("SERVER_PORT", "6000"))
_VM_HOST = os.environ.get("VM_HOST", "20.20.20.21")  # qemu-docker guest IP (DNAT target); NOT localhost
_VM_SERVER_PORT = 5000      # the guest VM's OSWorld Flask server (reached at _VM_HOST, the DNAT guest IP)
_VM_CHROMIUM_PORT = 9222
_VM_VLC_PORT = 8080

# The gated task_<id>.py files import `evaluation_examples.task_class.generated_task_utils` from the
# baked OSWorld-V2 repo root — but `pip install .` installs only the `desktop_env` package, not the
# `evaluation_examples` tree, so it isn't importable by default. Put the repo root on sys.path so
# those task imports resolve (the Dockerfile COPYs the V2 checkout to /opt/osworld-v2).
_V2_REPO_ROOT = os.environ.get("V2_REPO_ROOT", "/opt/osworld-v2")
if _V2_REPO_ROOT not in sys.path:
    sys.path.insert(0, _V2_REPO_ROOT)

app = FastAPI(title="cua-lite osworld_2 in-container eval server")

# Single-trajectory server (one container per trajectory) — a global env + lock is enough.
_env: Any = None
_lock = threading.Lock()


@app.exception_handler(Exception)
async def _all_exc(request: Request, exc: Exception) -> JSONResponse:
    # Surface the in-container traceback in the RPC response (else the host sees a bare 500).
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": traceback.format_exc()[-2500:]})


# ── vendored task loader (upstream task_loader.py is a repo-root module, not pip-exposed) ──
def _load_task_from_file(task_path: str) -> Any:
    from desktop_env.task_base import BaseTask
    name = f"osworld_2_task_{os.path.splitext(os.path.basename(task_path))[0]}"
    spec = importlib.util.spec_from_file_location(name, task_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load task module from {task_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "get_task") and callable(module.get_task):
        return module.get_task()
    if hasattr(module, "TASK_CLASS"):
        return module.TASK_CLASS()
    if hasattr(module, "Task") and callable(module.Task):
        return module.Task()
    for value in module.__dict__.values():
        if isinstance(value, type) and issubclass(value, BaseTask) and value is not BaseTask:
            return value()
    raise ValueError(f"no task class (get_task/TASK_CLASS/Task/BaseTask subclass) in {task_path}")


def _attached_desktop_env():
    """Build a DesktopEnv that ATTACHES to this container's guest VM (:5000) instead of spawning
    one — a pure controller/setup/evaluator façade so reset/step/evaluate run upstream-verbatim
    (the provider is never created; /reset expands the guest volume before task setup)."""
    from desktop_env.desktop_env import DesktopEnv
    from desktop_env.controllers.python import PythonController
    from desktop_env.controllers.setup import SetupController

    class _Attached(DesktopEnv):
        def __init__(self):
            # Replicate DesktopEnv.__init__ MINUS create_vm_manager_and_provider + _prepare_volume
            # — the docker provider connects to docker.sock eagerly (absent inside this container)
            # and _prepare_volume is provider-based. We never use the provider (attach + no-op close).
            # Volume tasks expand in /reset once the attached guest is ready.
            # Pinned dist → stable mirror. NOT a monkeypatch — a clean override.
            self.region = None
            self.provider_name = "docker"
            self.enable_proxy = False
            self.force_disable_vnc = False
            self.force_disable_recording = False
            self.volume_size = None
            self.client_password = _CLIENT_PASSWORD or "osworld-public-evaluation"
            self.screen_width, self.screen_height = _SCREEN_W, _SCREEN_H
            self.server_port, self.chromium_port, self.vnc_port, self.vlc_port = 5000, 9222, 8006, 8080
            self.current_use_proxy = False
            self.manager, self.provider = None, None
            self.os_type = "Ubuntu"
            self.is_environment_used = False
            self.path_to_vm = "__attached__"
            self.snapshot_name = "init_state"
            self.cache_dir_base = _CACHE_DIR
            self.headless = False
            self.require_a11y_tree = False
            self.require_terminal = False
            self._start_emulator()
            self.instruction = None
            self.action_space = "pyautogui"
            self._traj_no, self._step_no, self.action_history = -1, 0, []
            self.task_config = None
            self.user_simulator = None

        def _start_emulator(self):
            self.vm_ip = _VM_HOST
            self.server_port = _VM_SERVER_PORT
            self.chromium_port = _VM_CHROMIUM_PORT
            self.vlc_port = _VM_VLC_PORT
            self.vnc_port = 0
            self.controller = PythonController(vm_ip=self.vm_ip, server_port=self.server_port)
            self.setup_controller = SetupController(
                vm_ip=self.vm_ip, server_port=self.server_port,
                chromium_port=self.chromium_port, vlc_port=self.vlc_port,
                cache_dir=self.cache_dir_base, client_password=self.client_password,
                screen_width=self.screen_width, screen_height=self.screen_height)
            self._finalize_volume()

        def _revert_to_snapshot(self):
            # Attach model has no snapshot + no provider. reset() marks the env dirty
            # (is_environment_used=True) BEFORE task setup, so a setup-retry re-enters reset()'s
            # revert branch → upstream would call provider.revert_to_snapshot(None) → crash (masking
            # the real setup error). No-op: reset() re-attaches via _start_emulator() right after this,
            # and there's nothing to revert to anyway (one trajectory per freshly-booted, reaped container).
            pass

        def _finalize_volume(self):
            # /reset performs expansion once, before task setup; there is no provider here.
            pass

        def close(self):  # cua-lite reaps the container
            pass

    return _Attached()


def _encode_screenshot(obs: Any) -> str | None:
    if obs is None:
        return None
    from PIL import Image
    import numpy as np
    if isinstance(obs, (bytes, bytearray)):
        return base64.b64encode(bytes(obs)).decode()
    buf = io.BytesIO()
    if isinstance(obs, Image.Image):
        obs.save(buf, format="PNG")
    elif isinstance(obs, np.ndarray):
        Image.fromarray(obs).save(buf, format="PNG")
    else:
        return None
    return base64.b64encode(buf.getvalue()).decode()


# ── request models ──
class ResetBody(BaseModel):
    task_id: str


class StepBody(BaseModel):
    cmd: str
    pause: float = 0


@app.get("/healthz")
def healthz() -> dict:
    import requests
    vm_ready = False
    try:
        r = requests.get(f"http://{_VM_HOST}:{_VM_SERVER_PORT}/screenshot", timeout=(3, 3))
        vm_ready = r.status_code == 200
    except Exception:
        pass
    return {"ok": True, "vm_ready": vm_ready, "env_ready": _env is not None}


@app.post("/reset")
def reset(body: ResetBody) -> dict:
    global _env
    with _lock:
        path = os.path.join(_TASK_CLASS_DIR, f"task_{body.task_id}.py")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"task class not found: {path}")
        task = _load_task_from_file(path)
        if "freecad" in task.get("related_apps", []):
            from desktop_env.controllers.setup import SetupController
            # On the released VM, packagekitd can hold apt's lock just after boot.
            # Let the unmodified task's apt-get install start when that lock is free.
            SetupController(vm_ip=_VM_HOST, client_password=_CLIENT_PASSWORD).execute(
                command=("set -eu; command -v fuser >/dev/null; "
                         f"while printf '%s\\n' {shlex.quote(_CLIENT_PASSWORD)} | sudo -S -p '' fuser -s "
                         "/var/lib/apt/lists/lock /var/lib/dpkg/lock-frontend 2>/dev/null; "
                         "do sleep 2; done"),
                shell=True, timeout=180, check=True,
            )
        if _env is not None:
            try:
                _env.close()
            except Exception:
                pass
        _env = _attached_desktop_env()
        volume_size = task.get("volume_size")
        if volume_size is not None:
            if volume_size != int(os.environ.get("OSWORLD_VOLUME_SIZE", "0")):
                raise RuntimeError(f"task volume_size={volume_size} does not match VM disk")
            from desktop_env.providers.volume import expand_guest_volume
            expand_guest_volume(
                os_type=_env.os_type, controller=_env.controller,
                setup_controller=_env.setup_controller, client_password=_CLIENT_PASSWORD,
                expected_size_gb=volume_size,
            )
        _env.reset(task_config=task)
        # Upstream FreeCAD setup does not check the install command's exit code.
        # Reject a reset that completed without the app instead of scoring an unusable VM.
        if "freecad" in task.get("related_apps", []):
            _env.setup_controller.execute(["which", "freecad"], timeout=10, check=True)
        raw = _env.controller.get_screenshot()
        return {
            "instruction": task.get("instruction", ""),
            "screenshot_b64": _encode_screenshot(raw),
        }


@app.post("/step")
def step(body: StepBody) -> dict:
    if _env is None:
        raise HTTPException(status_code=409, detail="no active env; call /reset first")
    with _lock:
        _env.step(body.cmd, body.pause)
    return {"ok": True}


@app.post("/screenshot")
def screenshot() -> dict:
    if _env is None:
        raise HTTPException(status_code=409, detail="no active env")
    raw = _env.controller.get_screenshot()
    return {"screenshot_b64": _encode_screenshot(raw)}


@app.post("/evaluate")
def evaluate() -> dict:
    if _env is None:
        raise HTTPException(status_code=409, detail="no active env")
    with _lock:
        raw = _env.evaluate()
    # Match upstream's canonical extraction (lib_run_single.py): a dict result without "score"
    # scores 0.0, NOT a KeyError; a non-coercible score also scores 0.0 (upstream wraps the float()
    # in try/except (TypeError, ValueError)). `raw` may be a dict, float, or int.
    if isinstance(raw, dict):
        try:
            reward = float(raw.get("score", 0.0))
        except (TypeError, ValueError):
            reward = 0.0
    else:
        reward = float(raw)
    # JSON-safe the payload: v2 evaluator dicts can hold numpy scalars/arrays/Paths that would 500 the
    # response serializer AFTER the handler returns (losing the reward the host already has). Coerce.
    payload = json.loads(json.dumps(raw, default=str)) if isinstance(raw, dict) else None
    return {"reward": reward, "payload": payload}


@app.post("/close")
def close() -> dict:
    global _env
    with _lock:
        if _env is not None:
            try:
                _env.close()
            except Exception:
                pass
            _env = None
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=_SERVER_PORT, log_level="warning")
