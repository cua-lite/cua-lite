"""HTTP RPC layer hosting ``env_launcher``'s ``AsyncAndroidEnv`` + an
``android_world.task_evals`` task instance, running INSIDE the
androidworld container.

Why this exists
---------------
androidworld's ``a11y_grpc_wrapper`` opens a gRPC server in the
env-server's *host* Python process and asks the in-emulator a11y
forwarder APK to call back to ``10.0.2.2:<port>``. On rootless docker
nodes the slirp4netns layer disables the host-loopback path, so the
forwarder's reverse connection never lands. By co-locating both the
emulator AND the env_launcher-managed Python env inside the same
container, that callback now goes container-localhost-to-container-
localhost, sidestepping the rootless restriction entirely. The host
process talks to this RPC server over ``localhost:<api_port>`` —
which is forward traffic (host → container), the direction docker
*does* support under rootless.

API surface
-----------
All payloads are pickled Python objects so we preserve protobuf /
numpy fidelity without writing a parallel JSON schema.

  GET  /healthz                       -> {ok, env_ready, task_ready}
  POST /init  (pickled dict)          -> setup env_launcher + hide_automation_ui
  POST /env/reset  (pickled dict)     -> env.reset(go_home=...)
  POST /env/get_state                 -> pickled state (pixels, ui_elements, ...)
  POST /env/execute_action  (pkl)     -> env.execute_action(action)
  POST /env/set_interaction_cache(pkl)-> env.interaction_cache = text  (Q&A tasks)
  POST /env/execute_adb_call  (pkl)   -> env.controller.env.execute_adb_call(...)
  POST /env/attempt_enable_networking -> env.controller.env.attempt_enable_networking()
  POST /env/type_b64 (pkl)             -> UTF-8 text via ADBKeyBoard
  POST /env/launch_package (pkl)       -> launch a package's main activity
  POST /env/sqlite_count (pkl)         -> PhoneWorld verifier COUNT query
  POST /task/load  (pickled dict)     -> load TaskEval by class name + params
  POST /task/initialize               -> task.initialize_task(env)
  POST /task/is_successful            -> task.is_successful(env)  -> {reward: float}
  POST /task/tear_down                -> task.tear_down(env); drop instance
  POST /close                         -> tear_down + env.close()

Run (inside container):
    python /usr/local/bin/server.py --port 9554
"""
from __future__ import annotations

import argparse
import base64
import logging
import pickle
import re
import subprocess
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)


# ── adb hang mitigation ──────────────────────────────────────────────────────
# android_env's AdbController defaults to a 120s timeout per command and
# n_retries=2. On retry it calls _restart_server which itself runs 3 more
# adb sub-calls each subject to the same 2-try loop. Worst case for one
# hung device-specific adb call:
#   try1 (120s) + restart_server (3 × 2 × 120s = 720s) + try2 (120s) ≈ 16 min
#
# Under multi-tenant host contention (load avg > 1000 from neighbor ML
# workloads burst-loading model checkpoints), adb-bg subprocesses get
# D-state-blocked on disk/IO. The chain becomes:
#   adb hang → in-container uvicorn handler stuck → host env-server sees
#   ``Connection refused`` after a few seconds → host outer retry fires
#   (~14 s) → client step_timeout (120 s framework default; 180 s in eval
#   run.sh) fires before any recovery → task aborts with EnvTimeoutError.
#
# Mitigation: cap per-command adb timeout at 8 s AND drop n_retries to 0.
# Real adb commands in this workload finish in < 2 s on an unloaded host;
# 8 s catches genuine slow ops without burning the host step budget.
# n_retries=0 prevents the 3 × 2 × 8 s = 48 s ``_restart_server`` cascade
# so a single bad command fails in 8 s instead of 48-80 s. The host-side
# env-server's outer retry
# already absorbs transient errors at the /step level with jittered
# backoff — we don't need android_env's in-controller retry logic on top.
#
# Patch is applied here (inside the container, where AdbController is
# actually instantiated by env_launcher) instead of on the host because
# the host never calls AdbController directly — all real adb traffic
# happens inside this container.
def _patch_adb_controller_for_fast_hang_recovery() -> None:
    from android_env.components import adb_controller as _adb_ctrl
    if getattr(_adb_ctrl.AdbController.execute_command, "_lite_patched", False):
        return  # idempotent
    _orig_execute = _adb_ctrl.AdbController.execute_command

    def _patched_execute(self, args, timeout=None, device_specific=True):
        capped = 8.0 if timeout is None else min(timeout, 8.0)
        # AdbController reads ``self._n_retries`` on each call. Force
        # to 0 so a bad command fails fast instead of running the
        # 3 × 2 × N_seconds ``_restart_server`` cascade.
        try:
            self._n_retries = 0
        except Exception:
            pass
        return _orig_execute(self, args, timeout=capped, device_specific=device_specific)

    _patched_execute._lite_patched = True
    _adb_ctrl.AdbController.execute_command = _patched_execute


_patch_adb_controller_for_fast_hang_recovery()


app = FastAPI(title="cua-lite androidworld in-container server")

# Module-level state. One container = one env + one task; no concurrency.
_env: Any = None
_task: Any = None
# Params returned by ``/task/generate_params`` are stashed here as their
# real ``androidworld`` dataclass instances. ``/task/load`` reads from
# this slot instead of re-using whatever the host echoed back, because
# ``/task/generate_params`` returns a JSON-safe dict (dataclasses
# flattened) to keep the host's wire format ``androidworld``-free, and
# that flattening is lossy for downstream consumers — ``sqlite_validators
# .add_rows`` calls ``dataclasses.fields(row)`` and fails on plain dicts
# / SimpleNamespace with ``TypeError: must be called with a dataclass
# type or instance``. Cleared by ``/task/load`` after consumption and by
# ``/task/tear_down`` defensively. See androidworld/docker/server.py
# regression note for the empty-app UI symptom this fixes.
_pending_params: dict[str, Any] | None = None
_pending_task_class_name: str | None = None
_ADB = "/root/Android/Sdk/platform-tools/adb"
_DEVICE = "emulator-5554"


def _adb(
    *args: str,
    check: bool = True,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_ADB, "-s", _DEVICE, *args], capture_output=True, text=True,
        input=stdin, timeout=20, check=check,
    )


def _pickled_response(obj: Any) -> Response:
    return Response(
        content=pickle.dumps(obj), media_type="application/octet-stream"
    )


def _params_to_json_safe(obj: Any) -> Any:
    """Flatten a task-params tree (dataclass / list / dict / primitive)
    into pure JSON-safe nodes (plain dicts, no class references, no
    SimpleNamespace). Used for ``/task/generate_params``'s response —
    host logs these as ``task_params`` trajectory metadata, never
    re-instantiates the dataclasses from them.

    Distinct from ``_to_namespace`` (used by ``/env/get_state``):
    ``state.pixels`` style attribute access on host needs
    ``SimpleNamespace`` round-tripping, but task params need plain
    dicts so they survive JSON serialisation downstream.
    """
    import dataclasses as _dc
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return obj.tolist() if hasattr(obj, "tolist") else repr(obj)
    if isinstance(obj, (list, tuple)):
        return [_params_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _params_to_json_safe(v) for k, v in obj.items()}
    if _dc.is_dataclass(obj):
        return {f.name: _params_to_json_safe(getattr(obj, f.name)) for f in _dc.fields(obj)}
    return repr(obj)


def _to_namespace(obj: Any) -> Any:
    """Convert an ``androidworld`` dataclass / list / dict tree into plain
    ``SimpleNamespace`` / ``list`` / ``dict`` / primitive nodes.

    Why this exists: the host's Python venv intentionally has zero
    ``from android_world ...`` imports (host == orchestrator only; the
    library lives inside the docker image). A naive ``pickle.dumps(state)``
    on the wire would embed ``android_world.env.interface.State`` /
    ``android_world.env.representation_utils.UIElement`` /
    ``android_world.env.representation_utils.BoundingBox`` class
    references that the host can't resolve at unpickle time. Re-emitting
    the tree as ``SimpleNamespace`` keeps the host's attribute-access
    contract (``state.pixels``, ``el.bbox_pixels.x_min``) intact without
    requiring the host to import the library.

    NOT to be used for task-params: those need to round-trip back to
    ``/task/load`` and reach ``sqlite_validators.add_rows`` /
    ``dataclasses.fields(row)`` as real dataclass instances. See
    ``_params_to_json_safe`` + ``_pending_params`` slot for that path.
    """
    import dataclasses as _dc
    import types
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    # numpy arrays serialise natively via pickle; preserve as-is.
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_namespace(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_namespace(v) for k, v in obj.items()}
    if _dc.is_dataclass(obj):
        return types.SimpleNamespace(
            **{f.name: _to_namespace(getattr(obj, f.name)) for f in _dc.fields(obj)}
        )
    # Fall through (e.g. raw forest proto): drop the object — host code
    # only reads ``pixels`` / ``ui_elements`` so anything else is unused.
    # Returning ``None`` is safer than passing through an un-unpickleable
    # class reference.
    return None


async def _pickled_body(request: Request) -> Any:
    raw = await request.body()
    return pickle.loads(raw) if raw else {}


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "env_ready": _env is not None,
        "task_ready": _task is not None,
    }


@app.post("/init")
async def init(request: Request) -> dict[str, Any]:
    """Calls env_launcher.load_and_setup_env() + hide_automation_ui().

    Idempotent: a second call when env is already initialized returns
    ``{ok: true, already_initialized: true}`` without recreating.
    """
    global _env
    if _env is not None:
        return {"ok": True, "already_initialized": True}

    body = await _pickled_body(request)
    from android_world.env import env_launcher

    _env = env_launcher.load_and_setup_env(
        console_port=body["console_port"],
        grpc_port=body["grpc_port"],
        adb_path=body["adb_path"],
        emulator_setup=body.get("emulator_setup", False),
        freeze_datetime=body.get("freeze_datetime", True),
    )
    try:
        _env.hide_automation_ui()
    except Exception as e:
        logger.warning("hide_automation_ui failed: %s", e)
    return {"ok": True, "already_initialized": False}


# ── env passthrough ──────────────────────────────────────────────────────────


@app.post("/env/reset")
async def env_reset(request: Request) -> dict[str, Any]:
    if _env is None:
        raise HTTPException(409, "env not initialized")
    body = await _pickled_body(request)
    _env.reset(go_home=body.get("go_home", True))
    return {"ok": True}


@app.post("/env/get_state")
def env_get_state() -> Response:
    if _env is None:
        raise HTTPException(409, "env not initialized")
    return _pickled_response(_to_namespace(_env.get_state()))


@app.post("/env/execute_action")
async def env_execute_action(request: Request) -> dict[str, Any]:
    """Body: pickled dict of JSONAction fields (action_type, x, y, text,
    app_name, …). We construct the JSONAction inside the container so
    the host doesn't need to import ``android_world.env.json_action``.
    """
    if _env is None:
        raise HTTPException(409, "env not initialized")
    body = await _pickled_body(request)
    from android_world.env import json_action
    action = json_action.JSONAction(**body)
    _env.execute_action(action)
    return {"ok": True}


@app.post("/env/set_interaction_cache")
async def env_set_interaction_cache(request: Request) -> dict[str, Any]:
    if _env is None:
        raise HTTPException(409, "env not initialized")
    body = await _pickled_body(request)
    _env.interaction_cache = body["text"]
    return {"ok": True}


@app.post("/env/execute_adb_call")
async def env_execute_adb_call(request: Request) -> Response:
    """Body: pickled ``adb_pb2.AdbRequest``. Returns pickled AdbResponse."""
    if _env is None:
        raise HTTPException(409, "env not initialized")
    adb_request = await _pickled_body(request)
    return _pickled_response(_env.execute_adb_call(adb_request))


@app.post("/env/attempt_enable_networking")
def env_attempt_enable_networking() -> dict[str, Any]:
    if _env is None:
        raise HTTPException(409, "env not initialized")
    _env.controller.env.attempt_enable_networking()
    return {"ok": True}


@app.post("/env/check_airplane_mode")
def env_check_airplane_mode() -> dict[str, Any]:
    """Returns ``{airplane_on: bool}``. Mirrors host-side
    ``adb_utils.check_airplane_mode(env.controller.env)``; we run it
    in-container so the host doesn't need to import adb_utils."""
    if _env is None:
        raise HTTPException(409, "env not initialized")
    from android_world.env import adb_utils
    return {"airplane_on": bool(adb_utils.check_airplane_mode(_env.controller.env))}


@app.post("/env/exec_swipe")
async def env_exec_swipe(request: Request) -> dict[str, Any]:
    """Body: ``{sx, sy, ex, ey, duration_ms}``. Executes a precise
    swipe via raw ``adb input swipe`` (bypasses JSONAction's enum-only
    scroll direction). Mirrors host-side ``_exec_swipe_adb``."""
    if _env is None:
        raise HTTPException(409, "env not initialized")
    body = await _pickled_body(request)
    from android_world.env import adb_utils
    cmd = adb_utils.generate_swipe_command(
        body["sx"], body["sy"], body["ex"], body["ey"],
        duration_ms=body["duration_ms"],
    )
    adb_utils.issue_generic_request(cmd, _env.controller.env)
    return {"ok": True}


@app.post("/env/type_b64")
async def env_type_b64(request: Request) -> dict[str, Any]:
    """Type UTF-8 text through the active ADBKeyBoard IME, without Enter."""
    if _env is None:
        raise HTTPException(409, "env not initialized")
    body = await _pickled_body(request)
    text = body.get("text")
    if not isinstance(text, str):
        raise HTTPException(400, "text must be a string")
    ime = _adb("shell", "settings", "get", "secure", "default_input_method").stdout.strip()
    if ime != "com.android.adbkeyboard/.AdbIME":
        raise HTTPException(409, f"ADBKeyBoard is not active: {ime!r}")
    _adb(
        "shell", "am", "broadcast", "-a", "ADB_INPUT_B64",
        "--es", "msg", base64.b64encode(text.encode()).decode(),
    )
    return {"ok": True}


@app.post("/env/launch_package")
async def env_launch_package(request: Request) -> dict[str, Any]:
    """Launch an installed package's main activity."""
    if _env is None:
        raise HTTPException(409, "env not initialized")
    package = (await _pickled_body(request)).get("package", "")
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.]+", package):
        raise HTTPException(400, "invalid package")
    _adb(
        "shell", "monkey", "-p", package,
        "-c", "android.intent.category.LAUNCHER", "1",
    )
    return {"ok": True}


@app.post("/env/sqlite_count")
async def env_sqlite_count(request: Request) -> dict[str, Any]:
    """Run one trusted, normalized verifier COUNT query on the emulator."""
    if _env is None:
        raise HTTPException(409, "env not initialized")
    body = await _pickled_body(request)
    path = body.get("database_path", "")
    sql = body.get("sql", "")
    match = re.fullmatch(
        r"/data/data/(?P<package>com\.phoneuse\.[a-zA-Z][a-zA-Z0-9_]*)/"
        r"databases/[a-zA-Z0-9_.-]+",
        path,
    )
    if not match:
        raise HTTPException(400, "invalid app database path")
    if not isinstance(sql, str) or not sql.startswith("SELECT COUNT(*) FROM "):
        raise HTTPException(400, "invalid verifier query")
    if any(token in sql for token in (";", "--", "/*", "*/")):
        raise HTTPException(400, "invalid verifier query")
    _adb("shell", "am", "force-stop", match.group("package"))
    if _adb("shell", "test", "-f", path, check=False).returncode:
        return {"count": 0}
    # Feed SQL over stdin: ``adb shell`` joins argv into a remote shell
    # command, where an unquoted ``COUNT(*)`` is parsed as shell syntax.
    response = _adb("shell", "sqlite3", path, stdin=sql, check=False)
    if response.returncode:
        package = match.group("package")
        db_name = path.rsplit("/", 1)[-1]
        response = _adb(
            "shell", "run-as", package, "sqlite3", f"databases/{db_name}",
            stdin=sql, check=False,
        )
    if response.returncode:
        raise HTTPException(500, f"sqlite verifier failed: {response.stderr.strip()}")
    try:
        return {"count": int(response.stdout.strip())}
    except ValueError as exc:
        raise HTTPException(500, "sqlite verifier returned a non-integer") from exc


@app.post("/env/exec_menu")
def env_exec_menu() -> dict[str, Any]:
    """Press Android MENU key (KEYCODE_MENU=82) via raw adb. JSONAction
    has no menu action_type, so this bypasses it like swipe."""
    return _exec_keyevent(82)


@app.post("/env/exec_keyevent")
async def env_exec_keyevent(request: Request) -> dict[str, Any]:
    body = await _pickled_body(request)
    return _exec_keyevent(int(body["keycode"]))


def _exec_keyevent(keycode: int) -> dict[str, Any]:
    if _env is None:
        raise HTTPException(409, "env not initialized")
    from android_world.env import adb_utils
    adb_utils.issue_generic_request(
        ["shell", "input", "keyevent", str(keycode)], _env.controller.env,
    )
    return {"ok": True}


@app.post("/env/check_a11y")
def env_check_a11y() -> dict[str, Any]:
    """Run androidworld's ``get_a11y_tree`` in-container.

    The a11y tree fetch is implemented in upstream androidworld by
    polling ``env.accumulate_new_extras()`` (a method on the a11y-grpc
    wrapper that hangs off the underlying AndroidEnv). That call is
    not part of the slim ``_RemoteControllerEnv`` surface the host's
    proxy exposes, so the host's ``_ensure_a11y_healthy`` pre-flight
    routes the check through this endpoint instead. Returns ``ok=True``
    on a successful fetch; ``ok=False`` (with ``error`` string) is the
    host's cue to call ``_refresh_env``.
    """
    if _env is None:
        raise HTTPException(409, "env not initialized")
    from android_world.env import android_world_controller
    try:
        android_world_controller.get_a11y_tree(
            _env.controller.env, max_retries=5, sleep_duration=1.0,
        )
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ── task lifecycle ───────────────────────────────────────────────────────────


@app.post("/task/generate_params")
async def task_generate_params(request: Request) -> Response:
    """Body: pickled ``{task_class_name, seed?}``. Returns pickled
    JSON-safe params dict (dataclasses flattened to dicts).

    Round-trip contract: real ``androidworld`` dataclass instances
    (e.g. ``Expense``, ``Recipe``) are stashed in module-level
    ``_pending_params`` here and consumed by ``/task/load`` — the
    wire-format dict is **logging-only**. ``/task/load`` ignores the
    ``params`` field the host echoes back, because the lossy JSON
    flattening would otherwise break downstream
    ``sqlite_validators.add_rows`` → ``dataclasses.fields(row)``.

    Seeding semantics: the host's ``AndroidWorldEnv.reset()`` derives an
    instance_seed from ``md5(f"{task_name}:{base_seed}")`` and asks us
    to generate params under that seed. We save/restore the container's
    global RNG state around the call so this doesn't pollute other
    in-container RNG consumers (the apply_a11y_forwarder wrapper,
    etc.). When ``seed`` is None, params are random per call.
    """
    global _pending_params, _pending_task_class_name
    body = await _pickled_body(request)
    task_class_name = body["task_class_name"]
    seed = body.get("seed")

    import random

    from android_world import registry as aw_registry

    tasks = aw_registry.TaskRegistry().get_registry("android_world")
    if task_class_name not in tasks:
        raise HTTPException(404, f"unknown task class: {task_class_name}")
    task_class = tasks[task_class_name]

    if seed is not None:
        rng_state = random.getstate()
        try:
            random.seed(seed)
            params = task_class.generate_random_params()
            params["seed"] = seed
        finally:
            random.setstate(rng_state)
    else:
        params = task_class.generate_random_params()
    # Stash the real dataclass instances for /task/load to pick up.
    _pending_params = params
    _pending_task_class_name = task_class_name
    # Return only a JSON-safe flattening for host-side logging.
    return _pickled_response(_params_to_json_safe(params))


@app.post("/task/load")
async def task_load(request: Request) -> dict[str, Any]:
    """Body: pickled dict with ``task_class_name``. The host's echoed
    ``params`` field is **ignored** — we use the dataclass instances
    stashed by the most recent ``/task/generate_params`` call instead.

    Looks up the TaskEval class in androidworld's task registry,
    constructs an instance with the stashed params, and returns its
    static metadata (goal text, complexity, app_names, name).
    """
    global _task, _pending_params, _pending_task_class_name
    body = await _pickled_body(request)
    task_class_name = body["task_class_name"]

    if _pending_params is None or _pending_task_class_name != task_class_name:
        raise HTTPException(
            409,
            f"no stashed params for task_class={task_class_name!r}; call "
            f"/task/generate_params first (last stashed: "
            f"{_pending_task_class_name!r})",
        )

    from android_world import registry as aw_registry

    tasks = aw_registry.TaskRegistry().get_registry("android_world")
    if task_class_name not in tasks:
        raise HTTPException(404, f"unknown task class: {task_class_name}")
    task_class = tasks[task_class_name]
    _task = task_class(_pending_params)
    # Single-use: clear stash so a stale params can't accidentally
    # bind to a later mismatched task_class_name on /task/load.
    _pending_params = None
    _pending_task_class_name = None
    return {
        "ok": True,
        "goal": _task.goal,
        "complexity": _task.complexity,
        "app_names": list(_task.app_names),
        "name": _task.name,
    }


@app.post("/task/initialize")
def task_initialize() -> dict[str, Any]:
    if _task is None or _env is None:
        raise HTTPException(409, "task or env not initialized")
    _task.initialize_task(_env)
    return {"ok": True}


@app.post("/task/is_successful")
def task_is_successful() -> dict[str, Any]:
    if _task is None or _env is None:
        raise HTTPException(409, "task or env not initialized")
    return {"reward": float(_task.is_successful(_env))}


@app.post("/task/tear_down")
def task_tear_down() -> dict[str, Any]:
    global _task, _pending_params, _pending_task_class_name
    if _task is None or _env is None:
        # Still clear any orphan stash from an interrupted
        # generate_params → load sequence.
        _pending_params = None
        _pending_task_class_name = None
        return {"ok": True}
    try:
        _task.tear_down(_env)
    except Exception as e:
        logger.warning("task.tear_down failed: %s", e)
    _task = None
    _pending_params = None
    _pending_task_class_name = None
    return {"ok": True}


# ── shutdown ─────────────────────────────────────────────────────────────────


@app.post("/close")
def close() -> dict[str, Any]:
    global _env, _task, _pending_params, _pending_task_class_name
    if _task is not None and _env is not None:
        try:
            _task.tear_down(_env)
        except Exception:
            pass
    _task = None
    _pending_params = None
    _pending_task_class_name = None
    if _env is not None:
        try:
            _env.close()
        except Exception:
            pass
        _env = None
    return {"ok": True}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # log_level="warning" keeps the access log quiet — every reset/step
    # would otherwise print a line, drowning out actual errors.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
