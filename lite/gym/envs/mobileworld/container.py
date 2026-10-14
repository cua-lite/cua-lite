"""Docker container + factory for MobileWorld — one DinD benchmark box per env.

Each container is a self-contained MobileWorld environment: a rooted Android
emulator (AVD snapshot ``init_state`` baked into the image), a nested dockerd
hosting the app backends (Mattermost / Mastodon / Mall), and the upstream
FastAPI task/eval server on in-container port 6800. The host drives everything
through that one HTTP port — task init/eval, device actions, screenshots — so
no adb or reverse connectivity is ever needed (works on rootless docker).

Design:

  - Network: default bridge network, private netns per container. Only the
    backend port (:data:`_CPORT_API` = 6800) is published to the host via
    ``docker -p``; the in-container viewer (7860) and ADB relay (5556) stay
    private.

  - Privileges: ``--privileged --device /dev/kvm``. Privileged is required by
    the upstream entrypoint (nested dockerd, ``sysctl``, iptables); the
    explicit ``--device`` matters under rootless docker, where ``--privileged``
    alone does not map host devices into the container.

  - Boot: the image entrypoint (upstream ``entrypoint.sh``) starts nested
    dockerd, loads the app-backend images, boots the emulator from its
    snapshot, and launches the server. We poll ``POST /init`` until the
    controller answers 200 — that end-to-end probe covers dockerd + emulator +
    server, typically 2-5 min cold.

  - Reuse: the upstream server's ``/task/init`` reloads the ``init_state``
    AVD snapshot on every call, so ONE container is safely reused across
    tasks/episodes — that is upstream's own eval model (N long-lived
    containers, tasks queued onto them). Spawn is expensive; destroy only on
    close/reap.

Container lifecycle: spawned by :meth:`MobileWorldContainerFactory.acquire`,
destroyed by :meth:`MobileWorldContainer.destroy` (idempotent ``docker rm -f -v``;
the nested dockerd and all its state die with the container). Host-side port
goes through ``lite/gym/utils/backend/ports.py``'s shared reservation file (range
10600-10699 — see the port-range map at the top of that file).

Run (smoke, requires ``cua-lite/mobileworld:latest`` built via
``scripts/install.sh`` + KVM):

    uv run python -c "
    from lite.gym.envs.mobileworld.container import MobileWorldContainerFactory
    c = MobileWorldContainerFactory(task_id='Smoke').acquire()
    print('booted:', c.name, 'api:', c.base_url)
    c.destroy()
    "
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from lite.gym.container import LiteContainerBase, boot_with_retry
from lite.gym.errors import CapacityExhausted
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.docker import _rm_argv, docker_run_detached
from lite.gym.utils.backend.ports import allocate_ports
from lite.gym.utils.config.naming import format_container_name as _format_container_name

logger = logging.getLogger(__name__)

ENV_DIR = str(Path(__file__).resolve().parents[0])
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# MOBILE_WORLD_CONFIG=<abs-path | bundled-name>.
# ============================================================================
# --- env_kwargs (per-instance) ---
#: Image tag. Must match what ``scripts/install.sh`` tags.
DEFAULT_IMAGE = CFG.env_kwargs["image"]
# --- server_kwargs (per-deployment) ---
#: docker-run → ``POST /init`` 200 deadline. Covers the whole entrypoint
#: pipeline: nested dockerd up, app-backend images loaded, emulator booted
#: from snapshot, FastAPI server ready. Cold boot is typically 2-5 min;
#: allow for contended hosts.
BOOT_TIMEOUT_S = CFG.server_kwargs["boot_timeout"]
#: Per-container memory cap. The box holds an emulator (~3 GB) + nested
#: dockerd with Mattermost/Mastodon/Mall backends (+ their databases) +
#: the FastAPI server + the gradio viewer. Disable swap (memory ==
#: memory-swap, swappiness=0) so an OOM kills the offender instead of
#: dragging the host into swap thrash.
_MEMORY_LIMIT = CFG.server_kwargs["memory_limit"]
#: Max concurrent container spawns (docker run → ready) across this process.
#: The framework's ``docker_create_slot`` semaphore only exists in env-server
#: mode (serve_env.py configures it; direct mode is an explicit no-op), and a
#: mobileworld create is far heavier than most envs' — ``--privileged`` +
#: device mapping + a port-forward through rootless slirp4netns, followed by
#: an entrypoint that immediately starts nested dockerd, loads multi-GB
#: backend images, and boots the emulator. A c=32 direct rollout firing all
#: creates at once wedged the rootless daemon so hard that every plain
#: ``docker run -d`` exceeded 180 s, and the queued creates then landed
#: AFTER their callers had timed out and moved on — 180 zombie "Created"
#: containers + name-conflict retries (observed 2026-07-03). This env-local
#: gate bounds spawns in BOTH modes; queued resets simply wait their turn
#: (budget for that wait when setting the reset_timeout make-kwarg).
_SPAWN_CONCURRENCY = CFG.server_kwargs["spawn_concurrency"]
# ============================================================================

#: The env-local spawn gate (see ``_SPAWN_CONCURRENCY``). Module-level so all
#: factories in the process share one budget.
_SPAWN_SEMA = threading.Semaphore(_SPAWN_CONCURRENCY)


# ── Constants ────────────────────────────────────────────────────────────────

#: In-container port of the upstream task/eval server (fixed by the image).
_CPORT_API = 6800

#: Host-side range for the published API port (see the port-range map in
#: ``lite/gym/utils/backend/ports.py`` — carved from the 10555-10999 free buffer).
_API_PORT_RANGE = (10600, 10700)

#: adb device id of the baked emulator (fixed by the image's entrypoint).
ADB_DEVICE = "emulator-5554"

#: Simulated-user LLM plumbing (answers ``ask_user`` questions; agent-user-
#: interaction tasks only — GUI-only tasks never call it). The upstream code
#: inside the container reads the USER_AGENT_* names; on the host side the
#: credentials come from USER_AGENT_* or the standard OPENAI_* env vars (secrets stay
#: env-var-sourced per the config house rules) and the model name is a yaml
#: knob (``server_kwargs.user_agent_model``) injected at spawn.
_USER_AGENT_ENV_MAP = {           # container var ← host env var (skip if unset)
    "USER_AGENT_API_KEY": "OPENAI_API_KEY",
    "USER_AGENT_BASE_URL": "OPENAI_BASE_URL",
}
_USER_AGENT_MODEL = CFG.server_kwargs["user_agent_model"]

#: Fast-fail signatures in ``docker logs`` — waiting out the full boot
#: deadline is pointless once one of these appears.
_FATAL_LOG_SIGNATURES = (
    "dockerd failed to become functional",
    "requires hardware acceleration",
    "Could not access KVM kernel module",
    "failed to initialize KVM",
)


# ── Container ────────────────────────────────────────────────────────────────

@dataclass
class MobileWorldContainer(LiteContainerBase):
    """One MobileWorld DinD box (emulator + app backends + task server)."""

    name: str
    api_port: int
    image: str = DEFAULT_IMAGE

    #: ``api_port`` was reserved via ``backend.ports``; released on destroy().
    _ports_owned: tuple[int, ...] = field(default=(), repr=False)

    rm_label: ClassVar[str] = "mobileworld"

    @property
    def base_url(self) -> str:
        """Host URL of the in-container task/eval server."""
        return f"http://localhost:{self.api_port}"

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """``docker run`` the image (its entrypoint boots everything), then
        block until the server's controller answers ``POST /init``."""
        # Register BEFORE the container-creating call so the multi-minute
        # boot window below is covered (a KeyboardInterrupt there is a
        # BaseException that acquire()'s ``except Exception`` won't catch).
        # destroy() is idempotent, so failures/retries stay clean.
        self._register()
        kvm_gid = _kvm_gid()
        user_agent_env = {
            container_var: value
            for container_var, host_var in _USER_AGENT_ENV_MAP.items()
            if (value := os.environ.get(container_var) or os.environ.get(host_var))
        }
        docker_run_detached(
            name=self.name, image=self.image,
            auto_remove=True,
            # Nested dockerd + sysctl + iptables in the entrypoint need full
            # privileges; /dev/kvm must be mapped explicitly under rootless
            # docker (--privileged alone doesn't map host devices there).
            privileged=True,
            devices=("/dev/kvm",),
            memory=_MEMORY_LIMIT,
            group_add=(kvm_gid,) if kvm_gid is not None else (),
            env={**user_agent_env, "USER_AGENT_MODEL": _USER_AGENT_MODEL},
            ports=((self.api_port, _CPORT_API),),
            # Simulator-specific credentials take precedence over OPENAI_*.
            redact=("USER_AGENT_API_KEY",),
            timeout=180.0,
            label="mobileworld",
        )
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        """Poll ``POST /init`` until the controller answers 200.

        The upstream entrypoint boots nested dockerd → app backends →
        emulator → server strictly in that order, so a 200 from ``/init``
        (which builds an AndroidController and health-checks the device)
        proves the whole stack is up. Fast-fails on fatal signatures in
        ``docker logs`` (dockerd dead, KVM inaccessible) and on the
        container itself exiting.
        """
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        body = json.dumps({"device": ADB_DEVICE}).encode()
        last_err = ""
        while time.monotonic() < deadline:
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/init", data=body,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    if resp.status == 200:
                        logger.info(
                            "mobileworld server ready on %s", self.base_url
                        )
                        return
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"

            if not self._is_running():
                raise RuntimeError(
                    f"mobileworld container {self.name} exited during boot.\n"
                    f"--- docker logs tail ---\n{self._logs_tail()}"
                )
            log_tail = self._logs_tail()
            fatal = next(
                (s for s in _FATAL_LOG_SIGNATURES if s in log_tail), None
            )
            if fatal is not None:
                raise RuntimeError(
                    f"mobileworld container {self.name} hit a fatal boot error "
                    f"({fatal!r}).\n--- docker logs tail ---\n{log_tail[-1500:]}"
                )
            # 3s poll: each probe may build a controller server-side (adb
            # health check), so don't hammer it; boot is minutes anyway.
            time.sleep(3.0)

        # Still warming — recoverable; the client retries the same instance
        # (503 + Retry-After) instead of a terminal 500.
        raise CapacityExhausted.warming(
            f"mobileworld container {self.name} not ready in {BOOT_TIMEOUT_S}s "
            f"(last /init error: {last_err}).\n"
            f"--- docker logs tail ---\n{self._logs_tail()[-1500:]}"
        )

    def _is_running(self) -> bool:
        try:
            r = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}",
                 self.name],
                capture_output=True, timeout=10,
            )
        except subprocess.SubprocessError:
            return True  # daemon hiccup — keep waiting, don't false-fail
        return (r.stdout or b"").decode().strip() == "true"

    def _logs_tail(self) -> str:
        """Last ~4 KB of ``docker logs`` (the entrypoint tails the emulator /
        server / dockerd logs, so this carries actual failure context)."""
        try:
            r = subprocess.run(
                ["docker", "logs", "--tail", "60", self.name],
                capture_output=True, timeout=10,
            )
            return (r.stdout + r.stderr).decode("utf-8", errors="replace")
        except Exception:
            return "(could not read docker logs)"

    def adb_shell(self, *args: str, timeout: float = 15.0) -> str:
        """Run ``adb shell`` inside the MobileWorld container."""
        r = subprocess.run(
            [
                "docker", "exec", self.name,
                "adb", "-s", ADB_DEVICE, "shell", *map(str, args),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            detail = (r.stderr or r.stdout or "").strip()
            raise RuntimeError(f"mobileworld adb shell failed: {detail}")
        return r.stdout

    # destroy() inherited from LiteContainerBase (template: rm -f -v — the
    # ``-v`` reaps the anonymous ``/var/lib/docker`` volume the nested dockerd
    # writes — → release _ports_owned → de-register; idempotent, never raises).


# ── Factory ──────────────────────────────────────────────────────────────────

class MobileWorldContainerFactory:
    """Spawn one fresh :class:`MobileWorldContainer` per :meth:`acquire`.

    Holds shared kwargs (``image``, identity segments) so the env constructs
    the factory once and calls ``acquire()`` on (re-)spawn. Cross-episode
    reuse is NOT done here — the env keeps its container alive and relies on
    the server's ``/task/init`` snapshot reload between episodes.
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        task_id: str | None = None,
        session_id: str | None = None,
        token_hash: str | None = None,
        server_port: int | None = None,
    ):
        self.image = image
        self.task_id = task_id
        self.session_id = session_id
        self.token_hash = token_hash
        self.server_port = server_port

    def _make_name(self, api_port: int) -> str:
        return _format_container_name(
            env_id="mobileworld",
            task_id=self.task_id,
            suffix=str(api_port),
            session_id=self.session_id or os.environ.get("SESSION_ID"),
            token_hash=self.token_hash,
            server_port=self.server_port,
        )

    def acquire(self, *, max_attempts: int = 2) -> MobileWorldContainer:
        """Reserve a host port, ``docker run``, wait for readiness, return.

        Retries once on transient boot failures (fresh port + name each
        attempt). The whole per-attempt pipeline (zombie rm + docker run +
        boot wait) runs under TWO gates: the env-local ``_SPAWN_SEMA``
        (bounds concurrent spawns in every mode — direct rollouts have no
        framework semaphore, see ``_SPAWN_CONCURRENCY``) and the framework's
        ``docker_create_slot`` (env-server mode only; no-op otherwise).
        """
        from lite.gym.remote.admission import docker_create_slot

        def _build() -> MobileWorldContainer:
            api_port = allocate_ports(
                n=1,
                range_start=_API_PORT_RANGE[0],
                range_end=_API_PORT_RANGE[1],
            )[0]
            name = self._make_name(api_port)
            # Nuke any same-named zombie from a crashed prior run — a
            # stale name would fail ``docker run`` with "name already in
            # use". Best-effort: a timeout (wedged daemon) OR an OSError
            # (docker binary missing → FileNotFoundError) must NOT escape
            # here, else the ``allocate_ports`` reservation above leaks (the
            # container isn't constructed yet, so no ``destroy()`` releases
            # it). Swallow both; the run below then fails cleanly (on the
            # name conflict, or on the same missing-docker error inside
            # ``start()`` → ``destroy()`` releases the port) and retries.
            try:
                subprocess.run(
                    _rm_argv(name),
                    capture_output=True, timeout=60,
                )
            except (subprocess.SubprocessError, OSError) as e:
                logger.warning("pre-spawn rm of %s failed: %s", name, e)
            return MobileWorldContainer(
                name=name,
                api_port=api_port,
                image=self.image,
                _ports_owned=(api_port,),
            )

        def _start(c: MobileWorldContainer) -> None:
            with docker_create_slot():
                c.start()

        # attempt_gate holds the env-local spawn semaphore across the WHOLE
        # per-attempt pipeline (zombie rm + docker run + boot wait + failure
        # cleanup) — the documented two-gate contract. Retry classification is
        # single-sourced in boot_with_retry.
        return boot_with_retry(
            _build, start=_start, attempt_gate=lambda: _SPAWN_SEMA,
            max_attempts=max_attempts, label="mobileworld",
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _kvm_gid() -> int | None:
    """Host kvm group gid for ``--group-add`` (None if no kvm group)."""
    try:
        import grp
        return grp.getgrnam("kvm").gr_gid
    except Exception:
        return None
