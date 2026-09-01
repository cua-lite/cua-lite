"""Docker-container-per-worker emulator container + factory for androidworld.

Each container is a self-contained androidworld env: it hosts the
emulator AND the env_launcher-managed Python env AND the a11y gRPC
server (via ``docker/server.py``). Because every flow that needs
container→host reverse connectivity stays inside the container's
network namespace, this works on rootless docker even with the
``slirp4netns --disable-host-loopback`` default.

Design:

  - Network: default bridge network. Each container gets its own
    private net ns. The emulator's console/adb/gRPC always bind the
    same in-container ports (:attr:`_CPORT_CONSOLE` etc.) — no
    cross-container port contention, no need for a centralized
    locking pool. Only one port is exposed to the host:
    :attr:`api_port`, mapped via ``docker -p`` to the in-container
    server's :attr:`_CPORT_API`. That's the single channel the host
    process uses to drive the env.

  - Adb (Pattern Y): the in-container env-server hands env_launcher
    the in-container adb binary path directly, so all adb traffic
    stays inside the container's network namespace. The Android
    emulator's adbd binds 127.0.0.1 in-container; routing through
    docker exec sidesteps that completely. Host-side code talks to
    the env via HTTP-RPC, not adb.

  - In-container env-server: ``docker/server.py`` (FastAPI, baked
    into the image as ``/usr/local/bin/server.py``) hosts env_launcher
    + a TaskEval instance. The host's main.py drives it via HTTP-RPC
    (``_RemoteEnv`` / ``_RemoteTask`` proxies). a11y_grpc_wrapper's
    reverse gRPC server is created in that same in-container Python
    process, so the a11y forwarder APK's call to ``10.0.2.2:<port>``
    terminates at the container's own loopback. No host loopback ever
    needed.

  - Boot: ``-snapshot default_boot -no-snapshot-save`` — Quick Boot from
    the ``default_boot`` snapshot baked into the image at build time
    (``docker/apps.sh`` runs ``adb emu avd snapshot save default_boot``
    after installing the apps). We deliberately do NOT pass ``-read-only``
    (it would restore RAM but drop the userdata overlay on snapshot
    reload). A snapshot RAM-restore is ~5-20 s vs a true cold boot's
    60-120 s; see ``start()`` for the exact ``emulator`` flags.

Container lifecycle: **spawn per acquire, destroy on release** by default
(``_MAX_RESETS_PER_CONTAINER`` / ``max_resets_per_container`` = 0 — a
deliberate cleanliness tradeoff). The first spawn still pays ~35 s (boot
the snapshot off disk + the readiness pipeline); set
``max_resets_per_container`` >= 1 to reuse a live container across
episodes via ``reset_snapshot()`` (~5 s snapshot reload — see ``main.py``).

Port allocation: only the host-side :attr:`api_port` is dynamic; goes
through ``lite/gym/utils/backend/ports.py``'s shared flock + reservation
file. See the port-range map at the top of that file.

Rebuild semantics
-----------------
Image is built from ``lite/gym/envs/androidworld/docker/``. Changes
to ANY file under that directory — ``Dockerfile``, ``apps.sh``,
``server.py``, pip-installed deps — require
``bash lite/gym/envs/androidworld/scripts/install.sh rebuild``
to land in spawned containers. Changes to host-side code in this
file or ``main.py`` (the ``_RemoteEnv`` / ``_RemoteTask`` proxies,
the env-server orchestration) take effect on the next env-server
restart without rebuilding the image.

Run (smoke, requires ``cua-lite/androidworld:latest`` built via
``scripts/install.sh``):

    uv run python -c "
    from lite.gym.envs.androidworld.container import AndroidWorldContainerFactory
    factory = AndroidWorldContainerFactory(task_id='SmokeTest')
    c = factory.acquire()
    print('booted:', c.name, 'api:', c.base_url)
    print('packages installed:', len(c.adb_shell('pm', 'list', 'packages', '-3').splitlines()))
    c.destroy()
    "
"""

from __future__ import annotations

import logging
import os
import subprocess
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

# ENV_DIR is the env package root (package root); configs/ + docker/ are its siblings.
ENV_DIR = str(Path(__file__).resolve().parents[0])
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# ANDROID_WORLD_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
#: Image tag. Must match what ``scripts/install.sh`` tags.
DEFAULT_IMAGE = CFG.env_kwargs["image"]
# --- server_kwargs (per-deployment) ---
#: Boot-completion deadline. Cold boot inside a container is typically
#: 30-60 s; allow 4 min for slow hosts before failing the rollout.
BOOT_TIMEOUT_S = CFG.server_kwargs["boot_timeout"]
#: In-container HTTP API healthz poll deadline.
API_READY_TIMEOUT_S = CFG.server_kwargs["api_timeout"]
#: Android-system-services readiness deadline. ``sys.boot_completed=1``
#: fires when zygote starts (~30-60 s after docker run) but the system
#: services that ``env_launcher.load_and_setup_env()`` depends on —
#: PackageManager (used for ``pm path`` / ``pm install`` / ``pm grant``)
#: and AccessibilityManagerService (used for a11y_grpc reverse channel
#: registration) — become usable several seconds *later*. Under host
#: contention (KVM/vCPU starvation) that gap can stretch to 30 s+, and
#: an early /init call lands while ``service check accessibility`` still
#: returns ``not found`` → ``env_launcher`` raises → /init handler 500.
#: We probe both services explicitly and only declare ready when both
#: respond. See devs/envs/androidworld.md "Known Issue" for the full
#: discovery trail.
ANDROID_READY_TIMEOUT_S = CFG.server_kwargs["android_ready_timeout"]
#: Per-container memory cap. Each androidworld emulator typically uses
#: ~3 GB (emulator + adb + in-container server); 8g leaves ~5 GB headroom.
#: Sized so a v384 host with ~128 concurrent emulators (L2 + mobilegym
#: pool combined) stays within ~1 TB cgroup ceiling. Disable swap
#: (memory == memory-swap) and swappiness=0 so an OOM kills the offender
#: instead of dragging the host into swap thrash. Do not shrink it: an OOM
#: here targets the in-container python (highest rss after qemu — qemu itself
#: is mlocked) without touching the ``sleep 86400`` PID 1, so docker keeps the
#: container alive with no RPC server and every ``/step`` 500s. 8 GiB leaves
#: comfortable headroom while still bounding sticky-leak blast radius under a
#: 1.5 TB host budget.
_MEMORY_LIMIT = CFG.server_kwargs["memory_limit"]
# ============================================================================


# ── Constants ────────────────────────────────────────────────────────────────

#: AVD name baked into the image (see Dockerfile).
DEFAULT_AVD_NAME = "lite_avd_androidworld"

# Bind-mount the host's checked-in ``docker/server.py`` over the image's
# baked ``/usr/local/bin/server.py``. Keeps the running server in lock-step
# with the source tree so a server.py edit takes effect on the next docker
# run without needing a full image rebuild (the rebuild loop is expensive
# because the apps-install stage requires KVM in a privileged builder).
# Read-only so an accidental in-container write can't corrupt the host
# source. Resolved relative to this file: docker/ is a sibling of utils/.
_HOST_SERVER_PY = (Path(__file__).resolve().parent / "docker" / "server.py")

# In-container ports — constant across all containers because each
# container has its own private network namespace under bridge mode.
# Only the api port is exposed to the host (via docker -p mapping).
_CPORT_CONSOLE = 5554
_CPORT_ADB = 5555         # Android convention: console + 1
_CPORT_GRPC = 8554        # cua-lite convention: console + 3000
_CPORT_API = 9554         # docker/server.py listens here

#: Host-side port range for the published API port. ``backend.ports`` guards
#: cross-process allocation. See port-range map at the top of
#: ``lite/gym/utils/backend/ports.py``.
_API_PORT_RANGE = (9554, 9700)

# Paths inside the cua-lite/androidworld image (set by Dockerfile ENV +
# sdkmanager install).
_EMU_BIN = "/root/Android/Sdk/emulator/emulator"
_ADB_BIN = "/root/Android/Sdk/platform-tools/adb"


# Container-name prefix matches the repo-wide
# ``lite-env-{server_port?}-{token_hash?}-{session_id}-{env_id}-{task_id?}-{suffix}``
# convention so the env server's ``DELETE /instances`` and tier-1 cleanup.sh
# both find these. Name assembly lives in :mod:`lite.gym.utils.config.naming`
# (imported above).


# ── Container ────────────────────────────────────────────────────────────────

@dataclass
class AndroidWorldContainer(LiteContainerBase):
    """One emulator + in-container env-server inside one bridge-mode docker container.

    Each ``AndroidWorldContainerFactory.acquire()`` returns a freshly
    spawned container with:
      - the emulator booted (in-container ports
        :data:`_CPORT_CONSOLE` / :data:`_CPORT_ADB` / :data:`_CPORT_GRPC`)
      - ``docker/server.py`` running on in-container port
        :data:`_CPORT_API`, exposed to the host as :attr:`api_port`

    All in-container ports are fixed constants — each container has
    its own private network namespace, so there's no cross-container
    conflict. Only ``api_port`` is dynamic on the host.

    Public interface:

      ``console_port`` / ``adb_port`` / ``grpc_port``
          Constants — the in-container values that env_launcher
          consumes inside the in-container server.
      ``adb_serial``
          ``"emulator-5554"``. The in-container adb server discovers
          the emulator under this serial.
      ``base_url``
          Host-side URL of the in-container env-server.
    """

    name: str
    api_port: int
    image: str = DEFAULT_IMAGE
    avd_name: str = DEFAULT_AVD_NAME
    env_id: str = "androidworld"

    #: ``api_port`` was reserved via ``backend.ports``; released on destroy().
    _ports_owned: tuple[int, ...] = field(default=(), repr=False)

    rm_label: ClassVar[str] = "androidworld"

    # Fixed in-container emulator ports — exposed as plain attributes
    # (init=False) so callers can read them without going through a
    # property.
    console_port: int = field(default=_CPORT_CONSOLE, init=False)
    adb_port: int = field(default=_CPORT_ADB, init=False)
    grpc_port: int = field(default=_CPORT_GRPC, init=False)
    adb_serial: str = field(
        default=f"emulator-{_CPORT_CONSOLE}", init=False
    )

    @property
    def base_url(self) -> str:
        """Host URL of the in-container HTTP RPC server.

        The single channel through which the host process drives the
        container — forward traffic only, so rootless docker's
        ``slirp4netns --disable-host-loopback`` restriction (which
        blocks the reverse direction) doesn't apply.
        """
        return f"http://localhost:{self.api_port}"

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """``docker run`` (bridge net, ``-p api_port:9554``) + launch
        emulator + boot wait + spawn in-container server + write the
        adb wrapper script."""
        # Register BEFORE the container-creating call — covers the whole boot
        # path below (docker run + the multi-minute emulator boot waits): a
        # KeyboardInterrupt there is a BaseException that acquire()'s
        # ``except Exception`` won't catch; destroy() is idempotent, so
        # failures/retries stay clean.
        self._register()
        kvm_gid = _kvm_gid()
        # Read-only bind-mount keeps the host's docker/server.py canonical —
        # see _HOST_SERVER_PY for the rationale (avoids the 20+ min KVM-in-
        # builder rebuild loop for server.py edits).
        volumes = ([f"{_HOST_SERVER_PY}:/usr/local/bin/server.py:ro"]
                   if _HOST_SERVER_PY.exists() else [])
        # The docker-create semaphore is acquired in acquire() (wraps the
        # WHOLE boot path), not here. -p api_port:9554 is the only exposed
        # port: adb goes via `docker exec` (Pattern Y); gRPC + a11y reverse
        # channels live inside the container's netns. IPv6 is disabled so
        # qemu's emulator-modem chardev falls back to IPv4 127.0.0.1 —
        # glibc's getaddrinfo rejects the bracket-less ``::1:<port>`` string
        # qemu passes, the emulator then boots WITHOUT a SIM (modem
        # OUT_OF_SERVICE) and every Simple SMS task silently breaks.
        # memory == memory-swap → no swap: OOM kill preferred over swap
        # thrash that would drag the whole host down (_MEMORY_LIMIT comment).
        docker_run_detached(
            name=self.name, image=self.image,
            auto_remove=True,
            devices=("/dev/kvm",),
            sysctls=("net.ipv6.conf.all.disable_ipv6=1",
                     "net.ipv6.conf.lo.disable_ipv6=1"),
            memory=_MEMORY_LIMIT,
            group_add=(kvm_gid,) if kvm_gid is not None else (),
            volumes=volumes,
            ports=((self.api_port, _CPORT_API),),
            command=("sleep", "86400"),
            timeout=180.0,
            label=self.env_id,
        )

        # Launch the emulator inside the container.
        # ``-snapshot default_boot -no-snapshot-save`` loads the Quick Boot
        # snapshot baked by ``apps.sh`` at image build time (~5 s vs
        # ~30-60 s cold) and prevents writing snapshot updates back on
        # shutdown — every container starts from the baked baseline.
        # We deliberately do NOT pass ``-read-only`` (which would route
        # writes to a tmpfs overlay). ``-read-only`` breaks mid-runtime
        # ``adb emu avd snapshot load default_boot`` because the overlay
        # is NOT included in QuickBoot snapshot reload semantics — only
        # RAM is restored. Without ``-read-only``, writes hit
        # userdata-qemu.img directly, and the snapshot reload restores
        # both RAM AND the userdata delta → state truly returns to the
        # baked baseline. Container ephemeral (``docker rm`` on release)
        # so userdata bloat between resets is bounded by the K=N
        # force-respawn at the env layer.
        # Ports bind on the container's loopback only — that's fine
        # because env_launcher (also in-container) consumes them
        # locally.
        emu_cmd = (
            f"cd /root && {_EMU_BIN} -avd {self.avd_name} "
            "-snapshot default_boot -no-snapshot-save "
            "-no-window -no-audio -no-metrics "
            "-gpu swiftshader_indirect "
            f"-ports {_CPORT_CONSOLE},{_CPORT_ADB} "
            f"-grpc {_CPORT_GRPC} "
            ">/tmp/emu.log 2>&1"
        )
        r = subprocess.run(
            # bash -c (not -lc): login shell re-sources /etc/profile and
            # can clobber the Dockerfile's ENV PATH. See androidlab's
            # ``container._start_server`` for the failure-mode history.
            ["docker", "exec", "-d", self.name, "bash", "-c", emu_cmd],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            stderr = (r.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"docker exec (launch emulator) failed for {self.name}: {stderr}"
            )

        self._wait_until_booted()
        self._wait_until_android_ready()
        self._grant_missing_perms()
        self._start_server()
        self._wait_until_api_ready()
        # NOTE: _register() happens right after ``docker run`` above (so
        # the atexit backstop covers the boot window), NOT here.

    def _grant_missing_perms(self) -> None:
        """Hand-grant runtime perms that upstream ``apps.py`` setup misses.

        Upstream ``MarkorApp.setup()`` accepts Markor's storage permission
        via a UI flow (click NEXT×4 → DONE → OK → "Allow access to manage
        all files"). That UI flow runs at image build time inside our
        ``docker/apps.sh``; on this image it silently failed at the
        "Allow manage all files" step, leaving Markor with
        ``WRITE_EXTERNAL_STORAGE: granted=false`` and
        ``MANAGE_EXTERNAL_STORAGE: default``.

        Without these perms, Markor opens to a permission dialog instead
        of its file list, blocking every Markor task in androidworld's
        eval set. We compensate at container start by issuing the grants
        via adb — equivalent to what the UI flow would have done. Safe
        to re-run; ``pm grant`` is idempotent.
        """
        # Map: package → (runtime perms, appops to allow). Add more here
        # if other UI-driven setups in upstream apps.py fail similarly.
        # ``MANAGE_EXTERNAL_STORAGE`` is the "All files access" appop on
        # API 30+; needed by apps that write outside their sandbox
        # (e.g. Markor opening notes in /sdcard, Retro Music exporting
        # an .m3u playlist to /sdcard/Download).
        grants: list[tuple[str, list[str], list[str]]] = [
            (
                "net.gsantner.markor",
                ["android.permission.WRITE_EXTERNAL_STORAGE",
                 "android.permission.READ_EXTERNAL_STORAGE"],
                ["MANAGE_EXTERNAL_STORAGE"],
            ),
            (
                "code.name.monkey.retromusic",
                ["android.permission.WRITE_EXTERNAL_STORAGE",
                 "android.permission.READ_EXTERNAL_STORAGE"],
                ["MANAGE_EXTERNAL_STORAGE"],
            ),
        ]
        # Per-call cap raised 10→30 s. At c=32+ concurrent emulator boots the
        # rootless-docker daemon serialises ``docker exec ... adb`` setup,
        # pushing individual adb command latencies from <100 ms baseline to
        # 5-15 s. adb itself returns immediately; the extra ceiling only
        # kicks in under genuine contention.
        for pkg, runtime_perms, appops_perms in grants:
            for perm in runtime_perms:
                subprocess.run(
                    ["docker", "exec", self.name,
                     _ADB_BIN, "-s", self.adb_serial,
                     "shell", "pm", "grant", pkg, perm],
                    capture_output=True, timeout=30,
                )
            for op in appops_perms:
                subprocess.run(
                    ["docker", "exec", self.name,
                     _ADB_BIN, "-s", self.adb_serial,
                     "shell", "appops", "set", pkg, op, "allow"],
                    capture_output=True, timeout=30,
                )

    def _start_server(self) -> None:
        """Launch ``docker/server.py`` as a long-lived daemon inside
        the container, bound to the in-container API port.

        Backgrounded with ``docker exec -d`` so it lives as long as the
        container. The server hosts env_launcher + task_evals inside the
        container, sidestepping rootless docker's host-loopback block
        (see docker/server.py docstring).
        """
        cmd = (
            f"python /usr/local/bin/server.py "
            f"--port {_CPORT_API} > /tmp/server.log 2>&1"
        )
        # Same docker-exec contention rationale as _grant_missing_perms.
        r = subprocess.run(
            ["docker", "exec", "-d", self.name, "bash", "-c", cmd],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            stderr = (r.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"docker exec (launch server) failed for "
                f"{self.name}: {stderr}"
            )

    def _wait_until_api_ready(self) -> None:
        """Poll ``/healthz`` until uvicorn comes up (typically <5 s)."""
        deadline = time.monotonic() + API_READY_TIMEOUT_S
        last_err: str = ""
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.base_url}/healthz", timeout=2.0
                ) as resp:
                    if resp.status == 200:
                        logger.info(
                            "server ready on %s",
                            self.base_url,
                        )
                        return
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            # 0.5s poll: /healthz is cheap (in-container HTTP), detects
            # readiness ~0.5s sooner per env. At N=64 concurrent boots
            # that's ~30s aggregate wall time saved.
            time.sleep(0.5)

        # Failed — pull the server log out so the traceback is actionable.
        try:
            log_tail = subprocess.run(
                ["docker", "exec", self.name, "tail", "-100",
                 "/tmp/server.log"],
                capture_output=True, timeout=5,
            ).stdout.decode("utf-8", errors="replace")
        except Exception:
            log_tail = "(could not read server.log)"
        # Still warming: the in-container uvicorn just hasn't bound /healthz
        # yet. Recoverable — the client should retry the SAME instance while it
        # finishes coming up (CapacityExhausted → 503 + Retry-After), not get a
        # terminal 500 that ends the episode.
        raise CapacityExhausted.warming(
            f"androidworld server in {self.name} not HTTP-ready in "
            f"{API_READY_TIMEOUT_S}s (last error: {last_err}).\n"
            f"--- server.log tail ---\n{log_tail}"
        )

    def _read_emu_log(self) -> str:
        """Return the last ~4 KB of ``/tmp/emu.log`` (empty on read error)."""
        try:
            r = subprocess.run(
                ["docker", "exec", self.name, "tail", "-c", "4096", "/tmp/emu.log"],
                capture_output=True, timeout=5,
            )
            return (r.stdout or b"").decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _wait_until_booted(self) -> None:
        """Poll ``sys.boot_completed`` inside the container until it returns 1.

        Also fast-fails on KVM permission races: under rootless-docker
        cold-boot bursts, ``--group-add <kvm_gid>`` occasionally still
        fails to grant ``/dev/kvm`` access at runtime (qemu fails on
        ``KVM_CREATE_VM`` or the emulator wrapper's preflight rejects
        it). Both signatures surface within ~5-10 s in
        ``/tmp/emu.log``; without fast-fail the wait would burn the
        full ``BOOT_TIMEOUT_S`` (240 s) of either software-emulation
        crawl or exited-qemu silence. Mirrors
        :meth:`androidlab.AndroidLabContainer._wait_until_booted`.
        """
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        last_state = ""
        while time.monotonic() < deadline:
            try:
                r = subprocess.run(
                    ["docker", "exec", self.name, _ADB_BIN,
                     "-s", self.adb_serial,
                     "shell", "getprop", "sys.boot_completed"],
                    capture_output=True, timeout=30,
                )
                last_state = (r.stdout or b"").decode(errors="replace").strip()
                if last_state == "1":
                    logger.info("emulator booted in container %s", self.name)
                    return
            except subprocess.SubprocessError:
                pass
            # KVM fast-fail: check emu.log for the two known
            # permission-race signatures. Cost: one extra
            # ``docker exec tail`` per poll (~50-200 ms), throttle
            # bounds concurrent pollers.
            log_tail = self._read_emu_log()
            if log_tail and (
                "requires hardware acceleration" in log_tail
                or "doesn't have permissions to use KVM" in log_tail
                or "Could not access KVM kernel module" in log_tail
                or "failed to initialize KVM" in log_tail
            ):
                raise RuntimeError(
                    f"Emulator in {self.name} can't access /dev/kvm "
                    f"despite --group-add. Intermittent rootless-docker "
                    f"race; caller should destroy + retry. Fast-fail "
                    f"saves ~{BOOT_TIMEOUT_S}s of post-failure silence.\n"
                    f"  Log tail:\n{log_tail[-1500:]}"
                )
            # 1s poll: each `getprop sys.boot_completed` is a `docker
            # exec` (~50-200 ms daemon roundtrip). Halved from 2s for
            # ~1s-faster boot detect at the cost of double docker exec
            # rate — the throttle caps concurrent boots so daemon load
            # stays bounded.
            time.sleep(1.0)

        # Boot timed out without a KVM signature — the emulator is just still
        # warming up (cold boot under host contention can exceed the deadline).
        # Recoverable: retrying the same instance plausibly succeeds, so signal
        # warming (503 + Retry-After) rather than a terminal 500.
        log_tail = self._read_emu_log() or "(could not read /tmp/emu.log)"
        raise CapacityExhausted.warming(
            f"emulator in {self.name} not booted in {BOOT_TIMEOUT_S}s "
            f"(last sys.boot_completed={last_state!r}).\n"
            f"--- /tmp/emu.log tail ---\n{log_tail}"
        )

    def _wait_until_android_ready(self) -> None:
        """Poll until Android system services that env_launcher depends on
        are usable — PackageManager + AccessibilityManagerService.

        ``sys.boot_completed=1`` (checked by :meth:`_wait_until_booted`)
        only marks zygote startup. ``env_launcher.load_and_setup_env()``
        further uses ``adb shell pm`` and registers an a11y service over
        gRPC; on a contended host those become available several seconds
        *later*, so calling /init right after sys.boot_completed races
        and the in-container ``env_launcher`` raises → /init returns
        500 → host counts the task unfinished. See devs/envs/
        androidworld.md "Known Issue" for the full discovery (2026-
        05-26 on commit a0160069). Probing the two services directly
        closes that gap.
        """
        deadline = time.monotonic() + ANDROID_READY_TIMEOUT_S
        last_pm = last_a11y = ""
        while time.monotonic() < deadline:
            # PackageManager usable? ``pm path`` is idempotent + cheap.
            pm_ok = False
            try:
                r = subprocess.run(
                    ["docker", "exec", self.name, _ADB_BIN,
                     "-s", self.adb_serial,
                     "shell", "pm", "path", "com.android.settings"],
                    capture_output=True, timeout=15,
                )
                last_pm = (r.stdout or b"").decode(errors="replace").strip()
                pm_ok = r.returncode == 0 and last_pm.startswith("package:")
            except subprocess.SubprocessError:
                pass

            # AccessibilityManagerService registered?
            a11y_ok = False
            try:
                r = subprocess.run(
                    ["docker", "exec", self.name, _ADB_BIN,
                     "-s", self.adb_serial,
                     "shell", "service", "check", "accessibility"],
                    capture_output=True, timeout=15,
                )
                last_a11y = (r.stdout or b"").decode(errors="replace").strip()
                # adb prints ``Service accessibility: found`` when up,
                # ``Service accessibility: not found`` while still
                # registering.
                a11y_ok = r.returncode == 0 and "found" in last_a11y and "not found" not in last_a11y
            except subprocess.SubprocessError:
                pass

            if pm_ok and a11y_ok:
                logger.info("android services ready in %s", self.name)
                return
            time.sleep(1.0)

        # Still warming: zygote is up but PackageManager / a11y haven't
        # finished rehydrating yet. Recoverable — retry the same instance.
        raise CapacityExhausted.warming(
            f"android services in {self.name} not ready in "
            f"{ANDROID_READY_TIMEOUT_S}s "
            f"(pm.path={last_pm!r}, service.accessibility={last_a11y!r})."
        )

    # ── Pattern Y adb passthrough ────────────────────────────────────────

    def adb(
        self, *args: str, timeout: float = 15.0
    ) -> subprocess.CompletedProcess:
        """Run ``adb -s emulator-5554 <args>`` inside the container.

        Used by smoke tests and any host-side debugging that needs to
        poke the in-container emulator's adb. Runtime ``env_launcher``
        lives inside the container and talks to its local adb directly,
        so this path is rarely on the hot loop.
        """
        return subprocess.run(
            ["docker", "exec", self.name, _ADB_BIN,
             "-s", self.adb_serial, *args],
            capture_output=True, timeout=timeout, check=False,
        )

    def adb_shell(self, *args: str, timeout: float = 15.0) -> str:
        r = self.adb("shell", *args, timeout=timeout)
        return (r.stdout or b"").decode("utf-8", errors="replace").strip()

    def _exec_out(self, *args: str, timeout: float = 15.0) -> bytes:
        """``adb exec-out`` — binary-safe (screencap, raw XML file cat)."""
        r = subprocess.run(
            ["docker", "exec", self.name, _ADB_BIN,
             "-s", self.adb_serial, "exec-out", *args],
            capture_output=True, timeout=timeout, check=False,
        )
        return r.stdout or b""

    # ── Snapshot reset ───────────────────────────────────────────────────

    def reset_snapshot(self) -> bool:
        """Reload the baked ``default_boot`` Quick Boot snapshot.

        Used when ``max_resets_per_container`` allows snapshot reuse: instead
        of destroying the container on episode end and cold-booting a new one
        (~30-60 s per spawn), we hot-reload the snapshot state (~5 s) and let
        the host re-instantiate the next task via ``task.initialize_task``.

        Mirror of androidlab's ``AndroidLabContainer.reset_snapshot``.
        Issues ``adb emu avd snapshot load default_boot`` via the
        container's adb daemon; settles 1 s after for a11y service +
        settings observer to re-attach. Returns False on adb error so
        the caller (``AndroidWorldEnv.reset``) can fall back to
        destroy + cold-spawn.

        Prerequisite: the image must have ``default_boot`` baked in
        (see ``apps.sh``'s ``adb emu avd snapshot save`` step). Images
        built before that change will silently fail this call —
        symptom: every reset re-spawns, total wall regresses to MVP
        levels.
        """
        try:
            r = subprocess.run(
                ["docker", "exec", self.name,
                 _ADB_BIN, "-s", self.adb_serial,
                 "emu", "avd", "snapshot", "load", "default_boot"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning("reset_snapshot timeout on %s", self.name)
            return False
        if r.returncode != 0:
            logger.warning(
                "reset_snapshot failed on %s (rc=%d): %s",
                self.name, r.returncode, (r.stderr or "")[:300],
            )
            return False
        # Settle window. Snapshot load is fast (~2 s qemu side), but the
        # Android stack inside the restored VM needs more time to come
        # back online:
        # * SurfaceFlinger re-attaches GL/Vulkan contexts
        # * a11y service rebinds the reverse-gRPC channel to env_launcher
        # * settings provider re-reads /data/system/users/0/settings_*.xml
        # * launcher repaints
        # Empirically (matrix M1 5-rep test at 1s settle), some same-task
        # snapshot reuses on AudioRecorderRecordAudio gave reward=0 ~50 % of
        # the time — symptom of the a11y dump or initial UI capture
        # racing service re-attach. Bumped to 3 s here. androidlab's
        # 1 s works because its in-container RPC layer absorbs more
        # of the race; androidworld goes adb/a11y directly so the
        # settle needs to wait for the whole stack.
        time.sleep(3.0)

        # Defense-in-depth: re-apply ``hide_automation_ui`` settings here.
        # The PRIMARY fix is in ``AndroidWorldEnv.reset()`` *after*
        # ``self._env.reset(go_home=True)`` — that's the one that actually
        # sticks, because ``coordinator.rl_reset`` always relaunches
        # device_settings via ``_launch_simulator`` and would overwrite
        # whatever we set here. This call is a belt-and-suspenders for
        # the rare case where the relaunch path is skipped (e.g. a
        # future android_env that keeps the simulator hot across resets).
        # ``android_env``'s device_settings defaults
        # ``show_pointer_location=True`` / ``show_touches=True``
        # (config_classes.py:43-45); without these reapplied to 0, the
        # GPU pointer-trace overlay appears in obs.image AND
        # intercepts/shifts input delivery so baseline-recorded action
        # coordinates land on wrong UI elements. Empirically: M1 5-rep
        # AudioRecorderRecordAudio dropped to 1/5 success before the
        # ``main.py`` fix; 10/10 after.
        try:
            self.adb_shell("settings", "put", "system", "pointer_location", "0",
                           timeout=10.0)
            self.adb_shell("settings", "put", "system", "show_touches", "0",
                           timeout=10.0)
        except Exception as e:
            logger.warning("post-snapshot hide_automation_ui failed: %s", e)
        return True

    # ── Lifecycle tear-down ──────────────────────────────────────────────

    # ``release`` (main.py's emulator-handle spelling) is LiteContainerBase's
    # class-level alias for the template ``destroy()``.

    def _postmortem_snapshot(self, reason: str) -> None:
        """Diagnostic dump of in-container state to
        ``$CUA_LITE_DEBUG_POSTMORTEM_DIR/<name>__<reason>__<ts>/``.

        Production-gated, not "remove before release": this method
        always exists, but every call site reads
        ``CUA_LITE_DEBUG_POSTMORTEM_DIR`` first and silently no-ops
        when it's unset (the production default). Set the env var on
        a single env-server instance to capture postmortems without
        affecting any other instance.

        Robust extraction strategy: ``docker cp`` works at the container
        filesystem layer and doesn't require any process inside the
        container to be alive, so it survives the "alive container, dead
        workload" case that defeats ``docker exec``. We copy out
        ``/tmp/server.log`` and ``/tmp/emu.log`` (the two files that
        carry actual failure context) via ``docker cp``, and add
        ``docker exec``-based ps tree + meminfo as best-effort — they
        fail-soft if exec is unhealthy.

        Output layout::

            <pm_dir>/<container>__<reason>__<ts>/
                meta.txt          # reason, timestamps, healthz, ps tree
                server.log        # in-container /tmp/server.log (full)
                emu.log           # in-container /tmp/emu.log (full)
                inspect.json      # docker inspect output

        Called from two paths:

        * :meth:`destroy` — final state of every container being torn
          down by the env-server.
        * ``lite.gym.remote.server`` outer-retry exhaustion —
          captures the moment env-server gives up on a /step or
          /reset after the transient-error retry budget is spent.
        """
        pm_dir = os.environ.get("CUA_LITE_DEBUG_POSTMORTEM_DIR")
        if not pm_dir:
            return
        try:
            ts = int(time.time())
            out_dir = os.path.join(pm_dir, f"{self.name}__{reason}__{ts}")
            os.makedirs(out_dir, exist_ok=True)
            # ── docker cp: copies file from container filesystem to
            # host, survives dead in-container processes. Each cp is
            # independent — one failing doesn't kill the rest.
            for src in ("/tmp/server.log", "/tmp/emu.log"):
                dst = os.path.join(out_dir, os.path.basename(src))
                cp = subprocess.run(
                    ["docker", "cp", f"{self.name}:{src}", dst],
                    capture_output=True, timeout=10,
                )
                if cp.returncode != 0:
                    # mark missing so we can tell "exec failed" from
                    # "file genuinely not there" in offline analysis.
                    with open(dst + ".missing", "w") as f:
                        f.write(cp.stderr.decode(errors="replace"))
            # ── docker inspect: always works while container exists.
            inspect_out = subprocess.run(
                ["docker", "inspect", self.name,
                 "--format", "{{json .State}}"],
                capture_output=True, timeout=5,
            ).stdout.decode()
            with open(os.path.join(out_dir, "inspect.json"), "w") as f:
                f.write(inspect_out)
            # ── docker exec ps + meminfo: best-effort, often fails on
            # dead containers but gives extra signal when it works.
            exec_out = subprocess.run(
                ["docker", "exec", self.name, "sh", "-c",
                 "ps -ef --forest 2>/dev/null;"
                 " echo '== meminfo =='; cat /proc/meminfo | head -8 2>/dev/null;"
                 " echo '== /tmp ls =='; ls -la /tmp/ 2>/dev/null;"],
                capture_output=True, timeout=10,
            )
            hz_code = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-m", "3",
                 "-w", "%{http_code}",
                 f"http://localhost:{self.api_port}/healthz"],
                capture_output=True, timeout=5,
            ).stdout.decode().strip()
            with open(os.path.join(out_dir, "meta.txt"), "w") as f:
                f.write(f"reason: {reason}\n")
                f.write(f"container: {self.name}\n")
                f.write(f"api_port: {self.api_port}\n")
                f.write(f"timestamp: {ts}\n")
                f.write(f"healthz_from_host: {hz_code}\n")
                f.write("\n== docker exec output (best-effort) ==\n")
                f.write(exec_out.stdout.decode(errors="replace"))
                if exec_out.returncode != 0:
                    f.write(
                        f"\n[exec rc={exec_out.returncode}: "
                        f"{exec_out.stderr.decode(errors='replace')}]\n"
                    )
            logger.warning("POSTMORTEM saved: %s (reason=%s)", out_dir, reason)
        except Exception as e:
            logger.warning("postmortem dump failed for %s (reason=%s): %s",
                           self.name, reason, e)

    def _pre_destroy(self) -> None:
        # destroy() itself is LiteContainerBase's template (rm -f -v @ 60 s —
        # qemu's vCPU + KVM-session unwind after SIGKILL can take 30-90 s on a
        # stuck emulator; a 30 s ceiling caused false-positive "destroy timed
        # out" → orphan accumulation — → port release → de-register).
        # DEBUG hook: dump only when in-container python is genuinely dead
        # at destroy time. We do NOT dump on every destroy — that path
        # (~86 docker cp + docker exec per rollout) was empirically the
        # dominant docker-daemon-API load source under c=32 concurrent
        # task teardowns, pushing in-container adb commands past their
        # 8s timeout and triggering cascading /step failures. Real
        # mid-task failures are now captured earlier and more reliably
        # by the ``_postmortem_snapshot`` call inside main.py's
        # ``_dispatch_action`` except-clause (fires at the moment of
        # the Connection-refused, before the container is being
        # destroyed → docker cp can still read /tmp/server.log).
        if os.environ.get("CUA_LITE_DEBUG_POSTMORTEM_DIR"):
            try:
                ps_out = subprocess.run(
                    ["docker", "exec", self.name,
                     "sh", "-c", "pgrep -fc '[s]erver.py'"],
                    capture_output=True, timeout=5,
                ).stdout.decode().strip()
                if ps_out == "0":
                    self._postmortem_snapshot("destroy_py_dead")
            except Exception:
                pass    # never block destroy on probe failure


# ── Factory ──────────────────────────────────────────────────────────────────

def _wait_for_emulator_exit(container: AndroidWorldContainer) -> bool:
    """Block an internal retry until the failed attempt's qemu has exited."""
    deadline = time.monotonic() + container.rm_timeout_s
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container.name],
                capture_output=True,
                text=True,
                timeout=min(5.0, max(0.1, deadline - time.monotonic())),
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None:
            if result.returncode == 0 and result.stdout.strip().lower() == "false":
                return True
            if result.returncode != 0 and (
                "No such object" in result.stderr
                or "No such container" in result.stderr
            ):
                return True
        time.sleep(1.0)
    return False


def _block_overlapping_retry(container: AndroidWorldContainer) -> None:
    if not _wait_for_emulator_exit(container):
        # destroy() is intentionally best-effort and de-registers. Restore the
        # atexit backstop for this still-running failed attempt; the env-server
        # drift reaper independently sees it as an orphan.
        container._register()
        raise CapacityExhausted.warming(
            what=f"failed {container.env_id} attempt {container.name} is still running; "
                 "refusing to overlap its emulator with an internal retry",
        )


class AndroidWorldContainerFactory:
    """Spawn one fresh ``AndroidWorldContainer`` per :meth:`acquire`; caller
    calls ``container.destroy()`` (or relies on atexit) to release.

    Holds shared kwargs (``image``, ``avd_name``, ``session_id``,
    ``token_hash``, ``server_port``) so the env constructs the factory
    once and calls ``acquire()`` whenever a fresh container is needed
    (every reset under ``_MAX_RESETS_PER_CONTAINER=0``; every K-th reset
    above that). Cross-episode reuse is **not** done here — the value-add
    is :meth:`AndroidWorldContainer.reset_snapshot` (AVD snapshot reload)
    on a kept-alive container. If you want a real free-list pool, add
    one alongside this factory; the existing call sites would only need
    to consume a different acquire path.
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        avd_name: str = DEFAULT_AVD_NAME,
        task_id: str | None = None,
        session_id: str | None = None,
        token_hash: str | None = None,
        server_port: int | None = None,
        env_id: str = "androidworld",
    ):
        self.image = image
        self.avd_name = avd_name
        self.task_id = task_id
        # session_id + token_hash + server_port flow from gym.make →
        # ``env.identity`` (``EnvIdentity`` dataclass, set by
        # ``lite.gym.registry``). The env reads it via
        # ``getattr(self, "identity", None)`` and forwards the three
        # fields here so container naming aligns with the env-server's
        # cleanup scoping AND two same-token env-server instances on
        # the same host stay mutually isolated via the per-instance
        # ``server_port`` segment.
        self.session_id = session_id
        self.token_hash = token_hash
        self.server_port = server_port
        self.env_id = env_id

    def _make_name(self, api_port: int) -> str:
        """Canonical name: ``lite-env-[{server_port}-][{token_hash}-]{session_id}-androidworld-[{task_id}-]{api_port}``.

        Full segment contract + sanitization lives in
        :func:`lite.gym.utils.config.naming.format_container_name`. ``api_port``
        is the unique per-instance suffix (in-container emulator ports
        are constant; each container binds a different host port).
        """
        return _format_container_name(
            env_id=self.env_id,
            task_id=self.task_id,
            suffix=str(api_port),
            session_id=self.session_id or os.environ.get("SESSION_ID"),
            token_hash=self.token_hash,
            server_port=self.server_port,
        )

    def acquire(self, *, max_attempts: int = 3) -> AndroidWorldContainer:
        """Reserve one host-side API port, ``docker run``, boot emulator
        and in-container server, return.

        Retries up to ``max_attempts`` times on transient boot failures
        (emulator cold-boot timeout, adb-not-ready, in-container API
        readiness timeout). With KVM fast-fail in ``_wait_until_booted``
        each failed attempt aborts within ~5-10 s instead of waiting
        out the full ``BOOT_TIMEOUT_S``, so 3 attempts cap worst-case
        failure cost at ~30 s rather than ~720 s — parity with
        :meth:`androidlab.AndroidLabContainer.acquire`. Failed
        attempts are torn down completely before the next attempt —
        fresh port, fresh container name.
        ``RuntimeError`` + warming ``CapacityExhausted`` are retried; image-missing / docker-daemon
        errors propagate immediately.

        Under stress (32+ concurrent acquires fighting for KVM + disk
        I/O), emulator cold-boot can transiently exceed the boot
        deadline; the retry usually succeeds because the second attempt
        sees lower contention.

        Concurrent acquires across the whole process are gated by
        :func:`lite.gym.remote.admission.docker_create_slot` — a
        process-wide threading semaphore (default 8) that bounds
        ``docker run`` parallelism so the daemon doesn't thrash. The
        hold spans the WHOLE boot pipeline (docker run + emulator
        boot + API ready), not just ``docker run`` itself, because
        the disk-IO bottleneck during emulator boot is heavier than
        the daemon-side bottleneck.
        """
        from lite.gym.remote.admission import docker_create_slot

        def _build() -> AndroidWorldContainer:
            api_port = allocate_ports(
                n=1,
                range_start=_API_PORT_RANGE[0],
                range_end=_API_PORT_RANGE[1],
            )[0]
            name = self._make_name(api_port)
            # Nuke any same-named zombie from a crashed prior run before we
            # spawn a fresh container with the same name. ``docker run -d``
            # would otherwise fail with "name already in use". 60 s matches
            # ``destroy()`` — same qemu-vCPU-unwind reason.
            subprocess.run(_rm_argv(name), capture_output=True, timeout=60)
            return AndroidWorldContainer(
                name=name,
                api_port=api_port,
                image=self.image,
                avd_name=self.avd_name,
                env_id=self.env_id,
                _ports_owned=(api_port,),
            )

        def _start(c: AndroidWorldContainer) -> None:
            # structural docker semaphore — bounds concurrent ``docker run``
            # so the daemon doesn't thrash. Held across c.start() (docker run
            # + initial readiness wait) so the daemon API + NVMe burst is
            # serialized end-to-end.
            with docker_create_slot():
                c.start()

        # Retry classification (transient RuntimeError/CapacityExhausted →
        # fresh build + retry; else destroy + raise) is single-sourced in
        # boot_with_retry.
        return boot_with_retry(
            _build, start=_start, max_attempts=max_attempts, label=self.env_id,
            retry_barrier=_block_overlapping_retry,
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _kvm_gid() -> int | None:
    """Host kvm group gid, for ``--group-add`` so the in-container user
    can access ``/dev/kvm``. ``None`` if the kvm group doesn't exist (in
    which case we just skip ``--group-add``; the container's root may
    already have access)."""
    try:
        import grp
        return grp.getgrnam("kvm").gr_gid
    except Exception:
        return None
