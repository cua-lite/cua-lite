"""One local WindowsAgentArena VM per CUA-Lite environment instance.

Owns the QEMU-in-Docker container lifecycle: boot the prepared Windows 11 qcow2
(snapshot-restore when a ready snapshot is present, else cold boot), expose the
guest bridge, and tear the container down. Driven by ``main.py``; not run
directly. Prerequisite: ``scripts/install.sh`` has prepared the runner + qcow2.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from lite.gym.errors import CapacityExhausted
from lite.gym.remote.admission import docker_create_slot_async
from lite.gym.utils.backend.docker import _rm_argv
from lite.gym.utils.backend.ports import allocate_ports
from lite.gym.utils.config.identity import EnvIdentity
from lite.gym.utils.config.naming import container_name_prefix, format_container_name

logger = logging.getLogger(__name__)

BRIDGE_PORT = 5050
NOVNC_PORT = 8006
GUEST_IP = "20.20.20.21"
SLOT_ORPHAN_AGE_S = 600.0


async def _run(
    *args: str,
    check: bool = True,
    timeout: float | None = None,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        raise TimeoutError(f"command timed out: {' '.join(args)}") from None
    out = stdout.decode("utf-8", errors="replace")
    err = stderr.decode("utf-8", errors="replace")
    if check and proc.returncode != 0:
        detail = (err or out).strip()
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n{detail[-4000:]}"
        )
    return proc.returncode or 0, out, err


def reap_runtime_slots(
    runtime_root: Path,
    *,
    server_port: int | None,
    boot: bool,
) -> int:
    """Remove old slot directories after their owning container is gone."""
    if server_port is None:
        # Without a server port the naming prefix is only ``lite-env-``, which
        # would scan every env-server's WAA slots on the host. Direct-mode manual
        # cleanup is owned by scripts/cleanup.sh, where SESSION_ID provides scope.
        return 0
    slots_root = runtime_root.expanduser().resolve() / "slots"
    if not slots_root.is_dir():
        return 0
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    container_names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    prefix = container_name_prefix(server_port=server_port)
    now = time.time()
    removed = 0
    for slot in slots_root.iterdir():
        if not slot.is_dir() or not slot.name.startswith(prefix):
            continue
        if "-waa-" not in slot.name:
            continue
        if slot.name in container_names:
            continue
        if not boot and now - slot.stat().st_mtime <= SLOT_ORPHAN_AGE_S:
            continue
        shutil.rmtree(slot, ignore_errors=True)
        removed += 1
    return removed


@dataclass(frozen=True, slots=True)
class QemuConfig:
    base_disk: Path
    runner_image: str
    runtime_root: Path
    assets_dir: Path
    snapshot_dir: Path
    vcpus: int
    memory_gb: int
    shm_size: str
    bind_address: str
    ready_timeout_s: float
    readiness_poll_interval_s: float


def _snapshot_ready(cfg: QemuConfig) -> bool:
    """Whether a usable ready-snapshot exists for this base disk — if so, every VM
    restores from it (~15-50s) instead of cold-booting Windows (~60-90s).

    There is no on/off knob: a present, matching snapshot is always used. Returns
    False (→ cold boot) if the bundle is missing/incomplete or was built for a
    different base disk; never raises, so a stale/missing snapshot degrades
    gracefully. Build the bundle with scripts/utils/prepare_snapshot.py (install.sh
    does this automatically).
    """
    snap = cfg.snapshot_dir
    required = [snap / "ready.qcow2", snap / "ready.state", snap / "device", snap / "manifest.json"]
    if not all(p.exists() for p in required):
        logger.debug("no WAA ready snapshot at %s; cold-booting", snap)
        return False
    try:
        manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("base_disk_size") != cfg.base_disk.stat().st_size:
            logger.warning("WAA snapshot is stale (base disk changed); cold-booting")
            return False
    except (OSError, ValueError):
        return False
    return True


class QemuInstance:
    """Creates a disposable qcow2 overlay and runs it in the WAA bridge image."""

    def __init__(
        self,
        *,
        config: QemuConfig,
        task_id: str,
        identity: EnvIdentity,
    ) -> None:
        self._config = config
        self._task_id = task_id
        self._identity = identity
        self.name: str | None = None
        self.slot_root: Path | None = None
        self.bridge_url: str | None = None
        self.novnc_url: str | None = None

    async def start(self) -> None:
        cfg = self._config
        if not cfg.base_disk.is_file():
            raise FileNotFoundError(
                f"WindowsAgentArena base disk not found: {cfg.base_disk}. "
                "Run scripts/install.sh (or set base_disk in configs/default.yaml)."
            )
        if cfg.base_disk.stat().st_size == 0:
            raise ValueError(f"WindowsAgentArena base disk is empty: {cfg.base_disk}")
        marker = cfg.assets_dir / ".complete.json"
        if not marker.is_file():
            raise FileNotFoundError(
                f"WindowsAgentArena asset cache is incomplete: {marker}. "
                "Run scripts/install.sh."
            )
        if not Path("/dev/kvm").exists():
            raise RuntimeError("WindowsAgentArena local QEMU requires /dev/kvm")
        if shutil.which("docker") is None:
            raise RuntimeError("WindowsAgentArena local QEMU requires Docker")

        suffix = uuid.uuid4().hex[:8]
        self.name = format_container_name(
            env_id="waa",
            task_id=self._task_id,
            suffix=suffix,
            session_id=self._identity.resolved_session_id(),
            token_hash=self._identity.token_hash,
            server_port=self._identity.server_port,
        )
        self.slot_root = cfg.runtime_root / "slots" / self.name
        storage_dir = self.slot_root / "storage"
        storage_dir.mkdir(parents=True, exist_ok=False)

        use_snapshot = _snapshot_ready(cfg)
        snap = cfg.snapshot_dir
        ready_disk_mount = f"type=bind,src={snap}/ready.qcow2,dst=/images/ready.qcow2,readonly"
        ready_state_mount = f"type=bind,src={snap}/ready.state,dst=/snapshot/ready.state,readonly"

        try:
            # Each episode gets a fresh, disposable overlay so guest state never
            # leaks across episodes. In snapshot mode the overlay is backed by the
            # ready-state disk (itself backed by the base) and the VM is restored
            # from the saved RAM state instead of cold-booting; otherwise it is
            # backed directly by the base disk and cold-boots.
            if use_snapshot:
                for dev in sorted((snap / "device").iterdir()):
                    shutil.copy2(dev, storage_dir / dev.name)
                (storage_dir / ".waa-restore").touch()
            # The runner creates the overlay from these read-only backing mounts.
            async with docker_create_slot_async():
                # Reserve outside Linux's ephemeral range: rootless Docker's random
                # publication can collide with host outbound connections.
                bridge_port, novnc_port = allocate_ports(n=2, range_start=25000, range_end=26000)
                run_args = [
                    "docker", "run", "-d", "--pull=never", "--name", self.name,
                    "--device=/dev/kvm",
                    # dockur must set up a TAP+bridge so the guest is reachable at its
                    # static IP (GUEST_IP:5000 server, :9222 CDP). Its configureNAT path
                    # needs BOTH /dev/net/tun and net.ipv4.ip_forward=1; without either it
                    # falls back to usermode/SLIRP networking, under which GUEST_IP is
                    # unroutable from this container and the bridge can never reach the
                    # guest (reset would hang until ready_timeout, then warming-retry).
                    "--device=/dev/net/tun",
                    "--cap-add", "NET_ADMIN",
                    "--sysctl", "net.ipv4.ip_forward=1",
                    f"--shm-size={cfg.shm_size}",
                    "--mount", f"type=bind,src={cfg.base_disk},dst=/images/base.qcow2,readonly",
                    "--mount", f"type=bind,src={storage_dir},dst=/storage",
                    "--mount", f"type=bind,src={cfg.assets_dir},dst=/opt/waa-assets,readonly",
                    "--publish", f"{cfg.bind_address}:{bridge_port}:{BRIDGE_PORT}",
                    "--publish", f"{cfg.bind_address}:{novnc_port}:{NOVNC_PORT}",
                    "--env", f"RAM_SIZE={cfg.memory_gb}G",
                    "--env", f"CPU_CORES={cfg.vcpus}",
                    "--env", "CPU_MODEL=host",
                    "--env", "HV=N",
                    "--env", f"VM_NET_IP={GUEST_IP}",
                    "--env", f"WAA_GUEST_IP={GUEST_IP}",
                    "--env", f"WAA_GUEST_READY_TIMEOUT_S={int(cfg.ready_timeout_s)}",
                    "--env", f"HOST_PORTS={BRIDGE_PORT}",
                ]
                if use_snapshot:
                    run_args += [
                        "--mount", ready_disk_mount,
                        "--mount", ready_state_mount,
                        # Keep QMP (bridge screendumps) and add `-incoming defer`; the
                        # runner entrypoint feeds it the saved state via migrate_incoming.
                        "--env", "ARGUMENTS=-qmp tcp:0.0.0.0:7200,server,nowait -incoming defer",
                        "--env", "WAA_INCOMING_STATE=/snapshot/ready.state",
                    ]
                run_args.append(cfg.runner_image)
                await _run(*run_args, timeout=120)

            client_host = "127.0.0.1" if cfg.bind_address == "0.0.0.0" else cfg.bind_address
            self.bridge_url = f"http://{client_host}:{bridge_port}"
            self.novnc_url = f"http://{client_host}:{novnc_port}"
            await self._wait_ready()
            logger.info(
                "WindowsAgentArena VM ready: container=%s bridge=%s noVNC=%s",
                self.name,
                self.bridge_url,
                self.novnc_url,
            )
        except BaseException:
            await self.close()
            raise

    async def _wait_ready(self) -> None:
        assert self.name and self.bridge_url
        deadline = asyncio.get_running_loop().time() + self._config.ready_timeout_s
        last_error = ""
        async with httpx.AsyncClient(timeout=5.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                code, state, _ = await _run(
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Status}}",
                    self.name,
                    check=False,
                    timeout=60,
                )
                if code != 0 or state.strip() not in {"created", "running"}:
                    _, logs, log_err = await _run(
                        "docker", "logs", "--tail", "100", self.name, check=False, timeout=15
                    )
                    raise CapacityExhausted.warming(
                        what=f"WindowsAgentArena runner exited before readiness:\n"
                        f"{(logs or log_err)[-6000:]}"
                    )
                try:
                    response = await client.get(f"{self.bridge_url}/health")
                    if response.status_code == 200 and response.json().get("ready") is True:
                        return
                    last_error = response.text[-500:]
                except Exception as exc:
                    last_error = str(exc)
                await asyncio.sleep(self._config.readiness_poll_interval_s)

        _, logs, log_err = await _run(
            "docker", "logs", "--tail", "100", self.name, check=False, timeout=15
        )
        raise CapacityExhausted.warming(
            what=f"WindowsAgentArena bridge did not become ready at {self.bridge_url}: "
            f"{last_error}\n{(logs or log_err)[-6000:]}"
        )

    async def close(self) -> None:
        name, slot_root = self.name, self.slot_root
        if name:
            try:
                # Shield the removal so a cancellation of the caller still tears the
                # container down (otherwise it leaks until the reaper's next pass).
                code, out, err = await asyncio.shield(
                    _run(*_rm_argv(name), check=False, timeout=90)
                )
            except TimeoutError:
                logger.warning(
                    "timed out removing WindowsAgentArena container %s; preserving runtime slot %s",
                    name,
                    slot_root,
                )
                return
            detail = (err or out).strip()
            if code != 0 and "no such container" not in detail.lower():
                logger.warning(
                    "failed to remove WindowsAgentArena container %s; "
                    "preserving runtime slot %s: %s",
                    name,
                    slot_root,
                    detail,
                )
                return
        if slot_root:
            await asyncio.to_thread(shutil.rmtree, slot_root, True)
        self.name = None
        self.slot_root = None
        self.bridge_url = None
        self.novnc_url = None
