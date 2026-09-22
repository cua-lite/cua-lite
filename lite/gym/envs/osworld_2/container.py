"""OSWorld-V2 VM-in-Docker container lifecycle — spawn / name / boot-wait / reap.

One `cua-lite/osworld_2` container = one QEMU/KVM VM booting the gated v2 disk
`osworld-v2-ubuntu-x86.qcow2` + an in-container eval server (`docker/server.py`) that
hosts OSWorld-V2's `desktop_env` and drives the VM locally. The cua-lite HOST needs ZERO
`desktop_env` — it talks to the container over one HTTP-RPC port (the androidworld pattern),
which is what lets osworld_2 (V2) coexist with osworld v1 + lite.osworld (v1) in one uv.

Only the HOST-side API port is dynamic (mapped to the fixed in-container :6000). The VM's
own ports (5000/9222/8080) stay INTERNAL to the container — server.py reaches them at the
guest IP 20.20.20.21, so they need no `-p` mapping. cua-lite owns the container (naming + reaping);
names go through :func:`lite.gym.utils.config.naming.format_container_name` (`env_id="osworld_2"`,
disjoint from v1 `-osworld-` and lite `.osworld-`).

Run: not directly. Imported by ``lite/gym/envs/osworld_2/main.py``.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import ClassVar
from urllib.parse import urlsplit

import requests

from lite.gym.container import LiteContainerBase
from lite.gym.errors import CapacityExhausted
from lite.gym.utils.config.identity import EnvIdentity
from lite.gym.utils.config.naming import format_container_name
from lite.gym.utils.backend.docker import _rm_argv, docker_run_detached
from lite.gym.utils.backend.ports import allocate_ports

logger = logging.getLogger(__name__)

_IMAGE_DEFAULT = "cua-lite/osworld_2:osworld-v2.1-volume"

_CPORT_API = 6000       # docker/server.py listens here (fixed in-container)
_TASK_CLASS_PATH = "/opt/osworld-v2/evaluation_examples/task_class"
# Dedicated host-port band (only the RPC API port is published; VM ports stay in the container).
# Non-overlapping per lite/gym/utils/backend/ports.py's range map — do NOT use the default 20000-20999
# (that's the sandbox noVNC band). osworld v1 uses 22000-22999.
_API_PORT_RANGE = (23000, 23999)
_HOST_SERVER_PY = None  # set lazily to the checked-in docker/server.py (bind-mounted for hot edits)

_DOCKER_RUN_TIMEOUT_S = 180.0   # structural: `docker run -d` should return in <1s; generous for a daemon hiccup


def _kvm_gid() -> int | None:
    try:
        import grp
        return grp.getgrnam("kvm").gr_gid
    except Exception:
        return None


def _server_py_path() -> str:
    global _HOST_SERVER_PY
    if _HOST_SERVER_PY is None:
        from pathlib import Path
        _HOST_SERVER_PY = str(Path(__file__).resolve().parent / "docker" / "server.py")
    return _HOST_SERVER_PY


# ── Container ─────────────────────────────────────────────────────────────────
@dataclass
class OSWorldV2Container(LiteContainerBase):
    """One VM-in-Docker container + its in-container eval server.

    `OSWorldV2ContainerFactory.build()` returns an un-started container (api port + name
    reserved, NOT booted); `container.start()` then `docker run`s it, waits for the guest
    VM, launches `docker/server.py`, and waits for the server's `/healthz` to report the
    VM ready. The host then drives eval over `http://localhost:{api_port}`."""

    name: str
    api_port: int                 # host port → in-container :6000 (the ONLY mapped port)
    task_id: str
    image: str = _IMAGE_DEFAULT
    qcow2_path: str = ""
    task_class_dir: str = ""      # host dir with task_<id>.py → mounted in the upstream source tree
    asset_dir: str = ""           # official gated assets, mounted at the same absolute path
    ram_size: str = "4G"
    cpu_cores: str = "4"
    disk_size: str = "32G"
    volume_size: int | None = None
    website: bool = False
    boot_timeout: float = 300.0
    screen_width: int = 1920
    screen_height: int = 1080
    client_password: str = "osworld-public-evaluation"
    service_env: dict[str, str] = field(default_factory=dict)  # WEBSITE_HOST_SUFFIX / GITLAB_* → in-container task setup
    _ports_owned: tuple[int, ...] = field(default=(), repr=False)

    rm_label: ClassVar[str] = "osworld_2"

    @property
    def host(self) -> str:
        return "localhost"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.api_port}"

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        """`docker run -d` (KVM VM-in-Docker) + launch the in-container eval server +
        wait for `/healthz` vm_ready, held across the docker-create admission slot for the
        WHOLE boot. Registers (LiteContainerBase) BEFORE the create call (leak-safety)."""
        from lite.gym.remote.admission import docker_create_slot
        with docker_create_slot():
            self._start_locked()

    def _start_locked(self) -> None:
        self._register()   # atexit backstop covers the whole boot window below
        kvm_gid = _kvm_gid()
        docker_run_detached(
            name=self.name, image=self.image,
            auto_remove=True,
            cap_add=("NET_ADMIN",),
            devices=("/dev/kvm", "/dev/net/tun"),
            sysctls=("net.ipv4.ip_forward=1",),
            group_add=(kvm_gid,) if kvm_gid is not None else (),
            env={
                "DISK_SIZE": f"{self.volume_size}G" if self.volume_size else self.disk_size,
                **({"OSWORLD_VOLUME_SIZE": str(self.volume_size)} if self.volume_size else {}),
                "RAM_SIZE": self.ram_size,
                "CPU_CORES": self.cpu_cores,
                # eval-server config (read by docker/server.py)
                "SCREEN_W": str(self.screen_width),
                "SCREEN_H": str(self.screen_height),
                "CLIENT_PASSWORD": self.client_password,
                "TASK_CLASS_DIR": _TASK_CLASS_PATH,
                "SERVER_PORT": str(_CPORT_API),
                # Avoid qemu-docker falling back to usermode networking (guest
                # unreachable at 20.20.20.21, no port forwarding) when host inotify
                # instance limits prevent dnsmasq from watching resolv.conf. Mirrors v1.
                "DNSMASQ_OPTS": "--no-resolv --no-poll",
                # service knobs (website_host_suffix / gitlab) → read by the
                # in-container task setup (prepare_stateful_website_urls reads
                # os.getenv("WEBSITE_HOST_SUFFIX"), etc.)
                **self.service_env,
            },
            volumes=(
                f"{self.qcow2_path}:/System.qcow2:ro",
                f"{self.task_class_dir}/task_{self.task_id}.py:{_TASK_CLASS_PATH}/task_{self.task_id}.py:ro",
                *((f"{self.asset_dir}:{self.asset_dir}:ro",) if self.asset_dir else ()),
                # bind-mount the checked-in server.py over the baked copy
                # (hot edits, no rebuild)
                f"{_server_py_path()}:/usr/local/bin/server.py:ro",
            ),
            ports=((self.api_port, _CPORT_API),),   # the ONLY host-mapped port
            # service_env threads OPENAI_API_KEY / GITLAB_PRIVATE_TOKEN
            redact=("OPENAI_API_KEY", "GITLAB_PRIVATE_TOKEN", "CLIENT_PASSWORD"),
            timeout=_DOCKER_RUN_TIMEOUT_S,
            label="osworld_2",
        )
        self._launch_server()
        self._wait_ready()
        if self.website:
            site = urlsplit(f"http://{self.service_env.get('WEBSITE_HOST_SUFFIX', '')}")
            if site.port and site.port != 80:
                # The site may generate port-80 links even when served on a custom port.
                # The configured nip.io subdomains all resolve to this address.
                address = socket.gethostbyname(site.hostname or "")
                subprocess.run(
                    ["docker", "exec", self.name, "iptables", "-t", "nat", "-I", "OUTPUT",
                     "-p", "tcp", "-d", address, "--dport", "80", "-j", "DNAT",
                     "--to-destination", f"{address}:{site.port}"],
                    check=True, capture_output=True, timeout=30,
                )

    def _exclude_api_dnat(self) -> None:
        """The qemu-docker base image DNATs ALL container ports (except its own VNC) to the
        guest VM at 20.20.20.21, and (re)builds that DNAT DURING boot — flushing an earlier
        rule. So RE-ASSERT (idempotent check-then-insert) a PREROUTING RETURN for the API port,
        called each poll until it sticks — else host→:6000 is hijacked to the guest
        (RemoteDisconnected) and _wait_ready never sees the server."""
        try:
            subprocess.run(
                ["docker", "exec", self.name, "sh", "-c",
                 f"iptables -t nat -C PREROUTING -p tcp --dport {_CPORT_API} -j RETURN 2>/dev/null "
                 f"|| iptables -t nat -I PREROUTING 1 -p tcp --dport {_CPORT_API} -j RETURN"],
                capture_output=True, timeout=30,
            )
        except Exception:
            pass  # best-effort; re-asserted each poll, so a transient docker-exec stall just retries next iteration

    def _launch_server(self) -> None:
        """Start the in-container eval server (long-lived) via `docker exec -d`. It runs the
        venv python; it internally waits for the VM, so launching it immediately is fine."""
        r = subprocess.run(
            ["docker", "exec", "-d", self.name,
             "/opt/venv/bin/python", "/usr/local/bin/server.py"],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            stderr = (r.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"launch eval server in {self.name} failed: {stderr}")

    def _wait_ready(self) -> None:
        """Poll the in-container server's `/healthz` until it reports the guest VM ready.
        A still-booting VM is RECOVERABLE → `CapacityExhausted.warming` (retryable 503)."""
        url = f"{self.base_url}/healthz"
        deadline = time.monotonic() + self.boot_timeout
        last = "no response"
        while time.monotonic() < deadline:
            self._exclude_api_dnat()   # re-assert until the boot-time DNAT stops flushing it
            try:
                resp = requests.get(url, timeout=(5, 5))
                if resp.status_code == 200 and resp.json().get("vm_ready"):
                    logger.info("osworld_2 VM+server ready in %s", self.name)
                    return
                last = f"HTTP {resp.status_code} {resp.text[:80]}" if resp.status_code == 200 else f"HTTP {resp.status_code}"
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(2.0)
        raise CapacityExhausted.warming(
            what=f"osworld_2 VM/server in {self.name} not ready in {self.boot_timeout}s (last: {last})",
        )

# destroy() inherited from LiteContainerBase (template: _pre_destroy hook →
# docker rm -f -v → release _ports_owned → de-register; idempotent, never
# raises — and unlike the old body, a second destroy() can no longer
# re-release already-released ports).


# ── Factory ───────────────────────────────────────────────────────────────────
@dataclass
class OSWorldV2ContainerFactory:
    """Reserves ONE host port (api) + assigns the container name from `env.identity` scope +
    pre-reaps any same-named zombie (all in `build()`, fast/sync); `container.start()` then
    spawns under the docker-create admission slot + boots the VM + eval server. One
    `build()`+`start()` per trajectory (non-poolable)."""

    qcow2_path: str
    task_class_dir: str
    task_id: str
    asset_dir: str = ""
    image: str = _IMAGE_DEFAULT
    ram_size: str = "4G"
    cpu_cores: str = "4"
    disk_size: str = "32G"
    volume_size: int | None = None
    website: bool = False
    boot_timeout: float = 300.0
    screen_width: int = 1920
    screen_height: int = 1080
    client_password: str = "osworld-public-evaluation"
    service_env: dict[str, str] = field(default_factory=dict)   # WEBSITE_HOST_SUFFIX / GITLAB_* → container
    # scope, threaded from env.identity (server mode) or all-None (direct mode)
    session_id: str | None = None
    token_hash: str | None = None
    server_port: int | None = None

    def _make_name(self, suffix: str) -> str:
        identity = EnvIdentity(
            session_id=self.session_id, token_hash=self.token_hash, server_port=self.server_port,
        )
        return format_container_name(
            env_id="osworld_2", task_id=self.task_id, suffix=suffix,
            session_id=identity.resolved_session_id(), token_hash=identity.token_hash,
            server_port=identity.server_port,
        ).lower()

    def build(self) -> OSWorldV2Container:
        """Reserve the host api port + assign the name + pre-reap any same-named zombie.
        Fast + sync, NO `docker run`. The caller sets `_pending` to the returned (un-started)
        container so `external_resource_id` exposes its name before `start()` boots."""
        (api_port,) = allocate_ports(n=1, range_start=_API_PORT_RANGE[0], range_end=_API_PORT_RANGE[1])
        name = self._make_name(uuid.uuid4().hex[:6])
        try:
            subprocess.run(_rm_argv(name), capture_output=True, timeout=60)
        except Exception:
            pass  # best-effort pre-reap (name is uuid-unique); never leak the reserved port on a daemon stall
        return OSWorldV2Container(
            name=name, api_port=api_port, task_id=self.task_id, image=self.image,
            qcow2_path=self.qcow2_path, task_class_dir=self.task_class_dir, asset_dir=self.asset_dir,
            ram_size=self.ram_size, cpu_cores=self.cpu_cores, disk_size=self.disk_size,
            volume_size=self.volume_size,
            website=self.website,
            boot_timeout=self.boot_timeout, screen_width=self.screen_width,
            screen_height=self.screen_height, client_password=self.client_password,
            service_env=self.service_env,
            _ports_owned=(api_port,),
        )
