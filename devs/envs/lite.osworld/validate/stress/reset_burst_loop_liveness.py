#!/usr/bin/env python3
"""Manual reset-burst liveness harness for env-server loop stalls.

Never run this in CI. It starts its own env-server process, puts a temporary
``docker`` shim at the front of that process's PATH, and only cleans up the
server it started.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TextIO

import httpx

ROOT = Path(__file__).resolve().parents[5]
TOKEN = "pr9-liveness"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="manual env-server reset burst liveness probe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-id", default="lite.osworld")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--port", type=int, default=30991)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--inspect-delay-s", type=float, default=2.0)
    parser.add_argument("--server-start-timeout-s", type=float, default=180.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    return parser.parse_args()


def _write_docker_shim(tmpdir: Path, *, real_docker: str, delay_s: float) -> None:
    shim = tmpdir / "docker"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"image\" && \"$2\" == \"inspect\" ]]; then\n"
        f"  sleep {delay_s!r}\n"
        "fi\n"
        f"exec {real_docker!r} \"$@\"\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)


def _start_server(
    args: argparse.Namespace,
    shim_dir: Path,
    log_path: Path,
) -> tuple[subprocess.Popen, TextIO]:
    env = os.environ.copy()
    env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    env["CUA_LITE_RESET_JITTER_S"] = "0"
    env["CUA_LITE_BOOT_JITTER_S"] = "0"
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "serve_env.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--token",
            TOKEN,
            "--env-ids",
            args.env_id,
            "--max-live-envs",
            str(args.concurrency),
            "--idle-ttl-sec",
            "60",
        ],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log


async def _wait_for_server(
    base_url: str,
    headers: dict[str, str],
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base_url}/host_status", headers=headers)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"env-server did not answer at {base_url}")


async def _first_task_id(
    base_url: str,
    headers: dict[str, str],
    env_id: str,
) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{base_url}/envs/{env_id}/tasks", headers=headers)
        response.raise_for_status()
        splits = response.json()["splits"]
    for split in ("eval", "train", "rl"):
        task_ids = splits.get(split) or []
        if task_ids:
            return str(task_ids[0])
    raise RuntimeError(f"{env_id} has no registered tasks")


async def _poll_host_status(
    base_url: str,
    headers: dict[str, str],
    stop: asyncio.Event,
    samples: list[float],
    loop_samples: list[dict],
) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                response = await client.get(f"{base_url}/host_status", headers=headers)
                samples.append(time.monotonic() - t0)
                if response.status_code == 200:
                    loop_samples.append(response.json().get("loop", {}))
            except httpx.HTTPError as exc:
                samples.append(time.monotonic() - t0)
                print(f"host_status error: {exc}", file=sys.stderr)
            await asyncio.sleep(1.0)


async def _request_with_admission_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json: dict | None = None,
    budget_s: float,
) -> httpx.Response:
    deadline = time.monotonic() + budget_s
    attempt = 0
    while True:
        response = await client.request(method, url, headers=headers, json=json)
        if response.status_code not in {429, 503}:
            response.raise_for_status()
            return response
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        try:
            delay_s = float(retry_after) if retry_after else min(2.0**attempt, 30.0)
        except ValueError:
            delay_s = min(2.0**attempt, 30.0)
        await asyncio.sleep(min(delay_s, remaining))
        attempt += 1


async def _create_and_reset(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    headers: dict[str, str],
    env_key: str,
    index: int,
    retry_budget_s: float,
) -> None:
    response = await _request_with_admission_retries(
        client,
        "POST",
        f"{base_url}/instances",
        headers=headers,
        json={"env_key": env_key, "session_id": f"pr9-liveness-{index}"},
        budget_s=retry_budget_s,
    )
    instance_id = response.json()["id"]
    try:
        await _request_with_admission_retries(
            client,
            "POST",
            f"{base_url}/instances/{instance_id}/reset",
            headers=headers,
            budget_s=retry_budget_s,
        )
    finally:
        with contextlib.suppress(httpx.HTTPError):
            await client.delete(f"{base_url}/instances/{instance_id}", headers=headers)


async def _run_burst(args: argparse.Namespace) -> None:
    base_url = f"http://127.0.0.1:{args.port}"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    await _wait_for_server(base_url, headers, args.server_start_timeout_s)
    task_id = args.task_id or await _first_task_id(base_url, headers, args.env_id)
    env_key = f"{args.env_id}@{task_id}"

    stop = asyncio.Event()
    status_latencies: list[float] = []
    loop_samples: list[dict] = []
    poller = asyncio.create_task(
        _poll_host_status(base_url, headers, stop, status_latencies, loop_samples)
    )
    try:
        async with httpx.AsyncClient(timeout=args.request_timeout_s) as client:
            await asyncio.gather(
                *(
                    _create_and_reset(
                        client,
                        base_url=base_url,
                        headers=headers,
                        env_key=env_key,
                        index=i,
                        retry_budget_s=args.request_timeout_s,
                    )
                    for i in range(args.concurrency)
                )
            )
    finally:
        stop.set()
        await poller

    worst_status_ms = max(status_latencies, default=0.0) * 1000
    loop_recent = max((s.get("stall_max_60s") or 0.0 for s in loop_samples), default=0.0)
    loop_lifetime = max((s.get("stall_max_seconds") or 0.0 for s in loop_samples), default=0.0)
    print(
        f"env_key={env_key} concurrency={args.concurrency} "
        f"host_status_worst_ms={worst_status_ms:.1f} "
        f"server_loop_stall_60s={loop_recent:.3f} "
        f"server_loop_stall_lifetime={loop_lifetime:.3f}"
    )


def main() -> None:
    args = _parse_args()
    real_docker = shutil.which("docker")
    if not real_docker:
        raise RuntimeError("docker CLI not found")
    with tempfile.TemporaryDirectory(prefix="pr9_liveness_") as tmp:
        tmpdir = Path(tmp)
        _write_docker_shim(
            tmpdir,
            real_docker=real_docker,
            delay_s=args.inspect_delay_s,
        )
        log_path = tmpdir / "server.log"
        proc, log = _start_server(args, tmpdir, log_path)
        try:
            asyncio.run(_run_burst(args))
        except Exception:
            log.flush()
            with contextlib.suppress(Exception):
                print("--- server log tail ---", file=sys.stderr)
                lines = log_path.read_text(errors="replace").splitlines()
                print("\n".join(lines[-80:]), file=sys.stderr)
            raise
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
            log.close()
            print(f"server log: {log_path}")


if __name__ == "__main__":
    main()
