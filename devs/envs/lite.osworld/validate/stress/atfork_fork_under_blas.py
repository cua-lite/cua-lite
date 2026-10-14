"""Manual reproduction for fork-under-BLAS deadlocks. Never run this in CI.

The uncapped arm wedges instead of failing cleanly, so a CI worker cannot
reliably report the failure. Use this only during incident/debug work.

Arm A (control): uvloop forks (create_subprocess_exec, as the docker path does) while a
                 worker thread runs np.matmul with the full OpenBLAS pool -> known deadlock.
Arm B: identical, but caps the native numeric pools before numpy imports, matching the
       launcher fix.
"""
import asyncio
import os
import sys
import threading
import time

ARM = sys.argv[1].upper()
if ARM not in {"A", "B"}:
    raise SystemExit("usage: atfork_fork_under_blas.py A|B")

if ARM == "B":
    for _var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import uvloop  # noqa: E402


def spin(stop):
    a = np.random.rand(900, 900)
    while not stop.is_set():
        a @ a


async def main():
    stop = threading.Event()
    threading.Thread(target=spin, args=(stop,), daemon=True).start()
    time.sleep(1.0)
    worst = 0.0
    for i in range(12):
        t0 = time.monotonic()
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "/bin/true",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                ),
                timeout=20,
            )
            await asyncio.wait_for(proc.wait(), timeout=20)
        except TimeoutError:
            print(f"ARM {ARM}: HANG on fork #{i} after {time.monotonic()-t0:.1f}s", flush=True)
            os._exit(2)
        worst = max(worst, time.monotonic() - t0)
    stop.set()
    print(f"ARM {ARM}: 12/12 forks OK, worst {worst*1000:.1f} ms", flush=True)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
