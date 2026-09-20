"""Show why threadpool_limits before numpy import is not the launcher fix."""

import sys

from threadpoolctl import threadpool_info, threadpool_limits

print("numpy loaded at start?", "numpy" in sys.modules)
ctx = threadpool_limits(limits=1)  # applied while nothing is loaded
print("pools visible at that moment:", len(threadpool_info()))

import numpy as np  # noqa: E402  # OpenBLAS dlopens HERE

a = np.random.rand(64, 64)
a @ a  # force pool creation
info = threadpool_info()
for d in info:
    print(f"  {d['user_api']:8s} {d.get('internal_api'):10s} num_threads={d['num_threads']}")
print("VERDICT:", "capped" if all(d["num_threads"] == 1 for d in info) else "NOT capped")
