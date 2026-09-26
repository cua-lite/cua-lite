#!/usr/bin/python3
"""Run the official CAD scorer's Python with bounded CPU affinity."""

import os
import sys

cpus = sorted(os.sched_getaffinity(0))
width = min(4, len(cpus))
start = (os.getpid() % max(1, len(cpus) // width)) * width
os.sched_setaffinity(0, cpus[start : start + width])
os.environ["MALLOC_ARENA_MAX"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.execv("/opt/cad-score/bin/python", ["/opt/cad-score/bin/python", *sys.argv[1:]])
