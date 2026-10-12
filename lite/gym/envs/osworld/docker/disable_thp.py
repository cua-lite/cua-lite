"""Disable transparent huge pages for the VM process tree, then exec the entrypoint."""

import ctypes
import os
import sys

libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
    raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE")
os.execvp(sys.argv[1], sys.argv[1:])
