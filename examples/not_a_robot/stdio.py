"""Nonblocking input owned by the local controller transports.

Windows controller pipes can be read-only and reject changes to their mode.
Peek available bytes before reading, with this controller as the sole reader.
POSIX descriptors use nonblocking mode and restore it when the controller stops.
Never close the caller's descriptor or start a blocking reader thread.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


class ControllerInput:
    """Poll a controller pipe without a blocking reader thread or lost deadlines."""

    def __init__(self, descriptor: int):
        self.descriptor = descriptor
        self.console = os.name == "nt" and os.isatty(descriptor)
        self.console_line = ""
        self.windows_pipe = os.name == "nt" and not self.console
        self.blocking = None
        if self.windows_pipe:
            import msvcrt

            self.pipe_handle = msvcrt.get_osfhandle(descriptor)
            self.peek_named_pipe = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
            self.peek_named_pipe.argtypes = [
                wintypes.HANDLE,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.LPDWORD,
                wintypes.LPDWORD,
                wintypes.LPDWORD,
            ]
            self.peek_named_pipe.restype = wintypes.BOOL
        elif not self.console:
            self.blocking = os.get_blocking(descriptor)
            os.set_blocking(descriptor, False)

    def read(self) -> bytes | None:
        """Return available bytes, None while idle, or empty bytes at EOF."""
        if self.console:
            import msvcrt

            while msvcrt.kbhit():
                character = msvcrt.getwch()
                if character in ("\x00", "\xe0"):
                    msvcrt.getwch()
                elif character == "\x03":
                    raise KeyboardInterrupt
                elif character == "\x1a":
                    return b""
                elif character == "\b":
                    self.console_line = self.console_line[:-1]
                elif character in ("\r", "\n"):
                    line, self.console_line = self.console_line, ""
                    return (line + "\n").encode("utf-8")
                else:
                    self.console_line += character
            return None
        size = 65536
        if self.windows_pipe:
            available = wintypes.DWORD()
            if not self.peek_named_pipe(
                self.pipe_handle, None, 0, None, ctypes.byref(available), None
            ):
                error = ctypes.get_last_error()
                if error in (109, 232):  # ERROR_BROKEN_PIPE, ERROR_NO_DATA (pipe closing)
                    return b""
                raise ctypes.WinError(error)
            if not available.value:
                return None
            size = min(size, available.value)
        try:
            return os.read(self.descriptor, size)
        except BlockingIOError:
            return None
        except OSError as error:
            if self.windows_pipe and getattr(error, "winerror", None) in (109, 232):
                return b""
            raise

    def close(self) -> None:
        """Restore POSIX mode; Windows handles are unchanged and remain caller-owned."""
        if self.blocking is not None:
            os.set_blocking(self.descriptor, self.blocking)
