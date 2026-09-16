"""Real nonblocking transport checks on the current host, with no model calls."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from types import SimpleNamespace

import pytest

from examples.not_a_robot import stdio
from examples.not_a_robot.stdio import ControllerInput


@pytest.mark.parametrize("initial_blocking", [False, True])
def test_pipe_poll_chunks_eof_and_restore(initial_blocking):
    reader, writer = os.pipe()
    os.set_blocking(reader, initial_blocking)
    controller = ControllerInput(reader)
    try:
        assert controller.read() is None
        os.write(writer, b'{"partial":')
        assert controller.read() == b'{"partial":'
        assert controller.read() is None
        os.write(writer, b'1}\n{"second":2}\n')
        assert controller.read() == b'1}\n{"second":2}\n'
        os.close(writer)
        writer = None
        assert controller.read() == b""
        controller.close()
        assert os.get_blocking(reader) == initial_blocking
        assert os.read(reader, 1) == b""  # caller's fd is still open
    finally:
        controller.close()
        os.close(reader)
        if writer is not None:
            os.close(writer)


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only anonymous pipe regression")
def test_windows_read_only_anonymous_pipe_needs_no_mode_write():
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.DuplicateHandle.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel.DuplicateHandle.restype = wintypes.BOOL
    reader, writer = os.pipe()
    restricted_reader = None
    controller = None
    try:
        process = kernel.GetCurrentProcess()
        restricted = wintypes.HANDLE()
        # FILE_GENERIC_READ excludes FILE_WRITE_ATTRIBUTES needed to change pipe mode.
        assert kernel.DuplicateHandle(
            process,
            msvcrt.get_osfhandle(reader),
            process,
            ctypes.byref(restricted),
            0x120089,
            False,
            0,
        ), ctypes.WinError(ctypes.get_last_error())
        restricted_reader = msvcrt.open_osfhandle(restricted.value, os.O_RDONLY | os.O_BINARY)
        with pytest.raises(PermissionError):
            os.set_blocking(restricted_reader, False)
        controller = ControllerInput(restricted_reader)
        for _ in range(100):
            assert controller.read() is None
        os.write(writer, b'{"id":1}\n')
        assert controller.read() == b'{"id":1}\n'
        os.write(writer, b"tail")
        os.close(writer)
        writer = None
        assert controller.read() == b"tail"
        assert controller.read() == b""
        controller.close()
        assert os.read(restricted_reader, 1) == b""
    finally:
        if controller is not None:
            controller.close()
        if restricted_reader is not None:
            os.close(restricted_reader)
        os.close(reader)
        if writer is not None:
            os.close(writer)


@pytest.mark.skipif(os.name != "nt", reason="Windows PeekNamedPipe boundary")
def test_windows_pipe_never_changes_mode_and_reads_only_available_bytes(monkeypatch):
    reader, writer = os.pipe()
    controller = None
    reads = []

    def deny_mode_change(*args):
        raise PermissionError("Read-only inherited pipe cannot change mode")

    def observed_read(descriptor, size):
        reads.append(size)
        return os.read(descriptor, size)

    monkeypatch.setattr(
        stdio,
        "os",
        SimpleNamespace(
            name="nt",
            isatty=os.isatty,
            get_blocking=deny_mode_change,
            set_blocking=deny_mode_change,
            read=observed_read,
        ),
    )
    try:
        controller = ControllerInput(reader)
        assert controller.read() is None
        assert reads == []
        os.write(writer, b"abc")
        assert controller.read() == b"abc"
        assert reads == [3]
        assert controller.read() is None
        assert reads == [3]
        controller.close()
    finally:
        if controller is not None:
            controller.close()
        os.close(reader)
        os.close(writer)


@pytest.mark.skipif(os.name != "nt", reason="Windows PeekNamedPipe error codes")
@pytest.mark.parametrize("error_code", [109, 232, 5])
def test_windows_peek_distinguishes_eof_from_access_errors(monkeypatch, error_code):
    reader, writer = os.pipe()
    controller = ControllerInput(reader)

    def failed_peek(*args):
        ctypes.set_last_error(error_code)
        return False

    monkeypatch.setattr(controller, "peek_named_pipe", failed_peek)
    try:
        if error_code in (109, 232):
            assert controller.read() == b""
        else:
            with pytest.raises(PermissionError) as error:
                controller.read()
            assert error.value.winerror == error_code
    finally:
        controller.close()
        os.close(reader)
        os.close(writer)
