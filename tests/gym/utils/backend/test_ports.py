"""Bind-probe behavior for lite.gym.utils.backend.ports.

Guards the host-capability edges the probe must survive. On hosts without
``::1`` on loopback, every ``bind(("::1", port))`` raises EADDRNOTAVAIL,
which must NOT be read as a port conflict — otherwise every port counts
as in-use and ``allocate_ports`` can never hand out a port. A missing
``::1`` must not blind the probe to a v6-only wildcard holder on ``[::]``,
and a non-family failure of the ``[::]`` probe must count as in-use.
Genuine ``::1`` and IPv4 conflicts must always be detected.

Run: uv run pytest tests/gym/utils/backend/test_ports.py
"""

from __future__ import annotations

import errno
import socket

from lite.gym.utils.backend import ports


def _free_v4_port() -> int:
    """A port that is free right now on the IPv4 loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _V6BindFails:
    """AF_INET6 socket whose binds raise the errno configured per address.

    Addresses missing from *errnos* bind successfully.
    """

    def __init__(self, errnos: dict[str, int]) -> None:
        self._errnos = errnos

    def __enter__(self) -> _V6BindFails:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def bind(self, addr: tuple[str, int]) -> None:
        err = self._errnos.get(addr[0])
        if err is not None:
            raise OSError(err, f"bind {addr[0]} unavailable")

    def close(self) -> None:
        pass


def _patch_v6(monkeypatch, v6_socket: _V6BindFails | None, socket_errno: int | None) -> None:
    """Replace socket.socket: IPv4 sockets stay real; AF_INET6 returns
    *v6_socket* (bind-time failure) or raises *socket_errno* at creation."""
    real_socket = socket.socket

    def factory(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
        if family == socket.AF_INET6:
            if socket_errno is not None:
                raise OSError(socket_errno, "Address family not supported")
            return v6_socket
        return real_socket(family, type, proto, fileno)

    monkeypatch.setattr(ports.socket, "socket", factory)


def test_is_port_free_skips_unbindable_ipv6_loopback(monkeypatch) -> None:
    # Stack present but lo has no ::1: socket() succeeds, ::1 bind raises
    # EADDRNOTAVAIL for every port, and the [::] wildcard is free. Reading
    # ::1's failure as "in use" made every band on such a host permanently
    # exhausted.
    port = _free_v4_port()
    _patch_v6(monkeypatch, _V6BindFails({"::1": errno.EADDRNOTAVAIL}), socket_errno=None)
    assert ports._is_port_free(port) is True


def test_is_port_free_skips_absent_ipv6_stack(monkeypatch) -> None:
    # IPv6 compiled/disabled out entirely: socket(AF_INET6, ...) itself
    # raises EAFNOSUPPORT — same skip, different failure point.
    port = _free_v4_port()
    _patch_v6(monkeypatch, v6_socket=None, socket_errno=errno.EAFNOSUPPORT)
    assert ports._is_port_free(port) is True


def test_is_port_free_still_detects_ipv6_only_holders(monkeypatch) -> None:
    # The ::1 probe exists to catch containers bound to [::1]; a genuine
    # conflict (EADDRINUSE) there must still report in-use. The skip is
    # errno-scoped, not "ignore all IPv6 errors".
    port = _free_v4_port()
    _patch_v6(monkeypatch, _V6BindFails({"::1": errno.EADDRINUSE}), socket_errno=None)
    assert ports._is_port_free(port) is False


def test_is_port_free_detects_v6_wildcard_holder_without_loopback(monkeypatch) -> None:
    # A missing ::1 does not prove IPv6 is unusable: a v6-only listener on
    # [::] is invisible to the IPv4 probes and to the ::1 probe (which
    # cannot run here at all). The [::] probe must still report the port
    # busy.
    port = _free_v4_port()
    _patch_v6(
        monkeypatch,
        _V6BindFails({"::1": errno.EADDRNOTAVAIL, "::": errno.EADDRINUSE}),
        socket_errno=None,
    )
    assert ports._is_port_free(port) is False


def test_is_port_free_conservative_on_wildcard_probe_errors(monkeypatch) -> None:
    # A wildcard-probe failure that is neither a real holder (EADDRINUSE)
    # nor family-level unavailability (fd exhaustion, kernel memory, ...)
    # leaves the port unverified — report in-use instead of allocating it.
    port = _free_v4_port()
    _patch_v6(
        monkeypatch,
        _V6BindFails({"::1": errno.EADDRNOTAVAIL, "::": errno.EMFILE}),
        socket_errno=None,
    )
    assert ports._is_port_free(port) is False


def test_is_port_free_reports_v4_conflict_despite_skipped_v6_loopback(monkeypatch) -> None:
    # A real IPv4 listener must still be detected on a host whose ::1 probe
    # is skipped — the skip cannot mask IPv4 occupancy.
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("0.0.0.0", 0))
    holder.listen(1)
    try:
        port = int(holder.getsockname()[1])
        _patch_v6(monkeypatch, _V6BindFails({"::1": errno.EADDRNOTAVAIL}), socket_errno=None)
        assert ports._is_port_free(port) is False
    finally:
        holder.close()
