"""UDP MPEGTS ingest primitives.

This module intentionally owns only address validation, socket setup, and a small
file-like datagram reader. SCTE-35 stream integration is added by later ingest
layers so these lower-level behaviors remain deterministic and testable without
real multicast traffic.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

UDP_DATAGRAM_SIZE = 1316
_DEFAULT_INTERFACE_IP = "0.0.0.0"


class UDPAddressError(ValueError):
    """Raised when a UDP ingest address is malformed or unsafe to use."""


def parse_udp_url(url: str) -> tuple[str, int, bool]:
    """Return ``(host, port, used_at)`` for supported UDP address forms.

    Accepted forms are ``udp://@239.1.1.1:5000``, ``udp://239.1.1.1:5000``,
    and bare ``239.1.1.1:5000``. Query strings are rejected explicitly so
    private URL material never needs to appear in downstream errors.
    """
    address = url.strip()
    if not address:
        raise UDPAddressError("UDP address is required")

    if "://" in address and not address.startswith("udp://"):
        raise UDPAddressError("UDP address must use udp:// or host:port")

    parsed = urlparse(address if address.startswith("udp://") else f"//{address}")
    if parsed.query:
        raise UDPAddressError("UDP address must not include a query string")

    used_at = parsed.netloc.startswith("@")
    host = parsed.hostname
    if not host:
        raise UDPAddressError("UDP address host is required")

    try:
        port = parsed.port
    except ValueError as exc:
        raise UDPAddressError("UDP address port must be an integer") from exc

    if port is None:
        raise UDPAddressError("UDP address port is required")

    return host, port, used_at


def open_udp_socket(
    url: str,
    *,
    timeout: float = 2.0,
    interface_ip: str = _DEFAULT_INTERFACE_IP,
    recv_buffer_bytes: int | None = None,
) -> socket.socket:
    """Create and bind a UDP socket for unicast or multicast ingest.

    Multicast sockets bind to ``("", port)`` and join membership with the
    configured interface IP. Any socket setup failure closes the opened socket
    and re-raises the original exception for callers to classify.
    """
    host, port, used_at = parse_udp_url(url)
    should_join_multicast = used_at or _is_multicast_host(host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if recv_buffer_bytes is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buffer_bytes)

        sock.settimeout(timeout)
        sock.bind(("", port) if should_join_multicast else (host, port))

        if should_join_multicast:
            membership = socket.inet_aton(host) + socket.inet_aton(interface_ip)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)

        return sock
    except Exception:
        sock.close()
        raise


class UDPDatagramReader:
    """File-like reader that exposes UDP datagrams as a byte stream."""

    def __init__(self, sock: socket.socket, *, datagram_size: int = UDP_DATAGRAM_SIZE):
        self._sock = sock
        self._datagram_size = datagram_size
        self._buffer = bytearray()
        self._closed = False

    def read(self, size: int = UDP_DATAGRAM_SIZE) -> bytes:
        """Read up to ``size`` bytes, receiving datagrams only as needed."""
        if self._closed:
            return b""
        if size is None or size < 0:
            size = self._datagram_size
        if size == 0:
            return b""

        while len(self._buffer) < size:
            datagram = self._sock.recv(self._datagram_size)
            if datagram == b"":
                break
            self._buffer.extend(datagram)

        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def close(self) -> None:
        """Close the underlying socket once."""
        if self._closed:
            return
        self._closed = True
        self._sock.close()


def _is_multicast_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_multicast
    except ValueError:
        return False
