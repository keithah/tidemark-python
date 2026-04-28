import errno
import socket

import pytest

from tidemark.ingest.udp import (
    UDPAddressError,
    UDPDatagramReader,
    UDP_DATAGRAM_SIZE,
    open_udp_socket,
    parse_udp_url,
)


PRIVATE_UDP_URL = "udp://239.1.1.1:5000?token=secret&account=private"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("udp://@239.1.1.1:5000", ("239.1.1.1", 5000, True)),
        ("udp://239.1.1.1:5000", ("239.1.1.1", 5000, False)),
        ("239.1.1.1:5000", ("239.1.1.1", 5000, False)),
    ],
)
def test_parse_udp_url_accepts_udp_urls_and_bare_hostports(url, expected):
    assert parse_udp_url(url) == expected


@pytest.mark.parametrize(
    ("url", "expected_message"),
    [
        ("", "UDP address is required"),
        ("udp://:5000", "UDP address host is required"),
        ("udp://239.1.1.1", "UDP address port is required"),
        ("udp://239.1.1.1:not-a-port", "UDP address port must be an integer"),
        ("http://239.1.1.1:5000", "UDP address must use udp:// or host:port"),
        (PRIVATE_UDP_URL, "UDP address must not include a query string"),
    ],
)
def test_parse_udp_url_rejects_malformed_inputs_without_leaking_private_url(url, expected_message):
    with pytest.raises(UDPAddressError) as exc_info:
        parse_udp_url(url)

    message = str(exc_info.value)
    assert expected_message in message
    assert PRIVATE_UDP_URL not in message
    assert "token=secret" not in message
    assert "account=private" not in message


class FakeDatagramSocket:
    def __init__(self, datagrams):
        self.datagrams = list(datagrams)
        self.recv_sizes = []
        self.close_count = 0

    def recv(self, size):
        self.recv_sizes.append(size)
        if not self.datagrams:
            return b""
        return self.datagrams.pop(0)

    def close(self):
        self.close_count += 1


def test_udp_datagram_reader_reads_one_default_sized_datagram():
    sock = FakeDatagramSocket([b"a" * UDP_DATAGRAM_SIZE])
    reader = UDPDatagramReader(sock)

    assert reader.read() == b"a" * UDP_DATAGRAM_SIZE
    assert sock.recv_sizes == [UDP_DATAGRAM_SIZE]


def test_udp_datagram_reader_satisfies_smaller_reads_from_bounded_buffer():
    sock = FakeDatagramSocket([b"a" * UDP_DATAGRAM_SIZE])
    reader = UDPDatagramReader(sock)

    assert reader.read(188) == b"a" * 188
    assert reader.read(188) == b"a" * 188
    assert sock.recv_sizes == [UDP_DATAGRAM_SIZE]


def test_udp_datagram_reader_combines_datagrams_for_larger_reads():
    sock = FakeDatagramSocket([b"a" * UDP_DATAGRAM_SIZE, b"b" * UDP_DATAGRAM_SIZE])
    reader = UDPDatagramReader(sock)

    data = reader.read(1504)

    assert data == b"a" * UDP_DATAGRAM_SIZE + b"b" * (1504 - UDP_DATAGRAM_SIZE)
    assert sock.recv_sizes == [UDP_DATAGRAM_SIZE, UDP_DATAGRAM_SIZE]


def test_udp_datagram_reader_returns_eof_on_empty_datagram_and_closes_idempotently():
    sock = FakeDatagramSocket([b""])
    reader = UDPDatagramReader(sock)

    assert reader.read(188) == b""
    reader.close()
    reader.close()

    assert sock.close_count == 1


class RecordingSocket:
    instances = []

    def __init__(self, family, type_):
        self.family = family
        self.type = type_
        self.options = []
        self.timeout = None
        self.bound_to = None
        self.closed = False
        RecordingSocket.instances.append(self)

    def setsockopt(self, level, optname, value):
        self.options.append((level, optname, value))

    def settimeout(self, timeout):
        self.timeout = timeout

    def bind(self, address):
        self.bound_to = address

    def close(self):
        self.closed = True


def _patch_socket_factory(monkeypatch):
    RecordingSocket.instances = []

    def factory(family, type_):
        return RecordingSocket(family, type_)

    monkeypatch.setattr(socket, "socket", factory)


def test_open_udp_socket_sets_udp_options_timeout_recv_buffer_and_unicast_bind(monkeypatch):
    _patch_socket_factory(monkeypatch)

    sock = open_udp_socket("udp://127.0.0.1:5000", timeout=1.25, recv_buffer_bytes=262144)

    assert sock is RecordingSocket.instances[0]
    assert sock.family == socket.AF_INET
    assert sock.type == socket.SOCK_DGRAM
    assert (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) in sock.options
    if hasattr(socket, "SO_REUSEPORT"):
        assert (socket.SOL_SOCKET, socket.SO_REUSEPORT, 1) in sock.options
    assert (socket.SOL_SOCKET, socket.SO_RCVBUF, 262144) in sock.options
    assert sock.timeout == 1.25
    assert sock.bound_to == ("127.0.0.1", 5000)
    assert all(option[1] != socket.IP_ADD_MEMBERSHIP for option in sock.options)


@pytest.mark.parametrize("url", ["udp://@239.1.1.1:5000", "udp://239.1.1.1:5000"])
def test_open_udp_socket_binds_multicast_to_any_and_joins_membership(monkeypatch, url):
    _patch_socket_factory(monkeypatch)

    sock = open_udp_socket(url, interface_ip="192.0.2.10")

    membership_options = [
        value
        for level, optname, value in sock.options
        if level == socket.IPPROTO_IP and optname == socket.IP_ADD_MEMBERSHIP
    ]
    assert sock.bound_to == ("", 5000)
    assert len(membership_options) == 1
    assert membership_options[0] == socket.inet_aton("239.1.1.1") + socket.inet_aton("192.0.2.10")


def test_open_udp_socket_closes_socket_and_reraises_original_socket_errors(monkeypatch):
    class FailingMembershipSocket(RecordingSocket):
        def setsockopt(self, level, optname, value):
            super().setsockopt(level, optname, value)
            if level == socket.IPPROTO_IP and optname == socket.IP_ADD_MEMBERSHIP:
                raise OSError(errno.EADDRNOTAVAIL, "private url token=secret")

    RecordingSocket.instances = []

    def factory(family, type_):
        sock = FailingMembershipSocket(family, type_)
        RecordingSocket.instances.append(sock)
        return sock

    monkeypatch.setattr(socket, "socket", factory)

    with pytest.raises(OSError) as exc_info:
        open_udp_socket(PRIVATE_UDP_URL.replace("?token=secret&account=private", ""))

    assert exc_info.value.errno == errno.EADDRNOTAVAIL
    assert RecordingSocket.instances[0].closed is True
