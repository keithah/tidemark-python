import base64
import errno
import socket
import threading
import time
from pathlib import Path

import pytest
import threefive

from tidemark.ingest import udp as udp_module
from tidemark.ingest.udp import (
    UDPAddressError,
    UDPDatagramReader,
    UDP_DATAGRAM_SIZE,
    iter_udp_scte35_markers,
    open_udp_socket,
    parse_udp_url,
)
from tidemark.markers import decode_scte35_marker


PRIVATE_UDP_URL = "udp://239.1.1.1:5000?token=secret&account=private"
PRIVATE_STREAM_ERROR_TEXT = "private raw bytes from udp://239.1.1.1:5000?token=secret were malformed"
SPLICE_NULL = "/DARAAAAAAAAAP/wAAAAAHpPGuQ="
FIXTURE_PATH = Path("tests/fixtures/scte35_splice_null.ts")
EXPECTED_MARKER_KEYS = [
    "Type",
    "Classification",
    "Source",
    "Tag",
    "PTS",
    "Segment",
    "RawBase64",
    "Command",
    "Descriptors",
    "Tags",
    "Fields",
    "Timestamp",
]


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


def _decoded_splice_null_cue():
    cue = threefive.Cue(SPLICE_NULL)
    assert cue.decode() is True
    return cue


def _assert_udp_marker_contract(marker, *, timestamp):
    marker_dict = marker.to_dict()

    assert list(marker_dict) == EXPECTED_MARKER_KEYS
    assert marker_dict["Type"] == "SCTE35"
    assert marker_dict["Classification"] == "UNKNOWN"
    assert marker_dict["Source"] == "udp_multicast"
    assert marker_dict["Tag"] is None
    assert marker_dict["Segment"] is None
    assert marker_dict["Timestamp"] == timestamp
    assert marker_dict["RawBase64"] is not None
    base64.b64decode(marker_dict["RawBase64"], validate=True)
    decoded_marker = decode_scte35_marker(marker_dict["RawBase64"], source="fixture")
    assert decoded_marker.fields == {"CommandName": "Splice Null"}
    assert marker_dict["Fields"] == {"CommandName": "Splice Null"}


class RecordingUDPStream:
    calls = []

    def __init__(self, reader, *, show_null=True):
        self.calls.append({"reader": reader, "show_null": show_null})

    def decode_next(self):
        yield _decoded_splice_null_cue()


class ConstructionFailingUDPStream:
    def __init__(self, reader, *, show_null=True):
        raise RuntimeError(PRIVATE_STREAM_ERROR_TEXT)


class DecodeFailingUDPStream:
    def __init__(self, reader, *, show_null=True):
        pass

    def decode_next(self):
        raise RuntimeError(PRIVATE_STREAM_ERROR_TEXT)
        yield  # pragma: no cover


def _patch_udp_open(monkeypatch, sock):
    calls = []

    def open_socket(url, *, timeout=2.0, interface_ip="0.0.0.0", recv_buffer_bytes=None):
        calls.append(
            {
                "url": url,
                "timeout": timeout,
                "interface_ip": interface_ip,
                "recv_buffer_bytes": recv_buffer_bytes,
            }
        )
        return sock

    monkeypatch.setattr(udp_module, "open_udp_socket", open_socket)
    return calls


def test_iter_udp_scte35_markers_maps_stream_cues_and_closes_reader(monkeypatch):
    sock = FakeDatagramSocket([])
    open_calls = _patch_udp_open(monkeypatch, sock)
    RecordingUDPStream.calls = []
    monkeypatch.setattr(threefive, "Stream", RecordingUDPStream)

    markers = list(
        iter_udp_scte35_markers(
            "udp://@239.1.1.1:5000",
            timestamp_fn=lambda: 77.0,
            timeout=0.5,
            recv_buffer_bytes=262144,
        )
    )

    assert open_calls == [
        {
            "url": "udp://@239.1.1.1:5000",
            "timeout": 0.5,
            "interface_ip": "0.0.0.0",
            "recv_buffer_bytes": 262144,
        }
    ]
    assert len(RecordingUDPStream.calls) == 1
    assert isinstance(RecordingUDPStream.calls[0]["reader"], UDPDatagramReader)
    assert RecordingUDPStream.calls[0]["show_null"] is True
    assert sock.close_count == 1
    assert len(markers) == 1
    _assert_udp_marker_contract(markers[0], timestamp=77.0)


def test_iter_udp_scte35_markers_forwards_show_null_false_and_calls_timestamp_per_marker(
    monkeypatch,
):
    class TwoCueStream(RecordingUDPStream):
        def decode_next(self):
            yield _decoded_splice_null_cue()
            yield _decoded_splice_null_cue()

    timestamps = iter([10.0, 11.0])
    sock = FakeDatagramSocket([])
    _patch_udp_open(monkeypatch, sock)
    TwoCueStream.calls = []
    monkeypatch.setattr(threefive, "Stream", TwoCueStream)

    markers = list(
        iter_udp_scte35_markers(
            "udp://127.0.0.1:5000",
            timestamp_fn=lambda: next(timestamps),
            show_null=False,
        )
    )

    assert TwoCueStream.calls[0]["show_null"] is False
    assert [marker.timestamp for marker in markers] == [10.0, 11.0]
    assert sock.close_count == 1


@pytest.mark.parametrize("stream_cls", [ConstructionFailingUDPStream, DecodeFailingUDPStream])
def test_iter_udp_scte35_markers_wraps_stream_failures_without_leaking_private_details(
    monkeypatch, stream_cls
):
    sock = FakeDatagramSocket([])
    _patch_udp_open(monkeypatch, sock)
    monkeypatch.setattr(threefive, "Stream", stream_cls)

    with pytest.raises(ValueError) as exc_info:
        list(iter_udp_scte35_markers("udp://239.1.1.1:5000", timestamp_fn=lambda: 1.0))

    message = str(exc_info.value)
    assert message == "Unable to decode UDP SCTE-35 markers"
    assert "token=secret" not in message
    assert "private raw bytes" not in message
    assert "239.1.1.1:5000" not in message
    assert exc_info.value.__cause__ is not None
    assert sock.close_count == 1


def test_iter_udp_scte35_markers_closes_reader_when_generator_is_closed(monkeypatch):
    class EndlessStream(RecordingUDPStream):
        def decode_next(self):
            while True:
                yield _decoded_splice_null_cue()

    sock = FakeDatagramSocket([])
    _patch_udp_open(monkeypatch, sock)
    monkeypatch.setattr(threefive, "Stream", EndlessStream)

    marker_iter = iter_udp_scte35_markers("udp://127.0.0.1:5000", timestamp_fn=lambda: 1.0)
    marker = next(marker_iter)
    marker_iter.close()

    _assert_udp_marker_contract(marker, timestamp=1.0)
    assert sock.close_count == 1


def test_ingest_package_exports_udp_iterator_and_primitives():
    import tidemark.ingest as ingest

    assert ingest.UDPAddressError is UDPAddressError
    assert ingest.UDP_DATAGRAM_SIZE == UDP_DATAGRAM_SIZE
    assert ingest.UDPDatagramReader is UDPDatagramReader
    assert ingest.iter_udp_scte35_markers is iter_udp_scte35_markers
    assert ingest.open_udp_socket is open_udp_socket
    assert ingest.parse_udp_url is parse_udp_url


def test_iter_udp_scte35_markers_reads_tracked_fixture_over_udp_loopback(monkeypatch):
    if not FIXTURE_PATH.exists():
        pytest.fail(f"missing tracked MPEGTS fixture: {FIXTURE_PATH}")

    fixture_bytes = FIXTURE_PATH.read_bytes()
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_thread = None
    marker_iter = None
    try:
        receiver.settimeout(0.5)
        receiver.bind(("127.0.0.1", 0))
        host, port = receiver.getsockname()

        def open_prebound_socket(url, **kwargs):
            assert url == f"udp://{host}:{port}"
            return receiver

        def send_fixture():
            time.sleep(0.05)
            sender.sendto(fixture_bytes, (host, port))

        monkeypatch.setattr(udp_module, "open_udp_socket", open_prebound_socket)
        send_thread = threading.Thread(target=send_fixture)
        send_thread.start()

        marker_iter = iter_udp_scte35_markers(
            f"udp://{host}:{port}",
            timestamp_fn=lambda: 123.0,
            timeout=0.5,
            datagram_size=max(UDP_DATAGRAM_SIZE, len(fixture_bytes)),
        )
        marker = next(marker_iter)
    except OSError as exc:
        pytest.skip(f"UDP loopback unavailable in this environment: {exc}")
    finally:
        if marker_iter is not None:
            marker_iter.close()
        sender.close()
        if send_thread is not None:
            send_thread.join(timeout=2)
        receiver.close()

    _assert_udp_marker_contract(marker, timestamp=123.0)
