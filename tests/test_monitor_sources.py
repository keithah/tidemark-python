from io import BytesIO
from pathlib import Path

import pytest

from tidemark.ingest.udp import UDPAddressError
from tidemark.markers import decode_scte35_marker
from tidemark.monitor_sources import (
    MonitorSourceError,
    StreamType,
    detect_stream_type,
    iter_markers_for_source,
    normalize_stream_type,
)


PRIVATE_URL = "https://secret.example/live.ts?token=abc&account=private"
PRIVATE_ERROR_TEXT = "private token=abc payload bytes were malformed"
FIXTURE_PATH = Path("tests/fixtures/scte35_splice_null.ts")
HLS_PATH = Path("tests/fixtures/monitor_playlist.m3u8")
SPLICE_NULL = "/DARAAAAAAAAAP/wAAAAAHpPGuQ="
INLINE_HLS_MANIFEST = """#EXTM3U
#EXT-X-MEDIA-SEQUENCE:7
#EXT-X-CUE-OUT:DURATION=30
segments/seg7.ts
"""


def marker_from_raw(raw_base64=SPLICE_NULL, *, source="fixture"):
    return decode_scte35_marker(raw_base64, source=source)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("mpegts", StreamType.MPEGTS),
        ("ts", StreamType.MPEGTS),
        ("udp", StreamType.UDP),
        ("hls", StreamType.HLS),
        ("icy", StreamType.ICY),
        ("auto", StreamType.AUTO),
        (None, StreamType.AUTO),
    ],
)
def test_normalize_stream_type_accepts_supported_values(requested, expected):
    assert normalize_stream_type(requested) is expected


def test_normalize_stream_type_rejects_invalid_values_without_source_details():
    with pytest.raises(MonitorSourceError) as exc_info:
        normalize_stream_type("dash")

    message = str(exc_info.value)
    assert "invalid stream type" in message
    assert "dash" in message
    assert "token=" not in message


@pytest.mark.parametrize("requested", ["mpegts", "udp", "hls", "icy"])
def test_detect_stream_type_honors_explicit_stream_types(requested):
    assert detect_stream_type(PRIVATE_URL, requested=requested).value == requested


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("udp://@239.1.1.1:5000", StreamType.UDP),
        ("udp://239.1.1.1:5000", StreamType.UDP),
        ("239.1.1.1:5000", StreamType.UDP),
        ("@239.1.1.1:5000", StreamType.UDP),
        ("http://example.test/live/channel.ts", StreamType.MPEGTS),
        ("https://example.test/live/channel", StreamType.MPEGTS),
        ("http://example.test/live/playlist.m3u8", StreamType.HLS),
        ("https://example.test/live/playlist.M3U8?token=secret", StreamType.HLS),
    ],
)
def test_detect_stream_type_auto_routes_network_sources(source, expected):
    assert detect_stream_type(source, requested="auto") is expected


def test_detect_stream_type_auto_routes_existing_local_non_playlist_files_as_mpegts():
    assert FIXTURE_PATH.exists()

    assert detect_stream_type(FIXTURE_PATH, requested="auto") is StreamType.MPEGTS


def test_detect_stream_type_auto_routes_existing_local_playlist_files_as_hls(tmp_path):
    HLS_PATH.write_text("#EXTM3U\n", encoding="utf-8")
    try:
        assert detect_stream_type(HLS_PATH, requested="auto") is StreamType.HLS
    finally:
        HLS_PATH.unlink(missing_ok=True)


def test_detect_stream_type_auto_rejects_missing_local_paths_without_leaking_path_or_query():
    missing = Path("tests/fixtures/missing.ts?token=abc")

    with pytest.raises(MonitorSourceError) as exc_info:
        detect_stream_type(missing, requested="auto")

    message = str(exc_info.value)
    assert "source setup failed" in message
    assert "token=abc" not in message
    assert "missing.ts" not in message


def test_detect_stream_type_does_not_treat_ambiguous_bare_hostport_as_udp():
    with pytest.raises(MonitorSourceError) as exc_info:
        detect_stream_type("example.test:5000", requested="auto")

    message = str(exc_info.value)
    assert "source setup failed" in message
    assert "example.test" not in message


def test_iter_markers_for_source_delegates_mpegts_with_redacted_error_wrapping(monkeypatch):
    calls = []
    expected_marker = marker_from_raw(source="mpegts-fixture")

    def fake_mpegts(source, *, timestamp_fn, show_null=True, headers=None):
        calls.append(
            {
                "source": source,
                "timestamp": timestamp_fn(),
                "show_null": show_null,
                "headers": headers,
            }
        )
        yield expected_marker

    monkeypatch.setattr("tidemark.monitor_sources.iter_mpegts_scte35_markers", fake_mpegts)
    headers = {"Authorization": "Bearer secret"}

    markers = list(
        iter_markers_for_source(
            PRIVATE_URL,
            stream_type="mpegts",
            timestamp_fn=lambda: 12.5,
            timeout=3.0,
            headers=headers,
        )
    )

    assert markers == [expected_marker]
    assert calls == [
        {
            "source": PRIVATE_URL,
            "timestamp": 12.5,
            "show_null": True,
            "headers": headers,
        }
    ]


def test_iter_markers_for_source_wraps_mpegts_iterator_failures_without_private_details(monkeypatch):
    def failing_mpegts(source, *, timestamp_fn, show_null=True, headers=None):
        raise RuntimeError(PRIVATE_ERROR_TEXT)
        yield  # pragma: no cover

    monkeypatch.setattr("tidemark.monitor_sources.iter_mpegts_scte35_markers", failing_mpegts)

    with pytest.raises(MonitorSourceError) as exc_info:
        list(iter_markers_for_source(PRIVATE_URL, stream_type="mpegts", timestamp_fn=lambda: 1.0))

    message = str(exc_info.value)
    assert "mpegts source iteration failed" in message
    assert "token=abc" not in message
    assert "secret.example" not in message
    assert "private" not in message
    assert exc_info.value.__cause__ is not None


def test_iter_markers_for_source_delegates_udp_with_timeout(monkeypatch):
    calls = []
    expected_marker = marker_from_raw(source="udp-fixture")

    def fake_udp(source, *, timestamp_fn, show_null=True, timeout=2.0):
        calls.append(
            {
                "source": source,
                "timestamp": timestamp_fn(),
                "show_null": show_null,
                "timeout": timeout,
            }
        )
        yield expected_marker

    monkeypatch.setattr("tidemark.monitor_sources.iter_udp_scte35_markers", fake_udp)

    markers = list(
        iter_markers_for_source(
            "udp://@239.1.1.1:5000",
            stream_type="auto",
            timestamp_fn=lambda: 99.0,
            timeout=0.25,
        )
    )

    assert markers == [expected_marker]
    assert calls == [
        {
            "source": "udp://@239.1.1.1:5000",
            "timestamp": 99.0,
            "show_null": True,
            "timeout": 0.25,
        }
    ]


def test_iter_markers_for_source_wraps_malformed_udp_address_as_setup_error(monkeypatch):
    def failing_udp(source, *, timestamp_fn, show_null=True, timeout=2.0):
        raise UDPAddressError("UDP address token=secret is malformed")
        yield  # pragma: no cover

    monkeypatch.setattr("tidemark.monitor_sources.iter_udp_scte35_markers", failing_udp)

    with pytest.raises(MonitorSourceError) as exc_info:
        list(iter_markers_for_source("udp://239.1.1.1:5000", stream_type="udp"))

    message = str(exc_info.value)
    assert "udp source setup failed" in message
    assert "token=secret" not in message
    assert "239.1.1.1" not in message
    assert exc_info.value.__cause__ is not None


def test_iter_markers_for_source_wraps_udp_iterator_failures_without_private_details(monkeypatch):
    def failing_udp(source, *, timestamp_fn, show_null=True, timeout=2.0):
        raise RuntimeError(PRIVATE_ERROR_TEXT)
        yield  # pragma: no cover

    monkeypatch.setattr("tidemark.monitor_sources.iter_udp_scte35_markers", failing_udp)

    with pytest.raises(MonitorSourceError) as exc_info:
        list(iter_markers_for_source("udp://239.1.1.1:5000", stream_type="udp"))

    message = str(exc_info.value)
    assert "udp source iteration failed" in message
    assert "token=abc" not in message
    assert "private" not in message
    assert "239.1.1.1" not in message
    assert exc_info.value.__cause__ is not None


class FakeHttpResponse:
    def __init__(self, body: bytes = b"", headers: dict[str, str] | None = None):
        self._body = body
        self.headers = headers or {}
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            body, self._body = self._body, b""
            return body
        body, self._body = self._body[:size], self._body[size:]
        return body

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def test_normalize_stream_type_accepts_icecast_alias_for_icy():
    assert normalize_stream_type("icecast") is StreamType.ICY


def test_iter_markers_for_source_delegates_local_hls_manifest_and_resolves_relative_segments(tmp_path, monkeypatch):
    manifest_path = tmp_path / "playlist.m3u8"
    segment_path = tmp_path / "segments" / "seg7.ts"
    segment_path.parent.mkdir()
    segment_path.write_bytes(b"segment bytes")
    manifest_path.write_text(INLINE_HLS_MANIFEST, encoding="utf-8")
    expected_marker = marker_from_raw(source="hls-manifest")
    calls = []

    def fake_scte35(manifest_text, *, segment_loader, manifest_url, timestamp=0.0):
        calls.append(("scte35", manifest_text, manifest_url, timestamp, segment_loader("segments/seg7.ts")))
        yield expected_marker

    def fake_id3(manifest_text, *, segment_loader, manifest_url, timestamp=0.0):
        calls.append(("id3", manifest_text, manifest_url, timestamp, segment_loader("segments/seg7.ts")))
        return iter(())

    monkeypatch.setattr("tidemark.monitor_sources.iter_hls_manifest_scte35_markers", fake_scte35)
    monkeypatch.setattr("tidemark.monitor_sources.iter_hls_manifest_id3_markers", fake_id3)

    markers = list(iter_markers_for_source(manifest_path, stream_type="auto", timestamp_fn=lambda: 42.0))

    assert markers == [expected_marker]
    assert calls == [
        ("scte35", INLINE_HLS_MANIFEST, str(manifest_path), 42.0, b"segment bytes"),
        ("id3", INLINE_HLS_MANIFEST, str(manifest_path), 42.0, b"segment bytes"),
    ]


def test_iter_markers_for_source_delegates_http_hls_manifest_and_resolves_segments(monkeypatch):
    expected_marker = marker_from_raw(source="http-hls")
    opened = []
    calls = []

    def fake_urlopen(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else request
        opened.append((url, timeout))
        if url.endswith("playlist.m3u8?token=secret"):
            return FakeHttpResponse(INLINE_HLS_MANIFEST.encode("utf-8"))
        if url.endswith("segments/seg7.ts"):
            return FakeHttpResponse(b"segment bytes")
        raise AssertionError(f"unexpected url {url}")

    def fake_scte35(manifest_text, *, segment_loader, manifest_url, timestamp=0.0):
        calls.append((manifest_text, manifest_url, timestamp, segment_loader("segments/seg7.ts")))
        yield expected_marker

    monkeypatch.setattr("tidemark.monitor_sources.urlopen", fake_urlopen)
    monkeypatch.setattr("tidemark.monitor_sources.iter_hls_manifest_scte35_markers", fake_scte35)
    monkeypatch.setattr("tidemark.monitor_sources.iter_hls_manifest_id3_markers", lambda *args, **kwargs: iter(()))

    markers = list(
        iter_markers_for_source(
            "https://secret.example/live/playlist.m3u8?token=secret",
            stream_type="auto",
            timestamp_fn=lambda: 5.0,
            timeout=0.75,
        )
    )

    assert markers == [expected_marker]
    assert calls == [(INLINE_HLS_MANIFEST, "https://secret.example/live/playlist.m3u8?token=secret", 5.0, b"segment bytes")]
    assert opened == [
        ("https://secret.example/live/playlist.m3u8?token=secret", 0.75),
        ("https://secret.example/live/segments/seg7.ts", 0.75),
    ]


def test_iter_markers_for_source_routes_http_body_sniffed_playlist_as_hls(monkeypatch):
    expected_marker = marker_from_raw(source="sniffed-hls")

    def fake_urlopen(request, timeout=None):
        return FakeHttpResponse(INLINE_HLS_MANIFEST.encode("utf-8"), headers={})

    def fake_scte35(manifest_text, *, segment_loader, manifest_url, timestamp=0.0):
        assert manifest_text == INLINE_HLS_MANIFEST
        assert manifest_url == "https://example.test/live/channel"
        yield expected_marker

    monkeypatch.setattr("tidemark.monitor_sources.urlopen", fake_urlopen)
    monkeypatch.setattr("tidemark.monitor_sources.iter_hls_manifest_scte35_markers", fake_scte35)
    monkeypatch.setattr("tidemark.monitor_sources.iter_hls_manifest_id3_markers", lambda *args, **kwargs: iter(()))

    assert list(iter_markers_for_source("https://example.test/live/channel", stream_type="auto")) == [expected_marker]


def test_iter_markers_for_source_dedupes_hls_markers_with_stable_keys(monkeypatch):
    duplicate = marker_from_raw(source="hls_manifest")
    duplicate.segment = 7
    duplicate.tag = "#EXT-X-SCTE35"

    monkeypatch.setattr("tidemark.monitor_sources._load_hls_manifest_text", lambda *args, **kwargs: INLINE_HLS_MANIFEST)
    monkeypatch.setattr("tidemark.monitor_sources._build_hls_segment_loader", lambda *args, **kwargs: lambda uri: b"")
    monkeypatch.setattr("tidemark.monitor_sources.iter_hls_manifest_scte35_markers", lambda *args, **kwargs: iter([duplicate]))
    monkeypatch.setattr("tidemark.monitor_sources.iter_hls_manifest_id3_markers", lambda *args, **kwargs: iter([duplicate]))

    markers = list(iter_markers_for_source("https://example.test/live/playlist.m3u8", stream_type="hls"))

    assert markers == [duplicate]


def test_iter_markers_for_source_wraps_hls_loading_failures_without_private_details(monkeypatch):
    def failing_urlopen(request, timeout=None):
        raise RuntimeError("load failed for https://secret.example/live/playlist.m3u8?token=abc")

    monkeypatch.setattr("tidemark.monitor_sources.urlopen", failing_urlopen)

    with pytest.raises(MonitorSourceError) as exc_info:
        list(iter_markers_for_source("https://secret.example/live/playlist.m3u8?token=abc", stream_type="hls"))

    message = str(exc_info.value)
    assert "hls source setup failed" in message
    assert "secret.example" not in message
    assert "token=abc" not in message
    assert exc_info.value.__cause__ is not None


def test_iter_markers_for_source_wraps_hls_segment_failures_without_private_details(tmp_path, monkeypatch):
    manifest_path = tmp_path / "playlist.m3u8"
    manifest_path.write_text(INLINE_HLS_MANIFEST, encoding="utf-8")

    def fake_scte35(manifest_text, *, segment_loader, manifest_url, timestamp=0.0):
        segment_loader("missing-private-segment.ts?token=abc")
        return iter(())

    monkeypatch.setattr("tidemark.monitor_sources.iter_hls_manifest_scte35_markers", fake_scte35)
    monkeypatch.setattr("tidemark.monitor_sources.iter_hls_manifest_id3_markers", lambda *args, **kwargs: iter(()))

    with pytest.raises(MonitorSourceError) as exc_info:
        list(iter_markers_for_source(manifest_path, stream_type="hls"))

    message = str(exc_info.value)
    assert "hls source iteration failed" in message
    assert "missing-private-segment" not in message
    assert "token=abc" not in message
    assert exc_info.value.__cause__ is not None


def test_iter_markers_for_source_delegates_icy_with_request_headers_and_header_metaint(monkeypatch):
    stream = BytesIO(b"a" * 8)
    response = FakeHttpResponse(headers={"icy-metaint": "8"})
    requests = []
    calls = []
    expected_marker = marker_from_raw(source="icy-fixture")

    def fake_urlopen(request, timeout=None):
        requests.append((request, timeout))
        return response

    def fake_iter_icy(stream_arg, meta_int, source="icy_stream", timestamp=0.0):
        calls.append((stream_arg, meta_int, source, timestamp()))
        yield expected_marker

    monkeypatch.setattr("tidemark.monitor_sources.urlopen", fake_urlopen)
    monkeypatch.setattr("tidemark.monitor_sources.iter_icy_markers", fake_iter_icy)

    markers = list(
        iter_markers_for_source(
            "https://radio.example/live",
            stream_type="icy",
            timestamp_fn=lambda: 8.5,
            timeout=1.25,
        )
    )

    assert markers == [expected_marker]
    assert requests[0][0].full_url == "https://radio.example/live"
    assert requests[0][0].get_header("Icy-metadata") == "1"
    assert requests[0][1] == 1.25
    assert calls == [(response, 8, "icy_stream", 8.5)]


def test_iter_markers_for_source_auto_routes_icy_header_case_insensitively(monkeypatch):
    expected_marker = marker_from_raw(source="icy-auto")

    def fake_urlopen(request, timeout=None):
        return FakeHttpResponse(headers={"Icy-MetaInt": "16"})

    monkeypatch.setattr("tidemark.monitor_sources.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "tidemark.monitor_sources.iter_icy_markers",
        lambda stream, meta_int, source="icy_stream", timestamp=0.0: iter([expected_marker]) if meta_int == 16 else iter(()),
    )

    assert list(iter_markers_for_source("https://radio.example/live", stream_type="auto")) == [expected_marker]


def test_iter_markers_for_source_icy_missing_metaint_header_uses_default(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        return FakeHttpResponse(headers={})

    def fake_iter_icy(stream, meta_int, source="icy_stream", timestamp=0.0):
        calls.append(meta_int)
        return iter(())

    monkeypatch.setattr("tidemark.monitor_sources.urlopen", fake_urlopen)
    monkeypatch.setattr("tidemark.monitor_sources.iter_icy_markers", fake_iter_icy)

    assert list(iter_markers_for_source("https://radio.example/live", stream_type="icy")) == []
    assert calls == [16000]


def test_iter_markers_for_source_icy_rejects_invalid_metaint_without_private_details(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeHttpResponse(headers={"icy-metaint": "token=secret"})

    monkeypatch.setattr("tidemark.monitor_sources.urlopen", fake_urlopen)

    with pytest.raises(MonitorSourceError) as exc_info:
        list(iter_markers_for_source("https://radio.example/live?token=abc", stream_type="icy"))

    message = str(exc_info.value)
    assert "icy source setup failed" in message
    assert "token=secret" not in message
    assert "token=abc" not in message
    assert "radio.example" not in message


def test_iter_markers_for_source_wraps_icy_iterator_failures_without_private_details(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeHttpResponse(headers={"icy-metaint": "8"})

    def failing_iter_icy(stream, meta_int, source="icy_stream", timestamp=0.0):
        raise RuntimeError("raw ICY metadata StreamTitle='PRIVATE_TOKEN=abc'")
        yield  # pragma: no cover

    monkeypatch.setattr("tidemark.monitor_sources.urlopen", fake_urlopen)
    monkeypatch.setattr("tidemark.monitor_sources.iter_icy_markers", failing_iter_icy)

    with pytest.raises(MonitorSourceError) as exc_info:
        list(iter_markers_for_source("https://radio.example/live?token=abc", stream_type="icy"))

    message = str(exc_info.value)
    assert "icy source iteration failed" in message
    assert "PRIVATE_TOKEN" not in message
    assert "token=abc" not in message
    assert "radio.example" not in message
    assert exc_info.value.__cause__ is not None
