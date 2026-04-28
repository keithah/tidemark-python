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
