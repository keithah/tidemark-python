from pathlib import Path

import pytest
import threefive

from tidemark.ingest import iter_mpegts_scte35_markers
from tidemark.markers import decode_scte35_marker


SPLICE_NULL = "/DARAAAAAAAAAP/wAAAAAHpPGuQ="
PRIVATE_URL = "https://secret.example/live.ts?token=abc"
PRIVATE_ERROR_TEXT = "private raw bytes token=abc were malformed"


def marker_from_raw(raw_base64):
    return decode_scte35_marker(raw_base64, source="fixture")


class RecordingStream:
    calls = []

    def __init__(self, source, *, show_null=True, headers=None):
        self.calls.append(
            {
                "source": source,
                "show_null": show_null,
                "headers": headers,
            }
        )

    def decode_next(self):
        cue = threefive.Cue(SPLICE_NULL)
        assert cue.decode() is True
        yield cue


class ConstructionFailingStream:
    def __init__(self, source, *, show_null=True, headers=None):
        raise RuntimeError(PRIVATE_ERROR_TEXT)


class DecodeFailingStream:
    def __init__(self, source, *, show_null=True, headers=None):
        pass

    def decode_next(self):
        raise RuntimeError(PRIVATE_ERROR_TEXT)
        yield  # pragma: no cover


def test_iter_mpegts_scte35_markers_maps_local_path_strings(monkeypatch):
    RecordingStream.calls = []
    monkeypatch.setattr(threefive, "Stream", RecordingStream)

    markers = list(
        iter_mpegts_scte35_markers(
            "fixtures/local.ts",
            timestamp_fn=lambda: 123.5,
        )
    )

    assert RecordingStream.calls == [
        {"source": "fixtures/local.ts", "show_null": True, "headers": {}}
    ]
    assert len(markers) == 1
    marker = markers[0]
    marker_dict = marker.to_dict()

    assert marker_dict["Type"] == "SCTE35"
    assert marker_dict["Classification"] == "UNKNOWN"
    assert marker_dict["Source"] == "mpegts"
    assert marker_dict["Timestamp"] == 123.5
    assert marker_dict["RawBase64"] is not None
    decoded_marker = marker_from_raw(marker_dict["RawBase64"])
    assert decoded_marker.fields == {"CommandName": "Splice Null"}
    assert marker_dict["Command"]["name"] == "Splice Null"
    assert marker_dict["Fields"] == {"CommandName": "Splice Null"}


def test_iter_mpegts_scte35_markers_converts_path_objects_and_forwards_options(monkeypatch):
    RecordingStream.calls = []
    monkeypatch.setattr(threefive, "Stream", RecordingStream)
    headers = {"Authorization": "Bearer secret", "User-Agent": "tidemark-test"}

    markers = list(
        iter_mpegts_scte35_markers(
            Path("fixtures/local.ts"),
            timestamp_fn=lambda: 456.25,
            show_null=False,
            headers=headers,
        )
    )

    assert len(markers) == 1
    assert RecordingStream.calls == [
        {
            "source": "fixtures/local.ts",
            "show_null": False,
            "headers": headers,
        }
    ]
    assert RecordingStream.calls[0]["headers"] is headers
    assert headers == {"Authorization": "Bearer secret", "User-Agent": "tidemark-test"}


def test_iter_mpegts_scte35_markers_uses_fresh_empty_headers_for_none(monkeypatch):
    RecordingStream.calls = []
    monkeypatch.setattr(threefive, "Stream", RecordingStream)

    list(iter_mpegts_scte35_markers("one.ts", timestamp_fn=lambda: 1.0, headers=None))
    list(iter_mpegts_scte35_markers("two.ts", timestamp_fn=lambda: 2.0, headers=None))

    first_headers = RecordingStream.calls[0]["headers"]
    second_headers = RecordingStream.calls[1]["headers"]
    assert first_headers == {}
    assert second_headers == {}
    assert first_headers is not second_headers


def test_iter_mpegts_scte35_markers_forwards_http_sources_and_headers(monkeypatch):
    RecordingStream.calls = []
    monkeypatch.setattr(threefive, "Stream", RecordingStream)
    headers = {"X-Test": "1"}

    markers = list(
        iter_mpegts_scte35_markers(
            PRIVATE_URL,
            timestamp_fn=lambda: 10.0,
            headers=headers,
        )
    )

    assert len(markers) == 1
    assert RecordingStream.calls == [
        {"source": PRIVATE_URL, "show_null": True, "headers": headers}
    ]


@pytest.mark.parametrize("stream_cls", [ConstructionFailingStream, DecodeFailingStream])
def test_iter_mpegts_scte35_markers_wraps_failures_without_leaking_source_or_payload(
    monkeypatch, stream_cls
):
    monkeypatch.setattr(threefive, "Stream", stream_cls)

    with pytest.raises(ValueError) as exc_info:
        list(iter_mpegts_scte35_markers(PRIVATE_URL, timestamp_fn=lambda: 1.0))

    message = str(exc_info.value)
    assert "Unable to decode MPEGTS SCTE-35 markers" in message
    assert "secret.example" not in message
    assert "token=abc" not in message
    assert "private raw bytes" not in message
    assert exc_info.value.__cause__ is not None
