import json
from pathlib import Path

import pytest
import threefive

from tidemark.markers import AdMarker, decode_scte35_marker, decode_scte35_markers_from_mpegts
from tidemark.markers.scte35 import marker_from_scte35_cue


SPLICE_INSERT_OON_TRUE = "/DAvAAAAAAAA///wFAVIAACef+/+c2nALv4AUsz1AAAAAAAMAQpDVUVJAAABNWLbowo="
SPLICE_NULL = "/DARAAAAAAAAAP/wAAAAAHpPGuQ="
SPLICE_NULL_HEX = "0xFC301100000000000000FFF0000000007A4F1AE4"
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


def test_threefive_imports_current_local_package():
    assert threefive.__version__ == "3.0.77"
    assert Path(threefive.__file__).as_posix().endswith(
        "/threefive_is_scte35/threefive/__init__.py"
    )

    for exported_name in ("Cue", "Stream", "TagParser", "Segment"):
        assert hasattr(threefive, exported_name), f"threefive missing {exported_name}"


@pytest.mark.parametrize(
    "payload, expected_command_name",
    [
        (SPLICE_NULL, "Splice Null"),
        (SPLICE_INSERT_OON_TRUE, "Splice Insert"),
    ],
)
def test_fixture_cues_decode_with_expected_command_names(payload, expected_command_name):
    cue = threefive.Cue(payload)

    assert cue.decode() is True
    assert cue.command.get()["name"] == expected_command_name


def test_splice_insert_fixture_exposes_command_fields_and_descriptors():
    cue = threefive.Cue(SPLICE_INSERT_OON_TRUE)

    assert cue.decode() is True

    command = cue.command.get()
    assert command["name"] == "Splice Insert"
    assert command["pts_time"] > 0
    assert command["break_duration"] == pytest.approx(60.293567)
    assert command["splice_event_id"] == 1207959710
    assert command["out_of_network_indicator"] is True
    assert cue.get_descriptors()


def test_marker_from_scte35_cue_maps_decoded_splice_null_to_go_compatible_contract():
    cue = threefive.Cue(SPLICE_NULL)
    assert cue.decode() is True

    marker = marker_from_scte35_cue(
        cue,
        source="mpegts",
        timestamp=456.25,
        raw_base64=SPLICE_NULL,
    )

    assert isinstance(marker, AdMarker)
    marker_dict = marker.to_dict()

    assert list(marker_dict) == EXPECTED_MARKER_KEYS
    assert marker_dict["Type"] == "SCTE35"
    assert marker_dict["Classification"] == "UNKNOWN"
    assert marker_dict["Source"] == "mpegts"
    assert marker_dict["Tag"] is None
    assert marker_dict["Segment"] is None
    assert marker_dict["RawBase64"] == SPLICE_NULL
    assert marker_dict["Timestamp"] == 456.25
    assert marker_dict["PTS"] is None
    assert marker_dict["Command"]["name"] == "Splice Null"
    assert marker_dict["Descriptors"] == []
    assert marker_dict["Tags"] == []
    assert marker_dict["Fields"] == {"CommandName": "Splice Null"}


def test_decode_splice_insert_marker_preserves_go_compatible_contract():
    marker = decode_scte35_marker(
        SPLICE_INSERT_OON_TRUE,
        source="hls_manifest",
        tag="#EXT-X-SCTE35",
        segment=7,
        timestamp=123.0,
    )

    assert isinstance(marker, AdMarker)
    marker_dict = marker.to_dict()

    assert list(marker_dict) == EXPECTED_MARKER_KEYS
    assert marker_dict["Type"] == "SCTE35"
    assert marker_dict["Classification"] == "UNKNOWN"
    assert marker_dict["Source"] == "hls_manifest"
    assert marker_dict["Tag"] == "#EXT-X-SCTE35"
    assert marker_dict["Segment"] == 7
    assert marker_dict["RawBase64"] == SPLICE_INSERT_OON_TRUE
    assert marker_dict["Timestamp"] == 123.0
    assert marker_dict["PTS"] > 0
    assert marker_dict["Command"]["name"] == "Splice Insert"
    assert marker_dict["Descriptors"]
    assert marker_dict["Fields"]["CommandName"] == "Splice Insert"
    assert marker_dict["Fields"]["OutOfNetworkIndicator"] == "true"
    assert marker_dict["Fields"]["BreakDuration"] == "60.294"
    assert marker_dict["Fields"]["SpliceEventID"] == "0x4800009e"
    assert marker.break_duration == pytest.approx(60.293567)


def test_decode_splice_null_marker_has_default_containers_and_json_contract():
    marker = decode_scte35_marker(SPLICE_NULL, source="fixture")

    marker_dict = marker.to_dict()

    assert list(marker_dict) == EXPECTED_MARKER_KEYS
    assert marker_dict["Fields"]["CommandName"] == "Splice Null"
    assert marker_dict["PTS"] is None
    assert marker_dict["Descriptors"] == []
    assert marker_dict["Tags"] == []
    assert marker_dict["Fields"] == {"CommandName": "Splice Null"}

    decoded = json.loads(marker.to_json())
    assert decoded == marker_dict
    assert not ({"raw_base64", "break_duration", "classification"} & set(decoded))


def test_decode_splice_null_hex_marker_matches_base64_contract():
    base64_marker = decode_scte35_marker(SPLICE_NULL, source="fixture")
    hex_marker = decode_scte35_marker(
        SPLICE_NULL_HEX,
        source="hls_manifest",
        tag="#EXT-X-DATERANGE",
        segment=3,
        timestamp=45.0,
    )

    base64_dict = base64_marker.to_dict()
    hex_dict = hex_marker.to_dict()

    assert list(hex_dict) == EXPECTED_MARKER_KEYS
    assert hex_dict["Classification"] == "UNKNOWN"
    assert hex_dict["Source"] == "hls_manifest"
    assert hex_dict["Tag"] == "#EXT-X-DATERANGE"
    assert hex_dict["Segment"] == 3
    assert hex_dict["Timestamp"] == 45.0
    assert hex_dict["RawBase64"] == SPLICE_NULL
    assert hex_dict["Command"] == base64_dict["Command"]
    assert hex_dict["Descriptors"] == base64_dict["Descriptors"]
    assert hex_dict["Fields"] == base64_dict["Fields"]
    assert hex_dict["PTS"] == base64_dict["PTS"]


def test_decode_scte35_markers_from_mpegts_returns_empty_list_for_empty_segment_bytes():
    assert decode_scte35_markers_from_mpegts(b"") == []


def test_decode_scte35_markers_from_mpegts_maps_stream_cues(monkeypatch):
    class FakeStream:
        def __init__(self, stream_data):
            assert stream_data.read() == b"mpegts bytes"

        def decode(self, callback):
            cue = threefive.Cue(SPLICE_NULL)
            assert cue.decode() is True
            callback(cue)
            return True

    monkeypatch.setattr(threefive, "Stream", FakeStream)

    markers = decode_scte35_markers_from_mpegts(
        b"mpegts bytes",
        source="hls_segment",
        tag=None,
        segment=42,
        timestamp=9.5,
    )

    assert len(markers) == 1
    marker = markers[0]
    assert marker.source == "hls_segment"
    assert marker.tag is None
    assert marker.segment == 42
    assert marker.timestamp == 9.5
    assert marker.classification == "UNKNOWN"
    assert marker.raw_base64 is not None
    assert decode_scte35_marker(marker.raw_base64, source="fixture").fields == {
        "CommandName": "Splice Null"
    }
    assert marker.fields == {"CommandName": "Splice Null"}


def test_decode_scte35_markers_from_mpegts_rejects_non_bytes_without_leaking_content():
    with pytest.raises(TypeError) as exc_info:
        decode_scte35_markers_from_mpegts("private segment text")  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "bytes" in message
    assert "private segment text" not in message


def test_decode_scte35_markers_from_mpegts_wraps_stream_failures_without_leaking_bytes(monkeypatch):
    class FailingStream:
        def __init__(self, stream_data):
            stream_data.read()

        def decode(self, callback):
            raise RuntimeError("private raw bytes were malformed")

    monkeypatch.setattr(threefive, "Stream", FailingStream)

    with pytest.raises(ValueError) as exc_info:
        decode_scte35_markers_from_mpegts(b"private raw bytes")

    message = str(exc_info.value)
    assert "Unable to decode SCTE-35 markers from MPEGTS segment bytes" in message
    assert "private raw bytes" not in message


@pytest.mark.parametrize(
    "payload",
    ["", "not-valid-base64!!!", "0xFC301100", "0xnot-hex", b"\xff\xfe"],
)
def test_decode_scte35_marker_rejects_malformed_payload_without_leaking_payload(payload):
    with pytest.raises(ValueError) as exc_info:
        decode_scte35_marker(payload, source="https://secret.example/manifest.m3u8")

    message = str(exc_info.value)
    if isinstance(payload, str) and payload:
        assert payload not in message
    assert "secret.example" not in message
