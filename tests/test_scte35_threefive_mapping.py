import json
from pathlib import Path

import pytest
import threefive

from tidemark.markers import AdMarker, decode_scte35_marker


SPLICE_INSERT_OON_TRUE = "/DAvAAAAAAAA///wFAVIAACef+/+c2nALv4AUsz1AAAAAAAMAQpDVUVJAAABNWLbowo="
SPLICE_NULL = "/DARAAAAAAAAAP/wAAAAAHpPGuQ="


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

    assert list(marker_dict) == [
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

    assert marker_dict["Fields"]["CommandName"] == "Splice Null"
    assert marker_dict["PTS"] is None
    assert marker_dict["Descriptors"] == []
    assert marker_dict["Tags"] == []
    assert marker_dict["Fields"] == {"CommandName": "Splice Null"}

    decoded = json.loads(marker.to_json())
    assert decoded == marker_dict
    assert not ({"raw_base64", "break_duration", "classification"} & set(decoded))


@pytest.mark.parametrize("payload", ["", "not-valid-base64!!!"])
def test_decode_scte35_marker_rejects_malformed_payload_without_leaking_payload(payload):
    with pytest.raises(ValueError) as exc_info:
        decode_scte35_marker(payload, source="fixture")

    if payload:
        assert payload not in str(exc_info.value)
