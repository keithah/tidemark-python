import json

import pytest

from tidemark.markers import AdMarker


EXPECTED_KEYS = [
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


def test_ad_marker_serializes_with_go_compatible_key_order():
    marker = AdMarker(
        type="SCTE35",
        classification="UNKNOWN",
        source="fixture",
        tag="#EXT-X-CUE-OUT",
        pts=12.5,
        segment=42,
        raw_base64="/DAv...",
        command={"name": "time_signal"},
        descriptors=[{"tag": 2}],
        tags={"TIT2": "cue-out"},
        fields={"duration": 30.0},
        timestamp=1.0,
    )

    marker_dict = marker.to_dict()

    assert list(marker_dict) == EXPECTED_KEYS
    assert marker_dict == {
        "Type": "SCTE35",
        "Classification": "UNKNOWN",
        "Source": "fixture",
        "Tag": "#EXT-X-CUE-OUT",
        "PTS": 12.5,
        "Segment": 42,
        "RawBase64": "/DAv...",
        "Command": {"name": "time_signal"},
        "Descriptors": [{"tag": 2}],
        "Tags": {"TIT2": "cue-out"},
        "Fields": {"duration": 30.0},
        "Timestamp": 1.0,
    }


def test_ad_marker_json_round_trips_without_snake_case_keys():
    marker = AdMarker(
        type="SCTE35",
        classification="UNKNOWN",
        source="fixture",
        raw_base64="payload",
        timestamp=1.0,
    )

    encoded = marker.to_json()
    decoded = json.loads(encoded)

    assert decoded == marker.to_dict()
    assert list(decoded) == EXPECTED_KEYS
    assert not ({"type", "classification", "source", "raw_base64", "timestamp"} & set(decoded))
    assert "RawBase64" in decoded


@pytest.mark.parametrize("field_name, first_value, second_value", [
    ("Descriptors", {"tag": 1}, {"tag": 2}),
    ("Tags", ("TIT2", "first"), ("TXXX", "second")),
    ("Fields", ("first", 1), ("second", 2)),
])
def test_default_containers_are_independent_between_marker_instances(field_name, first_value, second_value):
    first = AdMarker(type="SCTE35", classification="UNKNOWN", source="fixture", timestamp=1.0)
    second = AdMarker(type="SCTE35", classification="UNKNOWN", source="fixture", timestamp=2.0)

    first_container = first.to_dict()[field_name]
    second_container = second.to_dict()[field_name]

    if isinstance(first_container, list):
        first_container.append(first_value)
        second_container.append(second_value)
    else:
        first_container[first_value[0]] = first_value[1]
        second_container[second_value[0]] = second_value[1]

    assert first.to_dict()[field_name] != second.to_dict()[field_name]
    assert second_value not in first.to_dict()[field_name]
    assert first_value not in second.to_dict()[field_name]
