import pytest

from tidemark.ingest.icy import (
    DEFAULT_META_INT,
    icy_marker_from_fields,
    icy_request_headers,
    parse_icy_metadata,
    sanitize_icy_metadata,
)


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


def test_icy_request_headers_request_metadata_with_go_compatible_casing():
    assert DEFAULT_META_INT == 16000
    assert icy_request_headers() == {"Icy-MetaData": "1"}


def test_sanitize_icy_metadata_strips_null_padding():
    assert sanitize_icy_metadata(b"StreamTitle='Song';\x00\x00\x00") == "StreamTitle='Song';"


def test_parse_icy_metadata_extracts_title_and_retains_additional_fields():
    fields = parse_icy_metadata(
        b"StreamTitle='Song';StreamUrl='https://station.example';\x00\x00"
    )

    assert fields == {
        "StreamTitle": "Song",
        "StreamUrl": "https://station.example",
    }


@pytest.mark.parametrize(
    "metadata",
    [
        b"",
        b"\x00\x00\x00",
        b"StreamUrl='https://station.example';",
        b"StreamTitle='';StreamUrl='https://station.example';",
    ],
)
def test_no_title_metadata_returns_no_marker(metadata):
    fields = parse_icy_metadata(metadata)

    assert icy_marker_from_fields(fields) is None


def test_invalid_utf8_sanitizes_to_binary_data_without_leaking_raw_bytes():
    invalid = b"StreamTitle='\xff\xfe';StreamUrl='https://station.example';"

    assert sanitize_icy_metadata(invalid) == "[binary data]"
    assert parse_icy_metadata(invalid) == {}
    assert icy_marker_from_fields(parse_icy_metadata(invalid)) is None


def test_icy_marker_uses_ad_marker_json_contract_without_snake_case_keys():
    fields = parse_icy_metadata(
        b"StreamTitle='Song';StreamUrl='https://station.example';\x00\x00"
    )

    marker = icy_marker_from_fields(fields, timestamp=12.5)

    assert marker is not None
    marker_dict = marker.to_dict()
    assert list(marker_dict) == EXPECTED_MARKER_KEYS
    assert marker_dict["Type"] == "ICY"
    assert marker_dict["Classification"] == "UNKNOWN"
    assert marker_dict["Source"] == "icy_stream"
    assert marker_dict["Fields"]["StreamTitle"] == "Song"
    assert marker_dict["Fields"]["StreamUrl"] == "https://station.example"
    assert marker_dict["Timestamp"] == 12.5
    assert not ({"type", "classification", "source", "fields", "timestamp"} & set(marker_dict))


def test_icy_marker_accepts_explicit_source_and_preserves_field_order():
    fields = parse_icy_metadata(b"StreamTitle='Ad';Extra='one';StreamUrl='two';")

    marker = icy_marker_from_fields(fields, source="fixture", timestamp=1.0)

    assert marker is not None
    assert marker.to_dict()["Source"] == "fixture"
    assert list(marker.to_dict()["Fields"]) == ["StreamTitle", "Extra", "StreamUrl"]
