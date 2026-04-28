from io import BytesIO

import pytest

from tidemark.ingest.icy import (
    DEFAULT_META_INT,
    icy_marker_from_fields,
    icy_request_headers,
    iter_icy_markers,
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


def build_icy_stream(meta_int: int, metadata_blocks: list[bytes]) -> bytes:
    stream = bytearray()
    audio = b"a" * meta_int
    for metadata in metadata_blocks:
        padding = (-len(metadata)) % 16
        padded = metadata + (b"\x00" * padding)
        stream.extend(audio)
        stream.append(len(padded) // 16)
        stream.extend(padded)
    return bytes(stream)


class NonBytesStream:
    def read(self, size: int) -> str:
        return "not bytes"


class TruncatedLengthStream:
    def __init__(self) -> None:
        self._calls = 0

    def read(self, size: int) -> bytes:
        self._calls += 1
        if self._calls == 1:
            return b"a" * size
        return b""


def marker_titles(markers):
    return [marker.to_dict()["Fields"]["StreamTitle"] for marker in markers]


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


def test_iter_icy_markers_yields_marker_for_one_title_with_static_timestamp():
    stream = build_icy_stream(16, [b"StreamTitle='Song';"])

    markers = list(iter_icy_markers(BytesIO(stream), meta_int=16, timestamp=12.5))

    assert marker_titles(markers) == ["Song"]
    assert markers[0].to_dict()["Timestamp"] == 12.5


def test_iter_icy_markers_preserves_content_ad_content_title_sequence():
    stream = build_icy_stream(
        16,
        [
            b"StreamTitle='Morning Show';",
            b"StreamTitle='Promo Spot';",
            b"StreamTitle='Morning Show';",
        ],
    )

    markers = list(iter_icy_markers(BytesIO(stream), meta_int=16))

    assert marker_titles(markers) == ["Morning Show", "Promo Spot", "Morning Show"]


def test_iter_icy_markers_suppresses_duplicate_consecutive_titles():
    stream = build_icy_stream(
        16,
        [
            b"StreamTitle='Song';",
            b"StreamTitle='Song';",
            b"StreamTitle='Next Song';",
        ],
    )

    markers = list(iter_icy_markers(BytesIO(stream), meta_int=16))

    assert marker_titles(markers) == ["Song", "Next Song"]


def test_iter_icy_markers_skips_zero_length_metadata_blocks_and_stops_after_complete_eof():
    stream = build_icy_stream(16, [b"", b"StreamTitle='Song';"])

    markers = list(iter_icy_markers(BytesIO(stream), meta_int=16))

    assert marker_titles(markers) == ["Song"]


def test_iter_icy_markers_uses_callable_timestamp_per_marker():
    timestamps = iter([1.0, 2.0])
    stream = build_icy_stream(
        16,
        [b"StreamTitle='First';", b"StreamTitle='Second';"],
    )

    markers = list(iter_icy_markers(BytesIO(stream), meta_int=16, timestamp=lambda: next(timestamps)))

    assert [marker.to_dict()["Timestamp"] for marker in markers] == [1.0, 2.0]


def test_iter_icy_markers_rejects_invalid_meta_int_without_reading_raw_data():
    with pytest.raises(ValueError, match="ICY metadata interval") as exc_info:
        list(iter_icy_markers(BytesIO(b"PRIVATE_TOKEN=abc123"), meta_int=0))

    assert "PRIVATE_TOKEN" not in str(exc_info.value)


def test_iter_icy_markers_rejects_non_bytes_stream_reads():
    with pytest.raises(TypeError, match="ICY stream audio read"):
        list(iter_icy_markers(NonBytesStream(), meta_int=16))


def test_iter_icy_markers_rejects_truncated_length_byte_with_redacted_error():
    with pytest.raises(ValueError, match="ICY metadata block length") as exc_info:
        list(iter_icy_markers(TruncatedLengthStream(), meta_int=16))

    assert "PRIVATE_TOKEN" not in str(exc_info.value)


def test_iter_icy_markers_rejects_truncated_metadata_block_with_redacted_error():
    metadata = b"StreamTitle='PRIVATE_TOKEN=abc123';"
    padded = metadata + (b"\x00" * ((-len(metadata)) % 16))
    stream = (b"a" * 16) + bytes([len(padded) // 16]) + padded[:-1]

    with pytest.raises(ValueError, match="ICY metadata block") as exc_info:
        list(iter_icy_markers(BytesIO(stream), meta_int=16))

    assert "PRIVATE_TOKEN" not in str(exc_info.value)
