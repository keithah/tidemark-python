import base64
from io import BytesIO

import pytest
from mutagen.id3 import ID3, PRIV, TIT2, TXXX

from tidemark.markers import decode_id3_markers_from_segment_bytes


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


def build_id3_tag(*, title="Segment marker", owner="com.example.owner", private_data=b"private-payload", txxx_desc="TIDEMARK", txxx_text=("AD", "START")):
    tag = ID3()
    tag.add(TIT2(encoding=3, text=[title]))
    tag.add(PRIV(owner=owner, data=private_data))
    tag.add(TXXX(encoding=3, desc=txxx_desc, text=list(txxx_text)))
    output = BytesIO()
    tag.save(output, v2_version=4, padding=lambda _: 0)
    return output.getvalue()


def assert_redacted(message: str):
    forbidden = [
        "private-payload",
        "private-secret",
        base64.b64encode(b"private-payload").decode("ascii"),
        base64.b64encode(b"prefix" + build_id3_tag() + b"suffix").decode("ascii"),
        "raw_base64",
    ]
    for value in forbidden:
        assert value not in message


def test_empty_and_no_id3_bytes_return_no_markers():
    assert decode_id3_markers_from_segment_bytes(b"") == []
    assert decode_id3_markers_from_segment_bytes(b"not an id3 tag") == []


def test_decodes_id3_tag_at_offset_zero_with_go_compatible_marker_fields():
    tag_bytes = build_id3_tag()

    markers = decode_id3_markers_from_segment_bytes(tag_bytes, segment=7, timestamp=12.5)

    assert len(markers) == 1
    marker = markers[0]
    assert marker.type == "ID3"
    assert marker.classification == "UNKNOWN"
    assert marker.source == "hls_segment"
    assert marker.tag == "ID3"
    assert marker.segment == 7
    assert marker.timestamp == 12.5
    assert marker.raw_base64 == base64.b64encode(tag_bytes).decode("ascii")

    marker_dict = marker.to_dict()
    assert list(marker_dict) == EXPECTED_MARKER_KEYS
    assert marker_dict["Type"] == "ID3"
    assert marker_dict["Classification"] == "UNKNOWN"
    assert marker_dict["Source"] == "hls_segment"
    assert marker_dict["Tag"] == "ID3"
    assert marker_dict["RawBase64"] == base64.b64encode(tag_bytes).decode("ascii")
    assert not ({"type", "classification", "source", "raw_base64"} & set(marker_dict))

    tags = marker_dict["Tags"]
    assert set(tags.keys()) == {"TIT2", "PRIV", "TXXX"}
    assert tags["TIT2"] == "Segment marker"
    assert tags["PRIV"] == "com.example.owner:" + b"private-payload".hex()
    assert tags["TXXX"] == "TIDEMARK:AD START"


def test_scans_prefixed_id3_tag_without_encoding_prefix_or_suffix_in_raw_payload():
    tag_bytes = build_id3_tag(private_data=b"private-secret")
    segment_bytes = b"prefix" + tag_bytes + b"suffix"

    markers = decode_id3_markers_from_segment_bytes(segment_bytes, source="fixture", tag="#EXTINF", segment=3)

    assert len(markers) == 1
    marker_dict = markers[0].to_dict()
    assert marker_dict["Source"] == "fixture"
    assert marker_dict["Tag"] == "#EXTINF"
    assert marker_dict["Segment"] == 3
    assert marker_dict["RawBase64"] == base64.b64encode(tag_bytes).decode("ascii")
    assert marker_dict["RawBase64"] != base64.b64encode(segment_bytes).decode("ascii")


def test_decodes_multiple_complete_id3_tags_in_byte_order_with_deterministic_frames():
    first = build_id3_tag(title="First", owner="owner-1", private_data=b"one", txxx_text=("ONE",))
    second = build_id3_tag(title="Second", owner="owner-2", private_data=b"two", txxx_text=("TWO",))

    markers = decode_id3_markers_from_segment_bytes(b"before" + first + b"middle" + second + b"after")

    assert [marker.raw_base64 for marker in markers] == [
        base64.b64encode(first).decode("ascii"),
        base64.b64encode(second).decode("ascii"),
    ]
    assert [set(m.tags.keys()) for m in markers] == [{"TIT2", "PRIV", "TXXX"}, {"TIT2", "PRIV", "TXXX"}]
    assert markers[0].tags["TIT2"] == "First"
    assert markers[1].tags["TIT2"] == "Second"


def test_rejects_non_synchsafe_id3_size_with_redacted_scan_error():
    malformed = b"prefixID3\x04\x00\x00\x80\x00\x00\x01private-secret"

    with pytest.raises(ValueError) as excinfo:
        decode_id3_markers_from_segment_bytes(malformed, segment=9)

    message = str(excinfo.value)
    assert "ID3 scan" in message
    assert "segment 9" in message
    assert_redacted(message)


def test_rejects_truncated_declared_id3_size_with_redacted_scan_error():
    truncated = b"ID3\x04\x00\x00\x00\x00\x00\x20private-secret"

    with pytest.raises(ValueError) as excinfo:
        decode_id3_markers_from_segment_bytes(truncated)

    message = str(excinfo.value)
    assert "ID3 scan" in message
    assert "truncated" in message.lower()
    assert_redacted(message)


def test_wraps_mutagen_parse_failures_without_leaking_parser_message(monkeypatch):
    tag_bytes = build_id3_tag(private_data=b"private-secret")

    import tidemark.markers.id3 as id3_module

    def fail_parse(_fileobj):
        raise RuntimeError("parser saw private-secret")

    monkeypatch.setattr(id3_module, "ID3", fail_parse)

    with pytest.raises(ValueError) as excinfo:
        decode_id3_markers_from_segment_bytes(tag_bytes, segment=11)

    message = str(excinfo.value)
    assert "ID3 parse" in message
    assert "segment 11" in message
    assert_redacted(message)
