import pytest

from tidemark.markers import AdMarker
from tidemark.markers.scte35 import decode_scte35_marker
from tidemark.ingest.hls import (
    HlsScte35Tag,
    direct_cue_marker,
    iter_hls_manifest_scte35_markers,
    parse_hls_scte35_tag,
)


SPLICE_INSERT_OON_TRUE = "/DAvAAAAAAAA///wFAVIAACef+/+c2nALv4AUsz1AAAAAAAMAQpDVUVJAAABNWLbowo="
SPLICE_NULL_HEX = "0xFC301100000000000000FFF0000000007A4F1AE4"


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            f"#EXT-X-SCTE35:CUE={SPLICE_INSERT_OON_TRUE}",
            HlsScte35Tag(
                tag="#EXT-X-SCTE35",
                payload=SPLICE_INSERT_OON_TRUE,
                attributes={"CUE": SPLICE_INSERT_OON_TRUE},
                direct_fields={},
            ),
        ),
        (
            f"#EXT-X-SCTE35:{SPLICE_INSERT_OON_TRUE}",
            HlsScte35Tag(
                tag="#EXT-X-SCTE35",
                payload=SPLICE_INSERT_OON_TRUE,
                attributes={},
                direct_fields={},
            ),
        ),
        (
            f"#EXT-OATCLS-SCTE35:{SPLICE_INSERT_OON_TRUE}",
            HlsScte35Tag(
                tag="#EXT-OATCLS-SCTE35",
                payload=SPLICE_INSERT_OON_TRUE,
                attributes={},
                direct_fields={},
            ),
        ),
        (
            f"#EXT-X-CUE-OUT-CONT:SCTE35={SPLICE_INSERT_OON_TRUE}",
            HlsScte35Tag(
                tag="#EXT-X-CUE-OUT-CONT",
                payload=SPLICE_INSERT_OON_TRUE,
                attributes={"SCTE35": SPLICE_INSERT_OON_TRUE},
                direct_fields={},
            ),
        ),
    ],
)
def test_parse_hls_scte35_payload_tag_families(line, expected):
    assert parse_hls_scte35_tag(line) == expected


def test_parse_daterange_preserves_scte35_hex_and_quoted_commas():
    line = (
        '#EXT-X-DATERANGE:ID="ad,break",CLASS="cue",'
        f'SCTE35-OUT={SPLICE_NULL_HEX},X-NOTE="alpha,beta"'
    )

    parsed = parse_hls_scte35_tag(line)

    assert parsed == HlsScte35Tag(
        tag="#EXT-X-DATERANGE",
        payload=SPLICE_NULL_HEX,
        attributes={
            "ID": "ad,break",
            "CLASS": "cue",
            "SCTE35-OUT": SPLICE_NULL_HEX,
            "X-NOTE": "alpha,beta",
        },
        direct_fields={},
    )

    marker = decode_scte35_marker(
        parsed.payload,
        source="hls_manifest",
        tag=parsed.tag,
        segment=3,
        timestamp=45.0,
    )
    assert marker.raw_base64 == "/DARAAAAAAAAAP/wAAAAAHpPGuQ="


def test_parse_daterange_preserves_base64_padding_when_splitting_attributes():
    line = f'#EXT-X-DATERANGE:ID="cue",SCTE35-IN={SPLICE_INSERT_OON_TRUE},PLANNED-DURATION=30.0'

    parsed = parse_hls_scte35_tag(line)

    assert parsed.payload == SPLICE_INSERT_OON_TRUE
    assert parsed.attributes["SCTE35-IN"] == SPLICE_INSERT_OON_TRUE
    assert parsed.attributes["PLANNED-DURATION"] == "30.0"


@pytest.mark.parametrize(
    "line, expected",
    [
        ("#EXT-X-CUE-OUT", ("#EXT-X-CUE-OUT", {})),
        ("#EXT-X-CUE-OUT:DURATION=30", ("#EXT-X-CUE-OUT", {"DURATION": "30"})),
        ("#EXT-X-CUE-IN", ("#EXT-X-CUE-IN", {})),
    ],
)
def test_parse_direct_cue_tags_for_later_segment_attachment(line, expected):
    tag, fields = expected

    parsed = parse_hls_scte35_tag(line)

    assert parsed == HlsScte35Tag(tag=tag, payload=None, attributes=fields, direct_fields=fields)


def test_direct_cue_marker_uses_ad_marker_contract_without_binary_decode(monkeypatch):
    def fail_binary_decode(*args, **kwargs):
        raise AssertionError("direct cue markers must not call the binary SCTE-35 decoder")

    monkeypatch.setattr("tidemark.markers.scte35.decode_scte35_marker", fail_binary_decode)

    marker = direct_cue_marker(
        "#EXT-X-CUE-OUT",
        {"DURATION": "30"},
        segment=9,
        timestamp=12.5,
    )

    assert isinstance(marker, AdMarker)
    assert marker.to_dict() == {
        "Type": "SCTE35",
        "Classification": "UNKNOWN",
        "Source": "hls_manifest",
        "Tag": "#EXT-X-CUE-OUT",
        "PTS": None,
        "Segment": 9,
        "RawBase64": None,
        "Command": None,
        "Descriptors": [],
        "Tags": [],
        "Fields": {"DURATION": "30"},
        "Timestamp": 12.5,
    }


@pytest.mark.parametrize(
    "line",
    [
        "",
        "#EXTM3U",
        "#EXTINF:6.006,",
        "#EXT-X-SCTE35:CUE=",
        "#EXT-OATCLS-SCTE35:",
        '#EXT-X-DATERANGE:ID="ad",CLASS="cue"',
        "https://private.example/media/segment0.ts",
    ],
)
def test_parse_unsupported_or_empty_scte_lines_returns_none(line):
    assert parse_hls_scte35_tag(line) is None


def test_iter_hls_manifest_scte35_markers_attaches_binary_tag_to_next_media_sequence():
    manifest = f"""
#EXTM3U
#EXT-X-MEDIA-SEQUENCE:7
#EXTINF:6.0,
#EXT-X-SCTE35:{SPLICE_INSERT_OON_TRUE}
segment7.ts
#EXTINF:6.0,
segment8.ts
"""

    markers = list(iter_hls_manifest_scte35_markers(manifest, timestamp=11.5))

    assert len(markers) == 1
    marker = markers[0]
    assert marker.source == "hls_manifest"
    assert marker.tag == "#EXT-X-SCTE35"
    assert marker.segment == 7
    assert marker.timestamp == 11.5
    assert marker.raw_base64 == SPLICE_INSERT_OON_TRUE
    assert marker.fields["CommandName"] == "Splice Insert"


def test_iter_hls_manifest_scte35_markers_attaches_multiple_pending_tags_to_one_segment():
    manifest = f"""
#EXTM3U
#EXT-X-MEDIA-SEQUENCE:4
#EXT-X-CUE-OUT:DURATION=30,ELAPSED=0
#EXT-X-DATERANGE:ID="ad",SCTE35-OUT={SPLICE_NULL_HEX}
segment4.ts
"""

    markers = list(iter_hls_manifest_scte35_markers(manifest, timestamp=3.25))

    assert [marker.tag for marker in markers] == ["#EXT-X-CUE-OUT", "#EXT-X-DATERANGE"]
    assert [marker.segment for marker in markers] == [4, 4]
    assert markers[0].source == "hls_manifest"
    assert markers[0].classification == "UNKNOWN"
    assert markers[0].fields == {"DURATION": "30", "ELAPSED": "0"}
    assert markers[0].raw_base64 is None
    assert markers[0].timestamp == 3.25
    assert markers[1].raw_base64 == "/DARAAAAAAAAAP/wAAAAAHpPGuQ="


def test_iter_hls_manifest_scte35_markers_resets_pending_tags_after_media_segment():
    manifest = f"""
#EXTM3U
#EXT-X-CUE-OUT:DURATION=15
segment0.ts
segment1.ts
#EXT-X-CUE-IN
segment2.ts
"""

    markers = list(iter_hls_manifest_scte35_markers(manifest))

    assert [(marker.tag, marker.segment, marker.fields) for marker in markers] == [
        ("#EXT-X-CUE-OUT", 0, {"DURATION": "15"}),
        ("#EXT-X-CUE-IN", 2, {}),
    ]


def test_iter_hls_manifest_scte35_markers_ignores_orphan_tags_without_media_uri():
    manifest = f"""
#EXTM3U
#EXT-X-MEDIA-SEQUENCE:12
#EXT-X-CUE-OUT:DURATION=30
#EXT-X-SCTE35:{SPLICE_INSERT_OON_TRUE}
"""

    assert list(iter_hls_manifest_scte35_markers(manifest)) == []


def test_iter_hls_manifest_scte35_markers_defaults_malformed_media_sequence_to_zero():
    manifest = """
#EXTM3U
#EXT-X-MEDIA-SEQUENCE:not-a-number
#EXT-X-CUE-IN
segment0.ts
"""

    markers = list(iter_hls_manifest_scte35_markers(manifest))

    assert [(marker.tag, marker.segment) for marker in markers] == [("#EXT-X-CUE-IN", 0)]


def test_iter_hls_manifest_scte35_markers_raises_redacted_error_for_malformed_binary_payload():
    private_payload = "not-valid-scte35-private-payload"
    manifest = f"""
#EXTM3U
#EXT-X-MEDIA-SEQUENCE:5
#EXT-X-SCTE35:{private_payload}
https://private.example/customer/session/segment5.ts?token=secret
"""

    with pytest.raises(ValueError) as exc_info:
        list(iter_hls_manifest_scte35_markers(manifest, manifest_url="https://origin.example/live/private.m3u8"))

    message = str(exc_info.value)
    assert "hls_manifest" in message
    assert "#EXT-X-SCTE35" in message
    assert "segment 5" in message
    assert private_payload not in message
    assert "private.example" not in message
    assert "origin.example" not in message


def test_iter_hls_manifest_scte35_markers_loads_resolved_segments_after_manifest_tags(monkeypatch):
    manifest = f"""
#EXTM3U
#EXT-X-MEDIA-SEQUENCE:21
#EXT-X-CUE-OUT:DURATION=30
segments/segment21.ts
"""
    loaded_uris = []

    def segment_loader(uri):
        loaded_uris.append(uri)
        return b"segment bytes"

    def fake_decode(data, *, source, tag=None, segment=None, timestamp=0.0):
        assert data == b"segment bytes"
        assert source == "hls_segment"
        assert tag is None
        assert segment == 21
        assert timestamp == 6.25
        return [
            AdMarker(
                type="SCTE35",
                classification="UNKNOWN",
                source=source,
                tag=tag,
                segment=segment,
                timestamp=timestamp,
            )
        ]

    monkeypatch.setattr("tidemark.ingest.hls.decode_scte35_markers_from_mpegts", fake_decode)

    markers = list(
        iter_hls_manifest_scte35_markers(
            manifest,
            manifest_url="https://cdn.example/live/master/playlist.m3u8",
            segment_loader=segment_loader,
            timestamp=6.25,
        )
    )

    assert loaded_uris == ["https://cdn.example/live/master/segments/segment21.ts"]
    assert [(marker.source, marker.segment, marker.classification) for marker in markers] == [
        ("hls_manifest", 21, "UNKNOWN"),
        ("hls_segment", 21, "UNKNOWN"),
    ]


def test_iter_hls_manifest_scte35_markers_keeps_relative_segment_uri_without_manifest_url(monkeypatch):
    manifest = """
#EXTM3U
segment0.ts
"""
    loaded_uris = []

    def segment_loader(uri):
        loaded_uris.append(uri)
        return b""

    monkeypatch.setattr(
        "tidemark.ingest.hls.decode_scte35_markers_from_mpegts",
        lambda data, **kwargs: [],
    )

    assert list(iter_hls_manifest_scte35_markers(manifest, segment_loader=segment_loader)) == []
    assert loaded_uris == ["segment0.ts"]


def test_iter_hls_manifest_scte35_markers_rejects_non_bytes_loader_result_without_content(monkeypatch):
    manifest = """
#EXTM3U
https://private.example/customer/session/segment0.ts?token=secret
"""

    def segment_loader(uri):
        return "private segment text"

    with pytest.raises(TypeError) as exc_info:
        list(iter_hls_manifest_scte35_markers(manifest, segment_loader=segment_loader))

    message = str(exc_info.value)
    assert "bytes" in message
    assert "private segment text" not in message
    assert "private.example" not in message


def test_iter_hls_manifest_scte35_markers_wraps_loader_errors_without_full_url():
    manifest = """
#EXTM3U
https://private.example/customer/session/segment0.ts?token=secret
"""

    def segment_loader(uri):
        raise RuntimeError(f"load failed for {uri}")

    with pytest.raises(ValueError) as exc_info:
        list(iter_hls_manifest_scte35_markers(manifest, segment_loader=segment_loader))

    message = str(exc_info.value)
    assert "Unable to load HLS segment bytes at segment 0" in message
    assert "private.example" not in message
    assert "token=secret" not in message
