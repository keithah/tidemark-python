import pytest

from tidemark.markers import AdMarker
from tidemark.markers.scte35 import decode_scte35_marker
from tidemark.ingest.hls import HlsScte35Tag, direct_cue_marker, parse_hls_scte35_tag


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
