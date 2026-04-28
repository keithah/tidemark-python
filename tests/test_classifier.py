from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import pytest

from tidemark.ingest.hls import direct_cue_marker
from tidemark.ingest.icy import iter_icy_markers
from tidemark.markers import (
    AD_END,
    AD_START,
    UNKNOWN,
    AdMarker,
    Classifier,
    classify_marker,
    classify_markers,
    decode_id3_markers_from_segment_bytes,
    decode_scte35_marker,
)
from tests.test_icy_ingest import build_icy_stream
from tests.test_id3_markers import build_id3_tag
from tests.test_scte35_threefive_mapping import SPLICE_INSERT_OON_TRUE, SPLICE_NULL


def marker(
    marker_type: str,
    *,
    fields: object | None = None,
    tags: object | None = None,
    tag: str | None = None,
    classification: str = UNKNOWN,
) -> AdMarker:
    ad_marker = AdMarker(
        type=marker_type,
        classification=classification,
        source="synthetic",
        tag=tag,
    )
    if fields is not None:
        ad_marker.fields = fields  # type: ignore[assignment]
    if tags is not None:
        ad_marker.tags = tags  # type: ignore[assignment]
    return ad_marker


def assert_classifies(ad_marker: AdMarker, expected: str, classifier: Classifier | None = None) -> None:
    original = replace(ad_marker)
    result = (classifier or Classifier()).classify(ad_marker)

    assert result == expected
    assert ad_marker.classification == expected
    assert ad_marker.type == original.type
    assert ad_marker.source == original.source
    assert ad_marker.tag == original.tag
    assert ad_marker.fields == original.fields
    assert ad_marker.tags == original.tags


def assert_only_classification_changes(ad_marker: AdMarker, expected: str, classifier: Classifier | None = None) -> None:
    before = ad_marker.to_dict()
    result = (classifier or Classifier()).classify(ad_marker)
    after = ad_marker.to_dict()

    assert result == expected
    assert after["Classification"] == expected
    assert list(after) == list(before)
    assert {key: value for key, value in after.items() if key != "Classification"} == {
        key: value for key, value in before.items() if key != "Classification"
    }


def test_exports_plain_go_classification_strings_and_helpers() -> None:
    assert UNKNOWN == "UNKNOWN"
    assert AD_START == "AD_START"
    assert AD_END == "AD_END"
    assert isinstance(Classifier(), Classifier)

    ad_marker = marker("SCTE35", fields={"CommandName": "Splice Null"})
    assert classify_marker(ad_marker) == UNKNOWN
    assert ad_marker.classification == UNKNOWN

    markers = [
        marker("SCTE35", fields={"CommandName": "Splice Insert", "OutOfNetworkIndicator": "true"}),
        marker("SCTE35", fields={"CommandName": "Splice Insert", "OutOfNetworkIndicator": "false"}),
    ]
    assert classify_markers(markers) == [AD_START, AD_END]
    assert [item.classification for item in markers] == [AD_START, AD_END]


def test_decoded_scte35_fixture_markers_classify_without_changing_serialization_contract() -> None:
    splice_insert = decode_scte35_marker(
        SPLICE_INSERT_OON_TRUE,
        source="hls_manifest",
        tag="#EXT-X-SCTE35",
        segment=7,
        timestamp=123.0,
    )
    splice_null = decode_scte35_marker(SPLICE_NULL, source="fixture")
    synthetic_splice_in = AdMarker(
        type="SCTE35",
        classification=UNKNOWN,
        source="synthetic",
        fields={"CommandName": "Splice Insert", "OutOfNetworkIndicator": "false"},
    )

    assert_only_classification_changes(splice_insert, AD_START)
    assert_only_classification_changes(splice_null, UNKNOWN)
    assert_only_classification_changes(synthetic_splice_in, AD_END)


def test_decoded_id3_fixture_markers_classify_from_real_frame_fields() -> None:
    ad_title = decode_id3_markers_from_segment_bytes(
        build_id3_tag(title="Next ad break", txxx_text=("station",)), source="fixture"
    )[0]
    ad_txxx = decode_id3_markers_from_segment_bytes(
        build_id3_tag(title="Episode segment", txxx_desc="Avail", txxx_text=("promo",)), source="fixture"
    )[0]
    ad_end = decode_id3_markers_from_segment_bytes(
        build_id3_tag(title="Episode segment", txxx_desc="cue", txxx_text=("ad_end",)), source="fixture"
    )[0]
    content_start = decode_id3_markers_from_segment_bytes(
        build_id3_tag(title="Episode segment", txxx_desc="cue", txxx_text=("content_start",)), source="fixture"
    )[0]
    benign = decode_id3_markers_from_segment_bytes(
        build_id3_tag(title="Episode segment", txxx_desc="chapter", txxx_text=("liner",)), source="fixture"
    )[0]

    assert_only_classification_changes(ad_title, AD_START)
    assert_only_classification_changes(ad_txxx, AD_START)
    assert_only_classification_changes(ad_end, AD_END)
    assert_only_classification_changes(content_start, AD_END)
    assert_only_classification_changes(benign, UNKNOWN)


def test_icy_fixture_marker_sequence_classifies_with_one_stream_classifier() -> None:
    stream = build_icy_stream(
        16,
        [
            b"StreamTitle='Morning Show';",
            b"StreamTitle='Promo Spot';",
            b"StreamTitle='Morning Show';",
        ],
    )
    markers = list(iter_icy_markers(BytesIO(stream), meta_int=16))
    classifier = Classifier()

    assert [classifier.classify(item) for item in markers] == [UNKNOWN, AD_START, AD_END]
    assert [item.classification for item in markers] == [UNKNOWN, AD_START, AD_END]


def test_direct_hls_cue_fixture_markers_classify_without_changing_marker_contract() -> None:
    cue_out = direct_cue_marker("#EXT-X-CUE-OUT", {"DURATION": "30"}, segment=9, timestamp=12.5)
    cue_in = direct_cue_marker("#EXT-X-CUE-IN", {}, segment=10, timestamp=18.5)

    assert_only_classification_changes(cue_out, AD_START)
    assert cue_out.source == "hls_manifest"
    assert cue_out.tag == "#EXT-X-CUE-OUT"
    assert cue_out.segment == 9
    assert cue_out.fields == {"DURATION": "30"}

    assert_only_classification_changes(cue_in, AD_END)
    assert cue_in.source == "hls_manifest"
    assert cue_in.tag == "#EXT-X-CUE-IN"
    assert cue_in.segment == 10
    assert cue_in.fields == {}


def test_scte35_splice_insert_out_of_network_rules() -> None:
    assert_classifies(
        marker("SCTE35", fields={"CommandName": "Splice Insert", "OutOfNetworkIndicator": "true"}),
        AD_START,
    )
    assert_classifies(
        marker("SCTE35", fields={"CommandName": "Splice Insert", "OutOfNetworkIndicator": "false"}),
        AD_END,
    )
    assert_classifies(marker("SCTE35", fields={"CommandName": "Splice Insert"}), AD_END)


@pytest.mark.parametrize("segmentation_id", ["0x22", "0x30", "0x34", "34", 0x30])
def test_scte35_time_signal_segmentation_start_rules(segmentation_id: object) -> None:
    assert_classifies(
        marker("SCTE35", fields={"CommandName": "Time Signal", "SegmentationTypeID": segmentation_id}),
        AD_START,
    )


@pytest.mark.parametrize("segmentation_id", ["0x23", "0x31", "0x35", "35", 0x31])
def test_scte35_time_signal_segmentation_end_rules(segmentation_id: object) -> None:
    assert_classifies(
        marker("SCTE35", fields={"CommandName": "Time Signal", "SegmentationTypeID": segmentation_id}),
        AD_END,
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"CommandName": "Splice Null"},
        {"CommandName": "Private Command"},
        {"CommandName": "Time Signal", "SegmentationTypeID": "not-a-number"},
        None,
        [],
        {"CommandName": ["Splice Insert"], "OutOfNetworkIndicator": "true"},
    ],
)
def test_scte35_malformed_or_unknown_shapes_fail_closed(fields: object | None) -> None:
    assert_classifies(marker("SCTE35", fields=fields), UNKNOWN)


@pytest.mark.parametrize(
    "candidate",
    [
        "ad_end",
        "content_start",
        "promo AD_END marker",
        "commercial content_start marker",
    ],
)
def test_id3_end_keywords_take_precedence_over_start_keywords(candidate: str) -> None:
    assert_classifies(marker("ID3", tags=[candidate]), AD_END)


@pytest.mark.parametrize("candidate", ["ad", "spot", "promo", "commercial", "next ad break"])
def test_id3_word_boundary_start_keywords(candidate: str) -> None:
    assert_classifies(marker("ID3", tags=[candidate]), AD_START)


@pytest.mark.parametrize("candidate", ["Administrator", "shadow", "adolescent", "promoção"])
def test_id3_prevents_substring_false_positives(candidate: str) -> None:
    assert_classifies(marker("ID3", tags=[candidate]), UNKNOWN)


def test_id3_extracts_text_arrays_and_txxx_description_text_from_fields_frames() -> None:
    assert_classifies(
        marker(
            "ID3",
            fields={
                "Frames": [
                    {"ID": "TIT2", "Text": ["station liner"]},
                    {"ID": "TXXX", "Description": "Avail", "Text": ["promo"]},
                ]
            },
        ),
        AD_START,
    )
    assert_classifies(
        marker(
            "ID3",
            fields={"Frames": [{"ID": "TXXX", "Description": "content_start", "Text": ["promo"]}]},
        ),
        AD_END,
    )


@pytest.mark.parametrize(
    "bad_fields,bad_tags",
    [
        ({"Frames": "promo"}, []),
        ({"Frames": [{"ID": "TXXX", "Description": 42, "Text": ["promo"]}]}, []),
        ({"Frames": [{"ID": "TXXX", "Text": 42}]}, []),
        ({"Frames": [{"ID": "TXXX"}]}, []),
        ({"Frames": [42]}, []),
        ({}, "promo"),
        ([], []),
    ],
)
def test_id3_malformed_shapes_do_not_raise_and_fail_closed(bad_fields: object, bad_tags: object) -> None:
    assert_classifies(marker("ID3", fields=bad_fields, tags=bad_tags), UNKNOWN)


def test_icy_stream_state_emits_start_once_and_end_on_content() -> None:
    classifier = Classifier()
    content = marker("ICY", fields={"StreamTitle": "The Morning Show"})
    promo = marker("ICY", fields={"StreamTitle": "Station Promo - free mugs"})
    repeated_spot = marker("ICY", fields={"StreamTitle": "Station Spot - free mugs"})
    resumed_content = marker("ICY", fields={"StreamTitle": "Back to Music"})

    assert classifier.classify(content) == UNKNOWN
    assert classifier.classify(promo) == AD_START
    assert classifier.classify(repeated_spot) == UNKNOWN
    assert classifier.classify(resumed_content) == AD_END
    assert [item.classification for item in [content, promo, repeated_spot, resumed_content]] == [
        UNKNOWN,
        AD_START,
        UNKNOWN,
        AD_END,
    ]


def test_icy_classifier_state_is_per_instance() -> None:
    in_ad_classifier = Classifier()
    assert in_ad_classifier.classify(marker("ICY", fields={"StreamTitle": "promo"})) == AD_START
    assert in_ad_classifier.classify(marker("ICY", fields={"StreamTitle": "spot"})) == UNKNOWN

    fresh_classifier = Classifier()
    assert fresh_classifier.classify(marker("ICY", fields={"StreamTitle": "spot"})) == AD_START


@pytest.mark.parametrize("tag", ["#EXT-X-CUE-OUT", "#EXT-X-CUE-OUT:DURATION=30", "ext-x-cue-out"])
def test_hls_cue_out_tags_classify_as_ad_start(tag: str) -> None:
    assert_classifies(marker("HLS", tag=tag), AD_START)


@pytest.mark.parametrize("tag", ["#EXT-X-CUE-IN", "ext-x-cue-in"])
def test_hls_cue_in_tags_classify_as_ad_end(tag: str) -> None:
    assert_classifies(marker("HLS", tag=tag), AD_END)


def test_unknown_marker_types_and_malformed_containers_fail_closed() -> None:
    assert_classifies(marker("OTHER", fields={"StreamTitle": "promo"}, tags=["ad"]), UNKNOWN)
    assert_classifies(marker("ICY", fields=[]), UNKNOWN)
    assert_classifies(marker("HLS", tag=None, tags="not-a-list"), UNKNOWN)
