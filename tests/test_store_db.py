import json
import sqlite3

import pytest

from tidemark.markers import AdMarker
from tidemark.store import SCHEMA_VERSION, insert_ad_event, migrate


EXPECTED_COLUMNS = [
    "id",
    "source_url",
    "marker_type",
    "classification",
    "source",
    "tag",
    "segment_seq",
    "pts",
    "break_duration",
    "raw_json",
    "ts",
    "created_at",
]


def table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "select name from sqlite_master where type = 'table' and name not like 'sqlite_%' order by name"
        )
    ]


def test_migrate_creates_ad_events_schema_and_sets_user_version():
    conn = sqlite3.connect(":memory:")

    migrate(conn)

    assert table_names(conn) == ["ad_events", "fingerprint_cache", "retained_audio", "segments", "songs", "transcript_words"]
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 4
    columns = [row[1] for row in conn.execute("PRAGMA table_info(ad_events)")]
    assert columns == EXPECTED_COLUMNS


def test_migrate_is_idempotent_and_does_not_downgrade_user_version():
    conn = sqlite3.connect(":memory:")

    migrate(conn)
    conn.execute("PRAGMA user_version = 99")
    migrate(conn)

    assert table_names(conn) == ["ad_events", "fingerprint_cache", "retained_audio", "segments", "songs", "transcript_words"]
    assert conn.execute("select count(*) from sqlite_master where type = 'table' and name = 'ad_events'").fetchone()[0] == 1
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 99


def test_insert_ad_event_writes_normalized_columns_and_raw_marker_json():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    marker = AdMarker(
        type="SCTE35",
        classification="AD_START",
        source="fixture",
        tag="#EXT-X-CUE-OUT",
        segment=7,
        pts=12.5,
        break_duration=30.0,
        raw_base64="/DAv...",
        command={"name": "time_signal"},
        descriptors=[{"tag": 2}],
        tags=["cue-out"],
        fields={"duration": 30.0},
        timestamp=1.25,
    )

    row_id = insert_ad_event(conn, "fixture://stream", marker)

    row = conn.execute(
        """
        select id, source_url, marker_type, classification, source, tag,
               segment_seq, pts, break_duration, raw_json, ts
        from ad_events
        """
    ).fetchone()
    assert row == (
        row_id,
        "fixture://stream",
        "SCTE35",
        "AD_START",
        "fixture",
        "#EXT-X-CUE-OUT",
        7,
        12.5,
        30.0,
        marker.to_json(),
        1.25,
    )
    assert json.loads(row[9]) == marker.to_dict()


def test_insert_ad_event_accepts_empty_optional_marker_fields_and_null_boundaries():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    marker = AdMarker(
        type="SCTE35",
        classification="UNKNOWN",
        source="fixture",
        tag="",
        segment=None,
        pts=None,
        timestamp=0.0,
    )

    insert_ad_event(conn, "fixture://stream", marker)

    row = conn.execute(
        "select tag, segment_seq, pts, break_duration, raw_json from ad_events"
    ).fetchone()
    assert row[:4] == ("", None, None, None)
    assert json.loads(row[4]) == marker.to_dict()
    assert "BreakDuration" not in json.loads(row[4])


def test_insert_ad_event_rejects_non_marker_inputs():
    conn = sqlite3.connect(":memory:")
    migrate(conn)

    with pytest.raises(TypeError, match="AdMarker"):
        insert_ad_event(conn, "fixture://stream", object())
