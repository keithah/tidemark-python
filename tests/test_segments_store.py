import json
import re
import sqlite3

import pytest

from tidemark.store import SCHEMA_VERSION, insert_segment, migrate
from tidemark.store.db import SegmentStoreRecord, get_segment


EXPECTED_AD_EVENT_COLUMNS = [
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

EXPECTED_SEGMENT_COLUMNS = [
    "id",
    "source_url",
    "sequence",
    "resolved_uri",
    "local_path",
    "start_ts",
    "duration_seconds",
    "byte_length",
    "sha256",
    "metadata_json",
    "created_at",
]


def table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "select name from sqlite_master where type = 'table' and name not like 'sqlite_%' order by name"
        )
    ]


def test_migrate_creates_ad_events_and_segments_schema_and_sets_user_version():
    conn = sqlite3.connect(":memory:")

    migrate(conn)

    assert table_names(conn) == ["ad_events", "segments"]
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 2
    assert [row[1] for row in conn.execute("PRAGMA table_info(ad_events)")] == EXPECTED_AD_EVENT_COLUMNS
    assert [row[1] for row in conn.execute("PRAGMA table_info(segments)")] == EXPECTED_SEGMENT_COLUMNS


def test_migrate_is_idempotent_and_does_not_downgrade_newer_user_version():
    conn = sqlite3.connect(":memory:")

    migrate(conn)
    conn.execute("PRAGMA user_version = 99")
    migrate(conn)

    assert table_names(conn) == ["ad_events", "segments"]
    assert conn.execute("select count(*) from sqlite_master where type = 'table' and name = 'segments'").fetchone()[0] == 1
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 99


def test_insert_segment_writes_normalized_columns_and_metadata_json():
    conn = sqlite3.connect(":memory:")
    migrate(conn)

    row_id = insert_segment(
        conn,
        source_url="https://example.test/live/stream.m3u8?token=secret",
        sequence=42,
        resolved_uri="https://cdn.example.test/media/seg42.ts",
        local_path="/tmp/tidemark/seg42.ts",
        start_ts=12.5,
        duration_seconds=6.006,
        byte_length=12345,
        sha256="a" * 64,
        metadata={"program_date_time": "2026-04-27T01:02:03Z", "discontinuity": False},
    )

    row = conn.execute(
        """
        select id, source_url, sequence, resolved_uri, local_path, start_ts,
               duration_seconds, byte_length, sha256, metadata_json
        from segments
        """
    ).fetchone()
    assert row == (
        row_id,
        "https://example.test/live/stream.m3u8?token=secret",
        42,
        "https://cdn.example.test/media/seg42.ts",
        "/tmp/tidemark/seg42.ts",
        12.5,
        6.006,
        12345,
        "a" * 64,
        json.dumps({"program_date_time": "2026-04-27T01:02:03Z", "discontinuity": False}, sort_keys=True, separators=(",", ":")),
    )


def test_insert_segment_accepts_direct_file_boundary_values_and_fetches_dataclass():
    conn = sqlite3.connect(":memory:")
    migrate(conn)

    row_id = insert_segment(
        conn,
        source_url="file:///fixtures/source.ts",
        sequence=0,
        resolved_uri="file:///fixtures/source.ts",
        local_path=None,
        start_ts=0,
        duration_seconds=0.0,
        byte_length=0,
        sha256="0" * 64,
        metadata=None,
    )

    stored = get_segment(conn, row_id)
    assert stored == SegmentStoreRecord(
        id=row_id,
        source_url="file:///fixtures/source.ts",
        sequence=0,
        resolved_uri="file:///fixtures/source.ts",
        local_path=None,
        start_ts=0.0,
        duration_seconds=0.0,
        byte_length=0,
        sha256="0" * 64,
        metadata=None,
    )


def test_insert_segment_allows_duplicate_content_without_unique_constraint():
    conn = sqlite3.connect(":memory:")
    migrate(conn)

    first_id = insert_segment(
        conn,
        source_url="fixture://stream",
        sequence=1,
        resolved_uri="fixture://segment.ts",
        local_path=None,
        start_ts=0,
        duration_seconds=1,
        byte_length=10,
        sha256="1" * 64,
    )
    second_id = insert_segment(
        conn,
        source_url="fixture://stream",
        sequence=1,
        resolved_uri="fixture://segment.ts",
        local_path=None,
        start_ts=0,
        duration_seconds=1,
        byte_length=10,
        sha256="1" * 64,
    )

    assert second_id != first_id
    assert conn.execute("select count(*) from segments").fetchone()[0] == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_url": ""}, "source_url"),
        ({"duration_seconds": -0.001}, "duration_seconds"),
        ({"sha256": "not-hex"}, "sha256"),
        ({"sha256": "g" * 64}, "sha256"),
        ({"sequence": 1.2}, "sequence"),
    ],
)
def test_insert_segment_rejects_malformed_inputs_without_leaking_source_or_metadata(kwargs, message):
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    values = {
        "source_url": "https://example.test/live/stream.m3u8?token=secret",
        "sequence": 1,
        "resolved_uri": "https://cdn.example.test/media/seg1.ts?private=secret",
        "local_path": "/private/tmp/seg1.ts",
        "start_ts": 0,
        "duration_seconds": 1,
        "byte_length": 10,
        "sha256": "2" * 64,
        "metadata": {"token": "secret"},
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message) as exc_info:
        insert_segment(conn, **values)

    public_message = str(exc_info.value)
    assert "secret" not in public_message
    assert "example.test" not in public_message
    assert "/private" not in public_message


def test_insert_segment_before_migration_fails_through_sqlite():
    conn = sqlite3.connect(":memory:")

    with pytest.raises(sqlite3.OperationalError, match=re.escape("no such table: segments")):
        insert_segment(
            conn,
            source_url="fixture://stream",
            sequence=1,
            resolved_uri="fixture://segment.ts",
            local_path=None,
            start_ts=0,
            duration_seconds=1,
            byte_length=10,
            sha256="3" * 64,
        )
