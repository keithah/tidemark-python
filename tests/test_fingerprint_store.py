import re
import sqlite3

import pytest

from tidemark.store import (
    FingerprintCacheRecord,
    RetainedAudioStoreRecord,
    SCHEMA_VERSION,
    SongStoreRecord,
    get_fingerprint_cache,
    get_retained_audio,
    get_song,
    insert_fingerprint_cache,
    insert_retained_audio,
    insert_segment,
    insert_song,
    migrate,
)


EXPECTED_TABLES = [
    "ad_events",
    "fingerprint_cache",
    "retained_audio",
    "segments",
    "songs",
    "transcript_words",
]
EXPECTED_SONG_COLUMNS = [
    "id",
    "segment_id",
    "source_url",
    "segment_sequence",
    "start_ts",
    "duration_seconds",
    "fingerprint",
    "acoustid_id",
    "recording_id",
    "title",
    "artist",
    "album",
    "score",
    "lookup_source",
    "created_at",
]
EXPECTED_FINGERPRINT_CACHE_COLUMNS = [
    "fingerprint",
    "acoustid_id",
    "recording_id",
    "title",
    "artist",
    "album",
    "score",
    "raw_status",
    "lookup_source",
    "cached_at",
]
EXPECTED_RETAINED_AUDIO_COLUMNS = [
    "id",
    "segment_id",
    "source_url",
    "segment_sequence",
    "path",
    "format",
    "sample_rate",
    "channels",
    "sample_format",
    "start_ts",
    "duration_seconds",
    "byte_length",
    "sha256",
    "created_at",
]


def table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "select name from sqlite_master where type = 'table' and name not like 'sqlite_%' order by name"
        )
    ]


def index_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return sorted(row[1] for row in conn.execute(f"PRAGMA index_list({table})"))


def insert_fixture_segment(conn: sqlite3.Connection) -> int:
    return insert_segment(
        conn,
        source_url="fixture://stream",
        sequence=3,
        resolved_uri="fixture://segment-3.ts",
        local_path=None,
        start_ts=18.0,
        duration_seconds=6.0,
        byte_length=2048,
        sha256="a" * 64,
    )


def test_migrate_creates_schema_v4_fingerprint_tables_and_indexes():
    conn = sqlite3.connect(":memory:")

    migrate(conn)

    assert SCHEMA_VERSION == 4
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
    assert table_names(conn) == EXPECTED_TABLES
    assert [row[1] for row in conn.execute("PRAGMA table_info(songs)")] == EXPECTED_SONG_COLUMNS
    assert [row[1] for row in conn.execute("PRAGMA table_info(fingerprint_cache)")] == EXPECTED_FINGERPRINT_CACHE_COLUMNS
    assert [row[1] for row in conn.execute("PRAGMA table_info(retained_audio)")] == EXPECTED_RETAINED_AUDIO_COLUMNS
    assert "idx_retained_audio_source_time" in index_names(conn, "retained_audio")


def test_song_helper_inserts_and_fetches_nullable_fingerprint_evidence():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    segment_id = insert_fixture_segment(conn)

    row_id = insert_song(
        conn,
        segment_id=segment_id,
        source_url="fixture://stream",
        segment_sequence=3,
        start_ts=18,
        duration_seconds=6,
        fingerprint="fingerprint-v1",
        acoustid_id=None,
        recording_id=None,
        title=None,
        artist=None,
        album=None,
        score=None,
        lookup_source="fixture",
    )

    stored = get_song(conn, row_id)
    assert stored == SongStoreRecord(
        id=row_id,
        segment_id=segment_id,
        source_url="fixture://stream",
        segment_sequence=3,
        start_ts=18.0,
        duration_seconds=6.0,
        fingerprint="fingerprint-v1",
        acoustid_id=None,
        recording_id=None,
        title=None,
        artist=None,
        album=None,
        score=None,
        lookup_source="fixture",
        created_at=stored.created_at if stored is not None else "",
    )
    assert stored is not None and stored.created_at


def test_fingerprint_cache_upserts_by_fingerprint():
    conn = sqlite3.connect(":memory:")
    migrate(conn)

    insert_fingerprint_cache(
        conn,
        fingerprint="fingerprint-v1",
        acoustid_id=None,
        recording_id=None,
        title=None,
        artist=None,
        album=None,
        score=None,
        raw_status="miss",
        lookup_source="acoustid",
    )
    insert_fingerprint_cache(
        conn,
        fingerprint="fingerprint-v1",
        acoustid_id="acoustid-1",
        recording_id="recording-1",
        title="Title",
        artist="Artist",
        album="Album",
        score=0.82,
        raw_status="ok",
        lookup_source="acoustid",
    )

    stored = get_fingerprint_cache(conn, "fingerprint-v1")
    assert stored == FingerprintCacheRecord(
        fingerprint="fingerprint-v1",
        acoustid_id="acoustid-1",
        recording_id="recording-1",
        title="Title",
        artist="Artist",
        album="Album",
        score=0.82,
        raw_status="ok",
        lookup_source="acoustid",
        cached_at=stored.cached_at if stored is not None else "",
    )
    assert conn.execute("select count(*) from fingerprint_cache").fetchone()[0] == 1


def test_retained_audio_helper_inserts_and_fetches_timestamp_indexed_metadata():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    segment_id = insert_fixture_segment(conn)

    row_id = insert_retained_audio(
        conn,
        segment_id=segment_id,
        source_url="fixture://stream",
        segment_sequence=3,
        path="retained/fixture-3.wav",
        format="wav",
        sample_rate=44100,
        channels=2,
        sample_format="s16le",
        start_ts=18.0,
        duration_seconds=6.0,
        byte_length=4096,
        sha256="b" * 64,
    )

    stored = get_retained_audio(conn, row_id)
    assert stored == RetainedAudioStoreRecord(
        id=row_id,
        segment_id=segment_id,
        source_url="fixture://stream",
        segment_sequence=3,
        path="retained/fixture-3.wav",
        format="wav",
        sample_rate=44100,
        channels=2,
        sample_format="s16le",
        start_ts=18.0,
        duration_seconds=6.0,
        byte_length=4096,
        sha256="b" * 64,
        created_at=stored.created_at if stored is not None else "",
    )
    index_columns = [
        row[2]
        for row in conn.execute("PRAGMA index_info(idx_retained_audio_source_time)")
    ]
    assert index_columns == ["source_url", "start_ts", "duration_seconds"]


@pytest.mark.parametrize(
    ("helper", "kwargs", "message"),
    [
        ("song", {"fingerprint": ""}, "fingerprint"),
        ("song", {"segment_sequence": -1}, "segment_sequence"),
        ("song", {"duration_seconds": -0.1}, "duration_seconds"),
        ("song", {"score": -0.01}, "score"),
        ("song", {"score": 1.01}, "score"),
        ("cache", {"fingerprint": ""}, "fingerprint"),
        ("cache", {"score": 1.01}, "score"),
        ("retained", {"path": ""}, "path"),
        ("retained", {"sample_rate": 0}, "sample_rate"),
        ("retained", {"channels": 0}, "channels"),
        ("retained", {"sha256": "not-hex"}, "sha256"),
    ],
)
def test_fingerprint_helpers_reject_malformed_inputs_without_leaking_private_values(helper, kwargs, message):
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    segment_id = insert_fixture_segment(conn)
    private_values = {
        "song": {
            "segment_id": segment_id,
            "source_url": "https://example.test/private/stream.m3u8?token=secret",
            "segment_sequence": 3,
            "start_ts": 18.0,
            "duration_seconds": 6.0,
            "fingerprint": "raw-secret-fingerprint",
            "acoustid_id": None,
            "recording_id": None,
            "title": "secret title",
            "artist": None,
            "album": None,
            "score": None,
            "lookup_source": "fixture",
        },
        "cache": {
            "fingerprint": "raw-secret-fingerprint",
            "acoustid_id": None,
            "recording_id": None,
            "title": "secret title",
            "artist": None,
            "album": None,
            "score": None,
            "raw_status": "secret status",
            "lookup_source": "fixture",
        },
        "retained": {
            "segment_id": segment_id,
            "source_url": "https://example.test/private/stream.m3u8?token=secret",
            "segment_sequence": 3,
            "path": "/private/tmp/raw-secret.wav",
            "format": "wav",
            "sample_rate": 44100,
            "channels": 2,
            "sample_format": "s16le",
            "start_ts": 18.0,
            "duration_seconds": 6.0,
            "byte_length": 4096,
            "sha256": "c" * 64,
        },
    }
    values = private_values[helper]
    values.update(kwargs)

    fn = {
        "song": insert_song,
        "cache": insert_fingerprint_cache,
        "retained": insert_retained_audio,
    }[helper]
    with pytest.raises((TypeError, ValueError), match=message) as exc_info:
        fn(conn, **values)

    public_message = str(exc_info.value)
    assert "secret" not in public_message
    assert "example.test" not in public_message
    assert "/private" not in public_message
    assert "raw-secret-fingerprint" not in public_message


def test_fingerprint_helpers_before_migration_fail_through_sqlite():
    conn = sqlite3.connect(":memory:")

    with pytest.raises(sqlite3.OperationalError, match=re.escape("no such table: songs")):
        insert_song(
            conn,
            segment_id=1,
            source_url="fixture://stream",
            segment_sequence=0,
            start_ts=0,
            duration_seconds=1,
            fingerprint="fingerprint-v1",
            lookup_source=None,
        )

    with pytest.raises(sqlite3.OperationalError, match=re.escape("no such table: fingerprint_cache")):
        insert_fingerprint_cache(
            conn,
            fingerprint="fingerprint-v1",
            lookup_source=None,
        )

    with pytest.raises(sqlite3.OperationalError, match=re.escape("no such table: retained_audio")):
        insert_retained_audio(
            conn,
            segment_id=1,
            source_url="fixture://stream",
            segment_sequence=0,
            path="retained.wav",
            format="wav",
            sample_rate=44100,
            channels=2,
            sample_format="s16le",
            start_ts=0,
            duration_seconds=1,
            byte_length=10,
            sha256="d" * 64,
        )
