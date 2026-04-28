import re
import sqlite3

import pytest

from tidemark.store import (
    SCHEMA_VERSION,
    TranscriptWordStoreRecord,
    get_transcript_words_for_segment,
    insert_segment,
    insert_transcript_words,
    migrate,
)
from tidemark.transcribe import WordToken


EXPECTED_TABLES = ["ad_events", "fingerprint_cache", "retained_audio", "segments", "songs", "transcript_words"]
EXPECTED_TRANSCRIPT_WORD_COLUMNS = [
    "id",
    "segment_id",
    "source_url",
    "segment_sequence",
    "word_index",
    "word_text",
    "start_ts",
    "end_ts",
    "confidence",
    "created_at",
]
EXPECTED_TRANSCRIPT_WORD_INDEXES = [
    "idx_transcript_words_segment_order",
    "idx_transcript_words_source_time",
    "idx_transcript_words_word",
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
        sequence=7,
        resolved_uri="fixture://segment-7.ts",
        local_path=None,
        start_ts=120.0,
        duration_seconds=6.0,
        byte_length=1024,
        sha256="a" * 64,
    )


def test_migrate_creates_schema_v4_transcript_words_table_and_indexes():
    conn = sqlite3.connect(":memory:")

    migrate(conn)

    assert SCHEMA_VERSION == 4
    assert table_names(conn) == EXPECTED_TABLES
    assert [row[1] for row in conn.execute("PRAGMA table_info(transcript_words)")] == EXPECTED_TRANSCRIPT_WORD_COLUMNS
    assert index_names(conn, "transcript_words") == EXPECTED_TRANSCRIPT_WORD_INDEXES
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 4


def test_insert_transcript_words_preserves_order_context_and_fetches_typed_records():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    segment_id = insert_fixture_segment(conn)
    words = (
        WordToken(text="alpha", start_ts=120.5, end_ts=120.8, confidence=0.91),
        WordToken(text="bravo", start_ts=120.9, end_ts=121.1, confidence=None),
        WordToken(text="charlie", start_ts=121.2, end_ts=121.7, confidence=1.0),
    )

    row_ids = insert_transcript_words(
        conn,
        segment_id=segment_id,
        source_url="fixture://stream",
        segment_sequence=7,
        words=words,
    )

    assert len(row_ids) == 3
    assert row_ids == tuple(sorted(row_ids))
    records = get_transcript_words_for_segment(conn, segment_id)

    assert tuple(record.created_at for record in records)
    assert tuple(
        TranscriptWordStoreRecord(
            id=record.id,
            segment_id=record.segment_id,
            source_url=record.source_url,
            segment_sequence=record.segment_sequence,
            word_index=record.word_index,
            word_text=record.word_text,
            start_ts=record.start_ts,
            end_ts=record.end_ts,
            confidence=record.confidence,
            created_at="<created>",
        )
        for record in records
    ) == (
        TranscriptWordStoreRecord(
            id=row_ids[0],
            segment_id=segment_id,
            source_url="fixture://stream",
            segment_sequence=7,
            word_index=0,
            word_text="alpha",
            start_ts=120.5,
            end_ts=120.8,
            confidence=0.91,
            created_at="<created>",
        ),
        TranscriptWordStoreRecord(
            id=row_ids[1],
            segment_id=segment_id,
            source_url="fixture://stream",
            segment_sequence=7,
            word_index=1,
            word_text="bravo",
            start_ts=120.9,
            end_ts=121.1,
            confidence=None,
            created_at="<created>",
        ),
        TranscriptWordStoreRecord(
            id=row_ids[2],
            segment_id=segment_id,
            source_url="fixture://stream",
            segment_sequence=7,
            word_index=2,
            word_text="charlie",
            start_ts=121.2,
            end_ts=121.7,
            confidence=1.0,
            created_at="<created>",
        ),
    )


def test_insert_transcript_words_accepts_empty_batch_without_creating_rows():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    segment_id = insert_fixture_segment(conn)

    row_ids = insert_transcript_words(
        conn,
        segment_id=segment_id,
        source_url="fixture://stream",
        segment_sequence=7,
        words=(),
    )

    assert row_ids == ()
    assert get_transcript_words_for_segment(conn, segment_id) == ()
    assert conn.execute("select count(*) from transcript_words").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"segment_id": -1}, "segment_id"),
        ({"segment_sequence": -1}, "segment_sequence"),
        ({"words": (WordToken(text="", start_ts=1.0, end_ts=1.1),)}, "word.text"),
        ({"words": (WordToken(text="secret transcript", start_ts=-1.0, end_ts=1.1),)}, "word.start_ts"),
        ({"words": (WordToken(text="secret transcript", start_ts=1.2, end_ts=1.1),)}, "word.end_ts"),
        ({"words": (WordToken(text="secret transcript", start_ts=1.0, end_ts=1.1, confidence=-0.1),)}, "word.confidence"),
        ({"words": (WordToken(text="secret transcript", start_ts=1.0, end_ts=1.1, confidence=1.1),)}, "word.confidence"),
        ({"words": (object(),)}, "WordToken"),
    ],
)
def test_insert_transcript_words_rejects_malformed_inputs_without_leaking_values(kwargs, message):
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    values = {
        "segment_id": 1,
        "source_url": "https://example.test/private/stream.m3u8?token=secret",
        "segment_sequence": 1,
        "words": (WordToken(text="secret transcript", start_ts=1.0, end_ts=1.1, confidence=0.5),),
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message) as exc_info:
        insert_transcript_words(conn, **values)

    public_message = str(exc_info.value)
    assert "secret" not in public_message
    assert "example.test" not in public_message
    assert "/private" not in public_message


def test_insert_transcript_words_before_migration_fails_through_sqlite():
    conn = sqlite3.connect(":memory:")

    with pytest.raises(sqlite3.OperationalError, match=re.escape("no such table: transcript_words")):
        insert_transcript_words(
            conn,
            segment_id=1,
            source_url="fixture://stream",
            segment_sequence=0,
            words=(WordToken(text="alpha", start_ts=0.0, end_ts=0.1),),
        )
