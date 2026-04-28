from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tidemark.search import (
    MalformedSearchQuery,
    TranscriptDatabaseEmpty,
    TranscriptDatabaseMissing,
    TranscriptSearchError,
    TranscriptSearchResult,
    search_transcript_db,
    search_transcripts,
)
from tidemark.store import insert_segment, insert_transcript_words, migrate
from tidemark.transcribe import WordToken


def _connect_migrated() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    return conn


def _insert_segment_words(
    conn: sqlite3.Connection,
    *,
    source_url: str = "fixture://stream-a",
    sequence: int = 1,
    start_ts: float = 0.0,
    words: tuple[str, ...],
) -> tuple[int, ...]:
    segment_id = insert_segment(
        conn,
        source_url=source_url,
        sequence=sequence,
        resolved_uri=f"fixture://segment-{sequence}.ts",
        local_path=None,
        start_ts=start_ts,
        duration_seconds=10.0,
        byte_length=1024,
        sha256=f"{sequence:064x}"[-64:],
    )
    tokens = tuple(
        WordToken(text=word, start_ts=start_ts + index, end_ts=start_ts + index + 0.5)
        for index, word in enumerate(words)
    )
    return insert_transcript_words(
        conn,
        segment_id=segment_id,
        source_url=source_url,
        segment_sequence=sequence,
        words=tokens,
    )


def test_search_transcripts_returns_timestamped_context_for_casefolded_adjacent_phrase() -> None:
    conn = _connect_migrated()
    row_ids = _insert_segment_words(
        conn,
        words=("Alpha", "bravo", "CHARLIE", "delta", "echo"),
    )

    results = search_transcripts(conn, "bravo charlie", context_seconds=1.0)

    assert results == (
        TranscriptSearchResult(
            source_url="fixture://stream-a",
            segment_id=1,
            segment_sequence=1,
            hit_start_ts=1.0,
            hit_end_ts=2.5,
            context_start_ts=0.0,
            context_end_ts=3.5,
            context_text="Alpha bravo CHARLIE delta",
            matched_text="bravo CHARLIE",
            word_ids=(row_ids[1], row_ids[2]),
        ),
    )


def test_search_transcripts_groups_by_source_and_orders_repeated_matches() -> None:
    conn = _connect_migrated()
    first_ids = _insert_segment_words(
        conn,
        source_url="fixture://stream-b",
        sequence=2,
        start_ts=20.0,
        words=("needle", "hay", "needle"),
    )
    second_ids = _insert_segment_words(
        conn,
        source_url="fixture://stream-a",
        sequence=1,
        start_ts=10.0,
        words=("needle", "hay"),
    )

    results = search_transcripts(conn, "needle", context_seconds=0)

    assert [(result.source_url, result.hit_start_ts, result.matched_text, result.word_ids) for result in results] == [
        ("fixture://stream-a", 10.0, "needle", (second_ids[0],)),
        ("fixture://stream-b", 20.0, "needle", (first_ids[0],)),
        ("fixture://stream-b", 22.0, "needle", (first_ids[2],)),
    ]
    assert [result.context_text for result in results] == ["needle", "needle", "needle"]


def test_search_transcripts_does_not_match_non_adjacent_phrase() -> None:
    conn = _connect_migrated()
    _insert_segment_words(conn, words=("alpha", "gap", "charlie"))

    assert search_transcripts(conn, "alpha charlie") == ()


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_search_transcripts_rejects_blank_query_without_leaking_value(query: str) -> None:
    conn = _connect_migrated()
    _insert_segment_words(conn, words=("secret",))

    with pytest.raises(MalformedSearchQuery, match="query") as exc_info:
        search_transcripts(conn, query)

    assert "secret" not in str(exc_info.value)


def test_search_transcripts_rejects_negative_context_before_reading_database() -> None:
    conn = sqlite3.connect(":memory:")

    with pytest.raises(MalformedSearchQuery, match="context_seconds"):
        search_transcripts(conn, "needle", context_seconds=-0.1)


def test_search_transcripts_raises_empty_for_migrated_database_without_transcript_words() -> None:
    conn = _connect_migrated()

    with pytest.raises(TranscriptDatabaseEmpty, match="transcript_words"):
        search_transcripts(conn, "needle")


def test_search_transcripts_wraps_database_read_failures_without_leaking_query() -> None:
    conn = sqlite3.connect(":memory:")

    with pytest.raises(TranscriptSearchError, match="database read") as exc_info:
        search_transcripts(conn, "private needle")

    assert "private needle" not in str(exc_info.value)


def test_search_transcript_db_raises_missing_without_creating_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite"

    with pytest.raises(TranscriptDatabaseMissing, match="database"):
        search_transcript_db(path, "needle")

    assert not path.exists()


def test_search_transcript_db_opens_migrates_and_closes_existing_database(tmp_path: Path) -> None:
    path = tmp_path / "transcripts.sqlite"
    conn = sqlite3.connect(path)
    try:
        migrate(conn)
        _insert_segment_words(conn, words=("open", "sesame"))
    finally:
        conn.close()

    results = search_transcript_db(path, "open sesame", context_seconds=0)

    assert [(result.context_text, result.matched_text) for result in results] == [("open sesame", "open sesame")]
    path.unlink()
    assert not path.exists()
