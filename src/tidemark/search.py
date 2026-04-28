"""Transcript search helpers for schema-v3 stored word timestamps.

The search layer is intentionally library-first and side-effect quiet: functions
return typed results or raise redacted exceptions, and never print/log query text,
paths, URLs, or transcript contents in diagnostics.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

from tidemark.store import initialize_db


@dataclass(frozen=True)
class TranscriptSearchResult:
    """One transcript phrase hit with timestamped context."""

    source_url: str
    segment_id: int
    segment_sequence: int
    hit_start_ts: float
    hit_end_ts: float
    context_start_ts: float
    context_end_ts: float
    context_text: str
    matched_text: str
    word_ids: tuple[int, ...]


class TranscriptSearchError(ValueError):
    """Base class for redacted transcript search errors."""


class MalformedSearchQuery(TranscriptSearchError):
    """Raised when query or context parameters are invalid."""


class TranscriptDatabaseMissing(TranscriptSearchError):
    """Raised when a path-level transcript database does not exist."""


class TranscriptDatabaseEmpty(TranscriptSearchError):
    """Raised when a migrated database has no transcript rows to search."""


class _WordRow(NamedTuple):
    id: int
    segment_id: int
    source_url: str
    segment_sequence: int
    word_index: int
    word_text: str
    start_ts: float
    end_ts: float


_ORDERED_TRANSCRIPT_WORDS_SQL = """
SELECT id, segment_id, source_url, segment_sequence, word_index, word_text, start_ts, end_ts
FROM transcript_words
ORDER BY source_url ASC, start_ts ASC, end_ts ASC, segment_sequence ASC, word_index ASC, id ASC
"""


def _validate_search_inputs(query: str, context_seconds: float) -> tuple[tuple[str, ...], float]:
    if not isinstance(query, str):
        raise MalformedSearchQuery("query must be a non-empty string")
    normalized_query = query.strip()
    if not normalized_query:
        raise MalformedSearchQuery("query must be a non-empty string")
    if not isinstance(context_seconds, (int, float)) or isinstance(context_seconds, bool):
        raise MalformedSearchQuery("context_seconds must be a non-negative number")
    normalized_context = float(context_seconds)
    if normalized_context < 0:
        raise MalformedSearchQuery("context_seconds must be a non-negative number")

    return tuple(token.casefold() for token in normalized_query.split()), normalized_context


def _load_words(conn: sqlite3.Connection) -> tuple[_WordRow, ...]:
    try:
        rows = conn.execute(_ORDERED_TRANSCRIPT_WORDS_SQL).fetchall()
    except sqlite3.Error as exc:
        raise TranscriptSearchError("database read failed during transcript search") from exc
    if not rows:
        raise TranscriptDatabaseEmpty("transcript_words table is empty")
    return tuple(
        _WordRow(
            id=int(row[0]),
            segment_id=int(row[1]),
            source_url=row[2],
            segment_sequence=int(row[3]),
            word_index=int(row[4]),
            word_text=row[5],
            start_ts=float(row[6]),
            end_ts=float(row[7]),
        )
        for row in rows
    )


def _same_source_groups(words: Iterable[_WordRow]) -> Iterable[tuple[_WordRow, ...]]:
    group: list[_WordRow] = []
    current_source: str | None = None
    for word in words:
        if current_source is None:
            current_source = word.source_url
        if word.source_url != current_source:
            yield tuple(group)
            group = []
            current_source = word.source_url
        group.append(word)
    if group:
        yield tuple(group)


def _words_overlap_window(words: tuple[_WordRow, ...], start_ts: float, end_ts: float) -> tuple[_WordRow, ...]:
    return tuple(word for word in words if word.end_ts >= start_ts and word.start_ts <= end_ts)


def _result_for_match(
    source_words: tuple[_WordRow, ...],
    match_start: int,
    query_length: int,
    context_seconds: float,
) -> TranscriptSearchResult:
    matched_words = source_words[match_start : match_start + query_length]
    first = matched_words[0]
    last = matched_words[-1]
    window_start = max(0.0, first.start_ts - context_seconds)
    window_end = last.end_ts + context_seconds
    context_words = _words_overlap_window(source_words, window_start, window_end)

    return TranscriptSearchResult(
        source_url=first.source_url,
        segment_id=first.segment_id,
        segment_sequence=first.segment_sequence,
        hit_start_ts=first.start_ts,
        hit_end_ts=last.end_ts,
        context_start_ts=context_words[0].start_ts,
        context_end_ts=context_words[-1].end_ts,
        context_text=" ".join(word.word_text for word in context_words),
        matched_text=" ".join(word.word_text for word in matched_words),
        word_ids=tuple(word.id for word in matched_words),
    )


def search_transcripts(
    conn: sqlite3.Connection,
    query: str,
    *,
    context_seconds: float = 5.0,
) -> tuple[TranscriptSearchResult, ...]:
    """Search stored transcript words for adjacent case-insensitive phrase hits.

    The caller owns the connection lifecycle. Validation happens before any table
    reads so malformed parameters fail without depending on database state.
    """
    query_tokens, normalized_context = _validate_search_inputs(query, context_seconds)
    words = _load_words(conn)
    query_length = len(query_tokens)
    results: list[TranscriptSearchResult] = []

    for source_words in _same_source_groups(words):
        source_tokens = tuple(word.word_text.casefold() for word in source_words)
        last_start = len(source_tokens) - query_length
        for index in range(last_start + 1):
            if source_tokens[index : index + query_length] == query_tokens:
                results.append(_result_for_match(source_words, index, query_length, normalized_context))

    return tuple(results)


def search_transcript_db(
    path: str | Path,
    query: str,
    *,
    context_seconds: float = 5.0,
) -> tuple[TranscriptSearchResult, ...]:
    """Open, migrate, search, and close an existing transcript database path."""
    db_path = Path(path)
    if not db_path.exists():
        raise TranscriptDatabaseMissing("database path does not exist")

    conn = initialize_db(db_path)
    try:
        return search_transcripts(conn, query, context_seconds=context_seconds)
    finally:
        conn.close()
