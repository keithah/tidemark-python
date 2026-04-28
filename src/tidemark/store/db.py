"""SQLite persistence helpers for ad marker events.

The store layer is intentionally explicit and side-effect isolated: callers provide
a path or connection, opt into migration, and receive standard sqlite3 exceptions
when an operation fails. Marker payloads and source URLs are never printed or
logged here.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from os import PathLike
from typing import Any

from tidemark.markers import AdMarker
from tidemark.transcribe import WordToken

SCHEMA_VERSION = 4

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class SegmentStoreRecord:
    """Normalized segment row returned from the SQLite handoff store."""

    id: int
    source_url: str
    sequence: int
    resolved_uri: str
    local_path: str | None
    start_ts: float
    duration_seconds: float
    byte_length: int
    sha256: str
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class TranscriptWordStoreRecord:
    """Normalized transcript word row returned from the SQLite handoff store."""

    id: int
    segment_id: int
    source_url: str
    segment_sequence: int
    word_index: int
    word_text: str
    start_ts: float
    end_ts: float
    confidence: float | None
    created_at: str


@dataclass(frozen=True)
class SongStoreRecord:
    """Normalized song/fingerprint evidence row returned from the store."""

    id: int
    segment_id: int
    source_url: str
    segment_sequence: int
    start_ts: float
    duration_seconds: float
    fingerprint: str
    acoustid_id: str | None
    recording_id: str | None
    title: str | None
    artist: str | None
    album: str | None
    score: float | None
    lookup_source: str | None
    created_at: str


@dataclass(frozen=True)
class FingerprintCacheRecord:
    """Normalized cache row for a fingerprint lookup result."""

    fingerprint: str
    acoustid_id: str | None
    recording_id: str | None
    title: str | None
    artist: str | None
    album: str | None
    score: float | None
    raw_status: str | None
    lookup_source: str | None
    cached_at: str


@dataclass(frozen=True)
class RetainedAudioStoreRecord:
    """Normalized retained-audio metadata row returned from the store."""

    id: int
    segment_id: int
    source_url: str
    segment_sequence: int
    path: str
    format: str
    sample_rate: int
    channels: int
    sample_format: str
    start_ts: float
    duration_seconds: float
    byte_length: int
    sha256: str
    created_at: str


_CREATE_AD_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS ad_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url TEXT NOT NULL,
    marker_type TEXT NOT NULL,
    classification TEXT NOT NULL,
    source TEXT NOT NULL,
    tag TEXT,
    segment_seq INTEGER,
    pts REAL,
    break_duration REAL,
    raw_json TEXT NOT NULL,
    ts REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"""

_INSERT_AD_EVENT_SQL = """
INSERT INTO ad_events (
    source_url,
    marker_type,
    classification,
    source,
    tag,
    segment_seq,
    pts,
    break_duration,
    raw_json,
    ts
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_CREATE_SEGMENTS_SQL = """
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    resolved_uri TEXT NOT NULL,
    local_path TEXT,
    start_ts REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    byte_length INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"""

_CREATE_TRANSCRIPT_WORDS_SQL = """
CREATE TABLE IF NOT EXISTS transcript_words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL,
    source_url TEXT NOT NULL,
    segment_sequence INTEGER NOT NULL,
    word_index INTEGER NOT NULL,
    word_text TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (segment_id) REFERENCES segments(id)
)
"""

_CREATE_SONGS_SQL = """
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL,
    source_url TEXT NOT NULL,
    segment_sequence INTEGER NOT NULL,
    start_ts REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    fingerprint TEXT NOT NULL,
    acoustid_id TEXT,
    recording_id TEXT,
    title TEXT,
    artist TEXT,
    album TEXT,
    score REAL,
    lookup_source TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (segment_id) REFERENCES segments(id)
)
"""

_CREATE_FINGERPRINT_CACHE_SQL = """
CREATE TABLE IF NOT EXISTS fingerprint_cache (
    fingerprint TEXT PRIMARY KEY,
    acoustid_id TEXT,
    recording_id TEXT,
    title TEXT,
    artist TEXT,
    album TEXT,
    score REAL,
    raw_status TEXT,
    lookup_source TEXT,
    cached_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"""

_CREATE_RETAINED_AUDIO_SQL = """
CREATE TABLE IF NOT EXISTS retained_audio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL,
    source_url TEXT NOT NULL,
    segment_sequence INTEGER NOT NULL,
    path TEXT NOT NULL,
    format TEXT NOT NULL,
    sample_rate INTEGER NOT NULL,
    channels INTEGER NOT NULL,
    sample_format TEXT NOT NULL,
    start_ts REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    byte_length INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (segment_id) REFERENCES segments(id)
)
"""

_CREATE_TRANSCRIPT_WORDS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_transcript_words_segment_order "
    "ON transcript_words(segment_id, word_index)",
    "CREATE INDEX IF NOT EXISTS idx_transcript_words_source_time "
    "ON transcript_words(source_url, start_ts, end_ts)",
    "CREATE INDEX IF NOT EXISTS idx_transcript_words_word "
    "ON transcript_words(word_text)",
)

_CREATE_RETAINED_AUDIO_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_retained_audio_source_time "
    "ON retained_audio(source_url, start_ts, duration_seconds)",
)

_CREATE_SEGMENTS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_segments_restart_evidence "
    "ON segments(source_url, sequence, sha256)",
)

_INSERT_SEGMENT_SQL = """
INSERT INTO segments (
    source_url,
    sequence,
    resolved_uri,
    local_path,
    start_ts,
    duration_seconds,
    byte_length,
    sha256,
    metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_TRANSCRIPT_WORD_SQL = """
INSERT INTO transcript_words (
    segment_id,
    source_url,
    segment_sequence,
    word_index,
    word_text,
    start_ts,
    end_ts,
    confidence
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_SONG_SQL = """
INSERT INTO songs (
    segment_id,
    source_url,
    segment_sequence,
    start_ts,
    duration_seconds,
    fingerprint,
    acoustid_id,
    recording_id,
    title,
    artist,
    album,
    score,
    lookup_source
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPSERT_FINGERPRINT_CACHE_SQL = """
INSERT INTO fingerprint_cache (
    fingerprint,
    acoustid_id,
    recording_id,
    title,
    artist,
    album,
    score,
    raw_status,
    lookup_source
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(fingerprint) DO UPDATE SET
    acoustid_id = excluded.acoustid_id,
    recording_id = excluded.recording_id,
    title = excluded.title,
    artist = excluded.artist,
    album = excluded.album,
    score = excluded.score,
    raw_status = excluded.raw_status,
    lookup_source = excluded.lookup_source,
    cached_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
"""

_INSERT_RETAINED_AUDIO_SQL = """
INSERT INTO retained_audio (
    segment_id,
    source_url,
    segment_sequence,
    path,
    format,
    sample_rate,
    channels,
    sample_format,
    start_ts,
    duration_seconds,
    byte_length,
    sha256
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def connect_db(path: str | bytes | PathLike[str] | PathLike[bytes]) -> sqlite3.Connection:
    """Open a SQLite connection for a caller-provided database path."""
    return sqlite3.connect(path)


def migrate(conn: sqlite3.Connection) -> None:
    """Create the SQLite store schema and advance user_version to the current version.

    The migration is idempotent. Existing databases with a newer user_version are
    left at that newer version so an older helper never downgrades schema state.
    """
    with conn:
        conn.execute(_CREATE_AD_EVENTS_SQL)
        conn.execute(_CREATE_SEGMENTS_SQL)
        conn.execute(_CREATE_TRANSCRIPT_WORDS_SQL)
        conn.execute(_CREATE_SONGS_SQL)
        conn.execute(_CREATE_FINGERPRINT_CACHE_SQL)
        conn.execute(_CREATE_RETAINED_AUDIO_SQL)
        for statement in _CREATE_TRANSCRIPT_WORDS_INDEX_SQL:
            conn.execute(statement)
        for statement in _CREATE_RETAINED_AUDIO_INDEX_SQL:
            conn.execute(statement)
        for statement in _CREATE_SEGMENTS_INDEX_SQL:
            conn.execute(statement)
        current_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if current_version < SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def initialize_db(path: str | bytes | PathLike[str] | PathLike[bytes]) -> sqlite3.Connection:
    """Open a SQLite database path, migrate it, and return the connection."""
    conn = connect_db(path)
    try:
        migrate(conn)
    except Exception:
        conn.close()
        raise
    return conn


def insert_ad_event(conn: sqlite3.Connection, source_url: str, marker: AdMarker) -> int:
    """Insert one ad marker event and return its SQLite row id."""
    if not isinstance(marker, AdMarker):
        raise TypeError("insert_ad_event() marker must be an AdMarker")

    raw_json = marker.to_json()
    values: tuple[Any, ...] = (
        source_url,
        marker.type,
        marker.classification,
        marker.source,
        marker.tag,
        marker.segment,
        marker.pts,
        marker.break_duration,
        raw_json,
        float(marker.timestamp),
    )
    with conn:
        cursor = conn.execute(_INSERT_AD_EVENT_SQL, values)
    return int(cursor.lastrowid)


def _require_non_empty_string(name: str, value: str, *, function_name: str = "insert_segment") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{function_name}() {name} must be a non-empty string")
    if not value.strip():
        raise ValueError(f"{function_name}() {name} must be a non-empty string")
    return value


def _require_optional_string(name: str, value: str | None, *, function_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{function_name}() {name} must be a string or None")
    return value


def _require_int(name: str, value: int, *, minimum: int | None = None, function_name: str = "insert_segment") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{function_name}() {name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{function_name}() {name} must be >= {minimum}")
    return value


def _require_number(
    name: str,
    value: int | float,
    *,
    minimum: float | None = None,
    function_name: str = "insert_segment",
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{function_name}() {name} must be a number")
    normalized = float(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{function_name}() {name} must be >= {minimum}")
    return normalized


def _require_score(score: int | float | None, *, function_name: str) -> float | None:
    if score is None:
        return None
    normalized = _require_number("score", score, function_name=function_name)
    if normalized < 0 or normalized > 1:
        raise ValueError(f"{function_name}() score must be between 0 and 1")
    return normalized


def _normalize_metadata(metadata: dict[str, Any] | None) -> str | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise TypeError("insert_segment() metadata must be a JSON object or None")
    try:
        return json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise TypeError("insert_segment() metadata must be JSON serializable") from exc


def _normalize_sha256(sha256: str, *, function_name: str = "insert_segment") -> str:
    if not isinstance(sha256, str):
        raise TypeError(f"{function_name}() sha256 must be a 64-character hex string")
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError(f"{function_name}() sha256 must be a 64-character hex string")
    return sha256.lower()


def insert_segment(
    conn: sqlite3.Connection,
    *,
    source_url: str,
    sequence: int,
    resolved_uri: str,
    local_path: str | None,
    start_ts: int | float,
    duration_seconds: int | float,
    byte_length: int,
    sha256: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Insert one resolved media segment and return its SQLite row id.

    Validation errors intentionally name only field names, not URLs, paths, or
    metadata values, so public exceptions remain useful without leaking inputs.
    """
    normalized_source_url = _require_non_empty_string("source_url", source_url)
    normalized_sequence = _require_int("sequence", sequence, minimum=0)
    normalized_resolved_uri = _require_non_empty_string("resolved_uri", resolved_uri)
    if local_path is not None and not isinstance(local_path, str):
        raise TypeError("insert_segment() local_path must be a string or None")
    normalized_start_ts = _require_number("start_ts", start_ts, minimum=0)
    normalized_duration = _require_number("duration_seconds", duration_seconds, minimum=0)
    normalized_byte_length = _require_int("byte_length", byte_length, minimum=0)
    normalized_sha256 = _normalize_sha256(sha256)
    metadata_json = _normalize_metadata(metadata)

    values: tuple[Any, ...] = (
        normalized_source_url,
        normalized_sequence,
        normalized_resolved_uri,
        local_path,
        normalized_start_ts,
        normalized_duration,
        normalized_byte_length,
        normalized_sha256,
        metadata_json,
    )
    with conn:
        cursor = conn.execute(_INSERT_SEGMENT_SQL, values)
    return int(cursor.lastrowid)


def get_segment(conn: sqlite3.Connection, row_id: int) -> SegmentStoreRecord | None:
    """Fetch one segment row by id as a typed record, or None when missing."""
    row = conn.execute(
        """
        SELECT id, source_url, sequence, resolved_uri, local_path, start_ts,
               duration_seconds, byte_length, sha256, metadata_json
        FROM segments
        WHERE id = ?
        """,
        (row_id,),
    ).fetchone()
    if row is None:
        return None
    return _segment_record_from_row(row)


def find_segment_by_restart_evidence(
    conn: sqlite3.Connection,
    *,
    source_url: str,
    sequence: int,
    sha256: str,
    byte_length: int,
) -> SegmentStoreRecord | None:
    """Fetch the oldest segment row matching immutable restart evidence.

    The lookup is read-only and intentionally validates only field names so
    callers can surface errors without leaking source URLs or content hashes.
    """
    function_name = "find_segment_by_restart_evidence"
    normalized_source_url = _require_non_empty_string("source_url", source_url, function_name=function_name)
    normalized_sequence = _require_int("sequence", sequence, minimum=0, function_name=function_name)
    normalized_sha256 = _normalize_sha256(sha256, function_name=function_name)
    normalized_byte_length = _require_int("byte_length", byte_length, minimum=0, function_name=function_name)
    row = conn.execute(
        """
        SELECT id, source_url, sequence, resolved_uri, local_path, start_ts,
               duration_seconds, byte_length, sha256, metadata_json
        FROM segments
        WHERE source_url = ?
          AND sequence = ?
          AND sha256 = ?
          AND byte_length = ?
        ORDER BY id ASC
        LIMIT 1
        """,
        (normalized_source_url, normalized_sequence, normalized_sha256, normalized_byte_length),
    ).fetchone()
    if row is None:
        return None
    return _segment_record_from_row(row)


def _segment_record_from_row(row: sqlite3.Row | tuple[Any, ...]) -> SegmentStoreRecord:
    metadata_json = row[9]
    metadata = json.loads(metadata_json) if metadata_json is not None else None
    return SegmentStoreRecord(
        id=int(row[0]),
        source_url=row[1],
        sequence=int(row[2]),
        resolved_uri=row[3],
        local_path=row[4],
        start_ts=float(row[5]),
        duration_seconds=float(row[6]),
        byte_length=int(row[7]),
        sha256=row[8],
        metadata=metadata,
    )


def _normalize_transcript_word(word: WordToken, index: int) -> tuple[int, str, float, float, float | None]:
    if not isinstance(word, WordToken):
        raise TypeError("insert_transcript_words() words must contain WordToken values")

    text = _require_non_empty_string("word.text", word.text, function_name="insert_transcript_words")
    start_ts = _require_number("word.start_ts", word.start_ts, minimum=0, function_name="insert_transcript_words")
    end_ts = _require_number("word.end_ts", word.end_ts, minimum=0, function_name="insert_transcript_words")
    if end_ts < start_ts:
        raise ValueError("insert_transcript_words() word.end_ts must be >= word.start_ts")

    confidence = word.confidence
    if confidence is not None:
        confidence = _require_number("word.confidence", confidence, function_name="insert_transcript_words")
        if confidence < 0 or confidence > 1:
            raise ValueError("insert_transcript_words() word.confidence must be between 0 and 1")

    return index, text, start_ts, end_ts, confidence


def insert_transcript_words(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    source_url: str,
    segment_sequence: int,
    words: tuple[WordToken, ...],
) -> tuple[int, ...]:
    """Insert transcript words for one segment and return their row ids.

    Empty word batches are accepted and return an empty tuple without writing any
    rows. Validation errors intentionally name only fields and function names;
    transcript text, URLs, paths, and metadata values are never echoed.
    """
    normalized_segment_id = _require_int(
        "segment_id", segment_id, minimum=0, function_name="insert_transcript_words"
    )
    normalized_source_url = _require_non_empty_string(
        "source_url", source_url, function_name="insert_transcript_words"
    )
    normalized_segment_sequence = _require_int(
        "segment_sequence", segment_sequence, minimum=0, function_name="insert_transcript_words"
    )
    if not isinstance(words, tuple):
        raise TypeError("insert_transcript_words() words must be a tuple of WordToken values")

    normalized_words = tuple(_normalize_transcript_word(word, index) for index, word in enumerate(words))
    if not normalized_words:
        return ()

    row_ids: list[int] = []
    with conn:
        for word_index, word_text, start_ts, end_ts, confidence in normalized_words:
            cursor = conn.execute(
                _INSERT_TRANSCRIPT_WORD_SQL,
                (
                    normalized_segment_id,
                    normalized_source_url,
                    normalized_segment_sequence,
                    word_index,
                    word_text,
                    start_ts,
                    end_ts,
                    confidence,
                ),
            )
            row_ids.append(int(cursor.lastrowid))
    return tuple(row_ids)


def get_transcript_words_for_segment(
    conn: sqlite3.Connection,
    segment_id: int,
) -> tuple[TranscriptWordStoreRecord, ...]:
    """Fetch transcript words for one segment ordered by their stored word index."""
    normalized_segment_id = _require_int(
        "segment_id", segment_id, minimum=0, function_name="get_transcript_words_for_segment"
    )
    rows = conn.execute(
        """
        SELECT id, segment_id, source_url, segment_sequence, word_index,
               word_text, start_ts, end_ts, confidence, created_at
        FROM transcript_words
        WHERE segment_id = ?
        ORDER BY word_index ASC, id ASC
        """,
        (normalized_segment_id,),
    ).fetchall()
    return tuple(
        TranscriptWordStoreRecord(
            id=int(row[0]),
            segment_id=int(row[1]),
            source_url=row[2],
            segment_sequence=int(row[3]),
            word_index=int(row[4]),
            word_text=row[5],
            start_ts=float(row[6]),
            end_ts=float(row[7]),
            confidence=None if row[8] is None else float(row[8]),
            created_at=row[9],
        )
        for row in rows
    )


def insert_song(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    source_url: str,
    segment_sequence: int,
    start_ts: int | float,
    duration_seconds: int | float,
    fingerprint: str,
    acoustid_id: str | None = None,
    recording_id: str | None = None,
    title: str | None = None,
    artist: str | None = None,
    album: str | None = None,
    score: int | float | None = None,
    lookup_source: str | None = None,
) -> int:
    """Insert one song/fingerprint evidence row and return its SQLite row id."""
    function_name = "insert_song"
    values: tuple[Any, ...] = (
        _require_int("segment_id", segment_id, minimum=0, function_name=function_name),
        _require_non_empty_string("source_url", source_url, function_name=function_name),
        _require_int("segment_sequence", segment_sequence, minimum=0, function_name=function_name),
        _require_number("start_ts", start_ts, minimum=0, function_name=function_name),
        _require_number("duration_seconds", duration_seconds, minimum=0, function_name=function_name),
        _require_non_empty_string("fingerprint", fingerprint, function_name=function_name),
        _require_optional_string("acoustid_id", acoustid_id, function_name=function_name),
        _require_optional_string("recording_id", recording_id, function_name=function_name),
        _require_optional_string("title", title, function_name=function_name),
        _require_optional_string("artist", artist, function_name=function_name),
        _require_optional_string("album", album, function_name=function_name),
        _require_score(score, function_name=function_name),
        _require_optional_string("lookup_source", lookup_source, function_name=function_name),
    )
    with conn:
        cursor = conn.execute(_INSERT_SONG_SQL, values)
    return int(cursor.lastrowid)


def get_song(conn: sqlite3.Connection, row_id: int) -> SongStoreRecord | None:
    """Fetch one song/fingerprint evidence row by id, or None when missing."""
    normalized_row_id = _require_int("row_id", row_id, minimum=0, function_name="get_song")
    row = conn.execute(
        """
        SELECT id, segment_id, source_url, segment_sequence, start_ts,
               duration_seconds, fingerprint, acoustid_id, recording_id,
               title, artist, album, score, lookup_source, created_at
        FROM songs
        WHERE id = ?
        """,
        (normalized_row_id,),
    ).fetchone()
    if row is None:
        return None
    return SongStoreRecord(
        id=int(row[0]),
        segment_id=int(row[1]),
        source_url=row[2],
        segment_sequence=int(row[3]),
        start_ts=float(row[4]),
        duration_seconds=float(row[5]),
        fingerprint=row[6],
        acoustid_id=row[7],
        recording_id=row[8],
        title=row[9],
        artist=row[10],
        album=row[11],
        score=None if row[12] is None else float(row[12]),
        lookup_source=row[13],
        created_at=row[14],
    )


def insert_fingerprint_cache(
    conn: sqlite3.Connection,
    *,
    fingerprint: str,
    acoustid_id: str | None = None,
    recording_id: str | None = None,
    title: str | None = None,
    artist: str | None = None,
    album: str | None = None,
    score: int | float | None = None,
    raw_status: str | None = None,
    lookup_source: str | None = None,
) -> str:
    """Upsert one fingerprint lookup cache row and return its fingerprint key."""
    function_name = "insert_fingerprint_cache"
    normalized_fingerprint = _require_non_empty_string("fingerprint", fingerprint, function_name=function_name)
    values: tuple[Any, ...] = (
        normalized_fingerprint,
        _require_optional_string("acoustid_id", acoustid_id, function_name=function_name),
        _require_optional_string("recording_id", recording_id, function_name=function_name),
        _require_optional_string("title", title, function_name=function_name),
        _require_optional_string("artist", artist, function_name=function_name),
        _require_optional_string("album", album, function_name=function_name),
        _require_score(score, function_name=function_name),
        _require_optional_string("raw_status", raw_status, function_name=function_name),
        _require_optional_string("lookup_source", lookup_source, function_name=function_name),
    )
    with conn:
        conn.execute(_UPSERT_FINGERPRINT_CACHE_SQL, values)
    return normalized_fingerprint


def get_fingerprint_cache(conn: sqlite3.Connection, fingerprint: str) -> FingerprintCacheRecord | None:
    """Fetch one fingerprint cache row by fingerprint, or None when missing."""
    normalized_fingerprint = _require_non_empty_string(
        "fingerprint", fingerprint, function_name="get_fingerprint_cache"
    )
    row = conn.execute(
        """
        SELECT fingerprint, acoustid_id, recording_id, title, artist, album,
               score, raw_status, lookup_source, cached_at
        FROM fingerprint_cache
        WHERE fingerprint = ?
        """,
        (normalized_fingerprint,),
    ).fetchone()
    if row is None:
        return None
    return FingerprintCacheRecord(
        fingerprint=row[0],
        acoustid_id=row[1],
        recording_id=row[2],
        title=row[3],
        artist=row[4],
        album=row[5],
        score=None if row[6] is None else float(row[6]),
        raw_status=row[7],
        lookup_source=row[8],
        cached_at=row[9],
    )


def insert_retained_audio(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    source_url: str,
    segment_sequence: int,
    path: str,
    format: str,
    sample_rate: int,
    channels: int,
    sample_format: str,
    start_ts: int | float,
    duration_seconds: int | float,
    byte_length: int,
    sha256: str,
) -> int:
    """Insert one retained-audio metadata row and return its SQLite row id."""
    function_name = "insert_retained_audio"
    values: tuple[Any, ...] = (
        _require_int("segment_id", segment_id, minimum=0, function_name=function_name),
        _require_non_empty_string("source_url", source_url, function_name=function_name),
        _require_int("segment_sequence", segment_sequence, minimum=0, function_name=function_name),
        _require_non_empty_string("path", path, function_name=function_name),
        _require_non_empty_string("format", format, function_name=function_name),
        _require_int("sample_rate", sample_rate, minimum=1, function_name=function_name),
        _require_int("channels", channels, minimum=1, function_name=function_name),
        _require_non_empty_string("sample_format", sample_format, function_name=function_name),
        _require_number("start_ts", start_ts, minimum=0, function_name=function_name),
        _require_number("duration_seconds", duration_seconds, minimum=0, function_name=function_name),
        _require_int("byte_length", byte_length, minimum=0, function_name=function_name),
        _normalize_sha256(sha256, function_name=function_name),
    )
    with conn:
        cursor = conn.execute(_INSERT_RETAINED_AUDIO_SQL, values)
    return int(cursor.lastrowid)


def get_retained_audio(conn: sqlite3.Connection, row_id: int) -> RetainedAudioStoreRecord | None:
    """Fetch one retained-audio metadata row by id, or None when missing."""
    normalized_row_id = _require_int("row_id", row_id, minimum=0, function_name="get_retained_audio")
    row = conn.execute(
        """
        SELECT id, segment_id, source_url, segment_sequence, path, format,
               sample_rate, channels, sample_format, start_ts, duration_seconds,
               byte_length, sha256, created_at
        FROM retained_audio
        WHERE id = ?
        """,
        (normalized_row_id,),
    ).fetchone()
    if row is None:
        return None
    return _retained_audio_record_from_row(row)


def find_retained_audio_covering(conn: sqlite3.Connection, at_seconds: int | float) -> RetainedAudioStoreRecord | None:
    """Return the deterministic retained-audio row covering ``at_seconds``.

    The query intentionally does not expose paths or source URLs in validation
    errors; callers that surface errors should keep DB and retained paths
    redacted as well.
    """
    normalized_at = _require_number("at_seconds", at_seconds, minimum=0, function_name="find_retained_audio_covering")
    row = conn.execute(
        """
        SELECT id, segment_id, source_url, segment_sequence, path, format,
               sample_rate, channels, sample_format, start_ts, duration_seconds,
               byte_length, sha256, created_at
        FROM retained_audio
        WHERE start_ts <= ? AND ? <= start_ts + duration_seconds
        ORDER BY start_ts DESC, duration_seconds ASC, id ASC
        LIMIT 1
        """,
        (normalized_at, normalized_at),
    ).fetchone()
    if row is None:
        return None
    return _retained_audio_record_from_row(row)


def _retained_audio_record_from_row(row: sqlite3.Row | tuple[Any, ...]) -> RetainedAudioStoreRecord:
    return RetainedAudioStoreRecord(
        id=int(row[0]),
        segment_id=int(row[1]),
        source_url=row[2],
        segment_sequence=int(row[3]),
        path=row[4],
        format=row[5],
        sample_rate=int(row[6]),
        channels=int(row[7]),
        sample_format=row[8],
        start_ts=float(row[9]),
        duration_seconds=float(row[10]),
        byte_length=int(row[11]),
        sha256=row[12],
        created_at=row[13],
    )
