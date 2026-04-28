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

SCHEMA_VERSION = 2

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


def _require_non_empty_string(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"insert_segment() {name} must be a non-empty string")
    if not value.strip():
        raise ValueError(f"insert_segment() {name} must be a non-empty string")
    return value


def _require_int(name: str, value: int, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"insert_segment() {name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"insert_segment() {name} must be >= {minimum}")
    return value


def _require_number(name: str, value: int | float, *, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"insert_segment() {name} must be a number")
    normalized = float(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"insert_segment() {name} must be >= {minimum}")
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


def _normalize_sha256(sha256: str) -> str:
    if not isinstance(sha256, str):
        raise TypeError("insert_segment() sha256 must be a 64-character hex string")
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError("insert_segment() sha256 must be a 64-character hex string")
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
