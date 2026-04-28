"""SQLite persistence helpers for ad marker events.

The store layer is intentionally explicit and side-effect isolated: callers provide
a path or connection, opt into migration, and receive standard sqlite3 exceptions
when an operation fails. Marker payloads and source URLs are never printed or
logged here.
"""

from __future__ import annotations

import sqlite3
from os import PathLike
from typing import Any

from tidemark.markers import AdMarker

SCHEMA_VERSION = 1

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


def connect_db(path: str | bytes | PathLike[str] | PathLike[bytes]) -> sqlite3.Connection:
    """Open a SQLite connection for a caller-provided database path."""
    return sqlite3.connect(path)


def migrate(conn: sqlite3.Connection) -> None:
    """Create the M001 ad_events schema and advance user_version to v1.

    The migration is idempotent. Existing databases with a newer user_version are
    left at that newer version so an older helper never downgrades schema state.
    """
    with conn:
        conn.execute(_CREATE_AD_EVENTS_SQL)
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
