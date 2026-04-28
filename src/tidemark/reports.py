"""Schema-v4 report helpers for persisted timeline rows.

The report layer is library-first and side-effect quiet: functions return typed,
immutable rows or raise redacted exceptions, and never print/log source URLs,
paths, SQL text, or private query values in diagnostics.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from tidemark.store import initialize_db


@dataclass(frozen=True)
class PlayReportRow:
    """One identified song play from the persisted timeline."""

    song_id: int
    source_url: str
    segment_id: int
    segment_sequence: int
    start_ts: float
    duration_seconds: float
    title: str
    artist: str | None
    album: str | None
    score: float
    acoustid_id: str | None
    recording_id: str | None
    lookup_source: str | None


@dataclass(frozen=True)
class RepeatReportRow:
    """A deterministic repeated-play grouping for an identified recording."""

    identity: str
    title: str
    artist: str | None
    album: str | None
    count: int
    first_start_ts: float
    last_start_ts: float
    source_urls: tuple[str, ...]
    song_ids: tuple[int, ...]
    best_score: float
    acoustid_id: str | None
    recording_id: str | None


@dataclass(frozen=True)
class AdSummaryReportRow:
    """A grouped ad-marker summary for a source/classification/type tuple."""

    source_url: str
    classification: str
    marker_type: str
    count: int
    first_ts: float
    last_ts: float
    total_break_duration: float


class ReportError(ValueError):
    """Base class for redacted report errors."""


class MalformedReportQuery(ReportError):
    """Raised when report filter parameters are invalid."""


class ReportDatabaseMissing(ReportError):
    """Raised when a path-level report database does not exist."""


class _PlayGroup(NamedTuple):
    identity: str
    rows: tuple[PlayReportRow, ...]


_WHITESPACE_RE = re.compile(r"\s+")
_DATABASE_READ_FAILED = "database read failed during report generation"

_IDENTIFIED_PLAYS_SQL = """
SELECT id, segment_id, source_url, segment_sequence, start_ts, duration_seconds,
       title, artist, album, score, acoustid_id, recording_id, lookup_source
FROM songs
WHERE title IS NOT NULL
  AND score IS NOT NULL
  AND score >= ?
  AND (? IS NULL OR start_ts >= ?)
  AND (? IS NULL OR source_url = ?)
ORDER BY source_url ASC, start_ts ASC, segment_sequence ASC, id ASC
"""

_AD_SUMMARY_SQL = """
SELECT source_url, classification, marker_type, COUNT(*) AS marker_count,
       MIN(ts) AS first_ts, MAX(ts) AS last_ts, COALESCE(SUM(break_duration), 0.0) AS total_break_duration
FROM ad_events
WHERE (? IS NULL OR ts >= ?)
  AND (? IS NULL OR source_url = ?)
GROUP BY source_url, classification, marker_type
ORDER BY source_url ASC, classification ASC, marker_type ASC
"""


def _validate_optional_since(since_seconds: int | float | None) -> float | None:
    if since_seconds is None:
        return None
    if not isinstance(since_seconds, (int, float)) or isinstance(since_seconds, bool):
        raise MalformedReportQuery("since_seconds must be a non-negative number")
    normalized = float(since_seconds)
    if normalized < 0:
        raise MalformedReportQuery("since_seconds must be a non-negative number")
    return normalized


def _validate_optional_source_url(source_url: str | None) -> str | None:
    if source_url is None:
        return None
    if not isinstance(source_url, str) or not source_url.strip():
        raise MalformedReportQuery("source_url must be a non-empty string")
    return source_url


def _validate_min_score(min_score: int | float) -> float:
    if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
        raise MalformedReportQuery("min_score must be between 0 and 1")
    normalized = float(min_score)
    if normalized < 0 or normalized > 1:
        raise MalformedReportQuery("min_score must be between 0 and 1")
    return normalized


def _validate_min_count(min_count: int) -> int:
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise MalformedReportQuery("min_count must be >= 1")
    return min_count


def _read_play_rows(
    conn: sqlite3.Connection,
    *,
    since_seconds: float | None,
    source_url: str | None,
    min_score: float,
) -> tuple[PlayReportRow, ...]:
    try:
        rows = conn.execute(
            _IDENTIFIED_PLAYS_SQL,
            (min_score, since_seconds, since_seconds, source_url, source_url),
        ).fetchall()
    except sqlite3.Error as exc:
        raise ReportError(_DATABASE_READ_FAILED) from exc

    return tuple(
        PlayReportRow(
            song_id=int(row[0]),
            segment_id=int(row[1]),
            source_url=row[2],
            segment_sequence=int(row[3]),
            start_ts=float(row[4]),
            duration_seconds=float(row[5]),
            title=row[6],
            artist=row[7],
            album=row[8],
            score=float(row[9]),
            acoustid_id=row[10],
            recording_id=row[11],
            lookup_source=row[12],
        )
        for row in rows
    )


def plays_report(
    conn: sqlite3.Connection,
    *,
    since_seconds: int | float | None = None,
    source_url: str | None = None,
    min_score: int | float = 0.8,
) -> tuple[PlayReportRow, ...]:
    """Return identified song plays ordered by source and stream timestamp.

    The caller owns the connection lifecycle. Validation happens before any table
    reads so malformed parameters fail without depending on database state.
    """
    normalized_since = _validate_optional_since(since_seconds)
    normalized_source_url = _validate_optional_source_url(source_url)
    normalized_min_score = _validate_min_score(min_score)
    return _read_play_rows(
        conn,
        since_seconds=normalized_since,
        source_url=normalized_source_url,
        min_score=normalized_min_score,
    )


def _normalize_text_identity_value(value: str | None) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", value.strip().casefold())


def _repeat_identity(row: PlayReportRow) -> str:
    if row.recording_id:
        return f"recording:{row.recording_id}"
    if row.acoustid_id:
        return f"acoustid:{row.acoustid_id}"
    artist = _normalize_text_identity_value(row.artist)
    title = _normalize_text_identity_value(row.title)
    return f"text:{artist}|{title}"


def _group_repeated_plays(rows: tuple[PlayReportRow, ...], min_count: int) -> tuple[_PlayGroup, ...]:
    grouped: dict[str, list[PlayReportRow]] = {}
    for row in rows:
        grouped.setdefault(_repeat_identity(row), []).append(row)

    groups = tuple(
        _PlayGroup(identity=identity, rows=tuple(group_rows))
        for identity, group_rows in grouped.items()
        if len(group_rows) >= min_count
    )
    return tuple(sorted(groups, key=lambda group: (group.rows[0].start_ts, group.identity)))


def repeats_report(
    conn: sqlite3.Connection,
    *,
    since_seconds: int | float | None = None,
    source_url: str | None = None,
    min_count: int = 2,
    min_score: int | float = 0.8,
) -> tuple[RepeatReportRow, ...]:
    """Return deterministic repeated-play groups for identified songs."""
    normalized_min_count = _validate_min_count(min_count)
    rows = plays_report(conn, since_seconds=since_seconds, source_url=source_url, min_score=min_score)
    repeated_groups = _group_repeated_plays(rows, normalized_min_count)

    results: list[RepeatReportRow] = []
    for group in repeated_groups:
        group_rows = group.rows
        first = group_rows[0]
        best = max(group_rows, key=lambda row: (row.score, -row.start_ts, -row.song_id))
        results.append(
            RepeatReportRow(
                identity=group.identity,
                title=first.title,
                artist=first.artist,
                album=first.album,
                count=len(group_rows),
                first_start_ts=min(row.start_ts for row in group_rows),
                last_start_ts=max(row.start_ts for row in group_rows),
                source_urls=tuple(sorted({row.source_url for row in group_rows})),
                song_ids=tuple(row.song_id for row in group_rows),
                best_score=best.score,
                acoustid_id=first.acoustid_id,
                recording_id=first.recording_id,
            )
        )
    return tuple(results)


def ads_report(
    conn: sqlite3.Connection,
    *,
    since_seconds: int | float | None = None,
    source_url: str | None = None,
) -> tuple[AdSummaryReportRow, ...]:
    """Return grouped ad marker summaries ordered by source/classification/type."""
    normalized_since = _validate_optional_since(since_seconds)
    normalized_source_url = _validate_optional_source_url(source_url)
    try:
        rows = conn.execute(
            _AD_SUMMARY_SQL,
            (normalized_since, normalized_since, normalized_source_url, normalized_source_url),
        ).fetchall()
    except sqlite3.Error as exc:
        raise ReportError(_DATABASE_READ_FAILED) from exc

    return tuple(
        AdSummaryReportRow(
            source_url=row[0],
            classification=row[1],
            marker_type=row[2],
            count=int(row[3]),
            first_ts=float(row[4]),
            last_ts=float(row[5]),
            total_break_duration=float(row[6]),
        )
        for row in rows
    )


def _open_existing_database(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise ReportDatabaseMissing("database path does not exist")
    try:
        return initialize_db(db_path)
    except sqlite3.Error as exc:
        raise ReportError(_DATABASE_READ_FAILED) from exc


def plays_report_db(
    path: str | Path,
    *,
    since_seconds: int | float | None = None,
    source_url: str | None = None,
    min_score: int | float = 0.8,
) -> tuple[PlayReportRow, ...]:
    """Open, migrate, report plays, and close an existing database path."""
    conn = _open_existing_database(path)
    try:
        return plays_report(conn, since_seconds=since_seconds, source_url=source_url, min_score=min_score)
    finally:
        conn.close()


def repeats_report_db(
    path: str | Path,
    *,
    since_seconds: int | float | None = None,
    source_url: str | None = None,
    min_count: int = 2,
    min_score: int | float = 0.8,
) -> tuple[RepeatReportRow, ...]:
    """Open, migrate, report repeated plays, and close an existing database path."""
    conn = _open_existing_database(path)
    try:
        return repeats_report(
            conn,
            since_seconds=since_seconds,
            source_url=source_url,
            min_count=min_count,
            min_score=min_score,
        )
    finally:
        conn.close()


def ads_report_db(
    path: str | Path,
    *,
    since_seconds: int | float | None = None,
    source_url: str | None = None,
) -> tuple[AdSummaryReportRow, ...]:
    """Open, migrate, report ad summaries, and close an existing database path."""
    conn = _open_existing_database(path)
    try:
        return ads_report(conn, since_seconds=since_seconds, source_url=source_url)
    finally:
        conn.close()
