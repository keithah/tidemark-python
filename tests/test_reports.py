from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tidemark.markers import AdMarker
from tidemark.reports import (
    AdSummaryReportRow,
    MalformedReportQuery,
    PlayReportRow,
    RepeatReportRow,
    ReportDatabaseMissing,
    ReportError,
    ads_report,
    ads_report_db,
    plays_report,
    plays_report_db,
    repeats_report,
)
from tidemark.store import insert_ad_event, insert_segment, insert_song, migrate


def _connect_migrated() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    return conn


def _insert_segment(
    conn: sqlite3.Connection,
    *,
    source_url: str = "fixture://stream-a",
    sequence: int = 1,
    start_ts: float = 0.0,
) -> int:
    return insert_segment(
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


def _insert_song(
    conn: sqlite3.Connection,
    *,
    source_url: str = "fixture://stream-a",
    sequence: int = 1,
    start_ts: float = 0.0,
    duration_seconds: float = 10.0,
    fingerprint: str | None = None,
    acoustid_id: str | None = "acoustid-a",
    recording_id: str | None = "recording-a",
    title: str | None = "Needle Song",
    artist: str | None = "Needle Artist",
    album: str | None = "Needle Album",
    score: float | None = 0.95,
    lookup_source: str | None = "fixture",
) -> int:
    segment_id = _insert_segment(conn, source_url=source_url, sequence=sequence, start_ts=start_ts)
    return insert_song(
        conn,
        segment_id=segment_id,
        source_url=source_url,
        segment_sequence=sequence,
        start_ts=start_ts,
        duration_seconds=duration_seconds,
        fingerprint=fingerprint or f"fingerprint-{sequence}-{start_ts}",
        acoustid_id=acoustid_id,
        recording_id=recording_id,
        title=title,
        artist=artist,
        album=album,
        score=score,
        lookup_source=lookup_source,
    )


def _marker(
    *,
    marker_type: str = "SCTE35",
    classification: str = "BREAK_START",
    source: str = "fixture",
    segment: int = 1,
    timestamp: float = 0.0,
    break_duration: float | None = None,
) -> AdMarker:
    return AdMarker(
        type=marker_type,
        classification=classification,
        source=source,
        segment=segment,
        pts=timestamp,
        break_duration=break_duration,
        timestamp=timestamp,
    )


def test_plays_report_returns_identified_rows_in_stable_order_and_thresholds() -> None:
    conn = _connect_migrated()
    below_threshold_id = _insert_song(
        conn,
        source_url="fixture://stream-a",
        sequence=1,
        start_ts=20.0,
        title="Below Threshold",
        score=0.799,
    )
    null_title_id = _insert_song(
        conn,
        source_url="fixture://stream-a",
        sequence=2,
        start_ts=21.0,
        title=None,
        score=0.99,
    )
    null_score_id = _insert_song(
        conn,
        source_url="fixture://stream-a",
        sequence=3,
        start_ts=22.0,
        title="Null Score",
        score=None,
    )
    first_id = _insert_song(
        conn,
        source_url="fixture://stream-b",
        sequence=2,
        start_ts=30.0,
        title="Later Source",
        artist="Artist B",
        score=0.8,
    )
    second_id = _insert_song(
        conn,
        source_url="fixture://stream-a",
        sequence=4,
        start_ts=40.0,
        title="Later Time",
        artist="Artist A",
        acoustid_id=None,
        recording_id=None,
        score=0.81,
    )

    results = plays_report(conn)

    assert [row.song_id for row in results] == [second_id, first_id]
    assert below_threshold_id not in [row.song_id for row in results]
    assert null_title_id not in [row.song_id for row in results]
    assert null_score_id not in [row.song_id for row in results]
    assert results == (
        PlayReportRow(
            song_id=second_id,
            source_url="fixture://stream-a",
            segment_id=5,
            segment_sequence=4,
            start_ts=40.0,
            duration_seconds=10.0,
            title="Later Time",
            artist="Artist A",
            album="Needle Album",
            score=0.81,
            acoustid_id=None,
            recording_id=None,
            lookup_source="fixture",
        ),
        PlayReportRow(
            song_id=first_id,
            source_url="fixture://stream-b",
            segment_id=4,
            segment_sequence=2,
            start_ts=30.0,
            duration_seconds=10.0,
            title="Later Source",
            artist="Artist B",
            album="Needle Album",
            score=0.8,
            acoustid_id="acoustid-a",
            recording_id="recording-a",
            lookup_source="fixture",
        ),
    )


def test_plays_report_filters_by_stream_relative_since_and_source_url() -> None:
    conn = _connect_migrated()
    _insert_song(conn, source_url="fixture://stream-a", sequence=1, start_ts=9.9, title="Too Early")
    kept_id = _insert_song(conn, source_url="fixture://stream-a", sequence=2, start_ts=10.0, title="Kept")
    _insert_song(conn, source_url="fixture://stream-b", sequence=3, start_ts=10.0, title="Wrong Source")

    results = plays_report(conn, since_seconds=10.0, source_url="fixture://stream-a")

    assert [row.song_id for row in results] == [kept_id]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"since_seconds": -0.1}, "since_seconds"),
        ({"source_url": "   "}, "source_url"),
        ({"min_score": -0.1}, "min_score"),
        ({"min_score": 1.1}, "min_score"),
    ],
)
def test_plays_report_rejects_malformed_inputs_before_reading_tables(kwargs: dict[str, object], message: str) -> None:
    conn = sqlite3.connect(":memory:")

    with pytest.raises(MalformedReportQuery, match=message):
        plays_report(conn, **kwargs)  # type: ignore[arg-type]


def test_empty_migrated_database_reports_return_empty_tuples() -> None:
    conn = _connect_migrated()

    assert plays_report(conn) == ()
    assert repeats_report(conn) == ()
    assert ads_report(conn) == ()


def test_repeats_report_groups_by_stable_identity_and_min_count() -> None:
    conn = _connect_migrated()
    _insert_song(conn, sequence=1, start_ts=10.0, title="Same Song", artist="Same Artist", recording_id="recording-1")
    _insert_song(conn, sequence=2, start_ts=30.0, title="Same Song", artist="Same Artist", recording_id="recording-1")
    _insert_song(conn, sequence=3, start_ts=50.0, title="Same Song", artist="Same Artist", recording_id="recording-1")
    _insert_song(
        conn,
        sequence=4,
        start_ts=70.0,
        title="Fallback Title",
        artist="Fallback Artist",
        acoustid_id=None,
        recording_id=None,
    )
    _insert_song(
        conn,
        sequence=5,
        start_ts=90.0,
        title=" fallback   title ",
        artist="FALLBACK ARTIST",
        acoustid_id=None,
        recording_id=None,
    )
    _insert_song(conn, sequence=6, start_ts=110.0, title="Single", artist="Artist", recording_id="recording-single")

    results = repeats_report(conn, min_count=2)

    assert results == (
        RepeatReportRow(
            identity="recording:recording-1",
            title="Same Song",
            artist="Same Artist",
            album="Needle Album",
            count=3,
            first_start_ts=10.0,
            last_start_ts=50.0,
            source_urls=("fixture://stream-a",),
            song_ids=(1, 2, 3),
            best_score=0.95,
            acoustid_id="acoustid-a",
            recording_id="recording-1",
        ),
        RepeatReportRow(
            identity="text:fallback artist|fallback title",
            title="Fallback Title",
            artist="Fallback Artist",
            album="Needle Album",
            count=2,
            first_start_ts=70.0,
            last_start_ts=90.0,
            source_urls=("fixture://stream-a",),
            song_ids=(4, 5),
            best_score=0.95,
            acoustid_id=None,
            recording_id=None,
        ),
    )


def test_repeats_report_rejects_invalid_min_count_before_reading_tables() -> None:
    conn = sqlite3.connect(":memory:")

    with pytest.raises(MalformedReportQuery, match="min_count"):
        repeats_report(conn, min_count=0)


def test_ads_report_groups_by_source_classification_and_marker_type() -> None:
    conn = _connect_migrated()
    insert_ad_event(
        conn,
        "fixture://stream-b",
        _marker(marker_type="ID3", classification="BUMPER", segment=2, timestamp=30.0, break_duration=None),
    )
    insert_ad_event(
        conn,
        "fixture://stream-a",
        _marker(marker_type="SCTE35", classification="BREAK_START", segment=1, timestamp=10.0, break_duration=15.0),
    )
    insert_ad_event(
        conn,
        "fixture://stream-a",
        _marker(marker_type="SCTE35", classification="BREAK_START", segment=2, timestamp=40.0, break_duration=30.0),
    )
    insert_ad_event(
        conn,
        "fixture://stream-a",
        _marker(marker_type="SCTE35", classification="BREAK_END", segment=3, timestamp=50.0, break_duration=None),
    )

    results = ads_report(conn, since_seconds=10.0)

    assert results == (
        AdSummaryReportRow(
            source_url="fixture://stream-a",
            classification="BREAK_END",
            marker_type="SCTE35",
            count=1,
            first_ts=50.0,
            last_ts=50.0,
            total_break_duration=0.0,
        ),
        AdSummaryReportRow(
            source_url="fixture://stream-a",
            classification="BREAK_START",
            marker_type="SCTE35",
            count=2,
            first_ts=10.0,
            last_ts=40.0,
            total_break_duration=45.0,
        ),
        AdSummaryReportRow(
            source_url="fixture://stream-b",
            classification="BUMPER",
            marker_type="ID3",
            count=1,
            first_ts=30.0,
            last_ts=30.0,
            total_break_duration=0.0,
        ),
    )


def test_report_db_path_wrapper_checks_missing_path_without_creating_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite"

    with pytest.raises(ReportDatabaseMissing, match="database"):
        plays_report_db(path)

    assert not path.exists()


def test_path_wrapper_opens_migrates_and_closes_existing_database(tmp_path: Path) -> None:
    path = tmp_path / "reports.sqlite"
    conn = sqlite3.connect(path)
    try:
        migrate(conn)
        row_id = _insert_song(conn, title="Path Wrapper")
    finally:
        conn.close()

    assert plays_report_db(path) == (
        PlayReportRow(
            song_id=row_id,
            source_url="fixture://stream-a",
            segment_id=1,
            segment_sequence=1,
            start_ts=0.0,
            duration_seconds=10.0,
            title="Path Wrapper",
            artist="Needle Artist",
            album="Needle Album",
            score=0.95,
            acoustid_id="acoustid-a",
            recording_id="recording-a",
            lookup_source="fixture",
        ),
    )
    path.unlink()
    assert not path.exists()


def test_schema_missing_sqlite_read_failure_is_redacted() -> None:
    conn = sqlite3.connect(":memory:")

    with pytest.raises(ReportError, match="database read failed during report generation") as exc_info:
        plays_report(conn, source_url="fixture://private-stream")

    message = str(exc_info.value)
    assert "private-stream" not in message
    assert "songs" not in message
    assert "no such table" not in message


def test_database_read_failure_preserves_original_exception_as_cause() -> None:
    conn = sqlite3.connect(":memory:")

    with pytest.raises(ReportError) as exc_info:
        ads_report(conn)

    assert isinstance(exc_info.value.__cause__, sqlite3.Error)
